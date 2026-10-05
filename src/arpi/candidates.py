"""The candidate engine. Deterministic, no model.

Input is evidence: the bars (a confidence map) and any other witnesses
(see evidence.py). Output is every EAN-13 that the surviving bars
allow, ranked by how well it explains them, after the structural constraints
have removed the ones that cannot exist:

    1. module geometry   each digit's 7 modules scored against all 10 (or 20)
                         patterns, so a half-read digit keeps 2-3 live options
    2. guards            fixed, and already used by the scanner to place the grid
    3. parity            the L/G pattern of the left half must be one of the 10
                         that encode a leading digit; 54 of 64 are illegal
    4. checksum          mod 10, weights 1 and 3
    5. GS1 prefix        unassigned prefix ranges are discarded
    6. known codes       optional. Score the customer's own list directly.

The search is an exact K-best dynamic programme over a state of
(checksum residue, parity bits so far). 640 states, K kept per state, so a digit
that is completely destroyed costs nothing extra: all of its options are
carried and the checksum decides later. A global beam would drop the true code
the moment one damaged digit made its prefix look unlikely.

Scores are log-likelihoods of the module evidence under each candidate. The
"share" reported next to each candidate is a softmax over the candidates found,
which is a ranking aid and not a calibrated probability. Calibrating it is the
job of the evaluation, not of this file.
"""
from dataclasses import dataclass, field

import numpy as np

from . import ean13, gs1
from .confmap import ConfidenceMap
from .evidence import Evidence, from_map, total

_LEAD = np.array(ean13.LEADING_FROM_BITS)
_PARITY = np.array([[0 if ch == "L" else 1 for ch in p] for p in ean13.PARITY])


def digit_likelihoods(cmap: ConfidenceMap):
    """Bars only, as (left, right). Kept for callers that want the arrays."""
    ev = from_map(cmap)
    return ev.left, ev.right


def _keep_top_k(score, state, k):
    """Indices of the k best hypotheses within each state."""
    order = np.lexsort((-score, state))
    s_sorted = state[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(s_sorted)]))
    rank = np.arange(len(order)) - group_start
    return order[rank < k]


def k_best(left, right, k=32, lead_ll=None):
    """Exact top-k per final state. Returns (codes as 13-char strings, scores),
    best first, every one structurally valid apart from the GS1 check.

    `lead_ll` is direct evidence about the leading digit (the printed text).
    The final state carries the full parity pattern, so the leading digit is a
    function of the state and adding its score at the end keeps the search
    exact."""
    score = np.zeros(1)
    residue = np.zeros(1, dtype=np.int64)
    parity = np.zeros(1, dtype=np.int64)
    digits = np.zeros((1, 0), dtype=np.int8)

    for pos in range(12):
        w = ean13.weight(pos + 1)            # EAN index pos+1; index 0 is implied
        if pos < 6:
            opt_ll = left[pos].reshape(-1)               # 20: [par*10 + d]
            opt_d = np.tile(np.arange(10), 2)
            opt_par = np.repeat([0, 1], 10)
        else:
            opt_ll = right[pos - 6]
            opt_d = np.arange(10)
            opt_par = np.zeros(10, dtype=np.int64)
        n, m = len(score), len(opt_ll)
        new_score = (score[:, None] + opt_ll[None, :]).ravel()
        new_res = ((residue[:, None] + w * opt_d[None, :]) % 10).ravel()
        if pos < 6:
            new_par = ((parity[:, None] << 1) | opt_par[None, :]).ravel()
        else:
            new_par = np.repeat(parity, m)
        src = np.repeat(np.arange(n), m)
        new_d = np.tile(opt_d, n)

        state = new_res * 64 + new_par
        keep = _keep_top_k(new_score, state, k)
        score, residue, parity = new_score[keep], new_res[keep], new_par[keep]
        digits = np.concatenate([digits[src[keep]], new_d[keep, None].astype(np.int8)],
                                axis=1)

    lead = _LEAD[parity]
    ok = (lead >= 0) & ((residue + np.where(lead >= 0, lead, 0)) % 10 == 0)
    score, digits, lead = score[ok], digits[ok], lead[ok]
    if lead_ll is not None:
        score = score + np.asarray(lead_ll)[lead]
    order = np.argsort(-score)
    codes = ["{}{}".format(lead[i], "".join(map(str, digits[i]))) for i in order]
    return codes, score[order]


def score_codes(left, right, codes, lead_ll=None):
    """Log-likelihood of each given 13-digit code. Used for the known-code
    list: scoring a few thousand codes directly is exact and cheaper than
    hoping they all made it into the K-best."""
    if not codes:
        return np.zeros(0)
    arr = np.array([[int(c) for c in code] for code in codes])
    par = _PARITY[arr[:, 0]]
    total_ = np.zeros(len(codes))
    for pos in range(6):
        total_ += left[pos, par[:, pos], arr[:, pos + 1]]
    for pos in range(6):
        total_ += right[pos, arr[:, pos + 7]]
    if lead_ll is not None:
        total_ += np.asarray(lead_ll)[arr[:, 0]]
    return total_


def score_evidence(ev: Evidence, codes):
    return score_codes(ev.left, ev.right, codes, ev.lead)


@dataclass
class Candidate:
    code: str
    log_likelihood: float
    share: float                       # softmax over the candidates returned
    symbology: str
    prefix: str                        # GS1 assignment, or "unassigned"
    in_known_list: bool = None         # None when no list was given
    survived: list = field(default_factory=list)
    # Per source, how much this candidate trails the best-scoring shown
    # candidate for that source, in nats. 0 means "this source's favourite".
    support: dict = field(default_factory=dict)


@dataclass
class Result:
    status: str          # "read" | "reconstructed" | "unique-in-list" | "none"
    candidates: list
    per_digit: list      # what each position looked like on its own
    message: str
    orientation: str = "forward"
    # How many codes explain the evidence within `list_margin` of the best,
    # which can be far more than the `top` shown. Capped by the K-best search.
    live_count: int = 0
    sources: list = field(default_factory=list)


def _softmax(x):
    if len(x) == 0:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def per_digit_view(ev: Evidence):
    """Each of the 13 positions judged in isolation, without the checksum.
    Position 1 (the leading digit) comes from the left half's parity plus
    any direct evidence for it. This is what gets reported when nothing
    survives, and it is what decides whether the read needed reconstruction."""
    out = []
    for i, ll in enumerate(ev.position_ll()):
        p = _softmax(ll)
        best = int(np.argmax(p))
        out.append({"position": i + 1, "best": best,
                    "confidence": round(float(p[best]), 3),
                    "alternatives": [int(d) for d in np.argsort(-p)[:3]
                                     if p[d] > 0.05]})
    return out


def _as_evidence(x):
    if isinstance(x, ConfidenceMap):
        return [from_map(x)], x.orientation
    if isinstance(x, Evidence):
        return [x], "forward"
    return list(x), "forward"


def solve(evidence, known_codes=None, top=10, k=32,
          gs1_filter=True, legible_threshold=0.99, digit_threshold=0.9,
          list_margin=6.0, outside_margin=15.0):
    """Rank the codes the evidence allows.

    `evidence` is a ConfidenceMap (bars only), one Evidence, or a list of
    them, one per source. Sources named "bars" decide whether this was a
    clean read; every other source can only make it a reconstruction.
    """
    sources, orientation = _as_evidence(evidence)
    ev = total(sources)
    bars = total([s for s in sources if s.source == "bars"])
    per_digit = per_digit_view(ev)

    survived_base = ["module geometry", "guards", "parity", "checksum"]
    outside_note = ""
    if known_codes is not None:
        known = sorted({ean13.normalise(c) for c in known_codes})
        known = [c for c in known if ean13.is_valid(c)]
        scores = score_evidence(ev, known)
        order = np.argsort(-scores)[:top]
        codes = [known[i] for i in order]
        scores = scores[order]
        # Only codes that explain the evidence about as well as the best one
        # are live. Everything in a list of thousands "fits" in the sense of
        # having a score; that is not the same thing as being a candidate.
        if len(scores):
            live = scores >= scores[0] - list_margin
            codes = [c for c, ok in zip(codes, live) if ok]
            scores = scores[live]
            # If something outside the list explains it far better, the list
            # may be missing this item, and saying so is the honest move.
            free_codes, free_scores = k_best(ev.left, ev.right, k=k, lead_ll=ev.lead)
            if len(free_scores) and free_scores[0] > scores[0] + outside_margin \
                    and free_codes[0] not in set(known):
                outside_note = (" A code outside the list ({}) fits much "
                                "better; the list may be incomplete."
                                .format(free_codes[0]))
    else:
        codes, scores = k_best(ev.left, ev.right, k=k, lead_ll=ev.lead)

    if gs1_filter:
        keep = [i for i, c in enumerate(codes) if gs1.is_assigned(c)]
        codes = [codes[i] for i in keep]
        scores = scores[keep]

    scores = np.asarray(scores)
    live_count = int((scores >= scores[0] - list_margin).sum()) if len(scores) else 0
    codes, scores = codes[:top], scores[:top]
    share = _softmax(scores)
    survived = survived_base + (["GS1 prefix"] if gs1_filter else []) \
        + (["known-code list"] if known_codes is not None else [])

    support = {}
    for s in sources:
        sc = score_evidence(s, codes)
        if len(sc):
            support[s.source] = np.round(sc - sc.max(), 2)

    cands = [Candidate(code=c, log_likelihood=round(float(s), 3),
                       share=round(float(p), 4), symbology=ean13.symbology(c),
                       prefix=gs1.assignment(c) or "unassigned",
                       in_known_list=(True if known_codes is not None else None),
                       survived=survived,
                       support={name: float(v[i]) for name, v in support.items()})
             for i, (c, s, p) in enumerate(zip(codes, scores, share))]
    names = [s.source for s in sources]

    if not cands:
        return Result("none", [], per_digit,
                      "Nothing survives the constraints. Readable digits are "
                      "listed per position.", orientation, 0, names)

    # "read" means what a conventional scanner would mean: the bars alone,
    # every digit clear, and the top candidate is exactly what each digit said
    # on its own. Anything else involved choosing, and choosing is
    # reconstruction - however many witnesses agreed.
    #
    # The per-digit bar is lower than the candidate bar on purpose: on a clean
    # read the left digits sit around 0.94 on their own, because an L pattern
    # and a G pattern of another digit can be one module apart. Parity is what
    # separates them, and parity is a whole-symbol property.
    bars_view = per_digit_view(bars) if any(n == "bars" for n in names) else None
    if bars_view is not None:
        top_code = cands[0].code
        independent = all(int(top_code[i]) == bars_view[i]["best"]
                          and bars_view[i]["confidence"] >= digit_threshold
                          for i in range(1, 13))
        bars_scores = score_evidence(bars, codes)
        bars_share = _softmax(bars_scores)[0] if len(bars_scores) else 0
        if independent and bars_share >= legible_threshold \
                and cands[0].share >= legible_threshold:
            return Result("read", cands[:1], per_digit,
                          "Read cleanly. No reconstruction.", orientation, 1, names)

    if known_codes is not None and len(cands) == 1 and not outside_note:
        return Result("unique-in-list", cands, per_digit,
                      "Reconstructed. One code in the known-code list fits. "
                      "Confirm before use.", orientation, 1, names)

    if live_count <= len(cands):
        shown = "{} candidates".format(len(cands))
    else:
        shown = "{} of {}{} equally plausible candidates".format(
            len(cands), live_count, "+" if live_count >= k else "")
    return Result("reconstructed", cands, per_digit,
                  "Reconstructed from damaged bars. {}, ranked. "
                  "Confirm before use.{}".format(shown, outside_note),
                  orientation, live_count, names)


def solve_either_way(cmap: ConfidenceMap, **kwargs):
    """The guard and digit-edge template is mirror symmetric, so the scanner
    cannot tell which way round the symbol is. The parity rule can: read
    backwards, the left half comes out all-G, which no leading digit encodes.
    Try both and keep the one whose best candidate explains the bars better."""
    results = []
    for m in (cmap, cmap.reversed()):
        r = solve(m, **kwargs)
        best = r.candidates[0].log_likelihood if r.candidates else -1e18
        results.append((best, r))
    return max(results, key=lambda t: t[0])[1]
