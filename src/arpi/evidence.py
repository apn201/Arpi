"""Evidence: what each witness says about each digit, in one currency.

Every source - the bars, the printed text, another frame, later the scene -
is turned into the same shape: a log-likelihood for every digit at every
position, with the left half split by parity because the bars can tell L from
G and nothing else can.

    left   (6, 2, 10)   positions 2-7, [parity L/G, digit]
    right  (6, 10)      positions 8-13
    lead   (10,)        position 1, which the bars only carry through parity

Sources add. They do not vote on finished answers: a vote would make the
checksum one opinion among several. Here the structural rules (parity,
checksum, GS1) stay hard in the search, and each source only moves weight
between codes those rules allow. Nothing a source says can produce a code the
rules forbid.

Each source is tempered so that no single soft witness can overrule clear
bars. That ceiling is the safety argument, and the evaluation checks it: a
source that raises the false-positive rate gets its weight cut or is removed.
"""
from dataclasses import dataclass, field

import numpy as np

from . import ean13
from .confmap import ConfidenceMap

_L = np.array([[int(b) for b in c] for c in ean13.L_CODES], dtype=float)
_G = np.array([[int(b) for b in c] for c in ean13.G_CODES], dtype=float)
_R = np.array([[int(b) for b in c] for c in ean13.R_CODES], dtype=float)
_PARITY = np.array([[0 if ch == "L" else 1 for ch in p] for p in ean13.PARITY])


@dataclass
class Evidence:
    source: str
    left: np.ndarray = field(default_factory=lambda: np.zeros((6, 2, 10)))
    right: np.ndarray = field(default_factory=lambda: np.zeros((6, 10)))
    lead: np.ndarray = field(default_factory=lambda: np.zeros(10))
    detail: dict = field(default_factory=dict)

    def __add__(self, other):
        return Evidence(self.source if self.source == other.source else "combined",
                        self.left + other.left, self.right + other.right,
                        self.lead + other.lead)

    def scaled(self, w):
        return Evidence(self.source, self.left * w, self.right * w, self.lead * w,
                        dict(self.detail))

    def position_ll(self):
        """(13, 10): best log-likelihood of each digit at each position,
        parity maximised out. Position 0 is what the left half's parity
        implies about the leading digit, plus any direct evidence for it."""
        out = np.zeros((13, 10))
        for d in range(10):
            out[0, d] = self.lead[d] + sum(self.left[p, _PARITY[d, p]].max()
                                           for p in range(6))
        out[1:7] = self.left.max(axis=1)
        out[7:] = self.right
        return out


def total(evidences):
    out = Evidence("total")
    for e in evidences:
        out = Evidence("total", out.left + e.left, out.right + e.right,
                       out.lead + e.lead)
    return out


# ---- sources ---------------------------------------------------------------

def from_map(cmap: ConfidenceMap, source="bars"):
    """The bars. Per position, how well each of the 10 (or 20) patterns
    explains the seven modules, given each module's p_bar and confidence."""
    q = cmap.effective()
    logq, log1q = np.log(q), np.log(1 - q)
    ev = Evidence(source)
    for pos in range(12):
        s = ean13.digit_start(pos)
        a, b = logq[s:s + 7], log1q[s:s + 7]
        if pos < 6:
            ev.left[pos, 0] = _L @ a + (1 - _L) @ b
            ev.left[pos, 1] = _G @ a + (1 - _G) @ b
        else:
            ev.right[pos - 6] = _R @ a + (1 - _R) @ b
    return ev


def from_digit_probs(probs, readable, source="text", weight=0.6, eps=0.05):
    """A reader that names digits directly: the printed text, a vision model,
    a person. `probs` is (13, 10), `readable` (13,) in 0..1.

    Per position the contribution is weight * readable * log(eps + p). With
    the defaults, the most a confidently wrong digit can cost the true code is
    0.6 * log(1.05 / 0.05) = 1.8 nats. One clearly read module in the bars is
    log(0.98 / 0.02) = 3.9 nats. Within one parity set a wrong digit differs
    from the right one in at least two modules (checked: L, G and R all have
    minimum distance 2). An L and a G pattern can be one module apart, but
    swapping one flips the parity pattern, which changes the leading digit,
    which the checksum then forces some other digit to absorb. So the text can
    settle what the bars leave open, and cannot outvote bars read clearly.
    """
    probs = np.asarray(probs, float)
    readable = np.clip(np.asarray(readable, float), 0, 1)
    ll = weight * readable[:, None] * np.log(eps + probs)
    ll -= ll.max(axis=1, keepdims=True)            # only differences matter
    ev = Evidence(source)
    ev.lead = ll[0]
    ev.left = np.repeat(ll[1:7, None, :], 2, axis=1)
    ev.right = ll[7:].copy()
    return ev


def orientation_score(ev: Evidence):
    """How well the evidence fits a legal symbol, without the checksum. Cheap,
    and enough to tell forward from reversed: read backwards, the left half
    comes out all-G, which no leading digit encodes, so this score collapses."""
    right = ev.right.max(axis=1).sum()
    left = max(sum(ev.left[p, _PARITY[d, p]].max() for p in range(6)) + ev.lead[d]
               for d in range(10))
    return float(left + right)
