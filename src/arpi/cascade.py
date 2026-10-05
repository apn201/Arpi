"""OpenCV first, ARPI as the fallback, and a check on OpenCV in between.

The product is a scanner. OpenCV's own barcode decoder is fast and reads the
easy symbols, so it goes first. ARPI takes every symbol OpenCV detected but
could not read, and every symbol only ARPI's localiser found.

What a plain cascade would miss: OpenCV is sometimes confidently wrong. In
the physical set it returned 4 wrong codes, every one passing its checksum -
three of them EAN-8 codes read out of part of a torn or angled EAN-13. A
cascade that trusts the first stage passes those straight through. So each
OpenCV read can be checked:

    verify="none"        trust OpenCV. The baseline a plain cascade gives.
    verify="symbology"   free: anything but EAN-13 / UPC-A goes to ARPI. In
                         this deployment every code is EAN-13 family, and the
                         partial EAN-8 read is the commonest wrong answer.
    verify="full"        also scan the symbol with ARPI and accept OpenCV's
                         code only if it explains the bars nearly as well as
                         ARPI's best candidate. Costs ARPI's time on every
                         symbol, and catches what the symbology rule cannot.

Every result says which path produced it, so the HUD can show the fast path
and the slow path differently and the report can count them.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ean13, vision
from .candidates import k_best, score_evidence
from .evidence import total
from .session import Session

EAN_FAMILY = {"EAN_13", "UPC_A"}

# How much worse than the best code the evidence allows OpenCV's read may be
# and still count as confirmed, in nats. Correct reads sit at the top (gap 0);
# the wrong reads caught in the physical set were 7 to 41 nats behind.
VERIFY_MARGIN = 3.0


def verify_read(code, sources, known_codes=None, margin=VERIFY_MARGIN):
    """Is OpenCV's read what the bars and the printed digits say?

    Compared against the best code the evidence allows anywhere - the free
    search - never against the best code in a list. The first version
    compared against the list's best: when no list code fitted (a code
    missing from the list), every list code scored terribly, a wrong OpenCV
    read beat them all, and it was shown as verified.

    Returns (accepted, reason, best code the evidence allows)."""
    if known_codes is not None:
        allowed = {ean13.normalise(c) for c in known_codes if len("".join(
            ch for ch in str(c) if ch.isdigit())) in (12, 13)}
        if code not in allowed:
            return False, "not in your list", None
    if not sources:
        return False, "nothing to check against", None
    ev = total(sources)
    codes, scores = k_best(ev.left, ev.right, k=32, lead_ll=ev.lead)
    if not codes:
        return False, "nothing fits the bars", None
    mine = score_evidence(ev, [code])[0]
    if mine >= scores[0] - margin:
        return True, "", codes[0]
    return False, "the evidence favours {} by {:.0f}".format(codes[0], scores[0] - mine), codes[0]


@dataclass
class Symbol:
    centre: tuple
    polygon: list
    code: str = None             # the answer, if one is asserted
    status: str = "none"         # read | reconstructed | unique-in-list | disputed | none
    path: str = ""               # opencv | opencv-verified | arpi | arpi-over-opencv
    candidates: list = field(default_factory=list)
    opencv_read: str = None      # what OpenCV said, kept even when overruled
    opencv_plausible: bool = False   # EAN-13 family, valid check digit
    note: str = ""
    result: object = None        # candidates.Result when ARPI ran


def _normalise(code, kind):
    if kind == "UPC_A" and len(code) == 12:
        return "0" + code
    return code


def _arpi(reading, known, use_text, with_session=False):
    s = Session(known_codes=known, use_text=use_text)
    s.add_reading(reading)
    return (s.result(), s) if with_session else s.result()


def _symbol_from_result(sym, r):
    sym.result = r
    sym.status = r.status if r else "none"
    sym.candidates = [c.code for c in r.candidates] if r else []
    if r and r.status in ("read", "unique-in-list"):
        sym.code = r.candidates[0].code
    return sym


def scan(gray, known_codes=None, verify="symbology", use_text=True,
         margin=VERIFY_MARGIN):
    """Every symbol in the frame. Returns a list of Symbol."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    det = cv2.barcode.BarcodeDetector()
    ok, decoded, kinds, pts = det.detectAndDecodeWithType(gray)
    polys = [np.asarray(p).reshape(-1, 2) for p in pts] if pts is not None else []
    decoded = list(decoded) if decoded is not None else [""] * len(polys)
    kinds = list(kinds) if kinds is not None else [""] * len(polys)

    out = []
    by_region = vision.read_frame(gray, regions=vision.propose(gray, polys))

    # Pair each OpenCV detection with the reading made from it, by centre.
    def reading_for(poly):
        c = poly.mean(0)
        best, dist = None, 1e18
        for rd in by_region:
            d = np.linalg.norm(rd.region.points.mean(0) - c)
            if d < dist:
                best, dist = rd, d
        return best if dist < 0.5 * np.ptp(poly[:, 0]) + 1 else None

    used = set()
    for poly, code, kind in zip(polys, decoded, kinds):
        sym = Symbol(centre=tuple(poly.mean(0)), polygon=poly.tolist())
        rd = reading_for(poly)
        if rd is not None:
            used.add(id(rd))
        if code:
            code = _normalise(code, kind)
            sym.opencv_read = code
            plausible = kind in EAN_FAMILY and len(code) == 13 and ean13.is_valid(code)
            sym.opencv_plausible = plausible
            if verify == "none" or (verify == "symbology" and plausible):
                sym.code, sym.status, sym.path = code, "read", "opencv"
                sym.candidates = [code]
                out.append(sym)
                continue
            if not plausible:
                sym.note = "OpenCV read {} ({}), not EAN-13 family; rejected".format(
                    code, kind)
            if rd is None:
                out.append(sym)
                continue
            r, sess = _arpi(rd, known_codes, use_text, with_session=True)
            if plausible and r is not None:
                ok, why, _ = verify_read(code, sess.sources(), known_codes, margin)
                if ok:
                    sym.code, sym.status, sym.path = code, "read", "opencv-verified"
                    sym.candidates = [code] + [c for c in
                                               (x.code for x in r.candidates) if c != code]
                    sym.result = r
                    out.append(sym)
                    continue
                sym.note = "OpenCV read {}: {}; overruled".format(code, why)
            _symbol_from_result(sym, r)
            sym.path = "arpi-over-opencv"
            if plausible:
                # Two decoders disagree. That is the safety rule's case
                # exactly: nothing is asserted, the person picks. OpenCV's
                # read stays on the list, second, so a correct read that the
                # bars happen to disfavour is one tap away, not gone.
                sym.code, sym.status = None, "disputed"
                if code not in sym.candidates:
                    sym.candidates.insert(min(1, len(sym.candidates)), code)
            out.append(sym)
            continue
        # Detected, not decoded: ARPI's job.
        if rd is not None:
            _symbol_from_result(sym, _arpi(rd, known_codes, use_text))
            sym.path = "arpi"
        out.append(sym)

    # Symbols only ARPI's own localiser found.
    for rd in by_region:
        if id(rd) in used:
            continue
        sym = Symbol(centre=tuple(rd.region.points.mean(0)),
                     polygon=rd.region.points.tolist())
        _symbol_from_result(sym, _arpi(rd, known_codes, use_text))
        sym.path = "arpi"
        out.append(sym)
    return out
