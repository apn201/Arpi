"""The live scanner: one aimed code, frame after frame, and everything the
HUD draws.

The normal use is one code held in front of the camera, so this works on the
symbol nearest the centre of the frame and nothing else. The cascade is the
same as cascade.py - OpenCV first, ARPI on what it cannot read, OpenCV's read
checked against the bars - but evidence accumulates across frames:

    phone  --frame + state-->  scan()  --result + overlay + state-->  phone

The state is the summed evidence so far (a few hundred numbers). The phone
keeps it and sends it back, so the server holds nothing between requests and
can be a Lambda. The phone starts a new state when the user starts a new
scan or the target jumps.

Everything geometric is returned in camera-image pixels, so the page only
has to scale it to the screen.
"""
import time

import cv2
import numpy as np

from . import ean13, vision
from .cascade import EAN_FAMILY, VERIFY_MARGIN, _normalise, verify_read
from .evidence import from_map, total
from .session import Session


def _target(gray, polys):
    """The region to scan: OpenCV's detection nearest the frame centre, else
    ARPI's own region nearest it."""
    h, w = gray.shape
    centre = np.array([w / 2, h / 2])
    if polys:
        i = int(np.argmin([np.linalg.norm(p.mean(0) - centre) for p in polys]))
        regs = vision.regions_from_polygons(gray, [polys[i]])
        if regs:
            return regs[0], i
    own = vision.localise(gray, max_regions=6)
    if not own:
        return None, None
    return min(own, key=lambda r: np.linalg.norm(r.points.mean(0) - centre)), None


# A frame counts as evidence only if the module grid actually fitted it. Clean
# codes in the physical set fit at about 0.95 with contrast 0.3 and up; the
# taped F04 fits at 0.83. The frames that sent the phone's candidates
# wandering fitted at 0.47-0.59 with contrast 0.02-0.11.
MIN_FIT = 0.70
MIN_CONTRAST = 0.12
# Positions on which this frame and the evidence so far must both be sure and
# disagree before the frame is taken to be a different code.
TARGET_CHANGED = 3
# A frame must fit at least this well to count as a different code. Clean
# codes fit at about 0.95; clutter that passes the 0.70 gate sits below this.
CHANGE_MIN_FIT = 0.80
SURE = 0.9


def _frame_ok(d):
    fit, contrast = d.get("grid_fit", 0), d.get("contrast", 0)
    if fit < MIN_FIT:
        return False, "grid did not fit (fit {:.2f})".format(fit)
    if contrast < MIN_CONTRAST:
        return False, "too little contrast ({:.2f})".format(contrast)
    return True, ""


def _sure_digits(sources):
    """Per position: (best digit, share) from all sources combined."""
    if not sources:
        return []
    out = []
    for row in total(sources).position_ll():
        p = np.exp(row - row.max())
        p /= p.sum()
        out.append((int(np.argmax(p)), float(p.max())))
    return out


def _disagreements(a_sources, b_sources):
    a, b = _sure_digits(a_sources), _sure_digits(b_sources)
    return sum(1 for (da, pa), (db, pb) in zip(a, b)
               if pa >= SURE and pb >= SURE and da != db)


def _grid_lines(reading):
    """Module boundaries 0..95 as line segments, top to bottom of the bars,
    in image pixels, following the warp if there is one."""
    a, b, k = reading.grid
    warp = reading.cmap.diagnostics.get("warp") or [0.0] * len(vision.SEGMENTS)
    seg = vision._SEG_OF
    u = np.arange(ean13.MODULES + 1, dtype=float)
    off = np.array([warp[seg[min(int(x), ean13.MODULES - 1)]] for x in u])
    xs = vision._positions(a, b, k, u + off)
    top, bottom = reading.bars
    pts_top = reading.to_image(np.column_stack([xs, np.full_like(xs, top)]))
    pts_bot = reading.to_image(np.column_stack([xs, np.full_like(xs, bottom)]))
    return np.round(np.hstack([pts_top, pts_bot]), 1).tolist()


def scan(gray, state=None, known_codes=None, use_text=True, margin=VERIFY_MARGIN):
    t0 = time.time()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    det = cv2.barcode.BarcodeDetector()
    ok, decoded, kinds, pts = det.detectAndDecodeWithType(gray)
    polys = [np.asarray(p).reshape(-1, 2) for p in pts] if pts is not None else []
    t_opencv = time.time() - t0

    region, det_index = _target(gray, polys)
    out = {"found": region is not None, "frame": [int(gray.shape[1]), int(gray.shape[0])],
           "timings": {"opencv_s": round(t_opencv, 3)}, "state": state or {}}
    if region is None:
        out["status"] = "searching"
        return out

    opencv_read, plausible = None, False
    if det_index is not None and decoded is not None and decoded[det_index]:
        opencv_read = _normalise(decoded[det_index], kinds[det_index])
        plausible = (kinds[det_index] in EAN_FAMILY and len(opencv_read) == 13
                     and ean13.is_valid(opencv_read))
        out["opencv"] = {"read": opencv_read, "type": kinds[det_index],
                         "plausible": plausible}
    out["polygon"] = np.round(region.points, 1).tolist()
    # Whether OpenCV's own detector found this symbol (not just ARPI's
    # localiser). The simple view says "barcode found, cannot decode" only
    # when OpenCV itself found it.
    out["opencv_detected"] = det_index is not None

    t1 = time.time()
    readings = vision.read_frame(gray, regions=[region])
    if not readings:
        out["status"] = "unreadable"
        return out
    reading = readings[0]
    # This frame on its own first: is it good enough to count, and is it the
    # same code as the evidence carried in?
    frame_only = Session(known_codes=known_codes, use_text=use_text)
    frame = frame_only.add_reading(reading)
    prior = Session(known_codes=known_codes, use_text=use_text).load_state(state)
    has_prior = bool(prior.prior)
    d = reading.cmap.diagnostics
    accepted, reason = _frame_ok(d)
    changed = False
    if not accepted:
        # A frame whose grid did not fit is not evidence about anything. It
        # is dropped, and the result so far stands. Hand shake and a target
        # sliding out of the box produced these on a phone, and summing them
        # in made the candidates jump.
        session = prior if has_prior else None
    elif has_prior and _disagreements(frame_only.sources(), prior.sources()) >= TARGET_CHANGED:
        # Confident disagreement on several positions: this is another code.
        # On a full sheet, a small hand movement makes the neighbour the code
        # nearest the centre, and adding its evidence to the first one's made
        # a mixture of two labels.
        if d.get("grid_fit", 0) >= CHANGE_MIN_FIT:
            session, changed = frame_only, True
        else:
            # Something else came into view, but not clearly a barcode: a
            # keyboard, text, the edge of the desk. On the phone that was
            # enough to throw away a finished result when the label was taken
            # away. It is skipped, and the result so far stands.
            accepted, reason = False, "something else in view, not clearly a code"
            session = prior
    elif has_prior:
        session = prior
        session.frames.append(frame)
    else:
        session = frame_only
    out["frame_quality"] = {"accepted": accepted, "reason": reason,
                            "grid_fit": d.get("grid_fit"), "contrast": d.get("contrast")}
    out["target_changed"] = changed
    if session is None:
        # Nothing trustworthy yet. Show where we looked, assert nothing.
        out.update({"status": "rejected", "code": None, "path": "",
                    "note": ("OpenCV read {}, not checked: this frame could not "
                             "be fitted".format(opencv_read) if opencv_read else ""),
                    "candidates": [], "candidate_order": [opencv_read] if opencv_read else [],
                    "live_count": 0, "per_digit": [], "ledger": {},
                    "grid": _grid_lines(reading),
                    "modules": {"p_bar": [round(float(x), 3) for x in reading.cmap.p_bar],
                                "confidence": [round(float(x), 3) for x in reading.cmap.confidence]},
                    "printed_digits": None, "diagnostics": d, "frames": 0,
                    "state": state or {}})
        out["timings"].update({"arpi_s": round(time.time() - t1, 3),
                               "total_s": round(time.time() - t0, 3)})
        return out
    result = session.result()
    t2 = time.time()

    # The cascade's check, on everything seen so far.
    status, code, path, note = None, None, "arpi", ""
    cands = [c.code for c in result.candidates] if result else []
    if opencv_read and plausible and result:
        ok, why, _ = verify_read(opencv_read, session.sources(), known_codes, margin)
        out["opencv"]["verdict"] = "verified" if ok else why
        if ok:
            status, code, path = "read", opencv_read, "opencv-verified"
            cands = [opencv_read] + [c for c in cands if c != opencv_read]
        elif why == "not in your list":
            # The list defines what can exist here; a read outside it is not
            # this code. ARPI's own answer stands, and nothing is disputed.
            note = "OpenCV read {}, not in your list; ignored".format(opencv_read)
        else:
            status, path = "disputed", "arpi-over-opencv"
            note = "OpenCV read {}; {}".format(opencv_read, why)
            if opencv_read not in cands:
                cands.insert(min(1, len(cands)), opencv_read)
    elif opencv_read and not plausible:
        note = "OpenCV read {} ({}), not EAN-13 family; ignored".format(
            opencv_read, out["opencv"]["type"])
    if status is None:
        status = result.status if result else "none"
        if result and status in ("read", "unique-in-list"):
            code = result.candidates[0].code

    cmap = reading.cmap
    out.update({
        "status": status, "code": code, "path": path, "note": note,
        "message": result.message if result else "",
        "candidates": [
            {"code": c.code, "share": c.share, "prefix": c.prefix,
             "symbology": c.symbology, "support": c.support}
            for c in (result.candidates if result else [])][:6],
        "candidate_order": cands[:6],
        "live_count": result.live_count if result else 0,
        "per_digit": result.per_digit if result else [],
        "ledger": session.ledger(),
        "grid": _grid_lines(reading),
        "modules": {"p_bar": [round(float(x), 3) for x in cmap.p_bar],
                    "confidence": [round(float(x), 3) for x in cmap.confidence]},
        "printed_digits": frame.digits.to_dict() if frame.digits else None,
        "diagnostics": cmap.diagnostics,
        "frames": max((s.detail.get("frames", 1) for s in session.sources()), default=1),
        "state": session.state(),
    })
    out["timings"].update({"arpi_s": round(t2 - t1, 3),
                           "total_s": round(time.time() - t0, 3)})
    return out
