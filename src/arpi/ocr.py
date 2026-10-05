"""The digits printed under the bars, read as a second, independent witness.

Model: OpenCV Zoo's CRNN text recogniser (EN, 2021sep, Apache 2.0), run with
cv2.dnn. `python tools/fetch_models.py` downloads it, pinned by hash.

What is unusual here is that we already know where every digit is. The bar
grid says exactly which columns belong to digit 7, so instead of asking a text
detector to find text and a recogniser to guess how many characters it holds,
the reader cuts each digit's cell out by geometry and lays the cells out as
two clean strips:

    strip 1   leading digit + the six left digits      (7 slots)
    strip 2   the six right digits                     (6 slots)

The guard bars that sit between those digits on the label never reach the
network. The CRNN emits 24 time steps per strip, each a distribution over
blank + 36 characters; the steps that fall inside a slot are pooled into that
slot's distribution over the 10 digits. Letters the model confuses with digits
(o, l, s, ...) are folded in, because the field is known to be numeric. What
is left over - other letters, mostly blank - is "could not read", and makes
the evidence for that position flat rather than wrong.

A slot whose digit was torn off reads as blank or junk and contributes
nothing. A slot that reads clearly contributes a lot, but never enough to
overrule clear bars: see `evidence.from_digit_probs`.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from . import ean13
from .vision import _positions

MODEL_NAME = "text_recognition_CRNN_EN_2021sep.onnx"
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
# Letters a general-purpose recogniser reads where a digit was printed.
FOLD = {"o": 0, "d": 0, "q": 9, "g": 9, "l": 1, "i": 1, "j": 1, "t": 7,
        "z": 2, "s": 5, "b": 6, "a": 4}
INPUT = (100, 32)
STEPS = 24

_NET = {}


def model_path():
    here = Path(__file__).resolve().parent
    candidates = [Path(os.environ["ARPI_MODELS"]) / MODEL_NAME] \
        if os.environ.get("ARPI_MODELS") else []
    candidates += [here.parent.parent / "models" / MODEL_NAME,   # repo
                   here.parent / "models" / MODEL_NAME]          # lambda bundle
    return next((p for p in candidates if p.exists()), None)


def available():
    return model_path() is not None


def _net():
    if "net" not in _NET:
        path = model_path()
        if path is None:
            raise FileNotFoundError("text model missing: python tools/fetch_models.py")
        _NET["net"] = cv2.dnn.readNet(str(path))
    return _NET["net"]


# ---- geometry -----------------------------------------------------------------

# Slot extents in module units, relative to module 0 of the symbol. The
# leading digit is printed in the quiet zone, left of the start guard.
SLOTS = [(-9.0, -2.0)] + [(ean13.digit_start(p), ean13.digit_start(p) + 7)
                          for p in range(12)]


def text_band(rect, grid, bars):
    """Rows of the printed digits: the ink just under the bars. Returns
    (top, bottom) or None when there is no ink there at all."""
    a, b, k = grid
    H, W = rect.shape
    top = bars[1]
    limit = min(H, int(top + 14 * b))
    if limit - top < 4:
        return None
    x0 = int(max(0, _positions(a, b, k, np.array([3.0]))[0]))
    x1 = int(min(W, _positions(a, b, k, np.array([92.0]))[0]))
    if x1 - x0 < 10:
        return None
    region = rect[top:limit, x0:x1]
    # Edge density, not darkness. "Darker than the paper" also describes the
    # table the label is lying on, and the first version of this cut the
    # digits off at the label edge and fed the network a strip of background.
    edges = np.abs(np.diff(region, axis=1)).mean(1)
    bar_edges = np.abs(np.diff(rect[bars[0]:bars[1], x0:x1], axis=1)).mean()
    if bar_edges <= 0:
        return None
    rows = edges / bar_edges
    thresh = max(0.08, 0.35 * rows.max())
    hot = rows > thresh
    if not hot.any():
        return None
    # The run of rows with the most edge energy. Not the first run: the bar
    # ends leave a sliver of hot rows just above the digits, and taking the
    # first run read that sliver and nothing else.
    runs, start = [], None
    for i, h in enumerate(np.r_[hot, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            runs.append((rows[start:i].sum(), start, i))
            start = None
    _, t0, t1 = max(runs)
    if t1 - t0 < 3:
        return None
    pad = max(1, int(0.12 * (t1 - t0)))
    return max(top, top + t0 - pad), min(H, top + t1 + pad)


def _rows_in(rect, x0, x1, y0, y1, bar_edges, min_rows=0):
    """Rows of ink in one column range: the run with the most horizontal edge
    energy, padded. None if nothing there looks like text.

    A run that starts on the window's first row and is shorter than
    `min_rows` is the bottom of the bars leaking in on a tilted label, not
    text, and is skipped. Bars carry more edges than digits, so letting them
    compete is how the first per-cell version read bars as text."""
    region = rect[y0:y1, max(0, int(x0)):int(x1)]
    if region.shape[0] < 4 or region.shape[1] < 3:
        return None
    rows = np.abs(np.diff(region, axis=1)).mean(1) / bar_edges
    hot = rows > max(0.08, 0.3 * rows.max())
    runs, start = [], None
    for i, h in enumerate(np.r_[hot, False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if not (start == 0 and i - start < min_rows):
                runs.append((rows[start:i].sum(), start, i))
            start = None
    if not runs:
        return None
    _, t0, t1 = max(runs)
    if t1 - t0 < 3:
        return None
    pad = max(1, int(0.15 * (t1 - t0)))
    return y0 + max(0, t0 - pad), y0 + min(region.shape[0], t1 + pad)


# Which warp segment each text slot moves with: the leading digit sits by the
# start guard, digits 2-7 and 8-13 under their own segments (see
# vision.SEGMENTS: 0 start guard, 1-6 left digits, 7 centre, 8-13 right).
SLOT_SEGMENT = [0] + list(range(1, 7)) + list(range(8, 14))


def _cells(rect, grid, band, bars, height=32, warp=None):
    """Each slot's cell, resampled to a common height.

    The rows are found per cell, not once for the whole strip: a label tilted
    away from the camera puts the digits on a slope in the crop, and one
    shared band cut the low end's digits in half while catching bar ends at
    the high end."""
    a, b, k = grid
    H, W = rect.shape
    y0 = bars[1]
    y1 = min(H, max(band[1], bars[1]) + int(5 * b))
    bar_edges = max(np.abs(np.diff(rect[bars[0]:bars[1]], axis=1)).mean(), 1e-6)
    out = []
    for slot, (lo, hi) in enumerate(SLOTS):
        # On a wrinkled or torn label the printed digit moved with its bars.
        shift = float(warp[SLOT_SEGMENT[slot]]) if warp else 0.0
        xl = _positions(a, b, k, np.array([lo + shift]))[0]
        xr = _positions(a, b, k, np.array([hi + shift]))[0]
        if xr - xl < 2 or xr <= 0 or xl >= W:
            out.append(None)
            continue
        rows = _rows_in(rect, xl, xr, y0, y1, bar_edges,
                        min_rows=int(2.5 * b)) or band
        top, bottom = rows
        src = np.float32([[xl, top], [xr, top], [xr, bottom]])
        w = int(round(height * (xr - xl) / max(bottom - top, 1)))
        dst = np.float32([[0, 0], [w, 0], [w, height]])
        M = cv2.getAffineTransform(src, dst)
        out.append(cv2.warpAffine(rect, M, (max(w, 4), height),
                                  flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
    return out


def _strip(cells, gap=4):
    """Cells side by side on paper-coloured background. Returns the strip and
    each slot's (start, end) column."""
    paper = float(np.median(np.concatenate([c.ravel() for c in cells if c is not None])))
    h = cells[0].shape[0] if cells[0] is not None else 32
    widths = [c.shape[1] if c is not None else h // 2 for c in cells]
    W = sum(widths) + gap * (len(cells) + 1)
    strip = np.full((h, W), paper, np.float32)
    spans, x = [], gap
    for c, w in zip(cells, widths):
        if c is not None:
            strip[:, x:x + w] = c
        spans.append((x, x + w))
        x += w + gap
    return strip, spans


def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _read_strip(strip, spans):
    """Per-slot distributions over the 10 digits, plus how readable each slot
    was (0..1), plus the raw greedy text for display."""
    img = np.clip(strip, 0, 255).astype(np.uint8)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, INPUT, interpolation=cv2.INTER_AREA),
                                 mean=127.5, scalefactor=1 / 127.5)
    net = _net()
    net.setInput(blob)
    probs = _softmax(net.forward()[:, 0, :])               # (24, 37)
    T, W = probs.shape[0], strip.shape[1]
    centres = (np.arange(T) + 0.5) * W / T

    greedy = "".join("-" if i == 0 else CHARSET[i - 1] for i in probs.argmax(1))
    dists, readable = [], []
    for x0, x1 in spans:
        # Steps whose centre falls in the slot, widened by half a step so a
        # narrow slot always gets at least one.
        half = 0.5 * W / T
        sel = (centres >= x0 - half) & (centres < x1 + half)
        if not sel.any():
            dists.append(np.full(10, 0.1))
            readable.append(0.0)
            continue
        p = probs[sel]
        nonblank = 1 - p[:, 0]
        chars = p[:, 1:] / np.maximum(nonblank[:, None], 1e-6)
        wsum = nonblank.sum()
        if wsum < 0.3:                      # nothing printed here, or erased
            dists.append(np.full(10, 0.1))
            readable.append(0.0)
            continue
        c = (chars * nonblank[:, None]).sum(0) / wsum
        digit = c[:10].copy()
        for letter, d in FOLD.items():
            digit[d] += c[10 + CHARSET[10:].index(letter)]
        conf = float(digit.sum())           # mass that looks like any digit
        dists.append(digit / max(digit.sum(), 1e-9))
        readable.append(conf * min(1.0, wsum))
    return dists, readable, greedy


class DigitRead:
    """What the printed text says, per EAN position 0..12."""

    def __init__(self, probs, readable, text, band, strips):
        self.probs = probs            # (13, 10)
        self.readable = readable      # (13,) 0..1
        self.text = text              # best guess per position, '?' if unreadable
        self.band = band              # rows in the crop
        self.strips = strips          # for the HUD

    def to_dict(self):
        return {"text": self.text,
                "readable": [round(float(r), 3) for r in self.readable],
                "best": [int(np.argmax(p)) for p in self.probs],
                "confidence": [round(float(p.max()), 3) for p in self.probs]}


def read_digits(reading, min_readable=0.35):
    """Read the printed digits of a reading that is already the right way up
    (see Reading.flip). Returns DigitRead, or None when there is no text."""
    band = text_band(reading.rect, reading.grid, reading.bars)
    if band is None or band[1] - band[0] < 4:
        return None
    cells = _cells(reading.rect, reading.grid, band, reading.bars,
                   warp=reading.cmap.diagnostics.get("warp"))
    if all(c is None for c in cells):
        return None
    s1, sp1 = _strip(cells[:7])
    s2, sp2 = _strip(cells[7:])
    d1, r1, g1 = _read_strip(s1, sp1)
    d2, r2, g2 = _read_strip(s2, sp2)
    probs = np.array(d1 + d2)
    readable = np.array(r1 + r2)
    text = "".join(str(int(np.argmax(p))) if r >= min_readable else "?"
                   for p, r in zip(probs, readable))
    return DigitRead(probs, readable, text, band, {"left": g1, "right": g2})
