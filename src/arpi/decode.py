"""One image in, one answer out. The local pipeline the Lambda will wrap.

    python -m arpi photo.jpg
    python -m arpi photo.jpg --known codes.csv
    python -m arpi frame1.jpg frame2.jpg frame3.jpg      # fused
"""
import csv
from dataclasses import asdict

import cv2

from .session import Session


def load_known(path):
    """A CSV or a newline-delimited list. Any cell that looks like a 12 or 13
    digit code counts. Deliberately dumb: no ERP connector, no schema."""
    codes = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            for cell in row:
                digits = "".join(ch for ch in cell if ch.isdigit())
                if len(digits) in (12, 13):
                    codes.append(digits)
    return codes


def _gray(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("could not read image")
        return img
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def scan(images, known_codes=None, **session_kwargs):
    """Any number of frames of the same label. Returns the Session."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    s = Session(known_codes=known_codes, **session_kwargs)
    for im in images:
        s.add(_gray(im))
    return s


def decode(image, known_codes=None, **session_kwargs):
    """Returns (Result, ConfidenceMap) for one image, or (None, None)."""
    s = scan(image, known_codes, **session_kwargs)
    if s.latest is None:
        return None, None
    return s.result(), s.latest.reading.cmap


def as_dict(result, cmap, session=None):
    if result is None:
        return {"status": "no-symbol", "message": "No barcode found in the image.",
                "candidates": []}
    d = asdict(result)
    d["diagnostics"] = cmap.diagnostics if cmap else {}
    d["digit_quality"] = cmap.digit_quality() if cmap else []
    if session is not None:
        d["ledger"] = session.ledger()
        f = session.latest
        if f is not None and f.digits is not None:
            d["printed_digits"] = f.digits.to_dict()
        d["frames"] = len(session.frames)
    return d
