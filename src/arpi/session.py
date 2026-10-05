"""One scan, from one frame or many. Where the witnesses meet.

Per frame:

    vision      localise, rectify, grid fit, per-module vote -> confidence map
    orientation from the bars alone (parity), never from the text: upside-down
                digits read as confident nonsense, which is how the first
                version of this got 200 digits wrong
    bars        evidence.from_map
    text        ocr.read_digits on the cells the grid says hold the digits

Across frames the evidence is summed in code space - per digit, per position -
so frames never need aligning pixel to pixel. A phone that moved between
frames has moved the glare too, and the modules it hid in one frame are
visible in another.

Frames of one label are not independent witnesses: the same smudge is in all
of them. Summing n frames as if they were would make the result n times as
sure of the smudge. So each source's sum is scaled by n ** -alpha. alpha = 1
is plain averaging (more frames never add certainty), alpha = 0 is full
independence. 0.5 is the starting point, and the evaluation sets it.
"""
import time

import numpy as np

from . import candidates, ocr, vision
from .evidence import Evidence, from_digit_probs, from_map, orientation_score, total


class Frame:
    def __init__(self, reading, bars, text, digits, timings):
        self.reading = reading        # vision.Reading, right way up
        self.bars = bars              # Evidence
        self.text = text              # Evidence or None
        self.digits = digits          # ocr.DigitRead or None
        self.timings = timings


class Session:
    def __init__(self, known_codes=None, use_text=True, text_weight=0.6,
                 alpha=0.5, max_frames=30):
        self.known_codes = known_codes
        self.use_text = use_text and ocr.available()
        self.text_weight = text_weight
        self.alpha = alpha
        self.max_frames = max_frames
        self.frames = []
        self.misses = 0

    # ---- per frame ---------------------------------------------------------
    def add(self, gray):
        t0 = time.time()
        readings = vision.read_frame(gray)
        t1 = time.time()
        best = None
        for r in readings:
            fwd = from_map(r.cmap)
            rev = from_map(r.cmap.reversed())
            sf, sr = orientation_score(fwd), orientation_score(rev)
            score = max(sf, sr)
            if best is None or score > best[0]:
                best = (score, r, sr > sf)
        if best is None:
            self.misses += 1
            return None
        _, reading, backwards = best
        return self.add_reading(reading, backwards, t0, t1)

    def add_reading(self, reading, backwards=None, t0=None, t1=None):
        """One already-localised symbol. Used directly when a frame holds
        many symbols (a sheet, a shelf) and each gets its own session."""
        t0 = t0 or time.time()
        t1 = t1 or t0
        if backwards is None:
            backwards = orientation_score(from_map(reading.cmap.reversed())) >                 orientation_score(from_map(reading.cmap))
        if backwards:
            reading.flip()
        bars = from_map(reading.cmap)
        text = digits = None
        t2 = time.time()
        if self.use_text:
            digits = ocr.read_digits(reading)
            if digits is not None:
                text = from_digit_probs(digits.probs, digits.readable,
                                        weight=self.text_weight)
        t3 = time.time()
        frame = Frame(reading, bars, text, digits,
                      {"vision_s": round(t1 - t0, 3), "text_s": round(t3 - t2, 3)})
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
        return frame

    # ---- across frames -----------------------------------------------------
    def sources(self):
        out = []
        for name, attr in (("bars", "bars"), ("text", "text")):
            evs = [getattr(f, attr) for f in self.frames if getattr(f, attr) is not None]
            if not evs:
                continue
            s = total(evs).scaled(len(evs) ** -self.alpha)
            s.source = name
            s.detail = {"frames": len(evs)}
            out.append(s)
        return out

    def result(self, **kwargs):
        srcs = self.sources()
        if not srcs:
            return None
        return candidates.solve(srcs, known_codes=self.known_codes, **kwargs)

    def ledger(self):
        """What each witness believes about each position, for the HUD and the
        report. Per source, per position: best digit and its share."""
        out = {}
        for s in self.sources():
            ll = s.position_ll()
            rows = []
            for i, row in enumerate(ll):
                p = np.exp(row - row.max())
                p /= p.sum()
                d = int(np.argmax(p))
                rows.append({"position": i + 1, "digit": d,
                             "share": round(float(p[d]), 3)})
            out[s.source] = rows
        return out

    @property
    def latest(self):
        return self.frames[-1] if self.frames else None
