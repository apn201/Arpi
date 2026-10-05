"""The per-module confidence map. The interface between seeing and reasoning.

The vision half produces one of these and stops. The candidate engine and the
agent consume it and never look at pixels. Keeping it that narrow is what lets
the synthetic tests drive the engine directly, and what lets the agent explain
its requests in terms a person can check ("the left half is unreadable").

Per module, two numbers, never collapsed to one:

    p_bar       how dark the module looked, 0 = clearly space, 1 = clearly bar
    confidence  how much to believe p_bar, 0 = no evidence at all, 1 = every
                scanline agreed and the contrast was good

A module painted over by a sticker reads as a confident space if nothing else
is known. Confidence is what lets the scanner say "this looked white, but the
run is seven modules long and EAN-13 never has one, so do not trust it".
"""
from dataclasses import asdict, dataclass, field
import json

import numpy as np

from . import ean13

LEGIBLE = "legible"
UNCERTAIN = "uncertain"
ILLEGIBLE = "illegible"


@dataclass
class ConfidenceMap:
    p_bar: list
    confidence: list
    symbology: str = "ean13"
    orientation: str = "forward"         # or "reversed": already flipped back
    module_px: float = 0.0               # module width in rectified pixels
    rows_used: int = 0
    # Measurements the agent reasons over. All 0..1 unless stated.
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.p_bar) != ean13.MODULES or len(self.confidence) != ean13.MODULES:
            raise ValueError("an EAN-13 map has exactly 95 modules")

    # ---- labels ------------------------------------------------------------
    @property
    def labels(self):
        out = []
        for p, c in zip(self.p_bar, self.confidence):
            if c < 0.2:
                out.append(ILLEGIBLE)
            elif c >= 0.6 and abs(p - 0.5) > 0.3:
                out.append(LEGIBLE)
            else:
                out.append(UNCERTAIN)
        return out

    def digit_quality(self):
        """Mean confidence over each encoded digit's seven modules. 12 values,
        left half first. The agent reads this to decide what to re-shoot."""
        c = np.asarray(self.confidence)
        return [float(c[ean13.digit_start(i):ean13.digit_start(i) + 7].mean())
                for i in range(12)]

    def effective(self, eps=0.02):
        """p_bar pulled toward 0.5 by lack of confidence, then kept off 0 and 1
        so that one confidently wrong module costs a fixed amount instead of
        vetoing the true answer outright."""
        p = np.asarray(self.p_bar, dtype=float)
        c = np.clip(np.asarray(self.confidence, dtype=float), 0, 1)
        q = 0.5 + c * (p - 0.5)
        return np.clip(q, eps, 1 - eps)

    # ---- io ----------------------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d["p_bar"] = [round(float(x), 4) for x in self.p_bar]
        d["confidence"] = [round(float(x), 4) for x in self.confidence]
        d["labels"] = self.labels
        return d

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        return cls(p_bar=list(d["p_bar"]), confidence=list(d["confidence"]),
                   symbology=d.get("symbology", "ean13"),
                   orientation=d.get("orientation", "forward"),
                   module_px=d.get("module_px", 0.0),
                   rows_used=d.get("rows_used", 0),
                   diagnostics=d.get("diagnostics", {}))

    @classmethod
    def from_bits(cls, bits, confidence=1.0):
        """A perfect read of a known module string. For tests."""
        p = [float(b) for b in bits]
        return cls(p_bar=p, confidence=[confidence] * len(p))

    def reversed(self):
        return ConfidenceMap(p_bar=list(self.p_bar[::-1]),
                             confidence=list(self.confidence[::-1]),
                             symbology=self.symbology,
                             orientation=("reversed" if self.orientation == "forward"
                                          else "forward"),
                             module_px=self.module_px, rows_used=self.rows_used,
                             diagnostics=dict(self.diagnostics))
