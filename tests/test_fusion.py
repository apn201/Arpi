"""Evidence fusion, the text reader and multi-frame sessions."""
import numpy as np
import pytest

from arpi import candidates, ean13, ocr, synth
from arpi.confmap import ConfidenceMap
from arpi.evidence import from_digit_probs, from_map, orientation_score
from arpi.session import Session

CODE = "6410405060457"
needs_model = pytest.mark.skipif(not ocr.available(),
                                 reason="python tools/fetch_models.py")


def erased(code, positions):
    cm = ConfidenceMap.from_bits(ean13.encode(code))
    c = list(cm.confidence)
    for pos in positions:
        s = ean13.digit_start(pos)
        c[s:s + 7] = [0.0] * 7
    return ConfidenceMap(list(cm.p_bar), c)


def text_saying(code, readable=1.0, sure=0.97):
    probs = np.full((13, 10), (1 - sure) / 9)
    for i, ch in enumerate(code):
        probs[i, int(ch)] = sure
    return from_digit_probs(probs, np.full(13, readable))


def test_text_settles_what_the_bars_leave_open():
    cm = erased(CODE, [0, 2, 8])
    bars_only = candidates.solve(cm)
    assert bars_only.live_count > 1
    fused = candidates.solve([from_map(cm), text_saying(CODE)])
    assert fused.candidates[0].code == CODE
    assert fused.status == "reconstructed"          # never "read"


def test_text_cannot_outvote_clear_bars():
    """A confidently wrong text reader against a clean symbol: the bars win.
    This is the ceiling in evidence.from_digit_probs, tested."""
    wrong = ean13.complete("6410405060" + "99")
    r = candidates.solve([from_map(ConfidenceMap.from_bits(ean13.encode(CODE))),
                          text_saying(wrong, sure=0.999)])
    assert r.candidates[0].code == CODE


def test_unreadable_text_changes_nothing():
    cm = erased(CODE, [2, 8])
    flat = from_digit_probs(np.full((13, 10), 0.1), np.zeros(13))
    a = candidates.solve(cm)
    b = candidates.solve([from_map(cm), flat])
    assert [c.code for c in a.candidates] == [c.code for c in b.candidates]


def test_support_names_each_source():
    r = candidates.solve([from_map(erased(CODE, [2])), text_saying(CODE)])
    assert set(r.candidates[0].support) == {"bars", "text"}
    assert r.candidates[0].support["text"] == 0.0     # text's favourite


def test_orientation_from_parity():
    ev = from_map(ConfidenceMap.from_bits(ean13.encode(CODE)))
    rev = from_map(ConfidenceMap.from_bits(ean13.encode(CODE)).reversed())
    assert orientation_score(ev) > orientation_score(rev)


@needs_model
def test_printed_digits_read_on_clean_label():
    from arpi import vision
    img, label = synth.sample(np.random.default_rng(14), "clean", 0)
    reading = vision.read_frame(img)[0]
    d = ocr.read_digits(reading)
    assert d is not None
    right = sum(a == b for a, b in zip(d.text, label.code))
    assert right >= 10


@needs_model
def test_session_fuses_frames():
    rng = np.random.default_rng(3)
    label = synth.render(synth.random_code(rng), module_px=3.0)
    synth.occlusion(label, rng, 0.12)
    s = Session()
    for _ in range(3):
        s.add(synth.photograph(label, rng, tilt=0.08, rotate=8, blur=0.5,
                               noise=3, jpeg=90))
    r = s.result()
    assert len(s.frames) == 3
    assert label.code in [c.code for c in r.candidates[:3]]
    assert set(s.ledger()) >= {"bars"}
