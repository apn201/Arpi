"""Module width from visible bars, half-hidden symbols, and the cascade."""
import cv2
import numpy as np

from arpi import cascade, ean13, synth, vision

REAL_DETECTOR = cv2.barcode.BarcodeDetector


def fake_detector(code, kind):
    """OpenCV's real detection, with its decode replaced."""
    class Fake:
        def detectAndDecodeWithType(self, gray):
            ok, dec, kinds, pts = REAL_DETECTOR().detectAndDecodeWithType(gray)
            return ok, [code for _ in dec], [kind for _ in dec], pts
    return Fake


def test_module_width_from_runs():
    """A rendered row at 5 px a module: the runs must say about 5."""
    bits = ean13.encode("6410405060457")
    row = np.repeat([0 if b == "1" else 255 for b in bits], 5).astype(float)
    row = np.r_[np.full(60, 255.0), row, np.full(60, 255.0)]
    b = vision.module_from_runs(row)
    assert b is not None and abs(b - 5.0) < 0.3


def test_module_width_survives_hiding_half_the_symbol():
    bits = ean13.encode("6410405060457")
    row = np.repeat([0 if b == "1" else 255 for b in bits], 5).astype(float)
    row[len(row) // 2:] = 235            # tape: low contrast, no bars
    b = vision.module_from_runs(np.r_[np.full(60, 255.0), row])
    assert b is not None and abs(b - 5.0) < 0.4


def test_taped_symbol_keeps_its_scale():
    """Tape over the right 40%: the localised box stops at the tape. The grid
    must still come out at the true module width, not squeezed into the
    visible part, which is what read 41 modules wrong on the physical set."""
    rng = np.random.default_rng(21)
    label = synth.render(synth.random_code(rng), module_px=3.0)
    x = int(label.x0 + 0.6 * ean13.MODULES * label.module_px)
    tape = label.image.astype(float)
    tape[:, x:] = 225 + 0.1 * (tape[:, x:] - 225)      # bars faint under tape
    label.image = tape.astype(np.uint8)
    img = synth.photograph(label, rng, tilt=0.05, rotate=5, blur=0.5, noise=3, jpeg=90)
    reading = vision.read_frame(img)[0]
    a, b, k = reading.grid
    # Rectified scale is unknown here, so compare against the visible bars'
    # own run lengths in the same crop.
    runs = vision.module_from_runs(np.median(
        reading.rect[reading.bars[0]:reading.bars[1]], axis=0))
    assert runs is not None and abs(b / runs - 1) < 0.1


def test_cascade_never_asserts_a_disputed_read(monkeypatch):
    """If OpenCV's read disagrees with the bars, nothing is asserted and
    OpenCV's read stays among the candidates."""
    img, label = synth.sample(np.random.default_rng(4), "clean", 0)
    wrong = ean13.complete(label.code[:11] + str((int(label.code[11]) + 3) % 10))

    monkeypatch.setattr(cascade.cv2.barcode, "BarcodeDetector",
                        fake_detector(wrong, "EAN_13"))
    syms = [s for s in cascade.scan(img, verify="full") if s.opencv_read]
    assert syms
    s = syms[0]
    assert s.status == "disputed" and s.code is None
    assert wrong in s.candidates and label.code == s.candidates[0]


def test_cascade_rejects_ean8_inside_ean13_symbology(monkeypatch):
    img, label = synth.sample(np.random.default_rng(5), "clean", 0)
    monkeypatch.setattr(cascade.cv2.barcode, "BarcodeDetector",
                        fake_detector("12345670", "EAN_8"))
    syms = [s for s in cascade.scan(img, verify="symbology") if s.opencv_read]
    assert syms and syms[0].path == "arpi-over-opencv"
    assert syms[0].code != "12345670"


def test_wrong_read_is_not_verified_when_the_list_misses_the_code():
    """The bug found on the phone page: with a list that does not contain the
    code, every list code fits badly, and a wrong OpenCV read used to beat
    them and pass as verified. It must be rejected, and with no list it must
    lose to the code the bars actually show."""
    from arpi.confmap import ConfidenceMap
    from arpi.evidence import from_map
    truth = "0350038585148"
    wrong = "0350038589900"
    bars = [from_map(ConfidenceMap.from_bits(ean13.encode(truth), 0.9))]
    unrelated_list = ["6978805497025", "9120547365484"]
    ok, why, _ = cascade.verify_read(wrong, bars, unrelated_list)
    assert not ok and why == "not in your list"
    ok, why, best = cascade.verify_read(wrong, bars, None)
    assert not ok and best == truth
    ok, _, _ = cascade.verify_read(truth, bars, None)
    assert ok
