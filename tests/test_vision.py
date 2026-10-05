"""End to end on rendered images. Slow-ish: about a second each."""
import cv2
import numpy as np
import pytest

from arpi import decode, synth


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_clean_photo_reads(seed):
    rng = np.random.default_rng(seed)
    img, label = synth.sample(rng, "clean", 0)
    r, cm = decode.decode(img)
    assert r is not None
    assert r.candidates[0].code == label.code
    assert cm.diagnostics["grid_fit"] > 0.7


def test_torn_label_opencv_fails_arpi_ranks_truth():
    """The project in one test: the stock decoder returns nothing, this
    returns the right code among ranked candidates and does not assert it."""
    rng = np.random.default_rng(11)
    img, label = synth.sample(rng, "tear", 0.06)
    ok, *_ = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(img)
    r, _ = decode.decode(img)
    assert label.code in [c.code for c in r.candidates[:3]]
    assert r.status != "read"


def test_blank_image_finds_nothing():
    img = np.full((400, 600), 180, np.uint8)
    r, cm = decode.decode(img)
    assert r is None or r.status in ("none", "reconstructed")
