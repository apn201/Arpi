"""The live scanner's lock-on rules: bad frames are not evidence, and another
code is not mixed into the first one."""
import numpy as np

from arpi import live, synth


def photo(code, seed, kind="clean"):
    rng = np.random.default_rng(seed)
    label = synth.render(code, module_px=3.2)
    if kind == "tear":
        synth.tear(label, rng, 0.08)
    return synth.photograph(label, rng, tilt=0.06, rotate=6, blur=0.5, noise=3, jpeg=90)


def test_evidence_accumulates_on_the_same_code():
    r1 = live.scan(photo("6410405060457", 1, "tear"))
    r2 = live.scan(photo("6410405060457", 2, "tear"), state=r1["state"])
    assert r2["frame_quality"]["accepted"] and not r2["target_changed"]
    assert r2["frames"] == 2


def test_another_code_starts_fresh_instead_of_mixing():
    r1 = live.scan(photo("6410405060457", 3))
    r2 = live.scan(photo("4006381333931", 4), state=r1["state"])
    assert r2["target_changed"]
    assert r2["frames"] == 1
    assert r2["candidate_order"][0] == "4006381333931"


def test_unfittable_frame_is_not_counted():
    """Blurred to mush, as a moving hand does: the grid cannot fit, the
    frame is dropped, and the evidence so far is returned unchanged."""
    import cv2
    r1 = live.scan(photo("6410405060457", 5))
    smear = cv2.GaussianBlur(photo("6410405060457", 6), (0, 0), 6)
    r2 = live.scan(smear, state=r1["state"])
    if r2.get("found"):
        assert r2["frame_quality"]["accepted"] is False
        assert r2["state"] == r1["state"]


def test_stripes_are_never_mixed_into_a_code():
    """Regular stripes fit about as well as a badly damaged code, so the
    fit gate may let them through. The identity check must not."""
    r1 = live.scan(photo("6410405060457", 7))
    stripes = (np.random.default_rng(8).random((480, 640)) * 255).astype(np.uint8)
    stripes[200:280, 150:490] = np.tile((np.arange(340) // 3 % 2) * 255, (80, 1))
    r2 = live.scan(stripes, state=r1["state"])
    if r2.get("found"):
        assert (not r2["frame_quality"]["accepted"]) or r2["target_changed"]
        assert r2["frames"] <= 1
