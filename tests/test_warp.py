"""The piecewise warp: segments shift independently, smoothly or in one jump,
and never on a label that did not move."""
import numpy as np

from arpi import synth, vision


def test_viterbi_finds_a_single_jump():
    """Segments 0-7 fit best unshifted, 8-14 fit best at +2 modules: a torn
    piece moved right. The path must be one clean step."""
    S = len(vision.SHIFTS)
    scores = np.zeros((15, S))
    zero = int(np.argmin(np.abs(vision.SHIFTS)))
    plus2 = int(np.argmin(np.abs(vision.SHIFTS - 2.0)))
    scores[:8, zero] = 7
    scores[8:, plus2] = 7
    shifts, _ = vision._viterbi(scores)
    assert list(shifts[:8]) == [0.0] * 8
    assert list(shifts[8:]) == [2.0] * 7


def test_viterbi_prefers_no_shift_without_evidence():
    shifts, _ = vision._viterbi(np.zeros((15, len(vision.SHIFTS))))
    assert not shifts.any()


def test_describe_warp_names_the_break():
    shifts = np.r_[np.zeros(7), np.full(8, 2.5)]
    notes = vision.describe_warp(shifts)
    assert notes == ["break between digit 7 and centre guard, +2.5 modules"]


def test_clean_labels_never_break_or_move_far():
    """A clean label may take up a fraction of a module of leftover
    perspective. It may never be given a tear, or slide half a module."""
    for seed in range(6):
        img, _ = synth.sample(np.random.default_rng(seed), "clean", 0)
        r = vision.read_frame(img)[0]
        warp = np.array(r.cmap.diagnostics["warp"])
        assert np.abs(warp).max() <= 0.5, seed
        assert not any("break" in n for n in r.cmap.diagnostics["warp_notes"]), seed


def test_flip_mirrors_the_warp():
    img, _ = synth.sample(np.random.default_rng(3), "displaced", 0.4)
    r = vision.read_frame(img)[0]
    before = list(r.cmap.diagnostics["warp"])
    r.flip()
    assert r.cmap.diagnostics["warp"] == [-x for x in before[::-1]]
