"""The symbology and the candidate engine, at module level. No pixels."""
import numpy as np
import pytest

from arpi import candidates, ean13, gs1
from arpi.confmap import ConfidenceMap

CODE = "6410405060457"


def damaged(code, positions, confidence=1.0):
    """A perfect read with whole encoded digits erased."""
    cm = ConfidenceMap.from_bits(ean13.encode(code), confidence)
    c = list(cm.confidence)
    for pos in positions:
        s = ean13.digit_start(pos)
        c[s:s + 7] = [0.0] * 7
    return ConfidenceMap(list(cm.p_bar), c)


# ---- symbology ---------------------------------------------------------------

@pytest.mark.parametrize("code", ["4006381333931", "5901234123457",
                                  "0012345678905", "9780306406157"])
def test_published_codes_validate(code):
    assert ean13.is_valid(code)
    assert len(ean13.encode(code)) == 95


def test_upc_a_is_ean13_with_leading_zero():
    assert ean13.normalise("012345678905") == "0012345678905"
    assert ean13.symbology("0012345678905") == "UPC-A"


def test_known_modules_hold_for_every_code():
    known = ean13.known_modules()
    assert len(known) == 35
    rng = np.random.default_rng(0)
    for _ in range(200):
        code = ean13.complete("".join(str(d) for d in rng.integers(0, 10, 12)))
        bits = ean13.encode(code)
        assert all(int(bits[i]) == b for i, b in known.items())


def test_ten_parity_patterns_of_sixty_four():
    assert sum(1 for d in ean13.LEADING_FROM_BITS if d >= 0) == 10


def test_gs1_gaps():
    assert gs1.assignment("6410405060457") == "Finland"
    assert not gs1.is_assigned("1400000000000")      # 140-199 unassigned
    assert not gs1.is_assigned("9600000000000")      # GTIN-8 range


# ---- engine --------------------------------------------------------------------

def test_clean_read_is_read():
    r = candidates.solve(ConfidenceMap.from_bits(ean13.encode(CODE)))
    assert r.status == "read"
    assert [c.code for c in r.candidates] == [CODE]


def test_two_destroyed_digits_recover_but_never_assert():
    r = candidates.solve(damaged(CODE, [2, 8]))
    assert r.status == "reconstructed"
    assert CODE in [c.code for c in r.candidates]


def test_one_destroyed_digit_is_unique_but_still_reconstructed():
    r = candidates.solve(damaged(CODE, [9]))
    assert r.status == "reconstructed"          # the safety rule, not "read"
    assert r.candidates[0].code == CODE
    assert r.live_count == 1


def test_known_list_resolves_three_destroyed_digits():
    cm = damaged(CODE, [0, 2, 8])
    free = candidates.solve(cm)
    assert free.live_count > 1
    r = candidates.solve(cm, known_codes=[CODE, "4006381333931", "5901234123457"])
    assert r.status == "unique-in-list"
    assert r.candidates[0].code == CODE


def test_known_list_neighbours_are_not_asserted():
    """Two codes in the list that differ only in a destroyed digit must both
    come back, and neither may be asserted. The twin's check digit differs
    too, so that has to be destroyed as well or the engine rightly rejects it."""
    cm = damaged(CODE, [10, 11])
    twin = ean13.complete(CODE[:11] + str((int(CODE[11]) + 1) % 10))
    r = candidates.solve(cm, known_codes=[CODE, twin])
    assert r.status == "reconstructed"
    assert {c.code for c in r.candidates} >= {CODE, twin}


def test_reversed_scan_is_recognised_by_parity():
    cm = damaged(CODE, [4]).reversed()
    r = candidates.solve_either_way(cm)
    assert r.candidates[0].code == CODE


def test_nothing_survives_reports_per_digit():
    cm = ConfidenceMap(p_bar=[0.0] * 95, confidence=[1.0] * 95)
    r = candidates.solve(cm, known_codes=[])
    assert r.status == "none"
    assert len(r.per_digit) == 13          # leading digit included


def test_confidence_map_round_trip():
    cm = damaged(CODE, [1])
    back = ConfidenceMap.from_dict(cm.to_dict())
    assert back.p_bar == cm.p_bar and back.confidence == cm.confidence
    assert len(cm.labels) == 95 and len(cm.digit_quality()) == 12
