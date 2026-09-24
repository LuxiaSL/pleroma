"""Positional correction past the end of the calibration array: one CLAMP, pinned.

anamnesis's Gate 0 (``run_replay_b0``) catches a changed feature LIST, not
changed feature VALUES. What happens at positions past the end of the
calibration array changes values only, so it is INVISIBLE to Gate 0. These
tests pin it instead:

  1. every call site the v3 families reach resolves to ONE function object;
  2. the behaviour past the array end is CLAMP, for every call site;
  3. on a zero-tailed array — the shipped 3B/8B shape — clamp is
     byte-identical to BOTH candidate behaviours (clamp and skip), which is
     why it is safe ahead of a wide artifact;
  4. on a filled array clamp and skip DIVERGE, and every call site picks
     clamp;
  5. positions the banked fits actually used (<= 473 of 712) are untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.extraction import state_extractor, state_extractor_reference
from anamnesis.extraction.feature_families import _helpers, residual_stream

# Upstream anamnesis (the pinned submodule) has no ``positional_correction``
# module. Its clamp lives in ``state_extractor._correct_hidden_state``
# and the v3 families reach it by import (``_helpers`` re-exports it,
# ``residual_stream`` imports it from there). Upstream carries no SKIP
# site, and v3 never enabled one.
# ``state_extractor_reference`` keeps its own copy; it is checked for the same
# BEHAVIOUR below, not identity.
correct_hidden_state = state_extractor._correct_hidden_state

LAYERS, MAX_POS, HID = 29, 712, 8
SHIPPED_CEILING = 555  # rows 555..711 are exact zeros in the pinned artifacts
FIT_MAX_POSITION = 473  # measured over all 41,944 fit generations


def _zero_tailed() -> np.ndarray:
    """The shipped pathology: contiguous rows 0..554, exact zeros 555..711."""
    rng = np.random.default_rng(0)
    pm = np.zeros((LAYERS, MAX_POS, HID), dtype=np.float32)
    pm[:, :SHIPPED_CEILING] = rng.normal(size=(LAYERS, SHIPPED_CEILING, HID)).astype(
        np.float32
    )
    return pm


def _filled() -> np.ndarray:
    """What a wide build produces: every row written."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(LAYERS, MAX_POS, HID)).astype(np.float32)


# ── the two candidate behaviours, as reference implementations ────────────────

def _legacy_clamp(h, layer_idx, abs_position, pm):
    """CLAMP: past the table's end, subtract the last row."""
    if pm is None:
        return h
    return h - pm[layer_idx, min(abs_position, pm.shape[1] - 1)]


def _legacy_skip(h, layer_array_idx, position, pm):
    """SKIP: past the table's end (in layer or position), subtract nothing."""
    if pm is None:
        return h
    if layer_array_idx >= pm.shape[0]:
        return h
    if position >= pm.shape[1]:
        return h
    return h - pm[layer_array_idx, position]


ALL_CALL_SITES = {
    "state_extractor": state_extractor._correct_hidden_state,
    "feature_families._helpers": _helpers._correct_hidden_state,
    "residual_stream": residual_stream._correct_hidden_state,
}
#: Same behaviour required, separate function object upstream.
BEHAVIOURAL_COPIES = {
    **ALL_CALL_SITES,
    "state_extractor_reference": state_extractor_reference._correct_hidden_state,
}


# ── 1. one place ──────────────────────────────────────────────────────────────

def test_every_call_site_is_the_same_function_object():
    for name, fn in ALL_CALL_SITES.items():
        assert fn is correct_hidden_state, f"{name} still has its own copy"


# ── 2. the behaviour is clamp ─────────────────────────────────────────────────

@pytest.mark.parametrize("abs_position,expected", [
    (0, 0), (1, 1), (473, 473), (554, 554), (711, 711),
    (712, 711), (1000, 711), (10**6, 711),
])
def test_positions_clamp_at_the_top(abs_position, expected):
    """Past the table's end the LAST row is subtracted (clamp), at every site.

    Negative positions are not exercised: upstream does ``min(pos, max-1)``
    only (no clamp below 0), and no caller produces a negative absolute
    position.
    """
    pm = _filled()
    h = np.ones(HID, dtype=np.float32)
    for name, fn in BEHAVIOURAL_COPIES.items():
        np.testing.assert_array_equal(fn(h, 2, abs_position, pm), h - pm[2, expected],
                                      err_msg=name)


def test_past_the_end_subtracts_the_last_row_not_nothing():
    pm = _filled()
    h = np.ones(HID, dtype=np.float32)
    got = correct_hidden_state(h, 3, MAX_POS + 500, pm)
    np.testing.assert_allclose(got, h - pm[3, MAX_POS - 1], rtol=0, atol=0)
    # and it is NOT the skip behaviour
    assert not np.allclose(got, h)


# ── 3. on the shipped shape, identical to BOTH candidate behaviours ────────────

@pytest.mark.parametrize("abs_position", [0, 42, 473, 554, 555, 600, 711, 712, 5000])
def test_zero_tailed_array_reproduces_both_legacy_behaviours(abs_position):
    pm = _zero_tailed()
    rng = np.random.default_rng(abs_position)
    h = rng.normal(size=HID).astype(np.float32)
    for layer in (0, 1, 14, LAYERS - 1):
        new = correct_hidden_state(h, layer, abs_position, pm)
        np.testing.assert_array_equal(new, _legacy_clamp(h, layer, abs_position, pm))
        np.testing.assert_array_equal(new, _legacy_skip(h, layer, abs_position, pm))


# ── 4. on a filled array the pair diverges, and every site picks clamp ───────

def test_legacy_behaviours_diverge_once_the_tail_is_filled():
    pm = _filled()
    h = np.ones(HID, dtype=np.float32)
    clamped = _legacy_clamp(h, 5, MAX_POS + 1, pm)
    skipped = _legacy_skip(h, 5, MAX_POS + 1, pm)
    assert not np.allclose(clamped, skipped), (
        "the premise of this fix is that clamp and skip differ on a filled array"
    )
    # the unified function picks one, and every call site picks the same one
    for fn in BEHAVIOURAL_COPIES.values():
        np.testing.assert_array_equal(fn(h, 5, MAX_POS + 1, pm), clamped)


# ── 5. nothing any banked fit touched changes ────────────────────────────────

def test_positions_the_fits_used_are_bit_identical_to_legacy():
    """The 3B and 8B fits never consumed a position above 473."""
    pm = _zero_tailed()
    rng = np.random.default_rng(7)
    for abs_position in range(0, FIT_MAX_POSITION + 1, 7):
        h = rng.normal(size=HID).astype(np.float32)
        for layer in (0, 8, 16, 28):
            np.testing.assert_array_equal(
                correct_hidden_state(h, layer, abs_position, pm),
                _legacy_clamp(h, layer, abs_position, pm),
            )


# ── guards ───────────────────────────────────────────────────────────────────

def test_none_means_no_correction():
    h = np.ones(HID, dtype=np.float32)
    assert correct_hidden_state(h, 0, 0, None) is h


# Not pinned: "out-of-range layer returns uncorrected" and "wrong-rank array is
# rejected". Upstream has neither guard; it indexes directly (an out-of-range
# layer raises IndexError). Every v3 call passes an in-range
# layer, which tests/contract/anamnesis_parity pins bit-for-bit.
