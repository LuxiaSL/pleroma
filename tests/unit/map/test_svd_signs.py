"""Canonical SVD signs: the same map gives the same code basis on any stack."""
from __future__ import annotations

import numpy as np
import pytest

from pleroma.map.svd import SIGN_CONVENTION, canonical_signs, is_canonical


def _factors(seed: int = 0, shape: tuple[int, int] = (40, 90), r: int = 8):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return u[:, :r], s[:r], vt[:r]


def test_product_unchanged_and_canonical():
    u, s, vt = _factors()
    cu, cs, cvt, rep = canonical_signs(u, s, vt)
    np.testing.assert_allclose((cu * cs) @ cvt, (u * s) @ vt, atol=1e-12)
    assert is_canonical(cvt)
    assert rep.convention == SIGN_CONVENTION


def test_arbitrary_flips_converge_to_one_orientation():
    """The failure this guards: same W, half the components sign-flipped."""
    u, s, vt = _factors(seed=3)
    flips = np.where(np.arange(s.size) % 2 == 0, -1.0, 1.0)
    a = canonical_signs(u, s, vt)
    b = canonical_signs(u * flips, s, vt * flips[:, None])
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[0], b[0])


def test_idempotent():
    u, s, vt = _factors(seed=5)
    once = canonical_signs(u, s, vt)
    twice = canonical_signs(*once[:3])
    np.testing.assert_array_equal(once[2], twice[2])
    assert twice[3].flipped == ()


def test_code_expands_to_same_lever_across_orientations():
    u, s, vt = _factors(seed=7)
    flips = np.array([-1.0, 1.0] * 4)
    _, _, vt_a, _ = canonical_signs(u, s, vt)
    _, _, vt_b, _ = canonical_signs(u * flips, s, vt * flips[:, None])
    lever = np.random.default_rng(1).standard_normal(vt.shape[1])
    code = vt_a @ lever
    np.testing.assert_allclose(code @ vt_b, code @ vt_a)


def test_ambiguous_row_is_reported():
    vt = np.array([[0.6, -0.6, 0.1], [0.9, 0.1, 0.0]])
    vt = vt / np.linalg.norm(vt, axis=1, keepdims=True)
    u = np.eye(2)
    _, _, _, rep = canonical_signs(u, np.array([2.0, 1.0]), vt)
    assert rep.ambiguous == (0,)


def test_spectral_gap_reported():
    u, s, vt = _factors()
    _, _, _, rep = canonical_signs(u, s, vt)
    assert rep.min_relative_gap is not None and rep.min_relative_gap > 0
    assert canonical_signs(u[:, :1], s[:1], vt[:1])[3].min_relative_gap is None


@pytest.mark.parametrize("bad", ["shape", "rank", "nan"])
def test_refuses_bad_factors(bad):
    u, s, vt = _factors()
    if bad == "shape":
        args = (u, s[:, None], vt)
    elif bad == "rank":
        args = (u[:, :3], s, vt)
    else:
        vt = vt.copy()
        vt[0, 0] = np.nan
        args = (u, s, vt)
    with pytest.raises(ValueError):
        canonical_signs(*args)
