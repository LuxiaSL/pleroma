"""The canonical ridge + rank truncation, as a spec for every solve in the build.

The build solves ridge in several places and spells "full rank" more than
once. This file states what each must compute — the centered closed form in
float64, and the full-rank convention ``r <= 0 or r >= min(W.shape)`` — and
holds the BUILD path's solves to it: `pleroma.map.build.regress.ridge_fit` /
`rank_truncate` (numpy), `pleroma.map.build.cv.cv_sweep` (torch float64, both
branches) and `pleroma.map.build.ridge.resolve_rank`. (fit_loom_map's and
v1a_export's inline solves are held to the same reference end-to-end in
test_fit_loom_map / test_v1a_export_e2e.)
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T


@pytest.fixture(scope="module")
def xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    x = rng.standard_normal((60, 10)) * rng.uniform(0.5, 3.0, 10) + 2.0
    y = x @ rng.standard_normal((10, 12)) + 0.3 * rng.standard_normal((60, 12)) - 1.0
    return x, y


def test_reference_is_the_normal_equations(xy) -> None:
    """The spec itself: W solves (XcᵀXc + λI) W = XcᵀYc, i.e. minimises
    ||Yc - Xc W||² + λ||W||² (gradient zero), on column-centered data."""
    x, y = xy
    w, mu_x, mu_y = F.ridge_reference(x, y, 7.0)
    xc, yc = x - mu_x, y - mu_y
    grad = xc.T @ (xc @ w - yc) + 7.0 * w
    assert np.abs(grad).max() < 1e-9 * np.abs(xc.T @ yc).max()


def test_regress_levers_ridge_fit_is_the_reference_on_centered_inputs(xy) -> None:
    """`ridge_fit(z, y, lam)` does NOT center: the caller passes centered z/y
    (as ``pleroma.map.build.regress`` does). On centered inputs it is the
    reference exactly."""
    x, y = xy
    w_ref, mu_x, mu_y = F.ridge_reference(x, y, 1e3)
    w = T.ridge_fit(x - mu_x, y - mu_y, 1e3)
    assert np.allclose(w, w_ref, atol=1e-12, rtol=1e-10)
    # ... and on UNcentered inputs it is a different (intercept-free) model
    assert not np.allclose(T.ridge_fit(x, y, 1e3), w_ref, atol=1e-6)


@pytest.mark.parametrize("rank", [-1, 0, 1, 4, 9, 10, 11, 64])
def test_rank_truncate_full_rank_convention(xy, rank: int) -> None:
    """`rank_truncate(w, r)`: best rank-r approximation, full rank iff
    ``r <= 0 or r >= min(w.shape)``; agrees with `resolve_rank` (the export's
    spelling) on every r, so `--rank 0` everywhere names the grid's ``r0``."""
    x, y = xy
    w, _, _ = F.ridge_reference(x, y, 1e3)
    got = T.rank_truncate(w, rank)
    assert np.allclose(got, F.truncate_reference(w, rank), atol=1e-12)
    r_eff = T.resolve_rank(rank, w.shape)
    assert r_eff == (min(w.shape) if rank <= 0 or rank >= min(w.shape) else rank)
    assert np.allclose(got, F.truncate_reference(w, r_eff), atol=1e-12)
    assert np.linalg.matrix_rank(got) == r_eff


def test_cv_sweep_is_the_reference_per_fold_both_rank_branches(xy) -> None:
    """`v1a_fit.cv_sweep` (torch float64 on CPU, stored float32): for every
    fold, λ and rank, pred_te = (x_te - mu_tr) rank_r(W_tr) + mu_y_tr with
    W_tr the reference fit on the other folds. r=3 exercises the truncated
    branch the main-run test cannot reach at z-dim 12."""
    x, y = xy
    fold = np.arange(x.shape[0]) % T.FIT_FOLDS
    preds = T.cv_sweep(x, y, fold, (1e1, 1e3), (0, 3), "cpu")
    assert set(preds) == {(1e1, 0), (1e1, 3), (1e3, 0), (1e3, 3)}
    for (lam, r), p in preds.items():
        assert p.dtype == np.float32 and p.shape == y.shape
        for f in range(T.FIT_FOLDS):
            te, tr = fold == f, fold != f
            w, mu_x, mu_y = F.ridge_reference(x[tr], y[tr], lam)
            want = (x[te] - mu_x) @ F.truncate_reference(w, r) + mu_y
            assert np.allclose(p[te], want, atol=1e-5 * np.abs(want).max(), rtol=0)


def test_lambda_changes_the_solution(xy) -> None:
    """The λ defaults differ across the stages (1e3 for regress's emitted
    predictions, 1e4 for the v1a fit and export),
    and on a realistic spectrum the choice is not cosmetic."""
    x, y = xy
    w3, _, _ = F.ridge_reference(x, y, 1e3)
    w4, _, _ = F.ridge_reference(x, y, 1e4)
    assert np.linalg.norm(w3 - w4) > 0.1 * np.linalg.norm(w3)
