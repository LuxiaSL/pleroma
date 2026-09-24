"""Tests for the v1a export — the identities the whole approach rests on.

★ THESE ARE LOAD-BEARING, NOT DECORATION. The v1a spot check runs entirely on
certified harnesses (`eval1_codes`, `eval1_run`, `/wear_code`) with no new
inference code, and the ONLY reason that is legitimate is two algebraic facts:

  1. the deployed one-vs-rest path computes `Vt @ (dz_i @ W)` — i.e. exactly
     v1a's registered LOO fan-centered input, because `mu_in`/`mu_y` cancel in
     the fan contrast;
  2. `lever_from_code(differential=True)` inverts that exactly, because the
     vector lives in W's row space and `Vt`'s orthonormal rows span it.

If either fails, every number downstream is measuring something else. So both
are pinned here against a REAL `LoomMap` built from a synthetic artifact in the
exporter's own format, plus the zero-padding no-op that makes the format fit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.map.build.export import (
    BINS_DIM,
    LAM,
    RANK,
    contrast_rows,
    deployment_contrast,
    zero_pad_factors,
)

Z_DIM_T = 40          # a small stand-in for 2713
BINS_DIM_T = 7        # ... and for 420
SITES_T = [8, 15, 19, 22]
HIDDEN_T = 5
RANK_T = 6


@pytest.fixture()
def synthetic_map(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    """A v1a-shaped artifact in the exporter's format + matching discriminants.

    Built by the same recipe `v1a_export.main` uses — fit a ridge on fan-centered
    inputs, truncate, zero-pad — so what the tests exercise is the real format
    and the real `LoomMap` reader, not a mock of either.
    """
    rng = np.random.default_rng(7)
    out_dim = len(SITES_T) * HIDDEN_T
    n = 90
    z = rng.standard_normal((n, Z_DIM_T))
    y = rng.standard_normal((n, out_dim))
    mu_in_z = z.mean(axis=0)
    mu_y = y.mean(axis=0)
    zc, yc = z - mu_in_z, y - mu_y
    w = np.linalg.solve(zc.T @ zc + 1e-2 * np.eye(Z_DIM_T), zc.T @ yc)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    u, s, vt = u[:, :RANK_T], s[:RANK_T], vt[:RANK_T]
    w_u, w_s, w_vt = zero_pad_factors(u, s, vt, BINS_DIM_T)

    disc = tmp_path / "disc.npz"
    np.savez(disc, FULL_mean=np.zeros(Z_DIM_T), FULL_scale=np.ones(Z_DIM_T))
    sha = hashlib.sha256(disc.read_bytes()).hexdigest()

    mapp = tmp_path / "v1a.npz"
    np.savez(
        mapp,
        # float64, as the exporter ships: a float32 Vt is only orthonormal to
        # ~1e-7, which would turn the row-space round trip from an identity into
        # an approximation for no reason (see v1a_export's FACTOR_DTYPE note).
        W_U=w_u, W_S=w_s, W_Vt=w_vt,
        mu_in=np.concatenate([mu_in_z, np.zeros(BINS_DIM_T)]),
        mu_y=mu_y,
        v3_mu=np.zeros(Z_DIM_T, dtype=np.float32),
        v3_sd=np.ones(Z_DIM_T, dtype=np.float32),
        v3_dead=np.zeros(Z_DIM_T, dtype=bool),
        bins_mu=np.zeros(BINS_DIM_T, dtype=np.float32),
        bins_sd=np.ones(BINS_DIM_T, dtype=np.float32),
        bins_dead=np.zeros(BINS_DIM_T, dtype=bool),
        sites=np.array(SITES_T),
        norm_ref=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        norm_ref_v1a_own=np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32),
        meta=np.array(json.dumps({
            "rank": RANK_T, "lam": 1e-2,
            "discriminants": str(disc), "discriminants_sha256": sha,
            "feature_order": f"z_corpus({Z_DIM_T}) then bins_std({BINS_DIM_T})",
        })),
    )
    # ★ the TRUNCATED W is what the artifact carries and therefore what every
    # identity below is stated against — a rank-r map does not reproduce the
    # full-rank ridge solution, and comparing to the latter would be testing
    # truncation, not the deployment identity.
    return mapp, disc, {"W_z": (u * s) @ vt, "W_full": w, "mu_in_z": mu_in_z}


# ── the zero-padding: the bins half of LoomMap's contract is a strict no-op ───

def test_zero_pad_factors_only_grows_U_and_pads_with_zeros() -> None:
    u = np.arange(12, dtype=np.float64).reshape(4, 3)
    s = np.array([3.0, 2.0, 1.0])
    vt = np.eye(3)
    pu, ps, pvt = zero_pad_factors(u, s, vt, bins_dim=5)
    assert pu.shape == (9, 3)
    assert np.array_equal(pu[:4], u)
    assert np.count_nonzero(pu[4:]) == 0
    assert np.array_equal(ps, s) and np.array_equal(pvt, vt)


def test_zero_pad_factors_refuses_inconsistent_ranks() -> None:
    with pytest.raises(ValueError, match="rank"):
        zero_pad_factors(np.zeros((4, 3)), np.zeros(2), np.zeros((3, 6)))


def test_perturbing_the_bins_input_changes_the_output_by_exactly_zero(
        synthetic_map) -> None:
    """★ The format-compatibility claim, as an equality and not an epsilon."""
    from pleroma.serve.legacy import LoomMap

    mapp, disc, _ = synthetic_map
    lm = LoomMap(mapp, disc)
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(Z_DIM_T)
    lev_a, raw_a, code_a = lm.lever_of(sig, np.zeros(BINS_DIM_T))
    lev_b, raw_b, code_b = lm.lever_of(sig, rng.standard_normal(BINS_DIM_T) * 1e3)
    assert code_a == code_b
    assert np.array_equal(lev_a, lev_b)
    assert raw_a == raw_b


# ── the deployment identity: contrast-after-encoding == encode-of-contrast ────

def test_contrastive_code_path_equals_W_times_fan_centered_dz(
        synthetic_map) -> None:
    """★ THE JUSTIFICATION FOR THE WHOLE APPROACH.

    Build a 16-future fan, run it through the certified path (`lever_of` ->
    `contrastive_code` -> `lever_from_code(differential=True)`), and compare with
    v1a's training-time quantity `W @ (z_i - mean_{j!=i} z_j)`. Equality here is
    what licenses using eval1's runner unchanged for the v1a arm.
    """
    from pleroma.probe.orchestrator import contrastive_code
    from pleroma.serve.legacy import LoomMap

    mapp, disc, truth = synthetic_map
    lm = LoomMap(mapp, disc)
    rng = np.random.default_rng(11)
    k = 16
    sigs = rng.standard_normal((k, Z_DIM_T))
    bins = rng.standard_normal((k, BINS_DIM_T)) * 50.0  # deliberately loud

    codes = [lm.lever_of(sigs[i], bins[i])[2] for i in range(k)]
    for sel in (0, 5, 15):
        ctr = contrastive_code(codes, sel)
        # unmatched reconstruction: the ruler is a wear-time transform
        via_code = (np.asarray(ctr) @ lm.Vt).reshape(lm.n_sites, lm.hidden)
        dz = sigs[sel] - sigs[[i for i in range(k) if i != sel]].mean(axis=0)
        direct = (dz @ truth["W_z"]).reshape(lm.n_sites, lm.hidden)
        # `lever_of` rounds the banked code to 4 dp, which is the only source of
        # disagreement; the identity itself is exact.
        assert np.allclose(via_code, direct, atol=1e-3, rtol=0)
        assert np.corrcoef(via_code.ravel(), direct.ravel())[0, 1] > 0.9999


def test_lever_from_code_inverts_Vt_exactly_on_the_row_space(
        synthetic_map) -> None:
    """Vtᵀ Vt v == v for any v in W's row space — exact, not merely close.

    Stated against the ARTIFACT's own W (`lm.W`, reconstructed from the stored
    factors), because that is the operator the deployment actually applies.
    """
    from pleroma.serve.legacy import LoomMap

    mapp, disc, _ = synthetic_map
    lm = LoomMap(mapp, disc)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(lm.W.shape[0]) @ lm.W
    round_tripped = (lm.Vt @ v) @ lm.Vt
    assert np.allclose(round_tripped, v, atol=1e-12, rtol=1e-12)


def test_deployment_contrast_matches_direct_dz_at_bank_scale(
        synthetic_map) -> None:
    """The vectorised form the exporter's self-check uses, on a whole `bank`."""
    from pleroma.serve.legacy import LoomMap

    mapp, disc, truth = synthetic_map
    lm = LoomMap(mapp, disc)
    rng = np.random.default_rng(5)
    n = 24
    z = rng.standard_normal((n, Z_DIM_T))
    fans = [np.arange(0, 8), np.arange(8, 16), np.arange(16, 24)]
    x = np.hstack([z, rng.standard_normal((n, BINS_DIM_T)) * 7.0])
    codes = (x - lm.mu_in) @ lm.W @ lm.Vt.T
    got = deployment_contrast(codes, lm.Vt, fans)
    # against the artifact's own W: the identity is exact in float64.
    want_artifact = contrast_rows(z, fans) @ lm.W[:Z_DIM_T]
    assert np.allclose(got, want_artifact, atol=1e-11, rtol=0)
    # and against the float64 fit itself: with float64 factors there is no
    # storage residue left to excuse a gap.
    want_fit = contrast_rows(z, fans) @ truth["W_z"]
    rel = np.max(np.abs(got - want_fit)) / np.median(
        np.linalg.norm(want_fit, axis=1))
    assert rel < 1e-12


# ── the contrast convention itself ────────────────────────────────────────────

def test_contrast_rows_is_leave_one_out_not_all_of_fan() -> None:
    x = np.array([[1.0], [2.0], [6.0]])
    out = contrast_rows(x, [np.arange(3)])
    assert out[0, 0] == pytest.approx(1.0 - 4.0)
    assert out[1, 0] == pytest.approx(2.0 - 3.5)
    assert out[2, 0] == pytest.approx(6.0 - 1.5)


def test_the_registered_operating_point_is_the_one_shipped() -> None:
    """Spot-check decision 2: the REGISTERED point, not the sweep's best cell."""
    assert (LAM, RANK) == (1e4, 64)
    assert BINS_DIM == 420


# ── the operating point as a PARAMETER (e.g. the λ=1e3 full-rank point) ───────

def test_resolve_rank_mirrors_v1a_fit_grid_convention() -> None:
    """★ `--rank 0` here must name the same model as the grid key `r0`.

    `v1a_fit.cv_sweep` takes the full-rank branch on `r <= 0 or r >= min(shape)`.
    If this helper disagreed, an export labelled "lambda=1e3, full rank" would
    not be the cell whose held-out .9837 is the reason to want it.
    """
    from pleroma.map.build.export import resolve_rank

    shape = (2713, 12288)
    assert resolve_rank(0, shape) == 2713          # the grid's "r0"
    assert resolve_rank(-1, shape) == 2713
    assert resolve_rank(99999, shape) == 2713      # >= min(shape) is full too
    assert resolve_rank(2713, shape) == 2713
    assert resolve_rank(64, shape) == 64           # the registered point
    assert resolve_rank(2712, shape) == 2712
    with pytest.raises(ValueError):
        resolve_rank(0, (0, 12288))


def test_heldout_grid_matches_the_registered_sweep() -> None:
    """The floors the exporter gates on are the v1a fit report's own numbers."""
    from pleroma.map.build.export import HELDOUT_GRID, SELFCHECK_FLOOR

    assert HELDOUT_GRID[(1e4, 64)] == 0.9339        # the registered point
    assert HELDOUT_GRID[(1e3, 0)] == 0.9837         # the full-rank candidate
    # the built-in floor is the registered point's bar and NOTHING else's
    assert SELFCHECK_FLOOR < HELDOUT_GRID[(1e4, 64)]
    assert SELFCHECK_FLOOR < HELDOUT_GRID[(1e3, 0)]


def test_overriding_the_point_without_a_floor_is_refused(
        monkeypatch, tmp_path: Path) -> None:
    """★ The under-gating trap, closed.

    A full-rank map scores .9837 held out. Reusing the registered point's .93
    floor would pass an export that lost four points somewhere in the factor
    path. The refusal fires before ANY input is read, so it is checked here
    against paths that do not exist.
    """
    import sys

    from pleroma.map.build import export as v1a_export

    argv = ["v1a_export",
            "--pairs-dir", str(tmp_path / "nope"),
            "--levers", str(tmp_path / "nope.npz"),
            "--hiddens", str(tmp_path / "nope.npz"),
            "--bins", str(tmp_path / "nope.npz"),
            "--wide-map", str(tmp_path / "nope.npz"),
            "--out", str(tmp_path / "out.npz"),
            "--lam", "1000", "--rank", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    assert v1a_export.main() == 1
    assert not (tmp_path / "out.npz").exists()

    # ... and an out-of-range floor is refused too, for the same reason
    monkeypatch.setattr(sys, "argv", argv + ["--selfcheck-floor", "1.5"])
    assert v1a_export.main() == 1


# ── the TARGET as a parameter (e.g. the sitenorm target) ─────────────────────

def test_build_target_reproduces_v1a_fits_own_sitenorm_arithmetic() -> None:
    """★ The export's target must be the one the fit SWEPT, not a lookalike.

    `v1a_fit.main` builds `d_sitenorm` inline. If `build_target` drifted from
    it — a different axis, a different epsilon, a reshape in the wrong order —
    the exported map would be a cell that no held-out number describes, and the
    `--selfcheck-floor` taken from the v1a fit report would be gating the wrong
    thing. So this recomputes the fit's expression literally and compares.
    """
    from pleroma.map.build.export import build_target

    rng = np.random.default_rng(7)
    n, n_sites, hidden = 11, 4, 5
    d_raw = rng.standard_normal((n, n_sites * hidden))

    # v1a_fit.main, lines "d_site = ...; norms = ...; d_sitenorm = ..."
    d_site = d_raw.reshape(n, n_sites, -1).copy()
    norms = np.maximum(np.linalg.norm(d_site, axis=2, keepdims=True), 1e-12)
    expected = (d_site / norms).reshape(n, -1)

    got = build_target(d_raw, "sitenorm", n_sites)
    assert got.shape == d_raw.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
    # every site block is unit norm, which is the whole point
    block_norms = np.linalg.norm(got.reshape(n, n_sites, hidden), axis=2)
    np.testing.assert_allclose(block_norms, 1.0, rtol=1e-12)


def test_build_target_raw_is_the_identity_and_unknown_targets_are_refused(
) -> None:
    from pleroma.map.build.export import build_target

    d_raw = np.arange(24, dtype=np.float64).reshape(3, 8)
    assert build_target(d_raw, "raw", 4) is d_raw          # not even a copy
    with pytest.raises(ValueError):
        build_target(d_raw, "l2", 4)
    with pytest.raises(ValueError):                         # 8 % 3 != 0
        build_target(d_raw, "sitenorm", 3)


def test_sitenorm_is_an_override_and_is_refused_without_a_floor(
        monkeypatch, tmp_path: Path) -> None:
    """★ Target is part of the registered point, so selecting it needs a bar.

    `v1a_fit.PRIMARY` names all four of input/target/lambda/rank. A sitenorm
    export at the registered lambda and rank is therefore NOT the registered
    point, and must not inherit the registered point's .93 floor — that floor
    is `dz|raw|lam10000|r64`'s bar and nothing else's. The refusal fires before
    any input is read, so nonexistent paths are enough to check it.
    """
    import sys

    from pleroma.map.build import export as v1a_export

    argv = ["v1a_export",
            "--pairs-dir", str(tmp_path / "nope"),
            "--levers", str(tmp_path / "nope.npz"),
            "--hiddens", str(tmp_path / "nope.npz"),
            "--bins", str(tmp_path / "nope.npz"),
            "--wide-map", str(tmp_path / "nope.npz"),
            "--out", str(tmp_path / "out.npz"),
            "--target", "sitenorm"]       # lam/rank left AT the registered pair
    monkeypatch.setattr(sys, "argv", argv)
    assert v1a_export.main() == 1
    assert not (tmp_path / "out.npz").exists()


def test_savez_compressed_is_byte_reproducible_for_identical_payloads(
        tmp_path: Path) -> None:
    """★ PINS THE CLAIM IN THE DOCSTRING, rather than asserting it in prose.

    The registered export must stay reproducible now that the operating point
    is a parameter. That rests on `np.savez_compressed` being a pure function
    of its payload — true only because numpy writes zip members with the zip
    format's fixed minimum timestamp instead of the wall clock. If a numpy upgrade started
    stamping mtimes, "byte-for-byte" would quietly become "content-for-content"
    and this test is where that is found out.
    """
    import time

    payload = {
        "W_U": np.arange(12.0).reshape(4, 3),
        "W_S": np.array([3.0, 2.0, 1.0]),
        "meta": np.array(json.dumps({"rank": 3, "lam": 1e4})),
    }
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, **payload)
    time.sleep(1.1)                      # cross a filesystem mtime tick
    np.savez_compressed(b, **payload)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == \
        hashlib.sha256(b.read_bytes()).hexdigest()


# ── what FULL RANK does to the served object ─────────────────────────────────

@pytest.fixture()
def synthetic_map_full(tmp_path: Path) -> tuple[Path, Path, int]:
    """The same recipe as `synthetic_map`, with the truncation switched OFF."""
    from pleroma.map.build.export import resolve_rank

    rng = np.random.default_rng(7)
    out_dim = len(SITES_T) * HIDDEN_T
    n = 90
    z = rng.standard_normal((n, Z_DIM_T))
    y = rng.standard_normal((n, out_dim))
    mu_in_z, mu_y = z.mean(axis=0), y.mean(axis=0)
    zc, yc = z - mu_in_z, y - mu_y
    w = np.linalg.solve(zc.T @ zc + 1e-2 * np.eye(Z_DIM_T), zc.T @ yc)
    r = resolve_rank(0, w.shape)
    assert r == out_dim == min(w.shape)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    w_u, w_s, w_vt = zero_pad_factors(u[:, :r], s[:r], vt[:r], BINS_DIM_T)

    disc = tmp_path / "disc_full.npz"
    np.savez(disc, FULL_mean=np.zeros(Z_DIM_T), FULL_scale=np.ones(Z_DIM_T))
    sha = hashlib.sha256(disc.read_bytes()).hexdigest()
    mapp = tmp_path / "v1a_full.npz"
    np.savez(
        mapp, W_U=w_u, W_S=w_s, W_Vt=w_vt,
        mu_in=np.concatenate([mu_in_z, np.zeros(BINS_DIM_T)]), mu_y=mu_y,
        v3_mu=np.zeros(Z_DIM_T, dtype=np.float32),
        v3_sd=np.ones(Z_DIM_T, dtype=np.float32),
        v3_dead=np.zeros(Z_DIM_T, dtype=bool),
        bins_mu=np.zeros(BINS_DIM_T, dtype=np.float32),
        bins_sd=np.ones(BINS_DIM_T, dtype=np.float32),
        bins_dead=np.zeros(BINS_DIM_T, dtype=bool),
        sites=np.array(SITES_T),
        norm_ref=np.array([1.0, 2.0, 3.0, 4.0]),
        meta=np.array(json.dumps({
            "rank": r, "lam": 1e3, "discriminants": str(disc),
            "discriminants_sha256": sha,
            "feature_order": f"z_corpus({Z_DIM_T}) then bins_std({BINS_DIM_T})",
        })),
    )
    return mapp, disc, r


def test_full_rank_map_serves_a_wider_code_and_still_reconstructs_exactly(
        synthetic_map_full) -> None:
    """The property the whole approach rests on survives turning truncation off.

    `Vtᵀ Vt v = v` needs v in W's row space; at full rank the row space is all
    of it, so the identity should be MORE robust, not less. Checked because a
    full-rank export changes `Vt` from 64 rows to min(W.shape) rows and an
    identity nobody re-ran is an assumption.
    """
    from pleroma.serve.legacy import LoomMap

    mapp, disc, r = synthetic_map_full
    lm = LoomMap(mapp, disc)
    assert lm.rank == r > RANK_T
    rng = np.random.default_rng(3)
    v = rng.standard_normal(lm.W.shape[0]) @ lm.W
    assert np.allclose((lm.Vt @ v) @ lm.Vt, v, atol=1e-12, rtol=1e-12)
    # and the served code is now r-d, which is the visible interface change
    _lev, _raw, code = lm.lever_of(rng.standard_normal(Z_DIM_T),
                                   np.zeros(BINS_DIM_T))
    assert len(code) == r


def test_a_rank64_code_is_REFUSED_by_a_full_rank_map(
        synthetic_map, synthetic_map_full) -> None:
    """★ THE BREAKING CHANGE, stated as a test rather than a caveat.

    `lever_from_code` gates on `c.size != self.rank`. Switching the served map
    to full rank therefore invalidates every banked code from the rank-64 map
    — `/wear_code`, the cross-conversation transplant, and eval1's banked fan
    codes all refuse. This is the cost of the operating point, and it must fail
    LOUDLY (it does) rather than reconstruct something plausible in the wrong
    basis (it must never).
    """
    from pleroma.serve.legacy import LoomMap

    narrow = LoomMap(*synthetic_map[:2])
    wide = LoomMap(*synthetic_map_full[:2])
    assert narrow.rank != wide.rank
    rng = np.random.default_rng(17)
    narrow_code = narrow.lever_of(rng.standard_normal(Z_DIM_T),
                                  np.zeros(BINS_DIM_T))[2]
    with pytest.raises(ValueError, match="coordinates but the map's rank"):
        wide.lever_from_code(narrow_code, differential=True)


def test_4dp_code_rounding_costs_more_at_full_rank(synthetic_map_full) -> None:
    """★ The lossy step the wider code makes worse, measured not assumed.

    `lever_of` rounds the banked code to 4 dp. Reconstruction error through
    `Vt` is the norm of the rounding vector, which grows like sqrt(rank): at 64
    coordinates it is negligible against the signal, and the question the
    full-rank drive has to answer is whether it still is at min(W.shape). The
    ordering is what this test pins; the magnitude on the REAL map depends on
    the real spectrum, so it is measured on that map, not here.
    """
    from pleroma.serve.legacy import LoomMap

    mapp, disc, r = synthetic_map_full
    lm = LoomMap(mapp, disc)
    rng = np.random.default_rng(23)
    sig = rng.standard_normal(Z_DIM_T)
    exact = lm.Vt @ ((np.concatenate([sig, np.zeros(BINS_DIM_T)]) - lm.mu_in)
                     @ lm.W)
    rounded = np.asarray(lm.lever_of(sig, np.zeros(BINS_DIM_T))[2])
    assert rounded.size == exact.size == r
    # rounding to 4 dp, and nothing else, separates them
    assert np.max(np.abs(rounded - exact)) <= 5e-5 + 1e-12
    # the reconstruction error is the rounding vector's norm (Vt is orthonormal)
    err = np.linalg.norm((rounded - exact) @ lm.Vt)
    assert err == pytest.approx(np.linalg.norm(rounded - exact), rel=1e-9)
    # ... and it scales with how many coordinates are kept
    err64 = np.linalg.norm((rounded - exact)[:RANK_T])
    assert err >= err64
