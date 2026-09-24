"""fit_loom_map (``pleroma.map.build.wide``) — the v0 "wide map" — pinned.

It stays on the golden path because the served map carries three things it
computes: the standardisation stats ``v3_mu/v3_sd/v3_dead`` + ``bins_*``, and
the shared ``norm_ref`` ruler. v1a_export either reads them back out of a
banked wide map or computes them with the same code
(``pleroma.map.build.shelf.compute_shelf``) — byte-identical either way. So
every quantity v1a_export takes from here is pinned exactly below, against
independent numpy references, on a synthetic bank with planted dead dims, a
no-lever [7,1] fan, a 2-pair fan and a non-member pair.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T
from tests.contract.build.conftest import npz_dict

WIDE_KEYS = {"W", "W_U", "W_S", "W_Vt", "mu_in", "mu_y", "v3_mu", "v3_sd", "v3_dead",
             "bins_mu", "bins_sd", "bins_dead", "sites", "norm_ref", "meta"}


def _argv(bank: F.Bank, out: Path, *extra: object, pairs_dir: Path | None = None,
          bins: Path | None = None) -> list[object]:
    return ["--pairs-dir", pairs_dir or bank.pairs_dir, "--levers", bank.levers,
            "--bins", bins or bank.bins, "--discriminants", bank.discriminants,
            "--norm-ref-bank", bank.levers, "--out", out, *extra]


def _joined(bank: F.Bank) -> list[int]:
    """Pairs rows fit_loom_map fits on: a lever member (group >= 0), in pairs order."""
    mo = bank.member_of
    return [r for r, g in enumerate(bank.pair_gen_ids.tolist())
            if g in mo and mo[g].group >= 0]


def _reference_w(bank: F.Bank, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _joined(bank)
    gids = bank.pair_gen_ids[rows]
    train = np.asarray([bank.pair_splits[r] == "train" for r in rows])
    b = np.stack([bank.bins_of[int(g)] for g in gids])
    mu, sd = b[train].mean(axis=0), b[train].std(axis=0)
    dead = sd < 1e-9
    b_std = (b - mu) / np.where(dead, 1.0, sd)
    b_std[:, dead] = 0.0
    x = np.hstack([bank.z_corpus[rows].astype(np.float64), b_std])
    mo = bank.member_of
    y = np.stack([(1.0 - 2.0 * mo[int(g)].cluster)
                  * bank.levers_arr[mo[int(g)].group].astype(np.float64).ravel()
                  for g in gids])
    return F.ridge_reference(x, y, lam)


# ── the artifact's format ─────────────────────────────────────────────────────

def test_output_keys_dtypes_and_shapes(small: F.Bank, small_wide: Path) -> None:
    """Pins the wide-map schema LoomMap and v1a_export read.

    Arrays ship float32 (v1a_export ships float64 by contrast —
    ``pleroma.map.build.export.FACTOR_DTYPE``), dead masks are bool, and BOTH
    the dense ``W`` and the rank-r factors are written (LoomMap prefers the
    dense W and re-derives its own SVD — the factors are not what it serves).
    """
    a = npz_dict(small_wide)
    assert set(a) == WIDE_KEYS
    zi, bi, out = small.z_dim, small.bins_dim, len(small.sites) * small.hidden
    shapes = {"W": (zi + bi, out), "W_U": (zi + bi, 8), "W_S": (8,), "W_Vt": (8, out),
              "mu_in": (zi + bi,), "mu_y": (out,), "v3_mu": (zi,), "v3_sd": (zi,),
              "v3_dead": (zi,), "bins_mu": (bi,), "bins_sd": (bi,), "bins_dead": (bi,),
              "sites": (len(small.sites),), "norm_ref": (len(small.sites),)}
    for k, shp in shapes.items():
        assert a[k].shape == shp, k
    for k in ("W", "W_U", "W_S", "W_Vt", "mu_in", "mu_y", "v3_mu", "v3_sd",
              "bins_mu", "bins_sd", "norm_ref"):
        assert a[k].dtype == np.float32, k
    assert a["v3_dead"].dtype == bool and a["bins_dead"].dtype == bool
    assert a["sites"].tolist() == small.sites
    # the factored product IS the stored (truncated) W, to float32 storage
    prod = (a["W_U"].astype(np.float64) * a["W_S"]) @ a["W_Vt"]
    assert np.allclose(prod, a["W"], atol=1e-6 * np.abs(a["W"]).max())
    vt = a["W_Vt"].astype(np.float64)
    assert np.allclose(vt @ vt.T, np.eye(8), atol=1e-6)


def test_meta_fields(small: F.Bank, small_wide: Path) -> None:
    """``meta`` is provenance v1a_export copies (discriminants + sha, bins_config).

    ``feature_order`` is DERIVED from the arrays, never a literal width: a
    hardcoded 3B width would mislabel any other model's map.
    ``n_rows`` counts every joined lever member: no MIN_FAN filter here, so the
    sparse 2-pair fan is IN (v1a_fit drops it — see test_v1a_fit_cv).
    """
    meta = json.loads(str(npz_dict(small_wide)["meta"]))
    assert meta["lam"] == 1e4 and meta["rank"] == 8
    assert meta["feature_order"] == f"z_corpus({small.z_dim}) then bins_std({small.bins_dim})"
    assert meta["discriminants"] == str(small.discriminants)
    assert meta["discriminants_sha256"] == F.sha256(small.discriminants)
    assert meta["bins_config"] == {"layers": [1, 2, 3], "n_bins": small.bins_dim,
                                   "synthetic": True}
    assert meta["n_rows"] == len(_joined(small)) == 1146
    assert {"pairs_dir", "levers", "in_sample_mean_cos", "sign_convention",
            "alpha_convention"} <= set(meta)


def test_synthetic_wide_map_is_a_format_twin(small: F.Bank, small_wide: Path,
                                             tmp_path: Path) -> None:
    """Holds `_fixtures.write_wide_map` (used by the export tests, where the real
    fit is too slow for the suite) to the real writer's keys, dtypes and meta
    fields — so the export tests exercise the format fit_loom_map actually emits."""
    real = npz_dict(small_wide)
    twin = npz_dict(F.write_wide_map(small, tmp_path / "twin.npz", rank=8))
    assert set(real) == set(twin)
    for k in real:
        if k == "meta":        # a JSON string: same kind, length is content
            assert real[k].dtype.kind == twin[k].dtype.kind == "U"
            continue
        assert real[k].dtype == twin[k].dtype, k
        assert real[k].shape == twin[k].shape, k
    assert set(json.loads(str(real["meta"]))) == set(json.loads(str(twin["meta"])))
    # the standardisation stats are not just same-shaped: they are the SAME
    for k in ("v3_mu", "v3_sd", "v3_dead", "bins_mu", "bins_sd", "bins_dead", "norm_ref"):
        assert np.array_equal(real[k], twin[k]), k


# ── the standardisation stats v1a_export copies ───────────────────────────────

def test_v3_stats_are_float32_train_row_stats_of_z_full(small: F.Bank,
                                                        small_wide: Path) -> None:
    """v3_mu/v3_sd = mean/std over TRAIN pairs of z_full cast to float32 FIRST
    (the pairs stage's own recipe) — bit-exact.

    Also pins that they are train-only: all-rows stats differ on this bank.
    """
    a = npz_dict(small_wide)
    assert np.array_equal(a["v3_mu"], small.v3_mu)
    assert np.array_equal(a["v3_sd"], small.v3_sd)
    assert np.array_equal(a["v3_dead"], small.v3_dead)
    assert not np.allclose(a["v3_mu"], small.z_full32.mean(axis=0))


def test_dead_dim_cutoffs_v3_1e8_vs_bins_1e9(small: F.Bank, small_wide: Path) -> None:
    """``v3_dead = sd < 1e-8`` (``pleroma.map.build.shelf.V3_DEAD_SD``) but
    ``bins_dead = sd < 1e-9`` (``BINS_DEAD_SD``). Pinned AS-IS: a dim with SD
    5e-9 is dead in the z block and LIVE (standardised by 1/5e-9) in the bins
    block. Unifying the two cutoffs changes this test deliberately, with the
    served-map reproduction gate as the arbiter.

    z dims: 1 (SD 5e-9) and 2 (degenerate FULL_scale) are dead. Bins: col 0
    (constant) dead, col 1 (SD 5e-9) live.
    """
    a = npz_dict(small_wide)
    assert np.nonzero(a["v3_dead"])[0].tolist() == [1, 2]
    assert np.nonzero(a["bins_dead"])[0].tolist() == [0]
    assert 1e-9 < float(a["bins_sd"][1]) < 1e-8
    assert 0.0 < float(small.v3_sd[1]) < 1e-8


def test_a_constant_raw_feature_is_NOT_dead_in_the_z_block(small: F.Bank,
                                                           small_wide: Path) -> None:
    """Pinned AS-IS. Raw dim 0 is the SAME
    value in every signature, yet ``v3_dead[0]`` is False: the float32 mean of a
    constant float32 column carries accumulation error e, so ``v3_sd = |e|``
    (~1e-5 here, far above the 1e-8 cutoff) and every z_corpus entry of that
    dim is ``-e/|e| = ±1`` — a constant, not a zero. Harmless to the fits
    (the centered ridge gives a constant column zero weight; the fan contrast
    cancels it) but it is exactly the float32 cast noise on near-constant dims
    the shelf stats inherit, and any unification of the dead cutoffs must
    decide it on purpose.
    """
    a = npz_dict(small_wide)
    assert not bool(a["v3_dead"][0])
    assert float(a["v3_sd"][0]) > 1e-8
    col = small.z_corpus[:, 0]
    assert np.unique(col).size == 1 and abs(float(col[0])) == 1.0
    # zero weight: the dim contributes nothing to W
    assert np.abs(a["W"][0]).max() < 1e-6 * np.abs(a["W"]).max()


def test_bins_stats_over_joined_train_rows_only(small: F.Bank, small_wide: Path) -> None:
    """bins_mu/bins_sd are over JOINED lever members in the train split
    — not all bins rows, not all pairs."""
    a = npz_dict(small_wide)
    rows = _joined(small)
    gids = [int(small.pair_gen_ids[r]) for r in rows if small.pair_splits[r] == "train"]
    b = np.stack([small.bins_of[g] for g in gids])
    assert np.array_equal(a["bins_mu"], b.mean(axis=0).astype(np.float32))
    assert np.array_equal(a["bins_sd"], b.std(axis=0).astype(np.float32))
    every = np.stack(list(small.bins_of.values()))
    assert not np.array_equal(a["bins_mu"], every.mean(axis=0).astype(np.float32))


def test_norm_ref_is_nanmedian_of_bank_lever_site_norms(small: F.Bank,
                                                         small_wide: Path) -> None:
    """THE RULER (alpha = 1) that v1a_export carries as its worn ``norm_ref``:
    per-site median over groups of ||lever|| in the --norm-ref-bank."""
    want = np.nanmedian(np.linalg.norm(small.levers_arr.astype(np.float64), axis=2), axis=0)
    assert np.array_equal(npz_dict(small_wide)["norm_ref"], want.astype(np.float32))


# ── the fit ───────────────────────────────────────────────────────────────────

def test_W_is_the_closed_form_centered_ridge_then_rank_truncated(
        small: F.Bank, small_wide: Path) -> None:
    """W = rank_r( (XcᵀXc + λI)⁻¹ XcᵀYc ) with X = [z_corpus | bins_std] over the
    joined rows and Y = m_sign * lever, m_sign = 1 - 2*cluster (the sign
    convention: toward the read trajectory's basin). mu_in/mu_y are the row
    means. Stated against `_fixtures.ridge_reference` — the one closed-form
    spec every ridge in the build is held to."""
    a = npz_dict(small_wide)
    w, mu_x, mu_y = _reference_w(small, 1e4)
    w8 = F.truncate_reference(w, 8)
    scale = np.abs(w8).max()
    assert np.allclose(a["W"], w8, atol=2e-6 * scale, rtol=0)
    assert np.allclose(a["mu_in"], mu_x, atol=1e-5)
    assert np.allclose(a["mu_y"], mu_y, atol=1e-5)


def test_output_is_byte_deterministic(small: F.Bank, small_wide: Path,
                                      tmp_path: Path) -> None:
    """Same inputs -> same FILE sha (numpy writes zip members at the 1980 epoch);
    ``--out`` is not in meta, so the path does not leak into the bytes."""
    out = tmp_path / "again.npz"
    assert T.run_fit_loom_map(_argv(small, out, "--lam", "10000", "--rank", "8")) == 0
    assert F.sha256(out) == F.sha256(small_wide)


def test_defaults_equal_the_live_wide_map_point(small: F.Bank, tmp_path: Path) -> None:
    """The profile default must be the point every banked wide map used."""
    out = tmp_path / "defaults.npz"
    assert T.run_fit_loom_map(_argv(small, out)) == 0
    meta = json.loads(str(npz_dict(out)["meta"]))
    # rank 64 is the REQUESTED default; on this 20-wide toy W it resolves to
    # full rank and is banked as such (effective rank, requested beside it)
    assert (meta["lam"], meta.get("rank_requested", meta["rank"])) == (1e4, 64)


def test_full_rank_writes_dense_W_only(small: F.Bank, tmp_path: Path) -> None:
    """``--rank 0`` (or >= min(W.shape)) skips truncation AND the factors:
    only the dense W ships, equal to the untruncated ridge; meta records the
    EFFECTIVE rank and the requested one beside it (a banked 0 would read back
    in LoomMap as an empty code basis)."""
    out = tmp_path / "full.npz"
    assert T.run_fit_loom_map(_argv(small, out, "--lam", "10000", "--rank", "0")) == 0
    a = npz_dict(out)
    assert set(a) == WIDE_KEYS - {"W_U", "W_S", "W_Vt"}
    w, _, _ = _reference_w(small, 1e4)
    assert np.allclose(a["W"], w, atol=2e-6 * np.abs(w).max(), rtol=0)
    meta = json.loads(str(a["meta"]))
    assert meta["rank"] == min(w.shape) and meta["rank_requested"] == 0


def test_full_rank_map_reads_back_with_its_effective_rank(small: F.Bank,
                                                          tmp_path: Path) -> None:
    """A full-rank map must read back as rank min(W.shape), not 0."""
    out = tmp_path / "full.npz"
    assert T.run_fit_loom_map(_argv(small, out, "--lam", "10000", "--rank", "0")) == 0
    lm = T.load_loom_map(out, small.discriminants)
    assert lm.rank == min(lm.W.shape)


# ── refusals ──────────────────────────────────────────────────────────────────

def _copy_pairs(bank: F.Bank, dst: Path) -> Path:
    shutil.copytree(bank.pairs_dir, dst)
    return dst


def test_refuses_pairs_not_built_with_standardize_corpus(small: F.Bank,
                                                         tmp_path: Path) -> None:
    """Exit 2 unless the pairs stage ran ``--standardize corpus``: without the
    corpus standardiser the discriminants' mismatched scale survives into z
    (median |z| ~3, a third of cells beyond |z| = 10) and the map retrieves
    nothing."""
    pdir = _copy_pairs(small, tmp_path / "pairs")
    man = json.loads((pdir / "pairs_manifest.json").read_text())
    man["params"]["standardize"] = "full"
    (pdir / "pairs_manifest.json").write_text(json.dumps(man))
    out = tmp_path / "m.npz"
    assert T.run_fit_loom_map(_argv(small, out, pairs_dir=pdir)) == 2
    assert not out.exists()


def test_refuses_when_recomputed_z_drifts_from_pairs_z(small: F.Bank,
                                                       tmp_path: Path) -> None:
    """Exit 1 if (z_full - v3_mu)/v3_sd recomputed from raw signatures differs
    from pairs_z by > 1e-3 on the verify rows
    (``pleroma.map.build.shelf.Z_VERIFY_TOL``)."""
    pdir = _copy_pairs(small, tmp_path / "pairs")
    with np.load(pdir / "pairs_z.npz", allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    payload["z"] = payload["z"] + np.float32(0.01)
    np.savez_compressed(pdir / "pairs_z.npz", **payload)
    out = tmp_path / "m.npz"
    assert T.run_fit_loom_map(_argv(small, out, pairs_dir=pdir)) == 1
    assert not out.exists()


def test_refuses_a_joined_gen_missing_from_bins(small: F.Bank, tmp_path: Path) -> None:
    """Exit 1 if any joined lever member has no bins row: absence raises, it
    is not imputed."""
    with np.load(small.bins, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    mo = small.member_of
    victim = next(k for k, g in enumerate(payload["generation_id"].tolist())
                  if g in mo and mo[g].group >= 0)
    keep = np.arange(payload["features"].shape[0]) != victim
    for k in ("features", "run_dir", "generation_id"):
        payload[k] = payload[k][keep]
    bins = tmp_path / "bins.npz"
    np.savez_compressed(bins, **payload)
    out = tmp_path / "m.npz"
    assert T.run_fit_loom_map(_argv(small, out, bins=bins)) == 1
    assert not out.exists()


def test_wide_map_factors_are_canonically_oriented(small_wide: Path) -> None:
    """The stored code basis has one orientation on every stack
    (pleroma.map.svd), so a re-fit cannot re-aim a banked code."""
    from pleroma.map.svd import SIGN_CONVENTION, is_canonical
    real = npz_dict(small_wide)
    assert is_canonical(real["W_Vt"])
    meta = json.loads(str(real["meta"]))
    assert meta["svd_orientation"]["convention"] == SIGN_CONVENTION
