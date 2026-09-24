"""v1a_export end to end — the SERVED map — on a synthetic 8.6k-gen bank.

The real ``pleroma.map.build.export.main`` (CPU torch) over a real pairs dir /
levers / hiddens / bins / wide map, invoked the way a live export is run. The
export may compute the wide map's stats itself instead of reading them, and
must then reproduce the served map, so what the served artifact IS gets pinned here:
its keys and dtypes, what it copies verbatim from the wide map, the ridge it
solves (against a closed-form numpy reference), the self-check gate both ways,
the file-sha reproducibility, and every flag the live command uses
(``--fan-source member_fan``, ``--z-dim``, ``--heldout-report``, ``--target``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T
from tests.contract.build.conftest import SITES, Run, export_argv, npz_dict, run_export

ARTIFACT_KEYS = {"W_U", "W_S", "W_Vt", "mu_in", "mu_y", "v3_mu", "v3_sd", "v3_dead",
                 "bins_mu", "bins_sd", "bins_dead", "sites", "norm_ref",
                 "norm_ref_v1a_own", "norm_ref_which", "meta"}
REPORT_KEYS = {"stage", "prereg_token", "out", "out_sha256", "n_rows", "n_fans",
               "operating_point", "self_check", "rulers", "sites", "hidden", "sources"}
# 540 prompts x 2 waves = 1,080 fans of 8. lever_group: 54 [7,1] fans dropped
# (432 gens), the sparse fan keeps 2 pairs and falls under MIN_FAN -> 8,200 / 1,025.
# member_fan: only the sparse fan goes -> 8,632 / 1,079.
N_LEVER, F_LEVER = 8200, 1025
N_MEMBER, F_MEMBER = 8632, 1079


def _kept(bank: F.Bank, *, member_fan: bool) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Independent re-derivation of the export's join: z, h, fans."""
    row_of = {int(g): r for r, g in enumerate(bank.pair_gen_ids.tolist())}
    ms = [m for m in bank.members if m.has_pair and m.fan != 1
          and (member_fan or m.group >= 0)]
    z = bank.z_corpus[[row_of[m.gen_id] for m in ms]].astype(np.float64)
    h = np.stack([bank.means[m.gen_id].reshape(-1) for m in ms]).astype(np.float64)
    fan = np.asarray([m.fan for m in ms])
    return z, h, [np.nonzero(fan == f)[0] for f in np.unique(fan)]


def _w_of(a: dict[str, np.ndarray], z_dim: int) -> np.ndarray:
    return (a["W_U"][:z_dim] * a["W_S"]) @ a["W_Vt"]


# ── the default (registered-point) export ─────────────────────────────────────

def test_default_export_passes_and_reports(export_default: Run, export_bank: F.Bank) -> None:
    """Exit 0; the artifact and ``<stem>_export_report.json`` exist; no
    ``<stem>.partial.npz`` staging file is left; the report pins its own file
    sha and the registered point, and — with no --heldout-report — a NULL
    held-out figure with the reason (never another model's held-out number)."""
    r = export_default
    assert r.rc == 0 and r.out.exists()
    assert not r.out.with_suffix(".partial.npz").exists()
    rep = r.report
    assert set(rep) == REPORT_KEYS
    assert rep["out_sha256"] == F.sha256(r.out)
    assert (rep["n_rows"], rep["n_fans"]) == (N_LEVER, F_LEVER)
    assert rep["operating_point"] == {"lambda": 1e4, "rank": 64, "input": "dz",
                                      "target": "raw"}
    sc = rep["self_check"]
    assert sc["verdict"] == "PASS" and sc["floor"] == T.SELFCHECK_FLOOR == 0.93
    assert sc["within_fan_top1_in_sample"] >= 0.93
    assert sc["chance"] == 0.125
    assert sc["bins_block_max_code_gap"] == 0.0
    assert sc["deployment_identity_rel_max_gap"] <= T.IDENTITY_TOL
    assert sc["registered_heldout_top1"] is None
    assert sc["heldout_source"] == T.HELDOUT_NOT_SUPPLIED
    assert rep["sites"] == export_bank.sites and rep["hidden"] == export_bank.hidden
    assert rep["sources"]["levers_sha256"] == F.sha256(export_bank.levers)


def test_artifact_keys_dtypes_and_what_is_copied_from_the_wide_map(
        export_default: Run, export_bank: F.Bank, export_wide: Path) -> None:
    """The served format:

    * factors and mu ship float64 (FACTOR_DTYPE — the Vtᵀ Vt identity);
    * W_U carries BINS_DIM (420) zero rows and mu_in a zero bins block, so
      the bins half of LoomMap's input is a strict no-op;
    * v3_*, bins_* are the wide map's arrays byte for byte, and ``norm_ref``
      (the WORN ruler) is the wide map's, cast to float64 — the dependency on
      the v0 chain that an export computing its own stats must reproduce.
    """
    a = npz_dict(export_default.out)
    w = npz_dict(export_wide)
    zd, bd = export_bank.z_dim, T.BINS_DIM
    out = len(export_bank.sites) * export_bank.hidden
    assert set(a) == ARTIFACT_KEYS
    for k, shp in {"W_U": (zd + bd, 64), "W_S": (64,), "W_Vt": (64, out),
                   "mu_in": (zd + bd,), "mu_y": (out,)}.items():
        assert a[k].shape == shp and a[k].dtype == np.float64, k
    assert np.count_nonzero(a["W_U"][zd:]) == 0
    assert np.count_nonzero(a["mu_in"][zd:]) == 0
    assert np.allclose(a["W_Vt"] @ a["W_Vt"].T, np.eye(64), atol=1e-12)
    for k in ("v3_mu", "v3_sd", "v3_dead", "bins_mu", "bins_sd", "bins_dead"):
        assert a[k].dtype == w[k].dtype and a[k].tobytes() == w[k].tobytes(), k
    assert a["norm_ref"].dtype == np.float64
    assert np.array_equal(a["norm_ref"], w["norm_ref"].astype(np.float64))
    assert a["sites"].tolist() == export_bank.sites
    assert "norm_ref_v1a_own is v1a's own and is NOT worn" in str(a["norm_ref_which"])


def test_meta_of_the_registered_point(export_default: Run, export_bank: F.Bank,
                                      export_wide: Path) -> None:
    """``meta`` at the registered point: provenance copied from the wide map
    (discriminants + sha, bins_config), feature_order from --z-dim, the
    operating point marked registered, and NO override-only keys (every such
    key is gated on leaving the registered point)."""
    meta = json.loads(str(npz_dict(export_default.out)["meta"]))
    wmeta = json.loads(str(npz_dict(export_wide)["meta"]))
    assert (meta["lam"], meta["rank"]) == (1e4, 64)
    assert (meta["n_rows"], meta["n_fans"]) == (N_LEVER, F_LEVER)
    assert meta["feature_order"] == f"z_corpus({export_bank.z_dim}) then bins_std(420)"
    assert meta["discriminants"] == wmeta["discriminants"]
    assert meta["discriminants_sha256"] == wmeta["discriminants_sha256"]
    assert meta["bins_config"] == wmeta["bins_config"]
    assert meta["prereg_token"] == "v1a-fit-001"
    op = meta["operating_point"]
    assert op["registered"] is True and (op["lambda"], op["rank"]) == (1e4, 64)
    assert not {"target_key", "grid_key", "overridden_from_registered"} & set(op)
    assert not {"fan_source", "selfcheck_floor", "full_rank",
                "target_normalization"} & set(meta)
    assert meta["sources"]["wide_map_sha256"] == F.sha256(export_wide)


def test_W_is_the_closed_form_ridge_on_fan_contrasted_inputs(
        export_default: Run, export_bank: F.Bank) -> None:
    """THE SERVED RIDGE, as a spec:
    W = rank64( (XcᵀXc + 1e4 I)⁻¹ XcᵀYc ) with X = dz (LOO fan-contrasted
    z_corpus) and Y = d_raw (LOO fan-contrasted mean hiddens), all joined rows
    — the export's torch float64 solve == numpy closed form to 1e-9.
    Because LOO contrasts sum to zero within every fan, mu_in and mu_y are 0
    up to roundoff. ``norm_ref_v1a_own`` is the
    per-site median norm of dz @ W (banked, NOT worn)."""
    a = npz_dict(export_default.out)
    z, h, fans = _kept(export_bank, member_fan=False)
    dz, d_raw = F.loo_center(z, fans), F.loo_center(h, fans)
    w_ref, mu_x, mu_y = F.ridge_reference(dz, d_raw, 1e4)
    w_ref = F.truncate_reference(w_ref, 64)
    got = _w_of(a, export_bank.z_dim)
    assert np.allclose(got, w_ref, atol=1e-9 * np.abs(w_ref).max(), rtol=0)
    assert np.abs(a["mu_in"]).max() < 1e-12 * max(1.0, np.abs(dz).max())
    assert np.abs(a["mu_y"]).max() < 1e-12 * max(1.0, np.abs(d_raw).max())
    pred = dz @ w_ref
    own = np.nanmedian(np.linalg.norm(pred.reshape(len(pred), 4, -1), axis=2), axis=0)
    assert np.allclose(a["norm_ref_v1a_own"], own, rtol=1e-9)


def test_loom_map_reads_the_artifact_and_bins_are_a_no_op(
        export_default: Run, export_bank: F.Bank, export_wide: Path) -> None:
    """The serving reader accepts it: rank 64, sites, hidden, and a bins row of
    any size moves the code by exactly 0 (the zero-padding contract)."""
    lm = T.load_loom_map(export_default.out, export_bank.discriminants)
    assert lm.rank == 64 and lm.sites == export_bank.sites and lm.hidden == export_bank.hidden
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(export_bank.z_dim)
    code_a = lm.lever_of(sig, np.zeros(T.BINS_DIM))[2]
    code_b = lm.lever_of(sig, rng.standard_normal(T.BINS_DIM) * 1e3)[2]
    assert code_a == code_b


# ── reproducibility and the no-op flags ───────────────────────────────────────

def test_export_file_sha_is_reproducible(export_default: Run, export_bank: F.Bank,
                                         export_wide: Path, tmp_path: Path) -> None:
    """Same inputs -> same FILE sha256 (zip members at the 1980 epoch, which
    is what lets a re-export be checked against a banked sha). ``--out`` is not
    in the npz, so a different output path does not change the bytes."""
    out = tmp_path / "again.npz"
    r = run_export(export_argv(export_bank, export_wide, out, z_dim=export_bank.z_dim), out)
    assert r.rc == 0
    assert F.sha256(out) == F.sha256(export_default.out)


def test_target_raw_explicit_is_byte_identical_to_the_default(
        export_default: Run, export_bank: F.Bank, export_wide: Path, tmp_path: Path) -> None:
    """``--target raw`` is the registered construction and a pure no-op:
    same file sha as passing nothing."""
    out = tmp_path / "raw.npz"
    r = run_export(export_argv(export_bank, export_wide, out, "--target", "raw",
                               z_dim=export_bank.z_dim), out)
    assert r.rc == 0
    assert F.sha256(out) == F.sha256(export_default.out)


def test_heldout_report_is_report_only(export_default: Run, export_bank: F.Bank,
                                       export_wide: Path, small_fit: Path,
                                       tmp_path: Path) -> None:
    """``--heldout-report`` (THIS model's v1a_fit report) fills the export
    REPORT's registered_heldout_top1 from the fit grid's dz|raw|lam10000|r64
    cell and names its source; the npz is byte-identical (the held-out figure
    lives in the report, never in the artifact)."""
    fit_report = small_fit / "v1a_report.json"
    out = tmp_path / "ho.npz"
    r = run_export(export_argv(export_bank, export_wide, out, "--heldout-report",
                               fit_report, z_dim=export_bank.z_dim), out)
    assert r.rc == 0
    grid = json.loads(fit_report.read_text())["grid"]
    assert r.report["self_check"]["registered_heldout_top1"] == \
        grid["dz|raw|lam10000|r64"]["retrieval_top1"]
    assert str(fit_report) in r.report["self_check"]["heldout_source"]
    assert F.sha256(out) == F.sha256(export_default.out)


# ── fan source ────────────────────────────────────────────────────────────────

def test_member_fan_keeps_the_no_lever_fans(export_member_fan: Run, export_default: Run,
                                            export_bank: F.Bank) -> None:
    """``--fan-source member_fan`` (the live served map's flag): the 54 [7,1]
    fans' 432 gens are IN (8,632 rows / 1,079 fans vs 8,200 / 1,025), the
    override is banked in meta and the report, the point stays registered,
    and on this bank the self-check still PASSes. The ridge is the same spec
    on the larger join."""
    r = export_member_fan
    assert r.rc == 0
    assert (r.report["n_rows"], r.report["n_fans"]) == (N_MEMBER, F_MEMBER)
    assert r.report["fan_source"] == "member_fan"
    assert r.report["self_check"]["verdict"] == "PASS"
    a = npz_dict(r.out)
    meta = json.loads(str(a["meta"]))
    assert meta["fan_source"]["source"] == "member_fan"
    assert meta["operating_point"]["registered"] is True
    assert F.sha256(r.out) != F.sha256(export_default.out)
    z, h, fans = _kept(export_bank, member_fan=True)
    w_ref, _, _ = F.ridge_reference(F.loo_center(z, fans), F.loo_center(h, fans), 1e4)
    w_ref = F.truncate_reference(w_ref, 64)
    assert np.allclose(_w_of(a, export_bank.z_dim), w_ref, atol=1e-9 * np.abs(w_ref).max())


def test_join_bank_row_counts_both_fan_sources(export_bank: F.Bank) -> None:
    """``join_bank`` is the one join: lever_group drops
    the no-lever fans and <MIN_FAN fans; member_fan drops only the latter."""
    a = T.join_bank(export_bank.pairs_dir, export_bank.levers, [export_bank.hiddens])
    b = T.join_bank(export_bank.pairs_dir, export_bank.levers, [export_bank.hiddens],
                    fan_source=T.FAN_SOURCE_MEMBER)
    assert (a.n, len(a.fans)) == (N_LEVER, F_LEVER)
    assert (b.n, len(b.fans)) == (N_MEMBER, F_MEMBER)
    assert all(f.size >= T.MIN_FAN for f in a.fans + b.fans)


# ── gates and refusals ────────────────────────────────────────────────────────

def test_selfcheck_refuses_a_map_that_does_not_retrieve(
        export_bank: F.Bank, export_wide: Path, tmp_path: Path) -> None:
    """Hidden means independent of z: in-sample top-1 falls to ~chance, below
    the .93 floor -> exit 1, NO artifact, the staging file removed, no report."""
    noise = F.write_noise_hiddens(export_bank)
    out = tmp_path / "noise.npz"
    r = run_export(export_argv(export_bank, export_wide, out, hiddens=noise,
                               z_dim=export_bank.z_dim), out)
    assert r.rc == 1
    assert not out.exists() and not out.with_suffix(".partial.npz").exists()
    assert r.report is None


def test_min_rows_gate_refuses_a_small_bank(small: F.Bank, small_wide: Path,
                                            tmp_path: Path) -> None:
    """Pinned AS-IS: ``MIN_ROWS = 8000`` (sized to the registered 3B bank's
    ~8,536 rows) refuses any join below it — a 3B-corpus constant in a
    model-agnostic exporter, so a newcomer's first small corpus cannot be
    exported."""
    assert T.EXPORT_MIN_ROWS == 8000
    out = tmp_path / "small.npz"
    r = run_export(export_argv(small, small_wide, out, z_dim=small.z_dim), out)
    assert r.rc == 1 and not out.exists()


def test_a_z_dim_that_disagrees_with_the_artifacts_refuses(
        export_bank: F.Bank, export_wide: Path, tmp_path: Path) -> None:
    """``--z-dim`` is only an assertion: the width comes from the wide map's
    v3_mu. A value that disagrees -> exit 1, no artifact (2713, the 3B width,
    on a 72-d bank is exactly that case)."""
    out = tmp_path / "wrongzdim.npz"
    r = run_export(export_argv(export_bank, export_wide, out, "--z-dim", "2713"), out)
    assert r.rc == 1 and not out.exists()


def test_z_dim_is_derived_from_the_artifacts(export_bank: F.Bank, export_wide: Path,
                                             export_default: Run, tmp_path: Path) -> None:
    """With no --z-dim the export must succeed and equal the --z-dim 72 export."""
    out = tmp_path / "derived.npz"
    r = run_export(export_argv(export_bank, export_wide, out), out)
    assert r.rc == 0
    assert F.sha256(out) == F.sha256(export_default.out)


# ── leaving the registered point ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def export_offpoint(export_bank: F.Bank, export_wide: Path, small_fit: Path,
                    tmp_path_factory) -> Run:
    """λ=1e3, FULL rank (the alternative operating point the fit grid offers),
    with a floor and THIS model's
    held-out report (the small bank's fit report stands in for it)."""
    out = tmp_path_factory.mktemp("offpoint") / "full.npz"
    return run_export(export_argv(
        export_bank, export_wide, out, "--lam", "1000", "--rank", "0",
        "--selfcheck-floor", "0.5", "--heldout-report", small_fit / "v1a_report.json",
        z_dim=export_bank.z_dim), out)


def test_offpoint_export_is_banked_as_an_override(export_offpoint: Run,
                                                  export_bank: F.Bank,
                                                  small_fit: Path) -> None:
    """Off the registered point: rank 0 resolves to min(W.shape) = 72 (full),
    meta marks it unregistered with grid_key/full_rank/selfcheck_floor, and the
    report's this_cell_heldout_top1 comes from --heldout-report's grid."""
    r = export_offpoint
    assert r.rc == 0
    meta = json.loads(str(npz_dict(r.out)["meta"]))
    op = meta["operating_point"]
    assert op["registered"] is False and op["rank"] == 72 == export_bank.z_dim
    assert op["grid_key"] == "dz|raw|lam1000|r0" and op["target_key"] == "raw"
    assert meta["selfcheck_floor"] == 0.5 and "full_rank" in meta
    rep = r.report
    assert rep["operating_point"]["registered_point"] is False
    assert rep["operating_point"]["full_rank"] is True
    assert rep["operating_point"]["code_dim_served"] == 72
    grid = json.loads((small_fit / "v1a_report.json").read_text())["grid"]
    assert rep["self_check"]["this_cell_heldout_top1"] == grid["dz|raw|lam1000|r0"]["retrieval_top1"]
    # meta carries THIS model's figure (from --heldout-report), not the 3B literal
    assert op["overridden_from_registered"]["heldout_top1_of_THIS_cell"] == \
        grid["dz|raw|lam1000|r0"]["retrieval_top1"]


def test_offpoint_meta_heldout_comes_from_this_models_report(
        export_offpoint: Run, small_fit: Path) -> None:
    meta = json.loads(str(npz_dict(export_offpoint.out)["meta"]))
    grid = json.loads((small_fit / "v1a_report.json").read_text())["grid"]
    assert meta["operating_point"]["overridden_from_registered"][
        "heldout_top1_of_THIS_cell"] == grid["dz|raw|lam1000|r0"]["retrieval_top1"]


def test_sitenorm_changes_only_the_regression_target(
        export_default: Run, export_bank: F.Bank, export_wide: Path, tmp_path: Path) -> None:
    """``--target sitenorm`` (with the floor it requires): same join, same dz,
    same copied v3_*/bins_*/norm_ref; W is the ridge onto per-site unit-norm
    deltas (build_target), meta says so, and the point is unregistered."""
    out = tmp_path / "sn.npz"
    r = run_export(export_argv(export_bank, export_wide, out, "--target", "sitenorm",
                               "--selfcheck-floor", "0.5", z_dim=export_bank.z_dim), out)
    assert r.rc == 0
    a, d = npz_dict(out), npz_dict(export_default.out)
    for k in ("v3_mu", "v3_sd", "v3_dead", "bins_mu", "bins_sd", "bins_dead", "norm_ref"):
        assert a[k].tobytes() == d[k].tobytes(), k
    meta = json.loads(str(a["meta"]))
    assert meta["operating_point"]["registered"] is False
    assert meta["operating_point"]["grid_key"] == "dz|sitenorm|lam10000|r64"
    assert "target_normalization" in meta
    z, h, fans = _kept(export_bank, member_fan=False)
    y = T.build_target(F.loo_center(h, fans), "sitenorm", len(export_bank.sites))
    w_ref, _, _ = F.ridge_reference(F.loo_center(z, fans), y, 1e4)
    w_ref = F.truncate_reference(w_ref, 64)
    assert np.allclose(_w_of(a, export_bank.z_dim), w_ref, atol=1e-9 * np.abs(w_ref).max())


def test_export_code_basis_is_canonically_oriented(export_default: Run) -> None:
    """The exported Vt rows (the code basis) follow SIGN_CONVENTION, and meta
    records it. An SVD's axis signs are arbitrary, so without this a re-export
    of the same map can flip half its code axes; canonicalised, a re-export
    agrees with the served map to float64 roundoff."""
    from pleroma.map.svd import SIGN_CONVENTION, is_canonical
    a = npz_dict(export_default.out)
    assert is_canonical(a["W_Vt"])
    orient = json.loads(str(a["meta"]))["svd_orientation"]
    assert orient["convention"] == SIGN_CONVENTION
    assert orient["ambiguous_components"] == []


# ── fold-in: the export computes the shelf stats itself ──────────────────────

def test_shelf_stats_computed_equal_the_wide_maps_bytes(small: F.Bank,
                                                        small_wide: Path) -> None:
    """`compute_shelf` is the code `wide` runs, so the stats a folded export
    computes are the wide map's arrays byte for byte (dtype included)."""
    from pleroma.map.build.shelf import ShelfStats, compute_shelf
    got = compute_shelf(small.pairs_dir, small.levers, small.bins,
                        small.discriminants, small.levers).stats
    ref = ShelfStats.from_wide_map(small_wide)
    for k in ("v3_mu", "v3_sd", "v3_dead", "bins_mu", "bins_sd", "bins_dead", "norm_ref"):
        a, b = getattr(got, k), getattr(ref, k)
        assert a.dtype == b.dtype and np.array_equal(a, b), k
    assert got.sites == ref.sites
    assert got.discriminants_sha256 == ref.discriminants_sha256
    assert got.bins_config == ref.bins_config


@pytest.fixture(scope="module")
def fold_bank(tmp_path_factory) -> tuple[F.Bank, Path]:
    """A small bank WITH raw signatures and the format's 420 bins, plus the
    real `wide` fit over it (the export needs both; `small` has 10 bins)."""
    bank = F.build_bank(tmp_path_factory.mktemp("foldbank"), n_prompts=80, z_dim=12,
                        bins_dim=T.BINS_DIM, sites=SITES, hidden=5)
    wide = bank.root / "wide.npz"
    assert T.run_fit_loom_map([
        "--pairs-dir", bank.pairs_dir, "--levers", bank.levers, "--bins", bank.bins,
        "--discriminants", bank.discriminants, "--norm-ref-bank", bank.levers,
        "--lam", "10000", "--rank", "8", "--out", wide]) == 0
    return bank, wide


def test_folded_export_equals_the_wide_map_export(fold_bank: tuple[F.Bank, Path],
                                                  tmp_path: Path) -> None:
    """The gate on the fold-in: dropping the wide map changes no array of the
    served artifact — only the provenance in meta."""
    small, small_wide = fold_bank
    extra = ["--rank", "8", "--selfcheck-floor", "0.01"]
    legacy_out, folded_out = tmp_path / "legacy.npz", tmp_path / "folded.npz"
    legacy = run_export(export_argv(small, small_wide, legacy_out, *extra), legacy_out)
    argv = [a for a in export_argv(small, small_wide, folded_out, *extra)]
    i = argv.index("--wide-map")
    argv[i:i + 2] = ["--discriminants", small.discriminants]
    folded = run_export(argv, folded_out)
    assert legacy.rc == 0 and folded.rc == 0
    a, b = npz_dict(legacy_out), npz_dict(folded_out)
    assert set(a) == set(b)
    for k in a:
        if k != "meta":
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
    src = json.loads(str(b["meta"]))["sources"]
    assert "wide_map" not in src and src["shelf"].startswith("computed")
    assert src["discriminants_sha256"] == F.sha256(small.discriminants)


def test_export_needs_exactly_one_shelf_source(small: F.Bank, small_wide: Path,
                                               tmp_path: Path) -> None:
    out = tmp_path / "x.npz"
    both = export_argv(small, small_wide, out, "--discriminants", small.discriminants)
    assert run_export(both, out).rc == 1 and not out.exists()
    neither = export_argv(small, small_wide, out)
    i = neither.index("--wide-map")
    del neither[i:i + 2]
    assert run_export(neither, out).rc == 1 and not out.exists()
