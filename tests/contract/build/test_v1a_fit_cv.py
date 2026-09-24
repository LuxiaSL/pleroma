"""v1a_fit — the registered grouped-CV validation — on a toy bank.

Runs the real ``pleroma.map.build.cv.main`` (CPU torch) with the real
fit_loom_map output as its shelf map, and pins: the registered constants, the
report/grid schema that ``v1a_export --heldout-report`` parses, WHICH rows the
join drops and why, the seeded-permutation fold assignment grouped by prompt,
that every prediction is held out and equals a numpy ridge fit on the other
folds, and that the shelf baseline standardises z ONCE.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T
from tests.contract.build.conftest import npz_dict

REPORT_KEYS = {"stage", "token", "n_gens", "n_fans", "fan_size_mean", "chance_top1",
               "folds", "fold_seed", "primary_point", "baseline_shelf", "grid",
               "baseline_v0_cv", "registered_test", "secondary_vs_v0cv",
               "fan_source"}  # fan_source: which fan key the join used


def _report(fit_dir: Path) -> dict:
    return json.loads((fit_dir / "v1a_report.json").read_text())


def _kept(small: F.Bank, fit_dir: Path) -> dict[str, np.ndarray]:
    """The fit's kept rows, re-derived from its npz, with bank ground truth."""
    d = npz_dict(fit_dir / "v1a_predicted_deltas.npz")
    row_of = {int(g): r for r, g in enumerate(small.pair_gen_ids.tolist())}
    gids = d["generation_ids"].astype(int)
    grp = d["group_index"].astype(int)
    fans = [np.nonzero(grp == g)[0] for g in np.unique(grp)]
    z = small.z_corpus[[row_of[int(g)] for g in gids]].astype(np.float64)
    h = np.stack([small.means[int(g)].reshape(-1) for g in gids]).astype(np.float64)
    prompts = np.asarray([small.member_of[int(g)].prompt_id for g in gids])
    return {**d, "gids": gids, "fans": fans, "z": z, "h": h, "prompts": prompts}


# ── constants and schema ──────────────────────────────────────────────────────

def test_registered_constants() -> None:
    """The frozen registration: the grid, primary point, folds, seeds and
    permutation count were fixed before the fit was run, and never move."""
    assert T.FIT_LAMBDAS == (1e3, 1e4, 3e4)
    assert T.FIT_RANKS == (0, 16, 32, 64, 128, 256)
    assert T.FIT_PRIMARY == {"input": "dz", "target": "raw", "lambda": 1e4, "rank": 64}
    assert (T.FIT_FOLDS, T.FIT_FOLD_SEED, T.FIT_PERM_SEED) == (5, 20260921, 20260921)
    assert T.FIT_N_PERMS == 20000 and T.MIN_FAN == 3


def test_report_schema_and_grid_keys(small_fit: Path) -> None:
    """The report shape v1a_export.heldout_from_fit_report parses: 72 grid cells
    keyed ``{dz|rawz}|{raw|sitenorm}|lam{λ:g}|r{r}``, each with retrieval_top1
    and pred_cos_vs_own_targets; ``registered_test.v1a_top1`` equals the
    primary cell (the consistency heldout_from_fit_report enforces)."""
    r = _report(small_fit)
    assert set(r) == REPORT_KEYS
    assert r["stage"] == "v1a_fit" and r["token"] == "v1a-fit-001"
    assert r["primary_point"] == T.FIT_PRIMARY
    want = {f"{i}|{t}|lam{lam:g}|r{k}" for i in ("dz", "rawz") for t in ("raw", "sitenorm")
            for lam in T.FIT_LAMBDAS for k in T.FIT_RANKS}
    assert set(r["grid"]) == want and len(want) == 72
    assert T.grid_key(1e4, 64) == "dz|raw|lam10000|r64" in r["grid"]
    for cell in r["grid"].values():
        assert set(cell) == {"retrieval_top1", "pred_cos_vs_own_targets"}
    assert r["registered_test"]["v1a_top1"] == r["grid"]["dz|raw|lam10000|r64"]["retrieval_top1"]
    assert r["registered_test"]["n_perms"] == 20000
    assert set(r["registered_test"]) == {"statistic", "observed", "v1a_top1", "shelf_top1",
                                         "n_perms", "one_sided_p", "verdict"}
    # and the export's own parser accepts it (the --heldout-report chain)
    got = T.heldout_from_fit_report(small_fit / "v1a_report.json", 1e4, 64)
    assert got["registered_heldout_top1"] == r["grid"]["dz|raw|lam10000|r64"]["retrieval_top1"]


def test_predicted_deltas_npz_schema(small: F.Bank, small_fit: Path) -> None:
    """``<fit>/v1a_predicted_deltas.npz`` (the ladder levers' source)."""
    d = npz_dict(small_fit / "v1a_predicted_deltas.npz")
    assert set(d) == {"pred", "measured", "corpus_keys", "generation_ids", "group_index",
                      "fold_idx", "sites"}
    n, w = 1144, len(small.sites) * small.hidden
    assert d["pred"].shape == d["measured"].shape == (n, w)
    assert d["pred"].dtype == d["measured"].dtype == np.float32
    assert d["sites"].tolist() == small.sites


# ── the row filter ────────────────────────────────────────────────────────────

def test_row_filter_drops_nonmembers_no_lever_fans_and_small_fans(
        small: F.Bank, small_fit: Path, small_wide: Path) -> None:
    """Pins WHICH rows the registered join keeps and why:

      1,281 pairs
      -   1  not in the lever file (``j is None``)
      - 128  members of the 16 [7,1] fans: ``m_group < 0`` — the filter
             that drops 4,344 of model C's 14,440 gens
      -   2  the sparse lever fan's 2 pairs: < MIN_FAN = 3
      = 1,144 gens in 143 fans of 8.

    fit_loom_map on the same bank joins 1,146 (no MIN_FAN), so the shelf map
    and the CV are fit on different row sets — pinned, not endorsed.
    """
    r = _report(small_fit)
    assert (r["n_gens"], r["n_fans"]) == (1144, 143)
    assert r["fan_size_mean"] == 8.0 and r["chance_top1"] == 0.125
    mo = small.member_of
    want = {m.gen_id for m in small.members
            if m.has_pair and m.group >= 0 and m.fan != 1}      # fan 1 = sparse
    got = set(npz_dict(small_fit / "v1a_predicted_deltas.npz")["generation_ids"].tolist())
    assert got == want
    assert small.extra_gen_id not in got
    assert not any(mo[g].group < 0 for g in got)
    assert json.loads(str(npz_dict(small_wide)["meta"]))["n_rows"] == 1146


def test_fit_can_cross_validate_the_served_member_fan_join(
        small: F.Bank, small_wide: Path, tmp_path: Path) -> None:
    """Under member_fan the [7,1] fans' 128 gens are IN: 1,144 + 128 = 1,272."""
    rc = T.run_v1a_fit([
        "--pairs-dir", small.pairs_dir, "--levers", small.levers,
        "--hiddens", small.hiddens, "--bins", small.bins, "--shelf-map", small_wide,
        "--out-dir", tmp_path, "--device", "cpu", "--fan-source", "member_fan"])
    assert rc == 0
    assert _report(tmp_path)["n_gens"] == 1272


# ── folds (seeded-permutation-mod, grouped by prompt) ─────────────────────────

def test_fold_assignment_is_seeded_permutation_over_sorted_prompts(
        small: F.Bank, small_fit: Path) -> None:
    """fold(prompt) = position of the prompt in rng(20260921).permutation over
    sorted(unique prompts), mod 5 — NOT regress's sorted-index-mod scheme
    (the two give different folds). Both waves of a prompt share a fold, so no
    fan straddles folds and no near-twin of a test row is trained on."""
    k = _kept(small, small_fit)
    uniq = sorted(set(k["prompts"].tolist()))
    order = np.random.default_rng(T.FIT_FOLD_SEED).permutation(len(uniq))
    fold_of = {uniq[int(j)]: i % T.FIT_FOLDS for i, j in enumerate(order)}
    assert k["fold_idx"].tolist() == [fold_of[p] for p in k["prompts"]]
    for p in uniq:
        assert len(set(k["fold_idx"][k["prompts"] == p].tolist())) == 1
    for sel in k["fans"]:
        assert len(set(k["fold_idx"][sel].tolist())) == 1
    assert np.bincount(list(fold_of.values())).tolist() == [16] * 5
    sorted_mod = {p: i % T.FIT_FOLDS for i, p in enumerate(uniq)}
    assert fold_of != sorted_mod


def test_heldout_predictions_are_numpy_ridge_fit_on_the_other_folds(
        small: F.Bank, small_fit: Path) -> None:
    """Every primary prediction is HELD OUT: for test fold f,
    pred = (dz_te - mu_tr) W_tr + mu_y_tr with W_tr the centered closed-form
    ridge (λ = 1e4) on folds != f (``cv_sweep``, torch float64; stored
    float32). Rank 64 >= min(W.shape) = 12 here, so the full-rank branch.
    Also pins ``measured`` = LOO fan-centred mean hiddens."""
    k = _kept(small, small_fit)
    dz = F.loo_center(k["z"], k["fans"])
    d_raw = F.loo_center(k["h"], k["fans"])
    assert np.allclose(k["measured"], d_raw, atol=1e-5 * np.abs(d_raw).max())
    for f in range(T.FIT_FOLDS):
        te, tr = k["fold_idx"] == f, k["fold_idx"] != f
        w, mu_x, mu_y = F.ridge_reference(dz[tr], d_raw[tr], 1e4)
        want = (dz[te] - mu_x) @ w + mu_y
        assert np.allclose(k["pred"][te], want, atol=1e-5 * np.abs(want).max(), rtol=0)
        # and it is NOT the in-sample fit (the held-out claim has teeth here)
        w_all, mx, my = F.ridge_reference(dz, d_raw, 1e4)
        assert not np.allclose(k["pred"][te], (dz[te] - mx) @ w_all + my,
                               atol=1e-6 * np.abs(want).max(), rtol=0)


# ── the shelf baseline ────────────────────────────────────────────────────────

def _shelf_top1(k: dict, small: F.Bank, wide: dict, twice: bool) -> float:
    b = np.stack([small.bins_of[int(g)] for g in k["gids"]])
    dead = wide["bins_dead"]
    bs = (b - wide["bins_mu"].astype(np.float64)) / np.where(dead, 1.0, wide["bins_sd"])
    bs[:, dead] = 0.0
    z = k["z"]
    if twice:        # the double-standardisation bug: z_corpus standardised again
        z = (z - wide["v3_mu"]) / np.where(wide["v3_dead"], 1.0, wide["v3_sd"])
    x = np.hstack([z, bs])
    w = (wide["W_U"].astype(np.float64) * wide["W_S"]) @ wide["W_Vt"]
    pred = F.loo_center(x, k["fans"]) @ w
    return F.top1_reference(pred, F.loo_center(k["h"], k["fans"]), k["fans"])


def test_shelf_baseline_standardises_z_once(small: F.Bank, small_fit: Path,
                                            small_wide: Path) -> None:
    """The shelf baseline feeds the pairs z
    (already z_corpus) to the map UNCHANGED and standardises only bins with
    the map's bins_mu/bins_sd. The report's number equals the single-
    standardised reference, and on this bank the double-standardised one is
    different — so this test would catch the bug coming back."""
    k = _kept(small, small_fit)
    wide = npz_dict(small_wide)
    got = _report(small_fit)["baseline_shelf"]["retrieval_top1"]
    once = _shelf_top1(k, small, wide, twice=False)
    twice = _shelf_top1(k, small, wide, twice=True)
    assert got == pytest.approx(once, abs=1e-12)
    assert abs(twice - once) > 1.0 / len(k["gids"])
    assert _report(small_fit)["registered_test"]["shelf_top1"] == round(once, 4)


def test_shelf_baseline_pred_function_contract() -> None:
    """``shelf_baseline_pred(z_corpus, raw_bins, shelf, fans)`` =
    fan_center([z | bins_std]) @ U S Vt, dead bins -> 0; refuses a z whose
    width is not the map's z block."""
    rng = np.random.default_rng(3)
    zd, bd, out, r = 5, 3, 6, 2
    shelf = {"v3_mu": rng.standard_normal(zd).astype(np.float32),
             "v3_sd": np.ones(zd, np.float32), "v3_dead": np.zeros(zd, bool),
             "bins_mu": np.array([1.0, 2.0, 3.0], np.float32),
             "bins_sd": np.array([2.0, 0.0, 4.0], np.float32),
             "bins_dead": np.array([False, True, False]),
             "W_U": rng.standard_normal((zd + bd, r)).astype(np.float32),
             "W_S": np.array([2.0, 1.0], np.float32),
             "W_Vt": rng.standard_normal((r, out)).astype(np.float32)}
    z = rng.standard_normal((6, zd))
    b = rng.standard_normal((6, bd)) * 5
    fans = [np.arange(3), np.arange(3, 6)]
    bs = (b - [1.0, 2.0, 3.0]) / [2.0, 1.0, 4.0]
    bs[:, 1] = 0.0
    w = (shelf["W_U"].astype(np.float64) * shelf["W_S"]) @ shelf["W_Vt"]
    want = F.loo_center(np.hstack([z, bs]), fans) @ w
    got = T.shelf_baseline_pred(z, b, shelf, fans)
    assert np.allclose(got, want, atol=1e-10)
    with pytest.raises(ValueError):
        T.shelf_baseline_pred(z[:, :4], b, shelf, fans)


def test_check_pairs_are_z_corpus() -> None:
    """corpus -> accept; anything else -> refusal string; unrecorded -> accept
    with a warning."""
    assert T.check_pairs_are_z_corpus({"params": {"standardize": "corpus"}}) is None
    assert isinstance(T.check_pairs_are_z_corpus({"params": {"standardize": "full"}}), str)
    assert T.check_pairs_are_z_corpus({}) is None


def test_refuses_a_pairs_dir_not_standardised_by_corpus(
        small: F.Bank, small_wide: Path, tmp_path: Path) -> None:
    """main returns 1 (no report) on a ``--standardize full`` pairs dir."""
    pdir = tmp_path / "pairs"
    shutil.copytree(small.pairs_dir, pdir)
    man = json.loads((pdir / "pairs_manifest.json").read_text())
    man["params"]["standardize"] = "full"
    (pdir / "pairs_manifest.json").write_text(json.dumps(man))
    out = tmp_path / "out"
    rc = T.run_v1a_fit(["--pairs-dir", pdir, "--levers", small.levers,
                        "--hiddens", small.hiddens, "--bins", small.bins,
                        "--shelf-map", small_wide, "--out-dir", out, "--device", "cpu"])
    assert rc == 1
    assert not (out / "v1a_report.json").exists()


# ── the hardcoded 4 sites ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def three_site(tmp_path_factory) -> tuple[F.Bank, Path]:
    bank = F.build_bank(tmp_path_factory.mktemp("three"), n_prompts=80, z_dim=12,
                        bins_dim=10, sites=[3, 5, 7], hidden=5, seed=4)
    wide = bank.root / "wide.npz"
    assert T.run_fit_loom_map([
        "--pairs-dir", bank.pairs_dir, "--levers", bank.levers, "--bins", bank.bins,
        "--discriminants", bank.discriminants, "--norm-ref-bank", bank.levers,
        "--lam", "10000", "--rank", "8", "--out", wide]) == 0
    return bank, wide


def test_fit_runs_on_a_three_site_install(three_site, tmp_path: Path) -> None:
    """Sites are an install property (70B: 22/42/52/62; 8B: 9/17/21/25), not 4."""
    bank, wide = three_site
    rc = T.run_v1a_fit(["--pairs-dir", bank.pairs_dir, "--levers", bank.levers,
                        "--hiddens", bank.hiddens, "--bins", bank.bins,
                        "--shelf-map", wide, "--out-dir", tmp_path, "--device", "cpu"])
    assert rc == 0
