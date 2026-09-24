"""The shelf baseline in `v1a_fit` standardizes z EXACTLY ONCE.

`load_pairs(...).z` is already z_corpus (build_pairs --standardize corpus;
fit_loom_map refuses to fit unless that holds), so applying the shelf map's
`(z - v3_mu)/v3_sd` to it again before projecting standardizes twice and
understates the shelf baseline — which overstates v1a's margin over it.
Pinned here, end to end through `main` on a small synthetic bank:

  * the reported shelf top-1 equals an INDEPENDENT single-standardized
    computation, and the fixture is built so that the double-standardized
    number is far from it — so this test FAILS on a double standardization;
  * v1a's own numbers (every grid cell, the registered v1a top-1, the CV-refit
    fairness baseline) are IDENTICAL whichever shelf input is used — the fix
    moves the baseline and nothing else;
  * a pairs dir that was not built with --standardize corpus is refused.

CPU only, no torch: `cv_sweep` is replaced by a numpy mirror of the same
grouped ridge + SVD, and `load_pairs` is monkeypatched at its module (main
imports it at call time). Hiddens, levers, bins and the shelf map are real npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pleroma.map.build import cv as v1a_fit

ZD = 12            # z_corpus width
BD = 3             # bins width
SITES = [2, 5, 7, 9]
HID = 3            # per-site hidden width -> 12-d flattened hiddens
N_FANS = 130       # x 8 members = 1,040 gens (main refuses < 1,000)
FAN = 8
CORPUS = "corpA"


def _numpy_cv_sweep(x, y, fold_idx, lambdas, ranks, device):
    """`v1a_fit.cv_sweep`, line for line, in numpy float64."""
    preds = {(lam, r): np.empty_like(y, dtype=np.float32)
             for lam in lambdas for r in ranks}
    d = x.shape[1]
    for f in range(v1a_fit.FOLDS):
        te = np.nonzero(fold_idx == f)[0]
        tr = np.nonzero(fold_idx != f)[0]
        mu_x, mu_y = x[tr].mean(axis=0), y[tr].mean(axis=0)
        xtr, ytr, xte = x[tr] - mu_x, y[tr] - mu_y, x[te] - mu_x
        gram, xty = xtr.T @ xtr, xtr.T @ ytr
        for lam in lambdas:
            w = np.linalg.solve(gram + lam * np.eye(d), xty)
            u, s, vt = np.linalg.svd(w, full_matrices=False)
            for r in ranks:
                if r <= 0 or r >= min(w.shape):
                    p = xte @ w + mu_y
                else:
                    p = ((xte @ u[:, :r]) * s[:r]) @ vt[:r] + mu_y
                preds[(lam, r)][te] = p.astype(np.float32)
    return preds


def _double_standardized_pred(z, bins, shelf, fans):
    """The WRONG shelf path, as a reference: z_corpus standardized a SECOND
    time with the map's v3 stats."""
    v3_mu = np.asarray(shelf["v3_mu"], dtype=np.float64)
    v3_sd = np.asarray(shelf["v3_sd"], dtype=np.float64)
    v3_dead = np.asarray(shelf["v3_dead"], dtype=bool)
    b_mu = np.asarray(shelf["bins_mu"], dtype=np.float64)
    b_sd = np.asarray(shelf["bins_sd"], dtype=np.float64)
    b_dead = np.asarray(shelf["bins_dead"], dtype=bool)
    z_std = (z - v3_mu) / np.where(v3_dead, 1.0, v3_sd)
    z_std[:, v3_dead] = 0.0
    b_std = (bins - b_mu) / np.where(b_dead, 1.0, b_sd)
    b_std[:, b_dead] = 0.0
    x_map = np.hstack([z_std, b_std])
    wu = np.asarray(shelf["W_U"], dtype=np.float64)
    ws = np.asarray(shelf["W_S"], dtype=np.float64)
    wvt = np.asarray(shelf["W_Vt"], dtype=np.float64)
    return v1a_fit.fan_center(x_map, fans) @ (wu * ws[None, :]) @ wvt


@pytest.fixture()
def bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    n = N_FANS * FAN
    gids = np.arange(1000, 1000 + n)
    grp = np.repeat(np.arange(N_FANS), FAN)
    z = rng.standard_normal((n, ZD)).astype(np.float32)   # z_corpus: ~N(0,1)
    bins = rng.standard_normal((n, BD)) * 3.0 + 5.0         # raw bins

    # hiddens: a linear function of z_corpus plus noise. The shelf map IS that
    # linear function (bins rows zero), so fed z_corpus it is near-oracle.
    a = rng.standard_normal((ZD, len(SITES) * HID))
    h = z.astype(np.float64) @ a + 1.6 * rng.standard_normal((n, len(SITES) * HID))
    w_full = np.vstack([a, np.zeros((BD, a.shape[1]))])
    u, s, vt = np.linalg.svd(w_full, full_matrices=False)
    # v3 stats far from identity (as at 3B/8B), so a second standardization
    # visibly re-weights coordinates: 1/v3_sd spans 1/20 .. 1/0.05
    v3_sd = np.geomspace(0.05, 20.0, ZD)
    shelf = {
        "v3_mu": rng.standard_normal(ZD), "v3_sd": v3_sd,
        "v3_dead": np.zeros(ZD, dtype=bool),
        "bins_mu": bins.mean(axis=0), "bins_sd": bins.std(axis=0),
        "bins_dead": np.zeros(BD, dtype=bool),
        "W_U": u, "W_S": s, "W_Vt": vt,
    }
    shelf_path = tmp_path / "shelf.npz"
    np.savez(shelf_path, **shelf)

    hid_path = tmp_path / "hiddens.npz"
    np.savez(hid_path, means=h.reshape(n, len(SITES), HID),
             generation_ids=gids, corpus_keys=np.array([CORPUS] * n),
             sites=np.array(SITES))
    lev_path = tmp_path / "levers.npz"
    np.savez(lev_path, member_corpus_keys=np.array([CORPUS] * n),
             member_generation_ids=gids, member_group_index=grp,
             member_clusters=(np.arange(n) % 2),
             levers=rng.standard_normal((N_FANS, len(SITES), HID)),
             sites=np.array(SITES))
    bins_path = tmp_path / "bins.npz"
    np.savez(bins_path, run_dir=np.array([f"/x/{CORPUS}"] * n),
             generation_id=gids, features=bins)

    meta = {"params": {"standardize": "corpus"}}
    pairs = [SimpleNamespace(run_dir=f"/x/{CORPUS}", generation_id=int(gids[i]),
                             row=i, prompt_id=f"p{grp[i]}")
             for i in range(n)]
    loaded = SimpleNamespace(pairs=pairs, z=z, meta=meta)

    from pleroma.map.build import pairs as build_pairs
    monkeypatch.setattr(build_pairs, "load_pairs", lambda _d: loaded)
    monkeypatch.setattr(v1a_fit, "cv_sweep", _numpy_cv_sweep)
    fans = [np.nonzero(grp == g)[0] for g in range(N_FANS)]
    return {"tmp": tmp_path, "z": z.astype(np.float64), "bins": bins, "h": h,
            "shelf": shelf, "fans": fans, "meta": meta,
            "argv": ["v1a_fit", "--pairs-dir", str(tmp_path / "pairs"),
                     "--levers", str(lev_path), "--hiddens", str(hid_path),
                     "--bins", str(bins_path), "--shelf-map", str(shelf_path),
                     "--device", "cpu"]}


def _run(bank: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tag: str) -> dict:
    out = bank["tmp"] / tag
    monkeypatch.setattr(sys, "argv", bank["argv"] + ["--out-dir", str(out)])
    assert v1a_fit.main() == 0
    return json.loads((out / "v1a_report.json").read_text())


def test_shelf_baseline_is_standardized_once(bank, monkeypatch) -> None:
    """★ FAILS on the pre-fix code: the report's shelf number is the
    single-standardized projection, computed here independently."""
    d_raw = v1a_fit.fan_center(bank["h"], bank["fans"])
    b = bank["bins"]
    b_std = (b - b.mean(axis=0)) / b.std(axis=0)
    w = (bank["shelf"]["W_U"] * bank["shelf"]["W_S"][None, :]) @ bank["shelf"]["W_Vt"]
    once = v1a_fit.fan_center(np.hstack([bank["z"], b_std]), bank["fans"]) @ w
    top1_once = float(v1a_fit.retrieval(once, d_raw, bank["fans"]).mean())
    twice = _double_standardized_pred(bank["z"], b, bank["shelf"], bank["fans"])
    top1_twice = float(v1a_fit.retrieval(twice, d_raw, bank["fans"]).mean())
    # the fixture must discriminate, or this test proves nothing
    assert top1_once - top1_twice > 0.05, (top1_once, top1_twice)

    rep = _run(bank, monkeypatch, "fixed")
    got = rep["baseline_shelf"]["retrieval_top1"]
    assert got == pytest.approx(top1_once, abs=1e-12)
    assert rep["registered_test"]["shelf_top1"] == round(top1_once, 4)


def test_the_fix_moves_the_baseline_and_nothing_else(bank, monkeypatch) -> None:
    """v1a's grid, its registered top-1 and the CV-refit fairness baseline are
    byte-identical under the old (double) and new (single) shelf input."""
    new = _run(bank, monkeypatch, "new")
    monkeypatch.setattr(v1a_fit, "shelf_baseline_pred", _double_standardized_pred)
    old = _run(bank, monkeypatch, "old")

    assert new["grid"] == old["grid"]
    assert new["baseline_v0_cv"] == old["baseline_v0_cv"]
    assert new["secondary_vs_v0cv"] == old["secondary_vs_v0cv"]
    assert new["registered_test"]["v1a_top1"] == old["registered_test"]["v1a_top1"]
    assert (new["baseline_shelf"]["retrieval_top1"]
            != old["baseline_shelf"]["retrieval_top1"])
    a = np.load(bank["tmp"] / "new" / "v1a_predicted_deltas.npz")
    b = np.load(bank["tmp"] / "old" / "v1a_predicted_deltas.npz")
    assert np.array_equal(a["pred"], b["pred"])


def test_shelf_baseline_pred_ignores_v3_stats_entirely(bank) -> None:
    """z_corpus enters the map unchanged: the banked v3 stats cannot move it."""
    s2 = dict(bank["shelf"], v3_mu=np.full(ZD, 99.0), v3_sd=np.full(ZD, 1e-3))
    p1 = v1a_fit.shelf_baseline_pred(bank["z"], bank["bins"], bank["shelf"], bank["fans"])
    p2 = v1a_fit.shelf_baseline_pred(bank["z"], bank["bins"], s2, bank["fans"])
    assert np.array_equal(p1, p2)


def test_shelf_baseline_pred_refuses_a_width_mismatch(bank) -> None:
    with pytest.raises(ValueError):
        v1a_fit.shelf_baseline_pred(bank["z"][:, :-1], bank["bins"],
                                    bank["shelf"], bank["fans"])


def test_non_corpus_pairs_are_refused(bank, monkeypatch) -> None:
    bank["meta"]["params"]["standardize"] = "full"
    monkeypatch.setattr(sys, "argv",
                        bank["argv"] + ["--out-dir", str(bank["tmp"] / "r")])
    assert v1a_fit.main() == 1
    assert not (bank["tmp"] / "r" / "v1a_report.json").exists()


def test_standardize_guard_cases() -> None:
    assert v1a_fit.check_pairs_are_z_corpus({"params": {"standardize": "corpus"}}) is None
    assert v1a_fit.check_pairs_are_z_corpus({"params": {}}) is None   # legacy: warn
    assert v1a_fit.check_pairs_are_z_corpus({}) is None
    assert "full" in v1a_fit.check_pairs_are_z_corpus(
        {"params": {"standardize": "full"}})
