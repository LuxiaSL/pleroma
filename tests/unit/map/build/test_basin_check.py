"""Contract tests for stage 4 against planted clusters and a fake discriminants npz.

The fixture builds three prompts of three classes, and a parameter says which of
them gets two planted basins:

    cc1  convergent_control   the calibration arm
    sf1  stance_fork          a fork class
    mf1  method_fork          a fork class

In the NORMAL fixture ``sf1`` is the one with two planted basins (four signatures
at +v, four at -v along the first discriminant direction) and the control is tight,
so a working 2-means probe must split sf1 4/4 and the calibration line must stay
quiet. In the REVERSED fixture the control is the one that splits, which is exactly
the "the spread measure is reading noise" case — the calibration line must fire.

The name-mismatch abort gets its own tests, because it is the assert every other
number in the stage depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.map.build import basin as bc

DIM = 6
NAMES: list[str] = [f"f{i}" for i in range(DIM)]
MODES: list[str] = ["m0", "m1", "m2"]
SEEDS_PER_PROMPT = 8

#: Per-prompt content component, identical across that prompt's seeds so it cancels
#: in Δ, but different BETWEEN prompts so between-prompt distance is not degenerate.
PROMPTS: list[tuple[str, str, list[float]]] = [
    ("cc1", "convergent_control", [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    ("sf1", "stance_fork", [0.0, 0.0, 1.0, -1.0, 1.0, -1.0]),
    ("mf1", "method_fork", [0.0, 0.0, -1.0, 1.0, -1.0, 1.0]),
    ("og1", "open_generative", [0.0, 0.0, -1.0, -1.0, 1.0, 1.0]),
]

#: Which shape each prompt's seeds are planted with, per fixture world.
NORMAL_SHAPES: dict[str, str] = {
    "cc1": "tight", "sf1": "bimodal", "mf1": "tight", "og1": "outlier",
}
REVERSED_SHAPES: dict[str, str] = {
    "cc1": "bimodal", "sf1": "tight", "mf1": "tight", "og1": "tight",
}


def _write_discriminants(path: Path, names: list[str]) -> Path:
    """A 2-discriminant, 3-mode fake: W picks out dims 0 and 1."""
    w = np.zeros((2, len(names)), dtype=np.float64)
    w[0, 0] = 1.0
    w[1, 1] = 1.0
    np.savez_compressed(
        path,
        labels=np.array(MODES),
        FULL_W=w,
        FULL_mean=np.zeros(len(names), dtype=np.float64),
        FULL_scale=np.ones(len(names), dtype=np.float64),
        FULL_names=np.array(names),
        FULL_centroid_m0=np.array([1.0, 0.0]),
        FULL_centroid_m1=np.array([-1.0, 0.0]),
        FULL_centroid_m2=np.array([0.0, 1.0]),
    )
    return path


def _write_run(root: Path, shapes: dict[str, str]) -> Path:
    """Plant each prompt's seeds in the shape ``shapes`` names for it.

    ``bimodal``  two balanced basins at +v / -v along the first discriminant axis
    ``outlier``  one seed far away, the other seven together (moves, but 7/1)
    ``tight``    everything within numerical noise of the content vector
    """
    (root / "signatures").mkdir(parents=True)
    cells: list[dict[str, Any]] = []
    gid = 0
    for prompt_idx, (pid, cls, content) in enumerate(PROMPTS):
        shape = shapes[pid]
        for i in range(SEEDS_PER_PROMPT):
            vec = np.array(content, dtype=np.float32)
            if shape == "bimodal":
                sign = 1.0 if i < SEEDS_PER_PROMPT // 2 else -1.0
                vec[0] = sign * (1.0 + 0.01 * i)
            elif shape == "outlier" and i == SEEDS_PER_PROMPT - 1:
                vec[0] = 10.0
            else:
                vec[1] = 1e-4 * (i - (SEEDS_PER_PROMPT - 1) / 2)
            _save_sig(root, gid, vec)
            cells.append(_cell(gid, pid, cls, prompt_idx, i))
            gid += 1
    (root / "cells.json").write_text(json.dumps(cells))
    return root


def _save_sig(root: Path, gid: int, vec: np.ndarray) -> None:
    _save_sig_with_names(root, gid, vec.astype(np.float32), NAMES)


def _save_sig_with_names(root: Path, gid: int, vec: np.ndarray, names: list[str]) -> None:
    np.savez_compressed(
        root / "signatures" / f"gen_{gid:03d}.npz",
        features=np.asarray(vec, dtype=np.float32),
        feature_names=np.array(names),
    )


def _cell(gid: int, pid: str, cls: str, prompt_idx: int, seed_idx: int) -> dict[str, Any]:
    return {
        "generation_id": gid,
        "prompt_id": pid,
        "prompt_class": cls,
        "prompt": f"synthetic prompt {pid}",
        "prompt_idx": prompt_idx,
        "seed_idx": seed_idx,
        "prompt_length": 3,
    }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Normal world: a fork class forks, the control is tight."""
    return _write_run(tmp_path / "replay", NORMAL_SHAPES)


@pytest.fixture()
def reversed_run_dir(tmp_path: Path) -> Path:
    """Broken world: the CONTROL is the one that splits."""
    return _write_run(tmp_path / "replay_reversed", REVERSED_SHAPES)


@pytest.fixture()
def discriminants(tmp_path: Path) -> Path:
    return _write_discriminants(tmp_path / "factor_directions_fake.npz", NAMES)


def _report(run_dir: Path, discriminants: Path, **kwargs: Any) -> dict[str, Any]:
    sigs = bc.load_signatures(run_dir)
    fd = bc.load_factor_directions_local(discriminants)
    return bc.analyse(
        sigs,
        fd,
        bc.mode_projection_local,
        kwargs.get("min_minority_frac", 0.25),
        kwargs.get("min_modes", 2),
        kwargs.get("control_class", bc.DEFAULT_CONTROL_CLASS),
        kwargs.get("control_margin", 0.25),
        kwargs.get("class_go_fraction", 0.5),
        kwargs.get("min_spread_ratio", 0.10),
    )


def _prompt(report: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    return next(p for p in report["prompts"] if p["prompt_id"] == prompt_id)


def _class(report: dict[str, Any], prompt_class: str) -> dict[str, Any]:
    return next(c for c in report["classes"] if c["prompt_class"] == prompt_class)


# ── loading ───────────────────────────────────────────────────────────────────


def test_signatures_load_with_a_runtime_dimension(run_dir: Path) -> None:
    sigs = bc.load_signatures(run_dir)
    assert sigs.n_gens == len(PROMPTS) * SEEDS_PER_PROMPT
    assert sigs.feature_names == NAMES
    assert sigs.features.shape == (len(PROMPTS) * SEEDS_PER_PROMPT, DIM)
    assert sigs.cells[0].prompt_id == "cc1"
    assert sigs.cells[0].prompt_class == "convergent_control"


def test_divergent_feature_names_across_gens_are_rejected(run_dir: Path) -> None:
    _save_sig_with_names(
        run_dir, 0, np.zeros(DIM, dtype=np.float32), [f"g{i}" for i in range(DIM)]
    )
    with pytest.raises(AssertionError, match="do not live in one space"):
        bc.load_signatures(run_dir)


def test_empty_signature_dir_refuses(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "signatures").mkdir(parents=True)
    (root / "cells.json").write_text(json.dumps([_cell(0, "cc1", "convergent_control", 0, 0)]))
    with pytest.raises(ValueError, match="no usable signatures"):
        bc.load_signatures(root)


# ── clustering: the PRIMARY measure ───────────────────────────────────────────


def test_planted_two_basin_prompt_splits_evenly(run_dir: Path, discriminants: Path) -> None:
    sf1 = _prompt(_report(run_dir, discriminants), "sf1")
    assert sf1["two_means_sizes"] == [4, 4]
    assert sf1["two_means_minority_fraction"] == pytest.approx(0.5)
    assert sf1["spread_shape"] == "bimodal"
    assert sf1["spread_ok"] is True
    assert sf1["verdict"] == "GO"


def test_tight_prompts_do_not_spread(run_dir: Path, discriminants: Path) -> None:
    """2-means splits noise evenly, so balance alone must NOT be enough."""
    report = _report(run_dir, discriminants)
    for pid in ("cc1", "mf1"):
        row = _prompt(report, pid)
        assert row["spread_shape"] == "tight"
        assert row["verdict"] == "NO-GO"
        # The even split is still there — it is just not evidence on its own.
        assert row["two_means_minority_fraction"] >= 0.25
        assert row["spread_ratio"] < 0.10


def test_movement_without_clustering_reads_as_diffuse(run_dir: Path, discriminants: Path) -> None:
    # og1's seeds move (one is far out) but do not split evenly — the
    # open_generative underdetermination shape, informative but not a GO.
    og1 = _prompt(_report(run_dir, discriminants), "og1")
    assert og1["spread_shape"] == "diffuse"
    assert og1["spread_ok"] is False
    assert og1["verdict"] == "NO-GO"
    assert og1["spread_ratio"] >= 0.10
    assert og1["two_means_minority_fraction"] < 0.25


def test_fork_prompt_spreads_more_than_the_control(run_dir: Path, discriminants: Path) -> None:
    report = _report(run_dir, discriminants)
    assert (
        _prompt(report, "sf1")["within_cosine_distance_mean"]
        > _prompt(report, "cc1")["within_cosine_distance_mean"]
    )


def test_separation_score_sees_the_planted_spread(run_dir: Path, discriminants: Path) -> None:
    # sf1's seeds sit further from each other than a tight prompt's do, so its
    # silhouette-style score is the lowest of the three.
    report = _report(run_dir, discriminants)
    sf1 = _prompt(report, "sf1")["separation_score"]
    cc1 = _prompt(report, "cc1")["separation_score"]
    assert sf1 is not None and cc1 is not None
    assert sf1 < cc1


# ── mode projection: SECONDARY colour ─────────────────────────────────────────


def test_the_two_basins_get_different_dominant_modes(run_dir: Path, discriminants: Path) -> None:
    sf1 = _prompt(_report(run_dir, discriminants), "sf1")
    assert sf1["n_distinct_dominant_modes"] == 2
    assert set(sf1["dominant_mode_counts"]) == {"m0", "m1"}
    assert sf1["dominant_mode_counts"]["m0"] == 4
    assert sf1["dominant_mode_counts"]["m1"] == 4
    assert sf1["mode_diversity_nats"] == pytest.approx(np.log(2))


def test_mode_diversity_never_gates_a_verdict(run_dir: Path, discriminants: Path) -> None:
    # An unreachable --min-modes must not withdraw a clustering GO: the
    # discriminants are a weak ruler off their fit distribution.
    report = _report(run_dir, discriminants, min_modes=99)
    sf1 = _prompt(report, "sf1")
    assert sf1["modes_ok_secondary"] is False
    assert sf1["verdict"] == "GO"
    assert _class(report, "stance_fork")["verdict"] == "GO"


def test_zero_scale_entries_do_not_produce_nan(discriminants: Path) -> None:
    fd = bc.load_factor_directions_local(discriminants)
    zeroed = bc.LocalFactorDirections(
        space=fd.space,
        labels=fd.labels,
        W=fd.W,
        mean=fd.mean,
        scale=np.zeros_like(fd.scale),   # every scale zero -> guard must hold
        names=fd.names,
        centroids=fd.centroids,
    )
    proj = bc.mode_projection_local(np.ones(DIM, dtype=np.float32), NAMES, zeroed)
    assert np.isfinite(proj.mode_fraction)
    assert np.isfinite(proj.delta_z_norm)
    assert all(np.isfinite(v) for v in proj.centroid_cosines.values())


def test_zero_delta_is_handled(discriminants: Path) -> None:
    fd = bc.load_factor_directions_local(discriminants)
    proj = bc.mode_projection_local(np.zeros(DIM, dtype=np.float32), NAMES, fd)
    assert proj.delta_z_norm == pytest.approx(0.0)
    assert np.isfinite(proj.mode_fraction)


# ── per-class rollup ──────────────────────────────────────────────────────────


def test_class_verdicts_follow_the_stated_thresholds(run_dir: Path, discriminants: Path) -> None:
    report = _report(run_dir, discriminants)
    assert _class(report, "stance_fork")["verdict"] == "GO"
    assert _class(report, "method_fork")["verdict"] == "NO-GO"
    assert _class(report, "open_generative")["verdict"] == "NO-GO"
    assert _class(report, "convergent_control")["verdict"] == "NO-GO"
    assert _class(report, "convergent_control")["is_control"] is True
    assert _class(report, "stance_fork")["is_control"] is False
    assert report["overall"]["go_classes"] == ["stance_fork"]
    assert report["thresholds"]["class_go_fraction"] == 0.5
    assert report["overall"]["spread_shape_counts"] == {
        "tight": 2, "diffuse": 1, "bimodal": 1
    }
    assert _class(report, "open_generative")["spread_shape_counts"]["diffuse"] == 1


def test_b1_shortlist_excludes_the_control_class(run_dir: Path, discriminants: Path) -> None:
    report = _report(run_dir, discriminants)
    assert [r["prompt_id"] for r in report["b1_shortlist"]] == ["sf1"]


def test_control_class_that_spreads_is_not_shortlisted(
    reversed_run_dir: Path, discriminants: Path
) -> None:
    report = _report(reversed_run_dir, discriminants)
    assert _class(report, "convergent_control")["verdict"] == "GO"
    assert "CONTROL" in _class(report, "convergent_control")["reading"]
    assert report["b1_shortlist"] == []


# ── the calibration line ──────────────────────────────────────────────────────


def test_calibration_stays_quiet_when_the_control_is_tight(
    run_dir: Path, discriminants: Path
) -> None:
    calibration = _report(run_dir, discriminants)["calibration"]
    assert calibration["status"] == "ok"
    assert calibration["control_class"] == "convergent_control"
    assert calibration["ratio"] < 1.0 - calibration["margin"]
    assert "CALIBRATION FLAG" not in calibration["message"]


def test_calibration_fires_when_the_control_is_the_one_that_spreads(
    reversed_run_dir: Path, discriminants: Path
) -> None:
    report = _report(reversed_run_dir, discriminants)
    calibration = report["calibration"]
    assert calibration["status"] == "FLAG"
    assert calibration["ratio"] > 1.0
    assert "CALIBRATION FLAG" in calibration["message"]
    assert "reading noise" in calibration["message"]
    assert report["overall"]["decision"].startswith("UNCALIBRATED")


def test_calibration_reports_absence_rather_than_guessing(
    run_dir: Path, discriminants: Path
) -> None:
    calibration = _report(run_dir, discriminants, control_class="not_a_class")["calibration"]
    assert calibration["status"] == "absent"
    assert "UNCALIBRATED" in calibration["message"]


# ── the assert everything else rests on ───────────────────────────────────────


def test_name_mismatch_aborts_the_analysis(run_dir: Path, tmp_path: Path) -> None:
    wrong = _write_discriminants(
        tmp_path / "wrong_names.npz", [f"OTHER{i}" for i in range(DIM)]
    )
    sigs = bc.load_signatures(run_dir)
    fd = bc.load_factor_directions_local(wrong)
    with pytest.raises(AssertionError, match="feature-name alignment failed"):
        bc.analyse(sigs, fd, bc.mode_projection_local, 0.25, 2)


def test_name_mismatch_reports_the_first_bad_index(tmp_path: Path) -> None:
    names = list(NAMES)
    names[3] = "DRIFTED"
    wrong = _write_discriminants(tmp_path / "one_off.npz", names)
    fd = bc.load_factor_directions_local(wrong)
    with pytest.raises(AssertionError, match="first mismatch at index 3"):
        bc.mode_projection_local(np.ones(DIM, dtype=np.float32), NAMES, fd)


def test_main_exits_3_on_a_name_mismatch(run_dir: Path, tmp_path: Path) -> None:
    wrong = _write_discriminants(tmp_path / "wrong2.npz", [f"X{i}" for i in range(DIM)])
    out = tmp_path / "report.json"
    assert _run_main(run_dir, wrong, out) == 3
    assert not out.exists()


def test_main_writes_a_report_and_returns_zero(
    run_dir: Path, discriminants: Path, tmp_path: Path
) -> None:
    out = tmp_path / "report" / "basin_check.json"
    assert _run_main(run_dir, discriminants, out) == 0
    report = json.loads(out.read_text())
    assert report["stage"] == "expB0_basin_check"
    assert report["feature_dim"] == DIM
    assert report["n_prompts"] == len(PROMPTS)
    assert report["n_classes"] == len(PROMPTS)
    assert "decision" in report["overall"]
    assert report["calibration"]["status"] == "ok"


def _run_main(run_dir: Path, discriminants: Path, out: Path) -> int:
    argv = [
        "basin_check",
        "--run-dir", str(run_dir),
        "--discriminants", str(discriminants),
        "--out", str(out),
    ]
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = argv
    try:
        return bc.main()
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
