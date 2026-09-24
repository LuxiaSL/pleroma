"""pleroma.harvest.lane_parity — the node gate's CPU steps (select, compare)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.harvest import lane_parity as lp

NAMES = [f"f{i}" for i in range(6)]
SCALE = np.array([1.0, 2.0, 0.5, 10.0, 0.0, 1.0])  # f4 is a constant feature


def _sig(path: Path, values, names=NAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.asarray(values, dtype=np.float32),
                        feature_names=np.array(names))


def _dep(**over) -> dict:
    dep = {"lane_id": "old-lane", "calibration_files": {"pm": "a", "pca": "b"},
           "model_files_sha256": {"config.json": "c"},
           "lane": {"torch": "2.8", "transformers": "4.56", "calibration": "d" * 64}}
    dep.update(over)
    return dep


def _banked(tmp_path: Path, gids=(0, 1, 2, 3, 4, 5, 6, 7), plen=5) -> Path:
    lane = tmp_path / "gpulane"
    for s, part in enumerate((gids[: len(gids) // 2], gids[len(gids) // 2:])):
        shard = lane / f"shard_{s:02d}"
        entries = {str(g): {"prompt_length": plen, "input_ids": list(range(plen + 3 + g))}
                   for g in part}
        (shard / "out").mkdir(parents=True)
        (shard / "manifest.json").write_text(json.dumps({"entries": entries}))
        (shard / "metadata.json").write_text(json.dumps({"generations": [
            {"generation_id": g, "prompt_id": f"p{g}"} for g in part]}))
        (shard / "out" / "deployment.json").write_text(json.dumps(_dep()))
        for g in part:
            _sig(shard / "out" / f"gen_{g:03d}.npz", np.arange(6) + g)
    return lane


def test_select_takes_the_extremes_and_a_seeded_sample(tmp_path: Path) -> None:
    index = lp.load_banked_index(_banked(tmp_path))
    got = lp.select_gens(index, 5, n_extremes=2, seed=0)
    assert len(got) == 5 and {0, 1, 6, 7} <= set(got)
    assert got == lp.select_gens(index, 5, n_extremes=2, seed=0)
    assert lp.select_gens(index, 99) == list(range(8))


def test_the_subset_manifest_carries_the_banked_bytes(tmp_path: Path) -> None:
    index = lp.load_banked_index(_banked(tmp_path))
    path = lp.write_subset_manifest(index, [1, 6], tmp_path / "w" / "manifest")
    entries = json.loads(path.read_text())["entries"]
    assert entries["6"] == index[6]["entry"]
    meta = json.loads((path.parent / "metadata.json").read_text())
    assert [m["generation_id"] for m in meta["generations"]] == [1, 6]


def _pair(tmp_path: Path, banked, new, names=NAMES) -> dict[int, tuple[Path, Path]]:
    _sig(tmp_path / "b.npz", banked)
    _sig(tmp_path / "n.npz", new, names)
    return {0: (tmp_path / "b.npz", tmp_path / "n.npz")}


def test_identical_values_are_exact(tmp_path: Path) -> None:
    r = lp.compare_runs(_pair(tmp_path, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]), NAMES, SCALE)
    assert r["verdict"] == "EXACT" and r["n_exact"] == 6


def test_a_small_difference_in_z_units_passes(tmp_path: Path) -> None:
    # f3 moves by 0.005 against a corpus scale of 10: z = 5e-4 < 1e-3
    r = lp.compare_runs(_pair(tmp_path, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4.005, 5, 6]), NAMES, SCALE)
    assert r["verdict"] == "PASS" and r["max_z"] == pytest.approx(5e-4, rel=1e-2)
    assert r["worst"][0]["feature"] == "f3"


def test_a_real_difference_fails_and_names_the_feature(tmp_path: Path) -> None:
    # f2 moves by 0.01 against a scale of 0.5: z = 0.02
    r = lp.compare_runs(_pair(tmp_path, [1, 2, 3, 4, 5, 6], [1, 2, 3.01, 4, 5, 6]), NAMES, SCALE)
    assert r["verdict"] == "FAIL" and r["worst"][0]["feature"] == "f2"


def test_a_constant_feature_is_held_to_atol(tmp_path: Path) -> None:
    r = lp.compare_runs(_pair(tmp_path, [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5.01, 6]), NAMES, SCALE)
    assert r["verdict"] == "FAIL" and any("zero-scale" in p for p in r["problems"])


def test_different_names_fail_even_at_equal_width(tmp_path: Path) -> None:
    renamed = NAMES[:5] + ["g5"]
    r = lp.compare_runs(_pair(tmp_path, [1] * 6, [1] * 6, renamed), NAMES, SCALE)
    assert r["verdict"] == "FAIL" and "names differ" in r["problems"][0]


def test_nan_must_coincide(tmp_path: Path) -> None:
    r = lp.compare_runs(_pair(tmp_path, [1, 2, np.nan, 4, 5, 6], [1, 2, 3, 4, 5, 6]), NAMES, SCALE)
    assert r["verdict"] == "FAIL" and any("NaN" in p for p in r["problems"])
    r = lp.compare_runs(_pair(tmp_path, [1, 2, np.nan, 4, 5, 6], [1, 2, np.nan, 4, 5, 6]),
                        NAMES, SCALE)
    assert r["verdict"] == "EXACT"


def test_preconditions_fail_on_other_bytes_and_warn_on_unrecorded() -> None:
    assert lp.preconditions(_dep(), _dep(lane_id="new")) == ([], [])
    fails, _ = lp.preconditions(_dep(), _dep(calibration_files={"pm": "x", "pca": "b"}))
    assert fails and "calibration" in fails[0]
    fails, warns = lp.preconditions(_dep(model_files_sha256=None), _dep())
    assert not fails and "checkpoint" in warns[0]


def test_the_compare_command_end_to_end(tmp_path: Path, capsys) -> None:
    lane = _banked(tmp_path)
    work = tmp_path / "work"
    assert lp.main(["select", "--banked-lane", str(lane), "--work", str(work), "--n", "4"]) == 0
    sel = json.loads((work / "manifest" / "selection.json").read_text())
    for tag in ("new", "repeat"):
        (work / tag).mkdir(parents=True)
        (work / tag / "deployment.json").write_text(json.dumps(_dep(lane_id="new-lane")))
        for g in sel["gen_ids"]:
            _sig(work / tag / f"gen_{g:03d}.npz", np.arange(6) + g)
    disc = tmp_path / "manifest.npz"
    np.savez(disc, FULL_names=np.array(NAMES), FULL_scale=SCALE.astype(np.float32))
    assert lp.main(["compare", "--work", str(work), "--discriminants", str(disc)]) == 0
    report = json.loads((work / "parity_report.json").read_text())
    assert report["verdict"] == "EXACT" and report["repeat"]["verdict"] == "EXACT"
    assert report["lane_id"] == {"banked": ["old-lane"], "new": "new-lane"}
    # a non-deterministic repeat poisons the verdict
    g = sel["gen_ids"][0]
    _sig(work / "repeat" / f"gen_{g:03d}.npz", np.arange(6) + g + 1e-3)
    assert lp.main(["compare", "--work", str(work), "--discriminants", str(disc)]) == 1
    # a different calibration is a precondition failure, not a verdict
    (work / "new" / "deployment.json").write_text(
        json.dumps(_dep(calibration_files={"pm": "zz", "pca": "b"})))
    assert lp.main(["compare", "--work", str(work), "--discriminants", str(disc)]) == 2
    assert "PRECONDITION" in capsys.readouterr().out
