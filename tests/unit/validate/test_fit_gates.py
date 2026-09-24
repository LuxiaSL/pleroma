"""fit gates: K-01 (join), E-01/E-02/E-03 (heldout), E-01 (ceiling)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pleroma.validate import fit

from .conftest import make_profile, write_json

ROOT = Path(__file__).resolve().parents[3]

BASE: dict[str, Any] = {
    "stage": "v1a_fit", "token": "v1a-fit-001", "n_gens": 4000, "n_fans": 500,
    "fan_size_mean": 8.0, "chance_top1": 0.125, "fan_source": "member_fan",
    "primary_point": {"input": "dz", "target": "raw", "lambda": 1e4, "rank": 4},
    "baseline_shelf": {"retrieval_top1": 0.55},
    "grid": {"dz|raw|lam10000|r4": {"retrieval_top1": 0.71, "pred_cos_vs_own_targets": 0.4}},
    "registered_test": {"observed": 0.16, "v1a_top1": 0.71, "shelf_top1": 0.55,
                        "n_perms": 20000, "one_sided_p": 5e-5, "verdict": "PASS"},
}


def report(tmp_path: Path, **patch: Any) -> Path:
    blob = copy.deepcopy(BASE)
    for dotted, value in patch.items():
        box = blob
        *head, last = dotted.split(".")
        for k in head:
            box = box[k]
        box[last] = value
    return write_json(tmp_path / "v1a_report.json", blob)


def by_gate(results) -> dict[str, Any]:
    return {r.gate: r for r in results}


def test_all_pass_on_a_healthy_report(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path), make_profile()))
    assert {g: r.verdict for g, r in rs.items()} == {
        "join": "PASS", "heldout": "PASS", "ceiling": "PASS"}
    assert "beats the shelf baseline" in rs["heldout"].reason


def test_join_fails_on_the_lever_join(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path, fan_source="lever_group"), make_profile()))
    assert rs["join"].verdict == "FAIL" and "K-01" in rs["join"].reason


def test_join_small_corpus_is_inconclusive(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path, n_gens=300), make_profile()))
    assert rs["join"].verdict == "INCONCLUSIVE" and "only 300 rows" in rs["join"].reason


def test_heldout_fails_when_the_map_does_not_beat_the_shelf(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path, **{
        "registered_test.observed": -0.01, "registered_test.one_sided_p": 0.7,
        "registered_test.verdict": "FAIL"}), make_profile()))
    assert rs["heldout"].verdict == "FAIL"
    assert "does not beat the zero-training shelf baseline" in rs["heldout"].reason


def test_heldout_fails_at_chance(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path, **{"registered_test.v1a_top1": 0.12}),
                             make_profile()))
    assert rs["heldout"].verdict == "FAIL" and "at or below chance" in rs["heldout"].reason


def test_heldout_inconclusive_off_the_served_point(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path), make_profile(**{"map.rank": 8})))
    assert rs["heldout"].verdict == "INCONCLUSIVE"
    assert "does not describe the served map" in rs["heldout"].reason


def test_ceiling_inconclusive_when_shelf_is_saturated(tmp_path: Path) -> None:
    rs = by_gate(fit.run_fit(report(tmp_path, **{"registered_test.shelf_top1": 0.91}),
                             make_profile()))
    assert rs["ceiling"].verdict == "INCONCLUSIVE"
    assert rs["ceiling"].reason.startswith("retrieval test at ceiling")


def test_missing_and_malformed_reports_are_inconclusive(tmp_path: Path) -> None:
    for path in (tmp_path / "nope.json",
                 write_json(tmp_path / "bad.json", {"stage": "v1a_fit"})):
        rs = fit.run_fit(path, make_profile())
        assert [r.verdict for r in rs] == ["INCONCLUSIVE"] * 3
    (tmp_path / "junk.json").write_text("{not json")
    rs = fit.run_fit(tmp_path / "junk.json", make_profile())
    assert "not valid JSON" in rs[0].reason


