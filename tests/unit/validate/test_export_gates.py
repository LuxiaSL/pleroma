"""export gates: S-06/X-04 (map-structure), X-02/X-06 (map-load), E-03/X-04 (export-report)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pleroma.validate import export

from tests.contract.map import _fixtures as mapfx

from .conftest import make_profile, write_json


def test_map_structure_pass(toy_map) -> None:
    m, _ = toy_map
    r = export.gate_map_structure(m, make_profile())
    assert r.verdict == "PASS", r.reason
    assert r.evidence["map_fingerprint"]


def test_map_structure_fails_on_site_mismatch(toy_map) -> None:
    m, _ = toy_map
    r = export.gate_map_structure(m, make_profile(**{"map.sites": [3, 7, 12]}))
    assert r.verdict == "FAIL" and "S-06" in r.reason


def test_map_structure_fails_on_hidden_mismatch(toy_map) -> None:
    m, _ = toy_map
    p = make_profile(**{"model.arch.hidden_dim": 32, "model.arch.head_dim": 16})
    r = export.gate_map_structure(m, p)
    assert r.verdict == "FAIL" and "built for a different model" in r.reason


def test_map_structure_fails_on_bad_ruler(tmp_path: Path) -> None:
    _, sha = mapfx.write_discriminants(tmp_path / "d")
    m = mapfx.write_v1a_map(tmp_path / "m", sha, norm_ref=[1.0, 0.0, 2.0])
    r = export.gate_map_structure(m, make_profile())
    assert r.verdict == "FAIL" and "α would mean nothing" in r.reason


def test_map_structure_not_a_map(tmp_path: Path) -> None:
    p = tmp_path / "x.npz"
    np.savez(p, foo=np.zeros(3))
    r = export.gate_map_structure(p, make_profile())
    assert r.verdict == "FAIL" and "not a readable loom map" in r.reason
    assert export.gate_map_structure(tmp_path / "absent.npz",
                                     make_profile()).verdict == "INCONCLUSIVE"


def test_map_load_pass(toy_map) -> None:
    m, d = toy_map
    r = export.gate_map_load(m, d)
    assert r.verdict == "PASS" and r.evidence["rank"] == mapfx.RANK


def test_map_load_fails_on_foreign_discriminants(toy_map, tmp_path: Path) -> None:
    m, _ = toy_map
    other, _ = mapfx.write_discriminants(tmp_path / "other", seed=99)
    r = export.gate_map_load(m, other)
    assert r.verdict == "FAIL" and "discriminants sha mismatch" in r.reason


def test_map_load_without_discriminants_is_inconclusive(toy_map) -> None:
    m, _ = toy_map
    r = export.gate_map_load(m, None)
    assert r.verdict == "INCONCLUSIVE" and "discriminants not supplied" in r.reason


def _report(tmp_path: Path, map_path: Path | None = None, **self_check) -> Path:
    sc = {"within_fan_top1_in_sample": 0.999, "floor": 0.93, "chance": 0.125,
          "registered_heldout_top1": 0.9983, "heldout_source": "v1a_report.json",
          "verdict": "PASS"}
    sc.update(self_check)
    blob = {"stage": "v1a_export", "self_check": sc}
    if map_path is not None:
        blob["out_sha256"] = hashlib.sha256(map_path.read_bytes()).hexdigest()
    return write_json(tmp_path / "export_report.json", blob)


def test_export_report_pass(tmp_path: Path, toy_map) -> None:
    m, _ = toy_map
    r = export.gate_export_report(_report(tmp_path, m), m)
    assert r.verdict == "PASS" and "sanity floor only" in r.reason


def test_export_report_fails_on_the_3b_literal(tmp_path: Path) -> None:
    r = export.gate_export_report(_report(tmp_path, registered_heldout_top1=0.9339,
                                          heldout_source=None))
    assert r.verdict == "FAIL" and "3B literal" in r.reason


def test_export_report_in_sample_only_is_inconclusive(tmp_path: Path) -> None:
    r = export.gate_export_report(_report(tmp_path, registered_heldout_top1=None))
    assert r.verdict == "INCONCLUSIVE" and "IN-SAMPLE" in r.reason


def test_export_report_fails_on_a_different_file(tmp_path: Path, toy_map) -> None:
    m, _ = toy_map
    rep = write_json(tmp_path / "r.json", {"out_sha256": "0" * 64, "self_check": {
        "verdict": "PASS", "registered_heldout_top1": 0.9, "heldout_source": "x"}})
    r = export.gate_export_report(rep, m)
    assert r.verdict == "FAIL" and "describes a different file" in r.reason


def test_export_report_fails_when_self_check_failed(tmp_path: Path) -> None:
    r = export.gate_export_report(_report(tmp_path, verdict="FAIL",
                                          within_fan_top1_in_sample=0.5))
    assert r.verdict == "FAIL" and "self-check did not pass" in r.reason
