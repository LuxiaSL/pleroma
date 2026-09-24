"""probe gate: G-01/G-02/F-01 (resolution)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pleroma.validate import probe

from .conftest import write_json


def res(**kw: Any) -> dict[str, Any]:
    base = {"resolved": True, "reps": 4, "n_candidates_used": 8,
            "n_candidates_excluded_for_reps": 0, "within_candidate_sd_nr": 0.3,
            "between_candidate_sd_cell": 0.25, "estimated_signal_sd": 0.2,
            "noise_share_of_cell_variance_at_this_reps": 0.36, "reps_needed_to_resolve": 3,
            "why": "the ordering is resolved at this rep count"}
    base.update(kw)
    return base


def test_resolved_passes_but_certifies_top1_only() -> None:
    r = probe.gate_resolution({"resolution": res(), "clean_regime": True})
    assert r.verdict == "PASS" and "ranks below #1 are NOT certified" in r.reason


def test_unresolved_names_the_reps_needed() -> None:
    r = probe.gate_resolution({"resolution": res(resolved=False, reps=3,
                                                 reps_needed_to_resolve=24)})
    assert r.verdict == "INCONCLUSIVE" and "needs about 24 reps" in r.reason


def test_flat_fan_is_named() -> None:
    r = probe.gate_resolution(res(resolved=False, between_candidate_sd_cell=0.0,
                                  within_candidate_sd_nr=0.0, reps_needed_to_resolve=None))
    assert r.verdict == "INCONCLUSIVE" and "fan did not fork" in r.reason


def test_worn_regime_is_inconclusive() -> None:
    r = probe.gate_resolution({"resolution": res(), "clean_regime": False})
    assert r.verdict == "INCONCLUSIVE" and "G-02" in r.reason


def test_one_rep_cannot_answer() -> None:
    r = probe.gate_resolution({"resolution": {"resolved": None, "why": "needs reps>=2"}})
    assert r.verdict == "INCONCLUSIVE" and "needs reps >= 2" in r.reason


def test_run_probe_on_files(tmp_path: Path) -> None:
    ok = write_json(tmp_path / "p.json", {"resolution": res(), "clean_regime": True})
    assert probe.run_probe(ok)[0].verdict == "PASS"
    junk = write_json(tmp_path / "j.json", {"ranking": [1, 2]})
    (r,) = probe.run_probe(junk)
    assert r.verdict == "INCONCLUSIVE" and "no 'resolution'" in r.reason
    assert probe.run_probe(None) == []
