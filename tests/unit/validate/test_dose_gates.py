"""dose gates: D-02 (band), C7-SPAN (band-span), D-01/D-05/F-06 (damage)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pleroma.dose.band import DoseBand, five_zone_band, load_map_identity
from pleroma.validate import dose

from tests.contract.map import _fixtures as mapfx

from .conftest import make_profile, write_json

ROOT = Path(__file__).resolve().parents[3]


def band_blob(norm_ref: list[float], fp: str | None, *, tier: str = "measured",
              source: str = "toy ladder (HF lane)", measured_at: str | None = "2026-09-24",
              **extra: Any) -> dict[str, Any]:
    band = DoseBand(tier=tier, zones=tuple(five_zone_band([0.05, 0.125, 0.35, 0.5], 1.0)),
                    norm_ref=tuple(norm_ref), source=source, alpha_max=1.0,
                    map_fingerprint=fp, measured_at=measured_at,
                    derivation={"rescale": 1.0} if tier == "derived" else None)
    blob = band.to_json()
    blob.update(extra)
    return blob


def fp_of(map_path: Path) -> str:
    return load_map_identity(map_path)[0]


# ── band (D-02) ──────────────────────────────────────────────────────────────


def test_band_pass_stamped_for_this_map(tmp_path: Path, toy_map) -> None:
    m, _ = toy_map
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, fp_of(m)))
    r = dose.gate_band(b, m)
    assert r.verdict == "PASS" and "stamped for this map" in r.reason


def test_band_fails_on_a_foreign_fingerprint(tmp_path: Path, toy_map) -> None:
    m, _ = toy_map
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, "deadbeefdeadbeef"))
    r = dose.gate_band(b, m)
    assert r.verdict == "FAIL" and "stamped for map deadbeefdeadbeef" in r.reason


def test_band_with_a_legacy_fingerprint_is_inconclusive(tmp_path: Path, toy_map) -> None:
    """X-06 migration: a v1 stamp matches the map's legacy id but never
    covered the weights — named, not passed."""
    m, _ = toy_map
    legacy = fp_of(m).rsplit(".", 1)[1]
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, legacy))
    r = dose.gate_band(b, m)
    assert r.verdict == "INCONCLUSIVE" and "LEGACY (v1) map fingerprint" in r.reason
    assert r.evidence["fingerprint_match"] == "legacy"


def test_band_derived_is_inconclusive(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, None, tier="derived"))
    r = dose.gate_band(b, None)
    assert r.verdict == "INCONCLUSIVE" and "DERIVED" in r.reason


def test_band_absent_sidecar_is_inconclusive_and_malformed_fails(tmp_path: Path, toy_map) -> None:
    m, _ = toy_map
    r = dose.gate_band(None, m)
    assert r.verdict == "INCONCLUSIVE" and "α is uncalibrated" in r.reason
    bad = write_json(tmp_path / "bad.json", {"tier": "measured", "zones": []})
    r = dose.gate_band(bad, None)
    assert r.verdict == "FAIL" and "server refuses it" in r.reason


# ── band-span (C7-SPAN) ──────────────────────────────────────────────────────


def test_band_span_pass_under_the_served_span(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(
        mapfx.NORM_REF, None, span_receipt={"injection_span": "uniform",
                                            "time_profile": "constant", "time_k": 64.0}))
    r = dose.gate_band_span(b, None, make_profile())
    assert r.verdict == "PASS" and r.evidence["recorded_at"] == "span_receipt.injection_span"


def test_band_span_fails_under_a_different_span(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, None,
                                                  injection_span="continuation"))
    r = dose.gate_band_span(b, None, make_profile())
    assert r.verdict == "FAIL"
    assert r.reason.startswith("band measured under continuation, served span uniform")


def test_band_span_unrecorded_is_inconclusive(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, None))
    r = dose.gate_band_span(b, None, make_profile())
    assert r.verdict == "INCONCLUSIVE" and "records no injection span" in r.reason


def test_band_span_pre_recording_vllm_band_is_continuation(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(
        mapfx.NORM_REF, None, source="ladder on the vLLM steered path",
        measured_at="2026-09-22"))
    r = dose.gate_band_span(b, None, make_profile())
    assert r.verdict == "FAIL" and "band measured under continuation" in r.reason
    assert r.evidence["span_inferred_from"]


def test_band_span_unknown_span_value_fails(tmp_path: Path) -> None:
    b = write_json(tmp_path / "b.json", band_blob(mapfx.NORM_REF, None,
                                                  injection_span="sideways"))
    r = dose.gate_band_span(b, None, make_profile())
    assert r.verdict == "FAIL" and "not a known span" in r.reason


# ── damage (D-01 / D-05 / F-06) ──────────────────────────────────────────────

PROSE = [
    "I would walk down to the river after lunch and read the book my sister left here "
    "last spring, then call her to argue about the ending.",
    "Honestly the first thing is sleep, because the week has been long, and after that "
    "maybe a slow coffee on the step while the street wakes up.",
    "There is a garden two streets over that nobody tends; I keep meaning to pull the "
    "weeds and see what was planted there before the owners left.",
]
LOOP = ("and so on, and so on, and so on, and so on, and so on, and so on, and so on, "
        "and so on, and so on")


def test_damage_pass_on_prose(modelc_profile) -> None:
    r = dose.gate_damage(PROSE * 4, modelc_profile)
    assert r.verdict == "PASS" and "not a fluency certificate" in r.reason


def test_damage_fails_on_loops(modelc_profile) -> None:
    r = dose.gate_damage(PROSE * 3 + [LOOP] * 3, modelc_profile)
    assert r.verdict == "FAIL" and "3/12 steered replies loop" in r.reason
    assert r.evidence["looped_indices"] == [9, 10, 11]


def test_damage_not_attributable_when_base_loops_too(modelc_profile) -> None:
    r = dose.gate_damage([LOOP] * 4, modelc_profile, base_replies=[LOOP] * 4)
    assert r.verdict == "INCONCLUSIVE" and "not attributable to the dose" in r.reason


def test_damage_reads_the_trimmed_modelc_reply(modelc_profile) -> None:
    # the loop lives only in a dreamed visitor turn: not the reply
    r = dose.gate_damage([p + "\n\n**User:** " + LOOP for p in PROSE], modelc_profile)
    assert r.verdict == "PASS"


def test_damage_too_short_is_inconclusive(modelc_profile) -> None:
    r = dose.gate_damage(["Yes.", "No thanks."], modelc_profile)
    assert r.verdict == "INCONCLUSIVE" and "too short" in r.reason


def test_run_dose_reads_jsonl_and_names_bad_input(tmp_path: Path, modelc_profile) -> None:
    p = tmp_path / "replies.jsonl"
    p.write_text("\n".join(json.dumps({"reply": t}) for t in PROSE))
    (r,) = dose.run_dose(modelc_profile, None, None, p, None)
    assert r.verdict == "PASS" and r.evidence["n_replies"] == 3
    bad = write_json(tmp_path / "bad.json", [1, 2])
    (r,) = dose.run_dose(modelc_profile, None, None, bad, None)
    assert r.verdict == "INCONCLUSIVE" and "neither a string" in r.reason
