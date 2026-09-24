"""Contract tests for dose-band calibration (`pleroma.dose.band`).

A dose alpha->zone table measured on ONE map does not transfer.
`LoomMap.lever_of` norm-matches per site to that map's own `norm_ref`, so
alpha means a fixed absolute perturbation WITHIN a map and nothing ACROSS maps
(docs/FINDINGS.md section 2) — pointing one map's table at a different map
misstates the dose by up to 2x. `pleroma.dose.band` replaces a hardcoded
table with data: a MEASURED band attached to a map, a
DERIVED band rescaled from another map's measured band via norm_ref ratio,
or NONE (uncalibrated) when neither exists — resolved fresh at server
startup and served on `GET /info`, never silently substituted.

What matters and is asserted here:

  * all three tiers resolve correctly through `resolve_dose_band` (measured
    wins, derived is the rescale fallback, none when there is nothing);
  * a measured band stamped for a DIFFERENT map fingerprint is refused, not
    silently downgraded or ignored;
  * the derived-band arithmetic matches the real w5-r32/old-map norm_ref
    values and the two numbers a live session validated (subliminal
    ceiling 0.236, overdriven ceiling 0.708);
  * `apply_overdriven_ceiling`/`overdriven_ceiling` cap a MEASURED band's
    top zone at the reference's damage point regardless of what the
    ladder's own accuracy suggested, and flag when it sampled into damage;
  * a malformed band (bad tier, non-contiguous zones, non-positive norm_ref,
    a schema newer than this reader, invalid JSON) is refused loudly, never
    half-applied;
  * `DoseBand.zone_for` classifies the actual w5-r32 fixture (subliminal
    <=0.125, threshold ~0.35, audible >=0.5, overdriven >=0.708) correctly
    — the same fixture and boundaries the legacy page's `zoneFor()`
    (`pleroma/serve/static/legacy_ui.html`) classifies, so the server and
    the page agree on every zone edge;
  * `loom_serve.build_info_payload` stays additive: every pre-existing
    `/info` field keeps its shape with `dose_band=None`, and the new key
    never displaces or renames an old one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.dose import band as db
from pleroma.serve import legacy as ls

# ── real norm_ref values and the live w5-r32 dose ladder ──────────────────

OLD_NORM_REF = (0.491, 1.044, 1.511, 2.096)
W5R32_NORM_REF = (1.012, 2.184, 3.239, 4.56)


def old_map_band() -> db.DoseBand:
    """The old map's measured band — the hardcoded ZONES table a single-map
    page carries, exactly (inert/subliminal/threshold/audible/
    overdriven at 0.125/0.5/0.875/1.2/1.5)."""
    return db.DoseBand(
        tier="measured",
        zones=tuple(db.five_zone_band([0.125, 0.5, 0.875, 1.2], 1.5)),
        norm_ref=OLD_NORM_REF,
        source="old map dose ladder (test fixture)",
        alpha_max=1.5,
        map_fingerprint="oldmapfp00000000",
    )


def w5r32_measured_band() -> db.DoseBand:
    """The real w5-r32 band attached to that map (see
    tests/fixtures/dose_bands/w5_r32_measured.json) — subliminal capped
    conservatively at 0.125 (the best-predicted lever, og01155, already
    reads .746 accuracy at alpha=0.25), and the top zone capped at the
    derived overdriven ceiling (~0.7082) rather than the ladder's own
    alpha_max=1.0, because the .75/1.0 rungs show damage cues."""
    old = old_map_band()
    ceiling = db.overdriven_ceiling(old, W5R32_NORM_REF)
    raw_zones = [
        db.DoseZone(lo=0.0, hi=0.125, hi_inclusive=True, key="subliminal",
                    label="subliminal", hint="<=0.25 measured chance-adjacent; pooled to 0.125"),
        db.DoseZone(lo=0.125, hi=0.5, hi_inclusive=False, key="threshold",
                    label="threshold", hint="~0.35 first rung clearing the catch floor"),
        db.DoseZone(lo=0.5, hi=1.0, hi_inclusive=True, key="audible",
                    label="audible", hint=">=0.5 accuracy .875-1.000"),
    ]
    capped, sampled = db.apply_overdriven_ceiling(raw_zones, ceiling, 1.0)
    assert sampled
    return db.DoseBand(
        tier="measured", zones=tuple(capped), norm_ref=W5R32_NORM_REF,
        source="w5-r32 dose ladder 2026-09-18 (test fixture)", alpha_max=1.0,
        map_fingerprint="w5r32fp000000000",
    )


# ── tier resolution ──────────────────────────────────────────────────────────


def test_measured_band_wins_when_present() -> None:
    measured = w5r32_measured_band()
    resolved = db.resolve_dose_band(
        measured=measured, reference=old_map_band(),
        target_norm_ref=W5R32_NORM_REF, target_map_fingerprint="w5r32fp000000000",
    )
    assert resolved is measured
    assert resolved.tier == "measured"


def test_derived_band_is_the_fallback_when_no_measured_band() -> None:
    resolved = db.resolve_dose_band(
        measured=None, reference=old_map_band(),
        target_norm_ref=W5R32_NORM_REF, target_map_fingerprint="w5r32fp000000000",
    )
    assert resolved is not None
    assert resolved.tier == "derived"


def test_none_when_neither_measured_nor_reference() -> None:
    resolved = db.resolve_dose_band(
        measured=None, reference=None,
        target_norm_ref=W5R32_NORM_REF, target_map_fingerprint="w5r32fp000000000",
    )
    assert resolved is None


def test_measured_band_for_a_different_map_is_refused_not_downgraded() -> None:
    """The silent-science failure this guard exists for: a sidecar sitting
    next to the wrong map must not quietly become a plausible-looking
    gauge — refuse loudly instead of falling back to derived or none."""
    measured = w5r32_measured_band()  # stamped w5r32fp000000000
    with pytest.raises(ValueError, match="refusing to serve"):
        db.resolve_dose_band(
            measured=measured, reference=old_map_band(),
            target_norm_ref=W5R32_NORM_REF, target_map_fingerprint="SOME-OTHER-MAP",
        )


def test_measured_band_with_no_fingerprint_is_trusted_as_is() -> None:
    """A band authored without access to the real map (no fingerprint
    stamped) cannot be cross-checked — the server trusts the operator's
    placement of the sidecar file. This is documented as a known weakness,
    not silently pretended away."""
    unstamped = db.DoseBand(
        tier="measured", zones=tuple(db.five_zone_band([0.125, 0.5, 0.875, 1.2], 1.5)),
        norm_ref=OLD_NORM_REF, source="unstamped fixture", alpha_max=1.5,
        map_fingerprint=None,
    )
    resolved = db.resolve_dose_band(
        measured=unstamped, reference=None,
        target_norm_ref=W5R32_NORM_REF, target_map_fingerprint="anything-at-all",
    )
    assert resolved is unstamped


# ── the derived-band arithmetic, against the REAL norm_ref values ───────────


def test_rescale_matches_the_validated_w5r32_numbers() -> None:
    """Applied to the old map's band with no access to the w5-r32 ladder,
    this exact method predicted a subliminal ceiling of 0.236 (measured
    <=0.25) and an overdriven ceiling of 0.708 (a live wear at alpha=0.75
    on w5-r32 produced visible word-salad, consistent with 0.75 already
    past 0.708) — the two numbers a live session validated."""
    derived = old_map_band().rescale(W5R32_NORM_REF, "w5r32fp000000000")
    assert derived.tier == "derived"
    by_key = {z.key: z for z in derived.zones}
    assert by_key["subliminal"].hi == pytest.approx(0.236065, abs=1e-4)
    assert derived.alpha_max == pytest.approx(0.708195, abs=1e-4)
    # every zone boundary scaled by the SAME factor — contiguity survives
    # (compared loosely: both sides are independently rounded to 6dp by rescale())
    assert by_key["inert"].hi == pytest.approx(0.125 * (by_key["subliminal"].hi / 0.5), abs=1e-6)


def test_rescale_scale_factor_is_mean_of_per_site_ratios_not_ratio_of_means() -> None:
    """Regression pin for the exact aggregation method: mean of the
    per-site (target/reference) ratios, inverted — NOT the ratio of the two
    means. The two differ by about 1% for this map pair and a live session
    specifically validated the mean-of-ratios version against real ladder
    numbers; silently drifting back to ratio-of-means would not raise a
    type error, just quietly answer a slightly different question."""
    ratios = [t / r for r, t in zip(OLD_NORM_REF, W5R32_NORM_REF)]
    expected_scale = 1.0 / (sum(ratios) / len(ratios))
    derived = old_map_band().rescale(W5R32_NORM_REF, None)
    # rescale() rounds each boundary to 6 decimal places
    assert derived.alpha_max == pytest.approx(1.5 * expected_scale, abs=1e-6)


def test_overdriven_ceiling_matches_rescaled_alpha_max() -> None:
    old = old_map_band()
    assert db.overdriven_ceiling(old, W5R32_NORM_REF) == pytest.approx(
        old.rescale(W5R32_NORM_REF, None).alpha_max)


def test_rescale_refuses_a_site_count_mismatch() -> None:
    with pytest.raises(ValueError, match="sites"):
        old_map_band().rescale([1.0, 2.0, 3.0], None)  # 3 sites, band has 4


def test_rescale_refuses_non_positive_target_norm_ref() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        old_map_band().rescale([1.0, 2.0, 3.0, 0.0], None)


def test_rescale_warns_on_high_per_site_spread(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="dose_band")
    # a wildly disagreeing target (one site much louder than the others
    # relative to the reference) should trip the >15% spread warning
    old_map_band().rescale([1.012, 2.184, 3.239, 40.0], None)
    assert any("disagree" in r.message for r in caplog.records)


# ── the overdriven ceiling: applied to a MEASURED band ───────────────────────


def test_apply_overdriven_ceiling_caps_and_flags_a_ladder_that_sampled_into_damage() -> None:
    zones = [
        db.DoseZone(lo=0.0, hi=0.5, hi_inclusive=False, key="threshold",
                    label="threshold", hint=""),
        db.DoseZone(lo=0.5, hi=1.0, hi_inclusive=True, key="audible",
                    label="audible", hint=""),
    ]
    capped, sampled = db.apply_overdriven_ceiling(zones, ceiling=0.7, alpha_max=1.0)
    assert sampled is True
    assert [z.key for z in capped] == ["threshold", "audible", "overdriven"]
    audible = capped[1]
    assert audible.hi == pytest.approx(0.7)
    assert audible.hi_inclusive is False
    overdriven = capped[2]
    assert (overdriven.lo, overdriven.hi, overdriven.hi_inclusive) == (0.7, 1.0, True)


def test_apply_overdriven_ceiling_is_a_noop_when_ceiling_is_not_reached() -> None:
    zones = [db.DoseZone(lo=0.0, hi=0.5, hi_inclusive=True, key="subliminal",
                          label="subliminal", hint="")]
    capped, sampled = db.apply_overdriven_ceiling(zones, ceiling=0.9, alpha_max=0.5)
    assert sampled is False
    assert capped == zones


def test_apply_overdriven_ceiling_does_not_flag_a_zone_already_called_overdriven() -> None:
    zones = [
        db.DoseZone(lo=0.0, hi=0.5, hi_inclusive=False, key="audible", label="audible", hint=""),
        db.DoseZone(lo=0.5, hi=1.5, hi_inclusive=True, key="overdriven",
                    label="overdriven", hint=""),
    ]
    capped, sampled = db.apply_overdriven_ceiling(zones, ceiling=0.7, alpha_max=1.5)
    assert sampled is False  # only 'audible' reaching past the ceiling would count
    assert [z.key for z in capped] == ["audible", "overdriven"]


# ── zone_for: the w5-r32 fixture ─────────────────────────────────────────────


@pytest.mark.parametrize("alpha, expected_key", [
    (0.0, "subliminal"),
    (0.125, "subliminal"),   # closed top edge
    (0.13, "threshold"),
    (0.35, "threshold"),
    (0.5, "audible"),
    (0.7, "audible"),
    (0.708195, "overdriven"),  # closed bottom edge, matches the live field report at 0.75
    (0.75, "overdriven"),
    (1.0, "overdriven"),
    (-1.0, "subliminal"),     # clamp below range
    (5.0, "overdriven"),      # clamp above range
])
def test_zone_for_classifies_the_w5r32_fixture(alpha: float, expected_key: str) -> None:
    band = w5r32_measured_band()
    assert band.zone_for(alpha).key == expected_key


def test_zone_for_with_no_zones_data_is_never_constructible() -> None:
    """A DoseBand with an empty zones tuple should never exist — caught at
    construction, not discovered later at gauge-render time."""
    with pytest.raises(ValueError, match="no zones"):
        db.DoseBand(tier="measured", zones=(), norm_ref=OLD_NORM_REF,
                    source="x", alpha_max=1.5)


# ── malformed bands are refused, never half-applied ──────────────────────────


def test_malformed_tier_is_refused() -> None:
    with pytest.raises(ValueError, match="tier"):
        db.DoseBand(tier="certified!!", zones=tuple(db.five_zone_band(
            [0.125, 0.5, 0.875, 1.2], 1.5)), norm_ref=OLD_NORM_REF, source="x", alpha_max=1.5)


def test_non_contiguous_zones_are_refused() -> None:
    zones = (
        db.DoseZone(lo=0.0, hi=0.5, hi_inclusive=False, key="a", label="a", hint=""),
        db.DoseZone(lo=0.6, hi=1.0, hi_inclusive=True, key="b", label="b", hint=""),  # gap
    )
    with pytest.raises(ValueError, match="not contiguous"):
        db.DoseBand(tier="measured", zones=zones, norm_ref=OLD_NORM_REF, source="x", alpha_max=1.0)


def test_first_zone_must_start_at_zero() -> None:
    zones = (db.DoseZone(lo=0.1, hi=1.0, hi_inclusive=True, key="a", label="a", hint=""),)
    with pytest.raises(ValueError, match="start at alpha=0"):
        db.DoseBand(tier="measured", zones=zones, norm_ref=OLD_NORM_REF, source="x", alpha_max=1.0)


def test_alpha_max_must_match_last_zone_hi() -> None:
    zones = (db.DoseZone(lo=0.0, hi=1.0, hi_inclusive=True, key="a", label="a", hint=""),)
    with pytest.raises(ValueError, match="alpha_max"):
        db.DoseBand(tier="measured", zones=zones, norm_ref=OLD_NORM_REF, source="x", alpha_max=2.0)


def test_non_positive_norm_ref_is_refused() -> None:
    zones = (db.DoseZone(lo=0.0, hi=1.0, hi_inclusive=True, key="a", label="a", hint=""),)
    with pytest.raises(ValueError, match="strictly positive"):
        db.DoseBand(tier="measured", zones=zones, norm_ref=(1.0, 0.0), source="x", alpha_max=1.0)


def test_zone_hi_must_exceed_lo() -> None:
    with pytest.raises(ValueError, match="hi <= lo"):
        db.DoseZone(lo=0.5, hi=0.5, hi_inclusive=True, key="a", label="a", hint="")


def test_five_zone_band_needs_exactly_four_boundaries() -> None:
    with pytest.raises(ValueError, match="4 boundaries"):
        db.five_zone_band([0.1, 0.2, 0.3], 1.0)


def test_five_zone_band_needs_strictly_increasing_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        db.five_zone_band([0.5, 0.2, 0.8, 1.0], 1.5)


def test_load_dose_band_file_refuses_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        db.load_dose_band_file(p)


def test_load_dose_band_file_refuses_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        db.load_dose_band_file(tmp_path / "nope.json")


def test_load_dose_band_file_refuses_malformed_content(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"tier": "measured", "zones": []}))  # empty zones
    with pytest.raises(ValueError, match="malformed dose band"):
        db.load_dose_band_file(p)


def test_load_dose_band_file_refuses_a_newer_schema(tmp_path: Path) -> None:
    p = tmp_path / "future.json"
    blob = old_map_band().to_json()
    blob["schema"] = db.CURRENT_SCHEMA + 1
    p.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="newer than this reader"):
        db.load_dose_band_file(p)


def test_save_and_load_dose_band_file_round_trips(tmp_path: Path) -> None:
    band = w5r32_measured_band()
    path = tmp_path / "sub" / "map.doseband.json"
    db.save_dose_band_file(band, path)
    back = db.load_dose_band_file(path)
    assert back.tier == band.tier
    assert back.alpha_max == band.alpha_max
    assert back.norm_ref == band.norm_ref
    assert [z.to_json() for z in back.zones] == [z.to_json() for z in band.zones]


# ── map identity (the discriminants-free npz read the CLI relies on) ────────


def _write_fake_map_npz(path: Path, norm_ref: tuple[float, ...]) -> None:
    np.savez(
        path,
        W_U=np.eye(4, 2),  # load_map_identity hashes the weights (v2, X-06)
        mu_y=np.zeros(len(norm_ref) * 8),
        sites=np.array([4, 8, 16, 24][: len(norm_ref)]),
        norm_ref=np.array(norm_ref),
        meta=json.dumps({"rank": 32, "discriminants_sha256": "irrelevant-here"}),
    )


def test_load_map_identity_matches_map_fingerprint(tmp_path: Path) -> None:
    p = tmp_path / "fake_map.npz"
    _write_fake_map_npz(p, W5R32_NORM_REF)
    fp, sites, norm_ref, meta = db.load_map_identity(p)
    expected = db.map_fingerprint(sites, norm_ref, np.zeros(len(W5R32_NORM_REF) * 8), meta)
    # v2 (weights included) with the legacy v1 id embedded — the migration path
    assert fp.startswith("v2-") and fp.endswith("." + expected)
    assert list(norm_ref) == pytest.approx(list(W5R32_NORM_REF))


def test_load_map_identity_refuses_a_non_map_npz(tmp_path: Path) -> None:
    p = tmp_path / "not_a_map.npz"
    np.savez(p, foo=np.zeros(3))
    with pytest.raises(KeyError):
        db.load_map_identity(p)


def test_loom_serve_reexports_the_same_map_fingerprint() -> None:
    """`ls.map_fingerprint` resolves to `pleroma.dose.band.map_fingerprint` —
    same function, imported back, not a divergent copy."""
    assert ls.map_fingerprint is db.map_fingerprint


# ── CLI: attach + show, end to end ───────────────────────────────────────────


def test_cli_attach_with_boundaries_and_overdriven_reference(tmp_path: Path, capsys: Any) -> None:
    map_path = tmp_path / "fake_map.npz"
    _write_fake_map_npz(map_path, W5R32_NORM_REF)
    ref_path = tmp_path / "old_ref.json"
    db.save_dose_band_file(old_map_band(), ref_path)
    out_path = tmp_path / "attached.json"

    db.main([
        "attach", "--map", str(map_path),
        "--boundaries", "0.125,0.5,0.875,1.2", "--alpha-max", "1.5",
        "--source", "cli test fixture", "--out", str(out_path),
        "--overdriven-reference", str(ref_path),
    ])
    captured = capsys.readouterr()
    assert "WARNING" in captured.out  # this ladder's own zones reach 1.5, past the ~0.708 ceiling
    assert out_path.exists()

    band = db.load_dose_band_file(out_path)
    assert band.tier == "measured"
    assert band.zones[-1].key == "overdriven"
    assert band.zones[-1].lo == pytest.approx(0.708195, abs=1e-4)
    assert band.map_fingerprint is not None  # the CLI stamps it — unlike this test module's fixtures


def test_cli_attach_rejects_both_boundaries_and_zones_json(tmp_path: Path) -> None:
    map_path = tmp_path / "fake_map.npz"
    _write_fake_map_npz(map_path, W5R32_NORM_REF)
    zones_path = tmp_path / "zones.json"
    zones_path.write_text(json.dumps([z.to_json() for z in db.five_zone_band(
        [0.125, 0.5, 0.875, 1.2], 1.5)]))
    with pytest.raises(SystemExit):
        db.main([
            "attach", "--map", str(map_path), "--source", "x",
            "--boundaries", "0.125,0.5,0.875,1.2", "--alpha-max", "1.5",
            "--zones-json", str(zones_path),
        ])


def test_cli_show_resolves_derived_tier(tmp_path: Path, capsys: Any) -> None:
    map_path = tmp_path / "fake_map.npz"
    _write_fake_map_npz(map_path, W5R32_NORM_REF)
    ref_path = tmp_path / "old_ref.json"
    db.save_dose_band_file(old_map_band(), ref_path)

    db.main(["show", "--map", str(map_path), "--dose-band-reference", str(ref_path)])
    out = capsys.readouterr().out
    assert "tier=derived" in out


def test_cli_show_reports_uncalibrated_with_nothing_attached(tmp_path: Path, capsys: Any) -> None:
    map_path = tmp_path / "fake_map.npz"
    _write_fake_map_npz(map_path, W5R32_NORM_REF)
    db.main(["show", "--map", str(map_path)])
    out = capsys.readouterr().out
    assert "tier=none" in out
    assert "uncalibrated" in out


# ── /info backward compatibility (loom_serve.build_info_payload) ────────────


def _base_info_kwargs() -> dict[str, Any]:
    return dict(
        map_path="outputs/loom/loom_map_w5_r32.npz", map_meta={"rank": 32},
        sites=[4, 8, 16, 24], branches=["base", "loom"], default_k=6,
        future_tokens=192, detach_wear_default=False, n_sessions=2,
        harvest_worker="127.0.0.1:8768", restored_sessions=1,
        persistence={"enabled": True, "dir": "/x/sessions"},
        auto_policies=[{"key": "distinct", "needs_wear": False}],
        loudness_ref=2.74875,
    )


def test_info_payload_keeps_every_preexisting_field_with_no_band() -> None:
    payload = ls.build_info_payload(**_base_info_kwargs(), dose_band=None)
    for key in ("map", "map_meta", "sites", "branches", "default_k", "future_tokens",
                "detach_wear_default", "n_sessions", "harvest_worker", "restored_sessions",
                "persistence", "auto_policies", "loudness_ref"):
        assert key in payload
    assert payload["map"] == "outputs/loom/loom_map_w5_r32.npz"
    assert payload["loudness_ref"] == pytest.approx(2.749, abs=1e-3)
    assert payload["dose_band"]["tier"] == "none"
    assert payload["dose_band"]["zones"] is None


def test_info_payload_carries_a_real_band_additively() -> None:
    band = w5r32_measured_band()
    payload = ls.build_info_payload(**_base_info_kwargs(), dose_band=band)
    # every old field is STILL there, unchanged in shape
    assert payload["sites"] == [4, 8, 16, 24]
    assert payload["loudness_ref"] == pytest.approx(2.749, abs=1e-3)
    db_json = payload["dose_band"]
    assert db_json["tier"] == "measured"
    assert db_json["alpha_max"] == 1.0
    assert isinstance(db_json["zones"], list) and len(db_json["zones"]) == 4
    assert db_json["lever_quality_caveat"]  # always present, never blank, on a real band
