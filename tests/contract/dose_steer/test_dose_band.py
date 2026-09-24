"""Contract: the dose band (``pleroma.dose.band``) — loading, zones,
thresholds, cross-model transfer, and the banked bands' golden numbers.

A band maps alpha to a zone: subliminal / threshold / audible / overdriven,
and damage can arrive before audibility, so "detected" and "broken" are not
the same edge. Everything a porter could change without noticing is pinned
here: edge inclusivity, the malformed-band refusal, the tier-none payload, and
the cross-model rescale reproducing the banked 8B derived band byte-for-byte.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import pytest

from tests.contract.dose_steer import targets as T

MODELC_FP = "b48ffbe7f75d3fde"
MODELC_SHA256 = "d32267353f8d6e7471823739b94e9d13c39f6a0e330a13e5e22b7db917d38106"


def _band_file(name: str) -> Any:
    return T.load_dose_band_file(T.DOSE_BANDS_DIR / name)


# ── the knob start (bandStartAlpha) ──────────────────────────────────────────


def _js_round_half_up(x: float) -> float:
    import math
    return math.floor(x + 0.5)


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host")
    return node


def _run_ui(prog: str) -> dict[str, Any]:
    r = subprocess.run([_node(), "-e", prog], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# ── resolution: what a server serves (measured > derived > none) ─────────────


def test_tier_none_payload_has_null_alpha_max() -> None:
    """An uncalibrated map serves tier "none" with alpha_max null and zones
    null — never another map's numbers. (The UI must treat null as absent,
    not as 0, or every wear would run at alpha 0.)"""
    info = T.dose_band_info_json(None)
    assert info["tier"] == "none"
    assert info["alpha_max"] is None and info["zones"] is None
    assert info == T.NONE_INFO_JSON and info is not T.NONE_INFO_JSON


def test_malformed_band_fails_loudly(tmp_path: Any) -> None:
    """load_dose_band_file raises on unreadable / invalid JSON / invalid band;
    a band that cannot be trusted is never half-applied.
    Also: zones must be contiguous from 0 and end at alpha_max."""
    bad = tmp_path / "b.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        T.load_dose_band_file(bad)
    with pytest.raises(ValueError, match="cannot read"):
        T.load_dose_band_file(tmp_path / "missing.json")
    blob = json.loads((T.DOSE_BANDS_DIR / "w5_r32_measured.json").read_text())
    blob["zones"][1]["lo"] = 0.2  # gap after subliminal
    bad.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="contiguous"):
        T.load_dose_band_file(bad)
    blob = json.loads((T.DOSE_BANDS_DIR / "w5_r32_measured.json").read_text())
    blob["alpha_max"] = None      # an uncalibrated null must not load as a measured band
    bad.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="alpha_max"):
        T.load_dose_band_file(bad)


def test_five_zone_band_edge_convention() -> None:
    """five_zone_band: subliminal's top is CLOSED, every other internal edge
    belongs to the zone above (the convention of the [0.125, 0.5] zone
    table). Boundaries must be strictly increasing below alpha_max."""
    zs = T.five_zone_band([0.06, 0.25, 0.41, 0.56], 1.0)
    assert [(z.key, z.hi_inclusive) for z in zs] == [
        ("inert", False), ("subliminal", True), ("threshold", False),
        ("audible", False), ("overdriven", True)]
    band = T.DoseBand(tier="measured", zones=tuple(zs), norm_ref=(1.0,), source="t", alpha_max=1.0)
    assert band.zone_for(0.06).key == "subliminal"
    assert band.zone_for(0.25).key == "subliminal"
    assert band.zone_for(0.41).key == "audible"
    with pytest.raises(ValueError):
        T.five_zone_band([0.3, 0.25, 0.41, 0.56], 1.0)


# ── cross-model / cross-map transfer ─────────────────────────────────────────


def test_rescale_reproduces_the_banked_8b_derived_band_exactly() -> None:
    """Golden: w5-r32 measured band -> rescale(8B v1a norm_ref, residual scale
    3B->8B) reproduces dose_bands/8b_v1a_derived.json EXACTLY (zones,
    alpha_max, derivation incl. cross_model, source string). scale =
    1/mean_s(target/ref) = 1.422757; disagreement abs vs rel 1.165373x, served
    side conservative. Alpha is not a portable unit across models: the served
    band is the absolute rescale, the relative one is carried beside it as an
    error bar, and here the served side puts every boundary lower."""
    w5 = _band_file("w5_r32_measured.json")
    banked = _band_file("8b_v1a_derived.json")
    rs = T.load_residual_scale_file(T.DOSE_BANDS_DIR / "residual_scale_3b_8b.json")
    got = w5.rescale(banked.norm_ref, banked.map_fingerprint, residual_scale=rs)
    assert got.tier == "derived"
    assert got.to_json() == banked.to_json()
    assert got.derivation["scale"] == 1.422757
    cm = got.derivation["cross_model"]
    assert cm["disagreement"] == 1.165373
    assert cm["served_is_conservative"] is True and cm["hypothesis_served"] == "absolute"


def test_rescale_is_mean_of_per_site_ratio_inverted() -> None:
    """scale = 1 / mean_s(target[s]/ref[s]) — the MEAN OF RATIOS, not the
    ratio of means — applied to every edge and alpha_max; tier always
    'derived'; never promoted. A residual_scale changes NO served number."""
    zs = T.five_zone_band([0.1, 0.2, 0.3, 0.4], 0.5)
    ref = T.DoseBand(tier="measured", zones=tuple(zs), norm_ref=(1.0, 2.0), source="r", alpha_max=0.5)
    tgt = [2.0, 8.0]  # ratios 2 and 4 -> mean 3 -> scale 1/3 (ratio of means would be 10/3)
    d = ref.rescale(tgt, "fp")
    assert d.derivation["mean_ratio"] == 3.0
    assert [z.hi for z in d.zones] == [round(x / 3, 6) for x in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert d.alpha_max == round(0.5 / 3, 6)
    rs = T.ResidualScale("a", (1.0, 1.0), "b", (3.0, 5.0), source="t")
    d2 = ref.rescale(tgt, "fp", residual_scale=rs)
    assert [z.to_json() for z in d2.zones] == [z.to_json() for z in d.zones]
    assert d2.derivation["cross_model"] is not None
    with pytest.raises(ValueError, match="same number of steering sites"):
        ref.rescale([1.0, 2.0, 3.0], "fp")


def test_overdriven_ceiling_old_map_to_w5_is_0708() -> None:
    """Golden: the old map's documented damage point (alpha_max 1.5) rescaled
    onto w5-r32's ruler = 0.708195 — the overdriven edge actually stored in
    dose_bands/w5_r32_measured.json (a wear at .75 on that map produces word
    salad)."""
    old = _band_file("old_map_measured.json")
    w5 = _band_file("w5_r32_measured.json")
    ceiling = T.overdriven_ceiling(old, w5.norm_ref)
    assert ceiling == 0.708195
    assert [z for z in w5.zones if z.key == "overdriven"][0].lo == ceiling


def test_apply_overdriven_ceiling_caps_and_flags() -> None:
    """apply_overdriven_ceiling: a non-overdriven zone straddling the ceiling
    is truncated there and a single overdriven zone runs to alpha_max; the
    flag reports the ladder sampled into damage. Ceiling >= alpha_max is a
    no-op."""
    zs = T.five_zone_band([0.1, 0.2, 0.3, 0.9], 1.0)
    out, sampled = T.apply_overdriven_ceiling(zs, 0.7, 1.0)
    assert sampled is True
    assert [(z.key, z.lo, z.hi) for z in out][-2:] == [("audible", 0.3, 0.7), ("overdriven", 0.7, 1.0)]
    same, s2 = T.apply_overdriven_ceiling(zs, 1.0, 1.0)
    assert same == list(zs) and s2 is False


def test_cross_model_bound_numbers() -> None:
    """cross_model_bound on the banked 3B->8B residual scale: the relative
    hypothesis divides the target ruler by the per-site ||h|| ratio; the
    residual mean ratio is 1.174465 (residual_scale_3b_8b.json). Pinned so a
    port keeps the arithmetic, not just the shape."""
    rs = T.load_residual_scale_file(T.DOSE_BANDS_DIR / "residual_scale_3b_8b.json")
    w5 = _band_file("w5_r32_measured.json")
    banked = _band_file("8b_v1a_derived.json")
    cm = T.cross_model_bound(w5.norm_ref, banked.norm_ref, rs)
    assert cm["residual_scale"]["mean_ratio_target_over_reference"] == 1.174465
    assert cm == banked.derivation["cross_model"]
