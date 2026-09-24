"""Contract tests for the CROSS-MODEL bound on a derived dose band, and for
freezing a derived band to a sidecar (`pleroma.dose.band`).

**The gap these close.** `DoseBand.rescale` transfers a judged band between
maps on one hypothesis: audibility tracks the ABSOLUTE norm of the injected
displacement, `alpha * norm_ref[s]`. That hypothesis is validated between
the old map and w5-r32 — but BOTH are Llama-3.2-3B, and
the competing hypothesis (audibility tracks `||delta|| / ||h||`, the fraction
of the residual stream displaced) makes *identical* predictions whenever the
two maps share a model, because `||h||` cancels. So that validation licenses
neither hypothesis for the 3B -> 8B transfer the 8B v1a map actually needs,
and a derived 8B band carries an assumption that has to be quantified.

`cross_model_bound` measures the size of that ambiguity from banked hidden
states (no GPU, no judges) and `rescale(..., residual_scale=...)` attaches it.
What is asserted here:

  * the bound CHANGES NO SERVED BOUNDARY — a band derived with a residual
    scale is zone-for-zone identical to one derived without it. This is the
    load-bearing property: quantifying an assumption must never be a covert
    way of switching to the unvalidated alternative;
  * a residual scale of all-ones (two models with the same residual norm)
    makes the two hypotheses coincide — `disagreement == 1.0` — which is
    exactly the same-model case the w5-r32 ladder covers;
  * the real 3B -> 8B numbers reproduce: absolute x1.4228 (what the live
    server serves), relative x1.6580, disagreement 1.165x, and the
    served side is the CONSERVATIVE one;
  * `served_is_conservative` tracks which hypothesis puts the boundaries
    lower, in both directions;
  * the bound survives a JSON round trip and reaches `/info` through
    `to_info_json`, because an operator who can see "derived" but not how far
    it reached cannot tell a 1.17x ambiguity from the 2x blowout
    `pleroma.dose.band` exists to prevent;
  * `derive` freezes a DERIVED band and refuses to overwrite a MEASURED one
    with a rescale, and never relabels a derived band as measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pleroma.map.identity import fingerprint_v2, weights_digest
import pytest

from pleroma.dose import band as db

# ── real numbers, all of them measured ───────────────────────────────────────

W5R32_NORM_REF = (1.012, 2.184, 3.239, 4.56)
# the 8B v1a map's ruler, as a live loom reports it in
# /info.ruler.norm_ref_in_force
NORM_REF_8B = (0.673457, 1.55648, 2.305945, 3.28943)
# tests/fixtures/dose_bands/residual_scale_3b_8b.json, measured off
# banked mean_hiddens (4288 3B generations, 11800 8B)
H_NORM_3B = (3.348894567418309, 6.130920276639283,
             8.925550724415299, 13.541854301206147)
H_NORM_8B = (3.32453403098369, 7.33091852089299,
             10.518061125114414, 18.023963677921387)

REPO = Path(__file__).resolve().parents[3]
BANDS = REPO / "tests" / "fixtures" / "dose_bands"


def w5r32_band() -> db.DoseBand:
    return db.load_dose_band_file(BANDS / "w5_r32_measured.json")


def residual_3b_8b() -> db.ResidualScale:
    return db.load_residual_scale_file(BANDS / "residual_scale_3b_8b.json")


def flat_residual(n_sites: int = 4, value: float = 7.0) -> db.ResidualScale:
    """Two models whose residual streams happen to carry the same norm — the
    algebraic stand-in for the SAME-model case, where the absolute and
    relative hypotheses are the same hypothesis."""
    return db.ResidualScale(
        reference_model="model-A", reference_h_norm=tuple([value] * n_sites),
        target_model="model-B", target_h_norm=tuple([value] * n_sites),
        source="test fixture")


# ── the load-bearing property: no served number moves ────────────────────────


def test_attaching_a_bound_changes_no_zone() -> None:
    ref = w5r32_band()
    plain = ref.rescale(NORM_REF_8B, "de6aa6f4b19e63d0")
    bounded = ref.rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                          residual_scale=residual_3b_8b())
    assert bounded.tier == "derived"
    assert bounded.alpha_max == plain.alpha_max
    assert [z.to_json() for z in bounded.zones] == [z.to_json() for z in plain.zones]
    assert bounded.norm_ref == plain.norm_ref


def test_the_served_zones_match_what_the_live_8b_loom_serves() -> None:
    """Pinned against a live 8B loom's own GET /info. If this drifts, the
    checked-in reference band or the 8B map's ruler changed and the band on a
    running instrument differs from the one in git."""
    band = w5r32_band().rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                                residual_scale=residual_3b_8b())
    assert [(z.key, round(z.lo, 6), round(z.hi, 6)) for z in band.zones] == [
        ("subliminal", 0.0, 0.177845),
        ("threshold", 0.177845, 0.711379),
        ("audible", 0.711379, 1.007589),
        ("overdriven", 1.007589, 1.422757),
    ]
    assert band.alpha_max == pytest.approx(1.422757)


# ── the bound itself ─────────────────────────────────────────────────────────


def test_equal_residual_norms_make_the_two_hypotheses_one() -> None:
    """The same-model case: `||h||` cancels, so the relative transfer IS the
    absolute one and the same-model w5-r32 validation covers it."""
    bound = db.cross_model_bound(W5R32_NORM_REF, NORM_REF_8B, flat_residual())
    assert bound["disagreement"] == pytest.approx(1.0)
    assert bound["relative"]["scale"] == pytest.approx(bound["absolute"]["scale"])
    assert bound["residual_scale"]["mean_ratio_target_over_reference"] == pytest.approx(1.0)


def test_the_real_3b_to_8b_bound_reproduces() -> None:
    bound = db.cross_model_bound(W5R32_NORM_REF, NORM_REF_8B, residual_3b_8b())
    assert bound["hypothesis_served"] == "absolute"
    assert bound["absolute"]["scale"] == pytest.approx(1.4228, abs=1e-4)
    assert bound["relative"]["scale"] == pytest.approx(1.6580, abs=1e-4)
    assert bound["disagreement"] == pytest.approx(1.1654, abs=1e-4)
    assert bound["residual_scale"]["mean_ratio_target_over_reference"] == pytest.approx(
        1.174465, abs=1e-5)
    # ★ 1.165x, not 2x. The failure this module exists to prevent is a gauge
    # wrong by 2x; this ambiguity is a fifth of that and is bounded, which is
    # the whole point of measuring it instead of worrying about it.
    assert bound["disagreement"] < 1.25


def test_the_served_side_is_the_conservative_one_for_the_8b() -> None:
    """The 8B's residual stream is LARGER than the 3B's, so the relative
    hypothesis would put every boundary HIGHER — i.e. would call a given alpha
    quieter. The absolute numbers we serve call it louder, which errs toward
    under-dosing rather than toward silent damage."""
    bound = db.cross_model_bound(W5R32_NORM_REF, NORM_REF_8B, residual_3b_8b())
    assert bound["served_is_conservative"] is True
    assert bound["conservative"] == "absolute"
    assert bound["relative"]["scale"] > bound["absolute"]["scale"]


def test_a_smaller_target_residual_flips_which_side_is_conservative() -> None:
    """The flag must be computed, not assumed. If the target model's residual
    stream were SMALLER, the relative hypothesis would put the boundaries
    lower and the served absolute band would be the LESS conservative one —
    and the operator has to be told that."""
    shrunk = db.ResidualScale(
        reference_model="ref", reference_h_norm=H_NORM_3B,
        target_model="tgt", target_h_norm=tuple(h * 0.5 for h in H_NORM_3B),
        source="test fixture")
    bound = db.cross_model_bound(W5R32_NORM_REF, NORM_REF_8B, shrunk)
    assert bound["served_is_conservative"] is False
    assert bound["conservative"] == "relative"
    band = w5r32_band().rescale(NORM_REF_8B, None, residual_scale=shrunk)
    assert "LESS CONSERVATIVE" in band.source


def test_the_source_string_names_the_transfer_and_the_alternative() -> None:
    band = w5r32_band().rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                                residual_scale=residual_3b_8b())
    assert "CROSS-MODEL" in band.source
    assert "llama-3.2-3b-instruct" in band.source
    assert "llama-3.1-8b-instruct" in band.source
    assert "CONSERVATIVE" in band.source
    assert "PROVISIONAL" in band.source, "the derived-band warning must survive"


# ── it must reach the operator ───────────────────────────────────────────────


def test_the_bound_is_served_on_info() -> None:
    band = w5r32_band().rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                                residual_scale=residual_3b_8b())
    info = band.to_info_json()
    assert info["tier"] == "derived"
    cross = info["derivation"]["cross_model"]
    assert cross["disagreement"] == pytest.approx(1.1654, abs=1e-4)
    assert cross["served_is_conservative"] is True
    assert json.dumps(info)  # JSON-safe, the /info contract


def test_none_info_json_still_answers_every_key_a_band_would() -> None:
    """`tier: none` must not be a differently-shaped object, or a UI that
    reads `derivation` on a real band throws on an uncalibrated one."""
    band = w5r32_band().rescale(NORM_REF_8B, None, residual_scale=residual_3b_8b())
    assert set(db.NONE_INFO_JSON) == set(band.to_info_json())
    assert db.dose_band_info_json(None)["derivation"] is None


def test_a_derived_band_without_a_residual_scale_says_so_rather_than_nothing() -> None:
    band = w5r32_band().rescale(NORM_REF_8B, None)
    assert band.derivation is not None
    assert band.derivation["cross_model"] is None
    assert band.derivation["scale"] == pytest.approx(1.422757, abs=1e-6)
    assert band.derivation["per_site_spread"] == pytest.approx(0.0795, abs=1e-4)


def test_a_measured_band_derives_nothing_and_claims_nothing() -> None:
    assert w5r32_band().derivation is None


def test_the_bound_survives_a_json_round_trip() -> None:
    band = w5r32_band().rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                                residual_scale=residual_3b_8b())
    back = db.DoseBand.from_json(json.loads(json.dumps(band.to_json())))
    assert back.derivation == band.derivation
    assert back.tier == "derived"


def test_resolve_forwards_the_residual_scale_only_on_the_derived_branch() -> None:
    measured = w5r32_band()
    got = db.resolve_dose_band(
        measured=measured, reference=None, target_norm_ref=W5R32_NORM_REF,
        target_map_fingerprint=measured.map_fingerprint or "x",
        residual_scale=residual_3b_8b())
    assert got is measured, "a measured band must be returned untouched"
    assert got.derivation is None

    derived = db.resolve_dose_band(
        measured=None, reference=measured, target_norm_ref=NORM_REF_8B,
        target_map_fingerprint="de6aa6f4b19e63d0",
        residual_scale=residual_3b_8b())
    assert derived is not None and derived.derivation is not None
    assert derived.derivation["cross_model"] is not None


# ── ResidualScale validation: it must refuse, not guess ──────────────────────


@pytest.mark.parametrize("kwargs", [
    {"reference_h_norm": (1.0, 2.0), "target_h_norm": (1.0, 2.0, 3.0)},
    {"reference_h_norm": (), "target_h_norm": ()},
    {"reference_h_norm": (1.0, 0.0), "target_h_norm": (1.0, 2.0)},
    {"reference_h_norm": (1.0, -2.0), "target_h_norm": (1.0, 2.0)},
    {"reference_h_norm": (1.0, float("nan")), "target_h_norm": (1.0, 2.0)},
])
def test_malformed_residual_scales_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        db.ResidualScale(reference_model="a", target_model="b",
                          source="test fixture", **kwargs)


def test_a_site_count_mismatch_against_the_maps_is_refused() -> None:
    with pytest.raises(ValueError, match="one entry per site"):
        db.cross_model_bound(W5R32_NORM_REF, NORM_REF_8B, flat_residual(n_sites=3))


def test_the_checked_in_residual_scale_file_loads_and_matches_the_measurement() -> None:
    rs = residual_3b_8b()
    assert rs.reference_h_norm == pytest.approx(np.asarray(H_NORM_3B))
    assert rs.target_h_norm == pytest.approx(np.asarray(H_NORM_8B))
    assert rs.reference_sites == (8, 15, 19, 22)
    assert rs.target_sites == (9, 17, 21, 25)
    assert rs.h_ratios() == pytest.approx(
        [0.992726, 1.195729, 1.178422, 1.330982], abs=1e-5)


def test_a_residual_scale_newer_than_this_reader_is_refused(tmp_path: Path) -> None:
    blob = json.loads((BANDS / "residual_scale_3b_8b.json").read_text())
    blob["schema"] = db.CURRENT_SCHEMA + 1
    p = tmp_path / "future.json"
    p.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="newer than this reader"):
        db.load_residual_scale_file(p)


def test_an_unreadable_residual_scale_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        db.load_residual_scale_file(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        db.load_residual_scale_file(bad)


# ── it must reach the operator THROUGH THE SERVER, not just through the band ──


def test_the_bound_reaches_get_info_through_build_info_payload() -> None:
    """The band object having the number is not enough — /info is what the UI
    and every other client reads."""
    from pleroma.serve import legacy as ls

    band = w5r32_band().rescale(NORM_REF_8B, "de6aa6f4b19e63d0",
                                residual_scale=residual_3b_8b())
    payload = ls.build_info_payload(
        map_path="corpus_8b/v1a/loom_map_v1a_r64.npz", map_meta={"rank": 64},
        sites=[9, 17, 21, 25], branches=["base", "loom"], default_k=12,
        future_tokens=96, detach_wear_default=False, n_sessions=0,
        harvest_worker="127.0.0.1:8784", restored_sessions=0,
        persistence={"enabled": True, "dir": "/x/sessions"},
        auto_policies=[], loudness_ref=1.956, dose_band=band)
    cross = payload["dose_band"]["derivation"]["cross_model"]
    assert cross["disagreement"] == pytest.approx(1.1654, abs=1e-4)
    assert cross["served_is_conservative"] is True
    assert json.dumps(payload)


def test_the_server_exposes_a_flag_for_the_residual_scale() -> None:
    """The bound is only ever attached if an operator can ask for it. Static,
    because loom_serve's parser is built inside main() and main() loads torch
    and a 9 GB map."""
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(
        (REPO / "pleroma" / "serve").rglob("*.py")))  # the server's modules
    assert "--dose-band-residual-scale" in src
    assert "load_residual_scale_file" in src
    assert "residual_scale=residual_scale" in src, (
        "the flag must actually reach resolve_dose_band")
    assert "changes NO served boundary" in src or "changes NO served" in src


# ── `derive`: freezing a band without promoting it ───────────────────────────


def _fake_map(path: Path, norm_ref: tuple[float, ...]) -> str:
    """A minimal loom_map_*.npz carrying only what `load_map_identity` reads."""
    sites = list(range(len(norm_ref)))
    mu_y = np.arange(len(norm_ref) * 3, dtype=np.float64)
    meta = {"rank": 64, "stage": "test fixture"}
    w_u = np.eye(3, 2)  # load_map_identity hashes the weights (v2, X-06)
    np.savez_compressed(
        path, mu_y=mu_y, sites=np.asarray(sites),
        norm_ref=np.asarray(norm_ref, dtype=np.float64),
        meta=np.array(json.dumps(meta)), W_U=w_u)
    return fingerprint_v2(sites, np.asarray(norm_ref), mu_y, meta,
                          weights_digest({"W_U": w_u}))


def test_derive_writes_a_derived_sidecar_with_the_bound(tmp_path: Path) -> None:
    map_path = tmp_path / "loom_map_8b.npz"
    fp = _fake_map(map_path, NORM_REF_8B)
    out = tmp_path / "band.json"
    db.main(["derive", "--map", str(map_path),
             "--dose-band-reference", str(BANDS / "w5_r32_measured.json"),
             "--residual-scale", str(BANDS / "residual_scale_3b_8b.json"),
             "--out", str(out)])
    frozen = db.load_dose_band_file(out)
    assert frozen.tier == "derived", "freezing a band must never promote it"
    assert frozen.map_fingerprint == fp
    assert frozen.derivation is not None
    assert frozen.derivation["cross_model"]["disagreement"] == pytest.approx(
        1.1654, abs=1e-4)
    # and the frozen file resolves to exactly what the server would have
    # computed on the fly
    live = db.resolve_dose_band(
        measured=None, reference=db.load_dose_band_file(BANDS / "w5_r32_measured.json"),
        target_norm_ref=NORM_REF_8B, target_map_fingerprint=fp,
        residual_scale=residual_3b_8b())
    assert live is not None
    assert [z.to_json() for z in frozen.zones] == [z.to_json() for z in live.zones]


def test_derive_refuses_to_overwrite_a_measured_sidecar_with_a_rescale(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "loom_map_8b.npz"
    fp = _fake_map(map_path, NORM_REF_8B)
    sidecar = db.default_sidecar_path(map_path)
    real = w5r32_band()
    db.save_dose_band_file(
        db.DoseBand(tier="measured", zones=real.zones, norm_ref=NORM_REF_8B,
                    source="a real 8B ladder", alpha_max=real.alpha_max,
                    map_fingerprint=fp),
        sidecar)
    with pytest.raises(SystemExit, match="will not"):
        db.main(["derive", "--map", str(map_path),
                 "--dose-band-reference", str(BANDS / "w5_r32_measured.json"),
                 "--out", str(tmp_path / "other.json")])


def test_derive_needs_a_reference(tmp_path: Path) -> None:
    map_path = tmp_path / "loom_map_8b.npz"
    _fake_map(map_path, NORM_REF_8B)
    with pytest.raises(SystemExit, match="dose-band-reference"):
        db.main(["derive", "--map", str(map_path)])


def test_derive_will_not_clobber_an_existing_file_without_force(tmp_path: Path) -> None:
    map_path = tmp_path / "loom_map_8b.npz"
    _fake_map(map_path, NORM_REF_8B)
    out = tmp_path / "band.json"
    out.write_text("{}")
    args = ["derive", "--map", str(map_path),
            "--dose-band-reference", str(BANDS / "w5_r32_measured.json"),
            "--out", str(out)]
    with pytest.raises(SystemExit, match="already exists"):
        db.main(args)
    db.main(args + ["--force"])
    assert db.load_dose_band_file(out).tier == "derived"


# ── the 8B MEASURED band, from the 8B confirmation ladder ───────────────────


