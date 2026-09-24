"""Contract tests for predicted-magnitude dosing (`dose_policy` on /wear).

WHAT IS BEING PINNED, AND WHY EACH ONE MATTERS.

`LoomMap.lever_of` ends with `out[s] = lever[s] * (norm_ref[s] / n[s])`: every
candidate of a fan leaves the map at a different predicted magnitude and is
injected at an identical one. The cost is measured: v1a predicts WITHIN-FAN displacement magnitude at r = .757 (v0:
.538) and the constant ruler deletes that prediction before it reaches the
model. `dose_policy: "predicted"` divides by the FAN's mean raw norm instead
of the candidate's own, restoring the spread while leaving the fan's average
dose exactly where it was.

Four things can go wrong, and all four would be silent:

  1. **the default changes.** Every dose band, ladder and published alpha in
     this project was measured under the flat ruler. If `flat` is not
     BYTE-IDENTICAL to plain `lever_of` output, every one of those numbers
     quietly stops applying. `test_flat_policy_is_byte_identical_to_the_ruler`
     and `test_flat_apply_returns_the_same_object` are that proof.
  2. **alpha stops meaning one thing.** The design is only sound because the
     per-site scales average to exactly 1 over the fan — so the FAN's mean
     dose is unchanged and the operator's alpha keeps its calibration.
     `test_the_fans_average_dose_is_unchanged` is that invariant.
  3. **a clamp fires silently.** A scale of 2.0 at alpha 0.45 injects the
     absolute magnitude of alpha 0.90, past the derived overdriven ceiling.
     `pleroma.dose.band`'s doctrine: a wrong dose gauge is worse than no gauge,
     because it is trusted. Every clamp must appear in the receipt.
  4. **a fan mean gets invented.** A mean borrowed from another draw, or
     conjured for a wear that has no draw at all (/wear_code), is a different
     unit wearing alpha's name. Those paths must REFUSE, not degrade.

Everything here is pure numpy — no model, no server, no GPU (the house
pattern: test_loom_detach_wear.py's header explains why the decision
functions live at module level).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.serve import legacy as ls

SITES = [8, 15, 19, 22]
HIDDEN = 12
NORM_REF = [1.0, 2.0, 3.0, 4.0]


# ── a real LoomMap, small enough to build in a tmpdir ────────────────────────
#
# The byte-identity proof has to run through the ACTUAL `lever_of`, not a
# stand-in: what it claims is that the worn vectors are the same bytes the
# old server produced, and the old server's vectors were `lever_of`'s output
# handed straight to `sess.worn["vectors"]`.


def write_map(tmp_path: Path, *, n_v3: int = 7, n_bins: int = 5,
              rank: int = 3) -> tuple[Path, Path]:
    """A minimal loom-map npz + matching discriminants, SVD-factored form
    (no dense `W`) — the same storage shape v1a ships, so `self.Vt` is the
    file's own `W_Vt` and banked codes stay in the basis they were made in.
    A map re-derived from a dense `W` recomputes its SVD, and banked codes
    can land in a different basis."""
    rng = np.random.default_rng(20260921)
    n_in, n_out = n_v3 + n_bins, len(SITES) * HIDDEN
    disc = tmp_path / "disc.npz"
    np.savez(disc,
             FULL_mean=rng.standard_normal(n_v3),
             FULL_scale=np.abs(rng.standard_normal(n_v3)) + 0.5)
    sha = hashlib.sha256(disc.read_bytes()).hexdigest()
    u = rng.standard_normal((n_in, rank))
    s = np.abs(rng.standard_normal(rank)) + 1.0
    vt = np.linalg.qr(rng.standard_normal((n_out, rank)))[0].T
    map_path = tmp_path / "map.npz"
    np.savez(
        map_path,
        W_U=u, W_S=s, W_Vt=vt,
        mu_in=rng.standard_normal(n_in),
        mu_y=rng.standard_normal(n_out),
        v3_mu=rng.standard_normal(n_v3).astype(np.float32),
        v3_sd=(np.abs(rng.standard_normal(n_v3)) + 0.5).astype(np.float32),
        v3_dead=np.zeros(n_v3, dtype=bool),
        bins_mu=rng.standard_normal(n_bins).astype(np.float32),
        bins_sd=(np.abs(rng.standard_normal(n_bins)) + 0.5).astype(np.float32),
        bins_dead=np.zeros(n_bins, dtype=bool),
        sites=np.asarray(SITES, dtype=np.int64),
        norm_ref=np.asarray(NORM_REF, dtype=np.float64),
        meta=json.dumps({"rank": rank, "discriminants_sha256": sha}),
    )
    return map_path, disc


@pytest.fixture
def loom_map(tmp_path: Path) -> ls.LoomMap:
    return ls.LoomMap(*write_map(tmp_path))


def a_fan(loom_map: ls.LoomMap, k: int = 6) -> list[dict[str, Any]]:
    """K candidates in the exact shape `do_loom` builds them."""
    rng = np.random.default_rng(7)
    out = []
    for j in range(k):
        sig = rng.standard_normal(loom_map.v3_mu.size)
        brow = rng.standard_normal(loom_map.bins_mu.size)
        lever, raw, code = loom_map.lever_of(sig, brow)
        out.append({"index": j, "lever": lever, "note": None,
                    "raw_norms": raw, "code": code})
    return out


# ══ 1. the default must not move ════════════════════════════════════════════


def test_flat_apply_returns_the_same_object() -> None:
    """Not `lever * 1.0` — the identical object. The cheapest possible
    guarantee that the default path is untouched."""
    lever = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert ls.flat_dose_scale(4).apply(lever) is lever


def test_flat_policy_is_byte_identical_to_the_ruler(loom_map: ls.LoomMap) -> None:
    """★ THE PROOF. On a FIXED input, the vectors a flat-policy wear stores
    are byte-for-byte `lever_of`'s output, unmodified — the vectors a server
    with no dose policy stores."""
    rng = np.random.default_rng(1234)
    for _ in range(8):
        sig = rng.standard_normal(loom_map.v3_mu.size)
        brow = rng.standard_normal(loom_map.bins_mu.size)
        pre_change, raw, _code = loom_map.lever_of(sig, brow)  # the old path
        dose = ls.dose_scale_for(policy="flat", raw_norms=raw,
                                 fan_mean=[9.0] * 4,  # present and IGNORED
                                 n_sites=loom_map.n_sites)
        worn = dose.apply(pre_change)
        assert worn.dtype == pre_change.dtype
        assert worn.shape == pre_change.shape
        assert worn.tobytes() == pre_change.tobytes()


def test_flat_is_the_module_default() -> None:
    assert ls.DEFAULT_DOSE_POLICY == "flat"
    assert ls.DOSE_POLICIES == ("flat", "predicted")
    assert next(p for p in ls.DOSE_POLICY_INFO if p["key"] == "flat")["default"]


def test_flat_receipt_says_flat(loom_map: ls.LoomMap) -> None:
    blob = ls.flat_dose_scale(loom_map.n_sites).to_json()
    assert blob["policy"] == "flat"
    assert blob["scale"] == [1.0] * loom_map.n_sites
    assert blob["clamped"] == []
    assert blob["fan_mean_raw_norms"] is None


# ══ 2. the predicted policy's arithmetic ════════════════════════════════════


def test_predicted_scale_is_candidate_over_fan_mean() -> None:
    dose = ls.dose_scale_for(policy="predicted",
                             raw_norms=[2.0, 6.0, 3.0, 8.0],
                             fan_mean=[2.0, 4.0, 4.0, 8.0], n_sites=4)
    assert dose.scale == pytest.approx([1.0, 1.5, 0.75, 1.0])
    assert dose.scale_raw == pytest.approx([1.0, 1.5, 0.75, 1.0])
    assert dose.clamped == ()
    assert dose.policy == "predicted"


def test_a_candidate_at_the_fan_average_gets_exactly_todays_dose(
    loom_map: ls.LoomMap,
) -> None:
    """The design's central promise: at the fan mean, predicted == flat, to
    the byte."""
    fan = a_fan(loom_map)
    mean = ls.fan_mean_raw_norms(fan)
    average_candidate = dict(fan[0])
    average_candidate["raw_norms"] = list(mean)
    dose = ls.dose_scale_for(policy="predicted",
                             raw_norms=average_candidate["raw_norms"],
                             fan_mean=mean, n_sites=loom_map.n_sites)
    assert dose.scale == pytest.approx([1.0] * loom_map.n_sites)
    worn = dose.apply(average_candidate["lever"])
    assert np.array_equal(worn, average_candidate["lever"])


def test_the_fans_average_dose_is_unchanged(loom_map: ls.LoomMap) -> None:
    """★ WHY ALPHA KEEPS ITS MEANING. Averaged over the fan, the per-site
    scale is exactly 1 — so the fan's mean injected magnitude under
    `predicted` equals its magnitude under `flat`. Only the spread moves."""
    fan = a_fan(loom_map, k=16)
    mean = ls.fan_mean_raw_norms(fan)
    scales = np.stack([
        ls.dose_scale_for(policy="predicted", raw_norms=c["raw_norms"],
                          fan_mean=mean, n_sites=loom_map.n_sites).scale_raw
        for c in fan
    ])
    assert scales.mean(axis=0) == pytest.approx([1.0] * loom_map.n_sites,
                                                rel=1e-12)
    # and it is a real redistribution, not a no-op
    assert scales.std(axis=0).max() > 1e-6


def test_louder_predicted_candidates_are_injected_harder(
    loom_map: ls.LoomMap,
) -> None:
    fan = a_fan(loom_map, k=10)
    mean = ls.fan_mean_raw_norms(fan)
    by_loudness = sorted(fan, key=lambda c: float(np.mean(c["raw_norms"])))
    quiet, loud = by_loudness[0], by_loudness[-1]
    dq = ls.dose_scale_for(policy="predicted", raw_norms=quiet["raw_norms"],
                           fan_mean=mean, n_sites=loom_map.n_sites)
    dl = ls.dose_scale_for(policy="predicted", raw_norms=loud["raw_norms"],
                           fan_mean=mean, n_sites=loom_map.n_sites)
    assert dl.scale_mean > dq.scale_mean
    # the injected per-site norms move with it
    nq = [float(np.linalg.norm(r)) for r in dq.apply(quiet["lever"])]
    nl = [float(np.linalg.norm(r)) for r in dl.apply(loud["lever"])]
    flat_norms = [float(np.linalg.norm(r)) for r in quiet["lever"]]
    assert flat_norms == pytest.approx(NORM_REF, rel=1e-9)  # the ruler, before
    assert all(a < b for a, b in zip(nq, nl))


def test_predicted_rescales_the_ruler_exactly(loom_map: ls.LoomMap) -> None:
    """The injected per-site norm must be norm_ref[s] * scale[s] — nothing
    else, so the dose stays readable against the band."""
    fan = a_fan(loom_map)
    mean = ls.fan_mean_raw_norms(fan)
    c = fan[3]
    dose = ls.dose_scale_for(policy="predicted", raw_norms=c["raw_norms"],
                             fan_mean=mean, n_sites=loom_map.n_sites)
    got = [float(np.linalg.norm(r)) for r in dose.apply(c["lever"])]
    want = [n * s for n, s in zip(NORM_REF, dose.scale)]
    assert got == pytest.approx(want, rel=1e-9)


# ══ 3. the clamp is never silent ════════════════════════════════════════════


def test_clamp_bounds_are_the_briefed_ones() -> None:
    assert (ls.DOSE_SCALE_MIN, ls.DOSE_SCALE_MAX) == (0.5, 2.0)


def test_clamp_fires_and_is_reported() -> None:
    dose = ls.dose_scale_for(policy="predicted",
                             raw_norms=[10.0, 0.1, 4.0, 4.0],
                             fan_mean=[2.0, 2.0, 4.0, 4.0], n_sites=4)
    assert dose.scale == pytest.approx([2.0, 0.5, 1.0, 1.0])
    assert dose.scale_raw == pytest.approx([5.0, 0.05, 1.0, 1.0])
    assert dose.was_clamped is True
    bounds = {c["site_index"]: c["bound"] for c in dose.clamped}
    assert bounds == {0: "max", 1: "min"}
    blob = dose.to_json()
    assert len(blob["clamped"]) == 2
    assert blob["clamp_range"] == [0.5, 2.0]
    # the RAW ask survives into the receipt: a clamp must be auditable
    assert blob["scale_raw"][0] == pytest.approx(5.0)


def test_no_clamp_means_an_empty_clamp_list() -> None:
    dose = ls.dose_scale_for(policy="predicted", raw_norms=[2.0, 2.0],
                             fan_mean=[2.0, 2.0], n_sites=2)
    assert dose.was_clamped is False
    assert dose.to_json()["clamped"] == []


# ══ 4. refusals — never invent a fan mean ═══════════════════════════════════


def test_predicted_refuses_without_a_fan_mean() -> None:
    with pytest.raises(ValueError, match="fan mean"):
        ls.dose_scale_for(policy="predicted", raw_norms=[1.0, 2.0],
                          fan_mean=None, n_sites=2)


def test_predicted_refuses_a_candidate_that_did_not_harvest() -> None:
    with pytest.raises(ValueError, match="raw map-output norms"):
        ls.dose_scale_for(policy="predicted", raw_norms=None,
                          fan_mean=[1.0, 2.0], n_sites=2)


@pytest.mark.parametrize("raw,mean,n,match", [
    ([1.0, 2.0], [1.0, 2.0, 3.0], 2, "disagree"),
    ([1.0, 2.0], [1.0, 2.0], 3, "disagree"),
    ([1.0, np.nan], [1.0, 2.0], 2, "non-finite"),
    ([1.0, 2.0], [1.0, 0.0], 2, "non-positive"),
    ([1.0, -2.0], [1.0, 2.0], 2, "non-positive"),
])
def test_predicted_refuses_malformed_inputs(raw: list[float], mean: list[float],
                                            n: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ls.dose_scale_for(policy="predicted", raw_norms=raw, fan_mean=mean,
                          n_sites=n)


def test_unknown_policy_refuses() -> None:
    with pytest.raises(ValueError, match="dose_policy must be one of"):
        ls.dose_scale_for(policy="loudest", raw_norms=[1.0], fan_mean=[1.0],
                          n_sites=1)


# ══ 5. the fan mean comes from the draw, and only from the draw ═════════════


def test_fan_mean_is_the_per_site_mean_over_harvested_candidates() -> None:
    fan = [{"raw_norms": [1.0, 2.0]}, {"raw_norms": [3.0, 6.0]}]
    assert ls.fan_mean_raw_norms(fan) == pytest.approx([2.0, 4.0])


def test_fan_mean_skips_candidates_that_did_not_harvest() -> None:
    fan = [{"raw_norms": [1.0, 2.0]}, {"raw_norms": None, "note": "boom"},
           {"raw_norms": [3.0, 6.0]}]
    assert ls.fan_mean_raw_norms(fan) == pytest.approx([2.0, 4.0])


def test_fan_mean_is_none_when_nothing_harvested() -> None:
    assert ls.fan_mean_raw_norms([{"raw_norms": None}, {"lever": None}]) is None


@pytest.mark.parametrize("rows,match", [
    ([{"raw_norms": [1.0, np.inf]}], "non-finite"),
    ([{"raw_norms": [1.0, 0.0]}], "non-positive"),
])
def test_fan_mean_refuses_a_broken_fan(rows: list[dict[str, Any]],
                                       match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ls.fan_mean_raw_norms(rows)


def test_a_session_carries_the_fan_mean_and_starts_without_one() -> None:
    sess = ls.LoomSession()
    assert sess.fan_mean_raw_norms is None  # a fresh session has no fan


# ══ 6. request-level resolution ═════════════════════════════════════════════


def test_body_overrides_the_server_default() -> None:
    assert ls.resolve_dose_policy("predicted", "flat") == "predicted"
    assert ls.resolve_dose_policy("flat", "predicted") == "flat"


def test_absent_falls_back_to_the_server_default() -> None:
    assert ls.resolve_dose_policy(None, "flat") == "flat"
    assert ls.resolve_dose_policy(None, "predicted") == "predicted"


@pytest.mark.parametrize("bad", [True, 1, 0.5, ["predicted"], {"k": 1}])
def test_a_non_string_dose_policy_is_refused(bad: Any) -> None:
    """The `request_detach_wear` doctrine: a truthy-but-wrong value silently
    changing how hard the model gets pushed is the whole risk."""
    with pytest.raises(ValueError, match="must be a string"):
        ls.resolve_dose_policy(bad, "flat")


@pytest.mark.parametrize("bad", ["", "Predicted ", "loudest", "none"])
def test_an_unknown_dose_policy_string_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ls.resolve_dose_policy(bad, "flat")


# ══ 7. the receipt survives a restart ═══════════════════════════════════════


def test_dose_receipt_round_trips(loom_map: ls.LoomMap) -> None:
    fan = a_fan(loom_map)
    mean = ls.fan_mean_raw_norms(fan)
    dose = ls.dose_scale_for(policy="predicted", raw_norms=fan[2]["raw_norms"],
                             fan_mean=mean, n_sites=loom_map.n_sites)
    back = ls.DoseScale.from_json(json.loads(json.dumps(dose.to_json())))
    assert back.policy == "predicted"
    assert back.scale == pytest.approx(dose.scale, rel=1e-5)
    assert back.fan_mean_raw_norms == pytest.approx(mean, rel=1e-5)


def test_worn_snapshot_carries_the_dose(loom_map: ls.LoomMap) -> None:
    fan = a_fan(loom_map)
    mean = ls.fan_mean_raw_norms(fan)
    dose = ls.dose_scale_for(policy="predicted", raw_norms=fan[1]["raw_norms"],
                             fan_mean=mean, n_sites=loom_map.n_sites)
    worn = {"index": 1, "alpha": 0.45, "loom_id": "abc-000",
            "vectors": dose.apply(fan[1]["lever"]), "code": fan[1]["code"],
            "loom_dir": "/x/loom_000", "map_fingerprint": "deadbeefdeadbeef",
            "dose": dose.to_json()}
    snap = ls.WornSnapshot.from_worn(worn)
    again = ls.WornSnapshot.from_json(json.loads(json.dumps(snap.to_json())))
    assert again.dose is not None
    assert again.dose["policy"] == "predicted"
    assert again.to_worn()["dose"] == snap.dose


def test_an_older_snapshot_with_no_dose_restores_as_flat() -> None:
    """Back-compat: a snapshot written without a dose block comes from a
    server with only one behaviour. A missing block can only mean flat."""
    snap = ls.WornSnapshot.from_json(
        {"index": 3, "alpha": 0.5, "loom_id": "x-000",
         "per_site_norms_at_alpha1": [1.0, 2.0, 3.0, 4.0]})
    assert snap.dose is None
    assert snap.to_worn()["dose"] is None


def test_worn_public_always_reports_the_dose(loom_map: ls.LoomMap) -> None:
    sess = ls.LoomSession()
    fan = a_fan(loom_map)
    dose = ls.flat_dose_scale(loom_map.n_sites)
    sess.worn = {"index": 0, "alpha": 0.5, "loom_id": "x-000",
                 "vectors": fan[0]["lever"], "dose": dose.to_json()}
    assert sess.worn_public()["dose"]["policy"] == "flat"


# ── rehydration: a restored predicted wear must come back at ITS magnitude ──


def write_loom_dir(root: Path, indices: list[int], n_feat: int = 5,
                   n_bins: int = 4) -> Path:
    (root / "signatures").mkdir(parents=True, exist_ok=True)
    for j in indices:
        np.savez(root / "signatures" / f"gen_{j:03d}.npz",
                 features=np.full(n_feat, float(j) + 1.0))
    np.savez(root / "bins.npz",
             features=np.stack([np.full(n_bins, float(j) + 0.5)
                                for j in indices]),
             generation_id=np.array(indices))
    return root


def fake_lever_of(sig: np.ndarray, brow: np.ndarray) -> tuple[
        np.ndarray, list[float], list[float]]:
    lever = np.outer(np.array([1.0, 2.0, 3.0]), np.concatenate([sig, brow]))
    return lever, [1.0, 2.0, 3.0], [0.0] * 8


def test_rehydrate_reapplies_a_persisted_predicted_dose(tmp_path: Path) -> None:
    """★ Without this, a session steered under `predicted` comes back from a
    restart wearing the FLAT magnitude under the same alpha — a dose change
    nobody asked for and nobody would see."""
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    dose = ls.DoseScale(policy="predicted", scale=(1.5, 0.8, 1.2),
                        scale_raw=(1.5, 0.8, 1.2))
    worn = {"index": 2, "alpha": 0.4, "loom_dir": str(loom_dir),
            "dose": dose.to_json()}
    got = ls.rehydrate_worn_vectors(worn, fake_lever_of)
    flat, _, _ = fake_lever_of(np.full(5, 3.0), np.full(4, 2.5))
    assert np.allclose(got, flat * np.array([[1.5], [0.8], [1.2]]))


def test_rehydrate_without_a_dose_block_is_unchanged(tmp_path: Path) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    worn = {"index": 2, "alpha": 0.4, "loom_dir": str(loom_dir)}
    got = ls.rehydrate_worn_vectors(worn, fake_lever_of)
    flat, _, _ = fake_lever_of(np.full(5, 3.0), np.full(4, 2.5))
    assert got.tobytes() == flat.tobytes()


def test_rehydrate_with_a_flat_dose_block_is_byte_identical(
    tmp_path: Path,
) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    worn = {"index": 2, "alpha": 0.4, "loom_dir": str(loom_dir),
            "dose": ls.flat_dose_scale(3).to_json()}
    got = ls.rehydrate_worn_vectors(worn, fake_lever_of)
    flat, _, _ = fake_lever_of(np.full(5, 3.0), np.full(4, 2.5))
    assert got.tobytes() == flat.tobytes()


def test_rehydrate_refuses_a_dose_that_does_not_fit_the_lever(
    tmp_path: Path,
) -> None:
    loom_dir = write_loom_dir(tmp_path / "loom_000", [0, 2])
    worn = {"index": 2, "alpha": 0.4, "loom_dir": str(loom_dir),
            "dose": ls.DoseScale(policy="predicted", scale=(1.5, 0.8),
                                 scale_raw=(1.5, 0.8)).to_json()}
    with pytest.raises(ValueError, match="refusing to re-wear"):
        ls.rehydrate_worn_vectors(worn, fake_lever_of)


# ══ 8. effective alpha — the band must be read at the right number ══════════


def test_effective_alpha_is_alpha_under_flat() -> None:
    blob = ls.effective_alpha_json(0.45, ls.flat_dose_scale(4))
    assert blob["alpha_effective_mean"] == pytest.approx(0.45)
    assert blob["alpha_effective_per_site"] == pytest.approx([0.45] * 4)
    assert "IS alpha" in blob["note"]


def test_effective_alpha_tracks_the_scale_under_predicted() -> None:
    dose = ls.dose_scale_for(policy="predicted", raw_norms=[3.0, 1.0],
                             fan_mean=[2.0, 2.0], n_sites=2)
    blob = ls.effective_alpha_json(0.45, dose)
    assert blob["alpha_effective_per_site"] == pytest.approx([0.675, 0.225])
    assert blob["alpha_effective_range"] == pytest.approx([0.225, 0.675])
    assert "not at alpha" in blob["note"]


def test_a_clamped_scale_can_push_effective_alpha_past_the_band() -> None:
    """Documented, not prevented: at the clamp a wear at alpha 0.45 injects
    the absolute magnitude of alpha 0.90. The operator must be able to SEE
    that, which is the whole reason this block is on the response."""
    dose = ls.dose_scale_for(policy="predicted", raw_norms=[9.0],
                             fan_mean=[1.0], n_sites=1)
    assert dose.scale == pytest.approx([2.0])
    assert ls.effective_alpha_json(0.45, dose)["alpha_effective_mean"] == \
        pytest.approx(0.9)


# ══ 9. what /info advertises ════════════════════════════════════════════════


def info(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        map_path="/m.npz", map_meta={"rank": 64}, sites=SITES,
        branches=["base", "loom"], default_k=16, future_tokens=96,
        detach_wear_default=False, n_sessions=0, harvest_worker=None,
        restored_sessions=0, persistence={}, auto_policies=ls.AUTO_POLICY_INFO,
        loudness_ref=2.68, dose_band=None,
    )
    base.update(over)
    return ls.build_info_payload(**base)


def test_info_still_answers_with_only_the_old_arguments() -> None:
    """Additive-only: a caller passing only the original arguments keeps
    working."""
    blob = info()
    for key in ("map", "sites", "branches", "default_k", "future_tokens",
                "detach_wear_default", "n_sessions", "persistence",
                "auto_policies", "loudness_ref", "dose_band"):
        assert key in blob


def test_info_lists_every_selectable_policy_with_a_status() -> None:
    keys = [p["key"] for p in info()["auto_policies"]]
    assert tuple(keys) == ls.AUTO_POLICIES  # the list and the behaviour agree
    for p in info()["auto_policies"]:
        assert p["description"] and p["status"]
        assert "needs_wear" in p  # the UI's own field, untouched


def test_the_policy_table_matches_pick_autos_behaviour() -> None:
    assert tuple(p["key"] for p in ls.AUTO_POLICY_INFO) == ls.AUTO_POLICIES


def test_info_says_gauge_is_adopted_and_NOW_WIRED() -> None:
    """★ The probed gauge policy is adopted by measurement AND wired.

    /info must (a) offer the policy, (b) carry its measured share of an
    oracle's picking skill, (c) carry the refusal of the same gauge as an
    INSTRUMENT (docs/FINDINGS.md section 8: a pick policy, not a measurement),
    and (d) price it, because a policy that costs seconds and does not say so
    misleads the operator.
    """
    blob = info()
    keys = [p["key"] for p in blob["auto_policies"]]
    assert "gauge" in keys
    # The "unavailable" list is empty — kept as a key, not removed, so a
    # client that reads it sees "nothing missing" rather than a missing field.
    assert blob["auto_policies_unavailable"] == []
    gauge = next(p for p in blob["auto_policies"] if p["key"] == "gauge")
    assert "ADOPTED BY MEASUREMENT" in gauge["status"]   # the adoption
    assert "71%" in gauge["status"]          # its measured share of ORACLE
    assert "AUC bar" in gauge["status"]                # refused as an INSTRUMENT
    assert "uncertified" in gauge["status"]  # the instrument-refusal label
    assert gauge["needs_probe"] is True
    assert "probe" in gauge["description"].lower()
    # ★ It must be priced, and priced in seconds. A FREE policy (`loudest`)
    # already captures 43%, so an operator cannot weigh the probe without
    # seeing what it costs.
    assert "NOT FREE" in gauge["cost"]
    assert " s " in gauge["cost"] or "s " in gauge["cost"]
    assert "docs/FINDINGS.md §8" in blob["auto_policy_caveat"]


def test_info_prices_the_probe_before_the_operator_commits() -> None:
    """The cost estimate is served, is in seconds, and grows with k."""
    blob = info(default_k=8)
    probe = blob["probe"]
    assert probe["available"] is True
    assert probe["policy_key"] == "gauge"
    assert probe["default_base"] in probe["base_modes"]
    est = probe["cost_estimate"]
    assert est["k"] == 8
    assert est["total_s"] > 0
    assert est["total_s"] == pytest.approx(est["generate_s"] + est["harvest_s"])
    assert "uncertified" in probe["label"]
    # more candidates must cost more, or the estimate is decorative
    assert info(default_k=16)["probe"]["cost_estimate"]["total_s"] > est["total_s"]


def test_defaults_stay_free_no_policy_and_no_probe_runs_unattended() -> None:
    """★ The standing rule: the probe is opt-in and the default turn is free.
    Pinned here so a later convenience change has to delete a test that says
    why, rather than quietly flipping a default."""
    blob = info()
    assert "DEFAULTS STAY FREE" in blob["auto_policy_caveat"]
    for key in ("distinct", "loudest"):
        p = next(x for x in blob["auto_policies"] if x["key"] == key)
        assert "free" in p["cost"].lower()


def test_loudest_carries_its_v1a_remeasurement() -> None:
    """Measured ON v1a, this policy captures 43% of an oracle's picking skill,
    up from 12% on the wide map. The status must carry the new number AND
    the caveat that the harvest ground truth came from OLD-MAP wear — an
    operator reading only "43%" would over-trust it."""
    loudest = next(p for p in info()["auto_policies"] if p["key"] == "loudest")
    assert "43%" in loudest["status"]        # the share measured on v1a
    assert "12%" in loudest["status"]        # the wide-map share it replaces
    assert "OLD-MAP" in loudest["status"]    # the caveat must survive edits
    assert "uncertified" in loudest["status"]


def test_distinct_no_longer_claims_to_be_the_best_wired_policy() -> None:
    """`loudest` on v1a outscores it, so the string must not say "best of the
    wired"."""
    distinct = next(p for p in info()["auto_policies"] if p["key"] == "distinct")
    assert "Best of the wired" not in distinct["status"]
    assert "best wired policy" not in distinct["status"].lower()
    assert "43%" in distinct["status"]       # points at what overtook it
    assert "71%" in distinct["status"]       # ...and at the probe above that


def test_info_advertises_the_dose_policies_and_the_default() -> None:
    blob = info(dose_policy_default="flat")
    keys = [p["key"] for p in blob["dose_policies"]]
    assert keys == list(ls.DOSE_POLICIES)
    assert blob["dose_policy_default"] == "flat"
    assert blob["dose_scale_clamp"] == [0.5, 2.0]
    predicted = next(p for p in blob["dose_policies"] if p["key"] == "predicted")
    assert "UNVALIDATED" in predicted["status"]
    assert "uncertified" in predicted["status"]


def test_info_surfaces_both_rulers_and_says_which_is_in_force() -> None:
    """★ v1a ships the WIDE map's ruler as `norm_ref` and its own beside it.
    An operator who cannot tell them apart cannot read a dose band."""
    blob = info(
        norm_ref=[1.022, 2.1676, 3.1607, 4.3787],
        norm_ref_which="norm_ref = WIDE MAP's ruler (decision 1)",
        norm_ref_alternatives={"norm_ref_v1a_own": [0.4609, 0.9801, 1.423,
                                                    1.989]},
    )
    ruler = blob["ruler"]
    assert ruler["norm_ref_in_force_mean"] == pytest.approx(2.6823, abs=1e-3)
    assert "WIDE MAP" in ruler["which"]
    own = ruler["alternatives"]["norm_ref_v1a_own"]
    assert own["in_force"] is False
    assert own["mean"] == pytest.approx(1.2133, abs=1e-3)
    assert "docs/FINDINGS.md §2" in ruler["note"]


def test_info_ruler_degrades_cleanly_for_a_map_with_one_ruler() -> None:
    blob = info(norm_ref=[0.491, 1.044, 1.511, 2.096])
    assert blob["ruler"]["alternatives"] == {}
    assert blob["ruler"]["which"] is None


# ══ 10. the map itself: v1a's shape, read with zero behaviour change ════════


def test_a_map_with_one_ruler_reads_no_alternatives(tmp_path: Path) -> None:
    m = ls.LoomMap(*write_map(tmp_path))
    assert m.norm_ref_alternatives == {}
    assert m.norm_ref_which is None
    assert m.norm_ref.tolist() == NORM_REF


def test_a_second_ruler_is_read_but_never_worn(tmp_path: Path) -> None:
    """The v1a shape. `norm_ref` stays the only thing `lever_of` applies."""
    map_path, disc = write_map(tmp_path)
    with np.load(map_path, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    payload["norm_ref_v1a_own"] = np.asarray([0.46, 0.98, 1.42, 1.99])
    payload["norm_ref_which"] = np.asarray("norm_ref = WIDE MAP's ruler")
    np.savez(map_path, **payload)
    m = ls.LoomMap(map_path, disc)
    assert m.norm_ref.tolist() == NORM_REF          # unchanged
    assert m.norm_ref_alternatives == {"norm_ref_v1a_own": [0.46, 0.98, 1.42,
                                                            1.99]}
    assert "WIDE MAP" in (m.norm_ref_which or "")
    rng = np.random.default_rng(3)
    lever, _, _ = m.lever_of(rng.standard_normal(m.v3_mu.size),
                             rng.standard_normal(m.bins_mu.size))
    got = [float(np.linalg.norm(r)) for r in lever]
    assert got == pytest.approx(NORM_REF, rel=1e-9)  # the ruler in force
