"""Contract: the loom's dose policy (``pleroma.dose.policy``) — `flat` /
`predicted`, the [0.5, 2] clamp, effective alpha, and the zone read at the
LOUDEST site.

predicted: per-site scale = candidate raw norm / fan-mean raw norm, clamped to
[0.5, 2]; effective alpha per site = knob x scale; the zone an operator must
read is at max_s(knob x scale[s]), not at the mean, because one site at the
clamp can sit in a louder zone than the average suggests. Clamps are reported,
never silent; predicted refuses without raw norms rather than inventing a
divisor. It exists because the ruler rescales every lever to the same
per-site norm and so discards the magnitude the map predicted; predicted puts
it back. It compounds with the knob: knob .5 x a predicted scale of 1.08-1.32
already reaches the overdriven zone.

The CODE default is `flat`; a deployment that wants `predicted` passes
--dose-policy-default predicted. Pinned as-is below.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from tests.contract.dose_steer import targets as T


# ── the constants and the policy switch ──────────────────────────────────────


def test_policy_constants() -> None:
    """Policies are exactly (flat, predicted); CODE default flat (a deployment
    may override it with --dose-policy-default); clamp [0.5, 2.0]. A porter
    changing any of these changes every served dose."""
    assert T.DOSE_POLICIES == ("flat", "predicted")
    assert T.DEFAULT_DOSE_POLICY == "flat"
    assert (T.DOSE_SCALE_MIN, T.DOSE_SCALE_MAX) == (0.5, 2.0)


@pytest.mark.parametrize(("raw", "default", "want"), [
    (None, "flat", "flat"), (None, "predicted", "predicted"),
    ("predicted", "flat", "predicted"), ("  flat ", "predicted", "flat"),
])
def test_resolve_dose_policy_body_beats_server_default(raw: Any, default: str, want: str) -> None:
    """resolve_dose_policy: the request body's value wins
    (whitespace-stripped), else the server's --dose-policy-default."""
    assert T.resolve_dose_policy(raw, default) == want


@pytest.mark.parametrize("raw", ["Predicted", "loud", 1, True, ["flat"]])
def test_resolve_dose_policy_fails_loudly(raw: Any) -> None:
    """A truthy-but-wrong policy must raise, never silently pick a dose."""
    with pytest.raises(ValueError):
        T.resolve_dose_policy(raw, "flat")


# ── flat ─────────────────────────────────────────────────────────────────────


def test_flat_is_the_identity_and_returns_the_same_object() -> None:
    """flat: scale all ones, no clamps, and DoseScale.apply returns the SAME
    ndarray object (byte-identity with the unscaled lever; every measured
    band/ladder alpha is under flat). flat ignores norms entirely."""
    lever = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    d = T.dose_scale_for(policy="flat", raw_norms=None, fan_mean=None, n_sites=4)
    assert d.policy == "flat" and d.scale == (1.0,) * 4 and not d.was_clamped
    assert d.apply(lever) is lever
    assert d == T.flat_dose_scale(4)


# ── predicted ────────────────────────────────────────────────────────────────


def test_predicted_scale_is_candidate_over_fan_mean() -> None:
    """predicted: scale[s] = raw[s] / fan_mean[s]; in-range
    values are unclamped; apply multiplies each site row by its scale."""
    raw = [1.2, 0.9, 1.0, 1.5]
    mean = [1.0, 1.0, 1.0, 1.0]
    d = T.dose_scale_for(policy="predicted", raw_norms=raw, fan_mean=mean, n_sites=4)
    assert d.scale == pytest.approx(tuple(raw))
    assert d.scale_raw == d.scale and d.clamped == ()
    lever = np.ones((4, 3), dtype=np.float32)
    out = d.apply(lever)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], raw, rtol=1e-6)


def test_predicted_clamps_to_half_and_two_and_reports_every_clamp() -> None:
    """Scales outside [0.5, 2] are clipped and EVERY clamp is in the
    receipt (site_index, raw, applied, bound). A silent clamp makes the
    receipt a lie."""
    d = T.dose_scale_for(policy="predicted", raw_norms=[0.2, 1.0, 5.0, 2.0],
                         fan_mean=[1.0, 1.0, 1.0, 1.0], n_sites=4)
    assert d.scale == (0.5, 1.0, 2.0, 2.0)
    assert d.scale_raw == (0.2, 1.0, 5.0, 2.0)
    assert d.was_clamped
    assert [(c["site_index"], c["bound"], c["applied"]) for c in d.clamped] == [
        (0, "min", 0.5), (2, "max", 2.0)]  # exactly-at-bound (site 3) is not a clamp
    j = d.to_json()
    assert j["clamp_range"] == [0.5, 2.0] and len(j["clamped"]) == 2


def test_predicted_fan_average_scale_is_one() -> None:
    """The design invariant: with every candidate's scale
    = raw / fan mean, the arithmetic mean of scales over the fan is exactly 1
    per site (unclamped), so alpha keeps its calibrated meaning on average."""
    rng = np.random.default_rng(0)
    fan = [{"raw_norms": list(rng.uniform(0.8, 1.4, size=4))} for _ in range(6)]
    mean = T.fan_mean_raw_norms(fan)
    scales = np.array([T.dose_scale_for(policy="predicted", raw_norms=c["raw_norms"],
                                        fan_mean=mean, n_sites=4).scale for c in fan])
    np.testing.assert_allclose(scales.mean(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize(("raw", "mean", "match"), [
    (None, [1.0] * 4, "did not harvest"),
    ([1.0] * 4, None, "SAME draw"),
    ([1.0] * 3, [1.0] * 4, "shapes disagree"),
    ([1.0, 0.0, 1.0, 1.0], [1.0] * 4, "non-positive"),
    ([1.0, float("nan"), 1.0, 1.0], [1.0] * 4, "non-finite"),
])
def test_predicted_refuses_rather_than_invents(raw: Any, mean: Any, match: str) -> None:
    """predicted REFUSES without this candidate's
    raw norms or the same draw's fan mean, and on malformed norms — inventing
    a divisor would make alpha mean different things on different turns."""
    with pytest.raises(ValueError, match=match):
        T.dose_scale_for(policy="predicted", raw_norms=raw, fan_mean=mean, n_sites=4)


def test_fan_mean_raw_norms() -> None:
    """fan_mean_raw_norms: per-site mean over HARVESTED candidates only
    (None-norm rows skipped); None when nothing harvested; raises on a
    non-positive norm. `key` selects the contrast lever's norms."""
    fan = [{"raw_norms": [1.0, 2.0]}, {"raw_norms": None}, {"raw_norms": [3.0, 4.0]},
           {"raw_norms": [9.0, 9.0], "contrast_raw_norms": [0.5, 0.5]}]
    assert T.fan_mean_raw_norms(fan[:3]) == [2.0, 3.0]
    assert T.fan_mean_raw_norms([{"raw_norms": None}]) is None
    assert T.fan_mean_raw_norms(fan, key="contrast_raw_norms") == [0.5, 0.5]
    with pytest.raises(ValueError):
        T.fan_mean_raw_norms([{"raw_norms": [1.0, -1.0]}])


def test_dose_scale_json_round_trip_and_legacy_snapshot() -> None:
    """The dose receipt persists in the worn snapshot; from_json(to_json(d))
    reproduces the applied scale, and a snapshot with no dose block
    restores as flat."""
    d = T.dose_scale_for(policy="predicted", raw_norms=[0.2, 1.0, 3.0],
                         fan_mean=[1.0, 1.0, 1.0], n_sites=3)
    back = T.DoseScale.from_json(json.loads(json.dumps(d.to_json())))
    assert back.policy == "predicted" and back.scale == d.scale
    assert back.clamp_range == (0.5, 2.0) and len(back.clamped) == 2
    assert T.DoseScale.from_json(None).policy == "flat"


# ── effective alpha and the loudest-site zone ────────────────────────────────


def test_effective_alpha_is_knob_times_scale_per_site() -> None:
    """effective_alpha_json: per-site = alpha x scale[s];
    mean = alpha x scale_mean; range = [min, max] of per-site. The UI reads
    the zone at range max (the loudest site). flat: every field is alpha."""
    d = T.dose_scale_for(policy="predicted", raw_norms=[0.9, 1.0, 1.1, 1.7],
                         fan_mean=[1.0, 1.0, 1.0, 1.0], n_sites=4)
    e = T.effective_alpha_json(0.315, d)
    assert e["alpha_effective_per_site"] == [round(0.315 * s, 6) for s in (0.9, 1.0, 1.1, 1.7)]
    assert e["alpha_effective_mean"] == round(0.315 * 1.175, 6)
    assert e["alpha_effective_range"] == [round(0.315 * 0.9, 6), round(0.315 * 1.7, 6)]
    f = T.effective_alpha_json(0.315, T.flat_dose_scale(4))
    assert f["alpha_effective_per_site"] == [0.315] * 4
    assert f["alpha_effective_range"] == [0.315, 0.315] and f["alpha_effective_mean"] == 0.315


# ── the lever-dependent half of a wear ───────────────────────────────────────


def test_fan_wear_fields_absolute_flat_is_the_candidate_lever_itself() -> None:
    """fan_wear_fields (used by /wear AND /loom auto-wear): absolute+flat hands
    back candidate['lever'] as the SAME object (an absolute flat wear carries
    the lever unmodified); predicted divides by the fan mean OF THE SAME KIND;
    contrast without a contrast lever REFUSES rather than silently wearing the
    absolute lever, which is not the fan contrast the map was fit on."""
    lever = np.ones((2, 3), dtype=np.float32)
    cand = {"index": 0, "lever": lever, "raw_norms": [2.0, 1.0], "code": [0.1],
            "contrast_lever": None, "contrast_note": "none"}
    f, dose = T.fan_wear_fields(cand, lever_kind="absolute", dose_policy="flat",
                                fan_mean=None, fan_mean_contrast=None, n_sites=2)
    assert f["vectors"] is lever and dose.policy == "flat" and f["lever_kind"] == "absolute"
    f2, d2 = T.fan_wear_fields(cand, lever_kind="absolute", dose_policy="predicted",
                               fan_mean=[1.0, 1.0], fan_mean_contrast=None, n_sites=2)
    np.testing.assert_allclose(f2["vectors"][:, 0], [2.0, 1.0])
    with pytest.raises(ValueError, match="refusing"):
        T.fan_wear_fields(cand, lever_kind="contrast", dose_policy="flat",
                          fan_mean=None, fan_mean_contrast=None, n_sites=2)


def test_injected_norm_is_knob_times_scale_times_ruler() -> None:
    """The whole dose chain on the loom's HF path: a ruler-matched lever
    (row s at norm_ref[s], what LoomMap.lever_of emits), x DoseScale.apply,
    through the loom's own attach (normalize=False) at alpha, lands in the
    residual stream with per-site norm alpha x scale[s] x norm_ref[s]
    (alpha is an absolute per-site norm, in units of the ruler)."""
    import torch

    from tests.contract.dose_steer import _toy

    model = _toy.ToyDecoder()
    sites = [1, 2, 4]
    norm_ref = np.array([2.0675, 4.5887, 6.4719])  # 70B-like ruler, 3 sites
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((3, _toy.HIDDEN))
    lever = (raw / np.linalg.norm(raw, axis=1, keepdims=True) * norm_ref[:, None]).astype(np.float32)
    d = T.dose_scale_for(policy="predicted", raw_norms=[0.7, 1.0, 3.0],
                         fan_mean=[1.0] * 3, n_sites=3)
    vectors = d.apply(lever)
    h, cp = _toy.prefill_input(5)
    model(h, cp)
    base = model.inputs()
    attach = T.loom_attach(model, sites)
    handles = attach(vectors, 0.315)
    try:
        model(h, cp)
        worn = model.inputs()
    finally:
        for hd in handles:
            hd.remove()
    # the FIRST site's input delta is exactly its write (nothing upstream);
    # later sites' deltas also carry the propagated upstream write, so their
    # write is read off the registered spec (scale 3.0 clamps to 2.0).
    delta0 = (worn[1] - base[1])[0]
    want0 = 0.315 * 0.7 * norm_ref[0]
    np.testing.assert_allclose(torch.linalg.vector_norm(delta0, dim=-1).numpy(), want0, rtol=1e-5)
    for s_i, handle in enumerate(handles):
        assert float(handle.spec.alpha) == 0.315
        vec = handle.spec.vector.numpy()
        np.testing.assert_allclose(np.linalg.norm(0.315 * vec),
                                   0.315 * d.scale[s_i] * norm_ref[s_i], rtol=1e-5)
