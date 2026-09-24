"""Dose policies: how hard a worn lever is pushed, per site (flat | predicted).

`pleroma.serve` applies these to every wear; they live here, torch-free, so
the dose arithmetic is testable without a model.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pleroma.dose.band import _floats, _obj
from pleroma.errors import CodedError, ErrorCode


# ── predicted-magnitude dosing ─────────────────────────────────────────────────
#
# ★ WHAT THE RULER THROWS AWAY. `LoomMap.lever_of` ends with
# `out[s] = lever[s] * (norm_ref[s] / n[s])` — every candidate of a fan leaves
# the map at a DIFFERENT predicted magnitude and is injected at an IDENTICAL
# one. That flattening is load-bearing and correct for comparing across maps
# (`docs/FINDINGS.md §2`: alpha must mean one absolute per-site norm),
# and it is what makes `alpha` portable. But it has a measured cost: v1a
# predicts WITHIN-FAN displacement magnitude at r = .757 (v0: .538), and the
# constant ruler deletes that prediction before it can reach the model. v1a is
# the arm that pays.
#
# ★ THE FIX, AND WHY IT KEEPS ALPHA'S MEANING. `dose_policy: "predicted"`
# divides by the FAN's mean raw norm instead of the candidate's own:
#
#       out[s] = lever[s] * norm_ref[s] / mean_over_fan(n[s])
#
# so the per-site injected norm becomes `norm_ref[s] * n[s]/fan_mean[s]`. A
# candidate at the fan average is injected at EXACTLY today's magnitude; a
# louder-predicted one gets proportionally more. The arithmetic mean of the
# per-site scales over the fan is exactly 1, so the fan's AVERAGE dose is
# unchanged and the operator's alpha keeps its calibrated meaning. What
# changes is the spread within one draw, which is precisely the signal the
# ruler was destroying.
#
# ★ WHY IT IS OPT-IN AND DEFAULT OFF. Nothing behavioural has been measured
# about it. The r=.757 is a FIT-SIDE correlation against banked displacement,
# not a judged reply. `flat` is the default and injects exactly what the
# ruler alone gives (pinned by tests/unit/serve/loom/test_loom_dose_policy.py).
#
# ★ WHY THE CLAMP IS LOUD. A scale of 2.0 at alpha 0.45 injects the absolute
# magnitude of alpha 0.90 — past the derived overdriven ceiling (~0.72 on this
# ruler). A dose gauge that is wrong is worse than no gauge because it is
# trusted (`pleroma.dose.band`), so every clamp is recorded per site, returned in the
# response, and persisted into the worn snapshot. A silent clamp would make
# the receipt a lie.
#
# Pure, torch-free, model-free — the house pattern (resolve_draw_wear,
# pick_auto, rehydrate_worn_vectors): the thing that decides how hard the
# model gets pushed must be testable on a laptop.

DOSE_POLICIES = ("flat", "predicted")
DEFAULT_DOSE_POLICY = "flat"
DOSE_SCALE_MIN = 0.5
DOSE_SCALE_MAX = 2.0

DOSE_POLICY_INFO: tuple[dict[str, Any], ...] = (
    {
        "key": "flat", "default": True, "needs_fan": False,
        "description": "every candidate injected at the map's own norm_ref — "
                       "identical per-site magnitude for all of them.",
        "status": "the default: the ruler alone, applied unchanged. Every "
                  "dose band, ladder and published alpha in this project was "
                  "measured under it.",
    },
    {
        "key": "predicted", "default": False, "needs_fan": True,
        "description": "divide by the FAN's mean raw norm instead of the "
                       "candidate's own, so a candidate the map predicts will "
                       "move further is injected proportionally harder. The "
                       "fan's average dose is unchanged, so alpha keeps its "
                       f"meaning. Per-site scale clamped to "
                       f"[{DOSE_SCALE_MIN}, {DOSE_SCALE_MAX}]; every clamp is "
                       "reported.",
        "status": "★ UNVALIDATED — estimated, uncertified. The magnitude "
                  "signal it restores is real and fit-side measured (v1a "
                  "within-fan r=.757 vs v0 .538) but NO behavioural result "
                  "exists for this "
                  "policy. It also breaks the 1:1 between alpha and the dose "
                  "band: a candidate's EFFECTIVE alpha is alpha x its scale, "
                  "reported on every /wear as `alpha_effective_*`.",
        "unavailable_for": "/wear_code — a bank group, a random direction or "
                           "a bare code has no fan, so there is no mean to "
                           "divide by. Asking for it there is refused, never "
                           "silently flattened.",
    },
)


def resolve_dose_policy(raw: Any, server_default: str) -> str:
    """The effective `dose_policy` for one request: the body's own value if it
    sent one, else the server-level `--dose-policy-default`.

    Fails loudly on anything that is not a known policy string — the
    `request_detach_wear` doctrine: a truthy-but-wrong value quietly changing
    how hard the model is pushed is exactly the corruption this feature must
    not introduce.
    """
    if raw is None:
        value = str(server_default)
    elif isinstance(raw, str):
        value = raw.strip()
    else:
        raise CodedError(
            ErrorCode.DOSE_POLICY_INVALID,
            f"dose_policy must be a string, got {type(raw).__name__}")
    if value not in DOSE_POLICIES:
        raise CodedError(
            ErrorCode.DOSE_POLICY_INVALID,
            f"dose_policy must be one of {DOSE_POLICIES}, got {value!r}")
    return value


def fan_mean_raw_norms(
    candidates: Sequence[Mapping[str, Any]],
    key: str = "raw_norms",
) -> list[float] | None:
    """Per-site MEAN of the raw (pre-ruler) map-output norms over every
    HARVESTED candidate of one draw — the divisor `dose_policy: "predicted"`
    uses in place of each candidate's own norm.

    Returns None when no candidate in the draw harvested (nothing to average),
    which the wear path turns into a refusal rather than a guess. Raises on a
    fan whose norms are non-finite or non-positive: `lever_of` already refuses
    a zero row, so either of those means the fan was assembled wrongly, and
    dividing by it would scale a dose by garbage.

    ★ It must come from the SAME draw. A mean borrowed from another fan is not
    a fan mean — it is a different unit wearing alpha's name.

    `key` (the lever kind) selects which raw norms: the default
    `raw_norms` is the absolute lever's, unchanged; `contrast_raw_norms` gives
    the contrast lever's fan mean. Same draw AND same kind, or it is not a
    mean of anything.
    """
    rows = [c.get(key) for c in candidates
            if c.get(key) is not None]
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            f"fan raw_norms must be [n_candidates, n_sites], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("fan raw_norms contain non-finite values")
    if (arr <= 0.0).any():
        raise ValueError("fan raw_norms contain a non-positive norm")
    return [float(x) for x in arr.mean(axis=0)]


@dataclass(frozen=True)
class DoseScale:
    """The per-site multiplier one wear applies ON TOP of the ruler, and the
    whole receipt for why. `scale` is what was actually applied; `scale_raw`
    is what the fan mean asked for before clamping; `clamped` is non-empty
    exactly when the two differ."""

    policy: str
    scale: tuple[float, ...]
    scale_raw: tuple[float, ...]
    fan_mean_raw_norms: tuple[float, ...] | None = None
    candidate_raw_norms: tuple[float, ...] | None = None
    clamped: tuple[dict[str, Any], ...] = ()
    clamp_range: tuple[float, float] = (DOSE_SCALE_MIN, DOSE_SCALE_MAX)

    @property
    def scale_mean(self) -> float:
        return float(np.mean(self.scale)) if self.scale else 1.0

    @property
    def was_clamped(self) -> bool:
        return bool(self.clamped)

    def apply(self, lever: np.ndarray) -> np.ndarray:
        """The ruler-flattened lever, rescaled per site.

        ★ `flat` returns the SAME OBJECT, untouched — not `lever * 1.0`. The
        default path must be byte-identical to the ruler-only lever, and
        the cheapest way to guarantee that is to never perform the
        multiplication at all (pinned by
        `tests/unit/serve/loom/test_loom_dose_policy.py`).
        """
        if self.policy == "flat":
            return lever
        arr = np.asarray(lever)
        if arr.shape[0] != len(self.scale):
            raise ValueError(
                f"dose scale has {len(self.scale)} sites but the lever has "
                f"{arr.shape[0]} — refusing to broadcast a dose")
        scale = np.asarray(self.scale, dtype=arr.dtype).reshape(-1, 1)
        return (arr * scale).astype(arr.dtype, copy=False)

    def to_json(self) -> dict[str, Any]:
        """Banked in the /wear response AND in the persisted worn state, so a
        restore reproduces the exact injected magnitude and a receipt can be
        read months later."""
        return {
            "policy": self.policy,
            "scale": [round(float(x), 6) for x in self.scale],
            "scale_raw": [round(float(x), 6) for x in self.scale_raw],
            "scale_mean": round(self.scale_mean, 6),
            "fan_mean_raw_norms": (
                None if self.fan_mean_raw_norms is None
                else [round(float(x), 6) for x in self.fan_mean_raw_norms]),
            "candidate_raw_norms": (
                None if self.candidate_raw_norms is None
                else [round(float(x), 6) for x in self.candidate_raw_norms]),
            "clamped": [dict(c) for c in self.clamped],
            "clamp_range": [float(self.clamp_range[0]),
                            float(self.clamp_range[1])],
            "note": (
                "flat: the incumbent ruler — every candidate at norm_ref."
                if self.policy == "flat" else
                "predicted: per-site norm_ref x (this candidate's raw norm / "
                "the fan's mean raw norm). Fan average dose unchanged; alpha "
                "keeps its meaning. UNVALIDATED — estimated, uncertified."
            ),
        }

    @classmethod
    def from_json(cls, value: Any) -> DoseScale:
        """Rebuild from a persisted worn snapshot. Tolerant of an absent or
        malformed block — an older snapshot has no dose at all and must
        restore as `flat`, which is what it was."""
        if not isinstance(value, Mapping):
            return flat_dose_scale(0)
        policy = str(value.get("policy") or DEFAULT_DOSE_POLICY)
        if policy not in DOSE_POLICIES:
            raise ValueError(f"persisted dose.policy {policy!r} is unknown")
        scale = _floats(value.get("scale") or [], "worn.dose.scale")
        raw = _floats(value.get("scale_raw") or scale, "worn.dose.scale_raw")
        fan = value.get("fan_mean_raw_norms")
        cand = value.get("candidate_raw_norms")
        clamp = value.get("clamp_range") or [DOSE_SCALE_MIN, DOSE_SCALE_MAX]
        clamped = value.get("clamped") or []
        if not isinstance(clamped, list):
            raise ValueError("worn.dose.clamped must be a list")
        return cls(
            policy=policy,
            scale=tuple(scale), scale_raw=tuple(raw),
            fan_mean_raw_norms=(
                None if fan is None
                else tuple(_floats(fan, "worn.dose.fan_mean_raw_norms"))),
            candidate_raw_norms=(
                None if cand is None
                else tuple(_floats(cand, "worn.dose.candidate_raw_norms"))),
            clamped=tuple(_obj(c, "worn.dose.clamped[]") for c in clamped),
            clamp_range=(float(clamp[0]), float(clamp[1])),
        )


def flat_dose_scale(n_sites: int) -> DoseScale:
    """The identity dose: every site scaled by 1.0, the `flat` policy."""
    ones = tuple(1.0 for _ in range(int(n_sites)))
    return DoseScale(policy="flat", scale=ones, scale_raw=ones)


def dose_scale_for(
    *,
    policy: str,
    raw_norms: Sequence[float] | None,
    fan_mean: Sequence[float] | None,
    n_sites: int,
    clamp: tuple[float, float] = (DOSE_SCALE_MIN, DOSE_SCALE_MAX),
) -> DoseScale:
    """The whole `dose_policy` decision for ONE candidate, pure.

    `flat` ignores everything else and returns the identity. `predicted`
    REFUSES — never silently degrades — when this candidate has no raw norms
    or the draw produced no fan mean: inventing a divisor is the one failure
    mode that would make alpha mean different things on different turns, and
    a wrong dose gauge is worse than no gauge (`pleroma.dose.band`).
    """
    if policy not in DOSE_POLICIES:
        raise CodedError(ErrorCode.DOSE_POLICY_INVALID,
                         f"dose_policy must be one of {DOSE_POLICIES}, "
                         f"got {policy!r}")
    if policy == "flat":
        return flat_dose_scale(n_sites)
    if raw_norms is None:
        raise CodedError(
            ErrorCode.DOSE_POLICY_UNAVAILABLE,
            "dose_policy='predicted' needs this candidate's raw map-output "
            "norms and it has none (it did not harvest) — refusing to invent "
            "a magnitude. Use dose_policy='flat'.")
    if fan_mean is None:
        raise CodedError(
            ErrorCode.DOSE_POLICY_UNAVAILABLE,
            "dose_policy='predicted' needs the fan mean raw norms from the "
            "SAME draw and this wear has none — a wear outside a fan context "
            "(/wear_code, or a restored session whose draw is gone) cannot be "
            "predicted-dosed. Use dose_policy='flat', or /loom again first.")
    cand = np.asarray(raw_norms, dtype=np.float64)
    mean = np.asarray(fan_mean, dtype=np.float64)
    if cand.shape != mean.shape or cand.size != int(n_sites):
        raise ValueError(
            f"dose shapes disagree: candidate norms {cand.shape}, fan mean "
            f"{mean.shape}, map sites {int(n_sites)}")
    if not (np.isfinite(cand).all() and np.isfinite(mean).all()):
        raise ValueError("dose inputs contain non-finite values")
    if (mean <= 0.0).any() or (cand <= 0.0).any():
        raise ValueError("dose inputs contain a non-positive norm")
    lo, hi = float(clamp[0]), float(clamp[1])
    raw_scale = cand / mean
    clipped = np.clip(raw_scale, lo, hi)
    clamps = tuple(
        {"site_index": int(i), "raw": round(float(raw_scale[i]), 6),
         "applied": round(float(clipped[i]), 6),
         "bound": "min" if raw_scale[i] < lo else "max"}
        for i in range(raw_scale.size)
        if not np.isclose(raw_scale[i], clipped[i], rtol=0.0, atol=0.0)
    )
    return DoseScale(
        policy="predicted",
        scale=tuple(float(x) for x in clipped),
        scale_raw=tuple(float(x) for x in raw_scale),
        fan_mean_raw_norms=tuple(float(x) for x in mean),
        candidate_raw_norms=tuple(float(x) for x in cand),
        clamped=clamps,
        clamp_range=(lo, hi),
    )


def effective_alpha_json(alpha: float, dose: DoseScale) -> dict[str, Any]:
    """What alpha actually MEANS for this wear once the dose scale is in.

    Under `flat` these are all just alpha, and the dose band's zones apply
    unchanged. Under `predicted` the per-site injected magnitude is
    `alpha * scale[s] * norm_ref[s]`, so the zone an operator should read off
    the band is the one containing `alpha * scale[s]` — NOT alpha. Reporting
    only alpha there would hand the operator a gauge that is wrong by up to
    the clamp, which is the failure `pleroma.dose.band` exists to prevent.
    """
    a = float(alpha)
    scales = dose.scale or (1.0,)
    per_site = [round(a * float(s), 6) for s in scales]
    return {
        "alpha": a,
        "alpha_effective_mean": round(a * dose.scale_mean, 6),
        "alpha_effective_per_site": per_site,
        "alpha_effective_range": [min(per_site), max(per_site)],
        "note": (
            "dose_policy=flat: effective alpha IS alpha; read the dose band "
            "at alpha." if dose.policy == "flat" else
            "dose_policy=predicted: read the dose band at "
            "alpha_effective_per_site, not at alpha — the band's zones are "
            "absolute-magnitude boundaries and this wear is off-nominal by "
            "its dose scale."
        ),
    }


