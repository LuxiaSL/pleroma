"""Stage `eval`: is a judged effect readable?

| gate              | catalog | question |
|-------------------|---------|----------|
| length-null       | J-01, J-02, T-01 | does the judged Δnr beat the LENGTH-ONLY ranker, paired by conversation? |
| positive-control  | J-05, J-06, T-01 | can this judge see a KNOWN effect, against its own measured floor? |

Inputs are already-judged artifacts (no API calls):

length-null JSON::

    {"arm": "steered", "base": "base",
     "judged":      {conv: {"base": [nr, ...], "steered": [nr, ...]}},
     "length_only": {conv: {"base": [lo, ...], "steered": [lo, ...]}}}

where `lo` is `pleroma.stats.length_only_rank` on CHARACTER counts (J-02) for
the same replies the judge ranked. Both Δnr's are computed by
`pleroma.stats.length_only_dnr` (mean base − mean arm per conversation), so the
two sides of the comparison are built the same way (T-02).

positive-control JSON::

    {"positive": [score, ...], "floor": [score, ...]}

`positive` = the judge's scores on a control set whose effect is known to be
real; `floor` = the same judge on the matched null/catch arm (its OWN floor —
never a fixed 0.5, J-05). Higher score = "detected".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pleroma.stats import boot_ci, length_only_dnr
from pleroma.stats.bootstrap import DEFAULT_SEED
from pleroma.validate._io import InputError, read_json
from pleroma.validate.result import GateResult, failed, guarded, inconclusive, passed

STAGE = "eval"
#: Below this many paired conversations a judge−length contrast is unpowered (T-01).
MIN_CONVERSATIONS: int = 5
#: Below this many scored items per arm a positive control reads nothing.
MIN_CONTROL_ITEMS: int = 8
ALPHA: float = 0.05
N_PERMS: int = 20000


class LengthNullInput(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    arm: str = Field(min_length=1)
    base: str = Field(default="base", min_length=1)
    judged: dict[str, dict[str, list[float]]]
    length_only: dict[str, dict[str, list[float]]]


class PositiveControlInput(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    positive: list[float]
    floor: list[float] | None = None


def _load(path: str | Path, model: type[BaseModel], what: str) -> Any:
    blob = read_json(path, what)
    try:
        return model.model_validate(blob)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise InputError(f"{what} malformed: {loc}: {first['msg']}") from exc


def permutation_p_greater(a: Sequence[float], b: Sequence[float], n_perms: int = N_PERMS,
                          seed: int = DEFAULT_SEED) -> float:
    """One-sided p that mean(a) > mean(b) under exchangeable labels (two
    independent samples; the +1 correction keeps p > 0)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    pooled = np.concatenate([x, y])
    obs = x.mean() - y.mean()
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perms, pooled.size)), axis=1)
    perm = pooled[idx]
    null = perm[:, :x.size].mean(axis=1) - perm[:, x.size:].mean(axis=1)
    return float(((null >= obs - 1e-12).sum() + 1) / (n_perms + 1))


@guarded(STAGE, "length-null")
def gate_length_null(data: LengthNullInput) -> GateResult:
    """J-01: the judge reads LENGTH, and steering moves length. An effect that
    does not beat the length-only ranker has not shown manner steering."""
    arm, base = data.arm, data.base
    try:
        judged = length_only_dnr(data.judged, arm, base)
        length = length_only_dnr(data.length_only, arm, base)
    except ValueError as exc:
        return inconclusive(STAGE, "length-null", f"cannot pair arms: {exc}", arm=arm, base=base)
    jp: Mapping[str, float] = judged["per_conversation"]  # type: ignore[assignment]
    lp: Mapping[str, float] = length["per_conversation"]  # type: ignore[assignment]
    convs = sorted(set(jp) & set(lp))
    diffs = [jp[c] - lp[c] for c in convs]
    ev: dict[str, Any] = dict(
        arm=arm, base=base, n_conversations=len(convs),
        judged_dnr=float(np.mean([jp[c] for c in convs])) if convs else None,
        length_only_dnr=float(np.mean([lp[c] for c in convs])) if convs else None,
        length_unit="characters", min_conversations=MIN_CONVERSATIONS)
    if len(convs) < MIN_CONVERSATIONS:
        return inconclusive(STAGE, "length-null",
                            f"only {len(convs)} conversation(s) carry both a judged and a "
                            f"length-only Δnr (< {MIN_CONVERSATIONS}): judge − length is "
                            "unpowered (T-01)", **ev)
    mean_diff = float(np.mean(diffs))
    lo, hi = boot_ci(diffs)
    ev.update(judge_minus_length=mean_diff, ci95=[lo, hi],
              n_conv_judge_beats_length=sum(1 for d in diffs if d > 0))
    if ev["judged_dnr"] <= 0:
        return inconclusive(STAGE, "length-null",
                            f"judged Δnr {ev['judged_dnr']:+.4f} ≤ 0: there is no effect for "
                            "the length null to explain", **ev)
    if mean_diff <= 0:
        return failed(STAGE, "length-null",
                      f"the length-only ranker explains the effect: length Δnr "
                      f"{ev['length_only_dnr']:+.4f} ≥ judged Δnr {ev['judged_dnr']:+.4f} "
                      "— the judge is reading reply length, not manner (J-01)", **ev)
    if not lo > 0:
        return inconclusive(STAGE, "length-null",
                            f"judge beats length by {mean_diff:+.4f} but the 95% CI "
                            f"[{lo:+.4f}, {hi:+.4f}] includes 0: length is not ruled out", **ev)
    return passed(STAGE, "length-null",
                  f"judged Δnr {ev['judged_dnr']:+.4f} beats the length-only ranker "
                  f"({ev['length_only_dnr']:+.4f}) by {mean_diff:+.4f}, 95% CI "
                  f"[{lo:+.4f}, {hi:+.4f}] over {len(convs)} conversations", **ev)


@guarded(STAGE, "positive-control")
def gate_positive_control(data: PositiveControlInput) -> GateResult:
    """Judge positive control FIRST: if the judge cannot separate a known
    effect from its own floor, a null on the real test means "this judge cannot
    see effects of this size", never "no effect". This gate never FAILs."""
    pos = [float(x) for x in data.positive if np.isfinite(x)]
    ev: dict[str, Any] = dict(n_positive=len(pos), min_items=MIN_CONTROL_ITEMS)
    if data.floor is None:
        return inconclusive(STAGE, "positive-control",
                            "no floor arm supplied: a judge's separation must be read against "
                            "its own measured catch/null floor, never a fixed 0.5 (J-05)", **ev)
    floor = [float(x) for x in data.floor if np.isfinite(x)]
    ev["n_floor"] = len(floor)
    if len(pos) < MIN_CONTROL_ITEMS or len(floor) < MIN_CONTROL_ITEMS:
        return inconclusive(STAGE, "positive-control",
                            f"too few scored items (positive {len(pos)}, floor {len(floor)}; "
                            f"need >= {MIN_CONTROL_ITEMS} each) to read the control", **ev)
    pm, fm = float(np.mean(pos)), float(np.mean(floor))
    p = permutation_p_greater(pos, floor)
    ev.update(positive_mean=pm, floor_mean=fm, separation=pm - fm, one_sided_p=p)
    if pm > fm and p < ALPHA:
        return passed(STAGE, "positive-control",
                      f"judge separates the known-positive control from its own floor "
                      f"({pm:.3f} vs {fm:.3f}, one-sided p={p:.3g})", **ev)
    return inconclusive(STAGE, "positive-control",
                        f"judge cannot see effects of this size: the known-positive control "
                        f"reads {pm:.3f} vs its own floor {fm:.3f} (p={p:.3g}) — a null from "
                        "this judge means nothing, not 'no effect'", **ev)


def run_eval(length_null: str | Path | None, positive_control: str | Path | None
             ) -> list[GateResult]:
    out: list[GateResult] = []
    if length_null is not None:
        try:
            out.append(gate_length_null(_load(length_null, LengthNullInput,
                                              "length-null input")))
        except InputError as exc:
            out.append(inconclusive(STAGE, "length-null", str(exc)))
    if positive_control is not None:
        try:
            out.append(gate_positive_control(_load(positive_control, PositiveControlInput,
                                                   "positive-control input")))
        except InputError as exc:
            out.append(inconclusive(STAGE, "positive-control", str(exc)))
    return out
