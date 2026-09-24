"""Lever kind: which lever a fan wear puts on (absolute | contrast).

`pleroma.serve.legacy` re-exports it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pleroma.dose.policy import DoseScale, dose_scale_for, fan_mean_raw_norms
from pleroma.errors import CodedError, ErrorCode


# ── lever kind: which lever a fan wear puts on ───────────────────────────────
#
# ★ THE FINDING. `lever_of` applies W to a candidate's OWN input row x_i. v1a
# was fit on the FAN CONTRAST x_i − mean_{j≠i} x_j (one-vs-rest,
# `pleroma.map.build.join.fan_center`), and for v1a mu_in = mu_y = 0 exactly, so the served
# lever is W·x_i = W·(x_i − x̄_rest) + W·x̄_rest: the contrast plus a fan-common
# offset. Measured on six live 70B fans: cos(served, contrast) median
# 0.55 [−0.14, 0.85]; the common offset W·x̄ is median 0.87 of the served
# lever's norm; served levers of different futures in one fan have mean
# pairwise cos 0.45–0.85 against ≈ −0.1 for contrast levers. So `absolute`
# mostly wears what the fan's members SHARE.
#
# ★ WHY A TOGGLE, AND WHY `absolute` IS THE DEFAULT. Every dose band, drive and
# conversation on record was steered with the absolute lever; nothing
# behavioural has been measured under `contrast` on a fan wear. With the
# server default in place and no body field sent, a /wear stores exactly the
# vectors, code and dose of an absolute wear (pinned by the lever-kind tests)
# plus the `lever_kind` label.
#
# The contrast lever is the SAME object /probe steers with and /wear_code
# {code_kind: "differential"} wears (LoomMap.contrast_lever_of's docstring;
# equality pinned in the tests), computed from the input rows rather than from
# the 4-place-rounded codes.

LEVER_KINDS = ("absolute", "contrast")
DEFAULT_LEVER_KIND = "absolute"

LEVER_KIND_INFO: tuple[dict[str, Any], ...] = (
    {
        "key": "absolute", "needs_fan": False,
        "description": "W applied to the candidate's OWN input row — the lever "
                       "every wear on record used. For v1a (mu = 0) it is the "
                       "fan contrast plus a fan-common offset W·x̄.",
        "status": "the incumbent; byte-identical to the pre-2026-09-22 "
                  "server. Every dose band and drive was steered with it.",
    },
    {
        "key": "contrast", "needs_fan": True,
        "description": "W applied to x_i − mean of the OTHER valid members of "
                       "the draw — the one-vs-rest object v1a was fit on, and "
                       "the lever /probe steers with. mu_y is not added back; "
                       "the ruler applies after the contrast, so alpha keeps "
                       "its per-site meaning.",
        "status": "★ UNVALIDATED behaviourally on a fan wear. Measured on six "
                  "live 70B fans: cos(absolute, contrast) median 0.55; the "
                  "fan-common offset is median 0.87 of the absolute lever's "
                  "norm; absolute levers of one fan share direction (mean "
                  "pairwise cos 0.45–0.85) where contrast levers do not "
                  "(≈ −0.1).",
        "unavailable_for": "a draw with fewer than 2 harvested candidates (no "
                           "rest to subtract), and /wear_code — use "
                           "code_kind='differential' there. Refused, never "
                           "silently worn absolute.",
    },
)


def resolve_lever_kind(raw: Any, server_default: str) -> str:
    """The effective `lever_kind` for one request: the body's own value if it
    sent one, else the server-level `--wear-lever-default`.

    Same doctrine as `resolve_dose_policy`: anything that is not a known kind
    string fails loudly. A typo quietly wearing the other lever is exactly the
    corruption a toggle must not introduce.
    """
    if raw is None:
        value = str(server_default)
    elif isinstance(raw, str):
        value = raw.strip()
    else:
        raise CodedError(
            ErrorCode.LEVER_KIND_INVALID,
            f"lever_kind must be a string, got {type(raw).__name__}")
    if value not in LEVER_KINDS:
        raise CodedError(
            ErrorCode.LEVER_KIND_INVALID,
            f"lever_kind must be one of {LEVER_KINDS}, got {value!r}")
    return value


def worn_lever_kind(worn: Mapping[str, Any]) -> str:
    """The kind a stored wear was made with. Absent = a wear that predates the
    field, which can only have been `absolute` (the only kind a wear without
    the field could have used). A present-but-unknown value refuses: guessing would rehydrate
    a different lever and call it the same wear."""
    raw = worn.get("lever_kind")
    if raw is None:
        return DEFAULT_LEVER_KIND
    if not isinstance(raw, str) or raw not in LEVER_KINDS:
        raise ValueError(f"worn.lever_kind {raw!r} is not one of {LEVER_KINDS}")
    return raw


def attach_fan_contrast(
    candidates: list[dict[str, Any]],
    x_rows: Sequence[np.ndarray | None],
    contrast_of: Callable[[Sequence[np.ndarray], int],
                          tuple[np.ndarray, list[float], list[float]]],
) -> tuple[list[float] | None, str | None]:
    """Give every candidate of ONE draw its fan-contrast lever, in place.

    The absolute fields (`lever`, `raw_norms`, `code`) are NOT touched. Each
    candidate gains `contrast_lever`, `contrast_raw_norms`, `contrast_code`,
    `contrast_note` (None when available) and `contrast_peers` — the draw
    indices of the valid members the contrast was taken over, which is what a
    restore needs to recompute exactly this lever.

    Valid = the candidate harvested (`lever` is not None) and has an input row.
    With fewer than 2 valid members there is no rest to subtract: contrast is
    unavailable for the whole fan, stated on every candidate.

    Returns (per-site fan mean of the CONTRAST raw norms — the divisor
    `predicted` dosing uses for a contrast wear — or None, a fan-level note or
    None). A failure is recorded, never raised: the absolute path outranks it.
    """
    if len(x_rows) != len(candidates):
        raise ValueError(
            f"{len(x_rows)} input rows for {len(candidates)} candidates")
    valid = [i for i, c in enumerate(candidates)
             if c.get("lever") is not None and x_rows[i] is not None]
    peers = [int(candidates[i]["index"]) for i in valid]
    fan_note: str | None = None
    if len(valid) < 2:
        fan_note = (f"lever_kind='contrast' needs at least 2 harvested "
                    f"candidates in the draw and this one has {len(valid)} — "
                    "there is no rest of the fan to subtract")
    xs = [np.asarray(x_rows[i]) for i in valid]
    for i, c in enumerate(candidates):
        c["contrast_lever"] = None
        c["contrast_raw_norms"] = None
        c["contrast_code"] = None
        c["contrast_peers"] = None
        if fan_note is not None:
            c["contrast_note"] = fan_note
            continue
        if i not in valid:
            c["contrast_note"] = ("this candidate did not harvest ("
                                  f"{c.get('note') or 'no input row'})")
            continue
        try:
            lever, raw, code = contrast_of(xs, valid.index(i))
        except Exception as exc:  # noqa: BLE001 — listed per candidate, not fatal
            c["contrast_note"] = f"{type(exc).__name__}: {exc}"
            continue
        c["contrast_lever"] = lever
        c["contrast_raw_norms"] = raw
        c["contrast_code"] = code
        c["contrast_peers"] = list(peers)
        c["contrast_note"] = None
    fan_mean: list[float] | None = None
    if fan_note is None:
        try:
            fan_mean = fan_mean_raw_norms(candidates, key="contrast_raw_norms")
        except ValueError as exc:
            fan_note = f"no contrast fan mean ({exc})"
    return fan_mean, fan_note


def fan_wear_fields(
    candidate: Mapping[str, Any],
    *,
    lever_kind: str,
    dose_policy: str,
    fan_mean: Sequence[float] | None,
    fan_mean_contrast: Sequence[float] | None,
    n_sites: int,
) -> tuple[dict[str, Any], DoseScale]:
    """The lever-dependent half of a fan wear, pure: which vectors, which code,
    which dose — for ONE candidate under ONE `lever_kind`.

    Used by BOTH /wear and /loom's auto-wear, so the two cannot drift.
    `predicted` dosing divides by the raw norms and fan mean OF THE SAME KIND:
    a contrast lever's own loudness over the fan's mean contrast loudness. A
    cross-kind ratio would be a number in no unit.

    ★ `absolute` returns `candidate["lever"]` through `DoseScale.apply`, which
    under `flat` is the SAME OBJECT — the absolute lever's bytes, untouched.
    """
    if lever_kind not in LEVER_KINDS:
        raise CodedError(
            ErrorCode.LEVER_KIND_INVALID,
            f"lever_kind must be one of {LEVER_KINDS}, got {lever_kind!r}")
    index = candidate.get("index")
    if lever_kind == "absolute":
        lever = candidate.get("lever")
        if lever is None:
            raise ValueError(f"candidate {index} unavailable "
                             f"({candidate.get('note')})")
        raw_norms = candidate.get("raw_norms")
        mean = fan_mean
        code = candidate.get("code")
    else:
        lever = candidate.get("contrast_lever")
        if lever is None:
            raise CodedError(
                ErrorCode.LEVER_KIND_UNAVAILABLE,
                f"lever_kind='contrast' is unavailable for candidate {index}: "
                f"{candidate.get('contrast_note') or 'this draw computed no contrast'}"
                " — refusing rather than wearing the absolute lever instead. "
                "Use lever_kind='absolute', or /loom again.")
        raw_norms = candidate.get("contrast_raw_norms")
        mean = fan_mean_contrast
        code = candidate.get("contrast_code")
    dose = dose_scale_for(policy=dose_policy, raw_norms=raw_norms,
                          fan_mean=mean, n_sites=n_sites)
    fields: dict[str, Any] = {"vectors": dose.apply(lever), "code": code,
                              "dose": dose.to_json(), "lever_kind": lever_kind}
    if lever_kind == "contrast":
        fields["contrast_peers"] = [int(x) for x in candidate["contrast_peers"]]
    return fields, dose


