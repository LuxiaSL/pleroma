"""Stage `probe`: can the gauge tell this fan's candidates apart?

| gate        | catalog | question |
|-------------|---------|----------|
| resolution  | G-01, G-02, F-01, X-05 | is the probe's ordering resolved, on a clean regime? |

Reads a probe receipt (what the loom's `/probe` returns: `resolution` from
`pleroma.probe.orchestrator.resolution_report`, plus `clean_regime`) — or the
bare `resolution` object. The resolution verdict is the headline, not the
ranking: `resolved: true` certifies the TOP pick only (G-01).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pleroma.validate._io import InputError, read_json, require_mapping
from pleroma.validate.result import GateResult, guarded, inconclusive, passed

STAGE = "probe"


def extract_resolution(blob: dict[str, Any]) -> tuple[dict[str, Any], bool | None]:
    """(resolution object, clean_regime or None if not recorded)."""
    if "resolution" in blob:
        res = require_mapping(blob["resolution"], "probe receipt 'resolution'")
        clean = blob.get("clean_regime")
    elif "resolved" in blob:
        res, clean = blob, blob.get("clean_regime")
    else:
        raise InputError("probe receipt has no 'resolution' object (and is not one): "
                         "was it produced by /probe with reps >= 2?")
    if clean is not None and not isinstance(clean, bool):
        raise InputError(f"probe receipt 'clean_regime' must be a boolean, got {clean!r}")
    return res, clean


@guarded(STAGE, "resolution")
def gate_resolution(blob: dict[str, Any]) -> GateResult:
    try:
        res, clean = extract_resolution(blob)
    except InputError as exc:
        return inconclusive(STAGE, "resolution", str(exc))
    resolved = res.get("resolved")
    ev: dict[str, Any] = {k: res.get(k) for k in (
        "resolved", "reps", "n_candidates_used", "n_candidates_excluded_for_reps",
        "within_candidate_sd_nr", "between_candidate_sd_cell", "estimated_signal_sd",
        "noise_share_of_cell_variance_at_this_reps", "reps_needed_to_resolve")}
    ev["clean_regime"] = clean
    if clean is False:
        return inconclusive(STAGE, "resolution",
                            "probe ran on a fan drawn under a wear its replies did not carry "
                            "(clean_regime false): cells mix 'code added' with 'wear removed' "
                            "(G-02) — unwear and redraw", **ev)
    if resolved is None:
        return inconclusive(STAGE, "resolution",
                            "probe cannot answer: it needs reps >= 2 on at least 2 candidates "
                            "to separate measurement noise from real difference", **ev)
    if not isinstance(resolved, bool):
        return inconclusive(STAGE, "resolution",
                            f"'resolved' must be true/false/null, got {resolved!r}", **ev)
    if not resolved:
        between = res.get("between_candidate_sd_cell")
        within = res.get("within_candidate_sd_nr")
        needed = res.get("reps_needed_to_resolve")
        if between is not None and within is not None and between <= 1e-12 and within <= 1e-12:
            reason = ("fan did not fork: every candidate scored identically, so no rep count "
                      "helps — reframe the prompt (F-01)")
        elif needed is None:
            reason = ("between-candidate spread is entirely measurement noise: these "
                      "candidates are not distinguishable by the gauge at any affordable rep "
                      "count")
        else:
            reason = (f"ordering NOT resolved at reps={res.get('reps')}: this fan needs about "
                      f"{needed} reps — treat the ranking as noise")
        return inconclusive(STAGE, "resolution", reason, **ev)
    note = " (weak evidence at reps=2: ~10% false-resolve rate on pure noise)" \
        if res.get("reps") == 2 else ""
    return passed(STAGE, "resolution",
                  f"top pick resolved at reps={res.get('reps')}{note}; ranks below #1 are NOT "
                  "certified (G-01)", **ev)


def run_probe(receipt: str | Path | None) -> list[GateResult]:
    if receipt is None:
        return []
    try:
        blob = require_mapping(read_json(receipt, "probe receipt"), "probe receipt")
    except InputError as exc:
        return [inconclusive(STAGE, "resolution", str(exc))]
    return [gate_resolution(blob)]
