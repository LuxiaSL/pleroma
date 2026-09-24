"""The GET /info payload: what this server is wired to, and what each of its
knobs costs or means, as one JSON object. Pure dict assembly — no model, no
I/O — so it is testable on a laptop and cannot throw on a live server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pleroma.probe import orchestrator as loom_probe
from pleroma.dose.band import DoseBand, dose_band_info_json
from pleroma.dose.policy import (
    DEFAULT_DOSE_POLICY,
    DOSE_POLICY_INFO,
    DOSE_SCALE_MAX,
    DOSE_SCALE_MIN,
)
from pleroma.levers.kind import DEFAULT_LEVER_KIND, LEVER_KIND_INFO
from pleroma.serve.policies import AUTO_POLICIES_UNAVAILABLE


def _probe_cost_or_note(default_k: int, harvest_width: int = 1) -> dict[str, Any]:
    """The probe's cost estimate, or an explanation of why there isn't one.

    Never raises. `/info` is the endpoint the UI loads first, so anything that
    can throw in here turns it into an error and takes the whole page down —
    see the call site.

    `harvest_width` is THIS server's armed pool width, not a constant: quoting
    a serial cost on a pooled server over-quotes by ~2x, and the UI shows this
    figure to an operator before they commit the time.
    """
    try:
        return loom_probe.estimate_cost_s(default_k, loom_probe.DEFAULT_REPS,
                                          harvest_width=harvest_width)
    except ValueError as exc:
        return {
            "unavailable": True,
            "why": f"no estimate for default_k={default_k}: {exc}",
            "note": "the probe needs a fan of at least 2 to build a "
                    "contrastive code; pass an explicit k on /loom.",
        }


def build_info_payload(
    *, map_path: str, map_meta: Mapping[str, Any], sites: Sequence[int],
    branches: Sequence[str], default_k: int, future_tokens: int,
    detach_wear_default: bool, n_sessions: int, harvest_worker: str | None,
    restored_sessions: int, persistence: Mapping[str, Any],
    auto_policies: Sequence[Mapping[str, Any]], loudness_ref: float,
    dose_band: DoseBand | None,
    # ── Everything below is keyword-only with a default, so a caller that
    # passes only the core fields still gets a complete, truthful payload.
    # See AUTO_POLICY_INFO / DOSE_POLICY_INFO for the copy itself.
    # `harvest_workers`: the pool, whose width tells an operator from /info
    # whether the fast path is actually armed rather than inferring it from
    # a latency they have to measure.
    harvest_workers: Sequence[str] | None = None,
    norm_ref: Sequence[float] | None = None,
    norm_ref_which: str | None = None,
    norm_ref_alternatives: Mapping[str, Sequence[float]] | None = None,
    # These two default to the module's own tables rather than to empty:
    # they are facts about this SERVER BUILD, not per-run configuration, and
    # a caller that forgets to pass them must still tell the truth about
    # what is wired and what is not.
    auto_policies_unavailable: Sequence[Mapping[str, Any]] = AUTO_POLICIES_UNAVAILABLE,
    dose_policies: Sequence[Mapping[str, Any]] = DOSE_POLICY_INFO,
    dose_policy_default: str = DEFAULT_DOSE_POLICY,
    dose_scale_clamp: Sequence[float] = (DOSE_SCALE_MIN, DOSE_SCALE_MAX),
    # ── --wear-lever-default. Same shape as the dose
    # policies: the table is this build's, the default is this run's.
    lever_kinds: Sequence[Mapping[str, Any]] = LEVER_KIND_INFO,
    lever_kind_default: str = DEFAULT_LEVER_KIND,
    # ── --prompt-mode. Defaults to "chat"/None so a caller that does not pass
    # it gets the truthful answer for a server in chat mode.
    prompt_mode: str = "chat",
    prompt_mode_detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The exact dict `GET /info` sends. Module level so it is testable on a
    laptop with pytest, no model or running server required. Never raises
    for a well-typed call (the probe cost estimate degrades to a note).
    `dose_band` degrades to ``pleroma.dose.band.NONE_INFO_JSON``-shaped output
    (tier="none") when nothing is loaded, so a client never sees a
    plausible-but-wrong band."""
    return {
        "map": map_path, "map_meta": dict(map_meta),
        "sites": list(sites), "branches": list(branches),
        "default_k": int(default_k),
        "future_tokens": int(future_tokens),
        # The unworn-draw protocol's server-wide
        # default a /loom request falls back to when its body
        # omits 'detach_wear'. Per-request always overrides this.
        "detach_wear_default": bool(detach_wear_default),
        "n_sessions": int(n_sessions),
        # ── WHICH PROMPT MODE IS IN FORCE. A fan drawn in
        # 'raw' is a different cell from the one every banked number was
        # measured in ('chat'), so this is not an implementation detail: it
        # belongs on the face of the instrument beside the ruler and the
        # dose band. `prompt_mode_detail` carries the modelc header in force
        # (and its stops and sampler) when that mode is on.
        "prompt_mode": str(prompt_mode),
        "prompt_mode_detail": (dict(prompt_mode_detail)
                               if prompt_mode_detail else None),
        "harvest_worker": harvest_worker or None,
        # The parallel-harvest pool. None = not armed, in which case every
        # harvest goes single worker -> subprocess.
        "harvest_workers": list(harvest_workers) if harvest_workers else None,
        "harvest_pool_width": len(harvest_workers) if harvest_workers else 0,
        # Durable sessions: how many were restored at startup, and the
        # store's state.
        "restored_sessions": restored_sessions,
        "persistence": dict(persistence),
        "auto_policies": [dict(p) for p in auto_policies],
        "loudness_ref": round(float(loudness_ref), 3),
        # See pleroma.dose.band's module docstring for the tier semantics.
        "dose_band": dose_band_info_json(dose_band),
        # ── WHICH RULER IS IN FORCE ───────────────────────────────────────
        # `norm_ref` is the one actually applied by
        # LoomMap.lever_of; `norm_ref_alternatives` is every other ruler the
        # map file ships. A map that carries two and wears one is a trap
        # unless both are visible: alpha is an ABSOLUTE per-site magnitude,
        # and reading a dose band against the wrong ruler turns one map's
        # audible dose into another's "destructive" one (docs/FINDINGS.md §2).
        "ruler": {
            "norm_ref_in_force": (
                None if norm_ref is None
                else [round(float(x), 6) for x in norm_ref]),
            "norm_ref_in_force_mean": (
                None if norm_ref is None
                else round(float(np.mean(np.asarray(norm_ref,
                                                    dtype=np.float64))), 6)),
            "which": norm_ref_which,
            "alternatives": {
                str(k): {
                    "norm_ref": [round(float(x), 6) for x in v],
                    "mean": round(float(np.mean(np.asarray(v,
                                                dtype=np.float64))), 6),
                    "in_force": False,
                }
                for k, v in (norm_ref_alternatives or {}).items()
            },
            "note": (
                "alpha means an absolute per-site injected norm of "
                "alpha x norm_ref_in_force[s]. NEVER compare a bare alpha "
                "across rulers — on two 3B maps the same alpha injects 2.139x "
                "apart (docs/FINDINGS.md §2). Any alternative listed "
                "here is NOT being worn."
            ),
        },
        # THE PICK POLICIES, HONESTLY. `auto_policies` is what /loom's
        # `auto.policy` will accept; `auto_policies_unavailable` names what
        # measurement adopted and this server does not have. Every figure is
        # labelled "estimated, uncertified": a picking-skill estimate, not a
        # certified measurement (docs/FINDINGS.md §8).
        "auto_policies_unavailable": [dict(p) for p in auto_policies_unavailable],
        "auto_policy_caveat": (
            "★ The pick policy this project adopted by measurement (the "
            "gauge probe, 71% of an oracle's picking skill, docs/FINDINGS.md §8) is wired as "
            "`gauge` — it is the probe, and it is the only policy here that "
            "costs seconds. The ranking of what is measured: gauge 71% "
            "(probed) > loudest 43% (free, v1a) > distinct 33% (free) > "
            "random floor (docs/FINDINGS.md §8). "
            "`stay` and `swerve` were never scored as pick "
            "policies at all. DEFAULTS STAY FREE: no policy runs unless the "
            "request asks for it, and the probe additionally requires an "
            "explicit opt-in. Estimated, uncertified."
        ),
        # What a probe will COST at these settings, published so the operator
        # prices it before committing. This has to be visible because
        # a free policy already captures 43% of the available picking skill, so
        # the probe's latency buys ~27 points and the operator must be able to
        # see both halves of that trade.
        "probe": {
            "available": True,
            "endpoint": "POST /probe (or \"probe\": true on /loom)",
            "policy_key": "gauge",
            "base_modes": list(loom_probe.BASE_MODES),
            "default_base": loom_probe.DEFAULT_BASE_MODE,
            "default_reps": loom_probe.DEFAULT_REPS,
            # ★ NEVER let a cost estimate take /info down. estimate_cost_s
            # refuses k < 2, and --default-k is an unvalidated int, so a
            # server started with --default-k 1 would turn a merely degraded
            # config (a 400 on /loom) into a failing /info — the UI loads /info
            # first, so a throw here means the UI fails to load at all. /info
            # is pure dict assembly and must stay that way.
            "cost_estimate": _probe_cost_or_note(
                int(default_k),
                len(harvest_workers) if harvest_workers else 1),
            "definition": (
                "gauge = normalized rank of the target future among the "
                "same-pass futures by cosine distance in z space; cell = mean "
                "base nr − mean steered nr (pleroma.probe.freegauge.gauge_nr, "
                "imported not re-implemented)."
            ),
            "label": "estimated, uncertified — this gauge failed its "
                     "registered test as an INSTRUMENT and is adopted only as "
                     "a pick POLICY.",
        },
        # PREDICTED-MAGNITUDE DOSING. Opt-in; `flat` is the default, the
        # ruler every banked result used.
        # ★ `default` is DERIVED from this server's runtime default, never taken
        # from the static table. The table's own flag marks `flat`, so taking
        # it verbatim would make a server started with --dose-policy-default
        # predicted publish two fields that contradict each other about the
        # same fact, and a client trusting the per-entry flag would describe
        # the wrong server. One truth, derived, never two copies.
        "dose_policies": [
            {**dict(p), "default": str(p.get("key")) == str(dose_policy_default)}
            for p in dose_policies
        ],
        "dose_policy_default": str(dose_policy_default),
        "dose_scale_clamp": [float(x) for x in dose_scale_clamp],
        # LEVER KIND. `absolute` is the default, the lever every banked result
        # wore. `default` per entry is
        # DERIVED from this run's default, for the reason dose_policies gives.
        "lever_kinds": [
            {**dict(p), "default": str(p.get("key")) == str(lever_kind_default)}
            for p in lever_kinds
        ],
        "lever_kind_default": str(lever_kind_default),
    }
