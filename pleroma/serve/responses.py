"""Typed response bodies for the loom API, and the contract check that binds
them to what the handlers actually send.

★ HOW THESE ARE USED. The route handlers build their own dicts (their key
order is part of what clients see). ``pleroma.serve.http`` VALIDATES each
dict through its model here and then serialises the dict itself — so the
bytes on the wire are what the handler built, and the model is the enforced, published
contract (``pleroma/serve/api-schema.json``). ``check_response`` has two modes:

* lenient (production default): a body that does not fit its model is
  logged as a CONTRACT VIOLATION and still sent. A GPU draw that has already
  happened, and a session that has already been mutated and persisted, must
  never be turned into a 500 by a typing mistake.
* strict (the test suite; ``PLEROMA_API_STRICT=1``): a misfit raises, and so
  does any difference between the model's own dump and the dict — keys and
  JSON values — so a field the model silently dropped or renamed is caught.

Every model forbids unknown keys: a handler that grows a field without the
model growing it fails strict mode, which is what keeps the schema honest.

Where a sub-shape is GENUINELY OPEN it is typed as ``dict[str, Any]`` and
says so in its description: ``map_meta`` (whatever the map file carries),
the probe receipt's ``resolution`` block
(``pleroma.probe.orchestrator.resolution_report``'s
variance decomposition, which grows with the estimator), worker RPC detail
(``harvest_worker_detail``), the /wear_code ``source`` (three shapes, a dead
path), the dose-band ``derivation``/``zones`` (owned by pleroma.dose.band),
the probe ``cost_estimate`` (``pleroma.probe.orchestrator.estimate_cost_s``, or an
"unavailable" note), the policy/kind/dose tables' rows beyond their ``key``,
and the dead ``/atlas`` report.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

logger = logging.getLogger("loom_serve")

#: Environment switch for strict contract checking (tests set it on).
STRICT_ENV = "PLEROMA_API_STRICT"


class ContractViolation(RuntimeError):
    """A handler's body does not match its published response model.

    A RuntimeError on purpose: it must surface as a 500, never as a 400."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TableRow(BaseModel):
    """A row of a copy table (/info's policies, kinds, dose policies): the
    ``key`` is the contract; the other fields are prose that grows."""

    model_config = ConfigDict(extra="allow")

    key: str


# ── shared blocks ─────────────────────────────────────────────────────────────


class DoseReceipt(_Model):
    """``DoseScale.to_json()`` — the dose one wear applied, and why."""

    policy: str
    scale: list[float]
    scale_raw: list[float]
    scale_mean: float
    fan_mean_raw_norms: list[float] | None
    candidate_raw_norms: list[float] | None
    clamped: list[dict[str, Any]] = Field(
        description="one record per clamped site (open: pleroma.dose.policy)")
    clamp_range: list[float]
    note: str


class EffectiveAlpha(_Model):
    """``effective_alpha_json`` — what alpha means once the dose scale is in."""

    alpha: float
    alpha_effective_mean: float
    alpha_effective_per_site: list[float]
    alpha_effective_range: list[float]
    note: str


class WornPublic(_Model):
    """``LoomSession.worn_public()`` — the session's wear, minus the vectors."""

    index: int = Field(description="candidate index; -1 for a /wear_code wear")
    alpha: float
    loom_id: str | None
    per_site_norms_at_alpha1: list[float]
    code: list[float] | None
    active: bool = Field(description="false: restored metadata, nothing attached")
    dose: DoseReceipt | None
    lever_kind: str | None


class WornRecord(_Model):
    """``WornSnapshot.to_json()`` — a wear as persisted (GET /sessions, on-disk rows)."""

    index: int
    alpha: float
    loom_id: str | None
    code: list[float] | None
    per_site_norms_at_alpha1: list[float]
    loom_dir: str | None
    map_fingerprint: str | None
    source: str | None
    dose: dict[str, Any] | None = Field(description="the persisted dose receipt, as stored")
    lever_kind: str | None
    contrast_peers: list[int] | None


class FutureScores(_Model):
    """Advisory per-future scores; each is present only when computable."""

    distinct: float | None = None
    loudness: float | None = None
    predicted_dose_scale: float | None = None
    predicted_dose_scale_contrast: float | None = None
    cos_to_worn: float | None = None
    gauge: float | None = Field(None, description="probe-derived; null = unscorable")


class Future(_Model):
    """One future of a fan (``futures[i]`` on /loom, /probe, /state, /restore)."""

    index: int
    n_tokens: int
    text: str
    harvested: bool
    note: str | None
    scores: FutureScores
    code: list[float] | None
    code_contrast: list[float] | None = Field(
        None, description="absent on futures persisted without a contrast code")
    # prompt-mode modelc only
    dream: str | None = None
    tail_kind: str | None = None
    modelc_dream_blocks: int | None = None
    reply_words: int | None = None


class HistoryMessage(_Model):
    role: Literal["user", "assistant", "system"]
    content: str
    dream: str | None = Field(None, description="modelc assistant turns only")
    tail_kind: str | None = Field(None, description="modelc assistant turns only")


class WornIndexAlpha(_Model):
    """/loom's ``drawn_under_wear``: what was ATTACHED for this draw."""

    index: int
    alpha: float
    loom_id: str | None


class GaugeCandidate(_Model):
    """``CandidateGauge.to_json()``."""

    index: int
    gauge: float | None
    gauge_base_nr: float | None
    gauge_steered_nr: float | None
    gauge_reps: int
    note: str | None


class ProbeTiming(_Model):
    generate: float
    harvest: float
    total: float


class ProbeReceipt(_Model):
    """The probe receipt (``run_probe``; ``worn``/``wall_s`` added by /probe)."""

    loom_id: str | None
    k: int
    reps: int
    alpha: float
    horizon: int
    base: str
    n_base: int
    n_spans_harvested: int
    candidates: list[GaugeCandidate]
    ranking: list[int]
    pick: int | None
    dead_rows: list[str]
    session_worn_during_draw: bool
    clean_regime: bool
    resolution: dict[str, Any] = Field(
        description="OPEN: loom_probe.resolution_report's variance decomposition")
    timing_s: ProbeTiming
    harvest_via: str
    harvest_worker_detail: dict[str, Any] | None = Field(
        description="OPEN: the harvest worker's own reply, or null")
    probe_dir: str
    absolute_comparable: bool
    label: str
    worn: WornPublic | None = Field(None, description="/probe with `wear` only")
    wall_s: float | None = Field(None, description="/probe only")


class ProbeFailure(_Model):
    """/loom's ``probe`` when the in-/loom probe failed; the draw survives."""

    error: str
    note: str


class LoomTiming(_Model):
    generate: float
    harvest: float


class Spread(_Model):
    mean_pairwise_distance: float
    n_scored: int
    loudness_ref: float


class AutoSelected(_Model):
    policy: str
    effective_policy: str
    index: int
    alpha: float
    scores: FutureScores
    dose: DoseReceipt
    lever_kind: str
    effective_alpha: EffectiveAlpha


class ProgressGens(_Model):
    n_gens: int
    harvested: list[int]


class LoomProgress(_Model):
    """A live /loom or /probe's ``{stage, done, total}`` (+ harvest detail)."""

    stage: str
    done: int
    total: int
    gens: ProgressGens | None = None
    origin: list[list[int | None]] | None = Field(
        None, description="probe harvest: gen id -> [candidate|null, rep]")


class StoreInfo(_Model):
    """``SessionStore.to_json()``."""

    enabled: bool
    dir: str
    writes: int
    write_failures: int
    restored_sessions: int
    failed_sessions: int
    last_error: str | None


# ── POST responses ────────────────────────────────────────────────────────────


class ChatResponse(_Model):
    """POST /chat. ``dream``/``tail_kind`` lead the body in modelc mode."""

    dream: str | None = None
    tail_kind: str | None = None
    session: str
    branch: str
    reply: str
    n_turns: int
    dropped_messages: int
    worn: WornPublic | None
    elapsed_s: float


class EditResponse(ChatResponse):
    """POST /edit: a /chat body plus what it replaced."""

    replaced_user: str
    replaced_assistant: str


class RerollResponse(_Model):
    dream: str | None = None
    tail_kind: str | None = None
    session: str
    branch: str
    reply: str
    replaced: str
    reroll_n: int
    n_turns: int
    dropped_messages: int
    worn: WornPublic | None
    elapsed_s: float


class PoppedTurn(_Model):
    user: str
    assistant: str


class UndoResponse(_Model):
    session: str
    branch: str
    popped: PoppedTurn
    n_turns: int


class TruncateResponse(_Model):
    session: str
    branch: str
    n_turns: int
    dropped_messages: int


class UnwearResponse(_Model):
    session: str
    worn: None


class ResetResponse(_Model):
    session: str
    cleared: Literal[True]
    worn: None


class RestoreResponse(_Model):
    session: str
    restored: Literal[True]
    file: str
    n_turns: dict[str, int]
    n_looms: int
    loom_id: str | None
    worn: WornPublic | None
    futures: list[Future]


class WearResponse(_Model):
    session: str
    worn: WornPublic
    sites: list[int]
    dose: DoseReceipt
    effective_alpha: EffectiveAlpha
    lever_kind: str


class WearCodeResponse(_Model):
    session: str
    worn: WornPublic
    sites: list[int]
    source: dict[str, Any] = Field(
        description="OPEN (dead path): explicit_code | random_direction | bank_group")


class LoomResponse(_Model):
    session: str
    loom_id: str
    n_futures: int
    horizon: int
    prompt_mode: str
    prompt_length: int
    bins_prompt_floor: int
    bins_feasible: bool
    detach_wear: bool
    drawn_under_wear: WornIndexAlpha | None
    futures: list[Future]
    spread: Spread | None
    fan_mean_raw_norms: list[float] | None
    fan_mean_raw_norms_contrast: list[float] | None
    contrast_note: str | None
    auto_selected: AutoSelected | None
    probe: ProbeReceipt | ProbeFailure | None
    worn: WornPublic | None
    timing_s: LoomTiming
    harvest_via: str
    harvest_worker_detail: dict[str, Any] | None = Field(
        description="OPEN: the harvest worker's own reply, or null")
    loom_dir: str


class ProbeResponse(_Model):
    session: str
    probe: ProbeReceipt
    futures: list[Future]
    worn: WornPublic | None


# ── GET responses ─────────────────────────────────────────────────────────────


class StateResponse(_Model):
    session: str
    histories: dict[str, list[HistoryMessage]]
    worn: WornPublic | None
    loom_id: str | None
    futures: list[Future]
    n_looms: int
    loom_in_progress: LoomProgress | None
    probe: ProbeReceipt | None
    probe_note: str | None


class SessionRow(_Model):
    """``SessionListing.to_json()``."""

    session: str
    file: str
    ok: bool
    in_memory: bool
    n_turns: dict[str, int]
    n_messages: int
    n_looms: int
    loom_id: str | None
    worn: WornPublic | WornRecord | None = Field(
        description="in-memory rows: the live wear; on-disk rows: the persisted record")
    modified_unix: float
    modified_iso: str
    size_bytes: int
    error: str | None


class SessionsResponse(_Model):
    sessions: list[SessionRow]
    n_sessions: int
    persistence: StoreInfo


class ProgressResponse(_Model):
    session: str
    progress: LoomProgress | None


class WorkerHealth(_Model):
    url: str
    state: Literal["ok", "busy", "down", "error"]
    latency_ms: float | None
    detail: str | None


class PoolResponse(_Model):
    configured: int
    alive: int
    workers: list[WorkerHealth]
    timeout_s: float
    probed_at: float
    pool_width_configured: int
    single_worker: str | None


class PromptModeDetail(_Model):
    temperature: float
    top_p: float
    stop_strings: list[str] | None
    chat_template_present: bool
    modelc_header_name: str | None = None
    modelc_header: str | None = None
    modelc_header_is_bare: bool | None = None
    raw_turn_join: str | None = None
    raw_history_note: str | None = None


class InfoPersistence(StoreInfo):
    restore_on_start: bool
    restore_errors: list[str]
    schema_: int = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DoseBandInfo(_Model):
    """``dose_band_info_json`` — ``tier: "none"`` means UNCALIBRATED."""

    tier: str
    alpha_max: float | None
    source: str | None
    measured_at: str | None
    notes: Any = Field(description="OPEN: the band file's own notes")
    map_fingerprint: str | None
    derivation: dict[str, Any] | None = Field(
        description="OPEN: pleroma.dose.band derivation record")
    zones: list[dict[str, Any]] | None = Field(description="OPEN: DoseZone.to_json rows")
    lever_quality_caveat: str | None


class RulerAlternative(_Model):
    norm_ref: list[float]
    mean: float
    in_force: bool


class Ruler(_Model):
    norm_ref_in_force: list[float] | None
    norm_ref_in_force_mean: float | None
    which: str | None
    alternatives: dict[str, RulerAlternative]
    note: str


class ProbeInfo(_Model):
    available: bool
    endpoint: str
    policy_key: str
    base_modes: list[str]
    default_base: str
    default_reps: int
    cost_estimate: dict[str, Any] = Field(
        description="OPEN: loom_probe.estimate_cost_s, or {unavailable, why, note}")
    definition: str
    label: str


class ApiInfo(_Model):
    """``/info.api`` — which API this server speaks (docs/API.md)."""

    version: Literal["1"]
    prefix: str
    schema_sha256: str = Field(description="sha256 of GET /api/v1/schema's canonical JSON")
    auth_required: bool = Field(description="/api/v1 routes need a bearer token")
    legacy_aliases_open: bool = Field(
        description="the un-prefixed aliases answer without a token")


class InfoResponse(_Model):
    map: str
    map_meta: dict[str, Any] = Field(description="OPEN: whatever the map file carries")
    sites: list[int]
    branches: list[str]
    default_k: int
    future_tokens: int
    detach_wear_default: bool
    n_sessions: int
    prompt_mode: str
    prompt_mode_detail: PromptModeDetail | None
    harvest_worker: str | None
    harvest_workers: list[str] | None
    harvest_pool_width: int
    restored_sessions: int
    persistence: InfoPersistence
    auto_policies: list[TableRow]
    loudness_ref: float
    dose_band: DoseBandInfo
    ruler: Ruler
    auto_policies_unavailable: list[TableRow]
    auto_policy_caveat: str
    probe: ProbeInfo
    dose_policies: list[TableRow]
    dose_policy_default: str
    dose_scale_clamp: list[float]
    lever_kinds: list[TableRow]
    lever_kind_default: str
    api: ApiInfo


class OpenObject(RootModel[dict[str, Any]]):
    """An object with no fixed shape (the dead /atlas report; the schema itself)."""


# ── the check ────────────────────────────────────────────────────────────────


def strict_default() -> bool:
    return os.environ.get(STRICT_ENV, "").strip() not in ("", "0", "false", "no")


def _json_view(value: Any) -> Any:
    """The body as a client sees it: tuples as lists, 1 == 1.0."""
    return json.loads(json.dumps(value))


def check_response(model: type[BaseModel], body: Any, *, route: str,
                   strict: bool | None = None) -> None:
    """Validate ``body`` against ``model``. Never alters ``body``.

    Lenient: log a misfit and return. Strict: raise ``ContractViolation`` on a
    misfit OR when the model's dump differs from the body (a dropped field)."""
    strict = strict_default() if strict is None else strict
    try:
        parsed = model.model_validate(body)
    except ValidationError as exc:
        if strict:
            raise ContractViolation(f"{route}: body does not fit {model.__name__}: "
                                    f"{exc}") from exc
        logger.error("CONTRACT VIOLATION on %s: body does not fit %s — sent anyway "
                     "(lenient mode): %s", route, model.__name__, exc)
        return
    if not strict:
        return
    dumped = _json_view(parsed.model_dump(mode="json", exclude_unset=True,
                                          by_alias=True))
    original = _json_view(body)
    if dumped != original:
        raise ContractViolation(
            f"{route}: {model.__name__} does not round-trip the body — "
            f"model {json.dumps(dumped, sort_keys=True)[:2000]} != body "
            f"{json.dumps(original, sort_keys=True)[:2000]}")
