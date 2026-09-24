"""Typed POST bodies for the loom server.

★ COERCION-EXACT. Each ``from_blob`` performs plain Python coercions —
``str(blob.get("branch") or "loom")``, ``int(blob.get("k") or default_k)``,
``int(blob["index"])`` … — in a fixed order, so a malformed field raises a
predictable exception with a predictable message (ValueError/KeyError -> 400,
TypeError -> 500) before any handler runs, and the first malformed field in
that order is the one reported. The model is then built from already-coerced
values under ``strict=True``, so pydantic never re-coerces anything and cannot
change a value or an error.

``auto`` and ``probe`` travel as the parsed objects (``dict``) and are typed
where they are READ: ``AutoSpec.from_mapping`` in ``do_loom`` (after the draw)
and ``ProbeSpec.from_mapping`` in ``run_probe``. That placement is the
contract: a malformed ``auto.alpha`` is a 400 AFTER the draw, and a malformed
in-/loom probe setting leaves the draw standing with ``probe: {error}``.
Parsing either any earlier would turn both into a 400 before the draw.

The ``*Body`` models at the bottom are the WIRE contract — what JSON each POST
accepts, in canonical types — used for the published schema
(``pleroma.serve.schema``, docs/API.md). The server still parses with the
coercion-exact ``from_blob`` above, which is more lenient than the schema says
(``"k": "6"`` works); clients should send the canonical types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from pleroma.probe import orchestrator as loom_probe
from pleroma.serve.draws import request_detach_wear, request_probe

#: The enums the wire schema names. Literal types need static values; a unit
#: test pins each against the runtime tuple it mirrors (DOSE_POLICIES,
#: LEVER_KINDS, AUTO_POLICIES, BRANCHES, loom_probe.BASE_MODES).
DosePolicy = Literal["flat", "predicted"]
LeverKind = Literal["absolute", "contrast"]
AutoPolicy = Literal["distinct", "loudest", "stay", "swerve", "gauge"]
Branch = Literal["base", "loom"]
ProbeBase = Literal["fan", "fresh"]

#: A field kept RAW (validated later by resolve_dose_policy/resolve_lever_kind,
#: which own the refusal sentence and its code) but published with its enum.
RawDosePolicy = Annotated[Any, WithJsonSchema(
    {"anyOf": [{"enum": ["flat", "predicted"], "type": "string"}, {"type": "null"}]})]
RawLeverKind = Annotated[Any, WithJsonSchema(
    {"anyOf": [{"enum": ["absolute", "contrast"], "type": "string"}, {"type": "null"}]})]


class _Body(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    session: str


class ChatRequest(_Body):
    """POST /chat and POST /edit."""

    branch: str
    text: str

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any]) -> ChatRequest:
        return cls(session=session, branch=str(blob.get("branch") or "loom"),
                   text=str(blob.get("text") or ""))


class BranchRequest(_Body):
    """POST /undo and POST /reroll."""

    branch: str

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any]) -> BranchRequest:
        return cls(session=session, branch=str(blob.get("branch") or "loom"))


class TruncateRequest(_Body):
    branch: str
    keep_turns: int

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any]) -> TruncateRequest:
        branch = str(blob.get("branch") or "loom")
        return cls(session=session, branch=branch, keep_turns=int(blob["keep_turns"]))


class LoomRequest(_Body):
    text: str
    k: int
    horizon: int
    auto: dict[str, Any] | None
    detach_wear: bool
    #: None = no probe; {} = probe with defaults; else the probe's settings.
    probe: dict[str, Any] | None

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any], *, default_k: int,
                  future_tokens: int, detach_wear_default: bool) -> LoomRequest:
        text = str(blob.get("text") or "")
        k = int(blob.get("k") or default_k)
        horizon = int(blob.get("horizon") or future_tokens)
        auto = blob.get("auto") if isinstance(blob.get("auto"), dict) else None
        detach = request_detach_wear(blob, bool(detach_wear_default))
        # Absent/false -> no probe, which is the default and
        # keeps the turn free. true -> probe defaults; an object -> settings.
        probe = request_probe(blob)
        return cls(session=session, text=text, k=k, horizon=horizon, auto=auto,
                   detach_wear=detach,
                   probe=None if probe is None else dict(probe))


class ProbeRequest(_Body):
    text: str
    #: The WHOLE request blob is the probe's settings object (alpha, reps,
    #: horizon, base, n_base, wear).
    cfg: dict[str, Any]

    @classmethod
    def from_blob(cls, session: str, blob: Any) -> ProbeRequest:
        text = str(blob.get("text") or "")
        return cls(session=session, text=text,
                   cfg=dict(blob) if isinstance(blob, dict) else {})


class WearRequest(_Body):
    index: int
    alpha: float
    loom_id: str | None
    #: absent -> the server default (itself 'flat' unless started with
    #: --dose-policy-default predicted). Validated by resolve_dose_policy.
    dose_policy: Any
    #: absent -> --wear-lever-default. Validated by resolve_lever_kind.
    lever_kind: Any

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any]) -> WearRequest:
        index = int(blob["index"])
        alpha = float(blob.get("alpha", 0.5))
        loom_id = str(blob["loom_id"]) if blob.get("loom_id") else None
        return cls(session=session, index=index, alpha=alpha, loom_id=loom_id,
                   dose_policy=blob.get("dose_policy"),
                   lever_kind=blob.get("lever_kind"))


class WearCodeRequest(_Body):
    alpha: float
    group: str | None
    axis: int | None
    pole: str | None
    rank: int
    random_seed: int | None
    code: list[float] | None
    code_kind: str
    dose_policy: Any

    @classmethod
    def from_blob(cls, session: str, blob: Mapping[str, Any]) -> WearCodeRequest:
        alpha = float(blob.get("alpha", 0.35))
        group = str(blob["group"]) if blob.get("group") else None
        axis = int(blob["axis"]) if blob.get("axis") is not None else None
        pole = str(blob["pole"]) if blob.get("pole") else None
        rank = int(blob.get("rank", 0))
        random_seed = (int(blob["random_seed"])
                       if blob.get("random_seed") is not None else None)
        code = ([float(x) for x in blob["code"]]
                if blob.get("code") is not None else None)
        code_kind = str(blob.get("code_kind", "absolute"))
        return cls(session=session, alpha=alpha, group=group, axis=axis,
                   pole=pole, rank=rank, random_seed=random_seed, code=code,
                   code_kind=code_kind, dose_policy=blob.get("dose_policy"))


# ── the typed readings of /loom's ``auto`` and the probe settings ───────────


class AutoSpec(BaseModel):
    """/loom's ``auto`` object: pick a candidate by ``policy`` and wear it.

    ``from_mapping`` performs the do_loom coercions in their original order;
    ``lever_kind``/``dose_policy`` stay raw here and are resolved (and refused,
    with their codes) by ``resolve_lever_kind``/``resolve_dose_policy`` exactly
    where they always were — after ``pick_auto``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    policy: str = Field(description=f"one of {list(AutoPolicy.__args__)}; "
                        "refused (400) otherwise",
                        json_schema_extra={"enum": list(AutoPolicy.__args__)})
    alpha: float = Field(0.35, description="the dose the pick is worn at")
    lever_kind: RawLeverKind = Field(
        None, description="absent -> --wear-lever-default")
    dose_policy: RawDosePolicy = Field(
        None, description="absent -> --dose-policy-default")

    @classmethod
    def from_mapping(cls, auto: Mapping[str, Any]) -> AutoSpec:
        policy = str(auto.get("policy") or "")
        alpha = float(auto.get("alpha", 0.35))
        return cls(policy=policy, alpha=alpha, lever_kind=auto.get("lever_kind"),
                   dose_policy=auto.get("dose_policy"))


class ProbeWearSpec(BaseModel):
    """/probe's optional ``wear``: wear the probe's pick when it has one."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    alpha: float | None = Field(None, description="absent -> the probe's alpha")
    dose_policy: RawDosePolicy = None
    lever_kind: RawLeverKind = None


class ProbeSpec(BaseModel):
    """The probe's settings: /loom's ``probe`` object, or /probe's whole body.

    ``from_mapping`` performs run_probe's coercions in their original order
    (alpha, reps, horizon, base, n_base); the range checks stay in run_probe.
    ``wear`` is read (and type-checked) by do_probe, after the probe ran — it
    is only meaningful on /probe.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    alpha: float = Field(0.35, description="dose of each candidate's contrastive code")
    reps: int = Field(loom_probe.DEFAULT_REPS, ge=1,
                      description="replies per candidate; >=2 for `resolution`")
    horizon: int = Field(64, ge=8, description="tokens per probe reply")
    base: ProbeBase = Field(loom_probe.DEFAULT_BASE_MODE,  # type: ignore[assignment]
                            description="'fan' (leave-one-out) or 'fresh' (unworn replies)")
    n_base: int = Field(loom_probe.DEFAULT_N_BASE, ge=1,
                        description="unworn replies when base='fresh'")
    wear: ProbeWearSpec | None = Field(
        None, description="/probe only: wear the pick with these settings")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ProbeSpec:
        alpha = float(cfg.get("alpha", 0.35))
        reps = int(cfg.get("reps", loom_probe.DEFAULT_REPS))
        horizon = int(cfg.get("horizon", 64))
        base = str(cfg.get("base", loom_probe.DEFAULT_BASE_MODE))
        n_base = int(cfg.get("n_base", loom_probe.DEFAULT_N_BASE))
        # model_construct: the values are already coerced, and the range and
        # base checks belong to run_probe (which owns their sentences).
        return cls.model_construct(alpha=alpha, reps=reps, horizon=horizon,
                                   base=base, n_base=n_base, wear=None)


# ── the wire contract: what each POST accepts (schema only; see module doc) ──


class _WireBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    session: str = Field(min_length=1, description="the session tag; required on every POST")


class SessionBody(_WireBody):
    """POST /unwear, /reset, /restore."""


class ChatBody(_WireBody):
    """POST /chat and /edit."""

    branch: Branch = "loom"
    text: str = ""


class BranchBody(_WireBody):
    """POST /undo and /reroll."""

    branch: Branch = "loom"


class TruncateBody(_WireBody):
    branch: Branch = "loom"
    keep_turns: int = Field(ge=0)


class LoomBody(_WireBody):
    text: str = Field(description="the contemplated next user turn (not committed)")
    k: int | None = Field(None, ge=2, description="absent -> --default-k")
    horizon: int | None = Field(None, description="absent -> --future-tokens")
    auto: AutoSpec | None = Field(None, description="pick and wear a candidate")
    detach_wear: bool | None = Field(
        None, description="absent -> --detach-wear-default")
    probe: bool | ProbeSpec | None = Field(
        None, description="absent/false: no probe; true: defaults; object: settings")


class ProbeBody(ProbeSpec):
    """POST /probe: the whole body is the probe's settings, plus the prefix."""

    model_config = ConfigDict(extra="allow")

    session: str = Field(min_length=1)
    text: str = Field(description="the SAME contemplated turn the fan was drawn against")


class WearBody(_WireBody):
    index: int
    alpha: float = 0.5
    loom_id: str | None = Field(
        None, description="the draw this wear belongs to; a stale id is refused (stale_loom)")
    dose_policy: DosePolicy | None = None
    lever_kind: LeverKind | None = None


class WearCodeBody(_WireBody):
    """POST /wear_code: wear a code from another conversation (no UI calls it; the API keeps it)."""

    alpha: float = 0.35
    group: str | None = None
    axis: int | None = None
    pole: Literal["low", "high"] | None = None
    rank: int = 0
    random_seed: int | None = None
    code: list[float] | None = None
    code_kind: Literal["absolute", "differential"] = "absolute"
    dose_policy: Literal["flat"] | None = None
