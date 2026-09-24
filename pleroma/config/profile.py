"""The ModelProfile: the single source of defaults for one model's loom.

A stage that declares its own defaults for preset, sites, format, sampling,
lengths, λ/rank and dose policy drifts from the values actually served (a
dose policy of `flat` in code against `predicted` live; 192 futures against
384; a z width of 2713, the 3B's, on another model; top_p .95 against Model
C's .98). Here the served values ARE the defaults, and a profile is validated
as a whole before anything loads.

Two layers, deliberately separate:

- `ModelProfile` — facts about the model and the method (architecture, layer
  geometry, format, sampling, map operating point, steering, dose). No
  machine paths beyond the model id, so a profile can ship publicly.
- `Deployment` — where THIS install's artifacts live and how it is served
  (paths, sha pins, host/port, worker). Private by nature.

Load with `load_profile(path)` / `load_deployment(path)` (TOML).
"""

from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Strict(BaseModel):
    """Unknown keys are errors: a typo in a profile must not silently fall
    back to a default (that is the exact failure this module exists to end)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ── enums ────────────────────────────────────────────────────────────────────


class PromptMode(str, Enum):
    """How a turn becomes model input. `modelc` is a document format (no chat
    template, ever); `chat` requires the tokenizer's template."""

    CHAT = "chat"
    RAW = "raw"
    MODELC = "modelc"


class InjectionSpan(str, Enum):
    """Which positions a worn lever is added to.

    UNIFORM (`start_pos=0`, prompt + continuation) is the design: the lever is
    meant to act from position 0, and uniform injection beats continuation-only
    injection at every scale ≥0.15 and every time profile measured. CONTINUATION
    exists only to describe what the legacy vLLM dose lanes did — the 70B band
    was measured that way, so a band records the span it was measured under.
    """

    UNIFORM = "uniform"
    CONTINUATION = "continuation"


class LeverKind(str, Enum):
    """ABSOLUTE = W·x_i (what WEAR serves by default). CONTRAST =
    W·(x_i − mean of the other fan members), what v1a was fit on and what
    every behavioural validation measured."""

    ABSOLUTE = "absolute"
    CONTRAST = "contrast"


class DosePolicy(str, Enum):
    FLAT = "flat"
    PREDICTED = "predicted"


# ── the model ────────────────────────────────────────────────────────────────


class Architecture(_Strict):
    """Straight from the checkpoint's config.json."""

    num_layers: PositiveInt
    hidden_dim: PositiveInt
    num_attention_heads: PositiveInt
    num_kv_heads: PositiveInt
    head_dim: PositiveInt
    eos_token_ids: tuple[int, ...] = Field(min_length=1)
    pad_token_id: int | None = None

    @model_validator(mode="after")
    def _check(self) -> Architecture:
        if self.num_attention_heads * self.head_dim != self.hidden_dim:
            raise ValueError(
                f"heads × head_dim = {self.num_attention_heads}×{self.head_dim} "
                f"≠ hidden_dim {self.hidden_dim}")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a multiple of num_kv_heads")
        # ★ pad == eos makes trimming eat real ends-of-text, and pad harvested as
        # generated text poisons every result built on that harvest.
        if self.pad_token_id is not None and self.pad_token_id in self.eos_token_ids:
            raise ValueError(f"pad_token_id {self.pad_token_id} is also an eos id")
        return self


class LayerGeometry(_Strict):
    """The anamnesis signature's sampled layers. For non-8B models these are
    the 8B preset's fractional depths rescaled, with the top layer anchored to
    `num_layers - 1` (the 70B preset's convention)."""

    sampled: tuple[int, ...] = Field(min_length=1)
    pca: tuple[int, ...] = Field(min_length=1)
    early_cutoff: int
    late_cutoff: int


class ModelSpec(_Strict):
    model_id: str = Field(min_length=1)
    preset_name: str = Field(min_length=1)
    #: anamnesis model-registry file holding ``preset_name`` when anamnesis does
    #: not ship it (relative paths resolve against the repo root). Launched
    #: processes get it on ``ANAMNESIS_MODELS`` (pleroma.config.legacy.legacy_env).
    anamnesis_registry: Path | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    arch: Architecture
    layers: LayerGeometry


# ── format and sampling ──────────────────────────────────────────────────────


class FormatSpec(_Strict):
    mode: PromptMode
    #: Model C document header key (default `pleroma.format.modelc.DEFAULT_HEADER`).
    header: str | None = None
    stops: tuple[str, ...] = ()
    #: Pinned into chat templates that take one; unpinned, the date baked
    #: into the prompt drifts with the wall clock (duplication-map cluster 5).
    date_string: str | None = None

    @model_validator(mode="after")
    def _check(self) -> FormatSpec:
        if self.mode is PromptMode.MODELC:
            if not self.stops:
                raise ValueError("modelc mode needs its stop strings")
            if not self.header:
                raise ValueError("modelc mode needs a document header key")
        return self


class Sampling(_Strict):
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)


class Lengths(_Strict):
    future_tokens: PositiveInt
    reply_tokens: PositiveInt
    default_k: PositiveInt = 6
    max_turns: PositiveInt = 12


# ── map, steering, dose ──────────────────────────────────────────────────────


class BinsSpec(_Strict):
    layers: tuple[int, ...] = Field(min_length=1)
    n_bins: PositiveInt = 20
    resolution: Literal["A", "B"] = "B"


class MapSpec(_Strict):
    """The v1a operating point. λ=1e4 / rank 64 is the REGISTERED point,
    not the sweep's best cell: choosing the best cell of a descriptive sweep
    would be selection."""

    sites: tuple[int, ...] = Field(min_length=1)
    lam: float = Field(default=1e4, gt=0)
    rank: PositiveInt = 64
    target: Literal["raw", "sitenorm"] = "raw"
    #: the build's own spelling (pleroma.map.build.join.FAN_SOURCES); the
    #: served 70B map is member_fan (every member row, lever or not)
    fan_source: Literal["member_fan", "lever_group"] = "member_fan"
    bins: BinsSpec

    @field_validator("sites")
    @classmethod
    def _sorted_unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if list(v) != sorted(set(v)):
            raise ValueError(f"sites must be strictly increasing, got {v}")
        return v


class SteerSpec(_Strict):
    injection_span: InjectionSpan = InjectionSpan.UNIFORM
    lever_kind: LeverKind = LeverKind.ABSOLUTE


class DoseSpec(_Strict):
    policy: DosePolicy = DosePolicy.PREDICTED
    #: `predicted` per-site scale clamp (`pleroma.dose.policy.DOSE_SCALE_MIN`/`DOSE_SCALE_MAX`).
    scale_clamp: tuple[float, float] = (0.5, 2.0)

    @field_validator("scale_clamp")
    @classmethod
    def _ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not 0 < v[0] < v[1]:
            raise ValueError(f"scale_clamp must satisfy 0 < lo < hi, got {v}")
        return v


class ModelProfile(_Strict):
    name: str = Field(min_length=1)
    description: str = ""
    model: ModelSpec
    format: FormatSpec
    sampling: Sampling
    lengths: Lengths
    map: MapSpec
    steer: SteerSpec = SteerSpec()
    dose: DoseSpec = DoseSpec()

    @model_validator(mode="after")
    def _layers_in_range(self) -> ModelProfile:
        n = self.model.arch.num_layers
        for label, layers in (("map.sites", self.map.sites),
                              ("map.bins.layers", self.map.bins.layers),
                              ("model.layers.sampled", self.model.layers.sampled),
                              ("model.layers.pca", self.model.layers.pca)):
            bad = [layer for layer in layers if not 0 <= layer < n]
            if bad:
                raise ValueError(f"{label} {bad} outside [0, {n}) for {self.model.model_id}")
        return self


# ── the deployment ───────────────────────────────────────────────────────────


class PinnedPath(_Strict):
    """An artifact path, optionally sha-pinned (checked by `pleroma validate`)."""

    path: Path
    sha256: Sha256 | None = None


class HarvestWorker(_Strict):
    address: str = Field(pattern=r"^[^:\s]+:\d+$")
    timeout_s: float = Field(default=900.0, gt=0)
    lane: Literal["replay", "gpu"] = "gpu"
    device: str = "cuda:0"
    bins_device: str | None = None


class BuildInputs(_Strict):
    """Where THIS install's build inputs live (the `pleroma shelf|fit|export`
    stages read these; every path lands sha-pinned in the stage receipt)."""

    pairs_dir: Path
    levers: PinnedPath
    hiddens: list[PinnedPath] = Field(min_length=1)
    bins: PinnedPath
    #: the shelf (wide) map: the registered CV's frozen primary baseline. It is
    #: NOT an export input: the export computes the shelf stats itself.
    shelf_map: PinnedPath | None = None
    #: the registered CV's report (export's --heldout-report)
    cv_report: PinnedPath | None = None
    #: where stage outputs go when no --out is given
    out_dir: Path


class Deployment(_Strict):
    profile: str = Field(min_length=1, description="ModelProfile.name this deploys")
    model_path: Path | None = Field(
        default=None, description="overrides profile.model.model_id on this machine")
    map: PinnedPath
    discriminants: PinnedPath
    calib_dir: Path
    dose_band: PinnedPath | None = None
    work_dir: Path
    host: str = "127.0.0.1"
    port: int = Field(default=8767, gt=0, lt=65536)
    allow_lan: bool = False
    worker: HarvestWorker | None = None
    build: BuildInputs | None = None


# ── loading ──────────────────────────────────────────────────────────────────


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with Path(path).open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"profile file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: not valid TOML: {exc}") from exc


def load_profile(path: str | Path) -> ModelProfile:
    return ModelProfile.model_validate(_read_toml(Path(path)))


def load_deployment(path: str | Path) -> Deployment:
    return Deployment.model_validate(_read_toml(Path(path)))
