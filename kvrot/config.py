"""Strongly-typed (pydantic) specs for architecture, RoPE, and eviction policy."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from torch import Tensor

from . import rope

LayerType = Literal["full_attention", "sliding_attention"]


class RoPESpec(BaseModel):
    """Everything needed to rebuild the exact ``inv_freq`` the model rotates with."""

    head_dim: int
    theta: float
    scaling_type: Literal["none", "llama3"] = "none"
    # llama3-only fields
    factor: float | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    original_max_position_embeddings: int | None = None

    def build_inv_freq(self, *, device=None) -> Tensor:
        if self.scaling_type == "none":
            return rope.default_inv_freq(self.head_dim, self.theta, device=device)
        if self.scaling_type == "llama3":
            missing = [
                k
                for k in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
                if getattr(self, k) is None
            ]
            if missing:
                raise ValueError(f"llama3 rope scaling missing fields: {missing}")
            return rope.llama3_inv_freq(
                self.head_dim,
                self.theta,
                factor=self.factor,  # type: ignore[arg-type]
                low_freq_factor=self.low_freq_factor,  # type: ignore[arg-type]
                high_freq_factor=self.high_freq_factor,  # type: ignore[arg-type]
                original_max_position_embeddings=self.original_max_position_embeddings,  # type: ignore[arg-type]
                device=device,
            )
        raise ValueError(f"unknown scaling_type {self.scaling_type!r}")

    @classmethod
    def from_hf_config(cls, cfg: Any) -> "RoPESpec":
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        # transformers 5.x relocates rope params INTO the rope dict (and may call
        # it rope_parameters); 5.x configs also *raise* on unset attrs, so a
        # top-level getattr default silently misreads theta (e.g. Llama-3.2:
        # rope_scaling["rope_theta"]=500000 but cfg.rope_theta absent -> the old
        # code built a theta=10000 table, 0.13 max error — caught by exp12 gate 2).
        scaling = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta_raw = None
        if isinstance(scaling, dict):
            theta_raw = scaling.get("rope_theta")
        if theta_raw is None:
            theta_raw = getattr(cfg, "rope_theta", None)
        theta = float(theta_raw) if theta_raw is not None else 10000.0
        if not scaling:
            return cls(head_dim=head_dim, theta=theta, scaling_type="none")
        rtype = scaling.get("rope_type", scaling.get("type"))
        # transformers 5.x standardizes rope_scaling to {"rope_type": "default"} even when the
        # config had null, so "default"/None == plain RoPE (no scaling).
        if rtype in (None, "default"):
            return cls(head_dim=head_dim, theta=theta, scaling_type="none")
        if rtype == "llama3":
            return cls(
                head_dim=head_dim,
                theta=theta,
                scaling_type="llama3",
                factor=scaling["factor"],
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                original_max_position_embeddings=scaling["original_max_position_embeddings"],
            )
        # Unknown/advanced scaling (yarn, dynamic, ...): don't block loading. The harness pulls
        # the real inv_freq off the live model and asserts the rotation homomorphism, so runtime
        # correctness holds regardless; 'none' is only an approximate build_inv_freq fallback.
        return cls(head_dim=head_dim, theta=theta, scaling_type="none")


class ArchSpec(BaseModel):
    """Per-layer attention topology — drives which layers get exact-free eviction."""

    num_hidden_layers: int
    num_kv_heads: int
    head_dim: int
    layer_types: list[LayerType]
    sliding_window: int | None
    rope: RoPESpec
    # Per-layer: is RoPE actually applied to this layer's keys? Critical for re-rotation:
    # we must only re-rotate layers whose stored keys were rotated. Most models rotate
    # every layer; afmoe/trinity rotates ONLY its sliding layers (global layers are NoPE),
    # so re-rotating those would corrupt them.
    applies_rope: list[bool]

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def sliding_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    @classmethod
    def from_hf_config(cls, cfg: Any) -> "ArchSpec":
        n = cfg.num_hidden_layers
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        sw = getattr(cfg, "sliding_window", None)
        raw_types = getattr(cfg, "layer_types", None)
        if raw_types:
            layer_types = [
                "sliding_attention" if "sliding" in t else "full_attention" for t in raw_types
            ]
        else:
            # uniform full attention (e.g. Llama-3.2) unless a sliding_window is set
            layer_types = ["full_attention"] * n

        # Which layers actually rotate their keys? afmoe (trinity) applies RoPE only on its
        # sliding layers; its global layers are NoPE. Verified in modeling_afmoe.py
        # (apply_rotary_pos_emb is gated by `is_local_attention`). Everything else: all layers.
        if getattr(cfg, "model_type", "") == "afmoe":
            applies_rope = [t == "sliding_attention" for t in layer_types]
        else:
            applies_rope = [True] * n

        return cls(
            num_hidden_layers=n,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
            layer_types=layer_types,  # type: ignore[arg-type]
            sliding_window=sw,
            rope=RoPESpec.from_hf_config(cfg),
            applies_rope=applies_rope,
        )


class EvictionSpec(BaseModel):
    """Phase-1 eviction policies (Tier 0/1)."""

    policy: Literal["none", "oldest", "sink_window"] = "sink_window"
    num_sink_tokens: int = Field(4, ge=0, description="leading tokens never evicted (attention sinks)")
    evict_count: int = Field(0, ge=0, description="# of post-sink prefix tokens to drop")
    recompact: bool = Field(
        True, description="re-rotate survivors to contiguous positions (StreamingLLM-style)"
    )
