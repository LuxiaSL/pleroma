"""KVSnapshot — a serving-engine-agnostic view of a KV cache, plus the surgery ops.

We deliberately work on our own ``KVSnapshot`` rather than directly on HF's
``DynamicCache`` so the surgery (evict + reindex) is pure tensor math, unit-testable on
CPU with no model, and trivially portable to vLLM/SGLang later. Adapters convert
to/from HF caches in the harness.

Layout per layer: keys/values are ``[batch, kv_heads, seq, head_dim]`` (the HF cache
layout). Positions are tracked **per layer** so hybrid models (trinity: 45 sliding + 15
full) can evict different sets on different layers — sliding layers can drop everything
older than the window *exactly*, while full layers keep sinks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .rope import reindex_keys


@dataclass
class KVSnapshot:
    keys: list[Tensor]       # per layer [B, kv_heads, S, head_dim]
    values: list[Tensor]     # per layer [B, kv_heads, S, head_dim]
    positions: list[Tensor]  # per layer [S] absolute positions (long)
    layer_types: list[str]   # per layer "full_attention" | "sliding_attention"
    applies_rope: list[bool] | None = None  # per layer; None => all True (rotate every layer)

    def __post_init__(self) -> None:
        n = len(self.keys)
        if not (len(self.values) == len(self.positions) == len(self.layer_types) == n):
            raise ValueError("KVSnapshot: per-layer list lengths must match")
        if self.applies_rope is None:
            self.applies_rope = [True] * n
        elif len(self.applies_rope) != n:
            raise ValueError("KVSnapshot: applies_rope length must match #layers")
        for li in range(n):
            if self.keys[li].shape[-2] != self.positions[li].shape[0]:
                raise ValueError(f"layer {li}: key seq != positions length")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def seq_len(self, layer: int = 0) -> int:
        return int(self.keys[layer].shape[-2])

    def clone(self) -> "KVSnapshot":
        return KVSnapshot(
            keys=[k.clone() for k in self.keys],
            values=[v.clone() for v in self.values],
            positions=[p.clone() for p in self.positions],
            layer_types=list(self.layer_types),
            applies_rope=list(self.applies_rope) if self.applies_rope is not None else None,
        )

    def next_position(self) -> int:
        """The position id the next generated token should use (max over layers + 1)."""
        return int(max(int(p.max().item()) for p in self.positions)) + 1


def _as_per_layer(x, num_layers: int) -> list:
    """Accept either one value applied to all layers, or a per-layer list."""
    if isinstance(x, (list, tuple)):
        if len(x) != num_layers:
            raise ValueError(f"expected {num_layers} per-layer entries, got {len(x)}")
        return list(x)
    return [x] * num_layers


def evict(snapshot: KVSnapshot, keep_indices) -> KVSnapshot:
    """Drop all but ``keep_indices`` along the sequence dim.

    ``keep_indices`` is a 1-D LongTensor (applied to every layer) or a per-layer list
    of them. Positions are carried along unchanged (use :func:`reindex` to renumber).
    """
    keeps = _as_per_layer(keep_indices, snapshot.num_layers)
    keys, values, positions = [], [], []
    for li in range(snapshot.num_layers):
        idx = keeps[li].to(snapshot.keys[li].device)
        keys.append(snapshot.keys[li].index_select(-2, idx))
        values.append(snapshot.values[li].index_select(-2, idx))
        # positions may live on a different device than this layer's keys (sharded models)
        positions.append(snapshot.positions[li].to(idx.device).index_select(0, idx))
    return KVSnapshot(
        keys=keys,
        values=values,
        positions=positions,
        layer_types=list(snapshot.layer_types),
        applies_rope=list(snapshot.applies_rope) if snapshot.applies_rope is not None else None,
    )


def reindex(snapshot: KVSnapshot, new_positions, inv_freq: Tensor) -> KVSnapshot:
    """Re-rotate keys so each token sits at its new position (Tier 0, exact).

    ``new_positions`` is a 1-D LongTensor (same for all layers) or a per-layer list.
    Values are untouched. ``delta = new_pos - old_pos`` per token.
    """
    news = _as_per_layer(new_positions, snapshot.num_layers)
    applies = snapshot.applies_rope or [True] * snapshot.num_layers
    keys, positions = [], []
    for li in range(snapshot.num_layers):
        new_p = news[li].to(snapshot.positions[li].device)
        if new_p.shape[0] != snapshot.positions[li].shape[0]:
            raise ValueError(f"layer {li}: new_positions length != current seq len")
        if applies[li]:
            delta = (new_p - snapshot.positions[li]).to(torch.float32)
            keys.append(reindex_keys(snapshot.keys[li], delta, inv_freq))
        else:
            # NoPE layer (e.g. trinity's global layers): keys carry no position, leave as-is.
            keys.append(snapshot.keys[li])
        positions.append(new_p.to(snapshot.positions[li].dtype))
    return KVSnapshot(
        keys=keys,
        values=[v for v in snapshot.values],
        positions=positions,
        layer_types=list(snapshot.layer_types),
        applies_rope=list(applies),
    )


# --- HF adapters (used by the GPU harness; thin, no heavy deps at import) -------

def from_hf_cache(
    past_key_values,
    *,
    positions: Tensor,
    layer_types: list[str],
    applies_rope: list[bool] | None = None,
) -> KVSnapshot:
    """Build a KVSnapshot from a HF Cache or legacy tuple. ``positions`` is the shared
    absolute-position vector for the current (pre-surgery) sequence. ``applies_rope`` gates
    which layers get re-rotated (None => all)."""
    keys, values = _extract_kv(past_key_values)
    n = len(keys)
    if len(layer_types) != n:
        raise ValueError(f"layer_types len {len(layer_types)} != #layers {n}")
    if applies_rope is not None and len(applies_rope) != n:
        raise ValueError(f"applies_rope len {len(applies_rope)} != #layers {n}")
    return KVSnapshot(
        keys=list(keys),
        values=list(values),
        # align each layer's positions to that layer's key device (sharded models split layers)
        positions=[positions.to(keys[i].device) for i in range(n)],
        layer_types=list(layer_types),
        applies_rope=list(applies_rope) if applies_rope is not None else None,
    )


def _extract_kv(past_key_values):
    # transformers >= 5: Cache has .layers, each layer exposes .keys / .values
    if hasattr(past_key_values, "layers"):
        layers = [ly for ly in past_key_values.layers if getattr(ly, "keys", None) is not None]
        return [ly.keys for ly in layers], [ly.values for ly in layers]
    # transformers < 5: Cache exposed key_cache / value_cache lists
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    # legacy tuple-of-tuples
    if isinstance(past_key_values, (list, tuple)):
        keys = [layer[0] for layer in past_key_values]
        values = [layer[1] for layer in past_key_values]
        return keys, values
    raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")


def to_hf_dynamic_cache(snapshot: KVSnapshot):
    """Materialise a transformers ``DynamicCache`` from a snapshot (import is lazy)."""
    try:
        from transformers import DynamicCache
    except ImportError as e:  # pragma: no cover - exercised only on the GPU harness
        raise ImportError("transformers required for HF cache materialisation (pip extra 'hf')") from e

    cache = DynamicCache()
    for li in range(snapshot.num_layers):
        cache.update(snapshot.keys[li], snapshot.values[li], li)
    return cache
