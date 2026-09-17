"""Per-head feature family: preserve head heterogeneity that head-averaging destroys.

The baseline tiers and the other families collapse attention over query heads
(`attn.mean(axis=0)`) and keys over KV heads (`k.mean(axis=0)`). Attention heads are
functionally specialized — attention-sink heads, positional heads, induction heads —
so the mean blends heterogeneous behaviors into one number. The raw tensors are
banked PER HEAD (attentions per query head, pre-RoPE keys per KV head), so these
features need NO re-extraction.

Three architecturally-motivated views, per sampled layer (8 features/layer):

  1. Head attention-entropy distribution — the division of labor across query heads
     (some razor-focused, some diffuse). Per-head entropy is length-normalized
     (÷ log seq_len), so it also carries the entropy-normalization that the dropped
     `temporal_dynamics` family used to host. Summaries: mean / std / min / max of
     the per-head time-averaged entropy. The **std** is the key new signal: head
     specialization (a sink head ~0 alongside a diffuse head ~1).
  2. Head-role stability — do heads keep their relative roles across the turn?
     Correlation of the per-head mean-entropy vector between the first and second
     half of the generation (high = stable roles; low = the head assignment
     reshuffles mid-generation).
  3. Per-head structural specialization —
       - `sink_head_std`: spread across query heads of per-head attention mass on
         position 0 (some heads are sinks, others aren't);
       - `kv_key_spread_head_{mean,std}`: per-KV-head key spread (mean cosine
         distance from that head's own centroid over time) and its spread across the
         GQA KV heads (do KV heads encode divergent content or collapse together).

All values are bounded/scale-aware so this family does not reintroduce a length
confound: entropies ∈ [0,1], stability ∈ [-1,1], sink/spread are fractions or
cosine distances.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# Number of features emitted per sampled layer (keep in sync with _per_head_names).
_N_PER_LAYER = 8
_MIN_T = 4


def _per_head_entropy_matrix(data: RawGenerationData, layer_idx: int, T: int) -> NDArray[np.float64]:
    """Length-normalized attention entropy per query head per step → [T, n_heads].

    Entropy of each head's attention distribution, divided by log(seq_len) so it is
    bounded in [0, 1] and does not drift with the growing context.
    """
    rows: list[NDArray[np.float64]] = []
    for t in range(T):
        attn = data.attentions[t][layer_idx].astype(np.float64)  # [n_heads, seq_len]
        if attn.ndim < 2 or attn.shape[0] == 0 or attn.shape[1] == 0:
            continue
        seq_len = attn.shape[1]
        row_sums = np.maximum(attn.sum(axis=1, keepdims=True), 1e-12)
        p = attn / row_sums
        p = np.maximum(p, 1e-30)
        ent = -(p * np.log(p)).sum(axis=1)  # [n_heads], nats
        ent = ent / np.log(max(seq_len, 2))  # → [0,1]
        rows.append(ent)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows)  # [T_valid, n_heads]


def _per_head_sink_matrix(data: RawGenerationData, layer_idx: int, T: int) -> NDArray[np.float64]:
    """Per-query-head attention mass on position 0 (the sink) per step → [T, n_heads]."""
    rows: list[NDArray[np.float64]] = []
    for t in range(T):
        attn = data.attentions[t][layer_idx].astype(np.float64)  # [n_heads, seq_len]
        if attn.ndim < 2 or attn.shape[0] == 0 or attn.shape[1] == 0:
            continue
        row_sums = np.maximum(attn.sum(axis=1), 1e-12)
        rows.append(attn[:, 0] / row_sums)  # fraction on position 0, per head
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows)  # [T_valid, n_heads]


def _per_kv_head_key_spread(data: RawGenerationData, layer_idx: int) -> NDArray[np.float64]:
    """Per-KV-head key spread → [n_kv_heads].

    For each KV head, mean cosine distance of its per-step pre-RoPE key from that
    head's own centroid. No head-averaging (unlike baseline `kv_key_spread`).
    """
    keys = data.pre_rope_keys.get(layer_idx)
    if keys is None or len(keys) < 2:
        return np.zeros(0, dtype=np.float64)
    K = np.stack([k for k in keys]).astype(np.float64)  # [T, n_kv_heads, head_dim]
    if K.ndim != 3:
        return np.zeros(0, dtype=np.float64)
    n_kv = K.shape[1]
    spreads = np.zeros(n_kv, dtype=np.float64)
    for h in range(n_kv):
        Kh = K[:, h, :]  # [T, head_dim]
        centroid = Kh.mean(axis=0, keepdims=True)
        kn = np.linalg.norm(Kh, axis=1)
        cn = float(np.linalg.norm(centroid))
        sims = (Kh * centroid).sum(axis=1) / np.maximum(kn * cn, 1e-12)
        spreads[h] = float((1.0 - sims).mean())
    return spreads


def _per_head_names(layer_idx: int) -> list[str]:
    p = f"ph_L{layer_idx}"
    return [
        f"{p}_head_entropy_mean",
        f"{p}_head_entropy_std",
        f"{p}_head_entropy_min",
        f"{p}_head_entropy_max",
        f"{p}_head_role_stability",
        f"{p}_sink_head_std",
        f"{p}_kv_key_spread_head_mean",
        f"{p}_kv_key_spread_head_std",
    ]


def extract_per_head(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
) -> FeatureFamilyResult:
    """Extract per-head heterogeneity features for each sampled layer.

    Parameters
    ----------
    data : RawGenerationData
        Per-token tensors (attentions per query head, keys per KV head).
    sampled_layers : list[int], optional
        Attention/key layer indices. Defaults to the 8B preset layers.
    """
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    T = len(data.attentions)
    features: list[float] = []
    names: list[str] = []

    if T < _MIN_T:
        for l_idx in sampled_layers:
            layer_names = _per_head_names(l_idx)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name="per_head",
        )

    num_attn_layers = data.attentions[0].shape[0] if T > 0 else 0

    for l_idx in sampled_layers:
        layer_names = _per_head_names(l_idx)
        if l_idx >= num_attn_layers:
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        try:
            ent = _per_head_entropy_matrix(data, l_idx, T)  # [T_valid, n_heads]
            sink = _per_head_sink_matrix(data, l_idx, T)     # [T_valid, n_heads]
            kv_spread = _per_kv_head_key_spread(data, l_idx)  # [n_kv_heads]

            if ent.size > 0:
                head_mean_ent = ent.mean(axis=0)  # [n_heads] per-head avg entropy
                e_mean = float(head_mean_ent.mean())
                e_std = float(head_mean_ent.std())
                e_min = float(head_mean_ent.min())
                e_max = float(head_mean_ent.max())
                # Role stability: corr of per-head mean entropy, first vs second half.
                half = ent.shape[0] // 2
                if half >= 1 and ent.shape[0] - half >= 1 and ent.shape[1] > 1:
                    a = ent[:half].mean(axis=0)
                    b = ent[half:].mean(axis=0)
                    if a.std() > 1e-12 and b.std() > 1e-12:
                        role_stability = float(np.corrcoef(a, b)[0, 1])
                    else:
                        role_stability = 0.0
                else:
                    role_stability = 0.0
            else:
                e_mean = e_std = e_min = e_max = role_stability = 0.0

            sink_head_std = float(sink.mean(axis=0).std()) if sink.size > 0 else 0.0
            kv_mean = float(kv_spread.mean()) if kv_spread.size > 0 else 0.0
            kv_std = float(kv_spread.std()) if kv_spread.size > 0 else 0.0

            vals = [e_mean, e_std, e_min, e_max, role_stability,
                    sink_head_std, kv_mean, kv_std]
            features.extend(vals)
            names.extend(layer_names)
        except Exception as e:  # pragma: no cover — defensive, keep dims consistent
            logger.warning(f"per_head failed at layer {l_idx}: {e}")
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)

    result = FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="per_head",
    )
    logger.info(
        f"Per-head: {len(result)} features "
        f"({len(sampled_layers)} layers × {_N_PER_LAYER})"
    )
    return result
