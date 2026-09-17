"""Temporal decomposition of core T2/T2.5 metrics.

Takes the per-step time series that state_extractor collapses to mean/std
and instead applies windowed statistics + STFT to capture temporal structure.

Key time series per layer:
    - attention_entropy: mean entropy across heads at each step
    - head_agreement: generalized JSD (head consensus) at each step
    - key_drift: cosine distance between consecutive pre-RoPE keys
    - key_novelty: cosine distance to running mean key
    - lookback_ratio: fraction of attention on prompt vs generation

These are the features shown to matter most (attn_entropy_std,
epoch_regularity_std were top discriminators). Temporal windowing should
capture WHEN mode information appears and how it evolves.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def _safe_entropy(probs: F32) -> float:
    """Entropy of a probability distribution, safe for zero/empty inputs."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(scipy_entropy(probs))


def _cosine_sim(a: F32, b: F32) -> float:
    """Cosine similarity, safe for zero vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_dist(a: F32, b: F32) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    return 1.0 - _cosine_sim(a, b)


def _vectorized_row_entropy(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute entropy for each row of a 2D probability array.

    Vectorized: no Python loops over heads. ~10-30x faster than
    calling _safe_entropy per row.

    Parameters
    ----------
    probs : array of shape [num_rows, seq_len]
        Each row is a probability distribution (may not be normalized).

    Returns
    -------
    Array of shape [num_rows] with entropy of each row.
    """
    # Normalize rows
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    p = probs / row_sums
    # Replace zeros to avoid log(0)
    p = np.maximum(p, 1e-30)
    return -np.sum(p * np.log(p), axis=1)


def _compute_attention_entropy_series(
    data: RawGenerationData,
    layer_idx: int,
    T: int,
) -> F32:
    """Compute mean attention entropy across heads at each step.

    Returns array of shape [T] with mean entropy per step.
    Vectorized across heads for performance.
    """
    series = np.zeros(T, dtype=np.float32)
    for t in range(T):
        attn = data.attentions[t][layer_idx]  # [num_heads, seq_len]
        if attn.ndim < 2 or attn.shape[0] == 0:
            continue
        # Vectorized entropy across all heads at once
        entropies = _vectorized_row_entropy(attn.astype(np.float64))
        series[t] = float(entropies.mean())
    return series


def _compute_head_agreement_series(
    data: RawGenerationData,
    layer_idx: int,
    T: int,
) -> F32:
    """Compute head agreement (1 - normalized JSD) at each step.

    Generalized JSD = H(mean_distribution) - mean(H(individual_heads)).
    Agreement = 1 - JSD / log(num_heads).
    Vectorized across heads for performance.
    """
    series = np.zeros(T, dtype=np.float32)
    for t in range(T):
        attn = data.attentions[t][layer_idx].astype(np.float64)  # [num_heads, seq_len]
        if attn.ndim < 2 or attn.shape[0] == 0:
            continue
        num_heads = attn.shape[0]

        # Normalize each head
        row_sums = attn.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        attn_norm = attn / row_sums

        # Vectorized: entropy of each head + entropy of mean
        head_entropies = _vectorized_row_entropy(attn_norm)  # [num_heads]
        mean_dist = attn_norm.mean(axis=0, keepdims=True)  # [1, seq_len]
        entropy_of_mean = float(_vectorized_row_entropy(mean_dist)[0])
        mean_of_entropies = float(head_entropies.mean())

        jsd = max(0.0, entropy_of_mean - mean_of_entropies)
        max_jsd = float(np.log(num_heads)) if num_heads > 1 else 1.0
        series[t] = float(1.0 - jsd / max(max_jsd, 1e-12))
    return series


def _compute_key_drift_series(
    data: RawGenerationData,
    layer_idx: int,
    T: int,
) -> F32:
    """Compute cosine distance between consecutive pre-RoPE keys.

    Returns array of shape [T] (first element is 0 since no previous key).
    Keys are averaged across KV heads before computing distance.
    """
    series = np.zeros(T, dtype=np.float32)
    keys = data.pre_rope_keys.get(layer_idx)
    if keys is None or len(keys) < 2:
        return series

    # Average across KV heads to get per-step key vector
    key_vectors = [k.mean(axis=0) for k in keys[:T]]

    for t in range(1, min(T, len(key_vectors))):
        series[t] = _cosine_dist(key_vectors[t - 1], key_vectors[t])
    return series


def _compute_key_novelty_series(
    data: RawGenerationData,
    layer_idx: int,
    T: int,
) -> F32:
    """Compute cosine distance to running centroid of pre-RoPE keys.

    Returns array of shape [T] (first element is 0).
    """
    series = np.zeros(T, dtype=np.float32)
    keys = data.pre_rope_keys.get(layer_idx)
    if keys is None or len(keys) < 2:
        return series

    key_vectors = [k.mean(axis=0) for k in keys[:T]]
    running_centroid = key_vectors[0].copy().astype(np.float64)

    for t in range(1, min(T, len(key_vectors))):
        series[t] = _cosine_dist(key_vectors[t], running_centroid.astype(np.float32))
        # Update running centroid (cumulative mean)
        running_centroid = (running_centroid * t + key_vectors[t].astype(np.float64)) / (t + 1)
    return series


def _compute_lookback_ratio_series(
    data: RawGenerationData,
    layer_idx: int,
    T: int,
) -> F32:
    """Compute lookback ratio (prompt attention / generation attention) per step.

    Returns array of shape [T].
    """
    series = np.zeros(T, dtype=np.float32)
    prompt_len = data.prompt_length

    for t in range(T):
        attn = data.attentions[t][layer_idx]  # [num_heads, seq_len]
        if attn.ndim < 2 or attn.shape[1] <= prompt_len:
            continue
        mean_attn = attn.mean(axis=0).astype(np.float64)
        prompt_mass = float(mean_attn[:prompt_len].sum())
        gen_mass = float(mean_attn[prompt_len:].sum())
        series[t] = prompt_mass / max(gen_mass, 1e-12)
    return series


# ── Feature names for consistent zero-padding ────────────────────────────────

_METRIC_NAMES = [
    "attn_entropy",
    "head_agreement",
    "key_drift",
    "key_novelty",
    "lookback_ratio",
]


def _temporal_dynamics_names(
    layer_idx: int,
    n_windows: int,
    include_stft: bool,
) -> list[str]:
    """Generate expected feature names for a single layer."""
    names: list[str] = []
    for metric in _METRIC_NAMES:
        prefix = f"td_L{layer_idx}_{metric}"
        # windowed_stats: 3 * n_windows features
        for w in range(n_windows):
            names.append(f"{prefix}_w{w}_mean")
            names.append(f"{prefix}_w{w}_std")
            names.append(f"{prefix}_w{w}_slope")
        # stft: 6 features
        if include_stft:
            for suffix in [
                "spectral_centroid", "bandwidth",
                "low_band_energy", "mid_band_energy", "high_band_energy",
            ]:
                names.append(f"{prefix}_{suffix}")
    return names


# ── Main extraction function ─────────────────────────────────────────────────

def extract_temporal_dynamics(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Extract temporal decomposition of core T2/T2.5 metrics.

    For each sampled layer, computes per-step time series of 5 metrics
    and applies windowed statistics + STFT temporal operators.

    Parameters
    ----------
    data : RawGenerationData
        Per-token tensors from a single generation.
    sampled_layers : list[int], optional
        Layer indices to compute features for.
        Defaults to [0, 8, 16, 20, 24, 28, 31].
    n_windows : int
        Number of temporal windows for windowed stats.
    include_stft : bool
        Whether to include STFT spectral features.
    """
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    T = len(data.attentions)
    features: list[float] = []
    names: list[str] = []

    if T < 4:
        # Not enough steps for meaningful temporal decomposition
        for l_idx in sampled_layers:
            layer_names = _temporal_dynamics_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name="temporal_dynamics",
        )

    # Metric extractors — each returns F32 of shape [T]
    metric_extractors = [
        ("attn_entropy", _compute_attention_entropy_series),
        ("head_agreement", _compute_head_agreement_series),
        ("key_drift", _compute_key_drift_series),
        ("key_novelty", _compute_key_novelty_series),
        ("lookback_ratio", _compute_lookback_ratio_series),
    ]

    for l_idx in sampled_layers:
        for metric_name, extractor in metric_extractors:
            prefix = f"td_L{l_idx}_{metric_name}"
            try:
                time_series = extractor(data, l_idx, T)
                op_features, op_names = apply_operators(
                    time_series,
                    prefix=prefix,
                    n_windows=n_windows,
                    include_stft=include_stft,
                )
                features.extend(op_features.tolist())
                names.extend(op_names)
            except Exception as e:
                logger.warning(
                    f"Failed {metric_name} at layer {l_idx}: {e}"
                )
                # Fall back to zeros with correct names
                fallback_names = []
                for w in range(n_windows):
                    fallback_names.extend([
                        f"{prefix}_w{w}_mean",
                        f"{prefix}_w{w}_std",
                        f"{prefix}_w{w}_slope",
                    ])
                if include_stft:
                    for suffix in [
                        "spectral_centroid", "bandwidth",
                        "low_band_energy", "mid_band_energy", "high_band_energy",
                    ]:
                        fallback_names.append(f"{prefix}_{suffix}")
                features.extend([0.0] * len(fallback_names))
                names.extend(fallback_names)

    result = FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="temporal_dynamics",
    )
    logger.info(
        f"Temporal dynamics: {len(result)} features "
        f"({len(sampled_layers)} layers × {len(_METRIC_NAMES)} metrics)"
    )
    return result
