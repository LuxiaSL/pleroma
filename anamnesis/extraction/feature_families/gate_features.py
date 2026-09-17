"""SwiGLU gate activation features.

Extracts statistics from the MLP gate_proj output (pre-SiLU activation).
SiLU is applied here to get actual gate values: gate = x * sigmoid(x).

These features capture:
    - Gate sparsity: how selective is the MLP (fraction of active dims)
    - Effective dimensionality: participation ratio of gate magnitudes
    - Gate drift: how much the activation pattern changes between steps
    - Cross-layer diversity: how much gate patterns differ across layers
    - Temporal dynamics: windowed + STFT decomposition of gate time series

Architecture-specific: only works for gated-MLP models (Llama, Mistral,
Gemma, Qwen). Returns empty result for models without gate hooks.
See research/planning/multi-model-feature-architecture.md for design notes.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families._helpers import _cosine_dist
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def _silu(x: NDArray) -> NDArray:
    """SiLU activation: x * sigmoid(x). Numerically stable."""
    # Clip to avoid overflow in exp for large negative values
    x_clipped = np.clip(x, -88.0, 88.0)
    return x * (1.0 / (1.0 + np.exp(-x_clipped)))


def extract_gate_features(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    sparsity_threshold: float = 0.01,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Extract SwiGLU gate activation statistics.

    IMPORTANT: data.gate_activations contains pre-SiLU gate_proj outputs.
    We apply SiLU here to get actual gate values.

    Per sampled layer:
      - gate_sparsity mean/std: fraction of dims with |SiLU(gate)| > threshold
      - gate_eff_dim mean/std: participation ratio of |SiLU(gate)|
      - gate_drift mean/std: cosine distance between consecutive gate vectors
      - gate_topk_overlap mean: Jaccard of top-k active dims across steps

    Cross-layer features:
      - gate_diversity: std of mean sparsity across layers
      - gate_layer_agreement: mean cosine sim of gate vectors across layer pairs

    Plus temporal operators on sparsity and drift time series.

    Parameters
    ----------
    data : RawGenerationData
        Must have gate_activations populated (from gate_proj hooks).
    sampled_layers : list[int], optional
        Which layers to compute gate features for.
    sparsity_threshold : float
        Threshold for |SiLU(gate)| to count as "active".
    n_windows : int
        Temporal decomposition windows.
    include_stft : bool
        Whether to include STFT features.
    """
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    # Check if gate data exists
    if data.gate_activations is None or len(data.gate_activations) == 0:
        logger.info("No gate activations available — returning empty gate features")
        return FeatureFamilyResult.empty("gate_features")

    T = len(data.hidden_states)
    if T < 2:
        return FeatureFamilyResult.empty("gate_features")

    features: list[float] = []
    names: list[str] = []

    # Per-layer features
    layer_sparsities: dict[int, float] = {}  # for cross-layer diversity

    for l_idx in sampled_layers:
        prefix = f"gate_L{l_idx}"
        gate_list = data.gate_activations.get(l_idx)

        if not gate_list or len(gate_list) < 2:
            # Layer not available — zeros
            layer_names = _gate_layer_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        # Apply SiLU to get actual gate values
        n_steps = min(T, len(gate_list))
        gates_silu = [_silu(gate_list[t].astype(np.float64)) for t in range(n_steps)]

        # ── Sparsity: fraction of active dimensions ──
        sparsity_ts: list[float] = []
        for g in gates_silu:
            active = float(np.mean(np.abs(g) > sparsity_threshold))
            sparsity_ts.append(active)
        sp_arr = np.array(sparsity_ts, dtype=np.float64)
        features.append(float(sp_arr.mean()))
        names.append(f"{prefix}_sparsity_mean")
        features.append(float(sp_arr.std()))
        names.append(f"{prefix}_sparsity_std")
        layer_sparsities[l_idx] = float(sp_arr.mean())

        # ── Effective dimensionality: participation ratio ──
        eff_dims: list[float] = []
        for g in gates_silu:
            magnitudes = np.abs(g)
            s2 = magnitudes ** 2
            s2_sum = s2.sum()
            if s2_sum > 1e-12:
                eff_dim = float((magnitudes.sum() ** 2) / (s2.sum()))
            else:
                eff_dim = 0.0
            eff_dims.append(eff_dim)
        ed_arr = np.array(eff_dims, dtype=np.float64)
        features.append(float(ed_arr.mean()))
        names.append(f"{prefix}_eff_dim_mean")
        features.append(float(ed_arr.std()))
        names.append(f"{prefix}_eff_dim_std")

        # ── Gate drift: cosine distance between consecutive steps ──
        drift_ts: list[float] = []
        for t in range(n_steps - 1):
            d = _cosine_dist(
                gates_silu[t].astype(np.float32),
                gates_silu[t + 1].astype(np.float32),
            )
            drift_ts.append(d)
        dr_arr = np.array(drift_ts, dtype=np.float64) if drift_ts else np.zeros(1)
        features.append(float(dr_arr.mean()))
        names.append(f"{prefix}_drift_mean")
        features.append(float(dr_arr.std()))
        names.append(f"{prefix}_drift_std")

        # ── Top-k overlap: Jaccard of top-k active dims ──
        top_k = min(100, gates_silu[0].shape[0] // 10)
        overlaps: list[float] = []
        for t in range(n_steps - 1):
            top_curr = set(np.argsort(np.abs(gates_silu[t]))[-top_k:])
            top_next = set(np.argsort(np.abs(gates_silu[t + 1]))[-top_k:])
            if len(top_curr | top_next) > 0:
                jaccard = len(top_curr & top_next) / len(top_curr | top_next)
            else:
                jaccard = 0.0
            overlaps.append(jaccard)
        features.append(float(np.mean(overlaps)) if overlaps else 0.0)
        names.append(f"{prefix}_topk_overlap_mean")

        # ── Temporal operators ──
        op_f, op_n = apply_operators(
            sp_arr.astype(np.float32),
            prefix=f"{prefix}_sparsity",
            n_windows=n_windows, include_stft=include_stft,
        )
        features.extend(op_f.tolist())
        names.extend(op_n)

        op_f, op_n = apply_operators(
            dr_arr.astype(np.float32),
            prefix=f"{prefix}_drift",
            n_windows=n_windows, include_stft=include_stft,
        )
        features.extend(op_f.tolist())
        names.extend(op_n)

    # ── Cross-layer features ──
    if len(layer_sparsities) >= 2:
        # Diversity: std of mean sparsity across layers
        sp_values = list(layer_sparsities.values())
        features.append(float(np.std(sp_values)))
        names.append("gate_cross_layer_sparsity_diversity")

        # Agreement: mean cosine sim of gate vectors across layer pairs
        # Use the mean gate vector per layer (averaged over time)
        layer_mean_gates: dict[int, F32] = {}
        for l_idx in sampled_layers:
            gate_list = data.gate_activations.get(l_idx)
            if gate_list and len(gate_list) > 0:
                n_steps = min(T, len(gate_list))
                mean_gate = np.mean(
                    [_silu(gate_list[t].astype(np.float64)) for t in range(n_steps)],
                    axis=0,
                ).astype(np.float32)
                layer_mean_gates[l_idx] = mean_gate

        agreements: list[float] = []
        layer_keys = sorted(layer_mean_gates.keys())
        for i, l1 in enumerate(layer_keys):
            for l2 in layer_keys[i + 1:]:
                sim = 1.0 - _cosine_dist(layer_mean_gates[l1], layer_mean_gates[l2])
                agreements.append(sim)
        features.append(float(np.mean(agreements)) if agreements else 0.0)
        names.append("gate_cross_layer_agreement")
    else:
        features.extend([0.0, 0.0])
        names.extend(["gate_cross_layer_sparsity_diversity", "gate_cross_layer_agreement"])

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="gate_features",
    )


def _gate_layer_names(
    layer_idx: int,
    n_windows: int,
    include_stft: bool,
) -> list[str]:
    """Generate feature names for a single layer's gate features."""
    prefix = f"gate_L{layer_idx}"
    names = [
        f"{prefix}_sparsity_mean",
        f"{prefix}_sparsity_std",
        f"{prefix}_eff_dim_mean",
        f"{prefix}_eff_dim_std",
        f"{prefix}_drift_mean",
        f"{prefix}_drift_std",
        f"{prefix}_topk_overlap_mean",
    ]
    # Temporal operators for sparsity and drift
    for ts_name in ["sparsity", "drift"]:
        for wi in range(n_windows):
            for stat in ["mean", "std", "slope"]:
                names.append(f"{prefix}_{ts_name}_w{wi}_{stat}")
        if include_stft:
            for feat in ["spectral_centroid", "bandwidth",
                          "low_band_energy", "mid_band_energy", "high_band_energy"]:
                names.append(f"{prefix}_{ts_name}_{feat}")
    return names
