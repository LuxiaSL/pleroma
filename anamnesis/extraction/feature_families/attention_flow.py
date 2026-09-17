"""Attention flow features: WHERE attention goes, not just how spread it is.

Captures how the model distributes attention across different regions of
context (system prompt, early generation, mid generation, recent tokens)
and how this distribution evolves over generation.

Key features:
    - System prompt attention trajectory and decay rate
    - Region-based attention decomposition
    - Per-head strategy diversity (head specialization)
    - Temporal operators on attention time series
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def extract_attention_flow(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Extract features about WHERE attention goes during generation.

    Per sampled layer:
      - system_prompt_mass mean/std/decay_rate
      - region_masses (4 regions × mean/std)
      - head_diversity for recency and system_prompt mass
      - temporal operators on sys_prompt_mass and recency_bias time series

    Parameters
    ----------
    data : RawGenerationData
        Per-token tensors from a single generation.
    sampled_layers : list[int], optional
        Attention layer indices to compute features for.
        Defaults to [0, 8, 16, 20, 24, 28, 31].
    n_windows : int
        Number of windows for temporal decomposition.
    include_stft : bool
        Whether to include STFT features.
    """
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    T = len(data.attentions)
    features: list[float] = []
    names: list[str] = []

    if T < 2:
        # Return consistent-length zero vector
        for l_idx in sampled_layers:
            layer_names = _attention_flow_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name="attention_flow",
        )

    num_attn_layers = data.attentions[0].shape[0]
    num_heads = data.attentions[0].shape[1]
    prompt_len = data.prompt_length

    for l_idx in sampled_layers:
        prefix = f"attn_flow_L{l_idx}"

        if l_idx >= num_attn_layers:
            layer_names = _attention_flow_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        # ── Build per-step time series ──
        sys_prompt_mass_ts: list[float] = []  # per-step
        recency_bias_ts: list[float] = []
        per_head_sys_masses: list[NDArray] = []  # [T] entries of [n_heads] f64
        per_head_recency: list[NDArray] = []

        # Region masses: [sys_prompt, early_gen, mid_gen, recent]
        region_masses: list[list[float]] = [[] for _ in range(4)]

        mean_rows = data.mean_attention(l_idx)  # shared per-(gen,layer) cache (C2)
        for t in range(T):
            attn = data.attentions[t][l_idx]  # [n_heads, seq_len]
            seq_len = attn.shape[1]

            if seq_len == 0:
                sys_prompt_mass_ts.append(0.0)
                recency_bias_ts.append(0.0)
                for r in region_masses:
                    r.append(0.0)
                continue

            mean_attn = mean_rows[t]  # [seq_len] float64
            total_mass = max(float(mean_attn.sum()), 1e-12)

            # System prompt mass
            sys_mass = float(mean_attn[:prompt_len].sum()) / total_mass
            sys_prompt_mass_ts.append(sys_mass)

            # Recency bias: mass on last 20% of positions
            cutoff = max(1, int(seq_len * 0.8))
            recency = float(mean_attn[cutoff:].sum()) / total_mass
            recency_bias_ts.append(recency)

            # Region decomposition
            gen_start = prompt_len
            gen_len = seq_len - prompt_len
            if gen_len > 0:
                third = max(1, gen_len // 3)
                r_sys = sys_mass
                r_early = float(mean_attn[gen_start:gen_start + third].sum()) / total_mass
                r_mid = float(mean_attn[gen_start + third:gen_start + 2 * third].sum()) / total_mass
                r_recent = float(mean_attn[gen_start + 2 * third:].sum()) / total_mass
            else:
                r_sys, r_early, r_mid, r_recent = sys_mass, 0.0, 0.0, 0.0

            region_masses[0].append(r_sys)
            region_masses[1].append(r_early)
            region_masses[2].append(r_mid)
            region_masses[3].append(r_recent)

            # Per-head values for diversity computation (vectorized over heads;
            # row-wise axis reductions are bit-identical to the per-head loop)
            attn64 = attn.astype(np.float64)  # [n_heads, seq_len]
            h_totals = np.maximum(attn64.sum(axis=1), 1e-12)
            per_head_sys_masses.append(attn64[:, :prompt_len].sum(axis=1) / h_totals)
            per_head_recency.append(attn64[:, cutoff:].sum(axis=1) / h_totals)

        # ── Summary statistics ──
        sys_arr = np.array(sys_prompt_mass_ts, dtype=np.float64)
        rec_arr = np.array(recency_bias_ts, dtype=np.float64)

        # System prompt mass: mean, std
        features.append(float(sys_arr.mean()))
        names.append(f"{prefix}_prompt_mass_mean")
        features.append(float(sys_arr.std()))
        names.append(f"{prefix}_prompt_mass_std")

        # System prompt decay rate (exponential fit)
        decay_rate = _fit_decay_rate(sys_arr)
        features.append(decay_rate)
        names.append(f"{prefix}_prompt_decay_rate")

        # Region masses: mean, std for each region
        region_labels = ["prompt", "early_gen", "mid_gen", "recent"]
        for ri, label in enumerate(region_labels):
            r_arr = np.array(region_masses[ri], dtype=np.float64)
            features.append(float(r_arr.mean()))
            names.append(f"{prefix}_region_{label}_mean")
            features.append(float(r_arr.std()))
            names.append(f"{prefix}_region_{label}_std")

        # Head diversity: std of per-head values (averaged over time)
        if per_head_sys_masses:
            sys_diversity = float(np.mean([np.std(step) for step in per_head_sys_masses]))
            rec_diversity = float(np.mean([np.std(step) for step in per_head_recency]))
        else:
            sys_diversity = 0.0
            rec_diversity = 0.0
        features.append(sys_diversity)
        names.append(f"{prefix}_head_diversity_prompt")
        features.append(rec_diversity)
        names.append(f"{prefix}_head_diversity_recency")

        # ── Temporal operators ──
        op_f, op_n = apply_operators(
            sys_arr.astype(np.float32),
            prefix=f"{prefix}_prompt_mass",
            n_windows=n_windows, include_stft=include_stft,
        )
        features.extend(op_f.tolist())
        names.extend(op_n)

        op_f, op_n = apply_operators(
            rec_arr.astype(np.float32),
            prefix=f"{prefix}_recency_bias",
            n_windows=n_windows, include_stft=include_stft,
        )
        features.extend(op_f.tolist())
        names.extend(op_n)

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="attention_flow",
    )


def _fit_decay_rate(time_series: NDArray) -> float:
    """Fit exponential decay rate to a time series.

    Fits log(y) = -lambda * t + c via linear regression.
    Returns lambda (positive = decaying).
    """
    ts = np.asarray(time_series, dtype=np.float64)
    mask = ts > 1e-10
    if mask.sum() < 2:
        return 0.0

    log_vals = np.log(ts[mask])
    t_vals = np.arange(len(ts), dtype=np.float64)[mask]

    try:
        coeffs = np.polyfit(t_vals, log_vals, 1)
        return float(-coeffs[0])  # positive = decaying
    except Exception:
        warnings.warn(
            "attention_flow._fit_decay_rate: polyfit failed on a degenerate "
            "window; emitting 0.0 sentinel — audit the run if this repeats",
            RuntimeWarning,
        )
        return 0.0


@lru_cache(maxsize=None)
def _attention_flow_names(
    layer_idx: int,
    n_windows: int,
    include_stft: bool,
) -> tuple[str, ...]:
    """All feature names for one layer's attention flow features (cached; C5).
    Returns a tuple — callers only len() and extend() from it."""
    prefix = f"attn_flow_L{layer_idx}"
    names = [
        f"{prefix}_prompt_mass_mean",
        f"{prefix}_prompt_mass_std",
        f"{prefix}_prompt_decay_rate",
    ]
    for label in ["prompt", "early_gen", "mid_gen", "recent"]:
        names.append(f"{prefix}_region_{label}_mean")
        names.append(f"{prefix}_region_{label}_std")
    names.append(f"{prefix}_head_diversity_prompt")
    names.append(f"{prefix}_head_diversity_recency")

    # Temporal operator names for two time series
    for ts_name in ["prompt_mass", "recency_bias"]:
        for wi in range(n_windows):
            for stat in ["mean", "std", "slope"]:
                names.append(f"{prefix}_{ts_name}_w{wi}_{stat}")
        if include_stft:
            for feat in ["spectral_centroid", "bandwidth",
                          "low_band_energy", "mid_band_energy", "high_band_energy"]:
                names.append(f"{prefix}_{ts_name}_{feat}")

    return tuple(names)
