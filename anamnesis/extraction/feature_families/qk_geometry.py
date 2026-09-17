"""Query / QK-space geometry — pre-RoPE query trajectory + q·k CONTENT alignment.

Queries are banked pre-RoPE at SAMPLED layers; keys pre-RoPE all-layer. q·k is the ONE cross-projection
dot product the model actually computes (attention logits), so — unlike v·k (never compared) or cross-layer
k·k (different bases) — comparing q and k here is legitimate.

RoPE scope (why we stay pre-RoPE, deliberately):
  - SELF-alignment q_t·k_t is between the SAME token at the SAME position, so RoPE rotates q and k by the
    same angle and the dot product is **RoPE-invariant** (pre-RoPE == post-RoPE). No RoPE needed.
  - Cross-position relative-RoPE geometry (q_t vs k_j, j<t — the actual attention logits over the cache)
    IS already captured by the ATTENTION surface (post-softmax). Recomputing it here would duplicate that
    and require O(T²) RoPE re-application. So we compute the **pre-RoPE CONTENT geometry** (position-free:
    what the token *looks for* vs what cached tokens *offer*), which the attention surface does NOT carry.
  Per-head q·k and explicit post-RoPE relative geometry are documented extensions (not v1).

GQA: q has n_q_heads, k has n_kv_heads (3:1/4:1). We use head-MEAN q/k in the shared head_dim space where
the model computes q·k (mirrors value/key geometry's head-mean). Reads `data.queries` + `data.pre_rope_keys`.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.operators import apply_operators
# reuse the canonical vector-trajectory primitives (mirror key/value geometry)
from anamnesis.extraction.feature_families.value_geometry import (
    _drift_halfcentroid, _drift_series, _eff_dim, _head_mean_matrix, _novelty_series, _spread,
)
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

_SCALARS = [
    "q_spread", "q_eff_dim", "q_drift_halfcentroid", "q_novelty_mean", "q_novelty_std",
    "qk_self_align_mean", "qk_self_align_std",         # cos(q_t, k_t) — RoPE-invariant content match
    "qk_cache_align_mean", "qk_cache_align_std",       # cos(q_t, running key centroid) — pre-RoPE content
]
_OP_SERIES = ["q_novelty", "qk_self_align"]


def _rowwise_cos(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    num = (A * B).sum(axis=1)
    den = np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)
    return num / den


def _qk_layer_names(l_idx: int, n_windows: int, include_stft: bool) -> list[str]:
    names = [f"qk_L{l_idx}_{s}" for s in _SCALARS]
    for ts in _OP_SERIES:
        for wi in range(n_windows):
            for stat in ["mean", "std", "slope"]:
                names.append(f"qk_L{l_idx}_{ts}_w{wi}_{stat}")
        if include_stft:
            for feat in ["spectral_centroid", "bandwidth",
                         "low_band_energy", "mid_band_energy", "high_band_energy"]:
                names.append(f"qk_L{l_idx}_{ts}_{feat}")
    return names


def extract_qk_geometry(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Pre-RoPE query-vector geometry + q·k content alignment per sampled layer."""
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    if data.queries is None or len(data.queries) == 0:
        logger.info("No queries banked — returning empty qk_geometry")
        return FeatureFamilyResult.empty("qk_geometry")

    features: list[float] = []
    names: list[str] = []
    layer_q_spread: dict[int, float] = {}

    for l_idx in sampled_layers:
        qseq = data.queries.get(l_idx)
        kseq = data.pre_rope_keys.get(l_idx) if data.pre_rope_keys else None
        if not qseq or len(qseq) < 2:
            layer_names = _qk_layer_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        Q = _head_mean_matrix(qseq)                 # [T, head_dim] pre-RoPE query content
        q_nov = _novelty_series(Q)
        q_drift = _drift_series(Q)
        spread = _spread(Q)
        layer_q_spread[l_idx] = spread

        # q·k content alignment (needs keys at this layer; keys are all-layer so usually present)
        if kseq and len(kseq) >= 2:
            K = _head_mean_matrix(kseq)
            n = min(len(Q), len(K))
            self_align = _rowwise_cos(Q[:n], K[:n])                       # RoPE-invariant content match
            cumsum = np.cumsum(K[:n], axis=0)
            running = cumsum / np.arange(1, n + 1, dtype=np.float64)[:, None]
            cache_align = _rowwise_cos(Q[:n], running)                    # pre-RoPE content vs cache centroid
        else:
            self_align = np.zeros(0)
            cache_align = np.zeros(0)

        scalar_vals = {
            "q_spread": spread,
            "q_eff_dim": _eff_dim(Q),
            "q_drift_halfcentroid": _drift_halfcentroid(Q),
            "q_novelty_mean": float(q_nov.mean()) if len(q_nov) else 0.0,
            "q_novelty_std": float(q_nov.std()) if len(q_nov) else 0.0,
            "qk_self_align_mean": float(self_align.mean()) if len(self_align) else 0.0,
            "qk_self_align_std": float(self_align.std()) if len(self_align) else 0.0,
            "qk_cache_align_mean": float(cache_align.mean()) if len(cache_align) else 0.0,
            "qk_cache_align_std": float(cache_align.std()) if len(cache_align) else 0.0,
        }
        for s in _SCALARS:
            features.append(float(scalar_vals[s]))
            names.append(f"qk_L{l_idx}_{s}")

        for ts_name, series in [("q_novelty", q_nov), ("qk_self_align", self_align)]:
            op_f, op_n = apply_operators(
                series.astype(np.float32), prefix=f"qk_L{l_idx}_{ts_name}",
                n_windows=n_windows, include_stft=include_stft,
            )
            features.extend(op_f.tolist())
            names.extend(op_n)

    # basis-free cross-layer dispersion of query spread (no cross-layer cosine — C4 trap)
    if len(layer_q_spread) >= 2:
        features.append(float(np.std(list(layer_q_spread.values()))))
    else:
        features.append(0.0)
    names.append("q_crosslayer_spread_diversity")

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="qk_geometry",
    )
