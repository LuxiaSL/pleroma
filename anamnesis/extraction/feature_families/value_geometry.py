"""Value-vector (v_proj) geometry — the OV-circuit storage surface.

The KV-cache "address" side (keys) is featurized in Tier-2.5 (`kv_key_*`); the **content** side
(values — what gets pulled back when the model attends) has **never been featurized**. v3 raw banks
`v_proj_values` all-layer per-KV-head, so this is CPU-only on banked tensors.

Mirrors the key-geometry math (spread / eff_dim / drift / novelty) on the head-mean value vector,
plus temporal operators (consistent with the v2 families) and a few value-specific stats.

METHODOLOGY — basis-free only. v3 deleted cross-layer key cosine (C4) because vectors from different
learned projections live in unrelated bases, so their cosine is uninterpretable. The SAME caveat
applies across surfaces (v_proj basis ≠ k_proj basis) and across layers. So we NEVER take a raw
cosine between value and key vectors, or between value vectors at different layers. Cross-surface /
cross-layer relations are expressed **basis-free**:
  - value↔key coupling  -> Pearson corr of their per-step *drift* time-series (scalar series, basis-free)
  - cross-head structure -> dispersion of per-head value *norms* (magnitudes, basis-free)
  - cross-layer structure -> dispersion of the per-layer *spread* scalar (basis-free)

Self-contained family module (same contract as gate_features). Reads `data.v_proj_values` and
`data.pre_rope_keys` (both all-layer in v3). Returns empty if values are absent.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families._helpers import _cosine_dist
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# per-layer scalar features, in extraction order (used for zero-fill name consistency)
_SCALARS = [
    "value_spread", "value_eff_dim", "value_drift_halfcentroid",
    "value_novelty_mean", "value_novelty_std",
    "value_crosshead_normcv_mean", "value_crosshead_normcv_std",
    "value_key_drift_corr",
]
_OP_SERIES = ["novelty", "drift"]   # time-series that get temporal operators


def _head_mean_matrix(seq: list[F32]) -> NDArray[np.float64]:
    """list of T arrays [n_kv_heads, head_dim] -> [T, head_dim] head-mean (mirrors key geometry)."""
    return np.stack([v.mean(axis=0) for v in seq]).astype(np.float64)


def _spread(M: NDArray[np.float64]) -> float:
    """Mean cosine distance from the centroid."""
    centroid = M.mean(axis=0, keepdims=True)
    m_norms = np.linalg.norm(M, axis=1)
    c_norm = float(np.linalg.norm(centroid))
    sims = (M * centroid).sum(axis=1) / np.maximum(m_norms * c_norm, 1e-12)
    return float((1.0 - sims).mean())


def _eff_dim(M: NDArray[np.float64]) -> float:
    """Participation ratio of singular values, bounded to a fraction of max rank (mirrors kv_key_eff_dim/B4)."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        s2 = s ** 2
        s2_sum = float(s2.sum())
        if s2_sum <= 1e-12:
            return 0.0
        pr = float((s2_sum ** 2) / float((s2 ** 2).sum()))
        return pr / max(min(M.shape[0], M.shape[1]), 1)
    except Exception:
        warnings.warn(
            "value_geometry._eff_dim: SVD raised; emitting 0.0 sentinel — "
            "audit the run if this repeats",
            RuntimeWarning,
        )
        return 0.0


def _novelty_series(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine distance of each vector from the running centroid of all prior vectors (vectorized)."""
    if len(M) < 2:
        return np.zeros(0, dtype=np.float64)
    cumsum = np.cumsum(M, axis=0)
    counts = np.arange(1, len(M) + 1, dtype=np.float64)[:, None]
    running = cumsum / counts
    a = M[1:]
    b = running[:-1]
    sims = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return 1.0 - sims


def _drift_series(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine distance between consecutive vectors (per-step drift)."""
    if len(M) < 2:
        return np.zeros(0, dtype=np.float64)
    a = M[:-1]
    b = M[1:]
    sims = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return 1.0 - sims


def _drift_halfcentroid(M: NDArray[np.float64]) -> float:
    mid = len(M) // 2
    if mid < 1 or mid >= len(M):
        return 0.0
    return _cosine_dist(M[:mid].mean(axis=0).astype(np.float32), M[mid:].mean(axis=0).astype(np.float32))


def _crosshead_norm_cv_series(seq: list[F32]) -> NDArray[np.float64]:
    """Per-step coefficient of variation (std/mean) of the per-head value-vector NORMS (basis-free).
    Captures whether some KV heads store much more than others, scale-free."""
    out = np.empty(len(seq), dtype=np.float64)
    for t, v in enumerate(seq):
        norms = np.linalg.norm(v.astype(np.float64), axis=1)   # [n_kv_heads]
        mean = float(norms.mean())
        out[t] = float(norms.std() / mean) if mean > 1e-12 else 0.0
    return out


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation of two scalar series; 0 if too short or either is constant (basis-free)."""
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _value_layer_names(l_idx: int, n_windows: int, include_stft: bool) -> list[str]:
    """Names for a single layer (extraction order), for zero-fill when a layer is missing."""
    prefix = f"value_L{l_idx}"
    names = [f"{prefix}_{s.replace('value_', '')}" if s.startswith("value_") else f"{prefix}_{s}"
             for s in _SCALARS]
    # the _SCALARS already start with "value_"; build clean names without double prefix:
    names = [f"value_L{l_idx}_{s[len('value_'):]}" for s in _SCALARS]
    for ts in _OP_SERIES:
        for wi in range(n_windows):
            for stat in ["mean", "std", "slope"]:
                names.append(f"value_L{l_idx}_{ts}_w{wi}_{stat}")
        if include_stft:
            for feat in ["spectral_centroid", "bandwidth",
                         "low_band_energy", "mid_band_energy", "high_band_energy"]:
                names.append(f"value_L{l_idx}_{ts}_{feat}")
    return names


def extract_value_geometry(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Extract v_proj value-vector geometry per sampled layer + basis-free cross-layer dispersion."""
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    if data.v_proj_values is None or len(data.v_proj_values) == 0:
        logger.info("No v_proj_values available — returning empty value_geometry")
        return FeatureFamilyResult.empty("value_geometry")

    features: list[float] = []
    names: list[str] = []
    layer_spread: dict[int, float] = {}      # for basis-free cross-layer dispersion

    for l_idx in sampled_layers:
        vseq = data.v_proj_values.get(l_idx)
        if not vseq or len(vseq) < 2:
            layer_names = _value_layer_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        M = _head_mean_matrix(vseq)                       # [T, head_dim]
        novelty = _novelty_series(M)
        drift = _drift_series(M)

        spread = _spread(M)
        layer_spread[l_idx] = spread
        nov_cv = _crosshead_norm_cv_series(vseq)

        # value↔key coupling: corr of value-drift vs key-drift series (basis-free)
        vk_corr = 0.0
        kseq = data.pre_rope_keys.get(l_idx) if data.pre_rope_keys else None
        if kseq and len(kseq) >= 3:
            vk_corr = _corr(drift, _drift_series(_head_mean_matrix(kseq)))

        scalar_vals = {
            "value_spread": spread,
            "value_eff_dim": _eff_dim(M),
            "value_drift_halfcentroid": _drift_halfcentroid(M),
            "value_novelty_mean": float(novelty.mean()) if len(novelty) else 0.0,
            "value_novelty_std": float(novelty.std()) if len(novelty) else 0.0,
            "value_crosshead_normcv_mean": float(nov_cv.mean()) if len(nov_cv) else 0.0,
            "value_crosshead_normcv_std": float(nov_cv.std()) if len(nov_cv) else 0.0,
            "value_key_drift_corr": vk_corr,
        }
        for s in _SCALARS:
            features.append(float(scalar_vals[s]))
            names.append(f"value_L{l_idx}_{s[len('value_'):]}")

        for ts_name, series in [("novelty", novelty), ("drift", drift)]:
            op_f, op_n = apply_operators(
                series.astype(np.float32), prefix=f"value_L{l_idx}_{ts_name}",
                n_windows=n_windows, include_stft=include_stft,
            )
            features.extend(op_f.tolist())
            names.extend(op_n)

    # ── basis-free cross-layer dispersion (NO cross-layer cosine — that's the C4 trap) ──
    if len(layer_spread) >= 2:
        features.append(float(np.std(list(layer_spread.values()))))
    else:
        features.append(0.0)
    names.append("value_crosslayer_spread_diversity")

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="value_geometry",
    )
