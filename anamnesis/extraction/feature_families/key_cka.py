"""Cross-layer KV-cache CKA — basis-invariant key/value structure agreement across depth.

v3 deleted cross-layer key COSINE (C4): k_proj outputs at different layers live in unrelated learned
bases, so a raw cosine between them is uninterpretable. The correct tool is **linear CKA** (Centered
Kernel Alignment), which is invariant to orthogonal transforms and isotropic scaling — so it measures
representational agreement across different bases. This is the principled replacement: how similar is the
KV-cache geometry across layer depths, basis-cleanly.

Linear CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F)  over the SAME rows (time steps),
columns mean-centered. Computed on per-head-mean key/value matrices [T, head_dim] for sampled-layer pairs.
Keys and values are both all-layer in v3. Reads `data.pre_rope_keys` and `data.v_proj_values`.

Whole-sequence CKA per pair (one scalar) — no temporal operators (windowed CKA is a documented extension).
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families.value_geometry import _head_mean_matrix
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def _linear_cka(X: NDArray[np.float64], Y: NDArray[np.float64]) -> float:
    """Basis-invariant (orthogonal-transform + isotropic-scale invariant) representational similarity."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(Yc.T @ Xc) ** 2)              # ||Yc^T Xc||_F^2
    den = float(np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))
    return num / max(den, 1e-12)


def _cka_for_surface(seq_by_layer: dict[int, list], sampled_layers: list[int], prefix: str
                     ) -> tuple[list[float], list[str]]:
    """Per-pair + summary cross-layer CKA for one surface (keys or values)."""
    feats: list[float] = []
    names: list[str] = []
    # build per-layer head-mean matrices for available layers
    mats: dict[int, NDArray[np.float64]] = {}
    for l in sampled_layers:
        seq = seq_by_layer.get(l)
        if seq and len(seq) >= 2:
            mats[l] = _head_mean_matrix(seq)
    avail = [l for l in sampled_layers if l in mats]

    pairwise: dict[tuple[int, int], float] = {}
    for a, b in combinations(avail, 2):
        n = min(len(mats[a]), len(mats[b]))
        cka = _linear_cka(mats[a][:n], mats[b][:n]) if n >= 2 else 0.0
        pairwise[(a, b)] = cka
        feats.append(cka)
        names.append(f"{prefix}_L{a}_L{b}")

    # summaries: early↔late, adjacent (consecutive sampled layers), overall mean
    third = max(1, len(avail) // 3)
    early, late = set(avail[:third]), set(avail[-third:])
    el = [v for (a, b), v in pairwise.items() if (a in early and b in late) or (a in late and b in early)]
    adj = [pairwise[(avail[i], avail[i + 1])] for i in range(len(avail) - 1)] if len(avail) >= 2 else []
    allv = list(pairwise.values())
    for suffix, vals in [("early_late", el), ("adjacent_mean", adj), ("overall_mean", allv)]:
        feats.append(float(np.mean(vals)) if vals else 0.0)
        names.append(f"{prefix}_{suffix}")
    return feats, names


def _expected_names(sampled_layers: list[int], prefix: str) -> list[str]:
    names = [f"{prefix}_L{a}_L{b}" for a, b in combinations(sampled_layers, 2)]
    names += [f"{prefix}_early_late", f"{prefix}_adjacent_mean", f"{prefix}_overall_mean"]
    return names


def extract_key_cka(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    **_ignored,
) -> FeatureFamilyResult:
    """Cross-layer CKA for keys and values over sampled-layer pairs (basis-invariant)."""
    if sampled_layers is None:
        sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    features: list[float] = []
    names: list[str] = []
    for surface, seq_by_layer, prefix in [
        ("keys", data.pre_rope_keys, "kv_key_cka"),
        ("values", data.v_proj_values, "kv_value_cka"),
    ]:
        if not seq_by_layer:
            # zero-fill to keep dims stable when a surface is absent
            exp = _expected_names(sampled_layers, prefix)
            features.extend([0.0] * len(exp))
            names.extend(exp)
            continue
        f, n = _cka_for_surface(seq_by_layer, sampled_layers, prefix)
        features.extend(f)
        names.extend(n)

    if not names:
        return FeatureFamilyResult.empty("kv_cka")
    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="kv_cka",
    )
