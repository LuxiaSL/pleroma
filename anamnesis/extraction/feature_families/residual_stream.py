"""Residual stream feature families.

Two sub-components:
    extract_residual_trajectory  — Geometric trajectory features (no learned params)
    extract_contrastive_projection — Supervised projection (requires trained model)
    train_contrastive_projection   — One-time training for contrastive model

Trajectory features capture HOW the representation moves through state space
during generation — velocity, curvature, directness. These are the "dynamical
motifs" that neuroscience literature identifies as diagnostic of computational
mode (Russo et al., Driscoll et al.).

Contrastive projection replaces PCA T3 with a mode-supervised projection that
finds directions maximizing mode separation in the residual stream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.feature_families._helpers import (
    _correct_hidden_state,
    _cosine_sim,
)
from anamnesis.extraction.feature_families.operators import apply_operators
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def extract_residual_trajectory(
    data: RawGenerationData,
    layer_indices: list[int] | None = None,
    positional_means: F32 | None = None,
    n_windows: int = 4,
    include_stft: bool = True,
) -> FeatureFamilyResult:
    """Compute geometric trajectory features from residual stream dynamics.

    Per sampled layer, from consecutive position-corrected hidden states h[t]:

    Raw features (~8 per layer):
        velocity_norm_mean/std    — ||h[t+1] - h[t]|| statistics
        direction_change_mean/std — angle between consecutive velocity vectors
        displacement_path_ratio   — ||h[T] - h[0]|| / sum(||v[t]||)
        acceleration_norm_mean/std — ||v[t+1] - v[t]|| statistics

    Temporal operator features (~36 per layer):
        apply_operators on velocity_norm time series (~18)
        apply_operators on direction_change time series (~18)

    Total: ~44 features per layer.

    Parameters
    ----------
    data : RawGenerationData
        Per-token tensors from a single generation.
    layer_indices : list[int], optional
        Transformer layer indices to compute trajectory for.
        Defaults to [8, 16, 20, 24, 28] (8B pca_layers).
    positional_means : array, optional
        For positional correction of hidden states.
    n_windows : int
        Number of windows for temporal decomposition.
    include_stft : bool
        Whether to include STFT features.
    """
    if layer_indices is None:
        layer_indices = [8, 16, 20, 24, 28]

    T = len(data.hidden_states)
    features: list[float] = []
    names: list[str] = []

    if T < 3:
        # Need at least 3 steps for velocity + direction change. Emit EXACTLY the name-set's
        # count of zeros per layer — the hardcoded n_raw+n_temporal (43) disagreed with the
        # actual _trajectory_feature_names count (41), so this path produced +2/layer features
        # with no names → features/names length divergence on short gens (caught 2026-07-18 on
        # M6 cold-temp pure_contrastive; latent for any model with T<3 gens).
        for l_idx in layer_indices:
            layer_names = _trajectory_feature_names(l_idx, n_windows, include_stft)
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name="residual_trajectory",
        )

    for l_idx in layer_indices:
        layer_features, layer_names = _compute_layer_trajectory(
            data, l_idx, positional_means, n_windows, include_stft,
        )
        features.extend(layer_features)
        names.extend(layer_names)

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="residual_trajectory",
    )


def _compute_layer_trajectory(
    data: RawGenerationData,
    layer_idx: int,
    positional_means: F32 | None,
    n_windows: int,
    include_stft: bool,
) -> tuple[list[float], list[str]]:
    """Compute trajectory features for a single layer."""
    T = len(data.hidden_states)
    prefix = f"res_traj_L{layer_idx}"
    arr_idx = layer_idx + 1  # hidden_states offset (0 = embedding)

    features: list[float] = []
    names: list[str] = []

    # Check that this layer has actual data (not zero-filled from raw_saver)
    sample_norm = float(np.linalg.norm(data.hidden_states[0][arr_idx]))
    if sample_norm < 1e-6:
        # Layer not saved in raw tensors — return zeros
        logger.debug(f"Layer {layer_idx} appears empty, returning zero trajectory features")
        n_expected = len(_trajectory_feature_names(layer_idx, n_windows, include_stft))
        return [0.0] * n_expected, _trajectory_feature_names(layer_idx, n_windows, include_stft)

    # Collect position-corrected hidden states
    h_corrected: list[F32] = []
    for t in range(T):
        abs_pos = data.prompt_length + t
        h = data.hidden_states[t][arr_idx]
        h_c = _correct_hidden_state(h, arr_idx, abs_pos, positional_means)
        h_corrected.append(h_c.astype(np.float64))

    # ── Velocity: v[t] = h[t+1] - h[t] ──
    velocities = [h_corrected[t + 1] - h_corrected[t] for t in range(T - 1)]
    velocity_norms = np.array([float(np.linalg.norm(v)) for v in velocities], dtype=np.float64)

    features.append(float(velocity_norms.mean()))
    names.append(f"{prefix}_velocity_norm_mean")
    features.append(float(velocity_norms.std()))
    names.append(f"{prefix}_velocity_norm_std")

    # ── Direction change: angle between consecutive velocity vectors ──
    direction_changes: list[float] = []
    for t in range(len(velocities) - 1):
        cos_sim = _cosine_sim(
            velocities[t].astype(np.float32),
            velocities[t + 1].astype(np.float32),
        )
        # Clamp for numerical safety
        cos_sim = max(-1.0, min(1.0, cos_sim))
        angle = float(np.arccos(cos_sim))  # radians, [0, pi]
        direction_changes.append(angle)

    dc_arr = np.array(direction_changes, dtype=np.float64) if direction_changes else np.zeros(1)
    features.append(float(dc_arr.mean()))
    names.append(f"{prefix}_direction_change_mean")
    features.append(float(dc_arr.std()))
    names.append(f"{prefix}_direction_change_std")

    # ── Displacement / path-length ratio (trajectory directness) ──
    displacement = float(np.linalg.norm(h_corrected[-1] - h_corrected[0]))
    path_length = float(velocity_norms.sum())
    directness = displacement / max(path_length, 1e-12)
    features.append(directness)
    names.append(f"{prefix}_directness")

    # ── Acceleration: a[t] = v[t+1] - v[t] ──
    if len(velocities) >= 2:
        accelerations = [velocities[t + 1] - velocities[t] for t in range(len(velocities) - 1)]
        accel_norms = np.array([float(np.linalg.norm(a)) for a in accelerations], dtype=np.float64)
        features.append(float(accel_norms.mean()))
        features.append(float(accel_norms.std()))
    else:
        features.extend([0.0, 0.0])
    names.append(f"{prefix}_acceleration_norm_mean")
    names.append(f"{prefix}_acceleration_norm_std")

    # ── Temporal operators on velocity_norm time series ──
    vn_ts = velocity_norms.astype(np.float32)
    op_f, op_n = apply_operators(
        vn_ts, prefix=f"{prefix}_velocity_norm",
        n_windows=n_windows, include_stft=include_stft,
    )
    features.extend(op_f.tolist())
    names.extend(op_n)

    # ── Temporal operators on direction_change time series ──
    dc_ts = dc_arr.astype(np.float32)
    op_f, op_n = apply_operators(
        dc_ts, prefix=f"{prefix}_direction_change",
        n_windows=n_windows, include_stft=include_stft,
    )
    features.extend(op_f.tolist())
    names.extend(op_n)

    return features, names


def _trajectory_feature_names(
    layer_idx: int,
    n_windows: int,
    include_stft: bool,
) -> list[str]:
    """Generate feature names for a single layer's trajectory features.

    Used for consistent zero-vector sizing when T is too short.
    """
    prefix = f"res_traj_L{layer_idx}"
    names = [
        f"{prefix}_velocity_norm_mean",
        f"{prefix}_velocity_norm_std",
        f"{prefix}_direction_change_mean",
        f"{prefix}_direction_change_std",
        f"{prefix}_directness",
        f"{prefix}_acceleration_norm_mean",
        f"{prefix}_acceleration_norm_std",
    ]

    # Temporal operator names for velocity_norm
    for wi in range(n_windows):
        for stat in ["mean", "std", "slope"]:
            names.append(f"{prefix}_velocity_norm_w{wi}_{stat}")
    if include_stft:
        for feat in ["spectral_centroid", "bandwidth",
                      "low_band_energy", "mid_band_energy", "high_band_energy"]:
            names.append(f"{prefix}_velocity_norm_{feat}")

    # Temporal operator names for direction_change
    for wi in range(n_windows):
        for stat in ["mean", "std", "slope"]:
            names.append(f"{prefix}_direction_change_w{wi}_{stat}")
    if include_stft:
        for feat in ["spectral_centroid", "bandwidth",
                      "low_band_energy", "mid_band_energy", "high_band_energy"]:
            names.append(f"{prefix}_direction_change_{feat}")

    return names
