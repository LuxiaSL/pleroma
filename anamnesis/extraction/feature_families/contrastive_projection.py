"""Contrastive projection of raw hidden states — T3 replacement.

Replaces content-calibrated PCA (which captures format, not modes) with
a trained contrastive projection that maximizes mode separation.

Architecture: Linear(hidden_dim, 256) → ReLU → Linear(256, 32) → L2-normalize
Trained with TripletMarginLoss on hidden states from fat extraction data.

INFERENCE IS PURE NUMPY — no torch dependency at feature extraction time.
Training requires torch (see scripts/train_contrastive_projection.py).

The projection model is trained once per model scale and saved to the
calibration directory as a .npz file with numpy weight matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


class ContrastiveProjectionInference:
    """Pure-numpy MLP inference for contrastive projection.

    Architecture: Linear → ReLU → Linear → L2-normalize
    Weights loaded from .npz file saved by training script.
    """

    def __init__(
        self,
        w1: F32,
        b1: F32,
        w2: F32,
        b2: F32,
        scaler_mean: F32 | None = None,
        scaler_scale: F32 | None = None,
    ):
        self.w1 = w1  # [hidden_dim_mlp, input_dim]
        self.b1 = b1  # [hidden_dim_mlp]
        self.w2 = w2  # [bottleneck_dim, hidden_dim_mlp]
        self.b2 = b2  # [bottleneck_dim]
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.bottleneck_dim = w2.shape[0]

    @classmethod
    def load(cls, path: Path) -> "ContrastiveProjectionInference":
        """Load projection model from .npz file."""
        data = np.load(path)
        scaler_mean = data.get("scaler_mean")
        scaler_scale = data.get("scaler_scale")
        # Handle numpy scalar arrays from npz
        if scaler_mean is not None and scaler_mean.size == 0:
            scaler_mean = None
        if scaler_scale is not None and scaler_scale.size == 0:
            scaler_scale = None
        return cls(
            w1=data["w1"].astype(np.float32),
            b1=data["b1"].astype(np.float32),
            w2=data["w2"].astype(np.float32),
            b2=data["b2"].astype(np.float32),
            scaler_mean=scaler_mean.astype(np.float32) if scaler_mean is not None else None,
            scaler_scale=scaler_scale.astype(np.float32) if scaler_scale is not None else None,
        )

    def project(self, x: F32) -> F32:
        """Project a hidden state vector through the MLP.

        Parameters
        ----------
        x : array of shape [hidden_dim] or [batch, hidden_dim]

        Returns
        -------
        Array of shape [bottleneck_dim] or [batch, bottleneck_dim], L2-normalized.
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        x = x.astype(np.float64)

        # Optional standardization
        if self.scaler_mean is not None and self.scaler_scale is not None:
            scale = self.scaler_scale.astype(np.float64)
            scale = np.where(scale < 1e-12, 1.0, scale)
            x = (x - self.scaler_mean.astype(np.float64)) / scale

        # Forward: Linear → ReLU → Linear → L2-normalize
        z = x @ self.w1.astype(np.float64).T + self.b1.astype(np.float64)
        z = np.maximum(z, 0.0)  # ReLU (no dropout at inference)
        z = z @ self.w2.astype(np.float64).T + self.b2.astype(np.float64)

        # L2 normalize
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        z = z / norms

        result = z.astype(np.float32)
        return result[0] if single else result


def _correct_hidden_state(
    hidden_state: F32,
    layer_array_idx: int,
    position: int,
    positional_means: F32 | None,
) -> F32:
    """Apply positional mean correction to a hidden state.

    Parameters
    ----------
    hidden_state : array [hidden_dim]
    layer_array_idx : int
        Index into the hidden_states array (0 = embedding, 1..N = transformer layers).
    position : int
        Token position in the sequence.
    positional_means : optional array [num_layers+1, max_pos, hidden_dim]
    """
    if positional_means is None:
        return hidden_state
    if layer_array_idx >= positional_means.shape[0]:
        return hidden_state
    if position >= positional_means.shape[1]:
        return hidden_state
    return hidden_state - positional_means[layer_array_idx, position]


def extract_contrastive_projection(
    data: RawGenerationData,
    projection_model_path: Path,
    layer_indices: list[int] | None = None,
    temporal_samples: int = 5,
    apply_positional_correction: bool = True,
) -> FeatureFamilyResult:
    """Project hidden states through trained contrastive model.

    At each (layer, temporal_sample) pair, extracts the hidden state,
    optionally applies positional mean correction, and projects through
    the trained MLP to produce a bottleneck_dim-dimensional embedding.

    Parameters
    ----------
    data : RawGenerationData
        Per-token tensors from a single generation.
    projection_model_path : Path
        Path to trained projection model (.npz).
    layer_indices : list[int], optional
        Transformer layer indices for projection. Defaults to [8, 16, 20, 24, 28].
    temporal_samples : int
        Number of evenly-spaced time points to sample.
    apply_positional_correction : bool
        Whether to subtract positional means before projection.

    Returns
    -------
    FeatureFamilyResult with shape [n_layers * temporal_samples * bottleneck_dim].
    """
    if layer_indices is None:
        layer_indices = [8, 16, 20, 24, 28]

    # Load model
    try:
        model = ContrastiveProjectionInference.load(projection_model_path)
    except Exception as e:
        logger.error(f"Failed to load contrastive projection model: {e}")
        return FeatureFamilyResult.empty("contrastive_projection")

    T = len(data.hidden_states)
    bottleneck_dim = model.bottleneck_dim

    features: list[float] = []
    names: list[str] = []

    if T < 2:
        # Return zeros with correct shape
        for l_idx in layer_indices:
            for ts in range(temporal_samples):
                for d in range(bottleneck_dim):
                    features.append(0.0)
                    names.append(f"cp_L{l_idx}_t{ts}_d{d}")
        return FeatureFamilyResult(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            family_name="contrastive_projection",
        )

    # Compute temporal sample indices (evenly spaced)
    if temporal_samples == 1:
        t_indices = [0]
    else:
        t_indices = [int(round(i * (T - 1) / (temporal_samples - 1)))
                     for i in range(temporal_samples)]

    for l_idx in layer_indices:
        # hidden_states uses array index = layer_idx + 1 (0 = embedding)
        arr_idx = l_idx + 1

        for ts_i, t in enumerate(t_indices):
            if t >= T:
                # Pad with zeros
                for d in range(bottleneck_dim):
                    features.append(0.0)
                    names.append(f"cp_L{l_idx}_t{ts_i}_d{d}")
                continue

            h = data.hidden_states[t][arr_idx].copy()

            # Positional correction
            if apply_positional_correction:
                position = data.prompt_length + t
                h = _correct_hidden_state(
                    h, arr_idx, position, data.positional_means,
                )

            # Project through MLP
            embedding = model.project(h)  # [bottleneck_dim]

            for d in range(bottleneck_dim):
                features.append(float(embedding[d]))
                names.append(f"cp_L{l_idx}_t{ts_i}_d{d}")

    result = FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="contrastive_projection",
    )
    logger.info(
        f"Contrastive projection: {len(result)} features "
        f"({len(layer_indices)} layers × {temporal_samples} times × {bottleneck_dim} dims)"
    )
    return result
