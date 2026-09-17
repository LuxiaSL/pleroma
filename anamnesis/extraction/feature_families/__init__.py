"""Pluggable feature families for GPU-free feature engineering.

Each module exports functions that accept RawGenerationData and return
FeatureFamilyResult. The feature_pipeline orchestrates calling enabled
families and concatenating results.

Modules:
    operators        — Reusable temporal operators (windowing, STFT)
    residual_stream  — Trajectory features + contrastive projection
    attention_flow   — System prompt tracking, region decomposition
    gate_features    — SwiGLU gate activation statistics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]


@dataclass
class FeatureFamilyResult:
    """Output from a pluggable feature family.

    Same contract as extract_tier*() in state_extractor.py:
    a flat feature vector paired with names.
    """

    features: F32  # flat feature vector
    feature_names: list[str]  # one name per dimension
    family_name: str  # e.g. "residual_trajectory", "attention_flow"

    def __len__(self) -> int:
        return len(self.features)

    @staticmethod
    def empty(family_name: str) -> "FeatureFamilyResult":
        """Return an empty result (e.g., when data is missing)."""
        return FeatureFamilyResult(
            features=np.array([], dtype=np.float32),
            feature_names=[],
            family_name=family_name,
        )
