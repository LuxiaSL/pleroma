"""Shared helper functions re-exported from state_extractor.

Avoids duplicating math utilities. state_extractor.py remains the
canonical source; this module re-exports for convenience.
"""

from anamnesis.extraction.state_extractor import (
    _cosine_dist,
    _cosine_sim,
    _correct_hidden_state,
    _safe_entropy,
    _trajectory_indices,
)

__all__ = [
    "_cosine_dist",
    "_cosine_sim",
    "_correct_hidden_state",
    "_safe_entropy",
    "_trajectory_indices",
]
