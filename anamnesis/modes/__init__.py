"""Mode definitions for extraction experiments.

Provides configurable mode sets for different experimental conditions:
  - run4: Original 5 modes (linear, analogical, socratic, contrastive, dialectical)
  - extended: Run 4 + adapted Run 3 modes (structured, compressed, associative) with format control
  - run3_original: Verbatim non-format-controlled Phase-0 Run-2 (paper) / Run-3 (local)
    process modes (associative, compressed, deliberative, pedagogical, structured);
    used for cross-format transfer experiments
  - prompt_swap: Prompt-swap pairs for confound testing
"""

from __future__ import annotations

from .extended_modes import EXTENDED_MODES, EXTENDED_MODE_INDEX
from .prompt_swap import PROMPT_SWAP_PAIRS, PromptSwapPair
from .run3_original_modes import (
    RUN3_ORIGINAL_MODE_INDEX,
    RUN3_ORIGINAL_MODES,
    RUN3_ORIGINAL_TOPICS,
)
from .run4_modes import RUN4_MODE_INDEX, RUN4_MODES

__all__ = [
    "RUN4_MODES",
    "RUN4_MODE_INDEX",
    "EXTENDED_MODES",
    "EXTENDED_MODE_INDEX",
    "RUN3_ORIGINAL_MODES",
    "RUN3_ORIGINAL_MODE_INDEX",
    "RUN3_ORIGINAL_TOPICS",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
]
