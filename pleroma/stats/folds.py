"""Grouped cross-validation folds — the registered scheme.

Folds are assigned per PROMPT by a seeded permutation, so every member of a
fan (and both waves of a prompt) lands in one fold. This is the construction
the registered v1a fit (`pleroma.map.build.cv`) uses; a sorted-mod assignment
gives different folds and would not reproduce its held-out numbers.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

FOLDS: int = 5
FOLD_SEED: int = 20260921


def make_folds(prompt_arr: np.ndarray, fans: Sequence[np.ndarray] = (), *,
               n_folds: int = FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold index per row. Raises if any fan (row-index array) straddles folds."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    prompts: list[Hashable] = list(np.asarray(prompt_arr).tolist())
    uniq = sorted(set(prompts))
    if len(uniq) < n_folds:
        raise ValueError(f"{len(uniq)} prompts cannot fill {n_folds} folds")
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_prompt = {uniq[int(j)]: int(i % n_folds) for i, j in enumerate(order)}
    fold_idx = np.asarray([fold_of_prompt[p] for p in prompts], dtype=int)
    for sel in fans:
        if len(set(fold_idx[sel].tolist())) != 1:
            raise ValueError("a fan is split across folds — fold construction drifted")
    return fold_idx
