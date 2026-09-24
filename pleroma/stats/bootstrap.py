"""Percentile bootstrap — the one implementation.

Its intervals are pinned by the contract in ``tests/contract/format_judge_stats``.
Separate implementations give different intervals from the same seed, so every
caller uses this one.
"""

from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np

#: The project's standing seed (the registered fold/permutation seed).
DEFAULT_SEED: int = 20260921


def boot_ci(vals: Sequence[float], n_boot: int = 20000, seed: int = DEFAULT_SEED,
            alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean. Non-finite values are dropped first; fewer
    than two finite values give (nan, nan) rather than a fake interval."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def distinct_resamples(k: int) -> int:
    """How many DISTINCT bootstrap resamples k values have: the multisets of
    size k from k items, C(2k−1, k). (k**k counts ordered draws — the error
    the spot-check report made; k=3 has 10 distinct resamples, not 27.)"""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return comb(2 * k - 1, k)
