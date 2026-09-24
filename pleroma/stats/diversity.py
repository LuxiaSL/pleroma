"""Fan diversity and degeneracy — the text-side instruments.

The window-Jaccard here is the instrument behind the 8B forking result (chat
.2467 → raw .1015, `docs/FINDINGS.md §11`), and `repetition_rate` is
the cheapest loop detector we have (a loop is damage that can arrive before a
dose is audible). Jaccard variants with different tokenizers or stopword
lists give numbers that are not comparable, so this is the only one in the
package.
"""

from __future__ import annotations

import re
import statistics as st
from itertools import combinations
from typing import Sequence

WORD_RE = re.compile(r"[a-z0-9']+")
#: Words compared per future. 60 is the value the banked forking numbers used.
DEFAULT_WINDOW: int = 60


def words(text: str) -> list[str]:
    """Lowercase [a-z0-9']+ tokens (so "Don't" is one word)."""
    return WORD_RE.findall(text.lower())


def ngrams(seq: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def jaccard(a: set[str], b: set[str]) -> float:
    """Set Jaccard; two empty sets are identical (1.0)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def fan_jaccard(texts: Sequence[str], window: int = DEFAULT_WINDOW) -> float:
    """Mean pairwise Jaccard of the first `window` words of each future.

    LOW = the fan forks; HIGH = the futures are one answer in k costumes. An
    empty future is kept and scores 0 against every non-empty sibling (the
    banked instrument's behaviour — skipping it would inflate the mean).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    sets = [set(words(t)[:window]) for t in texts]
    pairs = [jaccard(a, b) for a, b in combinations(sets, 2)]
    if not pairs:
        raise ValueError("fan_jaccard needs at least two futures")
    return st.fmean(pairs)


def distinct_n(ws: Sequence[str], n: int) -> float:
    """Distinct n-grams / n-grams; NaN when the text has none."""
    gs = ngrams(ws, n)
    return len(set(gs)) / len(gs) if gs else float("nan")


def repetition_rate(ws: Sequence[str], n: int = 4) -> float:
    """Fraction of n-grams that are repeats. High = a degenerate loop."""
    gs = ngrams(ws, n)
    return 1.0 - len(set(gs)) / len(gs) if gs else float("nan")
