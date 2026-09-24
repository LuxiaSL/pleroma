"""The LENGTH-ONLY null ranker — report it beside every Δnr.

Its definition is the one the banked site-norm analysis used, unchanged. The
rule it serves (`docs/FINDINGS.md §7`): *a Δnr that does not beat the
length-only ranker has not demonstrated manner steering.*

★ WHAT "LENGTH" IS. The banked number (+0.2571 on the 8B raw v1a arm, 12/13
conversations positive, vs the manner judge's +0.0848) was computed on
CHARACTER counts — the analysis calls this with `chars(...)` of both the
candidate futures and the reply. It is not a word count. `words` is kept here
only because the same analysis reported word counts beside chars. Use `chars`
to reproduce the banked number.

`tests/unit/stats/test_length_null.py` reproduces +0.2571 / +0.2053 from a
fixture extracted from the banked site-norm run.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def words(text: str) -> int:
    return len(str(text).split())


def chars(text: str) -> int:
    return len(str(text))


def length_only_rank(cand_lengths: Sequence[int], reply_length: int,
                     prescribed: int) -> float:
    """The nr a PURE LENGTH MATCHER assigns to `prescribed`.

    Candidates are ordered by |len(candidate) − len(reply)|, ties broken by
    lower fan index (deterministic, and independent of any judge). The result
    is on exactly the same 0..1 scale as `normalized_rank`, so Δnr computed
    from it is directly comparable to the judged Δnr.
    """
    k = len(cand_lengths)
    if k < 2:
        raise ValueError("a length ranking of one is not a ranking")
    order = sorted(range(k),
                   key=lambda i: (abs(cand_lengths[i] - reply_length), i))
    return order.index(prescribed) / (k - 1)


def length_only_dnr(by_conv: Mapping[str, Mapping[str, Sequence[float]]],
                    arm: str, base: str = "base") -> dict[str, object]:
    """Δnr of the length-only ranker for one arm, the way `sitenorm_analyze`'s
    `contrast_block(..., with_base=True)` computes its point estimate:

        per conversation   mean lo(base) − mean lo(arm)
        observed           unweighted mean over conversations that carry
                           both `base` and `arm`

    POSITIVE = the arm's replies sit closer IN LENGTH to their prescribed
    future than base's do. No inference here — the CI and permutation machinery
    stays with the caller's own analysis so it cannot drift.
    """
    convs = sorted(c for c, d in by_conv.items() if base in d and arm in d
                   and len(d[base]) and len(d[arm]))
    if not convs:
        raise ValueError(f"no conversation carries both {base!r} and {arm!r}")
    per = {c: sum(by_conv[c][base]) / len(by_conv[c][base])
           - sum(by_conv[c][arm]) / len(by_conv[c][arm]) for c in convs}
    return {
        "observed": sum(per.values()) / len(per),
        "per_conversation": per,
        "n_conversations": len(convs),
        "n_positive": sum(1 for v in per.values() if v > 0),
    }
