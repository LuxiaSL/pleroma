"""Fan-relative steering statistics: does wearing member j pull toward future j
more than toward j's fan-mates?

Every readout — a judge's resemblance ranking, a signature gauge, a
teacher-forced likelihood shift, a length-only ranker — reduces, per fan, to a
matrix ``NR[j, i]``: the NORMALIZED RANK of future ``i`` under the readout of
wear ``j`` (0 = most similar / most favoured, 1 = least; ``/(k-1)``). Rows are
the worn member slots, columns the fan's k futures. Averages over reps are
fine: nr is linear.

    Δnr(fan) = 0.5 − mean_j NR[j, j]

Positive = the readout of wear j sits closer to future j than chance. In a
BALANCED design (every slot worn equally), under "a lever does not pull toward
its own future" the member labels are exchangeable within a fan, so the null
permutes WHICH future each wear is prescribed to, independently per fan:

    Δnr_π(fan) = 0.5 − mean_j NR[j, π(j)]

and the pooled statistic is the mean over fans (spot70b_analyze's test,
generalized). With a base (unworn) readout ``NR0[i]`` the PAIRED form
``mean_j (NR0[j] − NR[j, j])`` removes each future's standing resemblance to
the model's default; it is permuted the same way.

Missing cells are NaN and are skipped (a slot the judge failed on drops out of
that fan's mean, observed and permuted alike).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

N_PERMS = 20000
SEED = 20260924


def normalized_ranks(scores: Sequence[float], *, higher_is_closer: bool = True) -> NDArray[np.float64]:
    """nr of every candidate from a score vector (ties share their mean rank).

    ``higher_is_closer`` — a likelihood gain or a similarity; pass False for a
    distance.
    """
    s = np.asarray(scores, dtype=np.float64)
    k = s.size
    if k < 2:
        raise ValueError(f"need at least 2 candidates, got {k}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    key = -s if higher_is_closer else s
    order = np.argsort(key, kind="stable")
    ranks = np.empty(k, dtype=np.float64)
    ranks[order] = np.arange(k, dtype=np.float64)
    for v in np.unique(key):  # average ties
        m = key == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks / (k - 1)


def nr_from_ranking(ranking: Sequence[int], k: int) -> NDArray[np.float64]:
    """nr of every candidate from a full ranking (candidate indices, most similar first)."""
    r = [int(x) for x in ranking]
    if sorted(r) != list(range(k)):
        raise ValueError(f"ranking {r} is not a permutation of range({k})")
    out = np.empty(k, dtype=np.float64)
    for pos, idx in enumerate(r):
        out[idx] = pos / (k - 1)
    return out


@dataclass(frozen=True)
class FanRankResult:
    """One readout's pooled fan-relative test."""

    delta_nr: float
    p_one_sided: float
    n_fans: int
    per_fan: dict[str, float]
    ci95: tuple[float, float]
    n_perms: int
    paired: bool
    null_mean: float
    null_sd: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"delta_nr": round(self.delta_nr, 4), "p_one_sided": self.p_one_sided,
                "n_fans": self.n_fans, "ci95": [round(x, 4) for x in self.ci95],
                "null_mean": round(self.null_mean, 4), "null_sd": round(self.null_sd, 4),
                "n_perms": self.n_perms, "paired": self.paired,
                "per_fan": {k: round(v, 4) for k, v in self.per_fan.items()},
                "notes": self.notes}


def _fan_stat(nr: NDArray[np.float64], perm: NDArray[np.int64],
              base: NDArray[np.float64] | None) -> float:
    rows = np.arange(nr.shape[0])
    vals = nr[rows, perm]
    if base is None:
        return float(0.5 - np.nanmean(vals))
    return float(np.nanmean(base[perm] - vals))


def fan_rank_test(nr_by_fan: Mapping[str, NDArray[np.floating]], *,
                  base_by_fan: Mapping[str, NDArray[np.floating]] | None = None,
                  n_perms: int = N_PERMS, seed: int = SEED) -> FanRankResult:
    """Pooled Δnr over fans with the within-fan label-permutation null.

    ``nr_by_fan[fan]`` is ``[k_worn, k]`` with worn slot j prescribed to
    future j (so ``k_worn <= k`` and row j's target is column j).
    ``base_by_fan[fan]`` (optional, length k) makes the test paired.
    """
    fans = sorted(nr_by_fan)
    if not fans:
        raise ValueError("no fans")
    mats, bases = [], []
    for f in fans:
        m = np.asarray(nr_by_fan[f], dtype=np.float64)
        if m.ndim != 2 or m.shape[0] > m.shape[1] or m.shape[0] < 2:
            raise ValueError(f"fan {f}: NR must be [k_worn>=2, k>=k_worn], got {m.shape}")
        if np.all(np.isnan(np.diag(m[:, : m.shape[0]]))):
            raise ValueError(f"fan {f}: no prescribed cell is observed")
        mats.append(m)
        if base_by_fan is not None:
            b = np.asarray(base_by_fan[f], dtype=np.float64)
            if b.shape != (m.shape[1],):
                raise ValueError(f"fan {f}: base must be length {m.shape[1]}, got {b.shape}")
            bases.append(b)
        else:
            bases.append(None)
    ident = [np.arange(m.shape[0]) for m in mats]
    per_fan = {f: _fan_stat(m, i, b) for f, m, i, b in zip(fans, mats, ident, bases, strict=True)}
    obs = float(np.mean(list(per_fan.values())))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for t in range(n_perms):
        acc = 0.0
        for m, b in zip(mats, bases, strict=True):
            # a random prescription of the worn slots onto distinct futures
            perm = rng.permutation(m.shape[1])[: m.shape[0]]
            acc += _fan_stat(m, perm, b)
        null[t] = acc / len(mats)
    p = float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perms))

    # fan-bootstrap CI of the pooled statistic
    vals = np.array(list(per_fan.values()))
    boots = rng.choice(vals, size=(4000, vals.size), replace=True).mean(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return FanRankResult(delta_nr=obs, p_one_sided=p, n_fans=len(fans), per_fan=per_fan,
                         ci95=ci, n_perms=n_perms, paired=base_by_fan is not None,
                         null_mean=float(null.mean()), null_sd=float(null.std()))


def paired_fan_contrast(a: Mapping[str, float], b: Mapping[str, float], *,
                        n_perms: int = N_PERMS, seed: int = SEED) -> dict[str, object]:
    """mean over shared fans of (a − b), one-sided sign-flip test (H1: a > b).

    Used for judged Δnr vs the length-only ranker's Δnr on the SAME replies:
    a judged effect that does not beat length has not shown manner steering.
    """
    fans = sorted(set(a) & set(b))
    if len(fans) < 2:
        raise ValueError("need at least 2 shared fans")
    d = np.array([float(a[f]) - float(b[f]) for f in fans])
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perms, d.size))
    null = (signs * d).mean(axis=1)
    p = float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perms))
    return {"contrast": round(obs, 4), "p_one_sided": p, "n_fans": len(fans),
            "positive_fans": int(np.sum(d > 0))}


def length_matched_test(rankings_by_fan: Mapping[str, Mapping[int, Sequence[int]]],
                        lengths_by_fan: Mapping[str, Sequence[float]], *, tol: float,
                        min_candidates: int = 3, n_perms: int = N_PERMS,
                        seed: int = SEED) -> dict[str, object]:
    """Δnr with length held (nearly) equal: the target is ranked only among the
    fan's futures whose length is within ``±tol`` (relative) of its own.

    ``rankings_by_fan[fan][j]`` is a readout's full ranking (future indices,
    most similar first) under wear j; ``lengths_by_fan[fan]`` the k futures'
    lengths. A wear whose target has fewer than ``min_candidates`` length
    peers (itself included) is skipped. The null permutes which future each
    wear is prescribed to within a fan, and re-derives the candidate set for
    the permuted target, so it is the same length-matched question asked of a
    wrong target.

    Matching is length-proof only when reply length tracks the target more
    loosely than ``tol``: a readout that knows nothing but length still ranks
    an exact-length target first among its peers. Run the length-only ranker
    through this same test as the control; the judged effect is read beside it.
    """
    if not 0 < tol < 1:
        raise ValueError(f"tol must be in (0, 1), got {tol}")
    fans = sorted(rankings_by_fan)
    lens = {f: np.asarray(lengths_by_fan[f], dtype=np.float64) for f in fans}

    def stat(assign: Mapping[str, Mapping[int, int]]) -> tuple[float, int]:
        vals = []
        for f in fans:
            length = lens[f]
            per = []
            for j, ranking in rankings_by_fan[f].items():
                t = assign[f][j]
                cand = {i for i in range(length.size)
                        if abs(length[i] - length[t]) <= tol * length[t]}
                if len(cand) < min_candidates:
                    continue
                order = [i for i in ranking if i in cand]
                per.append(order.index(t) / (len(cand) - 1))
            if per:
                vals.append(0.5 - float(np.mean(per)))
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    ident = {f: {int(j): int(j) for j in rankings_by_fan[f]} for f in fans}
    obs, n = stat(ident)
    if n < 2:
        raise ValueError(f"only {n} fans have a length-matched target at tol {tol}")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perms)
    for t in range(n_perms):
        assign = {}
        for f in fans:
            js = sorted(rankings_by_fan[f])
            perm = rng.permutation(lens[f].size)[: len(js)]
            assign[f] = {int(j): int(p) for j, p in zip(js, perm, strict=True)}
        null[t] = stat(assign)[0]
    null = null[np.isfinite(null)]
    p = float((1 + np.sum(null >= obs - 1e-12)) / (1 + null.size))
    return {"delta_nr": round(obs, 4), "p_one_sided": p, "n_fans": n, "tol": tol,
            "null_mean": round(float(null.mean()), 4), "n_perms": int(null.size)}
