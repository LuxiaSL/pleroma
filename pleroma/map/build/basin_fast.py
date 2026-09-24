"""The basin check's O(N²) core (`pleroma.map.build.basin`), as O(N·D + N·P) — EXACT, not sampled.

A full N×N cosine-distance matrix over every generation costs 250 MB and a
couple of minutes at 5,728 gens, 2.4 GB and ~20 min at 17,544, and 74 GB and
hours at ~96,000: the memory may exist, but the time does not close.

The check never needs the matrix. Both quantities the matrix feeds have closed
forms over UNIT rows u_i, so the better answer than a faster matrix is not to
compute it at all:

  BETWEEN-prompt mean distance for prompt p:
      mean_{j∉p} (1 − u_i·u_j)
        = 1 − u_i · (S − S_p)/(N − k),     S = Σ_j u_j, S_p = Σ_{j∈p} u_j
    so the mean over all members of p is 1 − (S_p/k) · (S − S_p)/(N − k).
    One dot product per prompt. O(N·D) total.

  SILHOUETTE b-term (mean distance from gen i to the members of another
  prompt q, minimised over q ≠ p):
      mean_{j∈q} (1 − u_i·u_j) = 1 − u_i · C_q,   C_q = S_q/|q|
    so one (N × P) matmul against the per-prompt centroid matrix, tiled.
    O(N·P) memory at worst, and P ≈ N/seeds.

WITHIN-prompt distances stay pairwise — they are k×k with k = seeds (8 or 32).

Equivalence is not asserted, it is TESTED: `verify_against_report` recomputes
a finished basin report's per-prompt numbers and reports the max absolute
deviation. Rounding differs in the last bits only because the matrix path clips
each pairwise cosine to [-1,1] before averaging while the closed form clips
after; on real signature data that is ~1e-16.

Optional GPU: --device cuda moves the two matmuls to torch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def unit_rows(x: NDArray[np.floating]) -> F64:
    """Row-normalise; a zero row stays zero (orthogonal to everything), which
    is exactly what cosine_distance_matrix's `np.where(norms < 1e-12, 1.0,...)`
    produces."""
    xf = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(xf, axis=1, keepdims=True)
    return xf / np.where(norms < 1e-12, 1.0, norms)


def group_index(prompt_ids: list[str]) -> tuple[list[str], list[NDArray[np.int64]]]:
    """Unique prompt ids in first-appearance order + member index arrays."""
    order: list[str] = []
    seen: dict[str, list[int]] = {}
    for i, p in enumerate(prompt_ids):
        if p not in seen:
            seen[p] = []
            order.append(p)
        seen[p].append(i)
    return order, [np.asarray(seen[p], dtype=np.int64) for p in order]


def between_means(unit: F64, members: list[NDArray[np.int64]]) -> list[float | None]:
    """Mean 1−cos from each prompt's members to every gen outside that prompt."""
    n = unit.shape[0]
    total = unit.sum(axis=0)
    out: list[float | None] = []
    for idx in members:
        k = int(idx.size)
        if n - k <= 0:
            out.append(None)
            continue
        s_p = unit[idx].sum(axis=0)
        # mean over (member, other) pairs = 1 - (mean member)·(mean other)
        mean_member = s_p / k
        mean_other = (total - s_p) / (n - k)
        out.append(float(1.0 - float(np.dot(mean_member, mean_other))))
    return out


def prompt_centroids(unit: F64, members: list[NDArray[np.int64]]) -> F64:
    return np.stack([unit[idx].mean(axis=0) for idx in members])


def silhouette_scores(
    unit: F64,
    members: list[NDArray[np.int64]],
    device: str = "cpu",
    tile: int = 4096,
) -> list[float | None]:
    """Per-prompt mean silhouette, matching the matrix path's own definition:
    a = mean distance from gen i to its OWN other members (self excluded);
    b = min over other prompts q of the mean distance from i to q's members.
    """
    cents = prompt_centroids(unit, members)          # [P, D]
    n, p = unit.shape[0], cents.shape[0]  # noqa: F841 — verbatim
    own_of = np.empty(n, dtype=np.int64)
    for gi, idx in enumerate(members):
        own_of[idx] = gi

    b_min = np.empty(n, dtype=np.float64)
    if device == "cuda":
        import torch

        u_t = torch.from_numpy(unit).to("cuda")
        c_t = torch.from_numpy(cents).to("cuda")
        for start in range(0, n, tile):
            stop = min(start + tile, n)
            sims = u_t[start:stop] @ c_t.T                    # [tile, P]
            d = 1.0 - sims
            rows = torch.arange(stop - start, device="cuda")
            d[rows, torch.from_numpy(own_of[start:stop]).to("cuda")] = float("inf")
            b_min[start:stop] = d.min(dim=1).values.cpu().numpy()
    else:
        for start in range(0, n, tile):
            stop = min(start + tile, n)
            d = 1.0 - unit[start:stop] @ cents.T
            d[np.arange(stop - start), own_of[start:stop]] = np.inf
            b_min[start:stop] = d.min(axis=1)

    out: list[float | None] = []
    for idx in members:
        k = int(idx.size)
        if k < 2:
            out.append(None)
            continue
        sub = unit[idx]
        # a_i = mean over own other members of (1 - u_i·u_j)
        sims = sub @ sub.T
        np.fill_diagonal(sims, 0.0)
        a = 1.0 - sims.sum(axis=1) / (k - 1)
        b = b_min[idx]
        denom = np.maximum(a, b)
        sil = np.where(denom > 1e-12, (b - a) / np.where(denom > 1e-12, denom, 1.0), 0.0)
        out.append(float(np.mean(sil)))
    return out


def within_pairs(unit: F64, members: list[NDArray[np.int64]]) -> list[F64]:
    """The k*(k-1)/2 within-prompt distances per prompt — k is the seed count,
    so this stays pairwise on purpose (the report needs mean, std AND max)."""
    out: list[F64] = []
    for idx in members:
        k = int(idx.size)
        if k < 2:
            out.append(np.array([], dtype=np.float64))
            continue
        sub = unit[idx]
        d = 1.0 - np.clip(sub @ sub.T, -1.0, 1.0)
        out.append(d[np.triu_indices(k, k=1)])
    return out


def within_means(unit: F64, members: list[NDArray[np.int64]]) -> list[float | None]:
    return [float(np.mean(v)) if v.size else None
            for v in within_pairs(unit, members)]


def verify_against_report(features: NDArray[np.floating], prompt_ids: list[str],
                          report: dict, device: str = "cpu") -> dict[str, float]:
    """Recompute a finished basin_report's per-prompt numbers the fast way and
    return max |deviation| per field. This is the equivalence proof."""
    unit = unit_rows(features)
    order, members = group_index(prompt_ids)
    fast = {
        "within_cosine_distance_mean": dict(zip(order, within_means(unit, members))),
        "between_cosine_distance_mean": dict(zip(order, between_means(unit, members))),
        "separation_score": dict(zip(order, silhouette_scores(unit, members, device))),
    }
    dev: dict[str, float] = {}
    for field, got in fast.items():
        worst = 0.0
        for row in report["prompts"]:
            pid = str(row["prompt_id"])
            old, new = row.get(field), got.get(pid)
            if old is None or new is None:
                continue
            worst = max(worst, abs(float(old) - float(new)))
        dev[field] = worst
    return dev
