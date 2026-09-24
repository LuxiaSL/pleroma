"""The ruler: renormalise each site row of a lever to the map's `norm_ref`.

The ruler is what makes alpha mean one absolute per-site norm across maps and
lever kinds: two maps of one model can have `norm_ref` differing by 2.1×, so
an alpha without its ruler is no dose at all (docs/FINDINGS.md §2).
Every wearable lever goes through this one function because separate copies
disagree exactly where it matters: on a row of norm 1e-10 (scale it up to full
loudness, zero it, or refuse it?) and on a NaN row (which would otherwise
become a silent NaN lever).

Policy: refuse a non-finite row, and refuse a row at or below `MIN_ROW_NORM` —
norm-matching that row would be amplifying noise to full loudness and calling
it a direction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: At or below this, a site row is noise, not a direction.
MIN_ROW_NORM: float = 1e-9


def apply_ruler(lever: np.ndarray, norm_ref: Sequence[float] | np.ndarray, *,
                sites: Sequence[int] | None = None,
                what: str = "lever") -> tuple[np.ndarray, list[float]]:
    """Scale each row `lever[s]` to norm `norm_ref[s]`.

    Returns (the renormalised lever in the input's dtype, the RAW per-site
    norms before matching — the lever's own loudness, kept for the record and
    for `predicted` dosing). `sites` only labels error messages (layer ids);
    without it the row index is used.
    """
    arr = np.asarray(lever)
    ref = np.asarray(norm_ref, dtype=np.float64).ravel()
    if arr.ndim != 2:
        raise ValueError(f"{what} must be [sites, hidden], got shape {arr.shape}")
    if arr.shape[0] != ref.size:
        raise ValueError(f"{what} has {arr.shape[0]} site rows but norm_ref has "
                         f"{ref.size} — refusing to norm-match across a site mismatch")
    out = np.empty_like(arr)
    raw: list[float] = []
    for s in range(arr.shape[0]):
        label = f"site {sites[s]}" if sites is not None else f"site index {s}"
        n = float(np.linalg.norm(arr[s]))
        if not np.isfinite(n):
            raise ValueError(f"{what} is non-finite at {label}")
        if n <= MIN_ROW_NORM:
            raise ValueError(
                f"{what} has a ~zero lever row at {label} (norm {n:.2e}) — "
                "refusing to norm-match noise up to full loudness")
        raw.append(n)
        out[s] = arr[s] * (ref[s] / n)
    return out, raw


def site_norms(vectors: np.ndarray) -> list[float]:
    """Per-site L2 norms of a ``[n_sites, hidden_dim]`` injection (float64 math).

    What the dose lanes record as ``injected_norms`` — the exact L2s that hit
    the stream. Raises ValueError unless ``vectors`` is 2-D.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected [n_sites, hidden_dim], got {arr.shape}")
    return [float(np.linalg.norm(arr[s])) for s in range(arr.shape[0])]
