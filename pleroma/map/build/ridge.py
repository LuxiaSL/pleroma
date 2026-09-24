"""Ridge conventions shared by the fit, the export and the wide map.

The export and the wide map both call `resolve_rank`, which mirrors the fit's
own full-rank test (`pleroma.map.build.cv.cv_sweep`), so "full rank" means one
thing everywhere and the rank banked in a map is the CONCRETE truncation: a map that banked the rank
as passed (0 for full rank) would be read by LoomMap as an empty code basis.
"""
from __future__ import annotations

def resolve_rank(rank: int, w_shape: tuple[int, ...]) -> int:
    """`--rank` as the fit's grid spells it -> a concrete SVD truncation.

    ★ `pleroma.map.build.cv.cv_sweep` encodes FULL RANK as
    ``r <= 0 or r >= min(w.shape)`` and the grid key for it is ``r0``. This
    mirrors that convention exactly, so ``--rank 0`` here and the
    ``dz|raw|lam1e3|r0`` cell of the fit report name the same model. Anything in ``1..min(shape)`` is a literal truncation;
    anything larger is full rank, again matching the fit. Raises ValueError
    when W has no usable rank.
    """
    m = int(min(w_shape))
    if m <= 0:
        raise ValueError(f"W has no usable rank: shape {tuple(w_shape)}")
    return m if int(rank) <= 0 or int(rank) >= m else int(rank)
