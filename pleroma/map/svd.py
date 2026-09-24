"""Canonical SVD orientation: a code must mean the same lever on every stack.

An SVD is unique only up to one sign per component: (u_k, v_k) and
(-u_k, -v_k) factor the same W. LAPACK/cuSOLVER builds pick different signs,
and that matters in practice: a 70B map reproduced on a second stack (W equal
to 2.6e-12) came back with 32 of 64 Vt rows flipped. A code is its
coordinates in span(Vt), so a banked code would expand to a different lever. Every exported map is therefore put in one deterministic orientation
before it is written.

Convention (``SIGN_CONVENTION``): in each Vt row, the entry of largest
magnitude is positive. It is read from Vt because Vt carries the code basis.
A row whose top two magnitudes are too close to call, and have opposite
signs, is ambiguous: noise could pick either. Such rows are reported, not
hidden.

Signs are not the only freedom. Where two singular values are (near-)equal,
any rotation inside their shared subspace is an equally valid SVD, and no sign
rule can fix that. `canonical_signs` reports the smallest relative spectral
gap so a caller can see when this applies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGN_CONVENTION: str = "vt_row_maxabs_positive"

#: Relative margin below which a row's top two |entries| are treated as tied.
AMBIGUITY_MARGIN: float = 1e-6


@dataclass(frozen=True)
class SignReport:
    """What canonicalisation did, for the export's meta block."""

    convention: str
    flipped: tuple[int, ...]
    ambiguous: tuple[int, ...]
    min_relative_gap: float | None

    def as_meta(self) -> dict[str, object]:
        return {
            "convention": self.convention,
            "n_flipped": len(self.flipped),
            "ambiguous_components": list(self.ambiguous),
            "min_relative_spectral_gap": self.min_relative_gap,
        }


def canonical_signs(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SignReport]:
    """Return (U, S, Vt) with each Vt row's largest-|.| entry positive.

    The product ``(U * S) @ Vt`` is unchanged: each component is flipped on
    both sides. Inputs are not modified in place.
    """
    u = np.asarray(u)
    s = np.asarray(s)
    vt = np.asarray(vt)
    if u.ndim != 2 or vt.ndim != 2 or s.ndim != 1:
        raise ValueError(f"bad factor shapes U{u.shape} S{s.shape} Vt{vt.shape}")
    r = s.size
    if u.shape[1] != r or vt.shape[0] != r:
        raise ValueError(f"factors disagree on rank: U{u.shape} S{s.shape} Vt{vt.shape}")
    if not (np.all(np.isfinite(vt)) and np.all(np.isfinite(u))):
        raise ValueError("non-finite SVD factors: refusing to orient them")

    flips = np.ones(r, dtype=vt.dtype)
    ambiguous: list[int] = []
    mag = np.abs(vt)
    for k in range(r):
        if vt.shape[1] == 0:
            break
        order = np.argsort(mag[k])[::-1]
        top = int(order[0])
        if vt[k, top] < 0:
            flips[k] = -1
        if vt.shape[1] > 1:
            second = int(order[1])
            a, b = mag[k, top], mag[k, second]
            if (a - b) <= AMBIGUITY_MARGIN * max(a, 1e-300) and \
                    np.sign(vt[k, top]) != np.sign(vt[k, second]):
                ambiguous.append(k)

    gap: float | None = None
    if r > 1:
        s64 = np.asarray(s, dtype=np.float64)
        denom = np.maximum(np.abs(s64[:-1]), 1e-300)
        gap = float(np.min(np.abs(s64[:-1] - s64[1:]) / denom))

    report = SignReport(
        convention=SIGN_CONVENTION,
        flipped=tuple(int(k) for k in np.flatnonzero(flips < 0)),
        ambiguous=tuple(ambiguous),
        min_relative_gap=gap,
    )
    return u * flips[None, :], s.copy(), vt * flips[:, None], report


def is_canonical(vt: np.ndarray) -> bool:
    """True when every Vt row's largest-|.| entry is non-negative."""
    vt = np.asarray(vt)
    if vt.ndim != 2 or vt.shape[1] == 0:
        return True
    idx = np.argmax(np.abs(vt), axis=1)
    return bool(np.all(vt[np.arange(vt.shape[0]), idx] >= 0))
