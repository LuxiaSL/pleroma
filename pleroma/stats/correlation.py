"""Partial correlation and residualization — the length control for spread measures.

WHY THIS EXISTS
---------------
``fpcd`` (the judge's fraction-of-pairs-called-different) and within-fan
length CV correlate at rho ~ +0.91 on the 3B outcome samples, and the
headline rho(fpcd, dnr) = +0.833 (spread predicts steerability,
`docs/FINDINGS.md §11`) falls to +0.509 (n=9, ns) once length CV is
partialled out. An edge like that stays unmeasured unless a length
residualization exists to compute it; this module is that residualization.

It is deliberately additive: nothing here changes an existing published
number. The intended use is that every place a spread measure is correlated
against a judged outcome ALSO reports the partial against length, so a future
reader never has to ask whether the control was run.

WHAT IT IS NOT
--------------
It does not decide *whether* a control belongs. Length is sometimes the outcome
under study rather than a nuisance — a framing comparison that measures how
long replies run treats LENGTH AS A MEASURED OUTCOME, not a control.
Residualizing there would destroy the thing being measured. The caller
chooses; this module only does the arithmetic, and does it with honest
degrees of freedom.

DEGREES OF FREEDOM
------------------
A first-order partial correlation costs one degree of freedom, so the t-test
uses ``df = n - 3``, not ``n - 2``; a k-th order partial uses
``df = n - 2 - k``. At the n=9 to n=12 that the outcome samples actually
have, this is not a rounding detail: it is most of the difference between
p=.14 and p=.20.

COLLINEARITY
------------
At rho(x, z) ~ 0.91 a partial cannot attribute causation in either direction —
neither variable survives controlling for the other, and that is a statement
about the data, not about either variable. :class:`PartialResult` therefore
carries ``collinearity`` and ``warning`` so a caller cannot report the partial
without the number that says how much to trust it. The project's own trap
applies: n ~ 10 is not enough to call a null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

try:  # scipy is present everywhere the analysis stages run, but keep the
    # import failure legible rather than an ImportError from a helper.
    from scipy import stats as _stats
except ImportError as exc:  # pragma: no cover - environment failure
    raise ImportError(
        "pleroma.stats.correlation needs scipy (scipy.stats.rankdata, "
        "spearmanr, t). Install it in the active venv — never with a bare pip."
    ) from exc

__all__ = [
    "MIN_N_FOR_PARTIAL",
    "HIGH_COLLINEARITY",
    "CorrelationResult",
    "PartialResult",
    "CovariateReport",
    "spearman",
    "residualize",
    "partial_spearman",
    "partial_pearson",
    "covariate_report",
    "length_covariate_block",
]

#: Below this many finite triples a partial is not reported at all. A first-order
#: partial needs df = n - 3 > 0 for a t-test to exist; 5 is the smallest n at
#: which the statistic is not simply degenerate, so outcome sets with n < 5 are
#: skipped.
MIN_N_FOR_PARTIAL = 5

#: |rho(x, z)| at or above which the partial is flagged as uninterpretable for
#: attribution. The 3B outcome samples sit at 0.90-0.92, where neither
#: variable survives controlling for the other.
HIGH_COLLINEARITY = 0.85


def _as_float_array(v: Iterable[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _align_finite(
    *arrays: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], int]:
    """Drop any index where ANY input is non-finite. Returns (arrays, n_dropped).

    Pairwise deletion would give the three correlations of a partial three
    different samples, which is how a partial silently stops being a partial.
    """
    n = arrays[0].size
    for a in arrays[1:]:
        if a.size != n:
            raise ValueError(
                f"length mismatch: {[int(a.size) for a in arrays]} — a partial "
                "needs one row per observation across every variable"
            )
    keep = np.ones(n, dtype=bool)
    for a in arrays:
        keep &= np.isfinite(a)
    return [a[keep] for a in arrays], int(n - int(keep.sum()))


@dataclass(frozen=True)
class CorrelationResult:
    """A plain Spearman, with the sample it was actually computed on."""

    rho: float
    p: float
    n: int
    n_dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartialResult:
    """rho(x, y | z), with everything needed to judge whether to believe it.

    ``collinearity`` is |rho(x, z)|. ``warning`` is non-None whenever the
    partial should not be read as attribution — either because the sample is
    too small or because x and z are too close to distinguish.
    """

    rho: float
    p: float
    n: int
    df: int
    order: int
    rho_xy: float
    rho_xz: float
    rho_yz: float
    collinearity: float
    n_dropped: int = 0
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CovariateReport:
    """One measure vs one outcome, with and without a covariate controlled.

    This is the shape meant to be banked next to every spread-vs-outcome number:
    the headline, the covariate's own edge, and the partial, together, so the
    reader never has to reconstruct which of the three was run.
    """

    measure: str
    outcome: str
    covariate: str
    raw: CorrelationResult
    covariate_vs_outcome: CorrelationResult
    measure_vs_covariate: CorrelationResult
    partial: PartialResult | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "outcome": self.outcome,
            "covariate": self.covariate,
            "raw": self.raw.to_dict(),
            "covariate_vs_outcome": self.covariate_vs_outcome.to_dict(),
            "measure_vs_covariate": self.measure_vs_covariate.to_dict(),
            "partial": self.partial.to_dict() if self.partial else None,
            "note": self.note,
        }


def spearman(x: Iterable[float], y: Iterable[float]) -> CorrelationResult:
    """Spearman rho with NaN rows dropped jointly. Never raises on a flat input."""
    ax = _as_float_array(x, "x")
    ay = _as_float_array(y, "y")
    (ax, ay), dropped = _align_finite(ax, ay)
    n = int(ax.size)
    if n < 3 or np.allclose(ax, ax[0]) or np.allclose(ay, ay[0]):
        return CorrelationResult(rho=float("nan"), p=float("nan"), n=n, n_dropped=dropped)
    r, p = _stats.spearmanr(ax, ay)
    return CorrelationResult(rho=float(r), p=float(p), n=n, n_dropped=dropped)


def residualize(
    y: Iterable[float],
    covariates: Iterable[float] | Sequence[Iterable[float]],
    *,
    rank: bool = True,
) -> NDArray[np.float64]:
    """Return y with the linear effect of ``covariates`` removed.

    With ``rank=True`` (the default, and what a Spearman partial means) both y
    and the covariates are rank-transformed first, so the removed effect is
    monotone rather than linear-in-the-raw-units. With ``rank=False`` this is an
    ordinary OLS residual.

    ``covariates`` may be a single 1-D sequence or a sequence of them. An
    intercept is always included, so the returned residual is mean-zero.

    Non-finite rows are NOT dropped here — they would change the array's length
    and break the caller's index alignment. They propagate as NaN, and
    :func:`partial_spearman` drops them jointly before calling this.
    """
    ay = _as_float_array(y, "y")
    cols = _covariate_columns(covariates, n=ay.size)
    if rank:
        ay = _stats.rankdata(ay).astype(np.float64)
        cols = [_stats.rankdata(c).astype(np.float64) for c in cols]
    design = np.column_stack([*cols, np.ones_like(ay)])
    if not (np.all(np.isfinite(design)) and np.all(np.isfinite(ay))):
        return np.full_like(ay, np.nan)
    beta, *_ = np.linalg.lstsq(design, ay, rcond=None)
    return np.asarray(ay - design @ beta, dtype=np.float64)


def _covariate_columns(
    covariates: Iterable[float] | Sequence[Iterable[float]], n: int
) -> list[NDArray[np.float64]]:
    arr = np.asarray(covariates, dtype=np.float64)
    if arr.ndim == 1:
        cols = [arr]
    elif arr.ndim == 2:
        # Accept either [k, n] or [n, k]; [n, k] only when unambiguous.
        if arr.shape[1] == n and arr.shape[0] != n:
            cols = [arr[i] for i in range(arr.shape[0])]
        elif arr.shape[0] == n:
            cols = [arr[:, j] for j in range(arr.shape[1])]
        else:
            raise ValueError(
                f"covariate array shape {arr.shape} matches neither [k, {n}] "
                f"nor [{n}, k]"
            )
    else:
        raise ValueError(f"covariates must be 1-D or 2-D, got shape {arr.shape}")
    for c in cols:
        if c.size != n:
            raise ValueError(f"covariate has {c.size} rows, expected {n}")
    return cols


def _partial_from_residuals(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    n: int,
    order: int,
) -> tuple[float, float, int]:
    """Correlate two residual vectors and t-test with df = n - 2 - order."""
    df = n - 2 - order
    if df <= 0 or np.allclose(ex, 0.0) or np.allclose(ey, 0.0):
        return float("nan"), float("nan"), max(df, 0)
    r = float(np.corrcoef(ex, ey)[0, 1])
    if not math.isfinite(r):
        return float("nan"), float("nan"), df
    r = max(-1.0, min(1.0, r))
    denom = max(1e-12, 1.0 - r * r)
    t = r * math.sqrt(df / denom)
    p = float(2.0 * _stats.t.sf(abs(t), df=df))
    return r, p, df


def partial_spearman(
    x: Iterable[float],
    y: Iterable[float],
    z: Iterable[float] | Sequence[Iterable[float]],
) -> PartialResult:
    """Spearman partial rho(x, y | z): rank-transform, residualize, correlate.

    ``z`` may be one covariate or several; ``order`` in the result records how
    many were controlled, and the t-test uses ``df = n - 2 - order``. This is
    the generalisation of ``length_partial_v1.partial_spearman``, whose
    first-order ``df = n - 3`` is the ``order == 1`` case.
    """
    ax = _as_float_array(x, "x")
    ay = _as_float_array(y, "y")
    zcols = _covariate_columns(z, n=ax.size)
    aligned, dropped = _align_finite(ax, ay, *zcols)
    ax, ay, *zcols = aligned
    n = int(ax.size)
    order = len(zcols)

    r_xy = spearman(ax, ay)
    # A single controlled variable is the common case (length alone); for
    # the multi-covariate case report the edges against the FIRST covariate, and
    # say so via `order`.
    z0 = zcols[0]
    r_xz = spearman(ax, z0)
    r_yz = spearman(ay, z0)
    collin = abs(r_xz.rho) if math.isfinite(r_xz.rho) else float("nan")

    warning: str | None = None
    if n < MIN_N_FOR_PARTIAL or n - 2 - order <= 0:
        return PartialResult(
            rho=float("nan"), p=float("nan"), n=n, df=max(n - 2 - order, 0),
            order=order, rho_xy=r_xy.rho, rho_xz=r_xz.rho, rho_yz=r_yz.rho,
            collinearity=collin, n_dropped=dropped,
            warning=(
                f"n={n} is below MIN_N_FOR_PARTIAL={MIN_N_FOR_PARTIAL} "
                f"(or df<=0 at order {order}) — no partial reported"
            ),
        )

    ex = residualize(ax, zcols, rank=True)
    ey = residualize(ay, zcols, rank=True)
    rho, p, df = _partial_from_residuals(ex, ey, n=n, order=order)

    if not math.isfinite(rho):
        # The residual is degenerate: x (or y) is a perfect monotone function of
        # the covariate, so there is literally nothing left to correlate. This
        # is NOT a null result and must never be reported as one.
        return PartialResult(
            rho=float("nan"), p=float("nan"), n=n, df=df, order=order,
            rho_xy=r_xy.rho, rho_xz=r_xz.rho, rho_yz=r_yz.rho,
            collinearity=collin, n_dropped=dropped,
            warning=(
                f"UNDEFINED: rho(x, z) = {r_xz.rho:+.3f} leaves no residual "
                "variance after ranking — x and the covariate carry the same "
                "ordering, so no partial exists. This is not a null."
            ),
        )

    if math.isfinite(collin) and collin >= HIGH_COLLINEARITY:
        warning = (
            f"rho(x, z) = {r_xz.rho:+.3f} — x and the covariate are collinear at "
            f"|rho| >= {HIGH_COLLINEARITY}. At this collinearity the partial "
            "cannot attribute the relationship to either variable; a small "
            "partial is NOT evidence that x is spurious."
        )
    elif n < 20:
        warning = (
            f"n={n}: the partial is reported but underpowered — n ~ 10 is not "
            "enough to call a null (project trap)."
        )
    return PartialResult(
        rho=rho, p=p, n=n, df=df, order=order,
        rho_xy=r_xy.rho, rho_xz=r_xz.rho, rho_yz=r_yz.rho,
        collinearity=collin, n_dropped=dropped, warning=warning,
    )


def partial_pearson(
    x: Iterable[float],
    y: Iterable[float],
    z: Iterable[float] | Sequence[Iterable[float]],
) -> PartialResult:
    """Pearson partial r(x, y | z) — same accounting, no rank transform."""
    ax = _as_float_array(x, "x")
    ay = _as_float_array(y, "y")
    zcols = _covariate_columns(z, n=ax.size)
    aligned, dropped = _align_finite(ax, ay, *zcols)
    ax, ay, *zcols = aligned
    n = int(ax.size)
    order = len(zcols)

    def _pear(a: NDArray[np.float64], b: NDArray[np.float64]) -> CorrelationResult:
        if n < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return CorrelationResult(float("nan"), float("nan"), n)
        r, p = _stats.pearsonr(a, b)
        return CorrelationResult(float(r), float(p), n)

    r_xy, r_xz, r_yz = _pear(ax, ay), _pear(ax, zcols[0]), _pear(ay, zcols[0])
    collin = abs(r_xz.rho) if math.isfinite(r_xz.rho) else float("nan")
    if n < MIN_N_FOR_PARTIAL or n - 2 - order <= 0:
        return PartialResult(
            rho=float("nan"), p=float("nan"), n=n, df=max(n - 2 - order, 0),
            order=order, rho_xy=r_xy.rho, rho_xz=r_xz.rho, rho_yz=r_yz.rho,
            collinearity=collin, n_dropped=dropped,
            warning=f"n={n} below MIN_N_FOR_PARTIAL={MIN_N_FOR_PARTIAL}",
        )
    ex = residualize(ax, zcols, rank=False)
    ey = residualize(ay, zcols, rank=False)
    rho, p, df = _partial_from_residuals(ex, ey, n=n, order=order)
    warning = None
    if math.isfinite(collin) and collin >= HIGH_COLLINEARITY:
        warning = (
            f"r(x, z) = {r_xz.rho:+.3f} — collinear at |r| >= {HIGH_COLLINEARITY}; "
            "the partial cannot attribute to either variable."
        )
    return PartialResult(
        rho=rho, p=p, n=n, df=df, order=order,
        rho_xy=r_xy.rho, rho_xz=r_xz.rho, rho_yz=r_yz.rho,
        collinearity=collin, n_dropped=dropped, warning=warning,
    )


def covariate_report(
    measure: str,
    outcome: str,
    covariate: str,
    x: Iterable[float],
    y: Iterable[float],
    z: Iterable[float],
    *,
    note: str | None = None,
) -> CovariateReport:
    """The full block: headline, covariate's own edge, collinearity, partial.

    Banking ``report.to_dict()`` beside a spread-vs-outcome number is the whole
    point of this module — a length confound is visible only when the banked
    rows carry the covariate (``length_cv`` per row) and someone computes the
    correlation; banking the block makes both the default.
    """
    ax, ay, az = (_as_float_array(v, n) for v, n in ((x, "x"), (y, "y"), (z, "z")))
    return CovariateReport(
        measure=measure,
        outcome=outcome,
        covariate=covariate,
        raw=spearman(ax, ay),
        covariate_vs_outcome=spearman(az, ay),
        measure_vs_covariate=spearman(ax, az),
        partial=partial_spearman(ax, ay, az),
        note=note,
    )


def length_covariate_block(
    rows: Sequence[dict[str, Any]],
    measure_key: str,
    outcome_key: str,
    *,
    length_key: str = "length_cv",
    measure_name: str | None = None,
    outcome_name: str | None = None,
    note: str | None = None,
) -> dict[str, Any] | None:
    """Convenience: pull three fields out of banked rows and report the block.

    Returns ``None`` (rather than raising) when ``length_key`` is absent from the
    rows, so wiring this into an existing stage cannot break a receipt that
    predates length being banked. Rows missing any of the three keys, or holding
    a non-numeric value for one, are dropped — and the count is reported in
    ``raw.n_dropped``.
    """
    if not rows:
        return None
    if not any(length_key in r for r in rows):
        return None

    def _num(v: Any) -> float:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
            else float("nan")

    xs = [_num(r.get(measure_key)) for r in rows]
    ys = [_num(r.get(outcome_key)) for r in rows]
    zs = [_num(r.get(length_key)) for r in rows]
    if not any(math.isfinite(v) for v in zs):
        return None
    rep = covariate_report(
        measure=measure_name or measure_key,
        outcome=outcome_name or outcome_key,
        covariate=length_key,
        x=xs, y=ys, z=zs,
        note=note,
    )
    return rep.to_dict()
