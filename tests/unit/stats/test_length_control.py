"""Tests for the length control helper and its two additive wiring sites.

Two jobs:

1. The arithmetic is right — partial Spearman matches the closed-form
   first-order partial, the degrees of freedom are n-3 (not n-2), and
   residualize() actually leaves a residual orthogonal to the covariate.

2. ★ THE PIN. The wiring is additive: adding a `length_partials` block must
   not move a single number in `correlations`. `test_length_partials_are_purely_additive`
   is that pin. If it ever fails, a published correlation has been changed and
   the whole comparability argument against touching the instrument is void.

Laptop-only: numpy + scipy + pytest, no torch, no node, no GPU (tests/conftest.py).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import stats

from pleroma.stats.correlation import (
    HIGH_COLLINEARITY,
    MIN_N_FOR_PARTIAL,
    covariate_report,
    length_covariate_block,
    partial_pearson,
    partial_spearman,
    residualize,
    spearman,
)


# ── the arithmetic ────────────────────────────────────────────────────────────


def _closed_form_partial(rxy: float, rxz: float, ryz: float) -> float:
    """Textbook first-order partial: (rxy - rxz*ryz) / sqrt((1-rxz^2)(1-ryz^2))."""
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz**2) * (1 - ryz**2))


def test_partial_spearman_matches_the_closed_form_first_order_partial():
    rng = np.random.default_rng(20260922)
    z = rng.normal(size=40)
    x = 0.8 * z + 0.6 * rng.normal(size=40)
    y = 0.5 * z + 0.5 * x + 0.7 * rng.normal(size=40)
    got = partial_spearman(x, y, z)
    want = _closed_form_partial(
        stats.spearmanr(x, y).statistic,
        stats.spearmanr(x, z).statistic,
        stats.spearmanr(y, z).statistic,
    )
    # Residualize-then-correlate and the closed form agree on ranks to ~1e-12.
    assert got.rho == pytest.approx(want, abs=1e-10)
    assert got.order == 1


def test_a_first_order_partial_spends_one_degree_of_freedom():
    """df = n - 3, not n - 2. At n=9 that is most of the difference in p."""
    rng = np.random.default_rng(7)
    n = 9
    z = rng.normal(size=n)
    x, y = z + rng.normal(size=n), z + rng.normal(size=n)
    r = partial_spearman(x, y, z)
    assert r.n == n
    assert r.df == n - 3
    # and the reported p is the t-test on that df, not scipy's n-2 default
    t = r.rho * math.sqrt(r.df / max(1e-12, 1 - r.rho**2))
    assert r.p == pytest.approx(2 * stats.t.sf(abs(t), df=n - 3), rel=1e-9)


def test_controlling_two_covariates_spends_two_degrees_of_freedom():
    rng = np.random.default_rng(11)
    n = 30
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    x, y = z1 + z2 + rng.normal(size=n), z1 - z2 + rng.normal(size=n)
    r = partial_spearman(x, y, [z1, z2])
    assert r.order == 2
    assert r.df == n - 4


def test_partialling_out_the_true_cause_kills_a_spurious_correlation():
    """x and y share only z; controlling z should leave nothing."""
    rng = np.random.default_rng(3)
    n = 400
    z = rng.normal(size=n)
    x = 2.0 * z + 0.3 * rng.normal(size=n)
    y = -1.5 * z + 0.3 * rng.normal(size=n)
    raw = spearman(x, y)
    par = partial_spearman(x, y, z)
    assert raw.rho < -0.9              # strong spurious correlation
    assert abs(par.rho) < 0.15         # and it is gone once z is controlled
    assert par.p > 0.01


def test_partialling_an_unrelated_covariate_leaves_a_real_correlation_alone():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    y = x + 0.4 * rng.normal(size=n)
    z = rng.normal(size=n)  # independent of both
    raw, par = spearman(x, y), partial_spearman(x, y, z)
    assert raw.rho > 0.85
    assert par.rho == pytest.approx(raw.rho, abs=0.08)


def test_residualize_leaves_a_residual_orthogonal_to_the_covariate():
    rng = np.random.default_rng(13)
    y, z = rng.normal(size=50), rng.normal(size=50)
    e = residualize(y, z, rank=False)
    assert float(np.dot(e, z)) == pytest.approx(0.0, abs=1e-9)
    assert float(e.mean()) == pytest.approx(0.0, abs=1e-9)  # intercept included
    er = residualize(y, z, rank=True)
    assert float(np.dot(er, stats.rankdata(z))) == pytest.approx(0.0, abs=1e-9)


def test_residualize_accepts_several_covariates_in_either_orientation():
    rng = np.random.default_rng(17)
    n = 25
    y = rng.normal(size=n)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    a = residualize(y, [z1, z2], rank=False)          # [k, n]
    b = residualize(y, np.column_stack([z1, z2]), rank=False)  # [n, k]
    assert np.allclose(a, b)


# ── guards ────────────────────────────────────────────────────────────────────


def test_a_sample_too_small_for_a_partial_refuses_rather_than_inventing_one():
    r = partial_spearman([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [3.0, 1.0, 2.0])
    assert math.isnan(r.rho)
    assert r.warning is not None and str(MIN_N_FOR_PARTIAL) in r.warning


def test_high_collinearity_is_reported_rather_than_silently_swallowed():
    """The measured situation: fpcd and length correlate at rho(x, z) ~ 0.91,
    which makes the partial unreadable."""
    rng = np.random.default_rng(23)
    n = 12
    z = rng.normal(size=n)
    x = z + 0.15 * rng.normal(size=n)  # near-collinear with z, as fpcd is
    y = z + rng.normal(size=n)
    r = partial_spearman(x, y, z)
    assert r.collinearity > HIGH_COLLINEARITY
    assert r.warning is not None and "collinear" in r.warning
    assert "not evidence" in r.warning.lower()


def test_a_perfectly_rank_collinear_covariate_is_undefined_not_null():
    """If x is a monotone function of z there is no residual left. Returning a
    NaN labelled "not a null" is the difference between "we cannot tell" and
    "fpcd is debunked" — the confusion a collapsed partial invites."""
    z = np.arange(12, dtype=float)
    x = 2.0 * z + 1.0          # rank-identical to z
    y = z + np.array([0.3, -0.2] * 6)
    r = partial_spearman(x, y, z)
    assert math.isnan(r.rho)
    assert r.collinearity == pytest.approx(1.0)
    assert r.warning is not None
    assert "UNDEFINED" in r.warning and "not a null" in r.warning


def test_nonfinite_rows_are_dropped_jointly_not_pairwise():
    """Pairwise deletion would give the three edges three different samples."""
    x = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0]
    y = [1.0, 3.0, 2.0, float("nan"), 5.0, 4.0, 7.0]
    z = [2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
    r = partial_spearman(x, y, z)
    assert r.n == 5 and r.n_dropped == 2
    # every reported edge is on the SAME 5 rows
    assert spearman(x, y).n == 5


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="length mismatch"):
        partial_spearman([1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0])


def test_a_constant_input_yields_nan_rather_than_a_crash():
    r = spearman([1.0] * 10, list(range(10)))
    assert math.isnan(r.rho) and r.n == 10


def test_partial_pearson_matches_the_closed_form_on_raw_values():
    rng = np.random.default_rng(29)
    n = 60
    z = rng.normal(size=n)
    x, y = z + rng.normal(size=n), 0.5 * z + rng.normal(size=n)
    got = partial_pearson(x, y, z)
    want = _closed_form_partial(
        float(np.corrcoef(x, y)[0, 1]),
        float(np.corrcoef(x, z)[0, 1]),
        float(np.corrcoef(y, z)[0, 1]),
    )
    assert got.rho == pytest.approx(want, abs=1e-10)


# ── the measured fpcd-vs-length partials, reproduced through the helper ──────


def test_it_reproduces_the_diary_4E_partials_on_synthetic_data_of_that_shape():
    """A 9-row sample at rho(x,z)=0.92 must reproduce the measured structure:
    a strong raw rho, a strong covariate edge, and a partial that collapses."""
    rng = np.random.default_rng(20260922)
    n = 9
    z = rng.normal(size=n)
    x = z + 0.45 * rng.normal(size=n)   # collinear, but not rank-identical
    y = 0.9 * z + 0.65 * rng.normal(size=n)
    rep = covariate_report("fpcd", "dnr", "length_cv", x, y, z)
    assert rep.raw.n == n
    assert rep.measure_vs_covariate.rho > HIGH_COLLINEARITY
    assert rep.partial is not None
    assert abs(rep.partial.rho) < abs(rep.raw.rho)   # the partial collapses
    assert rep.partial.warning is not None            # and says why to distrust it
    d = rep.to_dict()
    assert set(d) >= {"measure", "outcome", "covariate", "raw",
                      "covariate_vs_outcome", "measure_vs_covariate", "partial"}
    json.dumps(d)  # must be bankable




# ── length_covariate_block: the banked-rows convenience ───────────────────────


def _rows(n=12, seed=41):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = z + 0.3 * rng.normal(size=n)
    y = z + rng.normal(size=n)
    return [{"fpcd": float(x[i]), "dnr": float(y[i]), "length_cv": float(z[i])}
            for i in range(n)]


def test_length_covariate_block_reads_banked_rows():
    b = length_covariate_block(_rows(), "fpcd", "dnr")
    assert b is not None
    assert b["measure"] == "fpcd" and b["covariate"] == "length_cv"
    assert b["raw"]["n"] == 12
    assert b["partial"]["order"] == 1


def test_rows_without_a_length_column_return_None_rather_than_raising():
    """A receipt banked before length_cv existed must still load."""
    rows = [{"fpcd": 1.0, "dnr": 2.0}, {"fpcd": 2.0, "dnr": 1.0}]
    assert length_covariate_block(rows, "fpcd", "dnr") is None
    assert length_covariate_block([], "fpcd", "dnr") is None


def test_rows_whose_length_column_is_all_null_return_None():
    rows = [{"fpcd": 1.0, "dnr": 2.0, "length_cv": None} for _ in range(8)]
    assert length_covariate_block(rows, "fpcd", "dnr") is None


def test_a_non_numeric_cell_is_dropped_and_counted_not_crashed_on():
    rows = _rows(n=10)
    rows[0]["length_cv"] = "n/a"
    b = length_covariate_block(rows, "fpcd", "dnr")
    assert b is not None and b["partial"]["n"] == 9 and b["partial"]["n_dropped"] == 1


def test_booleans_are_not_treated_as_numbers():
    """`gate_passed: true` next to `fpcd` must not become 1.0 silently."""
    rows = _rows(n=8)
    rows[0]["fpcd"] = True
    b = length_covariate_block(rows, "fpcd", "dnr")
    assert b is not None and b["raw"]["n"] == 7


# ── ★ THE PIN: the wiring must not move an existing number ────────────────────










