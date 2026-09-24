"""STATS contract — the canonical implementations in ``pleroma.stats``
against scipy/numpy ground truth.

There is one implementation of each primitive. This is the spec each must
meet, plus the golden numbers the project's results rest on (the banked
length null +0.2571, the registered fold assignment, the forking Jaccard).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from . import targets as T

def test_exactly_one_implementation_of_each_survives() -> None:
    """One home per primitive."""
    assert list(T.SPEARMAN_IMPLS) == ["pleroma.stats.spearman"]
    assert set(T.PARTIAL_IMPLS) == {"pleroma.stats.partial_spearman",
                                    "pleroma.stats.partial_pearson"}
    assert list(T.BOOTSTRAP_IMPLS) == ["pleroma.stats.boot_ci"]


# ════════════════════════════════════════════════════════════════════════════
# Spearman
# ════════════════════════════════════════════════════════════════════════════

DATA: dict[str, tuple[list[float], list[float]]] = {
    "no_ties": ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [2, 1, 4, 3, 6, 5, 8, 7, 12, 9, 11, 10]),
    "ties_x": ([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
               [3, 1, 2, 5, 4, 6, 8, 7, 9, 12, 10, 11]),
    "ties_both": ([1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8],
                  [2, 2, 1, 3, 3, 5, 4, 4, 6, 8, 7, 7]),
    "negative_n8": ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                    [9, 7, 8, 5, 6, 3, 4, 1]),
}


def _rho_p(name: str, x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Normalise every implementation's return shape to (rho, p-or-None)."""
    fn = T.SPEARMAN_IMPLS[name]
    xf, yf = [float(v) for v in x], [float(v) for v in y]
    r = fn(xf, yf)
    return r.rho, r.p


#: smallest n at which each implementation reports a rho (its min-n policy).
MIN_N = {"pleroma.stats.spearman": 3}
RHO_TOL: dict[str, float] = {}
RHO_NONCONFORMING: dict[tuple[str, str], float] = {}


def _defined(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def test_reference_scipy_spearman_averages_ties() -> None:
    """The reference itself: scipy's Spearman == Pearson on average ranks."""
    for x, y in DATA.values():
        want = stats.pearsonr(stats.rankdata(x), stats.rankdata(y))[0]
        assert stats.spearmanr(x, y).statistic == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("dataset", sorted(DATA))
@pytest.mark.parametrize("impl", sorted(T.SPEARMAN_IMPLS))
def test_spearman_rho_against_scipy(impl: str, dataset: str) -> None:
    """Every Spearman vs scipy.stats.spearmanr with average-rank ties.
    Conforming impls match to their rounding; non-conforming ones are pinned
    as-is (RHO_NONCONFORMING); below an impl's min-n, rho is undefined."""
    x, y = DATA[dataset]
    rho, _ = _rho_p(impl, x, y)
    if len(x) < MIN_N[impl]:
        assert not _defined(rho)
        return
    if (impl, dataset) in RHO_NONCONFORMING:
        assert rho == pytest.approx(RHO_NONCONFORMING[(impl, dataset)], abs=1e-12)
        assert rho != pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-3)
        return
    assert rho == pytest.approx(stats.spearmanr(x, y).statistic, abs=RHO_TOL.get(impl, 1e-12))


@pytest.mark.parametrize("dataset", sorted(DATA))
@pytest.mark.parametrize("impl", sorted(T.SPEARMAN_IMPLS))
def test_spearman_p_against_scipy(impl: str, dataset: str) -> None:
    """p-values: scipy's t (n-2 df) is the reference. (A normal approximation
    of the null is anti-conservative by as much as nine orders of magnitude —
    p≈1e-14 where scipy gives ≈1e-5.)"""
    x, y = DATA[dataset]
    _, p = _rho_p(impl, x, y)
    if len(x) < MIN_N[impl]:
        assert not _defined(p)
        return
    assert p == pytest.approx(stats.spearmanr(x, y).pvalue, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("impl", sorted(T.SPEARMAN_IMPLS))
def test_spearman_min_n_policy_pinned(impl: str) -> None:
    """Each impl reports rho from exactly its own min-n upward. Data:
    x = 0..n-1, y = reversed with the first
    two swapped (never constant, never degenerate)."""
    for n in range(2, 13):
        x = [float(i) for i in range(n)]
        y = [float(v) for v in reversed(range(n))]
        y[0], y[1] = y[1], y[0]
        rho, _ = _rho_p(impl, x, y)
        assert _defined(rho) is (n >= MIN_N[impl]), n


def test_canonical_spearman_drops_non_finite_jointly_and_never_raises_on_flat() -> None:
    """pleroma.stats.spearman: NaN rows dropped jointly (n and n_dropped
    reported); a flat input gives NaN, not an exception."""
    x, y = DATA["no_ties"]
    xn = list(x) + [float("nan")]
    yn = list(y) + [3.0]
    r = T.SPEARMAN_IMPLS["pleroma.stats.spearman"](xn, yn)
    assert (r.n, r.n_dropped) == (12, 1)
    assert r.rho == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)
    flat = T.SPEARMAN_IMPLS["pleroma.stats.spearman"]([1.0] * 6, [1, 2, 3, 4, 5, 6])
    assert math.isnan(flat.rho) and math.isnan(flat.p)


# ════════════════════════════════════════════════════════════════════════════
# Partial correlation
# ════════════════════════════════════════════════════════════════════════════

PX = [-0.0, 0.9, -1.3, -1.3, -2.0, -2.0, -1.4, 1.2, -1.5, -0.4, 0.6, 0.2, -1.9, -1.4]
PY = [-0.0, 0.3, -1.7, -0.9, -1.2, -1.3, 1.1, -0.1, -0.3, 0.6, -0.3, 0.1, 0.2, -0.4]
PZ = [0.0, 0.3, -0.3, -0.9, -0.5, -1.0, 0.1, 1.3, -0.5, -0.6, 0.5, 0.4, 0.1, -0.9]


def _closed_form_partial(x: Any, y: Any, z: Any) -> float:
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def _t_p(r: float, n: int, order: int = 1) -> float:
    df = n - 2 - order
    return float(2 * stats.t.sf(abs(r * math.sqrt(df / (1 - r * r))), df))


REF_PARTIAL_SPEARMAN = _closed_form_partial(stats.rankdata(PX), stats.rankdata(PY),
                                            stats.rankdata(PZ))
REF_PARTIAL_PEARSON = _closed_form_partial(PX, PY, PZ)

#: which reference each implementation claims to compute
PARTIAL_KIND = {
    "pleroma.stats.partial_spearman": "spearman",
    "pleroma.stats.partial_pearson": "pearson",
}


def _partial(name: str, x: Any, y: Any, z: Any) -> tuple[float, float]:
    out = T.PARTIAL_IMPLS[name](np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    if hasattr(out, "rho"):
        return out.rho, out.p
    return out[0], out[1]


@pytest.mark.parametrize("impl", sorted(T.PARTIAL_IMPLS))
def test_partial_correlation_against_closed_form(impl: str) -> None:
    """First-order partial vs the closed form r_xy.z = (r_xy − r_xz r_yz) /
    √((1−r_xz²)(1−r_yz²)) on ranks (Spearman) or raw values (Pearson); p from
    Student t with df = n − 3 (n − 2 − the number of covariates).
    Both conform at n = 14 (ties present in x)."""
    rho, p = _partial(impl, PX, PY, PZ)
    ref = REF_PARTIAL_SPEARMAN if PARTIAL_KIND[impl] == "spearman" else REF_PARTIAL_PEARSON
    assert rho == pytest.approx(ref, abs=1e-10)
    assert p == pytest.approx(_t_p(ref, len(PX)), rel=1e-8)


#: small-n behaviour on the first n rows, pinned as-is. NaN = no value.
NAN = float("nan")
PARTIAL_SMALL_N = {
    3: {"pleroma.stats.partial_spearman": (NAN, NAN),
        "pleroma.stats.partial_pearson": (NAN, NAN)},
    4: {"pleroma.stats.partial_spearman": (NAN, NAN),
        "pleroma.stats.partial_pearson": (NAN, NAN)},
}


@pytest.mark.parametrize("n", sorted(PARTIAL_SMALL_N))
@pytest.mark.parametrize("impl", sorted(T.PARTIAL_IMPLS))
def test_partial_small_n_pinned(impl: str, n: int) -> None:
    """Below MIN_N_FOR_PARTIAL (5) there is no partial — NaN with a warning.
    (Without the n guard a partial at n=4 reports a spurious p≈6e-7.)"""
    rho, p = _partial(impl, PX[:n], PY[:n], PZ[:n])
    want_r, want_p = PARTIAL_SMALL_N[n][impl]
    for got, want in ((rho, want_r), (p, want_p)):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_canonical_partial_reports_df_order_collinearity_and_warnings() -> None:
    """pleroma.stats.partial_spearman: df = n − 2 − order, collinearity = |ρ(x,z)|,
    warning at |ρ(x,z)| ≥ 0.85, and a min-n refusal that says so (never a bare
    NaN a newcomer could read as a null)."""
    r = T.PARTIAL_IMPLS["pleroma.stats.partial_spearman"](PX, PY, PZ)
    assert (r.n, r.df, r.order) == (14, 11, 1)
    assert r.collinearity == pytest.approx(abs(stats.spearmanr(PX, PZ).statistic))
    assert r.warning is not None and "underpowered" in r.warning
    zc = [v + 0.01 * i for i, v in enumerate(PX)]
    hi = T.PARTIAL_IMPLS["pleroma.stats.partial_spearman"](PX, PY, zc)
    assert hi.collinearity >= T.LC_HIGH_COLLINEARITY and "collinear" in (hi.warning or "")
    small = T.PARTIAL_IMPLS["pleroma.stats.partial_spearman"](PX[:4], PY[:4], PZ[:4])
    assert T.LC_MIN_N_FOR_PARTIAL == 5 and "MIN_N_FOR_PARTIAL" in (small.warning or "")


def test_canonical_second_order_partial_uses_n_minus_4_df() -> None:
    """Two covariates: rank-residualise x and y on [1, rank z1, rank z2] and
    correlate; t-test df = n − 4 (the k-th order df, n − 2 − k)."""
    z2 = [((i * 5) % 7) - 3.0 for i in range(len(PX))]
    r = T.PARTIAL_IMPLS["pleroma.stats.partial_spearman"](PX, PY, [PZ, z2])
    Z = np.column_stack([np.ones(len(PX)), stats.rankdata(PZ), stats.rankdata(z2)])

    def res(v: Any) -> Any:
        a = stats.rankdata(v)
        return a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    ref = float(np.corrcoef(res(PX), res(PY))[0, 1])
    assert (r.order, r.df) == (2, 10)
    assert r.rho == pytest.approx(ref, abs=1e-10)
    assert r.p == pytest.approx(_t_p(ref, len(PX), order=2), rel=1e-8)


# ════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ════════════════════════════════════════════════════════════════════════════

BVALS = [0.3, -0.1, 0.25, 0.4, 0.05, 0.18, -0.02, 0.33]


def reference_bootstrap(vals: list[float], n_boot: int, seed: int,
                        alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean: numpy default_rng(seed), integers(0,k,(B,k)),
    np.quantile (linear) at alpha/2, 1-alpha/2."""
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _ci(name: str, vals: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    out = T.BOOTSTRAP_IMPLS[name](vals, n_boot, seed)
    if isinstance(out, dict):
        return out["lo"], out["hi"]
    return float(out[0]), float(out[1])


@pytest.mark.parametrize("impl", sorted(T.BOOTSTRAP_IMPLS))
def test_bootstrap_is_deterministic_under_seed(impl: str) -> None:
    """Same (values, n_boot, seed) -> the identical interval; a different seed
    moves it. Every bootstrap in the repo meets this."""
    a = _ci(impl, BVALS, 2000, 11)
    assert a == _ci(impl, BVALS, 2000, 11)
    assert a != _ci(impl, BVALS, 2000, 12)


def test_bootstrap_equals_the_reference() -> None:
    """Bit-identical to the reference percentile bootstrap."""
    for seed in (11, 20260921):
        assert _ci("pleroma.stats.boot_ci", BVALS, 2000, seed) == \
            reference_bootstrap(BVALS, 2000, seed)


def test_bootstrap_edge_policies_pinned() -> None:
    """Non-finite values are dropped; fewer than two finite values -> NaN."""
    ch = T.BOOTSTRAP_IMPLS["pleroma.stats.boot_ci"]
    assert ch([0.1, float("nan"), 0.4, 0.2], 1000, 3) == pytest.approx(
        reference_bootstrap([0.1, 0.4, 0.2], 1000, 3))
    assert all(math.isnan(v) for v in ch([0.1], 1000, 3))


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_distinct_resamples_is_the_multiset_count(k: int) -> None:
    """Resampling k units with replacement yields C(2k−1, k) distinct
    multisets (k=3 -> 10), not the k**k (27) ordered draws."""
    assert T.distinct_resamples(k) == math.comb(2 * k - 1, k)


# ════════════════════════════════════════════════════════════════════════════
# Grouped CV fold assignment
# ════════════════════════════════════════════════════════════════════════════

PROMPTS = np.asarray([f"p{i:02d}" for i in range(12) for _ in range(3)])
FANS = [np.arange(3 * i, 3 * i + 3) for i in range(12)]


def reference_seeded_folds(prompt_arr: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """The REGISTERED scheme, restated independently: sorted unique
    prompts, rng(seed).permutation, fold = position-in-permutation mod F."""
    uniq = sorted(set(prompt_arr.tolist()))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[int(j)]: int(i % folds) for i, j in enumerate(order)}
    return np.asarray([fold_of[p] for p in prompt_arr], dtype=int)


def _grouped_cv_properties(fold: np.ndarray, groups: np.ndarray, folds: int) -> None:
    for g in set(groups.tolist()):
        assert len(set(fold[groups == g].tolist())) == 1, f"group {g} split"
    per_fold = [len(set(groups[fold == f].tolist())) for f in range(folds)]
    assert min(per_fold) >= 1 and max(per_fold) - min(per_fold) <= 1


def test_registered_fold_constants() -> None:
    """5 folds, seed 20260921, identical in pleroma.stats.folds and
    pleroma.map.build.cv."""
    assert (T.FOLDS, T.FOLD_SEED) == (T.V1A_FOLDS, T.V1A_FOLD_SEED) == (5, 20260921)


def test_seeded_folds_equal_the_registered_scheme_and_are_grouped() -> None:
    """pleroma.stats.make_folds reproduces the registered construction exactly;
    no prompt (group) straddles folds; folds balanced to within one group.
    Golden assignment per prompt pinned (numpy PCG64 stream)."""
    got = T.make_folds_seeded(PROMPTS, FANS)
    assert np.array_equal(got, reference_seeded_folds(PROMPTS, T.FOLDS, T.FOLD_SEED))
    _grouped_cv_properties(got, PROMPTS, T.FOLDS)
    assert got[::3].tolist() == [1, 2, 0, 0, 1, 4, 1, 2, 3, 0, 3, 4]


def test_seeded_folds_refuse_a_fan_split_across_folds() -> None:
    """A 'fan' spanning two prompts that land in different folds is refused
    (ValueError), never silently accepted: a fan split across folds leaks its
    near-twins into training."""
    fold = T.make_folds_seeded(PROMPTS, FANS)
    a = next(i for i in range(12) if fold[3 * i] != fold[0])
    with pytest.raises(ValueError):
        T.make_folds_seeded(PROMPTS, FANS + [np.asarray([0, 3 * a])])


# ════════════════════════════════════════════════════════════════════════════
# Length-only null (pleroma.stats.length_null)
# ════════════════════════════════════════════════════════════════════════════

def test_length_only_rank_is_the_nr_of_a_pure_length_matcher() -> None:
    """length_only_rank: candidates ordered by |len − len(reply)|, ties to the
    lower fan index, position / (k − 1) — the same 0..1 scale as the judge's
    normalized_rank."""
    lens, reply = [90, 130, 95, 200], 100
    assert [T.length_only_rank(lens, reply, i) for i in range(4)] == [1 / 3, 2 / 3, 0.0, 1.0]
    assert T.length_only_rank([110, 90], 100, 0) == 0.0  # tie -> lower index first
    with pytest.raises(ValueError):
        T.length_only_rank([5], 5, 0)


def test_length_only_dnr_definition() -> None:
    """Δnr of the length ranker: per conversation mean(base) − mean(arm), then an
    unweighted mean over conversations carrying BOTH arms.
    POSITIVE = arm replies sit closer in length to their prescribed future."""
    by_conv = {"c1": {"base": [0.8, 0.6], "arm": [0.2]},
               "c2": {"base": [0.5], "arm": [0.7, 0.5]},
               "c3": {"base": [0.9]}}  # no arm: excluded
    out = T.length_only_dnr(by_conv, "arm")
    assert out["per_conversation"] == pytest.approx({"c1": 0.5, "c2": -0.1})
    assert out["observed"] == pytest.approx(0.2)
    assert (out["n_conversations"], out["n_positive"]) == (2, 1)
    with pytest.raises(ValueError):
        T.length_only_dnr({"c3": {"base": [0.9]}}, "arm")


def test_length_null_reproduces_the_banked_8b_number() -> None:
    """The rule's anchor (a Δnr that does not beat the length-only ranker has
    not demonstrated manner steering, because the judge reads length): on the
    banked sitenorm 8B fixture (in-repo artifact) the length-only ranker scores
    Δnr +0.2571 (12/13 positive) raw and +0.2053 (9/13) sitenorm — vs the
    manner judge's +0.0848. Lengths are CHARACTERS."""
    data = json.loads(T.LENGTH_NULL_FIXTURE.read_text())
    assert data["length_unit"].startswith("characters")
    by_conv: dict[str, dict[str, list[float]]] = {}
    for row in data["rows"]:
        lo = T.length_only_rank(data["cand_chars"][row["prompt_id"]], row["reply_chars"],
                                row["prescribed_index"])
        for key in row["keys"]:
            by_conv.setdefault(row["prompt_id"], {}).setdefault(key, []).append(lo)
    raw, sn = T.length_only_dnr(by_conv, "raw"), T.length_only_dnr(by_conv, "sitenorm")
    assert (round(raw["observed"], 4), raw["n_positive"], raw["n_conversations"]) == (0.2571, 12, 13)
    assert (round(sn["observed"], 4), sn["n_positive"]) == (0.2053, 9)


def test_length_unit_is_characters_not_words() -> None:
    """The length null counts CHARACTERS, not words; words() is kept only
    for reporting beside it."""
    assert T.length_chars("ab cd") == 5 and T.length_words("ab cd") == 2


# ════════════════════════════════════════════════════════════════════════════
# Fan-diversity Jaccard — the .1015 instrument (pleroma.stats.diversity)
# ════════════════════════════════════════════════════════════════════════════

TEXTS = ["The cat sat on the mat today.", "A dog sat on the rug, today!", "The cat ran."]


def test_fan_jaccard_is_the_headline_instrument() -> None:
    """pleroma.stats.diversity.fan_jaccard (the measure that reads 8B fans at
    .2467 under chat render and .1015 under raw render): lowercase [a-z0-9']+ words,
    first `window`, mean pairwise set Jaccard."""
    assert T.diversity_words("Don't STOP—now! 3.5") == ["don't", "stop", "now", "3", "5"]
    assert T.fan_jaccard(TEXTS, 60) == pytest.approx(0.2804232804232804, abs=1e-15)


def test_an_empty_future_counts_as_zero_overlap_not_skipped() -> None:
    """An empty future scores 0 against every sibling (.2804 -> .1402).
    Skipping it instead would read .2804 — not comparable."""
    assert T.fan_jaccard(TEXTS + [""], 60) == pytest.approx(0.1402116402116402, abs=1e-15)
    with pytest.raises(ValueError):
        T.fan_jaccard(["only one"], 60)
