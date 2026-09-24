"""Contract: the one-vs-rest contrastive code and each implementation of it.

`c_i − mean_{j≠i} c_j` is the zero-training contrastive probe: the code
`/probe` and `/wear_code {differential}` wear, and — in input space as
``pleroma.map.build.join.fan_center`` — the transform v1a was FIT on. Two
implementations compute it with different validation; these tests pin (a) that
each is the same function on well-formed input and (b) what each does on the
degenerate k=1 fan, where they diverge.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest

from . import targets as T


def codes_fixture(k: int = 5, r: int = 4) -> np.ndarray:
    return np.random.default_rng(8).standard_normal((k, r))


def reference(codes: np.ndarray, i: int) -> np.ndarray:
    return codes[i] - np.delete(codes, i, axis=0).mean(axis=0)


#: every implementation as f(codes [k, r], i) -> contrastive code of member i
COPIES: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "loom_probe.contrastive_code":
        lambda c, i: np.asarray(T.contrastive_loom_probe(c.tolist(), i)),
    "v1a_fit.fan_center":
        lambda c, i: T.fan_center(c, [np.arange(c.shape[0])])[i],
}


@pytest.mark.parametrize("copy", list(COPIES))
def test_every_copy_is_member_minus_mean_of_the_others(copy: str) -> None:
    """On a well-formed fan every implementation returns c_i − mean(c_rest) to
    float roundoff, for every member. (The export's 'deployment_identity'
    check relies on contrastive_code ≡ fan_center.)"""
    codes = codes_fixture()
    for i in range(codes.shape[0]):
        np.testing.assert_allclose(COPIES[copy](codes, i), reference(codes, i),
                                   rtol=1e-12, atol=1e-13)


def test_contrastive_codes_of_one_fan_sum_to_zero() -> None:
    """Σ_i (c_i − mean_{j≠i} c_j) = 0: the rest-mean is over k−1, not k."""
    codes = codes_fixture()
    total = sum(np.asarray(T.contrastive_loom_probe(codes.tolist(), i))
                for i in range(codes.shape[0]))
    np.testing.assert_allclose(total, 0.0, atol=1e-12)


def test_fan_center_centres_each_fan_independently() -> None:
    """fan_center applies the LOO contrast within each fan's index set; rows
    of different fans never mix (the property grouped CV and the export rely
    on)."""
    x = np.random.default_rng(9).standard_normal((7, 3))
    fans = [np.array([0, 2, 4]), np.array([1, 3, 5, 6])]
    out = T.fan_center(x, fans)
    for sel in fans:
        sub = x[sel]
        for pos, row in enumerate(sel):
            np.testing.assert_allclose(out[row], reference(sub, pos), atol=1e-12)


#: Behaviour on a fan of ONE member (no "rest"): the implementations diverge.
K1_EXPECTED: dict[str, str] = {
    "loom_probe.contrastive_code": "raise:ValueError",
    "v1a_fit.fan_center": "nan",
}


@pytest.mark.parametrize("copy", list(COPIES))
def test_a_fan_of_one_is_handled_as_pinned(copy: str) -> None:
    """k=1 has no rest-mean. The probe's contrastive_code refuses with
    ValueError; fan_center returns NaN with only a RuntimeWarning (it has no
    k≥2 check) — it is on the fit/export path and relies on the join's min-fan
    filter upstream. Adding that check changes this cell deliberately."""
    codes = codes_fixture(k=1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                out = COPIES[copy](codes, 0)
    except (ValueError, ZeroDivisionError) as exc:
        got = f"raise:{type(exc).__name__}"
    else:
        got = "nan" if np.isnan(out).all() else "value"
    assert got == K1_EXPECTED[copy]


def test_the_validated_copy_refuses_mixed_width_codes() -> None:
    """Mixed code widths mean codes from more than one map; refuse (ValueError)
    rather than zip-truncate."""
    with pytest.raises(ValueError):
        T.contrastive_loom_probe([[1.0, 2.0], [1.0, 2.0, 3.0], [0.0, 0.0]], 0)
