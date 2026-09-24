"""Contract: the lever math — absolute, from-code (both kinds), fan contrast.

The loom turns a sampled future into a steering vector in one of three ways
(``pleroma.map.loom_map.LoomMap``):

* absolute  `lever_of`            = W·x_i  (+ mu terms; 0 for v1a) — what /wear
                                    serves by default (NOT the fan contrast
                                    the map was fit on);
* from code `lever_from_code`     = code @ Vt (+ mu_y unless differential);
* contrast  `contrast_lever_of`   = W·(x_i − mean_{j≠i} x_j) — the object v1a
                                    was FIT on and what /probe and
                                    /wear_code{differential} wear.

Every returned lever is norm-matched per site to `norm_ref` (the ruler: alpha
is an absolute per-site norm), and the RAW per-site norms are returned beside
it. These tests are the identities any move of this code must keep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from . import _fixtures as fx
from . import targets as T


def fan_inputs(m: T.LoomMap, k: int = 5, seed: int = 7) -> list[np.ndarray]:
    return [m.input_of(s, b) for s, b in fx.raw_rows(k, seed)]


def exact_code(m: T.LoomMap, x: np.ndarray) -> np.ndarray:
    """The UNROUNDED code of input row x: Vt @ ((x − mu_in) @ W)."""
    return m.Vt @ ((x - m.mu_in) @ m.W)


@pytest.fixture(params=["v1a", "wide"])
def any_map(request: pytest.FixtureRequest, v1a: T.LoomMap,
            wide: T.LoomMap) -> T.LoomMap:
    """The served v1a form (factors, mu = 0) and the v0 wide form (dense W,
    mu != 0): identities that must hold for both."""
    return v1a if request.param == "v1a" else wide


def round_trip_tol(m: T.LoomMap) -> dict[str, float]:
    """Code <-> lever identities are exact to float64 roundoff on the v1a form
    (float64 factors: the export's meta "factor_dtype") but only to float32
    precision on fit_loom_map's wide form: its dense W is stored float32, so it
    is rank-r only up to ~1e-7 and the part of W·x outside span(Vt) is dropped
    by any code. Pinned per form so no refactor can silently loosen v1a."""
    if m.mu_y.any():  # the wide (v0) form
        return {"rtol": 1e-4, "atol": 1e-6}
    return {"rtol": 1e-9, "atol": 1e-12}


# ── absolute lever ───────────────────────────────────────────────────────────


def test_absolute_lever_is_W_x_norm_matched_per_site(any_map: T.LoomMap) -> None:
    """lever_of = ruler((x − mu_in)·W + mu_y) with raw per-site norms reported
    beside it. The absolute lever is what WEAR serves by default; every banked
    dose band was measured on these bytes."""
    m = any_map
    sig, brow = fx.raw_rows(1)[0]
    lever, raw, code = m.lever_of(sig, brow)
    flat = (m.input_of(sig, brow) - m.mu_in) @ m.W + m.mu_y
    rows = flat.reshape(m.n_sites, m.hidden)
    want_raw = np.linalg.norm(rows, axis=1)
    assert lever.shape == (len(fx.SITES), fx.HIDDEN)
    np.testing.assert_allclose(raw, want_raw, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(lever, axis=1), m.norm_ref,
                               rtol=1e-12)
    np.testing.assert_allclose(lever, rows * (m.norm_ref / want_raw)[:, None],
                               rtol=1e-12, atol=1e-15)
    assert isinstance(raw, list) and all(type(r) is float for r in raw)


def test_absolute_code_is_Vt_of_the_offset_free_output_rounded_to_4dp(
        any_map: T.LoomMap) -> None:
    """code = round(Vt @ (flat − mu_y), 4). The 4-dp rounding is a real lossy
    step (the export's meta "full_rank" note says so); stored codes in every
    receipt are these rounded values."""
    m = any_map
    x = fan_inputs(m, 1)[0]
    _, _, code = m.lever_of_input(x)
    assert isinstance(code, list) and len(code) == m.rank
    assert code == [round(float(c), 4) for c in exact_code(m, x)]


def test_lever_of_is_lever_of_input_of_input_of(any_map: T.LoomMap) -> None:
    """`lever_of(sig, bins)` ≡ `lever_of_input(input_of(sig, bins))` — the
    split that lets the contrast be built from exactly the absolute lever's
    rows."""
    m = any_map
    sig, brow = fx.raw_rows(1)[0]
    a = m.lever_of(sig, brow)
    b = m.lever_of_input(m.input_of(sig, brow))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


# ── code ↔ lever ─────────────────────────────────────────────────────────────


def test_absolute_code_round_trips_to_the_same_lever(any_map: T.LoomMap) -> None:
    """lever_from_code(Vt·(flat − mu_y), differential=False) reconstructs the
    absolute lever exactly, because Vt is orthonormal and flat − mu_y lies in
    W's row space (see ``LoomMap.lever_from_code``). This is
    what makes a stored code wearable later."""
    m = any_map
    x = fan_inputs(m, 1)[0]
    lever, raw, _ = m.lever_of_input(x)
    lever2, raw2 = m.lever_from_code(exact_code(m, x), differential=False)
    tol = round_trip_tol(m)
    np.testing.assert_allclose(lever2, lever, **tol)
    np.testing.assert_allclose(raw2, raw, rtol=tol["rtol"])


def test_code_to_lever_to_code_is_the_identity(any_map: T.LoomMap) -> None:
    """Vt Vtᵀ = I: any code in R^rank maps to a lever whose code is itself."""
    m = any_map
    c = np.random.default_rng(3).standard_normal(m.rank)
    lever, raw = m.lever_from_code(c, differential=True)
    flat = fx.raw_flat(lever, raw, m.norm_ref)
    np.testing.assert_allclose(m.Vt @ flat, c, rtol=0, atol=1e-12)


def test_differential_does_not_add_mu_y_back(wide: T.LoomMap) -> None:
    """differential=True: flat = code @ Vt; False: + mu_y. mu_y cancels in a code difference, so adding it back would put the
    map mean into every contrast wear. Needs a map with mu_y != 0 to show."""
    c = np.random.default_rng(4).standard_normal(wide.rank)
    d_lev, d_raw = wide.lever_from_code(c, differential=True)
    a_lev, a_raw = wide.lever_from_code(c, differential=False)
    np.testing.assert_allclose(fx.raw_flat(d_lev, d_raw, wide.norm_ref),
                               c @ wide.Vt, atol=1e-12)
    np.testing.assert_allclose(fx.raw_flat(a_lev, a_raw, wide.norm_ref),
                               c @ wide.Vt + wide.mu_y, atol=1e-12)


def test_lever_from_code_accepts_lists_and_any_shape_of_rank_elements(
        v1a: T.LoomMap) -> None:
    """The code is `np.asarray(code).ravel()` — a JSON list from a receipt, a
    [1, r] array and a flat array are the same code."""
    c = np.random.default_rng(5).standard_normal(v1a.rank)
    a = v1a.lever_from_code(c.tolist(), differential=True)
    b = v1a.lever_from_code(c.reshape(1, -1), differential=True)
    np.testing.assert_array_equal(a[0], b[0])


@pytest.mark.parametrize("size", [fx.RANK - 1, fx.RANK + 1, 0])
def test_a_code_of_the_wrong_rank_is_refused(v1a: T.LoomMap, size: int) -> None:
    """A code of the wrong length is refused before anything else: the rank
    is the first identity check a code gets."""
    with pytest.raises(ValueError, match="is this code from a different map"):
        v1a.lever_from_code(np.ones(size), differential=False)


def test_lever_from_code_returns_raw_norms_not_the_ruler(v1a: T.LoomMap) -> None:
    """The raw norms are the map-space loudness of the code (for a
    differential code: the contrastive loudness the reachability triage
    tracked) — reported, never the norm_ref values."""
    c = np.random.default_rng(6).standard_normal(v1a.rank) * 3.0
    lever, raw = v1a.lever_from_code(c, differential=True)
    want = np.linalg.norm((c @ v1a.Vt).reshape(v1a.n_sites, v1a.hidden), axis=1)
    np.testing.assert_allclose(raw, want, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(lever, axis=1), v1a.norm_ref,
                               rtol=1e-12)


def test_a_zero_code_is_refused_rather_than_amplified(v1a: T.LoomMap) -> None:
    """A ~zero row (≤ 1e-9) refuses: norm-matching noise up to full loudness
    would wear a random direction at the operator's α."""
    with pytest.raises(ValueError, match="~zero lever row"):
        v1a.lever_from_code(np.zeros(v1a.rank), differential=True)


# ── fan contrast ─────────────────────────────────────────────────────────────


def test_contrast_lever_is_W_of_member_minus_rest_mean(any_map: T.LoomMap) -> None:
    """contrast_lever_of(xs, i) = ruler(W·(x_i − mean_{j≠i} x_j)); mu_in cancels
    and mu_y is not added. This is the object v1a was fit on
    (``pleroma.map.build.join.fan_center``)."""
    m = any_map
    xs = fan_inputs(m, 5)
    for i in range(5):
        lever, raw, code = m.contrast_lever_of(xs, i)
        rest = np.mean([xs[j] for j in range(5) if j != i], axis=0)
        flat = (xs[i] - rest) @ m.W
        np.testing.assert_allclose(fx.raw_flat(lever, raw, m.norm_ref), flat,
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(lever, axis=1), m.norm_ref,
                                   rtol=1e-12)
        assert code == [round(float(c), 4) for c in m.Vt @ flat]


def test_contrast_lever_equals_lever_from_code_of_the_contrastive_code(
        any_map: T.LoomMap) -> None:
    """THE two-ways identity: the fan contrast in lever space equals
    lever_from_code(code_i − mean(code_rest), differential=True) in code space
    (contrast_lever_of docstring; v1a_export 'deployment_identity';
    test_loom_lever_kind). With unrounded codes the match is to float roundoff
    (float32 on the wide form, see round_trip_tol); with the 4-dp STORED codes
    it is to rounding."""
    m = any_map
    xs = fan_inputs(m, 6)
    codes = np.array([exact_code(m, x) for x in xs])
    stored = [m.lever_of_input(x)[2] for x in xs]
    for i in range(len(xs)):
        lever, raw, _ = m.contrast_lever_of(xs, i)
        c_i = codes[i] - np.delete(codes, i, axis=0).mean(axis=0)
        lev2, raw2 = m.lever_from_code(c_i, differential=True)
        tol = round_trip_tol(m)
        np.testing.assert_allclose(lev2, lever, **tol)
        np.testing.assert_allclose(raw2, raw, rtol=tol["rtol"])
        lev3, _ = m.lever_from_code(T.contrastive_loom_probe(stored, i),
                                    differential=True)
        cos = float((lev3 * lever).sum()
                    / (np.linalg.norm(lev3) * np.linalg.norm(lever)))
        assert cos > 1 - 1e-6


def test_absolute_minus_contrast_is_the_fan_common_offset(
        any_map: T.LoomMap) -> None:
    """Before the ruler: absolute_i − contrast_i = (mean_{j≠i} x_j − mu_in)·W
    + mu_y — the fan-common offset. For v1a (mu = 0) it is W·x̄_rest, measured
    at median .87 of the served lever's norm on live 70B fans. This
    decomposition is why the two kinds differ."""
    m = any_map
    xs = fan_inputs(m, 4)
    for i in range(4):
        a_lev, a_raw, _ = m.lever_of_input(xs[i])
        c_lev, c_raw, _ = m.contrast_lever_of(xs, i)
        rest = np.mean([xs[j] for j in range(4) if j != i], axis=0)
        offset = (rest - m.mu_in) @ m.W + m.mu_y
        np.testing.assert_allclose(
            fx.raw_flat(a_lev, a_raw, m.norm_ref)
            - fx.raw_flat(c_lev, c_raw, m.norm_ref),
            offset, rtol=1e-9, atol=1e-11)


def test_contrast_levers_of_a_fan_sum_to_zero_before_the_ruler(
        v1a: T.LoomMap) -> None:
    """Σ_i (x_i − mean_{j≠i} x_j) = 0, so the raw contrast levers of one fan
    cancel exactly — a structural check that the rest-mean is over the OTHER
    k−1 members, not all k."""
    xs = fan_inputs(v1a, 5)
    total = sum(fx.raw_flat(*v1a.contrast_lever_of(xs, i)[:2], v1a.norm_ref)
                for i in range(5))
    np.testing.assert_allclose(total, 0.0, atol=1e-10)


def test_the_contrast_ignores_mu_in_and_mu_y(tmp_path: Path, disc) -> None:
    """Two maps identical but for mu_in/mu_y give the SAME contrast lever:
    both offsets cancel in the member-minus-rest difference."""
    base = fx.write_v1a_map(tmp_path / "a", disc[1])
    with np.load(base) as z:
        arrays = {k: z[k] for k in z.files}
    rng = np.random.default_rng(11)
    arrays["mu_in"] = rng.standard_normal(fx.N_IN)
    arrays["mu_y"] = rng.standard_normal(fx.N_OUT)
    np.savez(tmp_path / "shifted.npz", **arrays)
    m0 = T.LoomMap(base, disc[0])
    m1 = T.LoomMap(tmp_path / "shifted.npz", disc[0])
    xs = fan_inputs(m0, 4)
    a, b = m0.contrast_lever_of(xs, 2), m1.contrast_lever_of(xs, 2)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("bad", ["k1", "pos_hi", "pos_neg", "ragged"])
def test_contrast_refuses_malformed_fans(v1a: T.LoomMap, bad: str) -> None:
    """k < 2, pos out of range and mixed row widths all refuse (ValueError)
    rather than wear something."""
    xs = fan_inputs(v1a, 3)
    args = {"k1": (xs[:1], 0), "pos_hi": (xs, 3), "pos_neg": (xs, -1),
            "ragged": (xs[:2] + [xs[2][:-1]], 0)}[bad]
    with pytest.raises(ValueError):
        v1a.contrast_lever_of(*args)


def test_a_member_identical_to_the_rest_refuses(v1a: T.LoomMap) -> None:
    """A member equal to the mean of the others has a zero contrast; the loom
    refuses rather than norm-matching noise to full loudness."""
    x = fan_inputs(v1a, 1)[0]
    with pytest.raises(ValueError, match="~zero lever row"):
        v1a.contrast_lever_of([x, x, x], 0)


# ── foreign and corrupted inputs ─────────────────────────────────────────────


def test_a_code_from_a_foreign_map_of_the_same_rank_is_refused(
        tmp_path: Path, disc) -> None:
    """A code produced by map B is not wearable on map A even when both are
    rank r: the code carries its map's fingerprint and a mismatch refuses.
    Without that check the foreign code would reconstruct an unrelated
    direction."""
    a = T.LoomMap(fx.write_v1a_map(tmp_path / "a", disc[1], seed=1), disc[0])
    b = T.LoomMap(fx.write_v1a_map(tmp_path / "b", disc[1], seed=2), disc[0])
    sig, brow = fx.raw_rows(1)[0]
    lever_b, _, code_b = b.lever_of(sig, brow)
    with pytest.raises(ValueError):
        a.lever_from_code(code_b, differential=False)


def test_the_foreign_code_really_wears_a_different_direction(
        tmp_path: Path, disc) -> None:
    """Why the fingerprint check exists (pinned as-is): stripped of its map id,
    the foreign same-rank code is ACCEPTED and its lever on map A is far from
    what it meant on map B (cos .068 between two real maps)."""
    a = T.LoomMap(fx.write_v1a_map(tmp_path / "a", disc[1], seed=1), disc[0])
    b = T.LoomMap(fx.write_v1a_map(tmp_path / "b", disc[1], seed=2), disc[0])
    sig, brow = fx.raw_rows(1)[0]
    lever_b, _, code_b = b.lever_of(sig, brow)
    # stripped of its map id (as a bare JSON list would be) it is accepted —
    # which is why every code must travel with its fingerprint
    lever_a, _ = a.lever_from_code(list(code_b), differential=False)
    cos = float((lever_a * lever_b).sum()
                / (np.linalg.norm(lever_a) * np.linalg.norm(lever_b)))
    assert abs(cos) < 0.9


def test_a_non_finite_signature_is_refused_by_the_absolute_lever(
        v1a: T.LoomMap) -> None:
    """The absolute path refuses a non-finite row exactly as
    contrast_lever_of does — the two kinds share conventions EXACTLY. Accepted,
    the NaN would flow into the lever and raw norms and be injected."""
    sig, brow = fx.raw_rows(1)[0]
    sig[0] = np.nan
    with pytest.raises(ValueError):
        v1a.lever_of(sig, brow)


def test_a_non_finite_code_is_refused(v1a: T.LoomMap) -> None:
    """/wear_code with a corrupted stored code (NaN from a bad receipt join)
    refuses instead of injecting NaN."""
    c = np.ones(v1a.rank)
    c[0] = np.nan
    with pytest.raises(ValueError):
        v1a.lever_from_code(c, differential=True)
