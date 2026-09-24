"""Contract: the per-site norm-matching ruler and every path that applies it.

The ruler rescales each site row of a lever to the map's `norm_ref[s]` so that
α means the same absolute per-site norm for every lever kind: α on its own is
not a unit, the absolute per-site norm is. The loom applies it through three
paths — two LoomMap methods and LeverBank — all sharing `pleroma.levers.ruler`.

The table below pins what EACH path does on the same edge rows, so any change
to the ruler's edge policy is visible cell by cell rather than discovered
later. When a path is deleted, delete its row. Non-finite handling on the
LoomMap paths is pinned in test_levers.py, so NaN is not tabulated for them
here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from . import _fixtures as fx
from . import targets as T

NORM_REF = np.asarray(fx.NORM_REF, dtype=np.float64)

Outcome = str  # "matched" | "zeros" | "nan" | "raise:<ExceptionType>"


def edge_lever(case: str) -> tuple[np.ndarray, np.ndarray]:
    """[S, H] lever whose site-0 row is the edge case; sites 1.. are ordinary.
    Returns (lever, norm_ref)."""
    rng = np.random.default_rng(42)
    lever = rng.standard_normal((len(fx.SITES), fx.HIDDEN)) * 3.0
    unit = lever[0] / np.linalg.norm(lever[0])
    ref = NORM_REF.copy()
    if case == "normal":
        lever[0] = unit * 3.0
    elif case == "zero":
        lever[0] = 0.0
    elif case == "tiny_1e-10":
        lever[0] = unit * 1e-10
    elif case == "tiny_1e-13":
        lever[0] = unit * 1e-13
    elif case == "nan":
        lever[0] = unit
        lever[0, 0] = np.nan
    elif case == "zero_target":
        ref[0] = 0.0
    else:  # pragma: no cover
        raise AssertionError(case)
    return lever, ref


# ── adapters: every path as f(lever, norm_ref, workdir) -> ruled lever ───────


def via_lever_of(lever: np.ndarray, ref: np.ndarray, work: Path) -> np.ndarray:
    """LoomMap.lever_of's ruler: a map that can never write site 0
    (zero Vt columns) so site 0's output is exactly mu_y's site-0 block."""
    disc, sha = fx.write_discriminants(work / "d")
    mu_y = np.zeros(fx.N_OUT)
    mu_y[: fx.HIDDEN] = lever[0]
    m = T.LoomMap(fx.write_v1a_map(work / "m", sha, zero_site=0, mu_y=mu_y,
                                   norm_ref=ref.tolist()), disc)
    sig, brow = fx.raw_rows(1)[0]
    return m.lever_of(sig, brow)[0]


def via_lever_from_code(lever: np.ndarray, ref: np.ndarray, work: Path) -> np.ndarray:
    """LoomMap.lever_from_code(differential=False)'s ruler, same map."""
    disc, sha = fx.write_discriminants(work / "d")
    mu_y = np.zeros(fx.N_OUT)
    mu_y[: fx.HIDDEN] = lever[0]
    m = T.LoomMap(fx.write_v1a_map(work / "m", sha, zero_site=0, mu_y=mu_y,
                                   norm_ref=ref.tolist()), disc)
    return m.lever_from_code(np.ones(m.rank), differential=False)[0]


def via_lever_bank(lever: np.ndarray, ref: np.ndarray, work: Path) -> np.ndarray:
    """LeverBank._match_norms through the public vectors_for (bank levers are
    stored float32 and returned float32)."""
    disc, sha = fx.write_discriminants(work / "d")
    m = T.LoomMap(fx.write_v1a_map(work / "m", sha, norm_ref=ref.tolist()), disc)
    bank = T.LeverBank(fx.write_lever_bank(work / "bank.npz", lever[None]), m)
    return bank.vectors_for("p000|orig")[0]


COPIES: dict[str, Callable[[np.ndarray, np.ndarray, Path], np.ndarray]] = {
    "loommap.lever_of": via_lever_of,
    "loommap.lever_from_code": via_lever_from_code,
    "loommap.LeverBank": via_lever_bank,
}

CASES = ["normal", "zero", "tiny_1e-10", "tiny_1e-13", "nan", "zero_target"]

#: What each path does with the site-0 edge row (the zero-row policy, measured).
#: None = not tabulated here (pinned in test_levers.py).
EXPECTED: dict[str, dict[str, Outcome | None]] = {
    # ── the loom's three paths share pleroma.levers.ruler: non-finite or
    # n <= 1e-9 refuses. A check of only n <= 0 would AMPLIFY a 1e-10 row to
    # full loudness, and a bank without the finiteness check would pass NaN
    # through.
    "loommap.lever_of": {
        "normal": "matched", "zero": "raise:ValueError",
        "tiny_1e-10": "raise:ValueError", "tiny_1e-13": "raise:ValueError",
        "nan": None, "zero_target": "zeros"},
    "loommap.lever_from_code": {
        "normal": "matched", "zero": "raise:ValueError",
        "tiny_1e-10": "raise:ValueError", "tiny_1e-13": "raise:ValueError",
        "nan": None, "zero_target": "zeros"},
    "loommap.LeverBank": {
        "normal": "matched", "zero": "raise:ValueError",
        "tiny_1e-10": "raise:ValueError", "tiny_1e-13": "raise:ValueError",
        "nan": "raise:ValueError", "zero_target": "zeros"},
}


def outcome(fn: Callable[[np.ndarray, np.ndarray, Path], np.ndarray],
            case: str, work: Path) -> Outcome:
    lever, ref = edge_lever(case)
    try:
        with np.errstate(all="ignore"):
            out = np.asarray(fn(lever, ref, work), dtype=np.float64)
    except (ValueError, SystemExit, ZeroDivisionError) as exc:
        return f"raise:{type(exc).__name__}"
    assert out.shape == lever.shape
    # the ordinary rows must be matched whatever happened to row 0
    np.testing.assert_allclose(np.linalg.norm(out[1:], axis=1), ref[1:],
                               rtol=1e-6)
    row = out[0]
    if np.isnan(row).any():
        return "nan"
    if not row.any():
        return "zeros"
    np.testing.assert_allclose(np.linalg.norm(row), ref[0], rtol=1e-5)
    return "matched"


@pytest.mark.parametrize("copy", list(COPIES))
@pytest.mark.parametrize("case", CASES)
def test_each_ruler_copy_handles_the_edge_row_as_pinned(
        tmp_path: Path, copy: str, case: str) -> None:
    """Per-path, per-edge-case behaviour. Every path refuses both a ZERO row
    and a 1e-10 row: silently zeroing the first drops a site from the wear,
    and amplifying the second wears noise at full loudness."""
    want = EXPECTED[copy][case]
    if want is None:
        pytest.skip("non-finite on the LoomMap paths is pinned in "
                    "test_levers.py")
    assert outcome(COPIES[copy], case, tmp_path) == want


def test_every_copy_agrees_on_an_ordinary_lever(tmp_path: Path) -> None:
    """On well-conditioned input all paths are the same function: each row
    scaled to norm_ref[s], direction preserved (float32 for LeverBank,
    float64 otherwise)."""
    lever, ref = edge_lever("normal")
    want = lever * (ref / np.linalg.norm(lever, axis=1))[:, None]
    for name in [n for n in COPIES if n not in ("loommap.lever_of", "loommap.lever_from_code")]:
        got = COPIES[name](lever, ref, tmp_path / name)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6,
                                   err_msg=name)


def test_output_dtypes_differ_by_copy(tmp_path: Path) -> None:
    """LeverBank returns float32; the LoomMap methods return float64. Pinned
    because a consolidated ruler has to pick one and the vLLM hooks cast
    anyway."""
    lever, ref = edge_lever("normal")
    dtypes = {name: np.asarray(fn(lever, ref, tmp_path / name)).dtype
              for name, fn in COPIES.items()}
    assert dtypes == {name: DTYPES[name] for name in COPIES}


#: each path's output dtype
DTYPES: dict[str, type] = {
    "loommap.lever_of": np.float64,
    "loommap.lever_from_code": np.float64,
    "loommap.LeverBank": np.float32,
}


def test_the_ruler_makes_alpha_an_absolute_per_site_norm(v1a: T.LoomMap) -> None:
    """After the ruler, two DIFFERENT levers (absolute,
    contrast, from-code) carry identical per-site norms, so at equal α they
    wear identical loudness and differ only in direction."""
    xs = [v1a.input_of(s, b) for s, b in fx.raw_rows(4)]
    kinds = [v1a.lever_of_input(xs[0])[0], v1a.contrast_lever_of(xs, 0)[0],
             v1a.lever_from_code(np.arange(1.0, v1a.rank + 1), differential=True)[0]]
    for lev in kinds:
        np.testing.assert_allclose(np.linalg.norm(lev, axis=1), v1a.norm_ref,
                                   rtol=1e-12)
