"""Contract: how a LoomMap npz is READ (``pleroma.map.loom_map.LoomMap``).

The loader decides what every downstream number means: which W, which code
basis, which ruler, and whether the frozen feature space is the one the map was
fitted in. These tests pin what the loader accepts, derives and refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from . import _fixtures as fx
from . import targets as T


def test_v1a_factored_map_loads_with_the_declared_geometry(v1a: T.LoomMap) -> None:
    """Sites come from the file, hidden is derived as mu_y.size // n_sites, rank
    from meta["rank"]. Sites must be READ from the artifact, never recomputed —
    a porter recomputing them (e.g. by a rounding convention on a deeper
    model) changes the geometry silently."""
    assert v1a.sites == fx.SITES
    assert all(type(s) is int for s in v1a.sites)
    assert v1a.n_sites == len(fx.SITES)
    assert v1a.hidden == fx.HIDDEN
    assert v1a.rank == fx.RANK
    assert v1a.W.shape == (fx.N_IN, fx.N_OUT)
    assert v1a.Vt.shape == (fx.RANK, fx.N_OUT)


def test_factored_W_is_the_product_of_the_factors_in_float64(
        v1a_path: Path, v1a: T.LoomMap) -> None:
    """With only W_U/W_S/W_Vt on disk, W = (U*S) @ Vt and Vt IS the stored
    W_Vt (no SVD recompute) — the v1a export's deployment identity depends on
    Vt being exactly the fitted basis (the export's meta "deployment_identity").
    Everything is promoted to float64 on read."""
    with np.load(v1a_path) as z:
        u, s, vt = z["W_U"], z["W_S"], z["W_Vt"]
    np.testing.assert_array_equal(v1a.Vt, vt[: fx.RANK])
    np.testing.assert_allclose(v1a.W, (u * s) @ vt, rtol=0, atol=1e-14)
    for arr in (v1a.W, v1a.Vt, v1a.mu_in, v1a.mu_y, v1a.v3_mu, v1a.v3_sd,
                v1a.bins_mu, v1a.bins_sd, v1a.norm_ref, v1a.full_mean,
                v1a.full_scale):
        assert arr.dtype == np.float64
    assert v1a.v3_dead.dtype == bool and v1a.bins_dead.dtype == bool


def test_v1a_map_has_zero_offsets_so_the_absolute_lever_is_W_x(v1a: T.LoomMap) -> None:
    """v1a's mu_in and mu_y are exactly 0, which is WHY the served absolute
    lever carries the fan-common offset W·x̄ (it is not the fan contrast)."""
    assert not v1a.mu_in.any() and not v1a.mu_y.any()


def test_code_basis_rows_are_orthonormal(v1a: T.LoomMap, wide: T.LoomMap) -> None:
    """Vt Vtᵀ = I is what makes code ↔ lever exact (see
    ``LoomMap.lever_from_code``). Holds for the float64 v1a factors and for the SVD
    the loader recomputes from a dense W."""
    for m in (v1a, wide):
        np.testing.assert_allclose(m.Vt @ m.Vt.T, np.eye(m.rank), atol=1e-12)


def test_only_norm_ref_is_worn_and_the_other_rulers_are_labels(
        v1a: T.LoomMap) -> None:
    """A map may ship more than one ruler; `norm_ref` is the ONLY one worn and
    `norm_ref_*` siblings are served as labels: which ruler is in force
    decides what alpha MEANS. `norm_ref_which` is read as text and excluded
    from the alternatives."""
    np.testing.assert_array_equal(v1a.norm_ref, fx.NORM_REF)
    assert v1a.norm_ref_alternatives == {"norm_ref_v1a_own": fx.NORM_REF_OWN}
    assert v1a.norm_ref_which == fx.NORM_REF_WHICH


def test_a_map_without_ruler_labels_reads_as_none(wide: T.LoomMap) -> None:
    """The wide map (fit_loom_map) carries no norm_ref_which / alternatives;
    the loader reports None / {} rather than inventing one."""
    assert wide.norm_ref_which is None
    assert wide.norm_ref_alternatives == {}


def test_meta_is_parsed_json(v1a: T.LoomMap) -> None:
    """`meta` is a JSON string array in the npz; the loader exposes it parsed.
    rank and discriminants_sha256 are the two keys the loader itself reads."""
    assert v1a.meta["rank"] == fx.RANK
    assert isinstance(v1a.meta["discriminants_sha256"], str)


def test_a_map_with_neither_W_nor_factors_is_refused(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """KeyError naming both accepted forms."""
    src = fx.write_v1a_map(tmp_path / "m", disc[1])
    with np.load(src) as z:
        keep = {k: z[k] for k in z.files if not k.startswith("W_")}
    bad = tmp_path / "bad.npz"
    np.savez(bad, **keep)
    with pytest.raises(KeyError, match="neither 'W' nor 'W_U/W_S/W_Vt'"):
        T.LoomMap(bad, disc[0])


@pytest.mark.parametrize("missing", ["mu_in", "v3_dead", "bins_sd", "norm_ref",
                                     "sites", "meta"])
def test_a_map_missing_a_required_array_does_not_load(
        tmp_path: Path, disc: tuple[Path, str], missing: str) -> None:
    """Every array the loader reads is required; a missing one raises KeyError
    at load (numpy's npz lookup), never a half-initialised map. (No schema
    object names the keys; the lookup is the check.)"""
    src = fx.write_v1a_map(tmp_path / "m", disc[1])
    with np.load(src) as z:
        keep = {k: z[k] for k in z.files if k != missing}
    bad = tmp_path / "bad.npz"
    np.savez(bad, **keep)
    with pytest.raises(KeyError):
        T.LoomMap(bad, disc[0])


def test_discriminants_from_a_different_frozen_space_are_refused(
        tmp_path: Path, v1a_path: Path) -> None:
    """The sha pin marries the map to the discriminants it was fitted in
    — the map's banked v3_mu/v3_sd are only meaningful composed with those
    exact FULL_mean/FULL_scale. Same shapes, different bytes -> ValueError."""
    other, _ = fx.write_discriminants(tmp_path / "other", seed=99)
    with pytest.raises(ValueError, match="discriminants sha mismatch"):
        T.LoomMap(v1a_path, other)


def test_dense_only_map_takes_its_code_basis_from_its_own_svd(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """With only a dense W on disk, Vt is the first `rank` rows of
    np.linalg.svd(W) in float64. Codes served by such a map live in THAT
    basis — the sign convention of numpy's SVD is part of the contract for
    every code banked against it."""
    path = fx.write_wide_map(tmp_path / "w", disc[1], store="dense")
    m = T.LoomMap(path, disc[0])
    with np.load(path) as z:
        w = np.asarray(z["W"], dtype=np.float64)
    np.testing.assert_array_equal(m.W, w)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    np.testing.assert_array_equal(m.Vt, vt[: m.rank])


def test_when_both_forms_are_stored_the_dense_W_is_the_one_applied(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """Every fit_loom_map wide map ships BOTH dense W and float32 factors; the
    loader applies the dense W (not the factor product). Which BASIS codes are
    served in is pinned below."""
    path = fx.write_wide_map(tmp_path / "w", disc[1], store="both")
    m = T.LoomMap(path, disc[0])
    with np.load(path) as z:
        w = np.asarray(z["W"], dtype=np.float64)
    np.testing.assert_array_equal(m.W, w)


def test_dense_only_map_spans_the_same_row_space_as_its_factors(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """A dense-only map and the factored form of the same W agree on W and on
    the SUBSPACE of the code basis (projector Vtᵀ Vt), even though individual
    basis rows may differ in sign. Levers are basis-independent; codes are
    not — see the stored-basis test below."""
    dense = T.LoomMap(fx.write_wide_map(tmp_path / "d", disc[1], store="dense"),
                      disc[0])
    fact = T.LoomMap(fx.write_wide_map(tmp_path / "f", disc[1], store="factors"),
                     disc[0])
    np.testing.assert_allclose(dense.W, fact.W, atol=1e-5)
    np.testing.assert_allclose(dense.Vt.T @ dense.Vt, fact.Vt.T @ fact.Vt,
                               atol=1e-5)


def test_a_map_whose_stored_factors_disagree_with_its_dense_W_is_refused_or_honoured(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """A map carrying both W and W_Vt either refuses to load when they are not
    one W, or serves codes in the STORED basis (so a code banked against W_Vt
    reconstructs its own lever) — never a silently recomputed basis."""
    path = fx.write_wide_map(tmp_path / "w", disc[1], store="both",
                             flip_factor_row=0)
    try:
        m = T.LoomMap(path, disc[0])
    except ValueError:
        return
    with np.load(path) as z:
        stored_vt = np.asarray(z["W_Vt"], dtype=np.float64)
    np.testing.assert_allclose(m.Vt, stored_vt[: m.rank], atol=1e-5)


def test_meta_rank_not_factor_width_sets_the_code_size(
        tmp_path: Path, disc: tuple[Path, str]) -> None:
    """`rank` is meta["rank"] and Vt is the first `rank` stored rows.
    A meta rank below the factor width truncates the
    code basis while W still uses every factor — pinned as current behaviour
    (the fit writers keep the two equal)."""
    path = fx.write_v1a_map(tmp_path / "m", disc[1], rank=fx.RANK)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["rank"] = fx.RANK - 1
    arrays["meta"] = np.array(json.dumps(meta))
    trimmed = tmp_path / "trimmed.npz"
    np.savez(trimmed, **arrays)
    m = T.LoomMap(trimmed, disc[0])
    assert m.rank == fx.RANK - 1
    assert m.Vt.shape == (fx.RANK - 1, fx.N_OUT)
    np.testing.assert_allclose(
        m.W, (arrays["W_U"] * arrays["W_S"]) @ arrays["W_Vt"], atol=1e-14)
