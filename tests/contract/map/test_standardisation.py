"""Contract: map-input standardisation (`z_corpus_of`, `input_of`).

This is where a silent bug is easiest to make: standardising an input that is
already z_corpus a second time (the shelf-baseline double standardisation)
produces plausible numbers in the wrong space. These tests pin the formulas
exactly, including the dead-dim and degenerate-scale policies.
"""

from __future__ import annotations

import numpy as np

from . import _fixtures as fx
from . import targets as T


def test_z_corpus_is_full_stage_then_corpus_stage_with_dead_dims_zeroed(
        v1a: T.LoomMap, disc) -> None:
    """z_corpus = ((sig − FULL_mean)/FULL_scale − v3_mu)/v3_sd, where a FULL
    dim with scale < 1e-12 is divided by 1 and zeroed, and a v3_dead dim is
    zeroed after the corpus stage. This is the space the map was fitted in;
    applying it twice is the double-standardisation bug."""
    with np.load(disc[0]) as d:
        full_mean = np.asarray(d["FULL_mean"], dtype=np.float64)
        full_scale = np.asarray(d["FULL_scale"], dtype=np.float64)
    sig = np.arange(1.0, fx.Z_DIM + 1.0) * 1.7
    degenerate = full_scale < 1e-12
    zf = (sig - full_mean) / np.where(degenerate, 1.0, full_scale)
    zf[degenerate] = 0.0
    v3_mu = v1a.v3_mu
    v3_sd = np.where(fx.V3_DEAD, 1.0, v1a.v3_sd)
    want = (zf - v3_mu) / v3_sd
    want[fx.V3_DEAD] = 0.0
    got = v1a.z_corpus_of(sig)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert got[fx.FULL_DEGENERATE_DIM] == -v3_mu[fx.FULL_DEGENERATE_DIM] / v3_sd[fx.FULL_DEGENERATE_DIM]
    assert got[fx.V3_DEAD].tolist() == [0.0]


def test_degenerate_full_dims_are_ignored_whatever_the_signature_says(
        v1a: T.LoomMap) -> None:
    """A degenerate FULL dim contributes the same constant however large the
    raw signature value is: 1e12 there moves nothing."""
    sig = np.ones(fx.Z_DIM)
    loud = sig.copy()
    loud[fx.FULL_DEGENERATE_DIM] = 1e12
    np.testing.assert_array_equal(v1a.z_corpus_of(sig), v1a.z_corpus_of(loud))


def test_dead_v3_dims_are_zero_even_with_zero_sd_and_no_warning(
        v1a: T.LoomMap) -> None:
    """The fixture's dead dim has sd exactly 0 (fit_loom_map marks sd < 1e-8
    dead). The loader divides dead dims by 1, so no inf/nan and no warning."""
    with np.errstate(all="raise"):
        z = v1a.z_corpus_of(np.full(fx.Z_DIM, 5.0))
    assert np.isfinite(z).all()
    assert (z[fx.V3_DEAD] == 0.0).all()


def test_input_of_is_z_corpus_then_standardised_bins(v1a: T.LoomMap) -> None:
    """x = [z_corpus | (bins − bins_mu)/bins_sd] with bins_dead dims zeroed.
    Width = z dim + bins dim; this row is what W multiplies, for the absolute
    AND the contrast lever."""
    sig, brow = fx.raw_rows(1)[0]
    x = v1a.input_of(sig, brow)
    assert x.shape == (fx.N_IN,)
    np.testing.assert_array_equal(x[: fx.Z_DIM], v1a.z_corpus_of(sig))
    bsd = np.where(fx.BINS_DEAD, 1.0, v1a.bins_sd)
    want_b = (brow - v1a.bins_mu) / bsd
    want_b[fx.BINS_DEAD] = 0.0
    np.testing.assert_allclose(x[fx.Z_DIM:], want_b, rtol=0, atol=1e-12)


def test_input_of_does_not_mutate_its_arguments(v1a: T.LoomMap) -> None:
    """Callers reuse banked signature/bins rows across candidates; the dead-dim
    zeroing must happen on copies."""
    sig, brow = fx.raw_rows(1)[0]
    sig0, brow0 = sig.copy(), brow.copy()
    v1a.input_of(sig, brow)
    np.testing.assert_array_equal(sig, sig0)
    np.testing.assert_array_equal(brow, brow0)


def test_a_live_dim_with_tiny_sd_is_divided_not_guarded(
        tmp_path, disc) -> None:
    """CURRENT policy: only the stored dead MASK is consulted at load/apply
    time; a dim the mask calls live is divided by its sd however small. The
    thresholds live in the FIT (``pleroma.map.build.shelf``: v3 1e-8, bins
    1e-9, two different cutoffs) — pinned so no shared cutoff silently starts
    re-deciding deadness at apply time."""
    path = fx.write_v1a_map(tmp_path / "m", disc[1])
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["bins_sd"] = arrays["bins_sd"].copy()
    arrays["bins_sd"][0] = np.float32(1e-10)
    np.savez(tmp_path / "tiny.npz", **arrays)
    m = T.LoomMap(tmp_path / "tiny.npz", disc[0])
    brow = np.array([float(m.bins_mu[0]) + 1.0, 0.0, 0.0])
    x = m.input_of(np.zeros(fx.Z_DIM), brow)
    np.testing.assert_allclose(x[fx.Z_DIM], 1.0 / float(np.float32(1e-10)),
                               rtol=1e-6)


def test_the_v1a_bins_block_is_a_strict_no_op(v1a: T.LoomMap) -> None:
    """v1a's W_U carries zero rows under the bins block
    (``pleroma.map.build.export.zero_pad_factors``; export report
    `bins_block_max_code_gap: 0.0`), so ANY
    bins row gives the bit-identical lever and code."""
    sig, brow = fx.raw_rows(1)[0]
    a = v1a.lever_of(sig, brow)
    b = v1a.lever_of(sig, brow * 1e3 + 7.0)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]
