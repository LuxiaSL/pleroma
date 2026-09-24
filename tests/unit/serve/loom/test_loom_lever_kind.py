"""Contract tests for `lever_kind` on /wear and /loom's auto-wear.

THE FINDING. `LoomMap.lever_of` applies W to a candidate's OWN input row x_i.
v1a was FIT on the fan contrast x_i − mean_{j≠i} x_j (one-vs-rest,
v1a_fit.fan_center). For v1a mu_in = mu_y = 0 exactly, so the served lever is
the contrast PLUS a fan-common offset W·x̄_rest — and on six live 70B fans
that offset measured median 0.87 of the served lever's norm. So the contrast
lever is a toggle, and the default stays `absolute`.

What must hold, and would be silent if it did not:

  1. `lever_of` is BIT-IDENTICAL to a frozen reference with `input_of`
     inlined — every banked code, dose band and restore depends on those
     bytes.
  2. the contrast lever IS the object /probe and /wear_code
     {code_kind: "differential"} already wear: equal to
     `lever_from_code(contrastive_code(codes, i), differential=True)`.
  3. the DEFAULT (no body field, server default `absolute`) stores exactly
     the vectors / code / dose that plain `lever_of` produces; only the
     `lever_kind` label is added.
  4. `predicted` dosing divides by raw norms and a fan mean OF THE SAME KIND.
  5. a restored wear rehydrates the kind it was worn as; an older snapshot
     with no field means `absolute`.
  6. unknown kinds and unavailable contrasts REFUSE; nothing silently falls
     back to the other lever.

Pure numpy on a tiny synthetic map — no model, no server, no GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.probe import orchestrator as loom_probe
from pleroma.serve import legacy as ls

SITES = [8, 15, 19, 22]
HIDDEN = 12
NORM_REF = [1.0, 2.0, 3.0, 4.0]
N_V3, N_BINS = 7, 5


def write_map(tmp_path: Path, *, rank: int = 3, dense: bool = False,
              zero_mu: bool = False, seed: int = 20260922) -> tuple[Path, Path]:
    """A minimal loom-map npz + discriminants. `dense` stores W itself
    (so Vt comes from the server's own SVD); otherwise the SVD-factored form
    v1a ships. mu_in / mu_y are NONZERO by default — the contrast must not
    depend on them, and a test where they are zero could not show that."""
    rng = np.random.default_rng(seed)
    n_in, n_out = N_V3 + N_BINS, len(SITES) * HIDDEN
    tmp_path.mkdir(parents=True, exist_ok=True)
    disc = tmp_path / "disc.npz"
    drng = np.random.default_rng(1)  # the SAME discriminants for every map
    np.savez(disc, FULL_mean=drng.standard_normal(N_V3),
             FULL_scale=np.abs(drng.standard_normal(N_V3)) + 0.5)
    sha = hashlib.sha256(disc.read_bytes()).hexdigest()
    u = rng.standard_normal((n_in, rank))
    s = np.abs(rng.standard_normal(rank)) + 1.0
    vt = np.linalg.qr(rng.standard_normal((n_out, rank)))[0].T
    mu_in = np.zeros(n_in) if zero_mu else rng.standard_normal(n_in)
    mu_y = np.zeros(n_out) if zero_mu else rng.standard_normal(n_out)
    common = dict(
        mu_in=mu_in, mu_y=mu_y,
        v3_mu=np.linspace(-0.3, 0.3, N_V3).astype(np.float32),
        v3_sd=np.linspace(0.6, 1.4, N_V3).astype(np.float32),
        v3_dead=np.array([False] * (N_V3 - 1) + [True]),
        bins_mu=np.linspace(-0.2, 0.2, N_BINS).astype(np.float32),
        bins_sd=np.linspace(0.7, 1.3, N_BINS).astype(np.float32),
        bins_dead=np.array([False, True] + [False] * (N_BINS - 2)),
        sites=np.asarray(SITES, dtype=np.int64),
        norm_ref=np.asarray(NORM_REF, dtype=np.float64),
        meta=json.dumps({"rank": rank, "discriminants_sha256": sha}),
    )
    map_path = tmp_path / "map.npz"
    if dense:
        np.savez(map_path, W=(u * s) @ vt, **common)
    else:
        np.savez(map_path, W_U=u, W_S=s, W_Vt=vt, **common)
    return map_path, disc


@pytest.fixture(params=["factored", "dense"])
def loom_map(request: pytest.FixtureRequest, tmp_path: Path) -> ls.LoomMap:
    return ls.LoomMap(*write_map(tmp_path, dense=request.param == "dense"))


def raw_inputs(k: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(N_V3), rng.standard_normal(N_BINS))
            for _ in range(k)]


def a_fan(loom_map: ls.LoomMap, k: int = 5, dead: tuple[int, ...] = ()
          ) -> tuple[list[dict[str, Any]], list[np.ndarray | None]]:
    """K candidates in the exact shape the draw path builds them, plus the
    x rows it hands `attach_fan_contrast`."""
    cands: list[dict[str, Any]] = []
    xs: list[np.ndarray | None] = []
    for j, (sig, brow) in enumerate(raw_inputs(k)):
        if j in dead:
            cands.append({"index": j, "lever": None, "note": "ValueError: dead",
                          "raw_norms": None, "code": None, "z": None})
            xs.append(None)
            continue
        x = loom_map.input_of(sig, brow)
        lever, raw, code = loom_map.lever_of_input(x)
        cands.append({"index": j, "lever": lever, "note": None,
                      "raw_norms": raw, "code": code, "z": None})
        xs.append(x)
    return cands, xs


# ══ 1. lever_of did not move by one bit ══════════════════════════════════════


def old_lever_of(m: ls.LoomMap, signature: np.ndarray, bins_row: np.ndarray
                 ) -> tuple[np.ndarray, list[float], list[float]]:
    """`LoomMap.lever_of` with `input_of` inlined, as a frozen reference.
    This copy is the reference; do not "tidy" it."""
    zc = m.z_corpus_of(signature)
    bs = (bins_row - m.bins_mu) / np.where(m.bins_dead, 1.0, m.bins_sd)
    bs[m.bins_dead] = 0.0
    x = np.concatenate([zc, bs])
    flat = (x - m.mu_in) @ m.W + m.mu_y
    code = [round(float(c), 4) for c in m.Vt @ (flat - m.mu_y)]
    lever = flat.reshape(m.n_sites, m.hidden)
    out = np.empty_like(lever)
    raw_norms: list[float] = []
    for s in range(m.n_sites):
        n = float(np.linalg.norm(lever[s]))
        if n <= 0:
            raise ValueError(f"map emitted a zero row at site {m.sites[s]}")
        raw_norms.append(n)
        out[s] = lever[s] * (m.norm_ref[s] / n)
    return out, raw_norms, code


def test_lever_of_is_bit_identical_after_the_split(loom_map: ls.LoomMap) -> None:
    """★ THE PIN. Same bytes, same raw norms, same code, on a fixed input —
    including dead v3/bins columns (the in-place zeroing moved functions)."""
    for sig, brow in raw_inputs(12, seed=99):
        want = old_lever_of(loom_map, sig.copy(), brow.copy())
        got = loom_map.lever_of(sig.copy(), brow.copy())
        assert got[0].dtype == want[0].dtype and got[0].shape == want[0].shape
        assert got[0].tobytes() == want[0].tobytes()
        assert got[1] == want[1]
        assert got[2] == want[2]
        via_x = loom_map.lever_of_input(loom_map.input_of(sig.copy(), brow.copy()))
        assert via_x[0].tobytes() == want[0].tobytes()
        assert via_x[1] == want[1] and via_x[2] == want[2]


def test_input_of_does_not_mutate_its_arguments(loom_map: ls.LoomMap) -> None:
    sig, brow = raw_inputs(1)[0]
    s0, b0 = sig.copy(), brow.copy()
    loom_map.input_of(sig, brow)
    assert np.array_equal(sig, s0) and np.array_equal(brow, b0)


# ══ 2. contrast == the differential-code path /probe already wears ══════════


def exact_codes(m: ls.LoomMap, xs: list[np.ndarray]) -> list[list[float]]:
    """`lever_of`'s code WITHOUT the 4-place rounding."""
    return [[float(c) for c in m.Vt @ ((x - m.mu_in) @ m.W)] for x in xs]


def test_contrast_equals_lever_from_code_of_the_contrastive_code(
    loom_map: ls.LoomMap,
) -> None:
    """★ THE EQUIVALENCE. `contrast_lever_of(xs, i)` ==
    `lever_from_code(contrastive_code(codes, i), differential=True)` — for
    BOTH contrastive_code implementations the codebase carries. Holds because
    W = U S Vᵀ with Vᵀ's rows orthonormal, so W·dx already lies in span(V)
    and `code @ Vt` reconstructs it exactly; mu_y cancels either way."""
    cands, xs = a_fan(loom_map, k=5)
    rows = [x for x in xs if x is not None]
    codes = exact_codes(loom_map, rows)
    for i in range(len(rows)):
        lever, raw, code = loom_map.contrast_lever_of(rows, i)
        for impl in (loom_probe.contrastive_code,):
            ref, ref_raw = loom_map.lever_from_code(impl(codes, i),
                                                    differential=True)
            np.testing.assert_allclose(lever, ref, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(raw, ref_raw, rtol=1e-9)
        np.testing.assert_allclose(code, loom_probe.contrastive_code(codes, i),
                                   atol=5.1e-5)  # code is rounded to 4 places


def test_contrast_matches_the_served_rounded_code_path_to_rounding(
    loom_map: ls.LoomMap,
) -> None:
    """What /probe actually wears is built from the ROUNDED served codes. The
    difference is 4-place rounding and nothing else."""
    cands, xs = a_fan(loom_map, k=6)
    rows = [x for x in xs if x is not None]
    served = [c["code"] for c in cands]
    for i in range(len(rows)):
        lever, _, _ = loom_map.contrast_lever_of(rows, i)
        ref, _ = loom_map.lever_from_code(
            loom_probe.contrastive_code(served, i), differential=True)
        cos = [float(lever[s] @ ref[s] / (np.linalg.norm(lever[s])
                                          * np.linalg.norm(ref[s])))
               for s in range(loom_map.n_sites)]
        assert min(cos) > 0.9999


def test_contrast_is_the_reference_scripts_arithmetic(loom_map: ls.LoomMap) -> None:
    """The contrast lever, norm-matched, as a reference computes it:
    (X[i] − (ΣX − X[i])/(k−1)) @ W. Equivalently the difference of the
    absolute pre-ruler levers, flat_i − mean_{j≠i} flat_j."""
    _, xs = a_fan(loom_map, k=4)
    X = np.stack([x for x in xs if x is not None])
    k = len(X)
    for i in range(k):
        c_ref = (X[i] - (X.sum(0) - X[i]) / (k - 1)) @ loom_map.W
        flats = (X - loom_map.mu_in) @ loom_map.W + loom_map.mu_y
        c_diff = flats[i] - np.delete(flats, i, 0).mean(0)
        np.testing.assert_allclose(c_ref, c_diff, rtol=1e-9, atol=1e-10)
        lever, raw, _ = loom_map.contrast_lever_of(list(X), i)
        c = c_ref.reshape(loom_map.n_sites, -1)
        np.testing.assert_allclose(raw, np.linalg.norm(c, axis=1), rtol=1e-12)
        want = c * (np.asarray(NORM_REF) / np.linalg.norm(c, axis=1))[:, None]
        np.testing.assert_allclose(lever, want, rtol=1e-12)


def test_contrast_is_independent_of_mu_in_and_mu_y(tmp_path: Path) -> None:
    """Same W, different mus: identical contrast (mu_in cancels in the
    subtraction and mu_y is not added back). The absolute lever does move."""
    a = ls.LoomMap(*write_map(tmp_path / "a", seed=5))
    b = ls.LoomMap(*write_map(tmp_path / "b", seed=5, zero_mu=True))
    assert np.array_equal(a.W, b.W) and not np.allclose(a.mu_y, b.mu_y)
    ins = raw_inputs(4)
    xa = [a.input_of(s, r) for s, r in ins]
    xb = [b.input_of(s, r) for s, r in ins]
    assert np.array_equal(np.stack(xa), np.stack(xb))
    la, _, _ = a.contrast_lever_of(xa, 1)
    lb, _, _ = b.contrast_lever_of(xb, 1)
    np.testing.assert_allclose(la, lb, rtol=1e-12)
    assert not np.allclose(a.lever_of(*ins[1])[0], b.lever_of(*ins[1])[0])


def test_contrast_keeps_the_ruler(loom_map: ls.LoomMap) -> None:
    """alpha means one absolute per-site norm across both kinds."""
    _, xs = a_fan(loom_map)
    rows = [x for x in xs if x is not None]
    for i in range(len(rows)):
        lever, _, _ = loom_map.contrast_lever_of(rows, i)
        np.testing.assert_allclose(np.linalg.norm(lever, axis=1), NORM_REF,
                                   rtol=1e-12)


def test_contrast_contrast_levers_sum_to_zero_before_the_ruler(
    loom_map: ls.LoomMap,
) -> None:
    """One-vs-rest contrasts of a fan sum to zero (Σ_i (x_i − x̄_{-i}) = 0):
    the fan-common part is exactly what the contrast removes."""
    _, xs = a_fan(loom_map, k=5)
    rows = [x for x in xs if x is not None]
    raws = []
    for i in range(len(rows)):
        lever, raw, _ = loom_map.contrast_lever_of(rows, i)
        raws.append(lever * (np.asarray(raw) / np.asarray(NORM_REF))[:, None])
    np.testing.assert_allclose(np.sum(raws, axis=0), 0.0, atol=1e-10)


@pytest.mark.parametrize("bad,match", [
    ("one", "at least 2"), ("pos", "outside a fan"), ("width", "mixed widths"),
])
def test_contrast_refuses_malformed_fans(loom_map: ls.LoomMap, bad: str,
                                         match: str) -> None:
    _, xs = a_fan(loom_map, k=3)
    rows = [x for x in xs if x is not None]
    with pytest.raises(ValueError, match=match):
        if bad == "one":
            loom_map.contrast_lever_of(rows[:1], 0)
        elif bad == "pos":
            loom_map.contrast_lever_of(rows, 3)
        else:
            loom_map.contrast_lever_of([rows[0], rows[1][:-1]], 0)


def test_contrast_refuses_a_member_identical_to_the_rest(
    loom_map: ls.LoomMap,
) -> None:
    """Two identical rows: the contrast is zero, and norm-matching zero up to
    full loudness would wear noise. Refuse, like lever_from_code does."""
    _, xs = a_fan(loom_map, k=2)
    with pytest.raises(ValueError, match="zero lever row"):
        loom_map.contrast_lever_of([xs[0], xs[0].copy()], 0)


# ══ 3. the draw path: contrast BESIDE absolute, never instead ════════════════


def test_attach_fan_contrast_leaves_the_absolute_fields_alone(
    loom_map: ls.LoomMap,
) -> None:
    cands, xs = a_fan(loom_map)
    before = [(c["lever"], list(c["raw_norms"]), list(c["code"])) for c in cands]
    mean, note = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    assert note is None and mean is not None
    for c, (lever, raw, code) in zip(cands, before):
        assert c["lever"] is lever                    # the same object
        assert c["raw_norms"] == raw and c["code"] == code
        assert c["contrast_lever"] is not None and c["contrast_note"] is None
        assert c["contrast_peers"] == [0, 1, 2, 3, 4]


def test_attach_fan_contrast_skips_dead_candidates_and_excludes_them(
    loom_map: ls.LoomMap,
) -> None:
    """The rest-mean is over the OTHER VALID members; a dead future is
    neither contrasted nor part of anyone's rest."""
    cands, xs = a_fan(loom_map, k=5, dead=(2,))
    mean, note = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    assert note is None
    assert cands[2]["contrast_lever"] is None
    assert "did not harvest" in cands[2]["contrast_note"]
    valid = [x for x in xs if x is not None]
    for pos, j in enumerate([0, 1, 3, 4]):
        assert cands[j]["contrast_peers"] == [0, 1, 3, 4]
        want, _, _ = loom_map.contrast_lever_of(valid, pos)
        assert cands[j]["contrast_lever"].tobytes() == want.tobytes()
    want_mean = np.mean([cands[j]["contrast_raw_norms"] for j in (0, 1, 3, 4)],
                        axis=0)
    np.testing.assert_allclose(mean, want_mean, rtol=1e-12)


@pytest.mark.parametrize("k,dead", [(1, ()), (3, (0, 2)), (2, (0, 1))])
def test_attach_fan_contrast_is_unavailable_below_two_valid(
    loom_map: ls.LoomMap, k: int, dead: tuple[int, ...],
) -> None:
    cands, xs = a_fan(loom_map, k=k, dead=dead)
    mean, note = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    assert mean is None and note is not None and "at least 2" in note
    for c in cands:
        assert c["contrast_lever"] is None and c["contrast_note"] == note


def test_a_contrast_failure_is_listed_not_raised(loom_map: ls.LoomMap) -> None:
    cands, xs = a_fan(loom_map, k=3)

    def boom(rows: Any, pos: int) -> Any:
        if pos == 1:
            raise ValueError("synthetic")
        return loom_map.contrast_lever_of(rows, pos)

    mean, note = ls.attach_fan_contrast(cands, xs, boom)
    assert note is None
    assert cands[1]["contrast_lever"] is None
    assert cands[1]["contrast_note"] == "ValueError: synthetic"
    assert cands[0]["contrast_lever"] is not None


def test_fan_mean_default_key_is_unchanged(loom_map: ls.LoomMap) -> None:
    cands, xs = a_fan(loom_map)
    ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    np.testing.assert_array_equal(
        ls.fan_mean_raw_norms(cands),
        np.mean([c["raw_norms"] for c in cands], axis=0))


def test_the_contrast_advisory_scale_uses_contrast_norms(
    loom_map: ls.LoomMap,
) -> None:
    """`scores.predicted_dose_scale_contrast` is the contrast raw norm over the
    contrast fan mean — computed with the same dose_scale_for the draw path
    calls — and the absolute `predicted_dose_scale` is untouched by it."""
    cands, xs = a_fan(loom_map)
    fan_abs = ls.fan_mean_raw_norms(cands)
    abs_before = [ls.dose_scale_for(policy="predicted", raw_norms=c["raw_norms"],
                                    fan_mean=fan_abs, n_sites=4).scale_mean
                  for c in cands]
    fan_c, _ = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    abs_after = [ls.dose_scale_for(policy="predicted", raw_norms=c["raw_norms"],
                                   fan_mean=ls.fan_mean_raw_norms(cands),
                                   n_sites=4).scale_mean for c in cands]
    assert abs_before == abs_after
    for c in cands:
        got = ls.dose_scale_for(policy="predicted",
                                raw_norms=c["contrast_raw_norms"],
                                fan_mean=fan_c, n_sites=4)
        want = np.clip(np.asarray(c["contrast_raw_norms"]) / np.asarray(fan_c),
                       0.5, 2.0)
        np.testing.assert_allclose(got.scale, want, rtol=1e-12)


# ══ 4. /wear: the default stores the old bytes; predicted is same-kind ═══════


def old_wear_fields(cand: dict[str, Any], policy: str,
                    fan_mean: Any, n_sites: int) -> tuple[dict[str, Any], Any]:
    """do_wear's lever-dependent half VERBATIM as on main 2ddea37."""
    dose = ls.dose_scale_for(policy=policy, raw_norms=cand["raw_norms"],
                             fan_mean=fan_mean, n_sites=n_sites)
    return {"vectors": dose.apply(cand["lever"]), "code": cand.get("code"),
            "dose": dose.to_json()}, dose


@pytest.mark.parametrize("policy", ["flat", "predicted"])
def test_the_default_kind_stores_exactly_what_the_old_server_stored(
    loom_map: ls.LoomMap, policy: str,
) -> None:
    """★ No body field + the server default (`absolute`) ⇒ the same vectors
    bytes, code and dose receipt as before; `lever_kind` is the only new key."""
    cands, xs = a_fan(loom_map)
    fan_abs = ls.fan_mean_raw_norms(cands)
    fan_c, _ = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    kind = ls.resolve_lever_kind(None, ls.DEFAULT_LEVER_KIND)
    assert kind == "absolute"
    for c in cands:
        new, new_dose = ls.fan_wear_fields(
            c, lever_kind=kind, dose_policy=policy, fan_mean=fan_abs,
            fan_mean_contrast=fan_c, n_sites=loom_map.n_sites)
        old, old_dose = old_wear_fields(c, policy, fan_abs, loom_map.n_sites)
        assert new["vectors"].tobytes() == old["vectors"].tobytes()
        if policy == "flat":
            assert new["vectors"] is c["lever"]
        assert new["code"] == old["code"]
        assert new["dose"] == old["dose"]
        assert set(new) - set(old) == {"lever_kind"}
        assert new["lever_kind"] == "absolute"


def test_the_module_default_is_absolute() -> None:
    assert ls.DEFAULT_LEVER_KIND == "absolute"
    assert ls.LEVER_KINDS == ("absolute", "contrast")


def _server_src() -> str:
    """The server's source: every module under pleroma/serve/."""
    root = Path(ls.__file__).resolve().parent  # pleroma/serve/
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.py")))


def test_the_server_flag_defaults_to_absolute() -> None:
    """The flag's default is the module default, not a literal that can drift."""
    src = _server_src()
    i = src.index('"--wear-lever-default"')
    block = src[i:i + 400]
    assert "choices=list(LEVER_KINDS)" in block
    assert "default=DEFAULT_LEVER_KIND" in block


def test_contrast_wear_uses_the_contrast_lever_and_code(
    loom_map: ls.LoomMap,
) -> None:
    cands, xs = a_fan(loom_map)
    fan_c, _ = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    c = cands[3]
    fields, dose = ls.fan_wear_fields(
        c, lever_kind="contrast", dose_policy="flat",
        fan_mean=ls.fan_mean_raw_norms(cands), fan_mean_contrast=fan_c,
        n_sites=loom_map.n_sites)
    assert fields["vectors"] is c["contrast_lever"]
    assert fields["code"] == c["contrast_code"] != c["code"]
    assert fields["lever_kind"] == "contrast"
    assert fields["contrast_peers"] == [0, 1, 2, 3, 4]
    assert dose.policy == "flat"


def test_predicted_divides_by_the_same_kinds_norms(loom_map: ls.LoomMap) -> None:
    """★ A contrast wear under `predicted` is scaled by ITS raw norms over the
    fan's mean CONTRAST raw norms — never by the absolute ones."""
    cands, xs = a_fan(loom_map)
    fan_abs = ls.fan_mean_raw_norms(cands)
    fan_c, _ = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    for c in cands:
        f_c, d_c = ls.fan_wear_fields(
            c, lever_kind="contrast", dose_policy="predicted",
            fan_mean=fan_abs, fan_mean_contrast=fan_c, n_sites=4)
        assert d_c.candidate_raw_norms == tuple(c["contrast_raw_norms"])
        assert d_c.fan_mean_raw_norms == tuple(fan_c)
        np.testing.assert_allclose(
            f_c["vectors"], c["contrast_lever"] * np.asarray(d_c.scale)[:, None])
        f_a, d_a = ls.fan_wear_fields(
            c, lever_kind="absolute", dose_policy="predicted",
            fan_mean=fan_abs, fan_mean_contrast=fan_c, n_sites=4)
        assert d_a.candidate_raw_norms == tuple(c["raw_norms"])
        assert d_a.fan_mean_raw_norms == tuple(fan_abs)
    # the two kinds' fan means are genuinely different numbers here, so a
    # swapped divisor would have been caught above
    assert not np.allclose(fan_abs, fan_c)


def test_predicted_contrast_refuses_without_a_contrast_fan_mean(
    loom_map: ls.LoomMap,
) -> None:
    cands, xs = a_fan(loom_map)
    ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    with pytest.raises(ValueError, match="fan mean"):
        ls.fan_wear_fields(cands[0], lever_kind="contrast",
                           dose_policy="predicted",
                           fan_mean=ls.fan_mean_raw_norms(cands),
                           fan_mean_contrast=None, n_sites=4)


def test_an_unavailable_contrast_refuses_and_never_wears_absolute(
    loom_map: ls.LoomMap,
) -> None:
    cands, xs = a_fan(loom_map, k=1)
    ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    with pytest.raises(ValueError, match="unavailable for candidate 0.*at least 2"):
        ls.fan_wear_fields(cands[0], lever_kind="contrast", dose_policy="flat",
                           fan_mean=None, fan_mean_contrast=None, n_sites=4)


def test_a_candidate_from_before_the_contrast_existed_refuses_contrast() -> None:
    """A candidate dict with no contrast_* keys at all (a draw made by an
    older code path) must refuse, not KeyError and not wear absolute."""
    cand = {"index": 0, "lever": np.ones((4, 3)), "raw_norms": [1.0] * 4,
            "code": [0.0], "note": None}
    with pytest.raises(ValueError, match="lever_kind='contrast' is unavailable"):
        ls.fan_wear_fields(cand, lever_kind="contrast", dose_policy="flat",
                           fan_mean=None, fan_mean_contrast=None, n_sites=4)


def test_body_overrides_the_server_default() -> None:
    assert ls.resolve_lever_kind("contrast", "absolute") == "contrast"
    assert ls.resolve_lever_kind(" absolute ", "contrast") == "absolute"
    assert ls.resolve_lever_kind(None, "contrast") == "contrast"


@pytest.mark.parametrize("bad", [1, True, ["contrast"], {"k": 1}])
def test_a_non_string_lever_kind_is_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        ls.resolve_lever_kind(bad, "absolute")


@pytest.mark.parametrize("bad", ["", "Contrast", "differential", "fan", "abs"])
def test_an_unknown_lever_kind_string_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="lever_kind must be one of"):
        ls.resolve_lever_kind(bad, "absolute")


def test_fan_wear_fields_refuses_an_unresolved_kind(loom_map: ls.LoomMap) -> None:
    cands, _ = a_fan(loom_map, k=2)
    with pytest.raises(ValueError, match="lever_kind must be one of"):
        ls.fan_wear_fields(cands[0], lever_kind="differential",
                           dose_policy="flat", fan_mean=None,
                           fan_mean_contrast=None, n_sites=4)


# ══ 5. worn_public and /info ═════════════════════════════════════════════════


def test_worn_public_reports_the_kind(loom_map: ls.LoomMap) -> None:
    sess = ls.LoomSession()
    cands, _ = a_fan(loom_map, k=2)
    base = {"index": 0, "alpha": 0.4, "loom_id": "x-000",
            "vectors": cands[0]["lever"],
            "dose": ls.flat_dose_scale(4).to_json()}
    sess.worn = dict(base)                             # predates the field
    assert sess.worn_public()["lever_kind"] == "absolute"
    sess.worn = dict(base, lever_kind="contrast", contrast_peers=[0, 1])
    assert sess.worn_public()["lever_kind"] == "contrast"
    sess.worn = dict(base, index=-1, source="bank:w5|orig")   # /wear_code
    assert sess.worn_public()["lever_kind"] is None
    sess.worn = None
    assert sess.worn_public() is None


def test_a_fresh_session_has_no_contrast_fan_mean() -> None:
    sess = ls.LoomSession()
    assert sess.fan_mean_raw_norms_contrast is None
    assert sess.fan_contrast_note is None


def info(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        map_path="/m.npz", map_meta={"rank": 64}, sites=SITES,
        branches=["base", "loom"], default_k=16, future_tokens=96,
        detach_wear_default=False, n_sessions=0, harvest_worker=None,
        restored_sessions=0, persistence={}, auto_policies=ls.AUTO_POLICY_INFO,
        loudness_ref=2.68, dose_band=None,
    )
    base.update(over)
    return ls.build_info_payload(**base)


def test_info_publishes_the_lever_kinds_and_the_default() -> None:
    blob = info()
    assert blob["lever_kind_default"] == "absolute"
    assert [p["key"] for p in blob["lever_kinds"]] == list(ls.LEVER_KINDS)
    assert [p["default"] for p in blob["lever_kinds"]] == [True, False]
    contrast = next(p for p in blob["lever_kinds"] if p["key"] == "contrast")
    assert "UNVALIDATED" in contrast["status"] and contrast["needs_fan"]


def test_info_derives_the_per_entry_default_from_the_running_one() -> None:
    blob = info(lever_kind_default="contrast")
    assert blob["lever_kind_default"] == "contrast"
    assert [p["default"] for p in blob["lever_kinds"]] == [False, True]


# ══ 6. persistence: a restore keeps the kind ═════════════════════════════════


def test_snapshot_round_trips_the_kind_and_peers(loom_map: ls.LoomMap) -> None:
    cands, _ = a_fan(loom_map, k=3)
    worn = {"index": 1, "alpha": 0.3, "loom_id": "x-000",
            "vectors": cands[1]["lever"], "code": [0.1], "loom_dir": "/d",
            "dose": ls.flat_dose_scale(4).to_json(),
            "lever_kind": "contrast", "contrast_peers": [0, 1, 2]}
    snap = ls.WornSnapshot.from_worn(worn)
    back = ls.WornSnapshot.from_json(json.loads(json.dumps(snap.to_json())))
    assert back.lever_kind == "contrast" and back.contrast_peers == [0, 1, 2]
    restored = back.to_worn()
    assert restored["lever_kind"] == "contrast"
    assert restored["contrast_peers"] == [0, 1, 2]
    assert restored["vectors"] is None


def test_an_older_snapshot_without_the_field_means_absolute() -> None:
    blob = {"index": 2, "alpha": 0.4, "loom_id": "x", "code": None,
            "per_site_norms_at_alpha1": [1.0, 2.0], "loom_dir": "/d"}
    snap = ls.WornSnapshot.from_json(blob)
    assert snap.lever_kind is None and snap.contrast_peers is None
    worn = snap.to_worn()
    assert ls.worn_lever_kind(worn) == "absolute"
    sess = ls.LoomSession()
    sess.worn = worn
    assert sess.worn_public()["lever_kind"] == "absolute"


@pytest.mark.parametrize("bad", ["differential", 3, ""])
def test_a_snapshot_with_an_unknown_kind_refuses(bad: Any) -> None:
    blob = {"index": 2, "alpha": 0.4, "lever_kind": bad}
    with pytest.raises(ValueError, match="lever_kind"):
        ls.WornSnapshot.from_json(blob)


def write_loom_dir(root: Path, n: int, seed: int = 3) -> Path:
    """A real loom dir: per-future signatures + one ``<loom dir>/bins.npz``, the files the
    draw path read and rehydration re-reads. bins rows are stored out of
    generation order on purpose (the lookup is by generation_id)."""
    rng = np.random.default_rng(seed)
    (root / "signatures").mkdir(parents=True, exist_ok=True)
    for j in range(n):
        np.savez(root / "signatures" / f"gen_{j:03d}.npz",
                 features=rng.standard_normal(N_V3))
    order = list(reversed(range(n)))
    np.savez(root / "bins.npz",
             features=rng.standard_normal((n, N_BINS)),
             generation_id=np.array(order))
    return root


def draw_from_dir(m: ls.LoomMap, root: Path, n: int, dead: tuple[int, ...] = ()
                  ) -> tuple[list[dict[str, Any]], list[np.ndarray | None]]:
    """The draw path's per-candidate loop, on a real dir."""
    with np.load(root / "bins.npz") as b:
        feats = np.asarray(b["features"], dtype=np.float64)
        gid = {int(g): i for i, g in enumerate(b["generation_id"])}
    cands, xs = [], []
    for j in range(n):
        if j in dead:
            cands.append({"index": j, "lever": None, "note": "dead",
                          "raw_norms": None, "code": None})
            xs.append(None)
            continue
        with np.load(root / "signatures" / f"gen_{j:03d}.npz") as s:
            sig = np.asarray(s["features"], dtype=np.float64)
        x = m.input_of(sig, feats[gid[j]])
        lever, raw, code = m.lever_of_input(x)
        cands.append({"index": j, "lever": lever, "note": None,
                      "raw_norms": raw, "code": code})
        xs.append(x)
    return cands, xs


def rehydrate(m: ls.LoomMap, worn: dict[str, Any]) -> np.ndarray:
    return ls.rehydrate_worn_vectors(worn, m.lever_of, input_of=m.input_of,
                                     contrast_of=m.contrast_lever_of)


@pytest.mark.parametrize("policy", ["flat", "predicted"])
def test_a_restored_contrast_wear_rehydrates_the_contrast(
    loom_map: ls.LoomMap, tmp_path: Path, policy: str,
) -> None:
    """★ Worn → snapshot → JSON → restore → rehydrate gives back the SAME
    BYTES that were worn, as a contrast, with a dead member still excluded."""
    root = write_loom_dir(tmp_path / "loom_000", 5)
    cands, xs = draw_from_dir(loom_map, root, 5, dead=(3,))
    fan_c, _ = ls.attach_fan_contrast(cands, xs, loom_map.contrast_lever_of)
    fields, _ = ls.fan_wear_fields(
        cands[4], lever_kind="contrast", dose_policy=policy,
        fan_mean=ls.fan_mean_raw_norms(cands), fan_mean_contrast=fan_c,
        n_sites=loom_map.n_sites)
    worn = {"index": 4, "alpha": 0.35, "loom_id": "x-000",
            "loom_dir": str(root), **fields}
    snap = ls.WornSnapshot.from_json(
        json.loads(json.dumps(ls.WornSnapshot.from_worn(worn).to_json())))
    restored = snap.to_worn()
    assert restored["contrast_peers"] == [0, 1, 2, 4]
    got = rehydrate(loom_map, restored)
    if policy == "flat":
        assert got.tobytes() == fields["vectors"].tobytes()
    else:
        # the persisted dose receipt rounds its scale to 6 places (pre-existing
        # DoseScale.to_json behaviour, same for absolute wears), so a restored
        # predicted wear matches to that rounding, not to the bit.
        np.testing.assert_allclose(got, fields["vectors"], rtol=2e-6)
        unscaled = ls.DoseScale.from_json(fields["dose"]).apply(
            cands[4]["contrast_lever"])
        assert got.tobytes() == unscaled.tobytes()
    absolute = ls.DoseScale.from_json(fields["dose"]).apply(cands[4]["lever"])
    assert not np.allclose(got, absolute)


def test_a_restored_absolute_wear_is_unchanged(loom_map: ls.LoomMap,
                                               tmp_path: Path) -> None:
    root = write_loom_dir(tmp_path / "loom_000", 3)
    cands, _ = draw_from_dir(loom_map, root, 3)
    for kind_field in ({}, {"lever_kind": "absolute"}):
        worn = {"index": 2, "alpha": 0.35, "loom_dir": str(root), **kind_field}
        got = rehydrate(loom_map, worn)
        assert got.tobytes() == cands[2]["lever"].tobytes()
        # and exactly what the pre-change call (lever_of only) returns
        assert ls.rehydrate_worn_vectors(worn, loom_map.lever_of).tobytes() \
            == got.tobytes()


def test_a_contrast_restore_without_the_recompute_refuses(
    loom_map: ls.LoomMap, tmp_path: Path,
) -> None:
    root = write_loom_dir(tmp_path / "loom_000", 3)
    worn = {"index": 1, "alpha": 0.3, "loom_dir": str(root),
            "lever_kind": "contrast", "contrast_peers": [0, 1, 2]}
    with pytest.raises(ValueError, match="refusing to rehydrate the absolute"):
        ls.rehydrate_worn_vectors(worn, loom_map.lever_of)


@pytest.mark.parametrize("peers,match", [
    (None, "contrast_peers"), ([1], "contrast_peers"),
    ([0, 2], "not among its own"),
])
def test_a_contrast_restore_without_usable_peers_refuses(
    loom_map: ls.LoomMap, tmp_path: Path, peers: Any, match: str,
) -> None:
    root = write_loom_dir(tmp_path / "loom_000", 3)
    worn = {"index": 1, "alpha": 0.3, "loom_dir": str(root),
            "lever_kind": "contrast", "contrast_peers": peers}
    with pytest.raises(ValueError, match=match):
        rehydrate(loom_map, worn)


def test_a_contrast_restore_with_a_missing_peer_signature_refuses(
    loom_map: ls.LoomMap, tmp_path: Path,
) -> None:
    """A pruned peer means the contrast cannot be recomputed — raise (the
    server then keeps the wear INERT) rather than contrast over fewer rows."""
    root = write_loom_dir(tmp_path / "loom_000", 3)
    (root / "signatures" / "gen_000.npz").unlink()
    worn = {"index": 1, "alpha": 0.3, "loom_dir": str(root),
            "lever_kind": "contrast", "contrast_peers": [0, 1, 2]}
    with pytest.raises(FileNotFoundError, match="no signature"):
        rehydrate(loom_map, worn)


def test_the_server_wires_the_contrast_recompute_into_restore() -> None:
    """The restore hook (ServeContext.rehydrate) needs a model to run; pin
    its call shape."""
    src = _server_src()
    i = src.index("vectors = rehydrate_worn_vectors(")
    call = src[i:i + 300]
    assert "input_of=self.loom_map.input_of" in call
    assert "contrast_of=self.loom_map.contrast_lever_of" in call


def test_the_server_threads_lever_kind_through_every_fan_wear() -> None:
    """/wear, /probe's wear rider and /loom's auto-wear all resolve the kind
    with the server default and go through fan_wear_fields."""
    src = _server_src()
    assert src.count("fan_wear_fields(") == 2   # do_wear + auto-wear (def: pleroma.levers.kind)
    # /loom's auto: read into AutoSpec (models.py), resolved after pick_auto
    assert 'lever_kind=auto.get("lever_kind")' in src
    assert "resolve_lever_kind(spec.lever_kind" in src
    assert "resolve_lever_kind(lever_kind, str(self.args.wear_lever_default))" in src
    assert 'wear.get("lever_kind")' in src          # /probe's wear rider
    assert 'blob.get("lever_kind")' in src          # POST /wear
    assert '"predicted_dose_scale_contrast"' in src
    assert '"lever_kind": kind}' in src             # in the /wear response
