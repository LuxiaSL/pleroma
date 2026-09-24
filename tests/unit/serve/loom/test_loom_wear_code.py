"""Tests for POST /wear_code's pure parts — the bank loader and atlas resolution.

WHY THESE EXIST. `/wear_code` makes a BANKED lever wearable on a live
conversation, which is the only way to get guaranteed manner contrast (a fan
may not contain any). Two things about it can fail
silently and both would poison an experiment rather than crash it:

  1. a bank whose `sites` disagree with the live map would inject at the WRONG
     LAYERS and still return 200;
  2. a lever that is not renormalised to the map's own `norm_ref` makes alpha mean
     one thing for a fan wear and another for a landmark wear — a 2.139x error
     between two maps' rulers (docs/FINDINGS.md section 2: alpha means nothing
     without its ruler).

So the loader refuses on (1) and the tests below pin (2) numerically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.serve.legacy import (
    LeverBank,
    LoomMap,
    check_atlas_matches_bank,
    resolve_atlas_landmark,
    strip_atlas_suffix,
)

SITES = [8, 15, 19, 22]
HIDDEN = 16
NORM_REF = [1.0, 2.0, 3.0, 4.0]


class FakeMap:
    """Just the surface LeverBank touches."""

    def __init__(self, sites: list[int] = SITES, hidden: int = HIDDEN,
                 norm_ref: list[float] | None = None) -> None:
        self.sites = list(sites)
        self.n_sites = len(sites)
        self.hidden = hidden
        self.norm_ref = np.asarray(norm_ref or NORM_REF, dtype=np.float64)


def write_bank(path: Path, *, n_groups: int = 3, sites: list[int] = SITES,
               hidden: int = HIDDEN, labels: list[tuple[str, str]] | None = None,
               zero_row: bool = False, **extra: Any) -> Path:
    rng = np.random.default_rng(0)
    lev = rng.standard_normal((n_groups, len(sites), hidden)).astype(np.float32)
    lev *= np.array([2.0, 5.0, 9.0, 13.0])[None, :len(sites), None]  # loud, unmatched
    if zero_row:
        lev[0, 1, :] = 0.0
    labels = labels or [(f"w5_p{i:03d}", "orig") for i in range(n_groups)]
    np.savez_compressed(
        path,
        levers=lev,
        group_prompt_ids=np.asarray([a for a, _ in labels], dtype=np.str_),
        group_waves=np.asarray([b for _, b in labels], dtype=np.str_),
        group_prompt_classes=np.asarray(["stance_fork"] * n_groups, dtype=np.str_),
        sites=np.asarray(sites, dtype=np.int64),
        **extra,
    )
    return path


# ── the ruler: the whole reason this endpoint is not a one-liner ───────────────

def test_lever_is_renormalised_to_the_maps_norm_ref(tmp_path: Path) -> None:
    """alpha must mean the same thing for a landmark as for a fan candidate."""
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    vectors, raw = bank.vectors_for("w5_p000|orig")
    got = [float(np.linalg.norm(r)) for r in vectors]
    assert got == pytest.approx(NORM_REF, rel=1e-5)
    # and the RAW norms are reported, not silently discarded
    assert len(raw) == len(SITES)
    assert all(r > 0 for r in raw)
    assert raw != pytest.approx(NORM_REF, rel=1e-2), \
        "fixture should be loud and unmatched, or the test proves nothing"


def test_renormalisation_preserves_direction(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    raw_lev = np.load(tmp_path / "b.npz")["levers"][0].astype(np.float64)
    vectors, _ = bank.vectors_for("w5_p000|orig")
    for s in range(len(SITES)):
        a, b = raw_lev[s], np.asarray(vectors[s], dtype=np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos == pytest.approx(1.0, abs=1e-6)


def test_every_group_is_matched_not_just_the_first(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz", n_groups=3), FakeMap())
    for i in range(3):
        vectors, _ = bank.vectors_for(f"w5_p{i:03d}|orig")
        assert [float(np.linalg.norm(r)) for r in vectors] == \
            pytest.approx(NORM_REF, rel=1e-5)


# ── refusals: each of these would otherwise be a silent wrong answer ──────────

def test_site_mismatch_refuses(tmp_path: Path) -> None:
    path = write_bank(tmp_path / "b.npz", sites=[4, 9, 14, 19])
    with pytest.raises(ValueError, match="wrong layers"):
        LeverBank(path, FakeMap(sites=SITES))


def test_hidden_dim_mismatch_refuses(tmp_path: Path) -> None:
    path = write_bank(tmp_path / "b.npz", hidden=HIDDEN)
    with pytest.raises(ValueError, match="but the map is"):
        LeverBank(path, FakeMap(hidden=HIDDEN + 1))


def test_missing_array_refuses_by_name(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, levers=np.zeros((2, 4, HIDDEN), dtype=np.float32))
    with pytest.raises(ValueError, match="missing"):
        LeverBank(p, FakeMap())


def test_unknown_group_names_the_bank_and_the_label_shape(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    with pytest.raises(ValueError, match="not in"):
        bank.vectors_for("nope|orig")


def test_zero_lever_row_refuses_rather_than_dividing(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz", zero_row=True), FakeMap())
    with pytest.raises(ValueError, match="zero lever row"):
        bank.vectors_for("w5_p000|orig")


def test_labels_and_classes_line_up(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz", n_groups=2,
                                labels=[("w5_aaa", "orig"), ("w5_bbb", "repl")]),
                     FakeMap())
    assert bank.labels == ["w5_aaa|orig", "w5_bbb|repl"]
    assert bank.index_of["w5_bbb|repl"] == 1
    assert bank.classes[1] == "stance_fork"


# ── atlas landmark resolution ─────────────────────────────────────────────────

ATLAS = {"axes": [
    {"axis": 0, "low_extreme": ["w5_lo0|orig(de)", "w5_lo1|repl(st)"],
     "high_extreme": ["w5_hi0|repl(op)"]},
    {"axis": 1, "low_extreme": [], "high_extreme": ["w5_x|orig"]},
]}


def test_strip_atlas_suffix() -> None:
    assert strip_atlas_suffix("w5_og01155|orig(op)") == "w5_og01155|orig"
    assert strip_atlas_suffix("w5_dc02683|orig") == "w5_dc02683|orig"
    assert strip_atlas_suffix(" w5_a|orig (de) ") == "w5_a|orig"


def test_resolve_both_poles_and_rank() -> None:
    assert resolve_atlas_landmark(ATLAS, 0, "low") == "w5_lo0|orig"
    assert resolve_atlas_landmark(ATLAS, 0, "low", 1) == "w5_lo1|repl"
    assert resolve_atlas_landmark(ATLAS, 0, "high") == "w5_hi0|repl"


@pytest.mark.parametrize("axis,pole,rank,match", [
    (0, "middle", 0, "pole must be"),
    (9, "low", 0, "no axis 9"),
    (0, "low", 5, "asked for rank"),
    (1, "low", 0, "has 0 entries"),
])
def test_resolution_refuses_loudly(axis: int, pole: str, rank: int,
                                   match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_atlas_landmark(ATLAS, axis, pole, rank)


def test_missing_axes_list_refuses() -> None:
    with pytest.raises(ValueError, match="no 'axes' list"):
        resolve_atlas_landmark({}, 0, "low")


# ── the random control arm ────────────────────────────────────────────────────
#
# The random arm asks whether the readout detects DIRECTION or merely
# PERTURBATION: if it scores above chance, the directional result is void
# whatever its value. That only works
# if the random vector passes through the SAME ruler as a real landmark, so these
# tests pin the renormalisation for the random path exactly as above.

def test_random_vectors_are_renormalised_to_norm_ref(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    vectors, raw = bank.random_vectors(7)
    assert [float(np.linalg.norm(r)) for r in vectors] == \
        pytest.approx(NORM_REF, rel=1e-5)
    assert len(raw) == len(SITES)
    assert all(r > 0 for r in raw)


def test_random_vectors_have_the_right_shape_and_dtype(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    vectors, _ = bank.random_vectors(1)
    assert vectors.shape == (len(SITES), HIDDEN)
    assert vectors.dtype == np.float32


def test_random_vectors_are_reproducible_from_the_seed(tmp_path: Path) -> None:
    """The seed is recorded in the run log; it has to mean something."""
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    a, _ = bank.random_vectors(20260919)
    b, _ = bank.random_vectors(20260919)
    c, _ = bank.random_vectors(20260920)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_vectors_are_unrelated_to_any_banked_lever(tmp_path: Path) -> None:
    """A control that accidentally aligned with the bank would not be a control."""
    bank = LeverBank(write_bank(tmp_path / "b.npz", n_groups=3), FakeMap())
    rnd, _ = bank.random_vectors(3)
    flat_r = np.asarray(rnd, dtype=np.float64).ravel()
    flat_r /= np.linalg.norm(flat_r)
    for i in range(3):
        lev, _ = bank.vectors_for(f"w5_p{i:03d}|orig")
        flat_l = np.asarray(lev, dtype=np.float64).ravel()
        flat_l /= np.linalg.norm(flat_l)
        assert abs(float(flat_r @ flat_l)) < 0.5


# ── explicit-code wear: lever_from_code (transplant + fan-centered probe) ─────
#
# `code = Vt @ (flat − mu_y)` with orthonormal Vt rows, so an ABSOLUTE code must
# reconstruct its candidate's lever exactly, and a DIFFERENTIAL code (a − b)
# must reconstruct flat_a − flat_b exactly (mu_y cancels). Both then pass the
# SAME per-site norm_ref ruler as every other wear. Any error
# here poisons the transplant/portability experiment silently, not loudly.

RANK = 8


class FakeCodeMap:
    """Just the surface lever_from_code touches, with a REAL orthonormal Vt."""

    def __init__(self, sites: list[int] = SITES, hidden: int = HIDDEN,
                 rank: int = RANK, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.sites = list(sites)
        self.n_sites = len(sites)
        self.hidden = hidden
        self.rank = rank
        q, _ = np.linalg.qr(rng.standard_normal((self.n_sites * hidden, rank)))
        self.Vt = q.T  # [rank, S*hidden], orthonormal rows
        self.mu_y = rng.standard_normal(self.n_sites * hidden) * 3.0
        self.norm_ref = np.asarray(NORM_REF, dtype=np.float64)

    def flat_of(self, code: np.ndarray, *, differential: bool) -> np.ndarray:
        flat = np.asarray(code, dtype=np.float64) @ self.Vt
        return flat if differential else flat + self.mu_y


def _row_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_absolute_code_reconstructs_its_lever_direction_exactly() -> None:
    fake = FakeCodeMap()
    rng = np.random.default_rng(1)
    code = rng.standard_normal(RANK)
    out, raw = LoomMap.lever_from_code(fake, code, differential=False)
    expected = fake.flat_of(code, differential=False).reshape(
        fake.n_sites, fake.hidden)
    for s in range(fake.n_sites):
        assert _row_cos(expected[s], out[s]) == pytest.approx(1.0, abs=1e-9)
        assert raw[s] == pytest.approx(float(np.linalg.norm(expected[s])),
                                       rel=1e-9)
    assert [float(np.linalg.norm(r)) for r in out] == \
        pytest.approx(NORM_REF, rel=1e-9)


def test_differential_code_is_the_difference_of_absolute_levers() -> None:
    """mu_y must cancel: wear(a−b, differential) ∝ flat(a) − flat(b) per site."""
    fake = FakeCodeMap()
    rng = np.random.default_rng(2)
    code_a, code_b = rng.standard_normal(RANK), rng.standard_normal(RANK)
    out, _ = LoomMap.lever_from_code(fake, code_a - code_b, differential=True)
    diff = (fake.flat_of(code_a, differential=False)
            - fake.flat_of(code_b, differential=False)).reshape(
                fake.n_sites, fake.hidden)
    for s in range(fake.n_sites):
        assert _row_cos(diff[s], out[s]) == pytest.approx(1.0, abs=1e-9)
    assert [float(np.linalg.norm(r)) for r in out] == \
        pytest.approx(NORM_REF, rel=1e-9)


def test_differential_does_not_add_mu_y_back() -> None:
    fake = FakeCodeMap()
    rng = np.random.default_rng(3)
    code = rng.standard_normal(RANK)
    out_d, raw_d = LoomMap.lever_from_code(fake, code, differential=True)
    expected = fake.flat_of(code, differential=True).reshape(
        fake.n_sites, fake.hidden)
    for s in range(fake.n_sites):
        assert _row_cos(expected[s], out_d[s]) == pytest.approx(1.0, abs=1e-9)
        assert raw_d[s] == pytest.approx(float(np.linalg.norm(expected[s])),
                                         rel=1e-9)


def test_fan_centered_identity_one_vs_rest() -> None:
    """The zero-training contrastive probe: lever(code_sel − mean(code_rest))
    equals lever(code_sel) − mean(lever(code_rest)) up to the ruler — linearity
    is the whole premise of running the probe without training anything."""
    fake = FakeCodeMap()
    rng = np.random.default_rng(4)
    codes = rng.standard_normal((16, RANK))
    sel, rest = codes[0], codes[1:]
    out, _ = LoomMap.lever_from_code(fake, sel - rest.mean(axis=0),
                                     differential=True)
    flats = np.stack([fake.flat_of(c, differential=False) for c in codes])
    centered = (flats[0] - flats[1:].mean(axis=0)).reshape(
        fake.n_sites, fake.hidden)
    for s in range(fake.n_sites):
        assert _row_cos(centered[s], out[s]) == pytest.approx(1.0, abs=1e-9)


def test_wrong_length_code_refuses_naming_the_rank() -> None:
    fake = FakeCodeMap()
    with pytest.raises(ValueError, match="rank is 8"):
        LoomMap.lever_from_code(fake, [1.0] * 5, differential=False)


def test_zero_differential_code_refuses_rather_than_amplifying_noise() -> None:
    """A no-op contrastive code must refuse, not get norm-matched to full
    loudness in a noise direction — the norm IS the wear-worthiness signal."""
    fake = FakeCodeMap()
    with pytest.raises(ValueError, match="zero lever row"):
        LoomMap.lever_from_code(fake, [0.0] * RANK, differential=True)


def test_raw_norms_are_reported_not_norm_ref() -> None:
    fake = FakeCodeMap()
    rng = np.random.default_rng(5)
    _, raw = LoomMap.lever_from_code(fake, rng.standard_normal(RANK) * 10,
                                     differential=True)
    assert len(raw) == len(SITES)
    assert raw != pytest.approx(NORM_REF, rel=1e-2), \
        "fixture should be loud and unmatched, or the test proves nothing"


# ── the wrong-atlas guard: an atlas built for a different bank is refused ─────

W4_STALE_ATLAS = {"axes": [
    {"axis": 0, "low_extreme": ["cc1|orig(co)", "mf1|orig(me)"],
     "high_extreme": ["dc14|orig(de)"]},
]}


def test_wrong_atlas_refuses_and_names_the_real_fault(tmp_path: Path) -> None:
    """The stale w4 atlas has 138 short ids; the w5 bank has 679 'w5_*' labels.

    Before this guard the failure surfaced from vectors_for as "group 'cc1|orig'
    is not in <bank>", which blames the group and never says the atlas is wrong.
    """
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    with pytest.raises(ValueError, match="does not describe the loaded bank"):
        check_atlas_matches_bank(W4_STALE_ATLAS, bank, "outputs/loom/atlas/x.json")


def test_wrong_atlas_error_shows_both_label_shapes(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    with pytest.raises(ValueError) as exc:
        check_atlas_matches_bank(W4_STALE_ATLAS, bank, "stale.json")
    msg = str(exc.value)
    assert "cc1|orig" in msg and "w5_p000|orig" in msg
    assert "--atlas-report" in msg, "the message must say how to fix it"


def test_matching_atlas_passes(tmp_path: Path) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz", n_groups=3), FakeMap())
    good = {"axes": [{"axis": 0, "low_extreme": ["w5_p000|orig(st)"],
                      "high_extreme": ["w5_p002|orig(st)"]}]}
    check_atlas_matches_bank(good, bank, "good.json")  # must not raise


def test_partially_matching_atlas_passes(tmp_path: Path) -> None:
    """One resolvable extreme is enough — a bank may legitimately be a subset."""
    bank = LeverBank(write_bank(tmp_path / "b.npz", n_groups=3), FakeMap())
    mixed = {"axes": [{"axis": 0, "low_extreme": ["nope|orig", "w5_p001|orig"],
                       "high_extreme": ["alsonope|repl"]}]}
    check_atlas_matches_bank(mixed, bank, "mixed.json")


@pytest.mark.parametrize("atlas,match", [
    ({}, "not an atlas report"),
    ({"axes": []}, "not an atlas report"),
    ({"axes": "nope"}, "not an atlas report"),
])
def test_malformed_atlas_refuses(tmp_path: Path, atlas: dict[str, Any],
                                 match: str) -> None:
    bank = LeverBank(write_bank(tmp_path / "b.npz"), FakeMap())
    with pytest.raises(ValueError, match=match):
        check_atlas_matches_bank(atlas, bank, "bad.json")
