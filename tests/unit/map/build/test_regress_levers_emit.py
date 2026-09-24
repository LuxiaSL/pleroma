"""B1.5 Build A: the emitted predictions are held out, and signed the way we say.

Two properties, and they are the two that decide whether the composed-chain
number means anything:

  * every emitted prediction comes from a fit that EXCLUDED its own fold —
    asserted against a literal per-fold refit, not against itself;
  * ``member_pred[i] == s_i * pred_i`` — the canonical orientation. If that sign
    were dropped, averaging a group's predictions would cancel cluster 0 against
    cluster 1 and Build A's matched arm would be noise around zero, which is a
    failure that LOOKS exactly like the negative result it would be mistaken for.

numpy + stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.map.build.levers import gen_key
from pleroma.map.build.regress import (
    cv_predictions,
    emit_predictions,
    ridge_fit,
)

N_GENS = 40
DIM = 12
N_SITES = 2
HIDDEN = 5
FOLDS = 4


def _world(seed: int = 0) -> dict[str, object]:
    """z with real signal, signed targets, whole-group folds, four planted groups.

    The levers share a common axis with a per-group wobble (which is the L2
    hypothesis itself: "does z carry where basins live, PORTABLY across
    prompts"), and z's first coordinate carries the member's side. A held-out
    group is then genuinely predictable — without that, the ridge would learn
    nothing but the intercept and these tests would be about an intercept.
    """
    rng = np.random.default_rng(seed)
    n_groups = 4
    common = rng.standard_normal((N_SITES, HIDDEN))
    levers = common[None] + 0.3 * rng.standard_normal((n_groups, N_SITES, HIDDEN))
    flat = levers.reshape(n_groups, -1)

    group_index = np.repeat(np.arange(n_groups), N_GENS // n_groups)
    signs = np.where(np.arange(N_GENS) % 2 == 0, 1.0, -1.0)
    clusters = (1 - signs) / 2  # sign +1 <-> cluster 0
    z = 0.4 * rng.standard_normal((N_GENS, DIM))
    z[:, 0] += 2.0 * signs
    y = signs[:, None] * flat[group_index]
    # Folds are whole groups, as the real grouped CV is.
    fold_idx = group_index % FOLDS
    return {
        "z": z,
        "y": y,
        "levers": levers,
        "group_index": group_index,
        "signs": signs,
        "clusters": clusters.astype(int),
        "fold_idx": fold_idx,
    }


def _levers_npz(tmp_path: Path, world: dict) -> Path:
    path = tmp_path / "levers.npz"
    n_groups = int(np.asarray(world["levers"]).shape[0])
    np.savez(
        path,
        levers=np.asarray(world["levers"], dtype=np.float32),
        member_group_index=np.asarray(world["group_index"], dtype=np.int64),
        member_clusters=np.asarray(world["clusters"], dtype=np.int64),
        member_corpus_keys=np.asarray(["toy"] * N_GENS, dtype=np.str_),
        member_generation_ids=np.arange(N_GENS, dtype=np.int64),
        member_prompt_ids=np.asarray(
            [f"p{g}" for g in np.asarray(world["group_index"])], dtype=np.str_
        ),
        member_prompt_classes=np.asarray(["dc"] * N_GENS, dtype=np.str_),
        group_prompt_ids=np.asarray([f"p{i}" for i in range(n_groups)], dtype=np.str_),
        group_waves=np.asarray(["orig"] * n_groups, dtype=np.str_),
        group_prompt_classes=np.asarray(["dc"] * n_groups, dtype=np.str_),
        group_corpus_keys=np.asarray(["toy"] * n_groups, dtype=np.str_),
        sites=np.asarray([8, 15], dtype=np.int64),
    )
    return path


# ── hold-out ───────────────────────────────────────────────────────────────────


def test_every_prediction_comes_from_a_fit_that_excluded_its_fold() -> None:
    world = _world()
    z, y, fold_idx = world["z"], world["y"], world["fold_idx"]
    pred = cv_predictions(z, y, fold_idx, FOLDS, lam=10.0, rank=0)

    for f in range(FOLDS):
        tr, te = fold_idx != f, fold_idx == f
        mu_z, mu_y = z[tr].mean(0), y[tr].mean(0)
        w = ridge_fit(z[tr] - mu_z, y[tr] - mu_y, 10.0)
        expected = (z[te] - mu_z) @ w + mu_y
        assert np.allclose(pred[te], expected, rtol=1e-8, atol=1e-10)


def test_rank_truncation_changes_the_prediction() -> None:
    """Otherwise --emit-rank would be a no-op flag on a bottleneck claim."""
    world = _world()
    full = cv_predictions(world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, 0)
    low = cv_predictions(world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, 2)
    assert not np.allclose(full, low)
    # A rank at or above the matrix rank is the untruncated fit again.
    wide = cv_predictions(world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, DIM + 5)
    assert np.allclose(full, wide, rtol=1e-8, atol=1e-10)


def test_a_row_in_no_test_fold_is_refused() -> None:
    """Sweeping fewer folds than the labels use would emit IN-SAMPLE rows silently."""
    world = _world()
    with pytest.raises(ValueError, match="never in a test fold"):
        # fold_idx runs 0..3 but only fold 0 is swept: folds 1-3 never get predicted.
        cv_predictions(world["z"], world["y"], world["fold_idx"], 1, 10.0, 0)


def test_folds_must_leave_training_rows() -> None:
    world = _world()
    with pytest.raises(ValueError, match="no training rows"):
        cv_predictions(world["z"], world["y"], np.zeros(N_GENS, dtype=int), 1, 10.0, 0)


# ── the sign convention ────────────────────────────────────────────────────────


def test_member_pred_is_the_signed_canonical_prediction(tmp_path: Path) -> None:
    world = _world()
    path = _levers_npz(tmp_path, world)
    out = tmp_path / "pred.npz"
    with np.load(path, allow_pickle=False) as lev:
        meta = emit_predictions(
            path=out,
            z=np.asarray(world["z"]),
            y=np.asarray(world["y"]),
            fold_idx=np.asarray(world["fold_idx"]),
            folds=FOLDS,
            lam=10.0,
            rank=0,
            use_gpu=False,
            members=list(range(N_GENS)),
            signs=np.asarray(world["signs"]),
            lever_npz=lev,
            levers=np.asarray(world["levers"]),
            sources={"pairs_dir": "x", "levers": str(path)},
        )

    raw = cv_predictions(
        world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, 0
    )
    signs = np.asarray(world["signs"])
    with np.load(out, allow_pickle=False) as npz:
        member_pred = npz["member_pred"]
        assert member_pred.shape == (N_GENS, N_SITES, HIDDEN)
        expected = (signs[:, None] * raw).reshape(N_GENS, N_SITES, HIDDEN)
        assert np.allclose(member_pred, expected, rtol=1e-5, atol=1e-6)
        # The raw ridge prediction is recoverable, which is what the docstring promises.
        recovered = npz["member_signs"][:, None, None] * member_pred
        assert np.allclose(
            recovered.reshape(N_GENS, -1), raw, rtol=1e-5, atol=1e-6
        )
        assert [str(x) for x in npz["member_keys"]] == [
            gen_key("toy", i) for i in range(N_GENS)
        ]
        assert [int(x) for x in npz["sites"]] == [8, 15]
        emitted = json.loads(str(npz["meta_json"]))
        assert emitted["rank"] == 0 and emitted["lambda"] == 10.0
        assert "s_i * pred_i" in emitted["sign_convention"]
    assert meta["n_gens"] == N_GENS


def test_the_canonical_orientation_is_what_makes_averaging_work(tmp_path: Path) -> None:
    """The failure this convention exists to prevent, demonstrated.

    Averaging the RAW predictions of a group cancels — its members predict
    opposite vectors. Averaging the canonical ones does not.
    """
    world = _world()
    raw = cv_predictions(world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, 0)
    signs = np.asarray(world["signs"])
    group_index = np.asarray(world["group_index"])
    flat = np.asarray(world["levers"]).reshape(4, -1)

    for gi in range(4):
        sel = group_index == gi
        canonical = (signs[sel, None] * raw[sel]).mean(0)
        naive = raw[sel].mean(0)
        true = flat[gi]
        cos = lambda a, b: float(  # noqa: E731 — local one-liner, read once
            a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )
        assert cos(canonical, true) > 0.5, "the canonical average must recover the lever"
        assert cos(canonical, true) > cos(naive, true)


def test_group_cosines_are_the_full_average_against_the_true_lever(tmp_path: Path) -> None:
    world = _world()
    path = _levers_npz(tmp_path, world)
    out = tmp_path / "pred.npz"
    with np.load(path, allow_pickle=False) as lev:
        emit_predictions(
            path=out, z=np.asarray(world["z"]), y=np.asarray(world["y"]),
            fold_idx=np.asarray(world["fold_idx"]), folds=FOLDS, lam=10.0, rank=0,
            use_gpu=False, members=list(range(N_GENS)),
            signs=np.asarray(world["signs"]), lever_npz=lev,
            levers=np.asarray(world["levers"]), sources={},
        )
    raw = cv_predictions(world["z"], world["y"], world["fold_idx"], FOLDS, 10.0, 0)
    signed = np.asarray(world["signs"])[:, None] * raw
    group_index = np.asarray(world["group_index"])
    flat = np.asarray(world["levers"]).reshape(4, -1)

    with np.load(out, allow_pickle=False) as npz:
        for gi in range(4):
            avg = signed[group_index == gi].mean(0)
            expected = float(
                avg @ flat[gi] / (np.linalg.norm(avg) * np.linalg.norm(flat[gi]))
            )
            assert float(npz["group_pred_cos"][gi]) == pytest.approx(expected, abs=1e-5)
            assert int(npz["group_n_predictions"][gi]) == N_GENS // 4
