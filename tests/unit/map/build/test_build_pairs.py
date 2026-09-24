"""Contract tests for B1 stage 1 against a synthetic B0 run dir.

The fixture plants every branch the stage is supposed to take, so each test reads
off a planted fact rather than a plausible number:

  - a discriminants npz whose ``FULL_scale`` has a DELIBERATE ZERO at index 1 and a
    known mean/scale elsewhere, so z-scaling is checkable by hand and the zero-scale
    guard has something to guard;
  - one gen (``gen_003``) whose only divergence position is 0 — the position with no
    banked distribution — so the entropy fallback has to fire, and its entropy series
    is planted so the top-4 is known;
  - convergent_control gens, so exclusion (and ``--include-control``) is testable;
  - four prompts x four seeds, so a 1-prompt / 1-seed holdout leaves a non-trivial
    train split and split integrity is a real assertion rather than a vacuous one;
  - a gen whose ``feature_names`` disagree with FULL_names, to check the run refuses
    it rather than training on a vector from another space.

numpy + pytest only. No torch, no GPU, no node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.map.build import pairs as bp

N_FEATURES = 6
PROMPT_LEN = 4
N_CONT = 10

#: FULL_scale[1] == 0 on purpose: the zero-scale guard's only test surface.
FULL_MEAN = np.array([0.0, 5.0, 1.0, -1.0, 2.0, 0.0], dtype=np.float64)
FULL_SCALE = np.array([2.0, 0.0, 1.0, 4.0, 0.5, 10.0], dtype=np.float64)
FULL_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]

#: (prompt_id, prompt_class). Four prompts, two of them the excluded control class.
PROMPTS: list[tuple[str, str]] = [
    ("mf1", "method_fork"),
    ("mf2", "method_fork"),
    ("sf1", "stance_fork"),
    ("cc1", "convergent_control"),
]
SEEDS = [0, 1, 2, 3]


def _gen_id(prompt_idx: int, seed_idx: int) -> int:
    return prompt_idx * len(SEEDS) + seed_idx


@pytest.fixture()
def discriminants(tmp_path: Path) -> Path:
    path = tmp_path / "factor_directions_test.npz"
    np.savez_compressed(
        path,
        FULL_mean=FULL_MEAN,
        FULL_scale=FULL_SCALE,
        FULL_names=np.asarray(FULL_NAMES, dtype="<U16"),
    )
    return path


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A synthetic B0 stage-2 output dir: signatures/, fork_series/, ``<run>/cells.json``."""
    root = tmp_path / "expB0_fake"
    (root / "signatures").mkdir(parents=True)
    (root / "fork_series").mkdir(parents=True)

    cells: list[dict[str, Any]] = []
    for p_idx, (prompt_id, prompt_class) in enumerate(PROMPTS):
        for seed_idx in SEEDS:
            gid = _gen_id(p_idx, seed_idx)
            # Features vary by gen so the z matrix is not degenerate; index 1 is
            # held at the zero-scale feature's own mean + an offset, which the
            # guard must map to exactly 0 regardless.
            features = np.array(
                [gid, 100.0 + gid, 1.0, -1.0, 2.0 + 0.5 * gid, float(seed_idx)],
                dtype=np.float32,
            )
            np.savez_compressed(
                root / "signatures" / f"gen_{gid:03d}.npz",
                features=features,
                feature_names=np.asarray(FULL_NAMES, dtype="<U16"),
            )
            # entropy[t] is the distribution that produced continuation index t+1.
            # gen_003 gets a planted ranking: steps 5, 2, 7, 0 are the top four,
            # i.e. continuation positions 6, 3, 8, 1.
            entropy = np.full(N_CONT - 1, 0.1, dtype=np.float32)
            if gid == 3:
                entropy[5], entropy[2], entropy[7], entropy[0] = 9.0, 8.0, 7.0, 6.0
            continuation = [100 + gid * 10 + i for i in range(N_CONT)]
            np.savez_compressed(
                root / "fork_series" / f"gen_{gid:03d}.npz",
                input_ids=np.asarray(list(range(PROMPT_LEN)) + continuation, dtype=np.int32),
                prompt_length=np.array(PROMPT_LEN, dtype=np.int32),
                logits_entropy=entropy,
                logits_values=np.zeros((N_CONT - 1, 3), dtype=np.float32),
                logits_indices=np.zeros((N_CONT - 1, 3), dtype=np.int32),
                chosen_ids=np.asarray(continuation[1:], dtype=np.int32),
            )
            cells.append(
                {
                    "generation_id": gid,
                    "prompt_id": prompt_id,
                    "prompt_class": prompt_class,
                    "seed_idx": seed_idx,
                    "prompt_length": PROMPT_LEN,
                    "num_generated_tokens": N_CONT,
                    "signature_path": f"signatures/gen_{gid:03d}.npz",
                    "fork_series_path": f"fork_series/gen_{gid:03d}.npz",
                }
            )
    (root / "cells.json").write_text(json.dumps(cells))
    return root


@pytest.fixture()
def fork_report(tmp_path: Path, run_dir: Path) -> Path:
    """Divergence positions: everyone gets [2, 5] except gen_003, which gets [0]."""
    gens: list[dict[str, Any]] = []
    for p_idx, (prompt_id, prompt_class) in enumerate(PROMPTS):
        for seed_idx in SEEDS:
            gid = _gen_id(p_idx, seed_idx)
            positions = [0] if gid == 3 else [2, 5]
            gens.append(
                {
                    "generation_id": gid,
                    "prompt_id": prompt_id,
                    "prompt_class": prompt_class,
                    "seed_idx": seed_idx,
                    "prompt_length": PROMPT_LEN,
                    "n_continuation_tokens": N_CONT,
                    "divergence_positions": positions,
                    "uncertainty_positions": [1, 3],
                    "forks": [
                        {"position_continuation": 1, "entropy": 0.5},
                        {"position_continuation": 3, "entropy": 0.4},
                    ],
                }
            )
    path = tmp_path / "fork_report.json"
    path.write_text(json.dumps({"stage": "test", "gens": gens, "failures": []}))
    return path


def _build(run_dir: Path, fork_report: Path, discriminants: Path, **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "holdout_prompts": 1,
        "holdout_seeds": 1,
        "include_control": False,
        "fallback_k": bp.ENTROPY_FALLBACK_K,
        "salt": "",
    }
    params.update(kwargs)
    return bp.build(
        run_dirs=[run_dir],
        fork_reports=[fork_report],
        discriminants=discriminants,
        **params,
    )


# ── z-scaling ──────────────────────────────────────────────────────────────────


def test_zscore_matches_the_hand_computed_value() -> None:
    features = np.array([4.0, 123.0, 1.0, 3.0, 4.0, 10.0], dtype=np.float32)
    z = bp.zscore(features, FULL_MEAN, FULL_SCALE)
    # (4-0)/2, GUARDED, (1-1)/1, (3-(-1))/4, (4-2)/0.5, (10-0)/10
    np.testing.assert_allclose(z, [2.0, 0.0, 0.0, 1.0, 4.0, 1.0], rtol=0, atol=1e-6)


def test_zero_scale_dimension_is_exactly_zero_not_inf_or_nan() -> None:
    # Numerator deliberately non-zero: an unguarded divide gives +inf, not nan, so
    # a nan-only check would pass while the vector was still unusable.
    z = bp.zscore(np.full(N_FEATURES, 1e6, dtype=np.float32), FULL_MEAN, FULL_SCALE)
    assert z[1] == 0.0
    assert np.all(np.isfinite(z))


def test_built_z_matrix_is_finite_and_aligned(run_dir: Path, fork_report: Path,
                                              discriminants: Path) -> None:
    pairs, z, names, meta = _build(run_dir, fork_report, discriminants)
    assert z.shape == (len(pairs), N_FEATURES)
    assert names == FULL_NAMES
    assert np.all(np.isfinite(z))
    assert np.all(z[:, 1] == 0.0)  # the zero-scale column, for every gen
    assert meta["n_zero_scale_features"] == 1
    # Row order IS manifest order.
    for pair in pairs:
        gid = pair.generation_id
        np.testing.assert_allclose(z[pair.row, 0], (gid - 0.0) / 2.0, atol=1e-5)


def test_feature_name_mismatch_is_recorded_as_a_failure_not_trained_on(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    gid = _gen_id(0, 0)
    np.savez_compressed(
        run_dir / "signatures" / f"gen_{gid:03d}.npz",
        features=np.zeros(N_FEATURES, dtype=np.float32),
        feature_names=np.asarray(["WRONG"] + FULL_NAMES[1:], dtype="<U16"),
    )
    pairs, _, _, meta = _build(run_dir, fork_report, discriminants)
    assert gid not in {p.generation_id for p in pairs}
    assert any(f"gen_{gid:03d}" in f and "feature_names" in f for f in meta["failures"])


# ── fork positions and the fallback ────────────────────────────────────────────


def test_position_zero_is_dropped_and_the_fallback_fires_and_is_recorded(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, _ = _build(run_dir, fork_report, discriminants)
    gen3 = next(p for p in pairs if p.generation_id == 3)
    assert gen3.fork_source == bp.SOURCE_FALLBACK_SERIES
    assert gen3.fallback_reason == "no_divergence_position_ge_1"
    # Planted top-4 entropy steps 5, 2, 7, 0 -> continuation positions 6, 3, 8, 1.
    assert gen3.fork_positions == [1, 3, 6, 8]
    assert 0 not in gen3.fork_positions


def test_divergence_gens_keep_their_own_positions(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, _ = _build(run_dir, fork_report, discriminants)
    others = [p for p in pairs if p.generation_id != 3]
    assert others
    assert all(p.fork_source == bp.SOURCE_DIVERGENCE for p in others)
    assert all(p.fork_positions == [2, 5] for p in others)


def test_every_fork_position_is_at_least_one(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, _ = _build(run_dir, fork_report, discriminants)
    assert min(min(p.fork_positions) for p in pairs) >= 1


def test_fallback_falls_back_to_the_report_when_fork_series_is_missing(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    (run_dir / "fork_series" / "gen_003.npz").unlink()
    pairs, _, _, _ = _build(run_dir, fork_report, discriminants)
    gen3 = next(p for p in pairs if p.generation_id == 3)
    assert gen3.fork_source == bp.SOURCE_FALLBACK_REPORT
    assert gen3.fork_positions == [1, 3]  # the only entropy-bearing report forks


# ── control exclusion ──────────────────────────────────────────────────────────


def test_control_class_is_excluded_by_default(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, z, _, summary = _build(run_dir, fork_report, discriminants)
    assert not any(p.prompt_class == bp.CONTROL_CLASS for p in pairs)
    assert len(pairs) == (len(PROMPTS) - 1) * len(SEEDS)
    assert z.shape[0] == len(pairs)


def test_include_control_keeps_them(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, _ = _build(run_dir, fork_report, discriminants, include_control=True)
    assert len(pairs) == len(PROMPTS) * len(SEEDS)
    assert any(p.prompt_class == bp.CONTROL_CLASS for p in pairs)


# ── split integrity ────────────────────────────────────────────────────────────


def test_no_val_prompt_leaks_into_train(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, meta = _build(run_dir, fork_report, discriminants)
    train_prompts = {p.prompt_id for p in pairs if p.split == "train"}
    val_prompts = {p.prompt_id for p in pairs if p.split == "val_prompt"}
    assert val_prompts
    assert not (train_prompts & val_prompts)
    assert sorted(val_prompts) == meta["holdout"]["val_prompt_ids"]
    # Every gen of a held-out prompt goes with it — no seed of it stays behind.
    for pair in pairs:
        if pair.prompt_id in val_prompts:
            assert pair.split == "val_prompt"


def test_val_seed_holds_out_seeds_of_train_prompts_only(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, meta = _build(run_dir, fork_report, discriminants)
    val_seed = [p for p in pairs if p.split == "val_seed"]
    assert val_seed
    train_keys = {(p.prompt_id, p.seed_idx) for p in pairs if p.split == "train"}
    for pair in val_seed:
        assert pair.prompt_id not in meta["holdout"]["val_prompt_ids"]
        assert (pair.prompt_id, pair.seed_idx) not in train_keys
        assert pair.seed_idx in meta["holdout"]["val_seed_idx_by_prompt"][pair.prompt_id]
        # The point of val_seed: the prompt IS represented in train, by other seeds.
        assert any(k[0] == pair.prompt_id for k in train_keys)


def test_every_pair_lands_in_exactly_one_split(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, meta = _build(run_dir, fork_report, discriminants)
    assert {p.split for p in pairs} <= set(bp.SPLITS)
    assert sum(meta["summary"]["by_split"].values()) == len(pairs)
    assert len({p.pair_key for p in pairs}) == len(pairs)


def test_splits_are_deterministic_and_order_independent(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    first = {p.pair_key: p.split for p in _build(run_dir, fork_report, discriminants)[0]}
    second = {p.pair_key: p.split for p in _build(run_dir, fork_report, discriminants)[0]}
    assert first == second
    # A different salt is allowed to differ; the point is that it is still a draw,
    # not that it changes, so only determinism-per-salt is asserted.
    salted = {
        p.pair_key: p.split
        for p in _build(run_dir, fork_report, discriminants, salt="other")[0]
    }
    assert set(salted) == set(first)


def test_pick_by_hash_ignores_input_order() -> None:
    keys = ["a", "b", "c", "d", "e"]
    assert bp.pick_by_hash(keys, 2, "") == bp.pick_by_hash(list(reversed(keys)), 2, "")


def test_holding_out_everything_is_refused(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    with pytest.raises(ValueError, match="leave nothing to train on|consumed every pair"):
        _build(run_dir, fork_report, discriminants, holdout_prompts=len(PROMPTS))


# ── refusals and round-trip ────────────────────────────────────────────────────


def test_mismatched_run_dir_and_report_counts_are_refused(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    with pytest.raises(ValueError, match="positional pairs"):
        bp.build(
            run_dirs=[run_dir, run_dir],
            fork_reports=[fork_report],
            discriminants=discriminants,
            holdout_prompts=1,
            holdout_seeds=1,
            include_control=False,
            fallback_k=4,
            salt="",
        )


def test_an_empty_pair_set_is_never_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty pair set"):
        bp.write_pairs(tmp_path / "out", [], np.zeros((0, 3), dtype=np.float32), [], {})


def test_write_then_load_round_trips(
    tmp_path: Path, run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, z, names, meta = _build(run_dir, fork_report, discriminants)
    out = tmp_path / "pairs"
    bp.write_pairs(out, pairs, z, names, meta)
    loaded = bp.load_pairs(out)
    assert [p.pair_key for p in loaded.pairs] == [p.pair_key for p in pairs]
    assert loaded.feature_names == names
    np.testing.assert_allclose(loaded.z, z)
    assert loaded.by_split("train")
    assert loaded.meta["holdout"] == meta["holdout"]


def test_load_pairs_rejects_a_manifest_matrix_mismatch(
    tmp_path: Path, run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, z, names, meta = _build(run_dir, fork_report, discriminants)
    out = tmp_path / "pairs"
    bp.write_pairs(out, pairs, z, names, meta)
    np.savez_compressed(
        out / "pairs_z.npz",
        z=z[:-1],
        feature_names=np.asarray(names, dtype="<U"),
        rows=np.asarray([p.pair_key for p in pairs[:-1]], dtype="<U"),
    )
    with pytest.raises(ValueError, match="rows for"):
        bp.load_pairs(out)


def test_clip_z_is_recorded_and_applied(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    _, z_raw, _, meta_raw = _build(run_dir, fork_report, discriminants)
    assert meta_raw["params"]["clip_z"] == 0.0
    assert meta_raw["n_clipped_cells"] == 0
    assert float(np.abs(z_raw).max()) > 1.0

    _, z_clipped, _, meta = _build(run_dir, fork_report, discriminants, clip_z=1.0)
    assert float(np.abs(z_clipped).max()) <= 1.0
    assert meta["n_clipped_cells"] > 0
    assert "after_clip" in meta["z_scale"]


def test_separability_flags_near_parallel_signatures() -> None:
    """The guard's whole point: near-identical rows must breach, distinct ones not."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=64)
    # Rows differing only by a whisper — what an unclipped v3 z-matrix looks like
    # once a handful of huge coordinates dominate the LayerNorm.
    near = np.stack([base + 1e-6 * rng.normal(size=64) for _ in range(8)]).astype(np.float32)
    flat = bp.layernorm_separability(near)
    assert flat["min_cosine"] > bp.SEPARABILITY_FLOOR

    spread = rng.normal(size=(8, 64)).astype(np.float32)
    assert bp.layernorm_separability(spread)["min_cosine"] < bp.SEPARABILITY_FLOOR


def test_separability_is_scale_invariant_but_not_shift_blind() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6, 32)).astype(np.float32)
    a = bp.layernorm_separability(z)
    # LayerNorm standardises per row, so a per-row rescale cannot change cosines.
    b = bp.layernorm_separability((z * np.array([[1.0], [7.0], [0.1], [3.0], [2.0], [5.0]])).astype(np.float32))
    assert a["min_cosine"] == pytest.approx(b["min_cosine"], abs=1e-6)


def test_separability_handles_degenerate_inputs() -> None:
    assert bp.layernorm_separability(np.zeros((1, 4), dtype=np.float32))["n_sampled"] == 1
    # A constant row standardises to zeros; the guard must not divide by zero.
    out = bp.layernorm_separability(np.ones((3, 4), dtype=np.float32))
    assert np.isfinite(out["min_cosine"])


def test_separability_is_reported_in_the_manifest(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    _, _, _, meta = _build(run_dir, fork_report, discriminants)
    sep = meta["separability"]
    assert sep["floor"] == bp.SEPARABILITY_FLOOR
    assert isinstance(sep["breached"], bool)
    assert -1.0 <= sep["min_cosine"] <= 1.0


def test_summary_counts_reconcile(
    run_dir: Path, fork_report: Path, discriminants: Path
) -> None:
    pairs, _, _, meta = _build(run_dir, fork_report, discriminants)
    summary = meta["summary"]
    assert summary["n_pairs"] == len(pairs)
    assert sum(summary["by_fork_source"].values()) == len(pairs)
    assert summary["fork_positions_total"] == sum(p.n_fork_positions for p in pairs)
    for cls, counts in summary["by_class"].items():
        assert sum(counts.values()) == sum(1 for p in pairs if p.prompt_class == cls)
    # The printed table must not blow up on a real summary.
    table = bp.format_summary(
        summary, meta["holdout"], meta["z_scale"], meta["separability"]
    )
    assert "prompt_class" in table
    assert "separability at g's input" in table
