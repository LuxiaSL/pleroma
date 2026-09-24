"""Contract tests for stage 3 against a synthetic fork_series fixture.

The fixture plants everything the stage claims to measure: a known divergence
point between siblings, a known entropy ranking, a divergence at continuation
index 0 (no banked distribution), and a realized token outside the banked top-k.
No torch, no anamnesis, no GPU.

Fixture layout (prompt = [1, 2, 3], so prompt_length = 3; all three gens are seeds
of one ``prompt_id``, which is what makes them positionally comparable):

    gen 000  continuation [10, 11, 12, 13, 14]   entropy [0.1, 2.0, 0.5, 1.0]
    gen 001  continuation [10, 11, 99, 98, 97]
    gen 002  continuation [20, 21, 22, 23, 24]

gen 000 therefore forks from 001 at continuation index 2 and from 002 at index 0.
Its top-2 entropy steps are t=1 and t=3, i.e. continuation indices 2 and 4 — so
index 2 is exactly where both notions agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.map.build import forks as ft

PROMPT: list[int] = [1, 2, 3]
TOP_K = 3
PROMPT_ID = "dc1"
PROMPT_CLASS = "decision_continuation"


def _write_gen(
    run_dir: Path,
    gen_id: int,
    continuation: list[int],
    entropy: list[float],
    logits_indices: list[list[int]],
    logits_values: list[list[float]],
) -> None:
    input_ids = PROMPT + continuation
    np.savez_compressed(
        run_dir / "fork_series" / f"gen_{gen_id:03d}.npz",
        input_ids=np.array(input_ids, dtype=np.int32),
        prompt_length=np.array(len(PROMPT), dtype=np.int32),
        logits_entropy=np.array(entropy, dtype=np.float32),
        logits_values=np.array(logits_values, dtype=np.float32),
        logits_indices=np.array(logits_indices, dtype=np.int32),
        # chosen_ids[t] = continuation[t + 1] — stage 2's banked alignment.
        chosen_ids=np.array(continuation[1:], dtype=np.int32),
    )


def _cell(
    gen_id: int, seed_idx: int, prompt_id: str = PROMPT_ID, prompt_class: str = PROMPT_CLASS
) -> dict[str, Any]:
    return {
        "generation_id": gen_id,
        "prompt_id": prompt_id,
        "prompt_class": prompt_class,
        "prompt": "a synthetic prompt",
        "prompt_idx": 0,
        "seed_idx": seed_idx,
        "prompt_length": len(PROMPT),
    }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A three-seed, one-prompt synthetic stage-2 output directory."""
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)

    # gen 000 — the trajectory under test.
    # step 0 chooses 11 (in top-k), step 1 chooses 12 (NOT in top-k -> flagged),
    # step 2 chooses 13, step 3 chooses 14.
    _write_gen(
        root,
        0,
        continuation=[10, 11, 12, 13, 14],
        entropy=[0.1, 2.0, 0.5, 1.0],
        logits_indices=[[11, 5, 6], [7, 8, 9], [13, 5, 6], [14, 5, 6]],
        logits_values=[[3.0, 1.0, 0.5], [3.0, 1.0, 0.5], [2.0, 1.0, 0.0], [1.0, 0.5, 0.25]],
    )
    # Siblings. Their own series content is irrelevant to gen 000's forks, but
    # must be well-formed (T = N - 1, chosen_ids aligned) to load at all.
    for gid, cont in ((1, [10, 11, 99, 98, 97]), (2, [20, 21, 22, 23, 24])):
        _write_gen(
            root,
            gid,
            continuation=cont,
            entropy=[0.2, 0.2, 0.2, 0.2],
            logits_indices=[[c, 0, 1] for c in cont[1:]],
            logits_values=[[2.0, 1.0, 0.0]] * 4,
        )

    (root / "cells.json").write_text(json.dumps([_cell(g, g) for g in (0, 1, 2)]))
    return root


def _gen(payload: dict[str, Any], gen_id: int) -> dict[str, Any]:
    return next(g for g in payload["gens"] if g["generation_id"] == gen_id)


def _fork(gen: dict[str, Any], position: int) -> dict[str, Any]:
    return next(f for f in gen["forks"] if f["position_continuation"] == position)


# ── divergence forks ──────────────────────────────────────────────────────────


def test_divergence_positions_are_the_planted_ones(run_dir: Path) -> None:
    payload = ft.analyse_run(run_dir, top_m=2)
    gen0 = _gen(payload, 0)
    assert gen0["divergence_positions"] == [0, 2]
    assert _fork(gen0, 2)["diverged_from"] == [1]
    assert _fork(gen0, 0)["diverged_from"] == [2]


def test_divergence_is_symmetric_between_siblings(run_dir: Path) -> None:
    payload = ft.analyse_run(run_dir, top_m=2)
    assert _gen(payload, 1)["divergence_positions"] == [0, 2]
    assert _fork(_gen(payload, 1), 2)["diverged_from"] == [0]


def test_siblings_are_grouped_by_prompt_id_not_class(tmp_path: Path) -> None:
    """Two prompts of the SAME class must not be compared position by position."""
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    for gid, cont in ((0, [10, 11, 12, 13, 14]), (1, [50, 51, 52, 53, 54])):
        _write_gen(
            root, gid, cont, [0.1, 0.2, 0.3, 0.4],
            [[c, 0, 1] for c in cont[1:]], [[2.0, 1.0, 0.0]] * 4,
        )
    # Same class, DIFFERENT prompt_id -> not siblings.
    (root / "cells.json").write_text(
        json.dumps([_cell(0, 0, "dc1"), _cell(1, 0, "dc2")])
    )
    payload = ft.analyse_run(root, top_m=2)
    assert _gen(payload, 0)["divergence_positions"] == []
    assert _gen(payload, 1)["divergence_positions"] == []
    assert payload["summary"]["n_prompts"] == 2
    assert payload["summary"]["n_classes"] == 1


def test_identical_siblings_produce_no_divergence(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    for gid in (0, 1):
        _write_gen(
            root, gid, [10, 11, 12, 13, 14], [0.1, 0.2, 0.3, 0.4],
            [[c, 0, 1] for c in [11, 12, 13, 14]], [[2.0, 1.0, 0.0]] * 4,
        )
    (root / "cells.json").write_text(json.dumps([_cell(g, g) for g in (0, 1)]))
    payload = ft.analyse_run(root, top_m=2)
    assert _gen(payload, 0)["divergence_positions"] == []
    assert payload["summary"]["divergence_in_uncertainty_micro"] is None


# ── uncertainty forks ─────────────────────────────────────────────────────────


def test_uncertainty_ranking_follows_entropy(run_dir: Path) -> None:
    gen0 = _gen(ft.analyse_run(run_dir, top_m=2), 0)
    # entropy [0.1, 2.0, 0.5, 1.0] at steps 0..3 -> continuation indices 1..4.
    assert gen0["uncertainty_positions"] == [2, 4]
    assert _fork(gen0, 2)["entropy_rank"] == 0   # the 2.0
    assert _fork(gen0, 4)["entropy_rank"] == 1   # the 1.0
    assert _fork(gen0, 2)["entropy"] == pytest.approx(2.0)


def test_top_m_larger_than_the_series_is_clamped(run_dir: Path) -> None:
    gen0 = _gen(ft.analyse_run(run_dir, top_m=99), 0)
    assert gen0["uncertainty_positions"] == [1, 2, 3, 4]  # all four banked steps


# ── surprise honesty ──────────────────────────────────────────────────────────


def test_token_outside_top_k_gets_null_surprise_and_a_flag(run_dir: Path) -> None:
    fork = _fork(_gen(ft.analyse_run(run_dir, top_m=2), 0), 2)
    # step 1's top-k is [7, 8, 9]; the realized token is 12.
    assert fork["surprise_nats_approx"] is None
    assert fork["surprise_flag"] == f"outside_top{TOP_K}"


def test_in_top_k_surprise_matches_the_truncated_log_softmax(run_dir: Path) -> None:
    fork = _fork(_gen(ft.analyse_run(run_dir, top_m=2), 0), 4)
    expected = float(-ft.log_softmax(np.array([1.0, 0.5, 0.25], dtype=np.float32))[0])
    assert fork["surprise_flag"] is None
    assert fork["surprise_nats_approx"] == pytest.approx(expected)


def test_continuation_index_zero_has_no_banked_distribution(run_dir: Path) -> None:
    fork = _fork(_gen(ft.analyse_run(run_dir, top_m=2), 0), 0)
    assert fork["entropy"] is None
    assert fork["surprise_nats_approx"] is None
    assert fork["surprise_flag"] == "no_banked_distribution"
    assert fork["notions"] == ["divergence"]


# ── positions, windows, summary ───────────────────────────────────────────────


def test_positions_are_reported_in_both_coordinate_systems(run_dir: Path) -> None:
    fork = _fork(_gen(ft.analyse_run(run_dir, top_m=2), 0), 2)
    assert fork["position_absolute"] == len(PROMPT) + 2
    assert fork["token_id"] == 12


def test_window_is_pm5_tokens_clipped_to_the_sequence(run_dir: Path) -> None:
    gen0 = _gen(ft.analyse_run(run_dir, top_m=2), 0)
    full = PROMPT + [10, 11, 12, 13, 14]
    fork = _fork(gen0, 2)          # absolute 5, window [0..10] clipped to len 8
    assert fork["window_start_absolute"] == 0
    assert fork["window_ids"] == full
    assert len(fork["window_ids"]) <= 2 * ft.WINDOW_RADIUS + 1


def test_overlap_of_the_two_notions(run_dir: Path) -> None:
    payload = ft.analyse_run(run_dir, top_m=2)
    gen0 = _gen(payload, 0)
    assert gen0["overlap_positions"] == [2]
    assert _fork(gen0, 2)["notions"] == ["divergence", "uncertainty"]
    # divergence {0, 2} vs uncertainty {2, 4}
    assert gen0["divergence_in_uncertainty"] == pytest.approx(0.5)
    assert gen0["jaccard"] == pytest.approx(1 / 3)
    assert gen0["chance_overlap"] == pytest.approx(2 / 5)
    summary = payload["summary"]
    assert summary["n_gens"] == 3
    assert summary["n_prompts"] == 1
    assert summary["n_classes"] == 1
    assert summary["n_positions_outside_top_k"] >= 1


def test_summary_is_broken_down_per_prompt_class(run_dir: Path) -> None:
    summary = ft.analyse_run(run_dir, top_m=2)["summary"]
    assert set(summary["by_class"]) == {PROMPT_CLASS}
    assert summary["by_class"][PROMPT_CLASS]["n_gens"] == 3
    assert summary["by_class"][PROMPT_CLASS]["first_divergence_position_mean"] == pytest.approx(0.0)


def test_alignment_offset_is_detected_not_assumed(run_dir: Path) -> None:
    series = ft.load_fork_series(run_dir / "fork_series" / "gen_000.npz")
    assert ft.detect_alignment_offset(series) == 1


# ── refusals ──────────────────────────────────────────────────────────────────


def test_missing_cells_json_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="cells.json"):
        ft.analyse_run(root, top_m=2)


def test_cells_without_prompt_id_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    row = _cell(0, 0)
    del row["prompt_id"]
    (root / "cells.json").write_text(json.dumps([row]))
    with pytest.raises(KeyError, match="prompt_id"):
        ft.analyse_run(root, top_m=2)


def test_empty_fork_series_refuses(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    (root / "cells.json").write_text(json.dumps([_cell(0, 0)]))
    with pytest.raises(ValueError, match="no usable fork series"):
        ft.analyse_run(root, top_m=2)


def test_misaligned_chosen_ids_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    (root / "fork_series").mkdir(parents=True)
    np.savez_compressed(
        root / "fork_series" / "gen_000.npz",
        input_ids=np.array(PROMPT + [10, 11, 12, 13, 14], dtype=np.int32),
        prompt_length=np.array(3, dtype=np.int32),
        logits_entropy=np.zeros(4, dtype=np.float32),
        logits_values=np.zeros((4, TOP_K), dtype=np.float32),
        logits_indices=np.zeros((4, TOP_K), dtype=np.int32),
        chosen_ids=np.array([777, 778, 779, 780], dtype=np.int32),  # match nothing
    )
    series = ft.load_fork_series(root / "fork_series" / "gen_000.npz")
    with pytest.raises(ValueError, match="offset"):
        ft.detect_alignment_offset(series)


def test_malformed_series_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gen_000.npz"
    np.savez_compressed(
        path,
        input_ids=np.array(PROMPT + [10, 11, 12], dtype=np.int32),
        prompt_length=np.array(3, dtype=np.int32),
        logits_entropy=np.zeros(2, dtype=np.float32),
        logits_values=np.zeros((5, TOP_K), dtype=np.float32),  # wrong T
        logits_indices=np.zeros((2, TOP_K), dtype=np.int32),
        chosen_ids=np.array([11, 12], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="logits_values"):
        ft.load_fork_series(path)


def test_main_writes_json_and_returns_zero(run_dir: Path, tmp_path: Path) -> None:
    import sys as _sys

    out = tmp_path / "out" / "fork_tokens.json"
    old = _sys.argv
    _sys.argv = [
        "fork_tokens", "--run-dir", str(run_dir), "--out", str(out), "--top-m-entropy", "2"
    ]
    try:
        assert ft.main() == 0
    finally:
        _sys.argv = old
    payload = json.loads(out.read_text())
    assert payload["stage"] == "expB0_fork_tokens"
    assert len(payload["gens"]) == 3
    assert payload["gens"][0]["prompt_id"] == PROMPT_ID
    assert payload["gens"][0]["prompt_class"] == PROMPT_CLASS
