"""Corpus stage 3: where did the loomed futures actually fork? (local CPU, numpy only)

Two fork notions, deliberately kept separate and both emitted per trajectory, so
the prediction that they agree is a measurement rather than a definition:

divergence forks
    Siblings — same ``prompt_id``, different ``seed_idx`` — share a byte-identical
    prompt, so their continuations are positionally comparable. For each sibling
    pair we take the FIRST index at which they differ. The set of those
    first-divergence points is the empirical fork structure of the prompt — what
    the loom actually did.

uncertainty forks
    The top-M positions by banked ``logits_entropy``, plus the realized surprise
    of the token the model went on to pick. Where the model *said* it was unsure.

The prediction (P ≈ .8) is that these agree. The summary block
reports the overlap against a chance baseline, so "they agree" is not read off a
number that would be large anyway — and it reports it per prompt CLASS as well as
pooled, because the classes are expected to fork at very different places (a
``decision_continuation`` seeds an explicit binary fork; a ``convergent_control``
is not expected to fork at all) and a pooled number would average over exactly the
variable under test. If the two notions *disagree*, that is the more interesting
result — commitments made at low-entropy tokens.

Two honesty constraints, both load-bearing:

  - Surprise is computed over the banked top-k truncation, not the full vocab, so
    it is an approximation and is named one (``surprise_nats_approx``). When the
    realized token is NOT in the top-k, the surprise is ``null`` with the flag
    ``outside_top50`` — never a clamped or invented "max surprise" number, which
    would be indistinguishable from a real measurement downstream.
  - Continuation index 0 has no banked predictive distribution: stage 2's replay
    banks T = N - 1 entries, entry ``t`` predicting continuation index ``t + 1``.
    A fork there gets ``null`` entropy with the flag
    ``no_banked_distribution``.

Reads stage 2's output only (``fork_series/`` + the cells manifest, see
`load_cells`); no torch, no anamnesis, no GPU.

Usage (local)::

    python -m pleroma.map.build.forks \\
        --run-dir outputs/expB0_3b/replay \\
        --out outputs/expB0_3b/fork_tokens.json --top-m-entropy 8
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy import (house rule; this stage is pure numpy).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB0.forks")

#: Half-width of the token-id window emitted around every fork position.
WINDOW_RADIUS = 5


@dataclass(frozen=True)
class ForkSeries:
    """One gen's banked per-position surface (stage 2's ``fork_series/gen_NNN.npz``).

    ``entropy``/``logits_values``/``logits_indices``/``chosen_ids`` are all indexed
    by the per-step index ``t`` (T = N - 1 of them); ``input_ids`` is the full
    realized sequence and ``prompt_length`` splits it.
    """

    generation_id: int
    input_ids: NDArray[np.int32]
    prompt_length: int
    entropy: NDArray[np.float32]
    logits_values: NDArray[np.float32]
    logits_indices: NDArray[np.int32]
    chosen_ids: NDArray[np.int32]

    @property
    def continuation(self) -> NDArray[np.int32]:
        return self.input_ids[self.prompt_length :]

    @property
    def n_steps(self) -> int:
        return int(self.entropy.shape[0])


@dataclass(frozen=True)
class GenCell:
    """Stage-1 metadata carried forward by stage 2's cells manifest (`load_cells`).

    ``prompt_id`` is the sibling key: same prompt_id + different seed_idx = the
    forks of one prompt, and the only pairs whose continuations are positionally
    comparable. ``prompt_class`` is the experimental variable stage 4 groups on.
    """

    generation_id: int
    prompt_id: str
    prompt_class: str
    seed_idx: int
    prompt: str | None = None
    prompt_idx: int | None = None
    prompt_length: int | None = None


@dataclass
class ForkEvent:
    """One fork position on one trajectory, with everything both notions know."""

    position_continuation: int
    position_absolute: int
    token_id: int | None
    notions: list[str]
    entropy: float | None = None
    entropy_rank: int | None = None
    surprise_nats_approx: float | None = None
    surprise_flag: str | None = None
    diverged_from: list[int] = field(default_factory=list)
    window_start_absolute: int = 0
    window_ids: list[int] = field(default_factory=list)


@dataclass
class GenForks:
    """Everything stage 4 and the pairs build need about one trajectory's forks."""

    generation_id: int
    prompt_id: str
    prompt_class: str
    seed_idx: int
    prompt_length: int
    n_continuation_tokens: int
    n_steps: int
    chosen_alignment_offset: int
    divergence_positions: list[int]
    uncertainty_positions: list[int]
    overlap_positions: list[int]
    divergence_in_uncertainty: float | None
    jaccard: float | None
    chance_overlap: float | None
    forks: list[ForkEvent]


def log_softmax(values: NDArray[np.float32]) -> NDArray[np.float64]:
    """Numerically stable log-softmax in float64 (expA house helper)."""
    x = np.asarray(values, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def load_fork_series(path: Path) -> ForkSeries:
    """Load one ``fork_series/gen_NNN.npz``, validating the shapes stage 3 relies on."""
    with np.load(path) as npz:
        required = (
            "input_ids",
            "prompt_length",
            "logits_entropy",
            "logits_values",
            "logits_indices",
            "chosen_ids",
        )
        missing = [k for k in required if k not in npz]
        if missing:
            raise KeyError(f"{path}: missing {missing} (has {list(npz.files)})")
        input_ids = np.asarray(npz["input_ids"], dtype=np.int32)
        prompt_length = int(np.asarray(npz["prompt_length"]).reshape(-1)[0])
        entropy = np.asarray(npz["logits_entropy"], dtype=np.float32)
        values = np.asarray(npz["logits_values"], dtype=np.float32)
        indices = np.asarray(npz["logits_indices"], dtype=np.int32)
        chosen = np.asarray(npz["chosen_ids"], dtype=np.int32)

    gen_id = int(path.stem.split("_")[-1])
    n_steps = int(entropy.shape[0])
    for name, arr in (("logits_values", values), ("logits_indices", indices)):
        if arr.ndim != 2 or arr.shape[0] != n_steps:
            raise ValueError(
                f"{path}: {name} has shape {arr.shape}, expected ({n_steps}, k)"
            )
    if chosen.shape[0] != n_steps:
        raise ValueError(
            f"{path}: chosen_ids has {chosen.shape[0]} entries, entropy has {n_steps}"
        )
    if not 0 < prompt_length < input_ids.shape[0]:
        raise ValueError(
            f"{path}: prompt_length {prompt_length} out of range for "
            f"{input_ids.shape[0]} input_ids"
        )
    return ForkSeries(
        generation_id=gen_id,
        input_ids=input_ids,
        prompt_length=prompt_length,
        entropy=entropy,
        logits_values=values,
        logits_indices=indices,
        chosen_ids=chosen,
    )


def load_cells(run_dir: Path) -> dict[int, GenCell]:
    """Read ``<run-dir>/cells.json``, stage 2's copy of the stage-1 metadata.

    Raises FileNotFoundError when the manifest is absent."""
    cells_path = run_dir / "cells.json"
    if not cells_path.exists():
        raise FileNotFoundError(
            f"{cells_path} not found — stage 2 copies stage-1 metadata there so this "
            "stage needs no --gen-dir"
        )
    blob = json.loads(cells_path.read_text())
    rows: list[dict[str, Any]] = blob["cells"] if isinstance(blob, dict) else blob
    out: dict[int, GenCell] = {}
    for row in rows:
        gid = int(row["generation_id"])
        for key in ("prompt_id", "prompt_class", "seed_idx"):
            if row.get(key) is None:
                raise KeyError(f"{cells_path}: gen {gid} is missing {key!r}")
        out[gid] = GenCell(
            generation_id=gid,
            prompt_id=str(row["prompt_id"]),
            prompt_class=str(row["prompt_class"]),
            seed_idx=int(row["seed_idx"]),
            prompt=row.get("prompt"),
            prompt_idx=(
                int(row["prompt_idx"]) if row.get("prompt_idx") is not None else None
            ),
            prompt_length=(
                int(row["prompt_length"]) if row.get("prompt_length") is not None else None
            ),
        )
    if not out:
        raise ValueError(f"{cells_path} holds no cells")
    return out


def detect_alignment_offset(series: ForkSeries) -> int:
    """Return ``o`` with ``chosen_ids[t] == continuation[t + o]`` for every t.

    Stage 2 banks ``o = 1`` (its chosen ids are g_1..g_{N-1}).
    We verify rather than assume, because every position this stage reports is
    stated in continuation coordinates and a silent off-by-one would put every
    fork one token away from the token that caused it.
    """
    cont = series.continuation
    n_steps = series.n_steps
    for offset in (1, 0):
        end = n_steps + offset
        if end <= cont.shape[0] and np.array_equal(cont[offset:end], series.chosen_ids):
            return offset
    raise ValueError(
        f"gen_{series.generation_id:03d}: chosen_ids match the continuation at neither "
        "offset 1 nor 0 — the banked series and input_ids disagree"
    )


def divergence_forks(series_by_gen: dict[int, ForkSeries]) -> dict[int, dict[int, list[int]]]:
    """First-divergence positions per gen, keyed by continuation index -> siblings.

    Callers pass ONE prompt_id's gens. Siblings whose prompts differ in length are
    skipped for that pair (they are not positionally comparable) rather than
    compared anyway; stage 1 already refuses to bank such a sibling, so this is a
    belt-and-braces guard on a hand-assembled run dir.
    """
    gen_ids = sorted(series_by_gen)
    result: dict[int, dict[int, list[int]]] = {g: {} for g in gen_ids}
    for i, a_id in enumerate(gen_ids):
        a = series_by_gen[a_id]
        for b_id in gen_ids[i + 1 :]:
            b = series_by_gen[b_id]
            if a.prompt_length != b.prompt_length:
                logger.warning(
                    "gen_%.3d/gen_%.3d share a prompt_id but not a prompt length "
                    "(%d vs %d) — skipping the pair",
                    a_id, b_id, a.prompt_length, b.prompt_length,
                )
                continue
            ca, cb = a.continuation, b.continuation
            n = int(min(ca.shape[0], cb.shape[0]))
            diff = np.nonzero(ca[:n] != cb[:n])[0]
            if diff.size:
                pos = int(diff[0])
            elif ca.shape[0] != cb.shape[0]:
                # Identical up to the shorter one's end: one continuation stopped.
                pos = n
            else:
                continue  # byte-identical siblings — no fork between these two
            result[a_id].setdefault(pos, []).append(b_id)
            result[b_id].setdefault(pos, []).append(a_id)
    return {g: {p: sorted(v) for p, v in sorted(d.items())} for g, d in result.items()}


def uncertainty_forks(series: ForkSeries, top_m: int, offset: int) -> tuple[list[int], dict[int, int]]:
    """Top-M positions by entropy, in continuation coordinates, plus their ranks.

    Ties broken by earliest position, so the selection is deterministic.
    """
    n_steps = series.n_steps
    if n_steps == 0 or top_m <= 0:
        return [], {}
    order = np.lexsort((np.arange(n_steps), -series.entropy.astype(np.float64)))
    chosen_steps = [int(t) for t in order[: min(top_m, n_steps)]]
    positions = [t + offset for t in chosen_steps]
    ranks = {pos: rank for rank, pos in enumerate(positions)}
    return sorted(positions), ranks


def realized_surprise(
    series: ForkSeries, step: int, token_id: int
) -> tuple[float | None, str | None]:
    """-log p(token) over the banked top-k truncation, or (None, flag).

    Approximate by construction: the denominator is the top-k mass, not the vocab,
    so this slightly UNDER-states surprise for in-top-k tokens. Outside the top-k
    there is no measurement at all, and we say so instead of clamping.
    """
    if step < 0 or step >= series.n_steps:
        return None, "no_banked_distribution"
    row = series.logits_indices[step]
    hit = np.nonzero(row == np.int32(token_id))[0]
    if hit.size == 0:
        return None, f"outside_top{row.shape[0]}"
    return float(-log_softmax(series.logits_values[step])[int(hit[0])]), None


def build_gen_forks(
    cell: GenCell,
    series: ForkSeries,
    divergences: dict[int, list[int]],
    top_m: int,
) -> GenForks:
    """Assemble one trajectory's fork record from both notions."""
    offset = detect_alignment_offset(series)
    cont = series.continuation
    n_cont = int(cont.shape[0])
    uncertainty_positions, ranks = uncertainty_forks(series, top_m, offset)

    div_set = set(divergences)
    unc_set = set(uncertainty_positions)
    overlap = sorted(div_set & unc_set)
    union = div_set | unc_set

    events: list[ForkEvent] = []
    for pos in sorted(union):
        notions = [
            name
            for name, present in (("divergence", pos in div_set), ("uncertainty", pos in unc_set))
            if present
        ]
        abs_pos = series.prompt_length + pos
        token_id = int(series.input_ids[abs_pos]) if abs_pos < series.input_ids.shape[0] else None
        step = pos - offset
        if 0 <= step < series.n_steps:
            entropy: float | None = float(series.entropy[step])
        else:
            entropy = None
        if token_id is None:
            surprise, flag = None, "past_end_of_sequence"
        elif entropy is None:
            surprise, flag = None, "no_banked_distribution"
        else:
            surprise, flag = realized_surprise(series, step, token_id)

        start = max(0, abs_pos - WINDOW_RADIUS)
        stop = min(int(series.input_ids.shape[0]), abs_pos + WINDOW_RADIUS + 1)
        events.append(
            ForkEvent(
                position_continuation=pos,
                position_absolute=abs_pos,
                token_id=token_id,
                notions=notions,
                entropy=entropy,
                entropy_rank=ranks.get(pos),
                surprise_nats_approx=surprise,
                surprise_flag=flag,
                diverged_from=list(divergences.get(pos, [])),
                window_start_absolute=start,
                window_ids=[int(x) for x in series.input_ids[start:stop]],
            )
        )

    n_div = len(div_set)
    return GenForks(
        generation_id=cell.generation_id,
        prompt_id=cell.prompt_id,
        prompt_class=cell.prompt_class,
        seed_idx=cell.seed_idx,
        prompt_length=series.prompt_length,
        n_continuation_tokens=n_cont,
        n_steps=series.n_steps,
        chosen_alignment_offset=offset,
        divergence_positions=sorted(div_set),
        uncertainty_positions=uncertainty_positions,
        overlap_positions=overlap,
        divergence_in_uncertainty=(len(overlap) / n_div if n_div else None),
        jaccard=(len(overlap) / len(union) if union else None),
        # If the uncertainty set were placed at random among the comparable
        # positions, this is the overlap we would expect — the baseline the
        # predicted P ≈ .8 has to beat to mean anything.
        chance_overlap=(len(unc_set) / n_cont if n_cont else None),
        forks=events,
    )


def _overlap_block(gens: list[GenForks]) -> dict[str, Any]:
    """The two notions' agreement over one group of trajectories."""
    per_gen = [g.divergence_in_uncertainty for g in gens if g.divergence_in_uncertainty is not None]
    total_div = sum(len(g.divergence_positions) for g in gens)
    total_overlap = sum(len(g.overlap_positions) for g in gens)
    chance = [g.chance_overlap for g in gens if g.chance_overlap is not None]
    jaccards = [g.jaccard for g in gens if g.jaccard is not None]
    outside = sum(
        1
        for g in gens
        for e in g.forks
        if e.surprise_flag and e.surprise_flag.startswith("outside_top")
    )
    return {
        "n_gens": len(gens),
        "divergence_forks_per_gen_mean": (
            float(np.mean([len(g.divergence_positions) for g in gens])) if gens else None
        ),
        "uncertainty_forks_per_gen_mean": (
            float(np.mean([len(g.uncertainty_positions) for g in gens])) if gens else None
        ),
        "forks_per_gen_mean": float(np.mean([len(g.forks) for g in gens])) if gens else None,
        "first_divergence_position_mean": (
            float(np.mean([min(g.divergence_positions) for g in gens if g.divergence_positions]))
            if any(g.divergence_positions for g in gens)
            else None
        ),
        # The P ~ .8 prediction reads off these two lines.
        "divergence_in_uncertainty_micro": (total_overlap / total_div if total_div else None),
        "divergence_in_uncertainty_macro": (float(np.mean(per_gen)) if per_gen else None),
        "jaccard_macro": float(np.mean(jaccards)) if jaccards else None,
        "chance_overlap_macro": float(np.mean(chance)) if chance else None,
        "n_positions_outside_top_k": outside,
    }


def summarize(gens: list[GenForks]) -> dict[str, Any]:
    """Run-level agreement, plus the same block per prompt class.

    Class is an experimental variable, and the classes differ
    in how early they fork (decision_continuation seeds an explicit binary fork;
    convergent_control is not expected to fork at all), so a single pooled overlap
    number would average over exactly the thing under test.
    """
    by_class: dict[str, list[GenForks]] = {}
    for gen in gens:
        by_class.setdefault(gen.prompt_class, []).append(gen)
    return {
        **_overlap_block(gens),
        "n_prompts": len({g.prompt_id for g in gens}),
        "n_classes": len(by_class),
        "by_class": {cls: _overlap_block(group) for cls, group in sorted(by_class.items())},
    }


def analyse_run(run_dir: Path, top_m: int) -> dict[str, Any]:
    """Load a stage-2 run dir and produce the full stage-3 payload."""
    cells = load_cells(run_dir)
    fork_dir = run_dir / "fork_series"
    if not fork_dir.is_dir():
        raise FileNotFoundError(f"{fork_dir} not found (stage 2 banks it for every gen)")

    series_by_gen: dict[int, ForkSeries] = {}
    failures: list[str] = []
    for path in sorted(fork_dir.glob("gen_*.npz")):
        try:
            series = load_fork_series(path)
        except (OSError, KeyError, ValueError) as exc:
            logger.error("%s unreadable: %s", path.name, exc)
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if series.generation_id not in cells:
            logger.warning("%s has no cells-manifest entry — skipping", path.name)
            failures.append(f"{path.name}: no cells.json entry")
            continue
        series_by_gen[series.generation_id] = series

    if not series_by_gen:
        raise ValueError(f"no usable fork series under {fork_dir}")

    # Siblings = same prompt_id, different seed_idx. Grouping on anything coarser
    # (class, say) would compare continuations of DIFFERENT prompts position by
    # position, which is meaningless.
    by_prompt: dict[str, dict[int, ForkSeries]] = {}
    for gid, series in series_by_gen.items():
        by_prompt.setdefault(cells[gid].prompt_id, {})[gid] = series
    logger.info(
        "%d gens over %d prompts / %d classes (top-M entropy = %d)",
        len(series_by_gen), len(by_prompt),
        len({cells[g].prompt_class for g in series_by_gen}), top_m,
    )

    gens: list[GenForks] = []
    for prompt_id in sorted(by_prompt):
        group = by_prompt[prompt_id]
        if len(group) < 2:
            logger.warning(
                "prompt_id %s has %d gen(s): no siblings, so no divergence forks",
                prompt_id, len(group),
            )
        divergences = divergence_forks(group)
        for gid in sorted(group):
            try:
                gens.append(
                    build_gen_forks(cells[gid], group[gid], divergences.get(gid, {}), top_m)
                )
            except ValueError as exc:
                logger.error("gen_%.3d skipped: %s", gid, exc)
                failures.append(f"gen_{gid:03d}: {type(exc).__name__}: {exc}")

    if not gens:
        raise ValueError("no trajectory produced a fork record")

    return {
        "stage": "expB0_fork_tokens",
        "run_dir": str(run_dir),
        "params": {
            "top_m_entropy": top_m,
            "window_radius": WINDOW_RADIUS,
            "surprise": (
                "-log p over the banked top-k truncation (APPROXIMATE: denominator is "
                "the top-k mass, not the vocab). null + outside_topK when the realized "
                "token is not in the top-k; never clamped."
            ),
            "alignment": (
                "positions are continuation-relative; entropy index t corresponds to "
                "continuation index t + chosen_alignment_offset (stage 2 banks offset 1)"
            ),
        },
        "summary": summarize(gens),
        "gens": [_gen_to_json(g) for g in gens],
        "failures": failures,
    }


def _gen_to_json(gen: GenForks) -> dict[str, Any]:
    """dataclasses.asdict, but with the fork list nested explicitly (typed, stable)."""
    payload = {k: v for k, v in vars(gen).items() if k != "forks"}
    payload["forks"] = [vars(e) for e in gen.forks]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="stage-2 out-dir (fork_series/ + cells.json)"
    )
    parser.add_argument("--out", type=Path, required=True, help="output JSON")
    parser.add_argument(
        "--top-m-entropy", type=int, default=8, help="uncertainty forks per trajectory"
    )
    args = parser.parse_args()

    if args.top_m_entropy <= 0:
        logger.error("--top-m-entropy must be > 0, got %s", args.top_m_entropy)
        return 2

    try:
        payload = analyse_run(args.run_dir, args.top_m_entropy)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("stage 3 failed: %s", exc)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    summary = payload["summary"]
    logger.info(
        "DONE — %d gens, %.2f divergence + %.2f uncertainty forks/gen",
        summary["n_gens"],
        summary["divergence_forks_per_gen_mean"] or 0.0,
        summary["uncertainty_forks_per_gen_mean"] or 0.0,
    )
    logger.info(
        "overlap: divergence-in-uncertainty %.3f micro / %.3f macro (chance %.3f) -> %s",
        summary["divergence_in_uncertainty_micro"] or 0.0,
        summary["divergence_in_uncertainty_macro"] or 0.0,
        summary["chance_overlap_macro"] or 0.0,
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
