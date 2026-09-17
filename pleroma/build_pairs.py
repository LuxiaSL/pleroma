"""Exp B1 stage 1: B0 run dirs -> the (z-signature, fork positions) training pairs.

Local CPU, numpy + stdlib only. No torch, no anamnesis, no GPU — this is the stage
that decides *what* g is trained on, so it must be runnable and re-runnable on a
laptop, and every choice it makes has to be visible in the manifest it writes.

What one pair is
    One B0 trajectory: its 2713-d v3 signature, z-scored into the space the 3B
    discriminants were fit in, plus the continuation positions at which that
    trajectory made a choice its siblings did not. B1's loss is the NLL of the
    trajectory's OWN tokens at exactly those positions, so the position list is
    half the experiment and is not recomputed downstream.

Three things this stage refuses to do, all on purpose:

REFUSES a feature-name mismatch. Every gen's ``feature_names`` must equal the
    discriminants' ``FULL_names`` exactly. B0's Gate 0 already asserted this at
    extraction time; asserting it again here is cheap and catches a run dir that
    was extracted under a different family config being mixed into one training
    set, which no downstream number would reveal.

REFUSES to invent a divergence at continuation index 0. Position 0 has no banked
    predictive distribution (B0 stage 2 banks T = N - 1 entries; entry ``t``
    predicts continuation index ``t+1``), so an NLL there is not computable from
    the same forward that produced the rest. p == 0 is dropped, and a gen left
    with nothing falls back to its top-4 entropy positions — with the fallback
    RECORDED per gen, because "this gen was trained on an entropy heuristic" and
    "this gen was trained on realized sibling divergences" are different data and
    the B0 results (§2) say so explicitly.

REFUSES to leak a held-out prompt into train. ``val_prompt`` holds out WHOLE
    prompt_ids (every seed of them); ``val_seed`` holds out seeds of prompts that
    are otherwise in train. The two answer different questions — "does g transfer
    to an unseen prompt" vs "does g pick the right basin among the seeds of a
    prompt it has seen" — and the headline B1 contrast (matched vs SIBLING) lives
    on ``val_seed``, where sibling and self share everything except the basin.

Splits are drawn by SORTED HASH, not by an RNG seed: the same prompt ids always
land in the same split for a given ``--hash-salt``, whatever order the run dirs
are passed in and however many gens survive loading.

One diagnostic here is load-bearing rather than decorative: **separability at g's
input**. The v3 discriminants' ``FULL_scale`` was fit on the expository mode
battery, and B0's 384-token fork-class gens sit tens of thousands of fit-sigmas out
on the length-sensitive families (STFT/spectral, gate band-energy). Those few
coordinates then dominate g's opening LayerNorm and flatten everything else, so two
different trajectories arrive at g as the SAME vector — at which point
``g(own signature)`` and ``g(sibling signature)`` are identical and B1's headline
contrast is pinned at zero no matter what training does. That would be a null about
scaling masquerading as a null about signatures. The stage measures the post-
LayerNorm pairwise cosine and WARNS when it breaches ``SEPARABILITY_FLOOR``;
``--clip-z`` is the recorded fix. On the B0 pilot the unclipped floor IS breached
(min cosine 0.9998) and ``--clip-z 10`` restores it to ~0.90.

convergent_control gens are excluded by default. B0's pilot found both control
prompts tight (RESULTS §1) — their seeds do not occupy distinguishable basins, so
there is no basin for a signature to carry and their pairs are noise with a label.
``--include-control`` keeps them for anyone who wants that null in the training
set on purpose.

Usage (local)::

    python -m pleroma.build_pairs \\
        --run-dirs outputs/expB0_pilot \\
        --fork-reports outputs/expB0_pilot/fork_report.json \\
        --discriminants ~/projects/anamnesis_exps/outputs/analysis/v3_audit/factor_directions_3b.npz \\
        --out outputs/expB1/pairs
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy import (house rule; this stage is pure numpy).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB1.pairs")

Split = Literal["train", "val_seed", "val_prompt"]
SPLITS: tuple[Split, ...] = ("train", "val_seed", "val_prompt")

#: prompt_class excluded unless --include-control. B0 RESULTS §1: both control
#: prompts came out tight, i.e. no within-prompt basin structure to invert.
CONTROL_CLASS = "convergent_control"

#: How many top-entropy positions stand in when a gen has no usable divergence.
ENTROPY_FALLBACK_K = 4

#: Fork-source labels. Written per pair so a later analysis can split on them.
SOURCE_DIVERGENCE = "divergence"
SOURCE_FALLBACK_SERIES = "entropy_fallback_from_fork_series"
SOURCE_FALLBACK_REPORT = "entropy_fallback_from_fork_report"


# ── Typed records ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairRecord:
    """One training pair, exactly as it appears in ``pairs_manifest.json``.

    ``row`` indexes ``pairs_z.npz::z`` — manifest order IS matrix order, and the
    loader re-asserts that rather than trusting it.

    ``fork_positions`` are CONTINUATION-relative and every one is >= 1. The token
    to score at continuation position ``p`` is ``input_ids[prompt_length + p]``
    and the distribution that produced it is the model's output at absolute index
    ``prompt_length + p - 1`` (B0 stage 2's banked alignment, offset 1).
    """

    row: int
    pair_key: str
    generation_id: int
    run_dir: str
    run_label: str
    prompt_id: str
    prompt_class: str
    seed_idx: int
    split: Split
    fork_positions: list[int]
    n_fork_positions: int
    fork_source: str
    fallback_reason: str | None
    n_tokens: int
    prompt_length: int
    signature_path: str
    fork_series_path: str

    @property
    def fork_series_abs(self) -> Path:
        return Path(self.run_dir) / self.fork_series_path

    @property
    def signature_abs(self) -> Path:
        return Path(self.run_dir) / self.signature_path


@dataclass
class LoadedPairs:
    """``pairs_manifest.json`` + ``pairs_z.npz``, loaded and cross-checked."""

    pairs: list[PairRecord]
    z: NDArray[np.float32]
    feature_names: list[str]
    meta: dict[str, Any]

    def by_split(self, split: str) -> list[PairRecord]:
        return [p for p in self.pairs if p.split == split]

    @property
    def feature_dim(self) -> int:
        return int(self.z.shape[1])


@dataclass
class _RawGen:
    """A gen after loading, before the split is drawn."""

    generation_id: int
    run_dir: Path
    run_label: str
    prompt_id: str
    prompt_class: str
    seed_idx: int
    z: NDArray[np.float32]
    fork_positions: list[int]
    fork_source: str
    fallback_reason: str | None
    n_tokens: int
    prompt_length: int
    signature_path: str
    fork_series_path: str

    @property
    def pair_key(self) -> str:
        return f"{self.run_label}:{self.generation_id:03d}"


# ── Deterministic, order-independent split drawing ─────────────────────────────


def stable_hash(key: str, salt: str) -> str:
    """blake2b digest of ``salt|key`` — stable across processes, unlike ``hash()``.

    Python's builtin ``hash`` is randomized per interpreter (PYTHONHASHSEED), so a
    split drawn from it would silently differ between the build run and any rerun.
    """
    return hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=16).hexdigest()


def pick_by_hash(keys: list[str], n: int, salt: str) -> list[str]:
    """The ``n`` keys with the smallest hash, returned sorted by key.

    Deterministic given (keys-as-a-set, n, salt) and independent of input order.
    Asking for more than exist is a caller error, not a silent truncation.
    """
    unique = sorted(set(keys))
    if n < 0:
        raise ValueError(f"holdout count must be >= 0, got {n}")
    if n > len(unique):
        raise ValueError(
            f"cannot hold out {n} of {len(unique)} ({unique}) — that would leave "
            "nothing to train on"
        )
    if n == 0:
        return []
    ordered = sorted(unique, key=lambda k: stable_hash(k, salt))
    return sorted(ordered[:n])


def assign_splits(
    gens: list[_RawGen], holdout_prompts: int, holdout_seeds: int, salt: str
) -> tuple[dict[str, Split], list[str], dict[str, list[int]]]:
    """Map pair_key -> split, plus the held-out prompt ids and per-prompt seeds.

    Whole prompts go to ``val_prompt`` FIRST; the seed holdout is then drawn only
    from the prompts that remain, so no prompt ever contributes to both val splits
    and no val_prompt trajectory can reach train by another route.
    """
    prompt_ids = [g.prompt_id for g in gens]
    val_prompts = pick_by_hash(prompt_ids, holdout_prompts, salt)
    val_prompt_set = set(val_prompts)

    seeds_by_prompt: dict[str, list[int]] = {}
    for gen in gens:
        if gen.prompt_id in val_prompt_set:
            continue
        seeds_by_prompt.setdefault(gen.prompt_id, []).append(gen.seed_idx)

    held_seeds: dict[str, list[int]] = {}
    for prompt_id, seeds in sorted(seeds_by_prompt.items()):
        keys = [str(s) for s in seeds]
        try:
            chosen = pick_by_hash(keys, holdout_seeds, f"{salt}|seed|{prompt_id}")
        except ValueError as exc:
            # A prompt with too few seeds cannot give up any: better a train-only
            # prompt than a prompt with an empty train slice.
            logger.warning("prompt %s: %s — holding out no seed for it", prompt_id, exc)
            chosen = []
        held_seeds[prompt_id] = sorted(int(s) for s in chosen)

    assignment: dict[str, Split] = {}
    for gen in gens:
        if gen.prompt_id in val_prompt_set:
            assignment[gen.pair_key] = "val_prompt"
        elif gen.seed_idx in held_seeds.get(gen.prompt_id, []):
            assignment[gen.pair_key] = "val_seed"
        else:
            assignment[gen.pair_key] = "train"
    return assignment, val_prompts, held_seeds


# ── Loading one B0 run dir ─────────────────────────────────────────────────────


def load_discriminants(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], str]:
    """(FULL_mean, FULL_scale, FULL_names, sha256) from the discriminants npz."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--discriminants {path} not found")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=True) as npz:
        missing = [k for k in ("FULL_mean", "FULL_scale", "FULL_names") if k not in npz]
        if missing:
            raise KeyError(f"{path}: missing {missing} (has {list(npz.files)})")
        mean = np.asarray(npz["FULL_mean"], dtype=np.float64)
        scale = np.asarray(npz["FULL_scale"], dtype=np.float64)
        names = [str(x) for x in npz["FULL_names"]]
    if not (mean.shape == scale.shape == (len(names),)):
        raise ValueError(
            f"{path}: FULL_mean {mean.shape} / FULL_scale {scale.shape} / "
            f"FULL_names {len(names)} disagree"
        )
    if mean.size == 0:
        raise ValueError(f"{path}: FULL_names is empty")
    return mean, scale, names, sha


def zscore(
    features: NDArray[np.floating],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float32]:
    """``(x - mean) / scale`` with a hard zero-scale guard.

    A zero scale means the fit saw no variance on that feature; dividing gives inf
    (or nan, when the numerator is zero too), and one non-finite entry makes the
    whole training batch non-finite. Those dimensions carry no information by
    construction, so they are set to exactly 0 — which is also their z-value under
    any non-degenerate scale.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.shape != mean.shape:
        raise ValueError(f"signature has {x.shape} features, discriminants have {mean.shape}")
    degenerate = scale == 0.0
    safe = np.where(degenerate, 1.0, scale)
    z = (x - mean) / safe
    z[degenerate] = 0.0
    return z.astype(np.float32)


def load_fork_report(path: Path) -> dict[int, dict[str, Any]]:
    """gen entries of a stage-3 ``fork_report.json``, keyed by generation_id."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--fork-reports entry {path} not found")
    blob = json.loads(path.read_text())
    gens = blob.get("gens")
    if not isinstance(gens, list) or not gens:
        raise ValueError(f"{path}: no 'gens' list (keys: {sorted(blob)})")
    out: dict[int, dict[str, Any]] = {}
    for row in gens:
        out[int(row["generation_id"])] = row
    return out


def load_cells(run_dir: Path) -> dict[int, dict[str, Any]]:
    """B0 stage 2's ``cells.json``, keyed by generation_id."""
    cells_path = run_dir / "cells.json"
    if not cells_path.exists():
        raise FileNotFoundError(
            f"{cells_path} not found — B0 stage 2 writes it next to signatures/"
        )
    blob = json.loads(cells_path.read_text())
    rows: list[dict[str, Any]] = blob["cells"] if isinstance(blob, dict) else blob
    if not rows:
        raise ValueError(f"{cells_path} holds no cells")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        gid = int(row["generation_id"])
        for key in ("prompt_id", "prompt_class", "seed_idx", "prompt_length"):
            if row.get(key) is None:
                raise KeyError(f"{cells_path}: gen {gid} is missing {key!r}")
        out[gid] = row
    return out


def entropy_fallback_positions(
    fork_series_path: Path, k: int
) -> tuple[list[int], str] | None:
    """Top-``k`` continuation positions by banked entropy, or None if unreadable.

    Alignment: entropy index ``t`` is the distribution that produced continuation
    index ``t + 1``, so every position this returns is >= 1 by construction — the
    p == 0 problem cannot recur through this path. Ties break toward the earlier
    position so the selection is deterministic.
    """
    if not fork_series_path.exists():
        return None
    try:
        with np.load(fork_series_path) as npz:
            if "logits_entropy" not in npz:
                return None
            entropy = np.asarray(npz["logits_entropy"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        logger.warning("%s unreadable for the entropy fallback: %s", fork_series_path, exc)
        return None
    if entropy.size == 0:
        return None
    order = np.lexsort((np.arange(entropy.size), -entropy))
    steps = [int(t) for t in order[: min(k, entropy.size)]]
    return sorted(t + 1 for t in steps), SOURCE_FALLBACK_SERIES


def report_fallback_positions(fork_entry: dict[str, Any], k: int) -> tuple[list[int], str] | None:
    """Entropy fallback from the fork report alone (no fork_series npz present).

    The report banks only the top-M entropy positions, so this is the top-k of a
    top-M and is labelled differently from the fork_series path for that reason.
    """
    scored: list[tuple[float, int]] = []
    for event in fork_entry.get("forks", []):
        pos = int(event["position_continuation"])
        ent = event.get("entropy")
        if pos >= 1 and ent is not None:
            scored.append((float(ent), pos))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    return sorted(p for _, p in scored[:k]), SOURCE_FALLBACK_REPORT


def load_run(
    run_dir: Path,
    fork_report_path: Path,
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    full_names: list[str],
    include_control: bool,
    fallback_k: int,
) -> tuple[list[_RawGen], list[str]]:
    """Load every usable gen of one B0 run dir. Returns (gens, failures).

    A gen that cannot be loaded is LOGGED and recorded in ``failures`` — never
    dropped silently. The caller refuses to write a result with no gens at all.
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dirs entry {run_dir} is not a directory")
    run_label = run_dir.name

    cells = load_cells(run_dir)
    fork_gens = load_fork_report(fork_report_path)
    logger.info(
        "%s: %d cells, %d fork-report gens", run_label, len(cells), len(fork_gens)
    )

    failures: list[str] = []
    gens: list[_RawGen] = []
    n_control_skipped = 0

    for gid in sorted(cells):
        cell = cells[gid]
        tag = f"{run_label}:gen_{gid:03d}"
        prompt_class = str(cell["prompt_class"])
        if prompt_class == CONTROL_CLASS and not include_control:
            n_control_skipped += 1
            continue
        try:
            fork_entry = fork_gens.get(gid)
            if fork_entry is None:
                raise KeyError(f"no fork_report entry for generation_id {gid}")
            if str(fork_entry.get("prompt_id", cell["prompt_id"])) != str(cell["prompt_id"]):
                raise ValueError(
                    f"fork_report prompt_id {fork_entry.get('prompt_id')!r} != cells.json "
                    f"{cell['prompt_id']!r} — the report does not belong to this run dir"
                )

            sig_rel = str(cell.get("signature_path") or f"signatures/gen_{gid:03d}.npz")
            sig_path = run_dir / sig_rel
            if not sig_path.exists():
                raise FileNotFoundError(f"{sig_path} not found")
            with np.load(sig_path, allow_pickle=True) as npz:
                if "features" not in npz or "feature_names" not in npz:
                    raise KeyError(
                        f"{sig_path}: needs features + feature_names (has {list(npz.files)})"
                    )
                features = np.asarray(npz["features"], dtype=np.float64)
                names = [str(x) for x in npz["feature_names"]]
            if names != full_names:
                first_bad = next(
                    (i for i, (a, b) in enumerate(zip(names, full_names)) if a != b), None
                )
                detail = (
                    f" first mismatch at {first_bad}: {names[first_bad]!r} vs "
                    f"{full_names[first_bad]!r}"
                    if first_bad is not None
                    else " (one list is a prefix of the other)"
                )
                raise ValueError(
                    "feature_names != discriminants FULL_names "
                    f"({len(names)} vs {len(full_names)} dims).{detail}"
                )

            z = zscore(features, mean, scale)
            if not np.all(np.isfinite(z)):
                n_bad = int((~np.isfinite(z)).sum())
                raise ValueError(
                    f"{n_bad} non-finite z-values after scaling — refusing to train on it"
                )

            positions = sorted(
                {int(p) for p in fork_entry.get("divergence_positions", []) if int(p) >= 1}
            )
            fork_source = SOURCE_DIVERGENCE
            fallback_reason: str | None = None
            if not positions:
                n_raw = len(fork_entry.get("divergence_positions", []))
                fallback_reason = (
                    "no_divergence_position_ge_1"
                    if n_raw
                    else "no_divergence_positions_at_all"
                )
                series_rel = str(
                    cell.get("fork_series_path") or f"fork_series/gen_{gid:03d}.npz"
                )
                picked = entropy_fallback_positions(run_dir / series_rel, fallback_k)
                if picked is None:
                    picked = report_fallback_positions(fork_entry, fallback_k)
                if picked is None:
                    raise ValueError(
                        "no divergence position >= 1 and no entropy fallback available "
                        "(fork_series npz unreadable and the report banks no entropy)"
                    )
                positions, fork_source = picked
                logger.info(
                    "%s: %s -> %s %s", tag, fallback_reason, fork_source, positions
                )

            n_tokens = int(
                fork_entry.get("n_continuation_tokens")
                or cell.get("num_generated_tokens")
                or 0
            )
            prompt_length = int(cell["prompt_length"])
            if n_tokens <= 0:
                raise ValueError(f"n_continuation_tokens resolved to {n_tokens}")
            out_of_range = [p for p in positions if p >= n_tokens]
            if out_of_range:
                raise ValueError(
                    f"fork positions {out_of_range} are past the continuation "
                    f"({n_tokens} tokens) — the report and the run dir disagree"
                )

            gens.append(
                _RawGen(
                    generation_id=gid,
                    run_dir=run_dir,
                    run_label=run_label,
                    prompt_id=str(cell["prompt_id"]),
                    prompt_class=prompt_class,
                    seed_idx=int(cell["seed_idx"]),
                    z=z,
                    fork_positions=positions,
                    fork_source=fork_source,
                    fallback_reason=fallback_reason,
                    n_tokens=n_tokens,
                    prompt_length=prompt_length,
                    signature_path=sig_rel,
                    fork_series_path=str(
                        cell.get("fork_series_path") or f"fork_series/gen_{gid:03d}.npz"
                    ),
                )
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.error("%s skipped: %s: %s", tag, type(exc).__name__, exc)
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")

    if n_control_skipped:
        logger.info(
            "%s: excluded %d %s gens (--include-control keeps them)",
            run_label, n_control_skipped, CONTROL_CLASS,
        )
    return gens, failures


# ── Summary + output ───────────────────────────────────────────────────────────


def z_scale_diagnostic(
    z: NDArray[np.float32], feature_names: list[str], top: int = 10
) -> dict[str, Any]:
    """How far outside the discriminant fit's distribution these signatures sit.

    This is not a formality. ``FULL_mean``/``FULL_scale`` were fit on the
    expository mode battery; B0's gens are 384-token continuations of *fork-class*
    prompts, and the length-sensitive families (STFT/spectral, gate band-energy)
    land tens of thousands of fit-sigmas out. g's first layer is a LayerNorm, which
    makes that survivable — but a coordinate at |z| ~ 1e4 dominates the LayerNorm
    statistics of every sample and squashes the other ~2700 dimensions toward zero,
    so the scale is a modelling fact, not a cosmetic one. It is reported here so
    the decision (leave it, clip it, refit the scaler) is taken with the number in
    view rather than after a flat training curve.
    """
    absz = np.abs(z.astype(np.float64))
    if absz.size == 0:
        return {"n": 0}
    per_dim = absz.max(axis=0)
    worst = np.argsort(-per_dim)[: min(top, per_dim.size)]
    return {
        "median_abs_z": float(np.median(absz)),
        "max_abs_z": float(absz.max()),
        "frac_abs_z_gt_10": float((absz > 10).mean()),
        "frac_abs_z_gt_100": float((absz > 100).mean()),
        "n_dims_absmax_gt_100": int((per_dim > 100).sum()),
        "n_dims": int(per_dim.size),
        "worst_features": [
            {"name": feature_names[i], "absmax_z": float(per_dim[i])} for i in worst
        ],
    }


def layernorm_separability(
    z: NDArray[np.float32], max_rows: int = 96, eps: float = 1e-5
) -> dict[str, Any]:
    """Pairwise cosine between signatures AS g's first layer will see them.

    g starts with a LayerNorm, which at initialisation is exactly per-row
    standardisation (unit gain, zero bias). This computes that and reports the
    off-diagonal cosine between rows — i.e. how distinguishable two trajectories
    are at g's input, which upper-bounds how distinguishable ``g(z_i)`` and
    ``g(z_j)`` can be for ANY weights.

    Why this number and not the raw-z one: B0's RESULTS §1 already warned that raw
    v3 signatures are nearly parallel globally and that "discrimination lives in a
    thin Δ-layer". If the standardised rows are still cosine ~1.0, then
    ``matched`` and ``sibling`` are the same input to within float noise and B1's
    headline contrast is pinned at zero BY CONSTRUCTION — a null that says nothing
    about signatures and everything about scaling. That is worth catching here,
    in a two-second CPU stage, rather than after a GPU training run.

    Rows are subsampled (deterministically, evenly) above ``max_rows`` to keep the
    O(n^2) comparison cheap; the sample size is reported.
    """
    x = np.asarray(z, dtype=np.float64)
    if x.shape[0] < 2:
        return {"n_sampled": int(x.shape[0])}
    if x.shape[0] > max_rows:
        x = x[np.linspace(0, x.shape[0] - 1, max_rows).astype(int)]
    x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(
        x.var(axis=1, keepdims=True) + eps
    )
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = x / norms
    cos = unit @ unit.T
    off = cos[~np.eye(cos.shape[0], dtype=bool)]
    return {
        "n_sampled": int(cos.shape[0]),
        "mean_cosine": float(off.mean()),
        "min_cosine": float(off.min()),
        "max_cosine": float(off.max()),
    }


#: Above this, two signatures are the same vector to g and no weights can separate
#: them. Chosen as "1 - 1e-3": at cosine 0.999 the between-signature signal is
#: ~0.1% of the norm, which bf16 training will not resolve.
SEPARABILITY_FLOOR = 0.999


def summarize(pairs: list[PairRecord]) -> dict[str, Any]:
    """Counts by split, by class, and the fork-source breakdown."""
    by_split: dict[str, int] = {s: 0 for s in SPLITS}
    by_class: dict[str, dict[str, int]] = {}
    by_source: dict[str, int] = {}
    fork_counts: list[int] = []
    for pair in pairs:
        by_split[pair.split] = by_split.get(pair.split, 0) + 1
        cls = by_class.setdefault(pair.prompt_class, {s: 0 for s in SPLITS})
        cls[pair.split] = cls.get(pair.split, 0) + 1
        by_source[pair.fork_source] = by_source.get(pair.fork_source, 0) + 1
        fork_counts.append(pair.n_fork_positions)
    return {
        "n_pairs": len(pairs),
        "by_split": by_split,
        "by_class": {k: by_class[k] for k in sorted(by_class)},
        "by_fork_source": {k: by_source[k] for k in sorted(by_source)},
        "n_prompts": len({p.prompt_id for p in pairs}),
        "fork_positions_per_pair_mean": (
            float(np.mean(fork_counts)) if fork_counts else None
        ),
        "fork_positions_total": int(sum(fork_counts)),
    }


def format_summary(
    summary: dict[str, Any],
    holdout: dict[str, Any],
    z_scale: dict[str, Any] | None = None,
    separability: dict[str, Any] | None = None,
) -> str:
    """The compact human-readable table printed at the end of a run."""
    lines: list[str] = []
    lines.append(f"pairs: {summary['n_pairs']} over {summary['n_prompts']} prompts")
    head = f"{'prompt_class':<24}" + "".join(f"{s:>12}" for s in SPLITS) + f"{'total':>8}"
    lines.append(head)
    lines.append("-" * len(head))
    for cls in sorted(summary["by_class"]):
        counts = summary["by_class"][cls]
        total = sum(counts.get(s, 0) for s in SPLITS)
        lines.append(
            f"{cls:<24}" + "".join(f"{counts.get(s, 0):>12}" for s in SPLITS) + f"{total:>8}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'ALL':<24}"
        + "".join(f"{summary['by_split'].get(s, 0):>12}" for s in SPLITS)
        + f"{summary['n_pairs']:>8}"
    )
    lines.append(f"fork source: {summary['by_fork_source']}")
    mean_forks = summary["fork_positions_per_pair_mean"]
    if mean_forks is None:
        lines.append("fork positions: none")
    else:
        lines.append(
            f"fork positions: {summary['fork_positions_total']} total, "
            f"{mean_forks:.2f}/pair"
        )
    lines.append(f"val_prompt prompt_ids: {holdout['val_prompt_ids']}")
    lines.append(f"val_seed seed_idx by prompt: {holdout['val_seed_idx_by_prompt']}")
    if z_scale and z_scale.get("n_dims"):
        lines.append(
            f"z scale: median |z| {z_scale['median_abs_z']:.2f}, max "
            f"{z_scale['max_abs_z']:.1f}, {100 * z_scale['frac_abs_z_gt_10']:.1f}% of "
            f"cells |z|>10, {z_scale['n_dims_absmax_gt_100']}/{z_scale['n_dims']} dims "
            "absmax>100"
        )
        worst = z_scale.get("worst_features") or []
        if worst:
            lines.append(
                "  worst: " + ", ".join(f"{w['name']} ({w['absmax_z']:.0f})" for w in worst[:3])
            )
    if separability and separability.get("min_cosine") is not None:
        lines.append(
            f"separability at g's input (n={separability['n_sampled']}): mean cosine "
            f"{separability['mean_cosine']:.5f}, min {separability['min_cosine']:.5f}"
        )
        if separability.get("breached"):
            lines.append(
                "  *** SEPARABILITY FLOOR BREACHED — g cannot tell these signatures "
                "apart; matched-minus-sibling would be ~0 by construction. "
                "Consider --clip-z 10. ***"
            )
    return "\n".join(lines)


def build(
    run_dirs: list[Path],
    fork_reports: list[Path],
    discriminants: Path,
    holdout_prompts: int,
    holdout_seeds: int,
    include_control: bool,
    fallback_k: int,
    salt: str,
    clip_z: float = 0.0,
    standardize: str = "full",
) -> tuple[list[PairRecord], NDArray[np.float32], list[str], dict[str, Any]]:
    """The whole stage: load -> z-score -> split -> manifest. Raises on empty."""
    if len(run_dirs) != len(fork_reports):
        raise ValueError(
            f"--run-dirs has {len(run_dirs)} entries but --fork-reports has "
            f"{len(fork_reports)}; they are positional pairs"
        )
    mean, scale, full_names, disc_sha = load_discriminants(discriminants)
    n_degenerate = int((scale == 0.0).sum())
    logger.info(
        "discriminants: %d dims, %d zero-scale (guarded to z=0), sha %s",
        len(full_names), n_degenerate, disc_sha[:12],
    )

    gens: list[_RawGen] = []
    failures: list[str] = []
    for run_dir, report in zip(run_dirs, fork_reports):
        run_gens, run_failures = load_run(
            run_dir, report, mean, scale, full_names, include_control, fallback_k
        )
        gens.extend(run_gens)
        failures.extend(run_failures)

    if not gens:
        raise ValueError(
            "no gen survived loading — refusing to write an empty pair set "
            f"({len(failures)} failures; first: {failures[0] if failures else 'none'})"
        )

    seen: dict[str, str] = {}
    for gen in gens:
        if gen.pair_key in seen:
            raise ValueError(
                f"duplicate pair_key {gen.pair_key} — two run dirs share a name; "
                "rename one, since the manifest keys on it"
            )
        seen[gen.pair_key] = str(gen.run_dir)

    assignment, val_prompts, held_seeds = assign_splits(
        gens, holdout_prompts, holdout_seeds, salt
    )

    gens.sort(key=lambda g: (g.run_label, g.generation_id))
    pairs: list[PairRecord] = []
    for row, gen in enumerate(gens):
        pairs.append(
            PairRecord(
                row=row,
                pair_key=gen.pair_key,
                generation_id=gen.generation_id,
                run_dir=str(gen.run_dir),
                run_label=gen.run_label,
                prompt_id=gen.prompt_id,
                prompt_class=gen.prompt_class,
                seed_idx=gen.seed_idx,
                split=assignment[gen.pair_key],
                fork_positions=list(gen.fork_positions),
                n_fork_positions=len(gen.fork_positions),
                fork_source=gen.fork_source,
                fallback_reason=gen.fallback_reason,
                n_tokens=gen.n_tokens,
                prompt_length=gen.prompt_length,
                signature_path=gen.signature_path,
                fork_series_path=gen.fork_series_path,
            )
        )
    z = np.stack([gen.z for gen in gens]).astype(np.float32)
    corpus_stats: dict[str, Any] | None = None
    if standardize == "corpus":
        # Re-standardize per dimension over the TRAIN rows only (no leakage).
        # Composing this affine map with the FULL z-scoring above is exactly
        # corpus-own standardization of the raw features; it whitens the thin
        # subspace where THIS corpus's trajectories actually differ, which the
        # foreign FULL_scale leaves buried under common mode (the B1 v1 null).
        train_rows = np.array([assignment[g.pair_key] == "train" for g in gens])
        if int(train_rows.sum()) < 2:
            raise ValueError("--standardize corpus needs >= 2 train rows")
        mu = z[train_rows].mean(axis=0)
        sd = z[train_rows].std(axis=0)
        dead = sd < 1e-8
        sd_safe = np.where(dead, 1.0, sd)
        z = ((z - mu) / sd_safe).astype(np.float32)
        z[:, dead] = 0.0
        corpus_stats = {
            "n_train_rows": int(train_rows.sum()),
            "n_dead_dims": int(dead.sum()),
        }
        logger.info(
            "--standardize corpus: %d train rows, %d dead dims zeroed",
            corpus_stats["n_train_rows"], corpus_stats["n_dead_dims"],
        )
    diagnostic = z_scale_diagnostic(z, full_names)
    n_clipped = 0
    if clip_z > 0:
        # Recorded, never silent: clipping changes g's input and therefore every
        # number downstream, so it is a flag with a count, not a default.
        n_clipped = int((np.abs(z) > clip_z).sum())
        z = np.clip(z, -clip_z, clip_z)
        logger.info("--clip-z %.1f clipped %d of %d cells", clip_z, n_clipped, z.size)
        diagnostic["after_clip"] = z_scale_diagnostic(z, full_names)
    logger.info(
        "z scale: median |z| %.2f, max %.1f, %.1f%% of cells |z|>10, %d/%d dims absmax>100",
        diagnostic["median_abs_z"], diagnostic["max_abs_z"],
        100 * diagnostic["frac_abs_z_gt_10"],
        diagnostic["n_dims_absmax_gt_100"], diagnostic["n_dims"],
    )
    separability = layernorm_separability(z)
    min_cos = separability.get("min_cosine")
    if min_cos is not None:
        logger.info(
            "separability at g's input (LayerNormed pairwise cosine, n=%d): "
            "mean %.5f, min %.5f",
            separability["n_sampled"], separability["mean_cosine"], min_cos,
        )
        if min_cos > SEPARABILITY_FLOOR:
            logger.warning(
                "SEPARABILITY FLOOR BREACHED: the closest pair of signatures has "
                "LayerNormed cosine %.5f > %.3f. g's first layer will see these "
                "trajectories as the SAME vector, so g(own) and g(sibling) cannot "
                "differ and eval_g's headline contrast is pinned at ~0 by "
                "construction — that null would be about scaling, not about "
                "signatures. A handful of length-sensitive dims (%s) carry almost "
                "all the norm; --clip-z 10 is the cheap fix and is recorded in the "
                "manifest when used.",
                min_cos, SEPARABILITY_FLOOR,
                ", ".join(w["name"] for w in diagnostic["worst_features"][:3]),
            )

    train_prompts = {p.prompt_id for p in pairs if p.split == "train"}
    leaked = train_prompts & set(val_prompts)
    if leaked:  # cannot happen by construction; assert it anyway
        raise AssertionError(f"held-out prompts {sorted(leaked)} reached train")
    if not any(p.split == "train" for p in pairs):
        raise ValueError(
            "the holdout consumed every pair — lower --holdout-prompts/--holdout-seeds"
        )

    meta: dict[str, Any] = {
        "stage": "expB1_build_pairs",
        "params": {
            "run_dirs": [str(Path(d).expanduser().resolve()) for d in run_dirs],
            "fork_reports": [str(Path(p).expanduser().resolve()) for p in fork_reports],
            "discriminants": str(Path(discriminants).expanduser().resolve()),
            "discriminants_sha256": disc_sha,
            "holdout_prompts": holdout_prompts,
            "holdout_seeds": holdout_seeds,
            "include_control": include_control,
            "entropy_fallback_k": fallback_k,
            "hash_salt": salt,
            "control_class": CONTROL_CLASS,
            "clip_z": clip_z,
            "standardize": standardize,
            "corpus_stats": corpus_stats,
        },
        "feature_dim": len(full_names),
        "n_zero_scale_features": n_degenerate,
        "n_clipped_cells": n_clipped,
        "z_scale": diagnostic,
        "separability": {
            **separability,
            "floor": SEPARABILITY_FLOOR,
            "breached": bool(min_cos is not None and min_cos > SEPARABILITY_FLOOR),
            "definition": (
                "off-diagonal cosine between signatures after per-row standardisation "
                "(= g's LayerNorm at init). Above the floor, g(own) and g(sibling) "
                "cannot differ and the matched-minus-sibling contrast is pinned at 0 "
                "by construction."
            ),
        },
        "alignment": (
            "fork_positions are continuation-relative and all >= 1; the token at "
            "continuation position p is input_ids[prompt_length + p] and is predicted "
            "by the model output at absolute index prompt_length + p - 1"
        ),
        "holdout": {
            "val_prompt_ids": val_prompts,
            "val_seed_idx_by_prompt": held_seeds,
        },
        "summary": summarize(pairs),
        "failures": failures,
    }
    return pairs, z, full_names, meta


def write_pairs(
    out_dir: Path,
    pairs: list[PairRecord],
    z: NDArray[np.float32],
    feature_names: list[str],
    meta: dict[str, Any],
) -> None:
    """Write ``pairs_manifest.json`` + ``pairs_z.npz`` + ``run_meta.json``."""
    if not pairs:
        raise ValueError("refusing to write an empty pair set")
    if z.shape[0] != len(pairs):
        raise ValueError(f"z has {z.shape[0]} rows for {len(pairs)} pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs_manifest.json").write_text(
        json.dumps({**meta, "pairs": [asdict(p) for p in pairs]}, indent=2)
    )
    np.savez_compressed(
        out_dir / "pairs_z.npz",
        z=z,
        feature_names=np.asarray(feature_names, dtype="<U"),
        rows=np.asarray([p.pair_key for p in pairs], dtype="<U"),
    )
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def load_pairs(pairs_dir: Path) -> LoadedPairs:
    """Read back a ``build_pairs`` output dir, re-checking manifest/matrix alignment.

    Used by ``train_g`` and ``eval_g``; lives here so there is exactly one reader
    of the format and torch never has to be importable to use it.
    """
    pairs_dir = Path(pairs_dir).expanduser()
    manifest_path = pairs_dir / "pairs_manifest.json"
    z_path = pairs_dir / "pairs_z.npz"
    for path in (manifest_path, z_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run build_pairs first")
    blob = json.loads(manifest_path.read_text())
    rows = blob.get("pairs")
    if not rows:
        raise ValueError(f"{manifest_path}: no pairs")
    known = set(PairRecord.__dataclass_fields__)
    pairs = [PairRecord(**{k: v for k, v in row.items() if k in known}) for row in rows]

    with np.load(z_path, allow_pickle=True) as npz:
        if "z" not in npz:
            raise KeyError(f"{z_path}: no 'z' array (has {list(npz.files)})")
        z = np.asarray(npz["z"], dtype=np.float32)
        feature_names = [str(x) for x in npz["feature_names"]] if "feature_names" in npz else []
        keys = [str(x) for x in npz["rows"]] if "rows" in npz else None

    if z.shape[0] != len(pairs):
        raise ValueError(f"{z_path} has {z.shape[0]} rows for {len(pairs)} manifest pairs")
    if keys is not None and keys != [p.pair_key for p in pairs]:
        raise ValueError(
            f"{z_path} row keys do not match manifest order — the pair dir is inconsistent"
        )
    for i, pair in enumerate(pairs):
        if pair.row != i:
            raise ValueError(
                f"manifest row {i} carries row={pair.row}; manifest order IS matrix order"
            )
    meta = {k: v for k, v in blob.items() if k != "pairs"}
    return LoadedPairs(pairs=pairs, z=z, feature_names=feature_names, meta=meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dirs", type=Path, nargs="+", required=True, help="B0 stage-2 output dirs"
    )
    parser.add_argument(
        "--fork-reports",
        type=Path,
        nargs="+",
        required=True,
        help="fork_report.json paths, positionally matched to --run-dirs",
    )
    parser.add_argument(
        "--discriminants",
        type=Path,
        required=True,
        help="factor_directions_3b.npz — FULL_mean/FULL_scale/FULL_names",
    )
    parser.add_argument("--out", type=Path, required=True, help="output DIRECTORY")
    parser.add_argument(
        "--holdout-prompts",
        type=int,
        default=2,
        help="whole prompt_ids held out into val_prompt (deterministic sorted hash)",
    )
    parser.add_argument(
        "--holdout-seeds",
        type=int,
        default=2,
        help="seed_idx values held out per remaining prompt into val_seed",
    )
    parser.add_argument(
        "--include-control",
        action="store_true",
        help=f"keep {CONTROL_CLASS} gens (excluded by default; B0 found them tight)",
    )
    parser.add_argument(
        "--entropy-fallback-k",
        type=int,
        default=ENTROPY_FALLBACK_K,
        help="positions used when a gen has no divergence position >= 1",
    )
    parser.add_argument(
        "--hash-salt",
        default="",
        help="changes the split draw reproducibly (same salt = same split)",
    )
    parser.add_argument(
        "--clip-z",
        type=float,
        default=0.0,
        help=(
            "clip z to +/- this many fit-sigmas (0 = off, the default). The B0 pilot "
            "lands ~37%% of cells beyond |z|=10 on length-sensitive families; clipping "
            "is a recorded decision, never automatic"
        ),
    )
    parser.add_argument(
        "--standardize",
        choices=("full", "corpus"),
        default="full",
        help=(
            "z-scaling basis: 'full' = the discriminants' FULL_mean/FULL_scale "
            "(default, discriminant-comparable); 'corpus' = re-standardize per dim "
            "over this build's TRAIN rows (whitens the corpus's own variance — the "
            "fix for the B1 v1 near-collinear-input null; not comparable to the "
            "discriminant space)"
        ),
    )
    args = parser.parse_args()

    if args.entropy_fallback_k <= 0:
        logger.error("--entropy-fallback-k must be > 0, got %s", args.entropy_fallback_k)
        return 2
    if args.holdout_prompts < 0 or args.holdout_seeds < 0:
        logger.error("--holdout-prompts/--holdout-seeds must be >= 0")
        return 2
    if args.clip_z < 0:
        logger.error("--clip-z must be >= 0 (0 = off), got %s", args.clip_z)
        return 2

    try:
        pairs, z, feature_names, meta = build(
            run_dirs=list(args.run_dirs),
            fork_reports=list(args.fork_reports),
            discriminants=args.discriminants,
            holdout_prompts=args.holdout_prompts,
            holdout_seeds=args.holdout_seeds,
            include_control=bool(args.include_control),
            fallback_k=args.entropy_fallback_k,
            salt=args.hash_salt,
            clip_z=float(args.clip_z),
            standardize=str(args.standardize),
        )
        write_pairs(args.out, pairs, z, feature_names, meta)
    except (OSError, KeyError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        logger.error("build_pairs failed: %s: %s", type(exc).__name__, exc)
        return 1

    print(
        format_summary(
            meta["summary"], meta["holdout"], meta.get("z_scale"), meta.get("separability")
        )
    )
    logger.info(
        "DONE — %d pairs x %d dims (%d failures) -> %s",
        len(pairs), z.shape[1], len(meta["failures"]), args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
