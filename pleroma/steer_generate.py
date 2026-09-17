"""Exp B1 stage 4a: FREE generation under g(signature), the basin-level readout's source.

The overnight eval (``eval_g``) scored teacher-forced NLL at ~3 fork tokens per
trajectory — a high-variance readout of a static whole-continuation bias. This
track asks the basin question directly instead: when the model generates FREELY
under ``g(z_i)``, does the trajectory it produces land nearer trajectory *i*'s
basin than an unsteered generation from the same prompt does?

Three arms per source trajectory, every one a real sampled continuation:

    base       no hooks at all — the frozen model's own draw
    matched    g(z_i)  — the source trajectory's OWN signature
    shuffled   g(z_k)  — a donor signature from a DIFFERENT prompt (the floor)
    sibling    g(z_j)  — same prompt, different seed (optional; not on by default)

``--reps`` independent draws per arm, and **the RNG seed is keyed on (source, rep)
and NOT on the arm**: base rep 3 and matched rep 3 consume the same random draws,
so the only thing that differs between them is the injected bias. That is what
makes a pair readable side by side in the felt-shift bundle, and it is what makes
the zero-bias self-check below a byte-identity test rather than a distributional one.

── the injection site, reconciled (READ THIS BEFORE CHANGING --inject-layers) ──

Two mechanisms, two conventions, off by one:

  * ``g_net.TrainableResidualInjector`` (g_net.py:178) registers a
    ``register_forward_hook`` on decoder layer ``i`` — it perturbs that layer's
    OUTPUT. For an HF Llama-class layer the output IS the post-MLP residual
    stream, so this is the stream ENTERING layer ``i+1``. ``--inject-layers
    7,14,18,21`` therefore trained biases that live at the inputs of layers
    8, 15, 19, 22 (g_net.py:32-39 says so explicitly, and the B1 handoff repeats it).

  * ``model_loader.attach_residual_write`` (model_loader.py:562) registers a
    ``register_forward_pre_hook`` on decoder layer ``spec.layer_idx`` and adds its
    delta to ``args[0]`` — the layer's INPUT hidden states (model_loader.py:436-450).

So the two agree only after a +1 shift:

    attach_layers = [i + 1 for i in train_meta.config.inject_layers]

i.e. biases trained at ``7,14,18,21`` are attached here at ``8,15,19,22``. That
shift is computed in ``attach_layers_for()``, asserted against the model's layer
count, recorded in every gen record AND in run_meta.json. A silent one-layer shift
would invalidate every number downstream, so it is never a literal anywhere.

Two further reconciliations with ``attach_residual_write``'s defaults:

  * ``normalize=False`` and ``alpha=1.0``. The reference normalises the vector and
    scales it by a scalar dose — right for a dose ladder, wrong here: the bias
    MAGNITUDE is part of what g learned (v3's whole tell was a bias L2 of ~0.05).
    We inject g's vector exactly as emitted.
  * ``start_pos = prompt_length``, ``end_pos = None``. Positional gating reads
    ``cache_position`` (absolute positions), which is correct under prefill and
    under every incremental decode step alike (model_loader.py:469-487) — this is
    precisely why the TRAINING injector cannot be reused here: it gates on a
    ``[B, S]`` mask built from a full-sequence teacher-forced forward and RAISES on
    a ``seq_len == 1`` step (g_net.py:228-234).

Position semantics match training exactly. Training injects at absolute positions
``>= prompt_length``; the distribution that produces continuation position 0 comes
from position ``prompt_length - 1``, which is NOT injected. Under ``generate()``
the prefill (positions ``0 .. prompt_length-1``) draws the first new token
un-steered, and every decode step from ``prompt_length`` on is injected. Same
boundary, same meaning.

── the not-a-perturbation gate ────────────────────────────────────────────────

Before a single real generation, ``self_check()`` draws the same continuation three
ways with one seed: no hooks; hooks attached with ``alpha=0`` (the reference's
early-return path); hooks attached with ``alpha=1`` and a ZERO vector (the full
clone-and-add arithmetic, delta == 0). All three token id lists must be identical,
and the third must report ``saw_cache_position`` with a non-zero injected-position
count — proving the gating fired under incremental decoding rather than silently
doing nothing. Any failure aborts the run (exit 3). A steering result is only
interesting if the apparatus is a no-op when the bias is.

── workflow ───────────────────────────────────────────────────────────────────

    1. steer_generate  (node GPU)  -> <steer-dir>/gen_records/gen_NNN.json
    2. run_replay_b0   (node GPU)  --gen-dir <steer-dir> --out-dir <steer-dir>
                                     --save-raw none   -> signatures/ + cells.json
    3. basin_readout + feltshift_bundle  (local CPU)

Step 2 is the EXISTING B0 stage 2, unmodified. It needs exactly ``generation_id``,
``input_ids``, ``prompt_length``, ``prompt_id``, ``prompt_class`` and ``seed_idx``
from a record (run_replay_b0.py:257-276) — no fork_report, no fork positions —
so the records written here carry all six plus the B0 optional fields, and the
steering metadata rides in a nested ``steering`` block that run_replay_b0 ignores.
Pointing ``--out-dir`` at the steer dir itself puts ``signatures/`` and
``cells.json`` beside ``gen_records/``, which is where ``basin_readout`` looks.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.steer_generate \\
        --pairs-dir outputs/b1_pairs_v2 \\
        --run-dirs outputs/pilot_replay outputs/wave2_replay \\
        --g-checkpoint outputs/b1_v3/g_state.pt \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --out-dir outputs/b1_steer_v3
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import: generation is GPU-bound, so one
# CPU thread costs nothing here but keeps the pools from exploding against whatever
# else shares the node (mirrors gen_forks.py:48-54).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from pleroma.build_pairs import PairRecord, load_pairs
from pleroma.eval_g import deterministic_choice
from pleroma.fork_tokens import load_cells
from pleroma.gen_forks import SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB1.steer")

#: Every arm this stage knows how to run. ``base`` never attaches a hook.
KNOWN_ARMS: tuple[str, ...] = ("base", "matched", "sibling", "shuffled")

#: The default three: the intervention, the floor, and the untouched model.
DEFAULT_ARMS: str = "base,matched,shuffled"

#: Arms whose bias comes from a donor pair rather than the source itself.
DONOR_ARMS: tuple[str, ...] = ("sibling", "shuffled")


# ── record schema ──────────────────────────────────────────────────────────────


class SteeringMeta(BaseModel):
    """What makes a steered record more than a B0 gen record.

    Everything the readout joins on. ``inject_layers_trained`` and ``attach_layers``
    are both banked per record precisely because their relationship is the one
    silent-failure mode that would invalidate the whole run (see the module
    docstring): a record that does not say where the bias went cannot be audited.
    """

    arm: str = Field(min_length=1)
    rep: int = Field(ge=0)
    rng_seed: int = Field(ge=0)
    source_pair_key: str = Field(min_length=1)
    source_generation_id: int = Field(ge=0)
    source_run_label: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    source_prompt_id: str = Field(min_length=1)
    source_seed_idx: int = Field(ge=0)
    donor_pair_key: str | None = None
    donor_prompt_id: str | None = None
    donor_seed_idx: int | None = None
    donor_pool: str | None = None
    inject_layers_trained: list[int]
    attach_layers: list[int]
    bias_l2: list[float]
    g_checkpoint: str


class SteerGenRecord(BaseModel):
    """``gen_records/gen_NNN.json`` — a B0 stage-1 record plus the steering block.

    The six fields ``run_replay_b0`` hard-requires (run_replay_b0.py:257-276) are
    all present and mean what that stage expects, with ONE overload stated here
    rather than discovered later: ``seed_idx`` is this record's own generation_id,
    not a B0 seed index. Steered gens have no B0 seed coordinate — the real
    coordinates are ``steering.{arm, rep, source_pair_key}`` — but the field is
    required downstream and must be unique per record inside a prompt_id, so the
    record id is what goes there. ``prompt_id``/``prompt_class`` ARE the source's:
    a steered gen continues the source's prompt, and the readout groups on it.
    """

    generation_id: int = Field(ge=0)
    prompt_id: str = Field(min_length=1)
    prompt_class: str = Field(min_length=1)
    prompt: str | None = None
    prompt_idx: int | None = None
    seed_idx: int = Field(ge=0)
    seed: int = Field(ge=0)
    system_prompt: None = None
    user_prompt: str | None = None
    prompt_length: int = Field(gt=0)
    num_generated_tokens: int = Field(ge=0)
    generated_text: str
    input_ids: list[int]
    sampling: SamplingParams
    model_id: str
    preset: str
    steering: SteeringMeta


# ── deterministic selection (no GPU, no torch — all of this is unit-tested) ─────


def make_steer_seed(source_pair_key: str, rep: int, seed_offset: int) -> int:
    """The RNG seed for one (source, rep) — the SAME seed for every arm.

    Keying on the arm would make base and matched independent draws, and the whole
    paired design (and the felt-shift bundle's side-by-side pair) rests on them
    being the same draw under two different interventions.

    Disjointness from every B0 wave comes from the domain prefix, not from
    arithmetic: B0's seeds are ``sha256("expB0_{prompt_id}_{seed_idx}")`` and these
    are ``sha256("expB1steer_{offset}_{pair_key}_{rep}")`` — different keyspaces
    entirely. ``--seed-offset`` is in the hashed string rather than added to the
    result, so bumping it redraws the whole wave cleanly instead of shifting it
    into a neighbouring one.
    """
    if rep < 0:
        raise ValueError(f"rep must be >= 0, got {rep}")
    if seed_offset < 0:
        raise ValueError(f"--seed-offset must be >= 0, got {seed_offset}")
    raw = f"expB1steer_{seed_offset}_{source_pair_key}_{rep}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def parse_arms(spec: str) -> list[str]:
    """Parse ``--arms``, preserving KNOWN_ARMS order so the run log is stable."""
    wanted = [a.strip() for a in spec.split(",") if a.strip()]
    if not wanted:
        raise ValueError("--arms is empty")
    unknown = [a for a in wanted if a not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; known arms are {list(KNOWN_ARMS)}")
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"duplicate arm in --arms: {wanted}")
    return [a for a in KNOWN_ARMS if a in wanted]


def parse_splits(spec: str) -> list[str]:
    """Parse ``--splits`` (comma-separated), refusing an empty selection."""
    splits = [s.strip() for s in spec.split(",") if s.strip()]
    if not splits:
        raise ValueError("--splits is empty")
    if len(set(splits)) != len(splits):
        raise ValueError(f"duplicate split in --splits: {splits}")
    return splits


def select_sources(
    pairs: Sequence[PairRecord], splits: Sequence[str], per_class: int
) -> list[PairRecord]:
    """Pick the source trajectories: up to ``per_class`` per prompt_class, 0 = all.

    Round-robin over prompt_id within each class, so ``--sources-per-class 4``
    spends its budget on four DIFFERENT prompts where four exist rather than on
    four seeds of one — the readout aggregates per class, and four seeds of one
    prompt is n=1 wearing an n=4 hat. Fully deterministic: candidates are sorted by
    pair_key and prompts by id, so the selection does not depend on manifest order.
    """
    wanted = set(splits)
    candidates = sorted(
        (p for p in pairs if p.split in wanted), key=lambda p: p.pair_key
    )
    if not candidates:
        raise ValueError(
            f"no pairs in split(s) {sorted(wanted)} — nothing to steer from "
            f"(manifest splits present: {sorted({p.split for p in pairs})})"
        )

    by_class: dict[str, list[PairRecord]] = {}
    for pair in candidates:
        by_class.setdefault(pair.prompt_class, []).append(pair)

    selected: list[PairRecord] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        if per_class <= 0:
            selected.extend(pool)
            continue
        by_prompt: dict[str, list[PairRecord]] = {}
        for pair in pool:
            by_prompt.setdefault(pair.prompt_id, []).append(pair)
        rings = [by_prompt[pid] for pid in sorted(by_prompt)]
        picked: list[PairRecord] = []
        depth = 0
        while len(picked) < per_class and any(len(r) > depth for r in rings):
            for ring in rings:
                if len(picked) >= per_class:
                    break
                if len(ring) > depth:
                    picked.append(ring[depth])
            depth += 1
        if len(picked) < per_class:
            logger.warning(
                "class %s has only %d source(s) in %s (asked for %d)",
                cls, len(picked), sorted(wanted), per_class,
            )
        selected.extend(picked)

    if not selected:
        raise ValueError("source selection came out empty; refusing to run")
    return selected


def pick_donor(
    source: PairRecord, pool: Sequence[PairRecord], arm: str, donor_pool: str
) -> tuple[PairRecord, str]:
    """(donor, pool_label) for one donor arm — deterministic per (arm, source).

    Mirrors ``eval_g.pick_donors``: prefer the source's own split (a donor from
    train is a signature g has already fit, which is not the control anyone means),
    fall back to the whole pool and RECORD that it happened.
    """
    if arm not in DONOR_ARMS:
        raise ValueError(f"{arm!r} is not a donor arm (donor arms: {list(DONOR_ARMS)})")
    if arm == "shuffled":
        def predicate(p: PairRecord) -> bool:
            return p.prompt_id != source.prompt_id
    else:
        def predicate(p: PairRecord) -> bool:
            return p.prompt_id == source.prompt_id and p.seed_idx != source.seed_idx

    key = f"steer-{arm}|{source.pair_key}"
    others = [p for p in pool if p.pair_key != source.pair_key]
    if donor_pool == "split":
        local = [p for p in others if p.split == source.split and predicate(p)]
        if local:
            return deterministic_choice(key, local), "split"
    wide = [p for p in others if predicate(p)]
    if not wide:
        raise ValueError(
            f"{source.pair_key}: no {arm} donor available "
            f"(prompt_id {source.prompt_id}, split {source.split})"
        )
    return deterministic_choice(key, wide), "all"


def attach_layers_for(inject_layers: Sequence[int], n_model_layers: int) -> list[int]:
    """Translate g's OUTPUT-side training sites into pre-hook INPUT-side sites.

    ``+1``, and the reason is the whole first half of the module docstring. The
    upper bound is a real constraint rather than a formality: a bias trained on the
    output of the LAST decoder layer has no "input of layer i+1" to be attached to
    (that stream goes straight to the final norm), so ``attach_residual_write``
    cannot express it and this refuses rather than clamping.
    """
    layers = [int(i) for i in inject_layers]
    if not layers:
        raise ValueError("checkpoint declares no injection layers")
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate injection layers in the checkpoint: {layers}")
    bad = [i for i in layers if i < 0 or i >= n_model_layers]
    if bad:
        raise ValueError(
            f"checkpoint injects at {bad}; the model has {n_model_layers} decoder layers"
        )
    attach = [i + 1 for i in layers]
    over = [i for i in attach if i >= n_model_layers]
    if over:
        raise ValueError(
            f"g was trained on the OUTPUT of layer(s) {[i - 1 for i in over]}, whose "
            f"input-side equivalent is layer {over} — out of range for a "
            f"{n_model_layers}-layer model. attach_residual_write is a forward_PRE_hook "
            "(model_loader.py:562) and cannot address the stream leaving the last layer; "
            "retrain g with an inject layer below the top, or extend this stage with an "
            "output-side hook."
        )
    return attach


def read_train_meta(checkpoint_path: Path) -> tuple[dict[str, Any], Path | None]:
    """``train_meta.json`` from beside the checkpoint — the declared g config.

    Dim/arch policy: nothing about g's shape is hard-coded here, and nothing is
    taken from a preset. ``train_meta.json`` is the human-readable declaration and
    the checkpoint's ``arch`` block is the machine one; ``main()`` cross-checks them
    and refuses to run on a disagreement, because "which inject layers" is exactly
    the question a silent mismatch would answer wrongly.
    """
    meta_path = checkpoint_path.expanduser().resolve().parent / "train_meta.json"
    if not meta_path.exists():
        return {}, None
    blob = json.loads(meta_path.read_text())
    if not isinstance(blob, dict):
        raise ValueError(f"{meta_path}: expected a JSON object")
    return blob, meta_path


def reconcile_g_config(
    arch: dict[str, Any], train_meta: dict[str, Any], meta_path: Path | None
) -> tuple[list[int], int]:
    """(inject_layers, hidden) agreed by the checkpoint and train_meta.json.

    Returns the checkpoint's values (it is what the weights were actually saved
    with) but only after train_meta.json agrees, when train_meta.json exists.
    """
    ckpt_layers = [int(i) for i in arch["inject_layers"]]
    ckpt_hidden = int(arch["hidden"])
    config = train_meta.get("config") or {}
    meta_layers = config.get("inject_layers")
    meta_hidden = config.get("hidden")
    if meta_path is None:
        logger.warning(
            "no train_meta.json beside the checkpoint — using the checkpoint's own "
            "arch block (inject_layers=%s, hidden=%d) with no cross-check",
            ckpt_layers, ckpt_hidden,
        )
        return ckpt_layers, ckpt_hidden
    if meta_layers is not None and [int(i) for i in meta_layers] != ckpt_layers:
        raise ValueError(
            f"{meta_path} declares inject_layers {list(meta_layers)} but the "
            f"checkpoint's arch says {ckpt_layers} — the checkpoint and its metadata "
            "describe different runs; refusing to guess which one the weights match"
        )
    if meta_hidden is not None and int(meta_hidden) != ckpt_hidden:
        raise ValueError(
            f"{meta_path} declares hidden {meta_hidden} but the checkpoint's arch says "
            f"{ckpt_hidden} — refusing to load g under a mismatched width"
        )
    logger.info(
        "g config agreed by %s and the checkpoint: inject_layers=%s, hidden=%d",
        meta_path, ckpt_layers, ckpt_hidden,
    )
    return ckpt_layers, ckpt_hidden


def plan_units(
    sources: Sequence[PairRecord], arms: Sequence[str], reps: int
) -> list[tuple[PairRecord, str, int]]:
    """The full (source, arm, rep) work list, in a stable order.

    Generation ids are assigned from this order, so a rerun with the same arguments
    writes the same gen_NNN.json for the same unit and ``--resume`` is meaningful.
    """
    if reps <= 0:
        raise ValueError(f"--reps must be > 0, got {reps}")
    return [
        (source, arm, rep)
        for source in sources
        for arm in arms
        for rep in range(reps)
    ]


# ── the GPU half ───────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="B0 stage-2 output dirs holding fork_series/*.npz (source prompts + tokens)",
    )
    parser.add_argument("--g-checkpoint", type=Path, required=True, help="train_g's g_state.pt")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b", help="anamnesis MODEL_PRESETS key")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="val_seed",
        help="comma-separated splits to draw source trajectories from",
    )
    parser.add_argument(
        "--sources-per-class",
        type=int,
        default=4,
        help="source trajectories per prompt_class, round-robin over prompts (0 = all)",
    )
    parser.add_argument(
        "--reps", type=int, default=4, help="free generations per arm per source"
    )
    parser.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        help=f"comma-separated subset of {list(KNOWN_ARMS)} (default: {DEFAULT_ARMS})",
    )
    parser.add_argument(
        "--donor-pool",
        default="split",
        choices=["split", "all"],
        help="draw donor signatures from the source's own split (default) or anywhere",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="matches the B0 waves, so base-arm gens are comparable to the corpus",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1000,
        help=(
            "hashed into every RNG seed; the domain prefix already separates this "
            "wave from B0's keyspace, and bumping this redraws the whole wave"
        ),
    )
    parser.add_argument(
        "--selfcheck-tokens",
        type=int,
        default=32,
        help="tokens drawn per arm of the zero-bias byte-identity gate at startup",
    )
    parser.add_argument(
        "--model-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="bf16 by default: the dtype g was trained against (train_meta model.dtype)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eos-ids", type=int, nargs="+", default=None, help="default = the preset's"
    )
    parser.add_argument("--limit", type=int, default=0, help="cap generations (0 = all)")
    parser.add_argument(
        "--no-resume", action="store_true", help="regenerate even when a record exists"
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        logger.error("--temperature must be > 0 (do_sample=True), got %s", args.temperature)
        return 2
    if not 0 < args.top_p <= 1:
        logger.error("--top-p must be in (0, 1], got %s", args.top_p)
        return 2
    if args.max_new_tokens <= 0:
        logger.error("--max-new-tokens must be > 0, got %s", args.max_new_tokens)
        return 2
    if args.selfcheck_tokens <= 0:
        logger.error("--selfcheck-tokens must be > 0, got %s", args.selfcheck_tokens)
        return 2

    try:
        arms = parse_arms(str(args.arms))
        splits = parse_splits(str(args.splits))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if args.reps <= 0:
        logger.error("--reps must be > 0, got %s", args.reps)
        return 2

    try:
        loaded = load_pairs(args.pairs_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load --pairs-dir %s: %s", args.pairs_dir, exc)
        return 2

    # Deferred so --help works on any machine, with or without the node env.
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA host-side graph builds per unseen kv_len (forensics 2026-09-16)
    torch.set_num_threads(1)

    from anamnesis.config import MODEL_PRESETS
    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pleroma import g_net

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
        return 2
    preset = MODEL_PRESETS[args.preset]

    eos_ids: list[int] = list(args.eos_ids) if args.eos_ids else list(preset.eos_token_ids)
    if not eos_ids:
        logger.error("no eos ids (preset %r has none; pass --eos-ids)", args.preset)
        return 2

    try:
        mapping = g_net.resolve_run_dirs(loaded.pairs, list(args.run_dirs))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    pool = [g_net.rebase(p, mapping) for p in loaded.pairs]
    row_of = {p.pair_key: p.row for p in loaded.pairs}

    try:
        sources = select_sources(pool, splits, int(args.sources_per_class))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    # Donors resolved up front: a missing donor is a configuration error, and it
    # should surface before a model is loaded, not 40 minutes into a GPU run.
    donors: dict[tuple[str, str], tuple[PairRecord, str]] = {}
    try:
        for source in sources:
            for arm in arms:
                if arm in DONOR_ARMS:
                    donors[(source.pair_key, arm)] = pick_donor(
                        source, pool, arm, str(args.donor_pool)
                    )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    units = plan_units(sources, arms, int(args.reps))
    if args.limit:
        units = units[: args.limit]
    class_counts: dict[str, int] = {}
    for source in sources:
        class_counts[source.prompt_class] = class_counts.get(source.prompt_class, 0) + 1
    logger.info(
        "%d sources over %d classes (%s) x %d arms %s x %d reps = %d generations",
        len(sources), len(class_counts),
        ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
        len(arms), arms, args.reps, len(units),
    )

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("--device %s but torch.cuda.is_available() is False", device)
        return 2

    # ── g ─────────────────────────────────────────────────────────────────────
    try:
        checkpoint = torch.load(args.g_checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("could not read --g-checkpoint %s: %s", args.g_checkpoint, exc)
        return 2
    arch = checkpoint.get("arch")
    if not arch:
        logger.error("%s has no 'arch' block — not a train_g checkpoint", args.g_checkpoint)
        return 2
    if int(arch["in_dim"]) != loaded.feature_dim:
        logger.error(
            "g was trained on %d-d signatures but the pairs dir has %d — different spaces",
            arch["in_dim"], loaded.feature_dim,
        )
        return 2
    ckpt_names = checkpoint.get("feature_names")
    if ckpt_names and list(ckpt_names) != loaded.feature_names:
        logger.error(
            "g's training feature_names differ from this pairs dir's — refusing to "
            "steer across two feature spaces"
        )
        return 2
    try:
        train_meta, train_meta_path = read_train_meta(args.g_checkpoint)
        inject_layers, g_hidden = reconcile_g_config(arch, train_meta, train_meta_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("g config unusable: %s", exc)
        return 2

    # ── the frozen model, loaded for GENERATION (not for training) ─────────────
    # Deliberately not g_net.load_frozen_model: that helper forces
    # ``config.use_cache = False`` (g_net.py:286-287), which is right for
    # teacher-forced training and ruinous here — every decode step would re-run the
    # full prefix. The cache_position-gated hook is correct either way; this is
    # purely about not being 300x slower.
    dtype = g_net.DTYPES[str(args.model_dtype)]
    logger.info("loading %s (sdpa, %s) for free generation", args.model_path, dtype)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = True
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
    except Exception as exc:  # noqa: BLE001 — loading is the commonest node-side failure
        logger.exception("could not load the model: %s", exc)
        return 1

    hidden_dim = g_net.model_hidden_dim(model)
    n_model_layers = len(g_net.decoder_layers(model))
    if int(arch["hidden_dim"]) != hidden_dim:
        logger.error(
            "g emits %d-d biases but the model's hidden_dim is %d",
            arch["hidden_dim"], hidden_dim,
        )
        return 2
    if int(preset.num_layers) != n_model_layers:
        logger.error(
            "preset %s says %d layers but %s has %d — refusing to run on a mismatched "
            "preset", args.preset, preset.num_layers, args.model_path, n_model_layers,
        )
        return 2
    try:
        attach_layers = attach_layers_for(inject_layers, n_model_layers)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "INJECTION SITE: g trained on the OUTPUT of decoder layers %s "
        "(g_net.py:178, forward hook) == the INPUT of layers %s; "
        "attach_residual_write is a forward_PRE_hook (model_loader.py:562), so this "
        "run attaches at %s. +1 shift applied.",
        inject_layers, attach_layers, attach_layers,
    )

    g = g_net.SignatureToBias(
        in_dim=int(arch["in_dim"]),
        hidden=g_hidden,
        n_layers=int(arch["n_layers"]),
        hidden_dim=hidden_dim,
    )
    g.load_state_dict(checkpoint["state_dict"])
    g.to(device=device, dtype=torch.float32).eval()
    g.requires_grad_(False)
    logger.info(
        "g loaded from %s (best epoch %s)", args.g_checkpoint, checkpoint.get("best_epoch")
    )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
    sampling = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        eos_ids=eos_ids,
        attn="sdpa",
        date_string=None,
    )

    # ── helpers closing over the loaded model ─────────────────────────────────

    def bias_vectors(z_row: np.ndarray) -> list[Any]:
        """g's per-layer bias for one signature row, as detached float32 vectors."""
        z = torch.from_numpy(
            np.ascontiguousarray(np.asarray(z_row, dtype=np.float32).reshape(1, -1))
        ).to(device)
        with torch.no_grad():
            bias = g(z)
        if int(bias.shape[1]) != len(inject_layers):
            raise RuntimeError(
                f"g emitted {int(bias.shape[1])} layer slots for "
                f"{len(inject_layers)} inject layers"
            )
        return [bias[0, i].detach().to(torch.float32).clone() for i in range(bias.shape[1])]

    def attach(vectors: list[Any], start_pos: int, alpha: float) -> list[Any]:
        """Attach one fixed-vector injection per trained layer, at the +1 site."""
        handles: list[Any] = []
        try:
            for vector, layer_idx in zip(vectors, attach_layers, strict=True):
                handles.append(
                    attach_residual_write(
                        model,
                        ResidualWriteSpec(
                            layer_idx=layer_idx,
                            vector=vector,
                            alpha=alpha,
                            start_pos=start_pos,
                            end_pos=None,
                            # g's bias magnitude is part of what g learned; the
                            # reference's unit-normalise + scalar dose is a dose
                            # ladder's convention, not ours.
                            normalize=False,
                        ),
                    )
                )
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def draw(prompt_ids: list[int], seed: int, max_new: int) -> list[int]:
        """One stock sampled generation. Returns the FULL realized sequence."""
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )
        return [int(x) for x in out[0].tolist()]

    # ── the not-a-perturbation gate ───────────────────────────────────────────
    probe_source = sources[0]
    try:
        probe_ids, probe_plen = g_net.load_sequence(probe_source)
    except (OSError, KeyError, ValueError) as exc:
        logger.error("could not load the self-check source %s: %s", probe_source.pair_key, exc)
        return 1
    probe_prompt = [int(x) for x in probe_ids[:probe_plen]]
    probe_seed = make_steer_seed(probe_source.pair_key, 0, int(args.seed_offset))
    n_probe = int(args.selfcheck_tokens)

    selfcheck: dict[str, Any] = {
        "source_pair_key": probe_source.pair_key,
        "seed": probe_seed,
        "n_tokens_requested": n_probe,
    }
    try:
        plain = draw(probe_prompt, probe_seed, n_probe)

        zero_vectors = [torch.zeros(hidden_dim, dtype=torch.float32) for _ in attach_layers]
        handles = attach(zero_vectors, probe_plen, 0.0)
        try:
            alpha_zero = draw(probe_prompt, probe_seed, n_probe)
        finally:
            for handle in handles:
                handle.remove()

        handles = attach(zero_vectors, probe_plen, 1.0)
        try:
            vector_zero = draw(probe_prompt, probe_seed, n_probe)
            stats = [dict(h.stats) for h in handles]
        finally:
            for handle in handles:
                handle.remove()
    except Exception as exc:  # noqa: BLE001 — the gate is allowed to fail loudly
        logger.exception("self-check could not run: %s", exc)
        return 3

    selfcheck["n_tokens_drawn"] = len(plain) - probe_plen
    selfcheck["alpha_zero_identical"] = plain == alpha_zero
    selfcheck["zero_vector_identical"] = plain == vector_zero
    selfcheck["hook_stats"] = stats
    saw_cache_position = all(bool(s.get("saw_cache_position")) for s in stats)
    injected = [int(s.get("positions", 0)) for s in stats]
    selfcheck["saw_cache_position"] = saw_cache_position
    selfcheck["positions_injected"] = injected

    if not selfcheck["alpha_zero_identical"] or not selfcheck["zero_vector_identical"]:
        logger.error(
            "SELF-CHECK FAILED — a zero bias changed the sampled tokens "
            "(alpha=0 identical: %s, zero-vector identical: %s). The injection path "
            "is a perturbation in its own right, so no steering number from this run "
            "would mean anything. Aborting.",
            selfcheck["alpha_zero_identical"], selfcheck["zero_vector_identical"],
        )
        return 3
    if not saw_cache_position or not all(n > 0 for n in injected):
        logger.error(
            "SELF-CHECK FAILED — the hooks were byte-neutral but never actually "
            "injected: saw_cache_position=%s, positions per layer=%s. A gate that "
            "never fires would make every arm the base arm. Aborting.",
            saw_cache_position, injected,
        )
        return 3
    if int(selfcheck["n_tokens_drawn"]) < 8:
        # An early EOS makes the identity test true over almost no tokens. It has not
        # FAILED, but it has not proved much either, and run_meta must not read as a
        # clean gate when only three tokens were compared.
        selfcheck["weak"] = True
        logger.warning(
            "SELF-CHECK is WEAK — the probe stopped after %d token(s), so byte "
            "identity was checked over almost nothing. Raise --selfcheck-tokens or "
            "pick a source whose continuation runs longer.",
            selfcheck["n_tokens_drawn"],
        )
    logger.info(
        "SELF-CHECK PASSED — %d tokens byte-identical under no-hook / alpha=0 / "
        "zero-vector; gating fired at %s positions per layer via cache_position",
        selfcheck["n_tokens_drawn"], injected,
    )

    # attach_residual_write logs one INFO line per registration (model_loader.py:565)
    # and the run re-attaches per (source, arm, rep) — four lines per generation would
    # bury the progress log. Quieted only AFTER the self-check, whose registrations are
    # the ones worth seeing, and only to WARNING: real problems still surface.
    from anamnesis.extraction import model_loader as _model_loader

    _model_loader.logger.setLevel(logging.WARNING)

    # ── the run ───────────────────────────────────────────────────────────────
    records_dir = args.out_dir / "gen_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    # Source-level things computed ONCE: the banked tokens, and g's bias per arm.
    source_state: dict[str, dict[str, Any]] = {}
    cells_cache: dict[str, dict[int, Any]] = {}
    n_done = 0
    n_skipped = 0
    failures: list[str] = []
    written: list[dict[str, Any]] = []
    started = time.time()

    for index, (source, arm, rep) in enumerate(units):
        out_path = records_dir / f"gen_{index:04d}.json"
        if out_path.exists() and not args.no_resume:
            n_skipped += 1
            continue
        try:
            state = source_state.get(source.pair_key)
            if state is None:
                ids, plen = g_net.load_sequence(source)
                # Prompt text for the felt-shift bundle: the run dir's cells.json is
                # authoritative; decoding the banked prompt ids is the fallback, and
                # is what a steer dir built from a run without cells.json gets.
                run_dir = str(Path(source.run_dir).expanduser().resolve())
                if run_dir not in cells_cache:
                    try:
                        cells_cache[run_dir] = load_cells(Path(run_dir))
                    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "no usable cells.json under %s (%s) — prompt text will be "
                            "decoded from the banked prompt ids", run_dir, exc,
                        )
                        cells_cache[run_dir] = {}
                cell = cells_cache[run_dir].get(source.generation_id)
                prompt_ids = [int(x) for x in ids[:plen]]
                decoded_prompt = tok.decode(prompt_ids, skip_special_tokens=True)
                state = {
                    "prompt_ids": prompt_ids,
                    "prompt_length": int(plen),
                    "prompt": (cell.prompt if cell is not None else None) or decoded_prompt,
                    "prompt_idx": cell.prompt_idx if cell is not None else None,
                    "user_prompt": decoded_prompt,
                    "bias": {},
                }
                source_state[source.pair_key] = state

            if arm == "base":
                vectors: list[Any] | None = None
                donor: PairRecord | None = None
                donor_pool_label: str | None = None
            else:
                if arm == "matched":
                    donor, donor_pool_label = None, None
                    z_row = loaded.z[row_of[source.pair_key]]
                else:
                    donor, donor_pool_label = donors[(source.pair_key, arm)]
                    z_row = loaded.z[row_of[donor.pair_key]]
                if arm not in state["bias"]:
                    state["bias"][arm] = bias_vectors(z_row)
                vectors = state["bias"][arm]

            seed = make_steer_seed(source.pair_key, rep, int(args.seed_offset))
            handles: list[Any] = []
            if vectors is not None:
                handles = attach(vectors, state["prompt_length"], 1.0)
            try:
                full_seq = draw(state["prompt_ids"], seed, int(args.max_new_tokens))
            finally:
                for handle in handles:
                    handle.remove()

            generated_ids = full_seq[state["prompt_length"]:]
            if len(generated_ids) < 2:
                # Same floor as B0 stage 1: replay_extract needs N >= 2 for a single
                # per-step transition, so a 0/1-token gen is dead weight downstream.
                raise RuntimeError(
                    f"only {len(generated_ids)} generated token(s) — the replay stage "
                    "needs >= 2 for a per-step transition"
                )

            record = SteerGenRecord(
                generation_id=index,
                prompt_id=source.prompt_id,
                prompt_class=source.prompt_class,
                prompt=state["prompt"],
                prompt_idx=state["prompt_idx"],
                seed_idx=index,
                seed=seed,
                user_prompt=state["user_prompt"],
                prompt_length=state["prompt_length"],
                num_generated_tokens=len(generated_ids),
                generated_text=tok.decode(generated_ids, skip_special_tokens=True),
                input_ids=full_seq,
                sampling=sampling,
                model_id=str(args.model_path),
                preset=str(args.preset),
                steering=SteeringMeta(
                    arm=arm,
                    rep=rep,
                    rng_seed=seed,
                    source_pair_key=source.pair_key,
                    source_generation_id=source.generation_id,
                    source_run_label=source.run_label,
                    source_split=source.split,
                    source_prompt_id=source.prompt_id,
                    source_seed_idx=source.seed_idx,
                    donor_pair_key=donor.pair_key if donor is not None else None,
                    donor_prompt_id=donor.prompt_id if donor is not None else None,
                    donor_seed_idx=donor.seed_idx if donor is not None else None,
                    donor_pool=donor_pool_label,
                    inject_layers_trained=list(inject_layers),
                    attach_layers=list(attach_layers),
                    bias_l2=(
                        [0.0] * len(attach_layers)
                        if vectors is None
                        else [float(v.norm()) for v in vectors]
                    ),
                    g_checkpoint=str(args.g_checkpoint),
                ),
            )
            out_path.write_text(record.model_dump_json())
            written.append(
                {
                    "generation_id": index,
                    "arm": arm,
                    "rep": rep,
                    "source_pair_key": source.pair_key,
                    "prompt_id": source.prompt_id,
                    "prompt_class": source.prompt_class,
                    "num_generated_tokens": record.num_generated_tokens,
                }
            )
            n_done += 1
            if (index + 1) % 20 == 0 or index == 0:
                elapsed = time.time() - started
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (len(units) - index - 1) / rate if rate > 0 else 0.0
                logger.info(
                    "%d/%d gen_%04d (%s %s rep %d): %d tok, %.0fs (%.2f gen/s, ETA %.0fs)",
                    index + 1, len(units), index, source.pair_key, arm, rep,
                    record.num_generated_tokens, elapsed, rate, eta,
                )
        except Exception as exc:  # noqa: BLE001 — a dead gen is recorded, never skipped
            logger.exception("gen_%04d failed", index)
            failures.append(f"gen_{index:04d}: {type(exc).__name__}: {exc}")

    if n_done == 0 and n_skipped == 0:
        logger.error("no generation succeeded; refusing to write an empty result")
        return 1

    (args.out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "stage": "expB1_steer_generate",
                "model_id": str(args.model_path),
                "preset": str(args.preset),
                "model_dtype": str(args.model_dtype),
                "attn_implementation": "sdpa",
                "pairs_dir": str(args.pairs_dir),
                "run_dirs": [str(d) for d in args.run_dirs],
                "g_checkpoint": str(args.g_checkpoint),
                "g_best_epoch": checkpoint.get("best_epoch"),
                "train_meta_path": str(train_meta_path) if train_meta_path else None,
                "injection": {
                    "inject_layers_trained": list(inject_layers),
                    "attach_layers": list(attach_layers),
                    "shift": "+1",
                    "note": (
                        "g_net.TrainableResidualInjector is a forward HOOK on the "
                        "layer OUTPUT (g_net.py:178); attach_residual_write is a "
                        "forward_PRE_hook on the layer INPUT (model_loader.py:562). "
                        "The same residual-stream site is layer i's output and layer "
                        "i+1's input, hence attach_layers = inject_layers + 1."
                    ),
                    "normalize": False,
                    "alpha": 1.0,
                    "start_pos": "per-source prompt_length (absolute, cache_position-gated)",
                },
                "selfcheck": selfcheck,
                "splits": splits,
                "arms": arms,
                "donor_pool": str(args.donor_pool),
                "sources_per_class": int(args.sources_per_class),
                "reps": int(args.reps),
                "seed_offset": int(args.seed_offset),
                "sampling": sampling.model_dump(),
                "class_counts": class_counts,
                "sources": [
                    {
                        "pair_key": s.pair_key,
                        "prompt_id": s.prompt_id,
                        "prompt_class": s.prompt_class,
                        "seed_idx": s.seed_idx,
                        "split": s.split,
                        "donors": {
                            arm: {
                                "pair_key": donors[(s.pair_key, arm)][0].pair_key,
                                "prompt_id": donors[(s.pair_key, arm)][0].prompt_id,
                                "pool": donors[(s.pair_key, arm)][1],
                            }
                            for arm in arms
                            if arm in DONOR_ARMS
                        },
                    }
                    for s in sources
                ],
                "n_units": len(units),
                "n_written": n_done,
                "n_resumed": n_skipped,
                "gens": written,
                "failures": failures,
                "elapsed_s": round(time.time() - started, 1),
            },
            indent=2,
        )
    )

    logger.info(
        "DONE — %d generations written (%d resumed, %d failed) in %.1fs -> %s",
        n_done, n_skipped, len(failures), time.time() - started, args.out_dir,
    )
    logger.info(
        "NEXT: run_replay_b0 --gen-dir %s --out-dir %s --save-raw none, then "
        "basin_readout + feltshift_bundle locally.", args.out_dir, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
