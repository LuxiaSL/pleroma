"""Exp B1 stage 2: train g — signature -> residual bias — against the FROZEN 3B.

Node GPU. One trainable object in the whole run: ``g``, a ~3M-parameter MLP that
reads a trajectory's z-scored v3 signature and emits an additive residual-stream
bias per injected layer. The 3B teacher is loaded frozen and in eval mode and is
never updated; the gradient reaches g THROUGH it.

The training signal, in one sentence: teacher-force the model on
(prompt + source continuation i), add ``g(sig_i)`` to the residual stream at the
continuation positions, and minimise the NLL of trajectory i's OWN tokens at the
positions where it diverged from its siblings.

Why that is not circular: the tokens are already fixed (B0 banked them), the model
is frozen, and the bias is a function of the signature alone — it never sees the
tokens. The only way g can lower the NLL is by mapping a signature onto a
perturbation that makes that trajectory's basin more likely. ``eval_g`` then asks
the question this loss cannot: does g(own signature) beat g(a SIBLING's signature)
— same prompt, same fork structure, different basin?

Two properties of the setup that make "it did nothing" legible:

  - g's output layer is ZERO-INITIALISED, so epoch 0 is bit-identical to alpha=0.
  - The alpha=0 validation NLL is computed once and cached. Every epoch reports
    both arms side by side, so a val curve that never separates from its own
    baseline is visible immediately rather than after eval.

Attention implementation is **sdpa**, not eager. B0 stage 2 needed eager because
its capture surface reads attention weights; B1 reads logits only, never attention,
so eager buys nothing here and costs a large constant factor in both time and
memory. (Stated explicitly because "the B0 script used eager" is the obvious wrong
inference.)

Model dtype defaults to **bfloat16**, not the 3b preset's float16. Backprop through
a frozen fp16 stack underflows; bf16 has the range and all four B1 arms use the
same dtype, so nothing is being compared across dtypes. ``--model-dtype float16``
restores the preset's value for anyone who wants the B0-identical forward.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.train_g \\
        --pairs-dir outputs/expB1/pairs \\
        --run-dirs outputs/expB0_pilot \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --inject-layers 7,14,18,21 --hidden 512 \\
        --epochs 8 --lr 1e-4 --batch 8 \\
        --out-dir outputs/expB1/g_v1 --seed 0
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import (house rule).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pleroma.build_pairs import PairRecord, load_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB1.train")

#: Default injection layers — the 3b preset's mid-depth sampled layers, where the
#: v3 signature's discriminative mass sits (expA). Not a dimension, so not derived.
DEFAULT_INJECT_LAYERS = "7,14,18,21"

#: Sequence cap. B0's pilot gens are 42 + 384 = 426 tokens; the cap is the
#: batching contract, not a property of the data.
DEFAULT_MAX_SEQ_LEN = 450

#: Weight on the optional all-continuation-positions NLL term.
ALL_POSITIONS_WEIGHT = 0.1


def parse_layers(spec: str) -> list[int]:
    """"7,14,18,21" -> [7, 14, 18, 21], rejecting duplicates and junk loudly."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--inject-layers {spec!r} names no layer")
    try:
        layers = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--inject-layers {spec!r} is not a comma-separated int list") from exc
    if any(i < 0 for i in layers):
        raise ValueError(f"--inject-layers {spec!r} has a negative index")
    if len(set(layers)) != len(layers):
        raise ValueError(f"--inject-layers {spec!r} repeats a layer")
    return layers


def usable_pairs(
    pairs: Sequence[PairRecord], max_seq_len: int
) -> tuple[list[PairRecord], list[str]]:
    """Drop pairs whose every fork position falls past the sequence cap.

    Returns (kept, dropped-with-reason). A dropped pair is recorded in train_meta,
    never skipped silently — "n pairs trained" must reconcile with the manifest.
    """
    kept: list[PairRecord] = []
    dropped: list[str] = []
    for pair in pairs:
        limit = min(pair.prompt_length + pair.n_tokens, max_seq_len)
        survivors = [p for p in pair.fork_positions if 0 < pair.prompt_length + p < limit]
        if survivors:
            kept.append(pair)
        else:
            dropped.append(
                f"{pair.pair_key}: fork positions {pair.fork_positions} all past the "
                f"{max_seq_len}-token cap (prompt_length {pair.prompt_length})"
            )
    return kept, dropped


def batches(indices: list[int], size: int) -> list[list[int]]:
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def main() -> int:  # noqa: C901 — one linear procedure, kept readable by sections
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs-dir", type=Path, required=True, help="build_pairs output dir")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="B0 output dirs holding fork_series/*.npz (input_ids + prompt_length)",
    )
    parser.add_argument("--model-path", required=True, help="local model dir")
    parser.add_argument("--preset", default="3b", help="anamnesis MODEL_PRESETS key (validation only)")
    parser.add_argument("--inject-layers", default=DEFAULT_INJECT_LAYERS)
    parser.add_argument("--hidden", type=int, default=512, help="g's hidden width")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="cap TRAIN pairs (0 = all)")
    parser.add_argument(
        "--permute-signatures",
        type=int,
        default=0,
        metavar="SEED",
        help=(
            "KILL RUNG (anamnesis ferry 2026-09-16 §4.2): non-zero seed pairs every "
            "trajectory with a RANDOM OTHER trajectory's signature (fixed-point-free "
            "permutation of z rows across all splits) before training. A g trained "
            "this way must show matched-minus-sibling ~= 0 under the standard eval; "
            "anything else means the design leaks prompt identity at train time. "
            "0 = off (normal training)."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 autocast around the frozen forward (default on)",
    )
    parser.add_argument(
        "--loss-all-positions",
        action="store_true",
        help=f"add {ALL_POSITIONS_WEIGHT}-weighted NLL over ALL continuation positions",
    )
    parser.add_argument(
        "--model-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="teacher weights dtype (bf16 by default; see the module docstring)",
    )
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument(
        "--patience", type=int, default=3, help="early stop after N epochs without val_seed gain"
    )
    parser.add_argument(
        "--lm-chunk",
        type=int,
        default=2048,
        help="positions per lm_head chunk (memory knob, not a result knob)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for name, value in (("--hidden", args.hidden), ("--batch", args.batch),
                        ("--epochs", args.epochs), ("--max-seq-len", args.max_seq_len),
                        ("--lm-chunk", args.lm_chunk)):
        if value <= 0:
            logger.error("%s must be > 0, got %s", name, value)
            return 2
    if args.lr <= 0:
        logger.error("--lr must be > 0, got %s", args.lr)
        return 2

    try:
        inject_layers = parse_layers(args.inject_layers)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        loaded = load_pairs(args.pairs_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load --pairs-dir %s: %s", args.pairs_dir, exc)
        return 2
    logger.info(
        "pairs: %d x %d dims (%s)",
        len(loaded.pairs), loaded.feature_dim,
        {s: len(loaded.by_split(s)) for s in ("train", "val_seed", "val_prompt")},
    )

    if args.permute_signatures:
        # Fixed-point-free permutation: rotate a seeded shuffle by one, so no
        # trajectory keeps its own signature. Applied to ALL splits, so the
        # training-time early stopping runs under the same broken mapping the
        # optimizer sees — the eval script loads the true mapping separately.
        rng = np.random.default_rng(int(args.permute_signatures))
        order = rng.permutation(len(loaded.pairs))
        rotated = np.roll(order, 1)
        row_map = {int(order[i]): int(rotated[i]) for i in range(len(order))}
        loaded = dataclasses.replace(
            loaded,
            pairs=[dataclasses.replace(p, row=row_map[p.row]) for p in loaded.pairs],
        )
        logger.warning(
            "KILL RUNG ACTIVE: --permute-signatures %d — every trajectory now wears "
            "a random OTHER trajectory's signature (fixed-point-free). matched-minus-"
            "sibling under standard eval must be ~0 or the design leaks.",
            args.permute_signatures,
        )

    # Deferred: torch/transformers live behind main() so --help works anywhere.
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA host-side graph builds per unseen kv_len (forensics 2026-09-16)
    from pleroma import g_net

    try:
        mapping = g_net.resolve_run_dirs(loaded.pairs, list(args.run_dirs))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    all_pairs = [g_net.rebase(p, mapping) for p in loaded.pairs]

    kept, dropped = usable_pairs(all_pairs, args.max_seq_len)
    for line in dropped:
        logger.error("dropped %s", line)
    if not kept:
        logger.error("no pair has a usable fork position; refusing to train")
        return 1
    keep_keys = {p.pair_key for p in kept}
    row_of = {p.pair_key: p.row for p in loaded.pairs}

    split_pairs: dict[str, list[PairRecord]] = {
        split: [p for p in kept if p.split == split]
        for split in ("train", "val_seed", "val_prompt")
    }
    if args.limit:
        split_pairs["train"] = split_pairs["train"][: args.limit]
    if not split_pairs["train"]:
        logger.error("no train pairs survived; refusing to train")
        return 1
    for split in ("val_seed", "val_prompt"):
        if not split_pairs[split]:
            logger.warning("%s is empty — its reported NLL will be null", split)

    # Reproducibility: seeds set before g is constructed, since g's first two
    # layers are randomly initialised (the third is zeros by design).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("--device %s but torch.cuda.is_available() is False", device)
        return 2

    # Preset is validation-only here: it must not silently disagree with the model
    # that is actually loaded, and no dimension is ever taken from it (dim policy).
    preset_layers: int | None = None
    try:
        from anamnesis.config import MODEL_PRESETS

        if args.preset not in MODEL_PRESETS:
            logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
            return 2
        preset_layers = int(MODEL_PRESETS[args.preset].num_layers)
    except ImportError:
        logger.warning("anamnesis not importable; skipping the preset cross-check")

    try:
        model = g_net.load_frozen_model(
            args.model_path, g_net.DTYPES[args.model_dtype], device
        )
    except Exception as exc:  # noqa: BLE001 — a load failure must name itself
        logger.exception("could not load the teacher: %s", exc)
        return 1

    hidden_dim = g_net.model_hidden_dim(model)
    n_model_layers = len(g_net.decoder_layers(model))
    if preset_layers is not None and preset_layers != n_model_layers:
        logger.error(
            "preset %s says %d layers but %s has %d — refusing to run on a "
            "mismatched preset", args.preset, preset_layers, args.model_path, n_model_layers,
        )
        return 2
    logger.info("teacher: %d layers x %d hidden", n_model_layers, hidden_dim)
    out_of_range = [i for i in inject_layers if i >= n_model_layers]
    if out_of_range:
        logger.error(
            "--inject-layers %s out of range: the teacher has %d layers",
            out_of_range, n_model_layers,
        )
        return 2

    pad_id = g_net.resolve_pad_token_id(model, args.model_path)

    g = g_net.SignatureToBias(
        in_dim=loaded.feature_dim,
        hidden=args.hidden,
        n_layers=len(inject_layers),
        hidden_dim=hidden_dim,
    ).to(device=device, dtype=torch.float32)
    n_params = sum(p.numel() for p in g.parameters())
    logger.info(
        "g: %d params (%d -> %d -> %d x %d), output layer zero-initialised",
        n_params, loaded.feature_dim, args.hidden, len(inject_layers), hidden_dim,
    )

    optimizer = torch.optim.AdamW(g.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def make_batch(pairs: Sequence[PairRecord]) -> Any:
        rows = np.asarray([row_of[p.pair_key] for p in pairs], dtype=np.int64)
        return g_net.build_batch(
            pairs,
            loaded.z[rows],
            pad_token_id=pad_id,
            max_seq_len=args.max_seq_len,
            include_all_positions=bool(args.loss_all_positions),
        ).to(device)

    all_w = ALL_POSITIONS_WEIGHT if args.loss_all_positions else 0.0

    def evaluate(pairs: Sequence[PairRecord], with_g: bool) -> float | None:
        """Mean per-trajectory fork NLL over a split, under one arm."""
        if not pairs:
            return None
        per_traj: list[float] = []
        g.eval()
        with torch.no_grad():
            for chunk in batches(list(range(len(pairs))), args.batch):
                batch = make_batch([pairs[i] for i in chunk])
                bias = g(batch.z) if with_g else None
                with g_net.autocast_context(device, bool(args.amp)):
                    result = g_net.run_arm(model, injector, batch, bias, 0.0, args.lm_chunk)
                per_traj.extend(
                    g_net.per_sample_means(result.fork_nll, result.fork_rows, batch.size)
                )
        return float(np.mean(per_traj)) if per_traj else None

    started = time.time()
    epochs_log: list[dict[str, Any]] = []
    failures: list[str] = list(dropped)
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stopped_early = False

    with g_net.TrainableResidualInjector(model, inject_layers) as injector:
        # alpha=0 baselines: the frozen model alone. Computed ONCE — it cannot
        # change, since nothing about the model or the data moves during training.
        base_val = {
            split: evaluate(split_pairs[split], with_g=False)
            for split in ("val_seed", "val_prompt")
        }
        base_train = evaluate(split_pairs["train"], with_g=False)
        logger.info(
            "alpha=0 baselines — train %s, val_seed %s, val_prompt %s",
            _fmt(base_train), _fmt(base_val["val_seed"]), _fmt(base_val["val_prompt"]),
        )
        if base_train is None:
            logger.error("the alpha=0 baseline produced no number; refusing to train")
            return 1

        order = list(range(len(split_pairs["train"])))
        for epoch in range(args.epochs):
            g.train()
            rng = random.Random(args.seed * 1000 + epoch)
            rng.shuffle(order)
            epoch_started = time.time()
            losses: list[float] = []
            n_failed_steps = 0

            for step, chunk in enumerate(batches(order, args.batch)):
                pairs = [split_pairs["train"][i] for i in chunk]
                try:
                    batch = make_batch(pairs)
                    bias = g(batch.z)
                    with g_net.autocast_context(device, bool(args.amp)):
                        result = g_net.run_arm(
                            model, injector, batch, bias, all_w, args.lm_chunk
                        )
                    loss = result.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss {float(loss)}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(g.parameters(), args.grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                except (RuntimeError, ValueError, OSError, KeyError, FloatingPointError) as exc:
                    # A dead step is recorded and the run continues; a dead EPOCH
                    # is caught below by the empty-losses check.
                    logger.exception("epoch %d step %d failed", epoch, step)
                    failures.append(f"epoch {epoch} step {step}: {type(exc).__name__}: {exc}")
                    n_failed_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            if not losses:
                logger.error("epoch %d produced no successful step; aborting", epoch)
                failures.append(f"epoch {epoch}: every step failed")
                break

            val_g = {
                split: evaluate(split_pairs[split], with_g=True)
                for split in ("val_seed", "val_prompt")
            }
            # Diagnostic only: how big a bias is g actually emitting? A run whose
            # val NLL never moves reads very differently at L2 ~ 0 (g stayed at
            # its zero init) than at L2 ~ 10 (g is steering, just not usefully).
            sample_rows = np.asarray(
                [row_of[p.pair_key] for p in split_pairs["train"][: args.batch]],
                dtype=np.int64,
            )
            norms = g.bias_norms(
                torch.from_numpy(np.ascontiguousarray(loaded.z[sample_rows])).to(device)
            )
            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "n_steps": len(losses),
                "n_failed_steps": n_failed_steps,
                "val_seed_nll_g": val_g["val_seed"],
                "val_seed_nll_base": base_val["val_seed"],
                "val_prompt_nll_g": val_g["val_prompt"],
                "val_prompt_nll_base": base_val["val_prompt"],
                "bias_l2_per_layer": {
                    str(layer): round(norm, 5) for layer, norm in zip(inject_layers, norms)
                },
                "elapsed_s": round(time.time() - epoch_started, 1),
            }
            epochs_log.append(record)
            logger.info(
                "epoch %d: train %.4f | val_seed %s (base %s, delta %s) | "
                "val_prompt %s (base %s, delta %s) | bias L2 %s | %.0fs",
                epoch, record["train_loss"],
                _fmt(val_g["val_seed"]), _fmt(base_val["val_seed"]),
                _fmt(_delta(val_g["val_seed"], base_val["val_seed"])),
                _fmt(val_g["val_prompt"]), _fmt(base_val["val_prompt"]),
                _fmt(_delta(val_g["val_prompt"], base_val["val_prompt"])),
                record["bias_l2_per_layer"], record["elapsed_s"],
            )

            # Early stop on val_seed — the split the headline contrast lives on.
            monitor = val_g["val_seed"] if val_g["val_seed"] is not None else record["train_loss"]
            if monitor < best_val - 1e-6:
                best_val = monitor
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in g.state_dict().items()}
            elif args.patience > 0 and epoch - best_epoch >= args.patience:
                logger.info(
                    "early stop: %d epochs without a val_seed gain (best epoch %d, %.4f)",
                    args.patience, best_epoch, best_val,
                )
                stopped_early = True
                break

    if not epochs_log:
        logger.error("no epoch completed; refusing to write an empty result")
        return 1
    if best_state is None:
        logger.warning("no epoch improved on init; saving the final state")
        best_state = {k: v.detach().cpu().clone() for k, v in g.state_dict().items()}
        best_epoch = epochs_log[-1]["epoch"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "arch": {
                "in_dim": loaded.feature_dim,
                "hidden": args.hidden,
                "n_layers": len(inject_layers),
                "hidden_dim": hidden_dim,
                "inject_layers": inject_layers,
            },
            "best_epoch": best_epoch,
            "feature_names": loaded.feature_names,
        },
        args.out_dir / "g_state.pt",
    )
    meta = {
        "stage": "expB1_train_g",
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "run_dirs": [str(d) for d in args.run_dirs],
            "inject_layers": inject_layers,
            "all_positions_weight": all_w,
        },
        "model": {
            "path": args.model_path,
            "dtype": args.model_dtype,
            "attn_implementation": "sdpa",
            "n_layers": n_model_layers,
            "hidden_dim": hidden_dim,
            "frozen": True,
        },
        "g": {"n_params": int(n_params), "output_layer_init": "zeros"},
        "pairs": {
            "dir": str(args.pairs_dir),
            "feature_dim": loaded.feature_dim,
            "n_manifest": len(loaded.pairs),
            "n_usable": len(keep_keys),
            "n_train": len(split_pairs["train"]),
            "n_val_seed": len(split_pairs["val_seed"]),
            "n_val_prompt": len(split_pairs["val_prompt"]),
            "build_meta": loaded.meta.get("params"),
            "holdout": loaded.meta.get("holdout"),
        },
        "permute_signatures_seed": int(args.permute_signatures) or None,
        "baselines_alpha0": {"train": base_train, **base_val},
        "epochs": epochs_log,
        "best_epoch": best_epoch,
        "best_val_seed_nll": None if best_val == float("inf") else best_val,
        "stopped_early": stopped_early,
        "injector_stats": dict(injector.stats),
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "DONE — %d epochs, best %d (val_seed %s vs base %s) in %.0fs -> %s",
        len(epochs_log), best_epoch, _fmt(best_val if best_val != float("inf") else None),
        _fmt(base_val["val_seed"]), time.time() - started, args.out_dir,
    )
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


if __name__ == "__main__":
    sys.exit(main())
