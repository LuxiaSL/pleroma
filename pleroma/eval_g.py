"""Exp B1 stage 3: does g's steering carry the BASIN, or just the prompt?

Node GPU. Four teacher-forced arms per held-out trajectory, all on the same frozen
model, the same tokens, the same fork positions, differing ONLY in what signature g
is conditioned on:

    base      alpha = 0 — the frozen model, no injection at all
    matched   g(z_i)   — the trajectory's OWN signature
    sibling   g(z_j)   — same prompt_id, DIFFERENT seed_idx
    shuffled  g(z_k)   — a different prompt entirely

``matched - base`` says the injection helps at all. It is the weak test: g could
be learning "make continuations of this prompt likelier", which any prompt-class
feature would give it.

``matched - sibling`` is the headline. A sibling shares the prompt, the prompt
class, the fork structure and the whole distribution of continuations — everything
except which basin the sampler fell into. If matched beats sibling, the 2713-d
signature carries basin-specific information and g found it. If matched and sibling
are indistinguishable while both beat base, g learned the PROMPT and nothing
finer — a real result, and the one this arm exists to be able to say.

``matched - shuffled`` is the coarse control kept for continuity with B0's plan.

**The headline number is matched-minus-sibling on val_seed** — held-out seeds of
prompts g trained on, so the prompt is not a confound and the sibling contrast is
as clean as the data allows. ``val_prompt`` (whole unseen prompts) answers the
harder transfer question and is reported beside it, never instead of it.

Sibling and shuffled donors are drawn DETERMINISTICALLY per gen (blake2b of the
pair key), so a rerun compares the same trajectories, and by default from the SAME
split — a donor from train would be a signature g has already fit, which is not the
control anyone means. ``--donor-pool all`` widens it; whichever pool was used is
recorded per gen.

Per-gen paired values are written out (one mean fork NLL per arm per trajectory),
so sign tests, paired bootstraps and per-class breakdowns run later off the JSON
without another GPU pass. This script deliberately does NOT compute a p-value: the
pairing is the thing that has to be banked correctly, and choosing the test after
seeing the numbers is a separate decision.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.eval_g \\
        --pairs-dir outputs/expB1/pairs \\
        --run-dirs outputs/expB0_pilot \\
        --g-checkpoint outputs/expB1/g_v1/g_state.pt \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --out outputs/expB1/g_v1/eval_report.json
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import (house rule).
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

from pleroma.build_pairs import PairRecord, load_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB1.eval")

ARMS: tuple[str, ...] = ("base", "matched", "sibling", "shuffled")
EVAL_SPLITS: tuple[str, ...] = ("val_seed", "val_prompt")
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("matched_minus_base", "base"),
    ("matched_minus_sibling", "sibling"),
    ("matched_minus_shuffled", "shuffled"),
)


def deterministic_choice(key: str, candidates: Sequence[PairRecord]) -> PairRecord:
    """Pick one candidate by hashing ``key`` — same draw on every rerun, any machine.

    Candidates are sorted by pair_key first, so the choice does not depend on the
    order the manifest happened to be in.
    """
    if not candidates:
        raise ValueError(f"no donor candidate for {key}")
    ordered = sorted(candidates, key=lambda p: p.pair_key)
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return ordered[int.from_bytes(digest, "big") % len(ordered)]


def pick_donors(
    pair: PairRecord, pool: list[PairRecord], donor_pool: str
) -> tuple[PairRecord, str, PairRecord, str]:
    """(sibling, sibling_pool, shuffled, shuffled_pool) for one gen.

    ``donor_pool == "split"`` restricts donors to the gen's own split and falls
    back to the full pool only when the split cannot supply one — recorded either
    way, because "the sibling came from train" changes what the arm means.
    """
    same_split = [p for p in pool if p.split == pair.split and p.pair_key != pair.pair_key]
    everything = [p for p in pool if p.pair_key != pair.pair_key]

    def choose(predicate: Any, label: str) -> tuple[PairRecord, str]:
        if donor_pool == "split":
            local = [p for p in same_split if predicate(p)]
            if local:
                return deterministic_choice(f"{label}|{pair.pair_key}", local), "split"
        wide = [p for p in everything if predicate(p)]
        if not wide:
            raise ValueError(
                f"{pair.pair_key}: no {label} donor available "
                f"(prompt_id {pair.prompt_id}, split {pair.split})"
            )
        return deterministic_choice(f"{label}|{pair.pair_key}", wide), "all"

    sibling, sib_pool = choose(
        lambda p: p.prompt_id == pair.prompt_id and p.seed_idx != pair.seed_idx, "sibling"
    )
    shuffled, shuf_pool = choose(lambda p: p.prompt_id != pair.prompt_id, "shuffled")
    return sibling, sib_pool, shuffled, shuf_pool


def paired_stats(values: list[float]) -> dict[str, Any]:
    """Mean/median/n/sign counts for one paired contrast. No p-value on purpose."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "n_negative": 0, "n_positive": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        # Negative = matched is LOWER NLL = the signature helped.
        "n_negative": int((arr < 0).sum()),
        "n_positive": int((arr > 0).sum()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm means and the three paired contrasts over a set of per-gen rows."""
    block: dict[str, Any] = {"n_gens": len(rows)}
    for arm in ARMS:
        vals = [r["nll"][arm] for r in rows if r["nll"].get(arm) is not None]
        block[f"mean_nll_{arm}"] = float(np.mean(vals)) if vals else None
    for name, other in CONTRASTS:
        block[name] = paired_stats(
            [
                r["nll"]["matched"] - r["nll"][other]
                for r in rows
                if r["nll"].get("matched") is not None and r["nll"].get(other) is not None
            ]
        )
    return block


def format_table(report: dict[str, Any]) -> str:
    """The compact human-readable table printed at the end of a run."""
    lines: list[str] = []
    header = (
        f"{'split / class':<28}{'n':>5}{'base':>9}{'matched':>9}{'sibling':>9}"
        f"{'shuffled':>10}{'m-base':>9}{'m-sib':>9}{'m-shuf':>9}{'sib<0':>8}"
    )

    def row(label: str, block: dict[str, Any]) -> str:
        def num(value: float | None, width: int, places: int = 4) -> str:
            return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{places}f}"

        sib = block["matched_minus_sibling"]
        sign = f"{sib['n_negative']}/{sib['n']}" if sib["n"] else "-"
        return (
            f"{label:<28}{block['n_gens']:>5}"
            + num(block["mean_nll_base"], 9)
            + num(block["mean_nll_matched"], 9)
            + num(block["mean_nll_sibling"], 9)
            + num(block["mean_nll_shuffled"], 10)
            + num(block["matched_minus_base"]["mean"], 9)
            + num(sib["mean"], 9)
            + num(block["matched_minus_shuffled"]["mean"], 9)
            + f"{sign:>8}"
        )

    lines.append(header)
    lines.append("-" * len(header))
    for split in EVAL_SPLITS:
        block = report["by_split"].get(split)
        if not block or not block["n_gens"]:
            lines.append(f"{split:<28}{0:>5}  (no gens)")
            continue
        lines.append(row(split, block))
        for cls in sorted(report["by_split_class"].get(split, {})):
            lines.append(row(f"  {cls}", report["by_split_class"][split][cls]))
    lines.append("-" * len(header))
    headline = report["headline"]
    lines.append(
        "HEADLINE matched-minus-sibling on val_seed: "
        + (
            "n/a (no val_seed gens)"
            if headline["mean"] is None
            else (
                f"{headline['mean']:+.4f} nats "
                f"({headline['n_negative']}/{headline['n']} trajectories negative). "
                "Negative = the signature carries basin-specific steering."
            )
        )
    )
    return "\n".join(lines)


def main() -> int:  # noqa: C901 — one linear procedure, kept readable by sections
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="B0 output dirs holding fork_series/*.npz",
    )
    parser.add_argument("--g-checkpoint", type=Path, required=True, help="train_g's g_state.pt")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b", help="anamnesis MODEL_PRESETS key (validation only)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON report")
    parser.add_argument("--limit", type=int, default=0, help="cap eval gens per split (0 = all)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--donor-pool",
        default="split",
        choices=["split", "all"],
        help="draw sibling/shuffled signatures from the gen's own split (default) or anywhere",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast"
    )
    parser.add_argument(
        "--model-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--max-seq-len", type=int, default=450)
    parser.add_argument("--lm-chunk", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch <= 0 or args.max_seq_len <= 0 or args.lm_chunk <= 0:
        logger.error("--batch/--max-seq-len/--lm-chunk must be > 0")
        return 2

    try:
        loaded = load_pairs(args.pairs_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load --pairs-dir %s: %s", args.pairs_dir, exc)
        return 2

    import torch


    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA host-side graph builds per unseen kv_len (forensics 2026-09-16)
    from pleroma import g_net
    from pleroma.train_g import batches, usable_pairs

    try:
        mapping = g_net.resolve_run_dirs(loaded.pairs, list(args.run_dirs))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    pool = [g_net.rebase(p, mapping) for p in loaded.pairs]
    kept, dropped = usable_pairs(pool, args.max_seq_len)
    for line in dropped:
        logger.error("dropped %s", line)
    row_of = {p.pair_key: p.row for p in loaded.pairs}
    # Donor pool: every pair that is loadable at all, including train pairs (only
    # reachable when --donor-pool all, but the filter has to see them to say so).
    donor_candidates = list(kept)

    eval_pairs: dict[str, list[PairRecord]] = {}
    for split in EVAL_SPLITS:
        rows = [p for p in kept if p.split == split]
        eval_pairs[split] = rows[: args.limit] if args.limit else rows
    n_eval = sum(len(v) for v in eval_pairs.values())
    if not n_eval:
        logger.error("no eval gens in %s; nothing to evaluate", EVAL_SPLITS)
        return 1
    logger.info("eval gens: %s", {k: len(v) for k, v in eval_pairs.items()})

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("--device %s but torch.cuda.is_available() is False", device)
        return 2

    # Preset is validation-only (dim policy: no dimension is taken from it).
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
            "evaluate across two feature spaces"
        )
        return 2
    inject_layers = [int(i) for i in arch["inject_layers"]]

    try:
        model = g_net.load_frozen_model(
            args.model_path, g_net.DTYPES[args.model_dtype], device
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("could not load the teacher: %s", exc)
        return 1

    hidden_dim = g_net.model_hidden_dim(model)
    n_model_layers = len(g_net.decoder_layers(model))
    if int(arch["hidden_dim"]) != hidden_dim:
        logger.error(
            "g emits %d-d biases but the teacher's hidden_dim is %d",
            arch["hidden_dim"], hidden_dim,
        )
        return 2
    out_of_range = [i for i in inject_layers if i >= n_model_layers]
    if out_of_range:
        logger.error("checkpoint injects at %s; the teacher has %d layers",
                     out_of_range, n_model_layers)
        return 2
    if preset_layers is not None and preset_layers != n_model_layers:
        logger.error(
            "preset %s says %d layers but %s has %d — refusing to run on a "
            "mismatched preset", args.preset, preset_layers, args.model_path, n_model_layers,
        )
        return 2

    g = g_net.SignatureToBias(
        in_dim=int(arch["in_dim"]),
        hidden=int(arch["hidden"]),
        n_layers=int(arch["n_layers"]),
        hidden_dim=hidden_dim,
    )
    g.load_state_dict(checkpoint["state_dict"])
    g.to(device=device, dtype=torch.float32).eval()
    logger.info(
        "g loaded from %s (best epoch %s), injecting at %s",
        args.g_checkpoint, checkpoint.get("best_epoch"), inject_layers,
    )

    pad_id = g_net.resolve_pad_token_id(model, args.model_path)
    started = time.time()
    failures: list[str] = list(dropped)
    per_gen: list[dict[str, Any]] = []

    with g_net.TrainableResidualInjector(model, inject_layers) as injector:
        for split in EVAL_SPLITS:
            pairs = eval_pairs[split]
            for chunk in batches(list(range(len(pairs))), args.batch):
                group = [pairs[i] for i in chunk]
                try:
                    rows = np.asarray([row_of[p.pair_key] for p in group], dtype=np.int64)
                    batch = g_net.build_batch(
                        group,
                        loaded.z[rows],
                        pad_token_id=pad_id,
                        max_seq_len=args.max_seq_len,
                        include_all_positions=False,
                    ).to(device)

                    donors: dict[str, list[int]] = {"sibling": [], "shuffled": []}
                    donor_meta: list[dict[str, Any]] = []
                    for pair in group:
                        sib, sib_pool, shuf, shuf_pool = pick_donors(
                            pair, donor_candidates, args.donor_pool
                        )
                        donors["sibling"].append(row_of[sib.pair_key])
                        donors["shuffled"].append(row_of[shuf.pair_key])
                        donor_meta.append(
                            {
                                "sibling_pair_key": sib.pair_key,
                                "sibling_seed_idx": sib.seed_idx,
                                "sibling_donor_pool": sib_pool,
                                "shuffled_pair_key": shuf.pair_key,
                                "shuffled_prompt_id": shuf.prompt_id,
                                "shuffled_donor_pool": shuf_pool,
                            }
                        )

                    arm_values: dict[str, list[float]] = {}
                    with torch.no_grad():
                        for arm in ARMS:
                            if arm == "base":
                                bias = None
                            else:
                                if arm == "matched":
                                    z_arm = batch.z
                                else:
                                    z_rows = np.asarray(donors[arm], dtype=np.int64)
                                    z_arm = torch.from_numpy(
                                        np.ascontiguousarray(loaded.z[z_rows])
                                    ).to(device)
                                bias = g(z_arm)
                            with g_net.autocast_context(device, bool(args.amp)):
                                result = g_net.run_arm(
                                    model, injector, batch, bias, 0.0, args.lm_chunk
                                )
                            arm_values[arm] = g_net.per_sample_means(
                                result.fork_nll, result.fork_rows, batch.size
                            )

                    for b, pair in enumerate(group):
                        per_gen.append(
                            {
                                "pair_key": pair.pair_key,
                                "generation_id": pair.generation_id,
                                "prompt_id": pair.prompt_id,
                                "prompt_class": pair.prompt_class,
                                "seed_idx": pair.seed_idx,
                                "split": pair.split,
                                "n_fork_positions": pair.n_fork_positions,
                                "fork_source": pair.fork_source,
                                "nll": {arm: arm_values[arm][b] for arm in ARMS},
                                **donor_meta[b],
                            }
                        )
                except (RuntimeError, ValueError, OSError, KeyError) as exc:
                    keys = [p.pair_key for p in group]
                    logger.exception("eval batch %s failed", keys)
                    failures.append(f"{keys}: {type(exc).__name__}: {exc}")
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            n_scored = sum(1 for r in per_gen if r["split"] == split)
            logger.info("%s: %d/%d gens scored", split, n_scored, len(pairs))

    if not per_gen:
        logger.error("no gen was scored; refusing to write an empty report")
        return 1

    by_split = {s: aggregate([r for r in per_gen if r["split"] == s]) for s in EVAL_SPLITS}
    by_split_class: dict[str, dict[str, Any]] = {}
    for split in EVAL_SPLITS:
        rows = [r for r in per_gen if r["split"] == split]
        by_split_class[split] = {
            cls: aggregate([r for r in rows if r["prompt_class"] == cls])
            for cls in sorted({r["prompt_class"] for r in rows})
        }

    report: dict[str, Any] = {
        "stage": "expB1_eval_g",
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "run_dirs": [str(d) for d in args.run_dirs],
            "inject_layers": inject_layers,
            "arms": list(ARMS),
        },
        "g_checkpoint": {
            "path": str(args.g_checkpoint),
            "best_epoch": checkpoint.get("best_epoch"),
            "arch": {k: v for k, v in arch.items()},
        },
        "pairs_dir": str(args.pairs_dir),
        "holdout": loaded.meta.get("holdout"),
        "n_gens_scored": len(per_gen),
        "by_split": by_split,
        "by_split_class": by_split_class,
        "by_class_pooled": {
            cls: aggregate([r for r in per_gen if r["prompt_class"] == cls])
            for cls in sorted({r["prompt_class"] for r in per_gen})
        },
        "headline": by_split["val_seed"]["matched_minus_sibling"],
        "headline_definition": (
            "mean over val_seed trajectories of (mean fork NLL under g(own signature) "
            "- mean fork NLL under g(a sibling's signature)); negative = the signature "
            "carries basin-specific steering. Per-gen paired values are in per_gen."
        ),
        "per_gen": per_gen,
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(format_table(report))
    logger.info(
        "DONE — %d gens, %d failures in %.0fs -> %s",
        len(per_gen), len(failures), time.time() - started, args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
