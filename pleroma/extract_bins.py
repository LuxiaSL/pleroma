"""bins-B extraction — the span-coverage coordinate class, pleroma-minimal.

Implements resolution B of the anamnesis span-coverage SPEC (ANSWER ferry,
2026-09-16, Luxia's sizing call): per generated token, attention mass into
N_BINS equal relative-position bins over the PROMPT, at a sampled set of
layers, with a 3-way head summary per bin — mean over heads, max over heads,
and the entropy of the per-bin head distribution (head *dispersion* without
head identity). Pooled over generated tokens by mean. 20 x 7 x 3 = 420 dims
at the defaults. Resolutions A (head-pooled) and C (per-head) are one config
change away, per the SPEC's own instruction — this script takes them as flags.

Scope note: this is pleroma's build, only as far as the amplifier needs
(scoping ruling in the ANSWER ferry). The standing-suite integration is
anamnesis's on its own schedule. Features are emitted RAW with family-tagged
names; standardization happens at the regression join, recorded there.

Conventions that are load-bearing:
- Eager attention is REQUIRED (SDPA exposes no weights); the cuDNN-SDPA fix
  is applied anyway as standing policy for any generate/forward on this stack.
- Bin density = (mass into bin) / (bin token count / prompt length): mass
  normalized by relative bin width, so a uniform-over-prompt row scores ~1 in
  every bin whatever the prompt length.
- Absence RAISES, never zero-fills: a gen whose prompt is shorter than the
  bin count, or with zero generated tokens, is a hard error naming the gen.
- Teacher-forced single forward per gen over the SAVED input_ids — no
  resampling, so the features describe exactly the trajectory the signature
  and the lever describe.

Usage (node):
    python -m pleroma.extract_bins \\
        --gen-dirs outputs/wave2_gen outputs/w3_gen_s0 \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --out outputs/bins/binsB_wave2_w3s0.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("extract_bins")

DEFAULT_LAYERS = "4,8,12,16,20,24,27"
DEFAULT_BINS = 20
SUMMARIES = ("hmean", "hmax", "hent")  # resolution B's per-bin head summary


@dataclass(frozen=True)
class GenRecord:
    """One saved trajectory, exactly as gen_forks wrote it."""

    run_dir: str
    generation_id: int
    prompt_id: str
    prompt_class: str
    seed_idx: int
    prompt_length: int
    input_ids: list[int]

    @staticmethod
    def load(run_dir: Path, path: Path) -> "GenRecord":
        d = json.loads(path.read_text())
        missing = [k for k in ("generation_id", "prompt_id", "prompt_class", "seed_idx",
                               "prompt_length", "input_ids") if k not in d]
        if missing:
            raise KeyError(f"{path}: gen record missing {missing}")
        return GenRecord(
            run_dir=str(run_dir), generation_id=int(d["generation_id"]),
            prompt_id=str(d["prompt_id"]), prompt_class=str(d["prompt_class"]),
            seed_idx=int(d["seed_idx"]), prompt_length=int(d["prompt_length"]),
            input_ids=[int(x) for x in d["input_ids"]],
        )


def bin_edges(prompt_length: int, n_bins: int) -> NDArray[np.int64]:
    """Token index edges of n_bins equal RELATIVE bins over [0, prompt_length).

    Equal in relative position; integer token counts differ by at most one.
    Raises when the prompt cannot give every bin at least one token — a
    zero-token bin would need a zero-fill, and absence raises here.
    """
    if prompt_length < n_bins:
        raise ValueError(
            f"prompt_length {prompt_length} < n_bins {n_bins} — cannot bin "
            "without empty bins (absence raises, never zero-fills)"
        )
    edges = np.floor(np.linspace(0, prompt_length, n_bins + 1)).astype(np.int64)
    if (np.diff(edges) <= 0).any():
        raise ValueError(f"degenerate bin edges for prompt_length {prompt_length}")
    return edges


TEMPORAL_SUMMARIES = ("tvrate", "qvrate")  # roughness of the bin-density path


def feature_names(
    family: str, layers: list[int], n_bins: int, summaries: tuple[str, ...]
) -> list[str]:
    """Family-tagged names, one per output dim, layer-major then bin then summary."""
    return [
        f"{family}|L{layer:02d}|bin{b:02d}|{s}"
        for layer in layers for b in range(n_bins) for s in summaries
    ]


def summarize_heads(bin_mass: NDArray[np.floating]) -> NDArray[np.floating]:
    """[heads, bins] per-head bin densities -> [bins, 3] (hmean, hmax, hent).

    hent is the entropy of the head distribution WITHIN a bin (which heads
    supply this bin's mass), normalized by log(n_heads) to [0, 1] — the "head
    dispersion without head identity" summary resolution B keeps.
    """
    n_heads = bin_mass.shape[0]
    hmean = bin_mass.mean(axis=0)
    hmax = bin_mass.max(axis=0)
    totals = bin_mass.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("a bin received zero mass from every head")
    p = bin_mass / totals[None, :]
    hent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=0) / np.log(n_heads)
    return np.stack([hmean, hmax, hent], axis=1)


def extract_one(
    rec: "GenRecord",
    model: Any,
    device: str,
    layers: list[int],
    n_bins: int,
    resolution: str,
    summaries: tuple[str, ...],
    temporal: bool,
) -> NDArray[np.float32]:
    """One gen's bins-B feature vector. Pulled out of ``main()`` (2026-09-17,
    loom latency work) so a persistent process (``harvest_worker.py``) can call
    the EXACT same per-gen numerics ``main()``'s loop calls, instead of a second
    implementation that could drift from this one. Raises on any of the same
    conditions the inline loop used to (short prompt, zero generated tokens,
    a bin with zero mass, a non-finite feature) — the caller decides whether
    that becomes a per-gen failure (as ``main()`` does) or an abort.
    """
    import torch  # deferred at module scope too; safe to re-import here

    edges = bin_edges(rec.prompt_length, int(n_bins))
    if len(rec.input_ids) <= rec.prompt_length + 1:
        raise ValueError(
            f"gen {rec.generation_id} ({rec.prompt_id}) has "
            f"{len(rec.input_ids) - rec.prompt_length} generated token(s)"
        )
    ids = torch.tensor([rec.input_ids], dtype=torch.long, device=str(device))
    with torch.no_grad():
        out = model(ids, output_attentions=True, use_cache=False)
    per_layer: list[NDArray[np.floating]] = []
    for layer in layers:
        # [heads, gen_rows, keys] — rows are the generated positions.
        attn = out.attentions[layer][0, :, rec.prompt_length:, :].float()
        attn = attn.cpu().numpy()
        # Mass into each prompt bin, density-normalized by relative width.
        widths = np.diff(edges).astype(np.float64) / rec.prompt_length
        mass = np.add.reduceat(
            attn[:, :, : rec.prompt_length], edges[:-1], axis=2
        )  # [heads, rows, bins]
        density = mass / widths[None, None, :]
        pooled = density.mean(axis=1)  # mean over generated tokens -> [heads, bins]
        if resolution == "A":
            block = pooled.mean(axis=0)[:, None]  # [bins, 1]
        else:
            block = summarize_heads(pooled)  # [bins, 3]
        if temporal:
            # Roughness of the head-mean bin-density PATH over generated
            # tokens — the axis the mean-pool above deliberately drops.
            # Per-step rates so path length divides out.
            path = density.mean(axis=0)  # [rows, bins]
            if path.shape[0] < 2:
                raise ValueError("temporal summaries need >= 2 generated tokens")
            steps = np.diff(path, axis=0)  # [rows-1, bins]
            tv = np.abs(steps).mean(axis=0)  # [bins]
            qv = (steps ** 2).mean(axis=0)  # [bins]
            block = np.concatenate([block, tv[:, None], qv[:, None]], axis=1)
        per_layer.append(block)
    del out
    vec = np.concatenate([x.reshape(-1) for x in per_layer])
    if not np.isfinite(vec).all():
        raise ValueError("non-finite feature value")
    return vec.astype(np.float32)


def load_bins_model(model_path: str, model_dtype: str, device: str) -> Any:
    """Load the eager-attention bins model exactly as ``main()`` does.

    Pulled out so ``harvest_worker.py`` loads this model ONCE and reuses this
    identical loading path (dtype/attn_implementation kwargs and the
    dtype=/torch_dtype= fallback both matter for byte-for-byte reproducing
    what ``main()`` would have loaded).
    """
    import torch
    from transformers import AutoModelForCausalLM

    from pleroma import g_net

    dtype = g_net.DTYPES[str(model_dtype)]
    common = {"attn_implementation": "eager", "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **common)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **common)
    model.to(str(device))
    model.eval()
    model.requires_grad_(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-dirs", type=Path, nargs="+", required=True,
                        help="gen_forks run dirs (each containing gen_records/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--layers", default=DEFAULT_LAYERS,
                        help="attention layer indices to sample (comma list)")
    parser.add_argument("--n-bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--resolution", choices=["A", "B"], default="B",
                        help="A = head-pooled mean only; B = mean/max/entropy over heads")
    parser.add_argument(
        "--temporal", action="store_true",
        help="ALSO emit per-(layer,bin) roughness of the head-mean bin-density "
        "path over generated tokens: total-variation rate (p=1) and quadratic-"
        "variation rate (p=2), both per-step so path length divides out. The "
        "order-sensitive coordinates the mean-pool discards (2026-09-17, "
        "Luxia's iterated-integral question).",
    )
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="cap gens (0 = all)")
    args = parser.parse_args()

    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    if len(set(layers)) != len(layers) or not layers:
        logger.error("--layers must be a non-empty list of distinct ints, got %r", args.layers)
        return 2
    if args.n_bins < 2:
        logger.error("--n-bins must be >= 2, got %d", args.n_bins)
        return 2

    records: list[GenRecord] = []
    for run_dir in args.gen_dirs:
        rec_dir = run_dir / "gen_records"
        if not rec_dir.is_dir():
            logger.error("%s has no gen_records/ — is it a gen_forks run dir?", run_dir)
            return 2
        paths = sorted(rec_dir.glob("gen_*.json"))
        if not paths:
            logger.error("%s/gen_records is empty", run_dir)
            return 2
        for p in paths:
            records.append(GenRecord.load(run_dir, p))
    if args.limit:
        records = records[: int(args.limit)]
    logger.info("%d gens from %d run dir(s)", len(records), len(args.gen_dirs))

    # Deferred imports so --help works anywhere.
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # standing policy on this stack
    torch.set_num_threads(1)
    from anamnesis.config import MODEL_PRESETS

    from pleroma import g_net

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
        return 2
    dtype = g_net.DTYPES[str(args.model_dtype)]
    logger.info("loading %s (EAGER attention — required for weights, %s)",
                args.model_path, dtype)
    model = load_bins_model(args.model_path, str(args.model_dtype), str(args.device))
    n_layers = len(g_net.decoder_layers(model))
    bad = [x for x in layers if not 0 <= x < n_layers]
    if bad:
        logger.error("--layers %s out of range for a %d-layer model", bad, n_layers)
        return 2

    base_summaries: tuple[str, ...] = SUMMARIES if args.resolution == "B" else ("hmean",)
    summaries: tuple[str, ...] = (
        base_summaries + TEMPORAL_SUMMARIES if args.temporal else base_summaries
    )
    dim = len(layers) * int(args.n_bins) * len(summaries)
    features = np.zeros((len(records), dim), dtype=np.float32)
    failures: list[str] = []
    started = time.time()

    for i, rec in enumerate(records):
        try:
            features[i] = extract_one(
                rec, model, str(args.device), layers, int(args.n_bins),
                str(args.resolution), summaries, bool(args.temporal),
            )
        except Exception as exc:  # noqa: BLE001 — recorded per gen, run continues
            failures.append(
                f"{rec.run_dir}/gen_{rec.generation_id:03d} ({rec.prompt_id}): "
                f"{type(exc).__name__}: {exc}"
            )
            features[i] = np.nan  # poisoned, never silently zero
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            rate = (i + 1) / max(time.time() - started, 1e-9)
            logger.info("%d/%d gens (%.1f/s)", i + 1, len(records), rate)

    if failures:
        logger.warning("%d gen(s) FAILED (features poisoned with NaN): %s",
                       len(failures), failures[:3])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        feature_names=np.array(feature_names(
            f"bins{args.resolution}", layers, int(args.n_bins), summaries
        )),
        run_dir=np.array([r.run_dir for r in records]),
        generation_id=np.array([r.generation_id for r in records], dtype=np.int64),
        prompt_id=np.array([r.prompt_id for r in records]),
        prompt_class=np.array([r.prompt_class for r in records]),
        seed_idx=np.array([r.seed_idx for r in records], dtype=np.int64),
        config=np.array(json.dumps({
            "family": f"bins{args.resolution}",
            "n_bins": int(args.n_bins), "layers": layers,
            "head_summaries": list(summaries),
            "temporal": bool(args.temporal),
            "pooling": "mean over generated tokens; tvrate/qvrate are per-step "
                       "roughness of the head-mean bin path over those tokens",
            "density_norm": "mass / relative bin width",
            "model_path": str(args.model_path), "preset": str(args.preset),
            "model_dtype": str(args.model_dtype),
            "attn_implementation": "eager",
            "standardization": "NONE here — raw densities; standardize at the join",
            "n_failures": len(failures),
        })),
        failures=np.array(failures),
    )
    logger.info("DONE — %s [%d gens x %d dims], %d failures in %.0fs",
                args.out, len(records), dim, len(failures), time.time() - started)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
