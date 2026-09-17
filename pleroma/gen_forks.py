"""Exp B0 stage 1: sample K forked continuations per bare prompt and bank the tokens.

The loomed futures. For each prompt we draw K independently-seeded continuations
from the *same* prompt, so the model — not a mode instruction — picks where each
trajectory goes. Bare prompts are the point: chat template, NO system prompt,
prompt text used verbatim as the user turn. Anything that tells the model how to
think would pin the basin we are trying to watch it choose.

Prompts come from ``pleroma/prompts_b0.json``, a CLASS-TAGGED battery, and
the class is an experimental variable (plan revised 2026-09-16). The old
"Write about: {topic}" battery was rejected because bare expository topics are
heavily-learned grooves — their mode diversity in the anamnesis corpus came from
the system prompts, not from sampling. ``convergent_control`` is retained as the
negative control: if it spreads as much as the fork classes, the spread measure is
reading noise, and stage 4 says so.

This phase does no extraction. There are no hooks, no capture surface and no
calibration here, so SDPA is safe (and the default); stage 2's replay is eager
regardless, and it is the replay that produces every number B0 reports. What this
script owes stage 2 is the FULL realized ``input_ids`` (prompt + continuation) per
gen, banked exactly, so the replay never has to re-tokenize or reconstruct g_0.

Sibling comparability is a hard invariant, not a hope: ``fork_tokens`` compares the
continuations of one ``prompt_id`` position-by-position, which is only meaningful
if every seed of that prompt saw a byte-identical prompt. Llama chat templates
render ``Today Date`` through ``strftime_now``, so a midnight rollover mid-run
would silently re-tokenize the prompt. We therefore check each gen's prompt ids
against the first seed of its prompt and FAIL the gen (loudly, into run_meta
failures) rather than bank an incomparable sibling. ``--date-string`` pins the
template date if a run has to straddle midnight.

Adapted from anamnesis ``scripts/run_gen_tokens.py`` (their canonical phase 1);
seed derivation mirrors ``extraction/generation_runner.make_seed``, keyed on the
prompt's string ``id`` so seeds survive edits to the battery.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.gen_forks \\
        --prompts-per-class 2 --seeds-per-prompt 16 \\
        --out-dir outputs/expB0_3b/gen --preset 3b \\
        --model-path <hf-id-or-local-dir>
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import: generation is GPU-bound, so one
# CPU thread costs nothing here but keeps the thread pools from exploding against
# whatever else shares the node (mirrors run_gen_tokens.py:25-31).
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
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB0.gen")

#: The class-tagged battery ships beside this script; --prompts-file overrides it.
DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parent / "prompts_b0.json"

#: Used when the battery file does not declare one. "{prompt}" = identity.
DEFAULT_USER_PROMPT_TEMPLATE = "{prompt}"


class PromptSpec(BaseModel):
    """One prompt in the battery. ``prompt_id`` is the stable key everywhere."""

    prompt_id: str = Field(min_length=1)
    prompt_class: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    prompt_idx: int = Field(ge=0)


class GenSpec(BaseModel):
    """One (prompt, seed) cell — the unit of work and of failure."""

    generation_id: int = Field(ge=0)
    prompt_id: str
    prompt_class: str
    prompt: str
    prompt_idx: int = Field(ge=0)
    seed_idx: int = Field(ge=0)
    seed: int = Field(ge=0)
    user_prompt: str


class SamplingParams(BaseModel):
    """Sampling knobs, banked per gen so a record is self-describing."""

    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_new_tokens: int = Field(gt=0)
    eos_ids: list[int]
    attn: Literal["sdpa", "eager"]
    date_string: str | None = None


class GenRecord(BaseModel):
    """What gets written to ``gen_records/gen_NNN.json``.

    ``input_ids`` is the FULL realized sequence (prompt + continuation); stage 2
    replays exactly this, so nothing downstream needs a tokenizer.
    """

    generation_id: int
    prompt_id: str
    prompt_class: str
    prompt: str
    prompt_idx: int
    seed_idx: int
    seed: int
    system_prompt: None = None
    user_prompt: str
    prompt_length: int = Field(gt=0)
    num_generated_tokens: int = Field(ge=0)
    generated_text: str
    input_ids: list[int]
    sampling: SamplingParams
    model_id: str
    preset: str


def make_seed(prompt_id: str, seed_idx: int) -> int:
    """Deterministic per-gen seed from the generation coordinates.

    Same construction as ``generation_runner.make_seed`` (sha256 of the coordinate
    string, first 8 hex digits) with a B0-specific prefix, keyed on the prompt's
    STRING id rather than its position: adding or reordering battery entries must
    not silently re-seed every other prompt's forks. Not Python's ``hash()``: that
    is salted per interpreter unless PYTHONHASHSEED is pinned, which would make the
    draw irreproducible across processes.
    """
    raw = f"expB0_{prompt_id}_{seed_idx}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def load_prompts(
    prompts_file: Path, prompts_per_class: int
) -> tuple[list[PromptSpec], str, dict[str, str]]:
    """Select the wave's prompts from the class-tagged battery.

    Takes the first ``prompts_per_class`` prompts of EACH class in file order (0 =
    every prompt), so the wave is balanced across classes by construction — class
    is the experimental variable, and an unbalanced wave would confound it with n.
    Returns (prompts, user_prompt_template, class descriptions).
    """
    blob: dict[str, Any] = json.loads(prompts_file.read_text())
    entries: list[dict[str, Any]] = blob.get("prompts", [])
    if not entries:
        raise ValueError(f"{prompts_file}: no 'prompts' list found")
    if blob.get("system_prompt"):
        raise ValueError(
            f"{prompts_file}: declares a system_prompt — B0 prompts are bare by "
            "design (a system prompt is what the revised plan removed)"
        )
    template = str(blob.get("user_prompt_template") or DEFAULT_USER_PROMPT_TEMPLATE)

    by_class: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        for key in ("id", "class", "prompt"):
            if not entry.get(key):
                raise ValueError(f"{prompts_file}: entry {i} is missing {key!r}")
        pid = str(entry["id"])
        if pid in seen_ids:
            raise ValueError(f"{prompts_file}: duplicate prompt id {pid!r}")
        seen_ids.add(pid)
        by_class.setdefault(str(entry["class"]), []).append(entry)

    selected: list[dict[str, Any]] = []
    for cls in by_class:  # dict preserves file order of first appearance
        pool = by_class[cls]
        if prompts_per_class > 0:
            if prompts_per_class > len(pool):
                raise ValueError(
                    f"--prompts-per-class {prompts_per_class} exceeds the "
                    f"{len(pool)} prompts in class {cls!r}"
                )
            pool = pool[:prompts_per_class]
        selected.extend(pool)

    prompts = [
        PromptSpec(
            prompt_id=str(e["id"]),
            prompt_class=str(e["class"]),
            prompt=str(e["prompt"]),
            prompt_idx=i,
        )
        for i, e in enumerate(selected)
    ]
    classes = {k: str(v) for k, v in (blob.get("classes") or {}).items()}
    return prompts, template, classes


def build_specs(
    prompts: list[PromptSpec], seeds_per_prompt: int, template: str,
    seed_offset: int = 0,
) -> list[GenSpec]:
    """Cross prompts with seeds. ``generation_id`` is stable under any later --limit.

    ``seed_offset`` shifts seed_idx (a replication wave over the same battery uses
    e.g. offset 16 for seed_idx 16..31 — the sha-derived RNG seeds are keyed on
    seed_idx, so an offset wave shares zero trajectories with the original).
    """
    if seeds_per_prompt <= 0:
        raise ValueError(f"--seeds-per-prompt must be > 0, got {seeds_per_prompt}")
    if seed_offset < 0:
        raise ValueError(f"--seed-offset must be >= 0, got {seed_offset}")
    return [
        GenSpec(
            generation_id=p.prompt_idx * seeds_per_prompt + s,
            prompt_id=p.prompt_id,
            prompt_class=p.prompt_class,
            prompt=p.prompt,
            prompt_idx=p.prompt_idx,
            seed_idx=seed_offset + s,
            seed=make_seed(p.prompt_id, seed_offset + s),
            user_prompt=template.format(prompt=p.prompt),
        )
        for p in prompts
        for s in range(seeds_per_prompt)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="class-tagged prompt battery (default: the one beside this script)",
    )
    parser.add_argument(
        "--prompts-per-class",
        type=int,
        default=2,
        help="prompts taken from EACH class, in file order (0 = all)",
    )
    parser.add_argument(
        "--seeds-per-prompt",
        "--seeds-per-topic",
        dest="seeds_per_prompt",
        type=int,
        default=16,
        help="K forks per prompt (--seeds-per-topic is the pre-battery alias)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="start seed_idx here (replication wave over the same battery, e.g. 16)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="longer paths give divergence room to compound (plan rev 2026-09-16)",
    )
    parser.add_argument("--preset", default="3b", help="MODEL_PRESETS key")
    parser.add_argument(
        "--model-path",
        default=None,
        help="local model dir; default = the preset's HF model_id",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "eager"],
        help="no capture happens here, so SDPA is safe; phase-2 replay is eager regardless",
    )
    parser.add_argument(
        "--eos-ids",
        type=int,
        nargs="+",
        default=None,
        help="default = the preset's eos_token_ids",
    )
    parser.add_argument(
        "--date-string",
        default=None,
        help=(
            "pin the chat template's Today Date (Llama renders it via strftime_now, "
            "so a midnight rollover mid-run would change the prompt tokens and break "
            "sibling comparability). Default None = template default."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap the number of gens (0 = all)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="regenerate even when a gen record already exists",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        logger.error("--temperature must be > 0 (do_sample=True), got %s", args.temperature)
        return 2
    if not 0 < args.top_p <= 1:
        logger.error("--top-p must be in (0, 1], got %s", args.top_p)
        return 2

    # Deferred so --help works on any machine, with or without the anamnesis env.
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA host-side graph builds per unseen kv_len (forensics 2026-09-16)
    torch.set_num_threads(1)
    from anamnesis.config import MODEL_PRESETS

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
        return 2
    preset = MODEL_PRESETS[args.preset]

    try:
        prompts, template, class_desc = load_prompts(
            args.prompts_file, args.prompts_per_class
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("could not load the prompt battery: %s", exc)
        return 2

    specs = build_specs(prompts, args.seeds_per_prompt, template, seed_offset=int(args.seed_offset))
    if args.limit:
        specs = specs[: args.limit]
    class_counts: dict[str, int] = {}
    for p in prompts:
        class_counts[p.prompt_class] = class_counts.get(p.prompt_class, 0) + 1
    logger.info(
        "pilot wave: %d prompts over %d classes x %d seeds = %d gens (%s)",
        len(prompts), len(class_counts), args.seeds_per_prompt, len(specs),
        ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
    )

    records_dir = args.out_dir / "gen_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    eos_ids: list[int] = list(args.eos_ids) if args.eos_ids else list(preset.eos_token_ids)
    if not eos_ids:
        logger.error("no eos ids (preset %r has none; pass --eos-ids)", args.preset)
        return 2
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        attn=args.attn,
        date_string=args.date_string,
    )

    model_path: str = args.model_path or preset.model_id
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(preset.torch_dtype), torch.float16)

    logger.info("loading %s (attn=%s, dtype=%s)", model_path, args.attn, preset.torch_dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.chat_template is None:
        logger.error(
            "%s has no chat template — B0's bare prompts are defined as the "
            "chat-templated user turn; refusing to silently fall back to raw text",
            model_path,
        )
        return 2
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation=args.attn
        )
        .to("cuda")
        .eval()
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]

    template_kwargs: dict[str, Any] = (
        {"date_string": args.date_string} if args.date_string else {}
    )

    # prompt_id -> the first seed's prompt ids; every sibling must match it exactly
    # or fork_tokens' positional comparison is comparing different prompts.
    prompt_ids_seen: dict[str, list[int]] = {}

    n_done = 0
    n_skipped = 0
    failures: list[str] = []
    written: list[dict[str, Any]] = []
    started = time.time()

    for i, spec in enumerate(specs):
        out_path = records_dir / f"gen_{spec.generation_id:03d}.json"
        if out_path.exists() and not args.no_resume:
            n_skipped += 1
            continue
        try:
            messages = [{"role": "user", "content": spec.user_prompt}]
            result = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
            input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
            input_ids = input_ids.to("cuda")
            prompt_length = int(input_ids.shape[1])
            prompt_list = [int(x) for x in input_ids[0].tolist()]

            first = prompt_ids_seen.setdefault(spec.prompt_id, prompt_list)
            if first != prompt_list:
                raise RuntimeError(
                    f"prompt tokens for prompt_id={spec.prompt_id!r} changed mid-run "
                    f"({len(first)} -> {len(prompt_list)} ids) — siblings would not be "
                    "positionally comparable. Pin --date-string and rerun."
                )

            torch.manual_seed(spec.seed)
            torch.cuda.manual_seed_all(spec.seed)
            np.random.seed(spec.seed % (2**32))

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=eos_ids,
                    pad_token_id=pad_id,
                )
            full_seq = [int(x) for x in out[0].tolist()]
            generated_ids = full_seq[prompt_length:]
            if len(generated_ids) < 2:
                # replay_extract needs N >= 2 for a single per-step transition; a
                # 0/1-token gen is dead weight downstream, so it fails here loudly.
                raise RuntimeError(
                    f"only {len(generated_ids)} generated token(s) — stage 2 replay "
                    "needs >= 2 for a per-step transition"
                )

            record = GenRecord(
                generation_id=spec.generation_id,
                prompt_id=spec.prompt_id,
                prompt_class=spec.prompt_class,
                prompt=spec.prompt,
                prompt_idx=spec.prompt_idx,
                seed_idx=spec.seed_idx,
                seed=spec.seed,
                user_prompt=spec.user_prompt,
                prompt_length=prompt_length,
                num_generated_tokens=len(generated_ids),
                generated_text=tok.decode(generated_ids, skip_special_tokens=True),
                input_ids=full_seq,
                sampling=sampling,
                model_id=model_path,
                preset=args.preset,
            )
            out_path.write_text(record.model_dump_json())
            written.append(
                {
                    "generation_id": record.generation_id,
                    "prompt_id": record.prompt_id,
                    "prompt_class": record.prompt_class,
                    "seed_idx": record.seed_idx,
                    "num_generated_tokens": record.num_generated_tokens,
                }
            )
            n_done += 1
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - started
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (len(specs) - i - 1) / rate if rate > 0 else 0.0
                logger.info(
                    "%d/%d gen_%03d (%s/%s seed %d): %d tok, %.0fs (%.2f gen/s, ETA %.0fs)",
                    i + 1, len(specs), spec.generation_id, spec.prompt_class,
                    spec.prompt_id, spec.seed_idx, record.num_generated_tokens,
                    elapsed, rate, eta,
                )
        except Exception as exc:  # noqa: BLE001 — a dead gen is recorded, never skipped
            logger.exception("gen_%03d failed", spec.generation_id)
            failures.append(f"gen_{spec.generation_id:03d}: {type(exc).__name__}: {exc}")

    if n_done == 0 and n_skipped == 0:
        logger.error("no gens succeeded; refusing to write an empty result")
        return 1

    (args.out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "stage": "expB0_gen_forks",
                "model_id": model_path,
                "preset": args.preset,
                "prompts_file": str(args.prompts_file),
                "prompts_per_class": int(args.prompts_per_class),
                "user_prompt_template": template,
                "class_descriptions": class_desc,
                "class_counts": class_counts,
                "prompts": [p.model_dump() for p in prompts],
                "seeds_per_prompt": int(args.seeds_per_prompt),
                "n_specs": len(specs),
                "n_written": n_done,
                "n_resumed": n_skipped,
                "sampling": sampling.model_dump(),
                "gens": written,
                "failures": failures,
                "elapsed_s": round(time.time() - started, 1),
            },
            indent=2,
        )
    )

    logger.info(
        "DONE — %d gens written (%d resumed, %d failed) in %.1fs -> %s",
        n_done, n_skipped, len(failures), time.time() - started, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
