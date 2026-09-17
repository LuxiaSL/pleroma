"""Standalone /loom latency benchmark — MEASURE first, then optimize, then MEASURE again.

Exercises exactly the work ``loom_serve.do_loom`` does for one ``/loom`` call
(K future draws from a shared prefix, then the harvest: v3 signatures +
bins-B features), timed per stage, WITHOUT going through the HTTP server —
this is the standalone instrument BENCH-LOOM-2026-09-17.md's numbers come
from. It shares no code with loom_serve.py beyond the tiny pure-Python
``loom_turn_seed`` (so the seeds it draws are reproducible against a real
server) and ``pleroma.duet_serve``'s message-building helpers.

Two independent knobs, matching the two optimizations in PLAN order:

    --gen-mode {sequential,batched}   K sequential model.generate() calls vs.
                                      ONE batched call (batch size K, shared
                                      prefix, one seed for the whole batch).
    --harvest-mode {subprocess,worker,none}
                                      the original two-cold-subprocess
                                      pipeline vs. one POST to a running
                                      harvest_worker.py vs. skip (generation
                                      timing only).

--repeats > 1 with --harvest-mode worker shows cold-vs-warm: the FIRST call
after the worker starts still pays nothing extra (the worker already paid
its model-load cost at ITS OWN startup, before this script ever ran), so
repeats mainly show run-to-run variance, not a warmup cliff — recorded
anyway because that is itself a claim worth checking, not assuming.

Uses an existing session-STYLE multi-turn context (hardcoded, not the live
server's real session data — this script never touches
outputs/loom/sessions, which belongs to the
live loom-v0 job) so the measured prefix length matches what /loom actually
re-prefills: several turns of chat, not a bare one-line prompt.

Usage (from the repo root, inside the project venv):
    python -m pleroma.bench_loom \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --calib-dir data/calibration/3b \\
        --discriminants factor_directions_3b.npz \\
        --work-dir bench_work --gen-mode sequential --harvest-mode subprocess \\
        --k 6 --horizon 192 --out logs/bench_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("bench_loom")

# A synthetic multi-turn history: session-STYLE context, not a bare prompt.
# Long enough (several hundred tokens once templated) that the re-prefill
# cost and the bins-B prompt-length assumptions (n_bins=20 needs
# prompt_length >= 20, trivially satisfied) both look like a real session.
BENCH_HISTORY: list[dict[str, str]] = [
    {"role": "user", "content": (
        "I've been thinking about how a model's own sampled continuations "
        "could be read back as a signal about what it's currently leaning "
        "toward saying. Does that make sense as a research direction?"
    )},
    {"role": "assistant", "content": (
        "Yes — the idea is that if you sample several continuations from the "
        "same prefix, the SET of continuations carries information about the "
        "model's current disposition that any single continuation doesn't. "
        "You could extract features from that set (attention patterns, "
        "hidden-state trajectories, lexical choices) and use them as a "
        "signature of 'where the model currently is' rather than only 'what "
        "it just said'. The interesting question is whether that signature "
        "is stable enough, and separable enough across different leanings, "
        "to be useful as a steering signal rather than just a description."
    )},
    {"role": "user", "content": (
        "Right, and then the follow-up idea is: what if you don't just read "
        "the signature, but push the model TOWARD one of the futures it "
        "already sampled — using the signature as a direction to bias the "
        "residual stream on subsequent turns. Does that closed loop seem "
        "sound, or is there an obvious way it goes wrong?"
    )},
    {"role": "assistant", "content": (
        "The main way it goes wrong is off-distribution transfer: whatever "
        "signature-to-bias map you fit was almost certainly trained on "
        "continuations of BARE single prompts, not multi-turn chat context, "
        "so applying it inside a running conversation is asking the map to "
        "generalize somewhere it was never fit. That doesn't make it "
        "meaningless, but it means the honest framing is 'exploratory "
        "instrument', not 'validated readout' — you'd want a held-out check "
        "that compares this against the same experiment run on bare prompts "
        "before trusting the multi-turn numbers on their own."
    )},
]
BENCH_NEXT_TURN = (
    "Given all that, if I wanted to test this on a small instruct model "
    "with a handful of sampled futures per turn, what would the smallest "
    "useful experiment look like, and what would falsify the idea?"
)


def loom_turn_seed(seed: int, session: str, branch: str, turn: int, draw: int = 0) -> int:
    """Mirrors loom_serve.loom_turn_seed exactly (duplicated, not imported,
    so this benchmark stays import-light and never needs torch to compute a
    seed)."""
    raw = f"loomv0_{seed}_{session}_{branch}_{turn}_{draw}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def build_env_cmds(py: str, repo: Path, loom_dir: Path, args: argparse.Namespace) -> list[list[str]]:
    """The exact two subprocess commands loom_serve.do_loom() runs today."""
    return [
        [py, "-m", "pleroma.run_replay_b0", "--gen-dir", str(loom_dir),
         "--out-dir", str(loom_dir), "--preset", str(args.preset),
         "--model-path", str(args.model_path), "--calib-dir", str(args.calib_dir),
         "--discriminants", str(args.discriminants),
         "--kvrot-path", str(args.kvrot_path), "--save-raw", "none"],
        [py, "-m", "pleroma.extract_bins", "--gen-dirs", str(loom_dir),
         "--model-path", str(args.model_path), "--preset", str(args.preset),
         "--out", str(loom_dir / "bins.npz")],
    ]


def main() -> int:  # noqa: C901 — one linear benchmark procedure
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--discriminants", required=True)
    parser.add_argument("--kvrot-path", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--session", default="bench-session")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gen-mode", choices=["sequential", "batched"], default="sequential")
    parser.add_argument("--harvest-mode", choices=["subprocess", "worker", "none"],
                        default="subprocess")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8768")
    parser.add_argument("--worker-connect-timeout", type=float, default=3.0)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None, help="append JSON results here")
    args = parser.parse_args()

    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # MANDATORY on this stack — never remove
    torch.set_num_threads(1)
    from anamnesis.config import MODEL_PRESETS
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pleroma import g_net
    from pleroma.duet_serve import build_messages

    preset = MODEL_PRESETS[args.preset]
    eos_ids = list(preset.eos_token_ids)
    device = str(args.device)
    dtype = g_net.DTYPES[str(args.model_dtype)]

    t_load0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(device).eval().requires_grad_(False)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
    load_s = time.time() - t_load0
    logger.info("model loaded in %.1fs (this cost is what harvest-mode=worker removes "
                "from the PER-CALL path — the worker pays it once, at ITS startup)", load_s)

    messages = build_messages(BENCH_HISTORY, BENCH_NEXT_TURN, None)
    result = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = (result if isinstance(result, torch.Tensor) else result["input_ids"]).to(device)
    plen = int(ids.shape[1])
    logger.info("prompt_length=%d tokens (session-style, %d prior turns)",
                plen, len(BENCH_HISTORY))

    py = sys.executable
    repo = Path(__file__).resolve().parent.parent

    results: list[dict[str, Any]] = []
    for rep in range(int(args.repeats)):
        sess_tag = hashlib.sha256(f"{args.session}-{rep}".encode()).hexdigest()[:10]
        loom_dir = args.work_dir / sess_tag / "loom_000"
        rec_dir = loom_dir / "gen_records"
        rec_dir.mkdir(parents=True, exist_ok=True)

        # ── stage 1: K futures ───────────────────────────────────────────
        t0 = time.time()
        futures: list[dict[str, Any]] = []
        if args.gen_mode == "sequential":
            for j in range(int(args.k)):
                seed = loom_turn_seed(args.seed, args.session, "loom", 0, draw=j + 1)
                torch.manual_seed(seed)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed % (2**32))
                with torch.no_grad():
                    out = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=int(args.horizon), do_sample=True,
                        temperature=float(args.temperature), top_p=float(args.top_p),
                        eos_token_id=eos_ids, pad_token_id=pad_id,
                    )
                full = [int(x) for x in out[0].tolist()]
                gen = full[plen:]
                text = tok.decode(gen, skip_special_tokens=True).strip()
                futures.append({"index": j, "seed": seed, "full": full, "gen": gen, "text": text})
        else:  # batched — ONE generate call, batch size k, ONE seed for the whole batch
            seed = loom_turn_seed(args.seed, args.session, "loom", 0, draw=0)
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))
            batch_ids = ids.repeat(int(args.k), 1)
            with torch.no_grad():
                out = model.generate(
                    batch_ids, attention_mask=torch.ones_like(batch_ids),
                    max_new_tokens=int(args.horizon), do_sample=True,
                    temperature=float(args.temperature), top_p=float(args.top_p),
                    eos_token_id=eos_ids, pad_token_id=pad_id,
                )
            eos_set = set(eos_ids)
            for j in range(int(args.k)):
                row = [int(x) for x in out[j].tolist()]
                gen_full = row[plen:]
                # Trim to the first eos, inclusive — matches what a batch-of-1
                # generate() would have produced (the loop stops right after
                # eos and never pads); a batched row that finished early is
                # otherwise followed by pad_token_id filler that never
                # existed in the sequential path and must not enter the
                # harvest (see BENCH-LOOM report, "determinism" section).
                stop = next((t for t, tid in enumerate(gen_full) if tid in eos_set), None)
                gen = gen_full[: stop + 1] if stop is not None else gen_full
                full = row[:plen] + gen
                text = tok.decode(gen, skip_special_tokens=True).strip()
                futures.append({"index": j, "seed": seed, "full": full, "gen": gen, "text": text})
        gen_s = time.time() - t0

        for f in futures:
            record = {
                "generation_id": f["index"], "prompt_id": "loom000",
                "prompt_class": "loom_context", "prompt": "<chat context>",
                "prompt_idx": 0, "seed_idx": f["index"], "seed": f["seed"],
                "system_prompt": None, "user_prompt": "<chat context>",
                "prompt_length": plen, "num_generated_tokens": len(f["gen"]),
                "generated_text": f["text"], "input_ids": f["full"],
                "sampling": {"temperature": float(args.temperature),
                             "top_p": float(args.top_p),
                             "max_new_tokens": int(args.horizon), "eos_ids": eos_ids,
                             "attn_implementation": "sdpa"},
                "model_id": str(args.model_path), "preset": str(args.preset),
            }
            (rec_dir / f"gen_{f['index']:03d}.json").write_text(json.dumps(record))

        # ── stage 2: harvest ────────────────────────────────────────────
        harvest_s = 0.0
        replay_s = bins_s = None
        harvest_error = None
        if args.harvest_mode == "subprocess":
            t1 = time.time()
            cmds = build_env_cmds(py, repo, loom_dir, args)
            stage_times = []
            for cmd in cmds:
                ts = time.time()
                proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
                stage_times.append(time.time() - ts)
                if proc.returncode != 0:
                    tail = (proc.stdout + proc.stderr)[-1500:]
                    harvest_error = f"harvest step failed ({cmd[2]}): ...{tail}"
                    logger.error(harvest_error)
                    break
            harvest_s = time.time() - t1
            if len(stage_times) >= 1:
                replay_s = round(stage_times[0], 3)
            if len(stage_times) >= 2:
                bins_s = round(stage_times[1], 3)
        elif args.harvest_mode == "worker":
            t1 = time.time()
            try:
                req = urllib.request.Request(
                    args.worker_url.rstrip("/") + "/harvest",
                    data=json.dumps({"loom_dir": str(loom_dir)}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=args.worker_connect_timeout + args.worker_timeout
                ) as resp:
                    blob = json.loads(resp.read())
                if not blob.get("ok"):
                    harvest_error = f"worker reported failure: {blob}"
                else:
                    replay_s = blob.get("replay_s")
                    bins_s = blob.get("bins_s")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                harvest_error = f"worker unreachable: {type(exc).__name__}: {exc}"
                logger.error(harvest_error)
            harvest_s = time.time() - t1
        else:
            logger.info("harvest-mode=none — skipping harvest stage")

        total_s = gen_s + harvest_s
        rec = {
            "rep": rep, "gen_mode": args.gen_mode, "harvest_mode": args.harvest_mode,
            "k": int(args.k), "horizon": int(args.horizon), "prompt_length": plen,
            "model_load_s": round(load_s, 2),
            "gen_s": round(gen_s, 3), "harvest_s": round(harvest_s, 3),
            "replay_s": replay_s, "bins_s": bins_s,
            "total_s_excl_model_load": round(total_s, 3),
            "harvest_error": harvest_error, "loom_dir": str(loom_dir),
        }
        logger.info("rep %d: gen=%.1fs harvest=%.1fs (replay=%s bins=%s) total=%.1fs",
                    rep, gen_s, harvest_s, replay_s, bins_s, total_s)
        results.append(rec)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if args.out.exists():
            try:
                existing = json.loads(args.out.read_text())
            except json.JSONDecodeError:
                existing = []
        existing.extend(results)
        args.out.write_text(json.dumps(existing, indent=2))
        logger.info("wrote %d result(s) -> %s", len(results), args.out)

    return 0 if all(r["harvest_error"] is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
