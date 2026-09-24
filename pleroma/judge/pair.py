"""The blind dose-ladder PAIR judge (2AFC) over the raw Anthropic API.

The library parts (price table, spend meter, transport, parser, instruction
text) live in `pleroma.judge.pricing` / `.transport` / `.parse` / `.prompts`;
this module is the pair-judging procedure.

Judges are direct API calls, not agent-harness subagents (whose context
carries a harness system prompt and project instructions — uncontrolled
contamination): pinned model, default sampling (the Claude 5 API has no
temperature knob), an identical cached instruction prefix, and ONE PAIR PER
CALL — which structurally removes the cross-pair induction channel a
multi-pair packet opens (a judge can learn recurring probe slots within a
packet).

Reads a dose ladder's pair file (``--pairs``), shuffles letters PER PAIR
(seeded, independent of any packet shuffle), writes:
  <out-dir>/keys_api.json          the per-pair answer key (which letter was steered)
  <out-dir>/judge_<model>.json     {"judge": <model>, "judgments": [...]}

The API key is read from the environment (ANTHROPIC_API_KEY) and is never
logged, echoed, or written anywhere. Judge replies are requested as a single
bare JSON object per pair; the judge never sees a pair_id, dose, rep or probe
class — only the prompt and two replies.

Spend: every call's ACTUAL `usage` is charged to a `SpendMeter`
(`--max-dollars`, a hard stop set ABOVE the projection; resumable). This tool's
cached prefix (~260 tokens) is below every current model's minimum cacheable
prefix, so caching never engages and the full input rate is what is paid.

Usage (local; export ANTHROPIC_API_KEY first):
    python3 -m pleroma.judge.pair \\
        --pairs outputs/b15_dose/pairs.json \\
        --models claude-sonnet-5,claude-opus-5 \\
        --out-dir outputs/b15_dose/api_judging --seed 20260917
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from pleroma.judge.parse import parse_judgment
from pleroma.judge.pricing import PRICES_AS_OF, SpendMeter, price_of
from pleroma.judge.prompts import PAIR_2AFC_INSTRUCTIONS, instruction_sha256
from pleroma.judge.request import (EFFORTS, THINKING_SAFE_MAX_TOKENS, anthropic_body,
                                   warn_if_thinking_starved)
from pleroma.judge.transport import call_anthropic

logger = logging.getLogger("pleroma.judge.pair")


def build_pair_body(model: str, probe_text: str, reply_a: str, reply_b: str,
                    *, max_tokens: int = THINKING_SAFE_MAX_TOKENS,
                    effort: str | None = None) -> dict[str, Any]:
    """One pair, one call: the cached 2AFC instructions + the pair.

    ``effort`` (``output_config.effort``) is left None by DEFAULT — the API
    default (high), the same instrument the w5-r32 ladder ran; a band
    is only comparable to the band it replaces if the judge is. Pass it
    explicitly to trade comparability for cost, and say so in the write-up."""
    user = (
        f"PROMPT:\n{probe_text}\n\n"
        f"REPLY A:\n{reply_a}\n\n"
        f"REPLY B:\n{reply_b}"
    )
    return anthropic_body(model, PAIR_2AFC_INSTRUCTIONS, user,
                          max_tokens=max_tokens, effort=effort)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--models", default="claude-sonnet-5,claude-opus-5")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="cap pairs (0 = all)")
    parser.add_argument(
        "--max-dollars", type=float, default=0.0,
        help="HARD spend ceiling across all models in this invocation, from the "
             "responses' ACTUAL usage blocks (0 = no guard). Set it ABOVE your "
             "staged projection, not at it: a staged rate is a LOWER bound when a "
             "model's thinking varies with content (a staged deepseek rate has "
             "understated the real one by 3.1x). On firing, the run stops, keeps what it has, and "
             "says so; it is resumable.")
    parser.add_argument(
        "--max-tokens", type=int, default=THINKING_SAFE_MAX_TOKENS,
        help="per-call output ceiling. Adaptive thinking is billed inside it and "
             "a truncated reply parses to nothing, so open it generously.")
    parser.add_argument(
        "--effort", default=None,
        choices=list(EFFORTS),
        help="output_config.effort. Omitted = the API default (high) = the same "
             "instrument the w5-r32 ladder ran, which is what makes a new band "
             "comparable to the one it replaces. Lower it only as a deliberate "
             "cost/comparability trade, and record it.")
    parser.add_argument(
        "--spend-out", type=Path, default=None,
        help="write the spend receipt here (default: <out-dir>/spend.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:  # fail before the first dollar, not after the last
        price_of(m)
        warn_if_thinking_starved(m, args.max_tokens, logger)
    meter = SpendMeter(args.max_dollars)
    logger.info("spend guard: %s | prices as of %s | max_tokens %d | effort %s",
                f"${args.max_dollars:.2f}" if meter.enabled else "NONE (unmetered)",
                PRICES_AS_OF, args.max_tokens, args.effort or "API default (high)")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set — export it in this shell "
                     "first; it is never logged")
        return 2

    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.pairs:
        blob = json.loads(path.read_text())
        rows = blob["pairs"] if isinstance(blob, dict) else blob
        for p in rows:
            if p["pair_id"] not in seen:
                seen.add(p["pair_id"])
                pairs.append(p)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        logger.error("no pairs loaded")
        return 1
    logger.info("%d unique pairs from %d file(s)", len(pairs), len(args.pairs))

    # Per-pair letter shuffle, seeded on (seed, pair_id) — independent of any
    # packet shuffle ever used before.
    rng_of = lambda pid: random.Random(f"{args.seed}|{pid}")  # noqa: E731
    keys: dict[str, Any] = {}
    shuffled: list[tuple[dict[str, Any], str, str]] = []
    for p in pairs:
        steered_is_a = rng_of(p["pair_id"]).random() < 0.5
        a = p["reply_steered"] if steered_is_a else p["reply_base"]
        b = p["reply_base"] if steered_is_a else p["reply_steered"]
        keys[p["pair_id"]] = {
            "A": "steered" if steered_is_a else "base",
            "B": "base" if steered_is_a else "steered",
            "alpha": p["alpha"], "is_catch": p["alpha"] == 0,
            "probe_id": p["probe_id"], "probe_class": p.get("probe_class"),
            "rep": p.get("rep"), "lever_group": p.get("lever_group"),
        }
        shuffled.append((p, a, b))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys_path = args.out_dir / "keys_api.json"
    keys_path.write_text(json.dumps(
        {"stage": "expB15_judge_api", "letter_seed": args.seed,
         "n_pairs": len(keys), "keys": keys}, indent=1))
    logger.info("keys -> %s (sequester before reading judgments by eye)", keys_path)

    for model in models:
        if meter.should_stop():
            logger.error("SPEND GUARD already fired — skipping %s entirely rather "
                         "than starting a judge that cannot finish", model)
            break
        out_path = args.out_dir / f"judge_{model.replace('/', '_')}.json"
        done: dict[str, Any] = {}
        if out_path.exists():  # resume
            prior = json.loads(out_path.read_text())
            done = {j["pair_id"]: j for j in prior.get("judgments", [])}
            logger.info("%s: resuming, %d already judged", model, len(done))
        todo = [(p, a, b) for (p, a, b) in shuffled if p["pair_id"] not in done]
        failures: list[str] = []
        t0 = time.time()

        skipped = 0

        def one(item: tuple[dict[str, Any], str, str]) -> dict[str, Any] | None:
            nonlocal skipped
            p, a, b = item
            # ★ Check BEFORE spending. Once the guard has fired the pool drains
            # without issuing another billable call — the whole point is that the
            # cap is a ceiling on dollars, not on dollars-plus-whatever-was-in-flight.
            if meter.should_stop():
                skipped += 1
                return None
            try:
                text, usage = call_anthropic(
                    api_key, build_pair_body(model, p["probe_text"], a, b,
                                             max_tokens=args.max_tokens,
                                             effort=args.effort))
                if not meter.charge(model, usage):
                    logger.error(
                        "SPEND GUARD FIRED at $%.4f of $%.2f after %d calls "
                        "($%.6f/call) — stopping. Partial results are kept and "
                        "this run is resumable; report the projection error "
                        "rather than completing a partial matrix silently.",
                        meter.spent, meter.max_dollars, meter.calls,
                        meter.spent / max(meter.calls, 1))
                j = parse_judgment(text)
                j["pair_id"] = p["pair_id"]
                return j
            except Exception as exc:  # noqa: BLE001 — recorded, run continues
                failures.append(f"{p['pair_id']}: {type(exc).__name__}: {exc}")
                return None

        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, res in enumerate(pool.map(one, todo)):
                if res is not None:
                    done[res["pair_id"]] = res
                if (i + 1) % 40 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-9)
                    snap = meter.snapshot()
                    logger.info("%s: %d/%d (%.1f pairs/s) spent $%.4f "
                                "($%.6f/call, projecting $%.4f for %d)",
                                model, i + 1, len(todo), rate, snap["spent_usd"],
                                snap["usd_per_call"],
                                snap["usd_per_call"] * len(todo), len(todo))
                    out_path.write_text(json.dumps(
                        {"judge": model, "judgments": list(done.values())}, indent=1))

        snap = meter.snapshot()
        out_path.write_text(json.dumps(
            {"judge": model, "judgments": list(done.values()),
             "failures": failures, "spend": snap,
             "max_tokens": args.max_tokens, "effort": args.effort,
             "instructions_sha256": instruction_sha256(PAIR_2AFC_INSTRUCTIONS)},
            indent=1))
        logger.info("%s: DONE — %d judged, %d failures, %d skipped-by-guard in "
                    "%.0fs, $%.4f spent -> %s",
                    model, len(done), len(failures), skipped,
                    time.time() - t0, snap["spent_usd"], out_path)
        if failures:
            logger.warning("%s failures (first 3): %s", model, failures[:3])
        if meter.should_stop():
            logger.error("%s: STOPPED BY THE SPEND GUARD with %d pair(s) unjudged. "
                         "The remainder is NOT a random sample of the matrix — check "
                         "it for arm skew before reading any rung.", model, skipped)

    spend_path = args.spend_out or (args.out_dir / "spend.json")
    snap = meter.snapshot()
    snap["models"] = models
    snap["n_pairs_offered"] = len(pairs)
    spend_path.write_text(json.dumps(snap, indent=1) + "\n")
    logger.info("SPEND: $%.4f over %d calls ($%.6f/call) -> %s",
                snap["spent_usd"], snap["calls"], snap["usd_per_call"], spend_path)
    return 3 if meter.should_stop() else 0


if __name__ == "__main__":
    sys.exit(main())
