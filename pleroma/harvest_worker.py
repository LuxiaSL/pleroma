"""LOOM v0 harvest worker — the persistent process behind /loom's 60s->10s win.

/loom's harvest stage used to be two cold subprocess launches per call:
``run_replay_b0`` (v3 signatures — loads the model with the full capture
surface, ~20-40s cold) and ``extract_bins`` (bins-B features — loads a SECOND
eager-attention copy of the model, ~20-30s cold). Both models get reloaded on
every single /loom call. This worker loads both ONCE, at startup, and then
serves harvest requests over loopback HTTP in a few seconds each.

**Numerics are untouched.** Every per-gen computation here calls the exact
functions the two frozen scripts already export:

    run_replay_b0.replay_extract / compute_features_v2_from_data /
        save_features / fork_series_arrays / check_gate0   (frozen — not
        rewritten; imported and called, same as the subprocess did)
    extract_bins.extract_one                                 (pulled out of
        extract_bins.main()'s loop in this same change, output-preserving)

Only the PLUMBING changed: one long-lived process instead of two cold ones
per call. The on-disk contract is identical too — the same
signatures/, fork_series/, cells.json, run_meta.json, and bins.npz land in
the loom dir, so a downstream reader (fit_loom_map / LoomMap.lever_of) cannot
tell which path produced them.

Protocol: stdlib http.server, loopback only (the host may be shared — see
``require_loopback``), single-threaded (HTTPServer, not Threading — GPU work
here must serialize, and the caller only ever has one harvest in flight per
/loom call anyway).

    GET  /health                      -> {"ok": true, "preset": ..., ...}
    POST /harvest {"loom_dir": "..."} -> {"ok": true, "replay_s":..,
                                           "bins_s":.., "n_gens":..,
                                           "failures_replay":[...],
                                           "failures_bins":[...]}
                                       -> {"ok": false, "error": "..."} (never
                                          a bare 500 with no body — loom_serve
                                          needs a body to decide fallback)

loom_serve.py's do_loom() tries this worker first (short connect timeout) and
falls back to the original two-subprocess pipeline, unchanged, if the worker
is unreachable or reports an error. This process is therefore OPTIONAL — the
server works exactly as before if you never start it.

Usage (from the repo root, inside the project venv):
    python -m pleroma.harvest_worker \\
        --preset 3b --model-path <hf-id-or-local-dir> \\
        --calib-dir data/calibration/3b \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --host 127.0.0.1 --port 8768
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("harvest_worker")


def main() -> int:  # noqa: C901 — one linear procedure, sectioned like loom_serve
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--discriminants", type=Path, required=True)
    parser.add_argument("--kvrot-path", default=None)
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logits-top-k", type=int, default=50,
                        help="matches run_replay_b0's --logits-top-k default")
    parser.add_argument("--bins-layers", default=None,
                        help="matches extract_bins --layers; default = extract_bins.DEFAULT_LAYERS")
    parser.add_argument("--bins-n-bins", type=int, default=None,
                        help="matches extract_bins --n-bins; default = extract_bins.DEFAULT_BINS")
    parser.add_argument("--bins-resolution", choices=["A", "B"], default="B")
    parser.add_argument("--bins-temporal", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    from pleroma.duet_serve import require_loopback

    host = require_loopback(args.host)

    # Pin CPU thread pools BEFORE numpy/torch import, matching run_replay_b0.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # MANDATORY on this stack — never remove
    torch.set_num_threads(1)

    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.feature_pipeline import (
        compute_features_v2_from_data,
        save_features,
    )
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_extract import replay_extract

    from pleroma import extract_bins, run_replay_b0

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
        return 2
    preset = MODEL_PRESETS[args.preset]

    # ── discriminants (Gate 0 target) ──────────────────────────────────────
    with np.load(args.discriminants.expanduser(), allow_pickle=True) as fd:
        if "FULL_names" not in fd:
            logger.error("%s has no FULL_names", args.discriminants)
            return 2
        full_names = [str(x) for x in fd["FULL_names"]]
    logger.info("discriminants: %d FULL_names from %s", len(full_names), args.discriminants)

    # ── calibration — REFUSES to run without it, same as run_replay_b0 ──────
    try:
        positional_means, pca_components, pca_mean, calib_provenance = (
            run_replay_b0.load_calibration_strict(args.calib_dir)
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.error("calibration unusable: %s", exc)
        return 2

    extraction_config, family_config, config_source = run_replay_b0.build_configs(
        args.preset, args.kvrot_path
    )
    logger.info("v3 config source: %s", config_source)

    # ── replay model: the SAME full-capture-surface load run_replay_b0 does ─
    all_layers = list(range(preset.num_layers))
    model_config = ModelConfig(
        model_id=args.model_path,
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    logger.info("loading %s with the full v3 capture surface (eager, replay model)",
                args.model_path)
    replay_loaded = load_model(
        model_config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=all_layers,
        value_layers=all_layers,
        query_layers=all_layers,
        attn_output_layers=all_layers,
    )
    logger.info("replay model loaded; sampled_layers=%s", preset.sampled_layers)

    # ── bins model: the SAME eager-attention load extract_bins.main() does ──
    bins_layers = (
        [int(x) for x in str(args.bins_layers).split(",") if x.strip()]
        if args.bins_layers
        else [int(x) for x in extract_bins.DEFAULT_LAYERS.split(",")]
    )
    bins_n_bins = int(args.bins_n_bins) if args.bins_n_bins else extract_bins.DEFAULT_BINS
    base_summaries = (
        extract_bins.SUMMARIES if args.bins_resolution == "B" else ("hmean",)
    )
    bins_summaries = (
        base_summaries + extract_bins.TEMPORAL_SUMMARIES
        if args.bins_temporal else base_summaries
    )
    bins_model = extract_bins.load_bins_model(
        args.model_path, str(args.model_dtype), str(args.device)
    )
    logger.info("bins model loaded (eager); layers=%s n_bins=%d resolution=%s",
                bins_layers, bins_n_bins, args.bins_resolution)

    # feature_names is fixed for this process's whole lifetime (the config
    # that determines it never changes after startup) — checked against
    # FULL_names ONCE, on the very first gen this process ever replays, then
    # just asserted equal on every gen after (cheap; catches a real drift
    # exactly as run_replay_b0's per-run check would, just amortized).
    state: dict[str, Any] = {"feature_names": None}

    def do_harvest(loom_dir: Path) -> dict[str, Any]:
        records = run_replay_b0.load_gen_records(
            run_replay_b0.resolve_gen_records_dir(loom_dir)
        )
        sig_dir = loom_dir / "signatures"
        fork_dir = loom_dir / "fork_series"
        sig_dir.mkdir(parents=True, exist_ok=True)
        fork_dir.mkdir(parents=True, exist_ok=True)

        # ── stage 1: v3 signatures, via run_replay_b0's own functions ───────
        t0 = time.time()
        cells: list[dict[str, Any]] = []
        replay_failures: list[str] = []
        for rec in records:
            gid = int(rec["generation_id"])
            try:
                input_ids = [int(x) for x in rec["input_ids"]]
                plen = int(rec["prompt_length"])
                raw_data = replay_extract(
                    replay_loaded, input_ids, plen, positional_means=positional_means
                )
                result = compute_features_v2_from_data(
                    raw_data, extraction_config, family_config, pca_components, pca_mean
                )
                names = [str(n) for n in result.feature_names]
                if state["feature_names"] is None:
                    state["feature_names"] = names
                    run_replay_b0.check_gate0(names, full_names)  # raises -> abort worker
                    logger.info("GATE 0 PASSED — feature_names == FULL_names (%d dims)",
                                len(names))
                elif names != state["feature_names"]:
                    raise AssertionError(
                        f"gen_{gid:03d}: feature names diverged mid-process — "
                        "signatures would not be comparable"
                    )
                series = run_replay_b0.fork_series_arrays(
                    raw_data, input_ids, plen, int(args.logits_top_k)
                )
                np.savez_compressed(fork_dir / f"gen_{gid:03d}.npz", **series)
                metadata: dict[str, Any] = {
                    "generation_id": gid,
                    "prompt_id": str(rec["prompt_id"]),
                    "prompt_class": str(rec["prompt_class"]),
                    "prompt": rec.get("prompt"),
                    "prompt_idx": rec.get("prompt_idx"),
                    "seed_idx": int(rec["seed_idx"]),
                    "seed": rec.get("seed"),
                    "user_prompt": rec.get("user_prompt"),
                    "prompt_length": plen,
                    "num_generated_tokens": int(len(input_ids) - plen),
                    "num_features": int(len(result.features)),
                    "tier_slices": {k: list(v) for k, v in result.tier_slices.items()},
                    "extraction_version": 3,
                    "v3_config_source": config_source,
                }
                save_features(gid, result, metadata, sig_dir)
                cells.append({
                    **{k: v for k, v in metadata.items() if k != "tier_slices"},
                    "generated_text": rec.get("generated_text"),
                    "n_steps": int(len(series["logits_entropy"])),
                    "signature_path": f"signatures/gen_{gid:03d}.npz",
                    "fork_series_path": f"fork_series/gen_{gid:03d}.npz",
                    "raw_saved": False,
                })
            except run_replay_b0.Gate0Failure:
                raise
            except Exception as exc:  # noqa: BLE001 — per-gen, run continues
                logger.exception("gen_%03d replay failed", gid)
                replay_failures.append(f"gen_{gid:03d}: {type(exc).__name__}: {exc}")
        replay_s = time.time() - t0

        if not cells:
            raise RuntimeError(f"no gens succeeded in replay stage: {replay_failures}")

        (loom_dir / "cells.json").write_text(json.dumps(cells, indent=2))
        (loom_dir / "run_meta.json").write_text(json.dumps({
            "stage": "harvest_worker_replay_b0",
            "model_id": args.model_path, "preset": args.preset,
            "sampled_layers": preset.sampled_layers, "gen_dir": str(loom_dir),
            "calib_dir": str(args.calib_dir), "calibration": calib_provenance,
            "discriminants": str(args.discriminants), "v3_config_source": config_source,
            "gate0": "passed", "feature_dim": len(state["feature_names"] or []),
            "n_records": len(records), "n_cells": len(cells),
            "logits_top_k": int(args.logits_top_k), "failures": replay_failures,
            "elapsed_s": round(replay_s, 1),
        }, indent=2))

        # ── stage 2: bins-B features, via extract_bins.extract_one ──────────
        t1 = time.time()
        bins_records = [
            extract_bins.GenRecord.load(loom_dir, p)
            for p in sorted((loom_dir / "gen_records").glob("gen_*.json"))
        ]
        dim = len(bins_layers) * bins_n_bins * len(bins_summaries)
        bins_features = np.zeros((len(bins_records), dim), dtype=np.float32)
        bins_failures: list[str] = []
        for i, rec in enumerate(bins_records):
            try:
                bins_features[i] = extract_bins.extract_one(
                    rec, bins_model, str(args.device), bins_layers, bins_n_bins,
                    str(args.bins_resolution), bins_summaries, bool(args.bins_temporal),
                )
            except Exception as exc:  # noqa: BLE001 — per-gen, run continues
                bins_failures.append(
                    f"gen_{rec.generation_id:03d} ({rec.prompt_id}): "
                    f"{type(exc).__name__}: {exc}"
                )
                bins_features[i] = np.nan
        bins_s = time.time() - t1

        np.savez_compressed(
            loom_dir / "bins.npz",
            features=bins_features,
            feature_names=np.array(extract_bins.feature_names(
                f"bins{args.bins_resolution}", bins_layers, bins_n_bins, bins_summaries
            )),
            run_dir=np.array([r.run_dir for r in bins_records]),
            generation_id=np.array([r.generation_id for r in bins_records], dtype=np.int64),
            prompt_id=np.array([r.prompt_id for r in bins_records]),
            prompt_class=np.array([r.prompt_class for r in bins_records]),
            seed_idx=np.array([r.seed_idx for r in bins_records], dtype=np.int64),
            config=np.array(json.dumps({
                "family": f"bins{args.bins_resolution}", "n_bins": bins_n_bins,
                "layers": bins_layers, "head_summaries": list(bins_summaries),
                "temporal": bool(args.bins_temporal),
                "model_path": str(args.model_path), "preset": str(args.preset),
                "model_dtype": str(args.model_dtype), "attn_implementation": "eager",
                "source": "harvest_worker.py (extract_bins.extract_one, in-process)",
                "n_failures": len(bins_failures),
            })),
            failures=np.array(bins_failures),
        )

        return {
            "ok": True, "n_gens": len(records),
            "replay_s": round(replay_s, 3), "bins_s": round(bins_s, 3),
            "failures_replay": replay_failures, "failures_bins": bins_failures,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            logger.info("http %s", fmt % a)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "preset": args.preset,
                                  "model_path": args.model_path})
            else:
                self._send(404, {"ok": False, "error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/harvest":
                self._send(404, {"ok": False, "error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                blob = json.loads(self.rfile.read(n) or b"{}")
                loom_dir = Path(str(blob["loom_dir"]))
                if not loom_dir.is_dir():
                    raise ValueError(f"loom_dir {loom_dir} does not exist")
                out = do_harvest(loom_dir)
                self._send(200, out)
            except Exception as exc:  # noqa: BLE001 — always a JSON body, never a bare 500
                logger.exception("harvest request failed")
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    server = HTTPServer((host, int(args.port)), Handler)
    logger.info("harvest worker listening on %s:%d", host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
