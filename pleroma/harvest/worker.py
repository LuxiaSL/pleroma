"""The loom harvest worker — a persistent process that keeps the harvest models warm.

Without it, /loom's harvest stage is two cold subprocess launches per call:
``pleroma.harvest.replay`` (v3 signatures — loads the model with the full
capture surface, ~20-40 s cold) and ``pleroma.harvest.bins`` (bins-B features —
loads a SECOND eager-attention copy of the model, ~20-30 s cold), both
reloading their models on every /loom call. This worker loads both ONCE, at
startup, and then serves harvest requests over loopback HTTP in a few seconds
each, which takes a loom turn from roughly 60 s to 10 s.

**Numerics are shared, not reimplemented.** Every per-gen computation here
calls the exact functions the two batch modules export:

    pleroma.harvest.replay: fork_series_arrays / check_gate0 / build_configs,
        with anamnesis' replay_extract / compute_features_with_families_from_data
        / save_features (imported and called, same as the subprocess does)
    pleroma.harvest.bins.extract_one        (the per-gen body of that
        module's ``main()`` loop)

Only the PLUMBING differs: one long-lived process instead of two cold ones
per call. The on-disk contract is identical too — the same
``<loom-dir>/signatures/``, ``<loom-dir>/fork_series/``, ``<loom-dir>/cells.json``,
``<loom-dir>/run_meta.json`` and ``<loom-dir>/bins.npz`` land in the loom dir,
so a downstream reader (``LoomMap.lever_of``, the map build) cannot tell which
path produced them.

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
                                          a bare 500 with no body — the server
                                          needs a body to decide fallback)

── SHARDING ─────────────────────────────────────────────────────────────────

``POST /harvest`` also accepts an optional ``"shard": "i/n"``. With it, this
worker extracts only its modulo slice of the loom dir's gen records — the same
``pleroma.harvest.replay.shard_records`` partition a sharded corpus replay
uses, so one definition of "a shard" serves both. N workers pointed at ONE
loom dir therefore cover it exactly once between them.

Per-gen artifacts (``signatures/gen_NNN.npz``, ``fork_series/gen_NNN.npz``) are
named by generation_id and the shards are disjoint, so no two workers can ever
target one path. The three WHOLE-CORPUS files cannot be written by any single
worker, so a sharded request writes them under ``shard_meta/`` instead:

    shard_meta/cells_shard<i>of<n>.json
    shard_meta/run_meta_shard<i>of<n>.json
    shard_meta/bins_shard<i>of<n>.npz

and ``pleroma.harvest.pool.harvest_via_pool`` merges those into the
``<loom-dir>/cells.json`` / ``<loom-dir>/run_meta.json`` / ``<loom-dir>/bins.npz``
that every downstream reader expects. An UNSHARDED request (no ``shard`` key,
or ``"0/1"``) is the single-worker path, unaffected by the pool.

**The numerics do not change with n.** Every span is extracted by exactly the
same per-record call chain regardless of which worker gets it; nothing in the
replay or bins stage carries state across records. That is asserted, not
assumed — see ``tests/unit/harvest/test_harvest_pool.py``
(``test_parallel_matches_serial_byte_for_byte``).

The server's harvest ladder (``pleroma.serve.harvest_client.HarvestLadder``)
tries this worker first (short connect timeout) and falls back to the
two-subprocess pipeline if the worker is unreachable or reports an error.
This process is therefore OPTIONAL — the server works without it, only slower.

── TWO HARVEST LANES ────────────────────────────────────────────────────────

``--harvest-lane {replay,gpu}``, default ``replay``.

**replay** (the default, and the path described above): stage 1 is
``pleroma.harvest.replay``'s v3 capture surface, which is FROZEN — pinned to
the 3B/8B discriminants, deliberately omitting value_geometry / qk_geometry /
kv_cka / expert_routing (see ``pleroma.harvest.replay.build_configs``). That
surface must not be widened: the live 8B loom's Gate 0 is that exact name list.

**gpu**: stage 1 is instead anamnesis' GPU ("fast") replay lane, driven
through ``pleroma.harvest.anamnesis_seam`` (``GpuHarvestLane``: the lane model,
calibration and checkpoint digests loaded once per worker; ``fork_series/``
read off the lane's own forward). The lane
emits three families the replay surface does not — value_geometry (295 names),
qk_geometry (301), kv_cka (118) — 4,086 names at 70B against the replay
surface's 3,441.

★ THIS IS A SECOND IMPLEMENTATION, NOT A FLAG ON A SHARED ONE. The lane has its
own extraction code (``anamnesis.extraction.fast``). Nothing about
``pleroma.harvest.replay`` changes when you pass ``--harvest-lane gpu``; it is
simply not imported into the harvest path. That is the whole point: a model C
(70B) loom can harvest on the lane its map was fit on WITHOUT the frozen
replay surface moving a byte under the 8B.

★ AND IT DOES NOT COST A THIRD MODEL COPY. In replay mode this worker holds the
model twice (replay + bins). In gpu mode it holds it twice as well (lane +
bins) — the replay copy is never loaded, because for a lane-fit map it is dead
weight. At 70B that is the difference between 2 GPUs and 5.

    stage 2 (bins) is IDENTICAL in both lanes. Only stage 1 differs.

Everything downstream of stage 1 — Gate 0 against the discriminants' FULL_names,
``signatures/gen_NNN.npz``, ``fork_series/gen_NNN.npz``, ``<loom-dir>/cells.json``,
``<loom-dir>/run_meta.json``, ``<loom-dir>/bins.npz``, the pool's shard_meta/
contract — is the same on-disk shape in both lanes, so the server and the map
build cannot tell which one ran except by reading the provenance that says so.

Usage (from the repo root, inside the project venv):
    python -m pleroma.harvest.worker \\
        --preset 3b --model-path <hf-id-or-local-dir> \\
        --calib-dir data/calibration/3b \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --host 127.0.0.1 --port 8768

    # 70B model C, lane harvest, two GPUs (lane on --device, a whole-model
    # placement the fast lane requires — and bins on cuda:1):
    CUBLAS_WORKSPACE_CONFIG=:16:8 python -m pleroma.harvest.worker \\
        --preset <70b preset> --model-path <dir> --model-dtype bfloat16 \\
        --harvest-lane gpu --map <loom_map.npz> \\
        --calib-dir <calib> --discriminants <manifest.npz> \\
        --device cuda:0 --bins-device cuda:1 \\
        --bins-layers 12,23,35,47,59,70,79 --port 8801
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("harvest_worker")

#: What goes in run_meta/cells as ``v3_config_source`` for a GPU-lane harvest.
#: The replay lane records its own label (``pleroma.harvest.replay.build_configs``);
#: a lane harvest must not be able to claim that string, because the two configs
#: are different objects producing different name lists. Owned by the seam (it
#: names the pin).
LANE_CONFIG_SOURCE = (
    "anamnesis GPU lane: extraction.replay_config.native_replay_configs via "
    "pleroma.harvest.anamnesis_seam (value_geometry + qk_geometry + kv_cka "
    "INCLUDED; not the frozen v3 replay surface)"
)


# ── the bins config must be the SERVED MAP's bins config ─────────────────────

def load_map_meta(map_path: Path) -> dict[str, Any]:
    """The `meta` blob the map build (`pleroma.map.build.export`) writes into a loom map npz."""
    with np.load(Path(map_path).expanduser(), allow_pickle=True) as z:
        if "meta" not in z.files:
            raise KeyError(f"{map_path} has no `meta` — not a loom map npz")
        meta = json.loads(str(z["meta"]))
    if not isinstance(meta, dict):
        raise ValueError(f"{map_path}: meta is {type(meta).__name__}, not an object")
    return meta


def bins_config_disagreements(
    map_meta: Mapping[str, Any],
    *,
    family: str,
    layers: Sequence[int],
    n_bins: int,
    summaries: Sequence[str],
    temporal: bool,
) -> list[str]:
    """Name-by-name comparison of THIS worker's bins config with the map's.

    ★ COUNTS ARE NOT AGREEMENT, AND THIS IS THE FUNCTION THAT SAYS SO.
    7 layers x 20 bins x 3 summaries is 420 dims whether the layers are
    ``[4, 8, 12, 16, 20, 24, 27]`` (``pleroma.harvest.bins``' 3B/8B default) or
    ``[12, 23, 35, 47, 59, 70, 79]`` (the 70B map's). A 70B worker run with the
    first against a map trained on the second — all seven layers in the bottom
    third of an 80-layer model — raises NOTHING, because the shape is right and
    only the meaning is wrong. A flag cannot guard that; this function guards
    the class, by refusing to trust the flag.

    Returns a list of human-readable disagreements; empty means agreement.
    """
    cfg = map_meta.get("bins_config")
    if not isinstance(cfg, Mapping):
        return ["the map's meta carries no `bins_config` to check against"]
    out: list[str] = []
    want_family = str(cfg.get("family", ""))
    if want_family and want_family != str(family):
        out.append(f"family: map wants {want_family!r}, worker has {family!r}")
    want_layers = [int(x) for x in (cfg.get("layers") or [])]
    got_layers = [int(x) for x in layers]
    if want_layers and want_layers != got_layers:
        same_dim = len(want_layers) == len(got_layers)
        out.append(
            f"layers: map wants {want_layers}, worker has {got_layers}"
            + (" — SAME COUNT, so the bins row would have the right width and "
               "the wrong meaning" if same_dim else "")
        )
    if cfg.get("n_bins") is not None and int(cfg["n_bins"]) != int(n_bins):
        out.append(f"n_bins: map wants {int(cfg['n_bins'])}, worker has {int(n_bins)}")
    want_sum = [str(x) for x in (cfg.get("head_summaries") or [])]
    got_sum = [str(x) for x in summaries]
    if want_sum and want_sum != got_sum:
        out.append(f"head_summaries: map wants {want_sum}, worker has {got_sum}")
    if cfg.get("temporal") is not None and bool(cfg["temporal"]) != bool(temporal):
        out.append(f"temporal: map wants {bool(cfg['temporal'])}, worker has "
                   f"{bool(temporal)}")
    return out


def discriminants_disagreements(
    map_meta: Mapping[str, Any], discriminants: Path
) -> list[str]:
    """Is this worker's Gate 0 target the one the map was fit against?

    Compares by sha256 (the map records ``discriminants_sha256``), not by path:
    the same file reachable under two paths is fine, and two different files at
    one path is exactly the failure worth catching.
    """
    import hashlib

    want = str(map_meta.get("discriminants_sha256") or "")
    if not want:
        return ["the map's meta carries no `discriminants_sha256` to check against"]
    path = Path(discriminants).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024**2), b""):
            digest.update(block)
    got = digest.hexdigest()
    if got == want:
        return []
    return [
        f"discriminants: map was fit against sha256 {want}, this worker loaded "
        f"{path} with sha256 {got} — the Gate 0 target and the map disagree"
    ]


# ── GPU lane: one lane run per harvest, through the seam ────────────────────

def run_gpu_lane(
    *,
    lane: Any,
    loom_dir: Path,
    gen_ids: Sequence[int] | None,
    shard_label: str,
    fork_top_k: int,
) -> Path:
    """Build the lane manifest for a loom dir and replay it through ``lane``.

    ``lane`` is the worker's ``pleroma.harvest.anamnesis_seam.GpuHarvestLane``:
    the lane model, calibration and checkpoint digests were paid ONCE at worker
    start, and this call only replays spans.

    The manifest transform is ``pleroma.harvest.gpu_manifests.write_manifests``
    — the SAME function that builds the corpus manifests, not a second copy of
    it. It is called with ``shards=1``: the worker's own shard partition is
    ``pleroma.harvest.replay.shard_records``' modulo slice, which is a DIFFERENT
    partition from ``write_manifests``' contiguous blocks, so the slice is expressed as
    ``gen_ids`` instead. Mixing the two partitions would give a pool of
    workers overlapping coverage that still looked complete.

    Returns the lane's output dir.
    """
    from pleroma.harvest import gpu_manifests as w6_gpu_manifests

    suffix = f"_{shard_label}" if shard_label else ""
    lane_root = Path(loom_dir) / f"lane{suffix}"
    # The seam REFUSES an output dir that exists (as upstream's runner does),
    # which is right for a corpus run and wrong for a retried draw. A loom dir
    # is written fresh per draw, so anything here is debris from a failed try.
    if lane_root.exists():
        logger.warning("clearing lane debris at %s", lane_root)
        shutil.rmtree(lane_root)
    man_root = lane_root / "manifest"
    n_gens = w6_gpu_manifests.write_manifests(Path(loom_dir), man_root, shards=1)
    manifest = man_root / "shard_00" / "manifest.json"
    out_dir = lane_root / "out"
    logger.info("gpu lane: %d gens in manifest, %s selected",
                n_gens, "all" if gen_ids is None else len(list(gen_ids)))
    lane.run(manifest, out_dir,
             gen_ids=None if gen_ids is None else sorted(int(g) for g in gen_ids),
             fork_top_k=int(fork_top_k))
    return out_dir


# ── GPU lane: adopt one lane run's output into the loom dir's harvest layout ──

def adopt_lane_output(
    lane_out: Path,
    loom_dir: Path,
    records: Sequence[Mapping[str, Any]],
    full_names: Sequence[str],
    expected_names: list[str] | None,
    check_gate0: Any,
    gate0_failure: type[BaseException],
) -> tuple[list[dict[str, Any]], list[str], list[str] | None, str | None]:
    """Move a GPU-lane run's per-gen artifacts into ``signatures/``/``fork_series/``.

    The lane writes ``<out>/gen_NNN.npz`` + ``<out>/gen_NNN.json`` flat (that is
    ``feature_pipeline.save_features``, the SAME function the replay lane
    calls) and ``<out>/fork_series/gen_NNN.npz`` (the seam reads the logits
    off the lane's own forward). The loom dir's harvest layout wants the features under
    ``signatures/`` and the fork series at the top level, so this moves them
    there and builds the ``cells`` rows from the gen records.

    Gate 0 runs on the FIRST gen adopted and every later gen is asserted equal
    to it — the same amortization the replay lane uses, and the same refusal:
    ``check_gate0`` raises and the worker aborts rather than harvesting into a
    space the map cannot read. ⚠️ It compares NAMES. ``len(a) == len(b)`` is not
    agreement; the 70B blocker this lane exists to clear was 3,441 vs 4,086 with
    one list a prefix of the other.

    Returns ``(cells, failures, feature_names, lane_id)``. ``feature_names`` is
    whatever was seen (``expected_names`` unchanged when nothing was adopted).
    """
    sig_dir = Path(loom_dir) / "signatures"
    fork_dir = Path(loom_dir) / "fork_series"
    sig_dir.mkdir(parents=True, exist_ok=True)
    fork_dir.mkdir(parents=True, exist_ok=True)
    lane_out = Path(lane_out)

    names_seen = expected_names
    lane_id: str | None = None
    cells: list[dict[str, Any]] = []
    failures: list[str] = []
    for rec in sorted(records, key=lambda r: int(r["generation_id"])):
        gid = int(rec["generation_id"])
        try:
            src_npz = lane_out / f"gen_{gid:03d}.npz"
            src_json = lane_out / f"gen_{gid:03d}.json"
            src_fork = lane_out / "fork_series" / f"gen_{gid:03d}.npz"
            if not src_npz.exists():
                raise FileNotFoundError(
                    f"the lane wrote no features for gen {gid} ({src_npz})"
                )
            if not src_fork.exists():
                raise FileNotFoundError(
                    f"the lane wrote no fork series for gen {gid} ({src_fork}). "
                    "Upstream's run_gpu_replay never writes fork_series/ — only "
                    "pleroma.harvest.anamnesis_seam.replay_manifest_to_dir "
                    "does; without it build_pairs/fork_tokens downstream have "
                    "nothing to read."
                )
            with np.load(src_npz, allow_pickle=True) as z:
                names = [str(n) for n in z["feature_names"]]
                n_features = int(np.asarray(z["features"]).shape[0])
            if names_seen is None:
                check_gate0(names, list(full_names))  # raises -> abort worker
                names_seen = names
                logger.info(
                    "GATE 0 PASSED (gpu lane) — feature_names == FULL_names "
                    "(%d dims)", len(names),
                )
            elif names != names_seen:
                raise AssertionError(
                    f"gen_{gid:03d}: feature names diverged mid-process — "
                    "signatures would not be comparable"
                )
            with np.load(src_fork) as fz:
                n_steps = int(np.asarray(fz["logits_entropy"]).shape[0])
            lane_meta: dict[str, Any] = {}
            if src_json.exists():
                try:
                    lane_meta = json.loads(src_json.read_text())
                except (OSError, ValueError):
                    lane_meta = {}
            if lane_id is None:
                lane_id = (str(lane_meta["lane_id"])
                           if lane_meta.get("lane_id") is not None else None)
            shutil.move(str(src_fork), str(fork_dir / f"gen_{gid:03d}.npz"))
            shutil.move(str(src_npz), str(sig_dir / f"gen_{gid:03d}.npz"))
            if src_json.exists():
                shutil.move(str(src_json), str(sig_dir / f"gen_{gid:03d}.json"))
            plen = int(rec["prompt_length"])
            cells.append({
                "generation_id": gid,
                "prompt_id": str(rec["prompt_id"]),
                "prompt_class": str(rec["prompt_class"]),
                "prompt": rec.get("prompt"),
                "prompt_idx": rec.get("prompt_idx"),
                "seed_idx": int(rec["seed_idx"]),
                "seed": rec.get("seed"),
                "user_prompt": rec.get("user_prompt"),
                "prompt_length": plen,
                "num_generated_tokens": int(len(rec["input_ids"]) - plen),
                "num_features": n_features,
                "extraction_version": 3,
                "v3_config_source": LANE_CONFIG_SOURCE,
                # Provenance that says WHICH instrument produced this row. A
                # cell that cannot name its own lane is unreadable later, and
                # mixing lanes under one map is the hazard this whole change
                # exists to keep impossible.
                "extraction_lane": "gpu",
                "lane_id": lane_meta.get("lane_id"),
                "generated_text": rec.get("generated_text"),
                "n_steps": n_steps,
                "signature_path": f"signatures/gen_{gid:03d}.npz",
                "fork_series_path": f"fork_series/gen_{gid:03d}.npz",
                "raw_saved": False,
            })
        except gate0_failure:
            raise
        except Exception as exc:  # noqa: BLE001 — per-gen, run continues
            logger.exception("gen_%03d lane adoption failed", gid)
            failures.append(f"gen_{gid:03d}: {type(exc).__name__}: {exc}")
    return cells, failures, names_seen, lane_id


def main() -> int:  # noqa: C901 — one linear procedure, sectioned by stage
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--discriminants", type=Path, required=True)
    parser.add_argument(
        "--harvest-lane",
        choices=["replay", "gpu"],
        default="replay",
        help=(
            "which instrument runs STAGE 1. 'replay' (default) is "
            "pleroma.harvest.replay's FROZEN v3 capture surface, the one the "
            "3B/8B maps are fit on. 'gpu' is anamnesis' GPU replay "
            "lane (pleroma.harvest.anamnesis_seam), which emits the three "
            "families the replay surface deliberately omits (value_geometry, "
            "qk_geometry, kv_cka) and is what a model-C map was fit on. Stage "
            "2 (bins) is identical either way. In 'gpu' mode the replay model "
            "is NEVER LOADED — for a lane-fit map it is dead weight, and at "
            "70B it is ~141 GB of it."
        ),
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        type=Path,
        default=None,
        help=(
            "the loom map this worker will be harvesting FOR. Optional, and "
            "worth more than any flag when given: the worker validates its "
            "bins layers / n_bins / head_summaries / temporal and its "
            "discriminants sha256 against the map's own `bins_config` and "
            "`discriminants_sha256`, and REFUSES TO START on a mismatch. "
            "★ 7x20x3 = 420 dims whether the layers are the 3B defaults or the "
            "70B map's, so this is the only check that catches a 70B worker "
            "harvesting the bottom third of an 80-layer model into a map "
            "trained on [12,23,35,47,59,70,79], which otherwise raises nothing."
        ),
    )
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bins-device",
        default=None,
        help=(
            "device for the BINS model; defaults to --device. ★ This worker "
            "holds the model TWICE — once for the v3 replay and once for bins "
            "(pleroma.harvest.bins.load_bins_model). On one device, at 3B/8B "
            "that is ~12-32 GB and fits. At 70B it is ~280 GB against a 178 GB "
            "card and the second load dies with a CUDA OOM whose traceback points at `model.to()` "
            "rather than at the duplication, which is the actual cause. Give "
            "the two models separate GPUs at 70B, e.g. --device cuda:0 "
            "--bins-device cuda:1."
        ),
    )
    parser.add_argument("--logits-top-k", type=int, default=50,
                        help="matches pleroma.harvest.replay's --logits-top-k default")
    parser.add_argument(
        "--allow-anamnesis-mismatch", action="store_true",
        help=("start even if the imported anamnesis is not the pinned "
              "vendor/anamnesis commit (loudly logged). Without it the worker "
              "REFUSES: every map it harvests for was fit in the pinned space."),
    )
    parser.add_argument("--bins-layers", default=None,
                        help="matches extract_bins --layers; default = extract_bins.DEFAULT_LAYERS")
    parser.add_argument("--bins-n-bins", type=int, default=None,
                        help="matches extract_bins --n-bins; default = extract_bins.DEFAULT_BINS")
    parser.add_argument("--bins-resolution", choices=["A", "B"], default="B")
    parser.add_argument("--bins-temporal", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--gpu-mem-fraction", type=float, default=0.0,
        help=(
            "RESOURCE BOUND: cap this process's CUDA allocator at this fraction "
            "of the visible device (0 = uncapped, the historical behaviour). A "
            "harvest POOL packs several workers onto the box; without a cap one "
            "worker's allocator can grow into the share another needs — or into "
            "the live server's. 0.12 comfortably holds 3B's two eager copies."
        ),
    )
    args = parser.parse_args()

    from pleroma.net import require_loopback

    host = require_loopback(args.host)

    # Pin CPU thread pools BEFORE numpy/torch import, matching pleroma.harvest.replay.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # MANDATORY on this stack — never remove
    torch.set_num_threads(1)

    # Resource bound, applied BEFORE any model load so the cap covers the
    # weights too. Refuses loudly rather than running uncapped on a bad value:
    # a pool member that silently ignores its ceiling is the one that starves
    # the box.
    if float(args.gpu_mem_fraction) > 0:
        frac = float(args.gpu_mem_fraction)
        if not 0 < frac <= 1:
            logger.error("--gpu-mem-fraction must be in (0, 1], got %s", frac)
            return 2
        if not torch.cuda.is_available():
            logger.error("--gpu-mem-fraction given but CUDA is not available")
            return 2
        torch.cuda.set_per_process_memory_fraction(frac)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        logger.info("CUDA allocator capped at %.2f of device 0 (%.1f GiB of %.1f GiB)",
                    frac, frac * total_gb, total_gb)

    from pleroma.harvest import anamnesis_seam

    try:
        anamnesis_seam.verify_anamnesis_pin(
            allow_mismatch=bool(args.allow_anamnesis_mismatch) or None)
    except anamnesis_seam.AnamnesisPinError as exc:
        logger.error("REFUSING TO START: %s", exc)
        return 2

    from anamnesis.config import ModelConfig, UnknownPresetError
    from anamnesis.extraction.feature_pipeline import (
        compute_features_with_families_from_data,
        save_features,
    )
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay.extract import replay_extract

    from pleroma.harvest import bins as extract_bins
    from pleroma.harvest import replay as run_replay_b0

    try:
        preset = anamnesis_seam.resolve_preset(args.preset)
    except UnknownPresetError as exc:
        logger.error("%s", exc)
        return 2

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

    gpu_lane = str(args.harvest_lane) == "gpu"

    # ── bins config, RESOLVED BEFORE ANY MODEL LOADS ────────────────────────
    # Resolved before the model loads so a config the served map cannot read
    # costs zero seconds to refuse instead of one 141 GB load. Pure argument
    # arithmetic.
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
    bins_device = str(args.bins_device or args.device)

    # ── does this worker agree with the MAP it will be harvesting for? ──────
    if args.map_path is not None:
        try:
            map_meta = load_map_meta(args.map_path)
        except (OSError, KeyError, ValueError) as exc:
            logger.error("--map %s unreadable: %s", args.map_path, exc)
            return 2
        problems = bins_config_disagreements(
            map_meta,
            family=f"bins{args.bins_resolution}",
            layers=bins_layers,
            n_bins=bins_n_bins,
            summaries=bins_summaries,
            temporal=bool(args.bins_temporal),
        ) + discriminants_disagreements(map_meta, args.discriminants)
        if problems:
            logger.error(
                "REFUSING TO START — this worker does not agree with the map it "
                "was pointed at (%s):", args.map_path,
            )
            for p in problems:
                logger.error("  * %s", p)
            logger.error(
                "Nothing here would have ERRORED at harvest time; the widths "
                "match and only the meaning differs. That is the whole reason "
                "this check exists."
            )
            return 2
        logger.info("map agreement OK: bins_config and discriminants sha256 "
                    "match %s", args.map_path)
    else:
        logger.warning(
            "no --map given, so the bins config (%s x %d bins x %s) and the "
            "discriminants are TRUSTED, not verified. Pass --map <the map this "
            "worker serves> and a wrong-layers harvest becomes a refusal "
            "instead of a silent 420 dims of the wrong thing.",
            bins_layers, bins_n_bins, list(bins_summaries),
        )

    gpu_harvest_lane: Any = None
    if gpu_lane:
        # ── GPU LANE: the replay model is NEVER LOADED ──────────────────────
        # Stage 1 is anamnesis' fast lane on ITS OWN eager copy of the model
        # (--device), loaded once here with calibration. Loading
        # pleroma.harvest.replay's capture surface too would be a THIRD 141 GB copy of
        # weights nothing in this mode ever reads.
        extraction_config = family_config = model_config = replay_loaded = None
        config_source = LANE_CONFIG_SOURCE
        logger.info("harvest lane: GPU (%s)", config_source)
        try:
            # Refuses without CUBLAS_WORKSPACE_CONFIG=:16:8 exported BEFORE
            # python started: part of the lane's deterministic identity
            # (deterministic algorithms, TF32 off), not a preference.
            gpu_harvest_lane = anamnesis_seam.GpuHarvestLane.open(
                preset=preset, model_path=str(args.model_path),
                calib_dir=args.calib_dir, device=str(args.device),
            )
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            logger.error("GPU lane unusable: %s", exc)
            return 2
        logger.info("GPU lane ready on %s (anamnesis %s)", args.device,
                    anamnesis_seam.PINNED_COMMIT[:12])
    else:
        extraction_config, family_config, config_source = run_replay_b0.build_configs(args.preset)
        logger.info("v3 config source: %s", config_source)

        # ── replay model: the SAME full-capture-surface load pleroma.harvest.replay does
        all_layers = list(range(preset.num_layers))
        model_config = ModelConfig.from_preset(preset, model_id=args.model_path)
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
    bins_model = extract_bins.load_bins_model(
        args.model_path, str(args.model_dtype), bins_device
    )
    logger.info("bins model loaded (eager) on %s; layers=%s n_bins=%d resolution=%s",
                bins_device, bins_layers, bins_n_bins, args.bins_resolution)
    if bins_device == str(args.device):
        logger.info("replay and bins models share %s — fine below ~13B, fatal at 70B "
                    "(see --bins-device)", bins_device)
    if gpu_harvest_lane is not None:
        # Hash the checkpoint ONCE, now, with the page cache still warm from the
        # bins load. The lane receipt stamps every safetensors shard's sha256;
        # at 70B that is ~141 GB — minutes — so it is paid here and held, never
        # on a draw (the server's harvest timeout is 120-900 s).
        gpu_harvest_lane.prewarm()

    # feature_names is fixed for this process's whole lifetime (the config
    # that determines it never changes after startup) — checked against
    # FULL_names ONCE, on the very first gen this process ever replays, then
    # just asserted equal on every gen after (cheap; catches a real drift
    # exactly as pleroma.harvest.replay's per-run check does, just amortized).
    state: dict[str, Any] = {"feature_names": None}

    def do_harvest(
        loom_dir: Path, shard_index: int = 0, shard_count: int = 1
    ) -> dict[str, Any]:
        records = run_replay_b0.load_gen_records(
            run_replay_b0.resolve_gen_records_dir(loom_dir)
        )
        n_corpus = len(records)
        # SHARD the static, sorted record list — the identical partition
        # a sharded corpus replay hands its workers. There is no resume filter
        # here (a loom dir is freshly generated every draw), so the ordering
        # hazard that module's docstring warns about cannot arise.
        if shard_count > 1:
            records = run_replay_b0.shard_records(records, shard_index, shard_count)
        shard_label = f"{shard_index}of{shard_count}" if shard_count > 1 else ""
        # The bins stage must see EXACTLY this record set, or the two halves of
        # one span's signature would come from different shards.
        shard_gids = {int(r["generation_id"]) for r in records}
        if not records:
            # More workers than gens. Not an error: return an empty, well-formed
            # shard so the merge still sees n shard files and the pool does not
            # mistake "nothing to do" for "a worker died".
            logger.info("shard %s has no gens (%d in corpus) — empty result",
                        shard_label, n_corpus)
            shard_dir = loom_dir / "shard_meta"
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / f"cells_shard{shard_label}.json").write_text("[]")
            (shard_dir / f"run_meta_shard{shard_label}.json").write_text(json.dumps({
                "stage": ("harvest_worker_gpu_lane" if gpu_lane
                          else "harvest_worker_replay_b0"),
                "harvest_lane": str(args.harvest_lane),
                "shard": shard_label,
                # NOT "passed": Gate 0 runs on a worker's first gen, and this
                # shard had none. Claiming a gate that never ran is how a real
                # space mismatch would get merged one day.
                "gate0": ("passed" if state["feature_names"] is not None
                          else "not_run (empty shard)"),
                "n_gens_extracted": 0,
                "feature_dim": len(state["feature_names"] or []),
                "calibration": calib_provenance, "n_corpus": n_corpus,
                "n_records": 0, "n_cells": 0, "failures": [], "elapsed_s": 0.0,
            }, indent=2))
            np.savez_compressed(
                shard_dir / f"bins_shard{shard_label}.npz",
                features=np.zeros((0, 0), dtype=np.float32),
                generation_id=np.zeros((0,), dtype=np.int64),
                empty_shard=np.array(True),
            )
            return {"ok": True, "n_gens": 0, "shard": shard_label,
                    "replay_s": 0.0, "bins_s": 0.0,
                    "failures_replay": [], "failures_bins": []}
        sig_dir = loom_dir / "signatures"
        fork_dir = loom_dir / "fork_series"
        sig_dir.mkdir(parents=True, exist_ok=True)
        fork_dir.mkdir(parents=True, exist_ok=True)

        # ── stage 1: signatures + fork series ───────────────────────────────
        # TWO IMPLEMENTATIONS, one on-disk contract. Both fill signatures/ and
        # fork_series/ and build `cells`; only the instrument differs. Stage 2
        # below is shared verbatim.
        t0 = time.time()
        cells: list[dict[str, Any]] = []
        replay_failures: list[str] = []
        lane_id: str | None = None
        if gpu_lane:
            out_dir = run_gpu_lane(
                lane=gpu_harvest_lane,
                loom_dir=loom_dir,
                gen_ids=sorted(shard_gids) if shard_count > 1 else None,
                shard_label=shard_label,
                fork_top_k=int(args.logits_top_k),
            )
            cells, replay_failures, names_seen, lane_id = adopt_lane_output(
                out_dir, loom_dir, records, full_names,
                state["feature_names"],
                run_replay_b0.check_gate0, run_replay_b0.Gate0Failure,
            )
            state["feature_names"] = names_seen
        else:
            stage1_replay(records, loom_dir, cells, replay_failures)
        replay_s = time.time() - t0

        if not cells:
            raise RuntimeError(
                f"no gens succeeded in stage 1 ({args.harvest_lane}): "
                f"{replay_failures}"
            )

        run_meta = {
            "stage": ("harvest_worker_gpu_lane" if gpu_lane
                      else "harvest_worker_replay_b0"),
            "harvest_lane": str(args.harvest_lane),
            "model_id": args.model_path, "preset": args.preset,
            "sampled_layers": preset.sampled_layers, "gen_dir": str(loom_dir),
            "calib_dir": str(args.calib_dir), "calibration": calib_provenance,
            "discriminants": str(args.discriminants), "v3_config_source": config_source,
            "gate0": "passed", "feature_dim": len(state["feature_names"] or []),
            "n_corpus": n_corpus,
            "n_records": len(records), "n_cells": len(cells),
            "logits_top_k": int(args.logits_top_k), "failures": replay_failures,
            "elapsed_s": round(replay_s, 1),
            **({"lane_id": lane_id,
                "bins_config_verified_against_map": (str(args.map_path)
                                                     if args.map_path else None)}
               if gpu_lane else {}),
        }
        write_stage1_meta(loom_dir, cells, run_meta, shard_count, shard_label)

        return stage2_bins(loom_dir, shard_gids, shard_count, shard_label,
                           records, n_corpus, replay_s, replay_failures)

    def stage1_replay(
        records: Sequence[Mapping[str, Any]], loom_dir: Path,
        cells: list[dict[str, Any]], replay_failures: list[str],
    ) -> None:
        """The FROZEN v3 replay surface. Untouched by the GPU-lane addition —
        this is character-for-character the loop that has always run, moved into
        a function so the two lanes are visibly two implementations rather than
        one function with a mode flag threaded through it."""
        sig_dir = loom_dir / "signatures"
        fork_dir = loom_dir / "fork_series"
        for rec in records:
            gid = int(rec["generation_id"])
            try:
                input_ids = [int(x) for x in rec["input_ids"]]
                plen = int(rec["prompt_length"])
                raw_data = replay_extract(
                    replay_loaded, input_ids, plen, positional_means=positional_means
                )
                result = compute_features_with_families_from_data(
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
                    # on-disk key stays "tier_slices" (STORED_BLOCK_SLICES_KEY)
                    "tier_slices": {k: list(v) for k, v in result.block_slices.items()},
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

    def write_stage1_meta(
        loom_dir: Path, cells: list[dict[str, Any]], run_meta: dict[str, Any],
        shard_count: int, shard_label: str,
    ) -> None:
        # Whole-corpus files: a sharded worker writes its slice under
        # shard_meta/ for the pool to merge; an unsharded one writes the real
        # files exactly where it always has.
        if shard_count > 1:
            run_meta["shard"] = shard_label
            shard_dir = loom_dir / "shard_meta"
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / f"cells_shard{shard_label}.json").write_text(
                json.dumps(cells, indent=2))
            (shard_dir / f"run_meta_shard{shard_label}.json").write_text(
                json.dumps(run_meta, indent=2))
        else:
            (loom_dir / "cells.json").write_text(json.dumps(cells, indent=2))
            (loom_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    def stage2_bins(
        loom_dir: Path, shard_gids: set[int], shard_count: int, shard_label: str,
        records: Sequence[Mapping[str, Any]], n_corpus: int, replay_s: float,
        replay_failures: list[str],
    ) -> dict[str, Any]:
        """bins-B features. IDENTICAL for both harvest lanes — it reads the
        gen records and the bins model, neither of which stage 1 touches."""
        # ── stage 2: bins-B features, via extract_bins.extract_one ──────────
        t1 = time.time()
        bins_records = [
            rec for rec in (
                extract_bins.GenRecord.load(loom_dir, p)
                for p in sorted((loom_dir / "gen_records").glob("gen_*.json"))
            )
            # SAME slice as the replay stage above. Filtering by the replay
            # stage's generation_ids rather than re-running shard_records keeps
            # the two stages tied to ONE partition by construction: a span's v3
            # signature and its bins row can never come from different workers.
            if shard_count == 1 or int(rec.generation_id) in shard_gids
        ]
        dim = len(bins_layers) * bins_n_bins * len(bins_summaries)
        bins_features = np.zeros((len(bins_records), dim), dtype=np.float32)
        bins_failures: list[str] = []
        for i, rec in enumerate(bins_records):
            try:
                bins_features[i] = extract_bins.extract_one(
                    # ⛔ `bins_device`, NOT `args.device`. extract_one builds the
                    # input tensor on the device it is handed and then calls the
                    # model with it, so this argument must name the device the
                    # BINS model is on. Below ~13B the two are the same device
                    # by default, so passing `args.device` here passes every
                    # small-model test; with split devices it makes every gen
                    # raise a device mismatch, land NaN in its row, and come
                    # back as a fan of dead futures with no map and no wear — a
                    # whole-draw failure that never says "device".
                    rec, bins_model, bins_device, bins_layers, bins_n_bins,
                    str(args.bins_resolution), bins_summaries, bool(args.bins_temporal),
                )
            except Exception as exc:  # noqa: BLE001 — per-gen, run continues
                bins_failures.append(
                    f"gen_{rec.generation_id:03d} ({rec.prompt_id}): "
                    f"{type(exc).__name__}: {exc}"
                )
                bins_features[i] = np.nan
        bins_s = time.time() - t1

        bins_path = (
            loom_dir / "shard_meta" / f"bins_shard{shard_label}.npz"
            if shard_count > 1 else loom_dir / "bins.npz"
        )
        bins_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bins_path,
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
            "shard": shard_label, "n_corpus": n_corpus,
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
                # Optional "shard": "i/n". Absent -> the unsharded path.
                # Parsed with pleroma.harvest.replay's OWN parser so a shard
                # means one thing in this repo.
                shard_index, shard_count = 0, 1
                if blob.get("shard"):
                    shard_index, shard_count = run_replay_b0.parse_shard(
                        str(blob["shard"])
                    )
                out = do_harvest(loom_dir, shard_index, shard_count)
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
