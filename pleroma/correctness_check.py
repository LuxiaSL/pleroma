"""Correctness check: harvest_worker.py vs the original subprocess pipeline.

The claim under test (BENCH-LOOM-2026-09-17.md): "only the plumbing changed"
— harvest_worker.py calls the exact same frozen functions
(run_replay_b0.replay_extract / compute_features_v2_from_data / save_features
/ fork_series_arrays / check_gate0, extract_bins.extract_one) that the
subprocess pipeline calls, just in one long-lived process instead of two
cold ones. This script is the proof: take ONE fixed set of gen_records
(fixed input_ids — no generation happens here, batching's RNG change is
irrelevant to this check), run it through BOTH harvest paths into separate
out-dirs, and diff every artifact that downstream code (LoomMap.lever_of)
actually reads:

    signatures/gen_NNN.npz "features"   (v3 signature, run_replay_b0)
    bins.npz "features"                 (bins-B features, extract_bins)
    LoomMap.lever_of(sig, bins_row)     (the fitted map's OUTPUT — the thing
                                          /wear actually uses)

Passes iff every one of those is bit-identical or within cosine distance
1e-4 (allows for float nondeterminism across process/thread pools, not
algorithmic drift).

Usage (from the repo root, inside the project venv):
    python -m pleroma.correctness_check \\
        --gen-records-dir bench_work/<tag>/loom_000/gen_records \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --calib-dir data/calibration/3b \\
        --discriminants factor_directions_3b.npz \\
        --map outputs/loom/loom_map.npz \\
        --worker-url http://127.0.0.1:8768 \\
        --work-dir correctness_work
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("correctness_check")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-records-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--discriminants", required=True)
    parser.add_argument("--kvrot-path", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--worker-url", default="http://127.0.0.1:8768")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tol-cosine", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.gen_records_dir.is_dir():
        logger.error("%s is not a directory", args.gen_records_dir)
        return 2

    sub_dir = args.work_dir / "via_subprocess"
    wrk_dir = args.work_dir / "via_worker"
    for d in (sub_dir, wrk_dir):
        if d.exists():
            shutil.rmtree(d)
        (d / "gen_records").mkdir(parents=True)
        for p in sorted(args.gen_records_dir.glob("gen_*.json")):
            shutil.copy(p, d / "gen_records" / p.name)
    logger.info("fixture: %d gen records copied into %s and %s",
                len(list(sub_dir.glob("gen_records/gen_*.json"))), sub_dir, wrk_dir)

    py = sys.executable
    repo = Path(__file__).resolve().parent.parent

    # ── path 1: the original two-subprocess pipeline ────────────────────────
    t0 = time.time()
    cmds = [
        [py, "-m", "pleroma.run_replay_b0", "--gen-dir", str(sub_dir),
         "--out-dir", str(sub_dir), "--preset", str(args.preset),
         "--model-path", str(args.model_path), "--calib-dir", str(args.calib_dir),
         "--discriminants", str(args.discriminants),
         "--kvrot-path", str(args.kvrot_path), "--save-raw", "none"],
        [py, "-m", "pleroma.extract_bins", "--gen-dirs", str(sub_dir),
         "--model-path", str(args.model_path), "--preset", str(args.preset),
         "--out", str(sub_dir / "bins.npz")],
    ]
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("subprocess path failed: %s", (proc.stdout + proc.stderr)[-2000:])
            return 1
    logger.info("subprocess path done in %.1fs", time.time() - t0)

    # ── path 2: the persistent worker ────────────────────────────────────────
    t0 = time.time()
    req = urllib.request.Request(
        args.worker_url.rstrip("/") + "/harvest",
        data=json.dumps({"loom_dir": str(wrk_dir)}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        blob = json.loads(resp.read())
    if not blob.get("ok"):
        logger.error("worker path failed: %s", blob)
        return 1
    logger.info("worker path done in %.1fs (worker reported replay=%.1fs bins=%.1fs)",
                time.time() - t0, blob.get("replay_s", -1), blob.get("bins_s", -1))

    # ── diff: signatures, bins, and the fitted map's OUTPUT ──────────────────
    from pleroma.loom_serve import LoomMap

    loom_map = LoomMap(args.map, Path(args.discriminants))

    gids = sorted(
        int(p.stem.split("_")[1]) for p in sub_dir.glob("gen_records/gen_*.json")
    )
    with np.load(sub_dir / "bins.npz", allow_pickle=True) as z:
        sub_bins_feat = np.asarray(z["features"], dtype=np.float64)
        sub_bins_gid = [int(x) for x in z["generation_id"]]
    with np.load(wrk_dir / "bins.npz", allow_pickle=True) as z:
        wrk_bins_feat = np.asarray(z["features"], dtype=np.float64)
        wrk_bins_gid = [int(x) for x in z["generation_id"]]
    sub_bins_of = {g: i for i, g in enumerate(sub_bins_gid)}
    wrk_bins_of = {g: i for i, g in enumerate(wrk_bins_gid)}

    rows: list[dict[str, Any]] = []
    all_ok = True
    for gid in gids:
        row: dict[str, Any] = {"generation_id": gid}
        try:
            with np.load(sub_dir / "signatures" / f"gen_{gid:03d}.npz", allow_pickle=True) as z:
                sub_sig = np.asarray(z["features"], dtype=np.float64)
            with np.load(wrk_dir / "signatures" / f"gen_{gid:03d}.npz", allow_pickle=True) as z:
                wrk_sig = np.asarray(z["features"], dtype=np.float64)
            sig_cos = cosine(sub_sig, wrk_sig)
            sig_max_abs = float(np.max(np.abs(sub_sig - wrk_sig)))
            row["sig_cosine"] = sig_cos
            row["sig_max_abs_diff"] = sig_max_abs

            sub_brow = sub_bins_feat[sub_bins_of[gid]]
            wrk_brow = wrk_bins_feat[wrk_bins_of[gid]]
            bins_cos = cosine(sub_brow, wrk_brow)
            bins_max_abs = float(np.max(np.abs(sub_brow - wrk_brow)))
            row["bins_cosine"] = bins_cos
            row["bins_max_abs_diff"] = bins_max_abs

            sub_lever = loom_map.lever_of(sub_sig, sub_brow)
            wrk_lever = loom_map.lever_of(wrk_sig, wrk_brow)
            lever_cos = [cosine(sub_lever[s], wrk_lever[s]) for s in range(sub_lever.shape[0])]
            lever_max_abs = float(np.max(np.abs(sub_lever - wrk_lever)))
            row["lever_cosine_per_site"] = lever_cos
            row["lever_max_abs_diff"] = lever_max_abs

            ok = (
                sig_cos > 1 - args.tol_cosine
                and bins_cos > 1 - args.tol_cosine
                and all(c > 1 - args.tol_cosine for c in lever_cos)
            )
            row["ok"] = ok
            all_ok = all_ok and ok
        except Exception as exc:  # noqa: BLE001 — recorded, not hidden
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            all_ok = False
        rows.append(row)
        logger.info("gen_%03d: sig_cos=%.8f bins_cos=%.8f lever_cos=%s ok=%s",
                    gid, row.get("sig_cosine", float("nan")),
                    row.get("bins_cosine", float("nan")),
                    row.get("lever_cosine_per_site"), row["ok"])

    logger.info("CORRECTNESS %s — %d/%d gens matched within cosine tol %.0e",
                "PASSED" if all_ok else "FAILED",
                sum(1 for r in rows if r["ok"]), len(rows), args.tol_cosine)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"tol_cosine": args.tol_cosine, "all_ok": all_ok,
                                        "rows": rows}, indent=2))
        logger.info("wrote %s", args.out)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
