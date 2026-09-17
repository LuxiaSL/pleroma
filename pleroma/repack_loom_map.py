"""Repack a full-W loom_map.npz into the factored ship-size form.

fit_loom_map rank-truncates W at fit time, so W = U_r S_r V_rᵀ exactly (up to
float32 roundoff). Storing the factors instead of the 3133x12288 product takes
the artifact from ~140MB to under 1MB with no semantic change: LoomMap
reconstructs W from the factors at load and uses V_rᵀ directly as the code
basis (it no longer re-derives it by SVD, which is also faster and gauge-stable).

The script verifies before writing: reconstruction error on W itself, and
end-to-end agreement of (lever, code) between the original and repacked
pipelines on random probe inputs, mirroring LoomMap.lever_of exactly. It
refuses to write if either check fails.

    python -m pleroma.repack_loom_map \\
        --in outputs/loom/loom_map.npz --out data/loom/loom_map_3b.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("repack_loom_map")

#: end-to-end agreement gates (float32 storage; the map's own outputs are
#: reported to 4 decimals, so 1e-3 absolute on codes is already generous)
W_REL_TOL = 1e-5
CODE_ABS_TOL = 1e-3
LEVER_REL_TOL = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=64)
    args = parser.parse_args()

    with np.load(args.inp, allow_pickle=True) as npz:
        keys = list(npz.files)
        if "W" not in keys:
            logger.error("%s has no full 'W' — nothing to repack (keys: %s)",
                         args.inp, keys)
            return 1
        data = {k: np.asarray(npz[k]) for k in keys}
    meta = json.loads(str(data["meta"]))
    rank = int(meta["rank"])
    w = data["W"].astype(np.float64)
    logger.info("W %s, rank %d (from meta)", w.shape, rank)

    u, s, vt = np.linalg.svd(w, full_matrices=False)
    u_r = u[:, :rank].astype(np.float32)
    s_r = s[:rank].astype(np.float32)
    vt_r = vt[:rank].astype(np.float32)
    tail = float(s[rank]) if len(s) > rank else 0.0
    logger.info("singular values kept: %s ... first dropped: %.3e (should be "
                "~0: W was already rank-truncated)",
                [round(float(x), 2) for x in s_r], tail)

    w_rec = (u_r.astype(np.float64) * s_r.astype(np.float64)) @ vt_r.astype(np.float64)
    rel = float(np.linalg.norm(w_rec - w) / np.linalg.norm(w))
    logger.info("W reconstruction relative error: %.3e (gate %.0e)", rel, W_REL_TOL)
    if rel > W_REL_TOL:
        logger.error("reconstruction gate FAILED — refusing to write")
        return 1

    # ── end-to-end probe: mirror LoomMap.lever_of on both forms ──────────────
    rng = np.random.default_rng(0)
    mu_in = data["mu_in"].astype(np.float64)
    mu_y = data["mu_y"].astype(np.float64)
    vt_orig = np.linalg.svd(w, full_matrices=False)[2][:rank]  # original code basis
    # Sign gauge: SVD columns are sign-arbitrary between two computations of
    # the same matrix; align probe comparison per-axis (the served basis is
    # whichever the server loaded — consistent within a server lifetime).
    axis_sign = np.sign(np.sum(vt_orig * vt_r.astype(np.float64), axis=1))
    if (axis_sign == 0).any():
        logger.error("degenerate axis alignment — refusing to write")
        return 1
    worst_code, worst_lever = 0.0, 0.0
    for _ in range(args.probes):
        x = rng.standard_normal(w.shape[0]) * 1.5
        flat_a = (x - mu_in) @ w + mu_y
        flat_b = (x - mu_in) @ w_rec + mu_y
        code_a = vt_orig @ (flat_a - mu_y)
        code_b = (vt_r.astype(np.float64) @ (flat_b - mu_y)) * axis_sign
        worst_code = max(worst_code, float(np.abs(code_a - code_b).max()))
        worst_lever = max(worst_lever, float(
            np.linalg.norm(flat_a - flat_b) / np.linalg.norm(flat_a - mu_y)))
    logger.info("probe agreement over %d draws: code max|Δ| %.3e (gate %.0e), "
                "lever rel %.3e (gate %.0e)",
                args.probes, worst_code, CODE_ABS_TOL, worst_lever, LEVER_REL_TOL)
    if worst_code > CODE_ABS_TOL or worst_lever > LEVER_REL_TOL:
        logger.error("probe gate FAILED — refusing to write")
        return 1

    out_data = {k: v for k, v in data.items() if k != "W"}
    out_data["W_U"], out_data["W_S"], out_data["W_Vt"] = u_r, s_r, vt_r
    meta["repacked_from"] = {"path": str(args.inp), "w_rel_err": rel,
                             "probe_code_max_abs": worst_code}
    out_data["meta"] = np.array(json.dumps(meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out_data)
    logger.info("DONE -> %s (%.2f MB, was %.2f MB)", args.out,
                args.out.stat().st_size / 1e6, args.inp.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
