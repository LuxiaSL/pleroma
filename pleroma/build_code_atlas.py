"""The CODE ATLAS — render the rank-8 code space and place the bank in it.

The map W = U S Vᵀ (rank 8). Two kinds of object live in ONE 8-dim space:
  - a candidate/future: c = S · Uᵀ (x − mu_in)      (input side, pre-expansion)
  - a banked TRUE lever: c = Vᵀ (lever − mu_y)       (output side)
Predicted levers lie exactly in mu_y + span(V), so these coordinates are the
entire content of what the map can say — the 8 numbers ARE the code.

This script: recovers U/S/V from the stored (already rank-truncated) W,
projects all banked levers (BOTH signs — a lever and its negation are both
real dispositions), and reports per-axis structure: variance, class
separation (ANOVA-style eta^2 over prompt classes), wave consistency
(orig-vs-repl twin coordinate correlation — an axis that flips between waves
of the same prompt is measuring noise), and the extreme groups per axis (the
raw material for HONEST naming — axes that earn no name stay anonymous).

Writes atlas.npz (V, U, S, bank coords, quantiles for UI normalization) and
atlas_report.json. Naming happens by eye AFTER reading the report, in
axes.json — never automatically.

    python -m pleroma.build_code_atlas \\
        --map outputs/loom/loom_map.npz \\
        --levers outputs/b15_levers_w3/levers.npz \\
        --out-dir outputs/loom/atlas
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
logger = logging.getLogger("build_code_atlas")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--levers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.map, allow_pickle=True) as npz:
        W = np.asarray(npz["W"], dtype=np.float64)
        mu_y = np.asarray(npz["mu_y"], dtype=np.float64)
        meta = json.loads(str(npz["meta"]))
    rank = int(meta["rank"])
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank]
    logger.info("recovered rank-%d factors: U %s, S %s, V %s (S: %s)",
                rank, U.shape, S.shape, Vt.T.shape,
                [round(float(s), 2) for s in S])

    lev = np.load(args.levers, allow_pickle=False)
    levers = np.asarray(lev["levers"], dtype=np.float64)  # [G, sites, D]
    flat = levers.reshape(levers.shape[0], -1)
    pids = [str(x) for x in lev["group_prompt_ids"]]
    waves = [str(x) for x in lev["group_waves"]]
    classes = [str(x) for x in lev["group_prompt_classes"]]
    ok = np.isfinite(flat).all(axis=1)

    coords = (flat[ok] - mu_y) @ Vt.T  # [G_ok, rank]
    kept = [i for i, o in enumerate(ok) if o]
    logger.info("projected %d/%d banked levers into the code space",
                len(kept), len(pids))

    # ── GAUGE FIX (first atlas pass found it): a lever's sign is which basin
    # 2-means happened to call cluster 0 — arbitrary PER (prompt, wave). A
    # disposition is an UNSIGNED axis; the bank's raw points carry coin-flip
    # signs (twins landed antipodally; all-negative twin_r was the tell).
    # Alignment: greedy, seeded by the largest-norm code; each lever (in
    # descending norm order) flips to positive cosine with the running mean
    # of the already-aligned set. Twins are aligned to each other FIRST.
    # Flips are recorded; the gauge is a convention, not a discovery.
    pid_arr0 = np.array([pids[i] for i in kept])
    wave_arr0 = np.array([waves[i] for i in kept])
    signs = np.ones(len(kept))
    for p in sorted(set(pid_arr0)):
        idx = np.nonzero(pid_arr0 == p)[0]
        if len(idx) == 2:
            a_, b_ = idx
            if float(coords[a_] @ coords[b_]) < 0:
                signs[b_] = -1.0  # align repl (second) to orig within the twin
    coords = coords * signs[:, None]
    order = np.argsort(-np.linalg.norm(coords, axis=1))
    running = coords[order[0]].copy()
    flips = int((signs < 0).sum())
    for j in order[1:]:
        if float(coords[j] @ running) < 0:
            coords[j] = -coords[j]
            signs[j] *= -1.0
            flips += 1
        running += coords[j]
    logger.info("gauge: %d of %d levers sign-flipped into the common convention",
                int((signs < 0).sum()), len(kept))

    # In-space energy: how much of each lever the 8 dims capture at all.
    recon = coords @ Vt + mu_y
    in_frac = 1.0 - (np.linalg.norm(flat[ok] - recon, axis=1)
                     / np.linalg.norm(flat[ok] - mu_y, axis=1)) ** 2

    report: dict[str, object] = {
        "rank": rank, "n_groups": len(kept),
        "singular_values": [round(float(s), 3) for s in S],
        "in_space_energy": {
            "mean": round(float(in_frac.mean()), 3),
            "median": round(float(np.median(in_frac)), 3),
            "note": "fraction of centered lever energy inside span(V) — how "
                    "much of a TRUE disposition the 8-dim code can even say",
        },
        "axes": [],
    }

    cls_arr = np.array([classes[i] for i in kept])
    pid_arr = np.array([pids[i] for i in kept])
    wave_arr = np.array([waves[i] for i in kept])
    uniq_cls = sorted(set(cls_arr))
    twins = sorted({p for p in pid_arr
                    if {"orig", "repl"} <= set(wave_arr[pid_arr == p])})

    for a in range(rank):
        v = coords[:, a]
        grand = v.mean()
        ss_tot = float(((v - grand) ** 2).sum())
        ss_between = float(sum(
            (v[cls_arr == c].mean() - grand) ** 2 * (cls_arr == c).sum()
            for c in uniq_cls))
        eta2 = ss_between / ss_tot if ss_tot > 0 else 0.0
        tw = []
        for p in twins:
            a_ = v[(pid_arr == p) & (wave_arr == "orig")]
            b_ = v[(pid_arr == p) & (wave_arr == "repl")]
            if len(a_) and len(b_):
                tw.append((float(a_[0]), float(b_[0])))
        twin_r = (float(np.corrcoef([x for x, _ in tw], [y for _, y in tw])[0, 1])
                  if len(tw) >= 3 else None)
        order = np.argsort(v)
        lo = [f"{pid_arr[i]}|{wave_arr[i]}({cls_arr[i][:2]})" for i in order[:4]]
        hi = [f"{pid_arr[i]}|{wave_arr[i]}({cls_arr[i][:2]})" for i in order[-4:]]
        by_class = {c: round(float(v[cls_arr == c].mean()), 3) for c in uniq_cls}
        report["axes"].append({
            "axis": a, "sv": round(float(S[a]), 3),
            "coord_std": round(float(v.std()), 3),
            "class_eta2": round(eta2, 3),
            "class_means": by_class,
            "twin_consistency_r": (round(twin_r, 3) if twin_r is not None else None),
            "low_extreme": lo, "high_extreme": hi,
            "quantiles": [round(float(q), 3)
                          for q in np.quantile(v, [.05, .25, .5, .75, .95])],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "atlas.npz",
        U=U.astype(np.float32), S=S.astype(np.float32), Vt=Vt.astype(np.float32),
        mu_y=mu_y.astype(np.float32),
        bank_coords=coords.astype(np.float32),
        bank_prompt_ids=pid_arr, bank_waves=wave_arr, bank_classes=cls_arr,
        in_space_energy=in_frac.astype(np.float32),
        gauge_signs=signs,
    )
    report["gauge"] = {
        "n_flipped": int((signs < 0).sum()),
        "note": "dispositions are unsigned axes; signs here are a convention "
                "(twin-aligned then greedy norm-ordered), not a discovery",
    }
    report["groups"] = [
        {"id": f"{pid_arr[i]}|{wave_arr[i]}", "class": str(cls_arr[i]),
         "coords": [round(float(c), 4) for c in coords[i]]}
        for i in range(len(kept))
    ]
    (args.out_dir / "atlas_report.json").write_text(json.dumps(report, indent=1))
    logger.info("DONE -> %s (read atlas_report.json BY EYE before naming "
                "anything in axes.json)", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
