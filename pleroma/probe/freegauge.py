"""The free gauge, stage 2: calibrate geometry against the banked fable eye.

The calibration protocol was fixed before any gauge number existed, and this
module executes it exactly as frozen:

  code-nr (``M1_code_nr``) — normalized rank of the target future among the SAME-PASS future
  codes by Euclidean distance in 32-d code space. Cell value = mean base nr −
  mean steered nr, the same construction as fable's Δnr.

  G1: within-conversation Spearman over the 16 gate cells, positive 3/3 AND
      mean ρ ≥ +0.40.
  G2: pooled AUC over 48 gate cells for movers (fable Δnr > own conversation's
      median) ≥ 0.70.
  G3: secondary, Spearman across the existence+rebaseline run-level cells > 0.

z-nr (``M2_z_nr``, cosine on float16 z) is computed ONLY if code-nr fails —
the protocol's conditional, honored in code so exploratory numbers never
exist to tempt.

Ties in distance rank are broken by future index (deterministic, declared).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("freegauge_calibration")

GATE_CONVS: tuple[str, ...] = ("secondhand", "trainlate", "window")
SHARED_DOSES: tuple[float, ...] = (0.35, 0.50)
G1_RHO_BAR = 0.40
G2_AUC_BAR = 0.70


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── the gauge's own normalized rank ──────────────────────────────────────────

def euclid(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_dist(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        raise ValueError("cosine distance of a zero vector")
    return 1.0 - sum(x * y for x, y in zip(a, b)) / (na * nb)


def gauge_nr(vec: list[float], future_vecs: list[list[float]], target: int,
             dist: Callable[[list[float], list[float]], float]) -> float:
    """Normalized rank (0 = nearest, /(k-1)) of `target` by distance; ties by
    index, declared in the module docstring."""
    k = len(future_vecs)
    if k < 2:
        raise ValueError("a fan of one is not a fan")
    if not 0 <= target < k:
        raise ValueError(f"target {target} outside fan of {k}")
    order = sorted(range(k), key=lambda j: (dist(vec, future_vecs[j]), j))
    return order.index(target) / (k - 1)


def cell_delta(base_vals: list[float], steered_vals: list[float]) -> float:
    if not base_vals or not steered_vals:
        raise ValueError("empty side in a cell")
    return (sum(base_vals) / len(base_vals)
            - sum(steered_vals) / len(steered_vals))


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC: P(score(pos) > score(neg)), ties at 0.5."""
    if not pos or not neg:
        raise ValueError("AUC needs both classes")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def z_of(row: dict[str, Any]) -> list[float]:
    import numpy as np
    raw = base64.b64decode(row["z_b64"])
    z = np.frombuffer(raw, dtype=np.float16).astype(np.float64)
    if z.size != int(row["z_dim"]):
        raise ValueError(f"{row['sid']}: z_dim mismatch")
    return list(z)


# ── fable's per-cell Δnr from banked rankings ────────────────────────────────

def fable_nr(rec: dict[str, Any], index: int) -> float:
    idx_of = {letter: int(i) for letter, i in rec["letter_to_index"].items()}
    positions = [idx_of[letter] for letter in rec["ranking"]]
    if index not in positions:
        raise ValueError(f"{rec.get('call_id')}: index {index} not ranked")
    return positions.index(index) / (len(positions) - 1)


def gate_cells_fable(rankings: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    """(conv, tidx) -> fable Δnr, from the gate's own base arm."""
    out: dict[tuple[str, int], float] = {}
    for conv in GATE_CONVS:
        base = [r for r in rankings if r["prompt_id"] == conv and r["arm"] == "base"]
        for t in range(16):
            steered = [r for r in rankings
                       if r["prompt_id"] == conv and r["arm"] == "sweep_ctr"
                       and int(r["prescribed_index"]) == t]
            if not steered:
                continue
            out[(conv, t)] = cell_delta([fable_nr(r, t) for r in base],
                                        [fable_nr(r, t) for r in steered])
    return out


def gate_cells_gauge(rows: list[dict[str, Any]],
                     futures: dict[str, list[list[float]]],
                     vec_of: Callable[[dict[str, Any]], list[float]],
                     dist: Callable[[list[float], list[float]], float],
                     ) -> tuple[dict[tuple[str, int], float],
                                dict[tuple[str, int], float]]:
    """(conv, tidx) -> gauge Δnr, plus the per-cell steered SD (probe pricing)."""
    deltas: dict[tuple[str, int], float] = {}
    sds: dict[tuple[str, int], float] = {}
    for conv in GATE_CONVS:
        fut = futures[conv]
        base_rows = [r for r in rows if r["source"] == "gate"
                     and r["prompt_id"] == conv and r["arm"] == "base"]
        for t in range(16):
            steered = [r for r in rows if r["source"] == "gate"
                       and r["prompt_id"] == conv and r["arm"] == "sweep_ctr"
                       and r["target_index"] == t]
            if not steered or not base_rows:
                continue
            b = [gauge_nr(vec_of(r), fut, t, dist) for r in base_rows]
            s = [gauge_nr(vec_of(r), fut, t, dist) for r in steered]
            deltas[(conv, t)] = cell_delta(b, s)
            m = sum(s) / len(s)
            sds[(conv, t)] = (sum((x - m) ** 2 for x in s) / (len(s) - 1)) ** 0.5 \
                if len(s) > 1 else 0.0
    return deltas, sds


def run_level_cells(rows: list[dict[str, Any]],
                    futures: dict[str, list[list[float]]],
                    vec_of: Callable[[dict[str, Any]], list[float]],
                    dist: Callable[[list[float], list[float]], float],
                    rankings_by: dict[str, list[dict[str, Any]]],
                    ) -> dict[str, dict[str, float]]:
    """existence + rebaseline cells: {cell_key: {gauge, fable}}.

    Base side for BOTH sources is the existence run's base arm. Fable base comes
    from the rank_readout rankings.
    """
    out: dict[str, dict[str, float]] = {}
    base_g: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["source"] == "existence" and r["arm"] == "base":
            base_g.setdefault(r["prompt_id"], []).append(r)
    for source, steered_arm, rank_key in (("existence", "prescribed", "existence"),
                                          ("rebaseline", "contrastive", "rebaseline")):
        by_conv: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if r["source"] == source and r["arm"] == steered_arm:
                by_conv.setdefault(r["prompt_id"], []).append(r)
        for conv, srows in sorted(by_conv.items()):
            brows = base_g.get(conv, [])
            if not brows or conv not in futures:
                logger.warning("%s/%s: missing base or futures — skipped",
                               source, conv)
                continue
            t = int(srows[0]["target_index"])
            fut = futures[conv]
            gauge = cell_delta(
                [gauge_nr(vec_of(r), fut, t, dist) for r in brows],
                [gauge_nr(vec_of(r), fut, t, dist) for r in srows])
            franks = rankings_by[rank_key]
            fbase = [r for r in rankings_by["existence_base"]
                     if r["prompt_id"] == conv]
            fsteer = [r for r in franks if r["prompt_id"] == conv
                      and float(r["alpha"]) in SHARED_DOSES]
            fable = cell_delta([fable_nr(r, t) for r in fbase],
                               [fable_nr(r, t) for r in fsteer])
            out[f"{source}:{conv}"] = {"gauge": gauge, "fable": fable,
                                       "target_index": t}
    return out


def evaluate_metric(name: str, rows: list[dict[str, Any]],
                    futures: dict[str, list[list[float]]],
                    vec_of: Callable[[dict[str, Any]], list[float]],
                    dist: Callable[[list[float], list[float]], float],
                    fable_gate: dict[tuple[str, int], float],
                    rankings_by: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from scipy import stats

    gauge_gate, cell_sds = gate_cells_gauge(rows, futures, vec_of, dist)
    keys = sorted(set(gauge_gate) & set(fable_gate))
    per_conv: dict[str, Any] = {}
    rhos: list[float] = []
    for conv in GATE_CONVS:
        ck = [k for k in keys if k[0] == conv]
        g = [gauge_gate[k] for k in ck]
        f = [fable_gate[k] for k in ck]
        rho, p = stats.spearmanr(g, f)
        rhos.append(float(rho))
        med = sorted(f)[len(f) // 2]
        per_conv[conv] = {
            "n_cells": len(ck), "spearman": round(float(rho), 4),
            "p": round(float(p), 4), "fable_median": round(med, 4),
            "mean_steered_sd_per_cell": round(
                sum(cell_sds[k] for k in ck) / len(ck), 4),
        }
    g1_pass = all(r > 0 for r in rhos) and (sum(rhos) / len(rhos)) >= G1_RHO_BAR
    movers = [gauge_gate[k] for k in keys
              if fable_gate[k] > per_conv[k[0]]["fable_median"]]
    duds = [gauge_gate[k] for k in keys
            if fable_gate[k] <= per_conv[k[0]]["fable_median"]]
    g2_auc = auc(movers, duds)
    run_cells = run_level_cells(rows, futures, vec_of, dist, rankings_by)
    g3_rho, g3_p = stats.spearmanr([c["gauge"] for c in run_cells.values()],
                                   [c["fable"] for c in run_cells.values()])
    return {
        "metric": name,
        "G1": {"per_conversation": per_conv,
               "mean_rho": round(sum(rhos) / len(rhos), 4),
               "bar": G1_RHO_BAR, "PASS": bool(g1_pass)},
        "G2": {"auc": round(float(g2_auc), 4), "n_movers": len(movers),
               "n_duds": len(duds), "bar": G2_AUC_BAR,
               "PASS": bool(g2_auc >= G2_AUC_BAR)},
        "G3_secondary": {"spearman": round(float(g3_rho), 4),
                         "p": round(float(g3_p), 4),
                         "positive": bool(g3_rho > 0),
                         "n_cells": len(run_cells)},
        "gate_cells": {f"{c}|{t}": {"gauge": round(gauge_gate[(c, t)], 4),
                                    "fable": round(fable_gate[(c, t)], 4)}
                       for (c, t) in keys},
        "run_cells": run_cells,
        "PASS": bool(g1_pass and g2_auc >= G2_AUC_BAR),
    }


def main() -> int:  # noqa: C901 — one linear calibration procedure
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gauge", type=Path, required=True)
    ap.add_argument("--gate-rankings", type=Path, required=True)
    ap.add_argument("--existence-rankings", type=Path, required=True,
                    help="rank_readout rankings (base + prescribed)")
    ap.add_argument("--rebaseline-rankings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.gauge)
    futures_rows = [r for r in rows if r["kind"] == "future"]
    reply_rows = [r for r in rows if r["kind"] == "reply"]
    futures_code: dict[str, list[list[float]]] = {}
    banked_check: list[float] = []
    for pid in sorted({r["prompt_id"] for r in futures_rows}):
        fr = sorted([r for r in futures_rows if r["prompt_id"] == pid],
                    key=lambda r: int(r["target_index"]))
        if [int(r["target_index"]) for r in fr] != list(range(16)):
            raise ValueError(f"{pid}: incomplete future set in gauge file")
        futures_code[pid] = [list(map(float, r["code"])) for r in fr]
        for r in fr:
            if r.get("banked_code"):
                banked_check.append(euclid(list(map(float, r["code"])),
                                           list(map(float, r["banked_code"]))))

    exist_rank = load_jsonl(args.existence_rankings)
    rankings_by = {
        "gate": load_jsonl(args.gate_rankings),
        "existence": [r for r in exist_rank if r["arm"] == "prescribed"],
        "existence_base": [r for r in exist_rank if r["arm"] == "base"],
        "rebaseline": load_jsonl(args.rebaseline_rankings),
    }
    fable_gate = gate_cells_fable(rankings_by["gate"])
    logger.info("%d gate cells (fable), %d conversations with futures, "
                "%d reply rows", len(fable_gate), len(futures_code),
                len(reply_rows))

    code_of: Callable[[dict[str, Any]], list[float]] = (
        lambda r: list(map(float, r["code"])))
    m1 = evaluate_metric("M1_code_nr", reply_rows, futures_code, code_of,
                         euclid, fable_gate, rankings_by)

    verdict: dict[str, Any] = {
        "prereg": "PREREG-2026-09-20-freegauge.md",
        "M1": m1,
        "banked_vs_samepass_future_codes": {
            "n": len(banked_check),
            "mean_euclid": round(sum(banked_check) / len(banked_check), 4)
            if banked_check else None,
            "max_euclid": round(max(banked_check), 4) if banked_check else None,
        },
    }
    if not m1["PASS"]:
        logger.info("code-nr failed a bar — evaluating z-nr, the protocol's fallback")
        futures_z: dict[str, list[list[float]]] = {}
        for pid, _ in futures_code.items():
            fr = sorted([r for r in futures_rows if r["prompt_id"] == pid],
                        key=lambda r: int(r["target_index"]))
            futures_z[pid] = [z_of(r) for r in fr]
        verdict["M2"] = evaluate_metric("M2_z_nr", reply_rows, futures_z,
                                        z_of, cosine_dist, fable_gate,
                                        rankings_by)
    else:
        verdict["M2"] = "not evaluated — M1 passed (prereg conditional)"

    winner = ("M1" if m1["PASS"] else
              ("M2" if isinstance(verdict["M2"], dict)
               and verdict["M2"]["PASS"] else None))
    verdict["ADOPTED"] = winner
    verdict["decision"] = (
        f"{winner} adopted as the picker's in-loop readout; fable demoted to "
        "offline eval + periodic audit" if winner else
        "no gauge passed — fable stays in the loop for the interim demo; "
        "local-ranker program opens under its own prereg")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=1))
    logger.info("wrote %s | ADOPTED=%s", args.out, winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
