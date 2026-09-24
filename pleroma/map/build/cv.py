"""v1a fit: the registered analysis, FROZEN (its report carries ``token: v1a-fit-001``).

Train the deployed object: fan-relative signatures -> member-vs-fan-mean
hidden displacements. Contrast in, contrast out: the lever is worn as a
member's difference from its fan, so the map is fit on exactly that object.

    input   dz_i = z_i - mean_{j!=i in fan} z_j     (registered; raw z ablation)
    target  d_i  = h_i - mean_{j!=i in fan} h_j     (raw; per-site-norm variant)

PRIMARY (registered): within-fan member retrieval top-1 at the FIXED operating
point (lambda=1e4, rank 64 — v0a-wide's shipped point, cleanest one-variable
comparison; grid is descriptive), vs the frozen baseline = the SHELF wide
map's zero-training projection (map factors applied to [z|bins], fan
contrast — mu_in/mu_y cancel in the contrast; cosine metrics are invariant
to the leave-one-out vs full-mean scalar; the pairs z IS z_corpus and enters
the map unchanged — `shelf_baseline_pred` says why it must not be
standardized twice). Paired by fan, sign-flip permutation (`N_PERMS` =
20,000), one-sided. Secondary fairness baseline: CV-refit 2-means ridge under the
SAME folds (the shelf map saw every fan in training; if v1a beats it anyway
the primary is conservative).

Retrieval candidates are ALWAYS the raw measured deltas, for every variant.
All predictions are held out: grouped 5-fold CV by prompt_id.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pleroma.map.build.join import (  # noqa: F401 — re-exported for callers
    FAN_SOURCE_LEVER,
    FAN_SOURCES,
    MIN_FAN,
    build_target,
    cosine_rows,
    fan_center,
    join_bank,
    load_hiddens,
    retrieval,
)
from pleroma.stats.folds import make_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v1a_fit")

LAMBDAS = (1e3, 1e4, 3e4)
RANKS = (0, 16, 32, 64, 128, 256)
PRIMARY = {"input": "dz", "target": "raw", "lambda": 1e4, "rank": 64}
FOLDS = 5
FOLD_SEED = 20260921
PERM_SEED = 20260921
N_PERMS = 20000
#: Below this the held-out figures are noisy; warned, not refused (policy 5).
MIN_CV_ROWS = 1000



def cv_sweep(
    x: np.ndarray, y: np.ndarray, fold_idx: np.ndarray, lambdas, ranks, device
) -> dict[tuple[float, int], np.ndarray]:
    """Held-out predictions for every (lambda, rank); one solve+SVD per (fold, lambda)."""
    import torch

    preds = {(lam, r): np.empty_like(y, dtype=np.float32) for lam in lambdas for r in ranks}
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    d = x.shape[1]
    eye = torch.eye(d, dtype=torch.float64, device=device)
    for f in range(FOLDS):
        te = np.nonzero(fold_idx == f)[0]
        tr = np.nonzero(fold_idx != f)[0]
        tr_t = torch.as_tensor(tr, device=device)
        te_t = torch.as_tensor(te, device=device)
        mu_x = xt[tr_t].mean(dim=0)
        mu_y = yt[tr_t].mean(dim=0)
        xtr = xt[tr_t] - mu_x
        ytr = yt[tr_t] - mu_y
        xte = xt[te_t] - mu_x
        gram = xtr.T @ xtr
        xty = xtr.T @ ytr
        for lam in lambdas:
            w = torch.linalg.solve(gram + lam * eye, xty)  # [D, T]
            u, s, vt = torch.linalg.svd(w, full_matrices=False)
            for r in ranks:
                if r <= 0 or r >= min(w.shape):
                    p = xte @ w + mu_y
                else:
                    p = ((xte @ u[:, :r]) * s[:r]) @ vt[:r] + mu_y
                preds[(lam, r)][te] = p.float().cpu().numpy()
        logger.info("fold %d done (%d test rows)", f, te.size)
    return preds


def check_pairs_are_z_corpus(meta: dict) -> str | None:
    """Refuse a pairs dir whose ``z`` is not already the shelf map's z_corpus.

    The shelf map's input contract is ``[z_corpus | bins_std]`` and the wide
    map fit only fits on a pairs dir whose ``z`` IS z_corpus
    (`pleroma.map.build.shelf.compute_shelf` recomputes
    ``(z_full - v3_mu)/v3_sd`` and refuses unless that matches ``loaded.z`` to
    `pleroma.map.build.shelf.Z_VERIFY_TOL`). That holds exactly when
    `pleroma.map.build.pairs` ran with ``--standardize corpus``. Returns an error string
    to refuse on, or None. A manifest too old to record the flag is let
    through with a warning (every banked v1a pairs dir records it).
    """
    params = meta.get("params") if isinstance(meta, dict) else None
    std = params.get("standardize") if isinstance(params, dict) else None
    if std is None:
        logger.warning("pairs manifest does not record params.standardize — "
                       "ASSUMING z is already z_corpus (the shelf map's input "
                       "space), as the wide map fit requires")
        return None
    if std != "corpus":
        return (f"pairs dir was built with --standardize {std!r}; the shelf "
                "map's input is z_corpus (pleroma.map.build.pairs --standardize corpus), so "
                "its baseline cannot be computed from this z without re-deriving "
                "z_corpus — refusing rather than guessing")
    return None


def shelf_baseline_pred(
    z_corpus: np.ndarray,
    bins: np.ndarray,
    shelf: dict[str, np.ndarray],
    fans: list[np.ndarray],
) -> np.ndarray:
    """The frozen primary baseline: the shelf map's zero-training projection.

    ``z_corpus`` is ``load_pairs(...).z`` — ALREADY corpus-standardized by
    ``pleroma.map.build.pairs --standardize corpus`` with exactly the map's banked
    ``v3_mu/v3_sd`` (`pleroma.map.build.shelf.compute_shelf` verifies this), so it enters the
    map UNCHANGED. Only the bins block is standardized here, with the map's
    banked ``bins_mu/bins_sd``, because the bins npz holds raw features.

    ★ DO NOT standardize ``z_corpus`` again with ``(z - v3_mu)/v3_sd``.
    Fan-centering cancels the shift, so the damage would be a per-coordinate
    rescaling of the fan-centered input by 1/v3_sd: the right directions with
    miscalibrated weights, which understates the shelf map (baseline top-1
    .6915 -> .6428 at 3B, .7300 -> .6672 at 8B, .9032 -> .9030 for model C)
    and so overstates v1a's margin over it. v1a's own input does not go
    through this path.
    """
    b_mu = np.asarray(shelf["bins_mu"], dtype=np.float64)
    b_sd = np.asarray(shelf["bins_sd"], dtype=np.float64)
    b_dead = np.asarray(shelf["bins_dead"], dtype=bool)
    z_corpus = np.asarray(z_corpus, dtype=np.float64)
    v3_mu = np.asarray(shelf["v3_mu"])
    if z_corpus.shape[1] != v3_mu.size:
        raise ValueError(f"z has {z_corpus.shape[1]} dims but the shelf map's "
                         f"z_corpus block is {v3_mu.size}-d")
    b_std = (np.asarray(bins, dtype=np.float64) - b_mu) / np.where(b_dead, 1.0, b_sd)
    b_std[:, b_dead] = 0.0
    x_map = np.hstack([z_corpus, b_std])
    wu = np.asarray(shelf["W_U"], dtype=np.float64)
    ws = np.asarray(shelf["W_S"], dtype=np.float64)
    wvt = np.asarray(shelf["W_Vt"], dtype=np.float64)
    if x_map.shape[1] != wu.shape[0]:
        raise ValueError(f"shelf input is {x_map.shape[1]}-d but W_U has "
                         f"{wu.shape[0]} rows")
    return fan_center(x_map, fans) @ (wu * ws[None, :]) @ wvt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--levers", type=Path, required=True)
    ap.add_argument("--hiddens", type=Path, nargs="+", required=True)
    ap.add_argument("--bins", type=Path, required=True)
    ap.add_argument("--shelf-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fan-source", choices=FAN_SOURCES, default=FAN_SOURCE_LEVER,
                    help="lever_group (the registered join) or member_fan (the "
                         "served map's join: every (corpus, prompt, wave) fan, "
                         "lever or not)")
    args = ap.parse_args()

    from pleroma.map.build.hiddens import corpus_key

    lev = np.load(args.levers, allow_pickle=True)
    m_corpus = [str(x) for x in lev["member_corpus_keys"]]
    m_genid = np.asarray(lev["member_generation_ids"], dtype=int)
    m_group = np.asarray(lev["member_group_index"], dtype=int)

    # ── join: THE v1a join (shared with the export) ───────────────────────────
    try:
        bank = join_bank(args.pairs_dir, args.levers, list(args.hiddens),
                         fan_source=args.fan_source)
    except ValueError as exc:
        logger.error("join refused: %s", exc)
        return 1
    n = bank.n
    if n < MIN_CV_ROWS:
        logger.warning(
            "only %d rows joined — below %d the held-out retrieval is noisy; "
            "reported, not refused", n, MIN_CV_ROWS)
    z, h, grp, keys, fans = bank.z, bank.h, bank.grp, bank.keys, bank.fans
    prompt_arr = bank.prompts
    fan_sizes = np.array([s_.size for s_ in fans])
    logger.info("%d fans (sizes %d-%d, mean %.2f), %d gens, fan source %s",
                len(fans), fan_sizes.min(), fan_sizes.max(), fan_sizes.mean(), n,
                args.fan_source)

    # ── targets and inputs ─────────────────────────────────────────────────────
    d_raw = fan_center(h, fans)  # measured LOO deltas
    d_sitenorm = build_target(d_raw, "sitenorm", len(bank.sites))
    dz = fan_center(z, fans)
    inputs = {"dz": dz, "rawz": z}
    targets = {"raw": d_raw, "sitenorm": d_sitenorm}

    # ── folds: grouped by prompt (both waves together); refuses a split fan ───
    fold_idx = make_folds(prompt_arr, fans, n_folds=FOLDS, seed=FOLD_SEED)

    # ── shelf-map baseline (frozen primary baseline) ───────────────────────────
    refusal = check_pairs_are_z_corpus(bank.meta)
    if refusal:
        logger.error("%s", refusal)
        return 1
    with np.load(args.shelf_map, allow_pickle=True) as m:
        shelf = {k: np.asarray(m[k]) for k in (
            "v3_mu", "v3_sd", "v3_dead", "bins_mu", "bins_sd", "bins_dead",
            "W_U", "W_S", "W_Vt")}
    ex = np.load(args.bins, allow_pickle=True)
    ex_run = [str(x) for x in ex["run_dir"]]
    ex_gid = np.asarray(ex["generation_id"], dtype=int)
    ex_of = {(corpus_key(Path(ex_run[k]).name), int(ex_gid[k])): k for k in range(len(ex_gid))}
    bins_idx = []
    for key in keys:
        k = ex_of.get(key)
        if k is None:
            logger.error("gen %s missing from bins — refusing", key)
            return 1
        bins_idx.append(k)
    bins = np.asarray(ex["features"], dtype=np.float64)[np.asarray(bins_idx)]
    try:
        base_pred = shelf_baseline_pred(z, bins, shelf, fans)  # z standardized ONCE
    except ValueError as exc:
        logger.error("shelf baseline refused: %s", exc)
        return 1
    base_hits = retrieval(base_pred, d_raw, fans)
    logger.info("shelf-map baseline retrieval top-1: %.4f", base_hits.mean())

    # ── sweep all variants ─────────────────────────────────────────────────────
    report: dict = {
        "stage": "v1a_fit", "token": "v1a-fit-001",
        "n_gens": int(n), "n_fans": len(fans),
        "fan_size_mean": float(fan_sizes.mean()),
        "chance_top1": float(np.mean(1.0 / fan_sizes.repeat(fan_sizes))),
        "folds": FOLDS, "fold_seed": FOLD_SEED,
        "fan_source": args.fan_source,
        "primary_point": PRIMARY,
        "baseline_shelf": {
            "retrieval_top1": float(base_hits.mean()),
            "map": str(args.shelf_map),
            "input": "[z_corpus (pairs z, used as-is) | bins_std] — z "
                     "standardized ONCE, never a second time",
        },
        "grid": {},
    }
    primary_pred = None
    for in_name, xin in inputs.items():
        for t_name, y in targets.items():
            logger.info("=== variant input=%s target=%s ===", in_name, t_name)
            preds = cv_sweep(xin, y, fold_idx, LAMBDAS, RANKS, args.device)
            for (lam, r), pr in preds.items():
                hits = retrieval(pr, d_raw, fans)
                cos = cosine_rows(pr.astype(np.float64), y)
                report["grid"][f"{in_name}|{t_name}|lam{lam:g}|r{r}"] = {
                    "retrieval_top1": round(float(hits.mean()), 4),
                    "pred_cos_vs_own_targets": round(float(cos.mean()), 4),
                }
                if (in_name == PRIMARY["input"] and t_name == PRIMARY["target"]
                        and lam == PRIMARY["lambda"] and r == PRIMARY["rank"]):
                    primary_pred = pr
                    primary_hits = hits

    # ── secondary fairness baseline: CV-refit 2-means ridge, same folds ───────
    # Needs a 2-means lever for every row; under member_fan the no-lever fans
    # have none, so the baseline is skipped there and SAYS so.
    j_of_key = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}
    jj = np.asarray([j_of_key[key] for key in keys])
    v0cv_hits = None
    if (m_group[jj] < 0).any():
        report["baseline_v0_cv"] = {
            "skipped": f"{int((m_group[jj] < 0).sum())} rows have no 2-means "
                       "lever (fan_source=member_fan) — the v0 baseline is "
                       "undefined on them"}
    else:
        levers_flat = np.asarray(lev["levers"], dtype=np.float64).reshape(
            len(lev["levers"]), -1)
        m_sign = 1.0 - 2.0 * np.asarray(lev["member_clusters"], dtype=np.float64)
        y_v0 = m_sign[jj, None] * levers_flat[m_group[jj]]
        v0_preds = cv_sweep(z, y_v0, fold_idx, (PRIMARY["lambda"],),
                            (PRIMARY["rank"],), args.device)
        v0_pred = v0_preds[(PRIMARY["lambda"], PRIMARY["rank"])]
        v0_contrast = fan_center(v0_pred.astype(np.float64), fans)
        v0cv_hits = retrieval(v0_contrast, d_raw, fans)
        report["baseline_v0_cv"] = {"retrieval_top1": round(float(v0cv_hits.mean()), 4)}

    # ── registered test: v1a primary vs shelf baseline, sign-flip by fan ──────
    def fan_diff(h1: np.ndarray, h0: np.ndarray) -> np.ndarray:
        return np.asarray([h1[s].mean() - h0[s].mean() for s in fans])

    rng_p = np.random.default_rng(PERM_SEED)
    diffs = fan_diff(primary_hits, base_hits)
    obs = diffs.mean()
    flips = rng_p.choice([-1.0, 1.0], size=(N_PERMS, diffs.size))
    null = (flips * diffs[None, :]).mean(axis=1)
    p_val = float(((null >= obs).sum() + 1) / (N_PERMS + 1))
    report["registered_test"] = {
        "statistic": "mean over fans of (v1a top1 - shelf top1)",
        "observed": round(float(obs), 4),
        "v1a_top1": round(float(primary_hits.mean()), 4),
        "shelf_top1": round(float(base_hits.mean()), 4),
        "n_perms": N_PERMS, "one_sided_p": round(p_val, 5),
        "verdict": "PASS" if (obs > 0 and p_val < 0.05) else "FAIL",
    }
    # same test vs the fairness baseline (secondary, descriptive)
    if v0cv_hits is not None:
        diffs2 = fan_diff(primary_hits, v0cv_hits)
        null2 = (rng_p.choice([-1.0, 1.0], size=(N_PERMS, diffs2.size))
                 * diffs2[None, :]).mean(axis=1)
        report["secondary_vs_v0cv"] = {
            "observed": round(float(diffs2.mean()), 4),
            "v0cv_top1": round(float(v0cv_hits.mean()), 4),
            "one_sided_p": round(float(((null2 >= diffs2.mean()).sum() + 1)
                                       / (N_PERMS + 1)), 5),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "v1a_predicted_deltas.npz",
        pred=primary_pred.astype(np.float32),
        measured=d_raw.astype(np.float32),
        corpus_keys=np.asarray([k[0] for k in keys], dtype=np.str_),
        generation_ids=np.asarray([k[1] for k in keys], dtype=np.int64),
        group_index=grp, fold_idx=fold_idx,
        sites=np.asarray(lev["sites"], dtype=int),
    )
    (args.out_dir / "v1a_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))
    best = sorted(report["grid"].items(), key=lambda kv: -kv[1]["retrieval_top1"])[:8]
    print("top grid cells by retrieval:")
    for k, v in best:
        print(f"  {k}: {v}")
    print("wrote", args.out_dir / "v1a_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
