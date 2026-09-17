"""B1.5 L2 — can the signature find the (now proven-real) carrier?

Regresses corpus-standardized signatures z onto the cluster-contrast levers L1
validated (RESULTS-B15-L1). Per gen the target is s_i * lever_g flattened over
sites (s_i = +1 for cluster A members, -1 for cluster B, matching eval_levers'
matched-arm convention), so cosine(prediction, target) > 0 means "z placed this
trajectory on the correct side of its prompt's basin axis."

Grouped CV by (prompt, wave): a held-out prompt's lever was NEVER a training
target, so nothing about the prompt's own basin geometry leaks — this is the
question "does z carry where basins LIVE in state space, portably across
prompts," which is exactly the map g needed and (per RESULTS-B1) never found
through end-to-end NLL.

Baselines: (a) train-mean-lever predictor (the "generic lever direction" floor);
(b) shuffled-z refits (permute z across gens within train, refit, evaluate) —
the chance floor for every metric. Rank-truncated variants (SVD of the ridge
coefficient matrix) probe the bottleneck hypothesis at 8/16/32 dims.

CPU + numpy only. Run where the pairs dir lives::

    python -m pleroma.regress_levers \\
        --pairs-dir outputs/b15_pairs_all --levers outputs/b15_levers/levers.npz \\
        --out outputs/b15_levers/regress_report.json
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("regress_levers")

# Mirrors build_levers' corpus normalization for the four original run dirs;
# anything else (e.g. the w3_replay_s* shards) falls through to the SAME
# corpus_key() bank_mean_hiddens/build_levers use, so new corpora join without
# this table growing (2026-09-17).
CORPUS_OF_BASENAME = {
    "pilot_replay": "pilot",
    "wave2_replay": "wave2",
    "repl_replay_a": "repl_a",
    "repl_replay_b": "repl_b",
    "expB0_pilot": "pilot",
    "expB0_wave2": "wave2",
}


def corpus_of_run_dir(run_dir: str) -> str:
    from pleroma.bank_mean_hiddens import corpus_key

    base = Path(run_dir).name
    return CORPUS_OF_BASENAME.get(base, corpus_key(base))


def ridge_fit(z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """W: [D, T] for centered z [N, D], y [N, T]."""
    d = z.shape[1]
    a = z.T @ z + lam * np.eye(d, dtype=np.float64)
    return np.linalg.solve(a, z.T @ y)


def rank_truncate(w: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0 or rank >= min(w.shape):
        return w
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na > 0) & (nb > 0), na * nb, 1.0)
    return np.einsum("ij,ij->i", a, b) / denom


def cv_predictions(
    z: np.ndarray,
    y: np.ndarray,
    fold_idx: np.ndarray,
    folds: int,
    lam: float,
    rank: int,
    use_gpu: bool = False,
) -> np.ndarray:
    """Held-out predictions [N, T] at ONE (lambda, rank), same grouped CV as run_cv.

    Build A's raw material. Every row is predicted by a model fitted on folds that
    excluded the row's whole (prompt, wave) group, so a prediction here has never
    seen its own prompt's basin geometry — which is why ``eval_levers --predicted``
    can skip the cluster_sums leave-one-out and still be held out.

    Deliberately a second pass rather than a hook into ``run_cv``: the sweep's job
    is a table of cosines over a grid, this one's job is one cell's vectors, and
    braiding them would make both harder to read for no saving that matters (the
    refit is seconds on the GPU path).
    """
    if folds <= 0:
        raise ValueError(f"folds must be > 0, got {folds}")
    pred = np.zeros_like(y, dtype=np.float64)
    seen = np.zeros(y.shape[0], dtype=bool)
    if use_gpu:
        import torch

        dev = torch.device("cuda")
        z_t = torch.from_numpy(z.astype(np.float32)).to(dev)
        y_t = torch.from_numpy(y.astype(np.float32)).to(dev)
    for f in range(folds):
        tr, te = fold_idx != f, fold_idx == f
        if not te.any():
            continue
        if not tr.any():
            raise ValueError(f"fold {f} leaves no training rows")
        if use_gpu:
            import torch

            tr_t = torch.from_numpy(tr).to(dev)
            te_t = torch.from_numpy(te).to(dev)
            ztr, ytr = z_t[tr_t], y_t[tr_t]
            mu_z, mu_y = ztr.mean(0), ytr.mean(0)
            zc, yc = ztr - mu_z, ytr - mu_y
            a = zc.T @ zc + lam * torch.eye(zc.shape[1], device=dev)
            w = torch.linalg.solve(a, zc.T @ yc)
            zte_c = z_t[te_t] - mu_z
            if rank <= 0 or rank >= min(w.shape):
                block = zte_c @ w + mu_y
            else:
                u, s, vt = torch.linalg.svd(w, full_matrices=False)
                block = ((zte_c @ u[:, :rank]) * s[:rank]) @ vt[:rank] + mu_y
            pred[te] = block.double().cpu().numpy()
        else:
            mu_z, mu_y = z[tr].mean(0), y[tr].mean(0)
            w = ridge_fit(z[tr] - mu_z, y[tr] - mu_y, lam)
            zte_c = z[te] - mu_z
            if rank <= 0 or rank >= min(w.shape):
                pred[te] = zte_c @ w + mu_y
            else:
                u, s, vt = np.linalg.svd(w, full_matrices=False)
                pred[te] = ((zte_c @ u[:, :rank]) * s[:rank]) @ vt[:rank] + mu_y
        seen[te] = True
    if not seen.all():
        raise ValueError(
            f"{int((~seen).sum())} row(s) were never in a test fold — the emitted "
            "predictions would not all be held out"
        )
    return pred


def emit_predictions(
    path: Path,
    z: np.ndarray,
    y: np.ndarray,
    fold_idx: np.ndarray,
    folds: int,
    lam: float,
    rank: int,
    use_gpu: bool,
    members: list[int],
    signs: np.ndarray,
    lever_npz: Any,
    levers: np.ndarray,
    sources: dict[str, str],
) -> dict[str, Any]:
    """Write Build A's ``predicted_levers.npz``: held-out predictions, per gen.

    ── THE SIGN CONVENTION, stated once and carried everywhere ────────────────

    The ridge target for gen *i* is ``s_i * lever_g`` — the vector that pushes
    *i* toward ITS OWN basin — so the raw prediction ``pred_i`` points the
    member's own way and two members of opposite clusters predict opposite
    vectors. Averaging those raw predictions would CANCEL. So what is banked is

        member_pred[i] = s_i * pred_i

    i.e. each gen's own estimate of its group's CANONICAL lever
    ``mean_h(cluster 0) - mean_h(cluster 1)``, the same orientation
    ``build_levers`` stores in ``levers``. Consequences, all of them wanted:
    averaging over a group is a plain mean; ``cos(member_pred, levers[g])`` is a
    plain cosine with no sign bookkeeping; and ``eval_levers`` recovers the
    matched arm exactly as it does in L1 — ``matched_sign(cluster) * lever`` —
    with the only difference being where the lever came from. ``member_signs``
    is banked beside it, so ``pred_i = s_i * member_pred[i]`` is recoverable.

    Every prediction is held out at the GROUP level: the CV folds are whole
    (prompt, wave) groups, so no member of gen i's group was in the fit that
    predicted it.
    """
    from pleroma.build_levers import gen_key

    names = set(lever_npz.files)

    def need(*cands: str) -> str:
        for c in cands:
            if c in names:
                return c
        raise KeyError(f"none of {cands} in levers npz (has {sorted(names)})")

    n_sites, hidden_dim = int(levers.shape[1]), int(levers.shape[2])
    if y.shape[1] != n_sites * hidden_dim:
        raise ValueError(
            f"targets are {y.shape[1]}-d but the levers are {n_sites}x{hidden_dim}"
        )
    logger.info(
        "emitting held-out predictions at lambda=%g rank=%s (%s backend), %d gens",
        lam, rank or "full", "torch-cuda" if use_gpu else "numpy", z.shape[0],
    )
    pred = cv_predictions(z, y, fold_idx, folds, lam, rank, use_gpu)

    signed = signs[:, None] * pred                       # canonical orientation
    flat_levers = levers.reshape(levers.shape[0], -1)
    m_group_all = np.asarray(lever_npz[need("member_group_index")], dtype=int)
    m_cluster_all = np.asarray(lever_npz[need("member_clusters")], dtype=int)
    m_corpus_all = [str(x) for x in lever_npz[need("member_corpus_keys")]]
    m_genid_all = np.asarray(lever_npz[need("member_generation_ids")], dtype=int)
    m_pid_all = [str(x) for x in lever_npz[need("member_prompt_ids")]]
    m_class_all = [str(x) for x in lever_npz[need("member_prompt_classes")]]

    group_index = np.asarray([int(m_group_all[j]) for j in members], dtype=np.int64)
    per_gen_cos = cosine_rows(signed, flat_levers[group_index])

    n_groups = int(levers.shape[0])
    group_cos = np.full(n_groups, np.nan, dtype=np.float64)
    group_cos_sites = np.full((n_groups, n_sites), np.nan, dtype=np.float64)
    group_n = np.zeros(n_groups, dtype=np.int64)
    for gi in range(n_groups):
        sel = np.nonzero(group_index == gi)[0]
        group_n[gi] = sel.size
        if sel.size == 0:
            continue
        # The FULL-group average (no leave-one-out): the dilution number Build A
        # is priced against. eval_levers records the LOO averages it actually used.
        avg = signed[sel].mean(axis=0)
        group_cos[gi] = float(cosine_rows(avg[None, :], flat_levers[gi][None, :])[0])
        avg_sites = avg.reshape(n_sites, hidden_dim)
        true_sites = levers[gi]
        group_cos_sites[gi] = cosine_rows(avg_sites, true_sites)

    meta = {
        "stage": "expB15_regress_levers_emit",
        "lambda": float(lam),
        "rank": int(rank),
        "rank_label": "full" if rank <= 0 else str(rank),
        "folds": int(folds),
        "backend": "torch-cuda" if use_gpu else "numpy",
        "sign_convention": (
            "member_pred[i] = s_i * pred_i, the gen's own held-out estimate of its "
            "group's CANONICAL lever mean_h(cluster 0) - mean_h(cluster 1) — the same "
            "orientation build_levers stores in `levers`. s_i = +1 for cluster-0 "
            "members, -1 for cluster-1. The raw ridge prediction is recoverable as "
            "s_i * member_pred[i]."
        ),
        "holdout": (
            "grouped CV by (prompt, wave): the fit that predicted gen i had NO member "
            "of i's group in training, so a prediction carries nothing of its own "
            "prompt's basin geometry"
        ),
        "sites": [int(x) for x in lever_npz[need("sites")]],
        "n_gens": int(signed.shape[0]),
        "n_groups_with_predictions": int((group_n > 0).sum()),
        "per_gen_cos_mean": float(per_gen_cos.mean()),
        "per_gen_cos_frac_positive": float((per_gen_cos > 0).mean()),
        "group_cos_mean": float(np.nanmean(group_cos)) if np.isfinite(group_cos).any() else None,
        "group_cos_frac_positive": (
            float(np.nanmean(group_cos > 0)) if np.isfinite(group_cos).any() else None
        ),
        "sources": sources,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        member_pred=signed.reshape(-1, n_sites, hidden_dim).astype(np.float32),
        member_signs=signs.astype(np.float32),
        member_clusters=np.asarray([int(m_cluster_all[j]) for j in members], dtype=np.int64),
        member_group_index=group_index,
        member_corpus_keys=np.asarray([m_corpus_all[j] for j in members], dtype=np.str_),
        member_generation_ids=np.asarray(
            [int(m_genid_all[j]) for j in members], dtype=np.int64
        ),
        member_keys=np.asarray(
            [gen_key(m_corpus_all[j], int(m_genid_all[j])) for j in members], dtype=np.str_
        ),
        member_prompt_ids=np.asarray([m_pid_all[j] for j in members], dtype=np.str_),
        member_prompt_classes=np.asarray([m_class_all[j] for j in members], dtype=np.str_),
        member_pred_cos=per_gen_cos.astype(np.float32),
        group_prompt_ids=lever_npz[need("group_prompt_ids")],
        group_waves=lever_npz[need("group_waves")],
        group_prompt_classes=lever_npz[need("group_prompt_classes")],
        group_corpus_keys=lever_npz[need("group_corpus_keys")],
        group_pred_cos=group_cos.astype(np.float32),
        group_pred_cos_sites=group_cos_sites.astype(np.float32),
        group_n_predictions=group_n,
        sites=lever_npz[need("sites")],
        meta_json=np.asarray(json.dumps(meta), dtype=np.str_),
    )
    logger.info(
        "PREDICTIONS -> %s | per-gen cos %+.4f (%.1f%% positive); CLUSTER-AVERAGED "
        "cos %+.4f (%.1f%% of %d groups positive) — the averaging gain is the whole "
        "bet of Build A",
        path, meta["per_gen_cos_mean"], 100 * meta["per_gen_cos_frac_positive"],
        meta["group_cos_mean"] or float("nan"),
        100 * (meta["group_cos_frac_positive"] or 0.0),
        meta["n_groups_with_predictions"],
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--levers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lambdas", default="10,100,1000,10000")
    parser.add_argument("--ranks", default="0,8,16,32")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--shuffle-seeds", type=int, default=3)
    parser.add_argument("--backend", choices=("numpy", "torch-cuda"), default="numpy")
    parser.add_argument(
        "--emit-predictions",
        type=Path,
        default=None,
        help=(
            "ALSO write the held-out predictions at (--emit-lambda, --emit-rank) to "
            "this npz — Build A's input. Stores the SIGNED-CANONICAL prediction "
            "s_i * pred_i, i.e. each gen's own estimate of its group's canonical "
            "lever (mean_h(cluster 0) - mean_h(cluster 1)), so member_pred lives in "
            "the same space as build_levers' `levers` and averaging is a plain mean"
        ),
    )
    parser.add_argument("--emit-lambda", type=float, default=1000.0)
    parser.add_argument(
        "--emit-rank", type=int, default=8, help="0 = full rank (no SVD truncation)"
    )
    parser.add_argument(
        "--emit-only",
        action="store_true",
        help="skip the lambda/rank sweep and the shuffled refits; emit and nothing else",
    )
    parser.add_argument(
        "--extra-features", type=Path, default=None,
        help="extract_bins npz, joined by (corpus, generation_id) — the amplifier arms",
    )
    parser.add_argument(
        "--fold-by", choices=["group", "prompt"], default="prompt",
        help="prompt (DEFAULT since the 2026-09-17 leak fix): both waves of a "
        "prompt share a fold. group = the old (prompt, wave) folds, which LEAK "
        "twin levers across folds — kept only to reproduce pre-fix numbers.",
    )
    parser.add_argument(
        "--features", choices=["z", "extra", "both"], default="z",
        help="z = signatures only (L2 as run); extra = the bins block alone; "
        "both = concatenated (the amplifier proper)",
    )
    args = parser.parse_args()

    if args.features != "z" and args.extra_features is None:
        logger.error("--features %s needs --extra-features", args.features)
        return 2

    if args.emit_only and args.emit_predictions is None:
        logger.error("--emit-only needs --emit-predictions")
        return 2

    from pleroma.build_pairs import load_pairs

    loaded = load_pairs(args.pairs_dir)
    lev = np.load(args.levers, allow_pickle=True)
    # Expected arrays (from build_levers): levers [G, S, D_h]; group keys; per-gen
    # cluster table. Discover the actual names defensively and fail loudly.
    names = set(lev.files)
    logger.info("levers npz arrays: %s", sorted(names))

    def need(*cands: str) -> str:
        for c in cands:
            if c in names:
                return c
        raise KeyError(f"none of {cands} in levers npz (has {sorted(names)})")

    levers = np.asarray(lev[need("levers")], dtype=np.float64)  # [G, S, Dh]
    g_prompt = [str(x) for x in lev[need("group_prompt_ids", "lever_prompt_ids", "prompt_ids")]]
    g_wave = [str(x) for x in lev[need("group_waves", "lever_waves", "waves")]]
    m_corpus = [str(x) for x in lev[need("member_corpus_keys", "member_corpora")]]
    m_genid = np.asarray(lev[need("member_generation_ids")], dtype=int)
    m_group = np.asarray(lev[need("member_group_index")], dtype=int)
    # lever = mean(cluster 0) - mean(cluster 1), so cluster-0 members carry +1
    # (matching eval_levers' matched-arm sign convention).
    m_sign = 1.0 - 2.0 * np.asarray(lev[need("member_clusters")], dtype=np.float64)

    group_key = {i: (g_prompt[i], g_wave[i]) for i in range(len(g_prompt))}
    flat_levers = levers.reshape(levers.shape[0], -1)  # [G, S*Dh]

    # Join pairs' z to lever membership on (corpus, generation_id).
    member_of = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}
    rows, targets, groups, signs = [], [], [], []
    members: list[int] = []            # index into the levers npz's member table
    joined_pairs = []                  # PairRecords, aligned with rows
    for p in loaded.pairs:
        corpus = corpus_of_run_dir(p.run_dir)
        j = member_of.get((corpus, p.generation_id))
        if j is None:
            continue
        gi = int(m_group[j])
        if gi < 0:  # member of a group that got no lever (min-cluster guard)
            continue
        rows.append(p.row)
        targets.append(m_sign[j] * flat_levers[gi])
        groups.append(group_key[gi])
        signs.append(m_sign[j])
        members.append(j)
        joined_pairs.append(p)
    if len(rows) < 100:
        logger.error("only %d joined gens — join is broken, refusing", len(rows))
        return 1
    z = np.asarray(loaded.z[rows], dtype=np.float64)

    if args.features != "z":
        # The amplifier arms (PLAN-amplifier-2026-09-17): an extra feature block
        # (extract_bins npz) joined by the SAME (corpus, generation_id) key as
        # the lever join, standardized on TRAIN-split rows only — no worse a
        # leak profile than z's own corpus standardization in build_pairs.
        from pleroma.bank_mean_hiddens import corpus_key as _corpus_key

        ex = np.load(args.extra_features, allow_pickle=True)
        ex_feat = np.asarray(ex["features"], dtype=np.float64)
        ex_run = [str(x) for x in ex["run_dir"]]
        ex_gid = np.asarray(ex["generation_id"], dtype=int)
        ex_of = {(_corpus_key(ex_run[k]), int(ex_gid[k])): k
                 for k in range(ex_feat.shape[0])}
        idx: list[int] = []
        missing: list[str] = []
        for p in joined_pairs:
            k = ex_of.get((corpus_of_run_dir(p.run_dir), p.generation_id))
            if k is None:
                missing.append(p.pair_key)
            else:
                idx.append(k)
        if missing:
            logger.error(
                "%d joined gens missing from --extra-features (first 3: %s) — "
                "refusing a silently-shrunk fit", len(missing), missing[:3],
            )
            return 1
        extra = ex_feat[np.asarray(idx, dtype=int)]
        if not np.isfinite(extra).all():
            logger.error(
                "--extra-features has non-finite rows among the joined gens "
                "(poisoned extractions) — refusing"
            )
            return 1
        train_mask = np.array([p.split == "train" for p in joined_pairs])
        if not train_mask.any():
            logger.error("no train-split rows to standardize the extra block on")
            return 1
        mu = extra[train_mask].mean(axis=0)
        sd = extra[train_mask].std(axis=0)
        dead = sd < 1e-9
        extra_std = (extra - mu) / np.where(dead, 1.0, sd)
        extra_std[:, dead] = 0.0
        logger.info(
            "extra block: %s — %d dims (%d dead), standardized on %d train rows",
            args.extra_features, extra.shape[1], int(dead.sum()),
            int(train_mask.sum()),
        )
        z = extra_std if args.features == "extra" else np.concatenate(
            [z, extra_std], axis=1
        )

    y = np.vstack(targets)
    logger.info("joined %d gens over %d lever groups; features=%s z %s -> y %s",
                len(rows), len(set(groups)), args.features, z.shape, y.shape)

    uniq = sorted(set(groups))
    if args.fold_by == "prompt":
        # LEAK FIX (2026-09-17, external review): folds by (prompt, wave) put
        # a prompt's two waves in DIFFERENT folds — sorted twins are adjacent,
        # i % folds never collides — so a held-out twin's sibling lever
        # (mean |cos| .40, if5 .82, sf3 .89) was a training target, inflating
        # the "held-out" tail. Grouping by prompt_id alone holds out BOTH
        # waves of a prompt together. The old mode is kept only to reproduce
        # the tainted numbers.
        prompts = sorted({p for p, _ in groups})
        fold_of_p = {p: i % args.folds for i, p in enumerate(prompts)}
        fold_idx = np.array([fold_of_p[p] for p, _ in groups])
    else:
        fold_of = {g: i % args.folds for i, g in enumerate(uniq)}
        fold_idx = np.array([fold_of[g] for g in groups])
    lambdas = [float(x) for x in args.lambdas.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    use_gpu = args.backend == "torch-cuda"
    if use_gpu:
        import torch

        dev = torch.device("cuda")
        z_t = torch.from_numpy(z.astype(np.float32)).to(dev)
        y_t = torch.from_numpy(y.astype(np.float32)).to(dev)

    def run_cv(z_in: np.ndarray, tag: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if use_gpu:
            import torch

            zz = torch.from_numpy(z_in.astype(np.float32)).to(dev) if z_in is not z else z_t
        for lam in lambdas:
            per_rank: dict[int, list[np.ndarray]] = {r: [] for r in ranks}
            base_cos: list[np.ndarray] = []
            order: list[int] = []
            for f in range(args.folds):
                tr, te = fold_idx != f, fold_idx == f
                if use_gpu:
                    import torch

                    tr_t = torch.from_numpy(tr).to(dev)
                    te_t = torch.from_numpy(te).to(dev)
                    ztr, ytr_ = zz[tr_t], y_t[tr_t]
                    mu_z_t, mu_y_t = ztr.mean(0), ytr_.mean(0)
                    zc, yc = ztr - mu_z_t, ytr_ - mu_y_t
                    a = zc.T @ zc + lam * torch.eye(zc.shape[1], device=dev)
                    w_t = torch.linalg.solve(a, zc.T @ yc)
                    u, s, vt = torch.linalg.svd(w_t, full_matrices=False)
                    zte_c = zz[te_t] - mu_z_t
                    yte = y_t[te_t]
                    def _cos(pred: "torch.Tensor") -> np.ndarray:
                        num = (pred * yte).sum(1)
                        den = pred.norm(dim=1) * yte.norm(dim=1)
                        return (num / torch.where(den > 0, den, torch.ones_like(den))).cpu().numpy()
                    base_cos.append(_cos(mu_y_t.expand_as(yte)))
                    for r in ranks:
                        if r <= 0 or r >= min(w_t.shape):
                            pred = zte_c @ w_t + mu_y_t
                        else:
                            pred = ((zte_c @ u[:, :r]) * s[:r]) @ vt[:r] + mu_y_t
                        per_rank[r].append(_cos(pred))
                    order.extend(np.where(te)[0].tolist())
                    continue
                mu_z, mu_y = z_in[tr].mean(0), y[tr].mean(0)
                w = ridge_fit(z_in[tr] - mu_z, y[tr] - mu_y, lam)
                # ONE SVD reused across ranks (was: full SVD per rank — 3x waste).
                uu, ss, vvt = np.linalg.svd(w, full_matrices=False)
                base_cos.append(cosine_rows(np.tile(mu_y, (te.sum(), 1)), y[te]))
                zte_c = z_in[te] - mu_z
                for r in ranks:
                    if r <= 0 or r >= min(w.shape):
                        pred = zte_c @ w + mu_y
                    else:
                        pred = ((zte_c @ uu[:, :r]) * ss[:r]) @ vvt[:r] + mu_y
                    per_rank[r].append(cosine_rows(pred, y[te]))
                order.extend(np.where(te)[0].tolist())
            inv = np.argsort(order)
            res = {}
            for r in ranks:
                cos = np.concatenate(per_rank[r])[inv]
                res[f"rank{r or 'full'}"] = {
                    "mean_cos": float(cos.mean()),
                    "frac_pos": float((cos > 0).mean()),
                }
                if tag == "real":
                    res[f"rank{r or 'full'}"]["per_gen_cos"] = cos.round(4).tolist()
            bc = np.concatenate(base_cos)[inv]
            res["mean_lever_baseline"] = {"mean_cos": float(bc.mean()),
                                          "frac_pos": float((bc > 0).mean())}
            out[f"lambda{lam:g}"] = res
        return out

    report: dict[str, Any] = {
        "n_gens": len(rows), "n_groups": len(uniq), "folds": args.folds,
        "real": {}, "shuffled": [],
    }
    if not args.emit_only:
        report["real"] = run_cv(z, "real")
        rng = np.random.default_rng(7)
        for s in range(args.shuffle_seeds):
            perm = rng.permutation(len(rows))
            report["shuffled"].append(run_cv(z[perm], f"shuf{s}"))

    # ── Build A: emit the held-out predictions at ONE cell ────────────────────
    if args.emit_predictions is not None:
        try:
            report["emit"] = emit_predictions(
                path=args.emit_predictions,
                z=z,
                y=y,
                fold_idx=fold_idx,
                folds=args.folds,
                lam=float(args.emit_lambda),
                rank=int(args.emit_rank),
                use_gpu=use_gpu,
                members=members,
                signs=np.asarray(signs, dtype=np.float64),
                lever_npz=lev,
                levers=levers,
                sources={"pairs_dir": str(args.pairs_dir), "levers": str(args.levers)},
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.error("could not emit predictions: %s", exc)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if args.emit_only:
        logger.info("emit-only: report -> %s", args.out)
        return 0

    logger.info("=" * 90)
    logger.info("L2 REGRESSION — cosine(predicted, true signed lever), grouped CV by (prompt, wave)")
    logger.info("cos > 0 = z placed the trajectory on the CORRECT side of its basin axis (chance .5)")
    for lam in lambdas:
        key = f"lambda{lam:g}"
        real = report["real"][key]
        shuf = [s[key] for s in report["shuffled"]]
        for r in ranks:
            rk = f"rank{r or 'full'}"
            sm = np.mean([s[rk]["mean_cos"] for s in shuf])
            sf = np.mean([s[rk]["frac_pos"] for s in shuf])
            logger.info("λ=%-7g %-9s mean_cos %+.4f (shuf %+.4f)  frac_pos %.3f (shuf %.3f)",
                        lam, rk, real[rk]["mean_cos"], sm, real[rk]["frac_pos"], sf)
        mb = real["mean_lever_baseline"]
        logger.info("λ=%-7g %-9s mean_cos %+.4f              frac_pos %.3f   <- train-mean-lever floor",
                    lam, "meanlever", mb["mean_cos"], mb["frac_pos"])
    logger.info("report -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
