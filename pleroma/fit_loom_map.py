"""Fit the FINAL loom map g: signature+bins -> lever, with full inference kit.

regress_levers emits held-out PREDICTIONS (the honest evaluation object); the
loom needs the fitted MAP plus every standardization statistic required to run
a brand-new trajectory through the identical pipeline:

    raw v3 signature x (2713)
      -> z_full   = (x - FULL_mean) / FULL_scale        [discriminants npz]
      -> z_corpus = (z_full - v3_mu) / v3_sd, dead -> 0 [train rows, recomputed
                                                         here and VERIFIED
                                                         against pairs_z]
    raw bins-B b (420)
      -> b_std    = (b - bins_mu) / bins_sd, dead -> 0  [joined train rows,
                                                         regress_levers' exact
                                                         convention]
    concat [z_corpus | b_std] -> centered ridge, rank-truncated W -> flat lever
    [S x 3072], sign convention: the output points TOWARD THE BASIN OF THE
    TRAJECTORY THAT WAS READ (targets were m_sign * lever, per regress_levers).

The fit uses ALL joined rows (a deployment map, not an evaluation one — the
held-out receipts live in RESULTS-AMPLIFIER). λ and rank default to the sweep's
winner (1000, 8).

Node CPU (numpy; reads the raw signature files):
    python -m pleroma.fit_loom_map \\
        --pairs-dir outputs/b15_pairs_w3 --levers outputs/b15_levers_w3/levers.npz \\
        --bins outputs/bins/binsB_all.npz \\
        --discriminants factor_directions_3b.npz \\
        --norm-ref-bank outputs/b15_levers_w3/levers.npz \\
        --out outputs/loom/loom_map.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("fit_loom_map")


def main() -> int:  # noqa: C901 — one linear procedure
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--levers", type=Path, required=True)
    parser.add_argument("--bins", type=Path, required=True)
    parser.add_argument("--discriminants", type=Path, required=True)
    parser.add_argument("--norm-ref-bank", type=Path, required=True,
                        help="bank whose median per-site lever norms define alpha=1")
    parser.add_argument("--lam", type=float, default=1000.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--verify-rows", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from pleroma.bank_mean_hiddens import corpus_key
    from pleroma.build_pairs import load_pairs
    from pleroma.regress_levers import corpus_of_run_dir

    loaded = load_pairs(args.pairs_dir)
    if loaded.meta.get("params", {}).get("standardize") != "corpus":
        logger.error("this fitter replicates --standardize corpus; pairs dir says %r",
                     loaded.meta.get("params", {}).get("standardize"))
        return 2

    # ── stage 1: recompute z_full from raw signatures ─────────────────────────
    with np.load(args.discriminants, allow_pickle=True) as npz:
        full_mean = np.asarray(npz["FULL_mean"], dtype=np.float64)
        full_scale = np.asarray(npz["FULL_scale"], dtype=np.float64)
    degenerate = full_scale < 1e-12
    scale_safe = np.where(degenerate, 1.0, full_scale)
    disc_sha = hashlib.sha256(args.discriminants.read_bytes()).hexdigest()

    z_full = np.zeros((len(loaded.pairs), len(full_mean)), dtype=np.float64)
    for i, p in enumerate(loaded.pairs):
        sig = Path(p.run_dir) / p.signature_path
        with np.load(sig, allow_pickle=True) as npz:
            x = np.asarray(npz["features"], dtype=np.float64)
        zf = (x - full_mean) / scale_safe
        zf[degenerate] = 0.0
        z_full[i] = zf
    train_mask = np.array([p.split == "train" for p in loaded.pairs])
    v3_mu = z_full[train_mask].mean(axis=0)
    v3_sd = z_full[train_mask].std(axis=0)
    v3_dead = v3_sd < 1e-8
    z_corpus = (z_full - v3_mu) / np.where(v3_dead, 1.0, v3_sd)
    z_corpus[:, v3_dead] = 0.0

    # Verification against the pairs matrix — recipe drift is a silent killer.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(loaded.pairs), size=min(args.verify_rows, len(loaded.pairs)),
                     replace=False)
    err = float(np.max(np.abs(z_corpus[idx] - loaded.z[idx].astype(np.float64))))
    if err > 1e-3:
        logger.error("recomputed z differs from pairs_z (max |Δ| %.3g) — the "
                     "standardization recipe drifted; refusing to emit a map "
                     "whose inference path is not the training path", err)
        return 1
    logger.info("z recipe verified on %d rows (max |Δ| %.2e)", len(idx), err)

    # ── bins block + lever join (regress_levers' conventions) ─────────────────
    ex = np.load(args.bins, allow_pickle=True)
    ex_feat = np.asarray(ex["features"], dtype=np.float64)
    ex_of = {(corpus_key(str(r)), int(g)): k for k, (r, g) in
             enumerate(zip(ex["run_dir"], ex["generation_id"]))}
    bins_config = json.loads(str(ex["config"]))

    lev = np.load(args.levers, allow_pickle=True)
    m_corpus = [str(x) for x in lev["member_corpus_keys"]]
    m_genid = np.asarray(lev["member_generation_ids"], dtype=int)
    m_group = np.asarray(lev["member_group_index"], dtype=int)
    m_sign = 1.0 - 2.0 * np.asarray(lev["member_clusters"], dtype=np.float64)
    levers = np.asarray(lev["levers"], dtype=np.float64)
    sites = [int(x) for x in lev["sites"]]
    flat_levers = levers.reshape(levers.shape[0], -1)
    member_of = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}

    rows, bins_rows, targets, joined_train = [], [], [], []
    for i, p in enumerate(loaded.pairs):
        corpus = corpus_of_run_dir(p.run_dir)
        j = member_of.get((corpus, p.generation_id))
        if j is None or int(m_group[j]) < 0:
            continue
        k = ex_of.get((corpus, p.generation_id))
        if k is None:
            logger.error("gen %s missing from bins npz — refusing", p.pair_key)
            return 1
        rows.append(i)
        bins_rows.append(k)
        targets.append(m_sign[j] * flat_levers[int(m_group[j])])
        joined_train.append(bool(train_mask[i]))
    logger.info("joined %d gens", len(rows))

    extra = ex_feat[np.asarray(bins_rows)]
    if not np.isfinite(extra).all():
        logger.error("non-finite bins rows among joined gens — refusing")
        return 1
    jt = np.asarray(joined_train)
    bins_mu = extra[jt].mean(axis=0)
    bins_sd = extra[jt].std(axis=0)
    bins_dead = bins_sd < 1e-9
    extra_std = (extra - bins_mu) / np.where(bins_dead, 1.0, bins_sd)
    extra_std[:, bins_dead] = 0.0

    z_in = np.concatenate([z_corpus[np.asarray(rows)], extra_std], axis=1)
    y = np.vstack(targets)

    # ── final centered ridge, rank truncated ──────────────────────────────────
    mu_in = z_in.mean(axis=0)
    mu_y = y.mean(axis=0)
    zc, yc = z_in - mu_in, y - mu_y
    a = zc.T @ zc + args.lam * np.eye(zc.shape[1])
    w = np.linalg.solve(a, zc.T @ yc)
    factored: dict[str, np.ndarray] = {}
    if 0 < args.rank < min(w.shape):
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        w = (u[:, : args.rank] * s[: args.rank]) @ vt[: args.rank]
        # Ship-size form: LoomMap accepts W_U/W_S/W_Vt in place of the full W
        # (the product IS the truncated W). ~300x smaller on disk at rank 8.
        factored = {
            "W_U": u[:, : args.rank].astype(np.float32),
            "W_S": s[: args.rank].astype(np.float32),
            "W_Vt": vt[: args.rank].astype(np.float32),
        }
    fit_cos = np.array([
        float(np.dot(p_, t_) / (np.linalg.norm(p_) * np.linalg.norm(t_)))
        for p_, t_ in zip(zc @ w + mu_y, y)
    ])
    logger.info("in-sample per-gen cos: mean %.3f (an upper bound, NOT a claim — "
                "held-out receipts are RESULTS-AMPLIFIER's)", float(fit_cos.mean()))

    ref = np.load(args.norm_ref_bank, allow_pickle=False)
    ref_norms = np.linalg.norm(np.asarray(ref["levers"], dtype=np.float64), axis=2)
    norm_ref = np.nanmedian(ref_norms, axis=0)  # [S]
    logger.info("alpha=1 per-site norm reference (bank median): %s",
                [round(float(x), 3) for x in norm_ref])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        W=w.astype(np.float32), **factored, mu_in=mu_in.astype(np.float32),
        mu_y=mu_y.astype(np.float32),
        v3_mu=v3_mu.astype(np.float32), v3_sd=v3_sd.astype(np.float32),
        v3_dead=v3_dead,
        bins_mu=bins_mu.astype(np.float32), bins_sd=bins_sd.astype(np.float32),
        bins_dead=bins_dead,
        sites=np.array(sites), norm_ref=norm_ref.astype(np.float32),
        meta=np.array(json.dumps({
            "lam": args.lam, "rank": args.rank, "n_rows": len(rows),
            "feature_order": "z_corpus(2713) then bins_std(420)",
            "discriminants": str(args.discriminants), "discriminants_sha256": disc_sha,
            "bins_config": bins_config,
            "pairs_dir": str(args.pairs_dir), "levers": str(args.levers),
            "in_sample_mean_cos": float(fit_cos.mean()),
            "sign_convention": "output points toward the basin of the trajectory "
                               "that was read (targets were m_sign * lever)",
            "alpha_convention": "norm-match each site row to norm_ref, then alpha "
                                "scales (dose axis of the dose ladders)",
        })),
    )
    logger.info("DONE -> %s (W %s)", args.out, w.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
