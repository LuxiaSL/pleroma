"""Fit the FINAL loom map g: signature+bins -> lever, with full inference kit.

`pleroma.map.build.regress` emits held-out PREDICTIONS (the honest evaluation object); the
loom needs the fitted MAP plus every standardization statistic required to run
a brand-new trajectory through the identical pipeline:

    raw v3 signature x (2713)
      -> z_full   = (x - FULL_mean) / FULL_scale        [discriminants npz]
      -> z_corpus = (z_full - v3_mu) / v3_sd, dead -> 0 [train rows, recomputed
                                                         here and VERIFIED
                                                         against pairs_z]
    raw bins-B b (420)
      -> b_std    = (b - bins_mu) / bins_sd, dead -> 0  [joined train rows,
                                                         the regression's exact
                                                         convention]
    concat [z_corpus | b_std] -> centered ridge, rank-truncated W -> flat lever
    [S x 3072], sign convention: the output points TOWARD THE BASIN OF THE
    TRAJECTORY THAT WAS READ (targets were m_sign * lever, as in the regression).

The fit uses ALL joined rows (a deployment map, not an evaluation one — held-out
evaluation is `pleroma.map.build.regress`'s job). λ and rank default to the
served wide-map point (1e4, rank 64).

CPU (numpy; reads the raw signature files):
    python -m pleroma.map.build.wide \\
        --pairs-dir outputs/b15_pairs_w3 --levers outputs/b15_levers_w3/levers.npz \\
        --bins outputs/bins/binsB_all.npz \\
        --discriminants factor_directions_3b.npz \\
        --norm-ref-bank outputs/b15_levers_w3/levers.npz \\
        --out outputs/loom/loom_map.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pleroma.map.build.ridge import resolve_rank
from pleroma.map.build.shelf import ShelfError, ShelfRecipeError, compute_shelf
from pleroma.map.svd import SIGN_CONVENTION, canonical_signs

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
    # the served wide-map point (every banked wide map: 1e4 / r64), not the
    # 3B sweep's best cell (1e3 / r8)
    parser.add_argument("--lam", type=float, default=1e4)
    parser.add_argument("--rank", type=int, default=64,
                        help="SVD truncation; 0 (or >= min(W.shape)) = full rank, "
                             "banked as the EFFECTIVE rank")
    parser.add_argument("--verify-rows", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # ★ The standardisers + ruler are pleroma.map.build.shelf's, shared with
    # the v1a export — see that module for every refusal.
    try:
        shelf = compute_shelf(args.pairs_dir, args.levers, args.bins,
                              args.discriminants, args.norm_ref_bank,
                              verify_rows=args.verify_rows)
    except ShelfError as exc:
        logger.error("shelf stats refused: %s", exc)
        return 2 if isinstance(exc, ShelfRecipeError) else 1
    st = shelf.stats
    v3_mu, v3_sd, v3_dead = st.v3_mu, st.v3_sd, st.v3_dead
    bins_mu, bins_sd, bins_dead = st.bins_mu, st.bins_sd, st.bins_dead
    sites, rows = st.sites, shelf.rows
    disc_sha, bins_config = st.discriminants_sha256, st.bins_config
    z_corpus, extra_std, targets = shelf.z_corpus, shelf.bins_std, shelf.targets

    z_in = np.concatenate([z_corpus[np.asarray(rows)], extra_std], axis=1)
    y = np.vstack(targets)

    # ── final centered ridge, rank truncated ──────────────────────────────────
    mu_in = z_in.mean(axis=0)
    mu_y = y.mean(axis=0)
    zc, yc = z_in - mu_in, y - mu_y
    a = zc.T @ zc + args.lam * np.eye(zc.shape[1])
    w = np.linalg.solve(a, zc.T @ yc)
    factored: dict[str, np.ndarray] = {}
    rank_eff = resolve_rank(args.rank, w.shape)
    if rank_eff < min(w.shape):
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        # One orientation on every stack (pleroma.map.svd): the stored
        # factors are the code basis LoomMap reads.
        u, s, vt, _ = canonical_signs(
            u[:, : args.rank], s[: args.rank], vt[: args.rank])
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
                "held-out evaluation is pleroma.map.build.regress's)", float(fit_cos.mean()))

    norm_ref = st.norm_ref  # [S] float32 — shelf.ruler_from_bank
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
            "lam": args.lam, "rank": rank_eff, "n_rows": len(rows),
            **({"svd_orientation": {"convention": SIGN_CONVENTION}}
               if factored else {}),
            **({} if rank_eff == args.rank else {"rank_requested": args.rank}),
            # ★ DERIVED, never a literal. 2713 is the 3B's signature width; the
            # 8B is 2769 and model C is 4086, so a literal
            # "z_corpus(2713) then bins_std(420)" would be wrong for every model
            # but one, and silently: the arrays would be right and only the
            # label would lie. The server does not parse this field, but /info
            # publishes it, and the UI must describe the map it is ACTUALLY
            # serving. A hardcoded dimension in a provenance field is the
            # exact inverse of that rule.
            "feature_order": f"z_corpus({len(v3_mu)}) then bins_std({len(bins_mu)})",
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
