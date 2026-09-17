"""AttnRes feature family — the kotodama-native cross-block routing surface (Block Attention Residuals).

Captured but never featurized: per-token routing softmax over source blocks at every routing point (source 0
= earliest committed = ANCHOR, source -1 = the running `partial` = RECENCY), plus block-boundary committed
snapshots. Kotodama-ONLY (Llama has no AttnRes) — gated behind `enable_attn_res`, reads NEW capture fields
(`attn_res_routing`, `attn_res_committed`); the Llama path leaves them None and never calls this.

Design (research/planning/attnres-feature-spec.md): fan wide, MID-GRAIN distributional (the per_head template —
never average a structured axis away, never per-token; summarize distributions + COARSE temporal). The source
axis is relative + an explicit ANCHOR(block-0)-vs-RECENCY(partial) contrast = the AttnRes echo of the anamnesis
how-axis (anchor/sink vs recency at mid layers). All values bounded ∈[0,1]/[-1,1] → no length confound. Fan
across blocks; the decompose prunes depth. FIXED output dim (zero-filled when a slot is absent).

Inputs:
  routing   : list of (tag:str, layer:int, w:np.ndarray[n_pos, n_src]) — per-producing-position routing weights.
              src 0 = anchor (block-0/embed), src -1 = partial (recency); n_src grows with depth.
  committed : list[np.ndarray[n_pos, D]] (block-boundary snapshots) or None.
  sampled_layers : the layers to emit per-layer features for (fixed → fixed dim).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult

F32 = NDArray[np.float32]
EPS = 1e-8
QNAMES = ("entropy", "top1", "eff_src", "anchor_w", "partial_w", "recency_cent")
SUMM = ("mean", "std", "drift")
N_COMMITTED_PAIRS = 6                      # 7 DD3B committed blocks → 6 consecutive pairs (fixed dim)


def _summ(ts: np.ndarray) -> list[float]:
    """Coarse-temporal summary of a per-position series: mean, std, half-split drift (NOT per-token)."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0:
        return [0.0, 0.0, 0.0]
    if ts.size < 2:
        return [float(ts.mean()), 0.0, 0.0]
    h = ts.size // 2
    return [float(ts.mean()), float(ts.std()), abs(float(ts[:h].mean()) - float(ts[h:].mean()))]


def _point_quantities(w: np.ndarray) -> dict[str, np.ndarray]:
    """w (n_pos, n_src) routing weights → 6 bounded per-position series. src0=anchor, src-1=recency."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] == 0:
        z = np.zeros(max(w.shape[0], 1))
        return {q: z for q in QNAMES}
    n_src = w.shape[1]
    w = w / (w.sum(1, keepdims=True) + EPS)                          # renormalize (safety)
    ent = -(w * np.log(w + EPS)).sum(1) / np.log(max(n_src, 2))       # ∈[0,1]: 0=focused, 1=diffuse
    eff = (1.0 / (np.square(w).sum(1) + EPS)) / n_src                 # participation ratio / n_src ∈(0,1]
    idx = np.arange(n_src) / max(n_src - 1, 1)                        # 0=anchor … 1=most-recent
    return {"entropy": ent, "top1": w.max(1), "eff_src": eff,
            "anchor_w": w[:, 0], "partial_w": w[:, -1], "recency_cent": (w * idx[None, :]).sum(1)}


def extract_attn_res(routing, committed=None, sampled_layers=None) -> FeatureFamilyResult:
    feats: list[float] = []
    names: list[str] = []
    by_layer: dict[int, list[np.ndarray]] = {}
    point_stats: list[tuple[int, float, float, float]] = []          # (layer, entropy_mean, anchor_mean, partial_mean)
    final_vec: np.ndarray | None = None

    for tag, layer, w in (routing or []):
        q = _point_quantities(w)
        vec = np.array([v for qn in QNAMES for v in _summ(q[qn])], dtype=np.float64)   # 6*3 = 18
        if tag == "final":
            final_vec = vec
            continue
        by_layer.setdefault(int(layer), []).append(vec)
        point_stats.append((int(layer), float(q["entropy"].mean()),
                            float(q["anchor_w"].mean()), float(q["partial_w"].mean())))

    sl = list(sampled_layers) if sampled_layers is not None else sorted(by_layer.keys())

    # (A–E) per-sampled-layer averaged distribution-summaries (the mid-grain core)
    for L in sl:
        vecs = by_layer.get(int(L))
        v = np.mean(vecs, axis=0) if vecs else np.zeros(len(QNAMES) * 3)
        for qi, qn in enumerate(QNAMES):
            for si, sm in enumerate(SUMM):
                feats.append(float(v[qi * 3 + si]))
                names.append(f"attnres_L{L}_{qn}_{sm}")

    # (D) depth profile: slope of entropy/anchor/partial vs layer + cross-point entropy spread
    if len(point_stats) >= 2:
        pm = np.array(point_stats, dtype=np.float64)                  # (n, 4)
        lay = pm[:, 0]
        for ci, cn in ((1, "entropy"), (2, "anchor_w"), (3, "partial_w")):
            slope = float(np.polyfit(lay, pm[:, ci], 1)[0]) if lay.std() > EPS else 0.0
            feats.append(slope)
            names.append(f"attnres_depthslope_{cn}")
        feats.append(float(pm[:, 1].std()))
        names.append("attnres_xpoint_entropy_std")
    else:
        for cn in ("entropy", "anchor_w", "partial_w"):
            feats.append(0.0)
            names.append(f"attnres_depthslope_{cn}")
        feats.append(0.0)
        names.append("attnres_xpoint_entropy_std")

    # (E) the final consolidation read (pre-output routing): its 6 mean quantities
    for qi, qn in enumerate(QNAMES):
        feats.append(float(final_vec[qi * 3]) if final_vec is not None else 0.0)
        names.append(f"attnres_final_{qn}_mean")

    # (F) committed-state geometry: consecutive-block direction preservation (cosine ∈[-1,1], bounded)
    cos_means = [0.0] * N_COMMITTED_PAIRS
    overall = [0.0, 0.0]
    if committed is not None and len(committed) >= 2:
        cos_all = []
        for b in range(min(len(committed) - 1, N_COMMITTED_PAIRS)):
            a = np.asarray(committed[b], dtype=np.float64)
            c = np.asarray(committed[b + 1], dtype=np.float64)
            if a.ndim == 2 and a.shape == c.shape and a.shape[0] > 0:
                den = np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + EPS
                cos = (a * c).sum(1) / den
                cos_means[b] = float(cos.mean())
                cos_all.append(cos)
        if cos_all:
            allc = np.concatenate(cos_all)
            overall = [float(allc.mean()), float(allc.std())]
    for b in range(N_COMMITTED_PAIRS):
        feats.append(cos_means[b])
        names.append(f"attnres_committed_cos_b{b}b{b + 1}_mean")
    feats.append(overall[0]); names.append("attnres_committed_cos_overall_mean")
    feats.append(overall[1]); names.append("attnres_committed_cos_overall_std")

    return FeatureFamilyResult(
        features=np.nan_to_num(np.array(feats, dtype=np.float32)),
        feature_names=names,
        family_name="attn_res",
    )
