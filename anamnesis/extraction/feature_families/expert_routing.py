"""MoE expert-routing features (Source.expert_routing; prefix `xrt_`).

The fifth substrate in the source×method×depth map — read only on models with a
Mixture-of-Experts MLP (vmb arm A7, M6 = DeepSeek-V2-Lite class: 2 shared + 64
routed experts, top-6 greedy). Reads the per-generated-token router allocation
distribution (RawGenerationData.router_dist), the shared/routed branch output
norms + cosine (RawGenerationData.router_branch_norms), and the pre-softmax router
logit norms (RawGenerationData.router_logit_norms) — all banked by the M6 capture
hooks. Dense models leave those fields None → this family returns empty (the
gate_features None-guard pattern), so it is safe to enable everywhere.

v2.1 ENRICHMENT (desk-approved 2026-07-18, 770d7b41 + 803eac37 + 48870af4): every
non-learned METHOD rung is now present on the routing source (v1 had only
distributional + magnitude — the "method asymmetry" the desk owned). Per sampled
MoE layer L (dense layers, e.g. DeepSeek layer 0, contribute NOTHING — they have no
router; feature_pipeline passes only banked-router layers, so L0 never reaches here):

  Static (per-layer means / span summaries):
    xrt_L{L}_alloc_entropy_mean            — mean entropy of per-token expert dist  [distributional]
    xrt_L{L}_top1_margin_mean              — mean (top1 − top2) allocation weight    [magnitude]
    xrt_L{L}_shared_mass_mean              — mean ‖shared‖/(‖shared‖+‖routed‖)        [magnitude]
    xrt_L{L}_coverage                      — unique experts selected / n_experts      [distributional]
    xrt_L{L}_load_kl                       — KL(selection hist ‖ uniform)             [distributional]
    xrt_L{L}_eff_experts                   — 1/Σp² of the load histogram (v2.1)       [geometry]
    xrt_L{L}_shared_routed_cos_mean        — mean cos(shared_out, routed_out) (v2.1)  [geometry]
    xrt_L{L}_topk_weight_entropy_mean      — entropy of renorm top-k weights (v2.1)   [distributional]
    xrt_L{L}_logit_norm_mean               — mean ‖router_logits‖ pre-softmax (v2.1)  [magnitude]
    xrt_L{L}_alloc_entropy_spectral_flatness — flatness of the entropy series (v2.1)  [spectral]
  Dynamic (dispersion / change over generation time):
    xrt_L{L}_alloc_entropy_std             — std of per-token entropy                 [distributional]
    xrt_L{L}_alloc_entropy_slope           — linear slope of per-token entropy        [distributional]
    xrt_L{L}_switch_rate                   — P(top1 expert_t ≠ top1 expert_{t−1})     [distributional]
    xrt_L{L}_hist_drift                    — JSD(first-half, second-half hist)        [distributional]
    xrt_L{L}_shared_mass_std               — std of shared_mass                       [magnitude]
    xrt_L{L}_shared_routed_cos_std         — std of the branch cosine series (v2.1)   [geometry]
    xrt_L{L}_set_churn_rate                — Jaccard churn of the selected set (v2.1) [distributional]
    xrt_L{L}_logit_norm_std                — std of ‖router_logits‖ (v2.1)            [magnitude]
    xrt_L{L}_switch_dominant_period        — dominant period of the switch series(v2.1)[spectral]

  Cross-layer (appended once, after the per-layer block):
    xrt_cka_L{i}_L{j}                      — linear-CKA of router_dist across adjacent sampled MoE
                                             layers (v2.1, geometry); ADJACENT-5 pairs (desk ruling 1).
    xrt_cka_global_mean                    — mean linear-CKA over ALL sampled-MoE pairs (v2.1, geometry).

Feature COUNT (M6, 6 banked MoE layers): 19 per-layer × 6 + 6 cross-layer = 120. (Older docs said
"~129": that was pre-desk arithmetic with the 15-pair CKA; the desk ruled adjacent-5+mean = 6, saving 9.)
After v2.1 the family is FROZEN for the program (deferred-documented: learned rung — encoder-on-raw
supersedes; per-expert internals + kv_b_proj — banked-optional; routing↔output-entropy coupling — NAMED
post-program idea). feature_map places every name; `FeatureMap.unclassified()` == 0 is the acceptance check.

Selection semantics: under greedy top-k routing (DeepSeek-V2-Lite topk_method "greedy") the selected
experts are exactly argtop-k of the banked distribution, so coverage/load/drift/churn/topk-weight derive
the selection in-module from router_dist. `top_k` (num_experts_per_tok, 6 for DSV2-Lite) is a parameter.

Pure numpy (no torch/model deps) — the state_extractor design constraint.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

# Per-layer feature suffixes in fixed order (a short-gen / dense layer emits zeros for the same
# names — the modal-vector guard expects fixed length). v1 suffixes first (unchanged), then v2.1.
_STATIC_SUFFIXES = (
    "alloc_entropy_mean",
    "top1_margin_mean",
    "shared_mass_mean",
    "coverage",
    "load_kl",
    # ── v2.1 static additions ──
    "eff_experts",
    "shared_routed_cos_mean",
    "topk_weight_entropy_mean",
    "logit_norm_mean",
    "alloc_entropy_spectral_flatness",
)
_DYNAMIC_SUFFIXES = (
    "alloc_entropy_std",
    "alloc_entropy_slope",
    "switch_rate",
    "hist_drift",
    "shared_mass_std",
    # ── v2.1 dynamic additions ──
    "shared_routed_cos_std",
    "set_churn_rate",
    "logit_norm_std",
    "switch_dominant_period",
)
N_FEATURES_PER_LAYER = len(_STATIC_SUFFIXES) + len(_DYNAMIC_SUFFIXES)


def _xrt_layer_names(layer_idx: int) -> list[str]:
    """Fixed-order feature names for one MoE layer."""
    prefix = f"xrt_L{layer_idx}"
    return [f"{prefix}_{s}" for s in (*_STATIC_SUFFIXES, *_DYNAMIC_SUFFIXES)]


def _cka_names(sampled_layers: list[int]) -> list[str]:
    """Cross-layer CKA feature names: adjacent sampled-MoE pairs + one global-mean scalar."""
    names = [f"xrt_cka_L{sampled_layers[i]}_L{sampled_layers[i + 1]}"
             for i in range(len(sampled_layers) - 1)]
    names.append("xrt_cka_global_mean")
    return names


def _entropy(p: F64) -> float:
    """Natural-log entropy of a distribution row (zeros safe)."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _jsd(p: F64, q: F64) -> float:
    """Jensen–Shannon divergence (natural log; range [0, ln 2]). p, q are histograms."""
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a: F64, b: F64) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _slope(y: F64) -> float:
    """Linear-regression slope of y vs its index (0 if <2 points)."""
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _spectral_flatness(ts: F64) -> float:
    """Spectral flatness (geometric/arithmetic mean of the PSD, DC removed) of a 1D series.

    ~1 = white/noisy (no rhythm); →0 = tonal/rhythmic. NaN-safe for short / constant series.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 4:
        return 0.0
    ts = ts - ts.mean()
    if not np.any(ts):
        return 0.0
    psd = np.abs(np.fft.rfft(ts)) ** 2
    psd = psd[1:]                                    # drop DC
    psd = psd[psd > 1e-12]
    if psd.size == 0:
        return 0.0
    gm = float(np.exp(np.mean(np.log(psd))))
    am = float(np.mean(psd))
    return gm / am if am > 0 else 0.0


def _dominant_period(ts: F64) -> float:
    """Dominant period (1 / peak non-DC frequency) of a 1D series. 0 if none / short / constant."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 4:
        return 0.0
    ts = ts - ts.mean()
    if not np.any(ts):
        return 0.0
    psd = np.abs(np.fft.rfft(ts)) ** 2
    freqs = np.fft.rfftfreq(ts.size)
    psd[0] = 0.0                                      # ignore DC
    peak = int(np.argmax(psd))
    f = float(freqs[peak])
    return float(1.0 / f) if f > 0 else 0.0


def _linear_cka(X: F64, Y: F64) -> float:
    """Linear CKA between two [T, d] matrices (column-centered). Range [0, 1]; 0 if degenerate."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] < 2 or X.shape[0] != Y.shape[0]:
        return 0.0
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    xty = X.T @ Y
    denom = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    if denom < 1e-12:
        return 0.0
    return float((np.linalg.norm(xty) ** 2) / denom)


def extract_expert_routing_features(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    top_k: int = 6,
) -> FeatureFamilyResult:
    """Extract MoE expert-routing statistics (v2.1: distributional + magnitude + geometry + spectral).

    Parameters
    ----------
    data : RawGenerationData
        Must have `router_dist` populated for MoE layers. `router_branch_norms` feeds the shared_mass
        + shared_routed_cos features (its absence zeros those; a 2-column form zeros only the cosine).
        `router_logit_norms` feeds the logit_norm features (absence zeros them). All degrade gracefully
        so the family runs on both v1-captured and v2.1-captured raw.
    sampled_layers : list[int], optional
        Which MoE layers to emit features for (feature_pipeline passes only banked-router layers).
    top_k : int
        num_experts_per_tok — the selection size (6 for DeepSeek-V2-Lite).
    """
    if sampled_layers is None:
        sampled_layers = [5, 11, 15, 18, 22, 26]

    # Dense model or capture not banked → empty family (safe-everywhere guard).
    if not data.router_dist:
        logger.info("No router_dist available — returning empty expert_routing features")
        return FeatureFamilyResult.empty("expert_routing")

    branch_norms = data.router_branch_norms or {}
    logit_norms = getattr(data, "router_logit_norms", None) or {}

    features: list[float] = []
    names: list[str] = []
    p_mats: dict[int, F64] = {}                       # layer → [T, n_experts] (for cross-layer CKA)

    for l_idx in sampled_layers:
        layer_names = _xrt_layer_names(l_idx)
        dist_list = data.router_dist.get(l_idx)

        # Layer has no router (dense) or too short a span → fixed-length zeros.
        if not dist_list or len(dist_list) < 1:
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        # [T, n_experts] per-token distribution over routed experts.
        P = np.asarray(np.stack(dist_list), dtype=np.float64)
        if P.ndim != 2 or P.shape[0] == 0:
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue
        T, n_experts = P.shape
        p_mats[l_idx] = P
        k = int(min(max(top_k, 1), n_experts))

        # ── Per-token entropy time series ──
        entropy_ts = np.array([_entropy(P[t]) for t in range(T)], dtype=np.float64)

        # ── Per-token top1/top2 margin ──
        if n_experts >= 2:
            part = np.partition(P, -2, axis=1)
            top1 = part[:, -1]
            top2 = part[:, -2]
        else:
            top1 = P[:, -1]
            top2 = np.zeros(T, dtype=np.float64)
        margin_ts = top1 - top2

        # ── Selected experts per token (argtop-k) — greedy selection == argtop-k(dist) ──
        sel = np.argpartition(P, -k, axis=1)[:, -k:]  # [T, k] (unordered, fine for sets/hist)
        top1_expert = np.argmax(P, axis=1)            # [T]

        # coverage: unique experts ever selected / n_experts
        coverage = float(np.unique(sel).size) / float(n_experts)

        # load histogram over the span (drives load_kl AND eff_experts)
        hist = np.bincount(sel.reshape(-1), minlength=n_experts).astype(np.float64)
        total = hist.sum()
        if total > 0:
            phist = hist / total
            uniform = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)
            mask = phist > 0
            load_kl = float((phist[mask] * np.log(phist[mask] / uniform[mask])).sum())
            eff_experts = float(1.0 / np.sum(phist ** 2))    # participation ratio [1, n_experts]
        else:
            load_kl = 0.0
            eff_experts = 0.0

        # topk_weight_entropy: entropy of the RENORMALIZED selected-k weights, mean over the span
        tkw = np.empty(T, dtype=np.float64)
        for t in range(T):
            w = P[t, sel[t]]
            s = w.sum()
            tkw[t] = _entropy(w / s) if s > 0 else 0.0

        # switch_rate: P(top1 changes step-to-step); switch series feeds the dominant period
        if T >= 2:
            switch_series = (top1_expert[1:] != top1_expert[:-1]).astype(np.float64)
            switch_rate = float(switch_series.mean())
        else:
            switch_series = np.zeros(0, dtype=np.float64)
            switch_rate = 0.0

        # set_churn_rate: mean over t of 1 − Jaccard(sel_set_t, sel_set_{t−1})
        if T >= 2:
            churn = np.empty(T - 1, dtype=np.float64)
            prev = set(int(e) for e in sel[0])
            for t in range(1, T):
                cur = set(int(e) for e in sel[t])
                union = len(prev | cur)
                churn[t - 1] = 1.0 - (len(prev & cur) / union if union else 0.0)
                prev = cur
            set_churn_rate = float(churn.mean())
        else:
            set_churn_rate = 0.0

        # hist_drift: JSD of first-half vs second-half selection histograms
        if T >= 2:
            half = T // 2
            if half >= 1:
                h1 = np.bincount(sel[:half].reshape(-1), minlength=n_experts).astype(np.float64)
                h2 = np.bincount(sel[half:].reshape(-1), minlength=n_experts).astype(np.float64)
                hist_drift = _jsd(h1, h2)
            else:
                hist_drift = 0.0
        else:
            hist_drift = 0.0

        # ── shared_mass + shared_routed_cos time series (need branch norms / cosine) ──
        bn = branch_norms.get(l_idx)
        shared_mass_ts = np.zeros(T, dtype=np.float64)
        cos_ts = np.zeros(T, dtype=np.float64)
        if bn and len(bn) >= 1:
            B = np.asarray(np.stack(bn), dtype=np.float64)     # [T, 2] or [T, 3] = (shared, routed[, cos])
            if B.ndim == 2 and B.shape[1] >= 2:
                m = min(B.shape[0], T)
                denom = B[:m, 0] + B[:m, 1]
                shared_mass_ts[:m] = np.divide(
                    B[:m, 0], denom,
                    out=np.zeros(m, dtype=np.float64),
                    where=denom > 1e-12,
                )
                if B.shape[1] >= 3:                            # v2.1 cosine column
                    cos_ts[:m] = B[:m, 2]

        # ── logit_norm time series (pre-softmax router commitment scale) ──
        ln = logit_norms.get(l_idx)
        if ln is not None and len(ln) >= 1:
            L = np.asarray(ln, dtype=np.float64).reshape(-1)
            logit_norm_mean = float(L.mean())
            logit_norm_std = float(L.std())
        else:
            logit_norm_mean = 0.0
            logit_norm_std = 0.0

        # ── Assemble in fixed suffix order (static block, then dynamic block) ──
        vals = [
            # static
            float(entropy_ts.mean()),                 # alloc_entropy_mean
            float(margin_ts.mean()),                  # top1_margin_mean
            float(shared_mass_ts.mean()),             # shared_mass_mean
            coverage,                                 # coverage
            load_kl,                                  # load_kl
            eff_experts,                              # eff_experts (v2.1)
            float(cos_ts.mean()),                     # shared_routed_cos_mean (v2.1)
            float(tkw.mean()),                        # topk_weight_entropy_mean (v2.1)
            logit_norm_mean,                          # logit_norm_mean (v2.1)
            _spectral_flatness(entropy_ts),           # alloc_entropy_spectral_flatness (v2.1)
            # dynamic
            float(entropy_ts.std()),                  # alloc_entropy_std
            _slope(entropy_ts),                       # alloc_entropy_slope
            switch_rate,                              # switch_rate
            hist_drift,                               # hist_drift
            float(shared_mass_ts.std()),              # shared_mass_std
            float(cos_ts.std()),                      # shared_routed_cos_std (v2.1)
            set_churn_rate,                           # set_churn_rate (v2.1)
            logit_norm_std,                           # logit_norm_std (v2.1)
            _dominant_period(switch_series),          # switch_dominant_period (v2.1)
        ]
        features.extend(vals)
        names.extend(layer_names)

    # ── Cross-layer routing CKA (v2.1 geometry): adjacent pairs + global mean over all pairs ──
    cka_names = _cka_names(sampled_layers)
    adj_vals: list[float] = []
    for i in range(len(sampled_layers) - 1):
        li, lj = sampled_layers[i], sampled_layers[i + 1]
        v = 0.0
        if li in p_mats and lj in p_mats:
            A, Bm = p_mats[li], p_mats[lj]
            n = min(A.shape[0], Bm.shape[0])
            if n >= 2:
                v = _linear_cka(A[:n], Bm[:n])
        adj_vals.append(v)
    all_pairs: list[float] = []
    layers_present = [l for l in sampled_layers if l in p_mats]
    for a in range(len(layers_present)):
        for b in range(a + 1, len(layers_present)):
            A, Bm = p_mats[layers_present[a]], p_mats[layers_present[b]]
            n = min(A.shape[0], Bm.shape[0])
            if n >= 2:
                all_pairs.append(_linear_cka(A[:n], Bm[:n]))
    cka_vals = [*adj_vals, float(np.mean(all_pairs)) if all_pairs else 0.0]
    features.extend(cka_vals)
    names.extend(cka_names)

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="expert_routing",
    )
