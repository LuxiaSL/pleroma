"""Pure feature extraction: raw tensors → flat feature vector.

This module has NO model awareness. It operates on pre-collected tensors
and can be tested offline with saved tensor samples.

Feature groups (LEGACY tier labels retired 2026-07-11 — the interpretable taxonomy is
source × method × depth, see analysis/feature_map.py and
research/methodology/tier-retirement-and-translation.md; the group names below are kept
only because feature-name prefixes and stored signatures still reference them):
  "T1"  — Activation norms (residual/magnitude), logit statistics (output/distributional)
  "T2"  — Attention entropy + head agreement (attention), residual deltas (residual),
          spectral graph features
  "T2.5"— cache_* attention-read profiles (SOURCE: attention) + kv_*/epoch_* key-space
          geometry (SOURCE: keys) — two different substrates in one legacy bin
  "T3"  — Residual stream PCA projections (residual/geometry)
  Baseline — kNN-LM single-layer signature
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.stats import entropy as scipy_entropy

from anamnesis.config import ExtractionConfig

logger = logging.getLogger(__name__)

# Type aliases
F32 = NDArray[np.float32]


@dataclass
class RawGenerationData:
    """All tensors needed for feature extraction, already on CPU as numpy.

    Shapes (after conversion from torch tensors):
        hidden_states: list of T arrays, each of shape [num_layers+1, hidden_dim]
            - T = number of generation steps (not including prefill)
            - Index 0 in the layer dim is the embedding layer
            - Transformer layers are at indices 1..num_layers
        attentions: list of T arrays, each of shape [num_layers, num_heads, current_seq_len]
            - current_seq_len grows by 1 each step
        logits: list of T arrays, each of shape [vocab_size]
        chosen_token_ids: array of shape [T] — the tokens actually generated
        pre_rope_keys: dict mapping layer_idx → list of T arrays,
            each of shape [num_kv_heads, head_dim]
            (generation steps only, prefill excluded)
        prompt_length: int — number of tokens in the prompt (for separating prompt/gen)
        positional_means: optional array [num_layers+1, max_calibrated_pos, hidden_dim]
            for positional decomposition correction
        v_proj_values: optional dict layer_idx → list of T arrays [num_kv_heads, head_dim]
            (attention values; OV-circuit surface; populated by replay-extract)
        queries: optional dict layer_idx → list of T arrays [num_attention_heads, head_dim]
            (PRE-RoPE queries; re-apply RoPE offline for QK geometry; replay-extract)
    """

    hidden_states: list[F32]       # T × [num_layers+1, hidden_dim]
    attentions: list[F32]          # T × [num_layers, num_heads, seq_len_at_step]
    logits: list[F32]              # T × [vocab_size]
    chosen_token_ids: F32          # [T]
    pre_rope_keys: dict[int, list[F32]]  # layer_idx → T × [num_kv_heads, head_dim]
    prompt_length: int
    positional_means: F32 | None = None  # [num_layers+1, max_pos, hidden_dim]
    gate_activations: dict[int, list[F32]] | None = None  # layer_idx → T × [intermediate_size]
    # gate_activations are pre-SiLU gate_proj outputs. Apply SiLU to get actual gate values.
    v_proj_values: dict[int, list[F32]] | None = None  # layer_idx → T × [num_kv_heads, head_dim]
    # v_proj_values are attention values (post-projection); the OV-circuit surface.
    queries: dict[int, list[F32]] | None = None  # layer_idx → T × [num_attention_heads, head_dim]
    # queries are PRE-RoPE q_proj outputs; re-apply RoPE offline for post-RoPE QK geometry.
    attn_outputs: dict[int, list[F32]] | None = None  # layer_idx → T × [hidden_dim]
    # attn_outputs are o_proj outputs: the attention sublayer's additive residual contribution
    # (pre residual-add). With block-boundary hidden states, mlp_out = resid_{l+1} − resid_l − attn_out.
    # ── AttnRes (kotodama Block Attention Residuals) — optional, kotodama-only; Llama leaves None ──
    attn_res_routing: list | None = None
    # list of (tag:str, layer:int, weights[n_pos, n_src]) — per-producing-position cross-block routing softmax;
    # source 0 = anchor (earliest committed/block-0), source -1 = `partial` (recency). n_src grows with depth.
    attn_res_committed: list | None = None
    # list of block-boundary committed snapshots, each [n_pos, hidden_dim].
    # ── MoE expert routing (vmb arm A7, M6 DeepSeek-V2-Lite class) — optional, MoE-layers only ──
    # Dense models (Llama/Qwen/OLMo/Gemma) leave these None; the xrt family returns empty when absent
    # (the gate_features None-guard pattern). Populated by the M6 capture hooks (see model_loader /
    # research/planning/HOOK-AUDIT-PLAN-M6-dsv2lite-2026-07-17.md).
    router_dist: dict[int, list[F32]] | None = None
    # layer_idx → T × [n_routed_experts]: per-generated-token expert-allocation distribution over the
    # routed experts. Banked reading = the DENSE pre-topk softmax (recomputed in the MoE-module pre-hook
    # from gate.weight, because DeepseekV2Moe bypasses gate.forward — F.linear(h, gate.weight)). Under
    # greedy top-k the selected set is exactly argtop-k of this vector, so coverage/load/drift are
    # derived in-module. Which reading was banked is stamped in run metadata (router_granularity).
    # Prefill step 0 excluded.
    router_branch_norms: dict[int, list[F32]] | None = None
    # layer_idx → T × [2 or 3] = (‖shared-expert branch output‖, ‖routed-expert sum output‖[, cos]) per
    # token, from forward hooks on mlp.shared_experts and mlp.experts outputs. Feeds xrt_shared_mass; the
    # optional 3rd column (v2.1) = per-token cos(shared_vec, routed_vec) → xrt_shared_routed_cos. MoE only.
    router_logit_norms: dict[int, list[F32]] | None = None
    # layer_idx → T × scalar ‖router_logits‖ (pre-softmax). Feeds xrt_logit_norm (v2.1 magnitude). MoE only.
    _hs_array_cache: F32 | None = field(default=None, repr=False, compare=False)
    # lazily-built np.stack(hidden_states); do not set directly — use hidden_states_array()
    _mean_attn_cache: dict[int, list[NDArray[np.float64]]] = field(
        default_factory=dict, repr=False, compare=False)
    # per-layer head-mean attention rows; do not set directly — use mean_attention()

    def mean_attention(self, l_idx: int) -> list[NDArray[np.float64]]:
        """Per-step head-mean attention rows for one layer: T entries of
        float64 [seq_len_t] — exactly attentions[t][l_idx].mean(axis=0)
        .astype(np.float64), built once per (gen, layer) and shared (C2: the
        tier-2.5 cache profiles, attention decay, spectral similarity, and
        attention_flow all re-derived this same reduction). Callers must not
        mutate the returned arrays."""
        if l_idx not in self._mean_attn_cache:
            self._mean_attn_cache[l_idx] = [
                self.attentions[t][l_idx].mean(axis=0).astype(np.float64)
                for t in range(len(self.attentions))
            ]
        return self._mean_attn_cache[l_idx]

    def hidden_states_array(self) -> F32:
        """[T, num_layers+1, hidden_dim] — np.stack(self.hidden_states), built once
        per gen and cached (tier1 and tier2 previously each rebuilt this ~270MB
        array per call). Callers must not mutate the returned array in place."""
        if self._hs_array_cache is None:
            self._hs_array_cache = np.stack(self.hidden_states)
        return self._hs_array_cache


@dataclass
class ExtractionResult:
    """Output of feature extraction."""

    features: F32                  # flat feature vector
    feature_names: list[str]       # one name per feature dimension
    tier_slices: dict[str, tuple[int, int]]  # tier name → (start, end) indices
    knnlm_baseline: F32 | None     # raw kNN-LM vector (before PCA)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_entropy(probs: F32, base: float | None = None) -> float:
    """Compute entropy, handling zeros and near-zeros safely."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()  # renormalize for safety
    return float(scipy_entropy(probs, base=base))


def _softmax(logits: F32) -> F32:
    """Numerically stable softmax."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _batch_softmax(logits_array: np.ndarray) -> np.ndarray:
    """Batch softmax over [T, vocab_size] in float64, returns float32."""
    x = logits_array.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@dataclass
class LogitFeatures:
    """All features derivable from per-step logits, computed in a single pass."""

    entropy: F32          # [T]
    top1_prob: F32        # [T]
    top5_mass: F32        # [T]
    chosen_rank: F32      # [T]
    chosen_prob: F32      # [T]
    surprise: F32         # [T]


def _compute_logit_features(
    logits: list[F32],
    chosen_token_ids: F32,
    vocab_size: int | None = None,
) -> LogitFeatures:
    """Compute all logit-derived features in a single per-row pass.

    Processes each timestep's [V] vector individually (~1.2MB in L2 cache)
    instead of creating [T, V] float64 arrays (~623MB, NUMA-hostile).
    One exp() per timestep reused for entropy, top-k, rank, and chosen_prob.

    Entropy uses the identity: H(softmax(x)) = log(Z) - sum(s*exp(s))/Z
    where s = x - max(x), Z = sum(exp(s)). Algebraically identical to
    softmax-then-entropy but avoids materializing the probability array.

    Rank comparison on raw logits is exact (softmax is monotonic).
    """
    T = len(logits)
    if vocab_size is None:
        vocab_size = logits[0].shape[0] if T > 0 else 0
    chosen_ids = chosen_token_ids.astype(np.intp)

    # Surface (don't silently mask) chosen ids outside the vocab — these indicate
    # an alignment bug; we still fall back to token 0 per-row below (v3/C8).
    if T > 0:
        n_oob = int(((chosen_ids[:T] < 0) | (chosen_ids[:T] >= vocab_size)).sum())
        if n_oob:
            logger.warning(
                f"_compute_logit_features: {n_oob}/{T} chosen token ids out of "
                f"[0,{vocab_size}); using token-0 fallback (possible alignment bug)."
            )

    entropy = np.empty(T, dtype=np.float32)
    top1_prob = np.empty(T, dtype=np.float32)
    top5_mass = np.empty(T, dtype=np.float32)
    chosen_rank = np.empty(T, dtype=np.float32)
    chosen_prob = np.empty(T, dtype=np.float32)

    for i in range(T):
        x = logits[i].astype(np.float64)
        max_x = x.max()
        s = x - max_x
        exp_s = np.exp(s)
        Z = exp_s.sum()
        inv_Z = 1.0 / Z

        entropy[i] = np.log(Z) - (s * exp_s).sum() * inv_Z
        top1_prob[i] = exp_s.max() * inv_Z

        top5_idx = np.argpartition(exp_s, -5)[-5:]
        top5_mass[i] = exp_s[top5_idx].sum() * inv_Z

        cid = int(chosen_ids[i]) if i < len(chosen_ids) and chosen_ids[i] < vocab_size else 0
        chosen_rank[i] = (x > x[cid]).sum()
        chosen_prob[i] = exp_s[cid] * inv_Z

    chosen_prob = np.maximum(chosen_prob, 1e-10).astype(np.float32)
    surprise = -np.log(chosen_prob).astype(np.float32)

    return LogitFeatures(
        entropy=entropy,
        top1_prob=top1_prob,
        top5_mass=top5_mass,
        chosen_rank=chosen_rank,
        chosen_prob=chosen_prob,
        surprise=surprise,
    )


def _batch_entropy(probs_2d: np.ndarray) -> np.ndarray:
    """Vectorized entropy over rows of a 2D probability array.

    Handles zeros safely. Input shape: [N, D], returns [N].
    """
    p = np.asarray(probs_2d, dtype=np.float64)
    p = np.maximum(p, 0.0)
    row_sums = p.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    p = p / row_sums
    log_p = np.zeros_like(p)
    mask = p > 0
    log_p[mask] = np.log(p[mask])
    return -(p * log_p).sum(axis=-1).astype(np.float32)


def _cosine_sim(a: F32, b: F32) -> float:
    """Cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_dist(a: F32, b: F32) -> float:
    return 1.0 - _cosine_sim(a, b)


def _trajectory_indices(T: int, n_points: int = 5) -> list[int]:
    """Return n_points evenly spaced indices in [0, T-1]."""
    if T <= 0:
        return []
    if T == 1:
        return [0] * n_points
    return [int(round(i * (T - 1) / (n_points - 1))) for i in range(n_points)]


def _correct_hidden_state(
    h: F32,
    layer_idx: int,
    abs_position: int,
    positional_means: F32 | None,
) -> F32:
    """Subtract positional mean from a hidden state vector."""
    if positional_means is None:
        return h
    max_pos = positional_means.shape[1]
    pos = min(abs_position, max_pos - 1)
    return h - positional_means[layer_idx, pos]


# ── Tier 1: Cheap, High Prior ──────────────────────────────────────────────────

def extract_tier1(
    data: RawGenerationData,
    config: ExtractionConfig,
) -> tuple[F32, list[str]]:
    """Extract Tier 1 features: activation norms, logit stats, token dynamics.

    Returns (feature_vector, feature_names).
    """
    features: list[float] = []
    names: list[str] = []
    T = len(data.hidden_states)
    num_layers_plus_embed = data.hidden_states[0].shape[0] if T > 0 else 29
    num_layers = num_layers_plus_embed - 1  # exclude embedding layer
    n_traj = config.trajectory_points

    if T == 0:
        # Return fixed-length zero vector so feature dimensions are consistent
        # 1.1: num_layers * (2 + n_traj)
        for l in range(num_layers):
            features.extend([0.0] * (2 + n_traj))
            names.append(f"activation_norm_mean_L{l}")
            names.append(f"activation_norm_std_L{l}")
            for i in range(n_traj):
                names.append(f"activation_norm_traj{i}_L{l}")
        # 1.2: 2 + n_traj + 2 + n_traj + 1 = 5 + 2*n_traj = 15
        features.extend([0.0] * (5 + 2 * n_traj))
        names.extend(["logit_entropy_mean", "logit_entropy_std"])
        for i in range(n_traj):
            names.append(f"logit_entropy_traj{i}")
        names.extend(["top1_prob_mean", "top1_prob_std"])
        for i in range(n_traj):
            names.append(f"top1_prob_traj{i}")
        names.append("top5_mass_mean")
        # 1.3: 4 + n_traj + 1 = 10
        features.extend([0.0] * (5 + n_traj))
        names.extend(["mean_chosen_rank", "std_chosen_rank", "mean_surprise", "std_surprise"])
        for i in range(n_traj):
            names.append(f"surprise_traj{i}")
        names.append("surprise_boundary_count")
        return np.array(features, dtype=np.float32), names

    traj_idx = _trajectory_indices(T, n_traj)

    # ── Pre-stack for vectorized ops (built once per gen, shared with tier2) ──
    hs_array = data.hidden_states_array()  # [T, num_layers+1, hidden_dim]

    # ── 1.1 Per-Layer Activation Norms ──
    positions = np.arange(T) + data.prompt_length
    for l in range(num_layers):
        h = hs_array[:, l + 1, :]  # [T, hidden_dim]
        if data.positional_means is not None:
            max_pos = data.positional_means.shape[1]
            pos_idx = np.minimum(positions, max_pos - 1)
            h = h - data.positional_means[l + 1, pos_idx, :]
        norms = np.linalg.norm(h, axis=1).astype(np.float32)  # [T]

        features.append(float(norms.mean()))
        names.append(f"activation_norm_mean_L{l}")
        features.append(float(norms.std()))
        names.append(f"activation_norm_std_L{l}")

        for i, ti in enumerate(traj_idx):
            features.append(float(norms[ti]) if ti < T else 0.0)
            names.append(f"activation_norm_traj{i}_L{l}")

    # ── 1.2 Output Logit Statistics ──
    # Single-pass per-row: entropy, top-k, rank, surprise from logits directly.
    # Avoids materializing [T, vocab_size] probability arrays (~1.9GB float64).
    lf = _compute_logit_features(data.logits, data.chosen_token_ids)

    ent_arr = lf.entropy
    top1_probs = lf.top1_prob
    top5_masses = lf.top5_mass

    features.append(float(ent_arr.mean()))
    names.append("logit_entropy_mean")
    features.append(float(ent_arr.std()))
    names.append("logit_entropy_std")
    for i, ti in enumerate(traj_idx):
        features.append(float(ent_arr[ti]) if ti < T else 0.0)
        names.append(f"logit_entropy_traj{i}")

    features.append(float(top1_probs.mean()))
    names.append("top1_prob_mean")
    features.append(float(top1_probs.std()))
    names.append("top1_prob_std")
    for i, ti in enumerate(traj_idx):
        features.append(float(top1_probs[ti]) if ti < T else 0.0)
        names.append(f"top1_prob_traj{i}")

    features.append(float(top5_masses.mean()))
    names.append("top5_mass_mean")

    # ── 1.3 Token Probability Dynamics ──
    rank_arr = lf.chosen_rank
    surp_arr = lf.surprise

    features.append(float(rank_arr.mean()))
    names.append("mean_chosen_rank")
    features.append(float(rank_arr.std()))
    names.append("std_chosen_rank")
    features.append(float(surp_arr.mean()))
    names.append("mean_surprise")
    features.append(float(surp_arr.std()))
    names.append("std_surprise")
    for i, ti in enumerate(traj_idx):
        features.append(float(surp_arr[ti]) if ti < T else 0.0)
        names.append(f"surprise_traj{i}")

    # Bayesian surprise: event boundary count
    if T >= config.surprise_window:
        window = config.surprise_window
        threshold_sigma = config.surprise_threshold_sigma
        running_mean = np.convolve(surp_arr, np.ones(window) / window, mode="valid")
        running_std = np.array([
            surp_arr[max(0, i - window + 1):i + 1].std()
            for i in range(window - 1, len(surp_arr))
        ], dtype=np.float32)
        # Count crossings
        boundary_count = 0
        for i in range(len(running_mean)):
            threshold = running_mean[i] + threshold_sigma * max(running_std[i], 1e-6)
            if surp_arr[i + window - 1] > threshold:
                boundary_count += 1
        features.append(float(boundary_count))
    else:
        features.append(0.0)
    names.append("surprise_boundary_count")

    return np.array(features, dtype=np.float32), names


# ── Tier 2: Moderate Cost ──────────────────────────────────────────────────────

def extract_tier2(
    data: RawGenerationData,
    config: ExtractionConfig,
) -> tuple[F32, list[str]]:
    """Extract Tier 2 features: attention entropy, head agreement, residual deltas, spectral.

    Returns (feature_vector, feature_names).
    """
    features: list[float] = []
    names: list[str] = []
    T = len(data.attentions)

    # Infer layer/head counts from data or fall back to defaults
    if T > 0:
        num_layers = data.attentions[0].shape[0]
        num_heads = data.attentions[0].shape[1]
    else:
        # Use hidden_states to infer layer count, default heads to 24
        num_layers = (data.hidden_states[0].shape[0] - 1) if data.hidden_states else 28
        num_heads = 24

    if T == 0:
        # Return fixed-length zero vector so feature dimensions are consistent
        # 2.1: num_layers * 2, 2.2: num_layers * 2, 2.3: (num_layers-1) * 3, 2.4: len(sampled) * 4
        for l in range(num_layers):
            features.extend([0.0, 0.0])
            names.extend([f"attn_entropy_mean_L{l}", f"attn_entropy_std_L{l}"])
        for l in range(num_layers):
            features.extend([0.0, 0.0])
            names.extend([f"head_agreement_mean_L{l}", f"head_agreement_std_L{l}"])
        for l in range(num_layers - 1):
            features.extend([0.0, 0.0, 0.0])
            names.extend([f"delta_norm_mean_L{l}", f"delta_norm_std_L{l}", f"delta_cosine_mean_L{l}"])
        for l_idx in config.sampled_layers:
            for feat_name in ["fiedler", "hfer", "spectral_entropy", "smoothness"]:
                features.append(0.0)
                names.append(f"spectral_{feat_name}_L{l_idx}")
        return np.array(features, dtype=np.float32), names

    # ── 2.1 Attention Entropy Per Layer ──
    attn_sample_steps = list(range(0, T, max(1, T // 60))) if T > 60 else list(range(T))
    for l in range(num_layers):
        all_ents: list[np.ndarray] = []
        for t in attn_sample_steps:
            attn = data.attentions[t][l]  # [num_heads, seq_len_t]
            all_ents.append(_batch_entropy(attn))  # [num_heads]
        ent_flat = np.concatenate(all_ents)

        features.append(float(ent_flat.mean()))
        names.append(f"attn_entropy_mean_L{l}")
        features.append(float(ent_flat.std()))
        names.append(f"attn_entropy_std_L{l}")

    # ── 2.2 Attention Head Agreement Per Layer ──
    agreement_sample_steps = list(range(0, T, max(1, T // 30))) if T > 30 else list(range(T))
    max_jsd = float(np.log(num_heads)) if num_heads > 1 else 1.0
    for l in range(num_layers):
        agreements_list: list[float] = []
        for t in agreement_sample_steps:
            attn = data.attentions[t][l].astype(np.float64)  # [num_heads, seq_len_t]
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            attn_norm = attn / row_sums

            mean_dist = attn_norm.mean(axis=0, keepdims=True)  # [1, seq_len]
            h_mean = float(_batch_entropy(mean_dist)[0])
            h_heads = _batch_entropy(attn_norm)  # [num_heads]
            mean_h = float(h_heads.mean())

            jsd_approx = max(0.0, h_mean - mean_h)
            agreements_list.append(1.0 - jsd_approx / max(max_jsd, 1e-12))

        agr_arr = np.array(agreements_list, dtype=np.float32)
        features.append(float(agr_arr.mean()))
        names.append(f"head_agreement_mean_L{l}")
        features.append(float(agr_arr.std()))
        names.append(f"head_agreement_std_L{l}")

    # ── 2.3 Layer-to-Layer Residual Stream Deltas ──
    hs_array = data.hidden_states_array()  # [T, num_layers+1, hidden_dim] (shared cache)
    num_model_layers = hs_array.shape[1] - 1
    positions = np.arange(T) + data.prompt_length
    for l in range(num_model_layers - 1):
        h_l = hs_array[:, l + 1, :]  # [T, hidden_dim] — view, no copy
        h_l1 = hs_array[:, l + 2, :]
        if data.positional_means is not None:
            max_pos = data.positional_means.shape[1]
            pos_idx = np.minimum(positions, max_pos - 1)
            h_l = h_l - data.positional_means[l + 1, pos_idx, :]
            h_l1 = h_l1 - data.positional_means[l + 2, pos_idx, :]
        delta = h_l1 - h_l  # [T, hidden_dim]
        dn = np.linalg.norm(delta, axis=1).astype(np.float32)  # [T]

        h_l_f64 = h_l.astype(np.float64)
        delta_f64 = delta.astype(np.float64)
        dots = (delta_f64 * h_l_f64).sum(axis=1)
        norm_d = np.linalg.norm(delta_f64, axis=1)
        norm_h = np.linalg.norm(h_l_f64, axis=1)
        denom = np.maximum(norm_d * norm_h, 1e-12)
        dc = (dots / denom).astype(np.float32)  # [T]

        features.append(float(dn.mean()))
        names.append(f"delta_norm_mean_L{l}")
        features.append(float(dn.std()))
        names.append(f"delta_norm_std_L{l}")
        features.append(float(dc.mean()))
        names.append(f"delta_cosine_mean_L{l}")

    # ── 2.4 Spectral Features ──
    # Only for sampled layers, using stacked attention matrices
    for l_idx in config.sampled_layers:
        if l_idx >= num_layers:
            # Pad with zeros if layer doesn't exist
            for feat_name in ["fiedler", "hfer", "spectral_entropy", "smoothness"]:
                features.append(0.0)
                names.append(f"spectral_{feat_name}_L{l_idx}")
            continue

        try:
            spectral_feats = _extract_spectral_features(data, l_idx, config)
            for feat_name, val in spectral_feats:
                features.append(val)
                names.append(f"spectral_{feat_name}_L{l_idx}")
        except Exception as e:
            logger.warning(f"Spectral extraction failed for layer {l_idx}: {e}")
            for feat_name in ["fiedler", "hfer", "spectral_entropy", "smoothness"]:
                features.append(0.0)
                names.append(f"spectral_{feat_name}_L{l_idx}")

    return np.array(features, dtype=np.float32), names


def _extract_spectral_features(
    data: RawGenerationData,
    layer_idx: int,
    config: ExtractionConfig,
) -> list[tuple[str, float]]:
    """Extract spectral features for a single layer.

    Build attention similarity matrix from subsampled generation steps,
    compute graph Laplacian eigenvalues.
    """
    T = len(data.attentions)
    num_heads = data.attentions[0].shape[1]

    # Subsample generation steps
    step = config.spectral_subsample_step
    sampled_steps = list(range(0, T, step))
    if len(sampled_steps) < 3:
        sampled_steps = list(range(T))

    n = len(sampled_steps)

    # Stack attention vectors for sampled steps, averaged across heads
    # Each attention at step t has shape [num_heads, seq_len_at_t]
    # Extract the attention pattern at the generated token position (last position)
    # and take last `n` positions of the sequence to build a square-ish matrix

    # Approach: build an n×n pairwise attention similarity matrix
    # For each pair of sampled steps (i, j), compute similarity of their
    # head-averaged attention distributions
    mean_rows = data.mean_attention(layer_idx)  # shared per-(gen,layer) cache (C2)
    attn_vectors = [mean_rows[t] for t in sampled_steps]  # [seq_len] f64 each

    # Pad to same length (max seq_len across sampled steps)
    max_len = max(v.shape[0] for v in attn_vectors)
    padded = np.zeros((n, max_len), dtype=np.float64)
    for i, v in enumerate(attn_vectors):
        padded[i, :v.shape[0]] = v

    # Build similarity matrix (cosine similarity)
    norms = np.linalg.norm(padded, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = padded / norms
    A = normalized @ normalized.T  # [n, n] cosine similarity
    A = np.maximum(A, 0)  # ensure non-negative for Laplacian

    # Symmetrize (should already be symmetric, but ensure)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0)  # no self-loops

    # Unnormalized graph Laplacian L = D - A. (The symmetric normalized Laplacian
    # collapses this near-complete attention-similarity graph to a near-constant
    # spectrum — verified WORSE for fiedler/spectral_entropy, v3/B2 — so keep L's
    # discriminative spectrum and make the SUMMARIES scale-free instead.)
    D = np.diag(A.sum(axis=1))
    L = D - A
    L += np.eye(n) * 1e-10  # numerical stability
    eigenvalues = la.eigvalsh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp numerical negatives

    total_energy = eigenvalues.sum()
    lam_max = float(eigenvalues[-1]) if n > 0 else 0.0

    # Fiedler normalized by the spectral radius → scale-free (the raw 2nd
    # eigenvalue scales with graph size n ∝ T) (v3/B2).
    fiedler = float(eigenvalues[1] / lam_max) if (n > 1 and lam_max > 1e-12) else 0.0

    # HFER: high-frequency energy ratio (a fraction, already scale-free).
    if total_energy > 1e-12:
        median_ev = float(np.median(eigenvalues))
        hfer = float(eigenvalues[eigenvalues > median_ev].sum() / total_energy)
    else:
        hfer = 0.0

    # Spectral entropy of the (sum-normalized, scale-free) eigenvalue
    # distribution, divided by log(n) so it doesn't grow with the node count
    # (= sampled steps ∝ T) (v3/B2).
    if total_energy > 1e-12 and n > 1:
        normalized_ev = eigenvalues / total_energy
        spec_entropy = float(scipy_entropy(normalized_ev + 1e-12) / np.log(n))
    else:
        spec_entropy = 0.0

    # Smoothness: Rayleigh quotient on the symmetric NORMALIZED Laplacian (bounded
    # [0,2], so it doesn't scale with n) of a meaningful graph signal — the
    # positionally-corrected hidden-state NORM per step (v3/C2). Old signal was
    # hidden_state.mean() = scalar mean of a ~4096-dim RMSNorm residual ≈ noise.
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    if len(data.hidden_states) > 0:
        h_signal = np.array([
            np.linalg.norm(
                _correct_hidden_state(
                    data.hidden_states[t][layer_idx + 1], layer_idx + 1,
                    data.prompt_length + t, data.positional_means,
                ).astype(np.float64)
            )
            for t in sampled_steps
        ], dtype=np.float64)
        xtLx = h_signal @ L_sym @ h_signal
        xtx = h_signal @ h_signal
        smoothness = float(xtLx / max(xtx, 1e-12))
    else:
        smoothness = 0.0

    return [
        ("fiedler", fiedler),
        ("hfer", hfer),
        ("spectral_entropy", spec_entropy),
        ("smoothness", smoothness),
    ]


# ── Tier 2.5: KV Cache Analysis ───────────────────────────────────────────────

def extract_tier2_5(
    data: RawGenerationData,
    config: ExtractionConfig,
) -> tuple[F32, list[str]]:
    """Extract Tier 2.5: KV cache attention profiles, key geometry, cross-layer, epochs.

    Returns (feature_vector, feature_names).
    """
    features: list[float] = []
    names: list[str] = []
    T = len(data.attentions)

    # Feature names for cache attention profiles (must match actual extraction order)
    _cache_profile_names = [
        "recency_bias", "sink_mass", "cache_coverage", "lookback_ratio",
        "attn_decay_rate",
        "recency_traj0", "recency_traj1", "recency_traj2", "recency_traj3",
    ]
    # Feature names for key geometry
    _key_geom_names = [
        "key_spread", "key_eff_dim", "key_drift",
        "key_novelty_mean", "key_novelty_std",
        "key_novelty_traj0", "key_novelty_traj1", "key_novelty_traj2",
        "key_novelty_traj3", "key_novelty_traj4",
    ]

    if T == 0:
        # Return fixed-length zero vector so feature dimensions are consistent
        for l_idx in config.sampled_layers:
            for name in _cache_profile_names:
                features.append(0.0)
                names.append(f"cache_{name}_L{l_idx}")
        for l_idx in config.sampled_layers:
            for name in _key_geom_names:
                features.append(0.0)
                names.append(f"kv_{name}_L{l_idx}")
        # Epoch detection (9 features)
        for base in ["epoch_n_transitions", "epoch_max_transition", "epoch_regularity"]:
            for suffix in ["_mean", "_max", "_std"]:
                features.append(0.0)
                names.append(f"{base}{suffix}")
        return np.array(features, dtype=np.float32), names

    num_layers = data.attentions[0].shape[0]
    num_heads = data.attentions[0].shape[1]

    # ── 2.5.1 Attention-Over-Cache-Position Distributions ──
    for l_idx in config.sampled_layers:
        if l_idx >= num_layers:
            for name in _cache_profile_names:
                features.append(0.0)
                names.append(f"cache_{name}_L{l_idx}")
            continue

        recency_biases = []
        sink_masses = []
        cache_coverages = []
        prompt_masses = []   # v3/B1: accumulate raw masses for ratio-of-means lookback
        gen_masses = []

        mean_rows = data.mean_attention(l_idx)  # shared per-(gen,layer) cache (C2)
        for t in range(T):
            mean_attn = mean_rows[t]  # [seq_len] float64
            seq_len = mean_attn.shape[0]

            if seq_len == 0:
                continue

            # Recency bias: fraction of mass on last 20% of positions
            cutoff = max(1, int(seq_len * 0.8))
            recency = float(mean_attn[cutoff:].sum() / max(mean_attn.sum(), 1e-12))
            recency_biases.append(recency)

            # Sink mass: attention to position 0 (BOS / attention sink). On this
            # corpus the sink dominates (argmax==0 ~100%), so this equals the old
            # "anchor_strength" = max(attention); named honestly now (v3/C1).
            sink_masses.append(float(mean_attn[0]))

            # Cache coverage: fraction of positions with > 1/N attention
            threshold = 1.0 / max(seq_len, 1)
            cache_coverages.append(float((mean_attn > threshold).sum() / seq_len))

            # Lookback: prompt vs generated attention mass. Accumulate raw masses
            # and aggregate as a ratio-of-means below (v3/B1) — the old
            # mean-of-ratios blew up at early steps where gen_mass≈0.
            prompt_masses.append(float(mean_attn[:data.prompt_length].sum()))
            gen_masses.append(float(mean_attn[data.prompt_length:].sum()))

        # Aggregate
        for arr, name in [
            (recency_biases, "recency_bias"),
            (sink_masses, "sink_mass"),
            (cache_coverages, "cache_coverage"),
        ]:
            features.append(float(np.mean(arr)) if arr else 0.0)
            names.append(f"cache_{name}_L{l_idx}")

        # Lookback ratio: ratio-of-means Σ(prompt_mass) / Σ(gen_mass) (v3/B1).
        total_gen = float(np.sum(gen_masses)) if gen_masses else 0.0
        total_prompt = float(np.sum(prompt_masses)) if prompt_masses else 0.0
        features.append(total_prompt / max(total_gen, 1e-12))
        names.append(f"cache_lookback_ratio_L{l_idx}")

        # Attention decay rate: fit exponential decay to mean attention vs distance
        try:
            decay_rate = _fit_attention_decay(data.mean_attention(l_idx), T,
                                              data.prompt_length)
        except Exception:
            # 0.0 sentinel preserves the fixed vector shape; warn so a degenerate
            # input can't silently masquerade as a real feature value (audit #10).
            warnings.warn(
                "cache_attn_decay_rate: decay computation raised; emitting 0.0 "
                "sentinel — audit the run if this repeats",
                RuntimeWarning,
            )
            decay_rate = 0.0
        features.append(decay_rate)
        names.append(f"cache_attn_decay_rate_L{l_idx}")

        # Recency bias trajectory (4 windows)
        if len(recency_biases) >= 4:
            window = len(recency_biases) // 4
            for wi in range(4):
                start = wi * window
                end = start + window if wi < 3 else len(recency_biases)
                features.append(float(np.mean(recency_biases[start:end])))
                names.append(f"cache_recency_traj{wi}_L{l_idx}")
        else:
            for wi in range(4):
                features.append(float(np.mean(recency_biases)) if recency_biases else 0.0)
                names.append(f"cache_recency_traj{wi}_L{l_idx}")

    # ── 2.5.2 Key Space Geometry (from pre-RoPE hooks) ──
    for l_idx in config.sampled_layers:
        keys = data.pre_rope_keys.get(l_idx, [])
        if len(keys) < 2:
            for name in [
                "key_spread", "key_eff_dim", "key_drift",
                "key_novelty_mean", "key_novelty_std",
                "key_novelty_traj0", "key_novelty_traj1", "key_novelty_traj2",
                "key_novelty_traj3", "key_novelty_traj4",
            ]:
                features.append(0.0)
                names.append(f"kv_{name}_L{l_idx}")
            continue

        # Stack keys: T × [num_kv_heads, head_dim] → average across heads
        key_matrix = np.stack([k.mean(axis=0) for k in keys])  # [T_keys, head_dim]

        # Key spread: mean cosine distance from centroid (vectorized)
        centroid = key_matrix.mean(axis=0, keepdims=True).astype(np.float64)  # [1, head_dim]
        km_f64 = key_matrix.astype(np.float64)
        km_norms = np.linalg.norm(km_f64, axis=1, keepdims=True)
        c_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
        sims = (km_f64 * centroid).sum(axis=1) / np.maximum(km_norms.squeeze() * c_norm.squeeze(), 1e-12)
        spread_dists = 1.0 - sims
        features.append(float(spread_dists.mean()))
        names.append(f"kv_key_spread_L{l_idx}")

        # Effective dimensionality (participation ratio of SVD singular values)
        try:
            s = np.linalg.svd(km_f64, compute_uv=False)
            s2 = s ** 2
            s2_sum = s2.sum()
            eff_dim = float((s2_sum ** 2) / (s2 ** 2).sum()) if s2_sum > 1e-12 else 0.0
            # Bound as a fraction of the max possible rank so it doesn't grow
            # toward head_dim with generation length (v3/B4). Residual length
            # coupling (PR still rises with T) is left to analysis-side
            # residualization; subsampling to fixed #keys is a later option.
            eff_dim = eff_dim / max(min(km_f64.shape[0], km_f64.shape[1]), 1)
        except Exception:
            warnings.warn(
                "kv_key_eff_dim: SVD/participation-ratio raised; emitting 0.0 "
                "sentinel — audit the run if this repeats",
                RuntimeWarning,
            )
            eff_dim = 0.0
        features.append(eff_dim)
        names.append(f"kv_key_eff_dim_L{l_idx}")

        # Key drift: cosine distance between first-half and second-half centroids
        mid = len(key_matrix) // 2
        centroid_first = km_f64[:mid].mean(axis=0)
        centroid_second = km_f64[mid:].mean(axis=0)
        features.append(_cosine_dist(centroid_first.astype(np.float32), centroid_second.astype(np.float32)))
        names.append(f"kv_key_drift_L{l_idx}")

        # Key novelty: vectorized via cumulative sum for running centroids
        cumsum = np.cumsum(km_f64, axis=0)  # [T, head_dim]
        counts = np.arange(1, len(km_f64) + 1, dtype=np.float64)[:, np.newaxis]
        running_centroids = cumsum / counts  # [T, head_dim]

        # Cosine distance between key[i] and running_centroid[i-1]
        keys_from_1 = km_f64[1:]   # [T-1, head_dim]
        cents_to_Tm1 = running_centroids[:-1]  # [T-1, head_dim]
        k_norms = np.linalg.norm(keys_from_1, axis=1)
        c_norms = np.linalg.norm(cents_to_Tm1, axis=1)
        dots = (keys_from_1 * cents_to_Tm1).sum(axis=1)
        novelty_sims = dots / np.maximum(k_norms * c_norms, 1e-12)
        novelties = (1.0 - novelty_sims).astype(np.float32)

        if len(novelties) > 0:
            features.append(float(novelties.mean()))
            features.append(float(novelties.std()))
        else:
            features.append(0.0)
            features.append(0.0)
        names.append(f"kv_key_novelty_mean_L{l_idx}")
        names.append(f"kv_key_novelty_std_L{l_idx}")

        traj_idx = _trajectory_indices(len(novelties), 5)
        for i, ti in enumerate(traj_idx):
            features.append(float(novelties[ti]) if ti < len(novelties) else 0.0)
            names.append(f"kv_key_novelty_traj{i}_L{l_idx}")

    # ── 2.5.3 KV Cache Epoch Detection ──
    # (v3/C3: cross-layer key agreement removed — it compared k_proj outputs from
    #  different layers, which live in unrelated learned bases, so the cosine is not
    #  interpretable as "agreement". _extract_cross_layer_key_agreement is retained
    #  below, unused, for a possible within-layer-CKA replacement later.)
    epoch_feats = _extract_epoch_features(data, config)
    for name, val in epoch_feats:
        features.append(val)
        names.append(name)

    return np.array(features, dtype=np.float32), names


def _fit_attention_decay(
    mean_rows: list[NDArray[np.float64]],
    T: int,
    prompt_length: int,
) -> float:
    """Fit exponential decay rate to mean attention vs distance from current position.

    mean_rows: the shared per-(gen,layer) head-mean rows (RawGenerationData
    .mean_attention) — same values the old (attentions, layer_idx) signature
    re-derived per call (C2)."""
    if T < 10:
        return 0.0

    # Sample attention patterns at a few steps
    sample_steps = list(range(T // 4, T, max(1, T // 10)))[:10]
    all_distances = []
    all_weights = []

    for t in sample_steps:
        attn = mean_rows[t]
        seq_len = attn.shape[0]
        if seq_len < 2:
            continue
        current_pos = prompt_length + t
        # Exclude position 0 (BOS / attention sink): it carries large mass at the
        # MAX distance, which flattens/biases the exponential-decay fit (v3/C1).
        distances = np.array([current_pos - i for i in range(1, seq_len)], dtype=np.float64)
        distances = np.maximum(distances, 1)
        all_distances.extend(distances.tolist())
        all_weights.extend(attn[1:].tolist())

    if len(all_distances) < 2:
        return 0.0

    # Simple log-linear regression: log(attn) ~ -lambda * distance
    dist_arr = np.array(all_distances, dtype=np.float64)
    weight_arr = np.array(all_weights, dtype=np.float64)
    mask = weight_arr > 1e-10
    if mask.sum() < 2:
        return 0.0

    log_w = np.log(weight_arr[mask])
    d = dist_arr[mask]
    # Linear regression: log_w = a - lambda * d
    try:
        coeffs = np.polyfit(d, log_w, 1)
        return float(-coeffs[0])  # decay rate (positive = faster decay)
    except Exception:
        warnings.warn(
            "_fit_attention_decay: polyfit failed on degenerate distance/weight "
            "data; emitting 0.0 sentinel",
            RuntimeWarning,
        )
        return 0.0


def _extract_cross_layer_key_agreement(
    data: RawGenerationData,
    config: ExtractionConfig,
) -> list[tuple[str, float]]:
    """Cross-layer key agreement: compare key vectors at same position across layers."""
    results: list[tuple[str, float]] = []
    sampled_layers = config.sampled_layers
    T = len(data.hidden_states)

    if T < 2 or len(sampled_layers) < 2:
        results.append(("cross_layer_early_late_agreement", 0.0))
        results.append(("cross_layer_adjacent_agreement", 0.0))
        results.append(("cross_layer_overall_coherence", 0.0))
        return results

    # Sample every 10th generation step
    sample_steps = list(range(0, T, 10))[:30]

    # Collect agreements — use config thresholds instead of hardcoded values
    early_cutoff = getattr(config, 'early_layer_cutoff', sampled_layers[len(sampled_layers) // 4])
    late_cutoff = getattr(config, 'late_layer_cutoff', sampled_layers[-(len(sampled_layers) // 4)])
    early_layers = [l for l in sampled_layers if l <= early_cutoff]
    late_layers = [l for l in sampled_layers if l >= late_cutoff]
    all_agreements = []
    early_late_agreements = []
    adjacent_agreements = []

    for t in sample_steps:
        for i, l1 in enumerate(sampled_layers):
            keys1 = data.pre_rope_keys.get(l1, [])
            if t >= len(keys1):
                continue
            k1 = keys1[t].mean(axis=0)  # average across KV heads → [head_dim]

            for l2 in sampled_layers[i + 1:]:
                keys2 = data.pre_rope_keys.get(l2, [])
                if t >= len(keys2):
                    continue
                k2 = keys2[t].mean(axis=0)
                sim = _cosine_sim(k1, k2)
                all_agreements.append(sim)

                if l1 in early_layers and l2 in late_layers:
                    early_late_agreements.append(sim)

                # Check if layers are adjacent in sampled set
                if sampled_layers.index(l2) == sampled_layers.index(l1) + 1:
                    adjacent_agreements.append(sim)

    results.append(("cross_layer_early_late_agreement",
                     float(np.mean(early_late_agreements)) if early_late_agreements else 0.0))
    results.append(("cross_layer_adjacent_agreement",
                     float(np.mean(adjacent_agreements)) if adjacent_agreements else 0.0))
    results.append(("cross_layer_overall_coherence",
                     float(np.mean(all_agreements)) if all_agreements else 0.0))

    return results


def _extract_epoch_features(
    data: RawGenerationData,
    config: ExtractionConfig,
) -> list[tuple[str, float]]:
    """KV cache epoch detection: find phase transitions in key space."""
    results: list[tuple[str, float]] = []
    window = config.epoch_window_size
    stride = config.epoch_stride

    all_n_transitions = []
    all_max_transitions = []
    all_regularities = []

    for l_idx in config.sampled_layers:
        keys = data.pre_rope_keys.get(l_idx, [])
        if len(keys) < window + stride:
            continue

        # Build windows
        key_matrix = np.stack([k.mean(axis=0) for k in keys])  # [T, head_dim]
        window_centroids = []
        for start in range(0, len(key_matrix) - window + 1, stride):
            window_centroids.append(key_matrix[start:start + window].mean(axis=0))

        if len(window_centroids) < 2:
            continue

        # Centroid-to-centroid distances between consecutive windows
        boundary_strengths = []
        for i in range(len(window_centroids) - 1):
            boundary_strengths.append(
                _cosine_dist(window_centroids[i], window_centroids[i + 1])
            )

        bs_arr = np.array(boundary_strengths, dtype=np.float32)
        bs_mean = float(bs_arr.mean())
        bs_std = float(bs_arr.std())

        # Count transitions above threshold, as a RATE (fraction of window
        # boundaries that are transitions) so it doesn't grow with generation
        # length (v3/B4).
        threshold = bs_mean + 1.5 * max(bs_std, 1e-6)
        n_trans = float((bs_arr > threshold).sum()) / len(bs_arr)
        all_n_transitions.append(n_trans)
        all_max_transitions.append(float(bs_arr.max()))
        all_regularities.append(bs_std)

    # Aggregate across layers: mean, max, std
    for arr, base_name in [
        (all_n_transitions, "epoch_n_transitions"),
        (all_max_transitions, "epoch_max_transition"),
        (all_regularities, "epoch_regularity"),
    ]:
        if arr:
            results.append((f"{base_name}_mean", float(np.mean(arr))))
            results.append((f"{base_name}_max", float(np.max(arr))))
            results.append((f"{base_name}_std", float(np.std(arr))))
        else:
            results.append((f"{base_name}_mean", 0.0))
            results.append((f"{base_name}_max", 0.0))
            results.append((f"{base_name}_std", 0.0))

    return results


# ── Tier 3: Residual PCA ───────────────────────────────────────────────────────

def extract_tier3(
    data: RawGenerationData,
    config: ExtractionConfig,
    pca_components: F32 | None,
    pca_mean: F32 | None,
) -> tuple[F32, list[str]]:
    """Extract Tier 3: project hidden states onto pre-fitted PCA basis.

    Args:
        pca_components: PCA basis. Either a single [n_components, hidden_dim] array (pooled,
            applied to every pca_layer — legacy) OR a dict {layer_idx: [n_components, hidden_dim]}
            for the C5 per-layer corrected-PCA fix (each layer gets its own basis fit on
            positionally-corrected calibration states).
        pca_mean: matching mean — [hidden_dim] array (legacy) or {layer_idx: [hidden_dim]} (per-layer).

    Returns (feature_vector, feature_names).
    """
    features: list[float] = []
    names: list[str] = []
    T = len(data.hidden_states)

    if T == 0 or pca_components is None or pca_mean is None:
        return np.array([], dtype=np.float32), []

    per_layer = isinstance(pca_components, dict)
    traj_idx = _trajectory_indices(T, config.pca_temporal_samples)

    for l_idx in config.pca_layers:
        layer_offset = l_idx + 1  # skip embedding layer
        if layer_offset >= data.hidden_states[0].shape[0]:
            continue

        if per_layer:
            if l_idx not in pca_components:
                continue
            comp_l, mean_l = pca_components[l_idx], pca_mean[l_idx]
        else:
            comp_l, mean_l = pca_components, pca_mean
        n_components = min(config.pca_components, comp_l.shape[0])
        comp_lt = comp_l[:n_components].T.astype(np.float64)
        mean_l64 = mean_l.astype(np.float64)

        for ti, t in enumerate(traj_idx):
            abs_pos = data.prompt_length + t
            h = data.hidden_states[t][layer_offset]
            h_corrected = _correct_hidden_state(h, layer_offset, abs_pos, data.positional_means)
            h_centered = h_corrected.astype(np.float64) - mean_l64
            projection = h_centered @ comp_lt
            for ci in range(n_components):
                features.append(float(projection[ci]))
                names.append(f"pca_L{l_idx}_t{ti}_c{ci}")

    return np.array(features, dtype=np.float32), names


# ── kNN-LM Baseline ───────────────────────────────────────────────────────────

def extract_knnlm_baseline(
    data: RawGenerationData,
) -> F32 | None:
    """Extract kNN-LM-style single-layer signature.

    Returns the hidden state at the last generated token from the final
    transformer layer (raw, before any PCA — PCA applied during analysis).
    """
    T = len(data.hidden_states)
    if T == 0:
        return None

    # Final transformer layer = index -1 (or num_layers, which is last in the +1 array)
    # hidden_states[t][layer_idx] where last transformer layer = hidden_states[t][-1]
    # But actually -1 is the last entry = final transformer layer output
    last_hidden = data.hidden_states[-1][-1]  # last step, last layer
    return last_hidden.copy()


# ── Full Extraction Pipeline ───────────────────────────────────────────────────

def extract_all_features(
    data: RawGenerationData,
    config: ExtractionConfig,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
) -> ExtractionResult:
    """Run all enabled tiers and concatenate into a single feature vector."""
    all_features: list[F32] = []
    all_names: list[str] = []
    tier_slices: dict[str, tuple[int, int]] = {}
    offset = 0

    if config.enable_tier1:
        f, n = extract_tier1(data, config)
        tier_slices["tier1"] = (offset, offset + len(f))
        all_features.append(f)
        all_names.extend(n)
        offset += len(f)
        logger.debug(f"Tier 1: {len(f)} features")

    if config.enable_tier2:
        f, n = extract_tier2(data, config)
        tier_slices["tier2"] = (offset, offset + len(f))
        all_features.append(f)
        all_names.extend(n)
        offset += len(f)
        logger.debug(f"Tier 2: {len(f)} features")

    if config.enable_tier2_5:
        f, n = extract_tier2_5(data, config)
        tier_slices["tier2_5"] = (offset, offset + len(f))
        all_features.append(f)
        all_names.extend(n)
        offset += len(f)
        logger.debug(f"Tier 2.5: {len(f)} features")

    if config.enable_tier3:
        f, n = extract_tier3(data, config, pca_components, pca_mean)
        tier_slices["tier3"] = (offset, offset + len(f))
        all_features.append(f)
        all_names.extend(n)
        offset += len(f)
        logger.debug(f"Tier 3: {len(f)} features")

    knnlm = None
    if config.enable_knnlm_baseline:
        knnlm = extract_knnlm_baseline(data)

    if all_features:
        combined = np.concatenate(all_features)
    else:
        combined = np.array([], dtype=np.float32)

    logger.info(f"Total features extracted: {len(combined)}")

    return ExtractionResult(
        features=combined,
        feature_names=all_names,
        tier_slices=tier_slices,
        knnlm_baseline=knnlm,
    )
