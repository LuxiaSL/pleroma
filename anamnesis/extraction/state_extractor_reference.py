"""Pure feature extraction: raw tensors → flat feature vector.

This module has NO model awareness. It operates on pre-collected tensors
and can be tested offline with saved tensor samples.

Tiers:
  1   — Activation norms, logit statistics, token probability dynamics (~221 features)
  2   — Attention entropy, head agreement, residual deltas, spectral features (~221 features)
  2.5 — KV cache: attention profiles, key space geometry, cross-layer agreement, epoch detection
  3   — Residual stream PCA projections
  Baseline — kNN-LM single-layer signature
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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

    # ── 1.1 Per-Layer Activation Norms ──
    # hidden_states[t][l+1] = transformer layer l at generation step t
    for l in range(num_layers):
        layer_norms = []
        for t in range(T):
            abs_pos = data.prompt_length + t
            h = data.hidden_states[t][l + 1]  # +1 to skip embedding layer
            h_corrected = _correct_hidden_state(h, l + 1, abs_pos, data.positional_means)
            layer_norms.append(float(np.linalg.norm(h_corrected)))

        if len(layer_norms) == 0:
            layer_norms = [0.0]

        norms = np.array(layer_norms, dtype=np.float32)
        features.append(float(norms.mean()))
        names.append(f"activation_norm_mean_L{l}")
        features.append(float(norms.std()))
        names.append(f"activation_norm_std_L{l}")

        # Trajectory: 5 temporal samples
        for i, ti in enumerate(traj_idx):
            features.append(layer_norms[ti] if ti < len(layer_norms) else 0.0)
            names.append(f"activation_norm_traj{i}_L{l}")

    # ── 1.2 Output Logit Statistics ──
    entropies = []
    top1_probs = []
    top5_masses = []

    for t in range(T):
        probs = _softmax(data.logits[t])
        ent = _safe_entropy(probs)
        entropies.append(ent)

        sorted_probs = np.sort(probs)[::-1]
        top1_probs.append(float(sorted_probs[0]))
        top5_masses.append(float(sorted_probs[:5].sum()))

    if len(entropies) == 0:
        entropies = [0.0]
        top1_probs = [0.0]
        top5_masses = [0.0]

    ent_arr = np.array(entropies, dtype=np.float32)
    t1_arr = np.array(top1_probs, dtype=np.float32)

    features.append(float(ent_arr.mean()))
    names.append("logit_entropy_mean")
    features.append(float(ent_arr.std()))
    names.append("logit_entropy_std")
    for i, ti in enumerate(traj_idx):
        features.append(entropies[ti] if ti < len(entropies) else 0.0)
        names.append(f"logit_entropy_traj{i}")

    features.append(float(t1_arr.mean()))
    names.append("top1_prob_mean")
    features.append(float(t1_arr.std()))
    names.append("top1_prob_std")
    for i, ti in enumerate(traj_idx):
        features.append(top1_probs[ti] if ti < len(top1_probs) else 0.0)
        names.append(f"top1_prob_traj{i}")

    features.append(float(np.mean(top5_masses)))
    names.append("top5_mass_mean")

    # ── 1.3 Token Probability Dynamics ──
    chosen_ranks = []
    surprises = []

    for t in range(T):
        probs = _softmax(data.logits[t])
        chosen_id = int(data.chosen_token_ids[t])

        # Rank of chosen token (0-indexed)
        sorted_indices = np.argsort(probs)[::-1]
        rank = int(np.where(sorted_indices == chosen_id)[0][0]) if chosen_id < len(probs) else len(probs) - 1
        chosen_ranks.append(rank)

        # Surprise = -log(p(chosen))
        chosen_prob = float(probs[chosen_id]) if chosen_id < len(probs) else 1e-10
        chosen_prob = max(chosen_prob, 1e-10)  # avoid log(0)
        surprises.append(-float(np.log(chosen_prob)))

    if len(chosen_ranks) == 0:
        chosen_ranks = [0]
        surprises = [0.0]

    rank_arr = np.array(chosen_ranks, dtype=np.float32)
    surp_arr = np.array(surprises, dtype=np.float32)

    features.append(float(rank_arr.mean()))
    names.append("mean_chosen_rank")
    features.append(float(rank_arr.std()))
    names.append("std_chosen_rank")
    features.append(float(surp_arr.mean()))
    names.append("mean_surprise")
    features.append(float(surp_arr.std()))
    names.append("std_surprise")
    for i, ti in enumerate(traj_idx):
        features.append(surprises[ti] if ti < len(surprises) else 0.0)
        names.append(f"surprise_traj{i}")

    # Bayesian surprise: event boundary count
    if len(surprises) >= config.surprise_window:
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
    # Subsample steps for efficiency (every 5th step, or all if T < 20)
    attn_sample_steps = list(range(0, T, max(1, T // 60))) if T > 60 else list(range(T))
    for l in range(num_layers):
        head_entropies_all = []
        for t in attn_sample_steps:
            attn = data.attentions[t][l]  # [num_heads, seq_len]
            for h in range(num_heads):
                head_entropies_all.append(_safe_entropy(attn[h]))

        ent_arr = np.array(head_entropies_all, dtype=np.float32)
        features.append(float(ent_arr.mean()))
        names.append(f"attn_entropy_mean_L{l}")
        features.append(float(ent_arr.std()))
        names.append(f"attn_entropy_std_L{l}")

    # ── 2.2 Attention Head Agreement Per Layer ──
    # Subsample steps (every 10th) and use vectorized JSD for speed
    agreement_sample_steps = list(range(0, T, max(1, T // 30))) if T > 30 else list(range(T))
    for l in range(num_layers):
        agreements = []
        for t in agreement_sample_steps:
            attn = data.attentions[t][l].astype(np.float64)  # [num_heads, seq_len]
            # Normalize each head's distribution
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            attn_norm = attn / row_sums
            # Vectorized mean JSD: average pairwise JSD across all head pairs
            # Use mean distribution trick: mean JSD ≈ entropy(mean) - mean(entropies)
            mean_dist = attn_norm.mean(axis=0)  # [seq_len]
            entropy_of_mean = _safe_entropy(mean_dist)
            mean_of_entropies = np.mean([_safe_entropy(attn_norm[h]) for h in range(num_heads)])
            # JSD(all heads) ≈ H(mean) - mean(H(individual))
            # This is the generalized JSD, bounded [0, log(num_heads)]
            jsd_approx = max(0.0, entropy_of_mean - mean_of_entropies)
            # Normalize to [0, 1]
            max_jsd = float(np.log(num_heads)) if num_heads > 1 else 1.0
            agreements.append(1.0 - jsd_approx / max(max_jsd, 1e-12))

        agr_arr = np.array(agreements, dtype=np.float32)
        features.append(float(agr_arr.mean()))
        names.append(f"head_agreement_mean_L{l}")
        features.append(float(agr_arr.std()))
        names.append(f"head_agreement_std_L{l}")

    # ── 2.3 Layer-to-Layer Residual Stream Deltas ──
    num_model_layers = data.hidden_states[0].shape[0] - 1 if T > 0 else 28
    for l in range(num_model_layers - 1):
        delta_norms = []
        delta_cosines = []
        for t in range(T):
            abs_pos = data.prompt_length + t
            h_l = _correct_hidden_state(
                data.hidden_states[t][l + 1], l + 1, abs_pos, data.positional_means
            )
            h_l1 = _correct_hidden_state(
                data.hidden_states[t][l + 2], l + 2, abs_pos, data.positional_means
            )
            delta = h_l1 - h_l
            delta_norms.append(float(np.linalg.norm(delta)))
            delta_cosines.append(_cosine_sim(delta, h_l))

        dn_arr = np.array(delta_norms, dtype=np.float32)
        dc_arr = np.array(delta_cosines, dtype=np.float32)
        features.append(float(dn_arr.mean()))
        names.append(f"delta_norm_mean_L{l}")
        features.append(float(dn_arr.std()))
        names.append(f"delta_norm_std_L{l}")
        features.append(float(dc_arr.mean()))
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
    attn_vectors = []
    for t in sampled_steps:
        attn_t = data.attentions[t][layer_idx]  # [num_heads, seq_len]
        # Average across heads
        mean_attn = attn_t.mean(axis=0).astype(np.float64)  # [seq_len]
        attn_vectors.append(mean_attn)

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

        for t in range(T):
            attn = data.attentions[t][l_idx]  # [num_heads, seq_len]
            mean_attn = attn.mean(axis=0).astype(np.float64)  # [seq_len]
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
            decay_rate = _fit_attention_decay(data.attentions, l_idx, T, data.prompt_length)
        except Exception:
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

        # Key spread: mean cosine distance from centroid
        centroid = key_matrix.mean(axis=0)
        dists = [_cosine_dist(k, centroid) for k in key_matrix]
        features.append(float(np.mean(dists)))
        names.append(f"kv_key_spread_L{l_idx}")

        # Effective dimensionality (participation ratio of SVD singular values)
        try:
            _, s, _ = np.linalg.svd(key_matrix.astype(np.float64), full_matrices=False)
            s2 = s ** 2
            s2_sum = s2.sum()
            if s2_sum > 1e-12:
                eff_dim = float((s2_sum ** 2) / (s2 ** 2).sum())
            else:
                eff_dim = 0.0
            # Bound as a fraction of the max possible rank so it doesn't grow
            # toward head_dim with generation length (v3/B4). Residual length
            # coupling (PR still rises with T) is left to analysis-side
            # residualization; subsampling to fixed #keys is a later option.
            eff_dim = eff_dim / max(min(key_matrix.shape[0], key_matrix.shape[1]), 1)
        except Exception:
            eff_dim = 0.0
        features.append(eff_dim)
        names.append(f"kv_key_eff_dim_L{l_idx}")

        # Key drift: cosine distance between first-half and second-half centroids
        mid = len(key_matrix) // 2
        centroid_first = key_matrix[:mid].mean(axis=0)
        centroid_second = key_matrix[mid:].mean(axis=0)
        features.append(_cosine_dist(centroid_first, centroid_second))
        names.append(f"kv_key_drift_L{l_idx}")

        # Key novelty time series
        running_centroid = key_matrix[0].copy().astype(np.float64)
        novelties = []
        for i in range(1, len(key_matrix)):
            nov = _cosine_dist(key_matrix[i], running_centroid.astype(np.float32))
            novelties.append(nov)
            # Update running centroid
            running_centroid = (running_centroid * i + key_matrix[i].astype(np.float64)) / (i + 1)

        if novelties:
            features.append(float(np.mean(novelties)))
            features.append(float(np.std(novelties)))
        else:
            features.append(0.0)
            features.append(0.0)
        names.append(f"kv_key_novelty_mean_L{l_idx}")
        names.append(f"kv_key_novelty_std_L{l_idx}")

        traj_idx = _trajectory_indices(len(novelties), 5)
        for i, ti in enumerate(traj_idx):
            features.append(novelties[ti] if ti < len(novelties) else 0.0)
            names.append(f"kv_key_novelty_traj{i}_L{l_idx}")

    # ── 2.5.3 Cross-Layer Key Agreement: DELETED (v3/C4) ──
    # Raw cross-layer key cosine compared vectors in unrelated learned bases; the
    # basis-invariant replacement is the kv_cka feature family. Removed from the
    # reference 2026-07-12 to restore equivalence with the optimized extractor
    # (prereg-vmb-v1 Stage A item v).

    # ── 2.5.4 KV Cache Epoch Detection ──
    epoch_feats = _extract_epoch_features(data, config)
    for name, val in epoch_feats:
        features.append(val)
        names.append(name)

    return np.array(features, dtype=np.float32), names


def _fit_attention_decay(
    attentions: list[F32],
    layer_idx: int,
    T: int,
    prompt_length: int,
) -> float:
    """Fit exponential decay rate to mean attention vs distance from current position."""
    if T < 10:
        return 0.0

    # Sample attention patterns at a few steps
    sample_steps = list(range(T // 4, T, max(1, T // 10)))[:10]
    all_distances = []
    all_weights = []

    for t in sample_steps:
        attn = attentions[t][layer_idx].mean(axis=0).astype(np.float64)
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
        return 0.0


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

        # Count transitions above threshold
        threshold = bs_mean + 1.5 * max(bs_std, 1e-6)
        n_trans = int((bs_arr > threshold).sum())
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
        pca_components: [n_components, hidden_dim] PCA basis vectors
        pca_mean: [hidden_dim] mean used during PCA fitting

    Returns (feature_vector, feature_names).
    """
    features: list[float] = []
    names: list[str] = []
    T = len(data.hidden_states)

    if T == 0 or pca_components is None or pca_mean is None:
        return np.array([], dtype=np.float32), []

    n_components = min(config.pca_components, pca_components.shape[0])
    traj_idx = _trajectory_indices(T, config.pca_temporal_samples)

    for l_idx in config.pca_layers:
        layer_offset = l_idx + 1  # skip embedding layer
        if layer_offset >= data.hidden_states[0].shape[0]:
            continue

        for ti, t in enumerate(traj_idx):
            abs_pos = data.prompt_length + t
            h = data.hidden_states[t][layer_offset]
            h_corrected = _correct_hidden_state(h, layer_offset, abs_pos, data.positional_means)
            h_centered = h_corrected.astype(np.float64) - pca_mean.astype(np.float64)
            projection = h_centered @ pca_components[:n_components].T
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
