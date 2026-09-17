"""Save and load raw per-token tensors to/from disk.

Enables GPU-free feature engineering iteration: extract once (GPU),
compute features many times (CPU) with different feature configs.

Saves RawGenerationData filtered to sampled layers in compressed npz format.
Load reconstructs a RawGenerationData compatible with state_extractor.

Disk format (per generation):
    raw_tensors/gen_NNN.npz:
        hidden_states:   [T, n_layers_saved, hidden_dim]      float16
        attentions:      [T, n_layers_attn, n_heads, max_seq]  float16 (padded)
        pre_rope_keys:   [T, n_kv_layers, n_kv_heads, head_dim] float16
        logits_values:   [T, top_k]                            float32
        logits_indices:  [T, top_k]                            int32
        chosen_ids:      [T]                                   int32
        actual_lengths:  [T]                                   int32  (for attention unpadding)
        prompt_length:   scalar                                int32
        saved_layers_hs: [n_layers_saved]                      int32  (which layers for hidden_states)
        saved_layers_attn: [n_layers_attn]                     int32  (which layers for attention)
        all_layers:      [total_layers+1]                      int32  (for index mapping)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def save_raw_tensors(
    raw_data: RawGenerationData,
    gen_id: int,
    output_dir: Path,
    config: ExtractionConfig,
    prompt_length: int,
    top_k_logits: int = 50,
    hidden_dtype: str = "float16",
) -> Path:
    """Save RawGenerationData to compressed npz, filtered to sampled layers.

    Parameters
    ----------
    raw_data : RawGenerationData
        The in-memory per-token tensors from a generation run.
    gen_id : int
        Generation ID for filename.
    output_dir : Path
        Directory to save into (e.g., raw_tensors/).
    config : ExtractionConfig
        Extraction config with sampled_layers and pca_layers.
    prompt_length : int
        Number of prompt tokens (stored in npz for reference).
    top_k_logits : int
        How many top logits to save per timestep (default 50).
    hidden_dtype : str
        Dtype for hidden states and attention weights (default float16).

    Returns
    -------
    Path to the saved npz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"gen_{gen_id:03d}.npz"

    T = len(raw_data.hidden_states)
    if T == 0:
        logger.warning(f"gen_{gen_id}: no generation steps, saving empty")
        np.savez_compressed(npz_path, T=np.array(0))
        return npz_path

    save_dtype = np.dtype(hidden_dtype)
    n_total_layers_plus_one = raw_data.hidden_states[0].shape[0]  # num_layers + 1 (incl embedding)

    # ── Determine which layers to save ──
    # Hidden states: sampled_layers ∪ pca_layers (for maximum flexibility)
    hs_layer_indices = sorted(set(config.sampled_layers) | set(config.pca_layers))
    # +1 offset: hidden_states index 0 = embedding, 1..N = transformer layers
    # sampled_layers/pca_layers refer to transformer layer indices
    # So to get transformer layer L, we index hidden_states[:, L+1, :]
    # But we also want the embedding (index 0) if layer 0 is in sampled_layers
    hs_array_indices = []
    for layer_idx in hs_layer_indices:
        arr_idx = layer_idx + 1  # transformer layer L is at array index L+1
        if arr_idx < n_total_layers_plus_one:
            hs_array_indices.append(arr_idx)
    # Also include embedding layer (index 0) if anyone might need it
    # Actually, let's just include it — it's one extra slice
    if 0 not in hs_array_indices:
        hs_array_indices = [0] + hs_array_indices
        hs_layer_indices = [-1] + hs_layer_indices  # -1 = embedding layer

    # Attention: sampled_layers only (these are what T2/T2.5 features use)
    attn_layer_indices = sorted(config.sampled_layers)
    # Attention array: index L directly (no +1 offset like hidden_states)

    # ── Stack hidden states ──
    # raw_data.hidden_states[t] has shape [num_layers+1, hidden_dim]
    # Select only the layers we want
    hs_stacked = np.stack([
        h[hs_array_indices, :] for h in raw_data.hidden_states
    ]).astype(save_dtype)  # [T, n_saved, hidden_dim]

    # ── Stack attentions (with padding) ──
    # raw_data.attentions[t] has shape [num_layers, num_heads, seq_len_at_step]
    # seq_len grows by 1 each step — need padding
    n_heads = raw_data.attentions[0].shape[1] if raw_data.attentions else 0
    actual_lengths = np.array([
        raw_data.attentions[t].shape[2] for t in range(T)
    ], dtype=np.int32)
    max_seq_len = int(actual_lengths.max()) if len(actual_lengths) > 0 else 0

    n_attn_layers = len(attn_layer_indices)
    attn_stacked = np.zeros(
        (T, n_attn_layers, n_heads, max_seq_len), dtype=save_dtype,
    )
    for t in range(T):
        attn_t = raw_data.attentions[t]  # [num_layers, num_heads, seq_len_t]
        seq_len_t = attn_t.shape[2]
        for i, layer_idx in enumerate(attn_layer_indices):
            if layer_idx < attn_t.shape[0]:
                attn_stacked[t, i, :, :seq_len_t] = attn_t[layer_idx, :, :seq_len_t].astype(save_dtype)

    # ── Stack pre-RoPE keys ──
    # raw_data.pre_rope_keys: dict[layer_idx → list of T arrays [n_kv_heads, head_dim]]
    kv_layer_indices = sorted(raw_data.pre_rope_keys.keys())
    if kv_layer_indices:
        n_kv_heads = raw_data.pre_rope_keys[kv_layer_indices[0]][0].shape[0]
        head_dim = raw_data.pre_rope_keys[kv_layer_indices[0]][0].shape[1]
        keys_stacked = np.zeros(
            (T, len(kv_layer_indices), n_kv_heads, head_dim), dtype=save_dtype,
        )
        for i, layer_idx in enumerate(kv_layer_indices):
            key_list = raw_data.pre_rope_keys[layer_idx]
            for t in range(min(T, len(key_list))):
                keys_stacked[t, i] = key_list[t].astype(save_dtype)
    else:
        keys_stacked = np.array([], dtype=save_dtype)

    # ── Extract top-k logits + pre-compute entropy ──
    # raw_data.logits[t] has shape [vocab_size]
    # Entropy needs full distribution — save it pre-computed since we discard most logits
    if raw_data.logits and len(raw_data.logits) > 0:
        k = min(top_k_logits, raw_data.logits[0].shape[0])
        logits_values = np.zeros((T, k), dtype=np.float32)
        logits_indices = np.zeros((T, k), dtype=np.int32)
        logits_entropy = np.zeros(T, dtype=np.float32)
        for t in range(min(T, len(raw_data.logits))):
            logit_vec = raw_data.logits[t]
            # Top-k extraction
            top_idx = np.argpartition(logit_vec, -k)[-k:]
            top_idx_sorted = top_idx[np.argsort(logit_vec[top_idx])[::-1]]
            logits_values[t] = logit_vec[top_idx_sorted]
            logits_indices[t] = top_idx_sorted
            # Pre-compute entropy from full distribution
            probs = np.exp(logit_vec - np.max(logit_vec)).astype(np.float64)
            probs /= probs.sum()
            probs = probs[probs > 0]
            logits_entropy[t] = float(-np.sum(probs * np.log(probs)))
    else:
        logits_values = np.array([], dtype=np.float32)
        logits_indices = np.array([], dtype=np.int32)
        logits_entropy = np.array([], dtype=np.float32)

    # ── Stack gate activations (SwiGLU gate_proj, pre-SiLU) ──
    gate_layer_indices: list[int] = []
    if raw_data.gate_activations:
        gate_layer_indices = sorted(raw_data.gate_activations.keys())
    if gate_layer_indices:
        intermediate_size = raw_data.gate_activations[gate_layer_indices[0]][0].shape[0]
        gates_stacked = np.zeros(
            (T, len(gate_layer_indices), intermediate_size), dtype=save_dtype,
        )
        for i, layer_idx in enumerate(gate_layer_indices):
            gate_list = raw_data.gate_activations[layer_idx]
            for t in range(min(T, len(gate_list))):
                gates_stacked[t, i] = gate_list[t].astype(save_dtype)
    else:
        gates_stacked = np.array([], dtype=save_dtype)

    # ── Chosen token IDs ──
    chosen_ids = raw_data.chosen_token_ids.astype(np.int32)

    # ── Save ──
    save_dict = {
        "hidden_states": hs_stacked,
        "attentions": attn_stacked,
        "pre_rope_keys": keys_stacked,
        "logits_values": logits_values,
        "logits_indices": logits_indices,
        "logits_entropy": logits_entropy,
        "chosen_ids": chosen_ids,
        "actual_lengths": actual_lengths,
        "prompt_length": np.array(prompt_length, dtype=np.int32),
        "saved_layers_hs": np.array(hs_layer_indices, dtype=np.int32),
        "saved_layers_attn": np.array(attn_layer_indices, dtype=np.int32),
        "kv_layer_indices": np.array(kv_layer_indices, dtype=np.int32),
        "all_layers_count": np.array(n_total_layers_plus_one, dtype=np.int32),
        "gate_activations": gates_stacked,
        "gate_layer_indices": np.array(gate_layer_indices, dtype=np.int32),
    }

    # Include positional means if available (needed for feature extraction)
    if raw_data.positional_means is not None:
        save_dict["positional_means"] = raw_data.positional_means.astype(save_dtype)

    np.savez_compressed(npz_path, **save_dict)

    size_mb = npz_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved raw tensors: {npz_path.name} ({size_mb:.1f} MB, T={T})")

    return npz_path


def save_raw_tensors_v3(
    raw_data: RawGenerationData,
    gen_id: int,
    output_dir: Path,
    prompt_length: int,
    top_k_logits: int = 50,
    hidden_dtype: str = "float16",
    input_ids: NDArray | list[int] | None = None,
) -> Path:
    """Save the v3 replay-extract capture surface (compressed npz).

    Differences from save_raw_tensors (the v2 generate-path saver):
      - hidden_states + attentions banked for **all layers** (not sampled∪pca / sampled)
      - pre_rope_keys + **v_proj_values** banked for all layers
      - **queries** (pre-RoPE) banked for whatever layers raw_data.queries holds
        (sampled pre-vmb; all layers from the vmb battery capture surface on)
      - **attn_outputs** (o_proj) banked for whatever layers raw_data.attn_outputs holds
        (vmb battery surface; absent in pre-vmb saves — loaders tolerate absence)
      - gate_activations banked for whatever layers raw_data.gate_activations holds (sampled)
      - positional_means is **NOT** banked per-gen (deduped — it is identical across all
        gens; lives once in the run's calibration dir, injected at feature-compute time)

    Per-surface layer-index arrays are stored so load_raw_tensors can reconstruct
    the full [num_layers(+1)] tensors. Tagged extraction_version=3.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"gen_{gen_id:03d}.npz"

    T = len(raw_data.hidden_states)
    if T == 0:
        logger.warning(f"gen_{gen_id}: no generation steps, saving empty")
        np.savez_compressed(npz_path, T=np.array(0))
        return npz_path

    save_dtype = np.dtype(hidden_dtype)
    n_total_layers_plus_one = raw_data.hidden_states[0].shape[0]  # num_layers + 1 (incl embedding)

    # ── Hidden states: ALL layers (array idx 0 = embedding, 1..N = transformer) ──
    hs_stacked = np.stack(
        [h.astype(save_dtype) for h in raw_data.hidden_states]
    )  # [T, n_total_layers_plus_one, hidden_dim]
    hs_layer_indices = [-1] + list(range(n_total_layers_plus_one - 1))  # -1 = embedding

    # ── Attentions: ALL layers, padded to max seq ──
    n_attn_layers = raw_data.attentions[0].shape[0] if raw_data.attentions else 0
    n_heads = raw_data.attentions[0].shape[1] if raw_data.attentions else 0
    actual_lengths = np.array(
        [raw_data.attentions[t].shape[2] for t in range(T)], dtype=np.int32
    )
    max_seq_len = int(actual_lengths.max()) if len(actual_lengths) > 0 else 0
    attn_stacked = np.zeros((T, n_attn_layers, n_heads, max_seq_len), dtype=save_dtype)
    for t in range(T):
        attn_t = raw_data.attentions[t]
        seq_len_t = attn_t.shape[2]
        attn_stacked[t, :, :, :seq_len_t] = attn_t[:, :, :seq_len_t].astype(save_dtype)
    attn_layer_indices = list(range(n_attn_layers))

    # ── Stack a per-layer dict {layer_idx → list of T arrays} into [T, n_layers, *shape] ──
    def _stack_layer_dict(d: dict[int, list] | None) -> tuple[NDArray, NDArray]:
        layer_list = sorted(d.keys()) if d else []
        if not layer_list:
            return np.array([], dtype=save_dtype), np.array([], dtype=np.int32)
        sample = np.asarray(d[layer_list[0]][0])
        stacked = np.zeros((T, len(layer_list), *sample.shape), dtype=save_dtype)
        for i, l in enumerate(layer_list):
            seq = d[l]
            for t in range(min(T, len(seq))):
                stacked[t, i] = np.asarray(seq[t]).astype(save_dtype)
        return stacked, np.array(layer_list, dtype=np.int32)

    keys_stacked, kv_layer_indices = _stack_layer_dict(raw_data.pre_rope_keys)
    values_stacked, value_layer_indices = _stack_layer_dict(raw_data.v_proj_values)
    queries_stacked, query_layer_indices = _stack_layer_dict(raw_data.queries)
    gates_stacked, gate_layer_indices = _stack_layer_dict(raw_data.gate_activations)
    attn_out_stacked, attn_out_layer_indices = _stack_layer_dict(raw_data.attn_outputs)
    # MoE expert routing (vmb arm A7, M6) — dense softmax [T, n_moe_layers, n_experts] + branch norms [T, n_moe_layers, 2]
    router_dist_stacked, router_layer_indices = _stack_layer_dict(raw_data.router_dist)
    router_norms_stacked, router_norms_layer_indices = _stack_layer_dict(raw_data.router_branch_norms)
    router_logit_stacked, router_logit_layer_indices = _stack_layer_dict(raw_data.router_logit_norms)

    # ── Top-k logits + precomputed exact entropy (full vocab discarded) ──
    if raw_data.logits and len(raw_data.logits) > 0:
        k = min(top_k_logits, raw_data.logits[0].shape[0])
        logits_values = np.zeros((T, k), dtype=np.float32)
        logits_indices = np.zeros((T, k), dtype=np.int32)
        logits_entropy = np.zeros(T, dtype=np.float32)
        for t in range(min(T, len(raw_data.logits))):
            logit_vec = raw_data.logits[t]
            top_idx = np.argpartition(logit_vec, -k)[-k:]
            top_idx_sorted = top_idx[np.argsort(logit_vec[top_idx])[::-1]]
            logits_values[t] = logit_vec[top_idx_sorted]
            logits_indices[t] = top_idx_sorted
            probs = np.exp(logit_vec - np.max(logit_vec)).astype(np.float64)
            probs /= probs.sum()
            probs = probs[probs > 0]
            logits_entropy[t] = float(-np.sum(probs * np.log(probs)))
    else:
        logits_values = np.array([], dtype=np.float32)
        logits_indices = np.array([], dtype=np.int32)
        logits_entropy = np.array([], dtype=np.float32)

    chosen_ids = raw_data.chosen_token_ids.astype(np.int32)

    save_dict = {
        "hidden_states": hs_stacked,
        "attentions": attn_stacked,
        "pre_rope_keys": keys_stacked,
        "v_proj_values": values_stacked,
        "queries": queries_stacked,
        "gate_activations": gates_stacked,
        "logits_values": logits_values,
        "logits_indices": logits_indices,
        "logits_entropy": logits_entropy,
        "chosen_ids": chosen_ids,
        "actual_lengths": actual_lengths,
        "prompt_length": np.array(prompt_length, dtype=np.int32),
        "saved_layers_hs": np.array(hs_layer_indices, dtype=np.int32),
        "saved_layers_attn": np.array(attn_layer_indices, dtype=np.int32),
        "kv_layer_indices": kv_layer_indices,
        "value_layer_indices": value_layer_indices,
        "query_layer_indices": query_layer_indices,
        "gate_layer_indices": gate_layer_indices,
        "attn_outputs": attn_out_stacked,
        "attn_output_layer_indices": attn_out_layer_indices,
        "router_dist": router_dist_stacked,
        "router_layer_indices": router_layer_indices,
        "router_branch_norms": router_norms_stacked,
        "router_norms_layer_indices": router_norms_layer_indices,
        "router_logit_norms": router_logit_stacked,
        "router_logit_layer_indices": router_logit_layer_indices,
        "all_layers_count": np.array(n_total_layers_plus_one, dtype=np.int32),
        "extraction_version": np.array(3, dtype=np.int32),
    }
    # Bank the full realized token sequence [prompt + generated] so any future
    # re-processing needs no re-tokenization (cheap: ~L int32 ≈ a few KB).
    if input_ids is not None:
        save_dict["input_ids"] = np.asarray(input_ids, dtype=np.int32)
    np.savez_compressed(npz_path, **save_dict)

    size_mb = npz_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved v3 raw tensors: {npz_path.name} ({size_mb:.1f} MB, T={T}, all-layer)")
    return npz_path


#: Surface names accepted by load_raw_tensors(surfaces=...).
VALID_SURFACES: frozenset[str] = frozenset(
    {"hidden", "attention", "keys", "values", "queries", "gate", "logits", "attn_out", "routing"}
)


def load_raw_tensors(
    gen_id: int,
    raw_dir: Path,
    *,
    surfaces: Sequence[str] | None = None,
    attn_layers: Sequence[int] | None = None,
) -> RawGenerationData:
    """Load saved raw tensors back into RawGenerationData.

    The returned object is compatible with state_extractor.extract_all_features(),
    with the caveat that only sampled layers are populated — features that require
    all layers (like per-layer norms in Tier 1) will only have data at saved layers.

    Parameters
    ----------
    gen_id : int
        Generation ID to load.
    raw_dir : Path
        Directory containing raw tensor npz files.
    surfaces : sequence of str, optional (keyword-only)
        Which surfaces to materialize. None (default) loads everything the npz
        holds — the exact historical behavior. Valid names (see VALID_SURFACES):
        "hidden", "attention", "keys", "values", "queries", "gate", "logits".
        Surfaces not requested are left empty ([] / {} / None) and — crucially —
        their npz members are never decompressed, so a hidden-only load skips
        the (large) attention/gate/logits work entirely. positional_means is
        loaded whenever present and "hidden" is requested (or surfaces is None),
        since it is a hidden-space correction.
    attn_layers : sequence of int, optional (keyword-only)
        If given (and "attention" is loaded), only these transformer layers are
        filled in the per-timestep attention arrays; the layer AXIS keeps its
        full [num_layers] extent (zeros elsewhere), so absolute-layer indexing
        in state_extractor and the feature families is unchanged. Layers not
        present in the npz are ignored. For v3 all-layer banks, passing
        config.sampled_layers avoids rebuilding ~25 layers nothing reads.
        None (default) fills every saved layer — the exact historical behavior.

    Returns
    -------
    RawGenerationData reconstructed from saved tensors.

    Raises
    ------
    FileNotFoundError if the npz is missing; ValueError on unknown surface names.
    """
    if surfaces is not None:
        want = frozenset(surfaces)
        unknown = want - VALID_SURFACES
        if unknown:
            raise ValueError(
                f"Unknown surface name(s) {sorted(unknown)}; "
                f"valid surfaces: {sorted(VALID_SURFACES)}"
            )
    else:
        want = VALID_SURFACES

    npz_path = raw_dir / f"gen_{gen_id:03d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Raw tensor file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    T_check = data.get("T")
    if T_check is not None and int(T_check) == 0:
        return RawGenerationData(
            hidden_states=[],
            attentions=[],
            logits=[],
            chosen_token_ids=np.array([], dtype=np.float32),
            pre_rope_keys={},
            prompt_length=0,
        )

    # ── Always-cheap metadata (npz member access decompresses lazily, per member) ──
    chosen_ids = data["chosen_ids"].astype(np.float32)      # [T]
    actual_lengths = data["actual_lengths"]                  # [T]
    prompt_length = int(data["prompt_length"])
    n_total_layers = int(data["all_layers_count"])           # num_layers + 1
    T = int(actual_lengths.shape[0])

    # ── Reconstruct hidden_states ──
    # Expand back to [num_layers+1, hidden_dim] per timestep; unsaved layers
    # stay zero (features won't use them). Vectorized: one upcast + one
    # fancy-indexed assignment instead of a per-(t, layer) Python loop.
    hidden_states: list[F32] = []
    if "hidden" in want:
        hs_stacked = data["hidden_states"]                   # [T, n_saved, hidden_dim] f16
        saved_layers_hs = data["saved_layers_hs"]            # layer indices (-1 = embedding)
        hidden_dim = hs_stacked.shape[2] if hs_stacked.ndim == 3 else 0
        src_pos: list[int] = []
        dst_idx: list[int] = []
        for i, layer_label in enumerate(saved_layers_hs):
            layer_label = int(layer_label)
            arr_idx = 0 if layer_label == -1 else layer_label + 1  # transformer layer offset
            if arr_idx < n_total_layers:
                src_pos.append(i)
                dst_idx.append(arr_idx)
        full_hs = np.zeros((T, n_total_layers, hidden_dim), dtype=np.float32)
        if src_pos:
            full_hs[:, dst_idx] = hs_stacked[:, src_pos].astype(np.float32)
        hidden_states = list(full_hs)  # per-t views, [num_layers+1, hidden_dim]

    # ── Reconstruct attentions ──
    # Expand to [num_layers_no_embed, num_heads, seq_len_t] per timestep.
    # Only the requested layers are filled (full layer axis preserved); the
    # f16→f32 upcast happens implicitly on assignment, never on the full bank.
    attentions: list[F32] = []
    if "attention" in want:
        attn_stacked = data["attentions"]                    # [T, n_attn, n_heads, max_seq] f16
        saved_layers_attn = data["saved_layers_attn"]        # layer indices
        n_layers_no_embed = n_total_layers - 1               # exclude embedding
        n_heads = attn_stacked.shape[2] if attn_stacked.ndim == 4 else 0
        wanted_attn = None if attn_layers is None else {int(l) for l in attn_layers}
        attn_src_pos: list[int] = []
        attn_dst_idx: list[int] = []
        for i, layer_idx in enumerate(saved_layers_attn):
            layer_idx = int(layer_idx)
            if layer_idx >= n_layers_no_embed:
                continue
            if wanted_attn is not None and layer_idx not in wanted_attn:
                continue
            attn_src_pos.append(i)
            attn_dst_idx.append(layer_idx)
        for t in range(T):
            seq_len_t = int(actual_lengths[t])
            full_a = np.zeros((n_layers_no_embed, n_heads, seq_len_t), dtype=np.float32)
            if attn_src_pos:
                full_a[attn_dst_idx, :, :seq_len_t] = attn_stacked[t, attn_src_pos, :, :seq_len_t]
            attentions.append(full_a)

    # ── Reconstruct logits ──
    # We only have top-k, not full vocab. Create a partial representation
    # (dense [vocab_size] with -1e9 sentinels — ~0.5MB per timestep). Skip the
    # "logits" surface if you only need the exact entropy: it is pre-computed
    # at save time and available via np.load(...)["logits_entropy"].
    logits: list[F32] = []
    if "logits" in want:
        logits_values = data["logits_values"]                # [T, k]
        logits_indices = data["logits_indices"]              # [T, k]
        if logits_values.size > 0:
            # Determine vocab size from max index
            vocab_size = int(logits_indices.max()) + 1 if logits_indices.size > 0 else 128256
            for t in range(T):
                # Create full logit vector with very negative values for unobserved positions
                full_logits = np.full(vocab_size, -1e9, dtype=np.float32)
                valid_mask = logits_indices[t] >= 0
                if np.any(valid_mask):
                    full_logits[logits_indices[t][valid_mask]] = logits_values[t][valid_mask]
                logits.append(full_logits)

    # ── Reconstruct pre-RoPE keys ──
    pre_rope_keys: dict[int, list[F32]] = {}
    if "keys" in want and data["pre_rope_keys"].size > 0:
        keys_stacked = data["pre_rope_keys"].astype(np.float32)
        kv_layer_indices = data["kv_layer_indices"] if "kv_layer_indices" in data else np.array([], dtype=np.int32)
        for i, layer_idx in enumerate(kv_layer_indices):
            pre_rope_keys[int(layer_idx)] = [keys_stacked[t, i] for t in range(T)]

    # ── Reconstruct v_proj values + pre-RoPE queries (v3 surface; absent in v2 npz) ──
    v_proj_values: dict[int, list[F32]] | None = None
    if "values" in want and "v_proj_values" in data and data["v_proj_values"].size > 0:
        values_stacked = data["v_proj_values"].astype(np.float32)
        value_layer_indices = data["value_layer_indices"] if "value_layer_indices" in data else np.array([], dtype=np.int32)
        v_proj_values = {}
        for i, layer_idx in enumerate(value_layer_indices):
            v_proj_values[int(layer_idx)] = [values_stacked[t, i] for t in range(T)]

    queries: dict[int, list[F32]] | None = None
    if "queries" in want and "queries" in data and data["queries"].size > 0:
        queries_stacked = data["queries"].astype(np.float32)
        query_layer_indices = data["query_layer_indices"] if "query_layer_indices" in data else np.array([], dtype=np.int32)
        queries = {}
        for i, layer_idx in enumerate(query_layer_indices):
            queries[int(layer_idx)] = [queries_stacked[t, i] for t in range(T)]

    # ── Reconstruct attention outputs (o_proj; vmb surface, absent in pre-vmb npz) ──
    attn_outputs: dict[int, list[F32]] | None = None
    if "attn_out" in want and "attn_outputs" in data and data["attn_outputs"].size > 0:
        attn_out_stacked = data["attn_outputs"].astype(np.float32)
        attn_out_layer_indices = (
            data["attn_output_layer_indices"]
            if "attn_output_layer_indices" in data
            else np.array([], dtype=np.int32)
        )
        attn_outputs = {}
        for i, layer_idx in enumerate(attn_out_layer_indices):
            attn_outputs[int(layer_idx)] = [attn_out_stacked[t, i] for t in range(T)]

    # ── Reconstruct gate activations ──
    gate_activations: dict[int, list[F32]] | None = None
    if "gate" in want:
        gate_layer_indices_arr = data.get("gate_layer_indices")
        gates_stacked_arr = data.get("gate_activations")
        if (
            gate_layer_indices_arr is not None
            and gates_stacked_arr is not None
            and gates_stacked_arr.size > 0
            and gate_layer_indices_arr.size > 0
        ):
            gates_stacked_f32 = gates_stacked_arr.astype(np.float32)
            gate_activations = {}
            for i, layer_idx in enumerate(gate_layer_indices_arr):
                gate_activations[int(layer_idx)] = [gates_stacked_f32[t, i] for t in range(T)]

    # ── Reconstruct MoE expert routing (vmb arm A7, M6; absent in non-MoE npz) ──
    router_dist: dict[int, list[F32]] | None = None
    router_branch_norms: dict[int, list[F32]] | None = None
    router_logit_norms: dict[int, list[F32]] | None = None
    if "routing" in want:
        rd_arr = data.get("router_dist")
        rd_idx = data.get("router_layer_indices")
        if rd_arr is not None and rd_arr.size > 0 and rd_idx is not None and rd_idx.size > 0:
            rd_f32 = rd_arr.astype(np.float32)
            router_dist = {int(l): [rd_f32[t, i] for t in range(T)] for i, l in enumerate(rd_idx)}
        rn_arr = data.get("router_branch_norms")
        rn_idx = data.get("router_norms_layer_indices")
        if rn_arr is not None and rn_arr.size > 0 and rn_idx is not None and rn_idx.size > 0:
            rn_f32 = rn_arr.astype(np.float32)
            router_branch_norms = {int(l): [rn_f32[t, i] for t in range(T)] for i, l in enumerate(rn_idx)}
        rl_arr = data.get("router_logit_norms")
        rl_idx = data.get("router_logit_layer_indices")
        if rl_arr is not None and rl_arr.size > 0 and rl_idx is not None and rl_idx.size > 0:
            rl_f32 = rl_arr.astype(np.float32)
            router_logit_norms = {int(l): [rl_f32[t, i] for t in range(T)] for i, l in enumerate(rl_idx)}

    # ── Positional means (hidden-space correction; rides with "hidden") ──
    positional_means: F32 | None = None
    if "hidden" in want and "positional_means" in data:
        positional_means = data["positional_means"].astype(np.float32)

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=positional_means,
        gate_activations=gate_activations,
        v_proj_values=v_proj_values,
        queries=queries,
        attn_outputs=attn_outputs,
        router_dist=router_dist,
        router_branch_norms=router_branch_norms,
        router_logit_norms=router_logit_norms,
    )


def list_raw_tensor_ids(raw_dir: Path) -> list[int]:
    """List all generation IDs that have saved raw tensors."""
    ids: list[int] = []
    for npz_path in sorted(raw_dir.glob("gen_*.npz")):
        stem = npz_path.stem  # gen_NNN
        try:
            gen_id = int(stem.split("_")[1])
            ids.append(gen_id)
        except (IndexError, ValueError):
            continue
    return ids
