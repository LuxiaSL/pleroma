"""Cached-bridge replay for ARM A4: teacher-force a continuation against an
INJECTED (possibly surgically-modified) KV cache and extract per-step states.

Ported from kv-rotation `kvrot.sigbridge.replay_extract_cached` (exp11, validated
there against the anamnesis extractor) and EXTENDED to the full vmb battery
capture surface: pre-RoPE keys + gates (exp11's surface) plus values, queries,
attention outputs (the 3,358-dim battery vector's extra families).

Alignment contract: prompt_length = cache length, so the extractor's
prompt/generated split lands exactly on the cache/continuation boundary — the
surgered-cache readout the key/attention features need. T = N-1 per-step entries
for an N-token continuation (same convention as replay_extract).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from anamnesis.extraction.model_loader import LoadedModel
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def cache_length(past_key_values) -> int:
    from anamnesis.extraction.cache_surgery import _extract_kv

    keys, _ = _extract_kv(past_key_values)
    if not keys:
        raise ValueError("cache has no layers")
    return int(keys[0].shape[-2])


def remap_positional_means(
    positional_means: F32 | None, cache_len: int, position_offset: int, n_steps: int
) -> F32 | None:
    """Make extractor lookups (index cache_len + t) return the means of the TRUE
    absolute positions (position_offset + t). Identity for FULL/ROT/REC; only
    NAIVE (survivors keep original positions => offset > cache_len) with a short
    cache needs the copy."""
    if positional_means is None or position_offset == cache_len:
        return positional_means
    if position_offset < cache_len:
        raise ValueError(f"position_offset {position_offset} < cache_len {cache_len}")
    max_pos = positional_means.shape[1]
    if cache_len >= max_pos - 1:
        return positional_means
    remapped = positional_means.copy()
    for t in range(n_steps):
        dst = cache_len + t
        if dst >= max_pos:
            break
        src = min(position_offset + t, max_pos - 1)
        remapped[:, dst, :] = positional_means[:, src, :]
    return remapped


def _slice_head_hook(d: dict[int, list[Tensor]], t_steps: int) -> dict[int, list[F32]] | None:
    """[1, heads, N, head_dim] captures (single cached forward) → {layer: T × [heads, head_dim]}."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in d.items():
        if not tensors:
            continue
        full = tensors[0]
        rows = full[0, :, :t_steps, :].float().cpu().numpy()  # [heads, T, head_dim]
        result[int(layer_idx)] = [rows[:, i, :] for i in range(t_steps)]
    return result or None


def _slice_seq_hook(d: dict[int, list[Tensor]], t_steps: int) -> dict[int, list[F32]] | None:
    """[1, N, width] captures → {layer: T × [width]} (gate_proj / o_proj shapes)."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in d.items():
        if not tensors:
            continue
        full = tensors[0]
        rows = full[0, :t_steps, :].float().cpu().numpy()  # [T, width]
        result[int(layer_idx)] = [rows[i] for i in range(t_steps)]
    return result or None


def replay_extract_cached(
    loaded: LoadedModel,
    past_key_values,
    cont_ids: Tensor | list[int] | NDArray,
    position_offset: int,
    positional_means: F32 | None = None,
) -> RawGenerationData:
    """Teacher-force cont_ids against an injected cache; extract per-step states.

    past_key_values is CONSUMED (the forward appends continuation KV) — pass a
    fresh cache per call (rebuild from the KVSnapshot; snapshot tensors are not
    mutated by to_hf_dynamic_cache). position_offset = absolute RoPE position of
    the first continuation token: cache length for FULL/ROT/REC, survivor-max+1
    for NAIVE.
    """
    device = next(loaded.model.parameters()).device
    if isinstance(cont_ids, torch.Tensor):
        ids = cont_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(np.asarray(cont_ids), dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError(f"replay_extract_cached expects batch=1, got {ids.shape[0]}")

    n = int(ids.shape[1])
    if n < 2:
        raise ValueError(f"need >= 2 continuation tokens, got {n}")
    t_steps = n - 1

    cache_len = cache_length(past_key_values)
    if cache_len <= 0:
        raise ValueError("injected cache is empty — prefill the context first")
    if position_offset < cache_len:
        raise ValueError(f"position_offset {position_offset} < cache_len {cache_len}")
    pm_eff = remap_positional_means(positional_means, cache_len, position_offset, t_steps)

    loaded.clear_hook_state()
    loaded.enable_hooks()
    try:
        with torch.no_grad():
            out = loaded.model(
                ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                position_ids=torch.arange(
                    position_offset, position_offset + n, device=device
                ).unsqueeze(0),
                cache_position=torch.arange(cache_len, cache_len + n, device=device),
            )
        loaded.flush_hooks_to_cpu()

        hs = out.hidden_states  # tuple(num_layers+1) of [1, N, hidden]
        n_hs_layers = len(hs)
        hs_rows = [hs[l][0, :t_steps].float().cpu().numpy() for l in range(n_hs_layers)]
        hidden_states: list[F32] = [
            np.stack([hs_rows[l][i] for l in range(n_hs_layers)]) for i in range(t_steps)
        ]

        att = out.attentions  # tuple(num_layers) of [1, H, N, cache_len + N]
        if att is None or len(att) == 0:
            raise RuntimeError("no attentions returned — model must be loaded eager")
        exp_cols = cache_len + n
        if int(att[0].shape[-1]) != exp_cols:
            raise RuntimeError(
                f"attention columns {int(att[0].shape[-1])} != cache_len + N ({exp_cols}) "
                "— cache injection did not take effect"
            )
        att_rows = [a[0, :, :t_steps, :].float().cpu().numpy() for a in att]  # [H, T, C+N]
        attentions: list[F32] = [
            np.stack([att_rows[l][:, i, : cache_len + i + 1] for l in range(len(att))])
            for i in range(t_steps)
        ]

        logits_rows = out.logits[0, :t_steps].float().cpu().numpy()
        logits: list[F32] = [logits_rows[i] for i in range(t_steps)]
        del out

        pre_rope_keys = _slice_head_hook(loaded.hook_state.pre_rope_keys, t_steps) or {}
        v_proj_values = _slice_head_hook(loaded.hook_state.v_proj_values, t_steps)
        queries = _slice_head_hook(loaded.hook_state.queries, t_steps)
        attn_outputs = _slice_seq_hook(loaded.hook_state.attn_outputs, t_steps)
        gate_activations = _slice_seq_hook(loaded.hook_state.gate_activations, t_steps)
    finally:
        loaded.clear_hook_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chosen_token_ids = ids[0, 1:n].float().cpu().numpy()

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_token_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=cache_len,
        positional_means=pm_eff,
        gate_activations=gate_activations,
        v_proj_values=v_proj_values,
        queries=queries,
        attn_outputs=attn_outputs,
    )
