"""Replay-extract: re-process a banked token sequence through ONE instrumented
forward and produce a RawGenerationData matching the generate-path alignment.

Why replay (vs re-generating):
    A decoder forward is a deterministic function of the input token sequence.
    Teacher-forcing the full realized sequence [prompt + generated] in a single
    causal forward reproduces every per-position internal state that incremental
    generation produced — so v2↔v3 signatures stay byte-comparable (modulo the
    ~1e-3 cache-vs-no-cache kernel noise, which washes out of aggregate features).
    One forward instead of an N-step autoregressive loop → much cheaper.

Alignment (must match extraction.generation_runner._convert_outputs_to_raw):
    Let the prompt be positions 0..P-1 and generated tokens g_0..g_{N-1} occupy
    positions P..P+N-1. The generate path banks T = N-1 per-step entries where
    entry i (i = 0..N-2) is the model state AT generated token g_i (position P+i),
    whose logits predict g_{i+1} = chosen_ids[i]. We reproduce that by slicing the
    single forward's outputs at positions P .. P+N-2.

The single forward uses use_cache=False so attention is the full causal matrix in
one pass; eager attention (required, set at load time) makes output_attentions work.
Hooks (k/v/q/gate) fire exactly once and capture the whole sequence.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import NDArray

from anamnesis.extraction.model_loader import LoadedModel
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def replay_extract(
    loaded: LoadedModel,
    full_token_ids: list[int] | NDArray | torch.Tensor,
    prompt_length: int,
    positional_means: F32 | None = None,
) -> RawGenerationData:
    """Teacher-force a full token sequence and extract per-step internal states.

    Parameters
    ----------
    loaded : LoadedModel
        Model loaded with eager attention + the desired hooks (for the full v3
        surface: key_layers=value_layers=all, query_layers=sampled, gate on sampled).
    full_token_ids : sequence of int
        The realized [prompt + generated] token ids, length L = P + N.
    prompt_length : int
        Number of prompt tokens P (states before this are not banked).
    positional_means : array, optional
        Calibration positional means, attached to the result for positional
        correction (deduped — not banked per-gen).

    Returns
    -------
    RawGenerationData with T = N - 1 per-step entries, aligned to the generate path.
    """
    P = int(prompt_length)
    device = next(loaded.model.parameters()).device

    if isinstance(full_token_ids, torch.Tensor):
        ids = full_token_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(np.asarray(full_token_ids), dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError(f"replay_extract expects a single sequence, got batch {ids.shape[0]}")

    L = int(ids.shape[1])
    N = L - P  # number of generated tokens
    if P <= 0 or P >= L:
        raise ValueError(f"prompt_length {P} out of range for sequence length {L}")
    if N < 2:
        raise ValueError(
            f"need >=2 generated tokens for a per-step transition (got N={N}, L={L}, P={P})"
        )
    T = N - 1  # number of banked per-step states (positions P .. P+N-2)

    # ── One instrumented forward over the full causal sequence ──
    loaded.clear_hook_state()
    loaded.enable_hooks()
    with torch.no_grad():
        out = loaded.model(
            ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    loaded.flush_hooks_to_cpu()

    # ── Hidden states (all layers): rows P .. P+T-1, per layer [T, hidden] ──
    hs = out.hidden_states  # tuple length num_layers+1, each [1, L, hidden]
    n_hs_layers = len(hs)
    hs_rows = [hs[l][0, P:P + T].float().cpu().numpy() for l in range(n_hs_layers)]  # [T, hidden]
    hidden_states: list[F32] = [
        np.stack([hs_rows[l][i] for l in range(n_hs_layers)]) for i in range(T)
    ]  # T × [num_layers+1, hidden]

    # ── Attentions (all layers): causal row at each gen position [num_layers, H, P+i+1] ──
    att = out.attentions  # tuple length num_layers, each [1, H, L, L]
    n_att_layers = len(att)
    att_rows = [att[l][0, :, P:P + T, :].float().cpu().numpy() for l in range(n_att_layers)]  # [H, T, L]
    attentions: list[F32] = [
        np.stack([att_rows[l][:, i, : P + i + 1] for l in range(n_att_layers)])
        for i in range(T)
    ]  # T × [num_layers, H, P+i+1]

    # ── Logits at gen positions: predicts g_{i+1} ──
    logits_rows = out.logits[0, P:P + T].float().cpu().numpy()  # [T, vocab]
    logits: list[F32] = [logits_rows[i] for i in range(T)]

    # ── Chosen ids = realized generated tokens g_1 .. g_{N-1} (positions P+1 .. P+N-1) ──
    chosen_token_ids = ids[0, P + 1:P + N].float().cpu().numpy()  # [T]

    # ── Hook surfaces (single capture each): slice the gen-position rows ──
    def _slice_head_hook(d: dict[int, list[torch.Tensor]]) -> dict[int, list[F32]] | None:
        """For dicts of [1, heads, L, head_dim] captures → {layer: T × [heads, head_dim]}."""
        result: dict[int, list[F32]] = {}
        for layer_idx, tensors in d.items():
            if not tensors:
                continue
            full = tensors[0]  # [1, heads, L, head_dim]
            rows = full[0, :, P:P + T, :].float().cpu().numpy()  # [heads, T, head_dim]
            result[int(layer_idx)] = [rows[:, i, :] for i in range(T)]
        return result or None

    pre_rope_keys = _slice_head_hook(loaded.hook_state.pre_rope_keys) or {}
    v_proj_values = _slice_head_hook(loaded.hook_state.v_proj_values)
    queries = _slice_head_hook(loaded.hook_state.queries)

    # ── Attention outputs (o_proj): [1, L, hidden] captures → {layer: T × [hidden]} ──
    attn_outputs: dict[int, list[F32]] | None = None
    if loaded.hook_state.attn_outputs:
        attn_outputs = {}
        for layer_idx, tensors in loaded.hook_state.attn_outputs.items():
            if not tensors:
                continue
            full = tensors[0]  # [1, L, hidden]
            rows = full[0, P:P + T, :].float().cpu().numpy()  # [T, hidden]
            attn_outputs[int(layer_idx)] = [rows[i] for i in range(T)]
        attn_outputs = attn_outputs or None

    # ── Gate activations: [1, L, intermediate] captures → {layer: T × [intermediate]} ──
    gate_activations: dict[int, list[F32]] | None = None
    if loaded.hook_state.gate_activations:
        gate_activations = {}
        for layer_idx, tensors in loaded.hook_state.gate_activations.items():
            if not tensors:
                continue
            full = tensors[0]  # [1, L, intermediate]
            rows = full[0, P:P + T, :].float().cpu().numpy()  # [T, intermediate]
            gate_activations[int(layer_idx)] = [rows[i] for i in range(T)]
        gate_activations = gate_activations or None

    # ── MoE router (vmb arm A7, M6): dense dist [1, L, n_experts] + branch norms [L] captured ONCE
    # in the single forward; slice the gen-position rows P..P+T-1 (same positions as gate). ──
    router_dist: dict[int, list[F32]] | None = None
    if loaded.hook_state.router_dist:
        router_dist = {}
        for layer_idx, tensors in loaded.hook_state.router_dist.items():
            if not tensors:
                continue
            full = tensors[0]  # [1, L, n_experts]
            rows = full[0, P:P + T, :].float().cpu().numpy()  # [T, n_experts]
            router_dist[int(layer_idx)] = [rows[i] for i in range(T)]
        router_dist = router_dist or None

    # branch norms [shared, routed] + the v2.1 per-token cos(shared_out, routed_out) 3rd column,
    # derived from the transient branch OUTPUT VECTORS (router_shared_vec / router_routed_vec), each
    # [n_tok_full, hidden] captured once in the single forward → slice P:P+T, cosine per row.
    router_branch_norms: dict[int, list[F32]] | None = None
    if loaded.hook_state.router_shared_norm and loaded.hook_state.router_routed_norm:
        router_branch_norms = {}
        for layer_idx in list(loaded.hook_state.router_shared_norm.keys()):
            sh = loaded.hook_state.router_shared_norm.get(layer_idx)
            ro = loaded.hook_state.router_routed_norm.get(layer_idx)
            if not sh or not ro:
                continue
            sh_rows = sh[0].reshape(-1)[P:P + T].float().cpu().numpy()  # [T]
            ro_rows = ro[0].reshape(-1)[P:P + T].float().cpu().numpy()  # [T]
            sv = loaded.hook_state.router_shared_vec.get(layer_idx)
            rv = loaded.hook_state.router_routed_vec.get(layer_idx)
            if sv and rv:
                S = sv[0].reshape(-1, sv[0].shape[-1])[P:P + T].float().cpu().numpy()  # [T, hidden]
                R = rv[0].reshape(-1, rv[0].shape[-1])[P:P + T].float().cpu().numpy()  # [T, hidden]
                den = np.linalg.norm(S, axis=1) * np.linalg.norm(R, axis=1)
                cos = np.where(den > 1e-12, (S * R).sum(axis=1) / den, 0.0)  # [T]
            else:
                cos = np.zeros(T, dtype=np.float64)
            router_branch_norms[int(layer_idx)] = [
                np.array([float(sh_rows[i]), float(ro_rows[i]), float(cos[i])], dtype=np.float32)
                for i in range(T)
            ]
        router_branch_norms = router_branch_norms or None

    # v2.1 magnitude: per-token ‖router_logits‖ [1, L] captured once → slice P:P+T.
    router_logit_norms: dict[int, list[F32]] | None = None
    if loaded.hook_state.router_logit_norm:
        router_logit_norms = {}
        for layer_idx, tensors in loaded.hook_state.router_logit_norm.items():
            if not tensors:
                continue
            ln_rows = tensors[0].reshape(-1)[P:P + T].float().cpu().numpy()  # [T]
            router_logit_norms[int(layer_idx)] = [np.float32(ln_rows[i]) for i in range(T)]
        router_logit_norms = router_logit_norms or None

    # Free GPU/structured outputs before returning
    del out
    loaded.clear_hook_state()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_token_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=P,
        positional_means=positional_means,
        gate_activations=gate_activations,
        v_proj_values=v_proj_values,
        queries=queries,
        attn_outputs=attn_outputs,
        router_dist=router_dist,
        router_branch_norms=router_branch_norms,
        router_logit_norms=router_logit_norms,
    )
