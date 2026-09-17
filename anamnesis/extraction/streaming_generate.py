"""Streaming generation with efficient internal state collection.

Replaces HuggingFace's model.generate() for our use case. HF's generate()
with output_hidden_states=True creates ~17,000 individual tensor objects for
a 512-token generation on a 32-layer model (steps × layers), adding ~10×
overhead vs normal generation. This module runs the same computation but
collects states efficiently via per-step GPU stacking and single transfers.

Two interfaces:
  streaming_generate() — returns pre-stacked numpy arrays for experiment use
  streaming_calibrate() — accumulates positional means on-the-fly for calibration

Both use the same core autoregressive loop. The key difference from HF generate:
  - One model() call per step (same as generate internally)
  - States are stacked on GPU per step → single .cpu() transfer per step
  - No tuple-of-tuples accumulation — just numpy list appends
  - k_proj hooks fire normally (registered on modules, not on generate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


@dataclass
class StreamingOutput:
    """Efficient generation output with pre-stacked numpy arrays.

    All lists are indexed by generation step, starting AFTER the prefill step.
    Step i corresponds to the model processing gen_token_i and producing
    gen_token_{i+1}. This matches the alignment contract of _convert_outputs_to_raw.

    For a generation producing N tokens:
      - generated_token_ids has N entries: [token_0, token_1, ..., token_{N-1}]
      - hidden_states has N-1 entries (prefill produces token_0, excluded)
      - logits has N-1 entries
      - attentions has N-1 entries (if collected)
    """

    sequences: torch.Tensor                    # [1, prompt_len + N]
    hidden_states: list[F32]                   # (N-1) × [n_layers+1, hidden_dim]
    attentions: list[F32]                      # (N-1) × [n_layers, n_heads, seq_len_at_step]
    logits: list[F32]                          # (N-1) × [vocab_size]
    generated_token_ids: list[int]             # N tokens (all generated, including first)
    prompt_length: int
    prefill_hidden_states: F32 | None = None   # [n_layers+1, prompt_len, hidden_dim] (calibration only)


def _sample_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """Sample a single token from logits with temperature and nucleus sampling.

    Args:
        logits: [vocab_size] raw logits (not softmaxed)
        temperature: Sampling temperature (>0)
        top_p: Nucleus sampling threshold

    Returns:
        Sampled token ID
    """
    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Zero out tokens beyond the nucleus
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order and sample
    logits_filtered = torch.full_like(scaled, float("-inf"))
    logits_filtered.scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def streaming_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    eos_token_ids: list[int] | None = None,
    output_hidden_states: bool = True,
    output_attentions: bool = False,
    collect_prefill_hidden_states: bool = False,
) -> StreamingOutput:
    """Generate tokens with efficient streaming state collection.

    Performs the same autoregressive generation as HF's model.generate(), but
    collects internal states (hidden_states, attentions, logits) efficiently
    by doing per-step GPU stacking and immediate CPU transfer.

    The k_proj hooks registered on the model fire normally during each forward
    pass — no special handling needed.

    Args:
        model: HuggingFace causal LM (already on GPU, eval mode)
        input_ids: [1, prompt_len] input token IDs on the model's device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_ids: Token IDs that signal end of generation
        output_hidden_states: Collect per-step hidden states
        output_attentions: Collect per-step attention weights
        collect_prefill_hidden_states: Also collect prefill hidden states
            (needed for calibration positional means, not for experiment)

    Returns:
        StreamingOutput with pre-stacked numpy arrays
    """
    device = input_ids.device
    eos_set = set(eos_token_ids or [])
    prompt_length = input_ids.shape[1]

    # GPU accumulators — kept on device during generation to avoid
    # per-step synchronous CPU transfers. Batch-converted after the loop.
    generated_ids: list[int] = []
    hidden_gpu: list[torch.Tensor] = []
    attn_gpu: list[torch.Tensor] = []
    logit_gpu: list[torch.Tensor] = []
    prefill_hs: F32 | None = None

    past_key_values = None
    current_input = input_ids

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

            step_logits = outputs.logits[0, -1]  # [vocab_size]

            next_token_id = _sample_top_p(step_logits, temperature, top_p)
            generated_ids.append(next_token_id)

            # ── Collect states on GPU ──────────────────────────────

            if step_idx == 0:
                if collect_prefill_hidden_states and output_hidden_states:
                    prefill_stacked = torch.stack([
                        outputs.hidden_states[l][0]
                        for l in range(len(outputs.hidden_states))
                    ])
                    prefill_hs = prefill_stacked.cpu().float().numpy()
                    del prefill_stacked

            else:
                if output_hidden_states and outputs.hidden_states is not None:
                    hs_stacked = torch.stack([
                        outputs.hidden_states[l][0, -1]
                        for l in range(len(outputs.hidden_states))
                    ])  # [n_layers+1, hidden_dim]
                    hidden_gpu.append(hs_stacked)

                if output_attentions and outputs.attentions is not None:
                    attn_stacked = torch.stack([
                        outputs.attentions[l][0, :, -1, :]
                        for l in range(len(outputs.attentions))
                    ])  # [n_layers, n_heads, current_seq_len]
                    attn_gpu.append(attn_stacked)

                logit_gpu.append(step_logits.clone())

            # ── Advance state ───────────────────────────────────────

            past_key_values = outputs.past_key_values
            current_input = torch.tensor(
                [[next_token_id]], device=device, dtype=input_ids.dtype
            )
            del outputs

            if next_token_id in eos_set:
                break

    # ── Batch GPU → CPU transfer ───────────────────────────────────
    hidden_list: list[F32] = []
    if hidden_gpu:
        hs_batch = torch.stack(hidden_gpu).cpu().float().numpy()
        hidden_list = [hs_batch[i] for i in range(hs_batch.shape[0])]
        del hidden_gpu, hs_batch

    logit_list: list[F32] = []
    if logit_gpu:
        lg_batch = torch.stack(logit_gpu).cpu().float().numpy()
        logit_list = [lg_batch[i] for i in range(lg_batch.shape[0])]
        del logit_gpu, lg_batch

    # Attention has variable seq_len per step — can't stack, transfer individually
    attn_list: list[F32] = [a.cpu().float().numpy() for a in attn_gpu]
    del attn_gpu

    if generated_ids:
        gen_tensor = torch.tensor(
            generated_ids, device=device, dtype=input_ids.dtype
        ).unsqueeze(0)
        sequences = torch.cat([input_ids, gen_tensor], dim=1)
    else:
        sequences = input_ids

    return StreamingOutput(
        sequences=sequences,
        hidden_states=hidden_list,
        attentions=attn_list,
        logits=logit_list,
        generated_token_ids=generated_ids,
        prompt_length=prompt_length,
        prefill_hidden_states=prefill_hs,
    )


def streaming_calibrate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    eos_token_ids: list[int] | None = None,
    pos_sums: np.ndarray,
    pos_counts: np.ndarray,
    pca_samples: list[np.ndarray],
    pca_layers: list[int],
    pca_sample_positions: list[int] | None = None,
) -> int:
    """Generate tokens and accumulate calibration data on-the-fly.

    Instead of storing all hidden states and post-processing, this accumulates
    positional means directly during generation. Much more memory-efficient
    and avoids the massive tensor storage overhead.

    Args:
        model: HuggingFace causal LM
        input_ids: [1, prompt_len] input token IDs
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token_ids: EOS token IDs
        pos_sums: [n_layers+1, max_positions, hidden_dim] float64 accumulator (MUTATED)
        pos_counts: [n_layers+1, max_positions] int64 count (MUTATED)
        pca_samples: List to append PCA sample arrays to (MUTATED)
        pca_layers: Layer indices for PCA sampling
        pca_sample_positions: Which generation steps to sample for PCA.
            If None, will be computed from num_gen_steps after generation.

    Returns:
        Number of tokens generated
    """
    device = input_ids.device
    eos_set = set(eos_token_ids or [])
    prompt_length = input_ids.shape[1]
    max_positions = pos_sums.shape[1]
    n_layers_plus_embed = pos_sums.shape[0]

    past_key_values = None
    current_input = input_ids
    num_generated = 0
    generated_ids: list[int] = []

    # Pre-compute PCA sample steps (estimates based on max_new_tokens)
    pca_step_indices = compute_pca_step_indices(max_new_tokens) if pca_layers else set()

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
            )

            # Sample next token
            step_logits = outputs.logits[0, -1]
            next_token_id = _sample_top_p(step_logits, temperature, top_p)
            generated_ids.append(next_token_id)
            num_generated += 1

            if step_idx == 0:
                # PREFILL: accumulate hidden states for all prompt positions
                n_layers = min(len(outputs.hidden_states), n_layers_plus_embed)
                # Stack all layers: [n_layers, prompt_len, hidden_dim]
                prefill_stacked = torch.stack([
                    outputs.hidden_states[l][0]
                    for l in range(n_layers)
                ]).cpu().float().numpy()

                n_pos = min(prefill_stacked.shape[1], max_positions)
                pos_sums[:n_layers, :n_pos] += prefill_stacked[:, :n_pos].astype(np.float64)
                pos_counts[:n_layers, :n_pos] += 1
                del prefill_stacked

            else:
                # GENERATION STEP: accumulate hidden state at this position
                abs_pos = prompt_length + step_idx - 1
                if abs_pos < max_positions:
                    n_layers = min(len(outputs.hidden_states), n_layers_plus_embed)
                    # Stack layers: [n_layers, hidden_dim]
                    hs_stacked = torch.stack([
                        outputs.hidden_states[l][0, -1]
                        for l in range(n_layers)
                    ]).cpu().float().numpy()

                    pos_sums[:n_layers, abs_pos] += hs_stacked.astype(np.float64)
                    pos_counts[:n_layers, abs_pos] += 1
                    del hs_stacked

                # Collect PCA samples at designated steps
                if step_idx in pca_step_indices:
                    for l_idx in pca_layers:
                        if l_idx + 1 < len(outputs.hidden_states):
                            h = outputs.hidden_states[l_idx + 1][0, -1].cpu().float().numpy()
                            pca_samples.append(h)

            # Advance state
            past_key_values = outputs.past_key_values
            current_input = torch.tensor(
                [[next_token_id]], device=device, dtype=input_ids.dtype
            )
            del outputs

            # Check EOS
            if next_token_id in eos_set:
                break

    return num_generated


def compute_pca_step_indices(max_new_tokens: int) -> set[int]:
    """Pre-compute which generation steps to sample for PCA.

    Uses the same heuristic as the original calibration: beginning, middle, end.
    Since we don't know the actual generation length upfront, we estimate
    based on max_new_tokens (most generations hit 400-512 tokens).
    """
    estimated_len = max_new_tokens
    positions = {
        1,
        max(1, estimated_len // 4),
        max(1, estimated_len // 2),
        max(1, 3 * estimated_len // 4),
        estimated_len,
    }
    return positions
