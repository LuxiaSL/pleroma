"""Orchestrates: prompt formatting → seed → generate → extract → save → cleanup.

Key design:
  - Simplified spec builder: 20 topics × 5 modes × 2 reps = 200 samples
  - Robust resume: checks existing files on disk, not just ID threshold
  - Incremental metadata saves: writes after every generation, not just at end
  - Progress logging with timing estimates
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from anamnesis.config import (
    ExperimentConfig,
    GenerationSpec,
    PROCESSING_MODES,
)
from anamnesis.extraction.model_loader import LoadedModel
from anamnesis.extraction.state_extractor import (
    ExtractionResult,
    RawGenerationData,
    extract_all_features,
)
from anamnesis.extraction.streaming_generate import (
    StreamingOutput,
    streaming_generate,
)

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]


def make_seed(topic_idx: int, mode_idx: int, rep_idx: int = 0, prompt_set: str = "8B") -> int:
    """Deterministic seed from generation coordinates.

    Uses "8B" as the prompt_set prefix to ensure seeds differ from the
    3B experiment even when topic/mode/rep match.
    """
    raw = f"{prompt_set}_{topic_idx}_{mode_idx}_{rep_idx}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def format_prompt(
    loaded: LoadedModel,
    system_prompt: str,
    user_prompt: str,
) -> tuple[torch.Tensor, int]:
    """Format prompt using chat template, return (input_ids, prompt_length)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = loaded.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(result, torch.Tensor):
        input_ids = result
    else:
        input_ids = result["input_ids"]
    prompt_length = input_ids.shape[1]
    device = next(loaded.model.parameters()).device
    input_ids = input_ids.to(device)
    return input_ids, prompt_length


def run_single_generation(
    loaded: LoadedModel,
    spec: GenerationSpec,
    config: ExperimentConfig,
    positional_means: F32 | None = None,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    save_raw: bool = False,
    raw_output_dir: Path | None = None,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Run a single generation and extract features.

    Returns:
        (extraction_result, metadata_dict)
    """
    t_start = time.time()

    # Set seed
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    np.random.seed(spec.seed % (2**32))

    # Format prompt
    input_ids, prompt_length = format_prompt(
        loaded, spec.system_prompt, spec.user_prompt,
    )

    # Clear hook state from previous generation
    loaded.clear_hook_state()
    loaded.enable_hooks()

    # Generate
    gen_config = config.generation
    with torch.no_grad():
        outputs = loaded.model.generate(
            input_ids,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            do_sample=gen_config.do_sample,
            eos_token_id=gen_config.eos_token_ids,
            output_hidden_states=gen_config.output_hidden_states,
            output_attentions=gen_config.output_attentions,
            output_logits=gen_config.output_logits,
            return_dict_in_generate=gen_config.return_dict_in_generate,
        )

    t_generate = time.time()

    loaded.flush_hooks_to_cpu()

    # Convert outputs to numpy for state_extractor
    raw_data = _convert_outputs_to_raw(
        outputs=outputs,
        prompt_length=prompt_length,
        hook_state=loaded.hook_state,
        sampled_layers=config.extraction.sampled_layers,
        positional_means=positional_means,
    )

    # Optionally save raw per-token tensors for GPU-free feature iteration
    if save_raw and raw_output_dir is not None:
        try:
            from anamnesis.extraction.raw_saver import save_raw_tensors
            save_raw_tensors(
                raw_data=raw_data,
                gen_id=spec.generation_id,
                output_dir=raw_output_dir,
                config=config.extraction,
                prompt_length=prompt_length,
            )
        except Exception as e:
            logger.warning(f"Failed to save raw tensors for gen {spec.generation_id}: {e}")

    # Extract features
    result = extract_all_features(
        raw_data,
        config.extraction,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )

    t_extract = time.time()

    # Decode generated text
    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Build metadata
    metadata = {
        "generation_id": spec.generation_id,
        "prompt_set": spec.prompt_set,
        "topic": spec.topic,
        "topic_idx": spec.topic_idx,
        "mode": spec.mode,
        "mode_idx": spec.mode_idx,
        "system_prompt": spec.system_prompt,
        "user_prompt": spec.user_prompt,
        "seed": spec.seed,
        "repetition": spec.repetition,
        "generated_text": generated_text,
        "num_generated_tokens": len(generated_ids),
        "prompt_length": prompt_length,
        "num_features": len(result.features),
        "tier_slices": result.tier_slices,
        "timing": {
            "generation_seconds": round(t_generate - t_start, 2),
            "extraction_seconds": round(t_extract - t_generate, 2),
            "total_seconds": round(t_extract - t_start, 2),
        },
    }

    # Cleanup GPU memory
    del outputs
    del input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result, metadata


def _router_fields_from_hooks(
    hook_state: Any, sampled_layers: list[int]
) -> tuple[dict[int, list[F32]] | None, dict[int, list[F32]] | None, dict[int, list[F32]] | None]:
    """Build (router_dist, router_branch_norms, router_logit_norms) for MoE models (arm A7, M6).

    Dense models never populate hook_state.router_dist → returns (None, None, None), so the xrt
    family stays inert everywhere else. Prefill (index 0) is skipped, matching get_generation_keys.
    router_dist[l] = T×[n_experts] dense softmax; router_branch_norms[l] = T×[‖shared‖, ‖routed‖, cos]
    (the v2.1 cos column derived from the transient branch vectors — 0.0 if vectors absent, e.g.
    v1-captured data); router_logit_norms[l] = T×scalar ‖router_logits‖ (v2.1 magnitude rung).
    """
    if not any(hook_state.router_dist.get(l, []) for l in sampled_layers):
        return None, None, None
    router_dist: dict[int, list[F32]] = {}
    for l_idx in sampled_layers:
        gen = hook_state.router_dist.get(l_idx, [])[1:]        # skip prefill
        if gen:
            router_dist[l_idx] = [
                s.reshape(-1, s.shape[-1])[-1].cpu().float().numpy().astype(np.float32)
                for s in gen
            ]

    # v2.1 magnitude: per-token ‖router_logits‖ (last position per step, prefill skipped).
    router_logit_norms: dict[int, list[F32]] | None = None
    if any(hook_state.router_logit_norm.get(l, []) for l in sampled_layers):
        router_logit_norms = {}
        for l_idx in sampled_layers:
            gen = hook_state.router_logit_norm.get(l_idx, [])[1:]
            if gen:
                router_logit_norms[l_idx] = [
                    np.float32(float(s.reshape(-1)[-1].cpu())) for s in gen
                ]
        router_logit_norms = router_logit_norms or None

    router_branch_norms: dict[int, list[F32]] | None = None
    if (any(hook_state.router_shared_norm.get(l, []) for l in sampled_layers)
            and any(hook_state.router_routed_norm.get(l, []) for l in sampled_layers)):
        have_vecs = (any(hook_state.router_shared_vec.get(l, []) for l in sampled_layers)
                     and any(hook_state.router_routed_vec.get(l, []) for l in sampled_layers))
        router_branch_norms = {}
        for l_idx in sampled_layers:
            sh = hook_state.router_shared_norm.get(l_idx, [])[1:]
            ro = hook_state.router_routed_norm.get(l_idx, [])[1:]
            if not (sh and ro):
                continue
            n = min(len(sh), len(ro))
            shv = hook_state.router_shared_vec.get(l_idx, [])[1:] if have_vecs else []
            rov = hook_state.router_routed_vec.get(l_idx, [])[1:] if have_vecs else []
            rows: list[F32] = []
            for t in range(n):
                cos = 0.0
                if have_vecs and t < len(shv) and t < len(rov):
                    sv = shv[t].reshape(-1, shv[t].shape[-1])[-1].cpu().float()
                    rv = rov[t].reshape(-1, rov[t].shape[-1])[-1].cpu().float()
                    denom = float(sv.norm()) * float(rv.norm())
                    cos = float((sv @ rv) / denom) if denom > 1e-12 else 0.0
                rows.append(np.array(
                    [float(sh[t].reshape(-1)[-1]), float(ro[t].reshape(-1)[-1]), cos],
                    dtype=np.float32))
            router_branch_norms[l_idx] = rows
    return (router_dist or None), router_branch_norms, router_logit_norms


def _convert_outputs_to_raw(
    outputs: Any,
    prompt_length: int,
    hook_state: Any,
    sampled_layers: list[int],
    positional_means: F32 | None,
) -> RawGenerationData:
    """Convert torch generation outputs to numpy arrays for state_extractor.

    Key indexing notes:
      - outputs.hidden_states[t][l]: t=generation step, l=layer (0=embedding, 1..N=transformer)
      - outputs.hidden_states[0] is the prefill step (shape [1, prompt_len, hidden_dim])
      - Generation steps start at index 1, each [1, 1, hidden_dim]
      - outputs.attentions[t][l]: [1, num_heads, 1, current_seq_len] at generation steps
      - outputs.logits[t]: [1, vocab_size] at each generation step (0-indexed, no prefill)
    """
    # Hidden states: for each generation step, stack all layers
    hidden_states_list: list[F32] = []
    for t in range(1, len(outputs.hidden_states)):
        layers = []
        for l_tensor in outputs.hidden_states[t]:
            layers.append(l_tensor[0, -1].cpu().float().numpy())
        hidden_states_list.append(np.stack(layers))

    # Attentions: for each generation step (skip prefill at index 0)
    attentions_list: list[F32] = []
    for t in range(1, len(outputs.attentions)):
        layers = []
        for l_tensor in outputs.attentions[t]:
            layers.append(l_tensor[0, :, -1, :].cpu().float().numpy())
        attentions_list.append(np.stack(layers))

    # Logits: skip prefill at index 0
    logits_list: list[F32] = []
    for t in range(1, len(outputs.logits)):
        logits_list.append(outputs.logits[t][0].cpu().float().numpy())

    # Chosen token IDs — skip first gen token to align with post-prefill lists
    chosen_ids = outputs.sequences[0, prompt_length + 1:].cpu().numpy().astype(np.float32)

    # Pre-RoPE keys from hooks
    pre_rope_keys: dict[int, list[F32]] = {}
    for l_idx in sampled_layers:
        gen_keys = hook_state.get_generation_keys(l_idx)
        if gen_keys:
            pre_rope_keys[l_idx] = [
                k[0].float().numpy().astype(np.float32)
                for k in gen_keys
            ]
            pre_rope_keys[l_idx] = [
                k[:, 0, :] if k.ndim == 3 else k
                for k in pre_rope_keys[l_idx]
            ]

    # Gate activations from hooks (SwiGLU gate_proj, pre-SiLU)
    gate_activations: dict[int, list[F32]] | None = None
    has_gates = any(hook_state.gate_activations.get(l, []) for l in sampled_layers)
    if has_gates:
        gate_activations = {}
        for l_idx in sampled_layers:
            gen_gates = hook_state.get_generation_gates(l_idx)
            if gen_gates:
                # gen_gates[t] shape: [1, 1, intermediate_size] during generation
                gate_activations[l_idx] = [
                    g[0, 0].float().numpy().astype(np.float32)
                    if g.ndim == 3 else g[0].float().numpy().astype(np.float32)
                    for g in gen_gates
                ]

    router_dist, router_branch_norms, router_logit_norms = _router_fields_from_hooks(
        hook_state, sampled_layers)

    return RawGenerationData(
        hidden_states=hidden_states_list,
        attentions=attentions_list,
        logits=logits_list,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=positional_means,
        gate_activations=gate_activations,
        router_dist=router_dist,
        router_branch_norms=router_branch_norms,
        router_logit_norms=router_logit_norms,
    )


def _convert_streaming_to_raw(
    stream_out: StreamingOutput,
    hook_state: Any,
    sampled_layers: list[int],
    positional_means: F32 | None,
) -> RawGenerationData:
    """Convert StreamingOutput to RawGenerationData for state_extractor.

    StreamingOutput already has the correct alignment:
      - hidden_states[i] = model state processing gen_token_i → produced gen_token_{i+1}
      - logits[i] = logits that produced gen_token_{i+1}
      - attentions[i] = attention when producing gen_token_{i+1}
    These are already numpy arrays, no GPU transfer needed.

    chosen_token_ids skips the first generated token (token_0 is produced by
    prefill, not by any state in our lists).
    """
    # chosen_ids: skip first generated token to match alignment
    # generated_token_ids = [token_0, token_1, ...], we want [token_1, token_2, ...]
    chosen_ids = np.array(
        stream_out.generated_token_ids[1:], dtype=np.float32
    ) if len(stream_out.generated_token_ids) > 1 else np.array([], dtype=np.float32)

    # Pre-RoPE keys from hooks — same extraction as before
    pre_rope_keys: dict[int, list[F32]] = {}
    for l_idx in sampled_layers:
        gen_keys = hook_state.get_generation_keys(l_idx)
        if gen_keys:
            pre_rope_keys[l_idx] = [
                k[0].float().numpy().astype(np.float32)
                for k in gen_keys
            ]
            pre_rope_keys[l_idx] = [
                k[:, 0, :] if k.ndim == 3 else k
                for k in pre_rope_keys[l_idx]
            ]

    router_dist, router_branch_norms, router_logit_norms = _router_fields_from_hooks(
        hook_state, sampled_layers)

    return RawGenerationData(
        hidden_states=stream_out.hidden_states,
        attentions=stream_out.attentions,
        logits=stream_out.logits,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=stream_out.prompt_length,
        positional_means=positional_means,
        router_dist=router_dist,
        router_branch_norms=router_branch_norms,
        router_logit_norms=router_logit_norms,
    )


def save_generation(
    gen_id: int,
    result: ExtractionResult,
    metadata: dict[str, Any],
    signatures_dir: Path,
) -> tuple[Path, Path]:
    """Save extraction result and metadata to disk."""
    npz_path = signatures_dir / f"gen_{gen_id:03d}.npz"
    json_path = signatures_dir / f"gen_{gen_id:03d}.json"

    save_dict: dict[str, Any] = {
        "features": result.features,
        "feature_names": np.array(result.feature_names),
    }
    if result.knnlm_baseline is not None:
        save_dict["knnlm_baseline"] = result.knnlm_baseline

    for tier_name, (start, end) in result.tier_slices.items():
        save_dict[f"features_{tier_name}"] = result.features[start:end]

    np.savez_compressed(npz_path, **save_dict)

    meta_copy = metadata.copy()
    meta_copy["tier_slices"] = {k: list(v) for k, v in metadata["tier_slices"].items()}
    with open(json_path, "w") as f:
        json.dump(meta_copy, f, indent=2, default=str)

    return npz_path, json_path


def build_generation_specs(
    config: ExperimentConfig,
    mode_dict: dict[str, str] | None = None,
    topics: list[str] | None = None,
    num_reps: int | None = None,
    prompt_set: str = "8B",
) -> list[GenerationSpec]:
    """Build generation specs. Defaults reproduce the 8B baseline.

    Parameters
    ----------
    config : ExperimentConfig
        Used only for `prompts_path`.
    mode_dict : dict[str, str] | None
        Maps mode name → system prompt. Defaults to `PROCESSING_MODES`
        (the canonical 5 format-controlled modes).
    topics : list[str] | None
        Topic strings. Defaults to `set_a + set_b` from `prompts_path`.
        Pass a subset for alternate experiments.
    num_reps : int | None
        Repetitions per (topic, mode) cell. Defaults to the `num_repetitions`
        field of `prompts_path`, or 2.
    prompt_set : str
        Tag stored on each spec; also used as the seed-namespace prefix so
        alternate experiments (e.g., "8B_r2") do not collide with baseline
        seeds.
    """
    with open(config.prompts_path) as f:
        prompts_file = json.load(f)

    if mode_dict is None:
        mode_dict = dict(PROCESSING_MODES)
    if topics is None:
        topics = [*prompts_file["topics"]["set_a"], *prompts_file["topics"]["set_b"]]
    if num_reps is None:
        num_reps = prompts_file.get("num_repetitions", 2)

    template = prompts_file.get("user_prompt_template", "Write about: {topic}")
    modes = list(mode_dict.keys())

    specs: list[GenerationSpec] = []
    gen_id = 0
    for rep in range(num_reps):
        for topic_idx, topic in enumerate(topics):
            for mode_idx, mode in enumerate(modes):
                specs.append(GenerationSpec(
                    generation_id=gen_id,
                    prompt_set=prompt_set,
                    topic=topic,
                    topic_idx=topic_idx,
                    mode=mode,
                    mode_idx=mode_idx,
                    system_prompt=mode_dict[mode],
                    user_prompt=template.format(topic=topic),
                    seed=make_seed(topic_idx, mode_idx, rep, prompt_set=prompt_set),
                    repetition=rep,
                ))
                gen_id += 1

    return specs


def find_completed_ids(signatures_dir: Path) -> set[int]:
    """Scan signatures directory for already-completed generations.

    A generation is complete if both .npz and .json exist.
    """
    completed: set[int] = set()
    if not signatures_dir.exists():
        return completed

    for npz_path in signatures_dir.glob("gen_*.npz"):
        gen_id_str = npz_path.stem.replace("gen_", "")
        try:
            gen_id = int(gen_id_str)
        except ValueError:
            continue
        json_path = signatures_dir / f"gen_{gen_id:03d}.json"
        if json_path.exists():
            completed.add(gen_id)

    return completed


def save_metadata_index(
    all_metadata: list[dict[str, Any]],
    failed_ids: list[int],
    config: ExperimentConfig,
) -> None:
    """Save master metadata index to disk."""
    metadata_path = config.metadata_path
    with open(metadata_path, "w") as f:
        json.dump({
            "total_generations": len(all_metadata),
            "failed_ids": failed_ids,
            "model": config.model.model_dump(),
            "generation_config": config.generation.model_dump(),
            "extraction_config": config.extraction.model_dump(),
            "generations": all_metadata,
        }, f, indent=2, default=str)


def run_experiment(
    loaded: LoadedModel,
    config: ExperimentConfig,
    positional_means: F32 | None = None,
    pca_components: F32 | None = None,
    pca_mean: F32 | None = None,
    specs: list[GenerationSpec] | None = None,
    save_raw: bool = False,
) -> list[dict[str, Any]]:
    """Run the experiment with robust resume support.

    Checks disk for already-completed generations and skips them.
    Saves metadata incrementally after every generation.

    Parameters
    ----------
    save_raw : bool
        If True, save raw per-token tensors alongside feature vectors.
        Enables GPU-free feature engineering iteration.
    """
    config.ensure_dirs()

    # Set up raw tensor output directory if requested
    raw_output_dir: Path | None = None
    if save_raw:
        raw_output_dir = config.signatures_dir.parent / "raw_tensors"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw tensor saving enabled → {raw_output_dir}")

    if specs is None:
        specs = build_generation_specs(config)

    # Check for already-completed generations on disk
    completed_ids = find_completed_ids(config.signatures_dir)
    if completed_ids:
        logger.info(f"Found {len(completed_ids)} completed generations on disk, resuming")

    specs_to_run = [s for s in specs if s.generation_id not in completed_ids]
    logger.info(
        f"Running {len(specs_to_run)} generations "
        f"({len(completed_ids)} already done, {len(specs)} total)"
    )

    if not specs_to_run:
        logger.info("All generations already complete!")
        return []

    # Load metadata from already-completed generations
    all_metadata: list[dict[str, Any]] = []
    for gen_id in sorted(completed_ids):
        json_path = config.signatures_dir / f"gen_{gen_id:03d}.json"
        try:
            with open(json_path) as f:
                all_metadata.append(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load metadata for gen {gen_id}: {e}")

    failed_ids: list[int] = []
    timing_history: list[float] = []

    for i, spec in enumerate(tqdm(specs_to_run, desc="Generating")):
        try:
            result, metadata = run_single_generation(
                loaded=loaded,
                spec=spec,
                config=config,
                positional_means=positional_means,
                pca_components=pca_components,
                pca_mean=pca_mean,
                save_raw=save_raw,
                raw_output_dir=raw_output_dir,
            )

            save_generation(
                gen_id=spec.generation_id,
                result=result,
                metadata=metadata,
                signatures_dir=config.signatures_dir,
            )

            all_metadata.append(metadata)
            elapsed = metadata["timing"]["total_seconds"]
            timing_history.append(elapsed)

            # Progress logging every 10 generations
            if (i + 1) % 10 == 0 or i == 0:
                avg_time = np.mean(timing_history[-20:])  # rolling average
                remaining = len(specs_to_run) - (i + 1)
                eta_minutes = (avg_time * remaining) / 60
                logger.info(
                    f"Gen {spec.generation_id} [{i+1}/{len(specs_to_run)}]: "
                    f"{spec.mode}/{spec.topic[:25]} rep={spec.repetition} "
                    f"— {metadata['num_generated_tokens']} tokens, "
                    f"{elapsed:.1f}s, {metadata['num_features']} features "
                    f"(avg {avg_time:.1f}s/gen, ETA {eta_minutes:.0f}min)"
                )

            # Incremental metadata save every 25 generations
            if (i + 1) % 25 == 0:
                save_metadata_index(all_metadata, failed_ids, config)
                logger.info(f"Metadata checkpoint saved ({len(all_metadata)} total)")

        except Exception as e:
            logger.error(f"Generation {spec.generation_id} failed: {e}", exc_info=True)
            failed_ids.append(spec.generation_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

    # Final metadata save
    save_metadata_index(all_metadata, failed_ids, config)

    logger.info(
        f"Experiment complete: {len(all_metadata)} succeeded, "
        f"{len(failed_ids)} failed. Metadata saved to {config.metadata_path}"
    )

    return all_metadata
