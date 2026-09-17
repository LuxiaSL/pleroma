"""Build a calibration dir (positional_means.npz + pca_model.pkl) for a preset.

This is the NEW-MODEL entry gate: every extraction stage refuses to run
without a calibration dir, and the shipped one is for Llama-3.2-3B only.
Adapted from anamnesis phase_0 ``scripts/run_calibration.py`` — the exact
procedure that produced the shipped ``data/calibration/3b`` — but driven by
the vendored ``anamnesis.config.MODEL_PRESETS`` instead of the phase-0
experiment config, so pointing it at a new preset is the whole job.

What it computes, over ~50 diverse prompts' generations:
  - positional_means: per-(layer, absolute position) mean hidden state
    [num_layers+1, max_pos, hidden_dim] — subtracted during extraction to
    remove positional artifacts (positions seen <6 times stay zero and are
    skipped by the extractor's bounds check).
  - pca_model.pkl: PCA (components + mean) fit on hidden states from the
    preset's pca_layers — tier-3 features. Legacy pooled dict format, the
    same one ``load_calibration_strict`` accepts.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.build_calibration \\
        --preset <your-preset> --out-dir data/calibration/<your-model> \\
        [--model-path <hf-id-or-local-dir>] [--prompts my_prompts.json]

Then freeze the feature space with run_replay_b0 --freeze-names-to.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("build_calibration")

F32 = NDArray[np.float32]

#: Diverse calibration prompts (verbatim from phase_0 run_calibration.py —
#: varied content to wash out content-specific signal). Override with --prompts.
CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Explain how photosynthesis works in plants.",
    "What are the main causes of the French Revolution?",
    "Describe the process of making traditional Japanese ramen.",
    "How do electric vehicles compare to gasoline cars?",
    "What is the significance of the Rosetta Stone?",
    "Explain the concept of supply and demand in economics.",
    "How does the human immune system fight infections?",
    "Describe the architecture of Gothic cathedrals.",
    "What are the principles of object-oriented programming?",
    "How do tides work and what causes them?",
    "Explain the theory of plate tectonics.",
    "What makes a good leader?",
    "How do birds navigate during migration?",
    "Describe the water cycle and its importance.",
    "What is quantum entanglement?",
    "How do vaccines work?",
    "Explain the causes and effects of inflation.",
    "What are the different types of clouds?",
    "How does a combustion engine work?",
    "Describe the life cycle of a star.",
    "What is machine learning and how does it differ from traditional programming?",
    "How do earthquakes happen?",
    "Explain the basics of music theory.",
    "What are renewable energy sources?",
    "How does the stock market work?",
    "Describe the process of fermentation.",
    "What are the effects of sleep deprivation?",
    "How do submarines work?",
    "Explain the concept of natural selection.",
    "What is the significance of pi in mathematics?",
    "How do 3D printers work?",
    "Describe the history of the internet.",
    "What causes aurora borealis?",
    "How do computers store and retrieve data?",
    "Explain the process of osmosis.",
    "What are the major types of rocks?",
    "How do airplanes fly?",
    "Describe the structure of DNA.",
    "What is cryptocurrency and how does blockchain work?",
    "How do telescopes work?",
    "Explain the greenhouse effect.",
    "What are the stages of grief?",
    "How does sonar work?",
    "Describe the Silk Road and its importance.",
    "What is dark matter?",
    "How do coral reefs form?",
    "Explain the basics of game theory.",
    "What are the layers of the atmosphere?",
    "How does a nuclear reactor work?",
    "Describe the process of cheese making.",
)

#: A position must be visited this often before its mean is trusted (phase_0 rule).
MIN_COUNT = 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", required=True, help="anamnesis MODEL_PRESETS key")
    parser.add_argument("--model-path", default=None,
                        help="HF id or local dir (default: the preset's model_id)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None,
                        help="JSON list of prompt strings (default: built-in 50)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--max-positions", type=int, default=0,
                        help="positions to calibrate (0 = max-new-tokens + 200)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="tokenize prompts bare (base models without a chat template)")
    args = parser.parse_args()

    # Deferred so --help works on any machine, with or without the anamnesis env.
    import torch

    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.model_loader import load_model

    # cuDNN SDPA builds a host-side graph per unseen shape — pathological for
    # token-by-token decode (2026-09-16 forensics). Never remove this.
    torch.backends.cuda.enable_cudnn_sdp(False)

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r (have: %s)", args.preset, sorted(MODEL_PRESETS))
        return 2
    preset = MODEL_PRESETS[args.preset]

    if args.prompts is not None:
        prompts = [str(p) for p in json.loads(args.prompts.read_text())]
        if not prompts:
            logger.error("--prompts %s is empty", args.prompts)
            return 2
    else:
        prompts = list(CALIBRATION_PROMPTS)
    logger.info("calibrating %r over %d prompts, %d new tokens each",
                args.preset, len(prompts), args.max_new_tokens)

    model_path = args.model_path or preset.model_id
    model_config = ModelConfig(
        model_id=model_path,
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    loaded = load_model(model_config, sampled_layers=[])
    loaded.disable_hooks()
    device = next(loaded.model.parameters()).device

    n_layers = preset.num_layers + 1  # + embedding layer
    max_pos = args.max_positions or (args.max_new_tokens + 200)
    pos_sums = np.zeros((n_layers, max_pos, preset.hidden_dim), dtype=np.float64)
    pos_counts = np.zeros((n_layers, max_pos), dtype=np.int64)
    pca_samples: list[F32] = []

    for prompt_idx, prompt_text in enumerate(prompts):
        if args.no_chat_template:
            enc = loaded.tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"]
        else:
            result = loaded.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, return_tensors="pt",
            )
            input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        prompt_length = int(input_ids.shape[1])
        input_ids = input_ids.to(device)
        torch.manual_seed(prompt_idx)

        with torch.no_grad():
            outputs = loaded.model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=preset.temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=preset.eos_token_ids,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Prefill: hidden_states[0] is a tuple of n_layers tensors [1, plen, hid].
        prefill = outputs.hidden_states[0]
        for layer in range(n_layers):
            h = prefill[layer][0].cpu().float().numpy()
            upto = min(h.shape[0], max_pos)
            pos_sums[layer, :upto] += h[:upto].astype(np.float64)
            pos_counts[layer, :upto] += 1

        # Decode steps: hidden_states[t] holds the single new position.
        for t in range(1, len(outputs.hidden_states)):
            abs_pos = prompt_length + t - 1
            if abs_pos >= max_pos:
                break
            step = outputs.hidden_states[t]
            for layer in range(min(len(step), n_layers)):
                h = step[layer][0, -1].cpu().float().numpy()
                pos_sums[layer, abs_pos] += h.astype(np.float64)
                pos_counts[layer, abs_pos] += 1

        # PCA samples: first/middle/last generated token, preset pca_layers.
        n_steps = len(outputs.hidden_states) - 1
        if n_steps > 0:
            for t in {1, max(1, n_steps // 2), n_steps}:
                for l_idx in preset.pca_layers:
                    if l_idx + 1 < len(outputs.hidden_states[t]):
                        pca_samples.append(
                            outputs.hidden_states[t][l_idx + 1][0, -1].cpu().float().numpy()
                        )

        del outputs, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if (prompt_idx + 1) % 10 == 0:
            logger.info("  %d/%d prompts", prompt_idx + 1, len(prompts))

    mask = pos_counts >= MIN_COUNT
    positional_means = np.zeros_like(pos_sums, dtype=np.float32)
    np.divide(pos_sums, pos_counts[:, :, None], where=mask[:, :, None],
              out=positional_means, casting="unsafe")
    covered = int(mask.any(axis=0).sum())
    logger.info("positional_means %s — %d/%d positions calibrated (count >= %d)",
                positional_means.shape, covered, max_pos, MIN_COUNT)
    if covered < 32:
        logger.error("almost nothing calibrated — generations too short? aborting")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "positional_means.npz",
                        positional_means=positional_means, pos_counts=pos_counts)
    logger.info("-> %s", args.out_dir / "positional_means.npz")

    if not pca_samples:
        logger.error("no PCA samples collected — tier-3 features would be empty; aborting")
        return 1
    from sklearn.decomposition import PCA

    pca_matrix = np.stack(pca_samples).astype(np.float64)
    n_components = min(args.pca_components, *pca_matrix.shape)
    pca = PCA(n_components=n_components)
    pca.fit(pca_matrix)
    with open(args.out_dir / "pca_model.pkl", "wb") as fh:
        pickle.dump({
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }, fh)
    logger.info("-> %s (components %s, explained %.3f)",
                args.out_dir / "pca_model.pkl", pca.components_.shape,
                float(pca.explained_variance_ratio_.sum()))
    logger.info("DONE — next: gen_forks for a corpus, then run_replay_b0 "
                "--freeze-names-to to freeze this model's feature space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
