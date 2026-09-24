"""Wrapper-aware lookups on a loaded HF causal LM: decoder layers, the decoder
stack, hidden size.

Torch-free: these only walk attributes, so they work
on any object shaped like a Llama/Qwen/OLMo or a multimodal wrapper around one.
"""

from __future__ import annotations

from typing import Any


def decoder_layers(model: Any) -> Any:
    """Decoder layer list, wrapper-aware (mirrors anamnesis' ``model_loader.decoder_layers``)."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner.layers
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__} — extend "
        "decoder_layers() for this architecture"
    )


def decoder_stack(model: Any) -> Any:
    """The module that returns ``last_hidden_state`` (i.e. everything but lm_head).

    For Llama/Qwen/OLMo ``model.model``; for a multimodal wrapper the text decoder
    nests one level deeper. Resolved rather than assumed so that calling the stack
    directly (to skip the full-vocab lm_head) does not quietly become architecture
    -specific.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner
    raise AttributeError(
        f"cannot locate the decoder stack on {type(model).__name__} — extend "
        "decoder_stack() for this architecture"
    )


def model_hidden_dim(model: Any) -> int:
    """hidden_size, wrapper-aware (Gemma-class configs nest it under text_config)."""
    cfg = model.config
    dim = getattr(cfg, "hidden_size", None) or getattr(
        getattr(cfg, "text_config", None), "hidden_size", None
    )
    if not dim:
        raise AttributeError(f"cannot read hidden_size off {type(cfg).__name__}")
    return int(dim)


# ── the 8B -> N-layer depth rescale (the 70B preset's layer rule) ────────────

#: The 8B preset's depth, and its final layer — the top-of-stack ANCHOR.
_8B_NUM_LAYERS = 32


def rescale_8b_layer(layer_8b: int, num_layers: int) -> int:
    """Map an 8B layer index to the same fractional depth in ``num_layers``.

    The rule every 70B layer list in this repo is built with: interior layers
    scale by fractional depth, ``round(layer * num_layers / 32)`` (Python's
    round-half-to-even), clamped into range; the 8B FINAL layer (31) is an
    anchor and maps to ``num_layers - 1`` — scaling it would give 77.5 -> 78 of
    80 and quietly stop sampling the top of the stack.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if layer_8b == _8B_NUM_LAYERS - 1:
        return num_layers - 1
    scaled = round(layer_8b * num_layers / _8B_NUM_LAYERS)
    return max(0, min(num_layers - 1, scaled))
