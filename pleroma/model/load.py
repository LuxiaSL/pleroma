"""Load an HF causal LM frozen for read-only forwards; dtype table; pad id.

**Imports torch at module scope** — import it lazily
(inside ``main()``) from entry points that must stay ``--help``-able without
torch.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("pleroma.model")

#: torch dtypes selectable from the CLI.
DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_frozen_model(model_path: str, dtype: torch.dtype, device: str) -> Any:
    """Load the model frozen and in eval mode, with SDPA attention.

    Frozen means ``requires_grad_(False)`` on every parameter AND ``eval()``: the
    callers only read activations/logits, and a stray dropout/train-mode
    difference between arms would show up as a fake effect in exactly the
    contrast being measured. ``config.use_cache`` is switched off (full-sequence
    forwards only).
    """
    from transformers import AutoModelForCausalLM

    logger.info("loading frozen teacher %s (%s, sdpa)", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable:
        raise RuntimeError(f"teacher is not frozen: {n_trainable} trainable parameters")
    return model


def resolve_pad_token_id(model: Any, model_path: str) -> int:
    """A pad id for right-padding. Masked out everywhere, so any valid id will do.

    Config first (no tokenizer load), tokenizer second, and 0 only as a last resort
    with a warning — never silently, because a pad id outside the vocab is an
    index error deep inside the embedding.
    """
    cfg = model.config
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(cfg, attr, None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            return int(value)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        value = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if value is not None:
            return int(value)
    except Exception as exc:  # noqa: BLE001 — the fallback below is still valid
        logger.warning("could not read a pad id from the tokenizer (%s)", exc)
    logger.warning("no pad/eos token id found; padding with 0 (masked out anyway)")
    return 0
