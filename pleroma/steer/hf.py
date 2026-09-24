"""HF backend: one :class:`~pleroma.steer.spec.Injection` -> residual-write
pre-hooks on a transformers decoder.

Built on anamnesis' ``attach_residual_write`` / ``ResidualWriteSpec`` — the
exact primitive the loom's WEAR, and every banked HF result, used — with the
settings every copy re-asserted by hand: one spec per site at
``layer_idx = site`` (a forward PRE-hook on ``layers[site]``: the map/lever
convention), ``alpha`` passed through (the primitive computes
``(alpha * row_f32).to(hidden dtype)``), ``normalize=False``, ``end_pos=None``,
and ``start_pos`` from the span: ``0`` (UNIFORM) or ``prompt_len``
(CONTINUATION). Gating reads ``cache_position``, so one spec is valid for
prefill, incremental decode steps and single-forward replay alike.

Attach is all-or-none: if any site fails, the handles already registered are
removed before the error propagates — never a half-worn model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from anamnesis.extraction.model_loader import (
    ResidualWriteHandle,
    ResidualWriteSpec,
    attach_residual_write,
    decoder_layers,
)

from pleroma.config.profile import InjectionSpan
from pleroma.steer.spec import (
    Injection,
    SpanError,
    TimeProfile,
    VectorShapeError,
    check_prompt_len,
    check_sites_in_range,
)

__all__ = ["attach", "detach", "hidden_size", "start_pos_for"]


def start_pos_for(span: InjectionSpan, prompt_len: int | None) -> int:
    """The absolute first position a span writes (anamnesis ``start_pos``)."""
    if span is InjectionSpan.UNIFORM:
        return 0
    if span is InjectionSpan.CONTINUATION:
        return check_prompt_len(prompt_len)
    raise SpanError(f"unhandled injection span {span!r}")


def hidden_size(model: Any) -> int:
    """``config.hidden_size`` (or ``config.text_config.hidden_size`` for
    multimodal wrappers), as anamnesis resolves it."""
    cfg = model.config
    hs = getattr(cfg, "hidden_size", None) or getattr(
        getattr(cfg, "text_config", None), "hidden_size", None)
    if hs is None:
        raise VectorShapeError("model.config has no hidden_size (nor text_config.hidden_size)")
    return int(hs)


def attach(model: Any, injection: Injection, *,
           prompt_len: int | None = None) -> list[ResidualWriteHandle]:
    """Register ``injection`` on ``model``; returns one handle per site
    (``.remove()`` each, or :func:`detach`, to unwear).

    ``prompt_len`` is required for CONTINUATION and ignored for UNIFORM.
    Raises :class:`~pleroma.steer.spec.SiteError` (site out of range),
    :class:`~pleroma.steer.spec.VectorShapeError` (hidden size mismatch) or
    :class:`~pleroma.steer.spec.SpanError` (missing prompt_len, or a
    non-constant time profile, which this primitive cannot express) BEFORE
    anything is registered.
    """
    if not isinstance(injection, Injection):
        raise TypeError(f"attach() takes an Injection, got {type(injection).__name__}")
    if injection.time_profile is not TimeProfile.CONSTANT:
        raise SpanError(f"time profile {injection.time_profile.value!r} is vLLM-only: the HF "
                        "write primitive (anamnesis ResidualWriteSpec) gates on a position "
                        "window, not a per-position weight. Use pleroma.steer.vllm, or "
                        "time_profile='constant'.")
    start = start_pos_for(injection.span, prompt_len)
    check_sites_in_range(injection.sites, len(decoder_layers(model)))
    h = hidden_size(model)
    if injection.hidden != h:
        raise VectorShapeError(f"vectors have hidden size {injection.hidden}, the model has "
                               f"{h} — wrong map/lever for this model?")
    handles: list[ResidualWriteHandle] = []
    try:
        for row, site in zip(injection.vectors, injection.sites, strict=True):
            handles.append(attach_residual_write(
                model,
                ResidualWriteSpec(
                    layer_idx=int(site),
                    vector=torch.from_numpy(np.ascontiguousarray(row, dtype=np.float32)),
                    alpha=float(injection.alpha),
                    start_pos=start, end_pos=None, normalize=False,
                ),
            ))
    except BaseException:
        detach(handles)
        raise
    return handles


def detach(handles: Iterable[Any]) -> None:
    """Remove every handle (idempotent per handle; torch ignores a second
    remove)."""
    for h in handles:
        h.remove()
