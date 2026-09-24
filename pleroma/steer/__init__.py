"""ONE steering implementation for both backends (HF and vLLM).

- :mod:`pleroma.steer.spec` — :class:`Injection` (sites, per-site vectors,
  alpha, span, time profile), the error types, receipts.
- :mod:`pleroma.steer.hf` — ``attach(model, injection, prompt_len=...)``:
  anamnesis residual-write pre-hooks on the INPUT of ``layers[site]`` (the
  loom's WEAR path, bit-identical to it).
- :mod:`pleroma.steer.vllm` — ``VllmInjectionState`` + ``install_pre_hooks``:
  a pre-hook on vLLM's ``(positions, hidden)`` call; no vllm import.

Span is UNIFORM by default (`docs/FINDINGS.md §4`); CONTINUATION only by name.
The backends are imported explicitly (``from pleroma.steer import hf``) so
this package does not pull torch/anamnesis in on import.
"""

from pleroma.config.profile import InjectionSpan
from pleroma.steer.spec import (
    TIME_K,
    Injection,
    SiteError,
    SpanError,
    SteerDTypeError,
    SteerError,
    TimeProfile,
    VectorShapeError,
    add_injection_span_arg,
    span_receipt,
)

__all__ = [
    "TIME_K", "Injection", "InjectionSpan", "SiteError", "SpanError", "SteerDTypeError",
    "SteerError", "TimeProfile", "VectorShapeError", "add_injection_span_arg", "span_receipt",
]
