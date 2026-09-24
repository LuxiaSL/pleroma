"""vLLM backend: a forward pre-hook on vLLM decoder layers.

vLLM calls a decoder layer as ``layer(positions, hidden_states, residual, ...)``
with the batch FLATTENED: ``positions: [num_tokens]`` (absolute positions),
``hidden_states: [num_tokens, H]``. A pre-hook registered with
``with_kwargs=True`` on ``layers[site]`` can therefore add a vector to arg 1
under a mask computed from arg 0 — one rule, valid for prefill AND every
decode step, no per-step toggling. It requires every sequence in flight to
share ``prompt_len`` when the span is CONTINUATION (the lanes submit one prompt
per request and batch with ``SamplingParams(n=...)``); UNIFORM does not read
``prompt_len`` at all.

Design (one hook covers every delivery the screens used): span
(uniform / continuation) x time profile (constant / step / ramp) and an
explicit dtype guard — the written vector is cast to the hidden states'
dtype/device, and the result is forced back to it (a float32 residual returned
into a bfloat16 model kills the next fused rms-norm).

Nothing here imports vllm: the hook is plain torch, testable on CPU tensors.

★ The hook is dead under CUDA graphs: construct the engine with
``enforce_eager=True`` and ``enable_prefix_caching=False`` and run the A/B/C
self-check (no hook == inert hook == zero vector; real vector != no hook)
before trusting a single steered sample. ``state.fired`` counts real writes
for that check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pleroma.config.profile import InjectionSpan
from pleroma.steer.spec import (
    TIME_K,
    Injection,
    SiteError,
    SteerDTypeError,
    SteerError,
    TimeProfile,
    VectorShapeError,
    as_sites,
    as_span,
    as_time_profile,
    check_prompt_len,
    check_sites_in_range,
    span_receipt,
)

__all__ = ["SitePreHook", "VllmInjectionState", "inject", "install_pre_hooks",
           "site_vectors", "time_weights"]


@dataclass
class VllmInjectionState:
    """The mutable state every site's pre-hook reads at call time.

    ``span`` / ``time_profile`` / ``time_k`` are fixed per run (they are what a
    receipt records); ``vecs`` (site -> 1-D tensor, already dosed) and
    ``prompt_len`` change per request via :meth:`wear`. An empty ``vecs`` is
    inert: the alpha-0 arm traverses the same code path and writes nothing.
    """

    span: InjectionSpan = InjectionSpan.UNIFORM
    time_profile: TimeProfile = TimeProfile.CONSTANT
    time_k: float = TIME_K
    vecs: dict[int, torch.Tensor] = field(default_factory=dict)
    prompt_len: int = 0
    fired: int = 0

    def __post_init__(self) -> None:
        self.span = as_span(self.span)
        self.time_profile = as_time_profile(self.time_profile)
        self.time_k = float(self.time_k)
        if not (np.isfinite(self.time_k) and self.time_k > 0):
            raise SteerError(f"time_k must be a positive finite token count, got {self.time_k}")
        self.wear(self.vecs, self.prompt_len)

    def wear(self, vecs: Mapping[int, torch.Tensor], prompt_len: int) -> None:
        """Arm the hooks for the next request: ``vecs`` (``{}`` = inert) and
        the request's prompt length (read only under CONTINUATION or a
        non-constant profile's time coordinate)."""
        clean: dict[int, torch.Tensor] = {}
        for site, v in dict(vecs).items():
            if not isinstance(v, torch.Tensor):
                raise VectorShapeError(f"vector for site {site} is a {type(v).__name__}, "
                                       "expected a torch.Tensor (see site_vectors)")
            if v.dim() != 1:
                raise VectorShapeError(f"vector for site {site} has shape {tuple(v.shape)}; "
                                       "expected 1-D [hidden]")
            if not v.is_floating_point():
                raise SteerDTypeError(f"vector for site {site} has dtype {v.dtype}; "
                                      "expected a floating dtype")
            clean[int(site)] = v
        self.vecs = clean
        self.prompt_len = check_prompt_len(prompt_len)

    def wear_injection(self, injection: Injection, prompt_len: int,
                       device: torch.device | str | None = None) -> None:
        """Arm from a typed :class:`Injection`: vectors dosed as
        ``(alpha * rows).astype(float32)`` (``{}`` at alpha 0, the catch arm).
        The injection's span/time profile must match this state's."""
        if (injection.span, injection.time_profile, injection.time_k) != (
                self.span, self.time_profile, self.time_k):
            raise SteerError(
                f"injection is {injection.receipt()} but the installed hooks run "
                f"{self.receipt()}; build the state from the same spec")
        vecs = {} if injection.alpha == 0.0 else site_vectors(
            injection.sites, injection.scaled_rows(), device=device)
        self.wear(vecs, prompt_len)

    def clear(self) -> None:
        self.vecs = {}

    def receipt(self) -> dict[str, Any]:
        return span_receipt(self.span, self.time_profile, self.time_k)


def site_vectors(sites: Sequence[int], rows: NDArray[np.floating] | Any,
                 device: torch.device | str | None = None) -> dict[int, torch.Tensor]:
    """``{site: float32 tensor}`` from ``[n_sites, hidden]`` rows (row j ->
    sites[j]), each ``torch.from_numpy(ascontiguousarray(row, float32))``
    moved to ``device`` — the construction both vLLM lanes used."""
    s = as_sites(sites)
    arr = np.asarray(rows)
    if arr.ndim != 2 or arr.shape[0] != len(s):
        raise VectorShapeError(f"rows must be [{len(s)}, hidden] for sites {list(s)}, "
                               f"got shape {arr.shape}")
    out: dict[int, torch.Tensor] = {}
    for row, site in zip(arr, s, strict=True):
        t = torch.from_numpy(np.ascontiguousarray(row, dtype=np.float32))
        out[site] = t.to(device) if device is not None else t
    return out


def time_weights(positions: torch.Tensor, span: InjectionSpan, time_profile: TimeProfile,
                 prompt_len: int, time_k: float = TIME_K) -> torch.Tensor:
    """Per-position float32 weight in [0, 1] (screen_delivery's rule).

    ``t = position`` (UNIFORM) or ``position - prompt_len`` (CONTINUATION).
    CONSTANT: 1 where the span writes; STEP: ``[t >= time_k]``; RAMP:
    ``clamp(t / time_k, 0, 1) * [t >= 0]``.
    """
    uniform = span is InjectionSpan.UNIFORM
    t = positions.to(torch.float32)
    if not uniform:
        t = t - float(prompt_len)
    if time_profile is TimeProfile.CONSTANT:
        return torch.ones_like(t) if uniform else (t >= 0).to(torch.float32)
    if time_profile is TimeProfile.STEP:
        return (t >= time_k).to(torch.float32)
    if time_profile is TimeProfile.RAMP:
        return torch.clamp(t / time_k, 0.0, 1.0) * (t >= 0).to(torch.float32)
    raise SteerError(f"unhandled time profile {time_profile!r}")


def inject(state: VllmInjectionState, site: int, args: tuple[Any, ...],
           kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """The pre-hook body. Returns ``None`` (vLLM keeps the original args) when
    inert: no vector for ``site``, fewer than 2 positional args, non-tensor
    positions/hidden, or no position selected. Otherwise returns
    ``((positions, h + write, *rest), kwargs)`` with a NEW hidden tensor (the
    caller's is never mutated) in the caller's dtype, and bumps
    ``state.fired``."""
    vec = state.vecs.get(site)
    if vec is None or len(args) < 2 or not isinstance(args[1], torch.Tensor):
        return None
    positions, h = args[0], args[1]
    if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return None
    if not h.is_floating_point():
        raise SteerDTypeError(f"site {site}: hidden states are {h.dtype}; a residual write "
                              "needs a floating dtype — is arg 1 really hidden_states?")
    if vec.shape[-1] != h.shape[-1]:
        raise VectorShapeError(f"site {site}: vector has {vec.shape[-1]} dims, hidden states "
                               f"have {h.shape[-1]} — wrong lever/map for this model?")
    if positions.dim() != 1 or h.dim() != 2 or h.shape[0] != positions.shape[0]:
        raise VectorShapeError(f"site {site}: positions {tuple(positions.shape)} vs hidden "
                               f"states {tuple(h.shape)}; expected vLLM's flattened layout "
                               "positions [num_tokens], hidden [num_tokens, H]")
    v = vec.to(device=h.device, dtype=h.dtype)
    if state.time_profile is TimeProfile.CONSTANT:
        if state.span is InjectionSpan.UNIFORM:
            h2 = h + v
        else:
            mask = (positions >= int(state.prompt_len)).to(h.device)
            if not bool(mask.any()):
                return None
            h2 = h.clone()
            h2[mask] = h2[mask] + v
    else:
        w = time_weights(positions, state.span, state.time_profile, state.prompt_len,
                         state.time_k).to(h.device)
        if not bool((w > 0).any()):
            return None
        h2 = h + w.to(h.dtype).unsqueeze(-1) * v.unsqueeze(0)
    if h2.dtype != h.dtype:  # ★ never hand a promoted residual back to a bf16 model
        h2 = h2.to(h.dtype)
    state.fired = int(state.fired) + 1
    return (args[0], h2) + tuple(args[2:]), kwargs


class SitePreHook:
    """``hook(module, args, kwargs)`` for ``layers[site]``, bound to a shared
    :class:`VllmInjectionState`. A module-level class (not a closure), so it
    is importable, picklable-by-reference and testable on its own."""

    __slots__ = ("site", "state")

    def __init__(self, state: VllmInjectionState, site: int) -> None:
        self.state = state
        self.site = int(site)

    def __call__(self, module: Any, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        return inject(self.state, self.site, args, kwargs)

    def __repr__(self) -> str:
        return f"SitePreHook(site={self.site}, span={self.state.span.value})"


def install_pre_hooks(layers: Sequence[Any], sites: Sequence[int],
                      state: VllmInjectionState) -> list[Any]:
    """Register one :class:`SitePreHook` per site on ``layers[site]``
    (``register_forward_pre_hook(..., with_kwargs=True)``); all-or-none.
    Returns the torch handles."""
    s = as_sites(sites)
    try:
        n = len(layers)
    except TypeError:
        raise SiteError("layers must be a sized sequence (vLLM's model.model.layers)") from None
    check_sites_in_range(s, n)
    handles: list[Any] = []
    try:
        for site in s:
            handles.append(layers[site].register_forward_pre_hook(
                SitePreHook(state, site), with_kwargs=True))
    except BaseException:
        for hd in handles:
            hd.remove()
        raise
    return handles
