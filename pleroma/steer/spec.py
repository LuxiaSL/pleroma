"""The typed description of one steering write, shared by both backends.

One :class:`Injection` says everything a residual write needs: WHERE (map
sites, i.e. ``hidden_states`` indices — the write is a forward PRE-hook on
``layers[site]``, the INPUT of that layer), WHAT
(one row per site, written VERBATIM: ``alpha * row``, never re-normalised — the
ruler and the dose already set its norm), HOW MUCH (``alpha``), WHICH POSITIONS
(``span``) and WHEN (``time_profile``).

Span: UNIFORM — every position, prompt included — is the design and the
loom's behaviour (`docs/FINDINGS.md §4`). CONTINUATION
(positions ``>= prompt_len``) exists only so old measurements (the 70B dose
band) can be reproduced; it must be asked for by name.

Why a frozen dataclass and not pydantic: the payload is an ndarray, which
pydantic can only carry as an opaque ``arbitrary_types_allowed`` field — every
check that matters (shape, finiteness, dtype) would be hand-written either way.
The scalar fields reuse the profile's :class:`~pleroma.config.InjectionSpan`.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pleroma.config.profile import InjectionSpan

#: Token scale of the non-constant time profiles (step onset / ramp length).
#: The scale the temporal-profile screen ran at (`docs/FINDINGS.md §4`).
TIME_K: float = 64.0


class TimeProfile(str, Enum):
    """Per-position weight on the written vector, as a function of the time
    coordinate ``t`` (``t = position`` under UNIFORM, ``position - prompt_len``
    under CONTINUATION).

    CONSTANT: weight 1 wherever the span writes (every copy's behaviour).
    STEP: weight ``[t >= time_k]``. RAMP: ``clamp(t / time_k, 0, 1)`` for
    ``t >= 0``. Non-constant profiles are vLLM-only: the HF write primitive
    (anamnesis ``ResidualWriteSpec``) has no time coordinate.
    """

    CONSTANT = "constant"
    STEP = "step"
    RAMP = "ramp"


# ── errors ───────────────────────────────────────────────────────────────────


class SteerError(ValueError):
    """Base for every refusal in pleroma.steer (a ValueError, so legacy
    ``except ValueError`` call sites keep working)."""


class SiteError(SteerError):
    """A site is not an addressable decoder layer, or sites are malformed."""


class VectorShapeError(SteerError):
    """Vectors do not match the sites or the model's hidden size."""


class SteerDTypeError(SteerError):
    """A tensor cannot carry a residual write (non-floating hidden states,
    non-finite vector)."""


class SpanError(SteerError):
    """A span was asked for without the information it needs (CONTINUATION
    without ``prompt_len``), or a backend cannot express it."""


# ── coercion helpers ─────────────────────────────────────────────────────────


def as_span(span: InjectionSpan | str) -> InjectionSpan:
    try:
        return InjectionSpan(span)
    except ValueError:
        raise SpanError(f"unknown injection span {span!r}; expected one of "
                        f"{[s.value for s in InjectionSpan]}") from None


def as_time_profile(profile: TimeProfile | str) -> TimeProfile:
    try:
        return TimeProfile(profile)
    except ValueError:
        raise SpanError(f"unknown time profile {profile!r}; expected one of "
                        f"{[p.value for p in TimeProfile]}") from None


def as_sites(sites: Sequence[int] | Any) -> tuple[int, ...]:
    """Sites as a tuple of distinct non-negative ints (order preserved: row j
    of the vectors goes to ``sites[j]``)."""
    try:
        out = tuple(sites)
    except TypeError:
        raise SiteError(f"sites must be a sequence of ints, got {type(sites).__name__}") from None
    if not out:
        raise SiteError("no sites given — a steering write needs at least one layer")
    norm: list[int] = []
    for s in out:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
            raise SiteError(f"site {s!r} is not an int (sites={list(out)})")
        if int(s) < 0:
            raise SiteError(f"site {int(s)} is negative (sites={list(out)})")
        norm.append(int(s))
    if len(set(norm)) != len(norm):
        raise SiteError(f"duplicate sites {norm}: each layer takes one row; sum the rows "
                        "yourself if a double write is intended")
    return tuple(norm)


def check_sites_in_range(sites: Sequence[int], n_layers: int) -> None:
    bad = [s for s in sites if not 0 <= int(s) < n_layers]
    if bad:
        raise SiteError(f"site(s) {bad} out of range: the model has {n_layers} decoder layers "
                        f"(valid 0..{n_layers - 1}). Map sites are hidden_states indices and "
                        "hook the INPUT of layers[site]; there is no pre-hook for the stream "
                        "leaving the last layer.")


def check_prompt_len(prompt_len: int | None) -> int:
    if prompt_len is None or isinstance(prompt_len, bool) or not isinstance(
            prompt_len, (int, np.integer)) or int(prompt_len) < 0:
        raise SpanError(f"prompt_len must be an int >= 0, got {prompt_len!r} (CONTINUATION "
                        "starts writing at the first generated position, prompt_len)")
    return int(prompt_len)


# ── the spec ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Injection:
    """One steering write: ``alpha * vectors[j]`` into the input of
    ``layers[sites[j]]`` at the positions ``span`` and ``time_profile`` select.

    ``vectors`` is stored as a C-contiguous float32 ``[n_sites, hidden]`` array
    (no copy when it already is one). It is NOT normalised: it already carries
    the ruler / dose magnitude.
    """

    sites: tuple[int, ...]
    vectors: NDArray[np.float32]
    alpha: float = 1.0
    span: InjectionSpan = InjectionSpan.UNIFORM
    time_profile: TimeProfile = TimeProfile.CONSTANT
    time_k: float = TIME_K

    def __post_init__(self) -> None:
        sites = as_sites(self.sites)
        try:
            vec = np.ascontiguousarray(self.vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VectorShapeError(f"vectors are not a numeric array: {exc}") from None
        if vec.ndim == 1 and len(sites) == 1:
            vec = vec[None, :]
        if vec.ndim != 2:
            raise VectorShapeError(f"vectors must be [n_sites, hidden], got shape {vec.shape}")
        if vec.shape[0] != len(sites):
            raise VectorShapeError(f"{vec.shape[0]} vector rows for {len(sites)} sites "
                                   f"{list(sites)}: one row per site, in site order")
        if vec.shape[1] == 0:
            raise VectorShapeError("vectors have hidden size 0")
        if not np.isfinite(vec).all():
            raise SteerDTypeError("vectors contain NaN/inf — refusing to write them into the "
                                  "residual stream (check the ruler row / lever build)")
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError):
            raise SteerError(f"alpha must be a real number, got {self.alpha!r}") from None
        if not math.isfinite(alpha):
            raise SteerError(f"alpha must be finite, got {alpha}")
        time_k = float(self.time_k)
        if not (math.isfinite(time_k) and time_k > 0):
            raise SteerError(f"time_k must be a positive finite token count, got {self.time_k}")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "vectors", vec)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "span", as_span(self.span))
        object.__setattr__(self, "time_profile", as_time_profile(self.time_profile))
        object.__setattr__(self, "time_k", time_k)

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def hidden(self) -> int:
        return int(self.vectors.shape[1])

    def scaled_rows(self) -> NDArray[np.float32]:
        """``alpha * vectors`` in float32 — the dose baked in, as the vLLM
        lanes have always done it (``(alpha * lever).astype(float32)``)."""
        return (self.alpha * self.vectors).astype(np.float32)

    def receipt(self) -> dict[str, Any]:
        """What a result must record to say which intervention it measured."""
        return span_receipt(self.span, self.time_profile, self.time_k)


def span_receipt(span: InjectionSpan | str, time_profile: TimeProfile | str = TimeProfile.CONSTANT,
                 time_k: float = TIME_K) -> dict[str, Any]:
    """The fields every steered-output receipt carries, so a band records the
    span it was measured under (checked by `pleroma.validate.dose`)."""
    return {"injection_span": as_span(span).value,
            "time_profile": as_time_profile(time_profile).value,
            "time_k": float(time_k)}


def add_injection_span_arg(ap: argparse.ArgumentParser,
                           default: InjectionSpan = InjectionSpan.UNIFORM) -> None:
    """``--injection-span {uniform,continuation}`` for a generator CLI."""
    ap.add_argument(
        "--injection-span", type=as_span, default=default,
        choices=list(InjectionSpan), metavar="{" + ",".join(s.value for s in InjectionSpan) + "}",
        help="which positions receive the steering write. uniform (default, the loom's "
             "span): every position, prompt included. continuation: only positions >= "
             "prompt_len — ONLY to reproduce bands measured before 2026-09-23. Recorded "
             "in the output receipt.")
