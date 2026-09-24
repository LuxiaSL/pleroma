"""Request bodies, with a thinking-aware ``max_tokens``.

★ WHY THE CEILING IS 8000. Every current Claude 5 judge (Sonnet 5, Opus 5,
Fable 5) runs ADAPTIVE thinking when ``thinking`` is omitted, ``budget_tokens``
is rejected with a 400, and thinking tokens are billed and counted INSIDE
``max_tokens``. A rank judge has spent 7,390 thinking tokens before a
216-char answer; a truncated response yields no parseable JSON, so a tight
ceiling buys nothing and pays for a wasted call (a 400-token ceiling has cost
725 calls for 480 answers). ``max_tokens`` is a ceiling, not a charge —
generous from call one is the cheaper setting.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger("pleroma.judge.request")

#: the thinking-safe ceiling (`default_max_tokens`, and the pair judge's default)
THINKING_SAFE_MAX_TOKENS = 8000

#: model-id prefixes that run adaptive thinking by default
ADAPTIVE_THINKING_PREFIXES: tuple[str, ...] = (
    "claude-sonnet-5", "claude-opus-5", "claude-fable-5")

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def is_adaptive_thinking(model: str) -> bool:
    """True for a model that thinks (inside ``max_tokens``) unless told not to."""
    return str(model).startswith(ADAPTIVE_THINKING_PREFIXES)


def default_max_tokens(model: str) -> int:
    """The per-call ceiling to use when the caller has no reason to choose one."""
    del model  # every model gets the safe ceiling; kept for call-site clarity
    return THINKING_SAFE_MAX_TOKENS


def thinking_starved(model: str, max_tokens: int) -> str | None:
    """A warning if ``max_tokens`` leaves an adaptive-thinking model too little
    room, else None. Callers log it; they do not silently raise the value."""
    if is_adaptive_thinking(model) and int(max_tokens) < THINKING_SAFE_MAX_TOKENS:
        return (f"max_tokens={max_tokens} for {model}, which runs adaptive thinking "
                f"billed INSIDE max_tokens (7,390 thinking tokens observed on one "
                f"answer): expect truncated, unparseable answers. The safe ceiling is "
                f"{THINKING_SAFE_MAX_TOKENS}.")
    return None


def warn_if_thinking_starved(model: str, max_tokens: int,
                             log: logging.Logger | None = None) -> bool:
    msg = thinking_starved(model, max_tokens)
    if msg:
        (log or logger).warning("%s", msg)
    return msg is not None


def anthropic_body(model: str, system: str, content: str | list[dict[str, Any]], *,
                   max_tokens: int | None = None, cache_system: bool = True,
                   effort: str | None = None) -> dict[str, Any]:
    """One Messages API body: a system prompt and ONE user message.

    ``cache_system`` puts the system text in an ephemeral-cached block (the
    judge pattern); False sends it as a plain string (the loom partner, whose
    system prompt changes per turn). ``effort`` (``output_config.effort``) is
    sent only when given — None is the API default (high), i.e. the same
    instrument every banked ladder ran."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS} or None, got {effort!r}")
    mt = default_max_tokens(model) if max_tokens is None else int(max_tokens)
    if mt <= 0:
        raise ValueError(f"max_tokens must be positive, got {mt}")
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": mt,
        "system": ([{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}]
                   if cache_system else system),
        "messages": [{"role": "user", "content": content}],
    }
    if effort is not None:
        body["output_config"] = {"effort": effort}
    return body


def messages_body(model: str, system: str, messages: list[dict[str, str]], *,
                  max_tokens: int | None = None) -> dict[str, Any]:
    """A multi-turn body with a plain-string system prompt (the loom partner)."""
    mt = default_max_tokens(model) if max_tokens is None else int(max_tokens)
    if mt <= 0:
        raise ValueError(f"max_tokens must be positive, got {mt}")
    return {"model": model, "max_tokens": mt, "system": system, "messages": messages}
