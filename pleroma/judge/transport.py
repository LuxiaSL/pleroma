"""ONE transport per provider, both returning ``(text, Usage)``.

There is deliberately no usage-dropping call: a call that returns the text
only lets every caller on it run with its spend unrecorded.

Retry skeleton (shared by both providers): a
retryable HTTP status backs off ``2 * 2**attempt + U(0,1)`` seconds; a
URLError/timeout backs off ``2 * 2**attempt``; anything else, or the last
attempt, raises `JudgeAPIError` with the response detail. Retry codes differ by
provider: Anthropic retries 429/500/502/503/529; OpenRouter also
retries 408/409/504 and a 200 whose body carries a retryable ``error`` member.

The API key is passed in by the caller (read from the environment there) and is
never logged. Tests never reach the network: they monkeypatch
``urllib.request.urlopen`` and this module's ``time.sleep``.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from pleroma.judge.usage import Usage

logger = logging.getLogger("pleroma.judge.transport")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_RETRY_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})
ANTHROPIC_TIMEOUT_S = 120.0

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_RETRY_CODES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
OPENROUTER_TIMEOUT_S = 600.0

DEFAULT_RETRIES = 5


class JudgeAPIError(RuntimeError):
    """A judge request that failed for good (non-retryable, or retries spent)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _backoff(attempt: int, jitter: bool) -> float:
    return 2.0 * (2 ** attempt) + (random.random() if jitter else 0.0)


def _post_json(url: str, headers: Mapping[str, str], body: Mapping[str, Any], *,
               retries: int, timeout: float, retry_codes: frozenset[int],
               label: str, error_member: bool = False) -> dict[str, Any]:
    """POST ``body`` as JSON with bounded retries; the decoded response object."""
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    payload = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=dict(headers))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code in retry_codes and attempt < retries - 1:
                wait = _backoff(attempt, jitter=True)
                logger.warning("%s HTTP %d, retry %d in %.1fs", label, exc.code,
                               attempt + 1, wait)
                time.sleep(wait)
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise JudgeAPIError(f"{label} {exc.code}: {detail}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(_backoff(attempt, jitter=False))
                continue
            raise
        if not isinstance(data, dict):
            raise JudgeAPIError(f"{label}: response is not a JSON object: "
                                f"{type(data).__name__}")
        # OpenRouter reports upstream failures as a 200 with an `error` member.
        err = data.get("error") if error_member else None
        if err:
            code = err.get("code") if isinstance(err, Mapping) else None
            if code in retry_codes and attempt < retries - 1:
                time.sleep(_backoff(attempt, jitter=True))
                continue
            raise JudgeAPIError(f"{label} error member: {str(err)[:300]}",
                                status=code if isinstance(code, int) else None)
        return data
    raise JudgeAPIError(f"{label}: retries exhausted")  # unreachable: last attempt raises


# ── Anthropic Messages API ───────────────────────────────────────────────────

def call_anthropic(api_key: str, body: Mapping[str, Any], retries: int = DEFAULT_RETRIES,
                   *, timeout: float = ANTHROPIC_TIMEOUT_S) -> tuple[str, Usage]:
    """One Messages API call -> (concatenated text blocks, Usage).

    Text blocks are joined in order; thinking blocks carry no ``text`` and
    contribute nothing. An empty string is a VALID answer (adaptive thinking can
    consume the whole ``max_tokens``; see `pleroma.judge.request`)."""
    if not api_key:
        raise JudgeAPIError("no Anthropic API key — set ANTHROPIC_API_KEY "
                            "in the environment; it is never logged")
    data = _post_json(
        ANTHROPIC_URL,
        {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION,
         "content-type": "application/json"},
        body, retries=retries, timeout=timeout,
        retry_codes=ANTHROPIC_RETRY_CODES, label="API")
    text = "".join(str(b.get("text", "")) for b in data.get("content", [])
                   if isinstance(b, Mapping))
    return text, Usage.from_response(data.get("usage"))


# ── OpenRouter (OpenAI chat-completions schema) ──────────────────────────────

def flatten_text(content: Any) -> str:
    """An Anthropic content value -> the exact text it renders to.

    A string passes through. A block list concatenates its ``text`` fields IN
    ORDER, which keeps an OpenAI-schema message byte-identical to the Anthropic
    one: ``cache_control`` is metadata, not content, so dropping it changes
    billing and nothing the judge reads."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                if block.get("type") not in (None, "text"):
                    raise ValueError(
                        f"cannot flatten a {block.get('type')!r} block into the "
                        "OpenAI chat schema without changing what the judge reads")
                parts.append(str(block.get("text", "")))
            else:
                raise ValueError(f"unexpected content block {type(block).__name__}")
        return "".join(parts)
    raise ValueError(f"unexpected content {type(content).__name__}")


def to_openai_body(anthropic_body: Mapping[str, Any], model: str) -> dict[str, Any]:
    """An Anthropic-shaped body -> an OpenAI chat-completions body.

    The system block becomes a system-role message and user content is
    flattened; ``max_tokens`` carries over. No sampling params are invented and
    ``reasoning_effort`` is never sent (provider default, on purpose)."""
    system = flatten_text(anthropic_body.get("system", ""))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for msg in anthropic_body.get("messages", []):
        messages.append({"role": str(msg["role"]),
                         "content": flatten_text(msg["content"])})
    body: dict[str, Any] = {"model": model, "messages": messages}
    if "max_tokens" in anthropic_body:
        body["max_tokens"] = int(anthropic_body["max_tokens"])
    return body


def call_openrouter(api_key: str, body: Mapping[str, Any], retries: int = DEFAULT_RETRIES,
                    *, timeout: float = OPENROUTER_TIMEOUT_S) -> tuple[str, Usage]:
    """One chat-completions call -> (first choice's content, Usage).

    ``Usage.counts`` holds prompt/completion/total/cached/reasoning tokens
    (nested detail fields flattened, missing ones 0); ``Usage.provider`` is the
    upstream OpenRouter routed to, banked per call."""
    if not api_key:
        raise JudgeAPIError("no OpenRouter API key — set OPENROUTER_API_KEY")
    data = _post_json(
        OPENROUTER_URL,
        {"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
        body, retries=retries, timeout=timeout,
        retry_codes=OPENROUTER_RETRY_CODES, label="OpenRouter", error_member=True)
    usage = data.get("usage") or {}
    counts = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cached_tokens": int(((usage.get("prompt_tokens_details") or {})
                              .get("cached_tokens", 0)) or 0),
        "reasoning_tokens": int(((usage.get("completion_tokens_details") or {})
                                 .get("reasoning_tokens", 0)) or 0),
    }
    choices = data.get("choices") or []
    text = ""
    if choices:
        text = str((choices[0].get("message") or {}).get("content") or "")
    provider = str(data["provider"]) if data.get("provider") is not None else None
    return text, Usage(counts=counts, provider=provider)


def fetch_openrouter_pricing(model: str, api_key: str, *,
                             timeout: float = 120.0) -> dict[str, Any]:
    """The LIVE per-token prices for ``model`` from OpenRouter's models endpoint.

    Raises ValueError if the id is not listed — ids are never guessed."""
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    for entry in data.get("data", []):
        if str(entry.get("id")) == model:
            pricing = entry.get("pricing") or {}
            return {
                "id": str(entry.get("id")),
                "name": entry.get("name"),
                "context_length": entry.get("context_length"),
                "prompt_per_token": float(pricing["prompt"]),
                "completion_per_token": float(pricing["completion"]),
                "input_cache_read_per_token": (
                    float(pricing["input_cache_read"])
                    if pricing.get("input_cache_read") is not None else None),
                "pricing_raw": pricing,
                "source": OPENROUTER_MODELS_URL,
            }
    available = sorted(str(e.get("id")) for e in data.get("data", [])
                       if "astra" in str(e.get("id", "")).lower())
    raise ValueError(
        f"model id {model!r} is not on {OPENROUTER_MODELS_URL} — do not guess an id. "
        f"Astra-like ids present: {available}")
