"""`Usage` — the token counts one judge response reports, verbatim.

Every transport in `pleroma.judge.transport` returns ``(text, Usage)``; there is
no usage-dropping call. Dollars are DERIVED from these counts
(`pleroma.judge.pricing`), never estimated — tokens are the ground truth.

The counts are kept as the response reported them (every numeric field of its
`usage` block, under the provider's own key names), so a receipt written from
``Usage.counts`` is byte-compatible with the ``usage`` dicts banked before the
refactor. Anthropic reports ``input_tokens`` / ``output_tokens`` /
``cache_creation_input_tokens`` / ``cache_read_input_tokens``; OpenRouter
reports ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
``cached_tokens`` / ``reasoning_tokens`` (flattened by the transport).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Usage(BaseModel):
    """One response's token counts (+ the upstream provider, when one is named)."""

    model_config = ConfigDict(frozen=True)

    #: numeric usage fields exactly as reported; bools and nested objects dropped
    counts: dict[str, int] = Field(default_factory=dict)
    #: OpenRouter routes one model id to several upstreams; Anthropic: None
    provider: str | None = None

    @field_validator("counts")
    @classmethod
    def _non_negative(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: n for k, n in v.items() if n < 0}
        if bad:
            raise ValueError(f"negative token counts in usage: {bad}")
        return v

    @classmethod
    def from_response(cls, raw: Mapping[str, Any] | None,
                      provider: str | None = None) -> "Usage":
        """The numeric fields of a response's `usage` block (bools excluded:
        ``True`` is an int in Python, not a token count)."""
        counts = {str(k): int(v) for k, v in (raw or {}).items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return cls(counts=counts, provider=provider)

    def get(self, key: str, default: int = 0) -> int:
        return self.counts.get(key, default)

    def items(self) -> Iterable[tuple[str, int]]:
        return self.counts.items()

    @property
    def input_tokens(self) -> int:
        return self.get("input_tokens")

    @property
    def output_tokens(self) -> int:
        return self.get("output_tokens")

    @property
    def cache_creation_input_tokens(self) -> int:
        return self.get("cache_creation_input_tokens")

    @property
    def cache_read_input_tokens(self) -> int:
        return self.get("cache_read_input_tokens")


def usage_counts(usage: "Usage | Mapping[str, Any]") -> dict[str, int]:
    """Plain ``{key: int}`` from a `Usage` or a banked usage dict."""
    if isinstance(usage, Usage):
        return dict(usage.counts)
    return {str(k): int(v) for k, v in usage.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def accumulate(totals: dict[str, int], usage: "Usage | Mapping[str, Any]") -> dict[str, int]:
    """Add one response's counts into a running ``totals`` dict, in place."""
    for k, v in usage_counts(usage).items():
        totals[k] = totals.get(k, 0) + v
    return totals
