"""ONE price table, ONE usage->dollars formula, ONE in-run spend guard.

Every judge caller prices through this module, because separate tables drift:
a copy that carries a superseded model's rates ($3/$15 per MTok against
claude-sonnet-5's $2/$10) over-states spend 1.5x and trips ``--max-dollars``
early, and a formula that bills prompt-cache WRITES at 1.0x instead of 1.25x
under-states it.

Source: the claude-api skill's pricing table (``PRICES_AS_OF`` records which
reading). Cache rates follow the standard 1.25x write (5-minute ephemeral) /
0.1x read of the input rate.

Refusal policy: pricing an UNKNOWN model is refused, not
metered at zero — a spend guard that silently prices a judge at $0 reports a
reassuring number while the real bill runs. `price_of` raises
`UnpricedModelError`, a `SystemExit` subclass so a CLI stops before its first
dollar. Ledger code that must keep running on an unpriced model (the rank
judge's ``costs_*.json``) uses `ledger_dollars`, which returns None and never
guesses.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pleroma.judge.usage import Usage, usage_counts

#: $ per million tokens, (input, output). Add a model only with a dated source.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
PRICES_AS_OF = ("2026-06-24 (claude-api skill pricing table); claude-opus-5-5 added "
                "2026-09-24 from platform.claude.com/docs/en/about-claude/pricing "
                "(its cache reads bill 0.05x, not 0.1x: moot here, since the judge "
                "prefix is below every model's minimum cacheable length)")

#: 5-minute ephemeral cache writes bill 1.25x the input rate; reads 0.1x.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


class ModelPrice(BaseModel):
    """$/MTok for one model."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class UnpricedModelError(SystemExit):
    """No price on file for a model a metered judge was asked to run.

    Subclasses `SystemExit` so a CLI that prices
    its model up front exits before the first billable call."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"pleroma.judge.pricing: no price on file for model {model!r} (known: "
            f"{sorted(PRICES)}). Refusing to run an unmetered judge — add it to "
            f"PRICES with a dated source, or the --max-dollars guard is a lie.")


def price_of(model: str) -> tuple[float, float]:
    """$/MTok (input, output) for a model, or `UnpricedModelError`."""
    if model not in PRICES:
        raise UnpricedModelError(model)
    return PRICES[model]


def resolve_price(model: str, price_in: float | None = None,
                  price_out: float | None = None) -> ModelPrice | None:
    """Explicit prices (BOTH given) beat the table; else the table; else None.

    This is the rank judge's ``--price-in/--price-out`` rule: one flag alone
    does not override, and an unpriced model with no flags has no price."""
    if price_in is not None and price_out is not None:
        return ModelPrice(input_per_mtok=price_in, output_per_mtok=price_out)
    if model in PRICES:
        pin, pout = PRICES[model]
        return ModelPrice(input_per_mtok=pin, output_per_mtok=pout)
    return None


def _cost(price: ModelPrice, usage: Usage | Mapping[str, Any]) -> float:
    u = usage_counts(usage)
    rate_in, rate_out = price.input_per_mtok, price.output_per_mtok
    return (
        u.get("input_tokens", 0) * rate_in
        + u.get("cache_creation_input_tokens", 0) * rate_in * CACHE_WRITE_MULTIPLIER
        + u.get("cache_read_input_tokens", 0) * rate_in * CACHE_READ_MULTIPLIER
        + u.get("output_tokens", 0) * rate_out
    ) / 1e6


def usage_cost(model: str, usage: Usage | Mapping[str, Any]) -> float:
    """Dollars for one response's (or a running total's) Anthropic usage, from
    ACTUAL token counts. Refuses an unpriced model (`UnpricedModelError`)."""
    pin, pout = price_of(model)
    return _cost(ModelPrice(input_per_mtok=pin, output_per_mtok=pout), usage)


def ledger_dollars(totals: Usage | Mapping[str, Any], model: str,
                   price_in: float | None = None, price_out: float | None = None,
                   *, ndigits: int = 4) -> float | None:
    """A ``costs_*.json`` ledger figure: `usage_cost` rounded to ``ndigits``, or
    None when the model has no price (`resolve_price`).

    None means "cannot be evaluated", never "nothing spent": a caller enforcing
    a cap must refuse on None (ontrunk_screen_fans.Judge does, at construction).
    """
    price = resolve_price(model, price_in, price_out)
    if price is None:
        return None
    return round(_cost(price, totals), ndigits)


def openrouter_dollars(totals: Mapping[str, Any], price: Mapping[str, Any]) -> float:
    """OpenRouter spend from banked counts and the ENDPOINT's own per-token prices.

    Cached prompt tokens bill at the cache-read rate when the endpoint
    publishes one AND reports them; otherwise every prompt token bills at the
    full prompt rate. ``fresh = max(prompt - cached, 0)``; rounded to 6 dp.
    OpenRouter prices come from https://openrouter.ai/api/v1/models at run time
    (`pleroma.judge.transport.fetch_openrouter_pricing`), not from `PRICES`."""
    prompt = int(totals.get("prompt_tokens", 0))
    cached = int(totals.get("cached_tokens", 0))
    completion = int(totals.get("completion_tokens", 0))
    read_rate = price.get("input_cache_read_per_token")
    if read_rate is None:
        cached, read_rate = 0, 0.0
    fresh = max(prompt - cached, 0)
    return round(fresh * float(price["prompt_per_token"])
                 + cached * float(read_rate)
                 + completion * float(price["completion_per_token"]), 6)


class SpendMeter:
    """Thread-safe running total, with a hard stop ABOVE the projection.

    `charge` returns False once the cap is reached; a worker pool then drains
    without issuing further calls (callers check `should_stop` BEFORE each
    billable call). ``max_dollars <= 0`` disables the guard but still meters.

    Set the cap above the staged projection, never at it: a staged rate is a
    LOWER bound whenever a model's thinking varies with content (a rate staged
    on a small batch has understated the real one 3.1x).
    """

    def __init__(self, max_dollars: float) -> None:
        self.max_dollars = float(max_dollars)
        self.spent = 0.0
        self.calls = 0
        self.tokens: dict[str, int] = {}
        self.stopped = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_dollars > 0

    def charge(self, model: str, usage: Usage | Mapping[str, Any]) -> bool:
        """Record one response; False once the guard has fired."""
        cost = usage_cost(model, usage)
        with self._lock:
            self.spent += cost
            self.calls += 1
            for k, v in usage_counts(usage).items():
                self.tokens[k] = self.tokens.get(k, 0) + v
            if self.enabled and self.spent >= self.max_dollars:
                self.stopped = True
            return not self.stopped

    def should_stop(self) -> bool:
        with self._lock:
            return self.stopped

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            per_call = self.spent / self.calls if self.calls else 0.0
            return {
                "spent_usd": round(self.spent, 6),
                "calls": self.calls,
                "usd_per_call": round(per_call, 8),
                "max_dollars": self.max_dollars if self.enabled else None,
                "guard_fired": self.stopped,
                "tokens": dict(self.tokens),
                "prices_as_of": PRICES_AS_OF,
            }
