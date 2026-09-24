"""ONE judge stack: every judge call prices, meters and parses through these modules.

- `pricing` — the one price table, `usage_cost` (cache writes 1.25x, reads
  0.1x), `ledger_dollars`, `openrouter_dollars`, the in-run `SpendMeter`.
- `ledger` — the cross-run `SpendLedger` (hard campaign cap).
- `transport` — `call_anthropic` / `call_openrouter`, both ``(text, Usage)``.
- `request` — bodies with a thinking-aware ``max_tokens`` (8000).
- `prompts` — the registered instruction texts + their sha256.
- `parse` — `parse_json_object`, typed `PairVerdict` / `RankVerdict`,
  `parse_judgment`, `parse_ranking`.
- `letters` — `letters_for`, `reply_rng` (resume-stable blind lettering).
- `pair` — the blind dose-ladder pair judge CLI (``python -m pleroma.judge.pair``).
"""

from pleroma.judge.letters import letters_for, reply_rng
from pleroma.judge.parse import (JudgeParseError, PairVerdict, RankVerdict,
                                 extract_json_object, parse_json_object,
                                 parse_judgment, parse_rank_verdict, parse_ranking)
from pleroma.judge.pricing import (PRICES, PRICES_AS_OF, SpendMeter, UnpricedModelError,
                                   ledger_dollars, openrouter_dollars, price_of,
                                   resolve_price, usage_cost)
from pleroma.judge.request import (THINKING_SAFE_MAX_TOKENS, anthropic_body,
                                   default_max_tokens, is_adaptive_thinking,
                                   messages_body, thinking_starved)
from pleroma.judge.transport import JudgeAPIError, call_anthropic, call_openrouter
from pleroma.judge.usage import Usage

__all__ = [
    "JudgeAPIError", "JudgeParseError", "PRICES", "PRICES_AS_OF", "PairVerdict",
    "RankVerdict", "SpendMeter", "THINKING_SAFE_MAX_TOKENS", "UnpricedModelError",
    "Usage", "anthropic_body", "call_anthropic", "call_openrouter",
    "default_max_tokens", "extract_json_object", "is_adaptive_thinking",
    "ledger_dollars", "letters_for", "messages_body", "openrouter_dollars",
    "parse_json_object", "parse_judgment", "parse_rank_verdict", "parse_ranking",
    "price_of", "reply_rng", "resolve_price", "thinking_starved", "usage_cost",
]
