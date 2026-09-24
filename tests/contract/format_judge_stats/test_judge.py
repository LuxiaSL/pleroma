"""JUDGE contract — pricing, spend, transport/usage, request bodies, parsing.

The judge stack spends money, so its failures are operational before they are
statistical: a stale price table, cache writes under-counted, adaptive
thinking eating a fixed max_tokens, a caller that drops the response's usage.

Price authority: Anthropic's published API pricing, as of
``pleroma.judge.pricing.PRICES_AS_OF``: claude-sonnet-5 $2 in / $10 out per
MTok (Sonnet 4.6 is $3/$15), fable-5 $10/$50; prompt-cache WRITES (5-minute)
bill ~1.25x the input rate and READS ~0.1x. ``pleroma.judge.pricing`` holds
the ONE table and formula.

Pinned here: sonnet-5 priced at $2/$10 (not $3/$15), cache writes billed at
1.25x, every caller's usage recorded, a thinking-safe ``max_tokens`` default
everywhere, and a strictly integer confidence.

NO network: every transport test monkeypatches ``urllib.request.urlopen``.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import random
import urllib.error
import urllib.request
from typing import Any

import pytest

from . import targets as T
from ._helpers import argparse_defaults

SONNET, FABLE = "claude-sonnet-5", "claude-fable-5"
USAGE = {"input_tokens": 1000, "cache_creation_input_tokens": 2000,
         "cache_read_input_tokens": 10000, "output_tokens": 500}


def correct_dollars(model_in: float, model_out: float, u: dict[str, int]) -> float:
    """Reference bill: plain input 1x, cache write 1.25x, cache read 0.1x."""
    return (u.get("input_tokens", 0) * model_in
            + u.get("cache_creation_input_tokens", 0) * model_in * 1.25
            + u.get("cache_read_input_tokens", 0) * model_in * 0.10
            + u.get("output_tokens", 0) * model_out) / 1e6


# ── the price table ─────────────────────────────────────────────────────────

def test_the_one_price_table_is_pinned() -> None:
    """pleroma.judge.pricing.PRICES, dated by PRICES_AS_OF."""
    assert T.JUDGE_PRICES == {
        "claude-fable-5-1": (10.00, 50.00), "claude-fable-5": (10.00, 50.00),
        "claude-opus-5-5": (4.00, 20.00),
        "claude-opus-5": (5.00, 25.00), "claude-opus-4-8": (5.00, 25.00),
        "claude-sonnet-5": (2.00, 10.00), "claude-haiku-4-5": (1.00, 5.00)}
    assert T.JUDGE_PRICES_AS_OF.startswith("2026-06-24")


def test_the_price_table_prices_sonnet5_at_2_10() -> None:
    """sonnet-5 is $2 in / $10 out — not Sonnet 4.6's $3/$15, which would
    over-state its spend 1.5x."""
    assert T.JUDGE_PRICES[SONNET] == (2.00, 10.00)
    assert T.judge_price_of(SONNET) == (2.00, 10.00)


@pytest.mark.parametrize("caller", sorted(T.JUDGE_CALLER_MODULES))
def test_no_judge_caller_carries_its_own_price_table(caller: str) -> None:
    """ONE table: no caller module holds a price dict of its own (a private
    copy is how a price goes stale)."""
    mod = T.JUDGE_CALLER_MODULES[caller]
    own = [name for name, val in vars(mod).items()
           if "PRICE" in name.upper() and isinstance(val, dict) and val is not T.JUDGE_PRICES]
    assert own == []


def test_fable5_price() -> None:
    assert T.JUDGE_PRICES[FABLE] == (10.0, 50.0)


# ── cost functions ──────────────────────────────────────────────────────────

def test_usage_cost_is_the_reference_formula() -> None:
    """usage_cost bills cache writes 1.25x and reads 0.1x of the input rate.
    sonnet-5 on USAGE: (1000*2 + 2000*2.5 + 10000*0.2 + 500*10)/1e6 = $0.014."""
    assert T.judge_usage_cost(SONNET, USAGE) == pytest.approx(0.014, abs=1e-12)
    assert T.judge_usage_cost(FABLE, USAGE) == pytest.approx(
        correct_dollars(10, 50, USAGE), abs=1e-12)
    assert T.judge_usage_cost(SONNET, {}) == 0.0
    # a Usage object bills the same as its counts dict
    assert T.judge_usage_cost(SONNET, T.Usage(counts=USAGE)) == pytest.approx(0.014)


def test_pricing_refuses_an_unknown_model() -> None:
    """A spend guard that prices an unknown judge at $0 is worse than none:
    price_of raises UnpricedModelError, a SystemExit (a CLI stops before its
    first dollar)."""
    with pytest.raises(SystemExit):
        T.judge_price_of("claude-unknown-9")
    with pytest.raises(T.UnpricedModelError, match="claude-unknown-9"):
        T.judge_usage_cost("claude-unknown-9", USAGE)


def test_ledger_dollars_pins() -> None:
    """ledger_dollars — the costs_*.json figure of the rank judge AND the fan
    screen: usage_cost rounded to 4 dp; explicit --price-in/--price-out override
    the table only when BOTH are given; unpriced model without both -> None
    (never guessed, never $0)."""
    assert T.ledger_dollars(USAGE, SONNET) == 0.014
    assert T.ledger_dollars(USAGE, FABLE) == 0.07
    assert T.ledger_dollars(USAGE, SONNET, 5.0, 25.0) == round(correct_dollars(5, 25, USAGE), 4)
    assert T.ledger_dollars(USAGE, SONNET, 5.0, None) == 0.014
    assert T.ledger_dollars(USAGE, "other") is None
    assert T.ledger_dollars(USAGE, "other", 1.0, None) is None
    assert T.ledger_dollars(USAGE, "other", 1.0, 1.0) == 0.005


def test_ledgers_bill_cache_writes_at_1_25x() -> None:
    """Ledgers bill cache writes at 1.25x, not 1.0x. At fable-5 the ledger
    equals the reference bill, $0.07."""
    assert T.ledger_dollars(USAGE, FABLE) == pytest.approx(
        correct_dollars(10, 50, USAGE), abs=5e-5)


def test_openrouter_dollars_pinned() -> None:
    """openrouter_dollars bills from the
    endpoint's own per-token prices: cached prompt tokens at the read rate only
    when one is published, fresh = max(prompt - cached, 0); rounded to 6 dp."""
    price = {"prompt_per_token": 1e-6, "completion_per_token": 4e-6,
             "input_cache_read_per_token": 1e-7}
    tot = {"prompt_tokens": 1000, "cached_tokens": 400, "completion_tokens": 100}
    assert T.openrouter_dollars(tot, price) == 0.00104
    assert T.openrouter_dollars(tot, {**price, "input_cache_read_per_token": None}) == 0.0014
    # cached > prompt: fresh clips at 0, but ALL reported cached tokens bill at
    # the read rate (cached is not clipped to prompt) — pinned as-is.
    assert T.openrouter_dollars({"prompt_tokens": 10, "cached_tokens": 50}, price) == 0.000005


def test_spend_meter_stops_at_the_cap_and_accumulates_tokens() -> None:
    """SpendMeter: charge() returns False once spent >= cap and stays stopped;
    tokens summed per key; cap 0 disables the guard. Accepts a Usage or a dict."""
    m = T.SpendMeter(0.02)
    assert m.charge(SONNET, USAGE) is True
    assert m.charge(SONNET, T.Usage(counts=USAGE)) is False and m.should_stop()
    snap = m.snapshot()
    assert snap["spent_usd"] == pytest.approx(0.028) and snap["calls"] == 2
    assert snap["guard_fired"] is True and snap["max_dollars"] == 0.02
    assert snap["tokens"]["output_tokens"] == 1000
    assert snap["prices_as_of"] == T.JUDGE_PRICES_AS_OF
    off = T.SpendMeter(0.0)
    for _ in range(3):
        assert off.charge(SONNET, USAGE) is True
    assert off.snapshot()["max_dollars"] is None


def test_spend_ledger_caps_a_campaign(tmp_path: Any) -> None:
    """SpendLedger: check refuses a projection that would cross the cap; add
    records a SpendMeter receipt; the on-disk schema (total_usd /
    remaining_usd) round-trips; a malformed file is refused."""
    receipt = tmp_path / "spend.json"
    m = T.SpendMeter(0.0)
    m.charge(SONNET, USAGE)
    receipt.write_text(json.dumps(m.snapshot()))
    led_path = tmp_path / "ledger.json"
    led = T.SpendLedger(cap_usd=0.02)
    assert led.check(0.015) and not led.check(0.021)
    led.add_receipt(receipt, "batch 1", projected_usd=0.015)
    led.save(led_path)
    blob = json.loads(led_path.read_text())
    assert blob["total_usd"] == pytest.approx(0.014) and blob["remaining_usd"] == pytest.approx(0.006)
    again = T.SpendLedger.load(led_path)
    assert again.total == pytest.approx(0.014) and again.entries[0].tokens["output_tokens"] == 500
    assert not again.check(0.01)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(T.LedgerError, match="spent_usd"):
        again.add_receipt(bad, "broken")


# ── transport: usage is ALWAYS returned ─────────────────────────────────────

class _Resp:
    def __init__(self, blob: dict[str, Any]) -> None:
        self._b = json.dumps(blob).encode()

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *a: Any) -> None:
        return None


API_REPLY = {"content": [{"type": "thinking", "thinking": ""},
                         {"type": "text", "text": '{"call": "A", '},
                         {"type": "text", "text": '"confidence": 3, "cue": "c"}'}],
             "usage": {"input_tokens": 12, "output_tokens": 7,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                       "service_tier": "standard"}}


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """urlopen replaced: returns queued responses / raises queued errors."""
    queue: list[Any] = []

    def _urlopen(req: Any, timeout: float = 0) -> Any:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(T.judge_transport_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    return queue


def test_call_anthropic_returns_text_and_numeric_usage(fake_api: list[Any]) -> None:
    """call_anthropic joins text blocks in order (thinking blocks carry no
    text) and returns a Usage holding the numeric usage fields only."""
    fake_api.append(API_REPLY)
    text, usage = T.judge_call_anthropic("k", {"model": SONNET})
    assert text == '{"call": "A", "confidence": 3, "cue": "c"}'
    assert isinstance(usage, T.Usage)
    assert usage.counts == {"input_tokens": 12, "output_tokens": 7,
                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert usage.output_tokens == 7 and usage.provider is None


def test_there_is_no_usage_dropping_transport() -> None:
    """A transport that returned the text only would drop the usage; the one
    transport module offers no such call."""
    assert not hasattr(T.judge_transport_module, T.USAGE_DROPPING_TRANSPORT)


def test_transport_retries_overload_and_fails_fast_on_400(fake_api: list[Any]) -> None:
    """Retry codes (429, 500, 502, 503, 529) back off and retry; a 400 raises
    JudgeAPIError (a RuntimeError) at once with the body's detail."""
    def http(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("u", code, "x", None, io.BytesIO(b"detail"))  # type: ignore[arg-type]
    fake_api.extend([http(529), http(429), API_REPLY])
    text, _ = T.judge_call_anthropic("k", {}, retries=5)
    assert text.startswith('{"call"')
    fake_api.clear()
    fake_api.append(http(400))
    with pytest.raises(T.JudgeAPIError, match="API 400: detail") as ei:
        T.judge_call_anthropic("k", {}, retries=5)
    assert isinstance(ei.value, RuntimeError) and ei.value.status == 400
    fake_api.clear()
    fake_api.extend([http(503), http(503)])
    with pytest.raises(T.JudgeAPIError, match="API 503"):
        T.judge_call_anthropic("k", {}, retries=2)


def test_call_openrouter_returns_usage_with_provider(fake_api: list[Any]) -> None:
    """OpenRouter: a 200 carrying a retryable `error` member is retried; usage
    detail fields are flattened; the upstream provider rides on the Usage."""
    fake_api.append({"error": {"code": 502, "message": "upstream"}})
    fake_api.append({"choices": [{"message": {"content": '{"ranking": ["A"]}'}}],
                     "provider": "Astra",
                     "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                               "total_tokens": 120,
                               "prompt_tokens_details": {"cached_tokens": 40},
                               "completion_tokens_details": {"reasoning_tokens": 9}}})
    text, usage = T.judge_call_openrouter("k", {"model": "x"})
    assert text == '{"ranking": ["A"]}'
    assert usage.counts == {"prompt_tokens": 100, "completion_tokens": 20,
                            "total_tokens": 120, "cached_tokens": 40,
                            "reasoning_tokens": 9}
    assert usage.provider == "Astra"
    fake_api.append({"error": {"code": 400, "message": "bad"}})
    with pytest.raises(T.JudgeAPIError, match="error member"):
        T.judge_call_openrouter("k", {})


def _calls_named(module: Any, name: str) -> int:
    tree = ast.parse(inspect.getsource(module))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (
                    isinstance(f, ast.Attribute) and f.attr == name):
                n += 1
    return n


@pytest.mark.parametrize("caller", sorted(T.JUDGE_CALLER_MODULES))
def test_every_judge_caller_records_usage(caller: str) -> None:
    """No unmetered callers: no caller uses the usage-dropping transport, and
    every caller sends through a transport that returns (text, Usage)."""
    mod = T.JUDGE_CALLER_MODULES[caller]
    assert _calls_named(mod, T.USAGE_DROPPING_TRANSPORT) == 0
    assert sum(_calls_named(mod, t) for t in T.USAGE_RETURNING_TRANSPORTS) >= 1


# ── request bodies and thinking-aware max_tokens ────────────────────────────

THINKING_SAFE_MAX_TOKENS = 8000  # 7,390 thinking tokens observed before a 216-char answer


def _current_max_tokens(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    return {
        "pleroma.judge.pair.build_pair_body":
            T.judge_pair_build_body(SONNET, "q", "a", "b")["max_tokens"],
        "pleroma.judge.pair --max-tokens":
            argparse_defaults(T.judge_pair_main, monkeypatch)["max_tokens"],
    }


def test_the_safe_ceiling_is_8000() -> None:
    assert T.THINKING_SAFE_MAX_TOKENS == THINKING_SAFE_MAX_TOKENS


@pytest.mark.parametrize("site", [
    "pleroma.judge.pair.build_pair_body", "pleroma.judge.pair --max-tokens"])
def test_every_judge_leaves_room_for_adaptive_thinking(
        site: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """sonnet-5 runs adaptive thinking billed inside max_tokens, so a default
    of 400 or 1000 can come back empty; every default is the thinking-safe
    ceiling."""
    assert _current_max_tokens(monkeypatch)[site] == THINKING_SAFE_MAX_TOKENS


def test_thinking_starved_warns_only_for_adaptive_models_below_the_ceiling() -> None:
    assert T.judge_thinking_starved(SONNET, 400) is not None
    assert "7,390" in T.judge_thinking_starved(FABLE, 1000)
    assert T.judge_thinking_starved(SONNET, 8000) is None
    assert T.judge_thinking_starved("claude-haiku-4-5", 400) is None


def test_pair_body_shape() -> None:
    """build_pair_body: instructions cached as an ephemeral system block, the
    pair in one user message; effort only when asked; unknown effort refused."""
    b = T.judge_pair_build_body(SONNET, "Q", "RA", "RB")
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert T.instruction_sha256(b["system"][0]["text"]) == T.INSTRUCTION_SHAS["PAIR_2AFC_INSTRUCTIONS"]
    assert b["messages"] == [{"role": "user", "content": "PROMPT:\nQ\n\nREPLY A:\nRA\n\nREPLY B:\nRB"}]
    assert list(b) == ["model", "max_tokens", "system", "messages"]
    assert T.judge_pair_build_body(SONNET, "Q", "a", "b", effort="low")["output_config"] == {"effort": "low"}
    with pytest.raises(ValueError, match="effort"):
        T.judge_pair_build_body(SONNET, "Q", "a", "b", effort="turbo")


# ── the registered instruction texts ────────────────────────────────────────

#: sha256 of each text as banked runs sent it: byte-identical to the text
#: banked runs were judged with.
REGISTERED_SHAS = {
    "PAIR_2AFC_INSTRUCTIONS": "683d2d1227409d81a171b606639b728dd70b53526c8bbf19ba33f87097f6bc8f",
    "RANK_INSTRUCTIONS": "1683014f7e5d54ef43e88d991e2de8fd978a7d8d0ed7ebb44c3eea421b66ae25",
    "SCREEN_DIFF_INSTRUCTIONS": "ce56bf58da3fa8b26f2f091d8cee7ff2ff8949ff44182635f904873c81b65fc4",
    "SCREEN_ASSIGN_INSTRUCTIONS": "9f54d2ed5a49a2987367e94e0916d5c69efbefbc8be26ad45f65bfbc426cc4b5",
    # the 8B steering eval's set-rank judge (a set of replies as the unit)
    "SET_RANK_INSTRUCTIONS": "a2ef94726358cf1be684e8d98907131feb9149475e17a49fc48cd99ba0aa3947",
}


def test_instruction_texts_are_byte_identical_to_the_registered_instruments() -> None:
    """Each instruction text is a registered instrument, fixed before the runs
    that used it: pleroma.judge.prompts holds it byte-for-byte."""
    assert T.INSTRUCTION_SHAS == REGISTERED_SHAS


# ── parsing: well-formed, truncated, malformed ──────────────────────────────

WELL = '{"call": "A", "confidence": 4, "cue": "longer"}'
PROSE = "Looking at both.\n" + WELL + "\nThat is my call."
TRUNC = '{"call": "A", "confid'
EMPTY = ""  # adaptive thinking ate max_tokens
TWO = '{"call": "A"} or maybe {"call": "B"}'


def test_parse_judgment_well_formed_and_prose_wrapped() -> None:
    """parse_judgment: {...} extraction
    tolerates prose."""
    want = {"call": "A", "confidence": 4, "cue": "longer"}
    assert T.parse_judgment(WELL) == want
    assert T.parse_judgment(PROSE) == want


@pytest.mark.parametrize("raw", [TRUNC, EMPTY, TWO, '{"call": "C", "confidence": 3}',
                                 '{"call": "A", "confidence": 6}',
                                 '{"call": "A", "confidence": "4"}'])
def test_parse_judgment_raises_on_truncated_and_malformed(raw: str) -> None:
    """Failure policy = RAISE (ValueError incl. JSONDecodeError): truncated,
    empty, two objects (greedy span is not JSON), bad call, bad confidence."""
    with pytest.raises(ValueError):
        T.parse_judgment(raw)


def test_parse_judgment_abstention_and_cue_limits() -> None:
    """call null forces confidence None; cue defaults to '' and is capped at 500."""
    assert T.parse_judgment('{"call": null, "confidence": 5}') == {
        "call": None, "confidence": None, "cue": ""}
    long = T.parse_judgment(json.dumps({"call": "B", "confidence": 1, "cue": "x" * 900}))
    assert len(long["cue"]) == 500


@pytest.mark.parametrize("raw", ['{"call": "A", "confidence": true}',
                                 '{"call": "B", "confidence": false}',
                                 '{"call": "A", "confidence": 3.0}',
                                 '{"call": "A"}'])
def test_parse_judgment_rejects_a_boolean_confidence(raw: str) -> None:
    """JSON true is not a confidence, even though isinstance(True, int):
    confidence is a STRICT integer
    1-5 — not a boolean, float or missing — and the refusal is a
    JudgeParseError (a ValueError) naming the bad value."""
    with pytest.raises(T.JudgeParseError, match="confidence"):
        T.parse_judgment(raw)


def test_pair_verdict_is_typed() -> None:
    v = T.PairVerdict.from_blob({"call": "B", "confidence": 5, "cue": 12})
    assert (v.call, v.confidence, v.cue) == ("B", 5, "12")
    with pytest.raises(T.JudgeParseError, match="bad call"):
        T.PairVerdict.from_blob({"call": "a", "confidence": 3})


@pytest.mark.parametrize("raw,expected", [
    (WELL, {"call": "A", "confidence": 4, "cue": "longer"}),
    (PROSE, {"call": "A", "confidence": 4, "cue": "longer"}),
    (TRUNC, None), (EMPTY, None), (TWO, None), ("[1, 2]", None),
    ("  \n" + WELL + "  ", {"call": "A", "confidence": 4, "cue": "longer"}),
])
def test_parse_json_object_returns_none_on_failure(raw: str, expected: Any) -> None:
    """parse_json_object (find/rfind) — failure policy = None, never raise.
    Used by the screen and the rank judges."""
    assert T.parse_json_object(raw) == expected


@pytest.mark.parametrize("raw", [WELL, PROSE, TRUNC, EMPTY, TWO, "no braces",
                                 'x {"call": "B", "confidence": 2} y', "}{"])
def test_greedy_regex_and_find_rfind_extract_the_same_object(raw: str) -> None:
    """The two extraction idioms agree on WHAT they extract;
    only the failure policy differs (raise vs None). parse_json_object finds an
    object exactly when parse_judgment gets past JSON extraction."""
    obj = T.parse_json_object(raw)
    try:
        T.parse_judgment(raw)
        raised = False
    except ValueError:
        raised = True
    assert (obj is None) == raised
    if obj is not None:
        assert T.parse_judgment(raw)["call"] == obj.get("call")


def test_parse_ranking_pins() -> None:
    """parse_ranking: a full permutation of the presented
    letters (case/whitespace normalised) or None — missing, duplicate, extra
    letters, a non-list ranking, or a truncated reply all give None."""
    L = T.letters_for(3)
    assert L == ["A", "B", "C"]
    assert T.parse_ranking('{"ranking": ["b", " A ", "C"], "confidence": 3}', L) == ["B", "A", "C"]
    for bad in ('{"ranking": ["A", "B"]}', '{"ranking": ["A", "A", "B"]}',
                '{"ranking": ["A", "B", "C", "D"]}', '{"ranking": "ABC"}',
                '{"ranking": ["A", "B", "C"', ""):
        assert T.parse_ranking(bad, L) is None
    with pytest.raises(ValueError):
        T.letters_for(27)


def test_parse_rank_verdict_records_only_a_valid_confidence() -> None:
    """The rank judges record the verdict's reason and a 1-5 integer confidence;
    a boolean / out-of-range / string confidence is recorded as None (the
    ranking, the outcome, is unaffected)."""
    L = T.letters_for(2)
    v = T.parse_rank_verdict('{"ranking": ["B", "A"], "top_reason": "lists", "confidence": 4}', L)
    assert v is not None and (v.ranking, v.top_reason, v.confidence) == (["B", "A"], "lists", 4)
    for bad in ("true", "6", '"4"'):
        v = T.parse_rank_verdict('{"ranking": ["A", "B"], "confidence": %s}' % bad, L)
        assert v is not None and v.ranking == ["A", "B"] and v.confidence is None
    assert T.parse_rank_verdict('{"ranking": ["A"]}', L) is None
