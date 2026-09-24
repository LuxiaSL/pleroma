"""The ONLY import surface for the format / judge / stats contract.

The primitives live in ``pleroma.format``, ``pleroma.judge`` and
``pleroma.stats``. When a function moves, edit the import HERE and nothing
else (tests/contract/README.md). Every test in this directory reaches code
under test through a name defined in this module.

Everything imported here is CPU-only and import-light (no weights, no
network, no GPU); the heaviest import is scipy via ``pleroma.stats.correlation``.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

# ── FORMAT (golden path: Model C document format) ───────────────────────────
from pleroma.format import modelc as _modelc_format
from pleroma.format import trim as _modelc_trim
from pleroma.serve import legacy as _loom_serve

# pleroma.format.trim — the canonical split / trim
split_modelc = _modelc_trim.split_modelc
modelc_reply = _modelc_trim.modelc_reply
modelc_reply_words = _modelc_trim.modelc_reply_words
trim_generated_ids = _modelc_trim.trim_generated_ids
TailKind = _modelc_trim.TailKind
ModelCSplit = _modelc_trim.ModelCSplit
DOC_STOPS = _modelc_trim.DOC_STOPS
VISITOR_NAMES = _modelc_trim.VISITOR_NAMES

# pleroma.format.modelc — the corpus-side renderer
format_render = _modelc_format.render
format_assert_well_formed = _modelc_format.assert_well_formed
FORMAT_HEADERS = _modelc_format.CORPUS_HEADERS
FORMAT_DEFAULT_HEADER_KEY = _modelc_format.DEFAULT_CORPUS_HEADER_KEY
FORMAT_STOPS = _modelc_format.STOPS
FORMAT_BRIDGE_LINE = _modelc_format.BRIDGE_LINE
FORMAT_TURN_END = _modelc_format.TURN_END
ModelCFormatError = _modelc_format.ModelCFormatError
empty_document_tokens = _modelc_format.empty_document_tokens

# pleroma.serve.legacy — the served render path (module-level, torch-free)
loom_render_modelc_text = _loom_serve.render_modelc_text
loom_render_raw_text = _loom_serve.render_raw_text
loom_render_prompt_text = _loom_serve.render_prompt_text
loom_render_prompt_ids = _loom_serve.render_prompt_ids
from pleroma.format.prompt import DEFAULT_CHAT_DATE as CHAT_DATE_DEFAULT  # noqa: E402,F401
loom_resolve_modelc_header = _loom_serve.resolve_modelc_header
loom_count_dream_blocks = _loom_serve.count_dream_blocks
loom_modelc_public_fields = _loom_serve.modelc_public_fields
loom_retrim_modelc_histories = _loom_serve.retrim_modelc_histories
LOOM_PROMPT_MODES = _loom_serve.PROMPT_MODES
LOOM_MODELC_HEADERS = _loom_serve.MODELC_HEADERS
LOOM_MODELC_DEFAULT_HEADER = _loom_serve.MODELC_DEFAULT_HEADER
LOOM_MODELC_STOPS = _loom_serve.MODELC_STOPS
LOOM_MODELC_TEMPERATURE = _loom_serve.MODELC_TEMPERATURE
LOOM_MODELC_TOP_P = _loom_serve.MODELC_TOP_P
LOOM_MODELC_BRIDGE = _loom_serve.MODELC_BRIDGE
LOOM_RAW_TURN_JOIN = _loom_serve.RAW_TURN_JOIN

# ── JUDGE (pleroma.judge — one stack) ──────────────────────────────────────
from pleroma.judge import letters as _jletters
from pleroma.judge import pair as _jpair
from pleroma.judge import parse as _jparse
from pleroma.judge import pricing as _jpricing
from pleroma.judge import prompts as _jprompts
from pleroma.judge import request as _jrequest
from pleroma.judge import transport as _jtransport
from pleroma.judge import usage as _jusage
from pleroma.judge import ledger as _jledger

# pricing / spend (ONE table, ONE formula)
JUDGE_PRICES = _jpricing.PRICES
JUDGE_PRICES_AS_OF = _jpricing.PRICES_AS_OF
judge_price_of = _jpricing.price_of
judge_usage_cost = _jpricing.usage_cost
UnpricedModelError = _jpricing.UnpricedModelError
SpendMeter = _jpricing.SpendMeter
#: the costs_*.json ledger figure the rank judge and the fan screen write
ledger_dollars = _jpricing.ledger_dollars
openrouter_dollars = _jpricing.openrouter_dollars
SpendLedger = _jledger.SpendLedger
LedgerError = _jledger.LedgerError

# transport (monkeypatched in tests — never reaches the network)
judge_transport_module: ModuleType = _jtransport
judge_call_anthropic = _jtransport.call_anthropic
judge_call_openrouter = _jtransport.call_openrouter
JudgeAPIError = _jtransport.JudgeAPIError
Usage = _jusage.Usage

# request bodies
THINKING_SAFE_MAX_TOKENS = _jrequest.THINKING_SAFE_MAX_TOKENS
judge_thinking_starved = _jrequest.thinking_starved
judge_pair_build_body = _jpair.build_pair_body
judge_pair_main = _jpair.main  # argparse defaults only

# instruction texts (registered instruments)
RANK_INSTRUCTIONS = _jprompts.RANK_INSTRUCTIONS
INSTRUCTION_SHAS = _jprompts.INSTRUCTION_SHAS
instruction_sha256 = _jprompts.instruction_sha256

# parsing
parse_judgment = _jparse.parse_judgment
JudgeParseError = _jparse.JudgeParseError
PairVerdict = _jparse.PairVerdict
parse_json_object = _jparse.parse_json_object
parse_ranking = _jparse.parse_ranking
parse_rank_verdict = _jparse.parse_rank_verdict
letters_for = _jletters.letters_for

#: Every module that sends a judge / partner request. The contract is that
#: each records the response's `usage` (no unmetered caller) and none
#: carries its own price table.
JUDGE_CALLER_MODULES: dict[str, ModuleType] = {
    "pleroma.judge.pair": _jpair,
}
#: The usage-DROPPING transport name: the transport defines no such function
#: and no caller calls one.
USAGE_DROPPING_TRANSPORT = "call_api"
#: The transports that return (text, Usage).
USAGE_RETURNING_TRANSPORTS = ("call_anthropic", "call_openrouter")

# ── STATS (pleroma.stats) ────────────────────────────────────────────────────
from pleroma.stats import bootstrap as _bootstrap
from pleroma.stats import correlation as _correlation
from pleroma.stats import diversity as _diversity
from pleroma.stats import folds as _folds
from pleroma.stats import length_null as _length_null
from pleroma.map.build import cv as _v1a_fit

SPEARMAN_IMPLS = {"pleroma.stats.spearman": _correlation.spearman}
PARTIAL_IMPLS = {
    "pleroma.stats.partial_spearman": _correlation.partial_spearman,
    "pleroma.stats.partial_pearson": _correlation.partial_pearson,
}
LC_MIN_N_FOR_PARTIAL = _correlation.MIN_N_FOR_PARTIAL
LC_HIGH_COLLINEARITY = _correlation.HIGH_COLLINEARITY

BOOTSTRAP_IMPLS = {"pleroma.stats.boot_ci": _bootstrap.boot_ci}
distinct_resamples = _bootstrap.distinct_resamples

# grouped CV folds: the registered scheme (``pleroma.map.build.cv`` carries its
# own FOLDS / FOLD_SEED constants; the contract holds the two equal)
make_folds_seeded = _folds.make_folds
FOLDS = _folds.FOLDS
FOLD_SEED = _folds.FOLD_SEED
V1A_FOLDS = _v1a_fit.FOLDS
V1A_FOLD_SEED = _v1a_fit.FOLD_SEED

# length-only null (a judge that reads length; words, not characters)
length_only_rank = _length_null.length_only_rank
length_only_dnr = _length_null.length_only_dnr
length_chars = _length_null.chars
length_words = _length_null.words
#: Banked fixture extracted from the sitenorm run (in-repo artifact).
LENGTH_NULL_FIXTURE: Path = (Path(__file__).resolve().parents[2] / "fixtures"
                             / "length_null_sitenorm_8b.json")

# fan diversity Jaccard — the .1015 instrument
fan_jaccard = _diversity.fan_jaccard
diversity_words = _diversity.words
