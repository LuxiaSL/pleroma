"""Judge-reply parsing: JSON extraction and typed verdicts.

Two failure policies, both kept because callers depend on them:

* `parse_json_object` — the outermost ``{...}`` (first ``{`` to last ``}``) or
  **None**. Never raises. Judges sometimes wrap JSON in prose; a truncated or
  empty reply (adaptive thinking ate ``max_tokens``) gives None and the caller
  re-judges it on a later pass.
* `parse_judgment` — the 2AFC pair verdict, or **raises** `JudgeParseError`
  (a ValueError). It extracts exactly the same span as `parse_json_object`
  (a greedy ``\\{.*\\}`` regex and find/rfind agree), then validates the
  object through `PairVerdict`.

★ Confidence is an INTEGER 1-5. JSON ``true`` is not a confidence:
``isinstance(True, int)`` holds, so a plain int check lets a boolean through
as confidence 1; strict validation refuses it.

`parse_ranking` (the rank judge's permutation parser) is golden: a full
permutation of the presented letters, case/whitespace-normalised, or None — a
partial or duplicated ranking is never repaired.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import (BaseModel, ConfigDict, Field, StrictInt, ValidationError,
                      model_validator)

#: judge confidence: a strict int in [1, 5] (bools and numeric strings refused)
Confidence = Annotated[StrictInt, Field(ge=1, le=5)]

CUE_MAX_CHARS = 500


class JudgeParseError(ValueError):
    """A judge reply that does not carry a valid verdict."""


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The outermost {...} in a reply, or None. Judges sometimes add prose."""
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        blob = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return blob if isinstance(blob, dict) else None


def extract_json_object(text: str) -> dict[str, Any]:
    """Like `parse_json_object`, but RAISES `JudgeParseError` saying why."""
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise JudgeParseError(f"no JSON object in judge reply: {text[:120]!r}")
    try:
        blob = json.loads(s[i:j + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError(
            f"judge reply's {{...}} span is not valid JSON ({exc.msg} at char "
            f"{exc.pos}); truncated by max_tokens, or two objects? {text[:120]!r}"
        ) from exc
    if not isinstance(blob, dict):  # pragma: no cover — a {..} span decodes to a dict
        raise JudgeParseError(f"judge reply is not a JSON object: {text[:120]!r}")
    return blob


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']} "
                     f"(got {e.get('input')!r})" for e in exc.errors())


class PairVerdict(BaseModel):
    """The 2AFC pair judge's verdict (`pleroma.judge.prompts.PAIR_2AFC_INSTRUCTIONS`)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    #: the reply called modified, or None = an honest abstention
    call: Literal["A", "B"] | None
    #: 1 = coin flip ... 5 = certain; None iff call is None
    confidence: Confidence | None
    cue: str = ""

    @classmethod
    def from_blob(cls, obj: dict[str, Any]) -> "PairVerdict":
        call = obj.get("call")
        if call not in ("A", "B", None):
            raise JudgeParseError(f"bad call {call!r} (want \"A\", \"B\" or null)")
        cue = str(obj.get("cue", ""))[:CUE_MAX_CHARS]
        if call is None:  # an abstention carries no confidence, whatever was sent
            return cls(call=None, confidence=None, cue=cue)
        conf = obj.get("confidence")
        try:
            return cls(call=call, confidence=conf, cue=cue)
        except ValidationError as exc:
            raise JudgeParseError(
                f"bad confidence {conf!r}: must be an integer 1-5 "
                f"({_validation_message(exc)})") from exc

    @model_validator(mode="after")
    def _confidence_iff_call(self) -> "PairVerdict":
        if self.call is not None and self.confidence is None:
            raise ValueError("confidence is required (integer 1-5) when call is not null")
        return self


def parse_judgment(text: str) -> dict[str, Any]:
    """The pair verdict as the banked ``judge_<model>.json`` schema:
    ``{"call", "confidence", "cue"}``. Raises `JudgeParseError` on anything else."""
    v = PairVerdict.from_blob(extract_json_object(text))
    return {"call": v.call, "confidence": v.confidence, "cue": v.cue}


class RankVerdict(BaseModel):
    """The rank judge's verdict: a full permutation + its stated reason."""

    model_config = ConfigDict(frozen=True)

    ranking: list[str]
    top_reason: str | None = None
    #: None when absent or not a valid 1-5 integer (the ranking is the outcome;
    #: an out-of-range or boolean confidence is not recorded, never coerced)
    confidence: Confidence | None = None


def parse_ranking(raw: str, expected_letters: Sequence[str]) -> list[str] | None:
    """A full permutation of the presented letters, or None (re-judged later)."""
    blob = parse_json_object(raw)
    if blob is None:
        return None
    ranking = blob.get("ranking")
    if not isinstance(ranking, list):
        return None
    got = [str(x).strip().upper() for x in ranking]
    if sorted(got) != sorted(expected_letters):
        return None
    return got


def parse_rank_verdict(raw: str, expected_letters: Sequence[str]) -> RankVerdict | None:
    """`parse_ranking` plus the reason and a validated confidence, or None."""
    ranking = parse_ranking(raw, expected_letters)
    if ranking is None:
        return None
    blob = parse_json_object(raw) or {}
    reason = blob.get("top_reason")
    conf = blob.get("confidence")
    try:
        return RankVerdict(ranking=ranking,
                           top_reason=None if reason is None else str(reason),
                           confidence=conf)
    except ValidationError:
        return RankVerdict(ranking=ranking,
                           top_reason=None if reason is None else str(reason),
                           confidence=None)
