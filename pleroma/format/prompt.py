"""Prompt rendering for the three prompt modes: chat | raw | modelc.

Pure functions of
(messages, mode, header) plus a tokenizer — no torch, no model — because a
mis-rendered document is the thing that silently corrupts everything
downstream, so it must be testable on a laptop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from pleroma.format import modelc

PROMPT_MODES: Final[tuple[str, ...]] = ("chat", "raw", "modelc")

#: Raw mode's join between consecutive turns: a paragraph break, NO role
#: markers. A "User:/Assistant:" scaffold is a different prompt (a few-shot
#: framing) and would change what the base model continues.
RAW_TURN_JOIN: Final[str] = "\n\n"

#: The date every chat render is pinned to unless a profile says otherwise.
#: It is the Llama-3.1 template's OWN default, so our 8B renders exactly as it
#: always has; a template that would otherwise read the wall clock (Llama-3.2)
#: renders the same prompt the same way on every day.
DEFAULT_CHAT_DATE: Final[str] = "26 Jul 2024"


def render_raw_text(messages: Sequence[Mapping[str, str]]) -> str:
    """Turn texts in order, blank-line joined, no role markers. A system message
    is refused (raw mode has nowhere honest to put it). At zero history this is
    exactly the new turn's text."""
    parts: list[str] = []
    for m in messages:
        if str(m.get("role", "")) == "system":
            raise ValueError(
                "prompt-mode 'raw' has no system role — a raw document is not "
                "a chat. Run without a system prompt, or use prompt-mode chat.")
        parts.append(str(m.get("content", "")).strip())
    return RAW_TURN_JOIN.join(p for p in parts if p)


def render_prompt_text(mode: str, messages: Sequence[Mapping[str, str]],
                       modelc_header: str = modelc.HEADERS[modelc.DEFAULT_HEADER]) -> str:
    """The prompt STRING for a template-free mode. 'chat' has no string form."""
    if mode == "raw":
        return render_raw_text(messages)
    if mode == "modelc":
        return modelc.render_document(messages, modelc_header)
    if mode == "chat":
        raise ValueError(
            "prompt-mode 'chat' renders through the tokenizer's chat template; "
            "it has no mode-owned string form. Use render_prompt_ids.")
    raise ValueError(f"unknown prompt mode {mode!r}")


def render_prompt_ids(mode: str, tok: Any, messages: Sequence[Mapping[str, str]],
                      modelc_header: str = modelc.HEADERS[modelc.DEFAULT_HEADER],
                      *, date_string: str = DEFAULT_CHAT_DATE) -> list[int]:
    """Token ids for one turn under one prompt mode. Never coerces.

    'chat' is `apply_chat_template(add_generation_prompt=True)` with the date
    PINNED (`date_string`); templates that take no date ignore it. The other
    modes encode their document with `add_special_tokens=True` (BOS once).
    """
    if mode == "chat":
        if getattr(tok, "chat_template", None) is None:
            raise RuntimeError(
                "prompt-mode 'chat' needs a chat template and this tokenizer "
                "carries none — use prompt-mode raw or modelc.")
        result = tok.apply_chat_template(list(messages), add_generation_prompt=True,
                                         date_string=date_string)
        ids = result["input_ids"] if hasattr(result, "keys") else result
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in ids]
    text = render_prompt_text(mode, messages, modelc_header)
    return [int(t) for t in tok.encode(text, add_special_tokens=True)]
