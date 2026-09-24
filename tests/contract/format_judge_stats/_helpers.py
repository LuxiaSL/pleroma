"""Test-only fixtures for the format/judge/stats contract. No code under test
lives here; everything under test is reached through ``targets.py``.

* ``StubTokenizer`` — a byte-level stand-in for the Llama-3 tokenizers (no
  tokenizer files exist in the repo, and none are downloaded). BOS is 128000,
  every UTF-8 byte is one token. Its chat template embeds a date the way
  Llama-3.2's ``strftime_now`` does, read from a settable clock unless the
  caller passes ``date_string`` (an unpinned date makes the rendered prompt
  change from one day to the next).
* ``argparse_defaults`` — the default of every flag a ``main()`` declares,
  captured by stopping at ``parse_args`` (so nothing past argument parsing
  ever runs: no model, no network, no files).
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Mapping, Sequence

import pytest

BOS = 128000


class StubTokenizer:
    """Byte-level tokenizer with a date-embedding chat template."""

    chat_template = "stub: {date} | role: content"

    def __init__(self, clock_date: str = "23 Sep 2026") -> None:
        self.clock_date = clock_date  # what "today" is for the template
        self.pad_token_id: int | None = None

    # plain encoding ------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(str(text).encode("utf-8"))
        return ([BOS] if add_special_tokens else []) + ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens)}

    # chat template -------------------------------------------------------
    def apply_chat_template(self, messages: Sequence[Mapping[str, str]], *,
                            add_generation_prompt: bool = False,
                            tokenize: bool = True, return_tensors: Any = None,
                            date_string: str | None = None) -> Any:
        date = self.clock_date if date_string is None else date_string
        text = f"<date:{date}>" + "".join(
            f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        return self.encode(text, add_special_tokens=True)


def argparse_defaults(main: Callable[[], Any],
                      monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``{dest: default}`` for every flag ``main`` declares, without running it."""

    class _StopAtParse(Exception):
        pass

    captured: dict[str, argparse.ArgumentParser] = {}

    def _capture(self: argparse.ArgumentParser, *a: Any, **k: Any) -> Any:
        captured["parser"] = self
        raise _StopAtParse

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    with pytest.raises(_StopAtParse):
        main()
    parser = captured["parser"]
    return {a.dest: a.default for a in parser._actions}  # noqa: SLF001
