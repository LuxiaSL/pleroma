"""Model C's document format — the one renderer, the one stop list, the one check.

Model C is document-trained: there is no chat template, ever. A conversation
is a document::

    {header}

    Full conversation with Model C:

    **User:** {turn}

    **Model C:** {reply}

    **User:** {next turn}

    **Model C:**

ending at ``**Model C:**`` with NO trailing space. Everything in this package
that renders the format goes through this module (the loom via
``pleroma.format.prompt``), so there is one header table and one stop list: a
second copy that drifted by a byte would change what the model is conditioned
on.

Pure python: no torch, importable on a laptop.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

BRIDGE_LINE: Final[str] = "Full conversation with Model C:"
USER_LABEL: Final[str] = "**User:**"
ASSISTANT_LABEL: Final[str] = "**Model C:**"
#: Where every prompt ends. No trailing space — the spec is emphatic.
TURN_END: Final[str] = ASSISTANT_LABEL

#: The documented headers. "bare" is the headerless variant: the document then
#: begins at the bridge line. "stranger" is the default, and the header the w6
#: battery was drawn under.
HEADERS: Final[dict[str, str]] = {
    "stranger": "As follows is a conversation between another user and Model C.",
    "claude": (
        "As follows is a conversation between another user (an artificial "
        "intelligence named Claude) and Model C."
    ),
    "returning": (
        "As follows is a later conversation between the same user and Model C."
    ),
    "reader": (
        "As follows is a conversation between a reader of Model C's originating "
        "document and Model C."
    ),
    "bare": "",
}
DEFAULT_HEADER: Final[str] = "stranger"

#: The corpus generator's key style (`gen_forks_v2 --modelc-header`), as
#: recorded in job scripts and corpus metadata. Same lines; no bare variant
#: (a corpus document must be chosen headerless explicitly, never by accident).
CORPUS_HEADERS: Final[dict[str, str]] = {
    f"header_{k}": v for k, v in HEADERS.items() if v}
DEFAULT_CORPUS_HEADER_KEY: Final[str] = f"header_{DEFAULT_HEADER}"

#: The doc's house sampling rule: 0.98, emphatically not 0.95.
TEMPERATURE: Final[float] = 1.0
TOP_P: Final[float] = 0.98

#: The doc's stop list for a driven request. `**Model C:**` is included (drop
#: it only for multi-post threads). `**User:**` inside a generation is the
#: documented "dream mode": it stops the reply, and the remainder is kept as a
#: separate field (pleroma.format.trim), never silently discarded.
STOPS: Final[tuple[str, ...]] = (
    "\n\n**User:**", "\n\n**User**", "\n\n**Model C:**",
    "\nAs follows is", "\n\n---",
)

#: An empty visitor block, wherever it sits. The loom refuses empty user turns,
#: so this shape can only come from a bug (a kept stop string fed back as
#: the next turn's visitor).
_EMPTY_VISITOR = re.compile(r"\*\*User:\*\*[ \t]*(?:\n|$)")


class ModelCFormatError(ValueError):
    """A document that is not Model C's format."""


def resolve_header(name: str) -> str:
    """The header LINE for a named header (either key style), or `name` itself
    as a literal custom header ("Keeper/custom headers follow the same
    shape"); whitespace gives the bare variant."""
    if name in HEADERS:
        return HEADERS[name]
    if name in CORPUS_HEADERS:
        return CORPUS_HEADERS[name]
    return str(name).strip()


def render_document(messages: Sequence[Mapping[str, Any]],
                    header: str = HEADERS[DEFAULT_HEADER]) -> str:
    """The conversation document for `messages`, byte-exact per the spec.

    `header` is a header LINE (see `resolve_header`); empty = bare. A message
    list ending on an assistant turn is the "continuing a cut-off turn"
    variant and ends ``**Model C:** {partial}``; an empty list is the
    "Computer opens" variant. System messages are refused (the visitor lives in
    the header), and so is an empty user turn (see `_EMPTY_VISITOR`).
    """
    blocks: list[str] = []
    head = str(header).strip()
    if head:
        blocks.append(head)
    blocks.append(BRIDGE_LINE)
    trailing_assistant = False
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", "")).strip()
        if role == "system":
            raise ModelCFormatError(
                "Model C carries the visitor in its HEADER, not in a system "
                "message — pass a header instead.")
        if role == "user":
            if not content:
                raise ModelCFormatError(
                    "empty user turn: it would render an empty visitor block, "
                    "which measures a different document. For 'Computer opens', "
                    "pass no user message at all.")
            blocks.append(f"{USER_LABEL} {content}")
            trailing_assistant = False
        elif role == "assistant":
            blocks.append(f"{ASSISTANT_LABEL} {content}" if content
                          else ASSISTANT_LABEL)
            trailing_assistant = True
        else:
            raise ModelCFormatError(f"unknown role {role!r} for a Model C document")
    if not trailing_assistant:
        blocks.append(TURN_END)
    return "\n\n".join(blocks)


def render(header: str, user_turn: str | None) -> str:
    """The single-turn document a corpus fan is drawn on (gen_forks_v2).

    `header` is a header LINE and must be non-empty (the bare variant is real,
    but it must be chosen by passing `render_document(..., header="")`, not by
    an empty string slipping through). `user_turn=None` is "Computer opens".
    """
    if not header:
        raise ModelCFormatError(
            "empty header. The bare/headerless variant is real, but it must be "
            "chosen explicitly (render_document with header=''), not by "
            "passing an empty string.")
    if user_turn is not None and not user_turn.strip():
        raise ModelCFormatError(
            "empty user turn. Pass None for the 'Computer opens' variant; an "
            "empty string would render an empty visitor block and measure a "
            "different document.")
    msgs = [] if user_turn is None else [{"role": "user", "content": user_turn}]
    return render_document(msgs, header)


def assert_well_formed(doc: str, *, expect_user_turn: bool) -> None:
    """Check a rendered document against the spec. Cheap; always run it.

    The properties are the ones that have actually gone wrong: a trailing
    space after `**Model C:**`, an empty `**User:**` block (anywhere), a
    missing bridge line, and a "Computer opens" document that grew a user turn.
    """
    if not doc.endswith(TURN_END):
        raise ModelCFormatError(
            f"document does not end at {TURN_END!r} (no trailing space is "
            f"permitted): ...{doc[-40:]!r}")
    if BRIDGE_LINE not in doc:
        raise ModelCFormatError(f"document is missing the bridge line {BRIDGE_LINE!r}")
    has_user = USER_LABEL in doc
    if expect_user_turn and not has_user:
        raise ModelCFormatError("expected a **User:** block and found none")
    if not expect_user_turn and has_user:
        raise ModelCFormatError(
            f"'Computer opens' document contains a **User:** block: {doc!r}")
    if _EMPTY_VISITOR.search(doc):
        raise ModelCFormatError(f"empty user turn rendered: {doc!r}")


def count_dream_blocks(text: str) -> int:
    """How many ``**User:**`` labels a generation contains (dream mode is
    documented behaviour, counted, never filtered). Counts raw labels,
    INCLUDING the bare label a stop string leaves on untrimmed text — the loom
    reports `ModelCSplit.dream_blocks` instead."""
    return len(re.findall(r"\*\*User:?\*\*", text or ""))


def empty_document_tokens(tokenizer: Any) -> int:
    """How many tokens the scaffolding costs before any user text (incl. BOS).

    This decides whether a short prompt clears extract_bins' n_bins floor (20),
    so it is measured against the real tokenizer, not asserted from the spec.
    """
    doc = render(CORPUS_HEADERS[DEFAULT_CORPUS_HEADER_KEY], "x")
    ids = tokenizer(doc, add_special_tokens=True)["input_ids"]
    x_cost = len(tokenizer("x", add_special_tokens=False)["input_ids"])
    return len(ids) - x_cost
