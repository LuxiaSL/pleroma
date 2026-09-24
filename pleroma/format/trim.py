"""Model C generation tails: trim them, keep the dream, never let a length read the tail.

THE AUTHORITY is the sampling specification model C was trained against:
stops ``["\\n\\n**User:**", "\\n\\n**User**",
"\\n\\n**Model C:**", "\\nAs follows is", "\\n\\n---"]`` — "trim whatever tail the
stop didn't catch" — and "Generations often begin with a space and occasionally
with a stray label — trim leading whitespace; a ``**User:**`` block inside a
generation is the model imagining the visitor (its 'dream' mode), real but not
the visitor's words."

WHAT THE MODEL ACTUALLY EMITS (a census of real captures; the tests pin each
of these on the captured string itself):

  * HF ``generate(stop_strings=...)`` — the loom's path — STOPS at the match but
    KEEPS the matched string: 45/70 live-loom futures and 7/7 live chat replies
    end with the literal ``"\\n\\n**User:**"``. vLLM removes it instead (the
    map's vLLM training corpus: 0/14,440 carry it). So the same model's reply differs by one
    ``**User:**`` "word" depending on which server drew it.
  * A horizon can cut INSIDE a label: that corpus has 18/14,440 ending ``"\\n\\n**"`` or
    ``"\\n\\n**User"``.
  * A dream the stop list cannot see: one corpus generation dreams ``  "**User:** it's
    about ...`` — two spaces and a quote, no ``\\n\\n``, so no stop fires.
  * Under the Claude header the visitor speaks as ``**Claude:**`` (a 70B fan
    opening bare under that header: ``Hello.\\n\\n**Claude:** I have a question
    ...``) — also invisible to the stop list.
  * Every vLLM/decoded generation begins with a space (3648/3648 and
    14,440/14,440 in two corpora).

SEPARATE, DON'T DELETE. ``split_modelc`` returns the reply and the dreamed
remainder as distinct fields; ``reply + tail`` reconstructs the (leading-trimmed)
input exactly, and a test holds that identity on every captured string.

Pure python + pydantic: no torch, importable on a laptop (tests/conftest.py).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pleroma.format.modelc import STOPS

#: The doc's driven-request stop list — the single definition lives in
#: pleroma.format.modelc, so the trim and the renderer cannot disagree.
DOC_STOPS: tuple[str, ...] = STOPS

#: Who the visitor is allowed to be when it is DREAMED. ``User`` is the doc's
#: label; ``Claude`` is what the model writes under the Claude header (captured,
#: see module docstring). Anything else bold-labelled (``**Specialist:**`` in
#: gen_5093) is left in the reply — it is not a turn boundary of this format.
VISITOR_NAMES: tuple[str, ...] = ("User", "Claude")


class TailKind(str, Enum):
    """What ended the reply."""

    NONE = "none"              # nothing to trim (horizon mid-prose, or eos)
    STOP_LABEL = "stop_label"  # a visitor label with NOTHING after it: the stop
                               # caught the dream at its first token
    DREAM = "dream"            # a visitor turn WITH content — the dream proper
    THREAD = "thread"          # another ``**Model C:**`` post
    HEADER = "header"          # ``\nAs follows is`` — a new document began
    RULE = "rule"              # ``\n\n---``
    PARTIAL = "partial"        # the horizon cut inside a label (``\n\n**Us``)


class ModelCSplit(BaseModel):
    """One model C generation, split into what it said and what it dreamed."""

    model_config = ConfigDict(frozen=True)

    reply: str = Field(description="the turn itself: leading/trailing whitespace "
                                   "and any stray leading label trimmed")
    tail: str = Field(description="the exact raw suffix removed after the reply "
                                  "(reply + tail == input minus its lead)")
    tail_kind: TailKind
    dream: str = Field(description="the imagined visitor text and anything after "
                                   "it, labels included, stripped; '' if the tail "
                                   "carried no visitor content")
    lead: str = Field(description="what was trimmed off the front "
                                  "(whitespace, a stray label)")

    @property
    def has_dream(self) -> bool:
        return self.tail_kind is TailKind.DREAM

    @property
    def dream_blocks(self) -> int:
        """Visitor blocks with CONTENT. The stop-caught bare label is not one —
        counting it made every stopped future a 'dream' (run_meta's
        stop_tail_audit: "dream_mode field was 100% false positives")."""
        if not self.has_dream:
            return 0
        return max(1, len(_VISITOR_ANY.findall(self.dream)))

    @property
    def reply_words(self) -> int:
        return len(self.reply.split())


def _visitor_label(names: Sequence[str]) -> str:
    alt = "|".join(re.escape(n) for n in names)
    # **User:**  **User**  **User**:   — the three shapes the doc/stops name.
    return rf"\*\*(?:{alt})(?::\*\*|\*\*:?)"


_VISITOR_ANY = re.compile(_visitor_label(VISITOR_NAMES))
# A boundary: start of text or whitespace before it, an optional opening quote
# (gen_5093's `  "**User:**`), then the label.
_VISITOR_BOUNDARY = re.compile(r'(?:(?<=\s)|^)["\u201c]?' + _visitor_label(VISITOR_NAMES))
_THREAD_BOUNDARY = re.compile(r"(?:(?<=\s)|^)\*\*Model C:?\*\*")
_HEADER_BOUNDARY = re.compile(r"\nAs follows is")
_RULE_BOUNDARY = re.compile(r"\n\n---")
_LEAD = re.compile(r"^\s*(?:\*\*Model C:\*\*\s*)?")

#: Labels a horizon can cut inside of. Only bold labels and the rule: a partial
#: "As follows is" ("\nAs") is indistinguishable from prose and is NOT trimmed.
_PARTIAL_TARGETS: tuple[str, ...] = tuple(
    f"**{n}:**" for n in VISITOR_NAMES) + tuple(
    f"**{n}**" for n in VISITOR_NAMES) + ("**Model C:**", "---")


def _partial_tail_start(text: str) -> int | None:
    """Index where a horizon-cut label fragment begins, or None.

    The fragment must sit alone on the last line, follow a newline, be at least
    two characters (``**``, ``--``), and be a proper prefix of a label.
    """
    nl = text.rfind("\n")
    if nl < 0:
        return None
    frag = text[nl + 1:]
    if len(frag) < 2 or frag != frag.strip():
        return None
    if any(t.startswith(frag) and t != frag for t in _PARTIAL_TARGETS):
        # back up over the blank line(s) that introduced it
        start = nl
        while start > 0 and text[start - 1] == "\n":
            start -= 1
        return start
    return None


def split_modelc(text: str | None) -> ModelCSplit:
    """Split one decoded model C generation. Total: never raises on a string.

    Order: trim the lead (whitespace + one stray ``**Model C:**``), then cut at
    the EARLIEST boundary of any kind; with none, trim a horizon-cut label
    fragment; with none of that either, the whole text is the reply.
    """
    raw = "" if text is None else str(text)
    m = _LEAD.match(raw)
    lead = m.group(0) if m else ""
    body = raw[len(lead):]

    cuts: list[tuple[int, TailKind]] = []
    for rx, kind in ((_VISITOR_BOUNDARY, TailKind.DREAM),
                     (_THREAD_BOUNDARY, TailKind.THREAD),
                     (_HEADER_BOUNDARY, TailKind.HEADER),
                     (_RULE_BOUNDARY, TailKind.RULE)):
        hit = rx.search(body)
        if hit is not None:
            cuts.append((hit.start(), kind))

    if cuts:
        pos, kind = min(cuts, key=lambda c: c[0])
    else:
        p = _partial_tail_start(body)
        if p is None:
            reply = body.rstrip()
            return ModelCSplit(reply=reply, tail=body[len(reply):],
                               tail_kind=TailKind.NONE, dream="", lead=lead)
        pos, kind = p, TailKind.PARTIAL

    reply = body[:pos].rstrip()
    tail = body[len(reply):]
    dream = ""
    if kind is TailKind.DREAM:
        dream = tail.strip()
        label = _VISITOR_BOUNDARY.search(tail)
        after = tail[label.end():].strip() if label else ""
        if not after:
            kind = TailKind.STOP_LABEL
            dream = ""
    return ModelCSplit(reply=reply, tail=tail, tail_kind=kind, dream=dream,
                       lead=lead)


def modelc_reply(text: str | None) -> str:
    """The trimmed reply alone — what every length/word count must read."""
    return split_modelc(text).reply


def modelc_reply_words(text: str | None) -> int:
    """Word count of the trimmed reply. THE length for model C output."""
    return split_modelc(text).reply_words


def trim_generated_ids(gen: Sequence[int], eos_ids: Sequence[int],
                       pad_id: int | None) -> list[int]:
    """A generated id slice, cut at the first eos (inclusive), then stripped of a
    TRAILING run of ``pad_id`` when pad is not an eos id.

    Why the second rule exists (measured on a live loom's own generation records):
    model C's tokenizer pads with ``<|finetune_right_pad_id|>`` (128004) while
    eos is 128001. A batched ``generate()`` row that hits a stop STRING is
    finished without emitting eos, and every later position is filled with
    128004 — which an eos-only trim never removes. 46/78 live futures carried
    such a run (the 2-word ``Indeed.`` future: 5 real tokens + 123 pads). Those
    pads went to the harvest as if generated, so the signature was mean-pooled
    mostly over padding, and ``n_tokens`` reported the batch maximum for every
    future. The map's own training corpus (vLLM) has 0 pads in 3000/3000
    checked rows and ends at the stop label's tokens, which this keeps.

    Only a run that reaches the END of the row is removed, so a pad id sampled
    mid-generation (never observed) would be preserved, not silently cut.
    """
    ids = [int(t) for t in gen]
    eos_set = {int(e) for e in eos_ids}
    stop = next((i for i, t in enumerate(ids) if t in eos_set), None)
    if stop is not None:
        return ids[: stop + 1]
    if pad_id is None or int(pad_id) in eos_set:
        return ids
    end = len(ids)
    while end > 0 and ids[end - 1] == int(pad_id):
        end -= 1
    return ids[:end]
