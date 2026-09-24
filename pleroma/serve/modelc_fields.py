"""Model C per-turn fields and the untrimmed-history repair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pleroma.format.trim import ModelCSplit, split_modelc


def modelc_public_fields(split: ModelCSplit) -> dict[str, Any]:
    """The per-turn fields a model C reply/future carries beside its text.

    ``text``/``content`` is ALWAYS the trimmed
    reply (``split.reply``); the dreamed remainder is its own field, so the UI
    can show or hide it and no length computation can read it by accident.
    ``modelc_dream_blocks`` counts visitor blocks WITH content — the bare
    ``**User:**`` a stop string leaves behind is ``tail_kind: stop_label`` and
    counts zero (``count_dream_blocks`` on the untrimmed text would count it
    as a dream on every stopped future).
    """
    return {"dream": split.dream, "tail_kind": split.tail_kind.value,
            "modelc_dream_blocks": split.dream_blocks,
            "reply_words": split.reply_words}


def retrim_modelc_histories(histories: Mapping[str, list[dict[str, str]]]) -> int:
    """Split assistant turns restored from an untrimmed snapshot, in place.

    A snapshot written without trimming stores model C replies
    with HF's kept stop (``...\n\n**User:**``); left alone, the next turn
    would render an empty visitor block into the document. A message that
    already carries ``tail_kind`` has been split and is not touched, so this is
    idempotent. Returns how many messages changed. modelc mode only.
    """
    changed = 0
    for rows in histories.values():
        for m in rows:
            if m.get("role") != "assistant" or "tail_kind" in m:
                continue
            sp = split_modelc(m.get("content", ""))
            m["content"] = sp.reply
            m["dream"] = sp.dream
            m["tail_kind"] = sp.tail_kind.value
            changed += 1
    return changed
