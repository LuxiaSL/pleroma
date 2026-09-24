"""Blind lettering for the rank judge: letters, and a resume-stable shuffle rng.

Per-reply (or per-conversation) letter shuffles derive from ``(seed, key)``
via crc32, NOT from one shared rng stream, so a resumed pass letters every
reply identically (Python's salted ``hash()`` and a shared stream's draw
order both break that). Golden for the rank judge's banked letterings: do not
change the constant.
"""

from __future__ import annotations

import random
import string
import zlib

MAX_LETTERS = 26


def letters_for(k: int) -> list[str]:
    """The first ``k`` capital letters — one per future; k > 26 is refused."""
    if k > MAX_LETTERS:
        raise ValueError(f"k={k} > {MAX_LETTERS} — one letter per future is the design")
    return list(string.ascii_uppercase[:k])


def reply_rng(seed: int, reply_id: str) -> random.Random:
    """Per-reply rng from (seed, reply_id): resume-stable letter assignment."""
    return random.Random(seed * 2_654_435_761 + zlib.crc32(reply_id.encode()))
