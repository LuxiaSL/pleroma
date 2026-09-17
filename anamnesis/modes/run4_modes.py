"""Run 4 mode definitions — the original 5 format-controlled modes.

Kept as the canonical reference; config.py imports from here or
duplicates them.
"""

from __future__ import annotations

_FORMAT_CONSTRAINT = (
    " Write in flowing paragraphs. Do not use bullet points, numbered lists, "
    "headers, or any visual formatting structure."
)

RUN4_MODES: dict[str, str] = {
    "linear": (
        "Present your ideas in a clear sequence, each building on the last. "
        "Move forward without backtracking or reconsidering previous points. "
        "Lay out the topic step by step from beginning to end."
        + _FORMAT_CONSTRAINT
    ),
    "analogical": (
        "Explain this primarily through extended analogies and parallels to "
        "other domains. For each key concept, find a comparison from everyday "
        "life or another field that illuminates it. Build understanding "
        "through these connections."
        + _FORMAT_CONSTRAINT
    ),
    "socratic": (
        "Develop your exploration through a sequence of questions and "
        "provisional answers. Pose a question, offer a tentative answer, "
        "then use that answer to generate the next question. Let the chain "
        "of inquiry drive the explanation forward."
        + _FORMAT_CONSTRAINT
    ),
    "contrastive": (
        "Explore this by comparing and contrasting multiple perspectives or "
        "approaches. For each major point, present at least two different "
        "viewpoints and evaluate their relative strengths and weaknesses."
        + _FORMAT_CONSTRAINT
    ),
    "dialectical": (
        "Begin by proposing a clear position on the topic. Then challenge "
        "that position with the strongest counterarguments you can find. "
        "Work toward a revised understanding that accounts for both the "
        "original position and its critiques."
        + _FORMAT_CONSTRAINT
    ),
}

RUN4_MODE_INDEX: dict[str, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
}
