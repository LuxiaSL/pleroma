"""Extended mode set — Run 4 modes + adapted Run 3 modes with format control.

Run 3's structured, compressed, and associative modes were originally
format-free (they produced distinctive output formats). Here they are
adapted with format control to match Run 4's constraint, making them
suitable for feature engineering where surface signal must be eliminated.

This gives 8 modes spanning the full computational diversity spectrum:
  - Easy modes (computationally distinctive): analogical, structured, compressed, associative
  - Hard modes (computationally similar): linear, socratic, contrastive, dialectical
"""

from __future__ import annotations

from .run4_modes import RUN4_MODES

_FORMAT_CONSTRAINT = (
    " Write in flowing paragraphs. Do not use bullet points, numbered lists, "
    "headers, or any visual formatting structure."
)

# Adapted Run 3 modes with format control applied
_RUN3_ADAPTED: dict[str, str] = {
    "structured": (
        "Organize your response as a systematic analysis. Identify the key "
        "components or dimensions of the topic, address each one methodically, "
        "and show how they relate to each other. Be thorough and organized "
        "in your coverage."
        + _FORMAT_CONSTRAINT
    ),
    "compressed": (
        "Express your ideas as densely and concisely as possible. Pack "
        "maximum information into minimum words. Every sentence should "
        "carry significant meaning — eliminate all filler, hedging, and "
        "unnecessary elaboration. Be precise and information-dense."
        + _FORMAT_CONSTRAINT
    ),
    "associative": (
        "Let your thinking flow freely through associations and connections. "
        "When one idea reminds you of another, follow that thread. Explore "
        "the topic through a stream of related concepts, tangents, and "
        "unexpected connections rather than a predetermined structure."
        + _FORMAT_CONSTRAINT
    ),
}

EXTENDED_MODES: dict[str, str] = {
    **RUN4_MODES,
    **_RUN3_ADAPTED,
}

EXTENDED_MODE_INDEX: dict[str, int] = {
    "linear": 0,
    "analogical": 1,
    "socratic": 2,
    "contrastive": 3,
    "dialectical": 4,
    "structured": 5,
    "compressed": 6,
    "associative": 7,
}
