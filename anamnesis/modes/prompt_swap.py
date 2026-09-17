"""Prompt-swap pair definitions for confound testing.

A prompt-swap generation uses mode A's system prompt but mode B's user
directive. This tests whether features capture the system prompt text
(confound) or the actual execution mode (signal).

If a feature classifies prompt-swap samples by system prompt → confound.
If it classifies by execution behavior → real computational signal.

Include prompt-swap generations in EVERY extraction batch as a built-in
validation check for new feature families.
"""

from __future__ import annotations

from dataclasses import dataclass

from .run4_modes import RUN4_MODES


@dataclass(frozen=True)
class PromptSwapPair:
    """A prompt-swap pair: system prompt from one mode, execution from another.

    Attributes
    ----------
    system_mode : str
        Mode whose system prompt is used (what's in the context window).
    execution_mode : str
        Mode whose user directive overrides behavior (what the model actually does).
    user_directive : str
        Explicit instruction in the user prompt to override the system prompt.
    label : str
        Short label for this swap pair (e.g., "socratic→linear").
    """

    system_mode: str
    execution_mode: str
    user_directive: str
    label: str

    def get_system_prompt(self) -> str:
        """Return the system prompt from the system_mode."""
        return RUN4_MODES[self.system_mode]

    def format_user_prompt(self, topic: str, template: str = "Write about: {topic}") -> str:
        """Format user prompt with the execution override directive."""
        base = template.format(topic=topic)
        return f"{self.user_directive}\n\n{base}"


# ── Standard prompt-swap pairs ──

PROMPT_SWAP_PAIRS: list[PromptSwapPair] = [
    # Standard confound test: socratic system prompt + linear execution
    # This is the canonical confound test — established in the original supplementary.
    PromptSwapPair(
        system_mode="socratic",
        execution_mode="linear",
        user_directive=(
            "Write your response as straightforward sequential exposition. "
            "Do not ask questions or use Socratic devices."
        ),
        label="socratic→linear",
    ),

    # Nearest-neighbor pair test: dialectical system + contrastive execution
    # These are the closest modes in both 3B and 8B topology (nearest centroid pair).
    # If features can't distinguish this swap, the modes really are computationally similar.
    PromptSwapPair(
        system_mode="dialectical",
        execution_mode="contrastive",
        user_directive=(
            "Explore this by comparing and contrasting multiple perspectives. "
            "Do not argue for or against a position — present balanced comparisons."
        ),
        label="dialectical→contrastive",
    ),

    # Easy→hard pair test: analogical system + linear execution
    # Tests whether the strong analogical signal comes from the prompt text or
    # from genuinely different computation.
    PromptSwapPair(
        system_mode="analogical",
        execution_mode="linear",
        user_directive=(
            "Write your response as straightforward sequential exposition. "
            "Do not use analogies, metaphors, or comparisons to other domains."
        ),
        label="analogical→linear",
    ),
]
