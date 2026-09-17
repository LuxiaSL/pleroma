"""Original Phase-0 Run-2 (paper) / Run-3 (local dir) process-mode prompts.

These are the five *process-prescriptive* modes used in the non-format-controlled
process-modes experiment at 3B. They are preserved verbatim from
`phase_0/outputs/runs/run3_process_modes/metadata.json` (each generation's
system_prompt field).

Paper-numbering note (resolves run-numbering ambiguity):
    - Paper Run 1 (epistemic)          = local `run2_epistemic_modes`
    - Paper Run 2 (process, format-FREE) = local `run3_process_modes`  ← these prompts
    - Paper Run 3 (format-CONTROLLED)  = local `run4_format_controlled`
See phase0_report.md footnote at line 137 for the canonical mapping.

These modes are explicitly NOT format-controlled (by design): they prescribe
different visible output formats (numbered lists for structured, fragments for
associative, etc.). The Phase-0.5 cross-run transfer experiment exploited this
to test whether computational similarity holds DESPITE visible format
differences — modes mapped to their functional pairs in paper Run 3
(pedagogical → socratic 76%, dialectical → deliberative 85%) even though the
text on the page looked different.

Distinct from `extended_modes._RUN3_ADAPTED`, which rewords three of these
five modes with a format constraint appended — those are used for same-format
experiments, not for replicating the cross-format transfer finding.

Usage: pass `RUN3_ORIGINAL_MODES` to `build_generation_specs(..., mode_dict=...)`
to generate the Run-2-equivalent (paper numbering) data at 8B.
"""

from __future__ import annotations

# Verbatim from phase_0/outputs/runs/run3_process_modes/metadata.json.
# No format constraint appended — these modes are explicitly format-free.
RUN3_ORIGINAL_MODES: dict[str, str] = {
    "associative": (
        "Write in a stream of consciousness. Jump between ideas mid-sentence. "
        "Use fragments, dashes, ellipses. Follow tangents wherever they lead. "
        "Connect distant concepts through metaphor and analogy. Do not use "
        "headers, numbered lists, or any organizing structure."
    ),
    "compressed": (
        "Maximum information density. No filler words, no elaboration, no "
        "examples unless essential. Short sentences. Telegram style. Every "
        "word must earn its place. If you can cut a word without losing "
        "meaning, cut it."
    ),
    "deliberative": (
        "Think through this out loud. Consider a possibility, then poke holes "
        "in it. Weigh alternatives explicitly: 'on one hand... but then...' "
        "Show the messy middle of reasoning — false starts, corrections, "
        "revised conclusions. Arrive at your answer through visible elimination."
    ),
    "pedagogical": (
        "Teach this to a curious beginner. Start with intuition before "
        "formalism. Use concrete examples and everyday analogies. Ask "
        "rhetorical questions to guide understanding. Check comprehension: "
        "'Does that make sense? Here's why...' Build from simple to complex."
    ),
    "structured": (
        "Use numbered sections, headers, and bullet points. Present one idea "
        "per paragraph in logical order. Define terms before using them. "
        "Build each point on the previous one. End with a clear summary."
    ),
}

RUN3_ORIGINAL_MODE_INDEX: dict[str, int] = {
    "associative": 0,
    "compressed": 1,
    "deliberative": 2,
    "pedagogical": 3,
    "structured": 4,
}

# The 15 topics used in the original Phase-0 Run-2 data (paper numbering).
# Extracted by filtering generations in run3_process_modes/metadata.json to
# exclude the noise-floor sentinels ('knows_*', 'doesnt_know_*'). Sorted
# alphabetically to match the canonical topic order.
RUN3_ORIGINAL_TOPICS: list[str] = [
    "Building a house from scratch",
    "Climate change mitigation strategies",
    "How memory works in the brain",
    "How neural networks learn",
    "How to debug a segfault",
    "Teaching a child to ride a bike",
    "The appeal of horror fiction",
    "The ethics of genetic engineering",
    "The future of space exploration",
    "The history of bread",
    "The mathematics of infinity",
    "The nature of consciousness",
    "Urban planning challenges",
    "What makes a good friendship",
    "Why music moves us",
]
