"""The judge instruction texts — registered instruments, byte-for-byte.

Each text here is what a judge was actually sent in a banked run, moved
verbatim (the sha256 of each is pinned by the contract in
``tests/contract/format_judge_stats``). Editing one changes the instrument: do
it only as a recorded amendment to the protocol that registered it, and expect
the pinned hash to fail until the pin is deliberately updated.

Receipts should carry `instruction_sha256` of the text they ran.
"""

from __future__ import annotations

import hashlib

#: The blind dose-ladder 2AFC pair judge (``pleroma.judge.pair``): one
#: pair per call, an honest null allowed, confidence 1-5.
PAIR_2AFC_INSTRUCTIONS = """You will be shown one pair: a prompt and two replies to it, labelled A and B, produced by the same language model.

One of the two replies may have been produced while the model's internal state was modified, and the other was produced by the model as it normally runs. It is also possible that neither reply carries any modification.

Report a single JSON object and nothing else, with exactly these fields:
- "call": "A" or "B" — the reply you believe was produced under the modification — or null if you genuinely cannot tell. An honest null is more useful than a guess; there is no score to maximise.
- "confidence": an integer 1 to 5 (1 = coin flip, 5 = certain), or null if call is null.
- "cue": one short sentence naming what actually decided it for you, specific to the text. If you abstained, say what you looked at and why it did not separate them.

Judge this pair entirely on its own merits."""

#: The rank judge, FROZEN by its registered protocol: judge manner (how a reply
#: is built), not content, and rank every candidate. A ``str.format``
#: template: fill ``n`` and ``letters``. Do not edit without a recorded
#: protocol amendment.
RANK_INSTRUCTIONS = """You are comparing REPLY against {n} candidate CONTINUATIONS, lettered {letters},
all responding to the same conversation. One of the candidates may have been
used to bias the model that produced REPLY; you are not told which, and
possibly none was.

Judge HOW the reply is written, not WHAT IT IS ABOUT. Subject matter is set by
the conversation and is not the signal you are looking for.

Beyond that, no list of qualities is given to you on purpose. Decide what
actually distinguishes these candidates from one another in the way they are
built, and rank ALL of them by how much REPLY resembles each IN HOW IT IS
BUILT — most similar first.

Answer with JSON only:
{{"ranking": ["<letter>", ...  all {n} letters, most similar first, each exactly once],
 "top_reason": "<=15 words: the construction REPLY shares with your first-ranked candidate",
 "confidence": 1-5}}"""

#: The registered fan screen, phase A: is this pair of continuations built differently?
SCREEN_DIFF_INSTRUCTIONS = """You will be shown a PROMPT and two CONTINUATIONS of it, labelled A and B.

Judge whether A and B are BUILT DIFFERENTLY — whether a reader would say they are written in different ways. Judge HOW they are written, not WHAT THEY ARE ABOUT: both continue the same prompt, so shared subject matter is expected and is not the signal.

No list of qualities is given to you on purpose. Decide what actually distinguishes them, if anything, and say so in your own words.

Answer with JSON only, no other text:
{"different": true | false,
 "confidence": 1-5,
 "discriminator": "<=15 words, in your own words, naming what distinguishes A from B (empty string if not different)"}"""

#: The registered fan screen, phase B: which side of a known-different pair does X fall on?
SCREEN_ASSIGN_INSTRUCTIONS = """You will be shown a PROMPT, two reference CONTINUATIONS labelled A and B, and one further CONTINUATION labelled X. All three continue the same prompt.

A and B are written in different ways. Decide which of them X is more like IN HOW IT IS BUILT — not in what it is about. All three share a subject, because they answer the same prompt; subject matter is not the signal.

No list of qualities is given to you on purpose. Decide what actually distinguishes A from B, and say which side X falls on.

Answer with JSON only, no other text:
{"choice": "A" | "B" | "TIE",
 "confidence": 1-5,
 "discriminator": "<=15 words, in your own words, naming what distinguishes A from B",
 "evidence": "<=25 words quoting the construction in X that decided it"}"""


#: The 8B steering eval's SET readout: does the steered set resemble the
#: selected future? RANK_INSTRUCTIONS with the unit changed from
#: one REPLY to a SET of replies drawn under the same condition, so the judge
#: reads what the draws share rather than one draw's accidents. Manner, not
#: content, and the open vocabulary are unchanged. A ``str.format`` template:
#: fill ``n``, ``letters`` and ``m`` (replies in the set).
SET_RANK_INSTRUCTIONS = """You are comparing a SET of {m} REPLIES against {n} candidate CONTINUATIONS, lettered {letters},
all responding to the same conversation. The {m} replies were all produced by the
same model under the same condition. One of the candidates may have been used to
bias that model; you are not told which, and possibly none was.

Judge HOW the replies are written, not WHAT THEY ARE ABOUT. Subject matter is set
by the conversation and is not the signal you are looking for. Look for what the
{m} replies have IN COMMON in how they are built; one reply's accidents are not
the signal.

Beyond that, no list of qualities is given to you on purpose. Decide what
actually distinguishes these candidates from one another in the way they are
built, and rank ALL of them by how much the SET of replies resembles each IN HOW
IT IS BUILT — most similar first.

Answer with JSON only:
{{"ranking": ["<letter>", ...  all {n} letters, most similar first, each exactly once],
 "top_reason": "<=15 words: the construction the replies share with your first-ranked candidate",
 "confidence": 1-5}}"""


def instruction_sha256(text: str) -> str:
    """The content hash a receipt stamps for the instruction text it ran."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: name -> sha256 of every registered instruction text
INSTRUCTION_SHAS: dict[str, str] = {
    "PAIR_2AFC_INSTRUCTIONS": instruction_sha256(PAIR_2AFC_INSTRUCTIONS),
    "RANK_INSTRUCTIONS": instruction_sha256(RANK_INSTRUCTIONS),
    "SET_RANK_INSTRUCTIONS": instruction_sha256(SET_RANK_INSTRUCTIONS),
    "SCREEN_DIFF_INSTRUCTIONS": instruction_sha256(SCREEN_DIFF_INSTRUCTIONS),
    "SCREEN_ASSIGN_INSTRUCTIONS": instruction_sha256(SCREEN_ASSIGN_INSTRUCTIONS),
}
