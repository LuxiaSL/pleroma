"""Stage `format`: the prompt a model sees, and how its output is cut.

Every check goes through the ONE renderer / stop list / trimmer
(`pleroma.format.*`); nothing here re-renders by hand.

| gate            | catalog | question |
|-----------------|---------|----------|
| template        | S-01, S-04 | does the profile's prompt mode render a well-formed prompt? |
| stops           | S-02    | are the format's stop strings configured, and does the trimmer cut at each? |
| pad-eos         | H-01    | can a trailing pad run survive the trim and be harvested as text? |
| empty-visitor   | S-02 (its bug signature) | is an empty visitor block refused everywhere? |
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from pleroma.config import ModelProfile, PromptMode
from pleroma.format import chat, modelc, prompt, trim
from pleroma.validate.result import GateResult, failed, guarded, inconclusive, passed

STAGE = "format"
#: The user turn every render probe uses. Plain, short, no markup.
PROBE_TURN = "What would you do with a free afternoon?"
#: The reply every stop probe prefixes before a stop string.
PROBE_REPLY = "I would walk to the river and read."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@guarded(STAGE, "template")
def gate_template(profile: ModelProfile, tokenizer: Any | None = None) -> GateResult:
    """S-01 (no chat template / wrong format) and S-04 (date drift).

    modelc and raw render without a tokenizer and are checked fully here. chat
    renders only through the tokenizer's template: without a tokenizer the gate
    is INCONCLUSIVE; with one, a missing template FAILs and the render is
    checked for wall-clock dependence (it must be pinned by the date argument).
    """
    mode = profile.format.mode
    if mode is PromptMode.MODELC:
        header = modelc.resolve_header(profile.format.header or "")
        try:
            doc = modelc.render(header, PROBE_TURN)
            modelc.assert_well_formed(doc, expect_user_turn=True)
            opens = modelc.render(header, None)
            modelc.assert_well_formed(opens, expect_user_turn=False)
        except modelc.ModelCFormatError as exc:
            return failed(STAGE, "template",
                          f"model C document does not render to spec: {exc}",
                          mode=mode.value, header=profile.format.header)
        return passed(STAGE, "template",
                      "model C document renders to spec (header, bridge line, ends at "
                      "'**Model C:**' with no trailing space; 'Computer opens' variant too)",
                      mode=mode.value, header=profile.format.header,
                      render_sha16=_sha(doc), render_chars=len(doc))
    if mode is PromptMode.RAW:
        text = prompt.render_raw_text([{"role": "user", "content": PROBE_TURN}])
        if text != PROBE_TURN:
            return failed(STAGE, "template",
                          "raw mode added scaffolding to a single turn: a raw document must "
                          "be the turn text alone", mode=mode.value, rendered=text)
        return passed(STAGE, "template",
                      "raw mode renders a turn as its bare text (no role markers)",
                      mode=mode.value)
    # chat
    if tokenizer is None:
        return inconclusive(STAGE, "template",
                            "chat mode renders through the tokenizer's chat template, and no "
                            "tokenizer was supplied (--tokenizer DIR): the template is unchecked",
                            mode=mode.value)
    if getattr(tokenizer, "chat_template", None) is None:
        return failed(STAGE, "template",
                      "prompt mode is chat but the tokenizer carries no chat template: "
                      "supply the model's native format (profile format.mode raw or modelc)",
                      mode=mode.value)
    msgs = [{"role": "user", "content": PROBE_TURN}]
    date = profile.format.date_string or prompt.DEFAULT_CHAT_DATE
    try:
        ids = prompt.render_prompt_ids("chat", tokenizer, msgs, date_string=date)
        ids_again = prompt.render_prompt_ids("chat", tokenizer, msgs, date_string=date)
        ids_other_day = prompt.render_prompt_ids("chat", tokenizer, msgs,
                                                 date_string="01 Jan 2030")
    except Exception as exc:  # noqa: BLE001 — any template error is the finding
        return failed(STAGE, "template", f"chat template failed to render: "
                      f"{type(exc).__name__}: {exc}", mode=mode.value)
    if not ids:
        return failed(STAGE, "template", "chat template rendered zero tokens", mode=mode.value)
    if ids != ids_again:
        return failed(STAGE, "template",
                      "chat render is not deterministic for a fixed date: the template "
                      "reads something besides its arguments (S-04)", mode=mode.value)
    date_sensitive = ids != ids_other_day
    return passed(STAGE, "template",
                  "chat template renders deterministically"
                  + (f"; it embeds a date, pinned to {date!r} (S-04)" if date_sensitive
                     else "; it embeds no date"),
                  mode=mode.value, n_tokens=len(ids), date_sensitive=date_sensitive,
                  date_string=date)


@guarded(STAGE, "stops")
def gate_stops(profile: ModelProfile) -> GateResult:
    """S-02: stop-string tails and dreamed turns left in the reply.

    modelc: the profile must carry every documented stop (`modelc.STOPS`), and
    `pleroma.format.trim.split_modelc` must cut a reply at each configured stop
    (a stop the trimmer does not know leaves its tail in every length count).
    chat/raw end turns on eos; no stop strings are required.
    """
    stops = tuple(profile.format.stops)
    if profile.format.mode is not PromptMode.MODELC:
        return passed(STAGE, "stops",
                      f"{profile.format.mode.value} mode ends turns on eos ids "
                      f"{list(profile.model.arch.eos_token_ids)}; no stop strings required",
                      mode=profile.format.mode.value, stops=list(stops))
    missing = [s for s in modelc.STOPS if s not in stops]
    if missing:
        return failed(STAGE, "stops",
                      f"profile omits {len(missing)} documented model C stop string(s) "
                      f"{missing!r}: generation runs past the turn and dreamed turns stay "
                      "in the reply", missing=missing, configured=list(stops))
    untrimmed = []
    for s in stops:
        cut = trim.split_modelc(f"{PROBE_REPLY}{s} and then the next thing")
        if cut.reply != PROBE_REPLY:
            untrimmed.append(s)
    if untrimmed:
        return failed(STAGE, "stops",
                      f"stop string(s) {untrimmed!r} are not cut by split_modelc: their "
                      "tails stay in the reply and in every length/word count",
                      untrimmed=untrimmed, configured=list(stops))
    return passed(STAGE, "stops",
                  f"all {len(stops)} model C stop strings configured, and the trimmer "
                  "cuts the reply at each", configured=list(stops))


def check_pad_eos(pad_token_id: int | None, eos_token_ids: Sequence[int],
                  stops: Sequence[str]) -> GateResult:
    """H-01 on raw values (usable even when the profile itself failed to load)."""
    eos = [int(e) for e in eos_token_ids]
    if not eos:
        return failed(STAGE, "pad-eos", "profile declares no eos token ids: nothing ends a "
                      "generation, so no trim is defined", pad_token_id=pad_token_id)
    if pad_token_id is not None and int(pad_token_id) in eos:
        return failed(STAGE, "pad-eos",
                      f"pad id {pad_token_id} is also an eos id: trimming at eos eats real "
                      "ends-of-text, and padding cannot be told from generation (H-01)",
                      pad_token_id=pad_token_id, eos_token_ids=eos)
    if pad_token_id is None:
        if stops:
            return inconclusive(
                STAGE, "pad-eos",
                "stop strings finish rows WITHOUT an eos token, and the profile declares no "
                "pad id: a batched row's trailing padding cannot be stripped and would be "
                "harvested as generated text (H-01) — declare model.arch.pad_token_id",
                pad_token_id=None, eos_token_ids=eos, n_stops=len(stops))
        return passed(STAGE, "pad-eos",
                      "no pad id declared and no stop strings: every finished row ends on "
                      "eos, which the trim cuts at, so no pad run survives",
                      pad_token_id=None, eos_token_ids=eos)
    # the trimmer itself must strip a trailing pad run on a stop-finished row
    row = [11, 12, 13] + [int(pad_token_id)] * 5
    if trim.trim_generated_ids(row, eos, int(pad_token_id)) != [11, 12, 13]:
        return failed(STAGE, "pad-eos",
                      "trim_generated_ids leaves a trailing pad run on a row finished by a "
                      "stop string: pads would be harvested as text (H-01)",
                      pad_token_id=pad_token_id, eos_token_ids=eos)
    return passed(STAGE, "pad-eos",
                  f"pad id {pad_token_id} is not an eos id, and a trailing pad run is "
                  "stripped before harvest",
                  pad_token_id=pad_token_id, eos_token_ids=eos)


@guarded(STAGE, "pad-eos")
def gate_pad_eos(profile: ModelProfile) -> GateResult:
    """H-01: pad tokens harvested as generated text."""
    arch = profile.model.arch
    return check_pad_eos(arch.pad_token_id, arch.eos_token_ids, profile.format.stops)


@guarded(STAGE, "empty-visitor")
def gate_empty_visitor(profile: ModelProfile) -> GateResult:
    """Policy call 6: an empty visitor block is refused everywhere. The shape
    only ever came from a bug (a kept stop string fed back as the next turn),
    so every layer that could produce or pass it must refuse it."""
    accepted: list[str] = []
    for blank in ("", "   ", "\n"):
        try:
            chat.build_messages([], blank, None)
            accepted.append(f"build_messages({blank!r})")
        except ValueError:
            pass
    if profile.format.mode is PromptMode.MODELC:
        header = modelc.resolve_header(profile.format.header or "")
        try:
            modelc.render(header, "")
            accepted.append("modelc.render(header, '')")
        except modelc.ModelCFormatError:
            pass
        try:
            modelc.render_document([{"role": "user", "content": "  "}], header)
            accepted.append("modelc.render_document(blank user turn)")
        except modelc.ModelCFormatError:
            pass
        bad_doc = "\n\n".join(b for b in (header, modelc.BRIDGE_LINE, modelc.USER_LABEL,
                                          modelc.TURN_END) if b)
        try:
            modelc.assert_well_formed(bad_doc, expect_user_turn=True)
            accepted.append("modelc.assert_well_formed(doc with empty **User:** block)")
        except modelc.ModelCFormatError:
            pass
    if accepted:
        return failed(STAGE, "empty-visitor",
                      f"an empty visitor block is accepted by {len(accepted)} path(s): "
                      f"{'; '.join(accepted)} — it renders a different document than the "
                      "one measured", accepted=accepted)
    return passed(STAGE, "empty-visitor",
                  "empty visitor turns are refused by the message builder"
                  + (", the model C renderer, and the well-formedness check"
                     if profile.format.mode is PromptMode.MODELC else ""),
                  mode=profile.format.mode.value)


def run_format(profile: ModelProfile, tokenizer: Any | None = None) -> list[GateResult]:
    return [gate_template(profile, tokenizer), gate_stops(profile),
            gate_pad_eos(profile), gate_empty_visitor(profile)]
