"""FORMAT contract — Model C's document format on the golden path.

Pins the served render (``pleroma.serve.legacy``, modelc mode), the corpus
renderer (``pleroma.format.modelc``), the stop list, sampling defaults, pad/eos
ids, ``trim_generated_ids`` and the dream split (``split_modelc``). The format
itself is stated in ``pleroma.format.modelc``'s module docstring: Model C is
document-trained, has no chat template, and a conversation is a document of
labelled blocks ending at ``**Model C:**``. What these tests guard against: a
chat template on a document model, stop strings and "dream" turns left in a
reply, a dream detector that fires on every stopped reply, a chat render that
reads the wall-clock date, and pad tokens harvested as generated text.
"""

from __future__ import annotations

import pytest

from . import targets as T
from ._helpers import BOS, StubTokenizer

SPEC_STOPS = ("\n\n**User:**", "\n\n**User**", "\n\n**Model C:**",
              "\nAs follows is", "\n\n---")
STRANGER = "As follows is a conversation between another user and Model C."
GOLDEN_ONE_TURN = (
    "As follows is a conversation between another user and Model C.\n\n"
    "Full conversation with Model C:\n\n"
    "**User:** What is a lighthouse for?\n\n"
    "**Model C:**"
)


# ── constants: stops, sampling, special ids ─────────────────────────────────

def test_stop_list_is_the_spec_list_in_every_home() -> None:
    """The five driven-request stops, verbatim and in order, under all three
    names (the loom's MODELC_STOPS, the trim's DOC_STOPS, the renderer's
    STOPS). One list, three names: none may drift from the others."""
    assert tuple(T.LOOM_MODELC_STOPS) == SPEC_STOPS
    assert tuple(T.DOC_STOPS) == SPEC_STOPS
    assert tuple(T.FORMAT_STOPS) == SPEC_STOPS


def test_loom_modelc_sampling_defaults_are_T1_top_p_098() -> None:
    """Format spec house rule: T 1.0, top_p 0.98 ("keeps the novel tail"), not
    the chat default 0.95."""
    assert T.LOOM_MODELC_TEMPERATURE == 1.0
    assert T.LOOM_MODELC_TOP_P == 0.98


# ── rendering ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("turn", ["What is a lighthouse for?",
                                  "  What is a lighthouse for?\n"])
def test_three_single_turn_renderers_are_byte_identical(turn: str) -> None:
    """The corpus renderer and the loom produce the same bytes for one user
    turn under the stranger header (GOLDEN_ONE_TURN, which every renderer of
    this format is held to). Turns are stripped."""
    a = T.format_render(T.FORMAT_HEADERS[T.FORMAT_DEFAULT_HEADER_KEY], turn)
    b = T.loom_render_modelc_text([{"role": "user", "content": turn}])
    assert a == b == GOLDEN_ONE_TURN


def test_prompt_ends_at_the_label_with_no_trailing_space() -> None:
    """The spec is emphatic: the document ends at ``**Model C:**`` with NO
    trailing space (``pleroma.format.modelc.TURN_END``)."""
    doc = T.loom_render_modelc_text([{"role": "user", "content": "hi"}])
    assert doc.endswith("**Model C:**") and not doc.endswith(" ")
    T.format_assert_well_formed(doc, expect_user_turn=True)


def test_multi_turn_document_golden() -> None:
    """History renders as alternating labelled blocks joined by a blank line,
    ending at the generation label (``render_modelc_text``)."""
    doc = T.loom_render_modelc_text([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ])
    assert doc == (f"{STRANGER}\n\nFull conversation with Model C:\n\n"
                   "**User:** one\n\n**Model C:** two\n\n**User:** three\n\n**Model C:**")


def test_continuing_a_cut_off_turn_ends_at_the_partial_text() -> None:
    """A history whose last message is the assistant's is the spec's
    'continuing a cut-off turn' variant: no new label is appended."""
    doc = T.loom_render_modelc_text([{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "part"}])
    assert doc.endswith("\n\n**Model C:** part")


def test_computer_opens_variant_agrees_between_renderers() -> None:
    """No user turn at all ('Computer opens'): the corpus render(h, None) and
    the loom's empty history render the same document, with no **User:** block."""
    a = T.format_render(STRANGER, None)
    b = T.loom_render_modelc_text([], STRANGER)
    assert a == b == f"{STRANGER}\n\nFull conversation with Model C:\n\n**Model C:**"
    T.format_assert_well_formed(a, expect_user_turn=False)


def test_header_tables_agree_on_text_but_not_on_keys() -> None:
    """Two header tables. Same four header LINES; the loom keys them
    'stranger'…, the corpus renderer keys them 'header_stranger'…, and only
    the loom has the documented 'bare' (empty) variant. Pinned so a merge of
    the two is deliberate."""
    for name in ("stranger", "claude", "returning", "reader"):
        assert T.LOOM_MODELC_HEADERS[name] == T.FORMAT_HEADERS[f"header_{name}"]
    assert T.LOOM_MODELC_HEADERS["bare"] == ""
    assert "header_bare" not in T.FORMAT_HEADERS
    assert T.LOOM_MODELC_DEFAULT_HEADER == "stranger"
    assert T.FORMAT_DEFAULT_HEADER_KEY == "header_stranger"


def test_bare_header_divergence_is_pinned() -> None:
    """The loom renders the headerless variant from an empty header (begins at
    the bridge line); the corpus renderer REFUSES an empty header. Divergence
    pinned as-is, so a merge of the two is deliberate."""
    doc = T.loom_render_modelc_text([{"role": "user", "content": "hi"}], "")
    assert doc == "Full conversation with Model C:\n\n**User:** hi\n\n**Model C:**"
    with pytest.raises(T.ModelCFormatError):
        T.format_render("", "hi")


def test_every_renderer_refuses_an_empty_user_turn() -> None:
    """One policy: an empty user turn is refused by every renderer — it would
    render an empty visitor block, which measures a different document. (In
    practice that shape comes from an untrimmed reply that kept HF's stop.)"""
    with pytest.raises(T.ModelCFormatError):
        T.format_render(STRANGER, "   ")
    with pytest.raises(T.ModelCFormatError):
        T.loom_render_modelc_text([{"role": "user", "content": "   "}])


def test_well_formed_check_rejects_an_empty_visitor_block_anywhere() -> None:
    """Any empty ``**User:**`` block is malformed, whichever way it got into
    the document — mid-document (the kept-stop bug's shape) or at the end."""
    head = f"{STRANGER}\n\nFull conversation with Model C:\n\n"
    for doc in (head + "**User:**\n\n**Model C:**",
                head + "**User:** hi\n\n**Model C:** yo\n\n**User:**\n\n**Model C:**",
                head + "**User:** \n\n**Model C:**"):
        with pytest.raises(T.ModelCFormatError):
            T.format_assert_well_formed(doc, expect_user_turn=True)


@pytest.mark.parametrize("role", ["system", "tool"])
def test_loom_modelc_refuses_system_and_unknown_roles(role: str) -> None:
    """The visitor lives in the HEADER, not a system message; unknown roles are
    refused rather than rendered."""
    with pytest.raises(ValueError):
        T.loom_render_modelc_text([{"role": role, "content": "x"}])


def test_header_resolution() -> None:
    """Named header -> its line; anything else is a literal custom header
    (stripped); whitespace -> the bare variant (``resolve_modelc_header``)."""
    assert T.loom_resolve_modelc_header("stranger") == STRANGER
    assert T.loom_resolve_modelc_header("  As follows is X.  ") == "As follows is X."
    assert T.loom_resolve_modelc_header("   ") == ""


def test_well_formed_check_catches_the_three_known_mistakes() -> None:
    """assert_well_formed: trailing space, missing bridge, and a
    'Computer opens' document that grew a user turn."""
    good = T.format_render(STRANGER, "hi")
    with pytest.raises(T.ModelCFormatError):
        T.format_assert_well_formed(good + " ", expect_user_turn=True)
    with pytest.raises(T.ModelCFormatError):
        T.format_assert_well_formed(good.replace(T.FORMAT_BRIDGE_LINE, ""),
                                    expect_user_turn=True)
    with pytest.raises(T.ModelCFormatError):
        T.format_assert_well_formed(good, expect_user_turn=False)


def test_modelc_ids_are_bos_plus_the_document() -> None:
    """render_prompt_ids('modelc') encodes the document with
    add_special_tokens=True: BOS exactly once, nothing else added."""
    tok = StubTokenizer()
    msgs = [{"role": "user", "content": "What is a lighthouse for?"}]
    ids = T.loom_render_prompt_ids("modelc", tok, msgs)
    assert ids == [BOS] + list(GOLDEN_ONE_TURN.encode("utf-8"))
    assert ids.count(BOS) == 1


def test_empty_document_token_cost_with_a_byte_tokenizer() -> None:
    """empty_document_tokens = scaffold tokens incl. BOS, minus the
    placeholder 'x' (the number that decides the bins floor: a prompt shorter
    than the floor produces fans that cannot steer). With a
    byte tokenizer that is BOS + len(scaffold bytes) - 1."""
    scaffold = (f"{STRANGER}\n\nFull conversation with Model C:\n\n"
                "**User:** x\n\n**Model C:**")
    assert T.empty_document_tokens(StubTokenizer()) == 1 + len(scaffold) - 1


def test_prompt_modes_and_mode_refusals() -> None:
    """Three modes. 'chat' has no string form; an unknown mode is refused; chat
    on a template-less tokenizer is refused (a document-format model has no
    chat template to apply)."""
    assert tuple(T.LOOM_PROMPT_MODES) == ("chat", "raw", "modelc")
    with pytest.raises(ValueError):
        T.loom_render_prompt_text("chat", [])
    with pytest.raises(ValueError):
        T.loom_render_prompt_text("nope", [])
    tok = StubTokenizer()
    tok.chat_template = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        T.loom_render_prompt_ids("chat", tok, [{"role": "user", "content": "x"}])


def test_raw_mode_is_blank_line_joined_with_no_role_markers() -> None:
    """Raw continuation (the .1015 render): turn texts, stripped, joined by a
    blank line; a system message is refused (``render_raw_text``)."""
    assert T.LOOM_RAW_TURN_JOIN == "\n\n"
    msgs = [{"role": "user", "content": " a "}, {"role": "assistant", "content": "b"}]
    assert T.loom_render_raw_text(msgs) == "a\n\nb"
    with pytest.raises(ValueError):
        T.loom_render_raw_text([{"role": "system", "content": "s"}])


# ── chat render and the date ────────────────────────────────────────────────

MSGS = [{"role": "user", "content": "hello"}]


def test_loom_chat_render_is_the_template_call_with_the_date_pinned() -> None:
    """The chat branch is apply_chat_template(add_generation_prompt=True) with
    date_string pinned to DEFAULT_CHAT_DATE — the Llama-3.1 template's own
    default, so the 8B renders exactly as every banked 8B number was measured."""
    tok = StubTokenizer("01 Jan 2026")
    assert T.CHAT_DATE_DEFAULT == "26 Jul 2024"
    assert T.loom_render_prompt_ids("chat", tok, MSGS) == tok.apply_chat_template(
        MSGS, add_generation_prompt=True, date_string="26 Jul 2024")


def test_chat_render_does_not_depend_on_the_wall_clock() -> None:
    """Rendering the same messages on two different days
    yields the same ids (the date is pinned, never read from the clock)."""
    day1 = T.loom_render_prompt_ids("chat", StubTokenizer("22 Sep 2026"), MSGS)
    day2 = T.loom_render_prompt_ids("chat", StubTokenizer("23 Sep 2026"), MSGS)
    assert day1 == day2


# ── trim_generated_ids ──────────────────────────────────────────────────────

EOS_C, PAD_C = 128001, 128004


def test_pad_run_after_a_stop_string_is_stripped() -> None:
    """A batched row that stopped on a stop STRING is pad-filled with 128004
    and never emits eos; the trailing pad run is not generated text
    (the 2-word 'Indeed.' future: 5 real
    tokens + 123 pads)."""
    row = [11, 12, 13, 14, 15] + [PAD_C] * 123
    assert T.trim_generated_ids(row, [EOS_C], PAD_C) == [11, 12, 13, 14, 15]


def test_first_eos_wins_inclusive() -> None:
    """The first eos is the true stopping point, kept inclusively; anything after
    it (pads included) is dropped."""
    assert T.trim_generated_ids([1, 2, EOS_C, PAD_C, PAD_C], [EOS_C], PAD_C) == [1, 2, EOS_C]


def test_pad_equal_to_eos_is_the_old_first_eos_rule() -> None:
    """On the 8B/3B stacks pad_id is eos_ids[0]; the rule reduces to a plain
    first-eos cut, and a row with no eos is returned whole."""
    eos = [128001, 128008, 128009]
    assert T.trim_generated_ids([1, 2, 128009, 128009], eos, 128009) == [1, 2, 128009]
    assert T.trim_generated_ids([1, 2, 3], eos, 128001) == [1, 2, 3]


def test_mid_row_pad_is_preserved_and_none_pad_is_a_no_op() -> None:
    """Only a TRAILING run is cut; pad_id=None never strips; an all-pad row
    trims to empty."""
    assert T.trim_generated_ids([1, PAD_C, 2], [EOS_C], PAD_C) == [1, PAD_C, 2]
    assert T.trim_generated_ids([1, PAD_C], [EOS_C], None) == [1, PAD_C]
    assert T.trim_generated_ids([PAD_C, PAD_C], [EOS_C], PAD_C) == []


# ── split_modelc: reply vs dream ────────────────────────────────────────────

SPLIT_CASES = {
    # key: (raw, reply, tail_kind, dream, dream_blocks)
    "hf_kept_stop": ("Indeed.\n\n**User:**", "Indeed.", "stop_label", "", 0),
    "vllm_leading_space": (" Why not?", "Why not?", "none", "", 0),
    "dream_proper": (" Hello.\n\n**User:** what next?\n\n**Model C:** fine",
                     "Hello.", "dream", "**User:** what next?\n\n**Model C:** fine", 1),
    "claude_visitor_dream": (" Hello.\n\n**Claude:** I have a question for you.\n\n**Model C:**",
                             "Hello.", "dream",
                             "**Claude:** I have a question for you.\n\n**Model C:**", 1),
    "thread": ("Reply.\n\n**Model C:** another post", "Reply.", "thread", "", 0),
    "new_document": ("Reply.\nAs follows is a conversation", "Reply.", "header", "", 0),
    "rule": ("Reply.\n\n---\nmore", "Reply.", "rule", "", 0),
    "horizon_in_label": ("Reply text.\n\n**Us", "Reply text.", "partial", "", 0),
    "horizon_bare_stars": ("Reply text.\n\n**", "Reply text.", "partial", "", 0),
    "horizon_mid_prose": ("cut off in the mid", "cut off in the mid", "none", "", 0),
    "partial_as_is_prose": ("Reply.\nAs", "Reply.\nAs", "none", "", 0),
    "invisible_dream": ('as follows:  "**User:** hello there', "as follows:", "dream",
                        '"**User:** hello there', 1),
    "other_bold_label_is_prose": ("Said **Specialist:** hi", "Said **Specialist:** hi",
                                  "none", "", 0),
    "stray_leading_label": ("**Model C:** Sure.", "Sure.", "none", "", 0),
}


@pytest.mark.parametrize("key", sorted(SPLIT_CASES))
def test_split_modelc_pins(key: str) -> None:
    """split_modelc on each tail shape found in banked model-C output:
    reply, tail kind, dream text and dream-block count; and the lossless identity
    lead + reply + tail == raw ("separate, don't delete")."""
    raw, reply, kind, dream, blocks = SPLIT_CASES[key]
    sp = T.split_modelc(raw)
    assert sp.reply == reply
    assert sp.tail_kind.value == kind
    assert sp.dream == dream
    assert sp.dream_blocks == blocks
    assert sp.has_dream is (kind == "dream")
    assert sp.lead + sp.reply + sp.tail == raw


@pytest.mark.parametrize("stop,kind", [
    ("\n\n**User:**", "stop_label"), ("\n\n**User**", "stop_label"),
    ("\n\n**Model C:**", "thread"), ("\nAs follows is", "header"), ("\n\n---", "rule")])
def test_every_spec_stop_is_a_split_boundary(stop: str, kind: str) -> None:
    """Each of the five stops, left in the text by HF generate(stop_strings=),
    is cut and classified (the tail begins with the stop)."""
    sp = T.split_modelc("A reply." + stop)
    assert sp.reply == "A reply." and sp.tail == stop and sp.tail_kind.value == kind


def test_split_is_total_and_length_reads_the_reply() -> None:
    """None/'' never raise; word counts read the trimmed reply, never the tail
    (modelc_reply_words is THE length for model C output)."""
    assert T.split_modelc(None).reply == "" and T.split_modelc("").tail_kind.value == "none"
    raw = "one two three\n\n**User:** four five"
    assert T.modelc_reply(raw) == "one two three"
    assert T.modelc_reply_words(raw) == 3


def test_loom_literal_dream_count_overcounts_the_kept_stop() -> None:
    """count_dream_blocks is a literal ``**User:**`` label
    count; on untrimmed HF text it calls every stopped future a dream, while
    ModelCSplit.dream_blocks (what the loom reports) counts 0. It also misses
    the ``**Claude:**`` visitor. Pinned as-is (kept by design)."""
    stopped = "Indeed.\n\n**User:**"
    assert T.loom_count_dream_blocks(stopped) == 1
    assert T.split_modelc(stopped).dream_blocks == 0
    assert T.loom_count_dream_blocks("x\n\n**Claude:** y") == 0
    assert T.loom_count_dream_blocks("a **User** b **User:** c") == 2


def test_public_fields_carry_the_dream_beside_the_reply() -> None:
    """modelc_public_fields: dream, tail_kind, modelc_dream_blocks, reply_words
    — the UI's dream toggle reads these."""
    sp = T.split_modelc(" Hi there.\n\n**User:** more?")
    assert T.loom_modelc_public_fields(sp) == {
        "dream": "**User:** more?", "tail_kind": "dream",
        "modelc_dream_blocks": 1, "reply_words": 2}


def test_pre_trim_history_is_split_once_and_stops_rendering_an_empty_visitor() -> None:
    """A history saved untrimmed keeps HF's stop; rendered as-is it puts an
    empty visitor block in the NEXT document. retrim_modelc_histories splits
    assistant turns once (idempotent, user turns untouched) and the re-rendered
    document is clean."""
    hist = {"loom": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": "Indeed.\n\n**User:**"}]}
    stale = T.loom_render_modelc_text(hist["loom"] + [{"role": "user", "content": "next"}])
    assert "**User:**\n\n**User:** next" in stale
    assert T.loom_retrim_modelc_histories(hist) == 1
    assert T.loom_retrim_modelc_histories(hist) == 0
    assert hist["loom"][1] == {"role": "assistant", "content": "Indeed.",
                               "dream": "", "tail_kind": "stop_label"}
    clean = T.loom_render_modelc_text(hist["loom"] + [{"role": "user", "content": "next"}])
    assert "**Model C:** Indeed.\n\n**User:** next\n\n**Model C:**" in clean
