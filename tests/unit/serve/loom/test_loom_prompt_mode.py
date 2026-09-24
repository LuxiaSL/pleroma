"""--prompt-mode {chat,raw,modelc} — the format contract, laptop-runnable.

THE PRIMARY SAFETY REQUIREMENT, and what this file exists to hold: **chat mode
is byte-identical to a plain chat-template render.** Every chat-mode result
was measured through ``apply_chat_template(messages,
add_generation_prompt=True)``, so the identity is asserted on TOKEN IDS against
that literal call — not on a summary, not on a string, and not against another
of our own helpers.

The other two modes are asserted against their written specs:

  raw     ``tok.encode(text)`` with special tokens on. At ZERO history the
          two must agree exactly, because that is the render the 8B raw-mode
          forking measurement (docs/FINDINGS.md section 11) used. The
          multi-turn join is this server's own decision (blank line, no role
          markers; see `pleroma.serve.legacy`'s module docstring for why) and
          is pinned here so it cannot drift silently.

  modelc  the worked example of the spec in `pleroma.format.modelc`,
          compared as a whole string, byte for byte, including the absence of
          a trailing space after ``**Model C:**``.

No torch, no GPU, no server: a tiny ``PreTrainedTokenizerFast`` carrying a
Llama-3-shaped chat template stands in for the model's tokenizer, and the
chat-mode identity claim is a claim about OUR code agreeing with
``apply_chat_template``, which is exactly what the server does.
"""

from __future__ import annotations

import pytest

from pleroma.serve import legacy as ls
from pleroma.format.chat import build_messages

# ── a stand-in tokenizer ─────────────────────────────────────────────────────
# Word-level, so `encode` is deterministic and inspectable, plus a chat
# template shaped like Llama-3's (header/eot wrappers around each turn and a
# generation prompt at the end) so the chat arm exercises a realistic render
# rather than a one-line stub that any bug would also satisfy.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def tok():  # type: ignore[no-untyped-def]
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    vocab = {
        "<bos>": 0, "<eos>": 1, "<unk>": 2,
        "<|start_header_id|>": 3, "<|end_header_id|>": 4, "<|eot_id|>": 5,
    }
    words = [
        "hello", "there", "how", "would", "you", "choose", "to", "speak",
        "system", "user", "assistant", "As", "follows", "is", "a",
        "conversation", "between", "another", "and", "Model", "C.", "Full",
        "with", "C:", "**User:**", "**Model", "the", "next", "thing", "fine",
        "yes", "no", "later", "same",
    ]
    for i, w in enumerate(words):
        vocab[w] = len(vocab) + i
    backing = tokenizers.Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backing.pre_tokenizer = Whitespace()
    # A BOS post-processor, as every Llama-3 tokenizer carries: without one
    # `add_special_tokens=True` would be a no-op here and the raw/modelc BOS
    # assertions below would pass for the wrong reason.
    backing.post_processor = TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backing,
        bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
        chat_template=CHAT_TEMPLATE,
    )


HISTORIES: list[tuple[list[dict[str, str]], str]] = [
    ([], "hello there"),
    ([{"role": "user", "content": "hello"},
      {"role": "assistant", "content": "there"}], "how would you choose"),
    ([{"role": "user", "content": "hello"},
      {"role": "assistant", "content": "there"},
      {"role": "user", "content": "how"},
      {"role": "assistant", "content": "fine"}], "to speak"),
    ([{"role": "user", "content": "yes no"},
      {"role": "assistant", "content": "later same"}], "the next thing"),
]


# ── ★ the byte-identity gate for chat mode ───────────────────────────────────
@pytest.mark.parametrize("history,text", HISTORIES)
def test_chat_mode_is_byte_identical_to_the_pre_flag_render(
    tok, history: list[dict[str, str]], text: str
) -> None:
    """Token ids equal to the EXACT call the server made before --prompt-mode.

    Written out literally rather than factored into a helper: a helper shared
    with the code under test could not detect the thing this asserts.
    """
    messages = build_messages(history, text, None)

    old = tok.apply_chat_template(messages, add_generation_prompt=True)
    old_ids = [int(x) for x in (old["input_ids"] if hasattr(old, "keys") else old)]

    new_ids = ls.render_prompt_ids("chat", tok, messages)

    assert new_ids == old_ids, (
        "chat mode MUST be byte-identical to the pre-2026-09-22 render — "
        "every number in RESULTS was measured through it"
    )
    assert len(new_ids) > 0


def test_chat_mode_identity_holds_for_every_history_at_once(tok) -> None:
    """The same claim as one assertion over the whole set, so a regression in
    a single history cannot hide behind a parametrised partial pass."""
    for history, text in HISTORIES:
        messages = build_messages(history, text, None)
        old = tok.apply_chat_template(messages, add_generation_prompt=True)
        old_ids = [int(x) for x in (old["input_ids"] if hasattr(old, "keys") else old)]
        assert ls.render_prompt_ids("chat", tok, messages) == old_ids


def test_chat_mode_refuses_a_tokenizer_with_no_template(tok) -> None:
    class NoTemplate:
        chat_template = None

    with pytest.raises(RuntimeError, match="chat.*needs a chat template"):
        ls.render_prompt_ids("chat", NoTemplate(), [{"role": "user", "content": "x"}])


# ── raw ──────────────────────────────────────────────────────────────────────
def test_raw_at_zero_history_is_exactly_tok_encode_of_the_text(tok) -> None:
    """★ The regime the .1015 was measured in: one bare prefix, BOS on.

    The measured raw render is ``tok.encode(prompt_text)``; if this drifts,
    the loom's raw cell stops being the cell the analysis measured.
    """
    text = "how would you choose to speak"
    messages = build_messages([], text, None)
    assert ls.render_raw_text(messages) == text
    assert (ls.render_prompt_ids("raw", tok, messages)
            == [int(t) for t in tok.encode(text)])


def test_raw_adds_bos_exactly_once(tok) -> None:
    ids = ls.render_prompt_ids("raw", tok, [{"role": "user", "content": "hello"}])
    assert ids[0] == tok.bos_token_id
    assert ids.count(tok.bos_token_id) == 1


def test_raw_joins_history_with_a_blank_line_and_no_role_markers() -> None:
    messages = build_messages(
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "there"}],
        "how would you choose", None,
    )
    out = ls.render_raw_text(messages)
    assert out == "hello\n\nthere\n\nhow would you choose"
    for marker in ("User:", "Assistant:", "**User:**", "<|"):
        assert marker not in out, (
            "raw mode must carry NO role markers — a role-marked join is the "
            "FEWSHOT cell of the forking analysis, not the raw one"
        )


def test_raw_join_constant_is_a_blank_line() -> None:
    """Pinned: the docstring's justification is about THIS value."""
    assert ls.RAW_TURN_JOIN == "\n\n"


def test_raw_refuses_a_system_message() -> None:
    with pytest.raises(ValueError, match="no system role"):
        ls.render_raw_text([{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"}])


def test_raw_drops_nothing_and_strips_each_turn() -> None:
    out = ls.render_raw_text([{"role": "user", "content": "  a  "},
                              {"role": "assistant", "content": "\nb\n"}])
    assert out == "a\n\nb"


# ── modelc ───────────────────────────────────────────────────────────────────
SPEC_WORKED_EXAMPLE = (
    "As follows is a conversation between another user and Model C.\n"
    "\n"
    "Full conversation with Model C:\n"
    "\n"
    "**User:** hello\n"
    "\n"
    "**Model C:**"
)


def test_modelc_reproduces_the_specs_worked_example_byte_for_byte() -> None:
    """The worked example of the Model C document spec (`pleroma.format.modelc`)."""
    doc = ls.render_modelc_text(build_messages([], "hello", None))
    assert doc == SPEC_WORKED_EXAMPLE


def test_modelc_prompt_ends_at_the_label_with_no_trailing_space() -> None:
    doc = ls.render_modelc_text(build_messages([], "hello", None))
    assert doc.endswith("**Model C:**")
    assert not doc.endswith("**Model C:** ")
    assert not doc.endswith("\n")


def test_modelc_multi_turn_document() -> None:
    doc = ls.render_modelc_text(build_messages(
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "Indeed."}],
        "and then?", None,
    ))
    assert doc == (
        "As follows is a conversation between another user and Model C.\n\n"
        "Full conversation with Model C:\n\n"
        "**User:** hello\n\n"
        "**Model C:** Indeed.\n\n"
        "**User:** and then?\n\n"
        "**Model C:**"
    )


def test_modelc_bare_header_begins_at_the_bridge_line() -> None:
    doc = ls.render_modelc_text(build_messages([], "hello", None),
                                ls.MODELC_HEADERS["bare"])
    assert doc.startswith("Full conversation with Model C:")
    assert "As follows is" not in doc


def test_modelc_computer_opens_variant_omits_the_user_block() -> None:
    doc = ls.render_modelc_text([])
    assert doc == (
        "As follows is a conversation between another user and Model C.\n\n"
        "Full conversation with Model C:\n\n"
        "**Model C:**"
    )
    assert "**User:**" not in doc


def test_modelc_continuing_a_cut_off_turn_ends_at_the_partial_text() -> None:
    doc = ls.render_modelc_text([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I'd say: 'think of"},
    ])
    assert doc.endswith("**Model C:** I'd say: 'think of")
    assert doc.count("**Model C:**") == 1


@pytest.mark.parametrize("name", sorted(ls.MODELC_HEADERS))
def test_every_named_header_resolves_and_is_selectable(name: str) -> None:
    line = ls.resolve_modelc_header(name)
    assert line == ls.MODELC_HEADERS[name]
    doc = ls.render_modelc_text(build_messages([], "hello", None), line)
    assert doc.endswith("**Model C:**")
    if line:
        assert doc.startswith(line + "\n\nFull conversation with Model C:")


def test_a_custom_header_line_passes_through_verbatim() -> None:
    custom = ("As follows is a conversation between another user and Model C "
              "about the reader's sadness.")
    assert ls.resolve_modelc_header(custom) == custom
    assert ls.render_modelc_text(build_messages([], "hi", None),
                                 custom).startswith(custom + "\n\n")


def test_the_default_header_is_the_stranger_header() -> None:
    assert ls.MODELC_DEFAULT_HEADER == "stranger"
    assert (ls.MODELC_HEADERS["stranger"]
            == "As follows is a conversation between another user and Model C.")


def test_modelc_sampling_and_stops_match_the_spec() -> None:
    assert ls.MODELC_TEMPERATURE == 1.0
    assert ls.MODELC_TOP_P == 0.98, "the doc is emphatic: 0.98, not 0.95"
    assert list(ls.MODELC_STOPS) == [
        "\n\n**User:**", "\n\n**User**", "\n\n**Model C:**",
        "\nAs follows is", "\n\n---",
    ]


def test_modelc_refuses_a_system_message() -> None:
    with pytest.raises(ValueError, match="HEADER"):
        ls.render_modelc_text([{"role": "system", "content": "x"},
                               {"role": "user", "content": "hi"}])


def test_modelc_encodes_with_bos(tok) -> None:
    ids = ls.render_prompt_ids("modelc", tok,
                               build_messages([], "hello", None))
    assert ids[0] == tok.bos_token_id
    assert ids.count(tok.bos_token_id) == 1


# ── dream-mode counting ──────────────────────────────────────────────────────
def test_dream_blocks_are_counted_not_filtered() -> None:
    gen = ("Indeed.\n\n**User:** and what of the walk?\n\n"
           "**Model C:** a 'walk' is a 'process'.\n\n**User** again")
    assert ls.count_dream_blocks(gen) == 2
    assert ls.count_dream_blocks("no visitor here") == 0
    assert ls.count_dream_blocks("") == 0
    assert ls.count_dream_blocks(None) == 0  # type: ignore[arg-type]


# ── dispatch / guards ────────────────────────────────────────────────────────
def test_prompt_modes_tuple_is_the_three_documented_modes() -> None:
    assert ls.PROMPT_MODES == ("chat", "raw", "modelc")


def test_the_bins_prompt_floor_still_matches_extract_bins() -> None:
    """The floor is DUPLICATED (extract_bins imports torch at module scope and
    loom_serve must stay laptop-importable), so it has to be pinned to the real
    value or the warning and `bins_feasible` start lying.

    A 19-token raw prompt fails bins for every future, and in chat mode the
    template's ~34 tokens of padding hide the floor.
    """
    pytest.importorskip("torch")
    from pleroma.harvest import bins as extract_bins

    assert ls.BINS_PROMPT_FLOOR == extract_bins.DEFAULT_BINS


def test_render_prompt_text_refuses_chat_and_unknown_modes() -> None:
    with pytest.raises(ValueError, match="no mode-owned string form"):
        ls.render_prompt_text("chat", [{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="unknown prompt mode"):
        ls.render_prompt_text("nonsense", [{"role": "user", "content": "x"}])


def test_render_prompt_ids_refuses_an_unknown_mode(tok) -> None:
    with pytest.raises(ValueError, match="unknown prompt mode"):
        ls.render_prompt_ids("nonsense", tok, [{"role": "user", "content": "x"}])


# ── /info surfaces the mode ──────────────────────────────────────────────────
def _info_kwargs() -> dict[str, object]:
    return {
        "map_path": "m.npz", "map_meta": {}, "sites": [10, 20],
        "branches": list(ls.BRANCHES), "default_k": 6, "future_tokens": 192,
        "detach_wear_default": False, "n_sessions": 0, "harvest_worker": None,
        "restored_sessions": 0, "persistence": {}, "auto_policies": [],
        "loudness_ref": 1.0, "dose_band": None,
    }


def test_info_defaults_to_chat_for_a_caller_that_predates_the_flag() -> None:
    payload = ls.build_info_payload(**_info_kwargs())  # type: ignore[arg-type]
    assert payload["prompt_mode"] == "chat"
    assert payload["prompt_mode_detail"] is None


def test_info_reports_the_modelc_header_in_force() -> None:
    payload = ls.build_info_payload(  # type: ignore[arg-type]
        **_info_kwargs(), prompt_mode="modelc",
        prompt_mode_detail={"modelc_header_name": "claude",
                            "modelc_header": ls.MODELC_HEADERS["claude"],
                            "top_p": 0.98},
    )
    assert payload["prompt_mode"] == "modelc"
    assert payload["prompt_mode_detail"]["modelc_header"] == ls.MODELC_HEADERS["claude"]
    assert payload["prompt_mode_detail"]["top_p"] == 0.98


def test_info_still_carries_every_pre_existing_key() -> None:
    """Additive only — the /info contract test_dose_band.py also guards."""
    before = ls.build_info_payload(**_info_kwargs())  # type: ignore[arg-type]
    after = ls.build_info_payload(**_info_kwargs(), prompt_mode="raw")  # type: ignore[arg-type]
    for k, v in before.items():
        if k.startswith("prompt_mode"):
            continue
        assert k in after and after[k] == v


# ── the CLI ──────────────────────────────────────────────────────────────────
def test_the_ui_surfaces_the_prompt_mode() -> None:
    from pathlib import Path
    ui = (Path(ls.__file__).resolve().parent / "static" / "legacy_ui.html").read_text()
    assert "info.prompt_mode" in ui, (
        "the composer says 'type the next thing you might say' — if the server "
        "is not in chat mode the operator has to be able to see it"
    )
