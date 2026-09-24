"""format gates: S-01 (template), S-02 (stops), H-01 (pad-eos), policy 6 (empty visitor)."""

from __future__ import annotations

from typing import Any

import pytest

from pleroma.format import chat, modelc
from pleroma.validate import format as fmt

from .conftest import MODELC_STOPS, make_profile


class FakeTok:
    """A chat tokenizer stand-in: renders messages (and the date) to ids."""

    def __init__(self, template: str | None = "tmpl", uses_date: bool = True) -> None:
        self.chat_template = template
        self.uses_date = uses_date

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool,
                            date_string: str) -> list[int]:
        text = "".join(m["content"] for m in messages)
        if self.uses_date:
            text += date_string
        return [ord(c) for c in text]


# ── template (S-01) ──────────────────────────────────────────────────────────


def test_template_modelc_pass(modelc_profile) -> None:
    r = fmt.gate_template(modelc_profile)
    assert r.verdict == "PASS" and "renders to spec" in r.reason


def test_template_modelc_fail_when_renderer_breaks(modelc_profile, monkeypatch) -> None:
    monkeypatch.setattr(modelc, "render", lambda h, t: "no bridge, trailing space **Model C:** ")
    r = fmt.gate_template(modelc_profile)
    assert r.verdict == "FAIL" and "does not render to spec" in r.reason


def test_template_raw_pass() -> None:
    r = fmt.gate_template(make_profile(**{"format.mode": "raw"}))
    assert r.verdict == "PASS"


def test_template_chat_without_tokenizer_is_inconclusive(chat_profile) -> None:
    r = fmt.gate_template(chat_profile)
    assert r.verdict == "INCONCLUSIVE" and "no tokenizer was supplied" in r.reason


def test_template_chat_without_template_fails(chat_profile) -> None:
    r = fmt.gate_template(chat_profile, FakeTok(template=None))
    assert r.verdict == "FAIL" and "carries no chat template" in r.reason


def test_template_chat_with_template_passes_and_names_the_pinned_date(chat_profile) -> None:
    r = fmt.gate_template(chat_profile, FakeTok())
    assert r.verdict == "PASS" and "embeds a date, pinned" in r.reason
    assert r.evidence["date_sensitive"] is True


# ── stops (S-02) ─────────────────────────────────────────────────────────────


def test_stops_pass(modelc_profile) -> None:
    r = fmt.gate_stops(modelc_profile)
    assert r.verdict == "PASS" and r.evidence["configured"] == MODELC_STOPS


def test_stops_fail_when_a_documented_stop_is_missing() -> None:
    p = make_profile(**{"format.stops": MODELC_STOPS[:2]})
    r = fmt.gate_stops(p)
    assert r.verdict == "FAIL" and "omits 3 documented model C stop" in r.reason


def test_stops_fail_when_the_trimmer_does_not_cut_a_configured_stop() -> None:
    p = make_profile(**{"format.stops": MODELC_STOPS + ["\n\n###"]})
    r = fmt.gate_stops(p)
    assert r.verdict == "FAIL" and "not cut by split_modelc" in r.reason
    assert r.evidence["untrimmed"] == ["\n\n###"]


def test_stops_chat_needs_none(chat_profile) -> None:
    assert fmt.gate_stops(chat_profile).verdict == "PASS"


# ── pad-eos (H-01) ───────────────────────────────────────────────────────────


def test_pad_eos_pass(modelc_profile) -> None:
    r = fmt.gate_pad_eos(modelc_profile)
    assert r.verdict == "PASS" and "is not an eos id" in r.reason


def test_pad_eos_fail_when_pad_is_eos() -> None:
    # a profile with pad == eos never validates, so the raw check is what runs
    r = fmt.check_pad_eos(2, [2], MODELC_STOPS)
    assert r.verdict == "FAIL" and "pad id 2 is also an eos id" in r.reason


def test_pad_eos_inconclusive_when_stops_but_no_pad() -> None:
    r = fmt.check_pad_eos(None, [2], MODELC_STOPS)
    assert r.verdict == "INCONCLUSIVE" and "declares no pad id" in r.reason


def test_pad_eos_chat_without_pad_passes(chat_profile) -> None:
    assert fmt.gate_pad_eos(chat_profile).verdict == "PASS"


# ── empty visitor (policy call 6) ────────────────────────────────────────────


def test_empty_visitor_pass(modelc_profile) -> None:
    r = fmt.gate_empty_visitor(modelc_profile)
    assert r.verdict == "PASS" and "refused" in r.reason


@pytest.mark.parametrize("target,attr,replacement", [
    (chat, "build_messages", lambda h, t, s: [{"role": "user", "content": t}]),
    (modelc, "assert_well_formed", lambda doc, expect_user_turn: None),
])
def test_empty_visitor_fail_when_a_layer_accepts_it(modelc_profile, monkeypatch, target,
                                                    attr: str, replacement: Any) -> None:
    monkeypatch.setattr(target, attr, replacement)
    r = fmt.gate_empty_visitor(modelc_profile)
    assert r.verdict == "FAIL" and "empty visitor block is accepted" in r.reason
    assert any(attr in a for a in r.evidence["accepted"])
