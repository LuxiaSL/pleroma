"""pleroma.format.chat — one turn's message list and its history trim."""

from __future__ import annotations

import pytest

from pleroma.format.chat import build_messages, trim_history


def test_build_messages_appends_the_user_turn_and_honours_the_system_prompt() -> None:
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    plain = build_messages(history, "c", None)
    assert [m["role"] for m in plain] == ["user", "assistant", "user"]
    assert plain[-1]["content"] == "c"
    # Prompts are bare by design, so a system message appears only if asked for.
    withsys = build_messages(history, "c", "be brief")
    assert withsys[0] == {"role": "system", "content": "be brief"}
    with pytest.raises(ValueError, match="empty user turn"):
        build_messages(history, "   ", None)
    # the caller's history is not mutated
    assert len(history) == 2


def test_trim_history_drops_the_oldest_and_keeps_the_system_message() -> None:
    messages = [{"role": "system", "content": "s"}]
    for i in range(5):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    kept, dropped = trim_history(messages, 2)
    assert dropped == 6
    assert kept[0]["role"] == "system"
    assert [m["content"] for m in kept[1:]] == ["u3", "a3", "u4", "a4"]
    # 0 = unbounded, and a short history is never touched.
    assert trim_history(messages, 0) == (messages, 0)
    assert trim_history(messages, 99) == (messages, 0)


def test_trim_history_without_a_system_message() -> None:
    messages = [{"role": r, "content": str(i)} for i, r in
                enumerate(["user", "assistant"] * 3)]
    kept, dropped = trim_history(messages, 1)
    assert dropped == 4 and [m["content"] for m in kept] == ["4", "5"]
