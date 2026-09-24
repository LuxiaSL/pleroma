"""Chat message assembly for one turn: the message list and its history trim.

The loom's chat path is its user.
Pure functions over ``[{"role", "content"}]`` lists; no tokenizer, no torch.
"""

from __future__ import annotations


def trim_history(
    messages: list[dict[str, str]], max_turns: int
) -> tuple[list[dict[str, str]], int]:
    """Keep the last ``max_turns`` user/assistant exchanges (0 = unbounded).

    Returns (messages, n_dropped_messages). A long conversation would otherwise
    grow its re-prefill without limit until the context or the GPU says no;
    dropping the OLDEST exchanges keeps the conversation alive and is reported per
    turn rather than done silently. A leading system message, if any, is always
    preserved.
    """
    if max_turns <= 0:
        return list(messages), 0
    system = [m for m in messages[:1] if m.get("role") == "system"]
    body = messages[len(system):]
    keep = max_turns * 2
    if len(body) <= keep:
        return list(messages), 0
    dropped = body[: len(body) - keep]
    return system + body[len(body) - keep:], len(dropped)


def build_messages(
    history: list[dict[str, str]], text: str, system_prompt: str | None
) -> list[dict[str, str]]:
    """The message list handed to the chat template for ONE turn.

    ``history`` is the conversation's own alternating record; ``text`` is the new
    user turn. The system prompt is prepended only when the caller configured
    one — prompts are BARE by design (``gen_forks``' whole premise: anything that
    tells the model how to think pins the basin we are watching it choose), so
    the default is no system message at all. An empty user turn is refused
    (ValueError): the model would continue from nothing the visitor said.
    """
    if not text.strip():
        raise ValueError("refusing to send an empty user turn")
    out: list[dict[str, str]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    out.extend(history)
    out.append({"role": "user", "content": text})
    return out
