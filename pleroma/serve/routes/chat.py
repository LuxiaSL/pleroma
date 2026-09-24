"""/chat /undo /truncate /reroll /edit — the conversation family.

The routes read ``self.*`` (``state``, ``args``, the runtime's ``render``/
``attach``/``draw``/``decode``/``finish_reply``).
"""

from __future__ import annotations

import time
from typing import Any

from pleroma.format.chat import build_messages, trim_history
from pleroma.serve.draws import loom_turn_seed
from pleroma.serve.errors import ApiError, ErrorCode
from pleroma.serve.routes._base import RouteBase
from pleroma.serve.session import BRANCHES


class ChatRoutes(RouteBase):

    def do_chat(self, session: str, branch: str, text: str) -> dict[str, Any]:
        rt = self.runtime
        if branch not in BRANCHES:
            raise ApiError(ErrorCode.UNKNOWN_BRANCH, f"unknown branch {branch!r}")
        sess = self.state.get(session)
        history, dropped = trim_history(
            list(sess.histories[branch]), int(self.args.max_turns)
        )
        messages = build_messages(history, text, None)
        ids = rt.render(messages)
        turn = sum(1 for m in sess.histories[branch] if m["role"] == "assistant")
        seed = loom_turn_seed(self.args.seed, session, branch, turn)
        vectors = sess.worn_vectors() if branch == "loom" else None
        handles = (rt.attach(vectors, sess.worn["alpha"])
                   if vectors is not None and sess.worn else [])
        t0 = time.time()
        try:
            full = rt.draw(ids, seed, int(self.args.max_new_tokens))
        finally:
            for h in handles:
                h.remove()
        reply, extras = rt.finish_reply(rt.decode(full[int(ids.shape[1]):]))
        sess.histories[branch].append({"role": "user", "content": text})
        sess.histories[branch].append({"role": "assistant", "content": reply,
                                       **extras})
        return {
            **extras,
            "session": session, "branch": branch, "reply": reply,
            "n_turns": turn + 1, "dropped_messages": dropped,
            "worn": sess.worn_public() if branch == "loom" else None,
            "elapsed_s": round(time.time() - t0, 1),
        }

    def do_undo(self, session: str, branch: str) -> dict[str, Any]:
        """Pop the last user/assistant exchange. The popped turn is RETURNED
        (never silently discarded) so the UI can offer redo-by-resend."""
        if branch not in BRANCHES:
            raise ApiError(ErrorCode.UNKNOWN_BRANCH, f"unknown branch {branch!r}")
        h = self.state.get(session).histories[branch]
        if len(h) < 2 or h[-1]["role"] != "assistant" or h[-2]["role"] != "user":
            raise ValueError("nothing to undo — history does not end in a "
                             "user/assistant exchange")
        popped_a = h.pop()
        popped_u = h.pop()
        return {"session": session, "branch": branch,
                "popped": {"user": popped_u["content"],
                           "assistant": popped_a["content"]},
                "n_turns": sum(1 for m in h if m["role"] == "assistant")}

    def do_truncate(self, session: str, branch: str, keep_turns: int) -> dict[str, Any]:
        """Cut a branch back to its first keep_turns exchanges (deep edit's
        first half: truncate to just before turn i, then /chat the new text)."""
        if branch not in BRANCHES:
            raise ApiError(ErrorCode.UNKNOWN_BRANCH, f"unknown branch {branch!r}")
        if keep_turns < 0:
            raise ValueError("keep_turns must be >= 0")
        h = self.state.get(session).histories[branch]
        self.state.get(session).histories[branch] = h[: keep_turns * 2]
        return {"session": session, "branch": branch, "n_turns": keep_turns,
                "dropped_messages": max(len(h) - keep_turns * 2, 0)}

    def do_reroll(self, session: str, branch: str) -> dict[str, Any]:
        """Redraw the last assistant reply: same user turn, fresh seed (a
        per-branch nonce keys it), current wear. The replaced reply is
        returned, not destroyed silently."""
        rt = self.runtime
        if branch not in BRANCHES:
            raise ApiError(ErrorCode.UNKNOWN_BRANCH, f"unknown branch {branch!r}")
        sess = self.state.get(session)
        h = sess.histories[branch]
        if len(h) < 2 or h[-1]["role"] != "assistant":
            raise ValueError("nothing to reroll — no assistant reply at the end")
        old = h.pop()
        user_text = h.pop()["content"]
        sess.reroll_n[branch] = sess.reroll_n.get(branch, 0) + 1
        nonce = 100000 + sess.reroll_n[branch]
        history, dropped = trim_history(list(h), int(self.args.max_turns))
        messages = build_messages(history, user_text, None)
        ids = rt.render(messages)
        turn = sum(1 for m in h if m["role"] == "assistant")
        seed = loom_turn_seed(self.args.seed, session, branch, turn, draw=nonce)
        vectors = sess.worn_vectors() if branch == "loom" else None
        handles = (rt.attach(vectors, sess.worn["alpha"])
                   if vectors is not None and sess.worn else [])
        t0 = time.time()
        try:
            full = rt.draw(ids, seed, int(self.args.max_new_tokens))
        finally:
            for hd in handles:
                hd.remove()
        reply, extras = rt.finish_reply(rt.decode(full[int(ids.shape[1]):]))
        h.append({"role": "user", "content": user_text})
        h.append({"role": "assistant", "content": reply, **extras})
        return {**extras,
                "session": session, "branch": branch, "reply": reply,
                "replaced": old["content"], "reroll_n": sess.reroll_n[branch],
                "n_turns": turn + 1, "dropped_messages": dropped,
                "worn": sess.worn_public() if branch == "loom" else None,
                "elapsed_s": round(time.time() - t0, 1)}

    def do_edit(self, session: str, branch: str, text: str) -> dict[str, Any]:
        """Replace the LAST user turn with new text and regenerate (undo+chat,
        atomically). Deeper edits: /truncate then /chat from the client."""
        undo = self.do_undo(session, branch)
        out = self.do_chat(session, branch, text)
        out["replaced_user"] = undo["popped"]["user"]
        out["replaced_assistant"] = undo["popped"]["assistant"]
        return out
