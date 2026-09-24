"""Pure helpers for a /loom draw: bookkeeping, seeds, and the unworn-draw
protocol (detach_wear), probe opt-in, and the scores read off a draw.

Torch-free by design: every function here is testable without a model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


#: ★ ``pleroma.harvest.bins.DEFAULT_BINS``, duplicated as a constant rather
#: than imported because the harvest's bins module is on the torch side of the
#: harvest and this module must stay importable on a laptop
#: (tests/conftest.py's contract). ``bin_edges`` REFUSES a prompt shorter than
#: this — "absence raises, never zero-fills" — so a fan drawn from a short
#: prompt has no bins-B features, no map output and no levers. A bare 19-token
#: raw prompt fails every future's bins; the chat template's ~34 extra tokens
#: keep chat-mode prompts above the floor. See do_loom's warning and
#: `bins_feasible`.
BINS_PROMPT_FLOOR = 20


def next_loom_index(sess_dir: Path, n_looms: int) -> int:
    """The index for a NEW loom dir: never reuses one that exists on disk.

    Derived from the FILESYSTEM, not the in-memory counter alone: /reset
    discards the session object, which rewinds ``n_looms`` to 0, and a counter
    alone would rebuild the identical ``loom_000`` path — ``mkdir(exist_ok=True)``
    would then silently overwrite a banked harvest (last writer wins,
    detectable only by text identity). A re-draw may always proceed; an
    existing dir is never reused. Regex, not a zero-padded three-digit glob,
    so indices past 999 (unpadded) still count.
    """
    idx = int(n_looms)
    if sess_dir.is_dir():
        on_disk = [
            int(m.group(1))
            for p in sess_dir.iterdir()
            if (m := re.fullmatch(r"loom_(\d+)", p.name))
        ]
        if on_disk:
            idx = max(idx, max(on_disk) + 1)
    return idx


def loom_turn_seed(seed: int, session: str, branch: str, turn: int, draw: int = 0) -> int:
    raw = f"loomv0_{seed}_{session}_{branch}_{turn}_{draw}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


# ── the unworn-draw protocol ─────────────────────────────────────────────────
#
# By default /loom draws its K futures UNDER whatever is currently worn — the
# signatures read the already-bent forward pass. `detach_wear` is the opt-in
# alternative: detach the steering for the draw only, on the theory that the
# accrued disposition already lives in the conversation TEXT (it is what the
# model actually said), so the vector is not needed to preserve it while
# previewing — only when the chosen future is re-worn and actually spoken.
#
# The three functions below are the whole decision, kept pure and torch-free
# so they are testable without the model (the house pattern for this module —
# see rehydrate_worn_vectors, map_fingerprint). do_loom wires them
# to the real attach()/draw_batch() closures; nothing here ever touches
# `sess.worn` — that is precisely why detaching a draw cannot leave a session
# unsteered afterwards, successful or not.


def resolve_draw_wear(
    worn: Mapping[str, Any] | None, detach_wear: bool
) -> Mapping[str, Any] | None:
    """What a /loom draw should attach to the model while sampling K futures.

    `worn` is the session's live wear (or None — nothing worn, or a restored
    wear that is still inert). Returns `worn` UNCHANGED — never a copy, never
    mutated — when the draw should be steered, else None. The caller's own
    `worn` reference, used afterwards for scores.cos_to_worn and the
    stay/swerve auto-policies, is untouched either way: this function only
    decides what gets attached for the generate() call, never what the worn
    code IS.
    """
    if worn is None or detach_wear:
        return None
    return worn


def attach_for_draw(
    draw_worn: Mapping[str, Any] | None,
    attach_fn: Callable[[np.ndarray, float], Sequence[Any]],
    draw_fn: Callable[[], Any],
) -> Any:
    """Attach `draw_worn` (if any) for the duration of draw_fn(), then ALWAYS
    detach — including when draw_fn raises. try/finally, not best-effort:
    a failed draw must never silently leave a later turn steered by handles
    nobody removed, or unsteered because a removal was skipped.

    When `draw_worn` is None (nothing worn, or the caller detached it via
    resolve_draw_wear), attach_fn is never called at all — no hook reaches
    the model, so the forward pass is genuinely unbent, not merely a hook
    that happens to write zeros. That is the honest mechanism: absence of a
    hook, not a zeroed alpha. (The server's own startup gate proves a
    zero-vector hook IS byte-identical to no hook at all — so either would
    have been *correct* — but "no hook attached" needs no such proof; it
    can't diverge from the unbent baseline by construction, and it costs
    nothing on a batch generate() call that was never going to use it.)
    """
    handles: Sequence[Any] = (
        attach_fn(draw_worn["vectors"], draw_worn["alpha"])
        if draw_worn is not None else ()
    )
    try:
        return draw_fn()
    finally:
        for h in handles:
            h.remove()


def request_detach_wear(blob: Mapping[str, Any], server_default: bool) -> bool:
    """The effective `detach_wear` for one /loom request: the body's own
    'detach_wear' if it sent one, else the server-level `--detach-wear-default`.
    Fails loudly on a body that sent something other than a boolean — a
    truthy string like "false" silently doing the wrong thing is exactly the
    kind of corruption this whole feature exists to avoid at the OTHER end.
    """
    raw = blob.get("detach_wear")
    if raw is None:
        return bool(server_default)
    if not isinstance(raw, bool):
        raise ValueError(
            f"detach_wear must be a boolean, got {type(raw).__name__}"
        )
    return raw


def prefix_fingerprint(history: Sequence[Mapping[str, str]], text: str) -> str:
    """A stable digest of the exact prefix a draw or a probe is built on.

    /probe must rank its replies against futures of the SAME prefix. Comparing
    the contemplated `text` is not sufficient, because the conversation behind
    it can move (/chat, /undo, /edit, /truncate, /reroll all mutate the loom
    history and none of them clear the fan). This hashes the trimmed history
    AND the contemplated turn, so any change to either is caught.
    """
    h = hashlib.sha256()
    for m in history:
        h.update(str(m.get("role", "")).encode())
        h.update(b"\x00")
        h.update(str(m.get("content", "")).encode())
        h.update(b"\x01")
    h.update(b"\x02")
    h.update(text.encode())
    return h.hexdigest()[:16]


def request_probe(blob: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The effective probe config for one /loom request, or None for no probe.

    Mirrors `request_detach_wear`'s contract deliberately: absent or false
    means OFF (the default turn stays free — the free `loudest` policy already
    captures 43% of an oracle's picking skill, docs/FINDINGS.md §8),
    `true` means "probe with
    defaults", and an object is the probe's settings.

    ★ `{}` is accepted and means DEFAULTS. A bare `if blob.get("probe")`
    would treat it as off, because an empty dict is falsy in Python — and the
    caller would then get a confusing refusal from `auto.policy: "gauge"`
    rather than the probe they asked for.

    Anything else raises. A truthy string like "false", or a stray 0, silently
    doing the wrong thing is exactly the class of bug the detach_wear guard
    exists to prevent — and here the wrong answer costs the operator tens of
    seconds rather than a wrong draw.
    """
    raw = blob.get("probe")
    if raw is None or raw is False:
        return None
    if raw is True:
        return {}
    if isinstance(raw, Mapping):
        return raw
    raise ValueError(
        f"probe must be omitted, a boolean, or an object of probe settings; "
        f"got {type(raw).__name__}")


def drawn_under_wear_field(
    draw_worn: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The `/loom` response's `drawn_under_wear` — what ACTUALLY got attached
    for this draw, never what merely happens to be worn in the session. None
    (falsy) both when nothing is worn and when detach_wear detached it; a
    caller that needs to tell those apart reads the response's `worn` (the
    session's persistent state, unaffected by detach_wear) alongside it.
    """
    if draw_worn is None:
        return None
    return {"index": draw_worn["index"], "alpha": draw_worn["alpha"],
            "loom_id": draw_worn.get("loom_id")}


def cos_to_worn(lever: np.ndarray, worn_vectors: np.ndarray) -> float:
    """Mean per-site cosine similarity between a candidate's lever and the
    currently worn one — scores.cos_to_worn, and what stay/swerve rank on.
    Always compares against `worn_vectors` (the session's actual wear); the
    caller passes that regardless of detach_wear — detaching a draw changes
    what gets attached to the MODEL, never what a future gets compared to.
    """
    pc = [float(np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_)))
          for x_, y_ in zip(lever, worn_vectors)]
    return round(float(np.mean(pc)), 3)
