"""Live session state: one ``LoomSession`` per session tag, in a locked map.

The durable subset of a session is ``pleroma.serve.persistence``'s business.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pleroma.levers.kind import DEFAULT_LEVER_KIND

if TYPE_CHECKING:
    from pleroma.serve.persistence import LoomSnapshot


BRANCHES = ("base", "loom")


# ── session state ──────────────────────────────────────────────────────────────


@dataclass
class LoomSession:
    histories: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {b: [] for b in BRANCHES}
    )
    candidates: list[dict[str, Any]] = field(default_factory=list)  # last /loom
    worn: dict[str, Any] | None = None  # {"index", "alpha", "loom_id", "vectors"}
    n_looms: int = 0
    loom_id: str | None = None          # id of the draw `candidates` belongs to
    last_futures: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, Any] | None = None  # live during /loom
    # The directory the CURRENT harvest is writing,
    # so /loom/progress can report which branches have landed; and, for a
    # probe, gen id -> [candidate, rep]. Live-only, never persisted.
    progress_dir: Path | None = None
    progress_origin: list[list[Any]] | None = None
    reroll_n: dict[str, int] = field(default_factory=dict)  # per-branch nonce
    looms: list[LoomSnapshot] = field(default_factory=list)  # loom metadata log
    last_loom_dir: str | None = None    # on-disk dir of the draw above
    # Predicted dosing: the per-site mean raw map-output
    # norm over the CURRENT draw's harvested candidates — the divisor
    # `dose_policy: "predicted"` uses. It belongs to `candidates` and is
    # replaced wholesale by every /loom, exactly like `loom_id`; a wear can
    # therefore never borrow another fan's mean. None = no fan (fresh session,
    # restored session, or a draw where nothing harvested).
    fan_mean_raw_norms: list[float] | None = None
    # lever_kind: the same divisor for the CONTRAST levers
    # of the same draw, and why the draw has none (fewer than 2 harvested).
    # Same lifetime as `fan_mean_raw_norms`: replaced by every /loom, never
    # persisted.
    fan_mean_raw_norms_contrast: list[float] | None = None
    fan_contrast_note: str | None = None
    # The probe's anchors. Both belong to `candidates`
    # and are replaced wholesale by every /loom, exactly like `loom_id`.
    # `last_loom_text` is the contemplated user turn the current fan was drawn
    # against — /probe checks its own `text` against it, because a probe run
    # on a different prefix would compare replies to futures of another
    # conversation and look perfectly healthy doing it. `last_probe` is the
    # most recent probe receipt, or None when this fan has not been probed.
    # ★ NEITHER IS PERSISTED, deliberately: SessionSnapshot names the fields
    # it stores, so these never reach the on-disk format — a snapshot written
    # after a probe has the same top-level keys as one written without a
    # probe. The consequence is that a RESTORED
    # session shows gauge scores (they ride in the persisted `last_futures`)
    # with no receipt behind them; /state flags exactly that as `probe_note`.
    last_loom_text: str | None = None
    last_probe: dict[str, Any] | None = None
    # ★ The PREFIX the fan was drawn against, fingerprinted. `last_loom_text`
    # alone is not enough: /chat, /undo, /edit, /truncate and /reroll all
    # mutate histories["loom"] WITHOUT clearing `candidates`, so a
    # /loom -> /chat -> /probe with the same contemplated text would rebuild a
    # prompt one turn longer than the one the futures came from, rank probe
    # replies against futures of a different conversation, and look perfectly
    # healthy doing it.
    last_loom_prefix_fp: str | None = None
    # how many probes have run against the CURRENT fan — feeds the probe
    # seed so a re-probe is a genuinely fresh draw, not a replay.
    n_probes: int = 0

    def worn_vectors(self) -> np.ndarray | None:
        """The levers actually attachable RIGHT NOW, or None.

        A session restored from disk carries the worn lever's METADATA with no
        vectors (they are not in the snapshot by design). Every attach site
        goes through here so a restored-but-not-rehydrated wear reads as "not
        wearing" to the model instead of raising mid-request.
        """
        if not self.worn:
            return None
        vectors = self.worn.get("vectors")
        return vectors if isinstance(vectors, np.ndarray) else None

    def worn_public(self) -> dict[str, Any] | None:
        if not self.worn:
            return None
        vectors = self.worn_vectors()
        if vectors is not None:
            norms = [round(float(np.linalg.norm(r)), 3) for r in vectors]
        else:  # restored metadata: report the norms the snapshot remembered
            norms = [round(float(x), 3)
                     for x in (self.worn.get("per_site_norms_at_alpha1") or [])]
        return {
            "index": self.worn["index"], "alpha": self.worn["alpha"],
            "loom_id": self.worn.get("loom_id"),
            "per_site_norms_at_alpha1": norms,
            "code": self.worn.get("code"),
            # False = the metadata survived a
            # restart but the vectors could not be rehydrated, so nothing is
            # actually being attached. /loom + /wear again to re-arm it.
            "active": vectors is not None,
            # The dose receipt, always present. A snapshot without one or an
            # untouched default reads policy="flat", which is what it was —
            # never null, so a client
            # never has to guess whether a wear was scaled.
            "dose": self.worn.get("dose"),
            # Which lever a fan wear put on. A fan wear with no recorded
            # kind is `absolute` (the default lever);
            # a /wear_code wear (it carries `source`) has no fan kind and
            # reports null — its own `source` says what it was.
            "lever_kind": self._public_lever_kind(),
        }

    def _public_lever_kind(self) -> str | None:
        if not self.worn:
            return None
        raw = self.worn.get("lever_kind")
        if raw is not None:
            return str(raw)
        return DEFAULT_LEVER_KIND if self.worn.get("source") is None else None


class State:
    def __init__(self) -> None:
        self.sessions: dict[str, LoomSession] = {}
        self.lock = threading.Lock()

    def get(self, session: str) -> LoomSession:
        with self.lock:
            return self.sessions.setdefault(session, LoomSession())

    def peek(self, session: str) -> LoomSession | None:
        """The session if it exists — never creates one (the /api/v1 GETs)."""
        with self.lock:
            return self.sessions.get(session)

    def put(self, session: str, sess: LoomSession) -> None:
        with self.lock:
            self.sessions[session] = sess

    def snapshot_names(self) -> list[str]:
        with self.lock:
            return sorted(self.sessions)


def n_assistant_turns(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")
