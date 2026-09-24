"""Durable sessions: the snapshot schema, the store, and wear rehydration."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import string
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.map.identity import same_map
from pleroma.errors import CodedError, ErrorCode
from pleroma.dose.policy import DoseScale
from pleroma.levers.kind import LEVER_KINDS, worn_lever_kind
from pleroma.serve.session import BRANCHES, LoomSession, State, n_assistant_turns

logger = logging.getLogger("loom_serve")


# ── durable sessions ───────────────────────────────────────────────────────────
#
# One small JSON file per session under <work-dir>/sessions/, written after
# every mutating request and reloaded at startup. What goes in: the turns, the
# worn lever's METADATA, the loom log. What stays out: every big array (raw
# signatures, hidden states, the 12,288-d levers themselves) — those already
# live in the loom dirs, and a snapshot you cannot read with `jq` is a snapshot
# nobody checks. Persistence is strictly best-effort: a failed write is logged
# loudly and the request still succeeds. The instrument staying up outranks the
# snapshot.

SESSION_SCHEMA_VERSION = 1
SESSIONS_DIRNAME = "sessions"
MAX_PERSISTED_LOOMS = 200  # the loom log is metadata; keep the recent tail
_FILENAME_SAFE = frozenset(string.ascii_letters + string.digits + "-_.")
_MAX_BASENAME_CHARS = 120
_ROLES = ("user", "assistant", "system")


def _iso(ts: float) -> str:
    return (datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def session_basename(session: str) -> str:
    """A session tag -> a safe, injective, single-component file name.

    Percent-escape everything outside [A-Za-z0-9-_.] (so `/`, `\\`, NUL and
    every unicode oddity become %XX), and escape a LEADING dot as well, so no
    tag can produce `.`, `..`, a dotfile, or a path that leaves the sessions
    directory. The escape is reversible and collision-free, which matters:
    two sessions must never land on one file. Over-long tags are truncated
    with a hash of the FULL tag appended, which keeps that property.

    Raises ValueError on a tag that cannot be a file name at all (empty or
    whitespace-only) — callers turn that into a 400, never a traversal.
    """
    if not isinstance(session, str):
        raise ValueError(f"session must be a string, got {type(session).__name__}")
    if not session.strip():
        raise ValueError("session must be a non-empty, non-whitespace string")
    chars: list[str] = []
    for i, ch in enumerate(session):
        if ch in _FILENAME_SAFE and not (i == 0 and ch == "."):
            chars.append(ch)
        else:
            chars.append("".join(f"%{b:02X}" for b in ch.encode("utf-8")))
    name = "".join(chars)
    if len(name) > _MAX_BASENAME_CHARS:
        digest = hashlib.sha256(session.encode("utf-8")).hexdigest()[:12]
        name = f"{name[: _MAX_BASENAME_CHARS - 13]}-{digest}"
    return f"{name}.json"


def _json_default(obj: Any) -> Any:
    """Last-ditch encoder: numpy scalars/arrays that slipped into a snapshot.

    Anything else still raises TypeError — silently stringifying unknown
    objects would let a schema bug write a file that cannot be read back.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _obj(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _opt_str(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string or null")
    return value


def _floats(value: Any, what: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{what} must be a list of numbers")
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a list of numbers ({exc})") from exc


#: Optional string keys an assistant message may carry (model C only): the
#: dreamed remainder and what ended the reply. See pleroma.format.trim.
_HISTORY_EXTRAS: tuple[str, ...] = ("dream", "tail_kind")


def _histories(value: Any) -> dict[str, list[dict[str, str]]]:
    blob = _obj(value, "histories")
    out: dict[str, list[dict[str, str]]] = {b: [] for b in BRANCHES}
    for branch in BRANCHES:
        rows = blob.get(branch, [])
        if not isinstance(rows, list):
            raise ValueError(f"histories[{branch!r}] must be a list")
        msgs: list[dict[str, str]] = []
        for k, row in enumerate(rows):
            msg = _obj(row, f"histories[{branch!r}][{k}]")
            role, content = msg.get("role"), msg.get("content")
            if role not in _ROLES:
                raise ValueError(
                    f"histories[{branch!r}][{k}].role must be one of {_ROLES}"
                )
            if not isinstance(content, str):
                raise ValueError(f"histories[{branch!r}][{k}].content must be a string")
            one: dict[str, str] = {"role": str(role), "content": content}
            # A model C reply's dreamed
            # remainder rides beside its content, never inside it — so the
            # next turn's document is rendered from the trimmed reply alone.
            for extra in _HISTORY_EXTRAS:
                if isinstance(msg.get(extra), str):
                    one[extra] = str(msg[extra])
            msgs.append(one)
        out[branch] = msgs
    return out


# `map_fingerprint` is defined in pleroma.dose.band so that module can
# fingerprint a map on its own, without importing the server — this module
# imports DoseBand/resolve_dose_band from pleroma.dose.band, so the reverse
# import would be circular. Imported above, so `map_fingerprint(...)` here and
# `loom_serve.map_fingerprint(...)` in the flat namespace are the same
# function.


@dataclass(frozen=True)
class WornSnapshot:
    """The worn lever, minus the lever. Everything /wear stores EXCEPT the
    12,288 numbers: index, dose, provenance, the 8-d code, the per-site norms,
    the loom dir the vectors can be recomputed from, and the fingerprint of
    the map that made them."""

    index: int
    alpha: float
    loom_id: str | None = None
    code: list[float] | None = None
    per_site_norms_at_alpha1: list[float] = field(default_factory=list)
    loom_dir: str | None = None
    map_fingerprint: str | None = None
    # Provenance for a /wear_code landmark wear, so a session
    # snapshot still says WHAT was worn when the vectors cannot be rehydrated.
    source: str | None = None
    # The dose receipt — policy, the
    # per-site scale actually applied, the fan mean it came from, and any
    # clamp. Without it a restored `predicted` wear would rehydrate to the
    # FLAT lever and quietly wear a different magnitude than the one the
    # session was steered with. None = a snapshot without the field, which
    # means flat.
    dose: dict[str, Any] | None = None
    # Which lever a fan wear put on, and — for
    # `contrast` — the draw indices of the valid members the contrast was
    # taken over, so a restore recomputes exactly that lever. None = a
    # snapshot without the field, which means `absolute` (see worn_lever_kind).
    lever_kind: str | None = None
    contrast_peers: list[int] | None = None

    @classmethod
    def from_worn(cls, worn: Mapping[str, Any]) -> WornSnapshot:
        vectors = worn.get("vectors")
        if isinstance(vectors, np.ndarray):
            norms = [round(float(np.linalg.norm(r)), 6) for r in vectors]
        else:
            norms = _floats(worn.get("per_site_norms_at_alpha1"),
                            "worn.per_site_norms_at_alpha1")
        code = worn.get("code")
        return cls(
            index=int(worn["index"]),
            alpha=float(worn["alpha"]),
            loom_id=_opt_str(worn.get("loom_id"), "worn.loom_id"),
            code=None if code is None else _floats(code, "worn.code"),
            per_site_norms_at_alpha1=norms,
            loom_dir=_opt_str(worn.get("loom_dir"), "worn.loom_dir"),
            map_fingerprint=_opt_str(worn.get("map_fingerprint"),
                                     "worn.map_fingerprint"),
            source=_opt_str(worn.get("source"), "worn.source"),
            dose=(None if worn.get("dose") is None
                  else _obj(worn.get("dose"), "worn.dose")),
            lever_kind=_opt_lever_kind(worn.get("lever_kind")),
            contrast_peers=_opt_ints(worn.get("contrast_peers"),
                                     "worn.contrast_peers"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index, "alpha": self.alpha, "loom_id": self.loom_id,
            "code": self.code,
            "per_site_norms_at_alpha1": self.per_site_norms_at_alpha1,
            "loom_dir": self.loom_dir,
            "map_fingerprint": self.map_fingerprint,
            "source": self.source,
            "dose": self.dose,
            "lever_kind": self.lever_kind,
            "contrast_peers": self.contrast_peers,
        }

    @classmethod
    def from_json(cls, value: Any) -> WornSnapshot:
        blob = _obj(value, "worn")
        if "index" not in blob or "alpha" not in blob:
            raise ValueError("worn needs 'index' and 'alpha'")
        try:
            index, alpha = int(blob["index"]), float(blob["alpha"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"worn.index/alpha must be numbers ({exc})") from exc
        code = blob.get("code")
        return cls(
            index=index, alpha=alpha,
            loom_id=_opt_str(blob.get("loom_id"), "worn.loom_id"),
            code=None if code is None else _floats(code, "worn.code"),
            per_site_norms_at_alpha1=_floats(
                blob.get("per_site_norms_at_alpha1"),
                "worn.per_site_norms_at_alpha1"),
            loom_dir=_opt_str(blob.get("loom_dir"), "worn.loom_dir"),
            map_fingerprint=_opt_str(blob.get("map_fingerprint"),
                                     "worn.map_fingerprint"),
            source=_opt_str(blob.get("source"), "worn.source"),
            dose=(None if blob.get("dose") is None
                  else _obj(blob.get("dose"), "worn.dose")),
            lever_kind=_opt_lever_kind(blob.get("lever_kind")),
            contrast_peers=_opt_ints(blob.get("contrast_peers"),
                                     "worn.contrast_peers"),
        )

    def to_worn(self) -> dict[str, Any]:
        """The runtime shape, with `vectors=None` — deliberately inert until
        something rehydrates it (see rehydrate_worn_vectors)."""
        return {
            "index": self.index, "alpha": self.alpha, "loom_id": self.loom_id,
            "vectors": None, "code": self.code,
            "per_site_norms_at_alpha1": self.per_site_norms_at_alpha1,
            "loom_dir": self.loom_dir,
            "map_fingerprint": self.map_fingerprint,
            "source": self.source,
            "dose": self.dose,
            "lever_kind": self.lever_kind,
            "contrast_peers": self.contrast_peers,
        }


def _opt_lever_kind(value: Any) -> str | None:
    """A persisted lever_kind: None stays None (an older snapshot = absolute);
    anything else must be a known kind — a restore that guessed would
    rehydrate a different lever under the same name."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in LEVER_KINDS:
        raise ValueError(f"worn.lever_kind {value!r} is not one of {LEVER_KINDS}")
    return value


def _opt_ints(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of integers")
    try:
        return [int(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a list of integers ({exc})") from exc


@dataclass
class LoomSnapshot:
    """One /loom draw, as metadata: what was asked for, what came back in
    aggregate, and which future (if any) ended up worn. The futures' TEXT is
    not here — it is in the loom dir's gen_records, and the live fan (with
    text) rides along separately as `last_futures`."""

    loom_id: str
    k: int
    horizon: int
    created_at: float
    loom_dir: str | None = None
    spread: dict[str, Any] | None = None
    futures: list[dict[str, Any]] = field(default_factory=list)
    auto_selected: dict[str, Any] | None = None
    worn_index: int | None = None
    worn_alpha: float | None = None
    detach_wear: bool = False  # was this draw's wear detached?

    @staticmethod
    def future_meta(future: Mapping[str, Any]) -> dict[str, Any]:
        """The small half of a public future: no text, no arrays."""
        return {
            "index": int(future.get("index", -1)),
            "n_tokens": int(future.get("n_tokens", 0)),
            "harvested": bool(future.get("harvested", False)),
            "note": future.get("note"),
            "scores": dict(future.get("scores") or {}),
            "code": future.get("code"),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "loom_id": self.loom_id, "k": self.k, "horizon": self.horizon,
            "created_at": round(self.created_at, 3),
            "created_at_iso": _iso(self.created_at),
            "loom_dir": self.loom_dir, "spread": self.spread,
            "futures": self.futures, "auto_selected": self.auto_selected,
            "worn_index": self.worn_index, "worn_alpha": self.worn_alpha,
            "detach_wear": self.detach_wear,
        }

    @classmethod
    def from_json(cls, value: Any) -> LoomSnapshot:
        blob = _obj(value, "looms[]")
        loom_id = blob.get("loom_id")
        if not isinstance(loom_id, str) or not loom_id:
            raise ValueError("looms[].loom_id must be a non-empty string")
        spread = blob.get("spread")
        auto = blob.get("auto_selected")
        futures = blob.get("futures") or []
        if not isinstance(futures, list):
            raise ValueError("looms[].futures must be a list")
        worn_index = blob.get("worn_index")
        worn_alpha = blob.get("worn_alpha")
        try:
            return cls(
                loom_id=loom_id,
                k=int(blob.get("k", 0)),
                horizon=int(blob.get("horizon", 0)),
                created_at=float(blob.get("created_at", 0.0)),
                loom_dir=_opt_str(blob.get("loom_dir"), "looms[].loom_dir"),
                spread=None if spread is None else _obj(spread, "looms[].spread"),
                futures=[cls.future_meta(_obj(f, "looms[].futures[]"))
                         for f in futures],
                auto_selected=(None if auto is None
                               else _obj(auto, "looms[].auto_selected")),
                worn_index=None if worn_index is None else int(worn_index),
                worn_alpha=None if worn_alpha is None else float(worn_alpha),
                # absent = a snapshot without detach_wear, which means the
                # default: drawn under wear.
                detach_wear=bool(blob.get("detach_wear", False)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"looms[] entry is malformed: {exc}") from exc


@dataclass
class SessionSnapshot:
    """The durable form of a LoomSession."""

    session: str
    histories: dict[str, list[dict[str, str]]]
    worn: WornSnapshot | None = None
    looms: list[LoomSnapshot] = field(default_factory=list)
    last_futures: list[dict[str, Any]] = field(default_factory=list)
    n_looms: int = 0
    loom_id: str | None = None
    last_loom_dir: str | None = None
    reroll_n: dict[str, int] = field(default_factory=dict)
    saved_at: float = 0.0
    schema: int = SESSION_SCHEMA_VERSION

    @classmethod
    def from_session(cls, session: str, sess: LoomSession,
                     now: float | None = None) -> SessionSnapshot:
        worn = None if not sess.worn else WornSnapshot.from_worn(sess.worn)
        return cls(
            session=session,
            histories={b: [dict(m) for m in sess.histories.get(b, [])]
                       for b in BRANCHES},
            worn=worn,
            looms=list(sess.looms[-MAX_PERSISTED_LOOMS:]),
            last_futures=[dict(f) for f in sess.last_futures],
            n_looms=int(sess.n_looms),
            loom_id=sess.loom_id,
            last_loom_dir=sess.last_loom_dir,
            reroll_n={str(k): int(v) for k, v in sess.reroll_n.items()},
            saved_at=time.time() if now is None else now,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": "loom-session",
            "session": self.session,
            "saved_at": round(self.saved_at, 3),
            "saved_at_iso": _iso(self.saved_at),
            "n_turns": {b: n_assistant_turns(self.histories.get(b, []))
                        for b in BRANCHES},
            "n_looms": self.n_looms,
            "loom_id": self.loom_id,
            "last_loom_dir": self.last_loom_dir,
            "reroll_n": self.reroll_n,
            "worn": None if self.worn is None else self.worn.to_json(),
            "histories": self.histories,
            "looms": [lm.to_json() for lm in self.looms],
            "last_futures": self.last_futures,
        }

    @classmethod
    def from_json(cls, value: Any) -> SessionSnapshot:
        blob = _obj(value, "snapshot")
        schema = blob.get("schema", SESSION_SCHEMA_VERSION)
        if not isinstance(schema, int):
            raise ValueError("schema must be an integer")
        if schema > SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema {schema} is newer than this server "
                f"understands ({SESSION_SCHEMA_VERSION})"
            )
        session = blob.get("session")
        if not isinstance(session, str) or not session.strip():
            raise ValueError("snapshot needs a non-empty 'session'")
        looms_raw = blob.get("looms") or []
        if not isinstance(looms_raw, list):
            raise ValueError("looms must be a list")
        last_futures = blob.get("last_futures") or []
        if not isinstance(last_futures, list):
            raise ValueError("last_futures must be a list")
        reroll_raw = _obj(blob.get("reroll_n") or {}, "reroll_n")
        try:
            reroll = {str(k): int(v) for k, v in reroll_raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reroll_n values must be integers ({exc})") from exc
        worn = blob.get("worn")
        try:
            n_looms = int(blob.get("n_looms", 0))
            saved_at = float(blob.get("saved_at", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"n_looms/saved_at must be numbers ({exc})") from exc
        return cls(
            session=session,
            histories=_histories(blob.get("histories") or {}),
            worn=None if worn is None else WornSnapshot.from_json(worn),
            looms=[LoomSnapshot.from_json(lm) for lm in looms_raw],
            last_futures=[_obj(f, "last_futures[]") for f in last_futures],
            n_looms=n_looms,
            loom_id=_opt_str(blob.get("loom_id"), "loom_id"),
            last_loom_dir=_opt_str(blob.get("last_loom_dir"), "last_loom_dir"),
            reroll_n=reroll,
            saved_at=saved_at,
            schema=schema,
        )

    def to_session(self) -> LoomSession:
        """A live LoomSession. `candidates` stays empty and the worn lever
        carries no vectors — both are arrays, both are recoverable from the
        loom dir, neither belongs in a snapshot."""
        sess = LoomSession()
        sess.histories = {b: [dict(m) for m in self.histories.get(b, [])]
                          for b in BRANCHES}
        sess.worn = None if self.worn is None else self.worn.to_worn()
        sess.looms = list(self.looms)
        sess.last_futures = [dict(f) for f in self.last_futures]
        sess.n_looms = self.n_looms
        sess.loom_id = self.loom_id
        sess.last_loom_dir = self.last_loom_dir
        sess.reroll_n = dict(self.reroll_n)
        return sess


def rehydrate_worn_vectors(
    worn: Mapping[str, Any],
    lever_of: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, Any, Any]],
    expect_map: str | None = None,
    *,
    input_of: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    contrast_of: Callable[[Sequence[np.ndarray], int],
                          tuple[np.ndarray, Any, Any]] | None = None,
) -> np.ndarray:
    """Recompute a restored wear's levers from the future's banked signature.

    The snapshot deliberately holds no arrays; the loom dir does. This runs
    the SAME map call /loom ran (signature + bins row -> norm-matched lever),
    so a rehydrated wear is bit-for-bit the lever that was worn — provided the
    loom dir and the map are the ones it was made with. Raises on anything
    missing or off; the caller keeps serving with an inert wear.

    ★ Predicted dosing: when the snapshot carries a
    `dose` block with a non-flat policy, its per-site scale is re-applied on
    top of the recomputed lever. Without that a session steered under
    `dose_policy: "predicted"` would come back from a restart wearing the
    FLAT magnitude under the same alpha — a silent dose change across a
    restart, which is precisely the class of bug the dose-band doctrine
    exists to prevent. The scale is READ FROM THE SNAPSHOT, never recomputed:
    the fan it came from may be long gone, and the receipt is the authority.

    ★ Lever kind: a `contrast` wear is recomputed as a
    contrast — `input_of` for every member in the persisted `contrast_peers`,
    then `contrast_of(rows, position of index)` — never as the absolute lever
    under the contrast label. A wear with no `lever_kind` is `absolute`,
    rehydrated from its own row alone. A contrast wear with no
    peers, or with no contrast callables supplied, REFUSES (the caller keeps
    serving it inert).
    """
    kind = worn_lever_kind(worn)
    loom_dir = worn.get("loom_dir")
    if not loom_dir:
        raise ValueError("worn has no loom_dir — nothing to rehydrate from")
    made_with = worn.get("map_fingerprint")
    match = same_map(made_with, expect_map) if expect_map is not None else None
    if match == "legacy":
        logger.warning(
            "worn snapshot carries a LEGACY map fingerprint (%s): it matches "
            "the running map's v1 id, but v1 never covered the weights",
            made_with)
    if match == "mismatch":
        raise ValueError(
            f"this wear was made with map {made_with} and the server is "
            f"running {expect_map} — refusing to recompute a different lever "
            "and call it the same wear"
        )
    if expect_map is not None and made_with is None:
        logger.warning(
            "worn snapshot predates map fingerprints — rehydrating against the "
            "CURRENT map (%s); if this session's lever was made elsewhere, "
            "/unwear and pick again", expect_map,
        )
    index = int(worn["index"])
    root = Path(str(loom_dir))
    bins_path = root / "bins.npz"
    bins_cache: list[tuple[np.ndarray, list[int]]] = []

    def banked(j: int) -> tuple[np.ndarray, np.ndarray]:
        # Check order is part of the contract: signature, then bins.
        sig_path = root / "signatures" / f"gen_{j:03d}.npz"
        if not sig_path.exists():
            raise FileNotFoundError(f"no signature at {sig_path}")
        if not bins_cache:
            if not bins_path.exists():
                raise FileNotFoundError(f"no bins at {bins_path}")
            with np.load(bins_path, allow_pickle=True) as bnpz:
                bins_cache.append(
                    (np.asarray(bnpz["features"], dtype=np.float64),
                     [int(x) for x in bnpz["generation_id"]]))
        feats, gids = bins_cache[0]
        with np.load(sig_path, allow_pickle=True) as snpz:
            sig = np.asarray(snpz["features"], dtype=np.float64)
        if j not in gids:
            raise ValueError(f"{bins_path} has no generation_id {j}")
        brow = feats[gids.index(j)]
        if not np.isfinite(brow).all() or not np.isfinite(sig).all():
            raise ValueError("banked features are non-finite")
        return sig, brow

    if kind == "absolute":
        sig, brow = banked(index)
        lever, _raw_norms, _code = lever_of(sig, brow)
    else:
        if input_of is None or contrast_of is None:
            raise ValueError(
                "this wear is lever_kind='contrast' and the caller supplied no "
                "contrast recompute — refusing to rehydrate the absolute lever "
                "under the contrast label")
        peers = worn.get("contrast_peers")
        if not isinstance(peers, (list, tuple)) or len(peers) < 2:
            raise ValueError(
                "a contrast wear needs its persisted contrast_peers (>= 2 draw "
                f"indices) to be recomputed, got {peers!r}")
        peers = [int(j) for j in peers]
        if index not in peers:
            raise ValueError(
                f"worn index {index} is not among its own contrast_peers {peers}")
        rows = [input_of(*banked(j)) for j in peers]
        lever, _raw_norms, _code = contrast_of(rows, peers.index(index))
    out = np.asarray(lever)
    dose_blob = worn.get("dose")
    if dose_blob is not None:
        dose = DoseScale.from_json(dose_blob)
        if dose.policy != "flat":
            if len(dose.scale) != out.shape[0]:
                raise ValueError(
                    f"persisted dose scale has {len(dose.scale)} sites but "
                    f"the rehydrated lever has {out.shape[0]} — refusing to "
                    "re-wear a different magnitude than the receipt records")
            out = dose.apply(out)
    return out


@dataclass(frozen=True)
class SessionListing:
    """One row of GET /sessions."""

    session: str
    file: str
    ok: bool
    in_memory: bool
    n_turns: dict[str, int] = field(default_factory=dict)
    n_messages: int = 0
    n_looms: int = 0
    loom_id: str | None = None
    worn: dict[str, Any] | None = None
    modified_unix: float = 0.0
    modified_iso: str = ""
    size_bytes: int = 0
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "session": self.session, "file": self.file, "ok": self.ok,
            "in_memory": self.in_memory, "n_turns": self.n_turns,
            "n_messages": self.n_messages, "n_looms": self.n_looms,
            "loom_id": self.loom_id, "worn": self.worn,
            "modified_unix": round(self.modified_unix, 3),
            "modified_iso": self.modified_iso,
            "size_bytes": self.size_bytes, "error": self.error,
        }


@dataclass(frozen=True)
class RestoreReport:
    restored: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"restored": self.restored, "failed": self.failed,
                "errors": self.errors}


class SessionStore:
    """Atomic, best-effort, one-file-per-session persistence.

    Every public method that runs inside a live request (`save`) swallows its
    own failures: a server that cannot write must keep answering. The methods
    an operator calls deliberately (`load`, `restore_one`) raise, because a
    /restore that quietly did nothing would be worse than a 400.
    """

    def __init__(self, work_dir: Path, enabled: bool = True) -> None:
        self.dir = Path(work_dir) / SESSIONS_DIRNAME
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._known: set[str] = set()  # files this process loaded or wrote
        self.writes = 0
        self.write_failures = 0
        self.restored_count = 0
        self.failed_count = 0
        self.last_error: str | None = None

    # ── paths ────────────────────────────────────────────────────────────────
    def path_for(self, session: str) -> Path:
        """The one legal path for this tag. Raises ValueError if anything at
        all about it would leave the sessions directory."""
        name = session_basename(session)
        if name in (".json", "..json") or "/" in name or "\\" in name:
            raise ValueError(f"session {session!r} does not map to a safe file name")
        path = self.dir / name
        # Belt and braces: the escape above already makes traversal
        # impossible, so this check should be unreachable — which is exactly
        # when it is worth having.
        if path.parent.resolve(strict=False) != self.dir.resolve(strict=False):
            raise ValueError(f"session {session!r} escapes {self.dir}")
        return path

    def ensure_dir(self) -> bool:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:
            self._note_error(f"cannot create {self.dir}: {exc}")
            return False

    def _note_error(self, msg: str) -> None:
        self.last_error = msg
        logger.error("PERSISTENCE: %s", msg)

    # ── writing ──────────────────────────────────────────────────────────────
    def save(self, session: str, sess: LoomSession) -> bool:
        """Snapshot a session. NEVER raises — returns False and logs loudly."""
        if not self.enabled:
            return False
        try:
            snapshot = SessionSnapshot.from_session(session, sess)
            body = json.dumps(snapshot.to_json(), indent=1, ensure_ascii=False,
                              default=_json_default)
            path = self.path_for(session)
        except Exception as exc:  # noqa: BLE001 — a bad snapshot must not 500
            self.write_failures += 1
            self._note_error(
                f"could not build snapshot for session {session!r} "
                f"({type(exc).__name__}: {exc}) — the live session is unharmed"
            )
            return False
        with self._lock:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._guard_unknown(path, session)
                self._atomic_write(path, body)
                self._known.add(path.name)
                self.writes += 1
                return True
            except Exception as exc:  # noqa: BLE001 — disk full, RO fs, races
                self.write_failures += 1
                self._note_error(
                    f"write failed for session {session!r} -> {path} "
                    f"({type(exc).__name__}: {exc}) — serving on, snapshot stale"
                )
                return False

    def _guard_unknown(self, path: Path, session: str) -> None:
        """Never clobber a snapshot this process has not read.

        Reachable with --no-restore, after a corrupt file was skipped, or if
        two servers share a work-dir. The old bytes get moved aside instead of
        overwritten — losing a conversation is the thing this feature exists
        to stop.
        """
        if path.name in self._known or not path.exists():
            return
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        backup = path.with_name(f"{path.name}.orphaned-{stamp}")
        try:
            os.replace(path, backup)
            logger.warning(
                "PERSISTENCE: %s existed but was never loaded this run "
                "(session %r) — moved to %s rather than overwritten",
                path, session, backup,
            )
        except OSError as exc:
            logger.warning(
                "PERSISTENCE: could not set aside pre-existing %s (%s) — "
                "it is about to be overwritten", path, exc,
            )

    @staticmethod
    def _atomic_write(path: Path, body: str) -> None:
        """temp file in the same dir + fsync + os.replace: a crash mid-write
        leaves either the old snapshot or the new one, never half of one."""
        tmp = path.with_name(
            f"{path.name}.tmp-{os.getpid()}-{threading.get_ident():x}"
        )
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    # ── reading ──────────────────────────────────────────────────────────────
    def load(self, session: str) -> SessionSnapshot:
        """Read one snapshot. RAISES (ValueError / OSError) on any problem."""
        path = self.path_for(session)
        return self._load_path(path)

    def _load_path(self, path: Path) -> SessionSnapshot:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CodedError(ErrorCode.NO_SNAPSHOT, f"no snapshot at {path}") from exc
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return SessionSnapshot.from_json(blob)

    def restore_one(self, session: str, state: State) -> LoomSession:
        """Load one snapshot into memory, replacing whatever is there.
        RAISES on a missing/corrupt file — /restore is an explicit act."""
        snapshot = self.load(session)
        sess = snapshot.to_session()
        state.put(session, sess)
        with self._lock:
            self._known.add(self.path_for(session).name)
        return sess

    def restore_all(
        self, state: State,
        after: Callable[[str, LoomSession], None] | None = None,
    ) -> RestoreReport:
        """Startup scan. NEVER raises: a corrupt file is skipped loudly and
        the other sessions still come back."""
        if not self.dir.exists():
            logger.info("PERSISTENCE: %s does not exist yet — starting empty",
                        self.dir)
            return RestoreReport()
        restored, failed, errors = 0, 0, []
        try:
            paths = sorted(p for p in self.dir.glob("*.json") if p.is_file())
        except OSError as exc:
            self._note_error(f"cannot list {self.dir}: {exc}")
            return RestoreReport(0, 1, [f"{self.dir}: {exc}"])
        for path in paths:
            try:
                snapshot = self._load_path(path)
                sess = snapshot.to_session()
                state.put(snapshot.session, sess)
                with self._lock:
                    self._known.add(path.name)
                restored += 1
                if after is not None:
                    try:
                        after(snapshot.session, sess)
                    except Exception as exc:  # noqa: BLE001 — post-hook is extra
                        logger.warning(
                            "PERSISTENCE: post-restore hook failed for %r "
                            "(%s: %s) — session restored anyway",
                            snapshot.session, type(exc).__name__, exc,
                        )
            except Exception as exc:  # noqa: BLE001 — one bad file, not a crash
                failed += 1
                msg = f"{path.name}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                self._note_error(f"SKIPPING unreadable snapshot {msg}")
        self.restored_count = restored
        self.failed_count = failed
        logger.info("PERSISTENCE: restored %d session(s) from %s (%d skipped)",
                    restored, self.dir, failed)
        return RestoreReport(restored, failed, errors)

    def list_sessions(self, state: State) -> list[SessionListing]:
        """Union of what is on disk and what is in memory. Never raises."""
        rows: dict[str, SessionListing] = {}
        live = set(state.snapshot_names())
        try:
            paths = sorted(p for p in self.dir.glob("*.json") if p.is_file()) \
                if self.dir.exists() else []
        except OSError as exc:
            self._note_error(f"cannot list {self.dir}: {exc}")
            paths = []
        for path in paths:
            try:
                stat = path.stat()
                mtime, size = stat.st_mtime, stat.st_size
            except OSError:
                mtime, size = 0.0, 0
            try:
                snapshot = self._load_path(path)
            except Exception as exc:  # noqa: BLE001 — list the broken ones too
                rows[path.name] = SessionListing(
                    session=path.stem, file=path.name, ok=False, in_memory=False,
                    modified_unix=mtime, modified_iso=_iso(mtime),
                    size_bytes=size, error=f"{type(exc).__name__}: {exc}",
                )
                continue
            rows[snapshot.session] = SessionListing(
                session=snapshot.session, file=path.name, ok=True,
                in_memory=snapshot.session in live,
                n_turns={b: n_assistant_turns(snapshot.histories.get(b, []))
                         for b in BRANCHES},
                n_messages=sum(len(v) for v in snapshot.histories.values()),
                n_looms=snapshot.n_looms, loom_id=snapshot.loom_id,
                worn=None if snapshot.worn is None else snapshot.worn.to_json(),
                modified_unix=mtime, modified_iso=_iso(mtime), size_bytes=size,
            )
        for name in sorted(live - set(rows)):  # in RAM, never written yet
            sess = state.get(name)
            rows[name] = SessionListing(
                session=name, file="", ok=True, in_memory=True,
                n_turns={b: n_assistant_turns(sess.histories.get(b, []))
                         for b in BRANCHES},
                n_messages=sum(len(v) for v in sess.histories.values()),
                n_looms=sess.n_looms, loom_id=sess.loom_id,
                worn=sess.worn_public(), error=None,
            )
        return sorted(rows.values(), key=lambda r: r.session)

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "dir": str(self.dir),
            "writes": self.writes, "write_failures": self.write_failures,
            "restored_sessions": self.restored_count,
            "failed_sessions": self.failed_count,
            "last_error": self.last_error,
        }
