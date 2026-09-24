"""Sessions at the request boundary: restore-time rehydration, persistence
after a mutating request, POST /restore, and the GET payloads (/state,
/sessions, /loom/progress, /info).

The GET payload builders take a ``create`` flag: the legacy alias creates a
session it is asked about, the versioned route never does.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pleroma.dose.policy import DOSE_POLICY_INFO, DOSE_SCALE_MAX, DOSE_SCALE_MIN
from pleroma.format.prompt import RAW_TURN_JOIN
from pleroma.levers.kind import LEVER_KIND_INFO, worn_lever_kind
from pleroma.serve.api import API_PREFIX, API_VERSION
from pleroma.serve.errors import ApiError, ErrorCode
from pleroma.serve.harvest_client import progress_payload
from pleroma.serve.info import build_info_payload
from pleroma.serve.modelc_fields import retrim_modelc_histories
from pleroma.serve.persistence import (
    SESSION_SCHEMA_VERSION,
    RestoreReport,
    rehydrate_worn_vectors,
)
from pleroma.serve.policies import AUTO_POLICIES_UNAVAILABLE, AUTO_POLICY_INFO
from pleroma.serve.routes._base import RouteBase
from pleroma.serve.schema import schema_sha256
from pleroma.serve.session import BRANCHES, LoomSession, n_assistant_turns

logger = logging.getLogger("loom_serve")


class SessionRoutes(RouteBase):

    def rehydrate(self, session: str, sess: LoomSession) -> None:
        """Best-effort: put the actual levers back under a restored wear.

        The snapshot holds the wear's metadata only; the vectors are recomputed
        from the future's banked signature in the loom dir, through the same
        map call. If the dir is gone (pruned outputs, another box, a different
        map), the wear stays visible but INERT — `worn.active: false` — and the
        operator re-wears. Never raises.
        """
        if self.runtime.prompt_mode == "modelc":
            try:
                n_fixed = retrim_modelc_histories(sess.histories)
                if n_fixed:
                    logger.info("session %r: split %d restored model C "
                                "repl%s into reply + dream (pre-trim snapshot)",
                                session, n_fixed, "y" if n_fixed == 1 else "ies")
            except Exception as exc:  # noqa: BLE001 — never block a restore
                logger.warning("session %r: could not re-trim restored model C "
                               "history (%s: %s)", session, type(exc).__name__, exc)
        if not sess.worn or sess.worn.get("vectors") is not None:
            return
        try:
            vectors = rehydrate_worn_vectors(
                sess.worn, self.loom_map.lever_of, expect_map=self.map_fp,
                input_of=self.loom_map.input_of,
                contrast_of=self.loom_map.contrast_lever_of)
            if vectors.shape != (self.loom_map.n_sites, self.loom_map.hidden):
                raise ValueError(
                    f"rehydrated lever is {vectors.shape}, expected "
                    f"{(self.loom_map.n_sites, self.loom_map.hidden)} — wrong map?"
                )
            sess.worn["vectors"] = vectors
            logger.info("restored wear for session %r: index %s alpha %s "
                        "lever_kind %s from %s",
                        session, sess.worn.get("index"), sess.worn.get("alpha"),
                        worn_lever_kind(sess.worn), sess.worn.get("loom_dir"))
        except Exception as exc:  # noqa: BLE001 — an inert wear beats a crash
            logger.warning(
                "session %r: worn lever could NOT be rehydrated (%s: %s) — "
                "its metadata is restored but nothing is attached; /loom and "
                "/wear again to re-arm it", session, type(exc).__name__, exc,
            )

    def persist(self, session: str) -> None:
        """Snapshot after a mutating request. Cannot fail the request: the
        store swallows its own errors and this is the second net under it.
        A /reset writes the emptied session, so a cleared conversation does
        not come back from the dead on the next restart — and reads the
        session WITHOUT creating it, so /reset still leaves nothing in RAM."""
        try:
            with self.state.lock:
                sess = self.state.sessions.get(session)
            self.store.save(session, sess if sess is not None else LoomSession())
        except Exception as exc:  # noqa: BLE001 — never propagate into a reply
            logger.error("PERSISTENCE: unexpected failure snapshotting %r "
                         "(%s: %s) — serving on", session, type(exc).__name__, exc)

    def do_restore(self, session: str) -> dict[str, Any]:
        """Reload one session from its snapshot, replacing what is in memory.
        Raises ValueError (-> 400) if there is no readable snapshot."""
        sess = self.store.restore_one(session, self.state)
        self.rehydrate(session, sess)
        return {
            "session": session, "restored": True,
            "file": str(self.store.path_for(session)),
            "n_turns": {b: n_assistant_turns(sess.histories[b]) for b in BRANCHES},
            "n_looms": sess.n_looms, "loom_id": sess.loom_id,
            "worn": sess.worn_public(),
            "futures": sess.last_futures,
        }

    def restore_on_start(self) -> RestoreReport:
        """The startup scan (``--restore``, the default) or its refusal
        (``--no-restore``). Stored on ``self.report`` for GET /info."""
        if self.args.restore:
            report = self.store.restore_all(self.state, after=self.rehydrate)
        else:
            report = RestoreReport()
            logger.info("PERSISTENCE: --no-restore — starting with no sessions "
                        "(snapshots in %s are left untouched)", self.store.dir)
        self.report = report
        return report

    # ── the GET payloads ─────────────────────────────────────────────────────

    def state_payload(self, session: str, *, create: bool = True) -> dict[str, Any]:
        """GET /state's body. Creates the session if absent (the legacy
        alias); ``create=False`` (``/api/v1/state``) refuses an
        unknown session with 404 ``no_session`` instead."""
        if create:
            sess = self.state.get(session)
        else:
            found = self.state.peek(session)
            if found is None:
                raise ApiError(ErrorCode.NO_SESSION,
                               f"no session {session!r} (GET /api/v1/state never "
                               "creates one — POST to it first)", status=404)
            sess = found
        return {
            "session": session,
            "histories": sess.histories,
            "worn": sess.worn_public(),
            "loom_id": sess.loom_id,
            "futures": sess.last_futures,
            "n_looms": sess.n_looms,
            "loom_in_progress": sess.progress,
            # The current fan's probe receipt, or
            # null when it has not been probed. The per-candidate
            # gauge scores also ride inside `futures[i].scores.gauge`.
            "probe": sess.last_probe,
            # ★ A RESTORED session is the one case where those two
            # disagree. `last_futures` IS persisted (it is small and
            # jq-readable) and carries its gauge scores across a
            # restart; the probe RECEIPT is not, by the same rule that
            # keeps big arrays out of a snapshot. So after /restore an
            # operator can see gauge numbers with no `resolution`
            # behind them — and "read resolution before ranking" is
            # the standing advice. Say so rather than let the scores
            # look freshly measured.
            "probe_note": (
                "these gauge scores were restored from a snapshot; the "
                "probe receipt (including its `resolution` — whether "
                "the ordering was distinguishable from noise at all) "
                "is NOT persisted. Re-probe before trusting them."
                if sess.last_probe is None and any(
                    (f.get("scores") or {}).get("gauge") is not None
                    for f in sess.last_futures)
                else None),
        }

    def sessions_payload(self) -> dict[str, Any]:
        """GET /sessions' body."""
        rows = self.store.list_sessions(self.state)
        return {
            "sessions": [r.to_json() for r in rows],
            "n_sessions": len(rows),
            "persistence": self.store.to_json(),
        }

    def progress_view(self, session: str, *, create: bool = True) -> dict[str, Any]:
        """GET /loom/progress' body: {stage, done, total}, plus — during a
        harvest — which branches have landed. Additive: a client reading only
        {stage, done, total} is unaffected.
        An empty session tag reads nothing (and creates nothing); with
        ``create=False`` (``/api/v1``) neither does an unknown one."""
        if not session:
            sess = None
        elif create:
            sess = self.state.get(session)
        else:
            sess = self.state.peek(session)
        return {"session": session,
                "progress": (progress_payload(
                    sess.progress, sess.progress_dir,
                    sess.progress_origin) if sess else None)}

    def info_payload(self) -> dict[str, Any]:
        """GET /info's body: ``build_info_payload`` over this server's state,
        plus the ``api`` block."""
        args, rt, loom_map, store = self.args, self.runtime, self.loom_map, self.store
        payload = build_info_payload(
            map_path=str(args.map), map_meta=loom_map.meta,
            sites=loom_map.sites, branches=list(BRANCHES),
            default_k=int(args.default_k), future_tokens=int(args.future_tokens),
            detach_wear_default=bool(args.detach_wear_default),
            n_sessions=len(self.state.sessions), harvest_worker=args.harvest_worker or None,
            harvest_workers=self.pool_urls or None,
            restored_sessions=self.report.restored,
            persistence={
                **store.to_json(),
                "restore_on_start": bool(args.restore),
                "restore_errors": self.report.errors,
                "schema": SESSION_SCHEMA_VERSION,
            },
            # The UI builds its selector from `key` and `needs_wear`;
            # `description` and `status` ride beside them — see
            # AUTO_POLICY_INFO.
            auto_policies=[dict(p) for p in AUTO_POLICY_INFO],
            loudness_ref=float(np.mean(loom_map.norm_ref)),
            dose_band=self.resolved_dose_band,
            norm_ref=loom_map.norm_ref,
            norm_ref_which=loom_map.norm_ref_which,
            norm_ref_alternatives=loom_map.norm_ref_alternatives,
            auto_policies_unavailable=AUTO_POLICIES_UNAVAILABLE,
            dose_policies=DOSE_POLICY_INFO,
            dose_policy_default=str(args.dose_policy_default),
            dose_scale_clamp=(DOSE_SCALE_MIN, DOSE_SCALE_MAX),
            lever_kinds=LEVER_KIND_INFO,
            lever_kind_default=str(args.wear_lever_default),
            prompt_mode=rt.prompt_mode,
            prompt_mode_detail={
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "stop_strings": list(rt.stop_strings) or None,
                "chat_template_present": rt.chat_template_present,
                **({"modelc_header_name": str(args.modelc_header),
                    "modelc_header": rt.modelc_header,
                    "modelc_header_is_bare": not rt.modelc_header}
                   if rt.prompt_mode == "modelc" else {}),
                **({"raw_turn_join": RAW_TURN_JOIN,
                    "raw_history_note": (
                        "turn texts concatenated in order, blank-line "
                        "joined, NO role markers. The .1015 was "
                        "measured at ZERO history, where this is "
                        "byte-identical to tok.encode(text); past "
                        "turn 0 the join is an extrapolation.")}
                   if rt.prompt_mode == "raw" else {}),
            },
        )
        # Which API this server speaks (docs/API.md). Last, so it never
        # shifts the position of an earlier key.
        payload["api"] = {
            "version": API_VERSION, "prefix": API_PREFIX,
            "schema_sha256": schema_sha256(),
            "auth_required": self.api_settings.auth_required,
            "legacy_aliases_open": self.api_settings.legacy_aliases_open,
        }
        return payload
