"""The HTTP layer: the versioned loom API over a ``ServeContext`` (docs/API.md).

It dispatches by the route table in ``pleroma.serve.api``:

* every route answers at ``/api/v1/<route>``; the un-prefixed path is a
  legacy alias (the bundled legacy UI page uses it);
* both methods match the PARSED path, so a query string never changes which
  route answers (``/info?x=1`` is ``/info``);
* GET has the same error mapping as POST (ValueError/KeyError -> 400, else
  500) and every error body is ``{"error", "code"}``;
* JSON carries ``Cache-Control: no-store``; CORS + bearer auth follow
  ``ApiSettings`` (the rule is written out on that class);
* each 200 body is validated against its response model
  (``pleroma.serve.responses.check_response``) and then sent AS BUILT, so the
  bytes are what the handler produced.

The legacy aliases share the versioned routes' handlers, status codes,
error sentences and persistence points; where a v1 route behaves differently
from its alias, ``Route.v1_note`` says how, and ``ApiSettings`` holds the
access rules for both.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

from pleroma.serve import paths
from pleroma.serve.api import (
    LEGACY_ALIAS_PATHS,
    MUTATING_PATHS,
    ROUTE_INDEX,
    ApiSettings,
    split_version,
)
from pleroma.serve.atlas import atlas_report_path
from pleroma.serve.errors import ApiError, ErrorCode, error_response
from pleroma.serve.models import (
    BranchRequest,
    ChatRequest,
    LoomRequest,
    ProbeRequest,
    TruncateRequest,
    WearCodeRequest,
    WearRequest,
)
from pleroma.serve.responses import check_response
from pleroma.serve.schema import build_schema

if TYPE_CHECKING:
    from pleroma.serve.context import ServeContext

logger = logging.getLogger("loom_serve")

__all__ = ["MUTATING_PATHS", "make_handler", "static_file"]

#: Methods and headers a preflight may ask for.
CORS_ALLOW_METHODS = "GET, POST, OPTIONS"
CORS_ALLOW_HEADERS = "Content-Type, Authorization"
CORS_MAX_AGE_S = "600"

_STATIC_TYPES: Mapping[str, str] = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".map": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".ico": "image/x-icon", ".woff": "font/woff", ".woff2": "font/woff2",
    ".ttf": "font/ttf", ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
}


def static_file(root: Path, url_path: str) -> Path | None:
    """The file under ``root`` that ``url_path`` names, or None.

    Refuses (None) anything that could leave ``root``: ``..`` or ``.``
    segments (also percent-encoded), dot-files, NUL, and — after resolving
    symlinks — any path that is not inside ``root``. Never raises."""
    try:
        rel = unquote(url_path).lstrip("/")
        if not rel or "\x00" in rel or "\\" in rel:
            return None
        parts = PurePosixPath(rel).parts
        if any(p in ("..", ".") or p.startswith(".") for p in parts):
            return None
        base = root.resolve()
        cand = (base / rel).resolve()
        if not cand.is_relative_to(base) or not cand.is_file():
            return None
        return cand
    except (OSError, ValueError):
        return None


def _query_session(query: str) -> str:
    q = parse_qs(query)
    return (q.get("session") or [""])[0].strip()


# ── POST handlers: (ctx, session, blob) -> body ──────────────────────────────


def _clear_progress(ctx: ServeContext, session: str) -> None:
    _s = ctx.state.get(session)
    _s.progress = None
    _s.progress_dir = None
    _s.progress_origin = None


def _post_chat(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    req = ChatRequest.from_blob(session, blob)
    return ctx.do_chat(session, req.branch, req.text)


def _post_loom(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    args = ctx.args
    try:
        lr = LoomRequest.from_blob(
            session, blob, default_k=args.default_k,
            future_tokens=args.future_tokens,
            detach_wear_default=bool(args.detach_wear_default))
        return ctx.do_loom(session, lr.text, lr.k, lr.horizon, auto=lr.auto,
                           detach_wear=lr.detach_wear, probe=lr.probe)
    finally:
        _clear_progress(ctx, session)


def _post_probe(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    try:
        pr = ProbeRequest.from_blob(session, blob)
        return ctx.do_probe(session, pr.text, pr.cfg)
    finally:
        _clear_progress(ctx, session)


def _post_wear(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    wr = WearRequest.from_blob(session, blob)
    return ctx.do_wear(session, wr.index, wr.alpha, wr.loom_id, wr.dose_policy,
                       wr.lever_kind)


def _post_wear_code(ctx: ServeContext, session: str,
                    blob: Mapping[str, Any]) -> dict[str, Any]:
    wc = WearCodeRequest.from_blob(session, blob)
    return ctx.do_wear_code(session, wc.alpha, wc.group, wc.axis, wc.pole, wc.rank,
                            wc.random_seed, wc.code, wc.code_kind, wc.dose_policy)


def _post_undo(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    br = BranchRequest.from_blob(session, blob)
    return ctx.do_undo(session, br.branch)


def _post_reroll(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    br = BranchRequest.from_blob(session, blob)
    return ctx.do_reroll(session, br.branch)


def _post_edit(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    req = ChatRequest.from_blob(session, blob)
    return ctx.do_edit(session, req.branch, req.text)


def _post_truncate(ctx: ServeContext, session: str,
                   blob: Mapping[str, Any]) -> dict[str, Any]:
    tr = TruncateRequest.from_blob(session, blob)
    return ctx.do_truncate(session, tr.branch, tr.keep_turns)


def _post_unwear(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    ctx.state.get(session).worn = None
    return {"session": session, "worn": None}


def _post_reset(ctx: ServeContext, session: str, blob: Mapping[str, Any]) -> dict[str, Any]:
    # Scope: clears BOTH branch histories, unwears, and forgets candidates —
    # the session starts over entirely. The snapshot written after this
    # records the CLEARED session, so a reset survives a restart too.
    with ctx.state.lock:
        ctx.state.sessions.pop(session, None)
    return {"session": session, "cleared": True, "worn": None}


def _post_restore(ctx: ServeContext, session: str,
                  blob: Mapping[str, Any]) -> dict[str, Any]:
    return ctx.do_restore(session)


PostHandler = Callable[["ServeContext", str, Mapping[str, Any]], dict[str, Any]]

POST_HANDLERS: Mapping[str, PostHandler] = {
    "/chat": _post_chat, "/loom": _post_loom, "/probe": _post_probe,
    "/wear": _post_wear, "/wear_code": _post_wear_code, "/undo": _post_undo,
    "/reroll": _post_reroll, "/edit": _post_edit, "/truncate": _post_truncate,
    "/unwear": _post_unwear, "/reset": _post_reset, "/restore": _post_restore,
}


def make_handler(ctx: ServeContext,
                 settings: ApiSettings | None = None) -> type[BaseHTTPRequestHandler]:
    """A request-handler class bound to one server's context.

    ``settings`` defaults to ``ctx.api_settings`` (built from the flags)."""
    api: ApiSettings = settings if settings is not None else ctx.api_settings

    class Handler(BaseHTTPRequestHandler):
        #: the allowed Origin to echo on this response, if any
        _cors_origin: str | None = None
        #: extra headers for this response (e.g. WWW-Authenticate on a 401)
        _extra_headers: list[tuple[str, str]]

        def log_message(self, fmt: str, *a: Any) -> None:
            # the request line only — never headers, so never the token
            logger.info("http %s", fmt % a)

        # ── response primitives ─────────────────────────────────────────────

        def _begin_response(self, status: int) -> None:
            self.send_response(status)
            if self._cors_origin is not None:
                self.send_header("Access-Control-Allow-Origin", self._cors_origin)
                self.send_header("Vary", "Origin")
            for k, v in getattr(self, "_extra_headers", []):
                self.send_header(k, v)

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self._begin_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, body: bytes, ctype: str) -> None:
            self._begin_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_exception(self, exc: BaseException) -> None:
            status, err = error_response(exc)
            if status >= 500:
                logger.error("request failed: %s %s", self.command,
                             urlsplit(self.path).path, exc_info=exc)
            if err.code == ErrorCode.UNAUTHORIZED:
                self._extra_headers = [("WWW-Authenticate", "Bearer")]
            self._send(status, err.model_dump(mode="json"))

        def _json(self) -> Any:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # ── access control ──────────────────────────────────────────────────

        def _same_origin(self, origin: str) -> bool:
            host = (self.headers.get("Host") or "").strip().lower()
            try:
                netloc = urlsplit(origin.strip()).netloc.lower()
            except ValueError:
                return False
            return bool(host) and netloc == host

        def _admit(self, route_path: str, versioned: bool, *,
                   preflight: bool = False) -> None:
            """Origin + auth, before anything is read or run. Raises ApiError."""
            self._cors_origin = None
            origin = self.headers.get("Origin")
            if origin is not None:
                if versioned and api.origin_allowed(origin):
                    self._cors_origin = origin.strip()
                elif not self._same_origin(origin):
                    raise ApiError(ErrorCode.FORBIDDEN_ORIGIN,
                                   f"origin {origin!r} is not allowed "
                                   + ("(start the server with --cors-origin to "
                                      "allow it)" if versioned else
                                      "— the un-prefixed routes are same-origin "
                                      "only; use /api/v1"), status=403)
            if preflight:
                return
            needs_token = (versioned or (route_path in LEGACY_ALIAS_PATHS
                                         and not api.legacy_aliases_open))
            if needs_token and api.auth_required and not api.token_ok(
                    self.headers.get("Authorization")):
                raise ApiError(ErrorCode.UNAUTHORIZED,
                               "missing or invalid bearer token "
                               "(Authorization: Bearer <token>)", status=401)

        # ── methods ─────────────────────────────────────────────────────────

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._extra_headers = []
            try:
                route_path, versioned = split_version(urlparse(self.path).path)
                origin = self.headers.get("Origin")
                if not versioned or origin is None or not api.origin_allowed(origin):
                    raise ApiError(ErrorCode.FORBIDDEN_ORIGIN,
                                   "cross-origin requests are allowed only on "
                                   "/api/v1 and only from a --cors-origin",
                                   status=403)
                self._admit(route_path, versioned, preflight=True)
                self._begin_response(204)
                self.send_header("Access-Control-Allow-Methods", CORS_ALLOW_METHODS)
                self.send_header("Access-Control-Allow-Headers", CORS_ALLOW_HEADERS)
                self.send_header("Access-Control-Max-Age", CORS_MAX_AGE_S)
                if (self.headers.get("Access-Control-Request-Private-Network")
                        or "").strip().lower() == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
                self._cors_origin = None
                self._send_exception(exc)

        def do_GET(self) -> None:  # noqa: N802
            self._extra_headers = []
            try:
                parsed = urlparse(self.path)
                route_path, versioned = split_version(parsed.path)
                self._admit(route_path, versioned)
                self._get(route_path, versioned, parsed.query)
            except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
                self._send_exception(exc)

        def _get(self, path: str, versioned: bool, query: str) -> None:
            route = ROUTE_INDEX.get(("GET", path))
            if route is not None and (versioned or route.legacy_alias):
                body = self._get_body(path, versioned, query)
                check_response(route.response, body, route=f"GET {path}")
                self._send(200, body)
                return
            if not versioned and path in ("/", "/ui"):
                self._send_ui()
                return
            if not versioned and api.ui_dir is not None:
                found = static_file(api.ui_dir, path)
                if found is not None:
                    self._send_bytes(200, found.read_bytes(), _STATIC_TYPES.get(
                        found.suffix.lower(), "application/octet-stream"))
                    return
            raise ApiError(ErrorCode.NOT_FOUND, "unknown path", status=404)

        def _get_body(self, path: str, versioned: bool, query: str) -> Any:
            if path == "/info":
                return ctx.info_payload()
            if path == "/state":
                session = _query_session(query)
                if not session:
                    raise ApiError(ErrorCode.SESSION_REQUIRED, "missing session")
                return ctx.state_payload(session, create=not versioned)
            if path == "/sessions":
                return ctx.sessions_payload()
            if path == "/loom/progress":
                return ctx.progress_view(_query_session(query), create=not versioned)
            if path == "/pool":
                # Pool health MEASURED on request. /info reports the launch
                # argument, which goes on saying 2 after both workers die.
                return ctx.pool_health_cached()
            if path == "/atlas":
                # ★ dead path, kept: see pleroma.serve.atlas
                ap = atlas_report_path(ctx.args.atlas_report)
                if not ap.exists():
                    raise ApiError(ErrorCode.NOT_FOUND, "no atlas built yet", status=404)
                return json.loads(ap.read_text())
            if path == "/schema":
                return build_schema()
            raise ApiError(ErrorCode.NOT_FOUND, "unknown path", status=404)

        def _send_ui(self) -> None:
            # Served from disk on EVERY request, deliberately: the UI is being
            # designed iteratively and a page reload should pick up the new
            # file without a server restart.
            if api.ui_dir is not None:
                ui = api.ui_dir / "index.html"
                missing = f"no index.html in --ui-dir {api.ui_dir}"
            else:
                ui = paths.UI_PATH
                missing = "loom_ui.html not present yet — the UI is being designed"
            if not ui.is_file():
                raise ApiError(ErrorCode.NOT_FOUND, missing, status=404)
            self._send_bytes(200, ui.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            self._extra_headers = []
            try:
                route_path, versioned = split_version(urlparse(self.path).path)
                self._admit(route_path, versioned)
                blob = self._json()
                session = str(blob.get("session") or "").strip()
                if not session:
                    raise ApiError(ErrorCode.SESSION_REQUIRED, "missing 'session'")
                route = ROUTE_INDEX.get(("POST", route_path))
                handler = POST_HANDLERS.get(route_path)
                if route is None or handler is None or not (
                        versioned or route.legacy_alias):
                    raise ApiError(ErrorCode.NOT_FOUND, "unknown path", status=404)
                out = handler(ctx, session, blob)
                if route.mutating:
                    ctx.persist(session)
                check_response(route.response, out, route=f"POST {route_path}")
                self._send(200, out)
            except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
                self._send_exception(exc)

    return Handler
