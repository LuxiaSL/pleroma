"""The loom API's route table and access settings (docs/API.md).

One ``Route`` per endpoint: its method, canonical path, request/response
models and whether it mutates a session. ``pleroma.serve.http`` dispatches by
this table and ``pleroma.serve.schema`` publishes it, so the served routes and
the published schema cannot drift apart.

Every route answers at ``/api/v1<path>``. The un-prefixed ``<path>`` is a
LEGACY ALIAS (the bundled single-file UI page,
``pleroma/serve/static/legacy_ui.html``, uses it); ``/api/v1/schema`` has no
alias. Where the v1 route behaves differently from its alias,
``Route.v1_note`` says how.

``ApiSettings`` is the access policy — CORS origins, the bearer token, the
UI directory — and the one place the auth rule is decided
(``legacy_aliases_open``).
"""

from __future__ import annotations

import argparse
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr

from pleroma.serve.models import (
    BranchBody,
    ChatBody,
    LoomBody,
    ProbeBody,
    SessionBody,
    TruncateBody,
    WearBody,
    WearCodeBody,
)
from pleroma.serve.responses import (
    ChatResponse,
    EditResponse,
    InfoResponse,
    LoomResponse,
    OpenObject,
    PoolResponse,
    ProbeResponse,
    ProgressResponse,
    RerollResponse,
    ResetResponse,
    RestoreResponse,
    SessionsResponse,
    StateResponse,
    TruncateResponse,
    UndoResponse,
    UnwearResponse,
    WearCodeResponse,
    WearResponse,
)

API_VERSION = "1"
API_PREFIX = "/api/v1"
#: The environment variable the bearer token is read from when --api-token is absent.
TOKEN_ENV = "PLEROMA_API_TOKEN"

# ── persistence, at the request boundary ──────────────────────────────────
#: Canonical paths whose successful POST is followed by a session snapshot.
MUTATING_PATHS = frozenset({
    "/chat", "/loom", "/wear", "/wear_code", "/unwear", "/reset",
    "/undo", "/reroll", "/edit", "/truncate",
    # /probe writes gauge scores onto the fan and may wear the pick.
    "/probe",
})


@dataclass(frozen=True)
class QueryParam:
    name: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class Route:
    method: Literal["GET", "POST"]
    #: canonical path, un-prefixed: "/chat". Served at API_PREFIX + path.
    path: str
    summary: str
    response: type[BaseModel]
    request: type[BaseModel] | None = None
    query: tuple[QueryParam, ...] = ()
    #: False = answers only under API_PREFIX (no un-prefixed alias).
    legacy_alias: bool = True
    deprecated: bool = False
    #: how the v1 route differs from its legacy alias, when it does
    v1_note: str | None = None

    @property
    def mutating(self) -> bool:
        return self.method == "POST" and self.path in MUTATING_PATHS

    @property
    def operation_id(self) -> str:
        words = [w for w in self.path.replace("/", " ").replace("_", " ").split() if w]
        return self.method.lower() + "".join(w.capitalize() for w in words)


_SESSION_Q = QueryParam("session", "the session tag", required=True)

ROUTES: tuple[Route, ...] = (
    # ── GET ──
    Route("GET", "/info", "what this server is: map, prompt mode, rulers, "
          "policies, dose band, API version", InfoResponse),
    Route("GET", "/state", "one session's histories, wear, current fan and probe",
          StateResponse, query=(_SESSION_Q,),
          v1_note="v1 NEVER creates the session: an unknown session is 404 "
                  "`no_session`. The legacy alias creates it."),
    Route("GET", "/sessions", "every session, on disk and in memory", SessionsResponse),
    Route("GET", "/loom/progress", "a running /loom or /probe's stage and harvest",
          ProgressResponse,
          query=(QueryParam("session", "the session tag; empty reads nothing"),),
          v1_note="v1 never creates the session (an unknown one reads "
                  "`progress: null`); the legacy alias creates it."),
    Route("GET", "/pool", "harvest worker health, probed on request (2 s cache)",
          PoolResponse),
    Route("GET", "/atlas", "the atlas report (dead path, kept for a decision)",
          OpenObject, deprecated=True),
    Route("GET", "/schema", "this document: the OpenAPI 3.1 description of /api/v1",
          OpenObject, legacy_alias=False),
    # ── POST ──
    Route("POST", "/chat", "say a turn on a branch (worn on `loom`)", ChatResponse,
          ChatBody),
    Route("POST", "/loom", "draw K futures of a contemplated turn; score, "
          "optionally probe and auto-wear", LoomResponse, LoomBody),
    Route("POST", "/probe", "score the current fan with the probe gauge (optionally wear the pick)",
          ProbeResponse, ProbeBody),
    Route("POST", "/wear", "wear a candidate of the current fan", WearResponse,
          WearBody),
    Route("POST", "/wear_code", "wear a lever from no draw (dead path, kept for a "
          "decision)", WearCodeResponse, WearCodeBody, deprecated=True),
    Route("POST", "/undo", "pop the last exchange (returned, not discarded)",
          UndoResponse, BranchBody),
    Route("POST", "/reroll", "redraw the last reply under the current wear",
          RerollResponse, BranchBody),
    Route("POST", "/edit", "replace the last user turn and regenerate", EditResponse,
          ChatBody),
    Route("POST", "/truncate", "cut a branch back to its first keep_turns exchanges",
          TruncateResponse, TruncateBody),
    Route("POST", "/unwear", "take the wear off", UnwearResponse, SessionBody),
    Route("POST", "/reset", "forget the session entirely (the snapshot records "
          "the cleared session)", ResetResponse, SessionBody),
    Route("POST", "/restore", "reload one session from its snapshot", RestoreResponse,
          SessionBody),
)

ROUTE_INDEX: Mapping[tuple[str, str], Route] = {(r.method, r.path): r for r in ROUTES}
#: canonical paths that have an un-prefixed alias (any method)
LEGACY_ALIAS_PATHS = frozenset(r.path for r in ROUTES if r.legacy_alias)


def split_version(path: str) -> tuple[str, bool]:
    """``/api/v1/chat`` -> (``/chat``, True); ``/chat`` -> (``/chat``, False).

    ``/api/v1`` and ``/api/v1/`` map to ``/`` (versioned), which is no route."""
    if path == API_PREFIX:
        return "/", True
    if path.startswith(API_PREFIX + "/"):
        return path[len(API_PREFIX):] or "/", True
    return path, False


# ── access settings ──────────────────────────────────────────────────────────


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]``, lowercased, no path/slash. Raises ValueError
    for anything else — ``*`` included: a wildcard with a bearer token is the
    one CORS configuration that must never be expressible here."""
    raw = str(origin).strip()
    if raw == "*":
        raise ValueError("--cors-origin '*' is refused: name each origin "
                         "(e.g. http://localhost:5173)")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"--cors-origin {origin!r} is not an origin "
                         "(expected scheme://host[:port])")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(f"--cors-origin {origin!r} has a path; an origin is "
                         "scheme://host[:port] only")
    return f"{parts.scheme}://{parts.netloc}".lower()


class ApiSettings(BaseModel):
    """Who may call the API, from where. Built once per server.

    THE RULE (docs/API.md "Auth and CORS"):

    * ``/api/v1/*`` needs ``Authorization: Bearer <token>`` whenever a token is
      configured (``--api-token`` or ``$PLEROMA_API_TOKEN``); CORS headers are
      sent only to a configured ``--cors-origin``.
    * The legacy un-prefixed aliases never send CORS headers and refuse a
      foreign browser Origin (403 ``forbidden_origin``). They stay OPEN (no
      token) only while the server is reachable from nowhere but its own
      origin: no token, or a token on a loopback-only server with no CORS
      origin. Once the server is opened
      to other origins (``--cors-origin``) or other hosts (``--allow-lan``), a
      configured token closes the aliases too — otherwise they would be an
      unauthenticated side door around it.
    * The served UI (``/``, ``/ui``, ``--ui-dir`` assets) never needs a token:
      a browser navigation cannot send one, and it is static.
    """

    model_config = ConfigDict(frozen=True)

    cors_origins: frozenset[str] = frozenset()
    token: SecretStr | None = None
    ui_dir: Path | None = None
    allow_lan: bool = False

    @property
    def auth_required(self) -> bool:
        return self.token is not None

    @property
    def legacy_aliases_open(self) -> bool:
        return self.token is None or (not self.cors_origins and not self.allow_lan)

    def origin_allowed(self, origin: str) -> bool:
        try:
            return normalize_origin(origin) in self.cors_origins
        except ValueError:
            return False

    def token_ok(self, authorization: str | None) -> bool:
        """Constant-time bearer check. True when no token is configured."""
        if self.token is None:
            return True
        if not authorization:
            return False
        scheme, _, value = authorization.strip().partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            return False
        return hmac.compare_digest(value.strip().encode(),
                                   self.token.get_secret_value().encode())

    @classmethod
    def from_args(cls, args: argparse.Namespace,
                  environ: Mapping[str, str] | None = None) -> ApiSettings:
        """From the parsed flags (absent flags = defaults, so a Namespace built
        without these flags still works) and the environment. Raises
        ValueError for a malformed ``--cors-origin``."""
        env = os.environ if environ is None else environ
        origins = frozenset(normalize_origin(o)
                            for o in (getattr(args, "cors_origin", None) or []))
        raw_token = getattr(args, "api_token", None) or env.get(TOKEN_ENV) or ""
        token = SecretStr(raw_token.strip()) if raw_token.strip() else None
        ui_dir = getattr(args, "ui_dir", None)
        return cls(cors_origins=origins, token=token,
                   ui_dir=None if ui_dir is None else Path(ui_dir),
                   allow_lan=bool(getattr(args, "allow_lan", False)))
