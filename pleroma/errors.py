"""Machine-readable error codes for refusals that a client reacts to.

A refusal on this project is a sentence written for the operator ("dose_policy
='predicted' needs the fan mean raw norms ..."), and those sentences are part
of the contract: they are shown verbatim. A client that had only the
sentence could tell WHICH refusal it got only by regex-matching it, and a
reworded sentence would silently break that match. ``CodedError`` carries a
stable ``code`` beside the unchanged sentence, raised at the very site that
decides to refuse, so a client keys on the code (docs/API.md) and the
sentence stays free to read well.

★ ``CodedError`` IS a ``ValueError`` and ``str(exc)`` is exactly the message,
so every existing ``except ValueError``, every ``pytest.raises(ValueError,
match=...)`` and the server's ValueError -> 400 mapping see no difference.

This module is dependency-free on purpose: domain code (``pleroma.dose``,
``pleroma.levers``) raises these without importing anything HTTP-shaped. The
HTTP status lives on ``pleroma.serve.errors.ApiError``.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Every ``code`` the loom API can put in an error body.

    Additive only (docs/API.md "Versioning"): a code, once published, keeps
    its meaning for the life of ``/api/v1``.
    """

    #: a malformed or refused request with no more specific code (400)
    BAD_REQUEST = "bad_request"
    #: no route at this path, or the resource is absent (/atlas, the UI) (404)
    NOT_FOUND = "not_found"
    #: the body/query carried no ``session`` (400)
    SESSION_REQUIRED = "session_required"
    #: ``GET state`` (API v1): no such session in memory — v1 never creates (404)
    NO_SESSION = "no_session"
    #: ``branch`` is not one of the session's branches (400)
    UNKNOWN_BRANCH = "unknown_branch"
    #: ``loom_id`` names a draw other than the session's latest (400)
    STALE_LOOM = "stale_loom"
    #: /wear or /probe before any /loom produced candidates (400)
    NO_CANDIDATES = "no_candidates"
    #: ``auto.policy: "gauge"`` on a fan that has no probe scores (400)
    GAUGE_NEEDS_PROBE = "gauge_needs_probe"
    #: ``lever_kind`` is not a known kind (400)
    LEVER_KIND_INVALID = "lever_kind_invalid"
    #: ``lever_kind: "contrast"`` on a draw that computed no contrast (400)
    LEVER_KIND_UNAVAILABLE = "lever_kind_unavailable"
    #: ``dose_policy`` is not a known policy (400)
    DOSE_POLICY_INVALID = "dose_policy_invalid"
    #: ``dose_policy: "predicted"`` with no fan mean / no raw norms (400)
    DOSE_POLICY_UNAVAILABLE = "dose_policy_unavailable"
    #: /probe's prefix is not the one the fan was drawn against (400)
    PROBE_PREFIX_MISMATCH = "probe_prefix_mismatch"
    #: /restore of a session with no readable snapshot (400)
    NO_SNAPSHOT = "no_snapshot"
    #: a v1 API route (or a closed legacy alias) without a valid bearer token (401)
    UNAUTHORIZED = "unauthorized"
    #: a browser Origin this server does not allow (403)
    FORBIDDEN_ORIGIN = "forbidden_origin"
    #: anything unexpected; the message names the exception type (500)
    INTERNAL = "internal"


class CodedError(ValueError):
    """A ``ValueError`` whose refusal has a stable machine-readable code.

    ``str(exc)`` is the message, unchanged — only ``.code`` is new.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = ErrorCode(code)
