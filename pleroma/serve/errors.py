"""HTTP errors for the loom API: ``ApiError`` and the ``{"error", "code"}`` body.

The mapping from exception to response is the same for GET and POST —
``ValueError``/``KeyError`` -> 400, anything else -> 500 ``"Type: msg"``. The
body carries ``error`` (the operator-facing sentence, verbatim) and ``code``
(``pleroma.errors.ErrorCode``), so a client can branch on the refusal without
regex-matching the sentence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pleroma.errors import CodedError, ErrorCode

__all__ = ["ApiError", "ErrorBody", "ErrorCode", "error_response"]


class ApiError(CodedError):
    """A ``CodedError`` that also names its HTTP status (default 400).

    A ``ValueError``: code that catches ValueError catches it too, and a 400
    ``ApiError`` answers exactly as a bare ValueError would, plus its code.
    """

    def __init__(self, code: ErrorCode, message: str, status: int = 400) -> None:
        super().__init__(code, message)
        self.status: int = int(status)


class ErrorBody(BaseModel):
    """Every non-2xx JSON body the loom API sends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: str = Field(description="the refusal, written for the operator; shown verbatim")
    code: ErrorCode = Field(description="stable machine-readable code (docs/API.md)")


def error_response(exc: BaseException) -> tuple[int, ErrorBody]:
    """(status, body) for an exception escaping a route handler.

    ``ApiError`` -> its own status; any other ``CodedError`` -> 400 with its
    code; ``ValueError``/``KeyError`` -> 400 ``bad_request`` with ``str(exc)``
    verbatim (so a KeyError reads ``"'index'"``); anything else ->
    500 ``internal`` with ``"Type: msg"``.
    """
    if isinstance(exc, ApiError):
        return exc.status, ErrorBody(error=str(exc), code=exc.code)
    if isinstance(exc, CodedError):
        return 400, ErrorBody(error=str(exc), code=exc.code)
    if isinstance(exc, (ValueError, KeyError)):
        return 400, ErrorBody(error=str(exc), code=ErrorCode.BAD_REQUEST)
    return 500, ErrorBody(error=f"{type(exc).__name__}: {exc}", code=ErrorCode.INTERNAL)
