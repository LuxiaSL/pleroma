"""Every serve test runs the response contract in STRICT mode: a body that
does not fit its published model (or that the model would not round-trip)
fails the request with a 500 instead of the production default of
log-and-send (pleroma/serve/responses.py). A developer's own
PLEROMA_API_TOKEN must not leak into the fixtures either."""

from __future__ import annotations

import pytest

from pleroma.serve.api import TOKEN_ENV
from pleroma.serve.responses import STRICT_ENV


@pytest.fixture(autouse=True)
def _strict_api_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STRICT_ENV, "1")
    monkeypatch.delenv(TOKEN_ENV, raising=False)
