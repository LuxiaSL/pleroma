"""pleroma.net.require_loopback — a server binds only to a loopback address."""

from __future__ import annotations

import pytest

from pleroma.net import require_loopback


def test_loopback_is_required() -> None:
    assert require_loopback("127.0.0.1") == "127.0.0.1"
    assert require_loopback("localhost") == "127.0.0.1"
    assert require_loopback("::1") == "::1"
    for bad in ("0.0.0.0", "10.0.0.5", "192.168.1.9"):
        with pytest.raises(ValueError, match="loopback"):
            require_loopback(bad)
    with pytest.raises(ValueError, match="not an IP address"):
        require_loopback("gpu-box.internal")
