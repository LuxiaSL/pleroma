"""Network-exposure guards for the servers (loom, harvest worker).

Every server in this package binds through ``require_loopback``, so none of
them can be exposed off-host by a flag.
"""

from __future__ import annotations

import ipaddress


def require_loopback(host: str) -> str:
    """Refuse any bind address that is not loopback. the host may be shared.

    A 0.0.0.0 bind would expose a GPU-backed endpoint with no auth to whatever
    can route to the node. The tunnel (``ssh -L 8765:localhost:8765 <host>``) is
    the supported way to reach this from a laptop, and it needs nothing wider.
    Returns the host to bind (``localhost`` → ``127.0.0.1``); raises ValueError
    for a non-IP name or a non-loopback address.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        if host == "localhost":
            return "127.0.0.1"
        raise ValueError(
            f"--host {host!r} is not an IP address; this server binds loopback only "
            "(use an ssh -L tunnel to reach it from elsewhere)"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            f"--host {host} is not a loopback address. the host may be shared: binding wider "
            "would publish an unauthenticated GPU chat endpoint. Use "
            "'ssh -L 8765:localhost:8765 <host>' instead."
        )
    return host
