"""Map identity: which fitted map a lever, code, band or wear was made with.

Fingerprint v1 (legacy, 16 hex) hashed meta, sites, norm_ref and mu_y, but
NOT the weights. v1a maps have mu_y = 0, so two exports with different W and
a repeated meta would share one id. A restored wear or a MEASURED
dose band could not tell them apart.

Fingerprint v2 hashes the stored weight arrays as well. It has the form
``v2-<16 hex>.<legacy 16 hex>``, and carrying the legacy id inside it is the
migration. A record stamped before v2, such as a dose band or a worn
snapshot, still matches its map on the v1 part. `same_map` reports that as
``"legacy"``, meaning the weights went unchecked, so callers can accept it
with a warning rather than refusing every banked artifact.

The weights are hashed AS STORED (dtype, shape, bytes of W / W_U / W_S /
W_Vt). Those stored factors are the code basis, so two exports that
differ only by SVD signs are different maps for code purposes, and they get
different ids.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

V2_PREFIX: str = "v2-"
WEIGHT_KEYS: tuple[str, ...] = ("W", "W_U", "W_S", "W_Vt")

MatchKind = Literal["exact", "legacy", "unstamped", "mismatch"]


def _legacy_hasher(sites: Sequence[int], norm_ref: np.ndarray, mu_y: np.ndarray,
                   meta: Mapping[str, Any]) -> "hashlib._Hash":
    h = hashlib.sha256()
    h.update(json.dumps(dict(meta), sort_keys=True, default=str).encode())
    h.update(np.asarray(list(sites), dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.asarray(norm_ref, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(mu_y, dtype=np.float64)).tobytes())
    return h


def legacy_fingerprint(sites: Sequence[int], norm_ref: np.ndarray, mu_y: np.ndarray,
                       meta: Mapping[str, Any]) -> str:
    """Fingerprint v1: 16 hex over meta/sites/norm_ref/mu_y (no weights)."""
    return _legacy_hasher(sites, norm_ref, mu_y, meta).hexdigest()[:16]


def weights_digest(arrays: Mapping[str, Any]) -> str:
    """sha256 over the stored weight arrays (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    found = False
    for key in WEIGHT_KEYS:
        if key not in arrays:
            continue
        a = np.ascontiguousarray(np.asarray(arrays[key]))
        h.update(key.encode())
        h.update(a.dtype.str.encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
        found = True
    if not found:
        raise KeyError(f"no weight arrays among {sorted(arrays)} (expected any of {WEIGHT_KEYS})")
    return h.hexdigest()


def fingerprint_v2(sites: Sequence[int], norm_ref: np.ndarray, mu_y: np.ndarray,
                   meta: Mapping[str, Any], weights: str) -> str:
    """``v2-<16 hex over everything incl. weights>.<legacy 16 hex>``."""
    h = _legacy_hasher(sites, norm_ref, mu_y, meta)
    h.update(b"weights:")
    h.update(weights.encode())
    return f"{V2_PREFIX}{h.hexdigest()[:16]}.{legacy_fingerprint(sites, norm_ref, mu_y, meta)}"


def legacy_part(fingerprint: str) -> str:
    """The v1 id inside a v2 fingerprint (or the v1 id itself)."""
    if fingerprint.startswith(V2_PREFIX) and "." in fingerprint:
        return fingerprint.rsplit(".", 1)[1]
    return fingerprint


def same_map(stored: str | None, current: str) -> MatchKind:
    """Compare a stamped fingerprint against the loaded map's.

    ``exact``: identical ids. ``legacy``: the record predates v2, and its v1 id
    matches the current map's v1 part (weights unverified). ``unstamped``: the
    record has no fingerprint. ``mismatch``: a different map.
    """
    if not stored:
        return "unstamped"
    if stored == current:
        return "exact"
    if not stored.startswith(V2_PREFIX) and current.startswith(V2_PREFIX) \
            and stored == legacy_part(current):
        return "legacy"
    return "mismatch"


class MapCode(list):  # type: ignore[type-arg]
    """A rank-r code (a list of floats, JSON-serialisable as-is) that remembers
    the fingerprint of the map that produced it. A code from
    a different map is refused, even when the ranks agree."""

    map_fingerprint: str

    def __init__(self, values: Sequence[float], map_fingerprint: str):
        super().__init__(values)
        self.map_fingerprint = map_fingerprint
