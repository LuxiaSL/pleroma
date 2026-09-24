"""The ONLY import surface for tests/contract/map (tests/contract/README.md).

When any of these moves inside ``pleroma/``, this file's imports change and
nothing else in tests/contract/map may. Every name is bound to the thing the
test is about, not to the module it lives in.

Where more than one implementation of a primitive exists (the one-vs-rest
contrastive code), each is imported here on purpose: the contract pins how
EACH behaves on the edge cases, so a consolidation can see exactly what it
changes. Deleting an implementation deletes its binding here, together with
the rows of the per-implementation tables that name it.
"""

from __future__ import annotations

# ── the canonical map + levers (pleroma.map / pleroma.levers) ───────────────
from pleroma.levers.bank import LeverBank
from pleroma.map.loom_map import LoomMap

# ── map identity (pleroma.dose.band / pleroma.map.identity) ─────────────────
from pleroma.dose.band import map_fingerprint
from pleroma.map.identity import MapCode, legacy_part, same_map

# ── contrastive (one-vs-rest) code implementations ──────────────────────────
from pleroma.probe.orchestrator import contrastive_code as contrastive_loom_probe
from pleroma.map.build.cv import fan_center

# ── export constants the served report must agree with ──────────────────────
from pleroma.map.build.export import (
    IDENTITY_TOL as EXPORT_IDENTITY_TOL,
    LAM as EXPORT_LAM,
    RANK as EXPORT_RANK,
    SELFCHECK_FLOOR as EXPORT_SELFCHECK_FLOOR,
    TARGET_DEFAULT as EXPORT_TARGET_DEFAULT,
)

__all__ = [
    "LoomMap", "LeverBank", "map_fingerprint",
    "MapCode", "legacy_part", "same_map",
    "contrastive_loom_probe", "fan_center",
    "EXPORT_IDENTITY_TOL", "EXPORT_LAM", "EXPORT_RANK",
    "EXPORT_SELFCHECK_FLOOR", "EXPORT_TARGET_DEFAULT",
]
