"""The attributes every route mixin reads — set by ``ServeContext.__init__``."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pleroma.dose.band import DoseBand
    from pleroma.levers.bank import LeverBank
    from pleroma.serve.api import ApiSettings
    from pleroma.map.loom_map import LoomMap
    from pleroma.serve.harvest_client import HarvestLadder, PoolHealthCache
    from pleroma.serve.persistence import RestoreReport, SessionStore
    from pleroma.serve.runtime import Runtime
    from pleroma.serve.session import State


class RouteBase:
    """Declarations only (no values): the inputs a route reads."""

    args: argparse.Namespace
    loom_map: LoomMap
    map_fp: str
    lever_bank: LeverBank | None
    runtime: Runtime
    state: State
    store: SessionStore
    harvester: HarvestLadder
    pool_health: PoolHealthCache
    pool_urls: list[str]
    resolved_dose_band: DoseBand | None
    report: RestoreReport
    api_settings: ApiSettings
