"""``ServeContext``: the one object every route reads its inputs from.

A route needs ``args``, ``state``, ``store``, ``loom_map``, ``map_fp``, the
model runtime and the harvest ladder; those are the fields below, set once at
startup by ``pleroma.serve.app.main``. The routes are methods, grouped by
family in ``pleroma.serve.routes.*`` mixins, so a test can build a context
with fakes and call a route directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pleroma.dose.band import DoseBand
from pleroma.levers.bank import LeverBank
from pleroma.map.loom_map import LoomMap
from pleroma.serve.api import ApiSettings
from pleroma.serve.harvest_client import HarvestLadder, PoolHealthCache
from pleroma.serve.persistence import RestoreReport, SessionStore
from pleroma.serve.routes.chat import ChatRoutes
from pleroma.serve.routes.loom import LoomRoutes
from pleroma.serve.routes.probe import ProbeRoutes
from pleroma.serve.routes.sessions import SessionRoutes
from pleroma.serve.routes.wear import WearRoutes
from pleroma.serve.runtime import Runtime
from pleroma.serve.session import State


class ServeContext(ChatRoutes, LoomRoutes, ProbeRoutes, WearRoutes, SessionRoutes):
    """Everything a request can reach. One per server process."""

    def __init__(self, *, args: argparse.Namespace, loom_map: LoomMap,
                 map_fp: str, lever_bank: LeverBank | None, runtime: Runtime,
                 state: State, store: SessionStore,
                 harvester: HarvestLadder, pool_health: PoolHealthCache,
                 pool_urls: Sequence[str],
                 resolved_dose_band: DoseBand | None,
                 report: RestoreReport | None = None,
                 api_settings: ApiSettings | None = None) -> None:
        self.args = args
        self.loom_map = loom_map
        self.map_fp = map_fp
        self.lever_bank = lever_bank
        self.runtime = runtime
        self.state = state
        self.store = store
        self.harvester = harvester
        self.pool_health = pool_health
        self.pool_urls = list(pool_urls)
        self.resolved_dose_band = resolved_dose_band
        #: set by ``restore_on_start`` (the startup scan), read by GET /info.
        self.report = report if report is not None else RestoreReport()
        #: who may call the API, from where (docs/API.md). Absent -> from
        #: ``args`` + ``$PLEROMA_API_TOKEN``.
        self.api_settings = (api_settings if api_settings is not None
                             else ApiSettings.from_args(args))

    def pool_health_cached(self, max_age_s: float = 2.0) -> dict:
        """GET /pool's body, probed on request (2 s cache)."""
        return self.pool_health.get(max_age_s)
