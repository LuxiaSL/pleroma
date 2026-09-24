"""``main()``: the loom server's startup, in a fixed order.

The sequence is parse, pool, bind check, map, bank, dose band, torch + model,
gate, sessions, serve. Exit codes: 2 for a bad pool list, a malformed API
setting, a chat-mode tokenizer with no chat template or a model whose hidden
size is not the map's; 3 when the byte-identity gate fails. The routes live
on ``ServeContext``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from http.server import ThreadingHTTPServer

from pleroma.harvest import pool as harvest_pool
from pleroma.dose.band import (
    DoseBand,
    ResidualScale,
    default_sidecar_path,
    load_dose_band_file,
    load_residual_scale_file,
    resolve_dose_band,
)
from pleroma.levers.bank import LeverBank
from pleroma.map.loom_map import LoomMap
from pleroma.net import require_loopback
from pleroma.serve.api import API_PREFIX, ApiSettings
from pleroma.serve.args import parse_args
from pleroma.serve.context import ServeContext
from pleroma.serve.harvest_client import HarvestLadder, PoolHealthCache
from pleroma.serve.http import make_handler
from pleroma.serve.paths import REPO_ROOT
from pleroma.serve.persistence import SessionStore
from pleroma.serve.runtime import load_runtime
from pleroma.serve.session import State

logger = logging.getLogger("loom_serve")

#: ★ The thread-pool pins. The harvest subprocesses the loom spawns
#: (``pleroma.harvest.bins``) inherit them and do not pin themselves.
#: ``pleroma.serve.legacy`` sets them at import; ``main()`` re-asserts them
#: (``setdefault``: idempotent) before its first torch import, so an entry
#: point that skips the shim is pinned too.
THREAD_POOL_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")


def pin_thread_pools() -> None:
    for _v in THREAD_POOL_ENV:
        os.environ.setdefault(_v, "1")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the loom server. ``argv`` defaults to ``sys.argv[1:]``."""
    pin_thread_pools()
    args = parse_args(argv)

    # ── the harvest pool, resolved ONCE at startup (3-4x faster, docs/FINDINGS.md §13) ─
    # Parsed EAGERLY and fatally: a malformed --harvest-workers would otherwise
    # surface as "the pool silently never engaged", which looks exactly like
    # "the speedup did not reproduce" and is the one failure mode that would
    # waste a measurement session.
    pool_urls: list[str] = []
    if args.harvest_workers:
        try:
            pool_urls = harvest_pool.parse_worker_list(args.harvest_workers)
        except ValueError as exc:
            logger.error("--harvest-workers: %s", exc)
            return 2
        if len(pool_urls) < 2:
            logger.error(
                "--harvest-workers needs >= 2 entries (got %d); a one-worker pool is "
                "just --harvest-worker, which is a path that already exists",
                len(pool_urls),
            )
            return 2
        if not args.harvest_worker:
            # Not fatal, but it removes a rung from the fallback ladder, and a
            # silently shorter ladder lets one failed harvest take a draw down.
            logger.warning(
                "--harvest-workers is set but --harvest-worker is NOT; if the pool "
                "fails, /loom degrades straight to the two-subprocess pipeline "
                "(~30-60s) instead of one warm worker. Pass --harvest-worker %s too.",
                pool_urls[0],
            )
        logger.info("harvest POOL: %d workers %s (falls back to %s, then subprocess)",
                    len(pool_urls), pool_urls, args.harvest_worker or "subprocess")

    # ── the API's access policy (docs/API.md), resolved EAGERLY: a malformed
    # --cors-origin must fail at startup, not on the first browser request.
    try:
        api_settings = ApiSettings.from_args(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    log_api_settings(api_settings)

    # GET /pool: every configured worker, probed, 2 s cache.
    pool_health = PoolHealthCache(pool_urls, args.harvest_worker)

    if args.allow_lan:
        host = str(args.host)
        logger.warning("binding %s — visible to the whole network, %s "
                       "(--allow-lan)", host,
                       "bearer token required" if api_settings.auth_required
                       else "NO AUTH")
    else:
        host = require_loopback(args.host)
    loom_map = LoomMap(args.map, args.discriminants)
    # v2 (covers the weights; the legacy v1 id rides inside it).
    map_fp = loom_map.fingerprint
    # The optional banked-lever source for POST /wear_code.
    # Loaded EAGERLY and fatally — a bank whose sites disagree with the map would
    # inject at the wrong layers, and finding that out on request #200 of an
    # experiment is worse than not starting.
    lever_bank = None if args.lever_bank is None else LeverBank(args.lever_bank,
                                                                loom_map)
    logger.info("map fingerprint %s (stamped on every wear; a restored wear "
                "made with another map is refused, not silently recomputed)",
                map_fp)
    logger.info("map: %d-d in, sites %s, rank %s, alpha=1 norms %s",
                loom_map.mu_in.size, loom_map.sites, loom_map.meta["rank"],
                [round(float(x), 3) for x in loom_map.norm_ref])

    resolved_dose_band = resolve_startup_dose_band(args, loom_map, map_fp)

    runtime = load_runtime(args, loom_map)
    if isinstance(runtime, int):
        return runtime
    if not runtime.gate(int(args.seed), int(args.selfcheck_tokens), loom_map.n_sites):
        return 3

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state = State()

    # ── durable sessions ──────────────────────────────────────────────────────
    store = SessionStore(args.work_dir, enabled=bool(args.persist))
    if store.enabled and not store.ensure_dir():
        logger.warning("PERSISTENCE DISABLED for this run — %s is not writable; "
                       "sessions will be lost on restart, as they always were",
                       store.dir)
        store.enabled = False

    ctx = ServeContext(
        args=args, loom_map=loom_map, map_fp=map_fp, lever_bank=lever_bank,
        runtime=runtime, state=state, store=store,
        harvester=HarvestLadder.from_args(args, pool_urls, sys.executable, REPO_ROOT),
        pool_health=pool_health, pool_urls=pool_urls,
        resolved_dose_band=resolved_dose_band, api_settings=api_settings)
    ctx.restore_on_start()

    server = ThreadingHTTPServer((host, int(args.port)), make_handler(ctx))
    logger.info("LOOM v0 listening on %s:%d — tunnel: ssh -L %d:localhost:%d <host>",
                host, args.port, args.port, args.port)
    server.serve_forever()
    return 0


def log_api_settings(api: ApiSettings) -> None:
    """One startup line per access decision. Never logs the token itself."""
    logger.info("API %s: auth %s; CORS origins %s; legacy un-prefixed routes %s; "
                "UI from %s", API_PREFIX,
                "REQUIRED (bearer token set)" if api.auth_required else "off",
                sorted(api.cors_origins) or "none (same-origin only)",
                "open (same-origin)" if api.legacy_aliases_open
                else "CLOSED without the token",
                api.ui_dir or "the bundled legacy UI")
    if not api.auth_required and (api.cors_origins or api.allow_lan):
        logger.warning(
            "API: %s with NO TOKEN — anything that can reach this port%s can "
            "drive the loom. Set PLEROMA_API_TOKEN (docs/API.md).",
            "--cors-origin" if api.cors_origins else "--allow-lan",
            " or any allowed origin" if api.cors_origins else "")


def resolve_startup_dose_band(args: argparse.Namespace, loom_map: LoomMap,
                              map_fp: str) -> DoseBand | None:
    """The dose band, resolved ONCE at startup (measured > derived > none).

    The server loads exactly one map for its whole run, so this is recomputed
    fresh every restart rather than cached anywhere. See ``pleroma.dose.band``'s
    module docstring for the tiers and why the resolution order never silently
    substitutes a different map's numbers. Raises (through the loaders) when an
    explicitly passed band, reference or residual-scale file is unreadable;
    returns None for tier ``none``.
    """
    dose_band_path = args.dose_band or default_sidecar_path(args.map)
    measured_band: DoseBand | None = None
    if args.dose_band is not None:
        measured_band = load_dose_band_file(dose_band_path)  # explicit: fail loudly
    elif dose_band_path.exists():
        measured_band = load_dose_band_file(dose_band_path)  # auto-discovered
    reference_band: DoseBand | None = (
        load_dose_band_file(args.dose_band_reference) if args.dose_band_reference else None
    )
    # The cross-model error bar. Explicit path only, fail
    # loudly — same doctrine as --dose-band. It cannot change a boundary, but a
    # malformed one would silently drop the one number that says how far a
    # cross-model rescale is reaching, and a band that has lost its error bar
    # looks exactly like one that never needed it.
    residual_scale: ResidualScale | None = (
        load_residual_scale_file(args.dose_band_residual_scale)
        if args.dose_band_residual_scale else None
    )
    resolved_dose_band = resolve_dose_band(
        measured=measured_band, reference=reference_band,
        target_norm_ref=loom_map.norm_ref, target_map_fingerprint=map_fp,
        residual_scale=residual_scale,
    )
    if residual_scale is not None and resolved_dose_band is not None:
        cross = (resolved_dose_band.derivation or {}).get("cross_model")
        if cross is None:
            logger.warning(
                "dose band: --dose-band-residual-scale was given but the resolved "
                "band is tier=%s and derived nothing, so no cross-model bound was "
                "attached — the flag had no effect", resolved_dose_band.tier)
        else:
            logger.info(
                "dose band cross-model bound (%s -> %s): served ABSOLUTE x%.4f, "
                "competing RELATIVE x%.4f, disagreement %.4fx, served side is %s",
                cross["residual_scale"]["reference_model"],
                cross["residual_scale"]["target_model"],
                cross["absolute"]["scale"], cross["relative"]["scale"],
                cross["disagreement"],
                "CONSERVATIVE" if cross["served_is_conservative"]
                else "★ NOT the conservative one")
    if resolved_dose_band is None:
        logger.warning("dose band: tier=none — UNCALIBRATED (no %s, and no "
                       "--dose-band-reference given). GET /info will report this "
                       "honestly; do not assume the old map's numbers apply.",
                       dose_band_path)
    else:
        logger.info("dose band: tier=%s — %s", resolved_dose_band.tier,
                    resolved_dose_band.source)
    return resolved_dose_band
