"""The harvest worker RPC, pool health, and harvest progress read off disk.

The harvest ladder (pool -> single worker -> two subprocesses) lives here
too, as ``HarvestLadder``, so /loom and /probe share one extraction path.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pleroma.harvest import pool as harvest_pool

if TYPE_CHECKING:
    from pleroma.serve.session import LoomSession

logger = logging.getLogger("loom_serve")


#: One-shot cache for ``subprocess_visible_presets`` — the answer cannot change
#: while this process lives, and the probe costs a python startup.
_SUBPROCESS_PRESETS: tuple[str, ...] | None | bool = False


def subprocess_visible_presets(py: str, cwd: Path) -> tuple[str, ...] | None:
    """Which anamnesis presets would a FRESH python process resolve?

    Presets come from anamnesis' registry: the model table anamnesis ships,
    plus every file on ``ANAMNESIS_MODELS``. Pleroma's own presets
    (``70b-modelc``) live in
    ``pleroma.config.anamnesis_registry.REGISTRY_FILE``, which this process may have put
    on the variable itself — a child inherits the variable, so it normally
    sees the same set, but a server launched without it (or with the variable
    scrubbed by a wrapper) would hand its fallback subprocess a preset the
    child cannot resolve. Asking the child is the only honest way to know.

    Returns None when the probe itself could not run: the caller then spawns
    the subprocess without a preflight, because a preflight that cannot
    answer must not be the thing that fails a draw. The answer (or None) is
    cached for the life of the process.
    """
    global _SUBPROCESS_PRESETS
    if _SUBPROCESS_PRESETS is not False:
        return _SUBPROCESS_PRESETS  # type: ignore[return-value]
    _SUBPROCESS_PRESETS = None
    try:
        proc = subprocess.run(
            [py, "-c",
             "import json,anamnesis.config as c;print(json.dumps(sorted(c.preset_names())))"],
            cwd=str(cwd), capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            _SUBPROCESS_PRESETS = tuple(json.loads(proc.stdout.strip().splitlines()[-1]))
            logger.info("a subprocess would see presets %s",
                        list(_SUBPROCESS_PRESETS))
        else:
            logger.warning("could not probe a subprocess's anamnesis presets (rc=%d): %s",
                           proc.returncode, (proc.stdout + proc.stderr)[-300:])
    except Exception as exc:  # noqa: BLE001 — a probe must never fail a draw
        logger.warning("could not probe a subprocess's anamnesis presets (%s: %s)",
                       type(exc).__name__, exc)
    return _SUBPROCESS_PRESETS  # type: ignore[return-value]


def harvest_via_worker(
    worker_url: str, loom_dir: Path, connect_timeout: float, total_timeout: float
) -> dict[str, Any]:
    """POST /harvest to a running harvest worker (``pleroma.harvest.worker``).
    Returns the worker's JSON body when it reports ``ok``. Raises on ANY problem
    (unreachable, non-ok body, timeout) — the caller's job is to catch that
    and fall back to the subprocess pipeline, never to half-trust a worker
    response.
    """
    req = urllib.request.Request(
        worker_url.rstrip("/") + "/harvest",
        data=json.dumps({"loom_dir": str(loom_dir)}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=connect_timeout + total_timeout) as resp:
        blob = json.loads(resp.read())
    if not blob.get("ok"):
        raise RuntimeError(f"harvest worker reported failure: {blob}")
    return blob


# ── live probes for the UI's top rig ──────────────────────────────────────────
#
# ★ WHY THESE EXIST. /info is a LAUNCH snapshot: it goes on advertising
# `harvest_pool_width: 2` after both workers have died. A rig cell that
# can go stale must be driven by a PROBE, not by a launch argument — so pool
# health is measured on request (GET /pool), and a harvest's per-branch progress
# is read off the disk it is being written to (GET /loom/progress), not guessed.
# Both are pure functions of their inputs so they are testable with no model.

WORKER_STATES = ("ok", "busy", "down", "error")


def _worker_hostport(url: str) -> tuple[str, int]:
    """'h:p' or 'http://h:p[/...]' -> (h, p). Raises ValueError on anything else."""
    s = str(url).strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"worker address {url!r} is not host:port")
    return host, int(port)


def probe_worker_health(url: str, timeout: float = 1.5) -> dict[str, Any]:
    """GET <worker>/health, classified into ONE of four states.

    The split between `busy` and `down` is the point, and it is measured, not
    assumed: the harvest worker serves on a single-threaded
    HTTPServer, so while it is harvesting, a TCP connect SUCCEEDS (the listen
    backlog accepts it) and the read then times out — curl rc=28 with
    connect=0.0002s. A dead worker refuses the connect outright (curl rc=7).
    So:
        ok     connected, and /health answered {"ok": true}
        busy   connected, but no answer inside `timeout` — alive and occupied
        down   the connect itself failed (refused / unreachable / no route)
        error  connected and answered, but not with a healthy body
    Never raises: a probe that throws would take the rig down with it.
    """
    import http.client
    import socket

    t0 = time.time()
    out: dict[str, Any] = {"url": str(url), "state": "error",
                           "latency_ms": None, "detail": None}
    try:
        host, port = _worker_hostport(url)
    except ValueError as exc:
        out["detail"] = str(exc)
        return out
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        try:
            conn.connect()
        except OSError as exc:  # refused, unreachable, connect timeout
            out["state"] = "down"
            out["detail"] = f"{type(exc).__name__}: {exc}"
            return out
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            raw = resp.read(4096)
        except (TimeoutError, socket.timeout):
            out["state"] = "busy"
            out["detail"] = f"connected; no answer in {timeout:g}s"
            return out
        except OSError as exc:  # reset mid-read: it was there and then was not
            out["state"] = "down"
            out["detail"] = f"{type(exc).__name__}: {exc}"
            return out
        out["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
        try:
            blob = json.loads(raw or b"{}")
        except ValueError:
            blob = None
        if resp.status == 200 and isinstance(blob, dict) and blob.get("ok") is True:
            out["state"] = "ok"
        else:
            out["detail"] = f"HTTP {resp.status}: {raw[:160]!r}"
        return out
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — closing a dead socket is not news
            pass


def pool_health_payload(
    urls: Sequence[str], timeout: float = 1.5,
    probe: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe every configured harvest worker concurrently -> the /pool body.

    `configured` is what the server was LAUNCHED with; `alive` is what answered
    just now. The UI draws the second and never the first. `busy` counts as
    alive (it is — it is harvesting). An empty `urls` means no worker is
    configured, which is a statement, not a failure: every harvest then takes
    the subprocess rung."""
    from concurrent.futures import ThreadPoolExecutor

    fn = probe or probe_worker_health
    uniq: list[str] = []
    for u in urls:
        if u and u not in uniq:
            uniq.append(str(u))
    rows: list[dict[str, Any]] = []
    if uniq:
        with ThreadPoolExecutor(max_workers=min(8, len(uniq))) as ex:
            for row in ex.map(lambda u: fn(u, timeout), uniq):
                rows.append(row if isinstance(row, dict)
                            else {"url": "?", "state": "error", "latency_ms": None,
                                  "detail": "probe returned a non-dict"})
    alive = sum(1 for r in rows if r.get("state") in ("ok", "busy"))
    return {"configured": len(uniq), "alive": alive, "workers": rows,
            "timeout_s": float(timeout), "probed_at": round(time.time(), 3)}


def harvest_completion(loom_dir: Path | None) -> dict[str, Any] | None:
    """Which generations of a RUNNING harvest have landed, in completion order.

    Stage 1 of every harvest rung (worker, pool shard, subprocess) writes
    `signatures/gen_NNN.npz` once per generation as it finishes it, into the
    draw's own directory — so the order those files appear in IS the order the
    branches were scanned. Read off disk, per request, no bookkeeping in the
    harvest path. Ordered by mtime_ns, then id (a tie is a same-tick write).

    Returns {"n_gens", "harvested": [gid, ...]} or None when there is no dir
    or it cannot be read. Never raises — progress is a nicety."""
    if loom_dir is None:
        return None
    try:
        d = Path(loom_dir)
        n_gens = sum(1 for _ in (d / "gen_records").glob("gen_*.json"))
        rows: list[tuple[int, int]] = []
        sig = d / "signatures"
        if sig.is_dir():
            for p in sig.glob("gen_*.npz"):
                m = re.fullmatch(r"gen_(\d+)\.npz", p.name)
                if not m:
                    continue
                try:
                    rows.append((p.stat().st_mtime_ns, int(m.group(1))))
                except OSError:  # vanished between glob and stat
                    continue
        rows.sort()
        return {"n_gens": int(n_gens), "harvested": [g for _, g in rows]}
    except OSError:
        return None


def progress_payload(
    progress: Mapping[str, Any] | None, loom_dir: Path | None,
    origin: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any] | None:
    """The /loom/progress `progress` block: the session's {stage, done, total},
    COPIED, plus — during a harvest stage only — which branches have landed
    (`gens`), and for a probe harvest which (candidate, rep) each gen is
    (`origin`, indexed by gen id; candidate null = the unworn base arm).
    Additive: a client that reads only {stage, done, total} sees no change."""
    if not progress:
        return None
    out = dict(progress)
    stage = str(out.get("stage") or "")
    if stage.endswith("harvest"):
        g = harvest_completion(loom_dir)
        if g is not None:
            out["gens"] = g
        if origin is not None and stage == "probe_harvest":
            out["origin"] = [list(o) for o in origin]
    return out


# ── the harvest ladder and the pool-health cache ─────────────────────────────


class PoolHealthCache:
    """GET /pool: every configured worker, probed. Cached for a couple of
    seconds so several tabs polling cannot hammer a worker that is
    single-threaded and busy harvesting. Thread-safe: one lock guards the
    probe and the cache.
    """

    def __init__(self, pool_urls: Sequence[str], harvest_worker: str | None,
                 probe: Callable[[str, float], dict[str, Any]] | None = None) -> None:
        self.pool_urls = list(pool_urls)
        self.harvest_worker = harvest_worker
        self._probe = probe
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {"at": 0.0, "body": None}

    def get(self, max_age_s: float = 2.0) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            body = self._cache["body"]
            if body is not None and now - float(self._cache["at"]) < max_age_s:
                return body
            configured = list(self.pool_urls) + (
                [str(self.harvest_worker)] if self.harvest_worker else [])
            body = pool_health_payload(configured, timeout=1.5, probe=self._probe)
            body["pool_width_configured"] = len(self.pool_urls)
            body["single_worker"] = str(self.harvest_worker) if self.harvest_worker else None
            self._cache.update(at=now, body=body)
            return body


class HarvestLadder:
    """The frozen harvest over a directory of gen_records: pool -> single worker
    -> two subprocesses. The fields are the launch settings every rung needs;
    ``run`` climbs down the ladder."""

    def __init__(self, *, pool_urls: Sequence[str], harvest_worker: str,
                 connect_timeout: float, timeout: float, preset: str,
                 model_path: str, calib_dir: str, discriminants: str,
                 py: str, repo: Path) -> None:
        self.pool_urls = list(pool_urls)
        self.harvest_worker = harvest_worker
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        self.preset = preset
        self.model_path = model_path
        self.calib_dir = calib_dir
        self.discriminants = discriminants
        self.py = py
        self.repo = repo

    @classmethod
    def from_args(cls, args: Any, pool_urls: Sequence[str], py: str,
                  repo: Path) -> HarvestLadder:
        return cls(pool_urls=pool_urls, harvest_worker=args.harvest_worker,
                   connect_timeout=float(args.harvest_worker_connect_timeout),
                   timeout=float(args.harvest_worker_timeout),
                   preset=str(args.preset), model_path=str(args.model_path),
                   calib_dir=str(args.calib_dir),
                   discriminants=str(args.discriminants),
                   py=py, repo=repo)

    def run(
        self, loom_dir: Path, sess: LoomSession, stage: str = "harvest"
    ) -> tuple[str, dict[str, Any] | None, float]:
        """Run the frozen harvest over a directory of gen_records.

        /probe and /loom both call this, so the probe's z-vectors come out of
        the EXACT extraction path the fan's did and are comparable to them by
        construction; a second path would break that. A POOL of >=2 workers
        (--harvest-workers) is tried first, then the persistent single worker,
        then the two-subprocess pipeline as the unconditional FALLBACK on ANY
        worker problem. The ladder is strictly degrading — every rung produces
        the same files:

            pool (N workers, sharded)  ->  single worker  ->  two subprocesses

        Returns ``(harvest_via, worker_detail, seconds)``: which rung ran, the
        worker's or pool's report (None for the subprocess rung) and the wall
        time. Sets ``sess.progress`` and ``sess.progress_dir`` while it runs.
        Raises RuntimeError when the subprocess rung cannot resolve the preset
        or a subprocess fails.
        """
        pool_urls = self.pool_urls
        sess.progress = {"stage": stage, "done": 0, "total": 2}
        sess.progress_dir = loom_dir  # GET /loom/progress reads landed branches here
        t1 = time.time()
        harvest_via = "subprocess"
        worker_detail: dict[str, Any] | None = None
        n_expected = len(list((loom_dir / "gen_records").glob("gen_*.json"))) or None
        if pool_urls:
            try:
                worker_detail = harvest_pool.harvest_via_pool(
                    pool_urls, loom_dir,
                    float(self.connect_timeout)
                    + float(self.timeout),
                    expected_gens=n_expected,
                )
                harvest_via = "pool"
                sess.progress = {"stage": stage, "done": 2, "total": 2}
            except Exception as exc:  # noqa: BLE001 — any pool problem degrades
                logger.warning(
                    "harvest POOL %s unavailable/failed (%s: %s) — falling back to "
                    "the single-worker path", ",".join(pool_urls),
                    type(exc).__name__, exc,
                )
                # Clear the half-written shard files. Nothing downstream reads
                # shard_meta/, so this is hygiene rather than correctness — but
                # leaving a partial set on disk is exactly the kind of debris a
                # later in-place retry could merge and believe.
                for stale in (loom_dir / "shard_meta").glob("*"):
                    try:
                        stale.unlink()
                    except OSError:  # a shared dir; never fail a draw over debris
                        logger.debug("could not remove stale shard file %s", stale)
        if harvest_via == "subprocess" and self.harvest_worker:
            try:
                worker_detail = harvest_via_worker(
                    f"http://{self.harvest_worker}", loom_dir,
                    float(self.connect_timeout),
                    float(self.timeout),
                )
                harvest_via = "worker"
                sess.progress = {"stage": stage, "done": 2, "total": 2}
            except Exception as exc:  # noqa: BLE001 — any worker problem falls back
                logger.warning(
                    "harvest worker %s unavailable/failed (%s: %s) — falling "
                    "back to the subprocess pipeline", self.harvest_worker,
                    type(exc).__name__, exc,
                )
        if harvest_via == "subprocess":
            # ⛔ THE BOTTOM RUNG SPAWNS A FRESH `pleroma.harvest.replay`, which resolves
            # the preset from anamnesis' registry in ITS environment. Pleroma's
            # presets (70b-modelc) come from pleroma's registry file via
            # ANAMNESIS_MODELS; a child without that variable dies with
            # `unknown model preset` — and the draw would report THAT instead
            # of whatever actually knocked the worker out. Checked BEFORE
            # spawning; spawns without the check if the probe cannot answer
            # (see subprocess_visible_presets).
            py, repo = self.py, self.repo
            stock = subprocess_visible_presets(py, repo)
            if stock is not None and str(self.preset) not in stock:
                raise RuntimeError(
                    f"harvest has no usable fallback: preset {self.preset!r} is "
                    "not resolvable in a child process (its registry file is not "
                    "on the child's ANAMNESIS_MODELS — see "
                    "pleroma/config/anamnesis_registry.py), so the subprocess rung "
                    f"would die with `unknown model preset {self.preset!r}` "
                    f"(child sees {list(stock)}) no matter what is wrong upstream. The "
                    "REAL failure is whatever took out the worker/pool above — "
                    "look at the warnings just before this line, not at this "
                    "one. Restart the harvest worker rather than expecting the "
                    "subprocess pipeline to cover for it."
                )
            env_cmds = [
                [py, "-m", "pleroma.harvest.replay", "--gen-dir", str(loom_dir),
                 "--out-dir", str(loom_dir), "--preset", str(self.preset),
                 "--model-path", str(self.model_path),
                 "--calib-dir", str(self.calib_dir),
                 "--discriminants", str(self.discriminants),
                 "--save-raw", "none"],
                [py, "-m", "pleroma.harvest.bins", "--gen-dirs", str(loom_dir),
                 "--model-path", str(self.model_path), "--preset", str(self.preset),
                 "--out", str(loom_dir / "bins.npz")],
            ]
            for si, cmd in enumerate(env_cmds):
                proc = subprocess.run(cmd, cwd=str(repo), capture_output=True,
                                      text=True)
                if proc.returncode != 0:
                    tail = (proc.stdout + proc.stderr)[-800:]
                    raise RuntimeError(f"harvest step failed ({cmd[2]}): ...{tail}")
                sess.progress = {"stage": stage, "done": si + 1, "total": 2}
        return harvest_via, worker_detail, time.time() - t1
