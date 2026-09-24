"""The top rig and the branch walk of the legacy loom page.

Two promises are pinned here, both of the form "the UI describes what it is
serving, not what it was launched with":

  * RIG. A cell that can go stale is driven by a PROBE. `/info` can advertise
    `harvest_pool_width: 2` for hours after both workers have died, so the
    rig's pool cell is drawn from `GET /pool`, which calls each worker's
    /health, and a server without /pool reads "unknown" — never the launch
    width.
  * WALK. The waiting animation draws only what the loom's `progress` route
    reported: a branch turns solid ("landed") only after the server has
    listed its signature, never before.

Server-side pieces are pure functions and are tested with no model. The page's
rig is rendered by headless Chrome against `loom_ui_stub_server`. The driven
walk needs a draw that takes real time and is not part of this suite.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pleroma.serve import legacy as ls

REPO = Path(__file__).resolve().parents[4]
UI = REPO / "pleroma" / "serve" / "static" / "legacy_ui.html"


@pytest.fixture(scope="module")
def ui_text() -> str:
    return UI.read_text(encoding="utf-8")


# ── the pool probe: four states, and busy is not down ─────────────────────

def _serve_health(status: int, body: bytes) -> tuple[http.server.HTTPServer, str]:
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a: Any) -> None:
            return

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


def test_a_healthy_worker_reads_ok() -> None:
    srv, url = _serve_health(200, b'{"ok": true, "preset": "8b"}')
    try:
        row = ls.probe_worker_health(url, timeout=2.0)
    finally:
        srv.shutdown()
    assert row["state"] == "ok"
    assert row["latency_ms"] is not None


def test_a_worker_that_accepts_but_never_answers_is_busy_not_down() -> None:
    """harvest_worker serves on ONE thread: mid-harvest, the TCP connect is
    accepted by the listen backlog and the read times out (measured on the GPU node:
    connect 0.2 ms, curl rc=28). That is an alive, occupied worker."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)                       # never accept() — exactly the busy shape
    try:
        row = ls.probe_worker_health(f"127.0.0.1:{sock.getsockname()[1]}", timeout=0.4)
    finally:
        sock.close()
    assert row["state"] == "busy", row


def test_a_refused_connect_is_down() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()                         # nothing listens there now
    row = ls.probe_worker_health(f"127.0.0.1:{port}", timeout=0.5)
    assert row["state"] == "down", row


@pytest.mark.parametrize("status,body", [(200, b'{"ok": false}'), (500, b"boom"), (200, b"not json")])
def test_an_unhealthy_answer_is_error(status: int, body: bytes) -> None:
    srv, url = _serve_health(status, body)
    try:
        row = ls.probe_worker_health(url, timeout=2.0)
    finally:
        srv.shutdown()
    assert row["state"] == "error", row


def test_a_malformed_address_never_raises() -> None:
    assert ls.probe_worker_health("not-an-address")["state"] == "error"
    assert ls.probe_worker_health("http://127.0.0.1:1/health", timeout=0.3)["state"] == "down"


def test_pool_payload_counts_busy_as_alive_and_dedupes() -> None:
    states = {"a:1": "ok", "b:2": "busy", "c:3": "down", "d:4": "error"}
    body = ls.pool_health_payload(
        ["a:1", "b:2", "c:3", "d:4", "a:1"], timeout=0.1,
        probe=lambda u, t: {"url": u, "state": states[u], "latency_ms": None, "detail": None})
    assert body["configured"] == 4, "a worker listed twice is one worker"
    assert body["alive"] == 2
    assert [w["state"] for w in body["workers"]] == ["ok", "busy", "down", "error"]


def test_no_configured_worker_is_a_statement_not_a_failure() -> None:
    body = ls.pool_health_payload([], timeout=0.1)
    assert body == {**body, "configured": 0, "alive": 0, "workers": []}


def test_the_pool_route_is_probe_driven_and_never_reads_the_launch_width() -> None:
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(
        (REPO / "pleroma" / "serve").rglob("*.py")))  # the server's modules
    i = src.index('if path == "/pool":')  # the GET dispatch in pleroma/serve/http.py
    route = "\n".join(l for l in src[i: i + 600].splitlines() if not l.strip().startswith("#"))
    assert "pool_health_cached()" in route
    assert "harvest_pool_width" not in route


def test_the_page_never_draws_pool_health_from_the_launch_width(ui_text: str) -> None:
    # prose about the incident is fine; READING the field is the bug
    assert not re.search(r"\.harvest_pool_width|\[\s*[\"']harvest_pool_width", ui_text), (
        "the rig must never read /info's launch argument as pool health")
    assert 'get("/pool"' in ui_text


# ── harvest progress read off the disk ────────────────────────────────────

def _fake_loom_dir(tmp: Path, n: int, landed_order: list[int]) -> Path:
    (tmp / "gen_records").mkdir(parents=True)
    (tmp / "signatures").mkdir()
    for g in range(n):
        (tmp / "gen_records" / f"gen_{g:03d}.json").write_text("{}")
    base = time.time() - 100
    for j, g in enumerate(landed_order):
        f = tmp / "signatures" / f"gen_{g:03d}.npz"
        f.write_bytes(b"x")
        os.utime(f, (base + j, base + j))
    (tmp / "signatures" / "run_meta.json").write_text("{}")   # not a gen: ignored
    return tmp


def test_harvest_completion_is_in_landing_order(tmp_path: Path) -> None:
    d = _fake_loom_dir(tmp_path, 6, [4, 0, 5])
    assert ls.harvest_completion(d) == {"n_gens": 6, "harvested": [4, 0, 5]}


def test_harvest_completion_degrades_rather_than_raising(tmp_path: Path) -> None:
    assert ls.harvest_completion(None) is None
    assert ls.harvest_completion(tmp_path / "nope") == {"n_gens": 0, "harvested": []}


def test_progress_payload_is_additive_and_never_mutates_the_session(tmp_path: Path) -> None:
    d = _fake_loom_dir(tmp_path, 3, [2])
    live = {"stage": "harvest", "done": 0, "total": 2}
    out = ls.progress_payload(live, d)
    assert live == {"stage": "harvest", "done": 0, "total": 2}, "must copy, not mutate"
    assert out == {**live, "gens": {"n_gens": 3, "harvested": [2]}}
    gen = ls.progress_payload({"stage": "generate", "done": 0, "total": 3}, d)
    assert "gens" not in gen, "per-branch detail only exists during a harvest"
    assert ls.progress_payload(None, d) is None
    origin = [[0, 0], [0, 1], [None, 0]]
    probe = ls.progress_payload({"stage": "probe_harvest", "done": 0, "total": 2}, d, origin)
    assert probe["origin"] == origin
    plain = ls.progress_payload({"stage": "harvest", "done": 0, "total": 2}, d, origin)
    assert "origin" not in plain, "origin describes a probe's replies only"


def test_the_server_wires_the_progress_dir_and_clears_it() -> None:
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(
        (REPO / "pleroma" / "serve").rglob("*.py")))  # the server's modules
    assert "sess.progress_dir = loom_dir" in src
    # one helper, called from both routes
    assert src.count("_s.progress_dir = None") == 1
    assert src.count("_clear_progress(ctx, session)") == 2, \
        "both /loom and /probe must clear it"
    assert src.count("finally:\n        _clear_progress(ctx, session)") == 2, \
        "…in a finally, so a failed draw clears it too"
    assert "progress_payload(\n" in src or "progress_payload(" in src


# ── static: motion is optional, hovers are budgeted ───────────────────────

def test_every_new_animation_honours_reduced_motion(ui_text: str) -> None:
    blocks = re.findall(r"@media \(prefers-reduced-motion: reduce\)\{(.*?)\n\}", ui_text, re.S)
    joined = "\n".join(blocks)
    for sel in (".walk .scan", ".walk .beam", ".gf .busy", ".gf .beat"):
        assert sel in joined, f"{sel} animates with no reduced-motion override"
    # the JS-driven growth is gated too, not only the CSS keyframes
    grow = re.search(r"function walkGrow\((.*?)\n\}", ui_text, re.S)
    assert grow and "WALK_REDUCED" in grow.group(1)
    scan = re.search(r"function walkScan\((.*?)\n\}", ui_text, re.S)
    assert scan and "WALK_REDUCED" in scan.group(1)


def test_the_walk_never_fills_a_tip_before_the_server_says_so(ui_text: str) -> None:
    """The growth is paced by the clock, so it must be CAPPED below the tips."""
    grow = re.search(r"function walkGrow\((.*?)\n\}", ui_text, re.S)
    assert grow is not None
    assert "0.92" in grow.group(1)
    assert "Math.min(0.92" in grow.group(1)


# ── rendered: the rig, against the stub server ────────────────────────────

CHROME = (shutil.which("google-chrome") or shutil.which("google-chrome-stable")
          or shutil.which("chromium") or shutil.which("chromium-browser"))


def _render(fx: Any, budget: int = 5000) -> str:
    from tests.unit.serve.loom.loom_ui_stub_server import serve
    harness = """<script>(function(){ window.__errs = [];
      window.addEventListener("error", (e) => window.__errs.push(String(e.message)));
      window.addEventListener("unhandledrejection", (e) => window.__errs.push("rej: " + e.reason));
      setTimeout(function(){ const d = document.createElement("div"); d.id = "__errbox";
        d.setAttribute("data-errs", JSON.stringify(window.__errs)); document.body.appendChild(d); }, 2500);
    })();</script>"""
    srv, url = serve(fx, harness)
    tmp = Path(tempfile.mkdtemp(prefix="loomui-rig-"))
    try:
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
             "--window-size=1680,1100", f"--virtual-time-budget={budget}",
             f"--user-data-dir={tmp / 'p'}", "--dump-dom", url],
            capture_output=True, text=True, timeout=240)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def _rig(dom: str) -> dict[str, tuple[str, str, str]]:
    """name -> (class, face, hover) for every gauge cell on the rig."""
    out: dict[str, tuple[str, str, str]] = {}
    for m in re.finditer(r'<div class="(cell g[^"]*)" data-g="([^"]+)"([^>]*)>(.*?)</div>', dom, re.S):
        t = re.search(r'title="([^"]*)"', m.group(3))
        v = re.search(r'<span class="v">([^<]*)</span>', m.group(4))
        out[m.group(2)] = (m.group(1), v.group(1) if v else "", t.group(1) if t else "")
    return out


def _errs(dom: str) -> list[str]:
    m = re.search(r'id="__errbox" data-errs="([^"]*)"', dom)
    assert m is not None, "the harness never ran — the page did not boot"
    return json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&"))


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_rig_splits_state_live_and_constants() -> None:
    from tests.unit.serve.loom.loom_ui_stub_server import Fixture
    fx = Fixture()
    fx.info["prompt_mode"] = "modelc"          # what both live looms run
    dom = _render(fx)
    assert _errs(dom) == []
    rig = _rig(dom)
    assert "warn" in rig["prompt"][0], "a non-chat framing is state that can be wrong"
    assert {"ruler", "dose", "band", "prompt", "srv", "pool", "draws", "map"} <= set(rig), rig.keys()
    # constants are NOT on the face any more — one `map` cell carries them
    for gone in ("lam", "n_rows", "n_fans", "sites", "k₀", "horizon₀", "code rank", "sessions"):
        assert gone not in rig
    assert rig["map"][1] == "r64"
    # every hover within the tier-2 budget
    for name, (_cls, _face, hover) in rig.items():
        assert len(hover) <= 100, (name, len(hover), hover)
    # groups, in order: state | live | map
    assert dom.index('id="rigState"') < dom.index('id="rigLive"') < dom.index('id="rigMap"')


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_a_server_without_a_pool_probe_claims_no_pool_width() -> None:
    """An older server has no /pool. The cell must say it does not know — a
    launch width displayed as health is the failure this guards."""
    from tests.unit.serve.loom.loom_ui_stub_server import Fixture
    fx = Fixture()
    fx.info["harvest_workers"] = ["127.0.0.1:8784", "127.0.0.1:8785"]
    fx.info["harvest_pool_width"] = 2
    dom = _render(fx)
    cls, face, hover = _rig(dom)["pool"]
    assert face == "?"
    assert "no /pool probe" in hover and "2" not in face


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome on this host")
def test_the_pool_cell_draws_what_the_probe_found() -> None:
    from tests.unit.serve.loom.loom_ui_stub_server import Fixture
    rows = [{"url": "127.0.0.1:8784", "state": "down", "latency_ms": None, "detail": "refused"},
            {"url": "127.0.0.1:8785", "state": "down", "latency_ms": None, "detail": "refused"},
            {"url": "127.0.0.1:8786", "state": "busy", "latency_ms": None, "detail": "no answer"}]
    fx = Fixture(pool={"configured": 3, "alive": 1, "workers": rows, "timeout_s": 1.5,
                       "probed_at": time.time()})
    fx.info["harvest_pool_width"] = 2      # the launch argument, still claiming health
    dom = _render(fx)
    cls, face, hover = _rig(dom)["pool"]
    assert face == "1/3"
    assert "warn" in cls and "alarmlink" in cls
    assert "2 of 3" in hover
    # one pip per worker, drawn by state: 2 crossed-out rings + 1 pulsing
    i = dom.index('data-g="pool"')
    cell = dom[i: dom.index("</div>", i)]
    assert cell.count('stroke="var(--alarm)"') == 4      # 2 rings + 2 crosses
    assert cell.count('class="busy"') == 1

