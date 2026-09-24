"""Every loom route, through `pleroma.serve`, over real HTTP, with no model.

A ``ServeContext`` takes a ``Runtime`` (Protocol), so these tests drive
``make_handler(ctx)`` on a real ``ThreadingHTTPServer`` with:

* a REAL ``LoomMap`` built from the contract fixtures (tests/contract/map),
* a fake runtime (deterministic token rows; ``attach`` records its calls),
* a fake harvester that writes the on-disk shape the real ladder produces
  (``signatures/gen_NNN.npz`` + ``<loom dir>/bins.npz``) so /loom and /probe
  run their real scoring, lever, contrast and gauge code.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pleroma.format.trim import trim_generated_ids
from pleroma.map.loom_map import LoomMap
from pleroma.serve import paths
from pleroma.serve.args import build_parser, resolve_sampling_defaults
from pleroma.serve.context import ServeContext
from pleroma.serve.harvest_client import PoolHealthCache
from pleroma.serve.http import MUTATING_PATHS, make_handler
from pleroma.serve.persistence import SessionStore
from pleroma.serve.session import LoomSession, State
from tests.contract.map import _fixtures as fx

EOS = 2
PLEN = 24


class _Handle:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def remove(self) -> None:
        self._log.append("remove")


@dataclass
class FakeRuntime:
    """The ``Runtime`` Protocol, deterministic and torch-free."""

    prompt_mode: str = "chat"
    modelc_header: str = ""
    stop_strings: list[str] = field(default_factory=list)
    eos_ids: list[int] = field(default_factory=lambda: [EOS])
    pad_id: int = EOS
    chat_template_present: bool = True
    attach_log: list[str] = field(default_factory=list)

    def render(self, messages: list[dict[str, str]]) -> np.ndarray:
        n = sum(len(m["content"]) for m in messages)
        return np.full((1, PLEN), 100 + n % 50, dtype=np.int64)

    def draw(self, input_ids: Any, seed: int, max_new: int) -> list[int]:
        return [int(x) for x in input_ids[0]] + [10 + seed % 7, 11, EOS]

    def draw_batch(self, input_ids: Any, batch: int, seed: int,
                   max_new: int) -> list[list[int]]:
        prefix = [int(x) for x in input_ids[0]]
        return [prefix + [20 + j, 30 + seed % 5, EOS] for j in range(batch)]

    def trim_to_eos(self, row: list[int], plen: int) -> list[int]:
        return trim_generated_ids(row[plen:], self.eos_ids, self.pad_id)

    def attach(self, vectors: np.ndarray, alpha: float) -> list[Any]:
        self.attach_log.append(f"attach {np.asarray(vectors).shape} {alpha}")
        return [_Handle(self.attach_log)]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"t{int(x)}" for x in ids if int(x) != EOS).strip()

    def finish_reply(self, decoded: str) -> tuple[str, dict[str, str]]:
        return decoded, {}


class FakeHarvester:
    """Writes what the real harvest ladder leaves on disk, from gen_records."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def run(self, loom_dir: Path, sess: LoomSession,
            stage: str = "harvest") -> tuple[str, dict[str, Any] | None, float]:
        self.calls.append((loom_dir, stage))
        sess.progress = {"stage": stage, "done": 0, "total": 2}
        sess.progress_dir = loom_dir
        gids = sorted(int(p.stem.split("_")[1])
                      for p in (loom_dir / "gen_records").glob("gen_*.json"))
        (loom_dir / "signatures").mkdir(exist_ok=True)
        rng = np.random.default_rng(len(self.calls))
        for g in gids:
            np.savez(loom_dir / "signatures" / f"gen_{g:03d}.npz",
                     features=rng.standard_normal(fx.Z_DIM) * 2.0)
        np.savez(loom_dir / "bins.npz",
                 features=rng.standard_normal((len(gids), fx.BINS_DIM)),
                 generation_id=np.asarray(gids))
        return "fake", None, 0.0


@dataclass
class Served:
    url: str
    ctx: ServeContext
    runtime: FakeRuntime
    harvester: FakeHarvester
    work: Path

    def get(self, path: str) -> tuple[int, Any]:
        try:
            with urllib.request.urlopen(self.url + path, timeout=30) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "")
                return r.status, (json.loads(body) if "json" in ctype else body)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path: str, body: Any) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


def _args(tmp: Path, map_path: Path, disc: Path, *extra: str) -> Any:
    args = build_parser().parse_args([
        "--map", str(map_path), "--discriminants", str(disc),
        "--calib-dir", str(tmp / "calib"), "--model-path", "fake/model",
        "--work-dir", str(tmp / "work"), *extra])
    resolve_sampling_defaults(args)
    return args


@pytest.fixture()
def served(tmp_path: Path) -> Iterator[Served]:
    disc, sha = fx.write_discriminants(tmp_path / "disc")
    map_path = fx.write_v1a_map(tmp_path / "map", sha)
    atlas = tmp_path / "atlas.json"
    atlas.write_text(json.dumps({"axes": [], "note": "fixture"}))
    args = _args(tmp_path, map_path, disc, "--atlas-report", str(atlas))
    args.work_dir.mkdir(parents=True)
    loom_map = LoomMap(map_path, disc)
    runtime, harvester = FakeRuntime(), FakeHarvester()
    ctx = ServeContext(
        args=args, loom_map=loom_map, map_fp="fp-test", lever_bank=None,
        runtime=runtime, state=State(), store=SessionStore(args.work_dir),
        harvester=harvester,  # type: ignore[arg-type]
        pool_health=PoolHealthCache([], None), pool_urls=[],
        resolved_dose_band=None)
    ctx.restore_on_start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield Served(f"http://127.0.0.1:{server.server_address[1]}", ctx,
                     runtime, harvester, args.work_dir)
    finally:
        server.shutdown()
        server.server_close()


# ── GET ──────────────────────────────────────────────────────────────────────


def test_info_is_build_info_payload_for_this_server(served: Served) -> None:
    status, body = served.get("/info")
    assert status == 200
    assert body == json.loads(json.dumps(served.ctx.info_payload()))
    assert body["prompt_mode"] == "chat"
    assert body["sites"] == fx.SITES
    assert body["dose_policy_default"] == "flat"
    assert body["lever_kind_default"] == "absolute"
    assert body["harvest_pool_width"] == 0
    assert body["persistence"]["restore_on_start"] is True
    assert body["prompt_mode_detail"]["top_p"] == 0.95


def test_get_dispatch(served: Served) -> None:
    """Every GET route matches the PARSED path (/info?x=1 is /info, not an
    unknown raw path); error bodies carry a machine-readable code."""
    status, body = served.get("/info?x=1")
    assert status == 200 and body["prompt_mode"] == "chat"
    assert served.get("/nope") == (404, {"error": "unknown path", "code": "not_found"})
    assert served.get("/state") == (
        400, {"error": "missing session", "code": "session_required"})
    status, body = served.get("/state?session=s1")
    assert status == 200 and body["session"] == "s1" and body["worn"] is None
    assert body["probe_note"] is None
    assert served.get("/loom/progress?session=") == (
        200, {"session": "", "progress": None})
    status, pool = served.get("/pool")
    assert status == 200 and pool["configured"] == 0 and pool["alive"] == 0
    assert pool["single_worker"] is None
    status, sess = served.get("/sessions")
    assert status == 200 and sess["n_sessions"] == 1   # /state created s1


def test_ui_is_read_from_disk_each_request(served: Served, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    ui = tmp_path / "ui.html"
    monkeypatch.setattr(paths, "UI_PATH", ui)
    assert served.get("/")[0] == 404
    ui.write_text("<p>v1</p>")
    assert served.get("/ui") == (200, b"<p>v1</p>")
    ui.write_text("<p>v2</p>")
    assert served.get("/") == (200, b"<p>v2</p>")


def test_atlas_route_serves_the_configured_report(served: Served) -> None:
    assert served.get("/atlas") == (200, {"axes": [], "note": "fixture"})
    Path(served.ctx.args.atlas_report).unlink()
    assert served.get("/atlas") == (
        404, {"error": "no atlas built yet", "code": "not_found"})


# ── POST: the conversation family ────────────────────────────────────────────


def test_chat_undo_reroll_edit_truncate(served: Served) -> None:
    s, r = served.post("/chat", {"session": "c", "text": "hello"})
    assert s == 200 and r["branch"] == "loom" and r["n_turns"] == 1
    assert r["reply"].startswith("t") and r["worn"] is None
    s, r = served.post("/chat", {"session": "c", "branch": "base", "text": "hi"})
    assert s == 200 and r["worn"] is None and r["branch"] == "base"
    s, r = served.post("/reroll", {"session": "c"})
    assert s == 200 and r["reroll_n"] == 1 and "replaced" in r
    s, r = served.post("/edit", {"session": "c", "text": "hello again"})
    assert s == 200 and r["replaced_user"] == "hello"
    s, r = served.post("/undo", {"session": "c"})
    assert s == 200 and r["popped"]["user"] == "hello again" and r["n_turns"] == 0
    assert served.post("/undo", {"session": "c"})[0] == 400
    s, r = served.post("/truncate", {"session": "c", "branch": "base", "keep_turns": 0})
    assert s == 200 and r["dropped_messages"] == 2
    assert served.post("/chat", {"session": "c", "branch": "x", "text": "a"}) == (
        400, {"error": "unknown branch 'x'", "code": "unknown_branch"})


def test_post_errors_map_to_the_same_statuses(served: Served) -> None:
    assert served.post("/chat", {"text": "a"}) == (
        400, {"error": "missing 'session'", "code": "session_required"})
    assert served.post("/nope", {"session": "e"}) == (
        404, {"error": "unknown path", "code": "not_found"})
    # KeyError -> 400 with the key, before the handler runs
    assert served.post("/wear", {"session": "e"}) == (
        400, {"error": "'index'", "code": "bad_request"})
    assert served.post("/truncate", {"session": "e"}) == (
        400, {"error": "'keep_turns'", "code": "bad_request"})
    # coercion precedes the handler: a bad alpha wins over "no candidates"
    s, r = served.post("/wear", {"session": "e", "index": 0, "alpha": "x"})
    assert s == 400 and "could not convert" in r["error"]
    assert served.post("/wear", {"session": "e", "index": 0}) == (
        400, {"error": "no candidates — /loom first", "code": "no_candidates"})
    # TypeError is NOT a 400 (pre-split behaviour, pinned as-is)
    s, r = served.post("/wear", {"session": "e", "index": None})
    assert s == 500 and r["error"].startswith("TypeError")
    s, r = served.post("/loom", {"session": "e", "text": "t", "detach_wear": "no"})
    assert s == 400 and "detach_wear must be a boolean" in r["error"]
    s, r = served.post("/loom", {"session": "e", "text": "t", "probe": 1})
    assert s == 400 and "probe must be omitted" in r["error"]


def test_unwear_reset_restore_and_persistence(served: Served) -> None:
    served.post("/chat", {"session": "p", "text": "keep me"})
    snap = served.ctx.store.path_for("p")
    assert snap.exists(), "a mutating request writes the snapshot"
    assert json.loads(snap.read_text())["n_turns"]["loom"] == 1
    assert served.post("/unwear", {"session": "p"}) == (
        200, {"session": "p", "worn": None})
    assert served.post("/reset", {"session": "p"}) == (
        200, {"session": "p", "cleared": True, "worn": None})
    assert "p" not in served.ctx.state.sessions, "/reset leaves nothing in RAM"
    assert json.loads(snap.read_text())["n_turns"]["loom"] == 0
    s, r = served.post("/restore", {"session": "p"})
    assert s == 200 and r["restored"] is True and r["n_turns"] == {"base": 0, "loom": 0}
    s, r = served.post("/restore", {"session": "never"})
    assert s == 400 and "no snapshot" in r["error"]
    assert "/restore" not in MUTATING_PATHS


# ── POST: the loom, wear and probe ───────────────────────────────────────────


def test_loom_then_wear_then_worn_chat(served: Served) -> None:
    s, r = served.post("/loom", {"session": "L", "text": "what next?", "k": 3})
    assert s == 200, r
    assert r["n_futures"] == 3 and r["prompt_length"] == PLEN
    assert r["bins_feasible"] is True and r["harvest_via"] == "fake"
    assert [f["harvested"] for f in r["futures"]] == [True, True, True]
    assert all(len(f["code"]) == fx.RANK for f in r["futures"])
    assert all("distinct" in f["scores"] and "loudness" in f["scores"]
               for f in r["futures"])
    assert r["fan_mean_raw_norms"] is not None and r["contrast_note"] is None
    assert r["drawn_under_wear"] is None and r["auto_selected"] is None
    sess = served.ctx.state.sessions["L"]
    assert sess.progress is None and sess.progress_dir is None, "cleared in finally"
    loom_dir = Path(r["loom_dir"])
    rec = json.loads((loom_dir / "gen_records" / "gen_000.json").read_text())
    assert rec["prompt_mode"] == "chat" and rec["sampling"]["top_p"] == 0.95

    s, w = served.post("/wear", {"session": "L", "index": 1, "alpha": 0.4,
                                 "loom_id": r["loom_id"]})
    assert s == 200, w
    assert w["worn"]["index"] == 1 and w["worn"]["active"] is True
    assert w["lever_kind"] == "absolute" and w["dose"]["policy"] == "flat"
    s, w2 = served.post("/wear", {"session": "L", "index": 1, "loom_id": "stale"})
    assert s == 400 and "not the latest draw" in w2["error"]

    served.runtime.attach_log.clear()
    s, c = served.post("/chat", {"session": "L", "text": "go"})
    assert s == 200 and c["worn"]["index"] == 1
    assert served.runtime.attach_log == [
        f"attach ({len(fx.SITES)}, {fx.HIDDEN}) 0.4", "remove"]


def test_loom_auto_contrast_and_detach(served: Served) -> None:
    s, r = served.post("/loom", {"session": "A", "text": "draw", "k": 4,
                                 "auto": {"policy": "loudest", "alpha": 0.3,
                                          "lever_kind": "contrast",
                                          "dose_policy": "predicted"}})
    assert s == 200, r
    auto = r["auto_selected"]
    assert auto["policy"] == "loudest" and auto["lever_kind"] == "contrast"
    assert auto["dose"]["policy"] == "predicted"
    assert r["worn"]["lever_kind"] == "contrast"
    # a second draw under that wear: detach_wear=true attaches nothing
    served.runtime.attach_log.clear()
    s, r2 = served.post("/loom", {"session": "A", "text": "draw", "k": 3,
                                  "detach_wear": True})
    assert s == 200 and r2["detach_wear"] is True and r2["drawn_under_wear"] is None
    assert served.runtime.attach_log == []
    assert all("cos_to_worn" in f["scores"] for f in r2["futures"])
    s, r3 = served.post("/loom", {"session": "A", "text": "draw", "k": 3})
    assert s == 200 and r3["drawn_under_wear"]["index"] == auto["index"]
    assert served.runtime.attach_log[-1] == "remove"


def test_probe_scores_the_fan_and_gauge_can_pick(served: Served) -> None:
    s, r = served.post("/loom", {"session": "P", "text": "fan me", "k": 3})
    assert s == 200
    s, p = served.post("/probe", {"session": "P", "text": "different"})
    assert s == 400 and "prefix does not match" in p["error"]
    s, p = served.post("/probe", {"session": "P", "text": "fan me", "reps": 2,
                                  "horizon": 16})
    assert s == 200, p
    probe = p["probe"]
    assert probe["k"] == 3 and probe["reps"] == 2 and probe["clean_regime"] is True
    assert probe["n_spans_harvested"] == 6
    assert served.harvester.calls[-1][1] == "probe_harvest"
    assert all("gauge" in f["scores"] for f in p["futures"])
    # the probe run inside /loom, feeding auto.policy='gauge' in one call
    s, r2 = served.post("/loom", {"session": "P", "text": "fan me", "k": 3,
                                  "probe": {"reps": 1, "horizon": 16},
                                  "auto": {"policy": "gauge"}})
    assert s == 200, r2
    assert r2["probe"]["k"] == 3 and r2["auto_selected"]["policy"] == "gauge"
    s, st = served.get("/state?session=P")
    assert st["probe"] is not None and st["probe_note"] is None


def test_wear_code_explicit_code_needs_no_draw(served: Served) -> None:
    s, r = served.post("/wear_code", {"session": "W", "code": [0.1] * fx.RANK,
                                      "code_kind": "differential", "alpha": 0.2})
    assert s == 200, r
    assert r["source"]["kind"] == "explicit_code"
    assert r["worn"]["index"] == -1 and r["worn"]["lever_kind"] is None
    s, r = served.post("/wear_code", {"session": "W", "group": "g"})
    assert s == 400 and "no lever bank loaded" in r["error"]
    s, r = served.post("/wear_code", {"session": "W", "code": [0.1] * fx.RANK,
                                      "dose_policy": "predicted"})
    assert s == 400 and "not available on /wear_code" in r["error"]


def test_the_legacy_argv_parses_through_the_moved_parser() -> None:
    """The argv pleroma/config/legacy.py renders for EVERY shipped profile
    parses through the server's own parser."""
    from pleroma.config import load_profile
    from pleroma.config.legacy import legacy_serve_argv
    from pleroma.config.profile import Deployment

    root = Path(__file__).resolve().parents[3]
    profiles = sorted((root / "profiles").glob("*.toml"))
    assert profiles
    for path in profiles:
        prof = load_profile(path)
        deploy = Deployment.model_validate({
            "profile": prof.name, "map": {"path": "m.npz"},
            "discriminants": {"path": "d.npz"}, "calib_dir": "c",
            "work_dir": "w"})
        argv = legacy_serve_argv(prof, deploy)
        assert argv[:2] == ["-m", "pleroma.serve.legacy"]
        args = build_parser().parse_args(argv[2:])
        resolve_sampling_defaults(args)
        assert args.prompt_mode == prof.format.mode.value, path
        assert args.top_p == prof.sampling.top_p, path
        assert args.temperature == prof.sampling.temperature, path
        assert args.default_k == prof.lengths.default_k, path
        assert args.wear_lever_default == prof.steer.lever_kind.value, path
