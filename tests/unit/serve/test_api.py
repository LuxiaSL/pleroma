"""The versioned loom API (docs/API.md): prefix + legacy aliases, CORS, the
bearer token, error codes, GET error handling, the typed-response contract,
the published schema, and --ui-dir.

Same rig as test_routes.py — a REAL fixture ``LoomMap``, a fake runtime and a
fake harvester behind a real ``ThreadingHTTPServer`` — with a raw
``http.client`` so headers, methods and un-normalised paths are visible.
Every request here runs the response contract in STRICT mode (conftest.py).
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr

from pleroma.probe import orchestrator as loom_probe
from pleroma.dose.policy import DOSE_POLICIES
from pleroma.errors import CodedError, ErrorCode
from pleroma.levers.kind import LEVER_KINDS
from pleroma.map.loom_map import LoomMap
from pleroma.serve import models, paths, responses, schema
from pleroma.serve.api import (
    API_PREFIX,
    MUTATING_PATHS,
    ROUTES,
    ApiSettings,
    normalize_origin,
)
from pleroma.serve.args import build_parser
from pleroma.serve.context import ServeContext
from pleroma.serve.harvest_client import PoolHealthCache
from pleroma.serve.http import make_handler, static_file
from pleroma.serve.persistence import SessionStore
from pleroma.serve.policies import AUTO_POLICIES
from pleroma.serve.session import BRANCHES, State
from tests.contract.map import _fixtures as fx
from tests.unit.serve.test_routes import FakeHarvester, FakeRuntime, _args

TOKEN = "s3cret-token-for-tests"
UI_ORIGIN = "http://localhost:5173"


@dataclass
class Reply:
    status: int
    headers: dict[str, str]
    raw: bytes

    @property
    def body(self) -> Any:
        return json.loads(self.raw) if self.raw else None


@dataclass
class Rig:
    port: int
    ctx: ServeContext
    runtime: FakeRuntime
    tmp: Path

    def call(self, method: str, path: str, body: Any = None,
             headers: dict[str, str] | None = None) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            data = None if body is None else json.dumps(body).encode()
            hdrs = dict(headers or {})
            if data is not None:
                hdrs.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=data, headers=hdrs)
            r = conn.getresponse()
            return Reply(r.status, {k.lower(): v for k, v in r.getheaders()}, r.read())
        finally:
            conn.close()

    def get(self, path: str, **kw: Any) -> Reply:
        return self.call("GET", path, **kw)

    def post(self, path: str, body: Any, **kw: Any) -> Reply:
        return self.call("POST", path, body, **kw)


RigFactory = Callable[..., Rig]


@pytest.fixture()
def rig_factory(tmp_path: Path) -> Iterator[RigFactory]:
    servers: list[ThreadingHTTPServer] = []

    def build(*extra: str, settings: ApiSettings | None = None) -> Rig:
        d = tmp_path / f"rig{len(servers)}"
        d.mkdir()
        disc, sha = fx.write_discriminants(d / "disc")
        map_path = fx.write_v1a_map(d / "map", sha)
        args = _args(d, map_path, disc, *extra)
        args.work_dir.mkdir(parents=True)
        runtime = FakeRuntime()
        ctx = ServeContext(
            args=args, loom_map=LoomMap(map_path, disc), map_fp="fp-test",
            lever_bank=None, runtime=runtime, state=State(),
            store=SessionStore(args.work_dir),
            harvester=FakeHarvester(),  # type: ignore[arg-type]
            pool_health=PoolHealthCache([], None), pool_urls=[],
            resolved_dose_band=None, api_settings=settings)
        ctx.restore_on_start()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctx))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return Rig(server.server_address[1], ctx, runtime, d)

    yield build
    for s in servers:
        s.shutdown()
        s.server_close()


@pytest.fixture()
def rig(rig_factory: RigFactory) -> Rig:
    return rig_factory()


def _bearer(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── prefix, aliases, query strings ───────────────────────────────────────────


def test_every_route_answers_under_the_prefix_like_its_alias(rig: Rig) -> None:
    assert rig.post("/chat", {"session": "a", "text": "hi"}).status == 200
    for path in ("/info", "/sessions", "/pool", "/state?session=a",
                 "/loom/progress?session=a", "/loom/progress?session="):
        legacy, v1 = rig.get(path), rig.get(API_PREFIX + path)
        assert (legacy.status, v1.status) == (200, 200), path
        assert legacy.body == v1.body, path
    # POST: the same sequence on two sessions answers with the same shapes
    for base in ("", API_PREFIX):
        s = f"s{base.replace('/', '_')}"
        r = rig.post(base + "/loom", {"session": s, "text": "what next?", "k": 3})
        assert r.status == 200, r.body
        w = rig.post(base + "/wear", {"session": s, "index": 0,
                                      "loom_id": r.body["loom_id"]})
        assert w.status == 200 and w.body["worn"]["index"] == 0
        for p, b in (("/chat", {"text": "go"}), ("/reroll", {}),
                     ("/edit", {"text": "go again"}), ("/undo", {}),
                     ("/truncate", {"keep_turns": 0}), ("/unwear", {}),
                     ("/reset", {}), ("/restore", {})):
            out = rig.post(base + p, {"session": s, **b})
            assert out.status == 200, (base, p, out.body)
    assert rig.get(API_PREFIX + "/nope").body == {"error": "unknown path",
                                                  "code": "not_found"}
    assert rig.post(API_PREFIX + "/nope", {"session": "x"}).status == 404


def test_error_parity_between_prefix_and_alias(rig: Rig) -> None:
    for base in ("", API_PREFIX):
        assert rig.post(base + "/chat", {"session": "e", "branch": "x"}).body == {
            "error": "unknown branch 'x'", "code": "unknown_branch"}
        assert rig.post(base + "/wear", {"session": "e"}).body == {
            "error": "'index'", "code": "bad_request"}


def test_query_strings_are_tolerated_everywhere(rig: Rig, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    ui = tmp_path / "ui.html"
    ui.write_text("<p>ui</p>")
    monkeypatch.setattr(paths, "UI_PATH", ui)
    assert rig.get("/info?x=1").status == 200
    assert rig.get(API_PREFIX + "/info?cache=bust").status == 200
    assert rig.get("/ui?v=2").raw == b"<p>ui</p>"
    assert rig.get("/?v=2").raw == b"<p>ui</p>"
    assert rig.post("/unwear?x=1", {"session": "q"}).status == 200


def test_schema_has_no_legacy_alias(rig: Rig) -> None:
    assert rig.get("/schema").status == 404
    assert rig.get(API_PREFIX + "/schema").status == 200


# ── GET error handling, error codes, headers ─────────────────────────────────


def test_get_errors_are_json_not_a_dropped_connection(
        rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(rig.ctx, "info_payload", boom)
    for path in ("/info", API_PREFIX + "/info"):
        r = rig.get(path)
        assert r.status == 500
        assert r.body == {"error": "RuntimeError: boom", "code": "internal"}

    def refuse() -> dict[str, Any]:
        raise ValueError("nope")

    monkeypatch.setattr(rig.ctx, "info_payload", refuse)
    assert rig.get("/info").body == {"error": "nope", "code": "bad_request"}


def test_json_is_never_cached(rig: Rig) -> None:
    for r in (rig.get("/info"), rig.get(API_PREFIX + "/sessions"), rig.get("/nope"),
              rig.post("/chat", {"text": "no session"})):
        assert r.headers["cache-control"] == "no-store"
        assert r.headers["content-type"] == "application/json"


def test_refusals_carry_codes_and_keep_their_sentences(rig: Rig) -> None:
    """Each refusal carries a machine-readable code, raised where the refusal
    is decided, and keeps its sentence, so the legacy page's regexes
    (`pleroma/serve/static/legacy_ui.html`) still match."""
    def post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        r = rig.post(API_PREFIX + path, body)
        return r.status, r.body

    assert post("/wear", {"session": "c", "index": 0})[1]["code"] == "no_candidates"
    assert post("/probe", {"session": "c", "text": "x"})[1]["code"] == "no_candidates"
    s, r = post("/loom", {"session": "c", "text": "what next?", "k": 3})
    assert s == 200
    loom_id = r["loom_id"]

    s, e = post("/wear", {"session": "c", "index": 0, "loom_id": "stale"})
    assert (s, e["code"]) == (400, "stale_loom") and "not the latest draw" in e["error"]
    s, e = post("/wear", {"session": "c", "index": 0, "lever_kind": "sideways"})
    assert (s, e["code"]) == (400, "lever_kind_invalid") and "lever_kind" in e["error"]
    s, e = post("/wear", {"session": "c", "index": 0, "lever_kind": 3})
    assert e["code"] == "lever_kind_invalid"
    s, e = post("/wear", {"session": "c", "index": 0, "dose_policy": "loud"})
    assert (s, e["code"]) == (400, "dose_policy_invalid") and "dose_policy" in e["error"]
    s, e = post("/wear_code", {"session": "c", "code": [0.1] * fx.RANK,
                               "dose_policy": "predicted"})
    assert e["code"] == "dose_policy_unavailable" and "dose_policy" in e["error"]
    s, e = post("/probe", {"session": "c", "text": "different"})
    assert e["code"] == "probe_prefix_mismatch"
    s, e = post("/restore", {"session": "never-saved"})
    assert (s, e["code"]) == (400, "no_snapshot")
    s, e = post("/chat", {"session": "c", "branch": "sideways"})
    assert e["code"] == "unknown_branch"
    s, e = post("/chat", {"text": "who am i"})
    assert e["code"] == "session_required"
    s, e = post("/loom", {"session": "c", "text": "what next?", "k": 3,
                          "auto": {"policy": "gauge"}})
    assert (s, e["code"]) == (400, "gauge_needs_probe")
    assert "gauge" in e["error"] and "probe" in e["error"]   # the UI's regex pair
    assert loom_id  # the first draw was fine


def test_contrast_unavailable_and_predicted_unavailable_codes(rig: Rig) -> None:
    sess = rig.ctx.state.get("u")
    sess.candidates = [{"index": 0, "lever": np.zeros((len(fx.SITES), fx.HIDDEN)),
                        "note": None, "raw_norms": None, "code": None,
                        "contrast_lever": None,
                        "contrast_note": "fewer than 2 harvested"}]
    sess.loom_id = "u-000"
    r = rig.post("/wear", {"session": "u", "index": 0, "lever_kind": "contrast"})
    assert (r.status, r.body["code"]) == (400, "lever_kind_unavailable")
    assert "lever_kind" in r.body["error"]
    r = rig.post("/wear", {"session": "u", "index": 0, "dose_policy": "predicted"})
    assert (r.status, r.body["code"]) == (400, "dose_policy_unavailable")


def test_coded_errors_are_still_value_errors() -> None:
    exc = CodedError(ErrorCode.STALE_LOOM, "the sentence")
    assert isinstance(exc, ValueError) and str(exc) == "the sentence"
    with pytest.raises(ValueError, match="the sentence"):
        raise exc


# ── v1 never creates sessions on read ────────────────────────────────────────


def test_v1_state_does_not_create_but_the_alias_still_does(rig: Rig) -> None:
    r = rig.get(API_PREFIX + "/state?session=ghost")
    assert r.status == 404 and r.body["code"] == "no_session"
    assert "ghost" not in rig.ctx.state.sessions
    r = rig.get(API_PREFIX + "/loom/progress?session=ghost")
    assert r.status == 200 and r.body == {"session": "ghost", "progress": None}
    assert "ghost" not in rig.ctx.state.sessions
    assert rig.get(API_PREFIX + "/state").body["code"] == "session_required"
    assert rig.get("/state?session=ghost").status == 200
    assert "ghost" in rig.ctx.state.sessions, "the legacy alias creates, as before"
    assert rig.get(API_PREFIX + "/state?session=ghost").status == 200


# ── CORS ─────────────────────────────────────────────────────────────────────


def test_no_cors_origin_configured_is_same_origin_only(rig: Rig) -> None:
    pre = rig.call("OPTIONS", API_PREFIX + "/info", headers={
        "Origin": UI_ORIGIN, "Access-Control-Request-Method": "GET"})
    assert pre.status == 403 and pre.body["code"] == "forbidden_origin"
    assert "access-control-allow-origin" not in pre.headers
    r = rig.get(API_PREFIX + "/info", headers={"Origin": UI_ORIGIN})
    assert r.status == 403 and "access-control-allow-origin" not in r.headers
    # a same-origin browser request (the old UI's POSTs carry Origin) passes
    same = f"http://127.0.0.1:{rig.port}"
    r = rig.post("/unwear", {"session": "o"}, headers={"Origin": same})
    assert r.status == 200 and "access-control-allow-origin" not in r.headers
    # no Origin at all (curl, scripts) is unaffected
    assert rig.get("/info").status == 200


def test_allowed_origin_gets_cors_on_v1_only(rig_factory: RigFactory) -> None:
    rig = rig_factory("--cors-origin", UI_ORIGIN + "/", "--cors-origin",
                      "http://127.0.0.1:4173")
    pre = rig.call("OPTIONS", API_PREFIX + "/chat", headers={
        "Origin": UI_ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization",
        "Access-Control-Request-Private-Network": "true"})
    assert pre.status == 204 and pre.raw == b""
    h = pre.headers
    assert h["access-control-allow-origin"] == UI_ORIGIN
    assert h["vary"] == "Origin"
    assert h["access-control-allow-methods"] == "GET, POST, OPTIONS"
    assert h["access-control-allow-headers"] == "Content-Type, Authorization"
    assert h["access-control-allow-private-network"] == "true"
    pre2 = rig.call("OPTIONS", API_PREFIX + "/info", headers={"Origin": UI_ORIGIN})
    assert pre2.status == 204
    assert "access-control-allow-private-network" not in pre2.headers

    r = rig.get(API_PREFIX + "/info", headers={"Origin": UI_ORIGIN})
    assert r.status == 200
    assert r.headers["access-control-allow-origin"] == UI_ORIGIN
    assert r.headers["vary"] == "Origin"
    # errors carry CORS too, so the app can read them
    r = rig.post(API_PREFIX + "/chat", {"text": "x"}, headers={"Origin": UI_ORIGIN})
    assert r.status == 400 and r.headers["access-control-allow-origin"] == UI_ORIGIN

    evil = "http://evil.example"
    r = rig.get(API_PREFIX + "/info", headers={"Origin": evil})
    assert r.status == 403 and "access-control-allow-origin" not in r.headers
    pre = rig.call("OPTIONS", API_PREFIX + "/info", headers={"Origin": evil})
    assert pre.status == 403 and "access-control-allow-origin" not in pre.headers

    # the legacy aliases stay same-origin only, even for an allowed origin
    r = rig.get("/info", headers={"Origin": UI_ORIGIN})
    assert r.status == 403 and "access-control-allow-origin" not in r.headers
    assert rig.call("OPTIONS", "/info", headers={"Origin": UI_ORIGIN}).status == 403
    assert rig.get("/info").status == 200


def test_origins_are_normalised_and_wildcards_refused() -> None:
    assert normalize_origin("HTTP://LocalHost:5173/") == "http://localhost:5173"
    for bad in ("*", "localhost:5173", "http://x/path", "ftp://x", ""):
        with pytest.raises(ValueError):
            normalize_origin(bad)
    args = build_parser().parse_args(["--map", "m", "--discriminants", "d",
                                      "--calib-dir", "c", "--model-path", "p",
                                      "--work-dir", "w", "--cors-origin", "*"])
    with pytest.raises(ValueError, match="refused"):
        ApiSettings.from_args(args, environ={})


# ── the bearer token ─────────────────────────────────────────────────────────


def test_token_guards_v1_and_leaves_loopback_aliases_open(
        rig_factory: RigFactory, caplog: pytest.LogCaptureFixture) -> None:
    rig = rig_factory("--api-token", TOKEN)
    r = rig.get(API_PREFIX + "/info")
    assert r.status == 401 and r.body["code"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"
    assert rig.get(API_PREFIX + "/info", headers=_bearer("wrong")).status == 401
    assert rig.get(API_PREFIX + "/info",
                   headers={"Authorization": TOKEN}).status == 401  # no scheme
    assert rig.post(API_PREFIX + "/chat", {"session": "t", "text": "x"}).status == 401
    assert "t" not in rig.ctx.state.sessions, "refused before anything ran"
    with caplog.at_level(logging.DEBUG, logger="loom_serve"):
        ok = rig.get(API_PREFIX + "/info", headers=_bearer())
    assert ok.status == 200
    assert ok.body["api"] == {
        "version": "1", "prefix": API_PREFIX, "schema_sha256": schema.schema_sha256(),
        "auth_required": True, "legacy_aliases_open": True}
    assert TOKEN not in caplog.text, "the token must never be logged"
    # loopback, no CORS: the legacy aliases (the old UI) answer as before
    assert rig.get("/info").status == 200
    assert rig.post("/unwear", {"session": "t"}).status == 200


@pytest.mark.parametrize("extra", [("--cors-origin", UI_ORIGIN), ("--allow-lan",)])
def test_token_closes_the_aliases_once_the_server_is_opened(
        rig_factory: RigFactory, extra: tuple[str, ...], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    rig = rig_factory("--api-token", TOKEN, *extra)
    r = rig.get("/info")
    assert r.status == 401 and r.body["code"] == "unauthorized"
    assert rig.post("/chat", {"session": "t", "text": "x"}).status == 401
    assert rig.get("/info", headers=_bearer()).status == 200
    assert rig.get(API_PREFIX + "/info", headers=_bearer()).body["api"][
        "legacy_aliases_open"] is False
    # the page itself is static and needs no token (a navigation cannot send one)
    ui = tmp_path / "ui.html"
    ui.write_text("<p>ui</p>")
    monkeypatch.setattr(paths, "UI_PATH", ui)
    assert rig.get("/").status == 200


def test_token_from_the_environment_and_default_off() -> None:
    args = build_parser().parse_args(["--map", "m", "--discriminants", "d",
                                      "--calib-dir", "c", "--model-path", "p",
                                      "--work-dir", "w"])
    off = ApiSettings.from_args(args, environ={})
    assert not off.auth_required and off.legacy_aliases_open and off.token_ok(None)
    on = ApiSettings.from_args(args, environ={"PLEROMA_API_TOKEN": " t0k "})
    assert on.auth_required and on.token_ok("Bearer t0k") and on.token_ok("bearer t0k")
    assert not on.token_ok("Bearer t0") and not on.token_ok(None)
    assert "t0k" not in repr(on) and "t0k" not in str(on.model_dump())
    flag = ApiSettings.from_args(
        build_parser().parse_args(["--map", "m", "--discriminants", "d",
                                   "--calib-dir", "c", "--model-path", "p",
                                   "--work-dir", "w", "--api-token", "f"]),
        environ={"PLEROMA_API_TOKEN": "env"})
    assert flag.token_ok("Bearer f"), "the flag wins over the environment"
    assert ApiSettings(token=SecretStr("x")).legacy_aliases_open


# ── the schema ───────────────────────────────────────────────────────────────


def test_committed_schema_is_fresh() -> None:
    committed = schema.SCHEMA_PATH.read_text(encoding="utf-8")
    assert committed == schema.schema_text(), (
        "pleroma/serve/api-schema.json is STALE — regenerate with "
        "`python -m pleroma.serve.schema > pleroma/serve/api-schema.json`")


def test_schema_route_and_info_fingerprint(rig: Rig) -> None:
    doc = rig.get(API_PREFIX + "/schema").body
    assert doc == json.loads(schema.schema_text())
    assert doc["openapi"] == "3.1.0" and doc["info"]["version"] == "1"
    for r in ROUTES:
        op = doc["paths"][API_PREFIX + r.path][r.method.lower()]
        assert op["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"].endswith("/" + r.response.__name__)
        assert op["x-mutating"] == (r.path in MUTATING_PATHS and r.method == "POST")
        if r.request is not None:
            assert r.request.__name__ in doc["components"]["schemas"]
    assert {r.path for r in ROUTES if r.mutating} == set(MUTATING_PATHS)
    assert rig.get("/info").body["api"]["schema_sha256"] == schema.schema_sha256()
    codes = doc["components"]["schemas"]["ErrorCode"]["enum"]
    assert set(codes) == {c.value for c in ErrorCode}


def test_wire_enums_mirror_the_runtime_tuples() -> None:
    assert models.DosePolicy.__args__ == DOSE_POLICIES
    assert models.LeverKind.__args__ == LEVER_KINDS
    assert models.AutoPolicy.__args__ == AUTO_POLICIES
    assert models.Branch.__args__ == BRANCHES
    assert models.ProbeBase.__args__ == loom_probe.BASE_MODES


def test_auto_and_probe_specs_coerce_exactly_as_before() -> None:
    a = models.AutoSpec.from_mapping({"policy": "loudest", "alpha": "0.4",
                                      "lever_kind": 3, "junk": 1})
    assert (a.policy, a.alpha, a.lever_kind, a.dose_policy) == ("loudest", 0.4, 3, None)
    assert models.AutoSpec.from_mapping({"alpha": 1}).policy == ""
    with pytest.raises(ValueError, match="could not convert"):
        models.AutoSpec.from_mapping({"policy": "stay", "alpha": "x"})
    p = models.ProbeSpec.from_mapping({"reps": "3", "horizon": 16.9, "base": "fresh"})
    assert (p.alpha, p.reps, p.horizon, p.base, p.n_base) == (
        0.35, 3, 16, "fresh", loom_probe.DEFAULT_N_BASE)
    assert models.ProbeSpec.from_mapping({"base": "odd"}).base == "odd", \
        "range/base checks stay in run_probe (it owns the sentence)"
    # the wire models accept what pleroma/serve/static/legacy_ui.html sends
    models.LoomBody.model_validate({"session": "s", "text": "t", "k": 6,
                                    "auto": {"policy": "gauge", "alpha": 0.35},
                                    "probe": True, "detach_wear": False})
    models.WearBody.model_validate({"session": "s", "index": 2, "alpha": 0.5,
                                    "loom_id": "abc-001", "lever_kind": "contrast"})


# ── the response contract, beyond test_routes' coverage ──────────────────────


def test_contract_covers_modelc_probe_wear_and_disk_rows(rig: Rig) -> None:
    rt = rig.runtime
    rt.prompt_mode = "modelc"
    rt.finish_reply = lambda decoded: (decoded, {"dream": "", "tail_kind": "none"})  # type: ignore[method-assign]
    r = rig.post("/chat", {"session": "m", "text": "hello"})
    assert r.status == 200 and list(r.body)[:2] == ["dream", "tail_kind"], \
        "extras still lead the body — the dict is sent as built"
    r = rig.post("/loom", {"session": "m", "text": "hello again", "k": 3})
    assert r.status == 200 and "tail_kind" in r.body["futures"][0]
    r = rig.post("/probe", {"session": "m", "text": "hello again", "reps": 2,
                            "horizon": 16, "wear": {"alpha": 0.3}})
    assert r.status == 200, r.body
    assert r.body["probe"]["worn"]["alpha"] == 0.3 and "wall_s" in r.body["probe"]
    st = rig.get(API_PREFIX + "/state?session=m")
    assert st.status == 200 and st.body["histories"]["loom"][1]["tail_kind"] == "none"
    assert st.body["probe"]["wall_s"] is not None
    # /sessions: an on-disk row carries the PERSISTED worn record
    rows = rig.get(API_PREFIX + "/sessions").body["sessions"]
    row = next(x for x in rows if x["session"] == "m")
    assert row["worn"]["map_fingerprint"] == "fp-test"
    # a live progress block, as /state and the loom's progress route report it
    rig.ctx.state.get("m").progress = {"stage": "generate", "done": 1, "total": 3}
    assert rig.get(API_PREFIX + "/state?session=m").body["loom_in_progress"][
        "done"] == 1
    assert rig.get(API_PREFIX + "/loom/progress?session=m").body["progress"][
        "stage"] == "generate"


def test_contract_covers_a_failed_in_loom_probe(rig: Rig,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a: Any, **k: Any) -> dict[str, Any]:
        raise RuntimeError("probe harvest died")

    monkeypatch.setattr(rig.ctx, "run_probe", broken)
    r = rig.post(API_PREFIX + "/loom", {"session": "f", "text": "t" * 30, "k": 3,
                                        "probe": True})
    assert r.status == 200
    assert r.body["probe"]["error"] == "RuntimeError: probe harvest died"


def test_lenient_mode_sends_a_misfit_body_and_logs_it(
        rig: Rig, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv(responses.STRICT_ENV)
    real = rig.ctx.sessions_payload
    monkeypatch.setattr(rig.ctx, "sessions_payload",
                        lambda: {**real(), "surprise": 1})
    with caplog.at_level(logging.ERROR, logger="loom_serve"):
        r = rig.get("/sessions")
    assert r.status == 200 and r.body["surprise"] == 1
    assert "CONTRACT VIOLATION" in caplog.text


def test_strict_mode_turns_a_misfit_into_a_500(rig: Rig,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    assert os.environ.get(responses.STRICT_ENV) == "1"
    real = rig.ctx.sessions_payload
    monkeypatch.setattr(rig.ctx, "sessions_payload",
                        lambda: {**real(), "surprise": 1})
    r = rig.get("/sessions")
    assert r.status == 500 and r.body["code"] == "internal"
    assert "ContractViolation" in r.body["error"]


# ── --ui-dir ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def ui_tree(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<p>new ui</p>")
    (root / "assets" / "app.js").write_text("console.log(1)")
    (root / ".env").write_text("SECRET=1")
    (tmp_path / "secret.txt").write_text("outside")
    (root / "assets" / "escape.txt").symlink_to(tmp_path / "secret.txt")
    return root


def test_ui_dir_serves_its_index_and_assets(rig_factory: RigFactory,
                                            ui_tree: Path) -> None:
    rig = rig_factory("--ui-dir", str(ui_tree))
    assert rig.get("/").raw == b"<p>new ui</p>"
    assert rig.get("/ui").raw == b"<p>new ui</p>"
    js = rig.get("/assets/app.js")
    assert js.status == 200 and js.raw == b"console.log(1)"
    assert js.headers["content-type"].startswith("text/javascript")
    # the API is unaffected and wins over any same-named file
    assert rig.get("/info").status == 200
    assert rig.get("/assets/missing.js").body["code"] == "not_found"


@pytest.mark.parametrize("path", [
    "/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt",
    "/assets/%2e%2e/%2e%2e/secret.txt", "/.env", "/assets/escape.txt",
    "/%00index.html", "/assets\\..\\..\\secret.txt",
])
def test_ui_dir_refuses_traversal(rig_factory: RigFactory, ui_tree: Path,
                                  path: str) -> None:
    rig = rig_factory("--ui-dir", str(ui_tree))
    r = rig.get(path)
    assert r.status == 404, (path, r.raw[:80])
    assert b"outside" not in r.raw and b"SECRET" not in r.raw
    assert static_file(ui_tree, path) is None


def test_default_ui_is_unchanged_without_ui_dir(rig: Rig) -> None:
    assert rig.ctx.api_settings.ui_dir is None
    assert rig.get("/assets/app.js").status == 404
