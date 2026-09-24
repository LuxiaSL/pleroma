"""The standalone front-end (ui/) against the Python-side contract.

The UI's typed client is GENERATED from pleroma/serve/api-schema.json, and its
mock backend serves ui/fixtures/. Both can drift silently when a response model
changes, so this suite catches it on the Python side:

* every ui/fixtures/*.json fits its response model in STRICT mode (no node needed);
* the schema fingerprint baked into the generated client
  (ui/src/api/schema-meta.ts) equals ``schema_sha256()`` — the value the server
  publishes as /info.api.schema_sha256 (no node needed);
* the fixtures are what ui/scripts/build_fixtures.py produces from the
  recordings (skipped where the recordings are absent, e.g. the public export);
* with node and the ui's node modules installed: ``npm run check:api`` (generated types
  current) and ``npm run typecheck`` (the client still compiles against them).
  Skipped otherwise, with the reason.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from pleroma.serve import responses as R
from pleroma.serve.schema import schema_sha256

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
FIXTURES = UI / "fixtures"

#: fixture file -> the response model it must fit
FIXTURE_MODELS: dict[str, type] = {
    "info.json": R.InfoResponse,
    "state_drawn.json": R.StateResponse,
    "state_worn.json": R.StateResponse,
    "loom_k16.json": R.LoomResponse,
    "loom_worn.json": R.LoomResponse,
    "wear.json": R.WearResponse,
}

pytestmark = pytest.mark.skipif(not UI.is_dir(), reason="no ui/ in this tree")


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ui_build_fixtures",
                                                  UI / "scripts" / "build_fixtures.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_fixture_has_a_model() -> None:
    present = {p.name for p in FIXTURES.glob("*.json")}
    assert present == set(FIXTURE_MODELS), (
        "ui/fixtures/ and FIXTURE_MODELS disagree — map every fixture to its model")


@pytest.mark.parametrize("name", sorted(FIXTURE_MODELS))
def test_fixture_fits_its_model_strictly(name: str) -> None:
    body = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    R.check_response(FIXTURE_MODELS[name], body, route=f"ui fixture {name}", strict=True)


def test_fixture_conversation_survived_conversion() -> None:
    """The recorded worn session has two turns per branch (a converter bug once emptied them)."""
    st = json.loads((FIXTURES / "state_worn.json").read_text(encoding="utf-8"))
    assert {b: len(m) for b, m in st["histories"].items()} == {"base": 4, "loom": 4}
    assert st["worn"]["index"] == 3


def test_generated_client_fingerprint_matches_server() -> None:
    meta = (UI / "src" / "api" / "schema-meta.ts").read_text(encoding="utf-8")
    m = re.search(r'SCHEMA_SHA256 = "([0-9a-f]{64})"', meta)
    assert m, "schema-meta.ts has no SCHEMA_SHA256"
    assert m.group(1) == schema_sha256(), (
        "ui/src/api/schema-meta.ts was generated from a different schema — "
        "run `npm run gen:api` in ui/ and commit")


def test_fixtures_are_what_the_converter_builds() -> None:
    builder = _load_builder()
    if not Path(builder.SRC).is_dir():
        pytest.skip("the recorded fixtures the converter reads are not in this tree")
    assert builder.main(["--check"]) == 0, "run `python ui/scripts/build_fixtures.py`"


def _npm(*args: str) -> subprocess.CompletedProcess[str]:
    npm = shutil.which("npm")
    if not shutil.which("node") or not npm:
        pytest.skip("node/npm not installed")
    if not (UI / "node_modules").is_dir():
        pytest.skip("ui/node_modules absent — run `npm ci` in ui/ to enable the front-end checks")
    return subprocess.run([npm, "run", "--silent", *args], cwd=UI, capture_output=True,
                          text=True, timeout=300)


def test_api_types_are_current() -> None:
    r = _npm("check:api")
    assert r.returncode == 0, r.stdout + r.stderr


def test_ui_typechecks() -> None:
    r = _npm("typecheck")
    assert r.returncode == 0, r.stdout + r.stderr
