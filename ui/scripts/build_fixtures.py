"""Convert the RECORDED live responses in ui_proto/fixtures/ into the mock
backend's fixtures (ui/fixtures/), fitted to the /api/v1 response models.

    python ui/scripts/build_fixtures.py          # rewrite ui/fixtures/*.json
    python ui/scripts/build_fixtures.py --check  # exit 1 if they would change

WHY A CONVERSION AND NOT A COPY. The recordings predate the typed API
(2026-09-23): they carry fields the models now forbid (the retired 2-means
`camp`/`camps` scores) and lack fields the models require (`prompt_mode`,
`lever_kind`, `/info.api`, ...). A mock that serves off-contract bodies would
let the UI grow against shapes no server sends, so every fixture here is
fitted to its pydantic model and then checked in STRICT mode
(``check_response(strict=True)`` — the same check the server's test suite runs
on every live response). ``tests/unit/test_ui_frontend.py`` re-validates them,
so a model change that the fixtures no longer fit fails the Python suite.

WHAT IS CHANGED, AND ONLY THIS:
  * keys a model forbids are DROPPED (recursively);
  * required keys a recording lacks are FILLED with the documented default of
    the server at the time of recording (`FILL` below, each with its reason);
  * private filesystem paths are REDACTED to `<redacted>/<basename>` so the
    public export's gate passes (tools/export_public.py).
The futures' texts, codes, scores, histories and dose receipts are untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pleroma.serve import responses as R  # noqa: E402
from pleroma.serve.schema import schema_sha256  # noqa: E402

SRC = ROOT / "ui_proto" / "fixtures"
DST = ROOT / "ui" / "fixtures"

#: private path prefixes → redacted. Anything absolute under these roots.
_PRIVATE = re.compile(r"(?:/(?:mnt|home|net|data|models)/)[^\s'\"]*")


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _PRIVATE.sub(lambda m: "<redacted>/" + m.group(0).rstrip("/").rsplit("/", 1)[-1],
                            value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def _shape(annotation: Any) -> tuple[str, type[BaseModel]] | None:
    """How a field holds a model: ("one", M) for M | None, ("list", M) for
    list[M], ("dict", M) / ("dictlist", M) for dict[str, M] / dict[str, list[M]].
    None for anything else (scalars, open dicts, unions of several models —
    those are left exactly as recorded)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ("one", annotation)
    origin = get_origin(annotation)
    args = [a for a in get_args(annotation) if a is not type(None)]
    if origin is list and len(args) == 1:
        inner = _shape(args[0])
        return ("list", inner[1]) if inner and inner[0] == "one" else None
    if origin is dict and len(args) == 2:
        inner = _shape(args[1])
        if inner and inner[0] == "one":
            return ("dict", inner[1])
        if inner and inner[0] == "list":
            return ("dictlist", inner[1])
        return None
    shapes = [sh for a in args if (sh := _shape(a)) is not None]
    return shapes[0] if len(shapes) == 1 and len(args) == 1 else None


def fit(model: type[BaseModel], body: Any, fill: dict[str, dict[str, Any]]) -> Any:
    """Drop keys `model` forbids, fill required keys from `fill[model name]`,
    recursing into nested models (directly, in lists, and in dict values)."""
    if not isinstance(body, dict):
        return body
    extra_ok = model.model_config.get("extra") != "forbid"
    out: dict[str, Any] = {}
    fields = model.model_fields
    by_alias = {(f.alias or name): name for name, f in fields.items()}
    for key, value in body.items():
        name = by_alias.get(key)
        if name is None:
            if extra_ok:
                out[key] = value
            continue
        shape = _shape(fields[name].annotation)
        if shape is None or value is None:
            out[key] = value
            continue
        kind, sub = shape
        if kind == "one":
            out[key] = fit(sub, value, fill)
        elif kind == "list" and isinstance(value, list):
            out[key] = [fit(sub, v, fill) for v in value]
        elif kind == "dict" and isinstance(value, dict):
            out[key] = {k: fit(sub, v, fill) for k, v in value.items()}
        elif kind == "dictlist" and isinstance(value, dict):
            out[key] = {k: [fit(sub, x, fill) for x in v] if isinstance(v, list) else v
                        for k, v in value.items()}
        else:
            out[key] = value
    for key, value in fill.get(model.__name__, {}).items():
        out.setdefault(key, value)
    return out


#: required fields the recordings lack, with the value the server of the time
#: would have sent. Keyed by model name.
FILL: dict[str, dict[str, Any]] = {
    "LoomResponse": {
        # the recordings were served in chat mode (loom_serve's default)
        "prompt_mode": "chat",
        # not recorded; illustrative, and not rendered as a measurement
        "prompt_length": 0, "bins_prompt_floor": 0, "bins_feasible": True,
        # recorded before code_contrast existed (2026-09-22): no contrast rows
        "fan_mean_raw_norms_contrast": None,
        "contrast_note": "recorded before the contrast lever existed; no contrast "
                         "rows on this fixture",
    },
    "WornPublic": {"lever_kind": "absolute"},
    "WearResponse": {"lever_kind": "absolute"},
    "InfoResponse": {
        "prompt_mode": "chat",
        "prompt_mode_detail": {"temperature": 1.0, "top_p": 0.95, "stop_strings": None,
                               "chat_template_present": True},
        "lever_kinds": [
            {"key": "absolute", "default": True,
             "description": "W applied to the future's own input row — every wear on "
                            "record before 2026-09-22."},
            {"key": "contrast", "default": False,
             "description": "W applied to the future's row minus the mean of the fan's "
                            "OTHER valid rows — the one-vs-rest object v1a was fit on. "
                            "UNVALIDATED behaviourally on a fan wear."}],
        "lever_kind_default": "absolute",
        "api": {"version": "1", "prefix": "/api/v1", "schema_sha256": "",
                "auth_required": False, "legacy_aliases_open": True},
    },
}


def load(name: str) -> Any:
    return json.loads((SRC / name).read_text(encoding="utf-8"))


def build() -> dict[str, tuple[type[BaseModel], Any]]:
    fill = json.loads(json.dumps(FILL))
    fill["InfoResponse"]["api"]["schema_sha256"] = schema_sha256()
    out: dict[str, tuple[type[BaseModel], Any]] = {}

    info = load("info.json")
    info["map_meta"] = {k: info["map_meta"][k] for k in
                        ("lam", "rank", "n_rows", "n_fans", "prereg_token", "stage")
                        if k in info["map_meta"]}
    out["info.json"] = (R.InfoResponse, fit(R.InfoResponse, info, fill))
    out["state_drawn.json"] = (R.StateResponse, fit(R.StateResponse, load("state.json"), fill))
    out["state_worn.json"] = (R.StateResponse, fit(R.StateResponse, load("state_worn.json"), fill))
    out["loom_k16.json"] = (R.LoomResponse, fit(R.LoomResponse, load("loom_k16.json"), fill))
    out["loom_worn.json"] = (R.LoomResponse, fit(R.LoomResponse, load("loom_worn.json"), fill))
    out["wear.json"] = (R.WearResponse, fit(R.WearResponse, load("wear.json"), fill))
    # invariants: the conversion only DROPS forbidden keys — the conversation
    # and the fan must survive it whole (a bug once emptied `histories`).
    for name, src in (("state_drawn.json", "state.json"), ("state_worn.json", "state_worn.json")):
        raw = load(src)
        got = out[name][1]
        assert got["histories"].keys() == raw["histories"].keys(), name
        for b, msgs in raw["histories"].items():
            assert len(got["histories"][b]) == len(msgs), (name, b)
    for name in ("loom_k16.json", "loom_worn.json"):
        raw = load(name)
        got = out[name][1]
        assert [f["text"] for f in got["futures"]] == [f["text"] for f in raw["futures"]], name
        assert [f["code"] for f in got["futures"]] == [f["code"] for f in raw["futures"]], name
    return {name: (model, redact(body)) for name, (model, body) in out.items()}


def render(body: Any) -> str:
    return json.dumps(body, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any fixture would change")
    a = ap.parse_args(argv)
    stale: list[str] = []
    for name, (model, body) in build().items():
        R.check_response(model, body, route=f"fixture {name}", strict=True)
        text = render(body)
        path = DST / name
        if a.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            DST.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)} ({model.__name__}, strict OK)")
    if stale:
        print("STALE fixtures: " + ", ".join(stale) + " — run python ui/scripts/build_fixtures.py",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
