"""The published schema of ``/api/v1``: an OpenAPI 3.1 document built from the
route table (``pleroma.serve.api.ROUTES``) and the pydantic models.

``GET /api/v1/schema`` serves it; ``/info.api.schema_sha256`` fingerprints it;
the committed copy ``pleroma/serve/api-schema.json`` is what a front-end
generates its client from, and a unit test fails when that copy is stale.

Regenerate::

    python -m pleroma.serve.schema > pleroma/serve/api-schema.json
    python -m pleroma.serve.schema --check     # exit 1 if the committed copy is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from pleroma.serve.api import API_PREFIX, API_VERSION, ROUTES, TOKEN_ENV, Route
from pleroma.serve.errors import ErrorBody

SCHEMA_PATH = Path(__file__).resolve().parent / "api-schema.json"
_REF = "#/components/schemas/{model}"


def _ref(model: type[BaseModel]) -> dict[str, str]:
    return {"$ref": _REF.format(model=model.__name__)}


def _operation(route: Route) -> dict[str, Any]:
    op: dict[str, Any] = {
        "operationId": route.operation_id,
        "summary": route.summary,
    }
    desc = []
    if route.legacy_alias:
        desc.append(f"Legacy alias (same-origin, un-prefixed): `{route.method} "
                    f"{route.path}`.")
    if route.v1_note:
        desc.append(route.v1_note)
    if route.mutating:
        desc.append("Mutating: a session snapshot is written after success.")
    if desc:
        op["description"] = " ".join(desc)
    if route.deprecated:
        op["deprecated"] = True
    if route.query:
        op["parameters"] = [
            {"name": q.name, "in": "query", "required": q.required,
             "description": q.description, "schema": {"type": "string"}}
            for q in route.query]
    if route.request is not None:
        op["requestBody"] = {"required": True, "content": {
            "application/json": {"schema": _ref(route.request)}}}
    op["responses"] = {
        "200": {"description": "OK",
                "content": {"application/json": {"schema": _ref(route.response)}}},
        "default": {"description": "error — `code` is machine-readable",
                    "content": {"application/json": {"schema": _ref(ErrorBody)}}},
    }
    op["x-legacy-alias"] = route.path if route.legacy_alias else None
    op["x-mutating"] = route.mutating
    return op


@lru_cache(maxsize=1)
def build_schema() -> dict[str, Any]:
    """The OpenAPI 3.1 document. Deterministic: no clock, no host, no config."""
    models: list[type[BaseModel]] = [ErrorBody]
    for r in ROUTES:
        for m in (r.request, r.response):
            if m is not None and m not in models:
                models.append(m)
    _, defs = models_json_schema([(m, "validation") for m in models],
                                 ref_template=_REF)
    paths: dict[str, dict[str, Any]] = {}
    for r in ROUTES:
        paths.setdefault(API_PREFIX + r.path, {})[r.method.lower()] = _operation(r)
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "pleroma loom API",
            "version": API_VERSION,
            "description": (
                "The loom inference server. Every route is served at "
                f"`{API_PREFIX}/<route>`; the un-prefixed paths are same-origin "
                "legacy aliases for loom_ui.html. Bearer auth applies when the "
                f"server has a token (--api-token / ${TOKEN_ENV}). See docs/API.md."),
        },
        "servers": [{"url": "/"}],
        "security": [{"bearer": []}, {}],
        "paths": paths,
        "components": {
            "schemas": defs.get("$defs", {}),
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
        },
    }


def schema_text() -> str:
    """The committed file's exact text."""
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


@lru_cache(maxsize=1)
def schema_sha256() -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace)."""
    canon = json.dumps(build_schema(), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="print (or --check) the /api/v1 schema")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if {SCHEMA_PATH.name} is not up to date")
    args = ap.parse_args(argv)
    text = schema_text()
    if args.check:
        try:
            current = SCHEMA_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read {SCHEMA_PATH}: {exc}", file=sys.stderr)
            return 1
        if current != text:
            print(f"{SCHEMA_PATH} is STALE — regenerate with "
                  "`python -m pleroma.serve.schema > pleroma/serve/api-schema.json`",
                  file=sys.stderr)
            return 1
        print(f"{SCHEMA_PATH.name} is up to date (sha256 {schema_sha256()})")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
