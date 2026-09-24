"""Reading gate inputs without tracebacks: a missing or unparseable input is a
named `InputError`, which each gate turns into INCONCLUSIVE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """An input the gate needs is absent or unreadable. The message is the
    named reason the gate prints."""


def read_json(path: str | Path, what: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise InputError(f"{what} not found: {p}")
    if not p.is_file():
        raise InputError(f"{what} is not a file: {p}")
    try:
        text = p.read_text()
    except OSError as exc:
        raise InputError(f"{what} unreadable ({p}): {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{what} is not valid JSON ({p}): {exc}") from exc


def read_json_or_jsonl(path: str | Path, what: str) -> Any:
    """JSON, or JSON Lines (one object/string per line) when the file is not a
    single JSON document."""
    p = Path(path)
    try:
        return read_json(p, what)
    except InputError as first:
        if not p.is_file():
            raise
        rows: list[Any] = []
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise InputError(f"{what} is neither JSON nor JSON Lines ({p}, line {n})") from first
        return rows


def sha256_file(path: str | Path) -> str | None:
    """sha256 of a file, or None when it cannot be read (receipt metadata only)."""
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def finite_float(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{what} must be a number, got {value!r}")
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise InputError(f"{what} is not finite: {value!r}")
    return f
