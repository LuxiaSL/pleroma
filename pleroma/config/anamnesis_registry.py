"""Pleroma's anamnesis model-registry file, and putting it on ``ANAMNESIS_MODELS``.

anamnesis resolves presets from a registry: the presets it ships plus
every file named in the ``ANAMNESIS_MODELS`` environment variable (``PATH``
syntax, merged additively; a file may add presets but never redefine a
shipped one). Presets an install needs that anamnesis does not ship live in
the install's own registry file, ``REGISTRY_FILE`` under ``profiles/``. That
file names the install's checkpoints, so it is local rather than part of the
package, and a tree without it resolves anamnesis' shipped presets only.

The variable is read by anamnesis at CALL time, and it is inherited by child
processes, which is the point: a subprocess (the harvest fallback, a pool
worker) resolves the same presets as its parent. An in-process mutation of
anamnesis' preset table would be invisible to every child.

Nothing here imports anamnesis, so this module is laptop-importable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any
from pathlib import Path

#: The variable anamnesis reads (``anamnesis.config.MODELS_ENV``), restated so
#: this module needs no anamnesis import; a test pins the two equal.
MODELS_ENV = "ANAMNESIS_MODELS"

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The install's registry file. It is local to an install (it names that
#: install's checkpoints) and may be absent; `resolve_preset` then loads only
#: anamnesis' shipped presets.
REGISTRY_FILE = REPO_ROOT / "profiles" / "anamnesis_models.json"


def _entries(value: str) -> list[str]:
    return [e.strip() for e in value.split(os.pathsep) if e.strip()]


def _same(a: str, b: Path) -> bool:
    try:
        return Path(a).expanduser().resolve() == b.resolve()
    except OSError:
        return False


def registry_env_value(path: Path = REGISTRY_FILE, current: str | None = None) -> str:
    """``ANAMNESIS_MODELS`` with ``path`` present exactly once (appended if absent).

    Order is preserved: files already named keep their place, so a caller's
    own extensions still merge in the order they chose.
    """
    path = Path(path)
    entries = _entries(current or "")
    if not any(_same(e, path) for e in entries):
        entries.append(str(path.resolve()))
    return os.pathsep.join(entries)


def ensure_registry(
    path: Path = REGISTRY_FILE, environ: MutableMapping[str, str] | None = None
) -> str:
    """Put ``path`` on ``ANAMNESIS_MODELS`` in this process (idempotent).

    Raises FileNotFoundError when the file is missing: a registry that silently
    did not load would make the preset lookup fail later with a less useful
    message, or — worse — resolve a different row.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"anamnesis registry file not found: {path}")
    env = os.environ if environ is None else environ
    value = registry_env_value(path, env.get(MODELS_ENV))
    env[MODELS_ENV] = value
    return value


def launch_env(registry: Path | None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment ADDITIONS a launched process needs for ``registry``.

    Empty when the profile names no registry (a preset anamnesis ships).
    """
    if registry is None:
        return {}
    reg = Path(registry)
    if not reg.is_absolute():
        reg = REPO_ROOT / reg
    src = os.environ if base is None else base
    return {MODELS_ENV: registry_env_value(reg, src.get(MODELS_ENV))}


def resolve_preset(name: Any) -> Any:
    """``anamnesis.config.resolve_preset`` with this registry on the path.

    Accepts a name or an ``anamnesis.config.ModelPreset``. Raises
    ``anamnesis.config.UnknownPresetError`` (a KeyError) for an unknown name,
    whose message lists every preset and the files they were read from.
    """
    # The file is local (it names this install's checkpoints); a tree without
    # it resolves anamnesis' shipped presets only.
    if REGISTRY_FILE.is_file():
        ensure_registry()
    from anamnesis.config import resolve_preset as _resolve

    return _resolve(name)
