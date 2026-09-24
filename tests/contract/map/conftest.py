"""Shared fixtures for the map contract: one discriminants file, one served
(v1a-shaped) map loaded through the canonical loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from . import _fixtures as fx
from . import targets as T


@pytest.fixture
def disc(tmp_path: Path) -> tuple[Path, str]:
    return fx.write_discriminants(tmp_path / "disc")


@pytest.fixture
def v1a_path(tmp_path: Path, disc: tuple[Path, str]) -> Path:
    return fx.write_v1a_map(tmp_path / "map", disc[1])


@pytest.fixture
def v1a(v1a_path: Path, disc: tuple[Path, str]) -> "T.LoomMap":
    return T.LoomMap(v1a_path, disc[0])


@pytest.fixture
def wide(tmp_path: Path, disc: tuple[Path, str]) -> "T.LoomMap":
    """The v0 wide map as fit_loom_map ships it (dense W + factors, mu != 0)."""
    return T.LoomMap(fx.write_wide_map(tmp_path / "wide", disc[1]), disc[0])
