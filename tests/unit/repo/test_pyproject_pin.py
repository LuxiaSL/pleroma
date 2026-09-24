"""The public package's anamnesis pin agrees with the code's and the submodule's.

anamnesis is recorded in three places: the pyproject dependency (what an
install gets), `pleroma.harvest.anamnesis_seam.PINNED_COMMIT` (what the
startup check demands), and the `vendor/anamnesis` gitlink (the readable
source). A pin bumped in one place and not the others would install one
extractor and refuse to start on it, or worse, fit maps in a space no install
reproduces. (The gitlink half is `tests/unit/harvest/test_anamnesis_seam.py`.)
"""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path

import pytest

OVERLAY = Path(__file__).resolve().parents[3]
DEP = re.compile(r"^anamnesis\s*@\s*git\+(?P<url>https://\S+?)@(?P<sha>[0-9a-f]{40})$")


def _anamnesis_dep() -> re.Match[str]:
    with (OVERLAY / "pyproject.toml").open("rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    hits = [m for d in deps if (m := DEP.match(d.strip()))]
    assert len(hits) == 1, f"expected one pinned anamnesis dependency, got {deps}"
    return hits[0]


def test_pyproject_pin_is_the_code_pin() -> None:
    seam = pytest.importorskip("pleroma.harvest.anamnesis_seam")
    assert _anamnesis_dep()["sha"] == seam.PINNED_COMMIT


def test_pyproject_url_is_the_submodule_url() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(OVERLAY / ".gitmodules")
    url = cfg['submodule "vendor/anamnesis"']["url"]
    assert _anamnesis_dep()["url"] == url.removesuffix(".git")
