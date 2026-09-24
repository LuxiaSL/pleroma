"""Session fixtures: one synthetic corpus per scale, each stage run once.

* ``small`` (1,281 pairs, z 12, bins 10, 4 sites x 5): big enough for
  ``v1a_fit``'s ``n >= 1000`` join gate, small enough for fit_loom_map's numpy
  ridge on a slow BLAS. Runs the real chain
  fit_loom_map -> v1a_fit.
* ``export`` (8,635 pairs, z 72, bins 420, 4 sites x 20): big enough for
  ``v1a_export.MIN_ROWS = 8000`` under the default ``lever_group`` join, with
  min(W.shape) = 72 > 64 so the registered rank-64 truncation is real. Its
  wide map is written in fit_loom_map's format by ``_fixtures.write_wide_map``
  (twin-tested against the real writer).

Everything is CPU, offline, and built in a session tmp dir.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T

SITES = [3, 5, 7, 9]      # four sites: v1a_fit hardcodes 4 (see test_v1a_fit_cv)


@dataclass
class Run:
    rc: int
    out: Path
    report: dict[str, Any] | None


@pytest.fixture(scope="session")
def small(tmp_path_factory) -> F.Bank:
    return F.build_bank(tmp_path_factory.mktemp("small"), n_prompts=80, z_dim=12,
                        bins_dim=10, sites=SITES, hidden=5)


@pytest.fixture(scope="session")
def small_wide(small: F.Bank, tmp_path_factory) -> Path:
    """The real fit_loom_map at the live wide-map point (1e4 / r8 < min(W) = 20)."""
    out = tmp_path_factory.mktemp("small_wide") / "wide.npz"
    rc = T.run_fit_loom_map([
        "--pairs-dir", small.pairs_dir, "--levers", small.levers, "--bins", small.bins,
        "--discriminants", small.discriminants, "--norm-ref-bank", small.levers,
        "--lam", "10000", "--rank", "8", "--out", out])
    assert rc == 0
    return out


@pytest.fixture(scope="session")
def small_fit(small: F.Bank, small_wide: Path, tmp_path_factory) -> Path:
    """v1a_fit's registered CV on the small bank, shelf = the real fit_loom_map map."""
    out_dir = tmp_path_factory.mktemp("small_fit")
    rc = T.run_v1a_fit([
        "--pairs-dir", small.pairs_dir, "--levers", small.levers,
        "--hiddens", small.hiddens, "--bins", small.bins, "--shelf-map", small_wide,
        "--out-dir", out_dir, "--device", "cpu"])
    assert rc == 0
    return out_dir


@pytest.fixture(scope="session")
def export_bank(tmp_path_factory) -> F.Bank:
    # 540 prompts x 2 waves = 1,080 fans x 8; every 20th fan is a [7,1] no-lever
    # fan (54 fans, 432 gens); one lever fan keeps only 2 pairs.
    return F.build_bank(tmp_path_factory.mktemp("exportbank"), n_prompts=540,
                        z_dim=72, bins_dim=T.BINS_DIM, sites=SITES, hidden=20,
                        nolever_every=20, seed=1, write_signatures=False)


@pytest.fixture(scope="session")
def export_wide(export_bank: F.Bank) -> Path:
    return F.write_wide_map(export_bank, export_bank.root / "wide.npz")


def export_argv(bank: F.Bank, wide: Path, out: Path, *extra: Any,
                hiddens: Path | None = None, z_dim: int | None = None) -> list[Any]:
    argv: list[Any] = [
        "--pairs-dir", bank.pairs_dir, "--levers", bank.levers,
        "--hiddens", hiddens or bank.hiddens, "--bins", bank.bins,
        "--wide-map", wide, "--device", "cpu", "--out", out]
    if z_dim is not None:
        argv += ["--z-dim", z_dim]
    return argv + list(extra)


def run_export(argv: list[Any], out: Path) -> Run:
    rc = T.run_v1a_export(argv)
    rpath = out.with_name(out.stem + "_export_report.json")
    report = json.loads(rpath.read_text()) if rpath.exists() else None
    return Run(rc=rc, out=out, report=report)


@pytest.fixture(scope="session")
def export_default(export_bank: F.Bank, export_wide: Path, tmp_path_factory) -> Run:
    """The live invocation shape minus ``--fan-source member_fan``."""
    out = tmp_path_factory.mktemp("export_default") / "v1a.npz"
    return run_export(export_argv(export_bank, export_wide, out, z_dim=export_bank.z_dim), out)


@pytest.fixture(scope="session")
def export_member_fan(export_bank: F.Bank, export_wide: Path, tmp_path_factory) -> Run:
    """The live served-map invocation: ``--fan-source member_fan``."""
    out = tmp_path_factory.mktemp("export_mf") / "v1a_mf.npz"
    return run_export(export_argv(export_bank, export_wide, out, "--fan-source",
                                  "member_fan", z_dim=export_bank.z_dim), out)


def npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}
