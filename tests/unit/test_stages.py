"""The golden-path build stages: profile + deployment -> argv -> receipt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.cli import main
from pleroma.config.profile import BuildInputs, Deployment, PinnedPath
from pleroma.stages import StageError, receipt_path, render, run_stage, sha256_path
from tests.contract.build import _fixtures as F
from tests.contract.build import targets as T
from tests.contract.build.conftest import SITES
from tests.unit.validate.conftest import make_profile


@pytest.fixture(scope="module")
def bank(tmp_path_factory) -> F.Bank:
    return F.build_bank(tmp_path_factory.mktemp("stagebank"), n_prompts=80, z_dim=12,
                        bins_dim=T.BINS_DIM, sites=SITES, hidden=5)


def _profile():
    return make_profile(**{"map.sites": list(SITES), "map.rank": 8})


def _deploy(bank: F.Bank, out_dir: Path, **build) -> Deployment:
    b = dict(pairs_dir=bank.pairs_dir, levers=PinnedPath(path=bank.levers),
             hiddens=[PinnedPath(path=bank.hiddens)], bins=PinnedPath(path=bank.bins),
             out_dir=out_dir)
    b.update(build)
    return Deployment(profile="toy-modelc", map=PinnedPath(path=out_dir / "m.npz"),
                      discriminants=PinnedPath(path=bank.discriminants),
                      calib_dir=out_dir, work_dir=out_dir,
                      build=BuildInputs(**b))


def test_render_uses_the_profile_operating_point(bank: F.Bank, tmp_path: Path) -> None:
    argv, _, out = render("export", _profile(), _deploy(bank, tmp_path))
    assert argv[3] == "pleroma.map.build.export"
    for flag, val in (("--rank", "8"), ("--lam", "10000"), ("--fan-source", "member_fan"),
                      ("--target", "raw"), ("--discriminants", str(bank.discriminants))):
        assert argv[argv.index(flag) + 1] == val
    assert "--wide-map" not in argv
    assert out == tmp_path / "loom_map_toy-modelc_v1a_r8.npz"


def test_fit_without_a_shelf_map_is_refused(bank: F.Bank, tmp_path: Path) -> None:
    with pytest.raises(StageError, match="pleroma shelf"):
        render("fit", _profile(), _deploy(bank, tmp_path))


def test_a_pin_mismatch_refuses_before_running(bank: F.Bank, tmp_path: Path) -> None:
    d = _deploy(bank, tmp_path, levers=PinnedPath(path=bank.levers, sha256="0" * 64))
    with pytest.raises(StageError, match="pins"):
        run_stage("shelf", _profile(), d)
    assert not receipt_path(tmp_path / "shelf_map_toy-modelc_r8.npz").exists()


def test_a_stage_without_a_build_block_is_refused(bank: F.Bank, tmp_path: Path) -> None:
    d = _deploy(bank, tmp_path).model_copy(update={"build": None})
    with pytest.raises(StageError, match=r"\[build\]"):
        render("export", _profile(), d)


def test_shelf_then_export_runs_end_to_end_with_receipts(bank: F.Bank, tmp_path: Path) -> None:
    """The golden path on a synthetic bank: the shelf stage, then an export
    that never reads the shelf map, each leaving a receipt whose shas are real."""
    prof, d = _profile(), _deploy(bank, tmp_path)
    assert run_stage("shelf", prof, d, device="cpu") == 0
    shelf = tmp_path / "shelf_map_toy-modelc_r8.npz"
    rc = run_stage("export", prof, d, device="cpu", extra=["--selfcheck-floor", "0.01"])
    assert rc == 0
    out = tmp_path / "loom_map_toy-modelc_v1a_r8.npz"
    rec = json.loads(receipt_path(out).read_text())
    assert rec["returncode"] == 0 and rec["output_sha256"] == sha256_path(out)
    assert rec["argv"][-2:] == ["--selfcheck-floor", "0.01"]
    roles = {i["role"]: i["sha256"] for i in rec["inputs"]}
    assert roles["levers"] == sha256_path(bank.levers)
    assert "shelf_map" not in roles
    # the folded export carries the same standardisers the shelf fit banked
    with np.load(shelf) as s, np.load(out) as e:
        for k in ("v3_mu", "v3_sd", "bins_mu", "norm_ref"):
            assert np.array_equal(s[k], e[k]), k




def test_cli_dry_run_prints_the_stage_command(capsys) -> None:
    """On the public fixture profile + deployment (the live 70B's shape)."""
    rc = main(["export", "--profile", "tests/fixtures/profiles/modelc-format-70b.toml",
               "--deploy", "tests/fixtures/deploy/modelc-format-70b.toml", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "-m pleroma.map.build.export" in out and "--heldout-report" in out
