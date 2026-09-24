"""`python -m pleroma`: serve/worker render exactly the bridge's argv.

Runs on the public fixture profile + deployment (tests/fixtures/profiles,
tests/fixtures/deploy): the live 70B shape with placeholder paths.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from pleroma.cli import main
from pleroma.config import load_deployment, load_profile
from pleroma.config.legacy import legacy_env, legacy_serve_argv, legacy_worker_argv
from tests.unit.validate.test_fit_gates import BASE

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
PROFILE = FIXTURES / "profiles" / "modelc-format-70b.toml"
DEPLOY = FIXTURES / "deploy" / "modelc-format-70b.toml"


@pytest.mark.parametrize("role,render", [("serve", legacy_serve_argv),
                                         ("worker", legacy_worker_argv)])
def test_dry_run_prints_the_bridge_command(role, render, capsys) -> None:
    assert main([role, "--profile", str(PROFILE), "--deploy", str(DEPLOY), "--dry-run"]) == 0
    printed = shlex.split(capsys.readouterr().out.strip())
    profile = load_profile(PROFILE)
    want = render(profile, load_deployment(DEPLOY))
    # this profile names no anamnesis registry file: no env prefix
    assert legacy_env(profile) == {}
    assert printed == [sys.executable, "-u", *want]
    assert want[:2] == ["-m", "pleroma.serve.legacy" if role == "serve"
                        else "pleroma.harvest.worker"]


def test_a_bad_profile_is_a_clean_exit_not_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n')
    with pytest.raises(SystemExit, match="pleroma:"):
        main(["serve", "--profile", str(bad), "--deploy", str(DEPLOY), "--dry-run"])


def test_profile_show_prints_the_resolved_profile(capsys) -> None:
    assert main(["profile", "show", str(PROFILE)]) == 0
    assert '"injection_span": "uniform"' in capsys.readouterr().out


def test_validate_subcommand_delegates(tmp_path: Path, capsys) -> None:
    """`pleroma validate` is the battery's own CLI, reached through the one entry point."""
    report = tmp_path / "v1a_report.json"
    report.write_text(json.dumps(BASE))
    rc = main(["validate", "--profile", str(PROFILE), "--cv-report", str(report)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fit/heldout" in out and "PASS" in out


def test_validate_deploy_fills_inputs_without_overriding_explicit_ones() -> None:
    from pleroma.cli import _validate_argv
    argv = _validate_argv(["--profile", "p.toml", "--deploy", str(DEPLOY),
                           "--band", "mine.json"])
    assert "--deploy" not in argv
    assert argv[argv.index("--band") + 1] == "mine.json" and argv.count("--band") == 1
    assert argv[argv.index("--map") + 1].endswith("loom_map_modelc_v1a_full_r64.npz")
    assert argv[argv.index("--cv-report") + 1].endswith("v1a_report.json")
    assert argv[argv.index("--profile") + 1] == "p.toml" and argv.count("--profile") == 1


def test_validate_deploy_fills_the_profile_it_names(tmp_path) -> None:
    from pleroma.cli import _validate_argv
    from pleroma.config.profile import load_deployment
    argv = _validate_argv(["--deploy", str(DEPLOY)])
    name = load_deployment(DEPLOY).profile
    got = argv[argv.index("--profile") + 1]
    assert got.endswith(f"profiles/{name}.toml")
