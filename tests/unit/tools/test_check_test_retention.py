"""G2 test retention, measured over throwaway git repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.check_test_retention import (
    MeasurementError,
    RetentionReport,
    Snapshot,
    build_report,
    main,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x_{n} = {n}\n" for n in range(count)), encoding="utf-8")


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Code of 600 lines and a suite of 300: the shape of the real ratio."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "test")
    git(root, "config", "commit.gpgsign", "false")
    write_lines(root / "anamnesis" / "kept.py", 270)
    write_lines(root / "anamnesis" / "dead.py", 330)
    write_lines(root / "tests" / "test_kept.py", 134)
    write_lines(root / "tests" / "test_dead.py", 166)
    commit(root, "base")
    return root


def report(repo: Path, base: str = "main", head: str = "HEAD") -> RetentionReport:
    return build_report(repo, base, head)


def branch(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "change")


def test_counts_suite_lines_and_code_lines(repo: Path) -> None:
    result = report(repo)
    assert (result.base.test_loc, result.base.code_loc) == (300, 600)
    assert result.head == result.base
    assert result.passed


def test_deleting_code_with_its_tests_passes(repo: Path) -> None:
    branch(repo)
    (repo / "anamnesis" / "dead.py").unlink()
    (repo / "tests" / "test_dead.py").unlink()
    commit(repo, "delete dead code with its tests")
    result = report(repo)
    assert (result.test_delta, result.code_delta) == (-166, -330)
    assert result.passed


def test_code_outside_the_package_counts_as_code(repo: Path) -> None:
    write_lines(repo / "tools" / "checker.py", 200)
    write_lines(repo / "tests" / "test_checker.py", 100)
    commit(repo, "a tool and its tests")
    branch(repo)
    (repo / "tools" / "checker.py").unlink()
    (repo / "tests" / "test_checker.py").unlink()
    commit(repo, "retire the tool with its tests")
    result = report(repo)
    assert (result.test_delta, result.code_delta) == (-100, -200)
    assert result.passed


def test_deleting_tests_while_the_code_stays_fails(repo: Path) -> None:
    branch(repo)
    (repo / "tests" / "test_dead.py").unlink()
    commit(repo, "delete tests only")
    result = report(repo)
    assert (result.test_delta, result.code_delta) == (-166, 0)
    assert not result.passed
    assert "the code did not shrink; restore the tests" in result.reason()


def test_deleting_tests_while_adding_code_fails(repo: Path) -> None:
    branch(repo)
    write_lines(repo / "tests" / "test_kept.py", 100)
    write_lines(repo / "anamnesis" / "new.py", 50)
    commit(repo, "trade tests for code")
    result = report(repo)
    assert not result.passed
    assert "grew by 50" in result.reason()


def test_suite_shrinking_more_than_the_code_fails(repo: Path) -> None:
    branch(repo)
    write_lines(repo / "anamnesis" / "dead.py", 300)
    write_lines(repo / "tests" / "test_dead.py", 130)
    commit(repo, "trim")
    result = report(repo)
    assert (result.test_delta, result.code_delta) == (-36, -30)
    assert not result.passed


def test_equal_shrinkage_is_the_boundary_and_passes(repo: Path) -> None:
    branch(repo)
    write_lines(repo / "anamnesis" / "dead.py", 294)
    write_lines(repo / "tests" / "test_dead.py", 130)
    commit(repo, "trim evenly")
    result = report(repo)
    assert (result.test_delta, result.code_delta) == (-36, -36)
    assert result.passed


def test_adding_code_without_tests_is_not_this_gate(repo: Path) -> None:
    branch(repo)
    write_lines(repo / "anamnesis" / "new.py", 500)
    commit(repo, "add code")
    assert report(repo).passed


def test_non_python_and_cache_files_do_not_count(repo: Path) -> None:
    branch(repo)
    (repo / "tests" / "test_dead.py").unlink()
    write_lines(repo / "tests" / "fixture.txt", 500)
    write_lines(repo / "tests" / "__pycache__" / "stale.py", 500)
    commit(repo, "swap a test for data")
    result = report(repo)
    assert result.head.test_loc == 134
    assert not result.passed


def test_untracked_files_count_on_neither_side(repo: Path) -> None:
    write_lines(repo / "tests" / "test_scratch.py", 999)
    result = report(repo)
    assert result.head.test_loc == 300


def test_base_is_the_merge_base_so_a_stale_branch_answers_only_for_itself(repo: Path) -> None:
    branch(repo)
    write_lines(repo / "tests" / "test_new.py", 10)
    commit(repo, "the change adds a test")
    git(repo, "checkout", "-q", "main")
    (repo / "tests" / "test_dead.py").unlink()
    (repo / "anamnesis" / "dead.py").unlink()
    commit(repo, "main moves on")
    git(repo, "checkout", "-q", "change")
    result = report(repo)
    assert result.base.test_loc == 300
    assert (result.test_delta, result.code_delta) == (10, 0)
    assert result.passed


def test_unknown_base_is_refused(repo: Path) -> None:
    with pytest.raises(MeasurementError, match="cannot resolve 'nowhere'"):
        report(repo, base="nowhere")


def test_missing_repo_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MeasurementError):
        build_report(tmp_path / "absent", "main")


def test_ratio_is_undefined_without_code() -> None:
    assert Snapshot(revision="x", test_loc=5, code_loc=0).ratio is None


def test_cli_exit_codes_and_receipt(repo: Path, tmp_path: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    branch(repo)
    (repo / "tests" / "test_dead.py").unlink()
    commit(repo, "delete tests only")
    receipt = tmp_path / "out" / "g2.json"
    argv = ["--repo", str(repo), "--base", base, "--json", str(receipt)]
    assert main(argv) == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G2"
    assert (payload["test_delta"], payload["code_delta"], payload["passed"]) == (-166, 0, False)

    (repo / "anamnesis" / "dead.py").unlink()
    commit(repo, "and the code they covered")
    assert main(argv) == 0
    assert main(["--repo", str(repo), "--base", "nowhere"]) == 2
