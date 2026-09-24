#!/usr/bin/env python3
"""G2 — test retention: a change does not shrink the suite on its own.

The rule, per change
--------------------
Count physical lines in the Python files under ``tests/`` (the suite) and in
every other Python file (the code: the package, ``tools/``, anything else the
suite covers), once at the base the change starts from and once at its head.
Then:

- If the suite did not shrink, the change passes.
- If the suite shrank, the code must have shrunk by **at least as many lines**.
  Otherwise the change fails.

So deleting a module together with its tests passes: the suite runs at roughly
half the size of the code, which means code removed with its own tests takes
about two code lines out for every test line, and the rule leaves that margin
in hand. What fails is removing tests while the code they covered stays,
or removing tests while adding code. If your change fails, either the tests
still describe code that exists — restore them — or the code they covered is
gone too, in which case delete it in the same change. Consolidating duplicate
tests without touching the code is a legitimate change this rule cannot tell
apart from dropping coverage; that is what a written waiver beside the receipt
is for.

The rule compares the change with its own base rather than with a pinned
number, so there is no floor that ages below the suite and no baseline file to
update. The base is the merge base of ``--base`` and ``--head``, which makes a
branch that has fallen behind ``main`` answer for its own lines only.

LOC is physical lines, as ``wc -l`` counts them, read from git objects so the
two sides are measured the same way and an untracked scratch file counts on
neither. Directories named ``__pycache__`` are skipped.

Usage
-----
    python -m tools.check_test_retention --base main
    python -m tools.check_test_retention --base <sha> --head HEAD --json receipt.json

Exit status: 0 pass, 1 fail, 2 the measurement could not be made.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

TESTS_ROOT = "tests"
SKIP_PARTS = frozenset({"__pycache__"})


class MeasurementError(RuntimeError):
    """A revision or repository the measurement cannot be made over."""


@dataclass(frozen=True)
class Snapshot:
    """Line counts for one revision."""

    revision: str
    test_loc: int
    code_loc: int

    @property
    def ratio(self) -> float | None:
        return None if self.code_loc == 0 else self.test_loc / self.code_loc


@dataclass(frozen=True)
class RetentionReport:
    """One change measured against its base, and the verdict."""

    repo: str
    base: Snapshot
    head: Snapshot

    @property
    def test_delta(self) -> int:
        return self.head.test_loc - self.base.test_loc

    @property
    def code_delta(self) -> int:
        return self.head.code_loc - self.base.code_loc

    @property
    def passed(self) -> bool:
        if self.test_delta >= 0:
            return True
        return -self.code_delta >= -self.test_delta

    def reason(self) -> str:
        """One sentence a contributor can act on."""
        if self.test_delta >= 0:
            return "the suite did not shrink"
        removed_tests = -self.test_delta
        removed_code = -self.code_delta
        if self.passed:
            return (
                f"the suite shrank by {removed_tests} lines and the code by "
                f"{removed_code}, at least as many"
            )
        if removed_code < 0:
            code_phrase = f"the code grew by {-removed_code} lines"
        elif removed_code == 0:
            code_phrase = "the code did not shrink"
        else:
            code_phrase = f"the code shrank by only {removed_code}"
        return (
            f"the suite shrank by {removed_tests} lines while {code_phrase}; "
            "restore the tests, or delete the code they covered in the same change"
        )

    def to_json(self) -> dict[str, object]:
        def side(snapshot: Snapshot) -> dict[str, object]:
            return {
                "revision": snapshot.revision,
                "test_loc": snapshot.test_loc,
                "code_loc": snapshot.code_loc,
                "test_to_code_ratio": snapshot.ratio,
            }

        return {
            "gate": "G2",
            "check": "test retention",
            "repo": self.repo,
            "tests_root": TESTS_ROOT,
            "base": side(self.base),
            "head": side(self.head),
            "test_delta": self.test_delta,
            "code_delta": self.code_delta,
            "reason": self.reason(),
            "passed": self.passed,
        }


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=stdin,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise MeasurementError("git is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise MeasurementError(f"git {' '.join(args)}: {detail}") from exc
    return completed.stdout


def resolve(repo: Path, revision: str) -> str:
    """The full commit id a revision names, or a refusal saying which one."""
    try:
        return _git(repo, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}").decode().strip()
    except MeasurementError as exc:
        raise MeasurementError(
            f"cannot resolve {revision!r} to a commit; fetch it, or pass --base "
            "with a revision this clone has"
        ) from exc


def merge_base(repo: Path, base: str, head: str) -> str:
    return _git(repo, "merge-base", base, head).decode().strip()


def python_blobs(repo: Path, commit: str) -> tuple[list[str], list[str]]:
    """Object ids of the `.py` files at `commit`: those under the suite, and the rest."""
    listing = _git(repo, "ls-tree", "-r", "-z", commit)
    tests: list[str] = []
    code: list[str] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, _, raw_path = entry.partition(b"\t")
        fields = meta.split()
        if len(fields) != 3 or fields[1] != b"blob":
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        parts = path.split("/")
        if not path.endswith(".py") or SKIP_PARTS.intersection(parts):
            continue
        (tests if parts[0] == TESTS_ROOT else code).append(fields[2].decode())
    return tests, code


def count_blob_lines(repo: Path, blobs: Sequence[str]) -> int:
    """Summed newline count of the given blobs, in one git process."""
    if not blobs:
        return 0
    payload = _git(repo, "cat-file", "--batch", stdin=("\n".join(blobs) + "\n").encode())
    total = 0
    offset = 0
    for _ in blobs:
        header_end = payload.index(b"\n", offset)
        header = payload[offset:header_end].split()
        if len(header) != 3:
            raise MeasurementError(f"unexpected git cat-file header: {header!r}")
        size = int(header[2])
        start = header_end + 1
        total += payload.count(b"\n", start, start + size)
        offset = start + size + 1
    return total


def snapshot(repo: Path, commit: str) -> Snapshot:
    tests, code = python_blobs(repo, commit)
    return Snapshot(
        revision=commit,
        test_loc=count_blob_lines(repo, tests),
        code_loc=count_blob_lines(repo, code),
    )


def build_report(repo: Path, base: str, head: str = "HEAD") -> RetentionReport:
    """Measure `head` against the merge base it shares with `base`."""
    repo = repo.resolve()
    if not repo.is_dir():
        raise MeasurementError(f"--repo is not a directory: {repo}")
    head_commit = resolve(repo, head)
    base_commit = merge_base(repo, resolve(repo, base), head_commit)
    return RetentionReport(
        repo=str(repo),
        base=snapshot(repo, base_commit),
        head=snapshot(repo, head_commit),
    )


def print_report(report: RetentionReport, stream: TextIO | None = None) -> None:
    out = sys.stdout if stream is None else stream

    def ratio(snap: Snapshot) -> str:
        return "n/a" if snap.ratio is None else f"{snap.ratio:.3f}"

    out.write("G2 test retention\n")
    out.write(f"  repo: {report.repo}\n")
    for label, snap in (("base", report.base), ("head", report.head)):
        out.write(
            f"  {label}: {snap.revision[:12]}  tests {snap.test_loc} LOC, "
            f"code {snap.code_loc} LOC, ratio {ratio(snap)}\n"
        )
    out.write(f"  change: tests {report.test_delta:+d}, code {report.code_delta:+d}\n")
    out.write(f"  {report.reason()}\n")
    out.write(f"  verdict: {'PASS' if report.passed else 'FAIL'}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_test_retention",
        description=(
            "G2: a change that shrinks the test suite must shrink the code by at "
            "least as many lines."
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository to measure")
    parser.add_argument(
        "--base",
        default="main",
        help="revision the change starts from; its merge base with --head is used",
    )
    parser.add_argument("--head", default="HEAD", help="revision holding the change")
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    args = parser.parse_args(argv)

    try:
        report = build_report(args.repo, args.base, args.head)
    except MeasurementError as exc:
        print(f"check_test_retention: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  receipt: {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
