"""Surface-area report: how much codebase there is to hold in a head.

Measures every ``.py`` and ``.md`` file under the given roots and reports,
grouped by top-level directory:

- **LOC** — physical lines.
- **code tokens** — Python tokens that carry program structure (names,
  operators, numbers, non-docstring strings), counted by the stdlib
  tokenizer. This is the reading load of the code itself.
- **doc words** — whitespace-separated words in comments, docstrings and
  Markdown files. This is the reading load of the prose.

The split matters because the two shrink by different means: code tokens by
consolidation and deletion, doc words by writing tighter documentation. A
single total would let one hide the other.

The report is informational and always exits 0 on a successful measurement;
it is a trend instrument, not a gate. ``--baseline`` prints deltas against an
earlier ``--json`` file so a receipt shows the direction of travel.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".eggs", "outputs"}

CODE_TOKEN_TYPES = frozenset({tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING})


class SurfaceError(RuntimeError):
    """A file or argument the report cannot proceed without."""


@dataclass
class FileMeasure:
    loc: int = 0
    code_tokens: int = 0
    doc_words: int = 0


@dataclass
class GroupMeasure:
    files: int = 0
    loc: int = 0
    code_tokens: int = 0
    doc_words: int = 0

    def add(self, m: FileMeasure) -> None:
        self.files += 1
        self.loc += m.loc
        self.code_tokens += m.code_tokens
        self.doc_words += m.doc_words


@dataclass
class Report:
    groups: dict[str, GroupMeasure] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    def total(self) -> GroupMeasure:
        t = GroupMeasure()
        for g in self.groups.values():
            t.files += g.files
            t.loc += g.loc
            t.code_tokens += g.code_tokens
            t.doc_words += g.doc_words
        return t

    def to_json(self) -> dict[str, object]:
        return {
            "groups": {
                name: {
                    "files": g.files,
                    "loc": g.loc,
                    "code_tokens": g.code_tokens,
                    "doc_words": g.doc_words,
                }
                for name, g in sorted(self.groups.items())
            },
            "total": {
                "files": self.total().files,
                "loc": self.total().loc,
                "code_tokens": self.total().code_tokens,
                "doc_words": self.total().doc_words,
            },
            "unreadable": self.unreadable,
        }


def docstring_starts(source: str) -> set[tuple[int, int]]:
    """(line, col) positions of docstring tokens: the first-statement string
    constant of a module, class or function body."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    starts: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            starts.add((first.value.lineno, first.value.col_offset))
    return starts


def measure_python(source: str) -> FileMeasure:
    """Token and word counts for one Python source text."""
    m = FileMeasure(loc=len(source.splitlines()))
    doc_starts = docstring_starts(source)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                m.doc_words += len(tok.string.lstrip("#").split())
            elif tok.type == tokenize.STRING and tok.start in doc_starts:
                m.doc_words += len(tok.string.split())
            elif tok.type in CODE_TOKEN_TYPES:
                m.code_tokens += 1
    except (tokenize.TokenError, IndentationError):
        # A file the tokenizer rejects still counts its lines; the parse
        # failure itself is surfaced through the caller's unreadable list.
        raise
    return m


def measure_markdown(text: str) -> FileMeasure:
    return FileMeasure(loc=len(text.splitlines()), doc_words=len(text.split()))


def iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".py", ".md"):
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            out.append(path)
    return out


def group_name(root: Path, path: Path) -> str:
    """Top-level directory under `root`, or the root's own name for direct children.

    A `root` that names one file measures that file alone, and `relative_to` then
    yields `.`, which has no parts at all. Such a measurement is grouped under the
    file's own parent, since one file admits no grouping below itself.
    """
    rel = path.relative_to(root)
    if not rel.parts:
        return f"{root.parent.name}/"
    if len(rel.parts) == 1:
        return f"{root.name}/"
    return f"{root.name}/{rel.parts[0]}/"


def run(roots: list[Path]) -> Report:
    report = Report()
    for root in roots:
        if not root.exists():
            raise SurfaceError(f"root does not exist: {root}")
        for path in iter_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                measure = measure_python(text) if path.suffix == ".py" else measure_markdown(text)
            except (OSError, tokenize.TokenError, IndentationError) as exc:
                report.unreadable.append(f"{path}: {exc}")
                continue
            report.groups.setdefault(group_name(root, path), GroupMeasure()).add(measure)
    return report


def print_table(report: Report, baseline: dict[str, object] | None) -> None:
    header = f"{'group':<40} {'files':>6} {'LOC':>8} {'code-tok':>10} {'doc-words':>10}"
    print("Surface report")
    print(header)
    for name, g in sorted(report.groups.items()):
        print(f"{name:<40} {g.files:>6} {g.loc:>8} {g.code_tokens:>10} {g.doc_words:>10}")
    t = report.total()
    print(f"{'TOTAL':<40} {t.files:>6} {t.loc:>8} {t.code_tokens:>10} {t.doc_words:>10}")
    if baseline is not None:
        base_total = baseline.get("total")
        if not isinstance(base_total, dict):
            raise SurfaceError("baseline file has no 'total' object")
        print("Against baseline:")
        for key, now in (
            ("files", t.files),
            ("loc", t.loc),
            ("code_tokens", t.code_tokens),
            ("doc_words", t.doc_words),
        ):
            before = base_total.get(key)
            if not isinstance(before, int):
                raise SurfaceError(f"baseline total lacks integer '{key}'")
            delta = now - before
            print(f"  {key}: {before} -> {now} ({delta:+d})")
    for line in report.unreadable:
        print(f"  unreadable: {line}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", required=True, type=Path, dest="roots",
                        help="directory (or file) to measure; repeatable")
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON here")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="a previous --json file; totals are printed as deltas against it")
    args = parser.parse_args(argv)

    baseline: dict[str, object] | None = None
    if args.baseline is not None:
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: baseline unreadable: {exc}", file=sys.stderr)
            return 2

    try:
        report = run(args.roots)
        print_table(report, baseline)
    except SurfaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json is not None:
        args.json.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"json written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
