#!/usr/bin/env python3
"""G3 — documentation timelessness: code says what is true now, not what changed.

The gate targets prose a reader is meant to believe. Three surfaces carry it, and
`documentation_text` is where the boundary is drawn:

  * **Comments**, from the tokenizer.
  * **Docstrings**, from the parse tree — the first-statement string of a module,
    class or function.
  * **The message a program says out loud**: the string arguments of a `raise`
    and of a logging call. A stranger meets that text at the moment something
    goes wrong, which is the worst moment to hand them a pointer they cannot
    follow or a date that explains nothing.

Every other string literal is data — a fixture, a dict key, a filename, a JSON
payload, a report line a run writes — and data is free to carry a date or a
phrase that prose may not; a checker that policed it would cry wolf on ported
code. Inside an f-string the substituted expressions are code, not prose, so the
braces and their contents are blanked before the rules read the line.

The rules, all scoped to that boundary:

  1. **Marker comments.** A comment opening with one of the three deferral
     markers is a note to a future reader that the code does not keep.
  2. **Changelog phrasing.** Prose that narrates an edit rather than the state.
     The phrase set lives in `CHANGELOG_RULES`.
  3. **Dates.** A date in prose, anywhere on the three surfaces. Dates belong in
     the record, not in the code: git carries when a line changed, and a dated
     measurement or decision is a document's job rather than a comment's. The
     allowlist beside this checker holds the few idioms where a date is part of
     something a reader can follow rather than a citation of its own; it is one
     regex per line, versioned in the repo, because that judgment accumulates
     rather than closing.

The patterns are written with the last character of each phrase bracketed
(`previousl[y]`), so this file does not match itself and can be scanned by the
same gate it implements.

The past-habitual rule carries a guard: the phrase is flagged only when a word
follows it and no passive auxiliary precedes it on the same line. In the passive
voice the same two words mean "for the purpose of" and state a present-tense
constraint, which passes; said of a subject they narrate a state the code has
left behind, which does not.

Usage
-----
    python tools/check_timelessness.py --root anamnesis tools --json g3.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TextIO

DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "timelessness_allowlist.txt"

SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"})

MARKER_RULE = "marker-comment"
DATED_RULE = "dated-prose"

MARKER_RE = re.compile(r"#\s*(TOD[O]|FIXM[E]|HAC[K])\b")
#: A calendar date, wherever prose puts it. The pattern carries no comment marker,
#: because a docstring and a raised message are prose as much as a `#` line is: the
#: surface is chosen by `documentation_text`, not by this rule.
DATE_RE = re.compile(r"(?:19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}")
USED_TO_RE = re.compile(r"\bused t[o]\s+([A-Za-z]+)", re.IGNORECASE)
PASSIVE_AUXILIARY_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|get|gets|got|become|becomes)\b", re.IGNORECASE
)

CHANGELOG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("changelog-prior-state", re.compile(r"\bpreviousl[y]\b", re.IGNORECASE)),
    ("changelog-changed-in", re.compile(r"\bchanged i[n]\b", re.IGNORECASE)),
    ("changelog-we-now", re.compile(r"\bwe no[w]\b", re.IGNORECASE)),
    ("changelog-no-longer", re.compile(r"\bno longe[r]\b", re.IGNORECASE)),
)

USED_TO_RULE = "changelog-used-to"


class TimelessnessError(RuntimeError):
    """A condition that makes the check unrunnable rather than failing."""


@dataclass(frozen=True)
class Violation:
    """One flagged line."""

    rule: str
    file: str
    line: int
    match: str
    text: str


@dataclass
class TimelessnessReport:
    """The result of a timelessness pass, serialisable as the G3 receipt."""

    roots: list[str]
    allowlist: str
    allowlist_size: int
    files_scanned: int = 0
    violations: list[Violation] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)
    exempted: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations and not self.read_errors

    def counts_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.rule] = counts.get(violation.rule, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> dict[str, object]:
        return {
            "gate": "G3",
            "roots": self.roots,
            "allowlist": self.allowlist,
            "allowlist_size": self.allowlist_size,
            "counts": {
                "files_scanned": self.files_scanned,
                "violations": len(self.violations),
                "by_rule": self.counts_by_rule(),
                "exempted_dates": len(self.exempted),
                "read_errors": len(self.read_errors),
            },
            "violations": [vars(v) for v in self.violations],
            "exempted_dates": [vars(v) for v in self.exempted],
            "read_errors": self.read_errors,
            "passed": self.passed,
        }


def load_allowlist(path: Path) -> list[re.Pattern[str]]:
    """Compile the date allowlist: one regex per line, `#` comments.

    A line matching one of these carries a date as part of something a reader can
    follow, rather than as a citation of its own. The shipped file is short on
    purpose; what earns an entry is stated at its top.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TimelessnessError(f"allowlist unreadable: {path} ({exc})") from exc
    patterns: list[re.Pattern[str]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            patterns.append(re.compile(stripped, re.IGNORECASE))
        except re.error as exc:
            raise TimelessnessError(f"{path}:{number}: invalid regex {stripped!r} ({exc})") from exc
    return patterns


def iter_python_files(root: Path) -> list[Path]:
    """Every `.py` file under `root`, excluding caches and virtual environments."""
    if root.is_file():
        return [root]
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

#: Method names that make an attribute call a logging call. A message handed to one
#: of these is read by a stranger, in a log, with nothing else to go on.
LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
)


@dataclass(frozen=True)
class Span:
    """One run of prose on one line, as a half-open column range."""

    line: int
    start: int
    end: int
    #: True when the span is an f-string, whose `{...}` substitutions are code.
    fstring: bool = False


def _char_column(line: str, byte_column: int) -> int:
    """An `ast` column (a UTF-8 byte offset) as an index into the line's characters.

    The parse tree counts bytes and the tokenizer counts characters. A docstring
    holding `─` or `α` before a column would put the two out of step, which would
    slice prose in the wrong place, so byte columns are converted rather than used.
    """
    if byte_column <= 0:
        return 0
    raw = line.encode("utf-8")
    if byte_column >= len(raw):
        return len(line)
    return len(raw[:byte_column].decode("utf-8", errors="ignore"))


def _node_spans(node: ast.expr, lines: Sequence[str]) -> list[Span]:
    """The columns a string node occupies, one span per line it covers."""
    first = node.lineno
    last = node.end_lineno or first
    is_fstring = isinstance(node, ast.JoinedStr)
    spans: list[Span] = []
    for number in range(first, last + 1):
        if number - 1 >= len(lines):
            break
        text = lines[number - 1]
        start = _char_column(text, node.col_offset) if number == first else 0
        if number == last and node.end_col_offset is not None:
            end = _char_column(text, node.end_col_offset)
        else:
            end = len(text)
        if end > start:
            spans.append(Span(line=number, start=start, end=end, fstring=is_fstring))
    return spans


def _string_literals(node: ast.expr) -> list[ast.expr]:
    """Every string literal an expression is built out of, or none.

    A message is often assembled rather than written whole: two literals joined with
    `+`, a `%` template, a `.format` on a literal, one of two literals chosen by a
    condition. Each piece is still prose a reader will see, so the walk goes through
    those forms and stops at anything else — a name, a subscript, a call on a
    variable — which is data this checker cannot read.
    """
    if isinstance(node, ast.JoinedStr):
        return [node]
    if isinstance(node, ast.Constant):
        return [node] if isinstance(node.value, str) else []
    if isinstance(node, ast.BinOp):
        return _string_literals(node.left) + _string_literals(node.right)
    if isinstance(node, ast.IfExp):
        return _string_literals(node.body) + _string_literals(node.orelse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            return _string_literals(node.func.value)
    return []


def _message_nodes(call: ast.Call) -> list[ast.expr]:
    """The string literals a call's positional arguments are built out of.

    Positional only: a keyword argument is as often a path or a code as a sentence.
    """
    return [node for arg in call.args for node in _string_literals(arg)]


def _prose_nodes(tree: ast.Module) -> list[ast.expr]:
    """Every string node that is prose: a docstring, or a message said out loud."""
    nodes: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    nodes.append(value)
        elif isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Call):
                nodes.extend(_message_nodes(node.exc))
            elif node.exc is not None:
                nodes.extend(_string_literals(node.exc))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS:
                nodes.extend(_message_nodes(node))
    return nodes


def _blank_substitutions(chars: list[str], start: int, end: int) -> None:
    """Blank every `{...}` inside an f-string span: a substitution is code, not prose.

    Brace depth is tracked so a nested format spec goes with its expression, and
    `{{`/`}}` are literal braces the prose keeps. An unclosed brace blanks to the
    end of the span, which is the safe direction: code read as prose is a false
    positive, and a gate that cries wolf gets switched off.
    """
    depth = 0
    index = start
    while index < end:
        char = chars[index]
        if depth == 0:
            if char == "{":
                if index + 1 < end and chars[index + 1] == "{":
                    index += 2
                    continue
                depth = 1
                chars[index] = " "
            elif char == "}" and index + 1 < end and chars[index + 1] == "}":
                index += 2
                continue
        else:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            chars[index] = " "
        index += 1


def documentation_text(path: Path, source: str) -> dict[int, str]:
    """Each prose-bearing line, with every column that is not prose blanked out.

    Comments come from the tokenizer; docstrings and the message arguments of a
    `raise` or a logging call come from the parse tree. Blanking rather than
    extracting keeps every column where it was, so a reported line number and a
    match still land on what the reader sees.

    Every other string literal is data: a fixture, a dict key, a filename, a JSON
    payload, a line a run writes into a report. Data may legitimately contain a
    date or a phrase this checker forbids in prose, so the rules stop at this
    boundary.
    """
    lines = source.splitlines()
    spans: list[Span] = []
    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                spans.append(Span(line=token.start[0], start=token.start[1], end=token.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise TimelessnessError(f"{path}: tokenize failed ({exc})") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise TimelessnessError(f"{path}: syntax error at line {exc.lineno} ({exc.msg})") from exc
    for node in _prose_nodes(tree):
        spans.extend(_node_spans(node, lines))

    masked: dict[int, list[str]] = {}
    for span in spans:
        if span.line - 1 >= len(lines):
            continue
        original = lines[span.line - 1]
        chars = masked.setdefault(span.line, [" "] * len(original))
        end = min(span.end, len(original))
        for index in range(max(span.start, 0), end):
            chars[index] = original[index]
        if span.fstring:
            _blank_substitutions(chars, max(span.start, 0), end)
    return {number: "".join(chars) for number, chars in sorted(masked.items())}


def used_to_violation(text: str) -> str | None:
    """The matched phrase when `used to` narrates an edit, else None."""
    match = USED_TO_RE.search(text)
    if match is None:
        return None
    prefix = text[: match.start()]
    if PASSIVE_AUXILIARY_RE.search(prefix):
        return None
    return match.group(0)


def scan_source(path: Path, source: str, allowlist: Sequence[re.Pattern[str]]) -> tuple[list[Violation], list[Violation]]:
    """Flag one file; returns (violations, exempted dates).

    The rules read the masked prose; what a violation quotes is the line as
    written, because that is what the reader has to go and fix.
    """
    prose = documentation_text(path, source)
    violations: list[Violation] = []
    exempted: list[Violation] = []
    file_str = str(path)
    for number, raw in enumerate(source.splitlines(), start=1):
        if number not in prose:
            continue
        text = prose[number]
        marker = MARKER_RE.search(text)
        if marker is not None:
            violations.append(
                Violation(rule=MARKER_RULE, file=file_str, line=number, match=marker.group(0), text=raw.strip())
            )
        for rule, pattern in CHANGELOG_RULES:
            found = pattern.search(text)
            if found is not None:
                violations.append(
                    Violation(rule=rule, file=file_str, line=number, match=found.group(0), text=raw.strip())
                )
        phrase = used_to_violation(text)
        if phrase is not None:
            violations.append(
                Violation(rule=USED_TO_RULE, file=file_str, line=number, match=phrase, text=raw.strip())
            )
        dated = DATE_RE.search(text)
        if dated is not None:
            record = Violation(
                rule=DATED_RULE, file=file_str, line=number, match=dated.group(0), text=raw.strip()
            )
            if any(pattern.search(text) for pattern in allowlist):
                exempted.append(record)
            else:
                violations.append(record)
    return violations, exempted


def build_report(
    roots: Sequence[Path],
    allowlist_path: Path = DEFAULT_ALLOWLIST,
) -> TimelessnessReport:
    """Run the check over every `.py` file under the given roots."""
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if not candidate.exists():
            raise TimelessnessError(f"--root does not exist: {candidate}")
        resolved.append(candidate)
    if not resolved:
        raise TimelessnessError("at least one --root is required")

    allowlist = load_allowlist(allowlist_path)
    report = TimelessnessReport(
        roots=[str(r) for r in resolved],
        allowlist=str(allowlist_path.resolve()),
        allowlist_size=len(allowlist),
    )

    seen: set[Path] = set()
    for root in resolved:
        for path in iter_python_files(root):
            if path in seen:
                continue
            seen.add(path)
            report.files_scanned += 1
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                report.read_errors.append(f"{path}: unreadable ({exc})")
                continue
            try:
                violations, exempted = scan_source(path, source, allowlist)
            except TimelessnessError as exc:
                report.read_errors.append(str(exc))
                continue
            report.violations.extend(violations)
            report.exempted.extend(exempted)
    report.violations.sort(key=lambda v: (v.file, v.line, v.rule))
    report.exempted.sort(key=lambda v: (v.file, v.line))
    return report


def print_report(report: TimelessnessReport, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G3 documentation timelessness")
    line(f"  roots: {', '.join(report.roots)}")
    line(f"  allowlist: {report.allowlist} ({report.allowlist_size} patterns)")
    line(f"  files scanned: {report.files_scanned}")
    line(f"  dates exempted by the allowlist: {len(report.exempted)}")
    line(f"  violations: {len(report.violations)}")
    for rule, count in report.counts_by_rule().items():
        line(f"    {rule}: {count}")
    for violation in report.violations:
        line(f"    {violation.file}:{violation.line} [{violation.rule}] {violation.text}")
    if report.read_errors:
        line(f"  READ ERRORS: {len(report.read_errors)}")
        for error in report.read_errors:
            line(f"    {error}")
    line(f"  verdict: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_timelessness.py",
        description="G3: no marker comments, no changelog phrasing, no dates in prose.",
    )
    parser.add_argument("--root", required=True, nargs="+", type=Path, help="directories or files to scan")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="citation-marker allowlist (one regex per line)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    args = parser.parse_args(argv)

    try:
        report = build_report(roots=args.root, allowlist_path=args.allowlist)
    except TimelessnessError as exc:
        print(f"check_timelessness: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  receipt: {args.json}")
    return 0 if report.passed else 1


def rule_names() -> Iterable[str]:
    """Every rule identifier this checker can emit."""
    return (MARKER_RULE, USED_TO_RULE, DATED_RULE, *(rule for rule, _ in CHANGELOG_RULES))


if __name__ == "__main__":
    raise SystemExit(main())
