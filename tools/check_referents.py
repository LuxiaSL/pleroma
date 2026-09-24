#!/usr/bin/env python3
"""G3 — referent reachability: every name a comment points at is in this repository.

The second documentation rule in `CONTRIBUTING.md`: if a comment or a docstring
names something, a reader holding only this repository must be able to go look at
it. A module path that exists here, a file in this tree, a symbol this package
defines, or a public URL all pass. A path into a private tree, a planning
document, or a phrase that defers the meaning to a conversation does not.

Scope is the same boundary G3's timelessness rule uses, and for the same reason:
`documentation_text` from `check_timelessness.py` is imported rather than
reimplemented, so both halves of the documentation rule agree on what counts as
prose — a comment, a docstring, and the message a `raise` or a logging call says
out loud. Every other string literal is data — a fixture, a dict key, a filename,
a JSON payload — and data names whatever it names.

How a line is read
------------------
1. **The allowlist masks first.** Each allow pattern from
   `referents_allowlist.txt` is blanked out of the line (replaced by spaces, so
   offsets survive). That is where the legitimate path-shaped things live: URLs,
   Hugging Face model ids, the on-disk data schema, the named external trees.
   Referent detection then runs over what is left, which means an over-broad
   allow pattern hides violations — the file says so at its top.
2. **Private-tree referents** are flagged: a path into the research tree, a
   home-relative path, an absolute path that does not resolve inside this
   repository, or a bare planning-document name (a run of capitalised
   hyphen-joined words, a title whose head is shouted and whose tail is not, or a
   `-memo` suffix). A planning-style name that *is* a file in this tree passes,
   because the rule asks for reachability and a file here is reachable — the shape
   of the name is only evidence.
3. **Path-like tokens** are resolved against an index of the tree. A token counts
   as path-like when it carries a known file extension, or starts at a top-level
   entry of the repository; a placeholder or a glob is a schema and never a
   referent. It resolves when some file or directory here ends with it, so a
   relative pointer written from a sibling module's point of view
   (`map/build/export.py` from inside `pleroma/`) still resolves.
4. **Dotted module paths** rooted at a package this repository ships resolve to a
   module file, or to a module plus a top-level symbol that module defines. The
   package roots are read off the tree, and the resolution is static (AST), never
   an import: the checker must run on a box with no torch and no model weights.
5. **Deferral phrases** — prose that points at a conversation instead of stating
   the substance — come from the `[defer]` section of the same data file. They
   are judgement, not syntax: which idioms bury meaning accumulates as the
   codebase is read, so they are versioned beside the checker rather than frozen
   in it, exactly as the date allowlist is.
6. **Provenance citations** — a section mark, a named arm, a pre-registration, a
   milestone code, a bare item code, a commit hash — come from the `[cite]`
   section of the same file. They are the other half of the deferral rule: a
   deferral points at a conversation, a provenance citation points at a document,
   and outside the tree that held them both leave the reader with nothing to
   open. The code is the receipt for what the code does, and the history of how
   it got here is git's to keep.

Matching is line-scoped, like the timelessness rules: a referent split across a
line break is not seen. Writing a path on one line is the cheaper half of that
trade.

The patterns for the private trees are written with one character bracketed
(`researc[h]`), so this file does not match itself and is scanned by the gate it
implements.

Usage
-----
    python tools/check_referents.py --root anamnesis tools tests --json g3-referents.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TextIO

from tools.check_timelessness import (
    SKIP_DIRS,
    TimelessnessError,
    documentation_text,
    iter_python_files,
)

DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "referents_allowlist.txt"
DEFAULT_REPO = Path(__file__).resolve().parent.parent

PRIVATE_RULE = "private-tree"
PATH_RULE = "unresolved-path"
MODULE_RULE = "unresolved-module"
DEFERRAL_RULE = "buried-deferral"
CITATION_RULE = "provenance-citation"

#: Extensions that make a token a path rather than prose, with or without a slash.
PATH_EXTENSIONS = frozenset(
    {
        "py", "md", "txt", "json", "jsonl", "yml", "yaml", "toml", "cfg", "ini",
        "npz", "npy", "pkl", "pt", "safetensors", "csv", "tsv", "sh", "lock",
        "tex", "pdf", "html", "css", "js", "rst",
    }
)

PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_~.][A-Za-z0-9_.~<>/*-]*")

RESEARCH_TREE_RE = re.compile(r"(?<![\w/-])researc[h]/[A-Za-z0-9_./<>*-]+")
HOME_RELATIVE_RE = re.compile(r"(?<![\w])~/[A-Za-z0-9_./<>*-]+")
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
#: Three or more capitalised hyphen-joined words: the shape of a planning document.
#: Three rather than two, so an ordinary compound adjective (`BH-FDR-corrected`,
#: whose tail is prose) does not read as a document name.
DOC_NAME_RE = re.compile(r"\b(?:[A-Z][A-Z0-9]+-){2,}[A-Z][A-Z0-9]+[A-Za-z0-9._-]*")
#: The other way a document title is written: a shouted head, then a hyphenated
#: sentence. `SPE[C]-path-receptacles-and-span-coverage` is the shape, and the tail is
#: what separates it from prose — a contrast name (`ANCHOR-versus-RECENCY`) or a
#: category label (`PORT-as-is`) joins two or three words, while a title spells out
#: what the document is about and runs to four or more. Three tail segments is the
#: line, measured against the compounds this codebase actually writes.
DOC_TITLE_RE = re.compile(r"\b[A-Z]{3,}(?:-[A-Za-z0-9]+){3,}")
#: Explicit planning-document prefixes, bracketed so this file does not match itself.
DOC_KEYWORD_RE = re.compile(
    r"\b(?:PORT-MANIFES[T]|ORIENTATIO[N]|HANDOF[F]|HANDOVE[R]|RULIN[G]|PREREG|SESSION)"
    r"-[A-Za-z0-9][A-Za-z0-9._-]*"
)
MEMO_NAME_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*-mem[o]\b")
#: A document title written with spaces rather than hyphens, tagged by a version-ish
#: token: ``v3 delta mem[o]``, ``a5 arm not[e]``. The version token is what separates a
#: title from ordinary prose — a bare noun names no document a reader could open, while a
#: version-tagged phrase does. A title carrying no such token reads exactly like prose to
#: a regular expression, so it is a reviewer's catch rather than this check's; the module
#: docstring says so. The examples above bracket a letter so this file does not match
#: itself, the same guard the timelessness patterns use.
SPACED_DOC_NAME_RE = re.compile(
    r"\b[a-z]*\d[a-z0-9]*(?:\s+[a-z]+){0,2}\s+(?:mem[o]|not[e]|pla[n]|spe[c]|handof[f])\b",
    re.IGNORECASE,
)

ALLOW_SECTION = "allow"
DEFER_SECTION = "defer"
CITE_SECTION = "cite"
SECTIONS = (ALLOW_SECTION, DEFER_SECTION, CITE_SECTION)


class ReferentError(RuntimeError):
    """A condition that makes the check unrunnable rather than failing."""


@dataclass(frozen=True)
class Violation:
    """One unreachable referent."""

    rule: str
    file: str
    line: int
    match: str
    text: str


@dataclass(frozen=True)
class Allowlist:
    """The three pattern sets the data file carries.

    `allow` patterns mask legitimate identifiers out of a line before referents
    are looked for; `defer` patterns are the prose idioms that bury meaning; `cite`
    patterns are the provenance idioms that point at a document instead of stating
    the fact.
    """

    path: Path
    allow: tuple[re.Pattern[str], ...]
    defer: tuple[re.Pattern[str], ...]
    cite: tuple[re.Pattern[str], ...]

    def mask(self, text: str) -> str:
        """Blank every allowed span, preserving line length so columns still line up."""
        chars = list(text)
        for pattern in self.allow:
            for match in pattern.finditer(text):
                for index in range(match.start(), match.end()):
                    chars[index] = " "
        return "".join(chars)


@dataclass
class ReferentReport:
    """The result of a referent pass, serialisable as the receipt."""

    roots: list[str]
    repo: str
    allowlist: str
    allow_size: int
    defer_size: int
    cite_size: int
    files_scanned: int = 0
    violations: list[Violation] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)

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
            "gate": "G3-referents",
            "roots": self.roots,
            "repo": self.repo,
            "allowlist": self.allowlist,
            "allow_patterns": self.allow_size,
            "defer_patterns": self.defer_size,
            "cite_patterns": self.cite_size,
            "counts": {
                "files_scanned": self.files_scanned,
                "violations": len(self.violations),
                "by_rule": self.counts_by_rule(),
                "read_errors": len(self.read_errors),
            },
            "violations": [vars(v) for v in self.violations],
            "read_errors": self.read_errors,
            "passed": self.passed,
        }


def load_allowlist(path: Path) -> Allowlist:
    """Compile the data file: one regex per line, `#` comments, `[allow]`/`[defer]` sections.

    Lines before the first section header are allow patterns, so the plain
    one-regex-per-line form remains valid. An unknown section name or an
    uncompilable regex is an error rather than a silently dropped line: a
    swallowed pattern would make the gate quietly weaker or quietly noisier.

    `[allow]` and `[cite]` patterns are case-sensitive and `[defer]` patterns are
    not, because the sections match different kinds of thing: capitalisation is
    often what separates an identifier or a code from the prose around it, while a
    deferral phrase is prose and stands at the head of a sentence as readily as
    inside one. A `[cite]` pattern that wants both spellings says so itself, with
    a character class or an inline `(?i:...)`.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReferentError(f"allowlist unreadable: {path} ({exc})") from exc
    buckets: dict[str, list[re.Pattern[str]]] = {
        ALLOW_SECTION: [],
        DEFER_SECTION: [],
        CITE_SECTION: [],
    }
    section = ALLOW_SECTION
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip().lower()
            if name not in SECTIONS:
                raise ReferentError(
                    f"{path}:{number}: unknown section {stripped!r}; expected one of {SECTIONS}"
                )
            section = name
            continue
        flags = re.IGNORECASE if section == DEFER_SECTION else 0
        try:
            buckets[section].append(re.compile(stripped, flags))
        except re.error as exc:
            raise ReferentError(f"{path}:{number}: invalid regex {stripped!r} ({exc})") from exc
    return Allowlist(
        path=path,
        allow=tuple(buckets[ALLOW_SECTION]),
        defer=tuple(buckets[DEFER_SECTION]),
        cite=tuple(buckets[CITE_SECTION]),
    )


@dataclass
class TreeIndex:
    """What this repository contains, from the point of view of a reader following a name."""

    repo: Path
    files: frozenset[str]
    directories: frozenset[str]
    top_level: frozenset[str]
    #: Matches a dotted path rooted at a package this repository actually ships.
    module_token: re.Pattern[str] | None

    @classmethod
    def build(cls, repo: Path) -> TreeIndex:
        """Index every file and directory under `repo`, skipping caches and environments.

        The package roots are read off the tree rather than named in the source, so
        the checker resolves dotted paths for whatever this repository contains.
        """
        if not repo.is_dir():
            raise ReferentError(f"--repo is not a directory: {repo}")
        files: set[str] = set()
        directories: set[str] = set()
        for path in repo.rglob("*"):
            relative = path.relative_to(repo)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            if path.is_dir():
                directories.add(relative.as_posix())
            else:
                files.add(relative.as_posix())
        top_level = {name.split("/", 1)[0] for name in files | directories}
        packages = sorted(
            name
            for name in top_level
            if name in directories and f"{name}/__init__.py" in files
        )
        module_token = (
            re.compile(
                r"\b(?:"
                + "|".join(re.escape(name) for name in packages)
                + r")(?:\.[a-z_][a-z0-9_]*)*\.[A-Za-z_][A-Za-z0-9_]*\b"
            )
            if packages
            else None
        )
        return cls(
            repo=repo,
            files=frozenset(files),
            directories=frozenset(directories),
            top_level=frozenset(top_level),
            module_token=module_token,
        )

    def contains(self, token: str) -> bool:
        """True when some file or directory here is, or ends with, `token`."""
        candidate = token.strip("/")
        if not candidate:
            return False
        if candidate in self.files or candidate in self.directories:
            return True
        suffix = "/" + candidate
        return any(name.endswith(suffix) for name in self.files) or any(
            name.endswith(suffix) for name in self.directories
        )

    def module_symbols(self, dotted: str) -> frozenset[str] | None:
        """Top-level names a module defines, or None when the module is not here.

        Parsed rather than imported: this runs where torch and the model weights
        are absent, and importing `anamnesis.extraction` to check a docstring
        would be a side effect a gate has no business having.
        """
        stem = dotted.replace(".", "/")
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if candidate not in self.files:
                continue
            try:
                source = (self.repo / candidate).read_text(encoding="utf-8")
                tree = ast.parse(source, filename=candidate)
            except (OSError, UnicodeDecodeError, SyntaxError):
                return frozenset()
            return frozenset(_top_level_names(tree))
        return None


def _top_level_names(tree: ast.Module) -> Iterable[str]:
    """Every name a module binds at module level, including re-exports."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target.id
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                yield alias.asname or alias.name.split(".", 1)[0]


def strip_edges(token: str) -> str:
    """Drop sentence punctuation a referent picks up from the prose around it.

    A leading dot survives, because it is part of the name in `.github`; a
    trailing one does not, because it is the end of the sentence.
    """
    return token.strip("'\"`()[]{}*,;:!?").rstrip(".")


def looks_like_path(token: str, index: TreeIndex) -> bool:
    """Whether a token is a pointer at a file rather than ordinary prose.

    Slashes alone are not enough — `attention/cache`, `Llama/Qwen` and `D(D-1)/2`
    all carry one, and a trailing slash is as often a units expression as a
    directory. A token qualifies on one of two signals: a named file type with
    something in front of it, or a first segment that is a top-level entry of
    this repository.

    A token carrying `<`, `>` or `*` is excluded whatever its shape: those are
    the marks of a schema — a placeholder or a glob — and a schema names a family
    of files a run writes rather than one file a reader can open.
    """
    if not token or token in {".", "..", "/"}:
        return False
    if any(mark in token for mark in "<>*"):
        return False
    basename = token.rsplit("/", 1)[-1]
    stem, _, extension = basename.rpartition(".")
    if stem and extension.lower() in PATH_EXTENSIONS:
        return True
    if "/" not in token:
        return False
    return token.split("/", 1)[0] in index.top_level


def scan_private(text: str, index: TreeIndex) -> list[tuple[str, str]]:
    """Referents that live outside this repository: (rule, matched text) pairs.

    The patterns overlap on purpose — a home-relative path is also an absolute one
    once the tilde is past, and a planning document's name is also part of the path
    it sits at. A match already covered by a wider one is dropped, so one buried
    referent is reported once.
    """
    found: list[tuple[str, str]] = []

    def record(matched: str) -> None:
        if any(matched in prior for _, prior in found):
            return
        found.append((PRIVATE_RULE, matched))

    for pattern in (RESEARCH_TREE_RE, HOME_RELATIVE_RE):
        for match in pattern.finditer(text):
            record(match.group(0))
    for pattern in (
        ABSOLUTE_PATH_RE,
        DOC_NAME_RE,
        DOC_TITLE_RE,
        DOC_KEYWORD_RE,
        MEMO_NAME_RE,
        SPACED_DOC_NAME_RE,
    ):
        for match in pattern.finditer(text):
            if index.contains(strip_edges(match.group(0))):
                continue
            record(match.group(0))
    return found


def scan_paths(text: str, index: TreeIndex, already: Sequence[str]) -> list[tuple[str, str]]:
    """In-repo-looking paths that no file or directory here answers to."""
    found: list[tuple[str, str]] = []
    for match in PATH_TOKEN_RE.finditer(text):
        raw = match.group(0)
        if any(raw in prior for prior in already):
            continue
        token = strip_edges(raw)
        if not looks_like_path(token, index):
            continue
        if index.contains(token):
            continue
        found.append((PATH_RULE, token))
    return found


def scan_modules(text: str, index: TreeIndex) -> list[tuple[str, str]]:
    """Dotted paths under a package root that resolve to neither a module nor a symbol."""
    found: list[tuple[str, str]] = []
    if index.module_token is None:
        return found
    for match in index.module_token.finditer(text):
        dotted = match.group(0)
        if index.module_symbols(dotted) is not None:
            continue
        parent, _, leaf = dotted.rpartition(".")
        symbols = index.module_symbols(parent) if parent else None
        if symbols is not None and leaf in symbols:
            continue
        found.append((MODULE_RULE, dotted))
    return found


def scan_source(
    path: Path,
    source: str,
    index: TreeIndex,
    allowlist: Allowlist,
    display: str | None = None,
) -> list[Violation]:
    """Flag one file's prose lines."""
    prose = documentation_text(path, source)
    file_str = display if display is not None else str(path)
    violations: list[Violation] = []
    for number, raw in enumerate(source.splitlines(), start=1):
        if number not in prose:
            continue
        text = allowlist.mask(prose[number])
        hits: list[tuple[str, str]] = scan_private(text, index)
        hits.extend(scan_paths(text, index, [m for _, m in hits]))
        hits.extend(scan_modules(text, index))
        for pattern in allowlist.defer:
            deferral = pattern.search(text)
            if deferral is not None:
                hits.append((DEFERRAL_RULE, deferral.group(0)))
        for pattern in allowlist.cite:
            citation = pattern.search(text)
            if citation is not None:
                hits.append((CITATION_RULE, citation.group(0)))
        for rule, matched in hits:
            violations.append(
                Violation(
                    rule=rule,
                    file=file_str,
                    line=number,
                    match=matched,
                    text=raw.strip(),
                )
            )
    return violations


def build_report(
    roots: Sequence[Path],
    repo: Path = DEFAULT_REPO,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
) -> ReferentReport:
    """Run the check over every `.py` file under the given roots."""
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if not candidate.exists():
            raise ReferentError(f"--root does not exist: {candidate}")
        resolved.append(candidate)
    if not resolved:
        raise ReferentError("at least one --root is required")

    repo_root = repo.resolve()
    index = TreeIndex.build(repo_root)
    allowlist = load_allowlist(allowlist_path)
    report = ReferentReport(
        roots=[str(r) for r in resolved],
        repo=str(repo_root),
        allowlist=str(allowlist_path.resolve()),
        allow_size=len(allowlist.allow),
        defer_size=len(allowlist.defer),
        cite_size=len(allowlist.cite),
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
                display = path.relative_to(repo_root).as_posix()
            except ValueError:
                display = str(path)
            try:
                report.violations.extend(
                    scan_source(path, source, index, allowlist, display=display)
                )
            except TimelessnessError as exc:
                report.read_errors.append(str(exc))
    report.violations.sort(key=lambda v: (v.file, v.line, v.rule, v.match))
    return report


def print_report(report: ReferentReport, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line("G3 referent reachability")
    line(f"  roots: {', '.join(report.roots)}")
    line(f"  repo: {report.repo}")
    line(
        f"  allowlist: {report.allowlist} "
        f"({report.allow_size} allow, {report.defer_size} defer, {report.cite_size} cite)"
    )
    line(f"  files scanned: {report.files_scanned}")
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


def rule_names() -> Iterable[str]:
    """Every rule identifier this checker can emit."""
    return (PRIVATE_RULE, PATH_RULE, MODULE_RULE, DEFERRAL_RULE, CITATION_RULE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_referents.py",
        description="G3: every referent in a comment or docstring is reachable from this repository.",
    )
    parser.add_argument("--root", required=True, nargs="+", type=Path, help="directories or files to scan")
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help="the tree a referent must resolve inside (default: this repository)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="allow/defer pattern file (one regex per line)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    args = parser.parse_args(argv)

    try:
        report = build_report(roots=args.root, repo=args.repo, allowlist_path=args.allowlist)
    except (ReferentError, TimelessnessError) as exc:
        print(f"check_referents: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  receipt: {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
