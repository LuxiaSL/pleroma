#!/usr/bin/env python3
"""G4 — import closure over a Python package: every module reachable, nothing dangling.

The gate: every module in the package must be reachable, by imports, from a
golden path (a script) or a test, and every intra-package import must resolve to
a module that exists. Exit status is 0 only when both counts are zero.

A purely static `import`/`from ... import` walk is not enough for this codebase,
because three dynamic idioms carry real import edges:

  (a) ``importlib.util.spec_from_file_location(name, <dir> / "mod.py")`` — the
      loader form a test harness uses to reach a script that is not on the
      import path.
  (b) ``sys.path.insert(0, <dir>)`` followed by a bare ``import mod`` in the same
      file — after the insertion, ``<dir>/mod.py`` is an importable target.
  (c) ``importlib.import_module(name, package=...)`` where ``name`` is built at
      run time from a registry of module-name strings held in the same module.
      When the argument is dynamic, every module-name-shaped string literal in
      that file is a candidate, matched against the sibling modules of the
      anchor package.

Without (a)-(c) the closure reports modules that a dispatch table reaches as
orphans, and reports harness tests as dependency-free. Dynamic edges carry their
idiom in the `kind` field, so a receipt shows which edges the verdict rests on.

Reachability follows Python's own semantics in one further respect: importing
``a.b.c`` executes ``a/__init__.py`` and ``a/b/__init__.py``, so the ancestor
packages of a reached module are reached too. The converse does not hold —
reaching a package does not reach its submodules.

Usage
-----
    python tools/check_import_closure.py \
        --package anamnesis --roots anamnesis/scripts tests --json g4.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO

MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"})

STATIC = "static"
STATIC_SUBMODULE = "static-submodule"
RELATIVE = "relative"
SPEC_FROM_FILE = "spec-from-file-location"
SYS_PATH = "sys-path-insert"
IMPORT_MODULE = "import-module"
IMPORT_MODULE_DYNAMIC = "import-module-dynamic"

DYNAMIC_KINDS = frozenset({SPEC_FROM_FILE, SYS_PATH, IMPORT_MODULE, IMPORT_MODULE_DYNAMIC})

AddEdge = Callable[[str, str, int], None]


class ClosureError(RuntimeError):
    """A condition that makes the closure unmeasurable rather than failing."""


@dataclass(frozen=True)
class ModuleInfo:
    """One importable module discovered inside the package tree."""

    name: str
    path: Path
    is_package: bool


@dataclass(frozen=True)
class Edge:
    """One import edge, static or dynamic, from a source file to a module."""

    source: str
    target: str
    kind: str
    file: str
    line: int


@dataclass(frozen=True)
class Dangling:
    """An intra-package import naming a module that does not exist."""

    source: str
    target: str
    kind: str
    file: str
    line: int


@dataclass(frozen=True)
class UnresolvedDynamic:
    """A dynamic import site whose target could not be identified."""

    kind: str
    detail: str
    file: str
    line: int


@dataclass
class ClosureReport:
    """The result of a closure pass, serialisable as the G4 receipt."""

    package: str
    package_dir: str
    roots: list[str]
    modules: list[str] = field(default_factory=list)
    root_modules: list[str] = field(default_factory=list)
    external_root_files: list[str] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    dangling: list[Dangling] = field(default_factory=list)
    unresolved_dynamic: list[UnresolvedDynamic] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    allowed_orphans: list[str] = field(default_factory=list)
    unused_allowances: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.orphans and not self.dangling and not self.parse_errors

    def edges_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.kind] = counts.get(edge.kind, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> dict[str, object]:
        return {
            "gate": "G4",
            "package": self.package,
            "package_dir": self.package_dir,
            "roots": self.roots,
            "counts": {
                "modules": len(self.modules),
                "root_modules": len(self.root_modules),
                "external_root_files": len(self.external_root_files),
                "edges": len(self.edges),
                "edges_by_kind": self.edges_by_kind(),
                "orphans": len(self.orphans),
                "allowed_orphans": len(self.allowed_orphans),
                "dangling_imports": len(self.dangling),
                "unresolved_dynamic": len(self.unresolved_dynamic),
                "parse_errors": len(self.parse_errors),
            },
            "orphans": self.orphans,
            "allowed_orphans": self.allowed_orphans,
            "unused_allowances": self.unused_allowances,
            "dangling_imports": [vars(d) for d in self.dangling],
            "unresolved_dynamic": [vars(u) for u in self.unresolved_dynamic],
            "dynamic_edges": [vars(e) for e in self.edges if e.kind in DYNAMIC_KINDS],
            "parse_errors": self.parse_errors,
            "passed": self.passed,
        }


def iter_python_files(root: Path) -> list[Path]:
    """Every `.py` file under `root`, excluding caches and virtual environments."""
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def discover_modules(package_dir: Path) -> dict[str, ModuleInfo]:
    """Map dotted module name to `ModuleInfo` for every module in the package."""
    base = package_dir.parent
    modules: dict[str, ModuleInfo] = {}
    for path in iter_python_files(package_dir):
        parts = list(path.relative_to(base).parts)
        is_package = parts[-1] == "__init__.py"
        if is_package:
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        name = ".".join(parts)
        if not name:
            continue
        modules[name] = ModuleInfo(name=name, path=path, is_package=is_package)
    if not modules:
        raise ClosureError(f"no Python modules found under {package_dir}")
    return modules


def parse_file(path: Path) -> ast.Module:
    """Parse one file, raising `ClosureError` naming the path on failure."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ClosureError(f"{path}: unreadable ({exc})") from exc
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ClosureError(f"{path}: syntax error at line {exc.lineno} ({exc.msg})") from exc


def dotted_name(node: ast.AST) -> str | None:
    """Render an attribute/name chain as a dotted string, or None."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def call_tail(node: ast.Call) -> str | None:
    """The final name of a call target: `a.b.c(...)` and `c(...)` both give `c`."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def assignment_map(tree: ast.Module) -> dict[str, ast.expr]:
    """Single-target `name = expr` bindings anywhere in a module.

    Path expressions are routinely split across two statements (bind a
    directory, then join a filename onto it), so resolving a dynamic import
    target means following one hop through the local bindings. Later bindings of
    the same name win, which matches what a reader of the file would assume.
    """
    bindings: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                bindings[node.target.id] = node.value
    return bindings


class PathResolver:
    """Static evaluation of filesystem-path expressions inside one file.

    Covers the composition forms this codebase uses: ``__file__``, ``Path(...)``,
    ``str(...)``, ``.resolve()``, ``.absolute()``, ``.parent``, ``.parents[n]``,
    ``os.path.dirname``/``join``, ``/`` joins against string literals, and one
    hop through a local name binding.
    """

    MAX_DEPTH = 12

    def __init__(self, file_path: Path, bindings: dict[str, ast.expr] | None = None) -> None:
        self.file_path = file_path
        self.bindings = bindings or {}

    def eval(self, node: ast.AST, depth: int = 0) -> Path | None:
        """The concrete path an expression denotes, or None when it is not static."""
        if depth > self.MAX_DEPTH:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = Path(node.value)
            return candidate if candidate.is_absolute() else (self.file_path.parent / candidate)
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self.file_path
            bound = self.bindings.get(node.id)
            return None if bound is None else self.eval(bound, depth + 1)
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value, depth + 1)
            if base is not None and node.attr == "parent":
                return base.parent
            return None
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
                base = self.eval(node.value.value, depth + 1)
                index = node.slice
                if base is None or not isinstance(index, ast.Constant):
                    return None
                if not isinstance(index.value, int) or isinstance(index.value, bool):
                    return None
                parents = base.parents
                if 0 <= index.value < len(parents):
                    return parents[index.value]
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self.eval(node.left, depth + 1)
            if left is None:
                return None
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                return left / node.right.value
            return None
        if isinstance(node, ast.Call):
            return self._eval_call(node, depth)
        return None

    def _eval_call(self, node: ast.Call, depth: int) -> Path | None:
        tail = call_tail(node)
        if tail is None:
            return None
        func = node.func
        if isinstance(func, ast.Attribute):
            if tail in {"resolve", "absolute", "expanduser"}:
                return self.eval(func.value, depth + 1)
            if tail == "joinpath" and node.args:
                base = self.eval(func.value, depth + 1)
                return None if base is None else self._append_literals(base, node.args)
            if tail == "dirname" and node.args:
                base = self.eval(node.args[0], depth + 1)
                return None if base is None else base.parent
            if tail == "join" and node.args:
                base = self.eval(node.args[0], depth + 1)
                return None if base is None else self._append_literals(base, node.args[1:])
            return None
        if tail in {"Path", "str", "fspath"} and node.args:
            return self.eval(node.args[0], depth + 1)
        if tail == "dirname" and node.args:
            base = self.eval(node.args[0], depth + 1)
            return None if base is None else base.parent
        if tail == "join" and node.args:
            base = self.eval(node.args[0], depth + 1)
            return None if base is None else self._append_literals(base, node.args[1:])
        return None

    @staticmethod
    def _append_literals(base: Path, nodes: Sequence[ast.expr]) -> Path | None:
        for extra in nodes:
            if not (isinstance(extra, ast.Constant) and isinstance(extra.value, str)):
                return None
            base = base / extra.value
        return base

    def literal_tail(self, node: ast.AST, depth: int = 0) -> str | None:
        """The trailing string literal of a path expression, if there is one."""
        if depth > self.MAX_DEPTH:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            bound = self.bindings.get(node.id)
            return None if bound is None else self.literal_tail(bound, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self.literal_tail(node.right, depth + 1)
        if isinstance(node, ast.Call):
            tail = call_tail(node)
            if tail is None or not node.args:
                return None
            if tail in {"Path", "str", "fspath"}:
                return self.literal_tail(node.args[0], depth + 1)
            if tail in {"join", "joinpath"}:
                return self.literal_tail(node.args[-1], depth + 1)
        return None


def string_constants(tree: ast.Module) -> list[str]:
    """Every string constant in a module, including f-string literal parts."""
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


class ModuleIndex:
    """Lookup helpers over the discovered module set."""

    def __init__(self, package: str, package_dir: Path, modules: dict[str, ModuleInfo]) -> None:
        self.package = package
        self.package_dir = package_dir
        self.modules = modules
        self.by_path: dict[Path, ModuleInfo] = {}
        self.by_basename: dict[str, list[ModuleInfo]] = {}
        for info in modules.values():
            self.by_path[info.path.resolve()] = info
            self.by_basename.setdefault(info.path.name, []).append(info)

    def is_package_name(self, name: str) -> bool:
        return name == self.package or name.startswith(self.package + ".")

    def is_namespace_package(self, name: str) -> bool:
        """True for a directory that holds modules but has no `__init__.py`.

        Importing one executes no code, so it is neither an edge target nor an
        orphan candidate — but it is also not a dangling import.
        """
        if name in self.modules:
            return False
        prefix = name + "."
        return any(other.startswith(prefix) for other in self.modules)

    def module_for_path(self, path: Path) -> ModuleInfo | None:
        return self.by_path.get(path.resolve())

    def modules_named(self, filename: str) -> list[ModuleInfo]:
        return list(self.by_basename.get(filename, ()))


@dataclass
class FileScan:
    """Edges and diagnostics harvested from a single file."""

    edges: list[Edge] = field(default_factory=list)
    dangling: list[Dangling] = field(default_factory=list)
    unresolved: list[UnresolvedDynamic] = field(default_factory=list)


def anchor_package_of(source_name: str, index: ModuleIndex) -> str:
    """The package a relative import inside `source_name` resolves against."""
    info = index.modules.get(source_name)
    if info is not None and info.is_package:
        return source_name
    return source_name.rsplit(".", 1)[0] if "." in source_name else ""


def resolve_relative(source_name: str, level: int, module: str | None, index: ModuleIndex) -> str | None:
    """Resolve a relative import to an absolute dotted name."""
    anchor = anchor_package_of(source_name, index)
    parts = anchor.split(".") if anchor else []
    for _ in range(level - 1):
        if not parts:
            return None
        parts = parts[:-1]
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def scan_file(path: Path, source_name: str, index: ModuleIndex, inside_package: bool) -> FileScan:
    """Collect every import edge (static and dynamic) out of one file."""
    scan = FileScan()
    tree = parse_file(path)
    file_str = str(path)
    literals = string_constants(tree)
    resolver = PathResolver(path, assignment_map(tree))
    inserted_dirs: list[Path] = []
    bare_imports: list[tuple[str, int]] = []

    def add_edge(target: str, kind: str, line: int) -> None:
        scan.edges.append(Edge(source=source_name, target=target, kind=kind, file=file_str, line=line))

    def record(name: str, kind: str, line: int, probe: bool = False) -> bool:
        """Record an edge for a dotted import name.

        `probe` marks the speculative ``from X import Y`` form, where `Y` is a
        submodule only some of the time; a miss there is a name, not a dangling
        import, and is never reported.
        """
        if name in index.modules:
            add_edge(name, kind, line)
            return True
        if index.is_namespace_package(name):
            return True
        if not probe and index.is_package_name(name):
            scan.dangling.append(
                Dangling(source=source_name, target=name, kind=kind, file=file_str, line=line)
            )
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not record(alias.name, STATIC, node.lineno) and "." not in alias.name:
                    bare_imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not inside_package:
                    continue
                base = resolve_relative(source_name, node.level, node.module, index)
                if base is None:
                    scan.unresolved.append(
                        UnresolvedDynamic(
                            kind=RELATIVE,
                            detail=f"relative import above package root (level={node.level})",
                            file=file_str,
                            line=node.lineno,
                        )
                    )
                    continue
                record(base, RELATIVE, node.lineno)
                for alias in node.names:
                    record(f"{base}.{alias.name}", RELATIVE, node.lineno, probe=True)
                continue
            if node.module is None:
                continue
            hit = record(node.module, STATIC, node.lineno)
            for alias in node.names:
                record(f"{node.module}.{alias.name}", STATIC_SUBMODULE, node.lineno, probe=True)
            if not hit and "." not in node.module:
                bare_imports.append((node.module, node.lineno))
        elif isinstance(node, ast.Call):
            tail = call_tail(node)
            if tail is None:
                continue
            dotted = dotted_name(node.func)
            if tail == "spec_from_file_location":
                scan_spec_from_file(node, resolver, index, add_edge, scan, file_str)
            elif tail == "import_module":
                scan_import_module(node, source_name, index, literals, add_edge, scan, file_str)
            elif dotted in {"sys.path.insert", "sys.path.append", "path.insert", "path.append"}:
                directory = sys_path_target(node, tail, resolver)
                if directory is None:
                    scan.unresolved.append(
                        UnresolvedDynamic(
                            kind=SYS_PATH,
                            detail="sys.path target not statically resolvable",
                            file=file_str,
                            line=node.lineno,
                        )
                    )
                else:
                    inserted_dirs.append(directory)

    for name, line in bare_imports:
        for directory in inserted_dirs:
            for candidate in (directory / f"{name}.py", directory / name / "__init__.py"):
                info = index.module_for_path(candidate)
                if info is not None:
                    add_edge(info.name, SYS_PATH, line)
    return scan


def sys_path_target(node: ast.Call, tail: str, resolver: PathResolver) -> Path | None:
    """The directory a `sys.path.insert`/`append` call adds, if it is static."""
    args = [a for a in node.args if not isinstance(a, ast.Starred)]
    if tail == "insert":
        candidate = args[1] if len(args) >= 2 else None
    else:
        candidate = args[0] if args else None
    return None if candidate is None else resolver.eval(candidate)


def scan_spec_from_file(
    node: ast.Call,
    resolver: PathResolver,
    index: ModuleIndex,
    add_edge: AddEdge,
    scan: FileScan,
    file_str: str,
) -> None:
    """Resolve `spec_from_file_location(name, location)` to a package module."""
    location: ast.AST | None = node.args[1] if len(node.args) >= 2 else None
    for kw in node.keywords:
        if kw.arg == "location":
            location = kw.value
    if location is None:
        scan.unresolved.append(
            UnresolvedDynamic(
                kind=SPEC_FROM_FILE,
                detail="no location argument found",
                file=file_str,
                line=node.lineno,
            )
        )
        return
    tail = resolver.literal_tail(location)
    if tail is None or not tail.endswith(".py"):
        scan.unresolved.append(
            UnresolvedDynamic(
                kind=SPEC_FROM_FILE,
                detail="location has no trailing '.py' literal",
                file=file_str,
                line=node.lineno,
            )
        )
        return
    resolved = resolver.eval(location)
    if resolved is not None:
        info = index.module_for_path(resolved)
        if info is not None:
            add_edge(info.name, SPEC_FROM_FILE, node.lineno)
            return
    candidates = index.modules_named(Path(tail).name)
    if candidates:
        for info in candidates:
            add_edge(info.name, SPEC_FROM_FILE, node.lineno)
        return
    scan.unresolved.append(
        UnresolvedDynamic(
            kind=SPEC_FROM_FILE,
            detail=f"no package module matches {Path(tail).name}",
            file=file_str,
            line=node.lineno,
        )
    )


def scan_import_module(
    node: ast.Call,
    source_name: str,
    index: ModuleIndex,
    literals: Sequence[str],
    add_edge: AddEdge,
    scan: FileScan,
    file_str: str,
) -> None:
    """Resolve `importlib.import_module(name, package=...)`, literal or built."""
    if not node.args:
        scan.unresolved.append(
            UnresolvedDynamic(
                kind=IMPORT_MODULE,
                detail="no name argument found",
                file=file_str,
                line=node.lineno,
            )
        )
        return
    anchor = import_module_anchor(node, source_name, index)
    arg = node.args[0]

    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        resolved = resolve_import_module_name(arg.value, anchor, index)
        if resolved is not None:
            add_edge(resolved, IMPORT_MODULE, node.lineno)
        else:
            scan.unresolved.append(
                UnresolvedDynamic(
                    kind=IMPORT_MODULE,
                    detail=f"literal name {arg.value!r} matches no package module",
                    file=file_str,
                    line=node.lineno,
                )
            )
        return

    prefix = dynamic_prefix(arg)
    matched: list[str] = []
    for literal in literals:
        if not MODULE_NAME_RE.match(literal):
            continue
        resolved = resolve_import_module_name(prefix + literal, anchor, index)
        if resolved is not None and resolved not in matched:
            matched.append(resolved)
    for name in matched:
        add_edge(name, IMPORT_MODULE_DYNAMIC, node.lineno)
    if not matched:
        scan.unresolved.append(
            UnresolvedDynamic(
                kind=IMPORT_MODULE_DYNAMIC,
                detail="no module-name literal in this file matches a package module",
                file=file_str,
                line=node.lineno,
            )
        )


def import_module_anchor(node: ast.Call, source_name: str, index: ModuleIndex) -> str:
    """The package a relative `import_module` name resolves against."""
    for kw in node.keywords:
        if kw.arg != "package":
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
        return anchor_from_name_dunder(source_name, index)
    if len(node.args) >= 2:
        second = node.args[1]
        if isinstance(second, ast.Constant) and isinstance(second.value, str):
            return second.value
    return anchor_from_name_dunder(source_name, index)


def anchor_from_name_dunder(source_name: str, index: ModuleIndex) -> str:
    """What `package=__name__` means inside `source_name`."""
    info = index.modules.get(source_name)
    if info is not None and info.is_package:
        return source_name
    return anchor_package_of(source_name, index)


def resolve_import_module_name(name: str, anchor: str, index: ModuleIndex) -> str | None:
    """Absolute module name for an `import_module` argument, when it exists."""
    if name.startswith("."):
        level = len(name) - len(name.lstrip("."))
        leaf = name[level:]
        parts = anchor.split(".") if anchor else []
        for _ in range(level - 1):
            if not parts:
                return None
            parts = parts[:-1]
        base = ".".join(parts)
        absolute = f"{base}.{leaf}" if base and leaf else (base or leaf)
    else:
        absolute = name
        if absolute not in index.modules and anchor:
            nested = f"{anchor}.{absolute}"
            if nested in index.modules:
                absolute = nested
    return absolute if absolute in index.modules else None


def dynamic_prefix(arg: ast.AST) -> str:
    """The leading literal of a built module name, e.g. the dot in f".{mod}"."""
    if isinstance(arg, ast.JoinedStr) and arg.values:
        head = arg.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
        return ""
    if isinstance(arg, ast.Call):
        func = arg.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            template = func.value
            if isinstance(template, ast.Constant) and isinstance(template.value, str):
                return template.value.split("{", 1)[0]
        return ""
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
        if isinstance(arg.left, ast.Constant) and isinstance(arg.left.value, str):
            return arg.left.value.split("%", 1)[0]
    return ""


def build_report(
    package_dir: Path,
    roots: Sequence[Path],
    allow_orphans: Iterable[str] = (),
) -> ClosureReport:
    """Run the closure and return the receipt."""
    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise ClosureError(f"--package is not a directory: {package_dir}")
    resolved_roots: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if not candidate.is_dir():
            raise ClosureError(f"--roots entry is not a directory: {candidate}")
        resolved_roots.append(candidate)
    if not resolved_roots:
        raise ClosureError("at least one --roots directory is required")

    modules = discover_modules(package_dir)
    index = ModuleIndex(package=package_dir.name, package_dir=package_dir, modules=modules)

    report = ClosureReport(
        package=package_dir.name,
        package_dir=str(package_dir),
        roots=[str(r) for r in resolved_roots],
        modules=sorted(modules),
    )

    def under_roots(path: Path) -> bool:
        return any(path.is_relative_to(root) for root in resolved_roots)

    report.root_modules = sorted(
        info.name for info in modules.values() if under_roots(info.path.resolve())
    )

    seen_paths = {info.path.resolve() for info in modules.values()}
    external_files: list[Path] = []
    for root in resolved_roots:
        for path in iter_python_files(root):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            external_files.append(resolved)
            seen_paths.add(resolved)
    report.external_root_files = sorted(str(p) for p in external_files)

    graph: dict[str, set[str]] = {name: set() for name in modules}
    seeds: set[str] = set(report.root_modules)

    for info in sorted(modules.values(), key=lambda m: m.name):
        try:
            scan = scan_file(info.path, info.name, index, inside_package=True)
        except ClosureError as exc:
            report.parse_errors.append(str(exc))
            continue
        report.edges.extend(scan.edges)
        report.dangling.extend(scan.dangling)
        report.unresolved_dynamic.extend(scan.unresolved)
        for edge in scan.edges:
            graph[info.name].add(edge.target)

    for path in sorted(external_files):
        try:
            scan = scan_file(path, f"<root>{path}", index, inside_package=False)
        except ClosureError as exc:
            report.parse_errors.append(str(exc))
            continue
        report.edges.extend(scan.edges)
        report.dangling.extend(scan.dangling)
        report.unresolved_dynamic.extend(scan.unresolved)
        for edge in scan.edges:
            seeds.add(edge.target)

    reached: set[str] = set()
    queue: deque[str] = deque()

    def visit(name: str) -> None:
        """Mark a module reached, with the ancestor packages its import executes."""
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            ancestor = ".".join(parts[:depth])
            if ancestor in modules and ancestor not in reached:
                reached.add(ancestor)
                queue.append(ancestor)

    for seed in sorted(seeds):
        if seed in modules:
            visit(seed)
    while queue:
        current = queue.popleft()
        for target in sorted(graph.get(current, ())):
            if target in modules and target not in reached:
                visit(target)

    allowed = sorted(set(allow_orphans))
    unreached = sorted(name for name in modules if name not in reached)
    report.orphans = [name for name in unreached if name not in allowed]
    report.allowed_orphans = [name for name in unreached if name in allowed]
    report.unused_allowances = [name for name in allowed if name not in unreached]
    return report


def print_report(report: ClosureReport, stream: TextIO | None = None) -> None:
    """Human-readable receipt, matching the JSON."""
    out = sys.stdout if stream is None else stream

    def line(text: str = "") -> None:
        out.write(text + "\n")

    line(f"G4 import closure — package {report.package} ({report.package_dir})")
    line(f"  roots: {', '.join(report.roots)}")
    line(f"  modules: {len(report.modules)}   root modules: {len(report.root_modules)}")
    line(f"  external root files: {len(report.external_root_files)}")
    line(f"  import edges: {len(report.edges)}")
    for kind, count in report.edges_by_kind().items():
        line(f"    {kind}: {count}")
    line(f"  dangling intra-package imports: {len(report.dangling)}")
    for dangling in report.dangling:
        line(f"    {dangling.file}:{dangling.line} -> {dangling.target}")
    line(f"  unresolved dynamic sites (reported, not gating): {len(report.unresolved_dynamic)}")
    for unresolved in report.unresolved_dynamic:
        line(f"    {unresolved.file}:{unresolved.line} [{unresolved.kind}] {unresolved.detail}")
    line(f"  allowed orphans: {len(report.allowed_orphans)}")
    for name in report.allowed_orphans:
        line(f"    {name}")
    if report.unused_allowances:
        line(f"  allowances matching no orphan: {len(report.unused_allowances)}")
        for name in report.unused_allowances:
            line(f"    {name}")
    line(f"  ORPHANS: {len(report.orphans)}")
    for name in report.orphans:
        line(f"    {name}")
    if report.parse_errors:
        line(f"  PARSE ERRORS: {len(report.parse_errors)}")
        for error in report.parse_errors:
            line(f"    {error}")
    line(f"  verdict: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_import_closure.py",
        description="G4: every package module reachable from a script or a test.",
    )
    parser.add_argument("--package", required=True, type=Path, help="package directory to close over")
    parser.add_argument(
        "--roots",
        required=True,
        nargs="+",
        type=Path,
        help="entry-point directories (scripts, tests) that seed the traversal",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the receipt here")
    parser.add_argument(
        "--allow-orphan",
        action="append",
        default=[],
        metavar="MODULE",
        help="dotted module name exempt from the orphan check (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(
            package_dir=args.package,
            roots=args.roots,
            allow_orphans=[value.strip() for value in args.allow_orphan if value.strip()],
        )
    except ClosureError as exc:
        print(f"check_import_closure: {exc}", file=sys.stderr)
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
