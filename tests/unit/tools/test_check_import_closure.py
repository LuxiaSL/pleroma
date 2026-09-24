"""Unit tests for the G4 import-closure checker."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.check_import_closure import (
    ClosureError,
    build_report,
    discover_modules,
    main,
)


def write(path: Path, body: str) -> Path:
    """Write a dedented source file, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def make_package(root: Path) -> Path:
    """A minimal package with a scripts entry point and one library module."""
    pkg = root / "demo"
    write(pkg / "__init__.py", '"""Demo package."""\n')
    write(pkg / "lib.py", "VALUE = 1\n")
    write(pkg / "scripts" / "run.py", "from demo.lib import VALUE\n")
    return pkg


def test_discover_modules_names_packages_and_modules(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    modules = discover_modules(pkg)
    assert set(modules) == {"demo", "demo.lib", "demo.scripts.run"}
    assert modules["demo"].is_package is True
    assert modules["demo.lib"].is_package is False


def test_static_closure_passes_with_no_orphans(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == []
    assert report.dangling == []
    assert report.passed is True


def test_unreached_module_is_an_orphan(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "stranded.py", "VALUE = 2\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == ["demo.stranded"]
    assert report.passed is False


def test_allow_orphan_exempts_and_reports_unused_allowance(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "stranded.py", "VALUE = 2\n")
    report = build_report(pkg, [pkg / "scripts"], allow_orphans=["demo.stranded", "demo.absent"])
    assert report.orphans == []
    assert report.allowed_orphans == ["demo.stranded"]
    assert report.unused_allowances == ["demo.absent"]
    assert report.passed is True


def test_dangling_intra_package_import_is_reported(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "scripts" / "broken.py", "from demo.gone import thing\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert [d.target for d in report.dangling] == ["demo.gone"]
    assert report.passed is False


def test_from_import_of_a_name_is_not_dangling(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "scripts" / "uses_name.py", "from demo.lib import VALUE\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.dangling == []


def test_namespace_package_import_is_not_dangling(tmp_path: Path) -> None:
    """A directory without `__init__.py` is importable and executes no code."""
    pkg = make_package(tmp_path)
    write(pkg / "tools" / "helper.py", "VALUE = 3\n")
    write(pkg / "scripts" / "via_namespace.py", "from demo.tools import helper\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.dangling == []
    assert "demo.tools.helper" not in report.orphans


def test_ancestor_packages_count_as_reached(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "deep" / "__init__.py", "")
    write(pkg / "deep" / "leaf.py", "VALUE = 4\n")
    write(pkg / "scripts" / "reach_leaf.py", "from demo.deep.leaf import VALUE\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == []


def test_reaching_a_package_does_not_reach_its_submodules(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "deep" / "__init__.py", "")
    write(pkg / "deep" / "leaf.py", "VALUE = 4\n")
    write(pkg / "scripts" / "reach_package.py", "import demo.deep\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == ["demo.deep.leaf"]


def make_dispatch_package(root: Path) -> Path:
    """A package whose sections are reached only through a string registry."""
    pkg = root / "demo"
    write(pkg / "__init__.py", "")
    write(
        pkg / "runner" / "__init__.py",
        '''
        """Runner dispatching sections named by a registry of strings."""
        import importlib

        SECTIONS = ["integrity", "classification"]


        def lazy_call(module: str, fn: str):
            mod = importlib.import_module(f".{module}", package=__name__)
            return getattr(mod, fn)


        def run() -> None:
            for name in SECTIONS:
                lazy_call(name, "run")()
        ''',
    )
    write(pkg / "runner" / "integrity.py", "def run() -> None:\n    return None\n")
    write(pkg / "runner" / "classification.py", "def run() -> None:\n    return None\n")
    write(pkg / "runner" / "unlisted.py", "def run() -> None:\n    return None\n")
    write(pkg / "scripts" / "go.py", "from demo.runner import run\n")
    return pkg


def test_dynamic_import_module_registry_is_followed(tmp_path: Path) -> None:
    pkg = make_dispatch_package(tmp_path)
    report = build_report(pkg, [pkg / "scripts"])
    dynamic = {edge.target for edge in report.edges if edge.kind == "import-module-dynamic"}
    assert dynamic == {"demo.runner.integrity", "demo.runner.classification"}
    assert report.orphans == ["demo.runner.unlisted"]


def test_literal_import_module_name_is_followed(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "section.py", "def run() -> None:\n    return None\n")
    write(
        pkg / "scripts" / "literal_dispatch.py",
        """
        import importlib

        importlib.import_module("demo.section")
        """,
    )
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == []
    kinds = {edge.kind for edge in report.edges if edge.target == "demo.section"}
    assert kinds == {"import-module"}


def test_spec_from_file_location_edge_from_an_external_test(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "harness_target.py", "def helper() -> int:\n    return 1\n")
    tests = tmp_path / "tests"
    write(
        tests / "test_harness.py",
        """
        import importlib.util
        from pathlib import Path

        package = Path(__file__).resolve().parents[1] / "demo"
        spec = importlib.util.spec_from_file_location(
            "harness_target", package / "harness_target.py")
        """,
    )
    report = build_report(pkg, [pkg / "scripts", tests])
    assert report.orphans == []
    edge_kinds = {edge.kind for edge in report.edges if edge.target == "demo.harness_target"}
    assert edge_kinds == {"spec-from-file-location"}


def test_spec_from_file_location_through_a_local_binding(tmp_path: Path) -> None:
    """The path is often bound one statement earlier; the resolver follows it."""
    pkg = make_package(tmp_path)
    write(pkg / "bound_target.py", "def helper() -> int:\n    return 1\n")
    tests = tmp_path / "tests"
    write(
        tests / "test_bound.py",
        """
        import importlib.util
        from pathlib import Path

        path = Path(__file__).parents[1] / 'demo/bound_target.py'
        spec = importlib.util.spec_from_file_location('bound_target', path)
        """,
    )
    report = build_report(pkg, [pkg / "scripts", tests])
    assert report.orphans == []


def test_unresolvable_spec_target_is_reported_not_silently_dropped(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    tests = tmp_path / "tests"
    write(
        tests / "test_outside.py",
        """
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "outsider", Path(__file__).parent / "not_in_the_package.py")
        """,
    )
    report = build_report(pkg, [pkg / "scripts", tests])
    kinds = [item.kind for item in report.unresolved_dynamic]
    assert kinds == ["spec-from-file-location"]
    assert report.passed is True


def test_sys_path_insert_then_bare_import_is_an_edge(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "scripts" / "sibling_helper.py", "VALUE = 5\n")
    write(
        pkg / "scripts" / "uses_sibling.py",
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))

        import sibling_helper
        """,
    )
    report = build_report(pkg, [pkg / "scripts"])
    edges = {(edge.target, edge.kind) for edge in report.edges}
    assert ("demo.scripts.sibling_helper", "sys-path-insert") in edges


def test_sys_path_insert_reaches_a_library_module(tmp_path: Path) -> None:
    """Without the sys.path edge the target would be reported as an orphan."""
    pkg = root_with_sys_path_only_consumer(tmp_path)
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == []


def root_with_sys_path_only_consumer(root: Path) -> Path:
    pkg = root / "demo"
    write(pkg / "__init__.py", "")
    write(pkg / "inner" / "__init__.py", "")
    write(pkg / "inner" / "only_bare.py", "VALUE = 6\n")
    write(
        pkg / "scripts" / "bare_consumer.py",
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inner"))

        from only_bare import VALUE
        """,
    )
    return pkg


def test_relative_imports_resolve_against_the_package(tmp_path: Path) -> None:
    pkg = tmp_path / "demo"
    write(pkg / "__init__.py", "from .lib import VALUE\n")
    write(pkg / "lib.py", "VALUE = 7\n")
    write(pkg / "scripts" / "run.py", "import demo\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert report.orphans == []
    assert any(edge.kind == "relative" for edge in report.edges)


def test_syntax_error_is_reported_and_fails_the_gate(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    write(pkg / "broken_syntax.py", "def oops(:\n")
    report = build_report(pkg, [pkg / "scripts"])
    assert len(report.parse_errors) == 1
    assert "broken_syntax.py" in report.parse_errors[0]
    assert report.passed is False


def test_missing_package_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ClosureError):
        build_report(tmp_path / "absent", [tmp_path])


def test_missing_root_directory_raises(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    with pytest.raises(ClosureError):
        build_report(pkg, [tmp_path / "absent"])


def test_empty_package_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ClosureError):
        build_report(empty, [empty])


def test_cli_exit_codes_and_json_receipt(tmp_path: Path) -> None:
    pkg = make_package(tmp_path)
    receipt = tmp_path / "out" / "g4.json"
    argv = ["--package", str(pkg), "--roots", str(pkg / "scripts"), "--json", str(receipt)]
    assert main(argv) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G4"
    assert payload["passed"] is True
    assert payload["counts"]["orphans"] == 0

    write(pkg / "stranded.py", "VALUE = 8\n")
    assert main(argv) == 1
    assert json.loads(receipt.read_text(encoding="utf-8"))["orphans"] == ["demo.stranded"]


def test_cli_reports_bad_arguments_with_status_two(tmp_path: Path) -> None:
    assert main(["--package", str(tmp_path / "absent"), "--roots", str(tmp_path)]) == 2
