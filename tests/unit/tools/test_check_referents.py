"""Unit tests for the G3 referent checker.

The fixtures spell out the referents in full, which is the point of the test; the
checker's own patterns are written so they do not match themselves. Each case is
built inside a throwaway tree with its own allowlist, so what passes and what
flags is a property of the rules rather than of this repository's contents.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.check_referents import (
    CITATION_RULE,
    DEFAULT_ALLOWLIST,
    DEFAULT_REPO,
    DEFERRAL_RULE,
    MODULE_RULE,
    PATH_RULE,
    PRIVATE_RULE,
    ReferentError,
    TreeIndex,
    build_report,
    load_allowlist,
    main,
    rule_names,
)

ALLOWLIST_BODY = """
# A trimmed copy of the shipped data file, enough to exercise every branch.
https?://[^\\s`'"<>)\\]]+
\\b(?:meta-llama|google|Qwen)/[A-Za-z0-9._-]+
\\bmetadata\\.json
\\bsignatures(?:_v[0-9]+)?/[A-Za-z0-9_.<>/*-]*

[defer]
\\bas discussed\\b
\\bsee the (?:earlier|previous)\\b

[cite]
§
(?i:prereg)
\\bA[1-7]\\b
"""


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A miniature repository: a package with one importable module and a symbol."""
    root = tmp_path / "repo"
    write(root / "pkg" / "__init__.py", "")
    write(
        root / "pkg" / "real.py",
        """
        WIDTH = 3


        def measure() -> int:
            return WIDTH
        """,
    )
    write(root / "docs" / "note.md", "a document that exists\n")
    write(root / "allow.txt", ALLOWLIST_BODY)
    return root


def check(tree: Path, body: str, name: str = "subject.py") -> list[tuple[str, str]]:
    """Scan one source file inside the fixture tree; returns (rule, match) pairs."""
    target = write(tree / name, body)
    report = build_report(
        roots=[target], repo=tree, allowlist_path=tree / "allow.txt"
    )
    assert not report.read_errors, report.read_errors
    return [(v.rule, v.match) for v in report.violations]


def test_private_tree_path_flags(tree: Path) -> None:
    hits = check(tree, '"""The reason is in research/notes/some-memo.md."""\n')
    assert any(rule == PRIVATE_RULE for rule, _ in hits)


def test_home_relative_and_absolute_paths_flag(tree: Path) -> None:
    hits = check(
        tree,
        """
        # The bank lives at ~/banks/vectors.npz.
        # The donor tree is /srv/private/anamnesis/notes.
        """,
    )
    assert [rule for rule, _ in hits] == [PRIVATE_RULE, PRIVATE_RULE]


def test_planning_document_name_flags(tree: Path) -> None:
    hits = check(tree, "# Ruled in ORIENTATION-brief-2026-01-01.md.\n")
    assert hits and hits[0][0] == PRIVATE_RULE


def test_a_shouted_head_with_a_spelled_out_tail_is_a_document_name(tree: Path) -> None:
    """The second title shape: capitals, then a hyphenated sentence saying the subject."""
    hits = check(
        tree,
        '"""The two sources are ruled in SPEC-path-receptacles-and-span-coverage.md."""\n',
    )
    assert any(rule == PRIVATE_RULE for rule, _ in hits), hits


def test_a_short_hyphenated_compound_is_prose_not_a_title(tree: Path) -> None:
    """A contrast name and a category label join two words; a title spells out a subject."""
    assert check(tree, "# The ANCHOR-versus-RECENCY contrast, over PORT-as-is items.\n") == []


def test_provenance_citations_flag(tree: Path) -> None:
    """A citation of a document outside the tree: the reader has nothing to open."""
    hits = check(
        tree,
        '''
        # The template is one per arm (prereg, one analysis template per model).
        """The row is read the way A3 defined it."""
        ''',
    )
    assert sorted(hits) == [
        (CITATION_RULE, "A3"),
        (CITATION_RULE, "prereg"),
    ], hits


def test_a_referent_in_a_raised_message_flags(tree: Path) -> None:
    """The message a program says out loud is read by a stranger, who can follow nothing."""
    hits = check(
        tree,
        '''
        def run(count: int) -> None:
            if not count:
                raise ValueError("no rows under arms — see research/notes/some-memo.md")
        ''',
    )
    assert [rule for rule, _ in hits] == [PRIVATE_RULE]


def test_a_referent_in_a_logged_message_flags(tree: Path) -> None:
    hits = check(
        tree,
        '''
        import logging

        LOGGER = logging.getLogger(__name__)


        def run() -> None:
            LOGGER.warning("falling back to pkg/missing.py")
        ''',
    )
    assert hits == [(PATH_RULE, "pkg/missing.py")]


def test_a_substitution_inside_a_message_is_code_not_prose(tree: Path) -> None:
    """A filename built in an f-string is data the program handles, not a claim."""
    hits = check(
        tree,
        '''
        from pathlib import Path


        def run(root: Path) -> None:
            raise FileNotFoundError(f"unreadable: {root / 'pkg/missing.py'}")
        ''',
    )
    assert hits == []


def test_nonexistent_repo_path_flags(tree: Path) -> None:
    hits = check(tree, '"""Handled in pkg/missing.py."""\n')
    assert hits == [(PATH_RULE, "pkg/missing.py")]


def test_existing_repo_path_passes(tree: Path) -> None:
    assert check(tree, '"""Handled in pkg/real.py, described in docs/note.md."""\n') == []


def test_resolvable_module_path_passes(tree: Path) -> None:
    assert check(tree, "# The width is pkg.real.WIDTH, read by pkg.real.measure.\n") == []


def test_unresolvable_module_path_flags(tree: Path) -> None:
    hits = check(tree, "# See pkg.ghost.measure for the fallback.\n")
    assert hits == [(MODULE_RULE, "pkg.ghost.measure")]


def test_module_without_the_named_symbol_flags(tree: Path) -> None:
    hits = check(tree, "# The constant is pkg.real.HEIGHT.\n")
    assert hits == [(MODULE_RULE, "pkg.real.HEIGHT")]


def test_url_passes(tree: Path) -> None:
    assert check(tree, "# The record is https://github.com/example/frozen/tree/main.\n") == []


def test_hugging_face_model_id_passes(tree: Path) -> None:
    assert (
        check(
            tree,
            "# Presets cover meta-llama/Llama-3.1-8B-Instruct and Qwen/Qwen2.5-7B.\n",
        )
        == []
    )


def test_data_shape_paths_pass(tree: Path) -> None:
    assert check(tree, '"""Reads signatures_v3/ beside metadata.json."""\n') == []


def test_referent_inside_a_string_literal_does_not_flag(tree: Path) -> None:
    """The same text as data: a fixture, a message, a key. Data is not prose."""
    body = """
        MESSAGE = "the reason is in research/notes/some-memo.md"
        PATHS = ["pkg/missing.py", "~/banks/vectors.npz"]


        def explain() -> str:
            note = "as discussed, pkg.ghost.measure handles it"
            return note + MESSAGE
        """
    assert check(tree, body) == []


def test_memo_style_name_flags(tree: Path) -> None:
    hits = check(tree, "# The revision is in the v3-delta-memo.\n")
    assert hits == [(PRIVATE_RULE, "v3-delta-memo")]


def test_deferral_phrase_flags(tree: Path) -> None:
    hits = check(tree, "# Kept as discussed, with the same threshold.\n")
    assert hits == [(DEFERRAL_RULE, "as discussed")]


def test_deferral_phrase_heading_a_sentence_flags(tree: Path) -> None:
    """A phrase is a phrase capitalised: [defer] patterns ignore case, [allow] do not."""
    hits = check(tree, '"""See the earlier reasoning for the threshold."""\n')
    assert hits == [(DEFERRAL_RULE, "See the earlier")]


def test_glob_and_placeholder_are_schema_not_referents(tree: Path) -> None:
    assert check(tree, '"""Writes pkg/<cell>/gen_*.npz per cell."""\n') == []


def test_relative_path_from_a_sibling_module_resolves(tree: Path) -> None:
    """A pointer written from inside the package still names a file here."""
    assert check(tree, "# The contract is in real.py.\n") == []


def test_the_checkers_pass_their_own_gate() -> None:
    """The rule applies to the checker and to everything beside it in `tools/`.

    The gate's own verdict over the package and the suite is the command in
    `CONTRIBUTING.md`, run per pull request; a unit test asserting the whole tree is
    clean would make one contributor's unrelated prose fail everybody's suite. What
    this pins is the property the checker cannot be excused from: it holds itself to
    the rule it enforces, which is why its patterns bracket a character.
    """
    # The gate modules only: the private exporter beside them in `tools/`
    # names the private trees because refusing to ship them is its job; it never
    # ships, and the export's own G3 run covers everything that does.
    gates = [DEFAULT_REPO / "tools" / f"{name}.py" for name in (
        "check_timelessness", "check_referents", "check_import_closure",
        "check_test_retention", "surface_report")]
    report = build_report(
        roots=gates,
        repo=DEFAULT_REPO,
        allowlist_path=DEFAULT_ALLOWLIST,
    )
    assert report.violations == [], report.violations
    assert report.read_errors == []
    assert report.files_scanned > 0


def test_unparseable_file_is_reported_not_skipped(tree: Path) -> None:
    target = write(tree / "broken.py", "def f(:\n    pass\n")
    report = build_report(roots=[target], repo=tree, allowlist_path=tree / "allow.txt")
    assert report.read_errors and "broken.py" in report.read_errors[0]
    assert not report.passed


def test_missing_root_is_an_error(tree: Path) -> None:
    with pytest.raises(ReferentError, match="does not exist"):
        build_report(roots=[tree / "absent"], repo=tree, allowlist_path=tree / "allow.txt")


def test_repo_that_is_not_a_directory_is_an_error(tree: Path) -> None:
    with pytest.raises(ReferentError, match="not a directory"):
        TreeIndex.build(tree / "allow.txt")


def test_allowlist_missing_is_an_error(tree: Path) -> None:
    with pytest.raises(ReferentError, match="unreadable"):
        load_allowlist(tree / "no-such-allowlist.txt")


def test_allowlist_invalid_regex_is_an_error(tree: Path) -> None:
    path = write(tree / "bad.txt", "# a comment\n[a-z\n")
    with pytest.raises(ReferentError, match="invalid regex"):
        load_allowlist(path)


def test_allowlist_unknown_section_is_an_error(tree: Path) -> None:
    path = write(tree / "sections.txt", "[allow]\nfoo\n[nonsense]\nbar\n")
    with pytest.raises(ReferentError, match="unknown section"):
        load_allowlist(path)


def test_allowlist_sections_are_read_into_their_own_buckets(tree: Path) -> None:
    loaded = load_allowlist(tree / "allow.txt")
    assert len(loaded.allow) == 4
    assert len(loaded.defer) == 2
    assert len(loaded.cite) == 3
    url = "https://example.com/x"
    assert loaded.mask(f"see {url} now") == "see " + " " * len(url) + " now"


def test_shipped_allowlist_compiles() -> None:
    loaded = load_allowlist(DEFAULT_ALLOWLIST)
    assert loaded.allow and loaded.defer


def test_cli_writes_a_receipt_and_exits_nonzero_on_a_violation(
    tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tree / "subject.py", "# Handled in pkg/missing.py.\n")
    receipt = tmp_path / "receipts" / "referents.json"
    code = main(
        [
            "--root",
            str(tree / "subject.py"),
            "--repo",
            str(tree),
            "--allowlist",
            str(tree / "allow.txt"),
            "--json",
            str(receipt),
        ]
    )
    assert code == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G3-referents"
    assert payload["passed"] is False
    assert payload["counts"]["by_rule"] == {PATH_RULE: 1}
    assert "unresolved-path" in capsys.readouterr().out


def test_cli_exits_two_when_the_check_cannot_run(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tree / "absent"), "--repo", str(tree)])
    assert code == 2
    assert "check_referents" in capsys.readouterr().err


def test_rule_names_are_distinct() -> None:
    names = list(rule_names())
    assert len(names) == len(set(names)) == 5


def test_a_version_tagged_document_title_written_with_spaces_flags(tree: Path) -> None:
    """A title a reader cannot open is flagged spaced as well as hyphenated."""
    hits = check(tree, '"""Read the gap against the v3 delta memo."""\n')
    assert any(rule == PRIVATE_RULE for rule, _ in hits), hits


def test_prose_naming_no_document_does_not_flag(tree: Path) -> None:
    """A bare noun names nothing openable, so it reads as prose rather than a title."""
    hits = check(tree, '"""Kept because the note in the header explains the bound."""\n')
    assert hits == []
