"""Unit tests for the G3 timelessness checker.

The fixtures spell the flagged phrases out in full, which is the point of the
test; the checker's own patterns are written so they do not match themselves.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.check_timelessness import (
    DEFAULT_ALLOWLIST,
    TimelessnessError,
    build_report,
    documentation_text,
    load_allowlist,
    main,
    used_to_violation,
)


def write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def rules_hit(report: object) -> set[str]:
    return {violation.rule for violation in report.violations}  # type: ignore[attr-defined]


def test_shipped_allowlist_compiles() -> None:
    patterns = load_allowlist(DEFAULT_ALLOWLIST)
    assert patterns


def test_marker_comments_are_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        """
        # TODO: tighten this
        VALUE = 1  # FIXME
        OTHER = 2  # HACK around the loader
        """,
    )
    report = build_report([tmp_path / "pkg"])
    assert len(report.violations) == 3
    assert rules_hit(report) == {"marker-comment"}
    assert report.passed is False


def test_changelog_phrases_in_comments_and_docstrings(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """This loop was previously two passes over the same file."""

        # we now cache the matrix once per generation
        VALUE = 1

        def f() -> str:
            """The split is no longer identical to the banked one."""
            return "ok"
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {
        "changelog-prior-state",
        "changelog-we-now",
        "changelog-no-longer",
    }


def test_string_data_is_not_documentation(tmp_path: Path) -> None:
    """Fixtures, log lines and error messages carry text the rules do not police."""
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        FIXTURE = "# TODO: tighten this\\nVALUE = 1\\n"
        LOG = "we now cache the matrix; the split is no longer identical"
        BANNER = "rewrote the loader 2026-07-18"
        PAYLOAD = {"note": "the family used to host these", "when": "2026-07-18"}


        def message(n: int) -> str:
            return f"{n} paths dropped - split no longer identical to the bank"
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert report.exempted == []
    assert report.passed is True


def test_a_raised_message_is_prose(tmp_path: Path) -> None:
    """A stranger meets this text when something breaks, with nothing else to go on."""
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        def check(value: int) -> None:
            if value < 0:
                raise ValueError("negative since 2026-07-18; we now refuse it")
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-prose", "changelog-we-now"}


def test_a_logged_message_is_prose(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        import logging

        LOGGER = logging.getLogger(__name__)


        def run() -> None:
            LOGGER.warning("the split is no longer identical to the banked one")
            LOGGER.info("calibration banked 2026-07-12")
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-no-longer", "dated-prose"}


def test_an_fstring_substitution_is_code_not_prose(tmp_path: Path) -> None:
    """Inside the braces is an expression: a filename there is data, not a claim."""
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        from pathlib import Path


        def load(root: Path) -> None:
            raise FileNotFoundError(f"unreadable: {root / '2026-07-18.json'}")
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert report.passed is True


@pytest.mark.parametrize(
    "body",
    [
        'raise ValueError("─ α " + "banked 2026-07-12")',
        'raise ValueError("banked 2026-07-12: %s" % value)',
        'raise ValueError("banked 2026-07-12: {}".format(value))',
        'raise ValueError("banked 2026-07-12" if value else "fresh")',
    ],
    ids=["joined", "percent", "format", "conditional"],
)
def test_an_assembled_message_is_read_piece_by_piece(tmp_path: Path, body: str) -> None:
    """A message written in parts is still one sentence a reader is handed.

    The first case also pins the column arithmetic: the parse tree counts UTF-8
    bytes, so a non-ASCII piece in front of the dated one would slice the line in
    the wrong place if the offsets were used as they come.
    """
    write(tmp_path / "pkg" / "mod.py", f"def f(value: int) -> None:\n    {body}\n")
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-prose"}


def test_a_dict_lookup_and_a_report_line_stay_data(tmp_path: Path) -> None:
    """Only the literal a call is handed is prose; a key or a written row is not."""
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        MESSAGES = {"stale 2026-07-18": "we now refuse it"}


        def explain(rows: list[str]) -> None:
            rows.append("floor observed bitwise zero 2026-07-12")
            raise ValueError(MESSAGES["stale 2026-07-18"])
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert report.passed is True


def test_masking_keeps_columns_and_blanks_the_rest(tmp_path: Path) -> None:
    """The mask preserves offsets, so a reported line still lines up with the file."""
    source = 'VALUE = "2026-07-18"  # banked 2026-07-12\n'
    masked = documentation_text(tmp_path / "mod.py", source)
    assert masked == {1: " " * 22 + "# banked 2026-07-12"}


def test_the_same_text_in_a_comment_or_docstring_is_flagged(tmp_path: Path) -> None:
    """The counterpart of the string-data test: prose gets no exemption."""
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """We now cache the matrix once per generation."""

        # TODO: tighten this
        # rewrote the loader 2026-07-18
        VALUE = 1


        def f() -> None:
            """The split is no longer identical to the banked one."""
            return None
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {
        "changelog-we-now",
        "changelog-no-longer",
        "marker-comment",
        "dated-prose",
    }


def test_only_the_first_statement_string_counts_as_a_docstring(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """The real docstring."""
        VALUE = 1
        """A loose string the parser does not treat as documentation: we now cache."""
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []


def test_class_and_method_docstrings_are_documentation(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        class Thing:
            """Holds what the loader previously rebuilt per generation."""

            def run(self) -> None:
                """No longer identical to the banked split."""
                return None
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-prior-state", "changelog-no-longer"}


def test_a_marker_inside_a_string_on_a_commented_line(tmp_path: Path) -> None:
    """A line's comment is what the marker rule reads, and the line qualifies."""
    write(tmp_path / "pkg" / "mod.py", 'VALUE = "plain"  # FIXME: tighten this\n')
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"marker-comment"}


def test_changed_in_is_flagged(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# the layer list changed in the second run\nVALUE = 1\n")
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-changed-in"}


def test_code_identifier_is_not_mistaken_for_prose(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "previously_seen = set()\nno_longer_valid = False\n")
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert report.passed is True


def test_used_to_narrating_an_edit_is_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """Summaries the windowed family used to host."""
        VALUE = 1
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"changelog-used-to"}


def test_used_to_as_a_present_tense_constraint_is_not_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "mod.py",
        '''
        """A lower bound is never reported as a floor or used to quote effect sizes."""
        VALUE = 1
        ''',
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []


def test_used_to_guard_directly() -> None:
    assert used_to_violation("the family used to host these summaries") == "used to host"
    assert used_to_violation("these states are used to compute features") is None
    assert used_to_violation("nothing followed by a verb here: used to") is None


def test_dated_comment_is_flagged(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# rewrote the loader 2026-07-18\nVALUE = 1\n")
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-prose"}


def test_a_date_that_cites_evidence_is_flagged_too(tmp_path: Path) -> None:
    """Dates come out of code whatever they date: the record holds the evidence."""
    write(
        tmp_path / "pkg" / "mod.py",
        """
        # vmb matrix completion pass 1 (prereg Stage A(ii), census 2026-07-12)
        VALUE = 1
        """,
    )
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-prose"}
    assert report.exempted == []
    assert report.passed is False


def test_a_date_inside_a_public_url_is_exempt(tmp_path: Path) -> None:
    """The one surviving exemption: strip the date and the link stops resolving."""
    write(
        tmp_path / "pkg" / "mod.py",
        "# The frozen record is https://example.com/2026-07-12/anamnesis.\nVALUE = 1\n",
    )
    report = build_report([tmp_path / "pkg"])
    assert report.violations == []
    assert len(report.exempted) == 1
    assert report.passed is True


def test_dated_docstring_is_flagged(tmp_path: Path) -> None:
    """A docstring is prose, so the date rule reads it: no `#` is needed."""
    write(tmp_path / "pkg" / "mod.py", '"""Banked 2026-07-12."""\nVALUE = 1\n')
    report = build_report([tmp_path / "pkg"])
    assert rules_hit(report) == {"dated-prose"}


def test_custom_allowlist_is_honoured(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "# rebuilt for the widget 2026-07-18\nVALUE = 1\n")
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# only widgets are citable here\n\\bwidget\\b\n", encoding="utf-8")
    report = build_report([tmp_path / "pkg"], allowlist_path=allowlist)
    assert report.violations == []
    assert len(report.exempted) == 1


def test_invalid_allowlist_regex_raises(tmp_path: Path) -> None:
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("[unclosed\n", encoding="utf-8")
    write(tmp_path / "pkg" / "mod.py", "VALUE = 1\n")
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "pkg"], allowlist_path=allowlist)


def test_missing_allowlist_raises(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "VALUE = 1\n")
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "pkg"], allowlist_path=tmp_path / "absent.txt")


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(TimelessnessError):
        build_report([tmp_path / "absent"])


def test_unparseable_file_is_reported_and_fails(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "mod.py", "def broken(:\n")
    report = build_report([tmp_path / "pkg"])
    assert len(report.read_errors) == 1
    assert "mod.py" in report.read_errors[0]
    assert report.passed is False


def test_a_single_file_root_is_accepted(tmp_path: Path) -> None:
    target = write(tmp_path / "pkg" / "mod.py", "# TODO: later\nVALUE = 1\n")
    report = build_report([target])
    assert len(report.violations) == 1


def test_cli_exit_codes_and_json_receipt(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    write(root / "clean.py", "VALUE = 1\n")
    receipt = tmp_path / "out" / "g3.json"
    argv = ["--root", str(root), "--json", str(receipt)]
    assert main(argv) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["gate"] == "G3"
    assert payload["passed"] is True

    write(root / "dirty.py", "# TODO: fix\nVALUE = 1\n")
    assert main(argv) == 1
    assert json.loads(receipt.read_text(encoding="utf-8"))["counts"]["violations"] == 1


def test_cli_reports_a_bad_allowlist_with_status_two(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    write(root / "mod.py", "VALUE = 1\n")
    assert main(["--root", str(root), "--allowlist", str(tmp_path / "absent.txt")]) == 2
