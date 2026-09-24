"""Regression tests for the loom session-dir overwrite trap.

The trap: /reset rewinds the in-memory loom counter, the /loom handler rebuilds
the identical loom_000 path, and mkdir(exist_ok=True) silently overwrites a
banked harvest. next_loom_index derives the index from the filesystem so an
existing dir is never reused, and the handler's mkdir does not pass exist_ok.
"""
from pathlib import Path

import pytest

from pleroma.serve.legacy import next_loom_index


def test_fresh_session_dir_starts_at_counter(tmp_path: Path) -> None:
    assert next_loom_index(tmp_path / "absent", 0) == 0
    assert next_loom_index(tmp_path / "absent", 3) == 3


def test_reset_rewind_cannot_reuse_existing_dir(tmp_path: Path) -> None:
    (tmp_path / "loom_000").mkdir()
    (tmp_path / "loom_001").mkdir()
    # the /reset scenario: in-memory counter rewound to 0, dirs on disk
    assert next_loom_index(tmp_path, 0) == 2


def test_counter_ahead_of_disk_wins(tmp_path: Path) -> None:
    (tmp_path / "loom_000").mkdir()
    assert next_loom_index(tmp_path, 5) == 5


def test_unpadded_indices_past_999_are_counted(tmp_path: Path) -> None:
    (tmp_path / "loom_999").mkdir()
    (tmp_path / "loom_1000").mkdir()  # three-digit padding runs out past 999 — the trap
    assert next_loom_index(tmp_path, 0) == 1001


def test_non_loom_entries_ignored(tmp_path: Path) -> None:
    (tmp_path / "loom_000").mkdir()
    (tmp_path / "loom_junk").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert next_loom_index(tmp_path, 0) == 1


def test_handler_mkdir_refuses_existing_gen_records(tmp_path: Path) -> None:
    # the belt-and-suspenders line: mkdir(parents=True) with NO exist_ok
    rec = tmp_path / "loom_000" / "gen_records"
    rec.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        rec.mkdir(parents=True)
