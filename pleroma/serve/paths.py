"""Filesystem anchors the server resolves, derived from this file's location.

Every anchor is computed from ``__file__`` rather than the working directory,
so the server finds the same files whichever directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

#: The repo root: harvest subprocesses run with this as cwd.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: The legacy single-file UI page, read from disk on every GET / so an edit to
#: it shows on the next reload without a restart.
UI_PATH: Path = Path(__file__).resolve().parent / "static" / "legacy_ui.html"

#: ★ The atlas fallback for GET /atlas and /wear_code's axis mode. On disk this
#: is a STALE rank-8 atlas whose group ids are not the served bank's labels, so
#: a lookup through it fails as "not in bank" rather than naming the wrong
#: atlas; pass --atlas-report instead. Nothing in the UI calls either route —
#: a dead path, kept because /wear_code's ``code`` mode is the only way to
#: wear a code from another conversation.
DEFAULT_ATLAS_REPORT: Path = REPO_ROOT / "outputs/loom/atlas/atlas_report.json"
