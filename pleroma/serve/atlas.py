"""Atlas landmarks for POST /wear_code and GET /atlas.

★ A DEAD PATH, kept: the UI calls neither route, and the default atlas path
holds a stale atlas (``pleroma.serve.paths.DEFAULT_ATLAS_REPORT``). It stays
because /wear_code's ``code`` mode is the only way to wear a code taken from
another conversation, and whether that capability goes is a science call.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pleroma.serve.paths import DEFAULT_ATLAS_REPORT

if TYPE_CHECKING:
    from pleroma.levers.bank import LeverBank


def strip_atlas_suffix(entry: str) -> str:
    """Atlas extremes read 'w5_og01155|orig(op)' — the trailing class tag is
    decoration, not part of the group label."""
    return entry.split("(", 1)[0].strip()


def resolve_atlas_landmark(atlas: Mapping[str, Any], axis: int, pole: str,
                           rank: int = 0) -> str:
    """Group label of the rank-th extreme group at one pole of one atlas axis."""
    if pole not in ("low", "high"):
        raise ValueError(f"pole must be 'low' or 'high', got {pole!r}")
    axes = atlas.get("axes")
    if not isinstance(axes, list):
        raise ValueError("atlas report has no 'axes' list")
    match = [a for a in axes if int(a.get("axis", -1)) == int(axis)]
    if not match:
        raise ValueError(f"atlas has no axis {axis} (has 0..{len(axes) - 1})")
    entries = match[0].get(f"{pole}_extreme") or []
    if rank >= len(entries):
        raise ValueError(f"axis {axis} {pole}_extreme has {len(entries)} "
                         f"entries, asked for rank {rank}")
    return strip_atlas_suffix(str(entries[rank]))


def check_atlas_matches_bank(atlas: Mapping[str, Any], bank: "LeverBank",
                             atlas_path: str) -> None:
    """Refuse an atlas whose groups are not THIS bank's groups, and say why.

    ★ WHY. `do_wear_code`'s atlas fallback is a fixed path, and
    on disk that path holds the **stale w4 atlas**: rank 8, 138 groups, ids like
    `cc1|orig` / `mf1|orig`, against a w5 bank of 679 `w5_*` groups. Resolving an
    axis pole against it fails in `vectors_for` with *"group 'dc14|orig' is not
    in …"* — loudly, but the message blames the group and never says **you are
    reading the wrong atlas**, which is the actual fault and the one a tired
    operator will not guess at 3am.

    Worse than the confusing message: if a future bank ever shares short ids with
    an old atlas, the resolve would SUCCEED and wear the wrong group. So this
    checks membership up front rather than trusting the label space.
    """
    axes = atlas.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError(f"{atlas_path}: no 'axes' list — not an atlas report")
    found = 0
    for ax in axes:
        for pole in ("low_extreme", "high_extreme"):
            for entry in (ax.get(pole) or []):
                if strip_atlas_suffix(str(entry)) in bank.index_of:
                    found += 1
    if found == 0:
        sample_atlas = [strip_atlas_suffix(str(e))
                        for ax in axes[:1]
                        for e in (ax.get("low_extreme") or [])[:2]]
        raise ValueError(
            f"{atlas_path} does not describe the loaded bank: not ONE of its "
            f"axis extremes is a group in {bank.path} "
            f"({len(bank.index_of)} groups). Atlas ids look like "
            f"{sample_atlas}, bank labels look like {bank.labels[:2]}. "
            "This is almost certainly the wrong atlas for this bank — pass "
            "--atlas-report pointing at the atlas built FROM this bank "
            "(e.g. outputs/loom/atlas_w5_r32/atlas_report.json), or address "
            "the landmark directly with {'group': '<label>'}.")


def atlas_report_path(atlas_report: str | None) -> Path:
    """``--atlas-report`` if given, else the fixed fallback path (stale w4 atlas).

    One resolution for both callers (GET /atlas and /wear_code's axis mode)."""
    return Path(atlas_report) if atlas_report is not None else DEFAULT_ATLAS_REPORT
