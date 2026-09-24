"""The ONLY import surface of tests/contract/build.

Every test in this area reaches the code under test through the names below.
The tests call each build stage by the name its argv parser carries:

* ``build_levers`` — ``pleroma.map.build.levers``, the lever bank;
* ``fit_loom_map`` — ``pleroma.map.build.wide``, the wide (shelf) map;
* ``v1a_fit`` — ``pleroma.map.build.cv``, the registered grouped-CV fit;
* ``v1a_export`` — ``pleroma.map.build.export``, the served map;
* the ridge — ``pleroma.map.build.regress``.

Moving any of these means re-pointing this file and nothing else. A test that
needs any other edit to survive a move is testing layout, not behaviour.

The CLI stages are called in-process through ``run_*`` (argv in, exit code
out), which is exactly how ``pleroma.stages`` invokes them
(``python -m ... <argv>``). ``SystemExit`` from argparse is turned into its
exit code so an unknown flag is an observable return value, not a crash.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from pleroma.map.build import levers as _build_levers
from pleroma.map.build import wide as _fit_loom_map
from pleroma.map.build import regress as _regress_levers
from pleroma.map.build import export as _v1a_export
from pleroma.map.build import cv as _v1a_fit
from pleroma.map.build.hiddens import corpus_key
from pleroma.map.build.pairs import PairRecord, load_pairs

# ── stage entry points (argv -> exit code) ────────────────────────────────────


def _run_main(prog: str, main: Callable[[], int], argv: Sequence[str]) -> int:
    old = sys.argv
    sys.argv = [prog, *[str(a) for a in argv]]
    try:
        return int(main())
    except SystemExit as exc:  # argparse refusals
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = old


def run_build_levers(argv: Sequence[str]) -> int:
    return _run_main("build_levers", _build_levers.main, argv)


def run_fit_loom_map(argv: Sequence[str]) -> int:
    return _run_main("fit_loom_map", _fit_loom_map.main, argv)


def run_v1a_fit(argv: Sequence[str]) -> int:
    return _run_main("v1a_fit", _v1a_fit.main, argv)


def run_v1a_export(argv: Sequence[str]) -> int:
    return _run_main("v1a_export", _v1a_export.main, argv)


def load_loom_map(path: Path, discriminants: Path) -> Any:
    """The serving reader of a map artifact (imported lazily: the serving stack is slow to import)."""
    from pleroma.map.loom_map import LoomMap

    return LoomMap(path, discriminants)


# ── build_levers ──────────────────────────────────────────────────────────────

BUILD_LEVERS_DEFAULT_MIN_CLUSTER: int = _build_levers.DEFAULT_MIN_CLUSTER

# ── ridge (the canonical closed-form solve and rank truncation) ───────────────

ridge_fit = _regress_levers.ridge_fit
rank_truncate = _regress_levers.rank_truncate
corpus_of_run_dir = _regress_levers.corpus_of_run_dir

# ── v1a_fit (registered grouped-CV validation) ────────────────────────────────

FIT_LAMBDAS = _v1a_fit.LAMBDAS
FIT_RANKS = _v1a_fit.RANKS
FIT_PRIMARY = _v1a_fit.PRIMARY
FIT_FOLDS = _v1a_fit.FOLDS
FIT_FOLD_SEED = _v1a_fit.FOLD_SEED
FIT_PERM_SEED = _v1a_fit.PERM_SEED
FIT_N_PERMS = _v1a_fit.N_PERMS
MIN_FAN = _v1a_fit.MIN_FAN
fan_center = _v1a_fit.fan_center
retrieval = _v1a_fit.retrieval
shelf_baseline_pred = _v1a_fit.shelf_baseline_pred
check_pairs_are_z_corpus = _v1a_fit.check_pairs_are_z_corpus
cv_sweep = _v1a_fit.cv_sweep

# ── v1a_export (the served map) ───────────────────────────────────────────────

EXPORT_LAM = _v1a_export.LAM
EXPORT_RANK = _v1a_export.RANK
BINS_DIM = _v1a_export.BINS_DIM
SELFCHECK_FLOOR = _v1a_export.SELFCHECK_FLOOR
EXPORT_MIN_ROWS = _v1a_export.MIN_ROWS
IDENTITY_TOL = _v1a_export.IDENTITY_TOL
HELDOUT_GRID = _v1a_export.HELDOUT_GRID
HELDOUT_NOT_SUPPLIED = _v1a_export.HELDOUT_NOT_SUPPLIED
FAN_SOURCE_LEVER = _v1a_export.FAN_SOURCE_LEVER
FAN_SOURCE_MEMBER = _v1a_export.FAN_SOURCE_MEMBER
TARGETS = _v1a_export.TARGETS
join_bank = _v1a_export.join_bank
member_fan_index = _v1a_export.member_fan_index
build_target = _v1a_export.build_target
resolve_rank = _v1a_export.resolve_rank
heldout_from_fit_report = _v1a_export.heldout_from_fit_report
grid_key = _v1a_export.grid_key

__all__ = [name for name in dir() if not name.startswith("_")] + [
    "PairRecord", "load_pairs", "corpus_key",
]
