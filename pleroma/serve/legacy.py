"""LOOM v0 — the model looms over its own futures and bends toward one you pick.

★ The flat ``loom_serve`` namespace. The server lives in the ``pleroma.serve``
package, split by concern (its package docstring is the full design). This
module re-exports every name of the server's flat namespace, sets the
thread-pool pins and logging setup at import, before anything heavy loads,
and is the launch target ``pleroma/config/legacy.py`` renders
(``python -m pleroma.serve.legacy``). New code imports from the owning
``pleroma`` module instead.

Usage (from the repo root, inside the project venv):
    python -m pleroma.serve.legacy \\
        --map data/loom/loom_map_3b.npz \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --calib-dir data/calibration/3b \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --work-dir outputs/loom/sessions --port 8767
Remote GPU box?  ssh -L 8767:localhost:8767 <host>  and browse localhost:8767.
"""

from __future__ import annotations

import argparse  # noqa: F401 — every flat-namespace module global stays importable
import datetime  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import logging
import os
import re  # noqa: F401
import string  # noqa: F401
import subprocess  # noqa: F401
import sys
import threading  # noqa: F401
import time  # noqa: F401
import types
import urllib.error  # noqa: F401
import urllib.request  # noqa: F401
from collections.abc import Callable, Mapping, Sequence  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any

import numpy as np  # noqa: F401

from pleroma.dose.band import (  # noqa: F401
    DoseBand,
    ResidualScale,
    default_sidecar_path,
    dose_band_info_json,
    load_dose_band_file,
    load_residual_scale_file,
    map_fingerprint,
    resolve_dose_band,
)
from pleroma.harvest import pool as harvest_pool  # noqa: F401
from pleroma.probe import orchestrator as loom_probe  # noqa: F401
# ── owned elsewhere in pleroma; re-exported here for flat-namespace callers ───
from pleroma.dose.policy import (  # noqa: F401
    DEFAULT_DOSE_POLICY,
    DOSE_POLICIES,
    DOSE_POLICY_INFO,
    DOSE_SCALE_MAX,
    DOSE_SCALE_MIN,
    DoseScale,
    dose_scale_for,
    effective_alpha_json,
    fan_mean_raw_norms,
    flat_dose_scale,
    resolve_dose_policy,
)
from pleroma.levers.bank import LeverBank  # noqa: F401
from pleroma.levers.kind import (  # noqa: F401
    DEFAULT_LEVER_KIND,
    LEVER_KIND_INFO,
    LEVER_KINDS,
    attach_fan_contrast,
    fan_wear_fields,
    resolve_lever_kind,
    worn_lever_kind,
)
from pleroma.map.loom_map import LoomMap  # noqa: F401
from pleroma.format import modelc
from pleroma.format.prompt import (  # noqa: F401 — re-exported for callers
    PROMPT_MODES,
    RAW_TURN_JOIN,
    render_prompt_ids,
    render_prompt_text,
    render_raw_text,
)
from pleroma.format.trim import ModelCSplit, split_modelc, trim_generated_ids  # noqa: F401
from pleroma.format.chat import build_messages, trim_history  # noqa: F401
from pleroma.net import require_loopback  # noqa: F401

from pleroma.serve.atlas import (  # noqa: F401
    check_atlas_matches_bank,
    resolve_atlas_landmark,
    strip_atlas_suffix,
)
from pleroma.serve.draws import (  # noqa: F401
    BINS_PROMPT_FLOOR,
    attach_for_draw,
    cos_to_worn,
    drawn_under_wear_field,
    loom_turn_seed,
    next_loom_index,
    prefix_fingerprint,
    request_detach_wear,
    request_probe,
    resolve_draw_wear,
)
from pleroma.serve.harvest_client import (  # noqa: F401
    WORKER_STATES,
    _worker_hostport,
    harvest_completion,
    harvest_via_worker,
    pool_health_payload,
    probe_worker_health,
    progress_payload,
    subprocess_visible_presets,
)
from pleroma.serve.info import _probe_cost_or_note, build_info_payload  # noqa: F401
from pleroma.serve.modelc_fields import (  # noqa: F401
    modelc_public_fields,
    retrim_modelc_histories,
)
from pleroma.serve.persistence import (  # noqa: F401
    _FILENAME_SAFE,
    _HISTORY_EXTRAS,
    _MAX_BASENAME_CHARS,
    _ROLES,
    MAX_PERSISTED_LOOMS,
    SESSION_SCHEMA_VERSION,
    SESSIONS_DIRNAME,
    LoomSnapshot,
    RestoreReport,
    SessionListing,
    SessionSnapshot,
    SessionStore,
    WornSnapshot,
    _floats,
    _histories,
    _iso,
    _json_default,
    _obj,
    _opt_ints,
    _opt_lever_kind,
    _opt_str,
    rehydrate_worn_vectors,
    session_basename,
)
from pleroma.serve.policies import (  # noqa: F401
    AUTO_POLICIES,
    AUTO_POLICIES_UNAVAILABLE,
    AUTO_POLICY_INFO,
    PROBE_ONLY_POLICIES,
    pick_auto,
)
from pleroma.serve.session import (  # noqa: F401
    BRANCHES,
    LoomSession,
    State,
    n_assistant_turns,
)

# Thread-pool pins, set at import: the harvest subprocesses the loom spawns
# (pleroma.harvest.bins) inherit them and do not pin themselves.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("loom_serve")

MODELC_BRIDGE = modelc.BRIDGE_LINE
MODELC_USER_LABEL = modelc.USER_LABEL
MODELC_ASSISTANT_LABEL = modelc.ASSISTANT_LABEL
MODELC_HEADERS = modelc.HEADERS
MODELC_DEFAULT_HEADER = modelc.DEFAULT_HEADER
MODELC_TEMPERATURE = modelc.TEMPERATURE
MODELC_TOP_P = modelc.TOP_P
MODELC_STOPS = modelc.STOPS
resolve_modelc_header = modelc.resolve_header
render_modelc_text = modelc.render_document
count_dream_blocks = modelc.count_dream_blocks


# ── module globals owned elsewhere AND rebound at runtime ─────────────────────
# `_SUBPROCESS_PRESETS` is a one-shot cache that lives in
# pleroma.serve.harvest_client. A plain re-export would be a stale copy, and a
# caller that resets it (`loom_serve._SUBPROCESS_PRESETS = False`, as
# test_harvest_worker_gpu_lane does) would reset nothing. So reads resolve to
# the live value (PEP 562 module __getattr__) and writes are forwarded.
from pleroma.serve import harvest_client as _harvest_client  # noqa: E402

_FORWARDED_GLOBALS: dict[str, Any] = {"_SUBPROCESS_PRESETS": _harvest_client}


def __getattr__(name: str) -> Any:
    owner = _FORWARDED_GLOBALS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(owner, name)


class _ForwardingModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        owner = _FORWARDED_GLOBALS.get(name)
        if owner is not None:
            setattr(owner, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ForwardingModule


# ── main: delegated to the package ─────────────────────────────────────────────


def main() -> int:
    """Run the loom server (``pleroma.serve.app.main``) on ``sys.argv[1:]``.

    A private launcher may rewrite ``sys.argv`` and call this with no
    arguments; ``pleroma/config/legacy.py`` renders ``-m
    pleroma.serve.legacy`` argv. Both work."""
    from pleroma.serve.app import main as _serve_main

    return _serve_main()


if __name__ == "__main__":
    sys.exit(main())
