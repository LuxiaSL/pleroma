"""The `pleroma.serve.legacy` shim keeps every name of the single-module server.

The server lives in the modules of `pleroma/serve/`; `pleroma.serve.legacy`
re-exports the names callers written against the single-module server import
(tests, launch scripts, the model-C lane). It must keep all 138 of its module
globals — frozen below, as `sorted(vars(module))` minus dunders.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

LEGACY_NAMES = """
AUTO_POLICIES AUTO_POLICIES_UNAVAILABLE AUTO_POLICY_INFO Any
BINS_PROMPT_FLOOR BRANCHES BaseHTTPRequestHandler Callable
DEFAULT_DOSE_POLICY DEFAULT_LEVER_KIND DOSE_POLICIES DOSE_POLICY_INFO
DOSE_SCALE_MAX DOSE_SCALE_MIN DoseBand DoseScale LEVER_KINDS LEVER_KIND_INFO
LeverBank LoomMap LoomSession LoomSnapshot MAX_PERSISTED_LOOMS
MODELC_ASSISTANT_LABEL MODELC_BRIDGE MODELC_DEFAULT_HEADER MODELC_HEADERS
MODELC_STOPS MODELC_TEMPERATURE MODELC_TOP_P MODELC_USER_LABEL Mapping
ModelCSplit PROBE_ONLY_POLICIES PROMPT_MODES Path RAW_TURN_JOIN
ResidualScale RestoreReport SESSIONS_DIRNAME SESSION_SCHEMA_VERSION Sequence
SessionListing SessionSnapshot SessionStore State ThreadingHTTPServer
WORKER_STATES WornSnapshot _FILENAME_SAFE _HISTORY_EXTRAS
_MAX_BASENAME_CHARS _ROLES _SUBPROCESS_PRESETS _floats _histories _iso
_json_default _obj _opt_ints _opt_lever_kind _opt_str _probe_cost_or_note _v
_worker_hostport annotations argparse attach_fan_contrast attach_for_draw
build_info_payload build_messages check_atlas_matches_bank cos_to_worn
count_dream_blocks dataclass datetime default_sidecar_path
dose_band_info_json dose_scale_for drawn_under_wear_field
effective_alpha_json fan_mean_raw_norms fan_wear_fields field
flat_dose_scale harvest_completion harvest_pool harvest_via_worker hashlib
json load_dose_band_file load_residual_scale_file logger logging loom_probe
loom_turn_seed main map_fingerprint modelc modelc_public_fields
n_assistant_turns next_loom_index np os pick_auto pool_health_payload
prefix_fingerprint probe_worker_health progress_payload re
rehydrate_worn_vectors render_modelc_text render_prompt_ids
render_prompt_text render_raw_text request_detach_wear request_probe
require_loopback resolve_atlas_landmark resolve_dose_band
resolve_dose_policy resolve_draw_wear resolve_lever_kind
resolve_modelc_header retrim_modelc_histories session_basename split_modelc
string strip_atlas_suffix subprocess subprocess_visible_presets sys
threading time trim_generated_ids trim_history urllib worn_lever_kind
""".split()


def test_the_frozen_list_is_the_one_captured() -> None:
    assert len(LEGACY_NAMES) == 138
    assert len(set(LEGACY_NAMES)) == 138


@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_every_legacy_name_is_still_importable(name: str) -> None:
    from pleroma.serve import legacy as loom_serve

    assert hasattr(loom_serve, name), name


def test_moved_names_are_the_same_objects() -> None:
    """A re-export, not a copy: identity with the new home."""
    from pleroma.serve import legacy as ls
    from pleroma.serve import draws, harvest_client, info, persistence, policies, session

    assert ls.LoomSession is session.LoomSession
    assert ls.State is session.State
    assert ls.SessionStore is persistence.SessionStore
    assert ls.rehydrate_worn_vectors is persistence.rehydrate_worn_vectors
    assert ls.pick_auto is policies.pick_auto
    assert ls.AUTO_POLICY_INFO is policies.AUTO_POLICY_INFO
    assert ls.build_info_payload is info.build_info_payload
    assert ls.attach_for_draw is draws.attach_for_draw
    assert ls.subprocess_visible_presets is harvest_client.subprocess_visible_presets


def test_the_preset_cache_reads_live_and_writes_through() -> None:
    """`_SUBPROCESS_PRESETS` is REBOUND at runtime, so the shim cannot hold a
    copy: reads see the live cache and a reset through the shim resets it."""
    from pleroma.serve import legacy as ls
    from pleroma.serve import harvest_client as hc

    saved = hc._SUBPROCESS_PRESETS
    try:
        hc._SUBPROCESS_PRESETS = ("3b",)
        assert ls._SUBPROCESS_PRESETS == ("3b",)
        ls._SUBPROCESS_PRESETS = False
        assert hc._SUBPROCESS_PRESETS is False
        assert "_SUBPROCESS_PRESETS" not in vars(ls)
    finally:
        hc._SUBPROCESS_PRESETS = saved


def test_importing_the_shim_or_the_package_does_not_import_torch() -> None:
    """loom_serve must stay laptop-importable (the GPU half is imported inside
    main()), and the thread-pool pins must be in place before torch loads."""
    code = (
        "import os, sys\n"
        "import pleroma.serve.legacy\n"
        "import pleroma.serve.app, pleroma.serve.http, pleroma.serve.context\n"
        "assert 'torch' not in sys.modules, 'torch imported at module scope'\n"
        "assert all(os.environ.get(v) == '1' for v in ('OMP_NUM_THREADS', "
        "'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))\n"
        "print('ok')\n")
    env = {k: v for k, v in __import__("os").environ.items()
           if not k.endswith("_NUM_THREADS")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


@pytest.mark.parametrize("module", ["pleroma.serve.legacy", "pleroma.serve"])
def test_both_entry_points_answer_help_without_torch(module: str) -> None:
    """``--help`` exits 0 before anything heavy loads, through the shim (the
    legacy launch) and through the package; both print the same parser."""
    proc = subprocess.run([sys.executable, "-m", module, "--help"], cwd=str(REPO),
                          capture_output=True, text=True, timeout=120,
                          env={**__import__("os").environ, "COLUMNS": "100"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--dose-band-residual-scale" in proc.stdout
    assert proc.stdout.startswith("usage: ")


def test_shim_main_delegates_to_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from pleroma.serve import legacy as loom_serve
    from pleroma.serve import app

    seen: list[object] = []
    monkeypatch.setattr(app, "main", lambda argv=None: seen.append(argv) or 7)
    assert loom_serve.main() == 7 and seen == [None]
