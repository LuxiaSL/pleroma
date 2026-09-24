"""The ONE place pleroma's GPU harvest lane calls anamnesis.

anamnesis is a git submodule at ``vendor/anamnesis``, pinned to an exact
upstream commit (``PINNED_COMMIT``). Everything the GPU ("fast") harvest lane needs
from it goes through this module, written against the pinned commit's PUBLIC
API, so a submodule bump is a change here and nowhere else:

  * :func:`verify_anamnesis_pin` — the startup check that the imported
    anamnesis IS the pinned commit; refuses on mismatch unless overridden.
  * :func:`resolve_preset` — preset lookup with pleroma's registry file on
    ``ANAMNESIS_MODELS`` (``pleroma.config.anamnesis_registry.REGISTRY_FILE``).
  * the loaded-model seam — ``read_lane_calibration``, ``load_lane_model``,
    ``prepare_fast_lane`` -> ``PreparedLane``, ``harvest_loaded`` ->
    ``HarvestResult``: upstream's own (LuxiaSL/anamnesis#16), re-exported
    lazily. The fork series is ``harvest_loaded``'s logit series, read from the
    lane's OWN forward.
  * :func:`replay_manifest_to_dir` — ``run_gpu_replay``'s per-span loop and
    on-disk layout, plus the fork series.
  * :class:`GpuHarvestLane` — the long-lived object a harvest worker holds.

This module patches nothing: it never assigns to an anamnesis module
attribute, never swaps ``sys.argv`` to drive an upstream ``main()``, and never
pins an upstream file's sha. The public API at the pinned commit covers every
call the lane makes, including the loaded-model composition
(https://github.com/LuxiaSL/anamnesis/pull/16).

Torch and anamnesis are imported inside functions, so this module imports on
a laptop with neither.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import numpy as np

from pleroma.config.anamnesis_registry import resolve_preset as _registry_resolve

if TYPE_CHECKING:  # pragma: no cover
    from anamnesis.config import ModelPreset
    from anamnesis.extraction.fast.runtime import PreparedLane

logger = logging.getLogger("pleroma.harvest.anamnesis_seam")

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_DIR = REPO_ROOT / "vendor" / "anamnesis"

#: The upstream commit the ``vendor/anamnesis`` gitlink pins. A bump changes
#: this constant AND the gitlink together; a test asserts they agree.
PINNED_COMMIT = "bc79f10c307e9fdab1113db0f09d843340168c05"

#: Setting this to 1 downgrades a pin mismatch from a refusal to a warning.
ALLOW_MISMATCH_ENV = "PLEROMA_ALLOW_ANAMNESIS_MISMATCH"

#: Top-k kept per fork-series step (the replay lane's ``--logits-top-k``
#: default, and the upstream lane config's ``raw_logits_top_k``).
DEFAULT_FORK_TOP_K = 50

#: Provenance label for rows harvested through this seam (``<out>/cells.json``,
#: ``<out>/run_meta.json``). Distinct from the replay lane's label on purpose.
LANE_CONFIG_SOURCE = (
    "anamnesis fast lane: extraction.replay_config.native_replay_configs via "
    "pleroma.harvest.anamnesis_seam (submodule pin " + PINNED_COMMIT[:12] + ")"
)


# ── the pin ─────────────────────────────────────────────────────────────────


class AnamnesisPinError(RuntimeError):
    """The imported anamnesis is not the pinned submodule commit."""


@dataclass(frozen=True)
class PinProbe:
    """What anamnesis this process imported, and how that was determined."""

    root: Path
    commit: str | None
    source: str
    dirty: bool = False

    @property
    def matches(self) -> bool:
        return self.commit == PINNED_COMMIT and not self.dirty


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _probe_git(root: Path, source: str) -> PinProbe | None:
    if not (root / ".git").exists():
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return PinProbe(root=root, commit=commit, source=source, dirty=bool(status))


def imported_anamnesis_commit() -> PinProbe:
    """Identify the anamnesis this process imported.

    1. The package's parent directory is a git checkout (the submodule, or an
       editable install of it): ``git rev-parse HEAD`` there, plus a dirty check
       on tracked files.
    2. Otherwise the installed distribution's ``<dist-info>/direct_url.json``: a VCS
       install's ``commit_id``, or an editable install's source directory
       (then as in 1).
    3. Otherwise the commit is unknown (``commit=None``), which never matches.
    """
    import anamnesis

    root = Path(anamnesis.__file__).resolve().parent.parent
    probe = _probe_git(root, f"git rev-parse in {root}")
    if probe is not None:
        return probe
    try:
        dist = importlib.metadata.distribution("anamnesis")
        raw = dist.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        raw = None
    if raw:
        try:
            info = json.loads(raw)
        except ValueError:
            info = {}
        vcs = info.get("vcs_info") or {}
        if vcs.get("commit_id"):
            return PinProbe(root=root, commit=str(vcs["commit_id"]),
                            source="direct_url.json vcs_info.commit_id")
        url = str(info.get("url", ""))
        if url.startswith("file://"):
            src = Path(unquote(urlparse(url).path))
            probe = _probe_git(src, f"direct_url.json -> git rev-parse in {src}")
            if probe is not None:
                return probe
    return PinProbe(root=root, commit=None, source="unknown (no git checkout, no direct_url.json)")


def verify_anamnesis_pin(*, allow_mismatch: bool | None = None) -> PinProbe:
    """Refuse unless the imported anamnesis is exactly :data:`PINNED_COMMIT`.

    A modified (dirty) checkout of the right commit is a mismatch too: the
    instrument is its code, not its commit label. ``allow_mismatch`` (or
    ``PLEROMA_ALLOW_ANAMNESIS_MISMATCH=1`` when it is None) turns the refusal
    into a loud warning, for deliberate experiments with another anamnesis.
    """
    if allow_mismatch is None:
        allow_mismatch = os.environ.get(ALLOW_MISMATCH_ENV, "") == "1"
    probe = imported_anamnesis_commit()
    if probe.matches:
        logger.info("anamnesis pin verified: %s (%s)", probe.commit, probe.source)
        return probe
    what = (f"commit {probe.commit}" if probe.commit else "an UNIDENTIFIABLE commit")
    msg = (
        f"anamnesis imported from {probe.root} is {what}"
        f"{' with local modifications' if probe.dirty else ''} ({probe.source}); "
        f"pleroma pins {PINNED_COMMIT}. Every shipped map/atlas/lever was fit in "
        "the pinned instrument's space. Run `git submodule update --init "
        "vendor/anamnesis` (and install it editable into the venv), or pass the "
        f"explicit override ({ALLOW_MISMATCH_ENV}=1 / --allow-anamnesis-mismatch)."
    )
    if not allow_mismatch:
        raise AnamnesisPinError(msg)
    logger.warning("ANAMNESIS PIN OVERRIDDEN — %s", msg)
    return probe


# ── presets ──────────────────────────────────────────────────────────────────


def resolve_preset(name: str | ModelPreset) -> ModelPreset:
    """``anamnesis.config.resolve_preset`` with pleroma's registry on the path."""
    return _registry_resolve(name)


# ── the loaded-model seam (https://github.com/LuxiaSL/anamnesis/pull/16) ─────
#
# Upstream's in-process API, re-exported so pleroma imports it from one place.
# The names and shapes are upstream's own code at the pinned commit.


def __getattr__(name: str) -> Any:
    """Lazy re-exports of upstream's loaded-model seam (torch stays unimported)."""
    if name in _RUNTIME_NAMES:
        from anamnesis.extraction.fast import runtime

        return getattr(runtime, name)
    if name in _HARVEST_NAMES:
        from anamnesis.extraction.fast import harvest

        return getattr(harvest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_RUNTIME_NAMES = frozenset({
    "LaneCalibration", "read_lane_calibration", "load_lane_model", "check_loaded_model",
    "PreparedLane", "prepare_fast_lane",
})
_HARVEST_NAMES = frozenset({"HarvestResult", "LogitSeries", "harvest_loaded"})


class ForkSeriesError(RuntimeError):
    """The fork-series capture and the span being replayed got out of step."""


def lane_weight_digests(model_path: str | Path) -> dict[str, str]:
    """sha256 of config.json + every safetensors shard (upstream's receipt input).

    At 70B this is ~141 GB of reading; :class:`GpuHarvestLane` pays it once.
    """
    from anamnesis.extraction.fast.runtime import weight_file_digests

    return weight_file_digests(model_path)


# ── the span loop ────────────────────────────────────────────────────────────


def _anamnesis_file_sha(relative: str) -> str:
    import anamnesis
    from anamnesis.provenance import file_sha

    return file_sha(Path(anamnesis.__file__).resolve().parent / relative)


def read_generation_metadata(path: Path | None) -> dict[int, dict[str, Any]]:
    """``metadata.json`` beside a lane manifest -> {generation_id: record}
    (``run_gpu_replay.read_generation_metadata``)."""
    from anamnesis.scripts.run_gpu_replay import read_generation_metadata as _read

    return _read(path)


def replay_manifest_to_dir(
    prepared: PreparedLane,
    manifest_path: Path,
    out_dir: Path,
    *,
    model_path: str,
    gen_ids: Iterable[int] | None = None,
    fork_top_k: int = DEFAULT_FORK_TOP_K,
) -> list[int]:
    """Replay a lane manifest (``pleroma.harvest.gpu_manifests`` shape) into ``out_dir``.

    Writes what ``anamnesis.scripts.run_gpu_replay.main`` writes at the pin —
    ``<out_dir>/deployment.json``, and per gen ``gen_NNN.npz`` + ``gen_NNN.json`` through
    ``feature_pipeline.save_features`` with the same metadata keys — plus
    ``fork_series/gen_NNN.npz``. ``out_dir`` must not exist. ``gen_ids=None``
    replays the whole manifest (sorted); otherwise exactly the ids given.
    Every selected span is validated against the calibration BEFORE any
    replay, as upstream's runner does. Returns the generation ids written.
    """
    import torch
    from anamnesis.extraction.fast.harvest import harvest_loaded
    from anamnesis.extraction.fast.runtime import resolve_lane_spans
    from anamnesis.extraction.feature_pipeline import save_features
    from anamnesis.provenance import file_sha
    from anamnesis.scripts.run_gpu_replay import select_ids

    manifest_path, out_dir = Path(manifest_path), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    entries = json.loads(manifest_path.read_text())["entries"]
    ids = select_ids(entries, None if gen_ids is None else [int(g) for g in gen_ids])
    meta_path = manifest_path.parent / "metadata.json"
    source_path = meta_path if meta_path.exists() else None
    source_metadata = read_generation_metadata(source_path)
    if source_path is not None and any(i not in source_metadata for i in ids):
        raise ValueError("source metadata is missing selected generation IDs")
    spans = resolve_lane_spans(
        entries, ids, positions_calibrated=prepared.calibration.positions_calibrated)

    out_dir.mkdir(parents=True, exist_ok=False)
    fork_dir = out_dir / "fork_series"
    fork_dir.mkdir()
    cuda = str(prepared.device).startswith("cuda")
    written: list[int] = []
    lane_ids: set[str] = set()
    for span in spans:
        i = span.gen_id
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = harvest_loaded(prepared, span.input_ids, prompt_len=span.prompt_length,
                                logit_series_top_k=int(fork_top_k))
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        series = result.series_arrays()
        if int(series["logits_entropy"].shape[0]) != span.n_steps:
            raise ForkSeriesError(
                f"gen {i}: fork series has {series['logits_entropy'].shape[0]} steps, "
                f"the lane replayed {span.n_steps}")
        metadata = dict(source_metadata.get(i, {}))
        metadata.update(
            generation_id=i, lane_id=result.lane_id, extraction_lane=result.receipt,
            replay_seconds=elapsed, mean_logprob=result.mean_logprob,
            raw_tensors_saved=False, certified=False,
        )
        save_features(i, result.extraction_result(), metadata, out_dir)
        np.savez_compressed(fork_dir / f"gen_{i:03d}.npz", **series)
        written.append(i)
        lane_ids.add(result.lane_id)
        logger.info("lane gen %d: %.2fs lane_id=%s", i, elapsed, result.lane_id)
    # The lane leaves its capture hooks enabled after a span; this model is the
    # lane's alone, but leave it quiet so nothing else accumulates captures.
    prepared.loaded.disable_hooks()
    prepared.loaded.clear_hook_state()

    # The lane identity (sources, configs, and the software stack: torch,
    # transformers, CUDA, determinism flags) is what the banked corpus records
    # under "lane"; lane_parity compares stacks through it.
    identities = {json.dumps(getattr(ln, "identity", {}), sort_keys=True, default=str)
                  for ln in getattr(prepared, "_lanes", {}).values()}
    provenance: dict[str, Any] = dict(
        lane_ids=sorted(lane_ids),
        lane_id=(next(iter(lane_ids)) if len(lane_ids) == 1 else None),
        lane=(json.loads(next(iter(identities))) if len(identities) == 1 else None),
        manifest_sha256=file_sha(manifest_path),
        **prepared.provenance(),
        model_path=model_path,
        runner="pleroma/harvest/anamnesis_seam.py",
        runner_sha256=file_sha(Path(__file__)),
        anamnesis_commit=PINNED_COMMIT,
        configuration_source_sha256=_anamnesis_file_sha("extraction/replay_config.py"),
        schema_source_sha256=_anamnesis_file_sha("extraction/fast/schema.py"),
        selected_ids=ids,
        source_metadata_sha256=file_sha(source_path) if source_path is not None else None,
        raw_tensors_saved=False,
        fork_series=f"logits top-{int(fork_top_k)} + entropy, read from the lane's own "
                    "forward by a per-span model forward hook (harvest_loaded)",
        certified=False,
    )
    (out_dir / "deployment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return written


# ── the long-lived lane ──────────────────────────────────────────────────────


class GpuHarvestLane:
    """What a harvest worker holds for its lifetime: a :class:`PreparedLane`
    (one lane model + one calibration, each paid once), the checkpoint
    digests (paid once, lazily or via :meth:`prewarm`), and :meth:`run` per
    harvest over a lane manifest.
    """

    def __init__(self, prepared: PreparedLane, model_path: str) -> None:
        self.prepared = prepared
        self.model_path = str(model_path)

    @classmethod
    def open(cls, *, preset: str | ModelPreset, model_path: str, calib_dir: Path,
             device: str) -> GpuHarvestLane:
        """Pin the arithmetic, read the calibration, load the lane model —
        cheap refusals first (arithmetic, preset, calibration), then the load
        (upstream ``prepare_fast_lane``'s order)."""
        from anamnesis.extraction.fast.runtime import prepare_fast_lane

        row = resolve_preset(preset)
        t0 = time.time()
        prepared = prepare_fast_lane(row, Path(calib_dir), device=str(device),
                                     model_path=str(model_path))
        logger.warning("lane prepared on %s in %.1f s (preset %r, calibration sha %s, "
                       "%d positions calibrated)", device, time.time() - t0, row.name,
                       prepared.calibration.calibration_sha256[:16],
                       prepared.calibration.positions_calibrated)
        return cls(prepared, str(model_path))

    def prewarm(self) -> None:
        """Hash the checkpoint now (ideally while the page cache is warm)."""
        if self.prepared.model_files is None:
            t0 = time.time()
            self.prepared.model_files = lane_weight_digests(self.model_path)
            logger.warning("lane provenance: sha256 over %d checkpoint files in %.1f s — "
                           "held for this process", len(self.prepared.model_files),
                           time.time() - t0)

    def run(self, manifest_path: Path, out_dir: Path, *,
            gen_ids: Iterable[int] | None = None,
            fork_top_k: int = DEFAULT_FORK_TOP_K) -> list[int]:
        self.prewarm()
        return replay_manifest_to_dir(self.prepared, manifest_path, out_dir,
                                      model_path=self.model_path, gen_ids=gen_ids,
                                      fork_top_k=fork_top_k)
