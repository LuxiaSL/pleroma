"""Replay stage: teacher-forced eager replay of the forked gens -> v3 signatures.

The generation stage banks the realized tokens; every signature a map is fit
on comes from here. The
model is loaded ONCE with the full v3 capture surface (all-layer k_proj/v_proj/
q_proj/o_proj hooks, gate hooks on sampled layers) and each gen's banked
``input_ids`` is replayed teacher-forced — the validated gateway, not a re-draw.

Three things this stage refuses to do, all of them on purpose:

REFUSES to run without calibration. ``--calib-dir`` must hold ``positional_means.npz``
    AND a PCA model. anamnesis' own loader *warns* and continues when positional
    means are missing, which produces a plausible-looking vector in a different
    space; the replay's whole premise is that our signatures live in the space the 3B
    discriminants were fit in, so a missing file is an abort, never a warning.

REFUSES to proceed past a feature-name mismatch (GATE 0). After the FIRST gen's
    extraction, ``feature_names`` must equal ``FULL_names`` from the discriminants
    npz exactly, order included. That single assert catches every calibration or
    family-config drift there is, including a PCA fit on pooled rather than
    position-corrected states. It fires on gen one so a 128-gen GPU run dies in seconds
    rather than producing a full directory of un-projectable vectors.

REFUSES to hard-code a dimension. The live suite's dimensionality drifts per
    model and as families land, so every shape here derives at runtime from
    ``feature_names`` / array shapes.

Raw banking is knob-controlled (``--save-raw``) because all-layer raw_tensors_v3 is
hundreds of MB per gen; the *fork series* (entropy, top-k logits, chosen ids,
input_ids — kilobytes) is banked for EVERY gen regardless, because that is what
the fork analysis consumes. The generation stage's metadata is copied into
``<out-dir>/cells.json`` so the map build (``pleroma.map.build.pairs``,
``pleroma.map.build.basin``) never needs ``--gen-dir``.

Adapted from anamnesis' replay-extraction script; the feature family flags
replicate kv-rotation's v3 family config (see ``build_configs``).

── --shard i/n and --resume ───────────────────────────────────────────────────

One gen is ~3-5 s on one GPU, and the cost is CPU feature math rather than the
forward, so the way to go faster is more processes, not a bigger batch. ``--shard
i/n`` takes every n-th record of the work list; a launcher runs
``len(gpus) x workers_per_gpu`` of them with one ``CUDA_VISIBLE_DEVICES`` each.

The filters compose IN THIS ORDER in every worker — ``--limit``, then the modulo,
then ``--resume`` (drop gens whose ``signatures/gen_NNN.npz`` exists). **The
modulo comes before the resume filter and must**: the record list is static, so
every worker partitions the same list whenever it happens to start, whereas the
set of already-extracted gens GROWS WHILE THE WORKERS RUN. Sharding a
resume-filtered list would give a worker that started two seconds later a shorter
list, a different partition, and a handful of gens that no worker owns (4
workers over 11 gens, sharded that way, extract 7 of them). The price is
that a RESUMED run is unbalanced — a worker whose shard is already done exits
immediately — which costs a little wall clock and nothing else.

Sharded workers share ONE ``--out-dir``: every per-gen artifact is named by
generation_id, so their writes are disjoint. The exceptions are
``<out-dir>/cells.json`` and ``<out-dir>/run_meta.json``, which describe the
WHOLE corpus and are therefore written per shard under ``shard_meta/`` for the
launcher to merge.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.harvest.replay \\
        --gen-dir outputs/3b/gen --out-dir outputs/3b/replay \\
        --preset 3b --model-path <hf-id-or-local-dir> \\
        --calib-dir data/calibration/3b \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --save-raw subset
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import: sharded workers each get one
# thread per pool, so N workers on one box do not oversubscribe its cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB0.replay")

F32 = NDArray[np.float32]

class Gate0Failure(RuntimeError):
    """Feature names diverged from the discriminants' FULL_names. Aborts the run."""


# ── Calibration: present or we do not run ──────────────────────────────────────


def find_pca_model(calib_dir: Path) -> Path:
    """Locate the tier-3 PCA pickle in a calibration dir, or raise.

    ``pca_model.pkl`` is the canonical name (kv-rotation's 3B calibration loader); a
    differently-named single ``*pca*.pkl`` is accepted with a log line so a staged
    calibration dir does not silently degrade to "tier 3 off".
    """
    canonical = calib_dir / "pca_model.pkl"
    if canonical.exists():
        return canonical
    candidates = sorted(p for p in calib_dir.glob("*pca*.pkl") if p.is_file())
    if len(candidates) == 1:
        logger.warning(
            "using non-canonical PCA model %s (expected pca_model.pkl)", candidates[0].name
        )
        return candidates[0]
    raise FileNotFoundError(
        f"no PCA model in {calib_dir} "
        f"(looked for pca_model.pkl, then *pca*.pkl; found {[p.name for p in candidates]})"
    )


def load_calibration_strict(calib_dir: Path) -> tuple[F32, Any, Any, dict[str, str]]:
    """Load positional_means + PCA, failing loudly on anything missing.

    Mirrors kv-rotation's 3B calibration loader: accepts the legacy pooled dict
    pickle ({"components", "mean"}), the C5 per-layer dict, and a fitted sklearn
    PCA object. Unlike anamnesis' ``_load_calibration`` it never warns-and-continues.
    """
    calib_dir = calib_dir.expanduser()
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"--calib-dir {calib_dir} is not a directory")

    pm_path = calib_dir / "positional_means.npz"
    if not pm_path.exists():
        raise FileNotFoundError(f"positional_means.npz not found in {calib_dir}")
    with np.load(pm_path) as npz:
        if "positional_means" not in npz:
            raise KeyError(f"{pm_path} has no 'positional_means' array (has {npz.files})")
        positional_means = npz["positional_means"].astype(np.float32)
    logger.info("positional_means %s from %s", positional_means.shape, pm_path)

    pca_path = find_pca_model(calib_dir)
    with open(pca_path, "rb") as fh:
        pca_bytes = fh.read()
    pca = pickle.loads(pca_bytes)
    # Provenance: the calibration choice is invisible to Gate 0 (names match under
    # any PCA; only tier-3 VALUES differ, e.g. a PCA fit on pooled rather than
    # position-corrected states), so the run must record exactly which bytes it used.
    provenance = {
        "positional_means_sha256": hashlib.sha256(pm_path.read_bytes()).hexdigest(),
        "pca_path": str(pca_path),
        "pca_sha256": hashlib.sha256(pca_bytes).hexdigest(),
    }
    if isinstance(pca, dict):
        values = list(pca.values())
        if values and isinstance(values[0], dict) and "components" in values[0]:
            pca_components: Any = {
                int(k): np.asarray(v["components"], dtype=np.float32) for k, v in pca.items()
            }
            pca_mean: Any = {
                int(k): np.asarray(v["mean"], dtype=np.float32) for k, v in pca.items()
            }
            provenance["pca_format"] = "c5_per_layer"
            logger.info("pca: C5 per-layer format, %d layers", len(pca_components))
        else:
            pca_components = np.asarray(pca["components"], dtype=np.float32)
            pca_mean = np.asarray(pca["mean"], dtype=np.float32)
            provenance["pca_format"] = "legacy_pooled"
            logger.info("pca: pooled components %s", pca_components.shape)
    else:
        pca_components = np.asarray(pca.components_, dtype=np.float32)
        pca_mean = np.asarray(pca.mean_, dtype=np.float32)
        provenance["pca_format"] = "sklearn_object"
        logger.info("pca: sklearn object, components %s", pca_components.shape)
    logger.info("calibration provenance: %s", provenance)
    return positional_means, pca_components, pca_mean, provenance


# ── The frozen v3 family/extraction config ─────────────────────────────────────


def build_configs(preset_name: str) -> tuple[Any, Any, str]:
    """(ExtractionConfig, FeaturePipelineConfig, provenance) for the frozen v3 space.

    The FLAGS are kv-rotation's ``v3_family_config``, replicated
    field-for-field; that function defined the space, and this package does not
    import kv-rotation, so the replication below is the definition in force.
    Only flags are replicated — every layer list still comes
    from the preset, and no dimension count appears here. The provenance label
    is unchanged so banked receipts stay comparable.
    """
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, resolve_preset

    preset = resolve_preset(preset_name)
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_residual_pca=True,  # was enable_tier3 (upstream rename, same flag)
    )
    # Replicated field-for-field from kv-rotation's v3_family_config().
    # NOTE the deliberate omissions: value_geometry / qk_geometry / kv_cka /
    # expert_routing are NOT set here even though anamnesis' replay-extraction
    # script enables them — those families are absent from the discriminant fit, and
    # enabling any of them changes feature_names, which
    # Gate 0 would (correctly) reject.
    family_config = FeaturePipelineConfig(
        include_core_blocks=True,  # was include_baseline_tiers (upstream rename)
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=True,
        enable_stft=True,
        enable_contrastive_projection=False,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    return extraction_config, family_config, "replicated from sigbridge.py:113"


# ── Generation-stage records ───────────────────────────────────────────────────


def resolve_gen_records_dir(gen_dir: Path) -> Path:
    """Accept either the stage-1 out-dir or its gen_records/ subdir."""
    nested = gen_dir / "gen_records"
    if nested.is_dir():
        return nested
    if gen_dir.is_dir() and any(gen_dir.glob("gen_*.json")):
        return gen_dir
    raise FileNotFoundError(
        f"no gen records under {gen_dir} (expected {nested} or gen_*.json directly)"
    )


def load_gen_records(records_dir: Path) -> list[dict[str, Any]]:
    """Load every gen_NNN.json, sorted by generation_id."""
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("gen_*.json")):
        rec = json.loads(path.read_text())
        for key in (
            "generation_id",
            "input_ids",
            "prompt_length",
            "prompt_id",
            "prompt_class",
            "seed_idx",
        ):
            if key not in rec:
                raise KeyError(f"{path}: stage-1 record is missing {key!r}")
        records.append(rec)
    if not records:
        raise FileNotFoundError(f"{records_dir} holds no gen_*.json records")
    records.sort(key=lambda r: int(r["generation_id"]))
    return records


def select_raw_prompts(records: list[dict[str, Any]], save_raw: str) -> set[str]:
    """Which prompt_ids get all-layer raw banking.

    ``subset`` (the default) = the first two distinct prompt_ids encountered, which
    keeps a feature-iteration sandbox (e.g. the bins-B extraction experiments)
    without tens of GB. "Encountered" is in generation_id order, so the subset is
    reproducible rather than set-ordering-dependent.
    """
    if save_raw == "none":
        return set()
    ordered: list[str] = []
    for rec in records:
        pid = str(rec["prompt_id"])
        if pid not in ordered:
            ordered.append(pid)
    if save_raw == "all":
        return set(ordered)
    return set(ordered[:2])


# ── Fork series: the small arrays the fork analysis lives on ───────────────────


def fork_series_arrays(
    raw_data: Any, input_ids: list[int], prompt_length: int, top_k: int
) -> dict[str, NDArray]:
    """Extract the fork analysis's per-position surface from an in-memory replay pass.

    Same construction as ``raw_saver.save_raw_tensors_all_layer``'s logits block
    (top-k by value, exact full-vocab entropy computed before the truncation),
    pulled out so it is banked even when raw banking is off.

    Alignment (as ``anamnesis.extraction.replay.extract`` banks it): entry ``t``
    is the state at continuation index ``t`` and its logits predict continuation
    index ``t+1``; ``chosen_ids[t]`` is therefore the realized token at
    continuation index ``t+1``, absolute ``prompt_length + t + 1``. T = N - 1,
    so continuation index 0 has no banked distribution — the fork analysis
    records that as null rather than inventing one.
    """
    logits = list(raw_data.logits or [])
    n_steps = len(logits)
    if n_steps == 0:
        raise ValueError("replay produced no per-step logits")
    k = int(min(top_k, logits[0].shape[0]))
    if k <= 0:
        raise ValueError(f"top_k resolved to {k}")

    values = np.zeros((n_steps, k), dtype=np.float32)
    indices = np.zeros((n_steps, k), dtype=np.int32)
    entropy = np.zeros(n_steps, dtype=np.float32)
    for t, vec in enumerate(logits):
        top = np.argpartition(vec, -k)[-k:]
        top = top[np.argsort(vec[top])[::-1]]
        values[t] = vec[top]
        indices[t] = top
        probs = np.exp(vec - np.max(vec)).astype(np.float64)
        probs /= probs.sum()
        probs = probs[probs > 0]
        entropy[t] = float(-np.sum(probs * np.log(probs)))

    return {
        "logits_values": values,
        "logits_indices": indices,
        "logits_entropy": entropy,
        "chosen_ids": np.asarray(raw_data.chosen_token_ids, dtype=np.int32),
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "prompt_length": np.array(prompt_length, dtype=np.int32),
    }


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``--shard i/n`` into ``(index, count)``, 0-based index.

    ``i/n`` rather than an explicit id list (which is what anamnesis'
    parallel replay passes its workers) because this stage's work list is
    already a deterministic sorted sequence: every worker can compute the same
    partition from two integers, and the launcher's command lines stay short
    enough to read in a log. That equivalence holds ONLY because the list being
    partitioned is static — see ``main()``'s shard-before-resume ordering.
    """
    text = str(spec).strip()
    if "/" not in text:
        raise ValueError(f"--shard must be i/n, got {spec!r}")
    left, _, right = text.partition("/")
    try:
        index, count = int(left), int(right)
    except ValueError as exc:
        raise ValueError(f"--shard must be two integers i/n, got {spec!r}") from exc
    if count <= 0:
        raise ValueError(f"--shard count must be > 0, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"--shard index {index} outside [0, {count})")
    return index, count


def signature_path(out_dir: Path, generation_id: int) -> Path:
    """Where ``anamnesis.extraction.feature_pipeline.save_features`` puts this gen's vector.

    The resume key, and the reason sharded workers can share ONE --out-dir:
    every per-gen artifact this stage writes is named by generation_id, so two
    shards can never target the same path. The whole-corpus indices
    (``<out-dir>/cells.json``, ``<out-dir>/run_meta.json``) are the exception
    and are written per shard, for the launcher to merge.
    """
    return Path(out_dir) / "signatures" / f"gen_{int(generation_id):03d}.npz"


def shard_records(
    records: list[dict[str, Any]], index: int, count: int
) -> list[dict[str, Any]]:
    """Modulo partition over the records IN THE ORDER GIVEN (sorted by gen id).

    Modulo rather than contiguous blocks: gens near each other in id share a prompt
    and therefore a length and a cost, so contiguous blocks would hand one worker
    all of an expensive prompt. The list handed in must be the STATIC one (after
    ``--limit``, before the resume filter) or the partition stops being a partition
    — see the module docstring.
    """
    if count <= 1:
        return list(records)
    return [rec for k, rec in enumerate(records) if k % int(count) == int(index)]


def check_gate0(feature_names: list[str], full_names: list[str]) -> None:
    """GATE 0: the extracted space must be the discriminants' space, exactly."""
    if list(feature_names) == list(full_names):
        return
    first_bad = next(
        (i for i, (a, b) in enumerate(zip(feature_names, full_names)) if a != b), None
    )
    detail = (
        f" first mismatch at index {first_bad}: "
        f"{feature_names[first_bad]!r} vs {full_names[first_bad]!r}"
        if first_bad is not None
        else " (one list is a prefix of the other)"
    )
    raise Gate0Failure(
        "GATE 0 FAILED — extracted feature_names != discriminants FULL_names: "
        f"lengths {len(feature_names)} vs {len(full_names)}.{detail}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-dir", type=Path, required=True, help="stage-1 out-dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", default="3b", help="anamnesis registry preset (anamnesis.config.preset_names())")
    parser.add_argument(
        "--model-path", default=None, help="local model dir; default = the preset's model_id"
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        required=True,
        help=(
            "calibration dir; positional_means.npz AND a PCA model must exist or the "
            "run aborts (anamnesis only warns — this replay refuses)"
        ),
    )
    parser.add_argument(
        "--discriminants",
        type=Path,
        default=None,
        help=(
            "the frozen-space manifest (e.g. factor_directions_3b.npz) — supplies "
            "FULL_names for GATE 0. Required unless bootstrapping a NEW model's "
            "space with --freeze-names-to."
        ),
    )
    parser.add_argument(
        "--freeze-names-to",
        type=Path,
        default=None,
        help=(
            "NEW-MODEL BOOTSTRAP: no external manifest exists yet, so freeze the "
            "space from THIS run — every gen is still gated against the first "
            "gen's feature names, and at the end a manifest npz (FULL_names + "
            "FULL_mean/FULL_scale over this run's signatures) is written here. "
            "Use it as --discriminants for every subsequent run in this space. "
            "Single-process only (mutually exclusive with --shard)."
        ),
    )
    parser.add_argument(
        "--save-raw",
        default="subset",
        choices=["none", "subset", "all"],
        help="all-layer raw_tensors_v3 banking; subset = the first 2 distinct prompt_ids",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap gens (0 = all)")
    parser.add_argument(
        "--logits-top-k",
        type=int,
        default=50,
        help="per-position logits retained in the fork series (and in raw)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help=(
            "THROUGHPUT: 'i/n' — take every n-th gen record starting at i, so n "
            "workers can share one --out-dir. Applied AFTER --limit and BEFORE the "
            "resume filter, in that order, by every worker alike. Gate 0 still runs "
            "in each worker on its own first gen."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "skip gens whose signatures/gen_NNN.npz already exists in --out-dir "
            "(default off, so a bare invocation still means 'replay everything')"
        ),
    )
    args = parser.parse_args()

    if args.logits_top_k <= 0:
        logger.error("--logits-top-k must be > 0, got %s", args.logits_top_k)
        return 2

    shard_index, shard_count = 0, 1
    if args.shard is not None:
        try:
            shard_index, shard_count = parse_shard(str(args.shard))
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
    shard_label = f"{shard_index}of{shard_count}" if shard_count > 1 else ""

    if (args.discriminants is None) == (args.freeze_names_to is None):
        logger.error("give exactly ONE of --discriminants / --freeze-names-to")
        return 2
    if args.freeze_names_to is not None and (args.shard is not None or args.resume):
        logger.error("--freeze-names-to needs one full pass (no --shard, no "
                     "--resume): the manifest's FULL_mean/FULL_scale must see "
                     "every gen of the run")
        return 2

    # Discriminants first: the cheapest possible failure for the commonest mistake.
    full_names: list[str] | None = None
    if args.discriminants is not None:
        try:
            with np.load(args.discriminants.expanduser(), allow_pickle=True) as fd:
                if "FULL_names" not in fd:
                    logger.error(
                        "%s has no FULL_names (has %s)", args.discriminants, list(fd.files)
                    )
                    return 2
                full_names = [str(x) for x in fd["FULL_names"]]
        except (OSError, ValueError) as exc:
            logger.error("could not read --discriminants %s: %s", args.discriminants, exc)
            return 2
        logger.info("discriminants: %d FULL_names from %s",
                    len(full_names), args.discriminants)
    else:
        logger.info("BOOTSTRAP mode: names freeze on the first gen; manifest -> %s",
                    args.freeze_names_to)

    try:
        records = load_gen_records(resolve_gen_records_dir(args.gen_dir))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load stage-1 records: %s", exc)
        return 2
    if args.limit:
        records = records[: args.limit]
    # Chosen from the WHOLE (post-limit) corpus, before the resume and shard
    # filters: "the first two distinct prompt_ids" has to mean the same two in
    # every worker and on every re-run, or the raw sandbox would depend on which
    # gens happened to be left.
    raw_prompts = select_raw_prompts(records, args.save_raw)
    n_corpus = len(records)

    # SHARD FIRST, RESUME SECOND. The record list is static; the set of extracted
    # signatures is not, and it grows while the other workers run. See the module
    # docstring — reversing these two lines silently loses gens.
    records = shard_records(records, shard_index, shard_count)
    n_shard = len(records)
    n_resumed = 0
    if args.resume:
        kept = [
            rec for rec in records
            if not signature_path(args.out_dir, int(rec["generation_id"])).exists()
        ]
        n_resumed = len(records) - len(kept)
        records = kept

    logger.info(
        "%d gens in the corpus%s%s -> %d for this worker over %d prompts / %d classes; "
        "raw banking: %s (prompt_id %s)",
        n_corpus,
        f", shard {shard_index}/{shard_count} takes {n_shard}" if shard_count > 1 else "",
        f", {n_resumed} of those already extracted (resume)" if args.resume else "",
        len(records),
        len({str(r["prompt_id"]) for r in records}),
        len({str(r["prompt_class"]) for r in records}),
        args.save_raw,
        sorted(raw_prompts) or "none",
    )
    if not records:
        # Nothing to do is not a failure when resuming or when a shard came up
        # empty (n workers > n remaining gens); it IS a failure otherwise, and
        # load_gen_records already refused an empty records dir upstream.
        logger.info(
            "no gens left for this worker (corpus %d, shard %d/%d took %d, resumed "
            "%d) — writing no output and exiting 0",
            n_corpus, shard_index, shard_count, n_shard, n_resumed,
        )
        return 0

    # Deferred so --help works on any machine, with or without the anamnesis env.
    from anamnesis.config import ModelConfig, UnknownPresetError, resolve_preset
    from anamnesis.extraction.feature_pipeline import (
        compute_features_with_families_from_data,
        save_features,
    )
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.raw_saver import save_raw_tensors_all_layer
    from anamnesis.extraction.replay.extract import replay_extract

    try:
        preset = resolve_preset(args.preset)
    except UnknownPresetError as exc:
        logger.error("%s", exc)
        return 2

    try:
        positional_means, pca_components, pca_mean, calib_provenance = load_calibration_strict(
            args.calib_dir
        )
    except (OSError, KeyError, ValueError, pickle.UnpicklingError) as exc:
        logger.error("calibration unusable: %s", exc)
        logger.error("refusing to run — a silently-uncorrected signature is not in the v3 space")
        return 2

    extraction_config, family_config, config_source = build_configs(args.preset)
    logger.info("v3 config source: %s", config_source)

    sig_dir = args.out_dir / "signatures"
    fork_dir = args.out_dir / "fork_series"
    raw_dir = args.out_dir / "raw_tensors_v3"
    sig_dir.mkdir(parents=True, exist_ok=True)
    fork_dir.mkdir(parents=True, exist_ok=True)
    if raw_prompts:
        raw_dir.mkdir(parents=True, exist_ok=True)

    model_path: str = args.model_path or preset.model_id
    all_layers = list(range(preset.num_layers))
    model_config = ModelConfig.from_preset(preset, model_id=model_path)
    logger.info("loading %s with the full v3 capture surface (eager)", model_path)
    loaded = load_model(
        model_config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=all_layers,
        value_layers=all_layers,
        query_layers=all_layers,
        attn_output_layers=all_layers,
    )
    logger.info("model loaded; sampled_layers=%s", preset.sampled_layers)

    cells: list[dict[str, Any]] = []
    failures: list[str] = []
    feature_names: list[str] | None = None
    boot_features: list[np.ndarray] = []  # freeze-names mode only
    n_raw_saved = 0
    started = time.time()

    for i, rec in enumerate(records):
        gid = int(rec["generation_id"])
        try:
            input_ids = [int(x) for x in rec["input_ids"]]
            plen = int(rec["prompt_length"])
            raw_data = replay_extract(
                loaded, input_ids, plen, positional_means=positional_means
            )

            save_raw = str(rec["prompt_id"]) in raw_prompts
            if save_raw:
                save_raw_tensors_all_layer(
                    raw_data,
                    gid,
                    raw_dir,
                    prompt_length=plen,
                    input_ids=input_ids,
                    top_k_logits=args.logits_top_k,
                )
                n_raw_saved += 1

            result = compute_features_with_families_from_data(
                raw_data, extraction_config, family_config, pca_components, pca_mean
            )

            if feature_names is None:
                feature_names = [str(n) for n in result.feature_names]
                logger.info("extracted %d features on gen_%03d", len(feature_names), gid)
                if full_names is not None:
                    check_gate0(feature_names, full_names)  # raises Gate0Failure -> abort
                    logger.info("GATE 0 PASSED — feature_names == FULL_names (%d dims)",
                                len(feature_names))
                else:
                    logger.info("BOOTSTRAP: %d feature names FROZEN from gen_%03d — "
                                "all later gens gate against these", len(feature_names), gid)
            elif [str(n) for n in result.feature_names] != feature_names:
                raise AssertionError(
                    f"gen_{gid:03d}: feature names diverged from the first gen — "
                    "signatures across gens would not be comparable"
                )
            if args.freeze_names_to is not None:
                boot_features.append(np.asarray(result.features, dtype=np.float64))

            series = fork_series_arrays(raw_data, input_ids, plen, args.logits_top_k)
            np.savez_compressed(fork_dir / f"gen_{gid:03d}.npz", **series)

            metadata: dict[str, Any] = {
                "generation_id": gid,
                "prompt_id": str(rec["prompt_id"]),
                "prompt_class": str(rec["prompt_class"]),
                "prompt": rec.get("prompt"),
                "prompt_idx": rec.get("prompt_idx"),
                "seed_idx": int(rec["seed_idx"]),
                "seed": rec.get("seed"),
                "user_prompt": rec.get("user_prompt"),
                "prompt_length": plen,
                "num_generated_tokens": int(len(input_ids) - plen),
                "num_features": int(len(result.features)),
                # on-disk key stays "tier_slices" (anamnesis STORED_BLOCK_SLICES_KEY)
                "tier_slices": {k: list(v) for k, v in result.block_slices.items()},
                "extraction_version": 3,
                "v3_config_source": config_source,
            }
            save_features(gid, result, metadata, sig_dir)

            # <out-dir>/cells.json carries the generation stage's metadata forward
            # so the map build never needs --gen-dir; the paths are relative to
            # --out-dir.
            cells.append(
                {
                    **{k: v for k, v in metadata.items() if k != "tier_slices"},
                    "generated_text": rec.get("generated_text"),
                    "n_steps": int(len(series["logits_entropy"])),
                    "signature_path": f"signatures/gen_{gid:03d}.npz",
                    "fork_series_path": f"fork_series/gen_{gid:03d}.npz",
                    "raw_saved": bool(save_raw),
                }
            )
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - started
                rate = len(cells) / elapsed if elapsed > 0 else 0.0
                eta = (len(records) - i - 1) / rate if rate > 0 else 0.0
                logger.info(
                    "%d/%d gen_%03d: %d feats, T=%d, %.0fs (ETA %.0fs)",
                    i + 1, len(records), gid, len(result.features),
                    len(series["logits_entropy"]), elapsed, eta,
                )
        except Gate0Failure as exc:
            # Not a per-gen failure: every later gen would fail identically, and the
            # vectors already written are un-projectable. Die here.
            logger.error("%s", exc)
            logger.error(
                "aborting the run. Check --calib-dir and the v3 family config "
                "(source: %s) before rerunning.", config_source,
            )
            return 3
        except Exception as exc:  # noqa: BLE001 — a dead gen is recorded, never skipped
            logger.exception("gen_%03d failed", gid)
            failures.append(f"gen_{gid:03d}: {type(exc).__name__}: {exc}")

    if not cells:
        logger.error("no gens succeeded; refusing to write an empty result")
        return 1

    assert feature_names is not None  # non-empty cells implies a first success
    meta = {
        "stage": "expB0_run_replay_b0",
        "model_id": model_path,
        "preset": args.preset,
        "sampled_layers": preset.sampled_layers,
        "gen_dir": str(args.gen_dir),
        "calib_dir": str(args.calib_dir),
        "calibration": calib_provenance,
        "discriminants": str(args.discriminants),
        "freeze_names_to": str(args.freeze_names_to) if args.freeze_names_to else None,
        "v3_config_source": config_source,
        "gate0": "passed" if full_names is not None else "frozen_from_first_gen",
        "feature_dim": len(feature_names),
        "n_corpus": n_corpus,
        "n_shard": n_shard,
        "n_resumed": n_resumed,
        "n_records": len(records),
        "n_cells": len(cells),
        "n_raw_saved": n_raw_saved,
        "save_raw": args.save_raw,
        "raw_prompt_ids": sorted(raw_prompts),
        "logits_top_k": int(args.logits_top_k),
        "resume": bool(args.resume),
        "shard": {"index": shard_index, "count": shard_count},
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
    }
    if shard_count > 1:
        # A shard holds a SLICE of the corpus, so it must not write the
        # whole-corpus index: <out-dir>/cells.json is what the map build reads,
        # and the last worker to finish would leave its own slice there under the
        # shared name. The launcher merges these shard files into it.
        shard_dir = args.out_dir / "shard_meta"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / f"cells_shard{shard_label}.json").write_text(
            json.dumps(cells, indent=2)
        )
        (shard_dir / f"run_meta_shard{shard_label}.json").write_text(
            json.dumps(meta, indent=2)
        )
    else:
        (args.out_dir / "cells.json").write_text(json.dumps(cells, indent=2))
        (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if args.freeze_names_to is not None:
        # The bootstrap manifest: names + standardization stats of THIS run's
        # signatures. Drop-in for --discriminants everywhere (this replay
        # reads FULL_names; LoomMap reads FULL_mean/FULL_scale). Near-zero
        # scales are kept as-is — every consumer already guards degenerate
        # dims itself (LoomMap.degenerate), and the manifest should not hide
        # which dims were dead in the bootstrap corpus.
        feats = np.stack(boot_features)
        full_mean = feats.mean(axis=0)
        full_scale = feats.std(axis=0)
        args.freeze_names_to.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.freeze_names_to,
            FULL_names=np.array(feature_names),
            FULL_mean=full_mean.astype(np.float32),
            FULL_scale=full_scale.astype(np.float32),
            manifest_meta=np.array(json.dumps({
                "bootstrapped_from": str(args.out_dir),
                "model_id": model_path, "preset": args.preset,
                "n_gens": int(feats.shape[0]),
                "n_dead_dims": int((full_scale < 1e-12).sum()),
            })),
        )
        logger.info("BOOTSTRAP manifest -> %s (%d names, %d gens, %d dead dims)",
                    args.freeze_names_to, len(feature_names), feats.shape[0],
                    int((full_scale < 1e-12).sum()))

    logger.info(
        "DONE%s — %d gens x %d features (%d raw, %d failures) in %.1fs -> %s",
        f" [shard {shard_label}]" if shard_label else "",
        len(cells), len(feature_names), n_raw_saved, len(failures),
        time.time() - started, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
