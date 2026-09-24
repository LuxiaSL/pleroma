"""The GPU-lane harvest path, and the two silent failures it is built against.

CPU-only: numpy, json, pathlib. Everything that needs the node (torch,
anamnesis, the 141 GB checkpoint) lives inside ``harvest_worker.main()`` or
behind a deferred import, so this file exercises the parts that DECIDE things —
the bins-config agreement check, Gate 0 over the lane's names, and the adoption
of a lane run's output into the loom dir's harvest layout.

★ THE TWO FAILURES BEING TESTED FOR ARE BOTH SILENT AT RUNTIME:

  1. ``7 layers x 20 bins x 3 summaries = 420`` whether the layers are the 3B
     defaults or the 70B map's. A worker harvesting the wrong third of an
     80-layer model produces a correctly shaped, meaningless row and NOTHING
     raises. Only a name-by-name comparison with the served map catches it.

  2. ``len(a) == len(b)`` is not agreement. The blocker this whole lane exists
     to clear was 3,441 names vs 4,086 with one list a PREFIX of the other —
     every shared name identical, and the join still wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pleroma.harvest import pool as harvest_pool
from pleroma.harvest import worker as harvest_worker
from pleroma.harvest import gpu_manifests as w6_gpu_manifests

# The 70B map's bins layers, and extract_bins' 3B/8B defaults. Both are 7 long.
MAP_LAYERS = [12, 23, 35, 47, 59, 70, 79]
DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24, 27]
SUMMARIES = ["hmean", "hmax", "hent"]


def _map_meta(**over: object) -> dict:
    meta = {
        "bins_config": {
            "family": "binsB", "n_bins": 20, "layers": list(MAP_LAYERS),
            "head_summaries": list(SUMMARIES), "temporal": False,
        },
        "discriminants_sha256": "0" * 64,
    }
    meta.update(over)
    return meta


# ── 1. the bins config must be the SERVED MAP's bins config ──────────────────


def test_agreeing_bins_config_reports_nothing() -> None:
    assert harvest_worker.bins_config_disagreements(
        _map_meta(), family="binsB", layers=MAP_LAYERS, n_bins=20,
        summaries=SUMMARIES, temporal=False,
    ) == []


def test_the_wrong_layers_are_caught_AND_named_as_the_same_width() -> None:
    """The failure this pins, exactly: same count, wrong third of the model.

    The message has to say SAME COUNT, because the first thing anyone does with
    a config mismatch is check the dimensions — and the dimensions agree. A
    report that only prints two lists invites "but they're both 420" as the
    conclusion.
    """
    out = harvest_worker.bins_config_disagreements(
        _map_meta(), family="binsB", layers=DEFAULT_LAYERS, n_bins=20,
        summaries=SUMMARIES, temporal=False,
    )
    assert len(out) == 1
    assert "SAME COUNT" in out[0]
    assert str(MAP_LAYERS) in out[0] and str(DEFAULT_LAYERS) in out[0]
    # and the widths really are equal, which is the whole point
    assert len(MAP_LAYERS) * 20 * len(SUMMARIES) == len(DEFAULT_LAYERS) * 20 * len(SUMMARIES)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"n_bins": 16}, "n_bins"),
        ({"summaries": ["hmean"]}, "head_summaries"),
        ({"temporal": True}, "temporal"),
        ({"family": "binsA"}, "family"),
    ],
)
def test_every_field_of_the_bins_config_is_compared(kwargs: dict, needle: str) -> None:
    base = dict(family="binsB", layers=MAP_LAYERS, n_bins=20,
                summaries=SUMMARIES, temporal=False)
    base.update(kwargs)
    out = harvest_worker.bins_config_disagreements(_map_meta(), **base)
    assert any(needle in line for line in out), out


def test_a_map_with_no_bins_config_is_reported_not_assumed_fine() -> None:
    out = harvest_worker.bins_config_disagreements(
        {}, family="binsB", layers=MAP_LAYERS, n_bins=20,
        summaries=SUMMARIES, temporal=False,
    )
    assert out and "no `bins_config`" in out[0]


def test_the_discriminants_are_compared_by_sha_not_by_path(tmp_path: Path) -> None:
    disc = tmp_path / "manifest.npz"
    disc.write_bytes(b"not really an npz, but it hashes")
    import hashlib
    real = hashlib.sha256(disc.read_bytes()).hexdigest()

    assert harvest_worker.discriminants_disagreements(
        _map_meta(discriminants_sha256=real), disc) == []
    wrong = harvest_worker.discriminants_disagreements(
        _map_meta(discriminants_sha256="f" * 64), disc)
    assert len(wrong) == 1 and real in wrong[0]


def test_load_map_meta_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "map.npz"
    np.savez_compressed(p, meta=np.array(json.dumps(_map_meta())), W=np.zeros((2, 2)))
    assert harvest_worker.load_map_meta(p)["bins_config"]["layers"] == MAP_LAYERS
    bare = tmp_path / "bare.npz"
    np.savez_compressed(bare, W=np.zeros((2, 2)))
    with pytest.raises(KeyError):
        harvest_worker.load_map_meta(bare)


# ── 2. adopting a lane run into the loom dir's harvest layout ────────────────


class _Gate0Failure(RuntimeError):
    pass


def _check_gate0(names: list[str], full: list[str]) -> None:
    """run_replay_b0.check_gate0's contract, restated so this file needs no torch."""
    if list(names) != list(full):
        raise _Gate0Failure(
            f"GATE 0 FAILED — lengths {len(names)} vs {len(full)}"
        )


def _lane_run(
    tmp_path: Path, gids: list[int], names: list[str], *, forks: bool = True,
    steps: int = 11,
) -> tuple[Path, Path, list[dict]]:
    """A loom dir plus a finished lane output dir, shaped as the lane writes it."""
    loom_dir = tmp_path / "loom_000"
    out = tmp_path / "lane" / "out"
    (out / "fork_series").mkdir(parents=True)
    records = []
    for g in gids:
        plen = 30
        records.append({
            "generation_id": g, "prompt_id": "loom000",
            "prompt_class": "loom_context", "prompt": "<chat context>",
            "prompt_idx": 0, "seed_idx": g, "seed": 7,
            "user_prompt": "<chat context>", "prompt_length": plen,
            "input_ids": list(range(plen + steps + 1)),
            "generated_text": f"future {g}",
        })
        np.savez_compressed(
            out / f"gen_{g:03d}.npz",
            features=np.arange(len(names), dtype=np.float32),
            feature_names=np.array(names),
        )
        (out / f"gen_{g:03d}.json").write_text(json.dumps(
            {"generation_id": g, "lane_id": "lane-abc123", "certified": False}))
        if forks:
            np.savez_compressed(
                out / "fork_series" / f"gen_{g:03d}.npz",
                logits_entropy=np.zeros(steps, dtype=np.float32),
                chosen_ids=np.zeros(steps, dtype=np.int32),
            )
    loom_dir.mkdir(parents=True, exist_ok=True)
    return loom_dir, out, records


def test_a_lane_run_becomes_the_same_on_disk_shape_a_replay_harvest_leaves(
    tmp_path: Path,
) -> None:
    names = [f"f{i}" for i in range(4086)]
    loom_dir, out, records = _lane_run(tmp_path, [0, 1, 2], names)

    cells, failures, seen, lane_id = harvest_worker.adopt_lane_output(
        out, loom_dir, records, names, None, _check_gate0, _Gate0Failure,
    )

    assert failures == []
    assert seen == names
    assert lane_id == "lane-abc123"
    assert [c["generation_id"] for c in cells] == [0, 1, 2]
    for g in (0, 1, 2):
        assert (loom_dir / "signatures" / f"gen_{g:03d}.npz").exists()
        assert (loom_dir / "signatures" / f"gen_{g:03d}.json").exists()
        assert (loom_dir / "fork_series" / f"gen_{g:03d}.npz").exists()
        # moved, not copied — the lane dir is scratch and must not double the bytes
        assert not (out / f"gen_{g:03d}.npz").exists()
    c = cells[0]
    assert c["signature_path"] == "signatures/gen_000.npz"
    assert c["fork_series_path"] == "fork_series/gen_000.npz"
    assert c["n_steps"] == 11
    assert c["num_features"] == 4086
    assert c["num_generated_tokens"] == 12
    assert c["prompt_length"] == 30
    assert c["raw_saved"] is False
    # provenance that names the instrument rather than inheriting the frozen
    # surface's string
    assert c["extraction_lane"] == "gpu"
    assert "sigbridge" not in c["v3_config_source"]
    assert "GPU lane" in c["v3_config_source"]


def test_gate0_refuses_a_prefix_rather_than_truncating_to_it(tmp_path: Path) -> None:
    """3,441 vs 4,086 with one list a prefix of the other — the real blocker.

    Every name they share is identical; only the tail is missing. Anything that
    compares counts-and-then-shrugs, or that zips to the shorter list, joins two
    different spaces and produces numbers that look fine.
    """
    lane_names = [f"f{i}" for i in range(4086)]
    replay_names = lane_names[:3441]
    loom_dir, out, records = _lane_run(tmp_path, [0], lane_names)
    with pytest.raises(_Gate0Failure, match="4086 vs 3441"):
        harvest_worker.adopt_lane_output(
            out, loom_dir, records, replay_names, None, _check_gate0, _Gate0Failure,
        )


def test_a_gen_whose_names_drift_mid_run_is_a_failure_not_a_silent_row(
    tmp_path: Path,
) -> None:
    names = [f"f{i}" for i in range(8)]
    loom_dir, out, records = _lane_run(tmp_path, [0, 1], names)
    # gen 1 comes back in a different space
    np.savez_compressed(
        out / "gen_001.npz",
        features=np.zeros(8, dtype=np.float32),
        feature_names=np.array([f"g{i}" for i in range(8)]),
    )
    cells, failures, seen, _ = harvest_worker.adopt_lane_output(
        out, loom_dir, records, names, None, _check_gate0, _Gate0Failure,
    )
    assert [c["generation_id"] for c in cells] == [0]
    assert len(failures) == 1 and "diverged" in failures[0]
    assert seen == names


def test_a_missing_fork_series_names_the_writer_that_should_have_written_it(
    tmp_path: Path,
) -> None:
    """Upstream's run_gpu_replay writes no fork_series (raw_tensors_saved=False).

    A lane output without fork_series/ gives build_pairs nothing to read, and
    the only symptom would be a downstream KeyError days later. The failure has
    to name the seam function that writes it.
    """
    names = [f"f{i}" for i in range(8)]
    loom_dir, out, records = _lane_run(tmp_path, [0], names, forks=False)
    cells, failures, _, _ = harvest_worker.adopt_lane_output(
        out, loom_dir, records, names, None, _check_gate0, _Gate0Failure,
    )
    assert cells == []
    assert len(failures) == 1
    assert "replay_manifest_to_dir" in failures[0]


def test_a_gen_the_lane_never_wrote_is_a_failure_not_a_missing_row(
    tmp_path: Path,
) -> None:
    names = [f"f{i}" for i in range(8)]
    loom_dir, out, records = _lane_run(tmp_path, [0, 1], names)
    (out / "gen_001.npz").unlink()
    cells, failures, _, _ = harvest_worker.adopt_lane_output(
        out, loom_dir, records, names, None, _check_gate0, _Gate0Failure,
    )
    assert [c["generation_id"] for c in cells] == [0]
    assert len(failures) == 1 and "gen_001" in failures[0]


def test_gate0_is_amortized_exactly_as_the_replay_lane_amortizes_it(
    tmp_path: Path,
) -> None:
    """A second harvest in the same process re-uses the first's verdict."""
    names = [f"f{i}" for i in range(8)]
    loom_dir, out, records = _lane_run(tmp_path, [0], names)
    calls: list[int] = []

    def counting_gate(n: list[str], f: list[str]) -> None:
        calls.append(1)
        _check_gate0(n, f)

    _, _, seen, _ = harvest_worker.adopt_lane_output(
        out, loom_dir, records, names, None, counting_gate, _Gate0Failure)
    loom2, out2, rec2 = _lane_run(tmp_path / "second", [0], names)
    _, failures, _, _ = harvest_worker.adopt_lane_output(
        out2, loom2, rec2, names, seen, counting_gate, _Gate0Failure)
    assert failures == []
    assert sum(calls) == 1


# ── 3. the manifest transform is REUSED, not reimplemented ───────────────────


def test_write_manifests_turns_gen_records_into_what_the_lane_consumes(
    tmp_path: Path,
) -> None:
    gen_dir = tmp_path / "loom_000"
    (gen_dir / "gen_records").mkdir(parents=True)
    for g in range(4):
        (gen_dir / "gen_records" / f"gen_{g:03d}.json").write_text(json.dumps({
            "generation_id": g, "prompt_id": "loom000",
            "prompt_class": "loom_context", "seed_idx": g,
            "prompt_length": 30, "input_ids": list(range(40 + g)),
        }))
    n = w6_gpu_manifests.write_manifests(gen_dir, tmp_path / "man", shards=1)
    assert n == 4
    man = json.loads((tmp_path / "man" / "shard_00" / "manifest.json").read_text())
    assert sorted(int(k) for k in man["entries"]) == [0, 1, 2, 3]
    assert man["entries"]["2"]["prompt_length"] == 30
    assert len(man["entries"]["2"]["input_ids"]) == 42
    meta = json.loads((tmp_path / "man" / "shard_00" / "metadata.json").read_text())
    assert [g["generation_id"] for g in meta["generations"]] == [0, 1, 2, 3]


def test_write_manifests_refuses_an_empty_dir_and_the_cli_still_returns_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "gen_records").mkdir()
    with pytest.raises(FileNotFoundError):
        w6_gpu_manifests.write_manifests(tmp_path, tmp_path / "out", shards=1)
    monkeypatch.setattr(sys, "argv", [
        "w6_gpu_manifests", "--gen-dir", str(tmp_path),
        "--out", str(tmp_path / "out"), "--shards", "1",
    ])
    assert w6_gpu_manifests.main() == 2


# ── 4. one pool, one lane ────────────────────────────────────────────────────


def _pool_metas(tmp_path: Path, lanes: list[str | None]) -> None:
    n = len(lanes)
    shard_dir = tmp_path / "shard_meta"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, lane in enumerate(lanes):
        meta = {
            "stage": "harvest_worker_replay_b0", "shard": f"{i}of{n}",
            "gate0": "passed", "feature_dim": 6,
            "calibration": {"positional_means_sha256": "deadbeef"},
            "n_corpus": 6, "n_records": 3, "n_cells": 3,
            "failures": [], "elapsed_s": 1.0, "model_id": "m", "preset": "3b",
        }
        if lane is not None:
            meta["harvest_lane"] = lane
        (shard_dir / f"run_meta_shard{i}of{n}.json").write_text(json.dumps(meta))


def test_a_pool_refuses_to_merge_two_different_lanes(tmp_path: Path) -> None:
    """Different lanes can agree on width by accident; they never agree on names."""
    _pool_metas(tmp_path, ["replay", "gpu"])
    with pytest.raises(ValueError, match="different lanes"):
        harvest_pool.merge_shard_run_meta(tmp_path, 2)


def test_a_pool_of_pre_lane_workers_still_merges(tmp_path: Path) -> None:
    """Shards written before the field existed report nothing, consistently."""
    _pool_metas(tmp_path, [None, None])
    merged = harvest_pool.merge_shard_run_meta(tmp_path, 2)
    assert merged["stage"] == "harvest_pool_replay_b0"
    assert merged["harvest_lane"] is None


def test_an_all_gpu_pool_says_so_in_its_merged_stage(tmp_path: Path) -> None:
    _pool_metas(tmp_path, ["gpu", "gpu"])
    merged = harvest_pool.merge_shard_run_meta(tmp_path, 2)
    assert merged["stage"] == "harvest_pool_gpu_lane"
    assert merged["harvest_lane"] == "gpu"


# ── 5. no monkeypatch of anamnesis survives in the harvest path ─────────────


def anamnesis_attribute_assignments(path: Path) -> list[str]:
    """Every `<anamnesis module>.<attr> = ...` assignment in `path`."""
    import ast

    tree = ast.parse(path.read_text())
    anamnesis_names = {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (getattr(node, "module", None) or "").startswith("anamnesis")
        for alias in node.names
    } | {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names if alias.name.startswith("anamnesis")
    }
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id in anamnesis_names):
                    hits.append(f"assigns {t.value.id}.{t.attr}")
    return hits


def test_nothing_in_the_harvest_path_patches_anamnesis() -> None:
    """The seam uses anamnesis's pinned public API and patches nothing in it.
    Guard against any `<anamnesis module>.<attr> = ...` assignment in the
    harvest path (file_sha, load_model, _load_calibration,
    run_gpu_replay.parser, _reduce_capture and save_features are the tempting
    ones)."""
    for rel in ("pleroma/harvest/worker.py", "pleroma/harvest/anamnesis_seam.py"):
        assert not anamnesis_attribute_assignments(REPO_ROOT / rel), rel


# ── 5b. the fallback rung that can never fire must SAY so ────────────────────


def _reset_preset_probe() -> None:
    from pleroma.serve import legacy as loom_serve

    loom_serve._SUBPROCESS_PRESETS = False


def test_subprocess_visible_presets_reports_the_CHILD_s_presets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """★ Not this process's. Pleroma's presets (70b-modelc) come from its
    registry file on ANAMNESIS_MODELS; a child spawned without the variable
    resolves only anamnesis' shipped rows, e.g.
    ['3b','8b','dsv2-lite','gemma3-27b','olmo2-7b','qwen-7b'].
    """
    import subprocess

    from pleroma.serve import legacy as loom_serve

    _reset_preset_probe()
    monkeypatch.setattr(loom_serve, "_SUBPROCESS_PRESETS", False, raising=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout='["3b", "8b", "dsv2-lite", "gemma3-27b", "olmo2-7b", "qwen-7b"]\n',
            stderr="",
        )

    monkeypatch.setattr(loom_serve.subprocess, "run", fake_run)
    got = loom_serve.subprocess_visible_presets("python", tmp_path)
    assert got is not None
    assert "70b-modelc" not in got
    assert "3b" in got and "8b" in got
    # Cached: the answer cannot change while the process lives, and the probe
    # costs a python startup on a path that runs inside a draw.
    again = loom_serve.subprocess_visible_presets("python", tmp_path)
    assert again == got and len(calls) == 1
    _reset_preset_probe()


def test_a_probe_that_cannot_answer_does_not_fail_the_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """None means 'could not tell', and the caller must then behave exactly as
    it did before this check existed. A preflight is allowed to be silent; it is
    not allowed to be the thing that kills a harvest."""
    from pleroma.serve import legacy as loom_serve

    _reset_preset_probe()
    monkeypatch.setattr(loom_serve, "_SUBPROCESS_PRESETS", False, raising=False)

    def boom(cmd, **kw):  # type: ignore[no-untyped-def]
        raise OSError("no python here")

    monkeypatch.setattr(loom_serve.subprocess, "run", boom)
    assert loom_serve.subprocess_visible_presets("python", tmp_path) is None
    _reset_preset_probe()


# ── 6. the default is the thing the live 8B loom already runs ────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]


def _help(module: str) -> str:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_the_worker_still_introspects_without_torch_or_anamnesis() -> None:
    """--help must work on a laptop: every heavy import stays inside main().

    The GPU lane goes through `pleroma.harvest.anamnesis_seam`; a module-level
    anamnesis/torch import would turn this entry point into an ImportError
    everywhere but the node, and would do it silently on the node itself.
    """
    text = _help("pleroma.harvest.worker")
    assert "--harvest-lane" in text
    assert "{replay,gpu}" in text
    assert "--map" in text


def test_harvest_lane_defaults_to_replay() -> None:
    """★ The live 8B loom and its pool run this worker with no --harvest-lane.

    A default of `gpu` would move the 3B/8B harvest onto a different instrument
    with a different name list, and Gate 0 would start failing on the demo
    instrument. The default is the frozen replay surface, stated in the help so
    it cannot drift without someone reading it.
    """
    text = " ".join(_help("pleroma.harvest.worker").split())
    assert "'replay' (default)" in text


def test_the_bins_stage_puts_its_input_on_the_BINS_device() -> None:
    """★ extract_one builds the input tensor on the device it is HANDED.

    So the third argument must name the device the bins model is on. It said
    `args.device` from the day --bins-device was added, and nothing caught it:
    below ~13B the two default to the same device, and the one 70B run that did
    split them (--device cuda:0 --bins-device cuda:4) aborted at Gate 0 in
    stage 1 and never reached the bins stage. With the lanes split the way they
    now are, that line runs — and a device mismatch would NaN every row and
    return a fan of dead futures without the word "device" appearing anywhere.

    Asserted on the source because the alternative needs two GPUs.
    """
    import ast

    src = (REPO_ROOT / "pleroma" / "harvest" / "worker.py").read_text()
    calls = [
        node for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "extract_one"
    ]
    assert len(calls) == 1, "expected exactly one extract_one call site"
    third = calls[0].args[2]
    assert isinstance(third, ast.Name) and third.id == "bins_device", (
        f"extract_one's `device` argument is {ast.dump(third)}; it must be "
        "bins_device, the device load_bins_model put the model on"
    )


def test_the_new_modules_import_without_torch() -> None:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         "import pleroma.harvest.worker,"
         " pleroma.harvest.anamnesis_seam, pleroma.config.anamnesis_registry;"
         "bad=[m for m in ('torch','transformers','anamnesis') if m in sys.modules];"
         "print('LEAKED', bad) if bad else print('CLEAN')"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "CLEAN", proc.stdout
