"""pleroma.harvest.anamnesis_seam — the pin, the registry, and the lane on CPU.

The lane tests run a REAL (tiny, random) Llama through the pinned anamnesis
fast lane on CPU, the way upstream's own equivalence tests do, so the seam's
composition is exercised end to end with no GPU:

  * the seam's loaded-model names ARE upstream's (#16), and harvest_loaded on
    an already-loaded model agrees with upstream's resolve_fast_lane — same
    lane_id, schema and calibration digest, byte-identical features;
  * the fork series read by the forward hook equals the replay lane's
    fork_series_arrays on the same model and tokens;
  * replay_manifest_to_dir leaves the on-disk shape adopt_lane_output reads.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from anamnesis.config import ModelPreset  # noqa: E402

from pleroma.config import anamnesis_registry  # noqa: E402
from pleroma.harvest import anamnesis_seam as seam  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

# ── the pin ──────────────────────────────────────────────────────────────────


def test_pinned_commit_is_the_gitlink() -> None:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "vendor/anamnesis"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("not a git checkout")
    mode, sha, *_ = out.stdout.split()
    assert mode == "160000", "vendor/anamnesis is not a submodule gitlink"
    assert sha == seam.PINNED_COMMIT


def test_the_imported_anamnesis_is_the_pin() -> None:
    probe = seam.verify_anamnesis_pin(allow_mismatch=False)
    assert probe.commit == seam.PINNED_COMMIT and not probe.dirty


def test_the_loaded_model_seam_is_upstreams() -> None:
    """No shim left: every loaded-model name the seam exposes is upstream's object."""
    from anamnesis.extraction.fast import harvest, runtime

    for name in ("LaneCalibration", "read_lane_calibration", "load_lane_model",
                 "check_loaded_model", "PreparedLane", "prepare_fast_lane"):
        assert getattr(seam, name) is getattr(runtime, name), name
    for name in ("HarvestResult", "LogitSeries", "harvest_loaded"):
        assert getattr(seam, name) is getattr(harvest, name), name
    with pytest.raises(AttributeError):
        seam.not_a_name  # noqa: B018


@pytest.mark.parametrize("probe", [
    seam.PinProbe(root=Path("/x"), commit="0" * 40, source="test"),
    seam.PinProbe(root=Path("/x"), commit=None, source="test"),
    seam.PinProbe(root=Path("/x"), commit=seam.PINNED_COMMIT, source="test", dirty=True),
])
def test_a_mismatch_is_refused_unless_overridden(monkeypatch, caplog, probe) -> None:
    monkeypatch.setattr(seam, "imported_anamnesis_commit", lambda: probe)
    monkeypatch.delenv(seam.ALLOW_MISMATCH_ENV, raising=False)
    with pytest.raises(seam.AnamnesisPinError, match=seam.PINNED_COMMIT):
        seam.verify_anamnesis_pin()
    assert seam.verify_anamnesis_pin(allow_mismatch=True) is probe
    monkeypatch.setenv(seam.ALLOW_MISMATCH_ENV, "1")
    assert seam.verify_anamnesis_pin() is probe
    assert "PIN OVERRIDDEN" in caplog.text


def test_direct_url_vcs_commit_is_read(monkeypatch, tmp_path) -> None:
    import anamnesis

    class Dist:
        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return json.dumps({"url": "https://x", "vcs_info": {"commit_id": "abc"}})

    fake_pkg = tmp_path / "site" / "anamnesis"
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(anamnesis, "__file__", str(fake_pkg / "__init__.py"))
    monkeypatch.setattr(seam.importlib.metadata, "distribution", lambda _n: Dist())
    probe = seam.imported_anamnesis_commit()
    assert probe.commit == "abc" and "vcs_info" in probe.source


# ── the registry ─────────────────────────────────────────────────────────────


def test_registry_env_constant_is_anamnesis_s() -> None:
    from anamnesis.config import MODELS_ENV

    assert anamnesis_registry.MODELS_ENV == MODELS_ENV


def test_ensure_registry_is_idempotent_and_keeps_other_files(tmp_path) -> None:
    """On a registry file of its own (the default registry is private and not
    exported)."""
    other = tmp_path / "mine.json"
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    env = {anamnesis_registry.MODELS_ENV: str(other)}
    v1 = anamnesis_registry.ensure_registry(registry, environ=env)
    v2 = anamnesis_registry.ensure_registry(registry, environ=env)
    assert v1 == v2
    parts = v1.split(":")
    assert parts[0] == str(other) and len(parts) == 2
    assert parts[1] == str(registry.resolve())


def test_a_missing_registry_file_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        anamnesis_registry.ensure_registry(tmp_path / "nope.json", environ={})


# ── the lane, on CPU ─────────────────────────────────────────────────────────

LAYERS, HIDDEN, HEADS, KV, HEAD_DIM, VOCAB, INTER = 3, 64, 4, 2, 16, 96, 128
PROMPT, MAX_POS = 9, 64


def _toy_preset() -> ModelPreset:
    return ModelPreset(
        name="toy-lane", model_id="toy", torch_dtype="float32", num_layers=LAYERS,
        hidden_dim=HIDDEN, num_attention_heads=HEADS, num_kv_heads=KV, head_dim=HEAD_DIM,
        sampled_layers=(0, 1, 2), pca_layers=(0, 1), trajectory_layers=(1,),
        contrastive_layers=(1,), early_layer_cutoff=0, late_layer_cutoff=2,
        temperature=1.0, top_p=1.0, max_new_tokens=16, eos_token_ids=(1,),
        calibration_root="outputs", calibration_dir="toy",
    )


def _toy_loaded(preset: ModelPreset):
    """A random tiny Llama hooked the way ``load_lane_model`` hooks one."""
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import (
        HookState,
        LoadedModel,
        _make_gate_proj_hook,
        _make_k_proj_hook,
        _make_q_proj_hook,
        _make_v_proj_hook,
    )

    torch.manual_seed(20260923)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV,
        head_dim=HEAD_DIM,
    )
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).eval()
    state, handles = HookState(), []
    for layer in preset.sampled_layers:
        attn = model.model.layers[layer].self_attn
        handles.append(attn.k_proj.register_forward_hook(_make_k_proj_hook(layer, state, KV, HEAD_DIM)))
        handles.append(attn.v_proj.register_forward_hook(_make_v_proj_hook(layer, state, KV, HEAD_DIM)))
        handles.append(attn.q_proj.register_forward_hook(_make_q_proj_hook(layer, state, HEADS, HEAD_DIM)))
        handles.append(model.model.layers[layer].mlp.gate_proj.register_forward_hook(
            _make_gate_proj_hook(layer, state)))
    return LoadedModel(model, None, state, handles,
                       ModelConfig.from_preset(preset, device_map="cpu"))


@pytest.fixture
def lane_env(monkeypatch, tmp_path):
    """CUBLAS pin in the environment, and torch's global arithmetic flags
    restored afterwards (require_lane_arithmetic sets them process-wide)."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    saved = (torch.are_deterministic_algorithms_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    rng = np.random.default_rng(3)
    calib = tmp_path / "calib"
    calib.mkdir()
    np.savez(calib / "positional_means.npz",
             positional_means=rng.normal(0, 0.01, (LAYERS + 1, MAX_POS, HIDDEN)).astype(np.float32))
    with open(calib / "pca_model.pkl", "wb") as fh:
        pickle.dump({"components": rng.normal(size=(50, HIDDEN)).astype(np.float32),
                     "mean": rng.normal(0, 0.01, HIDDEN).astype(np.float32)}, fh)
    rows = {
        str(g): {"prompt_length": PROMPT,
                 "input_ids": rng.integers(2, VOCAB, size=PROMPT + n).tolist()}
        for g, n in ((0, 12), (1, 20), (2, 12))
    }
    man = tmp_path / "manifest"
    man.mkdir()
    (man / "manifest.json").write_text(json.dumps({"entries": rows}))
    (man / "metadata.json").write_text(json.dumps({"generations": [
        {"generation_id": int(g), "prompt_id": f"p{g}"} for g in rows]}))
    try:
        yield calib, man / "manifest.json", rows
    finally:
        torch.use_deterministic_algorithms(saved[0])
        torch.backends.cuda.matmul.allow_tf32 = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]


def test_harvest_loaded_agrees_with_upstream_resolve_fast_lane(lane_env, monkeypatch) -> None:
    """Same lane_id, schema and calibration digest, byte-identical features."""
    from anamnesis.extraction import model_loader
    from anamnesis.extraction.fast.runtime import resolve_fast_lane

    calib, _manifest, rows = lane_env
    preset = _toy_preset()
    loaded = _toy_loaded(preset)
    # Upstream loads its own model; hand it ours (test-only substitution).
    monkeypatch.setattr(model_loader, "load_model", lambda *a, **k: loaded)
    upstream = resolve_fast_lane(preset=preset, model_path="toy", calib_dir=calib,
                                 entries=rows, gen_ids=[0, 2], device="cpu")
    prepared = seam.prepare_fast_lane(preset, calib, device="cpu", loaded=loaded)
    span = upstream.spans[0]
    lane, schema = prepared.lane(span.n_steps)
    assert lane.lane_id == upstream.lane.lane_id
    assert schema.feature_names == upstream.feature_names
    assert schema.family_slices == upstream.schemas[span.gen_id].family_slices
    assert prepared.calibration.calibration_sha256 == upstream.calibration_sha256
    assert prepared.calibration.calibration_files == upstream.calibration_files
    want = upstream.lane.replay_span(loaded, span.input_ids, span.prompt_length, span.end)
    got = seam.harvest_loaded(prepared, span.input_ids, prompt_len=span.prompt_length,
                              logit_series_top_k=7)
    assert got.features.tobytes() == want.features.tobytes()
    assert got.feature_names == tuple(want.feature_names)
    assert got.lane_id == want.metadata["lane_id"]


def test_the_fork_series_equals_the_replay_lanes(lane_env) -> None:
    """The hook-read series == run_replay_b0.fork_series_arrays on replay_extract."""
    from anamnesis.extraction.replay.extract import replay_extract

    from pleroma.harvest.replay import fork_series_arrays

    calib, _manifest, rows = lane_env
    preset = _toy_preset()
    loaded = _toy_loaded(preset)
    prepared = seam.prepare_fast_lane(preset, calib, device="cpu", loaded=loaded)
    ids = rows["1"]["input_ids"]
    got = seam.harvest_loaded(prepared, ids, prompt_len=PROMPT,
                              logit_series_top_k=10).series_arrays()
    loaded.disable_hooks()
    raw = replay_extract(loaded, ids, PROMPT, positional_means=None)
    want = fork_series_arrays(raw, ids, PROMPT, 10)
    assert set(got) == set(want)
    for key in ("chosen_ids", "input_ids", "prompt_length", "logits_indices"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    for key in ("logits_values", "logits_entropy"):
        np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-5, err_msg=key)
        assert got[key].dtype == want[key].dtype, key


def test_replay_manifest_leaves_the_shape_adopt_lane_output_reads(lane_env, tmp_path) -> None:
    from pleroma.harvest import worker as harvest_worker

    calib, manifest, rows = lane_env
    preset = _toy_preset()
    prepared = seam.prepare_fast_lane(preset, calib, device="cpu", loaded=_toy_loaded(preset))
    out = tmp_path / "lane_out"
    written = seam.replay_manifest_to_dir(prepared, manifest, out, model_path="toy",
                                          gen_ids=[2, 0])
    assert written == [2, 0]
    dep = json.loads((out / "deployment.json").read_text())
    assert dep["anamnesis_commit"] == seam.PINNED_COMMIT and dep["selected_ids"] == [2, 0]
    assert dep["lane_id"] and dep["certified"] is False
    # the stack block lane_parity compares (the banked corpus records it as "lane")
    from pleroma.harvest.lane_parity import STACK_KEYS
    assert set(STACK_KEYS) <= set(dep["lane"])
    for g in written:
        meta = json.loads((out / f"gen_{g:03d}.json").read_text())
        assert meta["prompt_id"] == f"p{g}" and meta["lane_id"] == dep["lane_id"]
        assert "tier_slices" in meta
        with np.load(out / "fork_series" / f"gen_{g:03d}.npz") as fz:
            assert fz["logits_entropy"].shape == (len(rows[str(g)]["input_ids"]) - PROMPT - 1,)
    # adopt it the way the worker does
    records = [{"generation_id": g, "prompt_id": f"p{g}", "prompt_class": "c", "seed_idx": 0,
                "prompt_length": PROMPT, "input_ids": rows[str(g)]["input_ids"]} for g in written]
    with np.load(out / "gen_000.npz") as z:
        names = [str(n) for n in z["feature_names"]]
    loom = tmp_path / "loom"
    cells, failures, seen, lane_id = harvest_worker.adopt_lane_output(
        out, loom, records, names, None, lambda a, b: None, RuntimeError)
    assert failures == [] and len(cells) == 2 and seen == names and lane_id == dep["lane_id"]
    assert (loom / "signatures" / "gen_002.npz").exists()
    assert (loom / "fork_series" / "gen_000.npz").exists()
    # an existing output dir is refused, as upstream's runner refuses it
    with pytest.raises(FileExistsError):
        seam.replay_manifest_to_dir(prepared, manifest, out, model_path="toy")


def test_a_loaded_model_of_the_wrong_shape_is_refused(lane_env) -> None:
    calib, _m, _r = lane_env
    preset = _toy_preset()
    wrong = preset.model_copy(update={"num_layers": 4, "sampled_layers": (0, 1, 2)})
    with pytest.raises(ValueError, match="num_hidden_layers"):
        seam.prepare_fast_lane(wrong, calib, device="cpu", loaded=_toy_loaded(preset))


def test_an_unsupported_span_is_refused_before_any_replay(lane_env, tmp_path) -> None:
    calib, manifest, rows = lane_env
    preset = _toy_preset()
    prepared = seam.prepare_fast_lane(preset, calib, device="cpu", loaded=_toy_loaded(preset))
    bad = json.loads(manifest.read_text())
    bad["entries"]["9"] = {"prompt_length": PROMPT, "input_ids": list(range(2, 2 + MAX_POS + 5))}
    mpath = tmp_path / "bad" / "manifest.json"
    mpath.parent.mkdir()
    mpath.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="outside supported"):
        seam.replay_manifest_to_dir(prepared, mpath, tmp_path / "o", model_path="toy")
    assert not (tmp_path / "o").exists()
