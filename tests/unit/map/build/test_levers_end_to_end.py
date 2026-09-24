"""The live half of the L1 ladder on a toy Llama: bank -> build, gates and all.

``bank_mean_hiddens`` and ``build_levers`` are individually tested (clustering
and the held-out algebra in ``test_build_levers``). What none of those can see is
the WIRING: that ``bank_mean_hiddens`` — loading through ``pleroma.model`` —
writes an npz ``build_levers`` can join, that its site-indexing gate actually
PASSES on a real transformers stack, and that the levers npz it produces is what
the dose lanes' loader (``pleroma.levers.npz.load_levers``) accepts.

So this builds a 4-layer randomly-initialised Llama (seconds, CPU, no weights
downloaded), plants a two-cluster corpus around it, and runs both ``main()``
functions. The NUMBERS are meaningless.

Needs torch AND anamnesis, so it skips on a bare laptop.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anamnesis.extraction.model_loader")
transformers = pytest.importorskip("transformers")
anamnesis_config = pytest.importorskip("anamnesis.config")

from tests.unit.map.build.test_build_levers import _plant  # noqa: E402

VOCAB = 64
HIDDEN = 32
LAYERS = 4
PROMPT_LEN = 6
CONT_LEN = 14
SITES = "1,2"


def _toy_model(path: Path) -> None:
    """A 4-layer random Llama plus a word-level tokenizer, both saved to ``path``.

    The tokenizer exists only because an HF model dir is expected to carry one;
    nothing downstream reads the strings, and a real tokenizer would be megabytes
    of download for a test that never looks at a word.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    LlamaForCausalLM(config).save_pretrained(path)

    backing = Tokenizer(WordLevel({f"t{i}": i for i in range(VOCAB)}, unk_token="t0"))
    backing.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backing,
        eos_token="t2",
        pad_token="t0",
        # A minimal chat template, so apply_chat_template works.
        chat_template=(
            "{% for m in messages %}{{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}t3 {% endif %}"
        ),
    ).save_pretrained(path)


def _run(module: str, argv: list[str]) -> int:
    """Call a stage's ``main()`` with a set argv, as the node would invoke it."""
    mod = importlib.import_module(module)
    old = sys.argv
    sys.argv = [module, *argv]
    try:
        return int(mod.main())
    finally:
        sys.argv = old


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A toy model, a planted stage-2 run dir, and matching stage-1 gen records.

    The anamnesis registry gets a 4-layer 'toy' row (a registry file of its own
    on ``ANAMNESIS_MODELS``, exactly how pleroma adds 70b-modelc): both node
    stages cross-check the preset's layer count against the loaded model and
    refuse a mismatch, which is the behaviour we want on the node and an
    obstacle here.
    """
    _toy_model(tmp_path / "model")
    registry = tmp_path / "toy_models.json"
    registry.write_text(json.dumps({"presets": {"toy": {
        "name": "toy", "model_id": str(tmp_path / "model"), "torch_dtype": "float32",
        "num_layers": LAYERS, "hidden_dim": HIDDEN, "num_attention_heads": 4,
        "num_kv_heads": 2, "head_dim": HIDDEN // 4,
        "sampled_layers": [0, 1, 2, 3], "pca_layers": [1, 2], "trajectory_layers": [1, 2],
        "contrastive_layers": [1, 2], "early_layer_cutoff": 1, "late_layer_cutoff": 2,
        "temperature": 1.0, "top_p": 1.0, "max_new_tokens": CONT_LEN,
        "eos_token_ids": [2], "calibration_root": "outputs", "calibration_dir": "toy",
    }}}))
    monkeypatch.setenv(anamnesis_config.MODELS_ENV, str(registry))

    planted = _plant(tmp_path, run_name="toy_replay", prompts=("mf1", "dc1"), per_cluster=4)
    run_dir = Path(planted["run_dir"])
    gen_records = tmp_path / "toy_gen" / "gen_records"
    gen_records.mkdir(parents=True)
    rng = np.random.default_rng(5)
    # One prompt per prompt_id, shared by all its seeds — as the real corpus is,
    # and as the (retired) steer_levers required (it reconstructed a group's prompt from any
    # member and refuses a group whose members disagree).
    prompt_of: dict[str, list[int]] = {}
    for cell in json.loads((run_dir / "cells.json").read_text()):
        prompt = prompt_of.setdefault(
            cell["prompt_id"], [int(x) for x in rng.integers(1, VOCAB, PROMPT_LEN)]
        )
        (gen_records / f"gen_{cell['generation_id']:03d}.json").write_text(
            json.dumps(
                {
                    "generation_id": cell["generation_id"],
                    "prompt_id": cell["prompt_id"],
                    "prompt_class": cell["prompt_class"],
                    "seed_idx": cell["seed_idx"],
                    "prompt_length": PROMPT_LEN,
                    "input_ids": prompt
                    + [int(x) for x in rng.integers(1, VOCAB, CONT_LEN)],
                }
            )
        )
    return {
        "root": tmp_path,
        "model": tmp_path / "model",
        "run_dir": run_dir,
        "gen_dir": tmp_path / "toy_gen",
    }


def test_bank_then_build_runs_end_to_end_and_the_site_gate_passes(
        world: dict[str, Path]) -> None:
    from pleroma.dose.ladder import dose_vector, find_group
    from pleroma.levers.npz import load_levers
    from pleroma.levers.ruler import site_norms

    root = world["root"]
    hiddens = root / "hiddens" / "mean_hiddens.npz"
    levers = root / "levers.npz"

    assert _run(
        "pleroma.map.build.hiddens",
        [
            "--gen-dirs", str(world["gen_dir"]),
            "--out-dir", str(root / "hiddens"),
            "--model-path", str(world["model"]),
            "--preset", "toy", "--sites", SITES, "--batch", "3",
            "--model-dtype", "float32", "--device", "cpu",
        ],
    ) == 0
    bank_meta = json.loads((root / "hiddens" / "bank_meta.json").read_text())
    # The gate that says hidden_states[s] IS the input of decoder layer s.
    assert bank_meta["site_indexing_gate"]["status"] == "passed"
    assert bank_meta["site_indexing_gate"]["n_hidden_states"] == LAYERS + 1
    assert bank_meta["n_banked"] == 16
    assert not bank_meta["failures"]

    assert _run(
        "pleroma.map.build.levers",
        ["--hiddens", str(hiddens), "--run-dirs", str(world["run_dir"]), "--out", str(levers)],
    ) == 0

    # The dose lanes' view of the same file: loads, validates, and a group
    # resolves to a dose vector whose norms scale with alpha.
    bank = load_levers(levers)
    assert bank.sites == [1, 2]
    assert bank.levers.shape == (bank.n_groups, 2, HIDDEN)
    gi = find_group(bank, "mf1|orig")
    base = site_norms(dose_vector(bank.levers[gi], 0.5, 1.0))
    assert site_norms(dose_vector(bank.levers[gi], 1.0, 1.0)) == pytest.approx(
        [2.0 * x for x in base], rel=1e-5)
    assert site_norms(dose_vector(bank.levers[gi], 0.0, 1.0)) == [0.0, 0.0]
