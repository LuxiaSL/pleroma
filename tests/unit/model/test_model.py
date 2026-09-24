"""pleroma.model — the layer, dtype and pad-token helpers the live path uses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pleroma.model.layers import decoder_layers, decoder_stack, model_hidden_dim

torch = pytest.importorskip("torch")


class _Stack:
    def __init__(self) -> None:
        self.layers = ["l0", "l1", "l2"]


class _CausalLM:
    """model.model.layers — the Llama/Qwen/OLMo shape."""

    def __init__(self, **cfg: Any) -> None:
        self.model = _Stack()
        self.config = SimpleNamespace(**cfg)


class _Wrapper:
    """model.model.language_model.model.layers — a multimodal wrapper."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(language_model=SimpleNamespace(model=_Stack()))
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=24))


def test_decoder_layers_and_stack_resolve_through_wrappers() -> None:
    plain = _CausalLM(hidden_size=8)
    assert decoder_layers(plain) == ["l0", "l1", "l2"]
    assert decoder_stack(plain) is plain.model
    wrapped = _Wrapper()
    assert decoder_layers(wrapped) == ["l0", "l1", "l2"]
    assert decoder_stack(wrapped) is wrapped.model.language_model.model
    with pytest.raises(AttributeError, match="decoder layers"):
        decoder_layers(SimpleNamespace())
    with pytest.raises(AttributeError, match="decoder stack"):
        decoder_stack(SimpleNamespace())


def test_model_hidden_dim_reads_nested_configs() -> None:
    assert model_hidden_dim(_CausalLM(hidden_size=8)) == 8
    assert model_hidden_dim(_Wrapper()) == 24
    with pytest.raises(AttributeError, match="hidden_size"):
        model_hidden_dim(_CausalLM())


def test_dtypes_table() -> None:
    from pleroma.model.load import DTYPES

    assert DTYPES == {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}


def test_resolve_pad_token_id_prefers_the_config() -> None:
    from pleroma.model.load import resolve_pad_token_id

    assert resolve_pad_token_id(_CausalLM(pad_token_id=0, eos_token_id=1),
                                "/does/not/exist") == 0
    # no pad -> eos; a list eos -> its first element
    assert resolve_pad_token_id(_CausalLM(pad_token_id=None, eos_token_id=[7, 9]),
                                "/does/not/exist") == 7


def test_resolve_pad_token_id_falls_back_to_zero_with_a_warning(
        caplog: pytest.LogCaptureFixture, tmp_path: Any) -> None:
    from pleroma.model.load import resolve_pad_token_id

    with caplog.at_level("WARNING", logger="pleroma.model"):
        assert resolve_pad_token_id(_CausalLM(), str(tmp_path / "no-tokenizer")) == 0
    assert "padding with 0" in caplog.text


def test_load_frozen_model_freezes_and_disables_the_cache(tmp_path: Any) -> None:
    transformers = pytest.importorskip("transformers")
    from pleroma.model.load import load_frozen_model

    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2,
                                   num_key_value_heads=1, max_position_embeddings=32)
    transformers.LlamaForCausalLM(cfg).save_pretrained(tmp_path / "m")
    model = load_frozen_model(str(tmp_path / "m"), torch.float32, "cpu")
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.config.use_cache
    assert len(decoder_layers(model)) == 2 and model_hidden_dim(model) == 16
