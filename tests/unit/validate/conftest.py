"""Synthetic fixtures for the validate battery: toy profiles that match the
contract's tiny map (3 sites x hidden 16, rank 4), and small artifact writers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pleroma.config import ModelProfile
from tests.contract.map import _fixtures as mapfx

MODELC_STOPS = ["\n\n**User:**", "\n\n**User**", "\n\n**Model C:**", "\nAs follows is",
                "\n\n---"]

TOY: dict[str, Any] = {
    "name": "toy-modelc",
    "model": {
        "model_id": "/models/toy", "preset_name": "toy",
        "arch": {"num_layers": 16, "hidden_dim": mapfx.HIDDEN, "num_attention_heads": 2,
                 "num_kv_heads": 1, "head_dim": mapfx.HIDDEN // 2,
                 "eos_token_ids": [2], "pad_token_id": 0},
        "layers": {"sampled": [0, 8, 15], "pca": [8], "early_cutoff": 4, "late_cutoff": 12},
    },
    "format": {"mode": "modelc", "header": "stranger", "stops": MODELC_STOPS},
    "sampling": {"temperature": 1.0, "top_p": 0.98},
    "lengths": {"future_tokens": 64, "reply_tokens": 64},
    "map": {"sites": list(mapfx.SITES), "lam": 1e4, "rank": mapfx.RANK,
            "bins": {"layers": [1, 2, 3]}},
    "steer": {"injection_span": "uniform"},
}


def make_profile(**overrides: Any) -> ModelProfile:
    """TOY with dotted-path overrides: make_profile(**{"format.mode": "chat"})."""
    blob = copy.deepcopy(TOY)
    for dotted, value in overrides.items():
        box = blob
        *head, last = dotted.split(".")
        for k in head:
            box = box.setdefault(k, {})
        if value is None:
            box.pop(last, None)
        else:
            box[last] = value
    return ModelProfile.model_validate(blob)


@pytest.fixture
def modelc_profile() -> ModelProfile:
    return make_profile()


@pytest.fixture
def chat_profile() -> ModelProfile:
    return make_profile(**{"format.mode": "chat", "format.header": None, "format.stops": [],
                           "model.arch.pad_token_id": None})


def write_json(path: Path, blob: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob))
    return path


@pytest.fixture
def toy_map(tmp_path: Path) -> tuple[Path, Path]:
    """(map npz, discriminants npz) in the served v1a on-disk schema."""
    disc, sha = mapfx.write_discriminants(tmp_path / "disc")
    return mapfx.write_v1a_map(tmp_path / "map", sha), disc
