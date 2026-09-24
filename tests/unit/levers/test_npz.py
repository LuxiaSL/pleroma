"""The levers-npz loader (`pleroma/levers/npz.py`) and
`pleroma.levers.ruler.site_norms`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.levers.npz import REQUIRED_KEYS, LeverNpz, group_label, load_levers
from pleroma.levers.ruler import site_norms

G, S, D = 2, 3, 5


def _write(path: Path, **override: np.ndarray) -> Path:
    rng = np.random.default_rng(1)
    arrays: dict[str, np.ndarray] = {
        "levers": rng.standard_normal((G, S, D)).astype(np.float32),
        "cluster_sums": rng.standard_normal((G, 2, S, D)),
        "cluster_sizes": np.full((G, 2), 3, dtype=np.int64),
        "lever_norms": np.ones((G, S), dtype=np.float32),
        "group_prompt_ids": np.asarray(["if5", "mf1"]),
        "group_waves": np.asarray(["orig", "orig"]),
        "group_prompt_classes": np.asarray(["a", "b"]),
        "group_corpus_keys": np.asarray(["c0", "c1"]),
        "member_keys": np.asarray(["k0", "k1"]),
        "member_clusters": np.asarray([0, 1]),
        "member_group_index": np.asarray([0, 1]),
        "member_prompt_ids": np.asarray(["if5", "mf1"]),
        "member_prompt_classes": np.asarray(["a", "b"]),
        "member_seed_idxs": np.asarray([0, 1]),
        "member_waves": np.asarray(["orig", "orig"]),
        "sites": np.asarray([4, 8, 12]),
        "meta_json": np.asarray(json.dumps({"stage": "test"})),
    }
    arrays.update(override)
    np.savez(path, **arrays)
    return path


def test_group_label_is_the_one_spelling() -> None:
    assert group_label("if3", "orig") == "if3|orig"
    assert group_label("if3", "repl") != group_label("if3", "orig")


def test_load_levers_round_trips_with_typed_fields(tmp_path: Path) -> None:
    bank = load_levers(_write(tmp_path / "l.npz"))
    assert isinstance(bank, LeverNpz)
    assert bank.n_groups == G and bank.sites == [4, 8, 12]
    assert bank.levers.dtype == np.float32 and bank.cluster_sums.dtype == np.float64
    assert bank.cluster_sizes.dtype == np.int64
    assert bank.group_prompt_ids == ["if5", "mf1"] and bank.meta == {"stage": "test"}
    assert all(isinstance(x, str) for x in bank.member_keys)


def test_load_levers_refuses_missing_keys_and_bad_shapes(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    arrays = dict(np.load(_write(tmp_path / "ok.npz")))
    arrays.pop("member_waves")
    np.savez(p, **arrays)
    with pytest.raises(KeyError, match="member_waves"):
        load_levers(p)
    with pytest.raises(ValueError, match="levers has shape"):
        load_levers(_write(tmp_path / "s.npz", sites=np.asarray([4, 8])))
    with pytest.raises(ValueError, match="cluster_sums"):
        load_levers(_write(tmp_path / "c.npz", cluster_sums=np.zeros((G, 2, S, D + 1))))
    with pytest.raises(ValueError, match="cluster_sizes"):
        load_levers(_write(tmp_path / "z.npz", cluster_sizes=np.zeros((G, 3), np.int64)))
    assert "sites" in REQUIRED_KEYS and "meta_json" not in REQUIRED_KEYS


def test_site_norms_are_per_row_l2_in_float64() -> None:
    v = np.asarray([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert site_norms(v) == [5.0, 0.0, 1.0]
    assert all(type(x) is float for x in site_norms(v))
    with pytest.raises(ValueError, match="n_sites, hidden_dim"):
        site_norms(v[0])
