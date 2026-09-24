"""The lever bank `pleroma.map.build.levers` writes, read whole and validated (the dose-ladder view).

:class:`LeverNpz` is the raw record: every array the file carries, with no map
and no ruler. :class:`pleroma.levers.bank.LeverBank` is the loom's *wearable*
view of the same file (ruler-matched to a live map); keep the two apart, since
a lever read here has no per-site meaning for alpha until a ruler is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

#: Arrays every levers npz must carry.
REQUIRED_KEYS: tuple[str, ...] = (
    "levers", "cluster_sums", "cluster_sizes", "lever_norms", "group_prompt_ids",
    "group_waves", "group_prompt_classes", "group_corpus_keys", "member_keys",
    "member_clusters", "member_group_index", "member_prompt_ids",
    "member_prompt_classes", "member_seed_idxs", "member_waves", "sites",
)


def group_label(prompt_id: str, wave: str) -> str:
    """``prompt|wave`` — the group key used in every file the lever stages touch."""
    return f"{prompt_id}|{wave}"


@dataclass(frozen=True)
class LeverNpz:
    """A lever bank (`pleroma.map.build.levers`), as the dose ladders need it."""

    levers: F32                      # [G, n_sites, d]
    cluster_sums: F64                # [G, 2, n_sites, d] — float64, for exact LOO
    cluster_sizes: NDArray[np.int64]  # [G, 2]
    lever_norms: F32                 # [G, n_sites]
    group_prompt_ids: list[str]
    group_waves: list[str]
    group_prompt_classes: list[str]
    group_corpus_keys: list[str]
    member_keys: list[str]
    member_clusters: NDArray[np.int64]
    member_group_index: NDArray[np.int64]
    member_prompt_ids: list[str]
    member_prompt_classes: list[str]
    member_seed_idxs: NDArray[np.int64]
    member_waves: list[str]
    sites: list[int]
    meta: dict[str, Any]

    @property
    def n_groups(self) -> int:
        return int(self.levers.shape[0])


def load_levers(path: Path) -> LeverNpz:
    """Load a levers npz, validating the shapes every consumer depends on."""
    p = Path(path).expanduser()
    with np.load(p, allow_pickle=False) as npz:
        missing = [k for k in REQUIRED_KEYS if k not in npz]
        if missing:
            raise KeyError(f"{p}: missing {missing} (has {list(npz.files)})")
        levers = np.asarray(npz["levers"], dtype=np.float32)
        sums = np.asarray(npz["cluster_sums"], dtype=np.float64)
        sizes = np.asarray(npz["cluster_sizes"], dtype=np.int64)
        sites = [int(x) for x in npz["sites"]]
        meta_raw = str(npz["meta_json"]) if "meta_json" in npz else "{}"
        bank = LeverNpz(
            levers=levers,
            cluster_sums=sums,
            cluster_sizes=sizes,
            lever_norms=np.asarray(npz["lever_norms"], dtype=np.float32),
            group_prompt_ids=[str(x) for x in npz["group_prompt_ids"]],
            group_waves=[str(x) for x in npz["group_waves"]],
            group_prompt_classes=[str(x) for x in npz["group_prompt_classes"]],
            group_corpus_keys=[str(x) for x in npz["group_corpus_keys"]],
            member_keys=[str(x) for x in npz["member_keys"]],
            member_clusters=np.asarray(npz["member_clusters"], dtype=np.int64),
            member_group_index=np.asarray(npz["member_group_index"], dtype=np.int64),
            member_prompt_ids=[str(x) for x in npz["member_prompt_ids"]],
            member_prompt_classes=[str(x) for x in npz["member_prompt_classes"]],
            member_seed_idxs=np.asarray(npz["member_seed_idxs"], dtype=np.int64),
            member_waves=[str(x) for x in npz["member_waves"]],
            sites=sites,
            meta=json.loads(meta_raw) if meta_raw else {},
        )
    if levers.ndim != 3 or levers.shape[1] != len(sites):
        raise ValueError(f"{p}: levers has shape {levers.shape}, expected (G, {len(sites)}, d)")
    if sums.shape != (levers.shape[0], 2, levers.shape[1], levers.shape[2]):
        raise ValueError(f"{p}: cluster_sums has shape {sums.shape}, expected (G, 2, S, d)")
    if sizes.shape != (levers.shape[0], 2):
        raise ValueError(f"{p}: cluster_sizes has shape {sizes.shape}, expected (G, 2)")
    if bank.n_groups == 0:
        raise ValueError(f"{p} holds no levers")
    return bank
