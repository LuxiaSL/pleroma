"""The lever bank build_levers writes — the schema its downstream consumers read.

The lever file is read by fit_loom_map (targets + norm_ref ruler), v1a_fit
(fan groups, the v0-CV fairness baseline) and v1a_export (fans, under either
``--fan-source``). The failure that matters here is the dropped fan: every
member of a group 2-means cannot split into two clusters of >= --min-cluster
gets ``member_group_index = -1`` — yet stays in the member arrays, which is
exactly what lets ``--fan-source member_fan`` recover it. Both halves are
pinned on a real ``pleroma.map.build.levers.main`` run over a planted run dir.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.contract.build import targets as T

SIG_DIM = 6
SITES = [8, 15]
HID = 4
RUN = "toy_replay"                   # corpus_key -> "toy"

# (prompt_id, seed base, per-member side list): side 0 -> +e0, side 1 -> +e1
PLAN = [
    ("p1", 0, [0, 0, 0, 0, 1, 1, 1, 1]),        # [4,4]: a lever
    ("p2", 0, [0, 0, 0, 0, 0, 0, 0, 1]),        # [7,1]: a lone outlier, no lever
    ("p3", 1000, [0, 0, 0, 1, 1, 1]),           # [3,3] in the repl wave
    ("p4", 0, [0, 0, 1, 1]),                    # [2,2]: a lever only at --min-cluster 2
]

CONSUMED = {
    # the wide map's lever join
    "member_corpus_keys", "member_generation_ids", "member_group_index",
    "member_clusters", "levers", "sites",
    # the export's join_bank / member_fan_index
    "member_prompt_ids", "member_waves",
}


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("levers")
    rng = np.random.default_rng(11)
    run_dir = root / RUN
    (run_dir / "signatures").mkdir(parents=True)
    names = [f"f{i}" for i in range(SIG_DIM)]
    mu = rng.standard_normal((len(SITES), HID)).astype(np.float32)
    u = rng.standard_normal((len(SITES), HID)).astype(np.float32)
    cells, means, gids, pids, seeds = [], [], [], [], []
    gid = 0
    for pid, base, sides in PLAN:
        for k, side in enumerate(sides):
            sig = np.zeros(SIG_DIM, dtype=np.float32)
            sig[side] = 1.0
            sig += rng.standard_normal(SIG_DIM).astype(np.float32) * 0.01
            np.savez(run_dir / "signatures" / f"gen_{gid:03d}.npz", features=sig,
                     feature_names=names)
            cells.append({"generation_id": gid, "prompt_id": pid,
                          "prompt_class": "decision_continuation",
                          "seed_idx": base + k, "prompt_length": 5})
            sign = 1.0 if side == 0 else -1.0
            means.append(mu + sign * u + rng.standard_normal(mu.shape).astype(np.float32) * 1e-3)
            gids.append(gid)
            pids.append(pid)
            seeds.append(base + k)
            gid += 1
    (run_dir / "cells.json").write_text(json.dumps(cells))
    hid = root / "mean_hiddens.npz"
    np.savez(hid, means=np.stack(means), generation_ids=np.asarray(gids, dtype=np.int64),
             prompt_ids=np.asarray(pids, dtype=np.str_),
             prompt_classes=np.asarray(["decision_continuation"] * gid, dtype=np.str_),
             seed_idxs=np.asarray(seeds, dtype=np.int64),
             corpus_keys=np.asarray(["toy"] * gid, dtype=np.str_),
             source_dir_labels=np.asarray([RUN] * gid, dtype=np.str_),
             sites=np.asarray(SITES, dtype=np.int64))
    outs = {}
    for label, extra in (("mc2", ["--min-cluster", "2"]), ("default", [])):
        out = root / label / "levers.npz"
        rc = T.run_build_levers(["--hiddens", hid, "--run-dirs", run_dir, "--out", out, *extra])
        assert rc == 0
        with np.load(out, allow_pickle=True) as npz:
            outs[label] = {k: npz[k] for k in npz.files}
    return {"outs": outs, "means": np.stack(means), "root": root}


def test_schema_keys_dtypes_shapes(planted) -> None:
    """The full key set the lever bank writes, with dtypes; every key a
    downstream stage reads is present (CONSUMED)."""
    a = planted["outs"]["mc2"]
    assert set(a) == {
        "levers", "lever_norms", "cluster_sums", "cluster_sizes", "group_prompt_ids",
        "group_waves", "group_prompt_classes", "group_corpus_keys", "member_keys",
        "member_generation_ids", "member_corpus_keys", "member_prompt_ids",
        "member_prompt_classes", "member_seed_idxs", "member_waves", "member_clusters",
        "member_group_index", "sites", "meta_json"}
    assert CONSUMED <= set(a)
    g, m = 3, sum(len(s) for _, _, s in PLAN)      # 3 lever groups, 26 members
    assert a["levers"].shape == (g, len(SITES), HID) and a["levers"].dtype == np.float32
    assert a["lever_norms"].shape == (g, len(SITES)) and a["lever_norms"].dtype == np.float32
    assert a["cluster_sums"].shape == (g, 2, len(SITES), HID)
    assert a["cluster_sums"].dtype == np.float64
    assert a["cluster_sizes"].dtype == np.int64
    for k in ("member_generation_ids", "member_seed_idxs", "member_clusters",
              "member_group_index", "sites"):
        assert a[k].dtype == np.int64, k
    for k in ("member_keys", "member_corpus_keys", "member_prompt_ids", "member_waves"):
        assert a[k].dtype.kind == "U" and a[k].shape == (m,), k
    assert a["sites"].tolist() == SITES
    assert a["member_keys"][0] == "toy#0000"          # gen_key: corpus, '#', zero-padded 4-digit id


def test_no_lever_group_members_are_kept_with_group_minus_one(planted) -> None:
    """The [7,1] fan gets NO lever even at --min-cluster 2 (its small side
    is 1), and its 8 members are still written, all with group -1. Lever groups
    are numbered 0.. in emission order; members are emitted group by group."""
    a = planted["outs"]["mc2"]
    pid = a["member_prompt_ids"]
    grp = a["member_group_index"]
    assert set(grp[pid == "p2"].tolist()) == {-1}
    assert (pid == "p2").sum() == 8
    assert grp[pid == "p1"].tolist() == [0] * 8
    assert grp[pid == "p3"].tolist() == [1] * 6
    assert grp[pid == "p4"].tolist() == [2] * 4
    assert a["group_prompt_ids"].tolist() == ["p1", "p3", "p4"]
    assert a["group_waves"].tolist() == ["orig", "repl", "orig"]
    assert set(a["member_waves"][pid == "p3"].tolist()) == {"repl"}
    assert set(a["member_corpus_keys"].tolist()) == {"toy"}


def test_lever_is_mean_cluster0_minus_mean_cluster1(planted) -> None:
    """lever[g] = mean_h(cluster 0) - mean_h(cluster 1) over the group's members
    — the target fit_loom_map regresses (times m_sign)."""
    a = planted["outs"]["mc2"]
    means = planted["means"]
    gid = a["member_generation_ids"]
    for g in range(a["levers"].shape[0]):
        sel = a["member_group_index"] == g
        cl = a["member_clusters"][sel]
        h = means[gid[sel]].astype(np.float64)
        want = h[cl == 0].mean(axis=0) - h[cl == 1].mean(axis=0)
        assert np.allclose(a["levers"][g], want, atol=1e-6)
        assert np.allclose(a["lever_norms"][g], np.linalg.norm(want, axis=1), atol=1e-5)


def test_default_min_cluster_is_3_and_evicts_the_2_2_group_as_is(planted) -> None:
    """Pinned AS-IS: the default ``--min-cluster`` is 3, which silently drops
    every [2,2] group (callers pass 2 to keep them). At 3 the [2,2] group loses its lever and its members join the -1
    pool (so a lever_group join drops them too); the meta records the value
    used, which is the only trace a consumer has of it."""
    assert T.BUILD_LEVERS_DEFAULT_MIN_CLUSTER == 3
    d = planted["outs"]["default"]
    assert json.loads(str(d["meta_json"]))["min_cluster"] == 3
    assert json.loads(str(planted["outs"]["mc2"]["meta_json"]))["min_cluster"] == 2
    assert d["levers"].shape[0] == 2
    assert set(d["member_group_index"][d["member_prompt_ids"] == "p4"].tolist()) == {-1}


def test_member_fan_index_accepts_real_build_levers_output(planted) -> None:
    """The (corpus, prompt_id, wave) fan key v1a_export --fan-source member_fan
    uses EXTENDS build_levers' own (prompt_id, wave) groups on a real lever file:
    4 fans, the no-lever [7,1] fan included."""
    a = planted["outs"]["mc2"]
    fan = T.member_fan_index(a["member_corpus_keys"].tolist(),
                             a["member_prompt_ids"].tolist(),
                             a["member_waves"].tolist(), a["member_group_index"])
    assert sorted(np.bincount(fan).tolist()) == [4, 6, 8, 8]


def test_sidecar_meta_json_matches_embedded(planted) -> None:
    """build_levers writes ``<stem>_meta.json`` beside the npz with the same meta."""
    side = json.loads((planted["root"] / "mc2" / "levers_meta.json").read_text())
    emb = json.loads(str(planted["outs"]["mc2"]["meta_json"]))
    assert side == emb
    assert emb["n_groups"] == 4 and emb["n_groups_with_lever"] == 3
    assert [g["group_index"] for g in emb["groups"]] == [0, -1, 1, 2]
