"""B1.5 L1b: the levers are basin_check's clusters, and the held-out algebra is exact.

Four properties, each the thing that would silently ruin the L1 readout if it
were wrong:

  * a PLANTED two-cluster geometry comes back out as the lever (if the contrast
    construction cannot recover a direction it was handed, nothing downstream
    means anything);
  * ``loo_lever`` equals a literal recompute over the remaining members — the
    held-out lever is the one property ``eval_levers`` cannot check for itself;
  * the cluster-size guard actually withholds a lever from a group too thin to
    survive leave-one-out;
  * the clustering is ``basin_check``'s, verified both against
    ``two_means_split`` directly and against a published ``<run>/basin_report.json``
    (including its label-flip freedom, and including a real disagreement, which
    must abort).

numpy + stdlib only: no torch, no anamnesis, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.map.build.hiddens import corpus_key
from pleroma.map.build.basin import load_signatures, two_means_split
from pleroma.map.build.levers import (
    DEFAULT_WAVE_BOUNDARY,
    build_groups,
    cluster_run_dir,
    gen_key,
    load_hidden_banks,
    loo_lever,
    matched_sign,
    parse_basin_report_arg,
    verify_against_basin_report,
    wave_of,
)

SIG_DIM = 6
N_SITES = 2
HIDDEN_DIM = 4


def _plant(
    tmp_path: Path,
    run_name: str = "toy_replay",
    prompts: tuple[str, ...] = ("p1", "p2"),
    per_cluster: int = 4,
    seed_base: int = 0,
    hidden_scale: float = 1.0,
) -> dict[str, object]:
    """A tiny stage-2 run dir + matching ``<run>/mean_hiddens.npz`` with a planted geometry.

    Cluster A's signatures point along +e0, cluster B's along +e1 (cosine 2-means
    separates those trivially), and the hidden means are ``mu +/- u`` so the true
    lever is ``2u`` up to the label order 2-means happens to choose.
    """
    rng = np.random.default_rng(11)
    run_dir = tmp_path / run_name
    sig_dir = run_dir / "signatures"
    sig_dir.mkdir(parents=True)
    names = [f"f{i}" for i in range(SIG_DIM)]

    mu = rng.standard_normal((N_SITES, HIDDEN_DIM)).astype(np.float32)
    u = rng.standard_normal((N_SITES, HIDDEN_DIM)).astype(np.float32) * hidden_scale

    cells: list[dict[str, object]] = []
    means: list[np.ndarray] = []
    gen_ids: list[int] = []
    pids: list[str] = []
    classes: list[str] = []
    seeds: list[int] = []
    intended: dict[str, int] = {}

    gid = 0
    for pid in prompts:
        for side in (0, 1):
            for k in range(per_cluster):
                sig = np.zeros(SIG_DIM, dtype=np.float32)
                sig[side] = 1.0
                sig += rng.standard_normal(SIG_DIM).astype(np.float32) * 0.01
                np.savez(sig_dir / f"gen_{gid:03d}.npz", features=sig, feature_names=names)
                cells.append(
                    {
                        "generation_id": gid,
                        "prompt_id": pid,
                        "prompt_class": "decision_continuation",
                        "seed_idx": seed_base + len(cells) % (2 * per_cluster),
                        "prompt_length": 5,
                    }
                )
                sign = 1.0 if side == 0 else -1.0
                means.append(
                    mu + sign * u + rng.standard_normal((N_SITES, HIDDEN_DIM)).astype(np.float32) * 0.001
                )
                gen_ids.append(gid)
                pids.append(pid)
                classes.append("decision_continuation")
                seeds.append(seed_base + (len(cells) - 1) % (2 * per_cluster))
                intended[gen_key(corpus_key(run_name), gid)] = side
                gid += 1
    (run_dir / "cells.json").write_text(json.dumps(cells))

    hiddens = tmp_path / f"{run_name}_hiddens.npz"
    corpus = corpus_key(run_name)
    np.savez(
        hiddens,
        means=np.stack(means).astype(np.float32),
        generation_ids=np.asarray(gen_ids, dtype=np.int64),
        prompt_ids=np.asarray(pids, dtype=np.str_),
        prompt_classes=np.asarray(classes, dtype=np.str_),
        seed_idxs=np.asarray(seeds, dtype=np.int64),
        corpus_keys=np.asarray([corpus] * len(gen_ids), dtype=np.str_),
        source_dir_labels=np.asarray([run_name] * len(gen_ids), dtype=np.str_),
        sites=np.asarray([8, 15], dtype=np.int64),
    )
    return {"run_dir": run_dir, "hiddens": hiddens, "u": u, "intended": intended}


# ── the planted lever ──────────────────────────────────────────────────────────


def test_lever_recovers_the_planted_direction(tmp_path: Path) -> None:
    world = _plant(tmp_path)
    bank = load_hidden_banks([world["hiddens"]])
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    groups = build_groups(clustered, bank, min_cluster=3)

    assert len(groups) == 2
    u = np.asarray(world["u"])
    for group in groups:
        assert group.lever is not None, group.reason
        assert group.cluster_sizes == (4, 4)
        for site in range(N_SITES):
            planted = 2.0 * u[site]
            got = group.lever[site]
            cosine = float(
                got @ planted / (np.linalg.norm(got) * np.linalg.norm(planted))
            )
            # |cos| because 2-means labels carry no order: the lever may come out
            # as A-B or B-A, and both are the same contrast.
            assert abs(cosine) > 0.99, (site, cosine)
            assert np.isclose(
                np.linalg.norm(got), np.linalg.norm(planted), rtol=0.05
            )


def test_cluster_assignment_is_two_means_split_itself(tmp_path: Path) -> None:
    """The levers must use basin_check's partition, not one of our own."""
    world = _plant(tmp_path)
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    ours = {g.key: g.cluster for g in clustered}
    intended = world["intended"]
    # Same partition up to a global label flip, per prompt.
    for pid in ("p1", "p2"):
        keys = [g.key for g in clustered if g.prompt_id == pid]
        mine = [ours[k] for k in keys]
        theirs = [intended[k] for k in keys]
        assert mine == theirs or mine == [1 - c for c in theirs]
    # And the labels ARE two_means_split's output on basin_check's own rows, in
    # basin_check's own order — not merely a partition that happens to agree.
    sigs = load_signatures(world["run_dir"])
    prompt_ids = [c.prompt_id for c in sigs.cells]
    pid_array = np.array(prompt_ids, dtype=object)
    for pid in ("p1", "p2"):
        members = np.nonzero(pid_array == pid)[0]
        expected, _ = two_means_split(sigs.features[members])
        got = [ours[gen_key("toy", int(sigs.gen_ids[i]))] for i in members]
        assert got == expected


# ── leave-one-out ──────────────────────────────────────────────────────────────


def test_loo_lever_equals_a_literal_recompute(tmp_path: Path) -> None:
    world = _plant(tmp_path)
    bank = load_hidden_banks([world["hiddens"]])
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    groups = build_groups(clustered, bank, min_cluster=3)
    rows = bank.row_index()

    for group in groups:
        sums = group.cluster_sums
        sizes = group.cluster_sizes
        for key, cluster in zip(group.member_keys, group.clusters):
            held_out = loo_lever(sums, sizes, cluster, bank.means[rows[key]])
            # The literal version: drop the member, average what is left.
            kept: dict[int, list[np.ndarray]] = {0: [], 1: []}
            for other, other_cluster in zip(group.member_keys, group.clusters):
                if other == key:
                    continue
                kept[other_cluster].append(bank.means[rows[other]].astype(np.float64))
            literal = (
                np.mean(np.stack(kept[0]), axis=0) - np.mean(np.stack(kept[1]), axis=0)
            )
            assert np.allclose(held_out, literal, rtol=1e-5, atol=1e-6)


def test_loo_lever_actually_removes_the_member(tmp_path: Path) -> None:
    """The point of the algebra: the held-out lever is NOT the full-set lever."""
    world = _plant(tmp_path)
    bank = load_hidden_banks([world["hiddens"]])
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    group = build_groups(clustered, bank, min_cluster=3)[0]
    rows = bank.row_index()
    key, cluster = group.member_keys[0], group.clusters[0]
    held_out = loo_lever(group.cluster_sums, group.cluster_sizes, cluster, bank.means[rows[key]])
    assert not np.allclose(held_out, group.lever)


def test_loo_lever_refuses_to_empty_a_cluster() -> None:
    sums = np.ones((2, N_SITES, HIDDEN_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="empties a cluster"):
        loo_lever(sums, (1, 4), 0, np.ones((N_SITES, HIDDEN_DIM), dtype=np.float32))


def test_loo_lever_validates_its_shapes() -> None:
    sums = np.ones((2, N_SITES, HIDDEN_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="own_cluster"):
        loo_lever(sums, (4, 4), 2, np.ones((N_SITES, HIDDEN_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="own_mean has shape"):
        loo_lever(sums, (4, 4), 0, np.ones((N_SITES + 1, HIDDEN_DIM), dtype=np.float32))


def test_matched_sign_is_toward_the_members_own_basin() -> None:
    assert matched_sign(0) == 1.0
    assert matched_sign(1) == -1.0
    with pytest.raises(ValueError):
        matched_sign(2)


# ── guards ─────────────────────────────────────────────────────────────────────


def test_a_cluster_below_min_cluster_gets_no_lever(tmp_path: Path) -> None:
    world = _plant(tmp_path, per_cluster=3)
    bank = load_hidden_banks([world["hiddens"]])
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)

    ok = build_groups(clustered, bank, min_cluster=3)
    assert all(g.lever is not None for g in ok)

    withheld = build_groups(clustered, bank, min_cluster=4)
    assert all(g.lever is None for g in withheld)
    assert all("min-cluster" in g.reason for g in withheld)


def test_a_group_spanning_two_corpora_is_refused(tmp_path: Path) -> None:
    """basin_check clustered per run dir; a group across two is not its partition."""
    a = _plant(tmp_path / "a", run_name="one_replay", prompts=("shared",))
    b = _plant(tmp_path / "b", run_name="two_replay", prompts=("shared",))
    bank = load_hidden_banks([a["hiddens"], b["hiddens"]])
    clustered = cluster_run_dir(a["run_dir"], DEFAULT_WAVE_BOUNDARY) + cluster_run_dir(
        b["run_dir"], DEFAULT_WAVE_BOUNDARY
    )
    with pytest.raises(ValueError, match="draws from corpora"):
        build_groups(clustered, bank, min_cluster=3)


def test_waves_separate_the_two_generation_sweeps(tmp_path: Path) -> None:
    assert wave_of(0) == "orig"
    assert wave_of(15) == "orig"
    assert wave_of(16) == "repl"
    assert wave_of(31) == "repl"
    assert wave_of(8, boundary=8) == "repl"

    # A run dir whose seeds straddle the boundary means the boundary is wrong.
    world = _plant(tmp_path, per_cluster=4, seed_base=14)
    with pytest.raises(ValueError, match="span waves"):
        cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)


def test_hidden_banks_refuse_disagreeing_sites(tmp_path: Path) -> None:
    world = _plant(tmp_path)
    other = tmp_path / "other.npz"
    with np.load(world["hiddens"]) as npz:
        fields = {k: npz[k] for k in npz.files}
    fields["sites"] = np.asarray([8, 16], dtype=np.int64)
    np.savez(other, **fields)
    with pytest.raises(ValueError, match="do not live in one space"):
        load_hidden_banks([world["hiddens"], other])


# ── the basin-report cross-check ───────────────────────────────────────────────


def _report(tmp_path: Path, run_name: str, clustered: list, flip: bool, corrupt: bool) -> Path:
    prompts: dict[str, list[dict[str, object]]] = {}
    for gen in clustered:
        cluster = 1 - gen.cluster if flip else gen.cluster
        prompts.setdefault(gen.prompt_id, []).append(
            {"generation_id": gen.generation_id, "cluster": cluster}
        )
    if corrupt:
        first = next(iter(prompts))
        prompts[first][0]["cluster"] = 1 - int(prompts[first][0]["cluster"])
    path = tmp_path / f"basin_report_{run_name}_{int(flip)}_{int(corrupt)}.json"
    path.write_text(
        json.dumps(
            {
                "run_dir": f"/somewhere/else/{run_name}",
                "prompts": [
                    {"prompt_id": pid, "seeds": seeds} for pid, seeds in prompts.items()
                ],
            }
        )
    )
    return path


@pytest.mark.parametrize("flip", [False, True])
def test_basin_report_agreement_is_label_order_free(tmp_path: Path, flip: bool) -> None:
    world = _plant(tmp_path)
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    by_key = {g.key: g for g in clustered}
    path = _report(tmp_path, "toy_replay", clustered, flip=flip, corrupt=False)
    result = verify_against_basin_report(path, by_key)
    assert result["n_prompts_verified"] == 2


def test_a_real_disagreement_with_the_basin_report_aborts(tmp_path: Path) -> None:
    world = _plant(tmp_path)
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    by_key = {g.key: g for g in clustered}
    path = _report(tmp_path, "toy_replay", clustered, flip=False, corrupt=True)
    with pytest.raises(ValueError, match="neither the report's nor its complement"):
        verify_against_basin_report(path, by_key)


def test_a_report_that_matches_nothing_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """A cross-check that checked zero prompts reads like a pass in the log."""
    world = _plant(tmp_path)
    clustered = cluster_run_dir(world["run_dir"], DEFAULT_WAVE_BOUNDARY)
    by_key = {g.key: g for g in clustered}
    path = _report(tmp_path, "toy_replay", clustered, flip=False, corrupt=False)
    blob = json.loads(path.read_text())
    blob["run_dir"] = "outputs/expB0_somewhere_else"   # the local-mirror naming trap
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="matched NO prompt"):
        verify_against_basin_report(path, by_key)
    # ... and the explicit corpus form rescues exactly that case.
    assert verify_against_basin_report(path, by_key, corpus="toy")["n_prompts_verified"] == 2


def test_parse_basin_report_arg_accepts_both_forms() -> None:
    assert parse_basin_report_arg("outputs/x/basin_report.json") == (
        None, Path("outputs/x/basin_report.json")
    )
    assert parse_basin_report_arg("pilot=outputs/x/basin_report.json") == (
        "pilot", Path("outputs/x/basin_report.json")
    )
    with pytest.raises(ValueError, match="corpus=path"):
        parse_basin_report_arg("=outputs/x.json")


def test_corpus_key_joins_gen_and_replay_dirs() -> None:
    assert corpus_key("pilot_gen") == corpus_key("pilot_replay") == "pilot"
    assert corpus_key("wave2_gen") == corpus_key("wave2_replay") == "wave2"
    assert corpus_key("repl_gen_a") == corpus_key("repl_replay_a") == "repl_a"
    assert corpus_key("repl_gen_b") == corpus_key("repl_replay_b") == "repl_b"
    assert corpus_key("repl_gen_a") != corpus_key("repl_gen_b")
    assert corpus_key("/abs/path/to/pilot_gen/") == "pilot"
    # A name made only of role tokens keeps itself rather than collapsing to "".
    assert corpus_key("gen") == "gen"
