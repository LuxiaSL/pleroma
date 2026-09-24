"""Tests for the parallel loom harvest (3-4x faster than serial; see
docs/FINDINGS.md section 13).

The risk this change carries is not "it might be slow". It is that a
parallelism bug could PERTURB FEATURES — quietly, by a little, on some spans —
and every downstream score (the map's lever inference, the probe's gauge, every
banked z-vector) would inherit the perturbation with nothing to flag it. A
harvest that is 3x faster and 0.1% wrong is worse than the slow one.

So the equivalence bar here is BYTE EQUALITY, not a tolerance. A tolerance
would be the wrong instrument twice over: the parallel path is supposed to run
the identical per-record call chain on the identical inputs, so any difference
at all is a bug rather than a rounding budget to be sized; and a tolerance
loose enough to pass float noise is also loose enough to pass a real
off-by-one in the merge's row ordering, which is the failure this actually
guards.

Two layers, because they catch different things:

  * THE MERGE (laptop, always runs). Serial cells/bins/meta are reconstructed
    from per-shard fixtures and compared to the serial reference exactly. This
    is where a parallelism bug would really live — the per-span math is
    untouched code, the merge is new code, and a row-order or dtype slip in it
    is exactly what would silently misalign every bins row against its
    signature.

  * THE REAL FAN (a GPU host, skipped without ``LOOM_POOL_TEST_URLS``). A fixed loom
    dir harvested serially and then in parallel, with every
    ``signatures/gen_NNN.npz``, every ``fork_series/gen_NNN.npz`` and every
    ``<loom dir>/bins.npz`` row compared byte-for-byte. The merge test cannot prove this
    one because it stubs out the GPU; this one cannot run on a laptop. Both are
    needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pleroma.harvest import pool as harvest_pool
from pleroma.harvest.replay import shard_records


# ── the partition, which everything else rests on ────────────────────────────


def test_shard_records_is_a_partition() -> None:
    """Every gen lands in exactly one shard, for every width the pool allows."""
    records = [{"generation_id": i} for i in range(17)]
    for n in range(1, harvest_pool.MAX_POOL_WORKERS + 1):
        shards = [shard_records(records, i, n) for i in range(n)]
        flat = [r["generation_id"] for s in shards for r in s]
        assert sorted(flat) == list(range(17)), f"n={n} is not a partition"
        assert len(flat) == len(set(flat)), f"n={n} duplicated a gen"


def test_shard_records_tolerates_more_workers_than_gens() -> None:
    """k=2 on a 4-worker pool must leave two shards EMPTY, not raise.

    The empty shard still writes a well-formed result (see harvest_worker), so
    the merge sees n files and cannot mistake "no work" for "a worker died".
    """
    records = [{"generation_id": i} for i in range(2)]
    shards = [shard_records(records, i, 4) for i in range(4)]
    assert [len(s) for s in shards] == [1, 1, 0, 0]


# ── --harvest-workers parsing ────────────────────────────────────────────────


def test_parse_worker_list_happy() -> None:
    assert harvest_pool.parse_worker_list("127.0.0.1:8768, 127.0.0.1:8769") == [
        "127.0.0.1:8768", "127.0.0.1:8769",
    ]


@pytest.mark.parametrize(
    "spec,fragment",
    [
        ("", "no entries"),
        ("  ,  ", "no entries"),
        # A duplicate is REFUSED, not deduped: an operator who wrote one worker
        # twice believes they have two, and a pool that silently halves its
        # width shows up only as a speedup that "did not reproduce".
        ("127.0.0.1:8768,127.0.0.1:8768", "more than once"),
        ("http://127.0.0.1:8768", "host:port, not a URL"),
        ("127.0.0.1", "not host:port"),
        ("127.0.0.1:abc", "not host:port"),
        (",".join(f"h:{9000 + i}" for i in range(harvest_pool.MAX_POOL_WORKERS + 1)),
         "ceiling"),
    ],
)
def test_parse_worker_list_refuses(spec: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        harvest_pool.parse_worker_list(spec)


# ── fixtures: a serial harvest, and the same thing cut into shards ───────────


def _write_shards(loom_dir: Path, n: int, gids: list[int], dim: int = 6) -> dict:
    """Write n shard files partitioning `gids`, plus return the SERIAL reference.

    The reference is built the way the unsharded worker builds it — ascending
    generation_id, one row per gen — so any disagreement is the merge's fault.
    """
    rng = np.random.default_rng(1234)
    feats = {g: rng.normal(size=dim).astype(np.float32) for g in gids}
    names = np.array([f"binsB_f{i}" for i in range(dim)])
    shard_dir = loom_dir / "shard_meta"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = [{"generation_id": g} for g in sorted(gids)]
    for i in range(n):
        mine = [int(r["generation_id"]) for r in shard_records(records, i, n)]
        (shard_dir / f"cells_shard{i}of{n}.json").write_text(json.dumps(
            [{"generation_id": g, "prompt_id": f"p{g}", "num_features": dim}
             for g in mine]
        ))
        (shard_dir / f"run_meta_shard{i}of{n}.json").write_text(json.dumps({
            "stage": "harvest_worker_replay_b0", "shard": f"{i}of{n}",
            "gate0": "passed", "feature_dim": dim,
            "calibration": {"positional_means_sha256": "deadbeef"},
            "n_corpus": len(gids), "n_records": len(mine), "n_cells": len(mine),
            "failures": [], "elapsed_s": 1.0, "model_id": "m", "preset": "3b",
        }))
        np.savez_compressed(
            shard_dir / f"bins_shard{i}of{n}.npz",
            features=(np.stack([feats[g] for g in mine]) if mine
                      else np.zeros((0, dim), dtype=np.float32)),
            generation_id=np.array(mine, dtype=np.int64),
            prompt_id=np.array([f"p{g}" for g in mine]),
            feature_names=names,
            failures=np.array([]),
        )
    return {
        "features": np.stack([feats[g] for g in sorted(gids)]),
        "generation_id": np.array(sorted(gids), dtype=np.int64),
        "feature_names": names,
    }


@pytest.mark.parametrize("n", [2, 3, 4, 8])
def test_merge_reconstructs_the_serial_object_exactly(tmp_path: Path, n: int) -> None:
    """The merged bins matrix is BIT-IDENTICAL to the serial one, at every width.

    Row ORDER is the substance of this test, not a detail. The bins matrix
    (``<loom dir>/bins.npz``) carries no per-row link back to a signature file
    other than its position, so a merge
    that returned the right rows in the wrong order would hand every span
    somebody else's bins features — and nothing downstream would notice.
    """
    gids = list(range(11))
    serial = _write_shards(tmp_path, n, gids)
    merged = harvest_pool.merge_shard_bins(tmp_path, n)

    assert merged["generation_id"].tolist() == serial["generation_id"].tolist()
    assert merged["features"].dtype == serial["features"].dtype
    # Byte equality, not allclose. See the module docstring.
    assert merged["features"].tobytes() == serial["features"].tobytes()
    assert np.array_equal(merged["feature_names"], serial["feature_names"])

    cells = harvest_pool.merge_shard_cells(tmp_path, n)
    assert [c["generation_id"] for c in cells] == sorted(gids)

    meta = harvest_pool.merge_shard_run_meta(tmp_path, n)
    assert meta["n_cells"] == len(gids)
    assert meta["n_records"] == len(gids)
    assert meta["feature_dim"] == 6
    assert meta["gate0"].startswith("passed")


def test_merge_survives_an_empty_shard(tmp_path: Path) -> None:
    """4 workers, 2 gens: the two empty shards contribute no rows and no error."""
    gids = [0, 1]
    serial = _write_shards(tmp_path, 4, gids)
    merged = harvest_pool.merge_shard_bins(tmp_path, 4)
    assert merged["features"].tobytes() == serial["features"].tobytes()
    assert harvest_pool.merge_shard_run_meta(tmp_path, 4)["n_cells"] == 2


def test_merge_accepts_gate0_not_run_ONLY_on_an_empty_shard(tmp_path: Path) -> None:
    """Gate 0 cannot run on a shard with no gens — but only then may it be absent.

    An empty shard honestly reports ``not_run``; the merge must accept that and
    nothing else. The second half is the load-bearing half: a shard that DID
    extract gens and reports ``not_run`` has un-poolable vectors, and accepting
    it would defeat the gate entirely.
    """
    _write_shards(tmp_path, 2, list(range(6)))
    p = tmp_path / "shard_meta" / "run_meta_shard1of2.json"
    meta = json.loads(p.read_text())
    meta["gate0"] = "not_run (empty shard)"

    # honest: no cells -> accepted
    meta["n_cells"] = 0
    p.write_text(json.dumps(meta))
    assert harvest_pool.merge_shard_run_meta(tmp_path, 2)["gate0"].startswith("passed")

    # dishonest: it extracted gens but skipped the gate -> refused
    meta["n_cells"] = 3
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="GATE 0"):
        harvest_pool.merge_shard_run_meta(tmp_path, 2)


# ── the refusals: every one of these is a silent-corruption hole if it passes ──


def test_merge_refuses_a_gen_in_two_shards(tmp_path: Path) -> None:
    _write_shards(tmp_path, 2, list(range(6)))
    bad = tmp_path / "shard_meta" / "cells_shard1of2.json"
    rows = json.loads(bad.read_text())
    rows.append({"generation_id": 0, "prompt_id": "p0"})  # also in shard 0
    bad.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="more than one shard"):
        harvest_pool.merge_shard_cells(tmp_path, 2)


def test_merge_refuses_a_missing_shard(tmp_path: Path) -> None:
    _write_shards(tmp_path, 3, list(range(6)))
    (tmp_path / "shard_meta" / "cells_shard2of3.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        harvest_pool.merge_shard_cells(tmp_path, 3)


def test_merge_refuses_mismatched_calibration(tmp_path: Path) -> None:
    """The pooled-vs-corrected PCA trap: Gate 0 cannot see it, only the hash can."""
    _write_shards(tmp_path, 2, list(range(6)))
    p = tmp_path / "shard_meta" / "run_meta_shard1of2.json"
    meta = json.loads(p.read_text())
    meta["calibration"] = {"positional_means_sha256": "0ther"}
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different calibrations"):
        harvest_pool.merge_shard_run_meta(tmp_path, 2)


def test_merge_refuses_a_failed_gate0(tmp_path: Path) -> None:
    _write_shards(tmp_path, 2, list(range(6)))
    p = tmp_path / "shard_meta" / "run_meta_shard0of2.json"
    meta = json.loads(p.read_text())
    meta["gate0"] = "FAILED"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="GATE 0"):
        harvest_pool.merge_shard_run_meta(tmp_path, 2)


def test_merge_refuses_mismatched_bins_feature_names(tmp_path: Path) -> None:
    _write_shards(tmp_path, 2, list(range(6)))
    p = tmp_path / "shard_meta" / "bins_shard1of2.npz"
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["feature_names"] = np.array([f"OTHER_{i}" for i in range(6)])
    np.savez_compressed(p, **cols)
    with pytest.raises(ValueError, match="not in one feature space"):
        harvest_pool.merge_shard_bins(tmp_path, 2)


# ── the failure path: a dead worker costs its shard, never the draw ───────────


def test_pool_redispatches_a_dead_shard(tmp_path: Path, monkeypatch) -> None:
    """A failure in one worker must not kill a whole draw.

    Worker 1 is dead on first contact. Its shard must be re-dispatched to the
    worker that answered, and the draw must complete with the SAME merged
    object a healthy pool would have produced — just slower.
    """
    gids = list(range(6))
    serial = _write_shards(tmp_path, 2, gids)
    (tmp_path / "gen_records").mkdir()
    calls: list[tuple[str, str | None]] = []

    def fake_post(url: str, loom_dir: Path, shard: str | None, timeout: float):
        calls.append((url, shard))
        if url.endswith(":8769") and calls.count((url, shard)) == 1:
            raise OSError("connection refused")
        i, n = (int(x) for x in str(shard).split("/"))
        return {"ok": True, "n_gens": 3, "shard": f"{i}of{n}",
                "replay_s": 1.0, "bins_s": 0.5,
                "failures_replay": [], "failures_bins": []}

    monkeypatch.setattr(harvest_pool, "post_harvest", fake_post)
    out = harvest_pool.harvest_via_pool(
        ["127.0.0.1:8768", "127.0.0.1:8769"], tmp_path, 30.0, expected_gens=len(gids)
    )
    assert out["ok"] and out["degraded"] is True
    assert out["n_shards_redispatched"] == 1
    assert out["n_gens"] == len(gids)
    # The shard that failed on :8769 was retried on the healthy :8768.
    assert ("http://127.0.0.1:8768", "1/2") in calls
    # And the merged artifact is the serial one, byte for byte.
    with np.load(tmp_path / "bins.npz") as z:
        assert z["features"].tobytes() == serial["features"].tobytes()
    assert (tmp_path / "cells.json").exists()
    assert json.loads((tmp_path / "run_meta.json").read_text())["pool"]["n_workers"] == 2


def test_pool_raises_when_every_worker_is_dead(tmp_path: Path, monkeypatch) -> None:
    """All dead -> raise, so loom_serve falls back rather than serving nothing."""
    def fake_post(url, loom_dir, shard, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(harvest_pool, "post_harvest", fake_post)
    with pytest.raises(RuntimeError, match="all 2 harvest workers failed"):
        harvest_pool.harvest_via_pool(["h:1", "h:2"], tmp_path, 5.0)


def test_pool_refuses_a_partial_harvest(tmp_path: Path, monkeypatch) -> None:
    """Covering fewer gens than the draw made must RAISE, not return quietly.

    A partial harvest is the one outcome a downstream reader cannot detect:
    ``<loom dir>/cells.json`` would simply describe a smaller fan, and the missing candidates
    would look like candidates that were never drawn.
    """
    _write_shards(tmp_path, 2, list(range(6)))

    def fake_post(url, loom_dir, shard, timeout):
        return {"ok": True, "n_gens": 3, "replay_s": 1.0, "bins_s": 0.5,
                "failures_replay": [], "failures_bins": []}

    monkeypatch.setattr(harvest_pool, "post_harvest", fake_post)
    with pytest.raises(RuntimeError, match="refusing a partial harvest"):
        harvest_pool.harvest_via_pool(["h:1", "h:2"], tmp_path, 5.0, expected_gens=8)


def test_pool_refuses_one_worker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ONE worker"):
        harvest_pool.harvest_via_pool(["h:1"], tmp_path, 5.0)


# ── the real fan: byte equality on the GPU node, skipped on a laptop ────────────────


POOL_URLS_ENV = "LOOM_POOL_TEST_URLS"
POOL_DIR_ENV = "LOOM_POOL_TEST_DIR"


#: Imported from the shipped module, never re-implemented here: the bench uses
#: the same function, so "byte-equal" cannot come to mean two different things
#: in the test and in the measurement that justifies the default.
_digest_tree = harvest_pool.digest_harvest_tree


@pytest.mark.skipif(
    not os.environ.get(POOL_URLS_ENV) or not os.environ.get(POOL_DIR_ENV),
    reason=f"needs live harvest workers: set {POOL_URLS_ENV} and {POOL_DIR_ENV}",
)
def test_parallel_matches_serial_byte_for_byte() -> None:
    """THE equivalence proof, on a real fan against real workers.

    Harvest one fixed loom dir serially, digest every artifact, wipe the
    artifacts, harvest the SAME dir through the pool, digest again, and require
    the two digest maps to be equal. Per-span byte equality is the claim; this
    is the measurement of it.
    """
    urls = harvest_pool.parse_worker_list(os.environ[POOL_URLS_ENV])
    src = Path(os.environ[POOL_DIR_ENV])
    assert (src / "gen_records").is_dir(), f"{src} has no gen_records/"
    n_gens = len(list((src / "gen_records").glob("gen_*.json")))
    assert n_gens >= 2

    def wipe() -> None:
        for sub in ("signatures", "fork_series", "shard_meta"):
            for p in (src / sub).glob("*"):
                p.unlink()
        for name in ("bins.npz", "cells.json", "run_meta.json"):
            (src / name).unlink(missing_ok=True)

    wipe()
    harvest_pool.post_harvest(f"http://{urls[0]}", src, None, 900.0)
    serial = _digest_tree(src)

    wipe()
    out = harvest_pool.harvest_via_pool(urls, src, 900.0, expected_gens=n_gens)
    assert out["ok"] and not out["degraded"]
    parallel = _digest_tree(src)

    assert set(serial) == set(parallel), (
        "the two paths produced different artifact SETS: "
        f"serial-only {sorted(set(serial) - set(parallel))}, "
        f"pool-only {sorted(set(parallel) - set(serial))}"
    )
    differing = [k for k in sorted(serial) if serial[k] != parallel[k]]
    assert not differing, (
        f"{len(differing)}/{len(serial)} artifacts DIFFER between serial and "
        f"parallel harvest: {differing[:10]} — the pool is perturbing features "
        "and must not ship"
    )
