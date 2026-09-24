"""Parallel loom harvest: fan ONE loom dir across N persistent harvest workers.

Harvest is where a loom turn's time goes: **88-95% of it** (k=8: ~1.5 s
generate, ~13.5 s harvest), and inside a 1.57 s span **79% is CPU feature
math, 15% GPU forward**. A 3B forward is cheap;
``compute_features_with_families_from_data`` is not. That shape is exactly the
one that parallelises: 4 concurrent harvest processes run at 1.67 s/span
against 1.57 s solo — ~3.8x throughput for ~6% per-span degradation.

── why a POOL and not a sharded corpus replay ───────────────────────────────

A sharded corpus replay (``pleroma.harvest.replay --shard i/n``) spreads a
corpus across cold subprocesses, and it is the right tool for a 21k-span
corpus wave where a 30 s model load amortises to nothing. It is the WRONG tool
for a loom turn: k=8 spans is ~13 s of work, and eight cold model loads would
cost more than the harvest they replace. That is the same reasoning behind
``pleroma.harvest.worker`` (two cold subprocesses per /loom call -> one
persistent process).

So the pool is N of those persistent workers, already warm, each handed a
``shard`` of the same loom dir over the worker's loopback protocol. The
partition is ``pleroma.harvest.replay.shard_records`` — literally the same
function a sharded corpus replay uses, so "a shard" means one thing in this
repo.

── the numerics are not touched, and that is asserted ───────────────────────

Nothing in the replay or bins stage carries state across records: every span
goes through ``replay_extract`` -> ``compute_features_with_families_from_data`` ->
``save_features`` on its own, and every bins row through ``extract_one`` on its
own. Which worker holds a span therefore cannot change that span's numbers, and
per-gen artifacts are named by generation_id so the writes are disjoint.

That is an argument, and arguments about numerics are worth what they cost.
``tests/unit/harvest/test_harvest_pool.py`` measures it instead: a fixed fan
harvested serially and in parallel, compared **byte-for-byte** on every
``signatures/gen_NNN.npz``, every ``fork_series/gen_NNN.npz`` and every
``<loom-dir>/bins.npz`` row. Byte equality, not a tolerance — see that test's
docstring for why nothing weaker would do.

── the failure path is the point ────────────────────────────────────────────

One failing part of a draw must not take the whole draw down. A worker that
dies, times out, or answers badly costs its shard, not the draw:

  1. its shard is RE-DISPATCHED, once, to a worker that answered (the loom dir
     is the only state, so a re-run of a shard is idempotent);
  2. if that also fails, ``harvest_via_pool`` raises — and the server's
     harvest ladder (``pleroma.serve.harvest_client.HarvestLadder``) falls back
     to the single-worker path and then to the two-subprocess pipeline.

The pool is therefore strictly additive: a box with no pool configured never
enters this module at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("harvest_pool")

#: Hard ceiling on pool width. The measured win is at 4 workers and the box
#: may be shared; a typo like ``--harvest-workers`` with 40 entries should be refused
#: at parse time, not discovered by the machine falling over.
MAX_POOL_WORKERS: int = 16


def _hash_array(a: np.ndarray) -> str:
    """sha256 over an array's dtype, shape and raw bytes.

    dtype and shape are in the hash, not just the bytes: the same bytes read as
    float32 and as int32 are different numbers, and a reshape that preserved
    bytes while moving rows is exactly the merge bug this is here to catch.
    """
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest_harvest_tree(loom_dir: Path) -> dict[str, str]:
    """Hash the CONTENTS of every artifact a harvest wrote, keyed by generation_id.

    This lives in the shipped module rather than in the test so that the test
    and ``bench_harvest_pool`` measure equivalence with ONE definition of it.
    Two copies would drift, and the copy that drifted would be the one that
    stopped catching a real perturbation.

    It hashes BOTH the whole file and every array inside it, which are two
    different claims worth making:

    * WHOLE-FILE bytes are the strongest statement available — "the parallel
      path wrote the identical file". This works because ``np.savez_compressed``
      is byte-reproducible: it writes entries through ``ZipInfo(name)``, whose
      ``date_time`` DEFAULTS to the 1980 epoch rather than the clock, so two
      harvests of identical arrays minutes apart produce identical bytes.
      (Checked, not assumed — an earlier draft of this function dropped
      whole-file hashing on the belief that npz embeds a real mtime. That
      belief was wrong, and dropping the check would have quietly weakened the
      strongest evidence this change has.)
    * PER-ARRAY hashes (dtype, shape, bytes) survive any future change to the
      container, and they name WHICH array moved instead of only which file.
      dtype and shape are in the hash because the same bytes read as float32
      and int32 are different numbers, and a reshape that preserved bytes while
      moving rows is exactly the merge bug this guards.

    ``<loom-dir>/bins.npz`` features are additionally split per ROW so a
    disagreement names the span it belongs to instead of the whole matrix.
    ``config`` and ``failures`` are excluded from the bins digest: both are
    metadata the pool legitimately rewrites (per-shard counts become pooled
    totals), and neither is a feature.
    """
    out: dict[str, str] = {}
    for sub in ("signatures", "fork_series"):
        for p in sorted((Path(loom_dir) / sub).glob("gen_*.npz")):
            out[f"{sub}/{p.name}#file"] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p, allow_pickle=False) as z:
                for key in sorted(z.files):
                    out[f"{sub}/{p.name}:{key}"] = _hash_array(np.asarray(z[key]))
    with np.load(Path(loom_dir) / "bins.npz", allow_pickle=False) as z:
        gids = np.asarray(z["generation_id"])
        feats = np.asarray(z["features"])
        for i, g in enumerate(gids.tolist()):
            out[f"bins/gen_{int(g):03d}"] = _hash_array(feats[i])
        for key in sorted(set(z.files) - {"features", "config", "failures"}):
            out[f"bins/{key}"] = _hash_array(np.asarray(z[key]))
    return out


def parse_worker_list(spec: str) -> list[str]:
    """``"h:1,h:2"`` -> ``["h:1", "h:2"]``, refusing the shapes that bite.

    Duplicates are refused rather than deduped: two entries pointing at one
    worker means the operator believes there are two, and a pool that silently
    halves its width would show up only as a missing speedup.
    """
    urls = [u.strip() for u in str(spec).split(",") if u.strip()]
    if not urls:
        raise ValueError(f"--harvest-workers parsed to no entries from {spec!r}")
    if len(urls) != len(set(urls)):
        dupes = sorted({u for u in urls if urls.count(u) > 1})
        raise ValueError(
            f"--harvest-workers lists {dupes} more than once; each entry must be a "
            "DISTINCT worker (two names for one process is not two workers)"
        )
    if len(urls) > MAX_POOL_WORKERS:
        raise ValueError(
            f"--harvest-workers lists {len(urls)} workers, over the {MAX_POOL_WORKERS} "
            "ceiling; this box is shared"
        )
    for url in urls:
        if "/" in url:
            raise ValueError(f"--harvest-workers entry {url!r} should be host:port, not a URL")
        host, _, port = url.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"--harvest-workers entry {url!r} is not host:port")
    return urls


def post_harvest(
    worker_url: str, loom_dir: Path, shard: str | None, timeout: float
) -> dict[str, Any]:
    """POST /harvest to one worker. Raises on ANY problem — see harvest_via_worker."""
    body: dict[str, Any] = {"loom_dir": str(loom_dir)}
    if shard is not None:
        body["shard"] = shard
    req = urllib.request.Request(
        worker_url.rstrip("/") + "/harvest",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = json.loads(resp.read())
    if not blob.get("ok"):
        raise RuntimeError(f"harvest worker {worker_url} reported failure: {blob}")
    return blob


def merge_shard_cells(loom_dir: Path, n: int) -> list[dict[str, Any]]:
    """Union the per-shard cell lists, ascending generation_id.

    Ascending gid is not cosmetic: the UNSHARDED worker writes cells in the
    order ``load_gen_records`` returns them, which is sorted by gid. Sorting
    here is what makes the merged ``<loom-dir>/cells.json`` identical to the serial one
    rather than merely equivalent to it.
    """
    shard_dir = Path(loom_dir) / "shard_meta"
    rows: dict[int, dict[str, Any]] = {}
    for i in range(n):
        path = shard_dir / f"cells_shard{i}of{n}.json"
        if not path.exists():
            raise ValueError(f"{path} missing — shard {i}/{n} wrote no cells")
        blob = json.loads(path.read_text())
        if not isinstance(blob, list):
            raise ValueError(f"{path}: expected a list of cells")
        for cell in blob:
            gid = int(cell["generation_id"])
            if gid in rows:
                raise ValueError(
                    f"generation_id {gid} appears in more than one shard — the "
                    "partition is not a partition; refusing to merge"
                )
            rows[gid] = cell
    return [rows[g] for g in sorted(rows)]


def merge_shard_bins(loom_dir: Path, n: int) -> dict[str, np.ndarray]:
    """Concatenate the per-shard bins arrays back into one, ascending gid.

    Refuses on a feature-name or width disagreement between shards: that would
    mean two workers were built from different configs, and the pooled matrix
    would silently mix two spaces — the trap ``merge_shard_run_meta`` guards
    for calibration, in the one place bins can hit it.
    """
    shard_dir = Path(loom_dir) / "shard_meta"
    parts: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    names: np.ndarray | None = None
    config: Any = None
    failures: list[str] = []
    for i in range(n):
        path = shard_dir / f"bins_shard{i}of{n}.npz"
        if not path.exists():
            raise ValueError(f"{path} missing — shard {i}/{n} wrote no bins")
        with np.load(path, allow_pickle=False) as z:
            if "empty_shard" in z.files:
                continue  # more workers than gens; contributes no rows
            gids = np.asarray(z["generation_id"])
            cols = {k: np.asarray(z[k]) for k in z.files}
            if names is None:
                names, config = cols["feature_names"], cols.get("config")
            elif not np.array_equal(names, cols["feature_names"]):
                raise ValueError(
                    f"shard {i}/{n} bins feature_names differ from shard 0's — the "
                    "shards are not in one feature space"
                )
            failures.extend(str(x) for x in cols.get("failures", []))
            parts.append((gids, cols))
    if not parts or names is None:
        raise ValueError("every bins shard was empty — nothing to merge")
    all_gids = np.concatenate([g for g, _ in parts])
    if len(set(all_gids.tolist())) != len(all_gids):
        raise ValueError("a generation_id appears in more than one bins shard")
    order = np.argsort(all_gids, kind="stable")
    out: dict[str, np.ndarray] = {}
    for key in ("features", "generation_id", "run_dir", "prompt_id",
                "prompt_class", "seed_idx"):
        if key not in parts[0][1]:
            continue
        out[key] = np.concatenate([cols[key] for _, cols in parts])[order]
    out["feature_names"] = names
    if config is not None:
        # Shard 0's config describes the pipeline, which every shard shares — but
        # its n_failures counts only shard 0's. Rewrite that one field to the
        # pooled total and say where the blob came from, so a reader of a merged
        # <loom-dir>/bins.npz is never told 0 failures when three shards had one each.
        try:
            blob = json.loads(str(config))
            blob["n_failures"] = len(failures)
            blob["source"] = (
                f"harvest_pool.merge_shard_bins over {n} shards "
                f"(per-shard: harvest_worker.py, extract_bins.extract_one)"
            )
            out["config"] = np.array(json.dumps(blob))
        except (ValueError, TypeError):
            out["config"] = config
    out["failures"] = np.array(failures)
    return out


def merge_shard_run_meta(loom_dir: Path, n: int) -> dict[str, Any]:
    """Aggregate the per-shard run metas, refusing the disagreements that matter.

    Two hard errors, the same ones a sharded corpus replay's merge makes: a
    shard whose GATE 0 did not pass has its vectors in another space, and shards
    on different CALIBRATION BYTES are un-poolable in a way Gate 0 cannot see.
    """
    shard_dir = Path(loom_dir) / "shard_meta"
    metas: list[dict[str, Any]] = []
    for i in range(n):
        path = shard_dir / f"run_meta_shard{i}of{n}.json"
        if not path.exists():
            raise ValueError(f"{path} missing — shard {i}/{n} wrote no run_meta")
        metas.append(json.loads(path.read_text()))
    # Gate 0 runs on a worker's FIRST gen. A shard with no gens (more workers
    # than candidates) therefore never ran it and says so; that is the one value
    # other than "passed" this accepts, and only for a shard that extracted
    # nothing. Anything else is a space mismatch and must not merge.
    bad = [
        str(m.get("gate0")) for m in metas
        if str(m.get("gate0")) != "passed"
        and not (str(m.get("gate0")).startswith("not_run")
                 and int(m.get("n_cells", 0)) == 0)
    ]
    if bad:
        raise ValueError(f"not every shard passed GATE 0 (saw {sorted(set(bad))})")
    calibs = {json.dumps(m.get("calibration", {}), sort_keys=True) for m in metas}
    if len(calibs) != 1:
        raise ValueError(
            f"the {n} shards used {len(calibs)} different calibrations — their "
            "signatures do not live in one space. Check --calib-dir on every worker."
        )
    dims = {int(m.get("feature_dim", -1)) for m in metas if int(m.get("n_cells", 0)) > 0}
    if len(dims) > 1:
        raise ValueError(f"the shards produced different feature dims: {sorted(dims)}")
    # ★ ONE POOL, ONE LANE. A worker can harvest on the frozen v3 replay
    # surface or on the GPU lane (pleroma.harvest.worker --harvest-lane), and
    # the two emit different feature-name lists — 3,441 vs 4,086 at 70B. Two
    # shards of one draw coming from different lanes is the "never mix lanes"
    # rule broken at serve time, and unlike a dim mismatch it CAN pass the check
    # above whenever the two happen to agree on width. A shard that does not
    # carry the field reports None, which is a consistent value, so a pool of
    # workers that all omit it still merges.
    lanes = {str(m.get("harvest_lane")) for m in metas}
    if len(lanes) != 1:
        raise ValueError(
            f"the {n} shards harvested on {len(lanes)} different lanes "
            f"({sorted(lanes)}) — their signatures are not one instrument's "
            "output and must not merge. Check --harvest-lane on every worker."
        )
    lane = lanes.pop()
    first = metas[0]
    return {
        **{k: first.get(k) for k in (
            "model_id", "preset", "sampled_layers", "gen_dir", "calib_dir",
            "calibration", "discriminants", "v3_config_source", "logits_top_k",
        )},
        "harvest_lane": None if lane == "None" else lane,
        "stage": ("harvest_pool_gpu_lane" if lane == "gpu"
                  else "harvest_pool_replay_b0"),
        "gate0": "passed (every shard, on its own first gen)",
        "feature_dim": dims.pop() if dims else int(first.get("feature_dim", 0)),
        "n_corpus": max(int(m.get("n_corpus", 0)) for m in metas),
        "n_records": sum(int(m.get("n_records", 0)) for m in metas),
        "n_cells": sum(int(m.get("n_cells", 0)) for m in metas),
        "failures": [f for m in metas for f in m.get("failures", [])],
        "n_shards": n,
        "shards": [
            {"shard": m.get("shard"), "n_records": m.get("n_records"),
             "n_cells": m.get("n_cells"), "elapsed_s": m.get("elapsed_s")}
            for m in metas
        ],
    }


def harvest_via_pool(
    worker_urls: Sequence[str],
    loom_dir: Path,
    timeout: float,
    expected_gens: int | None = None,
) -> dict[str, Any]:
    """Harvest ONE loom dir across N warm workers, then merge. Raises to fall back.

    The contract with the caller is deliberately all-or-nothing: either this
    returns having written ``<loom-dir>/cells.json``,
    ``<loom-dir>/run_meta.json`` and ``<loom-dir>/bins.npz`` exactly as a serial
    harvest would, or it raises and the caller takes the serial path. It never
    returns a partial harvest, because a partial harvest is the thing a
    downstream reader cannot detect.
    """
    urls = [f"http://{u}" if not str(u).startswith("http") else str(u) for u in worker_urls]
    n = len(urls)
    if n == 0:
        raise ValueError("harvest_via_pool called with no workers")
    if n == 1:
        raise ValueError("harvest_via_pool called with ONE worker — use the single-worker path")

    loom_dir = Path(loom_dir)
    results: dict[int, dict[str, Any]] = {}
    failed: list[tuple[int, str, Exception]] = []

    # Fan out. ThreadPoolExecutor, not asyncio: the work is entirely in the
    # workers, this side is blocking HTTP, and the pool is bounded by n.
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="harvest") as pool:
        futures = {
            pool.submit(post_harvest, urls[i], loom_dir, f"{i}/{n}", timeout): i
            for i in range(n)
        }
        for future, i in futures.items():
            try:
                results[i] = future.result()
            except Exception as exc:  # noqa: BLE001 — every failure mode degrades alike
                logger.warning("harvest shard %d/%d on %s FAILED (%s: %s)",
                               i, n, urls[i], type(exc).__name__, exc)
                failed.append((i, urls[i], exc))

    # ── degrade, don't die: re-dispatch a dead worker's shard to a live one ──
    if failed:
        healthy = [urls[i] for i in sorted(results)]
        if not healthy:
            raise RuntimeError(
                f"all {n} harvest workers failed; first: "
                f"{type(failed[0][2]).__name__}: {failed[0][2]}"
            )
        logger.warning(
            "%d/%d harvest shards failed — re-dispatching them serially to %d "
            "healthy worker(s); this draw will be SLOWER, not wrong",
            len(failed), n, len(healthy),
        )
        for k, (i, dead_url, _) in enumerate(failed):
            target = healthy[k % len(healthy)]
            try:
                results[i] = post_harvest(target, loom_dir, f"{i}/{n}", timeout)
                logger.info("shard %d/%d recovered on %s (was %s)", i, n, target, dead_url)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"harvest shard {i}/{n} failed on {dead_url} and again on the "
                    f"healthy worker {target} ({type(exc).__name__}: {exc}) — "
                    "falling back to serial harvest"
                ) from exc

    if len(results) != n:
        raise RuntimeError(f"only {len(results)}/{n} shards returned; refusing to merge")

    # ── merge into the three whole-corpus files downstream readers expect ────
    cells = merge_shard_cells(loom_dir, n)
    meta = merge_shard_run_meta(loom_dir, n)
    bins = merge_shard_bins(loom_dir, n)

    covered = {int(c["generation_id"]) for c in cells}
    n_corpus = int(meta.get("n_corpus") or 0)
    if expected_gens is not None and len(covered) != int(expected_gens):
        raise RuntimeError(
            f"the pool covered {len(covered)} gens but the draw made {expected_gens} — "
            "refusing a partial harvest"
        )
    if n_corpus and len(covered) != n_corpus:
        raise RuntimeError(
            f"the pool covered {len(covered)} gens of a {n_corpus}-gen loom dir — "
            "refusing a partial harvest"
        )
    if len(bins["generation_id"]) != len(covered):
        raise RuntimeError(
            f"bins has {len(bins['generation_id'])} rows for {len(covered)} cells"
        )

    meta["pool"] = {
        "workers": list(worker_urls),
        "n_workers": n,
        "n_shards_redispatched": len(failed),
        "redispatched_from": [u for _, u, _ in failed],
        "per_shard_s": {str(i): results[i].get("replay_s", 0) + results[i].get("bins_s", 0)
                        for i in sorted(results)},
    }
    (loom_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    (loom_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(loom_dir / "bins.npz", **bins)

    return {
        "ok": True,
        "n_gens": len(covered),
        "n_workers": n,
        "degraded": bool(failed),
        "n_shards_redispatched": len(failed),
        "replay_s": round(max(float(r.get("replay_s", 0)) for r in results.values()), 3),
        "bins_s": round(max(float(r.get("bins_s", 0)) for r in results.values()), 3),
        "failures_replay": [f for r in results.values() for f in r.get("failures_replay", [])],
        "failures_bins": [f for r in results.values() for f in r.get("failures_bins", [])],
    }
