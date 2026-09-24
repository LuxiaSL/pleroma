"""gen_records -> GPU-lane replay manifests: a pure data transform, no extraction here.

Reads a corpus's gen_records (the vLLM producer's contract) and writes N shard
directories, each holding exactly what `anamnesis.scripts.run_gpu_replay`
consumes: `<shard>/manifest.json` = {"entries": {gen_id: {prompt_length, input_ids}}}
plus `metadata.json` = {"generations": [...]} carrying the join keys
(prompt_id, prompt_class, seed_idx) so downstream banks can re-associate
features with prompts without ever re-reading gen_records.

Run on the host that holds the corpus (CPU only)::

    PYTHONPATH=. python3 -m pleroma.harvest.gpu_manifests \\
        --gen-dir <corpus>/gen_directive \\
        --out <corpus>/features_gpu/directive \\
        --shards 6 [--max-gens 25]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("w6_gpu_manifests")


def shard_of(gen_id: int, n_shards: int, n_total: int) -> int:
    """Contiguous blocks so shard sizes differ by at most 1."""
    base, extra = divmod(n_total, n_shards)
    boundary = 0
    for s in range(n_shards):
        boundary += base + (1 if s < extra else 0)
        if gen_id < boundary:
            return s
    raise ValueError(f"gen_id {gen_id} outside 0..{n_total - 1}")


def write_manifests(
    gen_dir: Path, out: Path, shards: int, max_gens: int | None = None
) -> int:
    """The transform itself: gen_records -> N lane shard dirs. Returns n_gens.

    Separate from ``main()`` so ``pleroma.harvest.worker``'s GPU-lane harvest
    builds a loom dir's manifest with THIS function rather than a second copy
    of the same transform. A second copy is exactly how a lane run and a corpus
    run come to disagree about ``prompt_length`` or ``input_ids`` truncation,
    with both sides looking locally correct.

    Raises ``FileNotFoundError`` when the dir holds no gen records — the CLI
    turns that into its historical ``return 2`` rather than a traceback.
    """
    records_dir = Path(gen_dir) / "gen_records"
    paths = sorted(records_dir.glob("gen_*.json"))
    if max_gens:
        paths = paths[:max_gens]
    if not paths:
        raise FileNotFoundError(f"no gen records under {records_dir}")
    n_total = len(paths)
    entries: list[dict[str, Any]] = [{"entries": {}, "generations": []}
                                     for _ in range(shards)]
    for p in paths:
        r = json.loads(p.read_text())
        gid = int(r["generation_id"])
        s = shard_of(gid if not max_gens else paths.index(p), shards, n_total)
        entries[s]["entries"][str(gid)] = {
            "prompt_length": int(r["prompt_length"]),
            "input_ids": [int(x) for x in r["input_ids"]],
        }
        entries[s]["generations"].append({
            "generation_id": gid, "prompt_id": str(r["prompt_id"]),
            "prompt_class": str(r["prompt_class"]),
            "seed_idx": int(r["seed_idx"]),
            "prompt_length": int(r["prompt_length"]),
        })
    for s, blob in enumerate(entries):
        d = Path(out) / f"shard_{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"entries": blob["entries"]}))
        (d / "metadata.json").write_text(
            json.dumps({"generations": blob["generations"]}))
        logger.info("shard_%02d: %d gens", s, len(blob["entries"]))
    logger.info("DONE: %d gens -> %d shards under %s", n_total, shards, out)
    return n_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--max-gens", type=int, default=None,
                    help="smoke mode: only the first N gen ids")
    args = ap.parse_args()

    try:
        write_manifests(args.gen_dir, args.out, args.shards, args.max_gens)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
