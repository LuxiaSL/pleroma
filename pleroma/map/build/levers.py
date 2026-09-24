"""Cluster-contrast levers (local CPU, numpy + stdlib only).

For one prompt's seeds, split into the two basins the basin check
(`pleroma.map.build.basin`) already found, the lever is the difference of the two clusters'
MEAN HIDDEN STATES at each inject site::

    lever[site] = mean_h(cluster 0)[site] - mean_h(cluster 1)[site]

That is the anamnesis desk's contrast-of-states construction — the one static
residual write that is PROVEN to put mode-class coordinates on the stream — aimed
at our question instead of theirs. It is not a learned map, it is not a read-side
direction, and it contains no NLL: each of those three was tried as the way to
put a basin on the stream, and each failed.

── the clusters are NOT re-invented here ──────────────────────────────────────

`pleroma.map.build.basin.two_means_split` is IMPORTED, not copied, and fed
exactly what `pleroma.map.build.basin.analyse_prompts` feeds it: the signatures of one
prompt's seeds **from one run dir**, in ``load_signatures`` order (sorted
``signatures/gen_*.npz``). Same rows, same order, no RNG in the algorithm — so
the assignment this stage computes is bit-for-bit the assignment the basin report
published. ``--basin-reports`` turns that from an argument into a check: every
prompt in a supplied report must agree with our labels exactly, or exactly
inverted (2-means labels carry no order), and a single disagreement aborts.

── waves, and why the grouping is (prompt_id, WAVE) ───────────────────────────

Clusters were defined PER RUN. The corpus is two waves of the same batteries —
pilot+wave2 at seeds 0-15, the frozen replication at seeds 16-31 — and a prompt's
seed-0-15 basins are not commensurable with its seed-16-31 basins: they are
separate 2-means fits over separate samples. So the unit here is
``(prompt_id, wave)`` and clustering happens WITHIN a wave. ``--wave-seed-
boundary`` (default 16) is the only place that split is expressed. Because the
run dirs partition the prompts (pilot and wave2 share no prompt_id; the same for
the two replication dirs), each ``(prompt_id, wave)`` group draws from exactly
ONE run dir — and if it ever does not, this stage REFUSES rather than clustering
across two runs and calling the result the basin check's.

── leave-one-out is the caller's job, and the algebra is here ─────────────────

``loo_lever()`` is exported because an evaluation must never score a gen under
a lever the gen helped build. Removing one member from a cluster mean is cheap
algebra on the banked cluster SUMS::

    mean_c^(-g) = (sum_c - h_g) / (n_c - 1)

which is why this stage banks ``cluster_sums`` beside ``levers``: the sums make
the held-out lever exact rather than approximate, at no GPU cost. They are banked
in **float64** although the means themselves are float32 — a lever is a
difference of two similar-magnitude means, so the cancellation is where precision
would go, and 8 MB of extra npz is cheaper than the doubt.

Usage (local)::

    python -m pleroma.map.build.levers \\
        --hiddens outputs/b15_hiddens_orig/mean_hiddens.npz \\
                  outputs/b15_hiddens_repl/mean_hiddens.npz \\
        --run-dirs outputs/pilot_replay outputs/wave2_replay \\
                   outputs/repl_replay_a outputs/repl_replay_b \\
        --basin-reports pilot=outputs/expB0_pilot/basin_report.json \\
        --out outputs/b15_levers/levers.npz
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy import (house rule; this stage is pure numpy).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pleroma.map.build.hiddens import corpus_key
from pleroma.map.build.basin import load_signatures, two_means_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB15.levers")

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

#: Seed index at which the frozen replication wave starts (pilot+wave2 = 0-15).
DEFAULT_WAVE_BOUNDARY = 16

#: Wave labels, in the order they are reported.
WAVE_ORIG = "orig"
WAVE_REPL = "repl"

#: Smallest either cluster may be for the group to carry a lever. Three, not two,
#: because leave-one-out removes a member: a 2-member cluster becomes a 1-member
#: "mean" the moment the held-out gen is subtracted, which is a single trajectory
#: wearing a cluster's hat, and its lever says nothing about a basin.
DEFAULT_MIN_CLUSTER = 3


def wave_of(seed_idx: int, boundary: int = DEFAULT_WAVE_BOUNDARY) -> str:
    """Which generation wave a seed index belongs to."""
    if boundary <= 0:
        raise ValueError(f"wave boundary must be > 0, got {boundary}")
    return WAVE_ORIG if int(seed_idx) < int(boundary) else WAVE_REPL


def gen_key(corpus: str, generation_id: int) -> str:
    """The join key between hidden means and signatures: corpus + generation id."""
    return f"{corpus}#{int(generation_id):04d}"


# ── the banked hidden means ────────────────────────────────────────────────────


@dataclass(frozen=True)
class HiddenBank:
    """Every hidden-means bank (`pleroma.map.build.hiddens`) given on the command line, concatenated.

    The npzs must agree on ``sites`` and hidden dim — two banks built at different
    sites do not live in one space, and silently stacking them would make every
    lever a mixture of two site conventions.
    """

    means: F32                       # [N, n_sites, d]
    generation_ids: NDArray[np.int64]
    prompt_ids: list[str]
    prompt_classes: list[str]
    seed_idxs: NDArray[np.int64]
    corpus_keys: list[str]
    source_dir_labels: list[str]
    sites: list[int]
    sources: list[str]

    @property
    def n_gens(self) -> int:
        return int(self.means.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.means.shape[2])

    def row_index(self) -> dict[str, int]:
        return {
            gen_key(self.corpus_keys[i], int(self.generation_ids[i])): i
            for i in range(self.n_gens)
        }


def load_hidden_banks(paths: Sequence[Path]) -> HiddenBank:
    """Load and concatenate hidden-means banks (`pleroma.map.build.hiddens`), refusing incompatible ones.

    Raises KeyError when a bank lacks a required array, ValueError when its
    ``means`` has the wrong shape or banks disagree on sites or hidden dim."""
    required = (
        "means", "generation_ids", "prompt_ids", "prompt_classes", "seed_idxs",
        "corpus_keys", "source_dir_labels", "sites",
    )
    means: list[F32] = []
    gids: list[NDArray[np.int64]] = []
    pids: list[str] = []
    classes: list[str] = []
    seeds: list[int] = []
    corpora: list[str] = []
    labels: list[str] = []
    sites: list[int] | None = None
    sources: list[str] = []

    for path in paths:
        p = Path(path).expanduser()
        with np.load(p, allow_pickle=False) as npz:
            missing = [k for k in required if k not in npz]
            if missing:
                raise KeyError(f"{p}: missing {missing} (has {list(npz.files)})")
            these_sites = [int(x) for x in npz["sites"]]
            block = np.asarray(npz["means"], dtype=np.float32)
            if block.ndim != 3 or block.shape[1] != len(these_sites):
                raise ValueError(
                    f"{p}: means has shape {block.shape}, expected "
                    f"(n_gens, {len(these_sites)}, hidden_dim)"
                )
            if sites is None:
                sites = these_sites
            elif these_sites != sites:
                raise ValueError(
                    f"{p} banks sites {these_sites} but an earlier npz banked {sites} "
                    "— these means do not live in one space"
                )
            if means and block.shape[2] != means[0].shape[2]:
                raise ValueError(
                    f"{p}: hidden dim {block.shape[2]} != {means[0].shape[2]}"
                )
            means.append(block)
            gids.append(np.asarray(npz["generation_ids"], dtype=np.int64))
            pids.extend(str(x) for x in npz["prompt_ids"])
            classes.extend(str(x) for x in npz["prompt_classes"])
            seeds.extend(int(x) for x in npz["seed_idxs"])
            corpora.extend(str(x) for x in npz["corpus_keys"])
            labels.extend(str(x) for x in npz["source_dir_labels"])
        sources.append(str(p))
        logger.info("%s: %d gens x %d sites x %d dims", p, block.shape[0], block.shape[1],
                    block.shape[2])

    if sites is None or not means:
        raise ValueError("no --hiddens given")
    bank = HiddenBank(
        means=np.concatenate(means, axis=0),
        generation_ids=np.concatenate(gids),
        prompt_ids=pids,
        prompt_classes=classes,
        seed_idxs=np.asarray(seeds, dtype=np.int64),
        corpus_keys=corpora,
        source_dir_labels=labels,
        sites=sites,
        sources=sources,
    )
    if bank.n_gens == 0:
        raise ValueError("the hidden banks are empty; refusing to build levers")
    index = bank.row_index()
    if len(index) != bank.n_gens:
        raise ValueError(
            "duplicate (corpus, generation_id) across --hiddens — two banks cover the "
            "same gens and the cluster sums would double-count them"
        )
    return bank


# ── clustering: the basin check's own 2-means, on its own rows ───────────────────


@dataclass(frozen=True)
class ClusteredGen:
    """One gen with the cluster label the basin check gave it."""

    generation_id: int
    corpus: str
    source_dir_label: str
    prompt_id: str
    prompt_class: str
    seed_idx: int
    wave: str
    cluster: int

    @property
    def key(self) -> str:
        return gen_key(self.corpus, self.generation_id)


def cluster_run_dir(run_dir: Path, wave_boundary: int) -> list[ClusteredGen]:
    """Re-run the basin check's per-prompt 2-means over one stage-2 run dir.

    Deliberately the same three lines `pleroma.map.build.basin.analyse_prompts`
    runs: prompt order is first-appearance in
    ``load_signatures`` order, members are ``np.nonzero(pid == pid_array)`` over
    that same order, and the split is ``two_means_split`` on the raw feature rows.
    Nothing is standardised, sub-selected or re-sorted in between — any of which
    would produce *a* clustering, but not *the* clustering the basin report named.
    """
    path = Path(run_dir).expanduser()
    label = path.resolve().name
    corpus = corpus_key(label)
    sigs = load_signatures(path)

    prompt_ids = [c.prompt_id for c in sigs.cells]
    unique_prompts = sorted(set(prompt_ids), key=lambda p: prompt_ids.index(p))
    pid_array = np.array(prompt_ids, dtype=object)

    out: list[ClusteredGen] = []
    for pid in unique_prompts:
        members = np.nonzero(pid_array == pid)[0]
        labels, _ = two_means_split(sigs.features[members])
        waves = {wave_of(sigs.cells[i].seed_idx, wave_boundary) for i in members}
        if len(waves) != 1:
            raise ValueError(
                f"{label} prompt {pid}: seeds span waves {sorted(waves)} at boundary "
                f"{wave_boundary}. The basin check clustered this run's seeds together, so "
                "a run dir that mixes waves means --wave-seed-boundary is wrong for "
                "this corpus."
            )
        wave = waves.pop()
        for i, cluster in zip(members, labels):
            cell = sigs.cells[i]
            out.append(
                ClusteredGen(
                    generation_id=int(sigs.gen_ids[i]),
                    corpus=corpus,
                    source_dir_label=label,
                    prompt_id=cell.prompt_id,
                    prompt_class=cell.prompt_class,
                    seed_idx=int(cell.seed_idx),
                    wave=wave,
                    cluster=int(cluster),
                )
            )
    logger.info(
        "%s -> corpus %r: %d gens over %d prompts clustered (the basin check's 2-means)",
        path, corpus, len(out), len(unique_prompts),
    )
    return out


def parse_basin_report_arg(spec: str) -> tuple[str | None, Path]:
    """``path`` or ``corpus=path``.

    The corpus a report belongs to is normally read off its own ``run_dir``
    field, but a report may have been written under a different directory name
    than the run dir given to ``--run-dirs`` (the local mirrors, for instance, sit
    under ``outputs/expB0_pilot`` while the run dir is ``pilot_replay``). The
    explicit form says which corpus it is rather than letting the check quietly
    match nothing.
    """
    text = str(spec)
    if "=" in text:
        corpus, _, path = text.partition("=")
        if not corpus.strip() or not path.strip():
            raise ValueError(f"--basin-reports entry {spec!r} is not 'corpus=path'")
        return corpus.strip(), Path(path.strip())
    return None, Path(text)


def verify_against_basin_report(
    report_path: Path, clustered: dict[str, ClusteredGen], corpus: str | None = None
) -> dict[str, Any]:
    """Assert our labels reproduce a published basin report (`pleroma.map.build.basin`), exactly.

    2-means labels are unordered, so a prompt passes if our assignment equals the
    report's OR is its exact complement. Anything else means the levers would be
    built on a different partition than the one the basin verdicts were published
    from, and this stage refuses to continue.

    A check that verified NOTHING also refuses: the commonest way for this to go
    wrong is a report whose corpus name does not match any ``--run-dirs`` entry,
    and a cross-check that silently matched zero prompts is worse than no
    cross-check at all — it reads in the log like a pass.
    """
    blob = json.loads(Path(report_path).expanduser().read_text())
    if corpus is None:
        corpus = corpus_key(str(blob.get("run_dir", report_path)))
    checked = 0
    skipped: list[str] = []
    for row in blob.get("prompts", []):
        pid = str(row["prompt_id"])
        ours: list[int] = []
        theirs: list[int] = []
        for seed in row.get("seeds", []):
            key = gen_key(corpus, int(seed["generation_id"]))
            mine = clustered.get(key)
            if mine is None or seed.get("cluster") is None:
                continue
            ours.append(int(mine.cluster))
            theirs.append(int(seed["cluster"]))
        if not ours:
            skipped.append(pid)
            continue
        same = ours == theirs
        flipped = ours == [1 - c for c in theirs]
        if not (same or flipped):
            raise ValueError(
                f"{report_path}: prompt {pid} — our 2-means assignment is neither the "
                f"report's nor its complement (ours {ours}, report {theirs}). The "
                "levers would not be built on the basin check's clusters."
            )
        checked += 1
    if checked == 0:
        raise ValueError(
            f"{report_path}: the cross-check matched NO prompt. It was read as corpus "
            f"{corpus!r}; the clustered corpora are "
            f"{sorted({g.corpus for g in clustered.values()})}. Pass the report as "
            "'corpus=path' if its run_dir names a directory other than the one given "
            "to --run-dirs."
        )
    logger.info(
        "basin-report cross-check %s (corpus %r): %d prompt(s) reproduced exactly%s",
        Path(report_path).name, corpus, checked,
        f", {len(skipped)} not covered by --run-dirs" if skipped else "",
    )
    return {
        "report": str(report_path),
        "corpus": corpus,
        "n_prompts_verified": checked,
        "prompts_not_covered": skipped,
    }


# ── levers ─────────────────────────────────────────────────────────────────────


@dataclass
class LeverGroup:
    """One ``(prompt_id, wave)`` unit: its members, its cluster sums, its lever."""

    prompt_id: str
    wave: str
    prompt_class: str
    corpus: str
    source_dir_label: str
    member_keys: list[str]
    clusters: list[int]
    cluster_sums: F64                   # [2, n_sites, d] — float64 on purpose
    cluster_sizes: tuple[int, int]
    missing_keys: list[str]
    lever: F32 | None = None            # [n_sites, d]
    lever_norms: list[float] | None = None
    reason: str = ""

    @property
    def key(self) -> str:
        return f"{self.prompt_id}|{self.wave}"


def loo_lever(
    cluster_sums: NDArray[np.floating],
    cluster_sizes: Sequence[int],
    own_cluster: int,
    own_mean: NDArray[np.floating],
) -> NDArray[np.float32]:
    """``mean(cluster 0) - mean(cluster 1)`` with ONE gen removed from its own side.

    The mandatory held-out construction: a lever built from a set containing the
    gen it is scored on has that gen's own hidden states inside it, and "the lever
    helps" would then be partly "the target is in the lever". Exact, not
    approximate — subtracting the member's banked mean from its cluster's banked
    sum reproduces the cluster mean over the remaining members bit-for-bit (up to
    float associativity), which is what ``test_build_levers`` asserts against a
    literal recompute.

    ``own_cluster`` must have at least 2 members before removal; the caller's
    ``min_cluster`` guard (>= 3) is what makes that true in practice.
    """
    sums = np.asarray(cluster_sums, dtype=np.float64).copy()
    if sums.shape[0] != 2:
        raise ValueError(f"cluster_sums must have 2 rows, got {sums.shape}")
    sizes = [int(n) for n in cluster_sizes]
    if len(sizes) != 2:
        raise ValueError(f"cluster_sizes must have 2 entries, got {cluster_sizes!r}")
    if own_cluster not in (0, 1):
        raise ValueError(f"own_cluster must be 0 or 1, got {own_cluster!r}")
    own = np.asarray(own_mean, dtype=np.float64)
    if own.shape != sums.shape[1:]:
        raise ValueError(
            f"own_mean has shape {own.shape}, expected {sums.shape[1:]}"
        )
    sums[own_cluster] -= own
    sizes[own_cluster] -= 1
    if sizes[own_cluster] < 1 or sizes[1 - own_cluster] < 1:
        raise ValueError(
            f"leave-one-out empties a cluster (sizes after removal {sizes}) — "
            "a group this small must not carry a lever"
        )
    return (sums[0] / sizes[0] - sums[1] / sizes[1]).astype(np.float32)


def matched_sign(own_cluster: int) -> float:
    """+1 for cluster 0, -1 for cluster 1 — "push toward my own basin".

    The stored lever always points 0 -> 1 (``mean_0 - mean_1``). A member of
    cluster 1 whose own basin is to be reinforced therefore gets the NEGATED
    lever, and its crossed arm is the stored one. Stating the sign in one function
    is the whole point: a flipped sign here would turn the headline inside out and
    nothing downstream could tell.
    """
    if own_cluster not in (0, 1):
        raise ValueError(f"own_cluster must be 0 or 1, got {own_cluster!r}")
    return 1.0 if own_cluster == 0 else -1.0


def build_groups(
    clustered: Sequence[ClusteredGen], bank: HiddenBank, min_cluster: int
) -> list[LeverGroup]:
    """One ``LeverGroup`` per ``(prompt_id, wave)``, lever where the split allows one."""
    if min_cluster < 2:
        raise ValueError(f"--min-cluster must be >= 2, got {min_cluster}")
    index = bank.row_index()
    n_sites, hidden_dim = bank.means.shape[1], bank.means.shape[2]

    ordered_keys: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[ClusteredGen]] = {}
    for gen in clustered:
        key = (gen.prompt_id, gen.wave)
        if key not in buckets:
            buckets[key] = []
            ordered_keys.append(key)
        buckets[key].append(gen)

    groups: list[LeverGroup] = []
    for key in ordered_keys:
        members = buckets[key]
        corpora = {g.corpus for g in members}
        if len(corpora) != 1:
            raise ValueError(
                f"prompt {key[0]} wave {key[1]} draws from corpora {sorted(corpora)}. "
                "The basin check clustered each run dir separately, so a group spanning "
                "two of them cannot reuse its assignment — split the run dirs or fix "
                "--wave-seed-boundary."
            )
        present = [g for g in members if g.key in index]
        missing = [g.key for g in members if g.key not in index]
        sums = np.zeros((2, n_sites, hidden_dim), dtype=np.float64)
        sizes = [0, 0]
        for gen in present:
            sums[gen.cluster] += bank.means[index[gen.key]].astype(np.float64)
            sizes[gen.cluster] += 1

        group = LeverGroup(
            prompt_id=key[0],
            wave=key[1],
            prompt_class=members[0].prompt_class,
            corpus=members[0].corpus,
            source_dir_label=members[0].source_dir_label,
            member_keys=[g.key for g in present],
            clusters=[g.cluster for g in present],
            cluster_sums=sums,
            cluster_sizes=(sizes[0], sizes[1]),
            missing_keys=missing,
        )
        if missing:
            logger.warning(
                "prompt %s wave %s: %d member(s) have no banked hidden mean and are "
                "excluded from the lever (%s)",
                key[0], key[1], len(missing), missing[:4],
            )
        if min(sizes) < min_cluster:
            group.reason = (
                f"cluster sizes {sizes} — the smaller side is below --min-cluster "
                f"{min_cluster}; leave-one-out would leave a near-single trajectory"
            )
        else:
            lever = (sums[0] / sizes[0] - sums[1] / sizes[1]).astype(np.float32)
            group.lever = lever
            group.lever_norms = [float(np.linalg.norm(lever[s])) for s in range(n_sites)]
            group.reason = "ok"
        groups.append(group)
    return groups


# ── reporting ──────────────────────────────────────────────────────────────────


def print_summary(groups: Sequence[LeverGroup], sites: Sequence[int], min_cluster: int) -> None:
    """The human-readable table; the npz is the record."""
    site_header = "".join(f"{'|L' + str(s):>10}" for s in sites)
    header = (
        f"{'prompt':<9}{'wave':<6}{'class':<22}{'n':>4}{'sizes':>8}{'lever':>7}" + site_header
    )
    logger.info("=" * len(header))
    logger.info(
        "CONTRAST LEVERS — %d group(s), %d with a lever (min cluster %d)",
        len(groups), sum(1 for g in groups if g.lever is not None), min_cluster,
    )
    logger.info("sites are decoder-layer INPUT indices (attach_residual_write layer_idx)")
    logger.info("-" * len(header))
    logger.info(header)
    for group in sorted(groups, key=lambda g: (g.wave, g.prompt_id)):
        norms = (
            "".join(f"{n:>10.2f}" for n in group.lever_norms)
            if group.lever_norms is not None
            else "".join(f"{'-':>10}" for _ in sites)
        )
        logger.info(
            "%-9s%-6s%-22s%4d%8s%7s%s",
            group.prompt_id, group.wave, group.prompt_class[:22],
            sum(group.cluster_sizes),
            f"{group.cluster_sizes[0]}/{group.cluster_sizes[1]}",
            "yes" if group.lever is not None else "no",
            norms,
        )
    logger.info("-" * len(header))
    skipped = [g for g in groups if g.lever is None]
    for group in sorted(skipped, key=lambda g: (g.wave, g.prompt_id)):
        logger.info("no lever for %s/%s: %s", group.prompt_id, group.wave, group.reason)
    logger.info("=" * len(header))


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hiddens", type=Path, nargs="+", required=True, help="mean_hiddens.npz file(s)"
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="stage-2 replay dirs (signatures/ + cells.json) matching the --hiddens corpora",
    )
    parser.add_argument("--out", type=Path, required=True, help="output levers npz")
    parser.add_argument(
        "--min-cluster",
        type=int,
        default=DEFAULT_MIN_CLUSTER,
        help="smallest either cluster may be for the group to carry a lever",
    )
    parser.add_argument(
        "--wave-seed-boundary",
        type=int,
        default=DEFAULT_WAVE_BOUNDARY,
        help="first seed_idx of the replication wave (pilot+wave2 are 0..boundary-1)",
    )
    parser.add_argument(
        "--basin-reports",
        nargs="*",
        default=[],
        metavar="[CORPUS=]PATH",
        help=(
            "basin report JSON(s) whose cluster assignments ours must reproduce "
            "exactly; prefix with 'corpus=' when the report's own run_dir names a "
            "different directory than --run-dirs (e.g. pilot=outputs/expB0_pilot/"
            "basin_report.json). A report that matches no prompt is an ERROR."
        ),
    )
    args = parser.parse_args()

    if args.min_cluster < 2:
        logger.error("--min-cluster must be >= 2, got %s", args.min_cluster)
        return 2
    if args.wave_seed_boundary <= 0:
        logger.error("--wave-seed-boundary must be > 0, got %s", args.wave_seed_boundary)
        return 2

    try:
        bank = load_hidden_banks(list(args.hiddens))
    except (OSError, KeyError, ValueError) as exc:
        logger.error("could not load --hiddens: %s", exc)
        return 2
    logger.info(
        "hidden bank: %d gens x %d sites %s x %d dims",
        bank.n_gens, len(bank.sites), bank.sites, bank.hidden_dim,
    )

    clustered: list[ClusteredGen] = []
    try:
        for run_dir in args.run_dirs:
            clustered.extend(cluster_run_dir(run_dir, int(args.wave_seed_boundary)))
    except (OSError, KeyError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        logger.error("clustering failed: %s", exc)
        return 2
    if not clustered:
        logger.error("no gen was clustered; refusing to build levers")
        return 1

    by_key: dict[str, ClusteredGen] = {}
    for gen in clustered:
        if gen.key in by_key:
            logger.error(
                "duplicate gen key %s across --run-dirs — two run dirs map to one "
                "corpus name", gen.key,
            )
            return 2
        by_key[gen.key] = gen

    checks: list[dict[str, Any]] = []
    try:
        for spec in args.basin_reports or []:
            report_corpus, report_path = parse_basin_report_arg(spec)
            checks.append(verify_against_basin_report(report_path, by_key, report_corpus))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("basin-report cross-check failed: %s", exc)
        return 3

    hidden_rows = bank.row_index()
    matched = sum(1 for key in by_key if key in hidden_rows)
    logger.info(
        "%d clustered gens, %d with a banked hidden mean (%d hidden rows unused)",
        len(by_key), matched, bank.n_gens - matched,
    )
    if matched == 0:
        logger.error(
            "no clustered gen has a banked hidden mean — --hiddens and --run-dirs "
            "describe different corpora (hidden corpora %s vs run-dir corpora %s)",
            sorted(set(bank.corpus_keys)), sorted({g.corpus for g in clustered}),
        )
        return 2

    try:
        groups = build_groups(clustered, bank, int(args.min_cluster))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    with_lever = [g for g in groups if g.lever is not None]
    if not with_lever:
        logger.error(
            "no (prompt, wave) group has two clusters of >= %d members; refusing to "
            "write an empty lever bank", args.min_cluster,
        )
        return 1

    print_summary(groups, bank.sites, int(args.min_cluster))

    # ── the npz ───────────────────────────────────────────────────────────────
    group_index = {g.key: i for i, g in enumerate(with_lever)}
    member_rows: list[dict[str, Any]] = []
    for group in groups:
        gi = group_index.get(group.key, -1)
        for key, cluster in zip(group.member_keys, group.clusters):
            gen = by_key[key]
            member_rows.append(
                {
                    "key": key,
                    "generation_id": gen.generation_id,
                    "corpus": gen.corpus,
                    "source_dir_label": gen.source_dir_label,
                    "prompt_id": gen.prompt_id,
                    "prompt_class": gen.prompt_class,
                    "seed_idx": gen.seed_idx,
                    "wave": gen.wave,
                    "cluster": cluster,
                    "group_index": gi,
                }
            )

    meta = {
        "stage": "expB15_build_levers",
        "hiddens": [str(p) for p in args.hiddens],
        "hidden_sources": bank.sources,
        "run_dirs": [str(p) for p in args.run_dirs],
        "sites": bank.sites,
        "site_convention": (
            "decoder-layer INPUT indices == attach_residual_write layer_idx; "
            f"equivalently the OUTPUT of layers {[s - 1 for s in bank.sites]}"
        ),
        "hidden_dim": bank.hidden_dim,
        "min_cluster": int(args.min_cluster),
        "wave_seed_boundary": int(args.wave_seed_boundary),
        "lever_definition": "mean_h(cluster 0) - mean_h(cluster 1), per site",
        "clustering": (
            "basin_check.two_means_split (basin_check.py:326) over one prompt's "
            "signatures from one run dir, in load_signatures order — the same rows in "
            "the same order the published basin report used"
        ),
        "basin_report_checks": checks,
        "n_clustered_gens": len(by_key),
        "n_gens_with_hidden": matched,
        "n_groups": len(groups),
        "n_groups_with_lever": len(with_lever),
        "n_members_missing_hidden": sum(len(g.missing_keys) for g in groups),
        "groups": [
            {
                "prompt_id": g.prompt_id,
                "wave": g.wave,
                "prompt_class": g.prompt_class,
                "corpus": g.corpus,
                "source_dir_label": g.source_dir_label,
                "cluster_sizes": list(g.cluster_sizes),
                "lever_norms": g.lever_norms,
                "group_index": group_index.get(g.key, -1),
                "missing_keys": g.missing_keys,
                "reason": g.reason,
            }
            for g in groups
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        levers=np.stack([g.lever for g in with_lever]).astype(np.float32),
        lever_norms=np.asarray(
            [g.lever_norms for g in with_lever], dtype=np.float32
        ),
        cluster_sums=np.stack([g.cluster_sums for g in with_lever]).astype(np.float64),
        cluster_sizes=np.asarray([list(g.cluster_sizes) for g in with_lever], dtype=np.int64),
        group_prompt_ids=np.asarray([g.prompt_id for g in with_lever], dtype=np.str_),
        group_waves=np.asarray([g.wave for g in with_lever], dtype=np.str_),
        group_prompt_classes=np.asarray([g.prompt_class for g in with_lever], dtype=np.str_),
        group_corpus_keys=np.asarray([g.corpus for g in with_lever], dtype=np.str_),
        member_keys=np.asarray([r["key"] for r in member_rows], dtype=np.str_),
        member_generation_ids=np.asarray(
            [r["generation_id"] for r in member_rows], dtype=np.int64
        ),
        member_corpus_keys=np.asarray([r["corpus"] for r in member_rows], dtype=np.str_),
        member_prompt_ids=np.asarray([r["prompt_id"] for r in member_rows], dtype=np.str_),
        member_prompt_classes=np.asarray(
            [r["prompt_class"] for r in member_rows], dtype=np.str_
        ),
        member_seed_idxs=np.asarray([r["seed_idx"] for r in member_rows], dtype=np.int64),
        member_waves=np.asarray([r["wave"] for r in member_rows], dtype=np.str_),
        member_clusters=np.asarray([r["cluster"] for r in member_rows], dtype=np.int64),
        member_group_index=np.asarray([r["group_index"] for r in member_rows], dtype=np.int64),
        sites=np.asarray(bank.sites, dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta), dtype=np.str_),
    )
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "DONE — %d/%d (prompt, wave) groups carry a lever over %d sites -> %s",
        len(with_lever), len(groups), len(bank.sites), args.out,
    )
    logger.info(
        "NEXT: evaluate the levers (--levers %s --hiddens %s) against the "
        "stage-1 gen dirs",
        args.out, " ".join(str(p) for p in args.hiddens),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
