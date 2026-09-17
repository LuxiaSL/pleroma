"""Exp B0 stage 4: do the K seeds of one prompt land in distinguishable basins?

The go/no-go, and — since the plan revision of 2026-09-16 — a go/no-go *per prompt
class*. B1 trains g on ``(signature, fork_tokens, continuation, prompt)`` pairs,
which is only worth doing if the K signatures of one prompt actually differ in a
structured way. If every seed of a prompt produced the same signature, B1's
"conditioning arms" would be conditioning on a constant.

Two readouts per prompt, and they are NOT equals:

(a) CLUSTERING — PRIMARY. Within-prompt pairwise cosine distance against a
    between-prompt reference, a silhouette-style separation score, and a 2-means
    split as a bimodality probe. Mode-agnostic, so it works whatever the prompt
    class. It takes BOTH a magnitude and a balance to count as forking, because
    2-means will split pure noise into two even halves and a balance-only rule
    would read a tight prompt as forked. Each prompt therefore gets a shape:
    ``tight`` (the seeds barely move), ``diffuse`` (they move but do not cluster —
    the open_generative underdetermination failure, informative but not a GO) or
    ``bimodal`` (they move AND split — a fork).

(b) MODE PROJECTION — SECONDARY COLOUR. Each gen's Δ = signature − the prompt's own
    seed-mean, projected onto the banked 3B LDA discriminants. Those discriminants
    were fit on *expository, system-prompted* battery gens, so they are a weak
    ruler for stance/interpretive/generative prompts: a low mode diversity here is
    not evidence of convergence. It never gates a verdict.

CALIBRATION ARM. ``convergent_control`` exists to test the measure, not the model.
Bare well-learned expository topics are predicted to cluster tightly; if their
within-prompt spread is NOT clearly below the fork classes', the spread measure is
reading noise and every other number in this report is suspect. That comparison is
computed, flagged loudly, and printed whether it passes or fails.

Feature-name equality against the discriminants' ``FULL_names`` is a hard assert on
both projection paths — the projection is meaningless in a misaligned space, and
stage 2's GATE 0 exists so this never fires in practice.

The plan gives a decision RULE but no numeric thresholds ("if basins don't spread
at temp 1.0, bump toward 1.2 and/or switch to ambiguous prompt frames before
scaling"), so the cut points live in ``--min-minority-frac``, ``--min-modes``,
``--class-go-fraction`` and ``--control-margin``, and are printed with every
verdict rather than buried.

Usage (local)::

    python -m pleroma.basin_check \\
        --run-dir outputs/expB0_3b/replay \\
        --discriminants outputs/v3_audit/factor_directions_3b.npz \\
        --out outputs/expB0_3b/basin_check.json
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from pleroma.fork_tokens import GenCell, load_cells
except ImportError:  # direct-script invocation rather than -m
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pleroma.fork_tokens import GenCell, load_cells

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB0.basin")

F64 = NDArray[np.float64]

#: Where to look for kvrot/sigbridge.py (mode_projection): the repo root
#: first (vendored kvrot/), then a sibling kv-rotation checkout.
DEFAULT_KVROT_PATHS: tuple[str, ...] = (
    ".",
    "../kv-rotation/src",
)

#: The class whose tight clustering calibrates the spread measure (plan rev 2026-09-16).
DEFAULT_CONTROL_CLASS = "convergent_control"


# ── Discriminants: sigbridge if importable, a cited local copy otherwise ───────


@dataclass(frozen=True)
class LocalFactorDirections:
    """Local stand-in for ``kvrot.sigbridge.FactorDirections`` (sigbridge.py:206)."""

    space: str
    labels: list[str]
    W: F64
    mean: F64
    scale: F64
    names: list[str]
    centroids: dict[str, F64] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalModeProjection:
    """Local stand-in for ``kvrot.sigbridge.ModeProjection`` (sigbridge.py:238)."""

    mode_fraction: float
    proj_w: list[float]
    delta_z_norm: float
    centroid_cosines: dict[str, float]


class FactorDirectionsLike(Protocol):
    """The slice of FactorDirections this module touches, either implementation."""

    labels: list[str]
    names: list[str]


def load_factor_directions_local(path: Path | str, space: str = "FULL") -> LocalFactorDirections:
    """Read factor_directions_*.npz. Replicates sigbridge.load_factor_directions:225."""
    if space not in ("FULL", "HAND"):
        raise ValueError(f"space must be FULL or HAND, got {space!r}")
    with np.load(Path(path).expanduser(), allow_pickle=True) as fd:
        labels = [str(x) for x in fd["labels"]]
        W = np.asarray(fd[f"{space}_W"], dtype=np.float64)
        mean = np.asarray(fd[f"{space}_mean"], dtype=np.float64)
        scale = np.asarray(fd[f"{space}_scale"], dtype=np.float64)
        names = [str(x) for x in fd[f"{space}_names"]]
        centroids = {
            m: np.asarray(fd[f"{space}_centroid_{m}"], dtype=np.float64)
            for m in labels
            if f"{space}_centroid_{m}" in fd
        }
    if W.shape[1] != len(names) or mean.shape[0] != len(names) or scale.shape[0] != len(names):
        raise ValueError(f"factor directions shape mismatch: W {W.shape}, names {len(names)}")
    return LocalFactorDirections(
        space=space, labels=labels, W=W, mean=mean, scale=scale, names=names, centroids=centroids
    )


def assert_names_match(feature_names: list[str], banked_names: list[str], space: str) -> None:
    """HARD ASSERT: the signature space must be the discriminants' space, exactly.

    Same failure text as sigbridge.mode_projection:264 so a mismatch reads the same
    whichever path produced it.
    """
    if list(feature_names) == list(banked_names):
        return
    first_bad = next(
        (i for i, (a, b) in enumerate(zip(feature_names, banked_names)) if a != b), None
    )
    detail = (
        f" ({feature_names[first_bad]!r} vs {banked_names[first_bad]!r})"
        if first_bad is not None
        else ""
    )
    raise AssertionError(
        f"feature-name alignment failed vs {space}_names: lengths "
        f"{len(feature_names)} vs {len(banked_names)}, first mismatch at index "
        f"{first_bad}{detail}"
    )


def mode_projection_local(
    delta: NDArray[np.floating], feature_names: list[str], fd: LocalFactorDirections
) -> LocalModeProjection:
    """Δz = Δ ⊘ scale projected onto the orthonormalized rows of W.

    Projection math only, replicated from ``kvrot.sigbridge.mode_projection``
    (sigbridge.py:252) for the case where sigbridge is not importable here (it
    imports torch at module level). Kept deliberately short and literal: scale
    divide with a zero-scale guard, QR of W's rows, projected fraction, centroid
    cosines against the centroid mean.
    """
    assert_names_match(feature_names, fd.names, fd.space)
    scale = np.where(np.abs(fd.scale) < 1e-12, 1.0, fd.scale)
    dz = np.asarray(delta, dtype=np.float64) / scale
    dz_norm = float(np.linalg.norm(dz))
    q, _ = np.linalg.qr(fd.W.T)          # [D, n_disc] orthonormal basis of the row space
    proj_orth = q.T @ dz
    frac = float((proj_orth @ proj_orth) / max(dz_norm**2, 1e-24))
    proj_w = fd.W @ dz
    cosines: dict[str, float] = {}
    if fd.centroids:
        cmean = np.mean(np.stack(list(fd.centroids.values())), axis=0)
        pn = float(np.linalg.norm(proj_w))
        for mode, centroid in fd.centroids.items():
            cc = centroid - cmean
            denom = pn * float(np.linalg.norm(cc))
            cosines[mode] = float((proj_w @ cc) / denom) if denom > 1e-12 else 0.0
    return LocalModeProjection(
        mode_fraction=frac,
        proj_w=[float(x) for x in proj_w],
        delta_z_norm=dz_norm,
        centroid_cosines=cosines,
    )


def resolve_projection_backend(
    discriminants: Path, space: str, kvrot_path: str | None
) -> tuple[Any, Any, str]:
    """(factor_directions, project_fn, provenance).

    Prefers kv-rotation's sigbridge — the module that defines this metric. Falls
    back to the local replication when sigbridge cannot be imported (it pulls in
    torch, which a local CPU box need not have).
    """
    candidates = [kvrot_path] if kvrot_path else []
    candidates.extend(DEFAULT_KVROT_PATHS)
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand).expanduser()
        if (path / "kvrot" / "sigbridge.py").exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from kvrot.sigbridge import (  # type: ignore[import-not-found]
            load_factor_directions,
            mode_projection,
        )

        return (
            load_factor_directions(discriminants, space=space),
            mode_projection,
            "kvrot.sigbridge.mode_projection",
        )
    except Exception as exc:  # noqa: BLE001 — torch-less box; the math is replicated
        logger.warning(
            "kvrot.sigbridge not importable (%s); using the local projection "
            "replicated from sigbridge.py:252", exc,
        )
        return (
            load_factor_directions_local(discriminants, space=space),
            mode_projection_local,
            "local replication of sigbridge.py:252",
        )


# ── Signatures ─────────────────────────────────────────────────────────────────


@dataclass
class SignatureSet:
    """Every gen's signature in one matrix, with the names they all share."""

    features: NDArray[np.float32]      # [n_gens, D]
    feature_names: list[str]           # [D]
    gen_ids: list[int]
    cells: list[GenCell]

    @property
    def n_gens(self) -> int:
        return int(self.features.shape[0])


def load_signatures(run_dir: Path) -> SignatureSet:
    """Load ``signatures/gen_NNN.npz`` + cells.json, asserting a shared name list.

    No dimension is asserted against a constant: D is whatever ``feature_names``
    says it is (dim policy, 2026-09-16).
    """
    cells = load_cells(run_dir)
    sig_dir = run_dir / "signatures"
    if not sig_dir.is_dir():
        raise FileNotFoundError(f"{sig_dir} not found (stage 2 writes it)")

    rows: list[NDArray[np.float32]] = []
    gen_ids: list[int] = []
    ordered_cells: list[GenCell] = []
    names: list[str] | None = None
    for path in sorted(sig_dir.glob("gen_*.npz")):
        gid = int(path.stem.split("_")[-1])
        if gid not in cells:
            logger.warning("%s has no cells.json entry — skipping", path.name)
            continue
        with np.load(path, allow_pickle=True) as npz:
            if "features" not in npz or "feature_names" not in npz:
                raise KeyError(f"{path}: needs 'features' and 'feature_names' (has {npz.files})")
            feats = np.asarray(npz["features"], dtype=np.float32).reshape(-1)
            these = [str(x) for x in npz["feature_names"]]
        if names is None:
            names = these
        elif these != names:
            raise AssertionError(
                f"{path.name}: feature_names differ from the first signature — "
                "these vectors do not live in one space"
            )
        if feats.shape[0] != len(these):
            raise ValueError(
                f"{path.name}: {feats.shape[0]} features but {len(these)} names"
            )
        rows.append(feats)
        gen_ids.append(gid)
        ordered_cells.append(cells[gid])

    if not rows or names is None:
        raise ValueError(f"no usable signatures under {sig_dir}")
    return SignatureSet(np.stack(rows), names, gen_ids, ordered_cells)


# ── Clustering: the PRIMARY spread measure ─────────────────────────────────────


def cosine_distance_matrix(x: NDArray[np.floating]) -> F64:
    """Pairwise 1 - cos over rows. Zero rows are treated as orthogonal to everything."""
    xf = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(xf, axis=1, keepdims=True)
    unit = xf / np.where(norms < 1e-12, 1.0, norms)
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def two_means_split(x: NDArray[np.floating], max_iter: int = 100) -> tuple[list[int], int]:
    """Deterministic 2-means on cosine distance; returns (labels, n_iterations).

    Seeded with the two most distant rows, so there is no RNG and no seed to
    report. A bimodality probe, and named one: balanced split sizes are evidence of
    forking, not proof of two basins.
    """
    xf = np.asarray(x, dtype=np.float64)
    n = xf.shape[0]
    if n < 2:
        return [0] * n, 0
    dist = cosine_distance_matrix(xf)
    a, b = np.unravel_index(int(np.argmax(dist)), dist.shape)
    centroids = xf[[a, b]].copy()
    labels = np.zeros(n, dtype=np.int64)
    for it in range(1, max_iter + 1):
        cn = np.linalg.norm(centroids, axis=1, keepdims=True)
        cu = centroids / np.where(cn < 1e-12, 1.0, cn)
        xn = np.linalg.norm(xf, axis=1, keepdims=True)
        xu = xf / np.where(xn < 1e-12, 1.0, xn)
        new_labels = np.argmax(np.clip(xu @ cu.T, -1.0, 1.0), axis=1)
        if it > 1 and np.array_equal(new_labels, labels):
            return [int(v) for v in labels], it
        labels = new_labels
        for k in (0, 1):
            members = xf[labels == k]
            if members.shape[0]:
                centroids[k] = members.mean(axis=0)
    return [int(v) for v in labels], max_iter


def dominant_mode_entropy(dominants: list[str], n_labels: int) -> tuple[float, float, int]:
    """(entropy_nats, normalized, n_distinct) over the dominant-mode counts."""
    if not dominants:
        return 0.0, 0.0, 0
    _, counts = np.unique(np.array(dominants), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    ent = float(-np.sum(p * np.log(p)))
    ceiling = float(np.log(max(min(n_labels, len(dominants)), 2)))
    return ent, (ent / ceiling if ceiling > 0 else 0.0), int(counts.shape[0])


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


# ── Per-prompt, per-class, and the calibration line ────────────────────────────


def analyse_prompts(
    sigs: SignatureSet,
    fd: Any,
    project: Any,
    min_minority_frac: float,
    min_modes: int,
    min_spread_ratio: float,
) -> list[dict[str, Any]]:
    """Per-prompt_id spread (primary) + mode projection (secondary)."""
    prompt_ids = [c.prompt_id for c in sigs.cells]
    unique_prompts = sorted(set(prompt_ids), key=lambda p: prompt_ids.index(p))
    pid_array = np.array(prompt_ids, dtype=object)
    dist = cosine_distance_matrix(sigs.features)
    labels: list[str] = [str(x) for x in fd.labels]

    rows: list[dict[str, Any]] = []
    for pid in unique_prompts:
        members = np.nonzero(pid_array == pid)[0]
        others = np.nonzero(pid_array != pid)[0]
        sub = sigs.features[members]
        k = int(members.shape[0])

        within = (
            dist[np.ix_(members, members)][np.triu_indices(k, k=1)]
            if k > 1
            else np.array([], dtype=np.float64)
        )
        between = dist[np.ix_(members, others)].reshape(-1) if others.size else np.array([])

        # Silhouette-style: own-prompt mean distance vs the nearest other prompt's.
        sil: list[float] = []
        for i, gi in enumerate(members):
            if k < 2:
                continue
            a = float(np.mean(np.delete(dist[gi, members], i)))
            b_candidates = [
                float(np.mean(dist[gi, np.nonzero(pid_array == other)[0]]))
                for other in unique_prompts
                if other != pid
            ]
            if not b_candidates:
                continue
            b = min(b_candidates)
            denom = max(a, b)
            sil.append((b - a) / denom if denom > 1e-12 else 0.0)

        split_labels, split_iters = two_means_split(sub)
        sizes = sorted([split_labels.count(0), split_labels.count(1)], reverse=True)
        minority_frac = sizes[1] / k if k else 0.0

        # Mode projection of each seed's delta from the prompt's own seed-mean.
        prompt_mean = sub.mean(axis=0)
        seeds: list[dict[str, Any]] = []
        dominants: list[str] = []
        for i, gi in enumerate(members):
            proj = project(sub[i] - prompt_mean, sigs.feature_names, fd)
            cosines: dict[str, float] = dict(proj.centroid_cosines)
            dominant = max(cosines, key=lambda m: cosines[m]) if cosines else None
            if dominant is not None:
                dominants.append(dominant)
            seeds.append(
                {
                    "generation_id": sigs.gen_ids[gi],
                    "seed_idx": sigs.cells[gi].seed_idx,
                    "cluster": split_labels[i],
                    "dominant_mode": dominant,
                    "dominant_cosine": float(cosines[dominant]) if dominant else None,
                    "mode_fraction": float(proj.mode_fraction),
                    "delta_z_norm": float(proj.delta_z_norm),
                    "centroid_cosines": {m: float(v) for m, v in cosines.items()},
                }
            )

        ent, ent_norm, n_distinct = dominant_mode_entropy(dominants, len(labels))

        # PRIMARY (clustering) decides the verdict; the mode arm is recorded beside
        # it as colour, never as a gate — the discriminants are a weak ruler off
        # the expository classes they were fit on.
        #
        # Balance ALONE is not enough: 2-means will happily split pure noise into
        # two even halves, so a tight prompt would read as forked. The magnitude
        # test is scale-free — how much of the typical BETWEEN-prompt distance does
        # one prompt's own seeds span — and the two together name three shapes:
        #   tight   : the seeds barely move (the convergent_control prediction)
        #   diffuse : they move, but not into clusters (the open_generative
        #             underdetermination failure — informative, not a GO)
        #   bimodal : they move AND split (a fork)
        within_mean = float(np.mean(within)) if within.size else None
        between_mean = float(np.mean(between)) if between.size else None
        spread_ratio = (
            within_mean / between_mean
            if within_mean is not None and between_mean is not None and between_mean > 1e-12
            else None
        )
        big_enough = spread_ratio is not None and spread_ratio >= min_spread_ratio
        balanced = minority_frac >= min_minority_frac
        if k < 2 or not big_enough:
            shape = "tight"
        elif not balanced:
            shape = "diffuse"
        else:
            shape = "bimodal"
        spread_ok = shape == "bimodal"
        modes_ok = n_distinct >= min_modes

        rows.append(
            {
                "prompt_id": pid,
                "prompt_class": sigs.cells[members[0]].prompt_class,
                "prompt": sigs.cells[members[0]].prompt,
                "n_seeds": k,
                "within_cosine_distance_mean": within_mean,
                "within_cosine_distance_std": float(np.std(within)) if within.size else None,
                "within_cosine_distance_max": float(np.max(within)) if within.size else None,
                "between_cosine_distance_mean": between_mean,
                "spread_ratio": spread_ratio,
                "spread_shape": shape,
                "separation_score": float(np.mean(sil)) if sil else None,
                "two_means_sizes": sizes,
                "two_means_minority_fraction": float(minority_frac),
                "two_means_iterations": split_iters,
                "dominant_mode_counts": {
                    m: int(dominants.count(m)) for m in sorted(set(dominants))
                },
                "mode_diversity_nats": ent,
                "mode_diversity_normalized": ent_norm,
                "n_distinct_dominant_modes": n_distinct,
                "mode_fraction_mean": _mean([s["mode_fraction"] for s in seeds]),
                "spread_ok": bool(spread_ok),
                "modes_ok_secondary": bool(modes_ok),
                "verdict": "GO" if spread_ok else "NO-GO",
                "seeds": seeds,
            }
        )
    return rows


def aggregate_classes(
    per_prompt: list[dict[str, Any]], control_class: str, class_go_fraction: float
) -> list[dict[str, Any]]:
    """Roll the per-prompt rows up to the experimental variable: prompt class."""
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in per_prompt:
        by_class.setdefault(row["prompt_class"], []).append(row)

    out: list[dict[str, Any]] = []
    for cls in sorted(by_class):
        group = by_class[cls]
        n_go = sum(1 for r in group if r["spread_ok"])
        go_fraction = n_go / len(group)
        if go_fraction >= class_go_fraction:
            verdict = "GO"
        elif go_fraction > 0:
            verdict = "MARGINAL"
        else:
            verdict = "NO-GO"
        is_control = cls == control_class
        if is_control:
            reading = (
                "CONTROL: a GO here is a WARNING about the measure, not a result "
                "about the model — see the calibration block."
            )
        else:
            reading = "fork class: GO means these prompts are usable for B1."
        out.append(
            {
                "prompt_class": cls,
                "is_control": is_control,
                "n_prompts": len(group),
                "n_gens": sum(r["n_seeds"] for r in group),
                "within_cosine_distance_mean": _mean(
                    [r["within_cosine_distance_mean"] for r in group]
                ),
                "between_cosine_distance_mean": _mean(
                    [r["between_cosine_distance_mean"] for r in group]
                ),
                "spread_ratio_mean": _mean([r["spread_ratio"] for r in group]),
                "spread_shape_counts": {
                    shape: sum(1 for r in group if r["spread_shape"] == shape)
                    for shape in ("tight", "diffuse", "bimodal")
                },
                "separation_score_mean": _mean([r["separation_score"] for r in group]),
                "two_means_minority_fraction_mean": _mean(
                    [r["two_means_minority_fraction"] for r in group]
                ),
                "n_prompts_spread_ok": n_go,
                "go_fraction": go_fraction,
                "mode_diversity_nats_mean": _mean([r["mode_diversity_nats"] for r in group]),
                "n_distinct_dominant_modes_mean": _mean(
                    [float(r["n_distinct_dominant_modes"]) for r in group]
                ),
                "verdict": verdict,
                "reading": reading,
                "prompt_ids": [r["prompt_id"] for r in group],
            }
        )
    return out


def calibration_check(
    per_prompt: list[dict[str, Any]], control_class: str, control_margin: float
) -> dict[str, Any]:
    """Is the control class clearly tighter than the fork classes?

    The spread measure is only trustworthy if the arm predicted to converge
    actually reads as converged. ``control_margin`` is how much below the fork
    classes' mean within-prompt distance the control has to sit before we call it
    "clearly below": ratio <= 1 - margin.
    """
    control = [r for r in per_prompt if r["prompt_class"] == control_class]
    fork = [r for r in per_prompt if r["prompt_class"] != control_class]
    if not control:
        return {
            "control_class": control_class,
            "status": "absent",
            "message": (
                f"no {control_class!r} prompts in this run — the spread measure is "
                "UNCALIBRATED; every verdict below is relative, not absolute."
            ),
        }
    if not fork:
        return {
            "control_class": control_class,
            "status": "absent",
            "message": (
                f"only {control_class!r} prompts in this run — nothing to calibrate "
                "against."
            ),
        }

    control_within = _mean([r["within_cosine_distance_mean"] for r in control])
    fork_within = _mean([r["within_cosine_distance_mean"] for r in fork])
    control_minority = _mean([r["two_means_minority_fraction"] for r in control])
    fork_minority = _mean([r["two_means_minority_fraction"] for r in fork])
    if control_within is None or fork_within is None or fork_within <= 1e-12:
        return {
            "control_class": control_class,
            "status": "absent",
            "message": "within-prompt distances are degenerate; cannot calibrate.",
            "control_within_mean": control_within,
            "fork_within_mean": fork_within,
        }

    ratio = control_within / fork_within
    ok = ratio <= (1.0 - control_margin)
    message = (
        f"control {control_class!r} within-prompt spread {control_within:.4f} vs fork "
        f"classes {fork_within:.4f} (ratio {ratio:.2f}, needs <= {1.0 - control_margin:.2f})"
    )
    if not ok:
        message = (
            "CALIBRATION FLAG — " + message + ". The arm predicted to CONVERGE is not "
            "clearly tighter than the fork classes, so the spread measure may be "
            "reading noise rather than basins. Treat every verdict below as suspect "
            "until this is resolved."
        )
    return {
        "control_class": control_class,
        "status": "ok" if ok else "FLAG",
        "control_within_mean": control_within,
        "fork_within_mean": fork_within,
        "ratio": ratio,
        "margin": control_margin,
        "control_minority_fraction_mean": control_minority,
        "fork_minority_fraction_mean": fork_minority,
        "message": message,
    }


def analyse(
    sigs: SignatureSet,
    fd: Any,
    project: Any,
    min_minority_frac: float,
    min_modes: int,
    control_class: str = DEFAULT_CONTROL_CLASS,
    control_margin: float = 0.25,
    class_go_fraction: float = 0.5,
    min_spread_ratio: float = 0.10,
) -> dict[str, Any]:
    """The full stage-4 report: per prompt, per class, plus the calibration line."""
    assert_names_match(sigs.feature_names, list(fd.names), getattr(fd, "space", "FULL"))

    per_prompt = analyse_prompts(
        sigs, fd, project, min_minority_frac, min_modes, min_spread_ratio
    )
    per_class = aggregate_classes(per_prompt, control_class, class_go_fraction)
    calibration = calibration_check(per_prompt, control_class, control_margin)

    fork_classes = [c for c in per_class if not c["is_control"]]
    go_classes = [c["prompt_class"] for c in fork_classes if c["verdict"] == "GO"]
    marginal = [c["prompt_class"] for c in fork_classes if c["verdict"] == "MARGINAL"]

    if go_classes:
        decision = (
            f"GO for {', '.join(go_classes)} — those classes fork at these settings; "
            "scale those and hand the shortlist to B1."
        )
    elif marginal:
        decision = (
            f"MARGINAL — only some prompts in {', '.join(marginal)} fork. Per the "
            "plan's decision rule, bump temperature toward 1.2 and/or lean on the "
            "more strongly forking classes (decision_continuation, interpretive_fork) "
            "before scaling; the GO prompts are still usable as a B1 pilot."
        )
    else:
        decision = (
            "NO-GO — no fork class spreads at these settings. Per the plan's decision "
            "rule: bump temperature toward 1.2 and/or move to more strongly "
            "underdetermined prompt frames before scaling."
        )
    if calibration["status"] == "FLAG":
        decision = "UNCALIBRATED — " + decision + " (See the calibration flag first.)"

    # B1 shortlist: forking prompts from the fork classes only, widest spread first.
    shortlist = sorted(
        (
            r
            for r in per_prompt
            if r["verdict"] == "GO" and r["prompt_class"] != control_class
        ),
        key=lambda r: -(r["within_cosine_distance_mean"] or 0.0),
    )

    return {
        "n_gens": sigs.n_gens,
        "n_prompts": len(per_prompt),
        "n_classes": len(per_class),
        "feature_dim": len(sigs.feature_names),
        "modes": [str(x) for x in fd.labels],
        "measure_note": (
            "PRIMARY = mode-agnostic clustering. A prompt is 'bimodal' (spread_ok) "
            "only if its seeds move far enough (spread_ratio = within/between "
            "cosine distance >= min_spread_ratio) AND split evenly (2-means minority "
            ">= min_minority_frac); 'diffuse' = moved without clustering (the "
            "underdetermination failure), 'tight' = did not move. SECONDARY = "
            "mode_projection against discriminants fit on expository, "
            "system-prompted gens: a weak ruler off that distribution, reported as "
            "colour and never gating a verdict."
        ),
        "thresholds": {
            "min_minority_frac": min_minority_frac,
            "min_spread_ratio": min_spread_ratio,
            "min_modes": min_modes,
            "class_go_fraction": class_go_fraction,
            "control_margin": control_margin,
        },
        "calibration": calibration,
        "overall": {
            "within_cosine_distance_mean": _mean(
                [r["within_cosine_distance_mean"] for r in per_prompt]
            ),
            "between_cosine_distance_mean": _mean(
                [r["between_cosine_distance_mean"] for r in per_prompt]
            ),
            "separation_score_mean": _mean([r["separation_score"] for r in per_prompt]),
            "spread_ratio_mean": _mean([r["spread_ratio"] for r in per_prompt]),
            "spread_shape_counts": {
                shape: sum(1 for r in per_prompt if r["spread_shape"] == shape)
                for shape in ("tight", "diffuse", "bimodal")
            },
            "n_prompts_spread_ok": sum(1 for r in per_prompt if r["spread_ok"]),
            "go_classes": go_classes,
            "decision": decision,
        },
        "b1_shortlist": [
            {
                "prompt_id": r["prompt_id"],
                "prompt_class": r["prompt_class"],
                "within_cosine_distance_mean": r["within_cosine_distance_mean"],
                "two_means_sizes": r["two_means_sizes"],
            }
            for r in shortlist
        ],
        "classes": per_class,
        "prompts": per_prompt,
    }


def print_summary(report: dict[str, Any]) -> None:
    """The human-readable version; the JSON is the record."""
    overall = report["overall"]
    thresholds = report["thresholds"]
    logger.info("=" * 92)
    logger.info(
        "BASIN CHECK — %d gens, %d prompts, %d classes, %d-dim signatures",
        report["n_gens"], report["n_prompts"], report["n_classes"], report["feature_dim"],
    )
    logger.info(
        "thresholds: spread ratio >= %.2f AND minority fraction >= %.2f | class GO at "
        ">= %.0f%% of its prompts | control margin %.2f",
        thresholds["min_spread_ratio"], thresholds["min_minority_frac"],
        100 * thresholds["class_go_fraction"], thresholds["control_margin"],
    )
    logger.info("PRIMARY measure = clustering; mode projection is secondary colour.")

    logger.info("-" * 92)
    logger.info(
        "%-8s %-22s %5s %7s %7s %6s %6s %-8s %5s  %s",
        "prompt", "class", "seeds", "within", "betwn", "ratio", "split", "shape",
        "modes", "verdict",
    )
    for row in report["prompts"]:
        logger.info(
            "%-8s %-22s %5d %7s %7s %6s %6s %-8s %5d  %s",
            row["prompt_id"], row["prompt_class"][:22], row["n_seeds"],
            _fmt(row["within_cosine_distance_mean"]),
            _fmt(row["between_cosine_distance_mean"]),
            _fmt(row["spread_ratio"], places=2),
            f"{row['two_means_sizes'][0]}/{row['two_means_sizes'][1]}",
            row["spread_shape"],
            row["n_distinct_dominant_modes"],
            row["verdict"],
        )

    logger.info("-" * 92)
    logger.info(
        "%-22s %8s %7s %6s %-24s %8s  %s",
        "class", "prompts", "within", "ratio", "shapes (tight/diff/bimod)", "spread", "verdict",
    )
    for row in report["classes"]:
        counts = row["spread_shape_counts"]
        logger.info(
            "%-22s %8d %7s %6s %-24s %8s  %s%s",
            row["prompt_class"], row["n_prompts"],
            _fmt(row["within_cosine_distance_mean"]),
            _fmt(row["spread_ratio_mean"], places=2),
            f"{counts['tight']}/{counts['diffuse']}/{counts['bimodal']}",
            f"{row['n_prompts_spread_ok']}/{row['n_prompts']}",
            row["verdict"],
            "  [CONTROL]" if row["is_control"] else "",
        )

    logger.info("-" * 92)
    calibration = report["calibration"]
    if calibration["status"] == "FLAG":
        logger.error("!!! %s", calibration["message"])
    elif calibration["status"] == "absent":
        logger.warning("CALIBRATION: %s", calibration["message"])
    else:
        logger.info("CALIBRATION ok: %s", calibration["message"])
    logger.info(
        "overall: within %s vs between %s, separation %s; %d/%d prompts spread",
        _fmt(overall["within_cosine_distance_mean"]),
        _fmt(overall["between_cosine_distance_mean"]),
        _fmt(overall["separation_score_mean"], sign=True),
        overall["n_prompts_spread_ok"], report["n_prompts"],
    )
    logger.info(
        "B1 shortlist: %s", [r["prompt_id"] for r in report["b1_shortlist"]] or "(empty)"
    )
    logger.info("DECISION: %s", overall["decision"])
    logger.info("=" * 92)


def _fmt(value: float | None, sign: bool = False, places: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{places}f}" if sign else f"{value:.{places}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="stage-2 out-dir (signatures/ + cells.json)"
    )
    parser.add_argument(
        "--discriminants", type=Path, required=True, help="factor_directions_3b.npz"
    )
    parser.add_argument("--out", type=Path, required=True, help="output JSON report")
    parser.add_argument("--space", default="FULL", choices=["FULL", "HAND"])
    parser.add_argument(
        "--min-minority-frac",
        type=float,
        default=0.25,
        help="smallest 2-means cluster, as a fraction of K, for 'spread_ok' (PRIMARY)",
    )
    parser.add_argument(
        "--min-spread-ratio",
        type=float,
        default=0.10,
        help=(
            "within/between cosine distance a prompt's seeds must span before the "
            "2-means split counts as anything (PRIMARY; 2-means will split pure "
            "noise evenly, so balance alone is not evidence)"
        ),
    )
    parser.add_argument(
        "--min-modes",
        type=int,
        default=2,
        help="distinct dominant modes for 'modes_ok_secondary' (reported, never gating)",
    )
    parser.add_argument(
        "--class-go-fraction",
        type=float,
        default=0.5,
        help="fraction of a class's prompts that must spread for a class-level GO",
    )
    parser.add_argument(
        "--control-class",
        default=DEFAULT_CONTROL_CLASS,
        help="the class predicted to converge; calibrates the spread measure",
    )
    parser.add_argument(
        "--control-margin",
        type=float,
        default=0.25,
        help=(
            "how far below the fork classes the control's within-prompt spread must "
            "sit (ratio <= 1 - margin) before calibration counts as passed"
        ),
    )
    parser.add_argument(
        "--kvrot-path", default=None, help="kv-rotation src/ dir, for sigbridge.mode_projection"
    )
    args = parser.parse_args()

    if not 0 < args.min_minority_frac <= 0.5:
        logger.error("--min-minority-frac must be in (0, 0.5], got %s", args.min_minority_frac)
        return 2
    if args.min_spread_ratio < 0:
        logger.error("--min-spread-ratio must be >= 0, got %s", args.min_spread_ratio)
        return 2
    if args.min_modes < 1:
        logger.error("--min-modes must be >= 1, got %s", args.min_modes)
        return 2
    if not 0 < args.class_go_fraction <= 1:
        logger.error("--class-go-fraction must be in (0, 1], got %s", args.class_go_fraction)
        return 2
    if not 0 <= args.control_margin < 1:
        logger.error("--control-margin must be in [0, 1), got %s", args.control_margin)
        return 2

    try:
        sigs = load_signatures(args.run_dir)
    except (OSError, KeyError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        logger.error("could not load signatures: %s", exc)
        return 1
    logger.info(
        "%d signatures x %d features from %s", sigs.n_gens, len(sigs.feature_names), args.run_dir
    )

    try:
        fd, project, provenance = resolve_projection_backend(
            args.discriminants, args.space, args.kvrot_path
        )
    except (OSError, KeyError, ValueError) as exc:
        logger.error("could not load --discriminants %s: %s", args.discriminants, exc)
        return 1
    logger.info("mode projection via %s (%s space)", provenance, args.space)

    try:
        report = analyse(
            sigs,
            fd,
            project,
            args.min_minority_frac,
            args.min_modes,
            args.control_class,
            args.control_margin,
            args.class_go_fraction,
            args.min_spread_ratio,
        )
    except AssertionError as exc:
        logger.error("%s", exc)
        logger.error(
            "refusing to report a projection in a misaligned space — stage 2's GATE 0 "
            "should have caught this; check --calib-dir and the v3 family config."
        )
        return 3
    except ValueError as exc:
        logger.error("basin analysis failed: %s", exc)
        return 1

    report["stage"] = "expB0_basin_check"
    report["run_dir"] = str(args.run_dir)
    report["discriminants"] = str(args.discriminants)
    report["projection_backend"] = provenance

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print_summary(report)
    logger.info("report -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
