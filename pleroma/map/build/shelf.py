"""The shelf statistics: standardisation stats + the ruler, computed ONCE.

The v1a export needs these statistics but none of a wide map's W, so they are
computed here rather than read back out of a whole second map fitted first
(`ShelfStats.from_wide_map` still reads a banked one). This module is the part
of `wide` that v1a consumes:

* ``v3_mu / v3_sd / v3_dead``: the corpus standardiser over TRAIN rows of
  z_full, computed in float32 exactly as `pleroma.map.build.pairs` does. The recomputed
  z_corpus is verified against the pairs matrix.
* ``bins_mu / bins_sd / bins_dead``: the bins standardiser over the
  lever_group-joined train rows. It is a no-op in v1a (bins rows of U are
  zero) but LoomMap's input contract requires it.
* ``norm_ref``: the ruler, the median per-site lever norm of a bank
  (docs/FINDINGS.md §2 says why alpha needs it).

`wide` and `export` both call `compute_shelf`, so the two cannot disagree. Every array is returned in its ON-DISK dtype (float32 stats,
bool masks). A folded export is therefore byte-identical in these arrays to
one that read them back out of a wide map.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: `pleroma.map.build.pairs`' dead cutoff on the corpus SD (float32 space).
V3_DEAD_SD: float = 1e-8
#: `pleroma.map.build.regress`' dead cutoff on the bins SD.
BINS_DEAD_SD: float = 1e-9
#: Max |Δ| allowed between recomputed z_corpus and the pairs matrix.
Z_VERIFY_TOL: float = 1e-3


class ShelfError(ValueError):
    """The inputs cannot produce trustworthy shelf statistics."""


class ShelfRecipeError(ShelfError):
    """The pairs dir was not built with ``--standardize corpus``."""


@dataclass(frozen=True)
class ShelfStats:
    """What a LoomMap carries besides W: standardisers, ruler, provenance."""

    v3_mu: np.ndarray        # float32 [z]
    v3_sd: np.ndarray        # float32 [z]
    v3_dead: np.ndarray      # bool [z]
    bins_mu: np.ndarray      # float32 [bins]
    bins_sd: np.ndarray      # float32 [bins]
    bins_dead: np.ndarray    # bool [bins]
    norm_ref: np.ndarray     # float32 [sites]
    sites: list[int]
    discriminants: str
    discriminants_sha256: str
    bins_config: Any
    source: str              # "computed" | path of the wide map read

    @property
    def z_dim(self) -> int:
        return int(self.v3_mu.size)

    @classmethod
    def from_wide_map(cls, path: Path) -> "ShelfStats":
        """Read the stats back out of a banked wide map (the legacy `--wide-map` path).

        Raises `ShelfError` when the file or any required key is unreadable."""
        try:
            with np.load(path, allow_pickle=True) as wm:
                meta = json.loads(str(wm["meta"]))
                return cls(
                    v3_mu=np.asarray(wm["v3_mu"], dtype=np.float32),
                    v3_sd=np.asarray(wm["v3_sd"], dtype=np.float32),
                    v3_dead=np.asarray(wm["v3_dead"], dtype=bool),
                    bins_mu=np.asarray(wm["bins_mu"], dtype=np.float32),
                    bins_sd=np.asarray(wm["bins_sd"], dtype=np.float32),
                    bins_dead=np.asarray(wm["bins_dead"], dtype=bool),
                    norm_ref=np.asarray(wm["norm_ref"], dtype=np.float32),
                    sites=[int(x) for x in wm["sites"]],
                    discriminants=str(meta["discriminants"]),
                    discriminants_sha256=str(meta["discriminants_sha256"]),
                    bins_config=meta.get("bins_config"),
                    source=str(path),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise ShelfError(f"cannot read shelf stats from wide map {path}: {exc}") from exc


@dataclass
class ShelfBuild:
    """`compute_shelf`'s full output: the stats, plus what `wide` fits on."""

    stats: ShelfStats
    z_corpus: np.ndarray             # float64 [n_pairs, z], dead -> 0
    rows: list[int]                  # pair indices joined to a lever group
    bins_std: np.ndarray             # float64 [len(rows), bins], dead -> 0
    targets: np.ndarray              # float64 [len(rows), S*H] = m_sign * lever
    verify_err: float
    extras: dict[str, Any] = field(default_factory=dict)


def ruler_from_bank(bank: Path) -> np.ndarray:
    """The ruler (alpha=1): the median per-site norm of a lever bank, as float32.

    Raises `ShelfError` when the bank is unreadable, its levers are not
    ``[G, S, H]``, or the ruler is not positive and finite."""
    try:
        with np.load(bank, allow_pickle=False) as ref:
            levers = np.asarray(ref["levers"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ShelfError(f"cannot read levers from ruler bank {bank}: {exc}") from exc
    if levers.ndim != 3:
        raise ShelfError(f"ruler bank levers must be [G, S, H], got {levers.shape}")
    norm_ref = np.nanmedian(np.linalg.norm(levers, axis=2), axis=0)
    if not np.all(np.isfinite(norm_ref)) or np.any(norm_ref <= 0):
        raise ShelfError(f"ruler from {bank} is not positive-finite: {norm_ref}")
    return norm_ref.astype(np.float32)


def compute_shelf(
    pairs_dir: Path,
    levers: Path,
    bins: Path,
    discriminants: Path,
    norm_ref_bank: Path,
    verify_rows: int = 25,
) -> ShelfBuild:
    """Recompute the wide map's standardisers and ruler from the raw inputs.

    Raises `ShelfError` on every refusal (`ShelfRecipeError` for a pairs dir
    not built with ``--standardize corpus``); callers decide how to exit.
    """
    from pleroma.map.build.hiddens import corpus_key
    from pleroma.map.build.regress import corpus_of_run_dir
    from pleroma.map.build.pairs import load_pairs

    loaded = load_pairs(pairs_dir)
    standardize = loaded.meta.get("params", {}).get("standardize")
    if standardize != "corpus":
        # ★ DO NOT relax this refusal to "fix" a mismatched FULL_scale.
        # The discriminants' standardiser IS badly matched to our corpus —
        # measured over 34,014 train rows, post-transform per-feature
        # SD is p50 1.10 / p95 19.4 / p99 93.7 / max 4053, with 14.7% of
        # features off by >3x and median location 3.0 fit-sigmas out (max
        # 13,780). It is harmless ONLY because the corpus stage below telescopes
        # it away: replacing the whole discriminants artifact with mean 0 /
        # scale 1 moves the shipped z-matrix by min row cosine 0.999999990821
        # (max |delta| 2.29e-3, float32 noise on near-constant gate dims).
        # Drop the corpus stage and you get median |z| 3.29, 36.5% of cells
        # beyond |z|=10, and LayerNorm pairwise cosine 0.949: inputs that
        # near-collinear make the fit return a null. The algebra is at the
        # ``--standardize corpus`` branch of `pleroma.map.build.pairs.build`.
        raise ShelfRecipeError(
            f"shelf stats replicate --standardize corpus; pairs dir says {standardize!r}")

    # ── stage 1: z_full from raw signatures, then the corpus standardiser ────
    with np.load(discriminants, allow_pickle=True) as npz:
        full_mean = np.asarray(npz["FULL_mean"], dtype=np.float64)
        full_scale = np.asarray(npz["FULL_scale"], dtype=np.float64)
    degenerate = full_scale < 1e-12
    scale_safe = np.where(degenerate, 1.0, full_scale)
    disc_sha = hashlib.sha256(Path(discriminants).read_bytes()).hexdigest()

    z_full = np.zeros((len(loaded.pairs), len(full_mean)), dtype=np.float64)
    for i, p in enumerate(loaded.pairs):
        sig = Path(p.run_dir) / p.signature_path
        with np.load(sig, allow_pickle=True) as npz:
            x = np.asarray(npz["features"], dtype=np.float64)
        zf = (x - full_mean) / scale_safe
        zf[degenerate] = 0.0
        z_full[i] = zf
    # MIRROR `pleroma.map.build.pairs`: float32 BEFORE the corpus stats
    # (near-constant dims amplify cast noise by 1/sd, so the other order
    # breaks the z verification below).
    z_full32 = z_full.astype(np.float32)
    train_mask = np.array([p.split == "train" for p in loaded.pairs])
    if not train_mask.any():
        raise ShelfError("no train rows in the pairs dir — no standardiser to fit")
    v3_mu = z_full32[train_mask].mean(axis=0)
    v3_sd = z_full32[train_mask].std(axis=0)
    v3_dead = v3_sd < V3_DEAD_SD
    sd_safe = np.where(v3_dead, 1.0, v3_sd)
    z_corpus = ((z_full32 - v3_mu) / sd_safe).astype(np.float32).astype(np.float64)
    z_corpus[:, v3_dead] = 0.0

    rng = np.random.default_rng(0)
    idx = rng.choice(len(loaded.pairs), size=min(verify_rows, len(loaded.pairs)),
                     replace=False)
    err = float(np.max(np.abs(z_corpus[idx] - loaded.z[idx].astype(np.float64))))
    if err > Z_VERIFY_TOL:
        raise ShelfError(
            f"recomputed z differs from pairs_z (max |Δ| {err:.3g}) — the "
            "standardization recipe drifted; refusing stats whose inference "
            "path is not the training path")
    logger.info("z recipe verified on %d rows (max |Δ| %.2e)", len(idx), err)

    # ── bins block + lever join (the regression's conventions) ───────────────
    with np.load(bins, allow_pickle=True) as ex:
        ex_feat = np.asarray(ex["features"], dtype=np.float64)
        ex_of = {(corpus_key(str(r)), int(g)): k for k, (r, g) in
                 enumerate(zip(ex["run_dir"], ex["generation_id"]))}
        bins_config = json.loads(str(ex["config"]))

    with np.load(levers, allow_pickle=True) as lev:
        m_corpus = [str(x) for x in lev["member_corpus_keys"]]
        m_genid = np.asarray(lev["member_generation_ids"], dtype=int)
        m_group = np.asarray(lev["member_group_index"], dtype=int)
        m_sign = 1.0 - 2.0 * np.asarray(lev["member_clusters"], dtype=np.float64)
        lever_arr = np.asarray(lev["levers"], dtype=np.float64)
        sites = [int(x) for x in lev["sites"]]
    flat_levers = lever_arr.reshape(lever_arr.shape[0], -1)
    member_of = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}

    rows: list[int] = []
    bins_rows: list[int] = []
    targets: list[np.ndarray] = []
    joined_train: list[bool] = []
    for i, p in enumerate(loaded.pairs):
        corpus = corpus_of_run_dir(p.run_dir)
        j = member_of.get((corpus, p.generation_id))
        if j is None or int(m_group[j]) < 0:
            continue
        k = ex_of.get((corpus, p.generation_id))
        if k is None:
            raise ShelfError(f"gen {p.pair_key} missing from bins npz")
        rows.append(i)
        bins_rows.append(k)
        targets.append(m_sign[j] * flat_levers[int(m_group[j])])
        joined_train.append(bool(train_mask[i]))
    if not rows:
        raise ShelfError("no pair joined a lever group — nothing to standardise bins on")
    logger.info("joined %d gens", len(rows))

    extra = ex_feat[np.asarray(bins_rows)]
    if not np.isfinite(extra).all():
        raise ShelfError("non-finite bins rows among joined gens")
    jt = np.asarray(joined_train)
    if not jt.any():
        raise ShelfError("no joined TRAIN rows — the bins standardiser is undefined")
    bins_mu = extra[jt].mean(axis=0)
    bins_sd = extra[jt].std(axis=0)
    bins_dead = bins_sd < BINS_DEAD_SD
    bins_std = (extra - bins_mu) / np.where(bins_dead, 1.0, bins_sd)
    bins_std[:, bins_dead] = 0.0

    norm_ref = ruler_from_bank(norm_ref_bank)
    if norm_ref.size != len(sites):
        raise ShelfError(f"ruler bank has {norm_ref.size} sites, levers have {len(sites)}")

    stats = ShelfStats(
        v3_mu=v3_mu.astype(np.float32), v3_sd=v3_sd.astype(np.float32),
        v3_dead=np.asarray(v3_dead, dtype=bool),
        bins_mu=bins_mu.astype(np.float32), bins_sd=bins_sd.astype(np.float32),
        bins_dead=np.asarray(bins_dead, dtype=bool),
        norm_ref=norm_ref, sites=sites,
        discriminants=str(discriminants), discriminants_sha256=disc_sha,
        bins_config=bins_config, source="computed",
    )
    return ShelfBuild(stats=stats, z_corpus=z_corpus, rows=rows, bins_std=bins_std,
                      targets=np.vstack(targets), verify_err=err)
