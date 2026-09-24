"""Tiny synthetic LoomMap artifacts in the REAL on-disk schema.

Pure numpy, no code under test imported here: these builders model the FILE
FORMAT (what ``pleroma.map.build.export`` and ``pleroma.map.build.wide``
write), so the contract still stands when the loader moves.

Two families, matching the two writers that feed `LoomMap`:

* `write_v1a_map` — the SERVED form:
  float64 rank-r factors `W_U/W_S/W_Vt` only, `W_U` carrying ZERO rows for the
  bins block (the bins half of the input is a strict no-op), `mu_in = mu_y = 0`,
  float32 `v3_*`/`bins_*` copied from the wide map, the WIDE ruler as
  `norm_ref` plus `norm_ref_v1a_own` and the `norm_ref_which` label.
* `write_wide_map` — fit_loom_map's v0 "wide" map: nonzero `mu_in`/`mu_y`,
  float32 everything, and BOTH the dense `W` and float32 factors (the wide
  build ships both when 0 < rank < min(W.shape), and the two can disagree on
  the factorisation's signs).

Dimensions are deliberately tiny: 3 sites x hidden 16, rank 4, z 6, bins 3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

SITES: list[int] = [3, 7, 11]
HIDDEN: int = 16
RANK: int = 4
Z_DIM: int = 6
BINS_DIM: int = 3
N_IN: int = Z_DIM + BINS_DIM
N_OUT: int = len(SITES) * HIDDEN
NORM_REF: list[float] = [1.5, 2.5, 4.0]            # the WIDE ruler (worn)
NORM_REF_OWN: list[float] = [0.9, 1.4, 2.2]        # v1a's own (banked, not worn)
NORM_REF_WHICH = ("norm_ref = WIDE MAP's ruler (decision 1); norm_ref_v1a_own "
                  "is v1a's own and is NOT worn")

#: v3 dim 4 and bins dim 1 are DEAD (sd ~ 0 at fit time), as fit_loom_map marks.
V3_DEAD = np.array([False, False, False, False, True, False])
BINS_DEAD = np.array([False, True, False])
#: FULL_scale dim 2 is degenerate (< 1e-12): LoomMap substitutes 1.0 and zeroes it.
FULL_DEGENERATE_DIM = 2


def write_discriminants(dirpath: Path, *, seed: int = 1) -> tuple[Path, str]:
    """`FULL_mean`/`FULL_scale` (+ the `FULL_names` contract key) and the sha
    the map's meta must carry (LoomMap refuses a mismatch)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scale = np.abs(rng.standard_normal(Z_DIM)) + 0.5
    scale[FULL_DEGENERATE_DIM] = 1e-13
    path = dirpath / f"disc_{seed}.npz"
    np.savez(path, FULL_names=np.array([f"f{i}" for i in range(Z_DIM)]),
             FULL_mean=rng.standard_normal(Z_DIM), FULL_scale=scale)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def orthonormal_rows(rng: np.random.Generator, rank: int, width: int,
                     zero_cols: slice | None = None) -> np.ndarray:
    """[rank, width] with orthonormal rows; `zero_cols` forces a column block
    to exactly zero, giving a site the map can never write to."""
    if zero_cols is None:
        return np.linalg.qr(rng.standard_normal((width, rank)))[0].T
    keep = np.ones(width, dtype=bool)
    keep[zero_cols] = False
    q = np.linalg.qr(rng.standard_normal((int(keep.sum()), rank)))[0].T
    out = np.zeros((rank, width))
    out[:, keep] = q
    return out


def standardisation_arrays() -> dict[str, np.ndarray]:
    """v3_*/bins_* as the wide map banks them (float32), with dead dims."""
    v3_sd = np.linspace(0.6, 1.4, Z_DIM).astype(np.float32)
    v3_sd[V3_DEAD] = 0.0
    bins_sd = np.linspace(0.7, 1.3, BINS_DIM).astype(np.float32)
    bins_sd[BINS_DEAD] = 0.0
    return dict(
        v3_mu=np.linspace(-0.3, 0.3, Z_DIM).astype(np.float32),
        v3_sd=v3_sd, v3_dead=V3_DEAD.copy(),
        bins_mu=np.linspace(-0.2, 0.2, BINS_DIM).astype(np.float32),
        bins_sd=bins_sd, bins_dead=BINS_DEAD.copy(),
    )


def v1a_factors(seed: int, rank: int = RANK, zero_site: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v1a's (U [N_IN, r] with zero bins rows, S [r], Vt [r, N_OUT])."""
    rng = np.random.default_rng(seed)
    u = np.vstack([rng.standard_normal((Z_DIM, rank)), np.zeros((BINS_DIM, rank))])
    s = np.sort(np.abs(rng.standard_normal(rank)) + 1.0)[::-1].copy()
    zc = (slice(zero_site * HIDDEN, (zero_site + 1) * HIDDEN)
          if zero_site is not None else None)
    vt = orthonormal_rows(rng, rank, N_OUT, zc)
    return u, s, vt


def write_v1a_map(dirpath: Path, disc_sha: str, *, seed: int = 20260921,
                  rank: int = RANK, name: str = "v1a.npz",
                  zero_site: int | None = None,
                  mu_y: np.ndarray | None = None,
                  norm_ref: list[float] | None = None,
                  meta_extra: dict[str, Any] | None = None) -> Path:
    """The served v1a LoomMap npz (the export's payload keys)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    u, s, vt = v1a_factors(seed, rank, zero_site)
    meta: dict[str, Any] = {
        "lam": 1e4, "rank": int(rank), "n_rows": 40, "n_fans": 5,
        "feature_order": f"z_corpus({Z_DIM}) then bins_std({BINS_DIM})",
        "discriminants": "disc.npz", "discriminants_sha256": disc_sha,
        "bins_config": None, "prereg_token": "v1a-fit-001",
        "operating_point": {"lambda": 1e4, "rank": int(rank), "input": "dz",
                            "target": "raw", "registered": True},
        "fit_seed": int(seed),
    }
    meta.update(meta_extra or {})
    path = dirpath / name
    np.savez_compressed(
        path,
        W_U=u.astype(np.float64), W_S=s.astype(np.float64),
        W_Vt=vt.astype(np.float64),
        mu_in=np.zeros(N_IN), mu_y=(np.zeros(N_OUT) if mu_y is None
                                    else np.asarray(mu_y, dtype=np.float64)),
        **standardisation_arrays(),
        sites=np.array(SITES),
        norm_ref=np.asarray(norm_ref or NORM_REF, dtype=np.float64),
        norm_ref_v1a_own=np.asarray(NORM_REF_OWN, dtype=np.float64),
        norm_ref_which=np.array(NORM_REF_WHICH),
        meta=np.array(json.dumps(meta)),
    )
    return path


def write_wide_map(dirpath: Path, disc_sha: str, *, seed: int = 20260917,
                   rank: int = RANK, name: str = "wide.npz",
                   store: Literal["dense", "factors", "both"] = "both",
                   flip_factor_row: int | None = None) -> Path:
    """fit_loom_map's wide map: nonzero mu, float32.

    `flip_factor_row` stores the factor pair with one singular vector's sign
    flipped relative to numpy's SVD of the stored dense W — the same W, a
    different (equally valid) factorisation — the situation a reader that
    re-derives its own SVD from the dense W runs into.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w_full = rng.standard_normal((N_IN, N_OUT))
    u, s, vt = np.linalg.svd(w_full, full_matrices=False)
    w = ((u[:, :rank] * s[:rank]) @ vt[:rank]).astype(np.float32)
    # factor what is actually STORED, exactly as a reader would
    u2, s2, vt2 = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    u2, s2, vt2 = u2[:, :rank].copy(), s2[:rank].copy(), vt2[:rank].copy()
    if flip_factor_row is not None:
        u2[:, flip_factor_row] *= -1.0
        vt2[flip_factor_row] *= -1.0
    arrays: dict[str, np.ndarray] = {}
    if store in ("dense", "both"):
        arrays["W"] = w
    if store in ("factors", "both"):
        arrays.update(W_U=u2.astype(np.float32), W_S=s2.astype(np.float32),
                      W_Vt=vt2.astype(np.float32))
    meta = {"lam": 1000.0, "rank": int(rank), "n_rows": 40,
            "feature_order": f"z_corpus({Z_DIM}) then bins_std({BINS_DIM})",
            "discriminants": "disc.npz", "discriminants_sha256": disc_sha}
    path = dirpath / name
    np.savez_compressed(
        path, **arrays,
        mu_in=rng.standard_normal(N_IN).astype(np.float32),
        mu_y=rng.standard_normal(N_OUT).astype(np.float32),
        **standardisation_arrays(),
        sites=np.array(SITES),
        norm_ref=np.asarray(NORM_REF, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    return path


def raw_rows(k: int, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (raw v3 signature, raw bins row) pairs — what the harvest hands the map."""
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(Z_DIM) * 2.0, rng.standard_normal(BINS_DIM) * 3.0)
            for _ in range(k)]


def raw_flat(lever: np.ndarray, raw_norms: list[float],
             norm_ref: np.ndarray) -> np.ndarray:
    """Undo the per-site ruler: the map's RAW output, flattened. Valid because
    every returned lever row is `raw_row * norm_ref[s] / raw_norm[s]`."""
    scale = np.asarray(raw_norms, dtype=np.float64) / np.asarray(norm_ref)
    return (lever * scale[:, None]).ravel()


def write_lever_bank(path: Path, levers: np.ndarray,
                     sites: list[int] | None = None) -> Path:
    """The lever-bank keys that `LeverBank` requires."""
    g = levers.shape[0]
    np.savez(path, levers=levers.astype(np.float32),
             group_prompt_ids=np.array([f"p{i:03d}" for i in range(g)]),
             group_waves=np.array(["orig"] * g),
             sites=np.array(sites or SITES))
    return path
