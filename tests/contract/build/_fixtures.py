"""Synthetic BUILD-chain inputs, written in the on-disk formats the stages read.

One builder, ``build_bank``, writes a whole corpus for one run:

    <root>/corpusA/signatures/gen_NNNNN.npz   raw v3 signatures (features, names)
    <pairs>/pairs_manifest.json               the pairs stage with ``--standardize corpus``
    <pairs>/pairs_z.npz                       (``<pairs>`` = ``<root>/pairs``)
    <root>/levers.npz                          the lever bank's schema, incl. -1 groups
    <root>/mean_hiddens.npz                    ``pleroma.map.build.hiddens``' schema
    <root>/bins.npz                            ``pleroma.harvest.bins``' schema
    <root>/disc.npz                            discriminants (FULL_mean/scale/names)

and returns a ``Bank`` carrying the ground truth every reference computation
in the tests needs. Nothing here imports the code under test: the formats are
written from what the readers read (``pleroma.map.build.pairs.load_pairs``,
``pleroma.map.build.join.load_hiddens``, the wide map's bins/levers joins), so a reader
drifting from its format shows up as a test failure, not a fixture edit.

The planted geometry (all deliberate, each pinned by some test):

* fans = (prompt, wave); every prompt has an ``orig`` and a ``repl`` fan, so a
  fold assignment that splits a prompt's waves is visible;
* every ``nolever_every``-th fan is a ``[7,1]`` fan with
  ``member_group_index = -1`` (a 7-vs-1 split: 2-means cannot make a lever
  from a singleton cluster);
* one lever fan (``sparse``) has only 2 of its 8 members in the pairs dir, so
  it survives ``fit_loom_map``'s join but not v1a's ``MIN_FAN = 3``;
* one extra pair (``px``) is in pairs/hiddens/bins but in no lever file;
* raw signature dim 0 is constant (NOT dead: float32 cast noise, see
  test_fit_loom_map), dim 1 has SD 5e-9 (dead under the v3 cutoff 1e-8),
  dim 2 has a degenerate ``FULL_scale`` (0.0);
* bins col 0 is constant, col 1 has SD 5e-9 (NOT dead under the bins cutoff
  1e-9 — the two stages' dead-feature cutoffs differ, pinned as-is);
* hidden means are a low-rank linear function of z_corpus plus noise, so the
  v1a fit genuinely retrieves (self-check can PASS) — or pure noise when
  ``signal=False`` (self-check must REFUSE).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FAN_SIZE = 8
WAVE_SEED_OFFSET = {"orig": 0, "repl": 1000}
CORPUS = "corpusA"


@dataclass
class Member:
    gen_id: int
    prompt_id: str
    wave: str
    seed_idx: int
    fan: int            # (prompt, wave) fan number, in member order
    group: int          # member_group_index (-1 = no lever)
    cluster: int
    has_pair: bool


@dataclass
class Bank:
    root: Path
    run_dir: Path
    pairs_dir: Path
    levers: Path
    hiddens: Path
    bins: Path
    discriminants: Path
    sites: list[int]
    hidden: int
    z_dim: int
    bins_dim: int
    members: list[Member]
    extra_gen_id: int
    # ground truth, aligned to PAIRS order (manifest row order)
    pair_gen_ids: np.ndarray
    pair_prompts: list[str]
    pair_splits: list[str]
    z_full32: np.ndarray        # [P, z] float32, the recipe build_pairs banks from
    z_corpus: np.ndarray        # [P, z] float32 == pairs_z
    v3_mu: np.ndarray
    v3_sd: np.ndarray
    v3_dead: np.ndarray
    # hiddens / bins / levers, keyed by generation id
    means: dict[int, np.ndarray] = field(default_factory=dict)      # [S, H]
    bins_of: dict[int, np.ndarray] = field(default_factory=dict)    # [B]
    levers_arr: np.ndarray | None = None                             # [G, S, H]
    full_mean: np.ndarray | None = None
    full_scale: np.ndarray | None = None

    @property
    def member_of(self) -> dict[int, Member]:
        return {m.gen_id: m for m in self.members}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_bank(
    root: Path,
    *,
    n_prompts: int,
    z_dim: int,
    bins_dim: int,
    sites: list[int],
    hidden: int,
    nolever_every: int = 10,
    sparse_fan: int = 1,
    seed: int = 0,
    signal: bool = True,
    latent_rank: int = 8,
    noise: float = 0.05,
    write_signatures: bool = True,
) -> Bank:
    rng = np.random.default_rng(seed)
    root = Path(root)
    run_dir = root / CORPUS
    sig_dir = run_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    n_sites = len(sites)
    out_dim = n_sites * hidden

    # ── members, fan by fan, in build_levers' emission order ──────────────────
    members: list[Member] = []
    gid = 0
    fan = 0
    group = 0
    for p in range(n_prompts):
        pid = f"p{p:03d}"
        for wave in ("orig", "repl"):
            nolever = (fan % nolever_every) == (nolever_every - 1)
            for k in range(FAN_SIZE):
                if nolever:
                    cluster = 0 if k < FAN_SIZE - 1 else 1          # [7,1]
                else:
                    cluster = 0 if k < FAN_SIZE // 2 else 1         # [4,4]
                has_pair = not (fan == sparse_fan and k >= 2)
                members.append(Member(
                    gen_id=gid, prompt_id=pid, wave=wave,
                    seed_idx=WAVE_SEED_OFFSET[wave] + k, fan=fan,
                    group=-1 if nolever else group, cluster=cluster,
                    has_pair=has_pair))
                gid += 1
            if not nolever:
                group += 1
            fan += 1
    extra_gen_id = gid          # a pair that is in no lever file
    all_ids = [m.gen_id for m in members] + [extra_gen_id]
    n_all = len(all_ids)

    # ── raw signatures + discriminants ────────────────────────────────────────
    scales = rng.uniform(0.1, 10.0, size=z_dim)
    offsets = rng.normal(0.0, 5.0, size=z_dim)
    prompt_off = {m.prompt_id: rng.normal(0.0, 0.5, size=z_dim) for m in members}
    prompt_off["px"] = np.zeros(z_dim)
    x = np.empty((n_all, z_dim), dtype=np.float32)
    for i, g in enumerate(all_ids):
        pid = members[g].prompt_id if g < len(members) else "px"
        row = offsets + scales * (rng.standard_normal(z_dim) + prompt_off[pid])
        x[i] = row.astype(np.float32)
    x[:, 0] = np.float32(4.25)                                           # constant
    x[:, 1] = (rng.standard_normal(n_all) * 5e-9).astype(np.float32)     # sd 5e-9
    full_mean = rng.normal(0.0, 1.0, size=z_dim)
    full_mean[1] = 0.0
    full_scale = rng.uniform(0.5, 3.0, size=z_dim)
    full_scale[1] = 1.0
    full_scale[2] = 0.0                                                  # degenerate
    names = np.asarray([f"feat_{j}" for j in range(z_dim)], dtype=np.str_)
    disc = root / "disc.npz"
    np.savez(disc, FULL_mean=full_mean, FULL_scale=full_scale, FULL_names=names)

    # ── pairs: which gens, in a SHUFFLED row order, with splits ───────────────
    in_pairs = [g for g in all_ids if g == extra_gen_id or members[g].has_pair]
    order = rng.permutation(len(in_pairs))
    pair_ids = np.asarray([in_pairs[int(j)] for j in order], dtype=int)

    def split_of(g: int) -> str:
        if g == extra_gen_id:
            return "train"
        m = members[g]
        if m.prompt_id == f"p{n_prompts - 1:03d}":
            return "val_prompt"
        return "val_seed" if (m.seed_idx % FAN_SIZE) == FAN_SIZE - 1 else "train"

    idx_of = {g: i for i, g in enumerate(all_ids)}
    xp = x[[idx_of[int(g)] for g in pair_ids]].astype(np.float64)
    degenerate = full_scale < 1e-12
    zf = (xp - full_mean) / np.where(degenerate, 1.0, full_scale)
    zf[:, degenerate] = 0.0
    z_full32 = zf.astype(np.float32)
    splits = [split_of(int(g)) for g in pair_ids]
    train = np.asarray([s == "train" for s in splits])
    v3_mu = z_full32[train].mean(axis=0)
    v3_sd = z_full32[train].std(axis=0)
    v3_dead = v3_sd < 1e-8
    z_corpus = ((z_full32 - v3_mu) / np.where(v3_dead, 1.0, v3_sd)).astype(np.float32)
    z_corpus[:, v3_dead] = 0.0

    if write_signatures:   # only fit_loom_map reads raw signatures
        for i, g in enumerate(all_ids):
            np.savez(sig_dir / f"gen_{g:05d}.npz", features=x[i], feature_names=names)

    pair_prompts = []
    rows = []
    for r, g in enumerate(pair_ids.tolist()):
        pid = "px" if g == extra_gen_id else members[g].prompt_id
        seed_idx = 0 if g == extra_gen_id else members[g].seed_idx
        pair_prompts.append(pid)
        rows.append({
            "row": r, "pair_key": f"{CORPUS}:{g:03d}", "generation_id": g,
            "run_dir": str(run_dir), "run_label": CORPUS, "prompt_id": pid,
            "prompt_class": "decision_continuation", "seed_idx": seed_idx,
            "split": splits[r], "fork_positions": [1], "n_fork_positions": 1,
            "fork_source": "synthetic", "fallback_reason": None, "n_tokens": 32,
            "prompt_length": 5, "signature_path": f"signatures/gen_{g:05d}.npz",
            "fork_series_path": f"fork_series/gen_{g:05d}.npz",
        })
    pairs_dir = root / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    (pairs_dir / "pairs_manifest.json").write_text(json.dumps({
        "stage": "synthetic build_pairs",
        "params": {"standardize": "corpus"},
        "pairs": rows,
    }))
    np.savez_compressed(pairs_dir / "pairs_z.npz", z=z_corpus,
                        feature_names=names,
                        rows=np.asarray([r["pair_key"] for r in rows], dtype=np.str_))

    # ── hidden means: low-rank linear in z_corpus (+ noise), or pure noise ────
    a = (rng.standard_normal((z_dim, latent_rank))
         @ rng.standard_normal((latent_rank, out_dim))) / np.sqrt(z_dim)
    z_of = {int(g): z_corpus[r].astype(np.float64) for r, g in enumerate(pair_ids)}
    means: dict[int, np.ndarray] = {}
    for g in all_ids:
        if signal and g in z_of:
            h = z_of[g] @ a + noise * rng.standard_normal(out_dim)
        else:
            h = rng.standard_normal(out_dim)
        means[g] = h.reshape(n_sites, hidden).astype(np.float32)
    hid = root / ("mean_hiddens.npz" if signal else "mean_hiddens_noise.npz")
    np.savez_compressed(
        hid,
        means=np.stack([means[g] for g in all_ids]),
        generation_ids=np.asarray(all_ids, dtype=np.int64),
        corpus_keys=np.asarray([CORPUS] * n_all, dtype=np.str_),
        prompt_ids=np.asarray([members[g].prompt_id if g < len(members) else "px"
                               for g in all_ids], dtype=np.str_),
        sites=np.asarray(sites, dtype=np.int64),
    )

    # ── bins (raw features, rows in their own order) ──────────────────────────
    bins_rows = rng.permutation(np.asarray(in_pairs, dtype=int))
    bfeat = rng.standard_normal((bins_rows.size, bins_dim)) * rng.uniform(
        0.5, 4.0, size=bins_dim) + rng.normal(0.0, 3.0, size=bins_dim)
    bfeat[:, 0] = 3.0                                                     # dead
    bfeat[:, 1] = 7.0 + rng.standard_normal(bins_rows.size) * 5e-9       # sd 5e-9
    bins_path = root / "bins.npz"
    bins_config = {"layers": [1, 2, 3], "n_bins": bins_dim, "synthetic": True}
    np.savez_compressed(
        bins_path,
        features=bfeat,
        run_dir=np.asarray([str(run_dir)] * bins_rows.size, dtype=np.str_),
        generation_id=bins_rows.astype(np.int64),
        config=np.asarray(json.dumps(bins_config)),
    )
    bins_of = {int(g): bfeat[k] for k, g in enumerate(bins_rows)}

    # ── levers npz, build_levers' schema (mean_0 - mean_1 per lever group) ────
    lever_groups = sorted({m.group for m in members if m.group >= 0})
    lev = []
    csums, csizes = [], []
    for gi in lever_groups:
        mem = [m for m in members if m.group == gi]
        s = np.zeros((2, n_sites, hidden))
        n = [0, 0]
        for m in mem:
            s[m.cluster] += means[m.gen_id]
            n[m.cluster] += 1
        csums.append(s)
        csizes.append(n)
        lev.append((s[0] / n[0] - s[1] / n[1]).astype(np.float32))
    levers_arr = np.stack(lev)
    first = {}
    for m in members:
        first.setdefault(m.group, m)
    levers = root / "levers.npz"
    np.savez_compressed(
        levers,
        levers=levers_arr,
        lever_norms=np.linalg.norm(levers_arr, axis=2).astype(np.float32),
        cluster_sums=np.stack(csums),
        cluster_sizes=np.asarray(csizes, dtype=np.int64),
        group_prompt_ids=np.asarray([first[g].prompt_id for g in lever_groups], dtype=np.str_),
        group_waves=np.asarray([first[g].wave for g in lever_groups], dtype=np.str_),
        group_prompt_classes=np.asarray(["decision_continuation"] * len(lever_groups),
                                        dtype=np.str_),
        group_corpus_keys=np.asarray([CORPUS] * len(lever_groups), dtype=np.str_),
        member_keys=np.asarray([f"{CORPUS}#{m.gen_id:04d}" for m in members], dtype=np.str_),
        member_generation_ids=np.asarray([m.gen_id for m in members], dtype=np.int64),
        member_corpus_keys=np.asarray([CORPUS] * len(members), dtype=np.str_),
        member_prompt_ids=np.asarray([m.prompt_id for m in members], dtype=np.str_),
        member_prompt_classes=np.asarray(["decision_continuation"] * len(members),
                                         dtype=np.str_),
        member_seed_idxs=np.asarray([m.seed_idx for m in members], dtype=np.int64),
        member_waves=np.asarray([m.wave for m in members], dtype=np.str_),
        member_clusters=np.asarray([m.cluster for m in members], dtype=np.int64),
        member_group_index=np.asarray([m.group for m in members], dtype=np.int64),
        sites=np.asarray(sites, dtype=np.int64),
        meta_json=np.asarray(json.dumps({"stage": "synthetic build_levers",
                                         "min_cluster": 2}), dtype=np.str_),
    )

    return Bank(
        root=root, run_dir=run_dir, pairs_dir=pairs_dir, levers=levers,
        hiddens=hid, bins=bins_path, discriminants=disc, sites=list(sites),
        hidden=hidden, z_dim=z_dim, bins_dim=bins_dim, members=members,
        extra_gen_id=extra_gen_id, pair_gen_ids=pair_ids, pair_prompts=pair_prompts,
        pair_splits=splits, z_full32=z_full32, z_corpus=z_corpus,
        v3_mu=v3_mu, v3_sd=v3_sd, v3_dead=v3_dead, means=means, bins_of=bins_of,
        levers_arr=levers_arr, full_mean=full_mean, full_scale=full_scale,
    )


def write_noise_hiddens(bank: Bank, seed: int = 99) -> Path:
    """Same keys as ``bank.hiddens``, values independent of z: nothing to retrieve."""
    rng = np.random.default_rng(seed)
    with np.load(bank.hiddens, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    payload["means"] = rng.standard_normal(payload["means"].shape).astype(np.float32)
    out = bank.root / "mean_hiddens_noise.npz"
    np.savez_compressed(out, **payload)
    return out


def write_wide_map(bank: Bank, out: Path, *, rank: int = 64, lam: float = 1e4,
                   seed: int = 5) -> Path:
    """A wide/shelf map in EXACTLY fit_loom_map's output format, without the fit.

    Used where running fit_loom_map is too slow for the suite (the export gate
    needs >= 8,000 rows x 420 bins dims). Standardisation stats are the bank's
    true ones with fit_loom_map's cutoffs (v3 1e-8, bins 1e-9); W is a random
    rank-``rank`` operator, since v1a_export reads only the stats, sites, ruler
    and meta from it. ``test_fit_loom_map::test_synthetic_wide_map_is_a_format_twin``
    holds this writer to the real one's keys, dtypes and meta fields.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(bank.sites)
    out_dim = n_sites * bank.hidden
    in_dim = bank.z_dim + bank.bins_dim
    mo = bank.member_of      # fit_loom_map: joined lever members, train split
    train_ids = [int(g) for g, s in zip(bank.pair_gen_ids, bank.pair_splits)
                 if s == "train" and int(g) in mo and mo[int(g)].group >= 0]
    b = np.stack([bank.bins_of[g] for g in train_ids])
    bins_mu, bins_sd = b.mean(axis=0), b.std(axis=0)
    bins_dead = bins_sd < 1e-9
    u, _ = np.linalg.qr(rng.standard_normal((in_dim, rank)))
    vt = np.linalg.qr(rng.standard_normal((out_dim, rank)))[0].T
    s = np.sort(rng.uniform(0.5, 3.0, size=rank))[::-1]
    w = (u * s) @ vt
    norm_ref = np.nanmedian(np.linalg.norm(bank.levers_arr.astype(np.float64), axis=2), axis=0)
    np.savez_compressed(
        out,
        W=w.astype(np.float32), W_U=u.astype(np.float32), W_S=s.astype(np.float32),
        W_Vt=vt.astype(np.float32),
        mu_in=np.zeros(in_dim, dtype=np.float32), mu_y=np.zeros(out_dim, dtype=np.float32),
        v3_mu=bank.v3_mu.astype(np.float32), v3_sd=bank.v3_sd.astype(np.float32),
        v3_dead=bank.v3_dead,
        bins_mu=bins_mu.astype(np.float32), bins_sd=bins_sd.astype(np.float32),
        bins_dead=bins_dead,
        sites=np.array(bank.sites), norm_ref=norm_ref.astype(np.float32),
        meta=np.array(json.dumps({
            "lam": lam, "rank": rank, "n_rows": len(bank.pair_gen_ids),
            "feature_order": f"z_corpus({bank.z_dim}) then bins_std({bank.bins_dim})",
            "discriminants": str(bank.discriminants),
            "discriminants_sha256": sha256(bank.discriminants),
            "bins_config": {"layers": [1, 2, 3], "n_bins": bank.bins_dim, "synthetic": True},
            "pairs_dir": str(bank.pairs_dir), "levers": str(bank.levers),
            "in_sample_mean_cos": 0.0,
            "sign_convention": "synthetic", "alpha_convention": "synthetic",
            "svd_orientation": {"convention": "vt_row_maxabs_positive"},
        })),
    )
    return out


# ── the closed-form references (the spec for pleroma.map ridge) ───────────────


def ridge_reference(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered ridge in float64 numpy: W = (XcᵀXc + λI)⁻¹ XcᵀYc; returns W, mu_x, mu_y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mu_x, y - mu_y
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ yc)
    return w, mu_x, mu_y


def truncate_reference(w: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-r approximation; r <= 0 or r >= min(shape) is full rank."""
    if rank <= 0 or rank >= min(w.shape):
        return w
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def loo_center(x: np.ndarray, fans: list[np.ndarray]) -> np.ndarray:
    """x_i - mean_{j != i in fan} x_j, written independently of v1a_fit.fan_center."""
    out = np.zeros_like(np.asarray(x, dtype=np.float64))
    for sel in fans:
        for i in sel:
            others = [j for j in sel if j != i]
            out[i] = x[i] - x[others].mean(axis=0)
    return out


def top1_reference(pred: np.ndarray, cand: np.ndarray, fans: list[np.ndarray]) -> float:
    hits = []
    for sel in fans:
        p = pred[sel] / np.linalg.norm(pred[sel], axis=1, keepdims=True)
        c = cand[sel] / np.linalg.norm(cand[sel], axis=1, keepdims=True)
        hits.extend((np.argmax(p @ c.T, axis=1) == np.arange(len(sel))).tolist())
    return float(np.mean(hits))
