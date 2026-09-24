"""The v1a bank join: pairs + lever-file members + mean hiddens -> fans.

The fit (`pleroma.map.build.cv`) and the export (`pleroma.map.build.export`)
both call `join_bank`, so the registered CV always runs on exactly the rows the
served map is fit on. Two copies of the join would drift, and a fan option
present in only one of them would put the CV and the served map on different
rows without either noticing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("v1a_join")

#: Target constructions (the fit's registered grid). `raw` = LOO
#: fan-centred mean hiddens (the registered default); `sitenorm` = each site
#: block of that delta scaled to unit L2 (a swept variant, not the default).
TARGETS: tuple[str, ...] = ("raw", "sitenorm")
TARGET_DEFAULT: str = "raw"


MIN_FAN = 3



def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-12)
    return (a * b).sum(axis=1) / denom



def load_hiddens(paths: list[Path]) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    sites_ref = None
    for p in paths:
        npz = np.load(p, allow_pickle=True)
        means = np.asarray(npz["means"], dtype=np.float64)
        gids = np.asarray(npz["generation_ids"], dtype=int)
        corp = [str(x) for x in npz["corpus_keys"]]
        sites = [int(x) for x in npz["sites"]]
        if sites_ref is None:
            sites_ref = sites
        elif sites != sites_ref:
            raise ValueError(f"{p}: sites {sites} != {sites_ref}")
        for i in range(means.shape[0]):
            key = (corp[i], int(gids[i]))
            if key in out:
                raise ValueError(f"duplicate hiddens key {key} (second source {p})")
            out[key] = means[i]
        logger.info("hiddens %s: %d gens (sites %s)", p, means.shape[0], sites)
    return out



def fan_center(x: np.ndarray, fans: list[np.ndarray]) -> np.ndarray:
    """Leave-one-out fan-centering, rowwise: x_i - mean_{j!=i in fan} x_j."""
    out = np.empty_like(x)
    for sel in fans:
        k = sel.size
        s = x[sel].sum(axis=0)
        out[sel] = (x[sel] * k - s[None, :]) / (k - 1)
    return out



def retrieval(pred: np.ndarray, cand: np.ndarray, fans: list[np.ndarray]) -> np.ndarray:
    """Per-row hit: does pred_i's nearest (cosine) fan candidate equal i?"""
    hits = np.zeros(pred.shape[0], dtype=bool)
    for sel in fans:
        p = pred[sel]
        c = cand[sel]
        pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        cm = pn @ cn.T
        hits[sel] = cm.argmax(axis=1) == np.arange(sel.size)
    return hits



def build_target(d_raw: np.ndarray, target: str, n_sites: int) -> np.ndarray:
    """The fit's target construction, re-exported so the two cannot drift.

    The fit (`pleroma.map.build.cv`, its ``d_raw`` / ``d_sitenorm``) and the
    export both build their targets here, including the same ``1e-12`` norm
    floor, so the export provably targets what the fit swept.

    ``raw``      : the LOO fan-centered mean hiddens, untouched.
    ``sitenorm`` : that same delta with each site's block scaled to unit L2, so
                   a member's per-site MAGNITUDE (which is driven by how far
                   its length sits from the fan mean) carries no weight in the
                   ridge and only its DIRECTION does.

    Raises ValueError for an unknown ``target`` or a width that does not
    split into ``n_sites`` equal blocks.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; known: {list(TARGETS)}")
    if target == "raw":
        return d_raw
    n, width = d_raw.shape
    if n_sites <= 0 or width % n_sites:
        raise ValueError(
            f"target width {width} is not divisible by {n_sites} sites — "
            "the per-site blocks sitenorm scales are not well defined")
    blocks = d_raw.reshape(n, n_sites, width // n_sites).copy()
    norms = np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
    return (blocks / norms).reshape(n, width)



@dataclass(frozen=True)
class BankJoin:
    """The registered v1a bank: z, mean hiddens, fans, and the join keys.

    One definition, used by the export and by every read-only diagnostic over
    the same bank, so a diagnostic can never disagree with the fit about which
    rows it is talking about.
    """

    z: np.ndarray                    # [N, z_dim] z_corpus, as the pairs build banked
    h: np.ndarray                    # [N, S*hidden] banked mean hiddens
    grp: np.ndarray                  # [N] lever-group index
    keys: list[tuple[str, int]]      # [N] (corpus_key, generation_id)
    fans: list[np.ndarray]           # row indices per fan, >= MIN_FAN members
    sites: list[int]
    prompts: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=str))
    #: [N] prompt_id per row (the grouped-CV key)
    meta: dict = field(default_factory=dict)  # the pairs manifest (pairs-build params)

    @property
    def n(self) -> int:
        return int(self.z.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.h.shape[1] // len(self.sites))



#: Where a fan's membership comes from. See `join_bank` and `member_fan_index`.
#: ``lever_group`` is the registered behaviour and the default.
FAN_SOURCE_LEVER: str = "lever_group"



FAN_SOURCE_MEMBER: str = "member_fan"



FAN_SOURCES: tuple[str, ...] = (FAN_SOURCE_LEVER, FAN_SOURCE_MEMBER)



def member_fan_index(corpus: Sequence[str], prompt_ids: Sequence[str],
                     waves: Sequence[str],
                     lever_group: np.ndarray) -> np.ndarray:
    """A fan index for EVERY lever-file member, lever or no lever.

    ★ WHY THIS EXISTS. v1a is one-vs-rest: its target is
    ``h_i - mean_{j!=i in fan} h_j`` and it never reads the v0 cluster lever.
    The lever file's ``member_group_index`` is a FAN definition only by
    borrowing — `pleroma.map.build.levers` writes ``-1`` for every member of a
    group it could not 2-means (`--min-cluster`), so a join on it evicts whole
    fans for failing a clustering step v1a does not perform. On model C that is
    543/1,805 fans, 4,344/14,440 gens: every ``[7,1]``/``[1,7]`` fan.

    The fan key is ``(corpus, prompt_id, wave)`` — the lever build's own group
    key ``(prompt_id, wave)`` plus the corpus, which is what keeps a merged
    lever bank correct (two banks concatenated from one cell share
    ``prompt_id`` and wave label but are distinct groups in distinct corpora). Every member row is
    already in the lever file with those three fields, lever or not, so no
    lever bank needs rebuilding.

    Numbering is by first appearance in member order. Because the lever build
    emits members group by group and a merge concatenates banks with offset
    indices, the kept groups keep their relative order — so restricted
    to lever members this reproduces `lever_group`'s fan ORDER, not just its
    partition.

    Refuses (``ValueError``) unless the fan partition EXTENDS the lever-group one
    without changing it: every fan that contains a lever member must
    consist ONLY of members of exactly one lever group, and every lever group
    must lie inside exactly one fan. So on the lever members the two sources
    agree by construction, checked, on every call.
    """
    n = len(corpus)
    if not (len(prompt_ids) == len(waves) == int(np.asarray(lever_group).size) == n):
        raise ValueError(
            f"member arrays disagree in length: corpus {n}, prompt_ids "
            f"{len(prompt_ids)}, waves {len(waves)}, group {np.asarray(lever_group).size}")
    grp = np.asarray(lever_group, dtype=int)
    fan_of: dict[tuple[str, str, str], int] = {}
    fan = np.empty(n, dtype=int)
    for j in range(n):
        key = (str(corpus[j]), str(prompt_ids[j]), str(waves[j]))
        idx = fan_of.get(key)
        if idx is None:
            idx = len(fan_of)
            fan_of[key] = idx
        fan[j] = idx
    groups_in_fan: dict[int, set[int]] = {}
    fans_of_group: dict[int, set[int]] = {}
    for f, g in zip(fan.tolist(), grp.tolist()):
        groups_in_fan.setdefault(f, set()).add(g)
        if g >= 0:
            fans_of_group.setdefault(g, set()).add(f)
    mixed = {f: sorted(gs) for f, gs in groups_in_fan.items()
             if len(gs) > 1}
    split = {g: sorted(fs) for g, fs in fans_of_group.items() if len(fs) > 1}
    if mixed or split:
        raise ValueError(
            f"(corpus, prompt_id, wave) fans do not extend the lever groups: "
            f"{len(mixed)} fan(s) mix groups (e.g. {list(mixed.items())[:3]}), "
            f"{len(split)} group(s) span fans (e.g. {list(split.items())[:3]}) "
            "— this lever file's group key is not (corpus, prompt, wave)")
    return fan



def join_bank(pairs_dir: Path, levers: Path,
              hiddens: Sequence[Path],
              fan_source: str = FAN_SOURCE_LEVER) -> BankJoin:
    """THE v1a join — used by the registered CV (`pleroma.map.build.cv`) AND the export, so
    the two can never disagree about which rows they are talking about.

    Raises rather than returning a partial bank: a member without banked
    hiddens, or a fan below ``MIN_FAN``, changes what the fit was registered on.

    ``fan_source``:
      * ``"lever_group"`` (DEFAULT, the registered behaviour): a fan is a lever
        group, and members whose group got no lever (``member_group_index <
        0``) are DROPPED. Every existing 3B/8B/model-C artifact was built this
        way and this path is unchanged.
      * ``"member_fan"``: a fan is ``(corpus, prompt_id, wave)`` for every
        member in the lever file, lever or not (`member_fan_index`). Nothing is
        dropped for 2-means viability. For lever members the fans are
        identical to ``lever_group``'s, checked.
    """
    from pleroma.map.build.pairs import load_pairs
    from pleroma.map.build.regress import corpus_of_run_dir

    if fan_source not in FAN_SOURCES:
        raise ValueError(f"fan_source {fan_source!r} not in {FAN_SOURCES}")
    loaded = load_pairs(pairs_dir)
    lev = np.load(levers, allow_pickle=True)
    m_corpus = [str(x) for x in lev["member_corpus_keys"]]
    m_genid = np.asarray(lev["member_generation_ids"], dtype=int)
    m_group = np.asarray(lev["member_group_index"], dtype=int)
    if fan_source == FAN_SOURCE_MEMBER:
        n_lever = int((m_group >= 0).sum())
        m_group = member_fan_index(
            m_corpus, [str(x) for x in lev["member_prompt_ids"]],
            [str(x) for x in lev["member_waves"]], m_group)
        logger.info(
            "fan source member_fan: %d members in %d (corpus, prompt, wave) "
            "fans — %d of them had no lever and are KEPT",
            m_group.size, len(np.unique(m_group)), m_group.size - n_lever)
    member_of = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}
    sites = [int(x) for x in lev["sites"]]

    hidden = load_hiddens(list(hiddens))

    seen: set[int] = set()
    rows: list[int] = []
    hs: list[np.ndarray] = []
    groups: list[int] = []
    keys: list[tuple[str, int]] = []
    prompts: list[str] = []
    dupes = missing_h = 0
    for p in loaded.pairs:
        corpus = corpus_of_run_dir(p.run_dir)
        j = member_of.get((corpus, p.generation_id))
        if j is None or int(m_group[j]) < 0:
            continue
        if j in seen:
            dupes += 1
            continue
        hh = hidden.get((corpus, p.generation_id))
        if hh is None:
            missing_h += 1
            continue
        seen.add(j)
        rows.append(p.row)
        hs.append(hh.reshape(-1))
        groups.append(int(m_group[j]))
        keys.append((corpus, p.generation_id))
        prompts.append(str(p.prompt_id))
    if missing_h:
        raise ValueError(f"{missing_h} joined members missing hiddens")
    if dupes:
        logger.warning("%d duplicate member rows in pairs (kept first)", dupes)

    z = np.asarray(loaded.z[rows], dtype=np.float64)
    h = np.asarray(hs, dtype=np.float64)
    grp = np.asarray(groups, dtype=int)
    n = z.shape[0]
    logger.info("joined %d gens", n)

    fans: list[np.ndarray] = []
    dropped = 0
    for g in np.unique(grp):
        sel = np.nonzero(grp == g)[0]
        if sel.size >= MIN_FAN:
            fans.append(sel)
        else:
            dropped += sel.size
    if not fans:
        raise ValueError(f"no fan survived the >={MIN_FAN}-member filter")
    keep = np.sort(np.concatenate(fans))
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(keep.size)
    fans = [remap[s] for s in fans]
    logger.info("%d fans, %d gens kept, %d dropped (<%d members)",
                len(fans), keep.size, dropped, MIN_FAN)
    return BankJoin(z=z[keep], h=h[keep], grp=grp[keep],
                    keys=[keys[i] for i in keep], fans=fans, sites=sites,
                    prompts=np.asarray([prompts[i] for i in keep]),
                    meta=dict(getattr(loaded, "meta", None) or {}))
