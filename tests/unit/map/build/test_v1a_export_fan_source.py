"""Tests for `v1a_export.join_bank`'s fan source.

The registered join takes fans from the lever file's ``member_group_index`` and
DROPS every member whose group got no 2-means lever — on the real corpus, 4,344
generations in lone-outlier fans. ``fan_source="member_fan"``
keeps them, with fan = (corpus, prompt_id, wave). Pinned here:

  * the DEFAULT path is the registered join — checked against an independent
    reference implementation of it, not against its own output;
  * member_fan keeps the no-lever fans, still honours MIN_FAN, and on lever
    members reproduces the default's rows, fans AND fan order;
  * a lever file whose groups are not (corpus, prompt, wave) is refused.

CPU only, small fixtures; `load_pairs`/`corpus_of_run_dir` are monkeypatched at
their modules (join_bank imports them at call time), hiddens are a real npz.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from pleroma.map.build import export as v1a_export
from pleroma.map.build.export import (
    FAN_SOURCE_LEVER,
    FAN_SOURCE_MEMBER,
    BankJoin,
    join_bank,
    member_fan_index,
)

SITES = [3, 7]
HID = 4
ZD = 6

# (corpus, prompt_id, wave, lever group index, n members)
#  - cA/p1: a lever group
#  - cA/p2: NO lever (the [7,1] case) -> dropped by default, kept by member_fan
#  - cB/p1: same prompt+wave as cA/p1 in another corpus (the merge_levers case)
#  - cB/p3: no lever AND below MIN_FAN -> dropped by both paths
#  - cA/p4 repl: a lever group in the other wave
FANS: list[tuple[str, str, str, int, int]] = [
    ("cA", "p1", "orig", 0, 4),
    ("cA", "p2", "orig", -1, 4),
    ("cB", "p1", "orig", 1, 3),
    ("cB", "p3", "orig", -1, 2),
    ("cA", "p4", "repl", 2, 3),
]


def _join_bank_reference(pairs_dir: Path, levers: Path,
                         hiddens: Sequence[Path]) -> BankJoin:
    """The registered join, as a frozen reference implementation — the default
    path must keep producing exactly this."""
    from pleroma.map.build.regress import corpus_of_run_dir
    from pleroma.map.build.pairs import load_pairs
    from pleroma.map.build.cv import MIN_FAN, load_hiddens

    loaded = load_pairs(pairs_dir)
    lev = np.load(levers, allow_pickle=True)
    m_corpus = [str(x) for x in lev["member_corpus_keys"]]
    m_genid = np.asarray(lev["member_generation_ids"], dtype=int)
    m_group = np.asarray(lev["member_group_index"], dtype=int)
    member_of = {(m_corpus[j], int(m_genid[j])): j for j in range(len(m_genid))}
    sites = [int(x) for x in lev["sites"]]

    hidden = load_hiddens(list(hiddens))

    seen: set[int] = set()
    rows: list[int] = []
    hs: list[np.ndarray] = []
    groups: list[int] = []
    keys: list[tuple[str, int]] = []
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
    if missing_h:
        raise ValueError(f"{missing_h} joined members missing hiddens")

    z = np.asarray(loaded.z[rows], dtype=np.float64)
    h = np.asarray(hs, dtype=np.float64)
    grp = np.asarray(groups, dtype=int)
    n = z.shape[0]

    fans: list[np.ndarray] = []
    for g in np.unique(grp):
        sel = np.nonzero(grp == g)[0]
        if sel.size >= MIN_FAN:
            fans.append(sel)
    if not fans:
        raise ValueError(f"no fan survived the >={MIN_FAN}-member filter")
    keep = np.sort(np.concatenate(fans))
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(keep.size)
    fans = [remap[s] for s in fans]
    return BankJoin(z=z[keep], h=h[keep], grp=grp[keep],
                    keys=[keys[i] for i in keep], fans=fans, sites=sites)


@pytest.fixture()
def bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rng = np.random.default_rng(23)
    corpus, gid, pid, wave, grp = [], [], [], [], []
    g = 0
    for c, p, w, gi, k in FANS:
        for _ in range(k):
            corpus.append(c)
            gid.append(g)
            pid.append(p)
            wave.append(w)
            grp.append(gi)
            g += 1
    n_mem = len(gid)
    levers = tmp_path / "levers.npz"
    np.savez_compressed(
        levers,
        member_corpus_keys=np.asarray(corpus, dtype=np.str_),
        member_generation_ids=np.asarray(gid, dtype=np.int64),
        member_group_index=np.asarray(grp, dtype=np.int64),
        member_prompt_ids=np.asarray(pid, dtype=np.str_),
        member_waves=np.asarray(wave, dtype=np.str_),
        sites=np.asarray(SITES, dtype=np.int64),
    )
    hid = tmp_path / "mean_hiddens.npz"
    np.savez_compressed(
        hid,
        means=rng.standard_normal((n_mem, len(SITES), HID)).astype(np.float32),
        generation_ids=np.asarray(gid, dtype=np.int64),
        corpus_keys=np.asarray(corpus, dtype=np.str_),
        sites=np.asarray(SITES, dtype=np.int64),
    )
    # pairs in a SHUFFLED order (row order is pairs order, which both paths
    # must respect), plus one pair that is in no lever file at all.
    order = rng.permutation(n_mem)
    pairs = [SimpleNamespace(row=r, run_dir=f"/runs/{corpus[m]}",
                             generation_id=gid[m], prompt_id=pid[m])
             for r, m in enumerate(order)]
    pairs.append(SimpleNamespace(row=n_mem, run_dir="/runs/cZ",
                                 generation_id=999, prompt_id="px"))
    loaded = SimpleNamespace(pairs=pairs,
                             z=rng.standard_normal((n_mem + 1, ZD)).astype(np.float32))

    import pleroma.map.build.pairs as bp
    import pleroma.map.build.regress as rl

    monkeypatch.setattr(bp, "load_pairs", lambda _d: loaded)
    monkeypatch.setattr(rl, "corpus_of_run_dir", lambda rd: Path(rd).name)
    return {"pairs_dir": tmp_path, "levers": levers, "hiddens": [hid],
            "n_mem": n_mem}


def _assert_same(a: BankJoin, b: BankJoin) -> None:
    assert a.z.tobytes() == b.z.tobytes()
    assert a.h.tobytes() == b.h.tobytes()
    assert a.grp.tobytes() == b.grp.tobytes()
    assert a.keys == b.keys
    assert a.sites == b.sites
    assert len(a.fans) == len(b.fans)
    for fa, fb in zip(a.fans, b.fans):
        assert fa.tobytes() == fb.tobytes()


def test_default_fan_source_is_the_registered_lever_group() -> None:
    sig = inspect.signature(join_bank)
    assert sig.parameters["fan_source"].default == FAN_SOURCE_LEVER == "lever_group"


def test_default_path_is_byte_identical_to_the_pre_change_join(bank) -> None:
    ref = _join_bank_reference(bank["pairs_dir"], bank["levers"], bank["hiddens"])
    new = join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"])
    explicit = join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"],
                         fan_source=FAN_SOURCE_LEVER)
    _assert_same(new, ref)
    _assert_same(explicit, ref)
    # and it still drops the no-lever fans: 4 + 3 + 3 lever members only
    assert new.n == 10
    assert {k[0] for k in new.keys} == {"cA", "cB"}
    assert sorted(f.size for f in new.fans) == [3, 3, 4]


def test_member_fan_keeps_the_no_lever_fan_and_still_honours_min_fan(bank) -> None:
    new = join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"],
                    fan_source=FAN_SOURCE_MEMBER)
    # cA/p2 (4, no lever) is recovered; cB/p3 (2 members) is still < MIN_FAN
    assert new.n == 14
    assert sorted(f.size for f in new.fans) == [3, 3, 4, 4]
    kept_gids = {k[1] for k in new.keys}
    assert set(range(4, 8)) <= kept_gids          # cA/p2's members
    assert not ({11, 12} & kept_gids)             # cB/p3's members
    # every fan is one (corpus, prompt) — no fan mixes corpora (merge case)
    corpora = np.asarray([k[0] for k in new.keys])
    for f in new.fans:
        assert len(set(corpora[f])) == 1


def test_member_fan_restricted_to_lever_members_equals_the_default(bank) -> None:
    ref = join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"])
    new = join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"],
                    fan_source=FAN_SOURCE_MEMBER)
    ref_keys = set(ref.keys)
    keep = np.asarray([k in ref_keys for k in new.keys])
    idx = np.nonzero(keep)[0]
    assert [new.keys[i] for i in idx] == ref.keys          # same rows, same order
    assert new.z[idx].tobytes() == ref.z.tobytes()
    assert new.h[idx].tobytes() == ref.h.tobytes()
    remap = -np.ones(new.n, dtype=int)
    remap[idx] = np.arange(idx.size)
    restricted = [remap[f] for f in new.fans if keep[f].all()]
    assert all(keep[f].all() or not keep[f].any() for f in new.fans)
    assert [f.tolist() for f in restricted] == [f.tolist() for f in ref.fans]


def test_member_fan_index_numbers_by_first_appearance() -> None:
    fan = member_fan_index(["a", "a", "b", "a", "b"], ["p", "p", "p", "q", "p"],
                           ["o", "o", "o", "o", "o"], np.array([0, 0, -1, 1, -1]))
    assert fan.tolist() == [0, 0, 1, 2, 1]


def test_member_fan_index_refuses_a_fan_that_mixes_lever_groups() -> None:
    with pytest.raises(ValueError, match="mix groups"):
        member_fan_index(["a"] * 4, ["p"] * 4, ["o"] * 4, np.array([0, 0, 1, 1]))
    # a fan with some lever and some no-lever members is also not a group
    with pytest.raises(ValueError, match="mix groups"):
        member_fan_index(["a"] * 4, ["p"] * 4, ["o"] * 4, np.array([0, 0, -1, -1]))


def test_member_fan_index_refuses_a_group_that_spans_fans() -> None:
    with pytest.raises(ValueError, match="span fans"):
        member_fan_index(["a", "a", "b", "b"], ["p"] * 4, ["o"] * 4,
                         np.array([0, 0, 0, 0]))


def test_member_fan_index_refuses_ragged_inputs() -> None:
    with pytest.raises(ValueError, match="disagree in length"):
        member_fan_index(["a", "a"], ["p"], ["o", "o"], np.array([0, 0]))


def test_unknown_fan_source_is_refused(bank) -> None:
    with pytest.raises(ValueError, match="fan_source"):
        join_bank(bank["pairs_dir"], bank["levers"], bank["hiddens"],
                  fan_source="prompt")


def test_fan_source_constants_are_what_the_cli_offers() -> None:
    assert v1a_export.FAN_SOURCES == (FAN_SOURCE_LEVER, FAN_SOURCE_MEMBER)
