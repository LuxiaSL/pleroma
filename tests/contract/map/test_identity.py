"""Contract: map identity — the fingerprint the loom stamps on every wear.

`map_fingerprint(sites, norm_ref, mu_y, meta)` (``pleroma.dose.band``) is
the legacy (v1) id; ``LoomMap.fingerprint`` (v2, ``pleroma.map.identity``)
adds the stored weights. The id is computed once at serve start and stamped on
every wear; a restored wear or a MEASURED dose band carrying a different
fingerprint is refused, so no path can silently serve one map's records
against another map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from . import _fixtures as fx
from . import targets as T


def fp(m: T.LoomMap) -> str:
    """Exactly the server's id: LoomMap.fingerprint (v2, weights included)."""
    return m.fingerprint


def test_fingerprint_is_16_hex_and_stable_across_reloads(
        v1a_path: Path, disc) -> None:
    """Deterministic: the same file loaded twice gives the same 16-hex id, so
    a restart can rehydrate its own wears."""
    a = fp(T.LoomMap(v1a_path, disc[0]))
    b = fp(T.LoomMap(v1a_path, disc[0]))
    assert a == b
    head, legacy = a.removeprefix("v2-").split(".")
    for part in (head, legacy):
        assert len(part) == 16 and all(ch in "0123456789abcdef" for ch in part)


@pytest.mark.parametrize("field", ["sites", "norm_ref", "mu_y", "meta"])
def test_fingerprint_changes_with_each_hashed_field(v1a: T.LoomMap,
                                                    field: str) -> None:
    """Each of the four inputs moves the id — in particular norm_ref, because
    a band measured under one ruler is meaningless under another."""
    base = T.map_fingerprint(v1a.sites, v1a.norm_ref, v1a.mu_y, v1a.meta)
    sites, ref, mu_y, meta = (list(v1a.sites), v1a.norm_ref.copy(),
                              v1a.mu_y.copy(), dict(v1a.meta))
    if field == "sites":
        sites[0] += 1
    elif field == "norm_ref":
        ref[0] *= 1.0001
    elif field == "mu_y":
        mu_y[0] += 1e-9
    else:
        meta["note"] = "x"
    assert T.map_fingerprint(sites, ref, mu_y, meta) != base


def test_two_maps_with_different_weights_have_different_fingerprints(
        tmp_path: Path, disc) -> None:
    """The id identifies the MAP, not its label. Two v1a
    exports with the same meta/sites/ruler but different W (e.g. a refit whose
    meta happened to repeat) must not share a fingerprint."""
    common = {"fit_seed": 0}
    a = T.LoomMap(fx.write_v1a_map(tmp_path / "a", disc[1], seed=1,
                                   meta_extra=common), disc[0])
    b = T.LoomMap(fx.write_v1a_map(tmp_path / "b", disc[1], seed=2,
                                   meta_extra=common), disc[0])
    assert not np.allclose(a.W, b.W)
    assert fp(a) != fp(b)


# ── v2 migration: records stamped with the legacy (v1) id ─────────────────────

def test_v2_embeds_the_legacy_id_so_old_stamps_still_match(v1a: T.LoomMap) -> None:
    """A band/wear stamped with the v1 id matches the loaded map as 'legacy'
    (weights unverified), never 'exact'."""
    legacy = T.map_fingerprint(v1a.sites, v1a.norm_ref, v1a.mu_y, v1a.meta)
    assert T.legacy_part(v1a.fingerprint) == legacy
    assert T.same_map(v1a.fingerprint, v1a.fingerprint) == "exact"
    assert T.same_map(legacy, v1a.fingerprint) == "legacy"
    assert T.same_map(None, v1a.fingerprint) == "unstamped"
    assert T.same_map("0123456789abcdef", v1a.fingerprint) == "mismatch"


def test_a_v2_stamp_from_another_map_is_a_mismatch_even_with_equal_legacy_ids(
        tmp_path: Path, disc) -> None:
    """The case v1 could not see: same meta/sites/ruler, different W."""
    common = {"fit_seed": 0}
    a = T.LoomMap(fx.write_v1a_map(tmp_path / "a", disc[1], seed=1,
                                   meta_extra=common), disc[0])
    b = T.LoomMap(fx.write_v1a_map(tmp_path / "b", disc[1], seed=2,
                                   meta_extra=common), disc[0])
    assert T.legacy_part(a.fingerprint) == T.legacy_part(b.fingerprint)
    assert T.same_map(b.fingerprint, a.fingerprint) == "mismatch"


def test_a_code_with_a_banked_foreign_fingerprint_is_refused(
        tmp_path: Path, disc) -> None:
    """A JSON code (bare list) banked beside its map id is checked too."""
    a = T.LoomMap(fx.write_v1a_map(tmp_path / "a", disc[1], seed=1), disc[0])
    b = T.LoomMap(fx.write_v1a_map(tmp_path / "b", disc[1], seed=2), disc[0])
    sig, brow = fx.raw_rows(1)[0]
    _, _, code_b = b.lever_of(sig, brow)
    assert isinstance(code_b, T.MapCode) and code_b.map_fingerprint == b.fingerprint
    with pytest.raises(ValueError, match="foreign code"):
        a.lever_from_code(list(code_b), differential=False,
                          map_fingerprint=b.fingerprint)
    # its own map, and a legacy stamp of its own map, still wear
    b.lever_from_code(code_b, differential=False)
    b.lever_from_code(list(code_b), differential=False,
                      map_fingerprint=T.legacy_part(b.fingerprint))
