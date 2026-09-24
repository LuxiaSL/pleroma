"""Stage `export`: is the map file the served map the profile describes?

| gate            | catalog | question |
|-----------------|---------|----------|
| map-structure   | S-06, X-04, X-06 | does the npz carry a loom map's keys, the profile's sites/rank/hidden, a sane ruler? |
| map-load        | X-02, X-06 | does the canonical loader (`LoomMap`) accept it with its discriminants (sha pin, W/factor agreement)? |
| export-report   | E-03, X-04 | did the export's self-check pass, and does its report carry a real held-out number for THIS file? |

The fingerprint printed as evidence is the map's v2 id (`load_map_identity`,
pleroma.map.identity): it covers the stored weights, with the legacy v1 id
embedded after the dot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pleroma.config import ModelProfile
from pleroma.dose.band import load_map_identity
from pleroma.map.loom_map import LoomMap
from pleroma.validate._io import InputError, read_json, require_mapping, sha256_file
from pleroma.validate.result import GateResult, failed, guarded, inconclusive, passed

STAGE = "export"
#: The 3B registered held-out number every pre-fix export report carried (X-04).
STALE_3B_HELDOUT: float = 0.9339
#: Keys `LoomMap` reads unconditionally (besides W or its factors).
REQUIRED_KEYS: tuple[str, ...] = ("mu_in", "mu_y", "v3_mu", "v3_sd", "v3_dead", "bins_mu",
                                  "bins_sd", "bins_dead", "sites", "norm_ref", "meta")


@guarded(STAGE, "map-structure")
def gate_map_structure(map_path: str | Path, profile: ModelProfile) -> GateResult:
    """Structure and profile agreement, without the (node-only, large)
    discriminants bundle."""
    path = Path(map_path)
    if not path.is_file():
        return inconclusive(STAGE, "map-structure", f"map file not found: {path}")
    try:
        fp, sites, norm_ref, meta = load_map_identity(path)
        with np.load(path, allow_pickle=True) as npz:
            files = set(npz.files)
            mu_y_size = int(np.asarray(npz["mu_y"]).size)
    except (KeyError, ValueError, OSError, EOFError) as exc:
        return failed(STAGE, "map-structure", f"not a readable loom map npz: {exc}",
                      path=str(path))
    ev: dict[str, Any] = dict(path=str(path), map_fingerprint=fp,
                              sites=sites, profile_sites=list(profile.map.sites),
                              rank=meta.get("rank"), profile_rank=profile.map.rank)
    missing = [k for k in REQUIRED_KEYS if k not in files]
    has_w = "W" in files or {"W_U", "W_S", "W_Vt"} <= files
    if missing or not has_w:
        return failed(STAGE, "map-structure",
                      f"map is missing required arrays {missing + ([] if has_w else ['W or W_U/W_S/W_Vt'])}: "
                      "the loader will refuse it", **ev)
    if list(sites) != list(profile.map.sites):
        return failed(STAGE, "map-structure",
                      f"map sites {sites} ≠ profile sites {list(profile.map.sites)}: sites "
                      "must be read from the map, never recomputed (S-06)", **ev)
    rank = meta.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool):
        return failed(STAGE, "map-structure",
                      f"map meta carries no integer rank ({rank!r}): the code basis is "
                      "undefined", **ev)
    if rank != profile.map.rank:
        return failed(STAGE, "map-structure",
                      f"map rank {rank} ≠ profile rank {profile.map.rank}: a code from this "
                      "map is not the profile's code", **ev)
    hidden = profile.model.arch.hidden_dim
    ev["mu_y_size"] = mu_y_size
    if mu_y_size != len(sites) * hidden:
        return failed(STAGE, "map-structure",
                      f"map output width {mu_y_size} ≠ {len(sites)} sites × hidden {hidden}: "
                      "built for a different model (X-04)", **ev)
    nr = np.asarray(norm_ref, dtype=np.float64).ravel()
    ev["norm_ref"] = nr.tolist()
    if nr.size != len(sites) or not np.all(np.isfinite(nr)) or np.any(nr <= 0):
        return failed(STAGE, "map-structure",
                      f"ruler norm_ref {nr.tolist()} is not one positive finite value per "
                      "site: α would mean nothing", **ev)
    if "discriminants_sha256" not in meta:
        return failed(STAGE, "map-structure",
                      "map meta carries no discriminants_sha256: nothing pins the frozen "
                      "feature space it was fit in (X-06)", **ev)
    return passed(STAGE, "map-structure",
                  f"loom map with the profile's sites {sites}, rank {rank}, hidden {hidden}, "
                  "and a positive per-site ruler", **ev)


@guarded(STAGE, "map-load")
def gate_map_load(map_path: str | Path, discriminants: str | Path | None) -> GateResult:
    """The canonical loader's own checks: discriminants sha pin (the frozen
    space) and, where the map ships both, W vs its stored factors (X-02)."""
    path = Path(map_path)
    if not path.is_file():
        return inconclusive(STAGE, "map-load", f"map file not found: {path}")
    if discriminants is None:
        return inconclusive(STAGE, "map-load",
                            "discriminants not supplied (--discriminants): the sha pin and "
                            "W/factor agreement are unchecked", path=str(path))
    disc = Path(discriminants)
    if not disc.is_file():
        return inconclusive(STAGE, "map-load", f"discriminants file not found: {disc}",
                            path=str(path))
    try:
        m = LoomMap(path, disc)
    except (KeyError, ValueError, OSError, EOFError) as exc:
        return failed(STAGE, "map-load", f"LoomMap refuses this map: {exc}",
                      path=str(path), discriminants=str(disc))
    return passed(STAGE, "map-load",
                  "LoomMap loads it: discriminants sha matches, stored factors agree with W",
                  path=str(path), discriminants=str(disc), rank=m.rank, hidden=m.hidden,
                  n_sites=m.n_sites, ruler=m.norm_ref_which)


@guarded(STAGE, "export-report")
def gate_export_report(report_path: str | Path, map_path: str | Path | None = None
                       ) -> GateResult:
    """E-03 (in-sample is a sanity floor, not an outcome) and X-04 (the 3B
    literal 0.9339 in every pre-fix report)."""
    try:
        rep = require_mapping(read_json(report_path, "export report"), "export report")
        sc = require_mapping(rep.get("self_check"), "export report 'self_check'")
    except InputError as exc:
        return inconclusive(STAGE, "export-report", str(exc), path=str(report_path))
    heldout = sc.get("registered_heldout_top1")
    ev: dict[str, Any] = dict(
        path=str(report_path), self_check_verdict=sc.get("verdict"),
        in_sample_top1=sc.get("within_fan_top1_in_sample"), floor=sc.get("floor"),
        chance=sc.get("chance"), registered_heldout_top1=heldout,
        heldout_source=sc.get("heldout_source"), out_sha256=rep.get("out_sha256"))
    if map_path is not None and rep.get("out_sha256"):
        actual = sha256_file(map_path)
        ev["map_sha256"] = actual
        if actual is not None and actual != rep["out_sha256"]:
            return failed(STAGE, "export-report",
                          "export report describes a different file (out_sha256 ≠ the map's "
                          "sha256): its self-check says nothing about this map", **ev)
    if sc.get("verdict") != "PASS":
        return failed(STAGE, "export-report",
                      f"export self-check did not pass (verdict {sc.get('verdict')!r}; "
                      f"in-sample top-1 {sc.get('within_fan_top1_in_sample')} vs floor "
                      f"{sc.get('floor')})", **ev)
    if heldout is None:
        return inconclusive(STAGE, "export-report",
                            "export report carries no held-out number: its only figure is "
                            "IN-SAMPLE, a sanity floor, not a quality claim (E-03) — re-export "
                            "with --heldout-report", **ev)
    if not sc.get("heldout_source") and abs(float(heldout) - STALE_3B_HELDOUT) < 1e-9:
        return failed(STAGE, "export-report",
                      f"held-out {STALE_3B_HELDOUT} with no source is the 3B literal every "
                      "pre-fix report carried, not this map's number (X-04): re-export", **ev)
    return passed(STAGE, "export-report",
                  f"export self-check passed; held-out top-1 {heldout} from "
                  f"{'a named CV report' if sc.get('heldout_source') else 'an unnamed source'} "
                  "(the in-sample figure is a sanity floor only)", **ev)


def run_export(profile: ModelProfile, map_path: str | Path | None,
               discriminants: str | Path | None, export_report: str | Path | None
               ) -> list[GateResult]:
    out: list[GateResult] = []
    if map_path is not None:
        out.append(gate_map_structure(map_path, profile))
        out.append(gate_map_load(map_path, discriminants))
    if export_report is not None:
        out.append(gate_export_report(export_report, map_path))
    return out
