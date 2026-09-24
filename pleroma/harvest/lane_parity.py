"""The GPU-lane parity gate: re-harvest banked gens through the seam, compare.

The served model-C map was fit on a corpus banked by an earlier build of the GPU
lane, one that hooked every layer. The lane in this package runs from the pinned
anamnesis submodule through ``pleroma.harvest.anamnesis_seam``, and it differs
from the banked build in two ways only real weights can test: the lane's source
(it hooks keys/values/queries/gates only at the preset's SAMPLED layers) and the
software stack of the GPU host's venv. This gate re-harvests N banked gens and
compares every feature of every gen to the banked signature.

Three steps, run in this order:

``select`` (CPU)   pick N banked gens — the shortest and longest spans plus a
                   seeded uniform sample — and write a lane manifest for them
                   from the banked shard manifests (same ``input_ids`` and
                   ``prompt_length`` bytes the corpus was replayed from).
``harvest`` (GPU)  open ONE ``GpuHarvestLane`` and run the manifest TWICE
                   (``new/`` and ``repeat/``): the repeat must be bit-exact,
                   which separates non-determinism from a real change.
``compare`` (CPU)  preconditions, then names, then values; writes
                   ``<work>/parity_report.json`` and exits 0 (EXACT/PASS), 1 (FAIL)
                   or 2 (a precondition failed — the comparison would not be
                   about the code).

PRECONDITIONS (exit 2, never a PASS or FAIL): the new run's calibration
digests and checkpoint digests equal the banked run's ``<run>/deployment.json``.
With a different calibration or checkpoint every value may differ for reasons
that have nothing to do with the lane.

TOLERANCE, AND WHY. Feature names must match in order, exactly. For values
the unit is the one the map reads: the map build z-scores every feature by
the corpus ``FULL_scale`` (from the corpus manifest npz), so a difference
matters only relative to that spread. Per feature, per gen::

    z = |new - banked| / FULL_scale        (FULL_scale > 0)
    |new - banked| <= ATOL = 1e-6           (FULL_scale == 0: a constant)

and NaN positions must coincide. Verdicts:

* ``EXACT`` — every value bit-identical. Expected when the host's stack
  (torch / transformers / CUDA / GPU type, recorded in both lane identities)
  matches the banked run's AND the sampled-layer hooking changed nothing.
* ``PASS`` — not exact, but ``max z <= Z_TOL`` (default ``1e-3``: a
  thousandth of a corpus standard deviation). The model forward is bf16,
  whose unit roundoff is 2^-8 ≈ 3.9e-3 relative, so one kernel choice that
  differs between torch builds moves a single activation by an ulp and a raw
  feature by up to ~1e-3 relative; aggregated and divided by the corpus
  spread that lands well under 1e-3 z. Meanwhile the map's discriminant
  directions separate classes by O(1) z, three orders above the tolerance —
  a real change in what a family measures (e.g. a capture that went missing
  at an unsampled layer) is O(0.1-1) z and cannot hide under it. The
  upstream CPU equivalence suite's fp32 tolerance (rtol 3e-5, atol 5e-6) is
  NOT the right yardstick here: it compares float32 paths, not a bf16 forward
  across software builds.
* ``FAIL`` — names differ, the repeat is not exact, or ``max z > Z_TOL``.
  The report lists the worst features by z and the per-family maxima, which
  is where a hooking difference would show (value_geometry / qk_geometry /
  kv_cka read the captures the hook change touches).

``lane_id`` WILL differ (it hashes the lane's source files); the report
records both. It is identity, not a verdict: rows from two lane_ids mix only
once the new lane is RULED equivalent, with this report as its evidence, in
the registry :mod:`pleroma.harvest.lane_equivalence` reads (``DEFAULT_REGISTRY``).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pleroma.harvest.lane_parity")

Z_TOL = 1e-3
ATOL = 1e-6
N_EXTREMES = 3
#: lane-identity keys that describe the software/hardware stack
STACK_KEYS = ("torch", "transformers", "numpy", "cuda_runtime", "cublas_workspace_config",
              "deterministic", "tf32", "preferred_blas_library", "device_type")


# ── the banked corpus ────────────────────────────────────────────────────────


def banked_shards(lane_dir: Path) -> list[Path]:
    shards = sorted(p for p in Path(lane_dir).glob("shard_*") if (p / "manifest.json").exists())
    if not shards:
        raise FileNotFoundError(f"no shard_*/manifest.json under {lane_dir}")
    return shards


def load_banked_index(lane_dir: Path) -> dict[int, dict[str, Any]]:
    """{gid: {entry, meta, npz, shard}} over every banked shard that has output."""
    index: dict[int, dict[str, Any]] = {}
    for shard in banked_shards(lane_dir):
        entries = json.loads((shard / "manifest.json").read_text())["entries"]
        meta_path = shard / "metadata.json"
        metas: dict[int, dict[str, Any]] = {}
        if meta_path.exists():
            for rec in json.loads(meta_path.read_text()).get("generations", []):
                metas[int(rec["generation_id"])] = rec
        for key, entry in entries.items():
            gid = int(key)
            npz = shard / "out" / f"gen_{gid:03d}.npz"
            if not npz.exists():
                continue
            if gid in index:
                raise ValueError(f"gen {gid} is banked in two shards")
            index[gid] = {"entry": entry, "meta": metas.get(gid), "npz": npz, "shard": shard}
    if not index:
        raise FileNotFoundError(f"no banked gen_NNN.npz under {lane_dir}/shard_*/out")
    return index


def select_gens(index: Mapping[int, Mapping[str, Any]], n: int, *,
                n_extremes: int = N_EXTREMES, seed: int = 0) -> list[int]:
    """The ``n_extremes`` shortest and longest spans, then a seeded sample."""
    def steps(g: int) -> int:
        e = index[g]["entry"]
        return len(e["input_ids"]) - int(e["prompt_length"]) - 1

    ordered = sorted(index, key=lambda g: (steps(g), g))
    if n >= len(ordered):
        return sorted(ordered)
    k = min(n_extremes, n // 2)
    chosen = set(ordered[:k]) | set(ordered[len(ordered) - k:])
    rest = [g for g in ordered if g not in chosen]
    chosen |= set(random.Random(seed).sample(rest, max(0, n - len(chosen))))
    return sorted(chosen)


def write_subset_manifest(index: Mapping[int, Mapping[str, Any]], gids: Sequence[int],
                          out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {str(g): index[g]["entry"] for g in gids}
    (out_dir / "manifest.json").write_text(json.dumps({"entries": entries}))
    gens = [index[g]["meta"] for g in gids if index[g]["meta"] is not None]
    if len(gens) == len(gids):
        (out_dir / "metadata.json").write_text(json.dumps({"generations": gens}))
    (out_dir / "selection.json").write_text(json.dumps({
        "gen_ids": list(gids),
        "banked": {str(g): str(index[g]["npz"]) for g in gids},
    }, indent=1))
    return out_dir / "manifest.json"


# ── comparison (pure) ────────────────────────────────────────────────────────


def _load_sig(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return [str(n) for n in z["feature_names"]], np.asarray(z["features"], dtype=np.float64)


def _family_of(name: str, slices: Mapping[str, Sequence[int]] | None, i: int) -> str:
    if slices:
        for fam, (a, b) in slices.items():
            if a <= i < b:
                return str(fam)
    return "?"


def compare_vectors(banked: np.ndarray, new: np.ndarray, scale: np.ndarray, *,
                    atol: float = ATOL) -> dict[str, np.ndarray]:
    """Elementwise diff, z (NaN where the scale is degenerate), and violations
    of the atol rule / NaN coincidence."""
    diff = np.abs(new - banked)
    both_nan = np.isnan(new) & np.isnan(banked)
    nan_mismatch = np.isnan(new) ^ np.isnan(banked)
    diff = np.where(both_nan, 0.0, diff)
    ok_scale = np.isfinite(scale) & (scale > 0)
    z = np.where(ok_scale, diff / np.where(ok_scale, scale, 1.0), np.nan)
    atol_bad = (~ok_scale) & (diff > atol)
    return {"diff": diff, "z": z, "nan_mismatch": nan_mismatch, "atol_bad": atol_bad,
            "exact": (diff == 0) & ~nan_mismatch}


def compare_runs(pairs: Mapping[int, tuple[Path, Path]], full_names: Sequence[str],
                 full_scale: np.ndarray, *, z_tol: float = Z_TOL, atol: float = ATOL,
                 slices: Mapping[str, Sequence[int]] | None = None,
                 worst: int = 20) -> dict[str, Any]:
    """Compare {gid: (banked_npz, new_npz)}; the verdict and its evidence."""
    full_names = [str(n) for n in full_names]
    scale_by_name = dict(zip(full_names, np.asarray(full_scale, dtype=np.float64), strict=True))
    problems: list[str] = []
    rows: list[tuple[float, int, str, float, float, float]] = []
    fam_max: dict[str, float] = {}
    n_values = n_exact = 0
    max_z = 0.0
    max_abs = 0.0
    names_ref: list[str] | None = None
    for gid, (b_path, n_path) in sorted(pairs.items()):
        b_names, b_vals = _load_sig(b_path)
        n_names, n_vals = _load_sig(n_path)
        if n_names != b_names:
            problems.append(f"gen {gid}: feature names differ (banked {len(b_names)}, "
                            f"new {len(n_names)}; first difference at "
                            f"{next((i for i, (a, b) in enumerate(zip(b_names, n_names)) if a != b), min(len(b_names), len(n_names)))})")
            continue
        if names_ref is None:
            names_ref = n_names
            missing = [n for n in n_names if n not in scale_by_name]
            if missing:
                problems.append(f"{len(missing)} feature names have no FULL_scale "
                                f"(e.g. {missing[:3]}) — wrong manifest?")
        scale = np.array([scale_by_name.get(n, np.nan) for n in n_names])
        c = compare_vectors(b_vals, n_vals, scale, atol=atol)
        n_values += len(n_names)
        n_exact += int(c["exact"].sum())
        if c["nan_mismatch"].any():
            idx = np.flatnonzero(c["nan_mismatch"])[:5]
            problems.append(f"gen {gid}: NaN in one run only at {[n_names[i] for i in idx]}")
        if c["atol_bad"].any():
            idx = np.flatnonzero(c["atol_bad"])[:5]
            problems.append(f"gen {gid}: zero-scale features differ by > {atol}: "
                            f"{[n_names[i] for i in idx]}")
        zfin = np.where(np.isfinite(c["z"]), c["z"], 0.0)
        max_z = max(max_z, float(zfin.max(initial=0.0)))
        max_abs = max(max_abs, float(c["diff"].max(initial=0.0)))
        for i in np.argsort(-zfin)[:worst]:
            if zfin[i] > 0:
                rows.append((float(zfin[i]), gid, n_names[i], float(b_vals[i]),
                             float(n_vals[i]), float(scale[i])))
        for i, zi in enumerate(zfin):
            fam = _family_of(n_names[i], slices, i)
            if zi > fam_max.get(fam, 0.0):
                fam_max[fam] = float(zi)
    rows.sort(reverse=True)
    if max_z > z_tol:
        problems.append(f"max z {max_z:.3g} > Z_TOL {z_tol:g}")
    if not problems and n_values and n_exact == n_values:
        verdict = "EXACT"
    elif not problems and n_values:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        if not n_values:
            problems.append("nothing was compared")
    return {
        "verdict": verdict,
        "n_gens": len(pairs),
        "n_values": n_values,
        "n_exact": n_exact,
        "max_z": max_z,
        "max_abs": max_abs,
        "z_tol": z_tol,
        "atol": atol,
        "per_family_max_z": dict(sorted(fam_max.items(), key=lambda kv: -kv[1])),
        "worst": [{"z": r[0], "gen": r[1], "feature": r[2], "banked": r[3], "new": r[4],
                   "full_scale": r[5]} for r in rows[:worst]],
        "problems": problems,
    }


def stack_diff(banked_identity: Mapping[str, Any], new_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {k: {"banked": banked_identity.get(k), "new": new_identity.get(k)}
            for k in STACK_KEYS if banked_identity.get(k) != new_identity.get(k)}


def preconditions(banked_dep: Mapping[str, Any],
                  new_dep: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(failures, warnings). Calibration: the files' digests, else the lane
    identity's ``calibration`` digest (the same digest_of_shas over the same
    two artifacts). Checkpoint: ``model_files_sha256``. A field the BANKED
    receipt never recorded is a warning (unverifiable), not a failure."""
    fails: list[str] = []
    warns: list[str] = []
    b_cal, n_cal = banked_dep.get("calibration_files"), new_dep.get("calibration_files")
    b_id, n_id = _identity(banked_dep).get("calibration"), _identity(new_dep).get("calibration")
    if b_cal and n_cal:
        if b_cal != n_cal:
            fails.append(f"calibration digests differ: banked {b_cal} vs new {n_cal}")
    elif b_id and n_id:
        if b_id != n_id:
            fails.append(f"calibration digest differs: banked {b_id} vs new {n_id}")
    else:
        warns.append("calibration digest not recorded by the banked run: unverifiable")
    b_mod, n_mod = banked_dep.get("model_files_sha256"), new_dep.get("model_files_sha256")
    if b_mod and n_mod:
        if b_mod != n_mod:
            fails.append("checkpoint digests differ (model_files_sha256)")
    else:
        warns.append("checkpoint digests not recorded by one run: unverifiable")
    return fails, warns


def _identity(dep: Mapping[str, Any]) -> dict[str, Any]:
    ident = dep.get("lane")
    if isinstance(ident, Mapping):
        return dict(ident)
    return {}


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cmd_select(args: argparse.Namespace) -> int:
    index = load_banked_index(args.banked_lane)
    gids = select_gens(index, args.n, seed=args.seed)
    path = write_subset_manifest(index, gids, args.work / "manifest")
    print(json.dumps({"selected": gids, "manifest": str(path)}))
    return 0


def _cmd_harvest(args: argparse.Namespace) -> int:
    from pleroma.harvest import anamnesis_seam

    anamnesis_seam.verify_anamnesis_pin(allow_mismatch=bool(args.allow_anamnesis_mismatch) or None)
    manifest = args.work / "manifest" / "manifest.json"
    lane = anamnesis_seam.GpuHarvestLane.open(
        preset=args.preset, model_path=str(args.model_path), calib_dir=args.calib_dir,
        device=str(args.device))
    for tag in ("new", "repeat"):
        written = lane.run(manifest, args.work / tag)
        logger.warning("%s: %d gens", tag, len(written))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    sel = json.loads((args.work / "manifest" / "selection.json").read_text())
    gids = [int(g) for g in sel["gen_ids"]]
    banked = {g: Path(sel["banked"][str(g)]) for g in gids}
    with np.load(args.discriminants, allow_pickle=False) as z:
        full_names = [str(n) for n in z["FULL_names"]]
        full_scale = np.asarray(z["FULL_scale"], dtype=np.float64)
    new_dir, rep_dir = args.work / "new", args.work / "repeat"
    new_dep = json.loads((new_dir / "deployment.json").read_text())
    shard_deps = {p.parent.parent for p in banked.values()}
    banked_deps = [json.loads((s / "out" / "deployment.json").read_text()) for s in sorted(shard_deps)]
    pre: list[str] = []
    pre_warn: list[str] = []
    for bd in banked_deps:
        f, w = preconditions(bd, new_dep)
        pre += f
        pre_warn += w
    pre_warn = sorted(set(pre_warn))
    slices = None
    first_json = new_dir / f"gen_{gids[0]:03d}.json"
    if first_json.exists():
        slices = json.loads(first_json.read_text()).get("tier_slices")
    repeat = compare_runs({g: (new_dir / f"gen_{g:03d}.npz", rep_dir / f"gen_{g:03d}.npz")
                           for g in gids}, full_names, full_scale, z_tol=0.0, atol=0.0,
                          slices=slices)
    main_cmp = compare_runs({g: (banked[g], new_dir / f"gen_{g:03d}.npz") for g in gids},
                            full_names, full_scale, z_tol=args.z_tol, atol=args.atol,
                            slices=slices)
    verdict = main_cmp["verdict"]
    if repeat["verdict"] != "EXACT":
        verdict = "FAIL"
        main_cmp["problems"].append("the repeat harvest is not bit-exact: the lane is "
                                    "not deterministic on this node, so no value "
                                    "comparison below is trustworthy")
    if pre:
        verdict = "PRECONDITION"
    report = {
        "verdict": verdict,
        "preconditions": pre,
        "precondition_warnings": pre_warn,
        "comparison": main_cmp,
        "repeat": {k: repeat[k] for k in ("verdict", "n_values", "n_exact", "max_abs", "problems")},
        "lane_id": {"banked": sorted({str(bd.get("lane_id")) for bd in banked_deps}),
                    "new": new_dep.get("lane_id")},
        "stack_diff": [stack_diff(_identity(bd), _identity(new_dep)) for bd in banked_deps],
        "anamnesis_commit": new_dep.get("anamnesis_commit"),
        "gen_ids": gids,
    }
    out = args.work / "parity_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    c = main_cmp
    print(f"GPU-LANE PARITY: {verdict}  gens={c['n_gens']} values={c['n_values']} "
          f"exact={c['n_exact']} max_z={c['max_z']:.3g} (tol {args.z_tol:g}) "
          f"max_abs={c['max_abs']:.3g} repeat={repeat['verdict']}")
    for p in pre + c["problems"]:
        print(f"  ! {p}")
    for p in pre_warn:
        print(f"  ? {p}")
    for fam, zmax in list(c["per_family_max_z"].items())[:12]:
        print(f"  family {fam:<24} max z {zmax:.3g}")
    print(f"  lane_id banked={report['lane_id']['banked']} new={report['lane_id']['new']} "
          "(differs by design; rows mix only once ruled equivalent in profiles/lane_equivalence.json)")
    if report["stack_diff"] and any(report["stack_diff"]):
        print(f"  stack differs from the banked run: {report['stack_diff']}")
    print(f"  report: {out}")
    return {"EXACT": 0, "PASS": 0, "FAIL": 1}.get(verdict, 2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="pick banked gens, write their lane manifest (CPU)")
    s.add_argument("--banked-lane", type=Path, required=True,
                   help="the banked lane dir (shard_NN/{manifest.json,out/})")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("harvest", help="run the selection twice through the seam (GPU)")
    h.add_argument("--work", type=Path, required=True)
    h.add_argument("--model-path", required=True)
    h.add_argument("--calib-dir", type=Path, required=True)
    h.add_argument("--preset", default="70b-modelc")
    h.add_argument("--device", default="cuda:0")
    h.add_argument("--allow-anamnesis-mismatch", action="store_true")
    c = sub.add_parser("compare", help="preconditions, names, values -> report (CPU)")
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--discriminants", type=Path, required=True,
                   help="the corpus manifest npz carrying FULL_names / FULL_scale")
    c.add_argument("--z-tol", type=float, default=Z_TOL)
    c.add_argument("--atol", type=float, default=ATOL)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    fn = {"select": _cmd_select, "harvest": _cmd_harvest, "compare": _cmd_compare}[args.cmd]
    return fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ATOL", "Z_TOL", "compare_runs", "compare_vectors", "load_banked_index", "main",
           "preconditions", "select_gens", "stack_diff", "write_subset_manifest"]
