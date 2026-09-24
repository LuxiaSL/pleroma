"""v1a EXPORT: the registered fit, refit on the whole bank, as a LoomMap.

The registered v1a analysis is FROZEN; its artifacts carry the token
``v1a-fit-001``. The registered fit beat the shelf baseline
held out (docs/FINDINGS.md §1); this module turns that fit into the map
the loom serves.

★ WHY THE PRODUCT IS A `LoomMap` npz AND NOT A NEW FORMAT. v1a is a LINEAR map
from fan-relative signatures to per-site hidden displacements, and the loom's
deployed one-vs-rest op is already a fan contrast computed in code space:

    contrastive_code(codes, i) = code_i - mean_{j != i} code_j
    code_i                     = Vt @ ((x_i - mu_in) @ W)          (`lever_of`)
    => contrastive_code         = Vt @ ((z_i - mean_{j!=i} z_j) @ W)
                                = Vt @ (dz_i @ W)

which is EXACTLY v1a's registered input construction
(`pleroma.map.build.join.fan_center`, leave-one-out on both sides). `mu_in`
and `mu_y` cancel in the difference. And because `dz_i @ W` lies in W's row space, which `Vt`'s orthonormal rows span,
`lever_from_code(..., differential=True)` reconstructs it EXACTLY (Vtᵀ Vt v = v),
not approximately. So every certified path — the probe's
`pleroma.probe.orchestrator.contrastive_code`, `/wear_code` — feeds v1a precisely
the quantity it was trained on, with zero new inference code.
`test_v1a_export.py` pins both identities numerically; they are the
justification for this whole approach, so those tests are load-bearing.

★ THE BINS BLOCK IS A FORMAT-LEVEL NO-OP. `LoomMap`'s input contract is
`[z_corpus(2713) | bins_std(420)]`, and v1a's W is 2713 -> 12288. The exported
`W_U` is therefore v1a's U with 420 ZERO ROWS appended, so the bins block
contributes exactly nothing to any code or lever; `bins_mu/bins_sd/bins_dead`
are copied verbatim from the wide map purely so the format is satisfied. A test
asserts that perturbing the bins input changes the output by exactly 0.0.

★ TWO RULERS, BOTH BANKED, ONLY ONE WORN. The `norm_ref` KEY carries the WIDE
MAP's ruler, because a comparison between maps norm-matches every map to one
reference so that at equal alpha only DIRECTION differs. v1a's own ruler, by
the usual convention (the per-site median of the L2 norms of v1a's own
predicted deltas over the bank), is banked beside it as `norm_ref_v1a_own` and
is NOT what a wear uses. `meta` states this in words as well as keys.

★ NO CROSS-VALIDATION HERE, ON PURPOSE. The registered claim (top-1 .9339 held
out, at 3B) was already made on grouped-CV predictions (`pleroma.map.build.cv`).
This is the DEPLOYMENT object, so it is fit on every joined row at the
REGISTERED operating point (lambda=1e4, rank 64, input dz, target raw — the
registered point, not the sweep's best cell, because picking the best cell of a
descriptive sweep would be selection). The self-check below is therefore
in-sample and is a PLUMBING gate, not evidence.

★ THE OPERATING POINT IS A PARAMETER, THE REGISTERED ONE IS THE DEFAULT.
`--lam`/`--rank` select any cell of v1a's registered grid, with `--rank 0`
meaning FULL RANK exactly as the fit grid's key `r0` does. Two rules keep this from eroding the registered artifact:

  1. running with no `--lam`/`--rank` reproduces the registered export's array
     payload EXACTLY — every added field is gated on the point being
     overridden (`test_v1a_export.py` pins this);
  2. overriding the point REQUIRES `--selfcheck-floor`, because the built-in
     .93 is the registered point's held-out bar and a better cell must be
     gated on its own (`HELDOUT_GRID`).

★ AND SO IS THE TARGET NORMALIZATION. `--target {raw,sitenorm}` selects the
other half of the fit's registered grid, under the same two rules plus one
more: retrieval candidates stay the RAW measured deltas for every variant (the
fit's own rule), so a `sitenorm` export's self-check is the same
readout a `raw` export reports and the two numbers are directly comparable.
`--target raw` is the default and a pure no-op. See `TARGETS`.

  ★ "byte-for-byte" is literal and was checked, not assumed:
  `np.savez_compressed` writes its zip members through `ZipFile.open(..., 'w')`,
  whose `ZipInfo` date is the fixed DOS epoch (the start of 1980) rather than
  the wall clock, so
  two runs that produce the same arrays produce the same FILE sha256. The tests
  pin the array payload, the `meta` JSON string AND the file sha.

On a GPU (the ridge + SVD are float64; ~1 GB of targets)::

    PYTHONPATH=. python3 -m pleroma.map.build.export \\
        --pairs-dir $B/map_wide/pairs \\
        --levers $B/map_wide/levers/levers.npz \\
        --hiddens $B/map_v0a/directive_hiddens/mean_hiddens.npz ... \\
        --bins $B/bins/binsB_w6wide_all.npz \\
        --wide-map $B/map_wide/loom_map_w6wide_r64.npz \\
        --out $B/v1a/loom_map_v1a_r64.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pleroma.map.build.ridge import resolve_rank  # noqa: F401 — re-exported
from pleroma.map.build.shelf import ShelfError, ShelfStats, compute_shelf
from pleroma.map.svd import canonical_signs
from pleroma.map.build.join import (  # noqa: F401 — re-exported for callers
    FAN_SOURCE_LEVER,
    FAN_SOURCE_MEMBER,
    FAN_SOURCES,
    TARGET_DEFAULT,
    TARGETS,
    BankJoin,
    build_target,
    fan_center,
    join_bank,
    member_fan_index,
    retrieval,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("v1a_export")

#: The REGISTERED operating point. Not the grid's best: choosing the best cell
#: of a descriptive sweep would be selection.
LAM: float = 1e4
RANK: int = 64
PREREG_TOKEN: str = "v1a-fit-001"


#: `LoomMap`'s input contract, and the width of the block v1a does not use.
Z_DIM: int = 2713
BINS_DIM: int = 420

#: The export refuses below this in-sample within-fan top-1. The registered
#: HELD-OUT number is .9339, so an in-sample re-derivation through the exported
#: artifact must be at least that good; anything under .93 means the export path
#: (zero-padding, factor order, mu handling, join) is wrong.
SELFCHECK_FLOOR: float = 0.93

#: ★ THE GATE IS SET PER OPERATING POINT. The floor above is `.93` because
#: .9339 is what the REGISTERED point scores held out. A different cell on the same registered grid has a different held-out
#: figure, and reusing .93 would let a badly-exported full-rank map — whose
#: held-out retrieval is .9837 — pass a gate it should fail by four points. So
#: the floor travels with the point: `--selfcheck-floor` is required whenever
#: the operating point is overridden, and `HELDOUT_GRID` is the registered
#: sweep (`outputs/v1a/v1a_report.json`) it must come from.
#: ★ THESE ARE THE 3B's FIGURES ONLY: another model's grid has other figures.
#: They feed the floor hint and the off-point `meta` note; the export REPORT's
#: held-out figures come from `--heldout-report` (`heldout_from_fit_report`),
#: never from this table.
HELDOUT_GRID: dict[tuple[float, int], float] = {
    (1e3, 0): 0.9837,      # dz|raw|lam1000|r0    — full rank
    (1e3, 64): 0.9585,     # dz|raw|lam1000|r64
    (1e4, 0): 0.9543,      # dz|raw|lam10000|r0
    (1e4, 64): 0.9339,     # dz|raw|lam10000|r64  — the REGISTERED point
    (3e4, 0): 0.9161,
    (3e4, 64): 0.8973,
}

#: The deployment-identity gate: the contrastive-code path must reproduce
#: `dz @ W` to float64 roundoff, relative to the vector's own scale.
IDENTITY_TOL: float = 1e-10

#: ★ THE FACTORS SHIP IN float64, unlike the wide map's float32. The whole
#: approach rests on `Vtᵀ Vt v = v` for v in W's row space, and a float32 `Vt` is
#: only orthonormal to ~1e-7 — which would silently demote an exact identity to
#: a 1e-7 approximation, in the one place the design leans hardest. Rank 64
#: factors are ~8 MB in float64, so the cost is nothing. `LoomMap` casts to
#: float64 on read either way, so no consumer can tell the difference except by
#: being more accurate.
FACTOR_DTYPE = np.float64

#: Refuse a join that is not the bank the fit was registered on.
MIN_ROWS: int = 8000


#: Where the export REPORT's held-out figures come from when no fit report is
#: supplied. Never a number: the export cannot know which model's registered
#: sweep applies, and a literal here would stamp the 3B's .9339 on every
#: model's export.
HELDOUT_NOT_SUPPLIED: str = (
    "not supplied — pass --heldout-report <this model's v1a_fit "
    "v1a_report.json>; v1a_export does not know which model's registered "
    "sweep applies and will not guess")


def grid_key(lam: float, rank: int) -> str:
    """The fit grid's key (`pleroma.map.build.cv`) for the registered input/target at (lam, rank)."""
    return f"dz|raw|lam{float(lam):g}|r{int(rank)}"


def heldout_from_fit_report(report: Path | None, lam: float,
                            rank: int) -> dict[str, Any]:
    """The export report's held-out figures, read from THIS model's fit report.

    Returns ``registered_heldout_top1`` (the registered point, lambda=1e4 rank
    64), ``this_cell_heldout_top1`` (the exported point's grid cell, None if the
    sweep has no such cell) and ``heldout_source``. With ``report=None`` both
    figures are None and the source says so. Raises ValueError on a report
    that is not a v1a_fit report at the registered construction, or whose
    grid disagrees with its own registered test.
    """
    if report is None:
        return {"registered_heldout_top1": None, "this_cell_heldout_top1": None,
                "heldout_source": HELDOUT_NOT_SUPPLIED}
    try:
        blob = json.loads(Path(report).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read held-out report {report}: {exc}") from exc
    if not isinstance(blob, dict) or blob.get("stage") != "v1a_fit":
        raise ValueError(f"{report} is not a v1a_fit report (stage "
                         f"{blob.get('stage') if isinstance(blob, dict) else None!r})")
    pp = blob.get("primary_point") or {}
    if (pp.get("input"), pp.get("target"), float(pp.get("lambda", -1)),
            int(pp.get("rank", -1))) != ("dz", "raw", float(LAM), int(RANK)):
        raise ValueError(f"{report}: primary_point {pp} is not the registered "
                         f"construction (dz|raw, lambda={LAM:g}, rank={RANK})")
    grid = blob.get("grid") or {}
    reg_cell = grid.get(grid_key(LAM, RANK))
    if reg_cell is None:
        raise ValueError(f"{report}: grid has no {grid_key(LAM, RANK)} cell")
    reg = float(reg_cell["retrieval_top1"])
    rt = (blob.get("registered_test") or {}).get("v1a_top1")
    if rt is not None and round(float(rt), 4) != round(reg, 4):
        raise ValueError(f"{report}: grid registered cell {reg} disagrees with "
                         f"registered_test.v1a_top1 {rt}")
    cell = grid.get(grid_key(lam, rank))
    return {"registered_heldout_top1": reg,
            "this_cell_heldout_top1": (None if cell is None
                                       else float(cell["retrieval_top1"])),
            "heldout_source": f"{report} (v1a_fit grid, "
                              f"{grid_key(LAM, RANK)} / {grid_key(lam, rank)})"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zero_pad_factors(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, bins_dim: int = BINS_DIM
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v1a's rank-r SVD factors -> `LoomMap`'s [z|bins] input width.

    Only ``U`` grows: it is indexed by the INPUT dimension, so appending
    ``bins_dim`` zero rows makes every bins coordinate multiply zero. ``S`` and
    ``Vt`` are untouched, and the product ``(U*S) @ Vt`` is v1a's W with a block
    of zeros under it — which is what makes the bins half of `LoomMap`'s input
    contract a strict no-op rather than a small contribution.
    """
    if u.ndim != 2 or vt.ndim != 2 or s.ndim != 1:
        raise ValueError(f"bad factor shapes U{u.shape} S{s.shape} Vt{vt.shape}")
    if u.shape[1] != s.size or vt.shape[0] != s.size:
        raise ValueError(
            f"factors disagree on rank: U{u.shape} S{s.shape} Vt{vt.shape}")
    if bins_dim < 0:
        raise ValueError(f"bins_dim {bins_dim} must be >= 0")
    pad = np.zeros((bins_dim, u.shape[1]), dtype=u.dtype)
    return np.vstack([u, pad]), s, vt






def contrast_rows(x: np.ndarray, fans: Sequence[np.ndarray]) -> np.ndarray:
    """Leave-one-out fan contrast, rowwise — `pleroma.map.build.join.fan_center`.

    The fit's own function rather than a reimplementation, so the export
    provably centers the way the fit did.
    """
    return fan_center(np.asarray(x, dtype=np.float64), list(fans))


def deployment_contrast(
    codes: np.ndarray, vt: np.ndarray, fans: Sequence[np.ndarray]
) -> np.ndarray:
    """The DEPLOYED path's predicted delta for every bank member.

    `pleroma.probe.orchestrator.contrastive_code` then
    `LoomMap.lever_from_code(differential=True)`, vectorised over the whole
    bank: contrast the per-member codes inside the fan, then expand back
    through Vt. No norm-matching — the retrieval readout is a cosine against
    measured deltas and the per-site ruler is a wear-time transform, not part
    of the map's prediction.
    """
    return contrast_rows(codes, fans) @ vt










def main() -> int:  # noqa: C901 — one linear procedure with explicit gates
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--levers", type=Path, required=True)
    ap.add_argument("--hiddens", type=Path, nargs="+", required=True)
    ap.add_argument("--bins", type=Path, required=True,
                    help="bins npz — read only to prove the bins block is a "
                         "no-op on REAL rows, never to fit anything")
    ap.add_argument("--wide-map", type=Path, default=None,
                    help="LEGACY: a banked wide map to read v3_mu/v3_sd/"
                         "v3_dead (the shared preprocessing), bins_* (format) "
                         "and the SHARED norm_ref ruler from. Omit it and "
                         "pass --discriminants: the export then computes the "
                         "same stats itself (pleroma.map.build.shelf) — no "
                         "wide map needs fitting.")
    ap.add_argument("--discriminants", type=Path, default=None,
                    help="the frozen discriminants npz; REQUIRED without "
                         "--wide-map (the shelf stats are computed from it)")
    ap.add_argument("--norm-ref-bank", type=Path, default=None,
                    help="lever bank whose median per-site norms are the "
                         "ruler (alpha=1); default: --levers, which is what "
                         "every banked wide map used")
    ap.add_argument("--verify-rows", type=int, default=25,
                    help="rows on which the recomputed z_corpus is checked "
                         "against the pairs matrix (folded mode only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--rank", type=int, default=RANK,
                    help="SVD truncation; 0 (or >= min(W.shape)) means FULL "
                         "RANK, the same convention the fit grid writes as "
                         "'r0'. The registered point is 64.")
    ap.add_argument("--selfcheck-floor", type=float, default=None,
                    help="in-sample within-fan top-1 the reloaded artifact must "
                         "reach. REQUIRED when the operating point is "
                         "overridden: the default .93 is the registered point's "
                         "bar and would under-gate a better cell. Use that "
                         "cell's HELD-OUT figure from the fit report.")
    ap.add_argument("--target", choices=list(TARGETS), default=TARGET_DEFAULT,
                    help="target normalization, as the fit grid spells it. "
                         "'raw' is the REGISTERED construction and the default "
                         "(a pure no-op: the registered export is unchanged "
                         "byte-for-byte). 'sitenorm' scales each site block of "
                         "the same LOO delta to unit L2 before the ridge, which "
                         "is meant to halve the target's loading on reply "
                         "length. "
                         "Selecting it OVERRIDES the registered "
                         "operating point and therefore REQUIRES "
                         "--selfcheck-floor, taken from THIS install's own fit "
                         "report, not from the 3B HELDOUT_GRID below.")
    ap.add_argument("--z-dim", type=int, default=None,
                    help="OPTIONAL assertion of this install's z_corpus width. "
                         "The width is DERIVED from the wide map (v3_mu.size: "
                         "3B 2713, 8B 2769, 70B 4086); passing a value that "
                         "disagrees refuses. BINS_DIM (420) is the format's "
                         "own contract.")
    ap.add_argument("--fan-source", choices=FAN_SOURCES,
                    default=FAN_SOURCE_LEVER,
                    help="where fan membership comes from. 'lever_group' "
                         "(default, registered): the lever file's "
                         "member_group_index, which DROPS every member of a "
                         "group the lever build could not 2-means. 'member_fan': "
                         "(corpus, prompt_id, wave) for every member in the "
                         "lever file, lever or not — v1a is one-vs-rest and "
                         "never reads the lever. Default output is unchanged "
                         "byte-for-byte; the override is banked in meta.")
    ap.add_argument("--heldout-report", type=Path, default=None,
                    help="THIS model's fit report (the v1a_report.json the fit "
                         "writes). The "
                         "export report's registered_heldout_top1 / "
                         "this_cell_heldout_top1 are read from it; without it "
                         "they are null. Report-only: the npz is unaffected.")
    args = ap.parse_args()
    try:
        heldout = heldout_from_fit_report(args.heldout_report, args.lam,
                                          args.rank)
    except ValueError as exc:
        logger.error("--heldout-report refused: %s", exc)
        return 1
    # ── the shelf stats: standardisers + ruler ───────────────────────────────
    # Either read back out of a banked wide map (legacy) or COMPUTED here by
    # the same code `wide` uses — byte-identical arrays either way.
    if (args.wide_map is None) == (args.discriminants is None):
        logger.error("pass exactly one of --wide-map (legacy) or "
                     "--discriminants (folded: stats computed here)")
        return 1
    try:
        if args.wide_map is not None:
            shelf = ShelfStats.from_wide_map(args.wide_map)
        else:
            shelf = compute_shelf(
                args.pairs_dir, args.levers, args.bins, args.discriminants,
                args.norm_ref_bank or args.levers,
                verify_rows=int(args.verify_rows)).stats
    except ShelfError as exc:
        logger.error("shelf stats refused: %s", exc)
        return 1
    folded = args.wide_map is None
    # ★ z_corpus width is DERIVED from the artifacts, never defaulted (a
    # default would be one model's width, the 3B's 2713, on every model): the
    # shelf's v3_mu size IS this install's width. A passed --z-dim is only an
    # assertion.
    z_dim = shelf.z_dim
    if args.z_dim is not None and int(args.z_dim) != z_dim:
        logger.error("--z-dim %d disagrees with the shelf's z_corpus width %d "
                     "— refusing (the flag is an assertion, not an override)",
                     int(args.z_dim), z_dim)
        return 1

    target = str(args.target)
    registered_point = bool(args.lam == LAM and args.rank == RANK
                            and target == TARGET_DEFAULT)
    if not registered_point:
        logger.warning(
            "operating point overridden to lambda=%g rank=%d target=%s — the "
            "REGISTERED point is lambda=%g rank=%d target=%s; the override "
            "is banked in meta",
            args.lam, args.rank, target, LAM, RANK, TARGET_DEFAULT)
    # ★ the floor must travel with the point (see HELDOUT_GRID): a full-rank
    # map that scores .9837 held out would sail through the registered point's
    # .93 bar while being four points short of what it should reach.
    if args.selfcheck_floor is None:
        if not registered_point:
            # ★ the suggestion comes from THIS model's fit report, never from
            # the 3B HELDOUT_GRID record, which is one model's figures.
            ref = (heldout["this_cell_heldout_top1"]
                   if target == TARGET_DEFAULT else None)
            logger.error(
                "--selfcheck-floor is REQUIRED when the operating point is "
                "overridden (lambda=%g rank=%d target=%s). The registered "
                "default %.2f is that point's bar only. %s",
                args.lam, args.rank, target, SELFCHECK_FLOOR,
                (f"This cell's held-out figure is {ref} "
                 f"(from the --heldout-report fit report) — pass "
                 f"--selfcheck-floor {ref}.") if ref is not None
                else "No held-out figure for this cell (pass THIS install's "
                     "fit report as --heldout-report); read its "
                     "retrieval_top1 out of that report's grid under the key "
                     f"'dz|{target}|lam{args.lam:g}|r{args.rank}' and pass it "
                     "explicitly, or do not export it.")
            return 1
        selfcheck_floor = SELFCHECK_FLOOR
    else:
        selfcheck_floor = float(args.selfcheck_floor)
        if not 0.0 < selfcheck_floor <= 1.0:
            logger.error("--selfcheck-floor %.4f is not a top-1 rate in (0,1]",
                         selfcheck_floor)
            return 1

    import torch

    try:
        bank = join_bank(args.pairs_dir, args.levers, list(args.hiddens),
                         fan_source=args.fan_source)
    except ValueError as exc:
        logger.error("join refused: %s", exc)
        return 1
    z, h, fans, sites = bank.z, bank.h, bank.fans, bank.sites
    n = bank.n
    if n < MIN_ROWS:
        # policy call 5: a first user's small corpus must be exportable; the
        # self-check floor still gates plumbing, and validate reports the size.
        logger.warning("join is %d rows, below %d (the 3B registered bank's "
                       "scale) — exporting anyway; held-out figures from a "
                       "bank this small are noisy", n, MIN_ROWS)
    if z.shape[1] != z_dim:
        logger.error("signatures are %d-d, not %d — LoomMap's input contract "
                     "would not hold", z.shape[1], z_dim)
        return 1

    # ── the registered constructions, both sides leave-one-out ───────────────
    dz = contrast_rows(z, fans)
    d_raw = contrast_rows(h, fans)
    if h.shape[1] % len(sites):
        logger.error("target width %d is not divisible by %d sites",
                     h.shape[1], len(sites))
        return 1
    hidden_dim = bank.hidden

    # ★ The REGRESSION TARGET. `d_raw` stays bound to the measured deltas
    # because it is ALSO the retrieval candidate set for every variant (see
    # TARGETS rule 3); `d_fit` is what the ridge is solved against.
    try:
        d_fit = build_target(d_raw, target, len(sites))
    except ValueError as exc:
        logger.error("target refused: %s", exc)
        return 1
    if target != TARGET_DEFAULT:
        logger.info(
            "TARGET = %s: fitting against per-site unit-norm deltas (median "
            "per-site raw norm %s), while retrieval candidates remain the RAW "
            "measured deltas — the self-check below is therefore the same "
            "readout the raw export reports, and the two are comparable",
            target,
            [round(float(x), 4) for x in np.median(
                np.linalg.norm(
                    d_raw.reshape(n, len(sites), hidden_dim), axis=2), axis=0)])

    # ── the deployment fit: ALL rows, centered ridge, rank-truncated ──────────
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        logger.warning("cuda unavailable — falling back to %s", device)
    xt = torch.as_tensor(dz, dtype=torch.float64, device=device)
    yt = torch.as_tensor(d_fit, dtype=torch.float64, device=device)
    mu_in_z = xt.mean(dim=0)
    mu_y_t = yt.mean(dim=0)
    xc = xt - mu_in_z
    yc = yt - mu_y_t
    gram = xc.T @ xc
    gram += args.lam * torch.eye(xc.shape[1], dtype=torch.float64, device=device)
    w_full = torch.linalg.solve(gram, xc.T @ yc)
    u_t, s_t, vt_t = torch.linalg.svd(w_full, full_matrices=False)
    try:
        r = resolve_rank(int(args.rank), tuple(w_full.shape))
    except ValueError as exc:
        logger.error("rank refused: %s", exc)
        return 1
    full_rank = r == int(min(w_full.shape))
    if full_rank:
        logger.warning(
            "FULL RANK: the SVD truncation is OFF (r=%d = min%s). The factors "
            "carry the whole spectrum, the served CODE is %d-d rather than %d, "
            "and every consumer that banked a rank-%d code from this family "
            "will be REFUSED by lever_from_code's dimensionality check.",
            r, tuple(w_full.shape), r, RANK, RANK)
    u = u_t[:, :r].cpu().numpy()
    s = s_t[:r].cpu().numpy()
    vt = vt_t[:r].cpu().numpy()
    # ★ One orientation on every stack: without this, re-exporting the same
    # map on another stack can flip code axes (32 of 64 on a 70B map;
    # pleroma.map.svd).
    u, s, vt, sign_report = canonical_signs(u, s, vt)
    if sign_report.ambiguous:
        logger.warning("SVD sign convention is ambiguous on components %s "
                       "(top-2 |Vt| entries tied with opposite signs): those "
                       "code axes may still differ across stacks",
                       list(sign_report.ambiguous))
    mu_in_z_np = mu_in_z.cpu().numpy()
    mu_y = mu_y_t.cpu().numpy()
    del xt, yt, xc, yc, gram, w_full, u_t, s_t, vt_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    w_trunc = (u * s) @ vt  # [2713, 12288] — v1a's deployed W
    logger.info("fit W %s at lambda=%g rank=%d target=%s (spectrum head %s)",
                w_trunc.shape, args.lam, r, target,
                [round(float(x), 2) for x in s[: min(5, r)]])

    # ── the wide map's banked arrays (shared preprocessing + shared ruler) ────
    v3_mu, v3_sd, v3_dead = shelf.v3_mu, shelf.v3_sd, shelf.v3_dead
    bins_mu, bins_sd, bins_dead = shelf.bins_mu, shelf.bins_sd, shelf.bins_dead
    wide_norm_ref = np.asarray(shelf.norm_ref, dtype=np.float64)
    wide_sites = list(shelf.sites)
    if wide_sites != sites:
        logger.error("wide map sites %s != levers sites %s — one ruler across "
                     "arms is impossible if the install geometry differs",
                     wide_sites, sites)
        return 1
    if v3_mu.size != z_dim or bins_mu.size != BINS_DIM:
        logger.error("wide map declares z=%d bins=%d, expected %d/%d",
                     v3_mu.size, bins_mu.size, z_dim, BINS_DIM)
        return 1

    # ── v1a's OWN ruler: median per-site norm of its predicted deltas ─────────
    pred_own = dz @ w_trunc
    pred_site_norms = np.linalg.norm(
        pred_own.reshape(n, len(sites), hidden_dim), axis=2)
    norm_ref_own = np.nanmedian(pred_site_norms, axis=0)
    logger.info("v1a's OWN norm_ref (banked, NOT worn): %s",
                [round(float(x), 4) for x in norm_ref_own])
    logger.info("WIDE map's norm_ref (the ruler this artifact carries): %s",
                [round(float(x), 4) for x in wide_norm_ref])

    # ── assemble the LoomMap-format artifact ─────────────────────────────────
    w_u, w_s, w_vt = zero_pad_factors(u, s, vt, BINS_DIM)
    mu_in = np.concatenate([mu_in_z_np, np.zeros(BINS_DIM, dtype=np.float64)])
    meta: dict[str, Any] = {
        "lam": float(args.lam),
        "rank": int(r),
        "n_rows": int(n),
        "n_fans": len(fans),
        "feature_order": f"z_corpus({z_dim}) then bins_std({BINS_DIM})",
        "discriminants": shelf.discriminants,
        "discriminants_sha256": shelf.discriminants_sha256,
        "bins_config": shelf.bins_config,
        "prereg": ("the registered operating point (dz -> raw, lambda 1e4, rank 64), "
                   "fixed before the fit was run"),
        "prereg_token": PREREG_TOKEN,
        "stage": "v1a_export — the deployment refit of the registered v1a fit",
        "svd_orientation": sign_report.as_meta(),
        "operating_point": {
            "lambda": float(args.lam), "rank": int(r),
            "input": "dz (LOO fan-centered z_corpus)",
            "target": (
                "raw (LOO fan-centered mean hiddens)" if target == "raw" else
                "sitenorm (that same LOO fan-centered delta with each site "
                "block scaled to unit L2 before the ridge; retrieval "
                "candidates are still the RAW measured deltas)"),
            "registered": bool(args.lam == LAM and r == RANK
                               and target == TARGET_DEFAULT),
            "why": "the REGISTERED point, not the "
                   "sweep's best cell (lambda=1e3, full rank, .9889), which "
                   "would be selection on a descriptive sweep; rank 64 also "
                   "matches the wide map's rank, keeping the arms like-for-like",
        },
        "cv": "NONE — this is the deployment object; the registered claim was "
              "made on grouped-CV held-out predictions (docs/FINDINGS.md §1)",
        "input_transform": (
            "raw v3 -> (x - FULL_mean)/FULL_scale -> float32 -> (. - v3_mu)/"
            "v3_sd, dead->0 -> fan-contrast -> W. v3_* are COPIED from the "
            "wide map: build_pairs' corpus "
            "standardization IS v1a's training space, so v1a and the wide map "
            "share byte-identical preprocessing and only the learned W differs."
        ),
        "factor_dtype": (
            "float64 (fit_loom_map ships float32). The deployment identity is "
            "Vtᵀ Vt v = v on W's row space, and a float32 Vt is only "
            "orthonormal to ~1e-7; rank-64 float64 factors cost ~8 MB, so the "
            "identity is kept exact instead of approximate."
        ),
        "bins_block": (
            f"NO-OP. v1a's W is {z_dim}->{mu_y.size}; W_U carries {BINS_DIM} "
            "ZERO ROWS so every bins coordinate multiplies zero. bins_mu/"
            "bins_sd/bins_dead are copied from the wide map to satisfy "
            "LoomMap's format only, and mu_in's bins block is zeros because "
            "it multiplies those zero rows. Perturbing the bins input changes "
            "nothing (pinned by test_v1a_export)."
        ),
        "norm_ref_source": (
            "THE WIDE MAP'S RULER: every arm compared at one alpha "
            "norm-matches per site to ONE reference so that at equal alpha "
            "they wear identical per-site magnitudes and only DIRECTION "
            "differs. A per-family ruler would confound direction with loudness."
        ),
        "norm_ref_wide_map": (str(args.wide_map) if not folded else
                              "NONE — folded: computed from the ruler bank"),
        "norm_ref_v1a_own": [float(x) for x in norm_ref_own],
        "norm_ref_v1a_own_note": (
            "v1a's OWN ruler, by the same convention — the per-site median of the L2 "
            "norms of v1a's predicted deltas over the bank. BANKED FOR THE "
            "RECORD AND NOT WORN: the `norm_ref` key is the wide map's."
        ),
        "deployment_identity": (
            "contrastive_code(codes, i) = Vt @ (dz_i @ W) because mu_in/mu_y "
            "cancel in the fan contrast, and lever_from_code(differential="
            "True) inverts Vt exactly on W's row space. The certified "
            "one-vs-rest path therefore feeds v1a exactly what it was "
            "trained on, with no new inference code."
        ),
        "sign_convention": (
            "the output is the predicted displacement of the CONTRASTED member "
            "relative to its fan mean (h_i - mean_{j!=i} h_j), i.e. it points "
            "toward the prescribed future's own basin"
        ),
        "alpha_convention": "norm-match each site row to norm_ref, then alpha "
                            "scales (the dose axis of the dose ladders)",
        "sources": {
            "pairs_dir": str(args.pairs_dir),
            "levers": str(args.levers),
            "levers_sha256": sha256_of(args.levers),
            "hiddens": [str(p) for p in args.hiddens],
            "hiddens_sha256": [sha256_of(p) for p in args.hiddens],
            "bins": str(args.bins),
            **({"wide_map": str(args.wide_map),
                "wide_map_sha256": sha256_of(args.wide_map)} if not folded else {
                "shelf": "computed (pleroma.map.build.shelf)",
                "discriminants": str(args.discriminants),
                "discriminants_sha256": sha256_of(args.discriminants),
                "bins_sha256": sha256_of(args.bins),
                "norm_ref_bank": str(args.norm_ref_bank or args.levers),
                "norm_ref_bank_sha256": sha256_of(args.norm_ref_bank or args.levers),
            }),
        },
    }

    # ★ EVERY key below is added ONLY off the registered point, so the
    # registered export's `meta` JSON — and therefore every array in the
    # artifact — is unchanged by this parameterisation. Pinned by
    # `test_v1a_export.py::test_registered_point_payload_is_unchanged_*`.
    if not registered_point:
        meta["operating_point"]["target_key"] = target
        meta["operating_point"]["grid_key"] = (
            f"dz|{target}|lam{args.lam:g}|r{args.rank}")
        meta["operating_point"]["overridden_from_registered"] = {
            "registered_lambda": float(LAM), "registered_rank": int(RANK),
            "registered_target": TARGET_DEFAULT,
            "heldout_top1_of_THIS_cell": (
                heldout["this_cell_heldout_top1"]
                if target == TARGET_DEFAULT else None),
            "source": heldout["heldout_source"],
            "source_note": (
                "heldout_top1_of_THIS_cell comes from THIS model's v1a_fit "
                "report (--heldout-report), and is null without "
                "one or for a target other than 'raw' (the fit grid's dz|raw "
                "sweep does not speak for another target). The bar actually "
                "enforced is `selfcheck_floor`."),
        }
        if target != TARGET_DEFAULT:
            meta["target_normalization"] = (
                "SITENORM. The ONLY difference from the registered export is "
                "the regression target: the same bank, the same join, the same "
                "fans, the same dz input, the same lambda and rank, the same "
                "v3_*/bins_*/norm_ref arrays copied from the same wide map. "
                "Each site block of the LOO fan-centered delta is scaled to "
                "unit L2 before the ridge, so per-site target MAGNITUDE — "
                "which tracks how far a member's "
                "length sits from its fan mean — carries no weight and only "
                "DIRECTION does. Because `norm_ref` is byte-identical to the "
                "raw export's, `lever_from_code` norm-matches both maps to the "
                "SAME per-site magnitudes, so at equal alpha the two artifacts "
                "wear identical loudness and differ only in direction.")
        meta["selfcheck_floor"] = float(selfcheck_floor)
        if full_rank:
            meta["full_rank"] = (
                f"THE SVD TRUNCATION IS OFF. W_U/W_S/W_Vt carry the whole "
                f"spectrum (rank {r} = min(W.shape)), so the code LoomMap "
                f"serves is {r}-d, not {RANK}-d. Consequences a reader must "
                f"know: (a) `lever_from_code` refuses any code whose size is "
                f"not {r}, so codes banked under a rank-{RANK} map of this "
                f"same family are NOT wearable here and vice versa; (b) "
                f"`lever_of` rounds the emitted code to 4 dp, and at this rank "
                f"the tail singular values are small enough that some "
                f"coordinates round to exactly zero — the rounding is a real "
                f"lossy step, not a display convention, and is measured in "
                f"`v1a_fullrank_probe.py`; (c) the artifact is ~340 MB against "
                f"the rank-{RANK} map's ~7 MB."
            )

    # ★ same rule as the operating point: added ONLY off the default, so the
    # registered export's meta JSON is untouched.
    if args.fan_source != FAN_SOURCE_LEVER:
        meta["fan_source"] = {
            "source": str(args.fan_source),
            "definition": "fan = (corpus, prompt_id, wave) over EVERY member of "
                          "the lever file, whether or not build_levers could "
                          "form a 2-means lever for its group",
            "why": "v1a is one-vs-rest and never reads the v0 lever; the "
                   "registered join took fans from member_group_index and so "
                   "dropped every group that failed --min-cluster (model C: "
                   "543/1,805 fans, 4,344/14,440 gens). On lever members the "
                   "fans are identical to the registered ones (checked in "
                   "member_fan_index).",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "W_U": w_u.astype(FACTOR_DTYPE),
        "W_S": w_s.astype(FACTOR_DTYPE),
        "W_Vt": w_vt.astype(FACTOR_DTYPE),
        "mu_in": mu_in.astype(FACTOR_DTYPE),
        "mu_y": mu_y.astype(FACTOR_DTYPE),
        "v3_mu": v3_mu, "v3_sd": v3_sd, "v3_dead": v3_dead,
        "bins_mu": bins_mu, "bins_sd": bins_sd, "bins_dead": bins_dead,
        "sites": np.array(sites),
        "norm_ref": wide_norm_ref.astype(np.float64),
        "norm_ref_v1a_own": norm_ref_own.astype(np.float64),
        "norm_ref_which": np.array(
            "norm_ref = WIDE MAP's ruler (decision 1); norm_ref_v1a_own is "
            "v1a's own and is NOT worn"),
        "meta": np.array(json.dumps(meta)),
    }
    # ★ the staging name must END in .npz: np.savez_compressed APPENDS ".npz"
    # to anything that does not, so `<out>.npz.partial` writes
    # `<out>.npz.partial.npz` and the read-back gate then opens a file that was
    # never created. Caught on the first run.
    tmp = args.out.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **payload)

    # ── GATE: re-derive the registered readout THROUGH the exported artifact ──
    # Deliberately loaded back from disk and driven by LoomMap, so the gate
    # exercises the float32 cast, the zero padding, the factor order and the
    # meta the deployment will actually read.
    from pleroma.map.loom_map import LoomMap

    disc = Path(shelf.discriminants)
    if not disc.exists() and args.wide_map is not None:
        disc = Path(args.wide_map).parent / Path(shelf.discriminants).name
    lm = LoomMap(tmp, disc)
    if list(lm.sites) != sites or lm.hidden != hidden_dim or lm.rank != r:
        logger.error("exported artifact reads back as sites=%s hidden=%d "
                     "rank=%d, expected %s/%d/%d — refusing",
                     lm.sites, lm.hidden, lm.rank, sites, hidden_dim, r)
        return 1

    rng = np.random.default_rng(20260921)
    # `lever_of`'s code, associated as x @ (W @ Vtᵀ) so the 12,288-wide
    # intermediate is 3,133x64 instead of 8,536x12,288.
    enc = lm.W @ lm.Vt.T
    bins_probe = rng.standard_normal((n, BINS_DIM)) * 10.0
    codes = (np.hstack([z, bins_probe]) - lm.mu_in) @ enc
    codes_zero = (np.hstack([z, np.zeros((n, BINS_DIM))]) - lm.mu_in) @ enc
    bins_gap = float(np.max(np.abs(codes - codes_zero)))
    pred_dep = deployment_contrast(codes, lm.Vt, fans)    # the worn direction
    if bins_gap != 0.0:
        logger.error("the bins block is NOT a no-op: random bins moved the "
                     "codes by %.3g — refusing", bins_gap)
        return 1
    scale = float(np.median(np.linalg.norm(pred_own, axis=1)))
    ident = float(np.max(np.abs(pred_dep - pred_own))) / max(scale, 1e-12)
    if not np.isfinite(ident) or ident > IDENTITY_TOL:
        logger.error("the deployed contrastive path does not reproduce dz @ W "
                     "(relative max gap %.3g > %.0e) — refusing",
                     ident, IDENTITY_TOL)
        return 1
    hits = retrieval(pred_dep, d_raw, fans)
    top1 = float(hits.mean())
    chance = float(np.mean([1.0 / s_.size for s_ in fans
                            for _ in range(s_.size)]))
    logger.info("SELF-CHECK: within-fan top-1 %.4f (in-sample; chance %.4f; "
                "floor %.4f; this cell's held-out figure %s); bins no-op gap "
                "%.1f; deployment identity rel-gap %.2e",
                top1, chance, selfcheck_floor,
                heldout["this_cell_heldout_top1"],
                bins_gap, ident)
    if top1 < selfcheck_floor:
        logger.error(
            "SELF-CHECK FAILED: in-sample within-fan top-1 %.4f < %.4f. This "
            "operating point's HELD-OUT value is %s, so an in-sample "
            "re-derivation through the reloaded artifact must be at least "
            "that good — something in the export path (padding, factor "
            "order, mu handling, join) is wrong. REFUSING to emit the "
            "artifact.", top1, selfcheck_floor,
            heldout["this_cell_heldout_top1"])
        tmp.unlink(missing_ok=True)
        return 1

    tmp.replace(args.out)
    report = {
        "stage": "v1a_export", "prereg_token": PREREG_TOKEN,
        "out": str(args.out), "out_sha256": sha256_of(args.out),
        "n_rows": int(n), "n_fans": len(fans),
        "operating_point": {"lambda": float(args.lam), "rank": int(r),
                            "input": "dz", "target": target},
        "self_check": {
            "within_fan_top1_in_sample": round(top1, 4),
            "floor": selfcheck_floor,
            "chance": round(chance, 4),
            # ★ per-model, never a literal: read from THIS
            # model's fit report, or null with the reason in heldout_source
            "registered_heldout_top1": heldout["registered_heldout_top1"],
            "heldout_source": heldout["heldout_source"],
            "bins_block_max_code_gap": bins_gap,
            "deployment_identity_rel_max_gap": ident,
            "verdict": "PASS",
        },
        "rulers": {
            "norm_ref_WORN_wide_map": [float(x) for x in wide_norm_ref],
            "norm_ref_v1a_own_BANKED_NOT_WORN": [float(x) for x in norm_ref_own],
        },
        "sites": sites, "hidden": int(hidden_dim),
        "sources": meta["sources"],
    }
    if not registered_point:
        report["operating_point"]["registered_point"] = False
        report["operating_point"]["full_rank"] = bool(full_rank)
        report["operating_point"]["code_dim_served"] = int(r)
        report["operating_point"]["grid_key"] = (
            f"dz|{target}|lam{args.lam:g}|r{args.rank}")
        # ★ from THIS model's fit report, and only for `raw`:
        # the fit grid is the dz|raw sweep and cannot speak for sitenorm.
        report["self_check"]["this_cell_heldout_top1"] = (
            heldout["this_cell_heldout_top1"]
            if target == TARGET_DEFAULT else None)
        report["artifact_bytes"] = int(args.out.stat().st_size)
    if args.fan_source != FAN_SOURCE_LEVER:
        report["fan_source"] = str(args.fan_source)
    rpath = args.out.with_name(args.out.stem + "_export_report.json")
    rpath.write_text(json.dumps(report, indent=1))
    logger.info("DONE -> %s (+ %s)", args.out, rpath)
    print(json.dumps(report["self_check"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
