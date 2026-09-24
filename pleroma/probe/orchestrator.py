"""THE PROBE ORCHESTRATOR — the z-nr probe gauge as a live pick policy.

★ WHAT THIS IS. The free gauge FAILS as an INSTRUMENT (its z-nr metric misses
the registered AUC bar by .0016) and works as a PICK POLICY: 71% of the
ORACLE's picking skill against `distinct`'s 33%, and on `window` it picks the
oracle's exact top-4 while the static policies pick BELOW random. That
division is the standing one — fable = the instrument, the gauge = the
policy, the human chooses (`docs/FINDINGS.md §8`).

The probe gauge cannot be read off the fan, because it is PROBE-DERIVED: you
have to actually steer toward a candidate, generate, extract the reply's
signature, and measure where that reply LANDS relative to the fan. This module
is that measurement.

★ THE DEFINITION, TAKEN FROM THE SOURCE, NOT RE-INVENTED.
`pleroma.probe.freegauge.gauge_nr` / `cosine_dist` / `cell_delta` are
imported and called here — the same functions the calibration runs — so the
live policy cannot drift from the calibrated one by a paraphrase. The gauge
(``M2_z_nr`` in the calibration's verdict) is:

    nr(reply, t) = normalized rank (0 = nearest, /(k-1)) of future t among
                   the SAME-PASS future representations, by COSINE distance
                   in z (corpus-standardized signature) space

    cell(t)      = mean base nr(·, t) − mean steered nr(·, t)

which is fable's Δnr construction with geometry substituted for the judge.
HIGHER cell = steering toward t moved the reply toward t more. The policy
ranks candidates by cell, descending.

★ THE BASE TERM, STATED PRECISELY (the one real design choice).
The cell is base-minus-steered, so a base arm is required, and the base nr
for target t is a per-target correction: some futures sit centrally in the
fan and a random reply lands near them by geometry alone. Two modes:

  "fan" (DEFAULT, FREE) — leave-one-out over the fan itself. For target t the
      base rows are the k-1 OTHER futures of this same draw. These are real
      unsteered continuations of the very same prefix, already harvested by
      the draw, so this base costs nothing. The bias is known and bounded: a
      fan member j sits in the reference set at distance 0 from itself, so it
      occupies rank 0 and every OTHER target is pushed up by exactly one
      slot — a CONSTANT +1/(k-1) added to every base nr alike. A constant
      that is identical across targets cancels in the RANKING, which is all a
      pick policy consumes. ★ It does NOT cancel in the absolute value, so a
      "fan"-based cell is a ranking statistic and MUST NOT be reported as a
      Δnr; `absolute_comparable` is False on these and the caller says so.

  "fresh" (OPT-IN, COSTS n_base spans) — generate n_base genuinely unworn
      replies at the PROBE's own horizon and use them as the base arm for
      every target. Length-matched to the steered arm, no self-distance
      artifact, absolute-comparable. This is the calibration's own construction.

Everything here is PURE — no torch, no model, no HTTP — which is the house
pattern for anything that decides how the model gets pushed (resolve_draw_wear,
pick_auto, dose_scale_for, rehydrate_worn_vectors): the deciding logic must be
testable on a laptop. `pleroma.serve.routes.probe` supplies the generations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pleroma.probe.freegauge import cell_delta, cosine_dist, gauge_nr

BASE_MODES: tuple[str, ...] = ("fan", "fresh")
DEFAULT_BASE_MODE = "fan"
# ★ DEFAULT_REPS = 2, not 1, and the reason is a measurement rather than a
# taste. At reps=1 there is exactly one reading per candidate, so the probe
# CANNOT separate its own measurement noise from a real difference and
# `resolution_report` has to refuse. On at least one real v1a fan that
# refusal would have hidden something load-bearing: nearly all of the
# apparent between-candidate spread at reps=1 was sampling noise (see
# resolution_report's docstring for the reps=1/2/3/6 sweep). Two reps is the
# cheapest setting at which the probe reports whether its own answer means
# anything on THIS fan. Raising it further is the operator's call, priced by
# estimate_cost_s.
DEFAULT_REPS = 2
DEFAULT_N_BASE = 3

# ★ MEASURED ON THE PROBE'S OWN PATH: B200, llama-3.2-3b-instruct,
# a real 3-turn session prefix (~1065 prompt tokens), probe horizon 64, the
# serial harvest worker. k=8 reps=1 came in at generate 7.9 s / harvest 10.6 s
# over 8 spans. These are what `/info` and the UI quote, so an operator can
# price a probe BEFORE committing to it — the trade (43% of ORACLE for free vs
# 71% for seconds, `docs/FINDINGS.md §8`) is unreadable without the seconds.
#
# They are deliberately measured on the PROBE's spans, not the draw's: a probe
# reply is shorter than a future, so a draw-derived 1.57 s/span would overstate
# the bill. An over-estimate is a lie in the operator's favour and still a lie.
PER_SPAN_HARVEST_S = 1.33   # v3 replay + feature math, serial, one probe span
PER_GEN_S = 0.99            # one steered generation, horizon 64, reps batched


def contrastive_code(codes: Sequence[Sequence[float]], index: int) -> list[float]:
    """Candidate `index`'s CONTRASTIVE code: its own code minus the mean of
    the rest of the fan.

    This is the arm the calibration measured (`arm == "sweep_ctr"`), and the
    same construction `/wear_code`'s `code_kind: "differential"` wears — so the
    probe steers with exactly the object the calibration scored, not with the
    absolute code. mu_y cancels in the difference and is not added back; that
    is handled downstream by `LoomMap.lever_from_code(..., differential=True)`.
    """
    k = len(codes)
    if k < 2:
        raise ValueError("a contrastive code needs at least 2 candidates")
    if not 0 <= index < k:
        raise ValueError(f"index {index} outside a fan of {k}")
    rest = [j for j in range(k) if j != index]
    width = len(codes[index])
    for j in rest:
        if len(codes[j]) != width:
            raise ValueError(
                f"code width mismatch: candidate {index} has {width}, "
                f"candidate {j} has {len(codes[j])} — different maps?")
    return [float(codes[index][d]) - sum(float(codes[j][d]) for j in rest) / len(rest)
            for d in range(width)]


def base_nr_fan(future_z: Sequence[Sequence[float]], target: int) -> list[float]:
    """Leave-one-out base nrs for `target`: every OTHER future of this draw,
    ranked against the same fan. See the module docstring for the constant
    +1/(k-1) self-distance offset and why it cancels in a ranking."""
    k = len(future_z)
    if k < 3:
        raise ValueError(
            "base='fan' needs k>=3 (leave-one-out over 2 futures leaves one "
            "base row and no spread); draw a wider fan or use base='fresh'")
    return [gauge_nr(list(future_z[j]), [list(z) for z in future_z], target,
                     cosine_dist)
            for j in range(k) if j != target]


def base_nr_fresh(future_z: Sequence[Sequence[float]],
                  base_z: Sequence[Sequence[float]], target: int) -> list[float]:
    """Base nrs for `target` from genuinely unworn probe-horizon replies."""
    if not base_z:
        raise ValueError("base='fresh' needs at least one harvested base reply")
    return [gauge_nr(list(b), [list(z) for z in future_z], target, cosine_dist)
            for b in base_z]


@dataclass
class CandidateGauge:
    """One candidate's probe verdict."""

    index: int
    cell: float | None                 # base − steered; None = unscorable
    steered_nr: list[float] = field(default_factory=list)
    base_nr_mean: float | None = None
    steered_nr_mean: float | None = None
    n_reps_scored: int = 0
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "gauge": None if self.cell is None else round(self.cell, 4),
            "gauge_base_nr": (None if self.base_nr_mean is None
                              else round(self.base_nr_mean, 4)),
            "gauge_steered_nr": (None if self.steered_nr_mean is None
                                 else round(self.steered_nr_mean, 4)),
            "gauge_reps": self.n_reps_scored,
            "note": self.note,
        }


def score_candidates(
    future_z: Sequence[Sequence[float]],
    steered_z: Mapping[int, Sequence[Sequence[float]]],
    *,
    base_mode: str = DEFAULT_BASE_MODE,
    base_z: Sequence[Sequence[float]] | None = None,
) -> list[CandidateGauge]:
    """The probe-gauge cell for every candidate we have a steered reply for.

    `steered_z[i]` is the list of z-vectors of the replies generated while
    wearing candidate i's contrastive code (one per rep). A candidate with no
    usable reply gets cell=None and a note — listed, never hidden, exactly as
    a dead future is in /loom.
    """
    if base_mode not in BASE_MODES:
        raise ValueError(f"base must be one of {BASE_MODES}, got {base_mode!r}")
    if base_mode == "fresh" and not base_z:
        raise ValueError("base='fresh' requires base_z (harvested unworn replies)")
    k = len(future_z)
    if k < 2:
        raise ValueError("a fan of one is not a fan")

    out: list[CandidateGauge] = []
    for i in range(k):
        reps = list(steered_z.get(i) or [])
        if not reps:
            out.append(CandidateGauge(index=i, cell=None,
                                      note="no harvested probe reply"))
            continue
        try:
            base = (base_nr_fan(future_z, i) if base_mode == "fan"
                    else base_nr_fresh(future_z, base_z or [], i))
            steered = [gauge_nr(list(z), [list(f) for f in future_z], i,
                                cosine_dist) for z in reps]
            out.append(CandidateGauge(
                index=i, cell=cell_delta(base, steered), steered_nr=steered,
                base_nr_mean=sum(base) / len(base),
                steered_nr_mean=sum(steered) / len(steered),
                n_reps_scored=len(steered)))
        except (ValueError, ZeroDivisionError) as exc:
            out.append(CandidateGauge(
                index=i, cell=None,
                note=f"{type(exc).__name__}: {exc}"))
    return out


def rank_by_gauge(gauges: Sequence[CandidateGauge]) -> list[int]:
    """Candidate indices best-first. Higher cell wins; ties and unscorables
    break by index, deterministically. Unscorables always sort last — a
    candidate the probe could not measure is never picked over one it could."""
    scored = [g for g in gauges if g.cell is not None]
    dead = [g for g in gauges if g.cell is None]
    ordered = sorted(scored, key=lambda g: (-float(g.cell or 0.0), g.index))
    return [g.index for g in ordered] + [g.index for g in sorted(
        dead, key=lambda g: g.index)]


def pick_gauge(gauges: Sequence[CandidateGauge]) -> int:
    """The single candidate the probe gauge picks. Raises when nothing scored —
    the caller must never silently fall back to a different policy."""
    order = rank_by_gauge(gauges)
    scored = {g.index for g in gauges if g.cell is not None}
    for i in order:
        if i in scored:
            return i
    raise ValueError(
        "the probe scored no candidate — refusing to pick. (Every probe reply "
        "failed to harvest; /loom again, or pick with a free policy.)")


def resolution_report(gauges: Sequence[CandidateGauge]) -> dict[str, Any]:
    """★ CAN THIS PROBE TELL THESE CANDIDATES APART AT ALL?

    Needs reps >= 2, and then it answers from the probe's OWN readings rather
    than from anybody's prior. Variance decomposition:

        within  = pooled SD of one candidate's steered nr across its reps
                  = the per-rep measurement noise
        between = SD of the candidates' cells
                  = signal + noise/reps

    so  signal_sd^2 ~= between^2 - within^2/reps, and the reps needed for the
    noise on a candidate's MEAN to fall below the true between-candidate
    spread is (within / signal_sd)^2.

    ★ WHY THIS SHIPS RATHER THAN BEING A FOOTNOTE IN A DOC: because it is a
    property of the FAN, not a constant of the instrument, and the spread
    between fans is enormous. Two real v1a fans on this server, both k=8:

      fan A  within .299 / signal .200 -> RESOLVED at reps=2
      fan B  within ~.35 / signal ~.071 -> needs ~24 reps (fitted across a
             reps = 1/2/3/6 sweep: cell SD .358/.280/.212/.160, which is
             almost exactly 1/sqrt(reps) — i.e. nearly all of the apparent
             between-candidate spread at reps=1 was sampling noise)

    At ~1.33 s/span, 24 reps at k=8 is ~256 s of harvest for ONE turn. So on
    some fans the probe answers cheaply and on others it cannot answer at any
    price an interactive loom can pay — and NOTHING VISIBLE ON THE DRAW
    DISTINGUISHES THEM IN ADVANCE. That is `docs/FINDINGS.md §11`
    arriving from a new direction (fan spread predicts steerability, ICC .82,
    and a flat fan has nothing to pick between); here it becomes a per-turn
    readout instead of a corpus statistic. A probe that cannot resolve its own
    candidates has to say so in its own receipt, on the fan it actually ran on.
    """
    # ★ `cells` MUST come from `scored`, not from every scored-at-all
    # candidate. A candidate that survived only ONE rep carries the full
    # per-rep noise in its cell, but the correction below subtracts only
    # within^2/reps with reps >= 2 — so mixing them under-subtracts the noise,
    # over-states the signal, and reports `resolved: true` on pure noise. That
    # is reachable in production, because a probe span that fails to harvest
    # leaves its candidate short a rep. A simulation over 2000 pure-noise fans
    # puts the false-"resolved" rate at ~62-75% for mixed rep counts against
    # ~13% for uniform ones.
    scored = [g for g in gauges if g.cell is not None and len(g.steered_nr) >= 2]
    cells = [float(g.cell) for g in scored]
    if len(scored) < 2 or len(cells) < 2:
        return {
            "resolved": None,
            "why": "needs reps>=2 on at least 2 candidates; with one rep each "
                   "there is no way to separate measurement noise from real "
                   "between-candidate difference.",
        }
    reps = min(len(g.steered_nr) for g in scored)
    within_var = 0.0
    dof = 0
    for g in scored:
        m = sum(g.steered_nr) / len(g.steered_nr)
        within_var += sum((x - m) ** 2 for x in g.steered_nr)
        dof += len(g.steered_nr) - 1
    within_sd = (within_var / dof) ** 0.5 if dof else 0.0
    mc = sum(cells) / len(cells)
    between_sd = (sum((c - mc) ** 2 for c in cells) / (len(cells) - 1)) ** 0.5
    # ★ A MARGIN, because `between^2 > within^2/reps` is itself a noisy
    # comparison and comes out positive by chance far too often. The sampling
    # variance of a variance estimate on n candidates is ~2/(n-1) of its own
    # square, so the noise floor is inflated by 1 + Z*sqrt(2/(n-1)) before the
    # comparison. Z = 3.0 was calibrated by simulation (4000 pure-noise fans
    # per cell, k=8, sigma=.35) rather than chosen:
    #
    #   FALSE "resolved" on PURE NOISE : reps=2 10.2%  reps=4 3.8%  reps=6 3.0%
    #   POWER, signal == noise (.35)   : reps=2 57.3%              reps=6 91.0%
    #
    # ★ Note the residual ~10% at reps=2 — the DEFAULT. Two readings simply
    # cannot do better, and that is itself an argument for more reps rather
    # than something to hide: `resolved: true` at reps=2 is weak evidence, and
    # `resolved: false` at reps=2 is strong.
    noise_floor = (within_sd ** 2) / reps
    margin = 1.0 + 3.0 * (2.0 / max(1, len(cells) - 1)) ** 0.5
    signal_var = (between_sd ** 2 - noise_floor
                  if between_sd ** 2 > noise_floor * margin else 0.0)
    signal_sd = signal_var ** 0.5 if signal_var > 0 else 0.0
    # at least 1: "0 reps" is not an answer to "how many readings do I need"
    needed = (max(1, int(round((within_sd / signal_sd) ** 2)))
              if signal_sd > 1e-9 else None)
    resolved = bool(signal_sd > 0 and needed is not None and needed <= reps)
    # A fan whose candidates read IDENTICALLY is a different thing from one
    # whose spread is drowned in noise, and saying "noise" for both would be
    # wrong about the flat case — a fan that did not fork, the case worth naming.
    flat = between_sd <= 1e-12 and within_sd <= 1e-12
    return {
        "resolved": resolved,
        "reps": reps,
        # how many candidates this verdict is actually based on — candidates
        # with fewer than 2 surviving reps are excluded from BOTH terms, so a
        # fan that lost rows to dead harvests says so instead of quietly
        # reporting a verdict computed on half of it.
        "n_candidates_used": len(scored),
        "n_candidates_excluded_for_reps": sum(
            1 for g in gauges
            if g.cell is not None and len(g.steered_nr) < 2),
        "within_candidate_sd_nr": round(within_sd, 4),
        "between_candidate_sd_cell": round(between_sd, 4),
        "estimated_signal_sd": round(signal_sd, 4),
        "noise_share_of_cell_variance_at_this_reps": (
            round(min(1.0, (within_sd ** 2 / reps) / between_sd ** 2), 3)
            if between_sd > 0 else None),
        "reps_needed_to_resolve": needed,
        "why": (
            "the ordering is resolved at this rep count"
            if resolved else
            ("every candidate scored IDENTICALLY — the gauge sees no "
             "difference at all between these futures. Not a noise problem: "
             "this fan did not fork (docs/FINDINGS.md §11). No rep count helps."
             if flat else
             "the between-candidate spread is entirely explained by "
             "measurement noise — these candidates are not distinguishable "
             "by the gauge at ANY rep count this loom can afford"
             if signal_sd <= 0 else
             f"the ordering is NOT resolved at reps={reps}: this fan needs "
             f"about {needed} reps for the noise on a candidate's mean to "
             "fall under the real between-candidate spread. Treat the "
             "ranking as provisional.")),
    }


#: Per-span degradation when harvest is sharded across a warm pool rather than
#: run serially: measured 1.67 s/span across 4 workers vs 1.57 solo, i.e. ~6%
#: Applied only when width > 1.
POOL_PER_SPAN_DEGRADATION = 1.06


def estimate_cost_s(k: int, reps: int, *, base_mode: str = DEFAULT_BASE_MODE,
                    n_base: int = DEFAULT_N_BASE,
                    per_gen_s: float = PER_GEN_S,
                    per_span_s: float = PER_SPAN_HARVEST_S,
                    harvest_width: int = 1) -> dict[str, Any]:
    """What a probe at these settings will cost, in seconds, on THIS server.

    Published by /info and shown in the UI BEFORE the operator commits, which
    is the whole point: the free policy gets 43% of ORACLE, so the probe's
    extra 27 points have to be bought knowingly. Generation is one batched
    call per candidate (reps ride in the batch, so reps are nearly free on the
    generate side and cost a full span each on the harvest side — which is
    where 88-95% of a loom turn lives; see ``pleroma.harvest.pool``).
    """
    if k < 2 or reps < 1:
        raise ValueError("k must be >= 2 and reps >= 1")
    if base_mode not in BASE_MODES:
        raise ValueError(f"base must be one of {BASE_MODES}")
    width = max(1, int(harvest_width))
    n_extra_base = n_base if base_mode == "fresh" else 0
    gen_s = k * per_gen_s + (per_gen_s if n_extra_base else 0.0)
    # ★ The harvest term is SHARDED when a pool is armed. Quoting a serial
    # worker unconditionally over-quotes a pooled probe by ~2x (k=4 reps=2
    # quotes 14.6 s serial and takes 7.9 s on a 2-worker pool). Over-quoting is
    # the safe direction but it is still wrong, and the UI shows this number to
    # an operator deciding whether to spend the time, so it has to describe the
    # server that is actually running.
    degradation = POOL_PER_SPAN_DEGRADATION if width > 1 else 1.0
    harvest_s = (k * reps + n_extra_base) * per_span_s * degradation / width
    # Round the PARTS, then sum the rounded parts — a published breakdown that
    # does not add up is a receipt nobody can check by hand, and this one is
    # shown to an operator deciding whether to spend the time.
    gen_r, harvest_r = round(gen_s, 1), round(harvest_s, 1)
    return {
        "k": int(k), "reps": int(reps), "base": base_mode,
        # published so a client can reproduce this arithmetic exactly rather
        # than assuming the server's current default base mode
        "n_base": int(n_extra_base),
        "n_spans_harvested": int(k * reps + n_extra_base),
        "generate_s": gen_r,
        "harvest_s": harvest_r,
        "total_s": round(gen_r + harvest_r, 1),
        "harvest_width": int(width),
        "basis": {
            "per_gen_s": per_gen_s, "per_span_harvest_s": per_span_s,
            "harvest_width": int(width),
            "pool_per_span_degradation": (degradation if width > 1 else None),
            "measured": "B200, llama-3.2-3b-instruct, 3-turn "
                        "session prefix (~1.1k prompt tokens), probe horizon "
                        "64, ONE serial harvest worker (k=8 reps=1 observed at "
                        "7.9 s generate / 10.6 s harvest). The per-span figure "
                        "is that measurement; the harvest term is divided by "
                        "harvest_width and multiplied by the pool's ~6% "
                        "per-span degradation when a pool is armed.",
            "caveat": "the per-span basis was measured on llama-3.2-3b; a "
                      "larger model harvests more slowly per span, so on an 8B "
                      "or 70B server read this as an ORDER OF MAGNITUDE, not a "
                      "prediction.",
        },
        "note": ("ON TOP of the /loom draw itself. The harvest term dominates "
                 "— it is 88-95% of any loom turn — and is sharded across the "
                 f"{width} harvest worker(s) this server has armed"
                 + (" (serial: no pool)." if width == 1 else
                    ", at ~6% per-span degradation.")),
    }
