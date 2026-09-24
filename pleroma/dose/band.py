"""Dose-band calibration — attach, derive, and resolve the alpha->zone map.

**The problem this closes.** A dose band (inert / subliminal / threshold /
audible / overdriven, in alpha) is measured on ONE specific map.
`LoomMap.lever_of` (`pleroma.map.loom_map`) norm-matches every predicted
lever per site to that map's own `norm_ref`, so the vector actually injected
at site `s` has norm `alpha * norm_ref[s]` — a fixed unit WITHIN a map, but
not ACROSS maps, because `norm_ref` differs map to map (`docs/FINDINGS.md §2`;
read it before touching `rescale()` below). Point an alpha->zone table
measured on one map at a *different* map and it quietly lies: the w5 rank-32
map's `norm_ref` runs ~2.1x the old map's, and a ladder measured against it
confirmed audibility tracks ABSOLUTE perturbation magnitude, not map-relative
alpha — its thresholds landed at roughly HALF the old map's alphas, not the
same ones. A dose gauge wrong by 2x is worse than no gauge, because it is
trusted.

**The fix.** A dose band is not a UI constant. It is data that can be
attached to a map (this module's CLI), discovered by the server at startup,
served over `GET /info` as an explicit, honestly-tiered object, and rendered
by the UI with the tier always visible:

  * **measured** — attached from an actual judged dose ladder. Authoritative.
  * **derived**  — provisional: rescaled from ANOTHER map's measured band by
    the ratio of the two maps' `norm_ref` (`DoseBand.rescale`). Marked
    provisional everywhere it is surfaced.
  * **none**      — no band available for this map. The UI must say
    "uncalibrated" and must NOT fall back to some other map's numbers.

**Where a measured band lives, and why.** A sidecar JSON file next to the
map: `<map>.doseband.json` by convention (`default_sidecar_path`), or any
path via `--dose-band`. NOT inside the `.npz` — these maps are ~144MB and
rewriting one to attach five numbers is exactly the kind of operation that
should not exist; NOT a server flag alone, because a flag does not survive
choosing to load a different map next time and does not `ls` next to the
thing it describes. A small JSON file: survives a server restart (read
fresh from disk at startup, same as the map and discriminants), diffs
cleanly in git, is hand-editable in a pinch, and can be attached to a map
that already exists on disk without touching the map at all — exactly the
"existing maps and existing ladders" case this feature exists for.

**How a band transfers.** `DoseBand.rescale()` implements
`alpha_new = alpha_old / mean_s(norm_ref_new[s] / norm_ref_old[s])` — the
MEAN of the per-site ratio `target/reference`, inverted, as a single scalar
multiplier on every alpha boundary. This is a simplification — `norm_ref` is
a per-site vector (4 sites here) and the per-site ratios are not identical —
but it is not a hand-wave: for the one pair actually measured so far (old
map vs w5-r32), this exact method — applied to the old map's band alone, no
access to the w5-r32 ladder — predicted a subliminal ceiling of 0.236
(measured: <=0.25) and, via `overdriven_ceiling` below, a damage line of
0.708 (a live `/wear` at alpha=0.75 on w5-r32 produced visible word-salad —
"you may not have the muscle memory of a professional chef, but your body is
still full of muscle fibers that..." — consistent with 0.75 already past
0.708). Agreement within one ladder rung. `rescale()`
still computes the per-site spread and logs a WARNING (does not refuse) when
it exceeds 15%, because a future map pair may not agree this well and the
single-scalar approximation would be weaker for it. The result is always
tier="derived" and is never returned as "measured" no matter how good the
fit looks.

**And a transfer ACROSS MODELS carries one more assumption than that
validation can see.** The old map and w5-r32 are both Llama-3.2-3B, so
"audibility tracks the absolute injected norm" and "audibility tracks the
norm as a FRACTION of the residual stream" make identical predictions
there — the w5-r32 ladder cannot choose between them and therefore
licenses neither for a 3B -> 8B transfer. Pass
`rescale(..., residual_scale=...)` with a `ResidualScale` (both models'
per-site median ||h||, measured off banked mean_hiddens for free) and the
returned band's `derivation.cross_model` states how far apart the two
hypotheses put it, which side the served numbers sit on, and the
residual-scale ratio behind both. **It changes no served boundary.** Measured
for 3B -> 8B: the hypotheses disagree by 1.165x and the served absolute
numbers are the conservative side — an error bar, not the 2x blowout this
module exists to prevent (the per-site norms are listed above `ResidualScale`).

**A derived band can also be FROZEN to a sidecar** (`python -m pleroma.dose.band derive`)
instead of being recomputed from its reference at every server startup. Same
`resolve_dose_band`, same numbers, but the band an operator reads off the
gauge then exists in git, diffs, and can carry provenance. `derive` refuses
to overwrite a measured sidecar with a rescale, and what it writes is
`tier: "derived"` — freezing a band does not promote it.

**The overdriven ceiling is special-cased, not inferred from ladder
accuracy.** A psychometric curve keeps climbing as alpha rises into damage —
a broken, word-salad reply is if anything MORE reliably distinguishable from
the control than a clean but subtle one, so accuracy alone cannot tell
"audible-and-legible" from "audible-because-it-broke" (`docs/FINDINGS.md §7`;
the w5-r32 ladder's own judge free-text at alpha 0.75-1.0 says "garbled
details", "word-salad sentence", "cuts off mid-sentence" — accuracy .988/
1.000 there is substantially damage detection). So the overdriven boundary
is always DERIVED from the reference map's own known damage point — the old
map's alpha=1.5, where its replies turn to word-salad — rescaled by
`overdriven_ceiling()`, for BOTH tiers: a "derived" band gets it for free
(rescaling the reference's whole zone table rescales its top edge along with
everything else); a "measured" band being attached from a real ladder has it
applied explicitly by `apply_overdriven_ceiling`, which caps and relabels
any zone the ladder itself pushed past the ceiling, and flags the run as
having sampled into damage rather than dose.

Usage — attach a MEASURED band once a ladder has been read into zones (a
manual judgement call: a reader separates register content, which is the
dose working, from coherence damage, which is the dose breaking the model;
this module does not automate that read)::

    python -m pleroma.dose.band attach \\
        --map loom_map_w5_r32.npz \\
        --zones-json w5_r32_zones.json \\
        --source "w5-r32 dose ladder, 1008 pairs, sonnet-5+opus-5" \\
        --ladder-json judge/report.json

or, for the classic 5-zone shape (inert/subliminal/threshold/audible/
overdriven), a one-line shorthand instead of --zones-json::

    python -m pleroma.dose.band attach \\
        --map loom_map_w5_r32.npz \\
        --boundaries 0.06,0.25,0.41,0.56 --alpha-max 1.0 \\
        --source "..."

Inspect what a server would resolve for a map (exercises the exact same
`resolve_dose_band` the server calls at startup)::

    python -m pleroma.dose.band show --map <path> \\
        [--dose-band <path>] [--dose-band-reference <path>]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.map.identity import (
    fingerprint_v2, legacy_fingerprint, same_map, weights_digest,
)

logger = logging.getLogger("dose_band")

CURRENT_SCHEMA = 1
TIERS = ("measured", "derived")
# how much per-site norm_ref ratio disagreement rescale() tolerates before
# it warns that the mean-scalar approximation is getting rough
_SITE_RATIO_SPREAD_WARN = 0.15

# A dose band is a per-MAP average over an unknown mix of future lever
# quality, not a per-lever guarantee. The w5-r32 ladder judged three levers
# (held-out cos-to-true .208/.458/.573) and got three visibly different
# curves — the best-predicted lever (cos .573) read .746 accuracy at
# alpha=0.25, already well clear of chance, while the worst (cos .208)
# needed 0.5 to get there. A band fit on one lever, or pooled
# across a few, is silent about where the CURRENTLY WORN code's own
# prediction quality sits in that range. Rather than trust every attach
# call to remember to say so, this string is surfaced on EVERY band's
# `/info` payload (see `to_info_json`) — cheap, and it means the UI never
# has to render a band as more exact than it is.
LEVER_QUALITY_CAVEAT = (
    "this band is a per-map average over levers of varying held-out "
    "prediction quality (cos-to-true), not a guarantee for the specific "
    "code currently worn — a well-predicted future can read audible well "
    "below this band's threshold line, and a poorly-predicted one well above it"
)


# ── map identity — fingerprinting and a discriminants-free npz read ─────────


def map_fingerprint(sites: Sequence[int], norm_ref: np.ndarray, mu_y: np.ndarray,
                     meta: Mapping[str, Any]) -> str:
    """A short, stable id for the fitted map a lever (or a dose band) was
    made with. It lives here so this module can fingerprint a map on its
    own: `pleroma.serve.app` imports `DoseBand`/`resolve_dose_band` from
    here, so importing the server back would be circular.

    Rehydration recomputes a lever from a banked signature; run through a
    DIFFERENT map that produces a different lever of the same shape, and the
    session would silently wear something nobody chose. The fingerprint turns
    that into a refusal — and, here, lets a dose band refuse to be served as
    MEASURED for a map it was not measured on.

    This is fingerprint v1 (LEGACY: it does not hash the map's weights). A
    loaded map's id is `LoomMap.fingerprint` / `load_map_identity` (v2,
    pleroma.map.identity), which embeds this one so records stamped with it
    still match.
    """
    return legacy_fingerprint(sites, norm_ref, mu_y, meta)


def load_map_identity(path: Path) -> tuple[str, list[int], np.ndarray, dict[str, Any]]:
    """Read just the fields `map_fingerprint()` needs straight out of a
    `loom_map_*.npz` — sites, norm_ref, mu_y, meta — WITHOUT the paired
    discriminants file a full `LoomMap` load requires. `LoomMap.__init__`
    needs the discriminants only to build `full_mean`/`full_scale` and check
    the discriminants sha, neither of which fingerprinting or dose-band
    derivation touches. This is what lets the dose-band CLI attach a band to
    a map on a machine that has the (smaller) map file but not the
    (much larger) discriminants bundle.

    Returns (fingerprint, sites, norm_ref, meta).
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        missing = [k for k in ("mu_y", "sites", "norm_ref", "meta") if k not in npz.files]
        if missing:
            raise KeyError(f"{path} is missing {missing} (has {npz.files}) — not a "
                            "loom_map_*.npz?")
        mu_y = np.asarray(npz["mu_y"], dtype=np.float64)
        sites = [int(x) for x in npz["sites"]]
        norm_ref = np.asarray(npz["norm_ref"], dtype=np.float64)
        meta = json.loads(str(npz["meta"]))
        try:
            wd = weights_digest({k: npz[k] for k in npz.files})
        except KeyError as exc:
            raise KeyError(f"{path}: {exc} — not a loom_map_*.npz?") from exc
    fp = fingerprint_v2(sites, norm_ref, mu_y, meta, wd)
    return fp, sites, norm_ref, meta


# ── tiny JSON validation helpers ─────────────────────────────────────────────
#
# Deliberately local: importing look-alike helpers from `pleroma.serve`
# would create the same import cycle `map_fingerprint` above is dodging.
# `pleroma.dose.policy` imports `_obj`/`_floats` from here. Small, stable
# functions — the duplication costs less than the cycle would.


def _obj(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _opt_str(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string or null")
    return value


def _num(value: Any, what: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a number ({exc})") from exc


def _floats(value: Any, what: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{what} must be a list of numbers")
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a list of numbers ({exc})") from exc


# ── the cross-model question: is ||delta|| even the right ruler? ─────────────
#
# `rescale()` transfers a judged band by
# the ratio of two maps' `norm_ref`, i.e. it assumes AUDIBILITY TRACKS THE
# ABSOLUTE INJECTED NORM. Between two maps of the SAME model that is measured:
# the old map -> w5-r32 transfer (ratio 2.118) predicted a subliminal ceiling
# of 0.236 against a measured <=0.25 and a damage line of 0.708 against a live
# word-salad at 0.75.
#
# ★ But that validation CANNOT see the axis an 8B transfer needs. Both of those
# maps are Llama-3.2-3B: same hidden width, same residual-stream scale, so the
# absolute hypothesis and the RELATIVE one ("audibility tracks ||delta|| /
# ||h||, the fraction of the residual stream the injection displaces") make
# IDENTICAL predictions there and the ladder could not choose between them.
# Across model sizes they come apart, because the residual stream does not.
#
# Measured from banked mean_hiddens (no GPU, no judges): per-site median ||h|| is
# [3.35, 6.13, 8.93, 13.54] for the 3B at sites [8,15,19,22] and
# [3.32, 7.33, 10.52, 18.02] for the 8B at sites [9,17,21,25] — a mean ratio of
# 1.1745. So the two hypotheses disagree about the 8B's band by a factor of
# ~1.165, NOT by the 2x that this module exists to prevent, and the absolute
# rescale is the CONSERVATIVE of the two (it puts every boundary lower, so it
# calls a given alpha louder than the relative hypothesis would).
#
# ★ This function does NOT change which band is served. Switching the served
# numbers to an unvalidated alternative would be exactly the move this module
# forbids. It quantifies the assumption so the band can carry its own error
# bar, and it is the cheapest honest thing available: it turns "an unvalidated
# cross-model stretch" into "a stretch whose competing hypothesis is bounded at
# 1.165x, on the safe side".


@dataclass(frozen=True)
class ResidualScale:
    """Per-site residual-stream norms for the reference and target models —
    the denominator the absolute-magnitude rescale leaves out.

    `h_norm` is a per-site central ||h|| (the median over a banked corpus).
    Both arms must have one entry per steering site, in site order, measured
    the same way — the RATIO is the load-bearing quantity, so a consistent
    reduction matters more than which reduction it is.
    """

    reference_model: str
    reference_h_norm: tuple[float, ...]
    target_model: str
    target_h_norm: tuple[float, ...]
    source: str
    reference_sites: tuple[int, ...] | None = None
    target_sites: tuple[int, ...] | None = None
    measured_at: str | None = None
    schema: int = CURRENT_SCHEMA

    def __post_init__(self) -> None:
        if len(self.reference_h_norm) != len(self.target_h_norm):
            raise ValueError(
                f"residual scale: reference has {len(self.reference_h_norm)} sites, "
                f"target has {len(self.target_h_norm)} — a cross-model bound only "
                "makes sense site-for-site")
        if not self.reference_h_norm:
            raise ValueError("residual scale needs at least one site")
        for name, vals in (("reference_h_norm", self.reference_h_norm),
                           ("target_h_norm", self.target_h_norm)):
            if any(not np.isfinite(v) or v <= 0 for v in vals):
                raise ValueError(f"residual scale {name} must be finite and strictly "
                                  f"positive, got {list(vals)}")

    @property
    def n_sites(self) -> int:
        return len(self.reference_h_norm)

    def h_ratios(self) -> list[float]:
        """Per-site target/reference residual norm."""
        return [t / r for r, t in zip(self.reference_h_norm, self.target_h_norm)]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "source": self.source,
            "measured_at": self.measured_at,
            "reference": {"model": self.reference_model,
                          "sites": (None if self.reference_sites is None
                                    else list(self.reference_sites)),
                          "h_norm": [float(x) for x in self.reference_h_norm]},
            "target": {"model": self.target_model,
                       "sites": (None if self.target_sites is None
                                 else list(self.target_sites)),
                       "h_norm": [float(x) for x in self.target_h_norm]},
        }

    @classmethod
    def from_json(cls, value: Any) -> ResidualScale:
        blob = _obj(value, "residual scale")
        schema = blob.get("schema", 1)
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise ValueError(f"residual scale 'schema' must be an integer, got {schema!r}")
        if schema > CURRENT_SCHEMA:
            raise ValueError(
                f"residual scale schema {schema} is newer than this reader supports "
                f"(schema {CURRENT_SCHEMA}) — upgrade pleroma")
        ref = _obj(blob.get("reference"), "residual scale 'reference'")
        tgt = _obj(blob.get("target"), "residual scale 'target'")

        def _sites(raw: Any, what: str) -> tuple[int, ...] | None:
            if raw is None:
                return None
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"{what} must be a list of integers or null")
            try:
                return tuple(int(x) for x in raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{what} must be a list of integers ({exc})") from exc

        return cls(
            reference_model=_str(ref.get("model"), "residual scale reference.model"),
            reference_h_norm=tuple(_floats(ref.get("h_norm"),
                                            "residual scale reference.h_norm")),
            target_model=_str(tgt.get("model"), "residual scale target.model"),
            target_h_norm=tuple(_floats(tgt.get("h_norm"),
                                         "residual scale target.h_norm")),
            source=_str(blob.get("source"), "residual scale 'source'"),
            reference_sites=_sites(ref.get("sites"), "residual scale reference.sites"),
            target_sites=_sites(tgt.get("sites"), "residual scale target.sites"),
            measured_at=_opt_str(blob.get("measured_at"), "residual scale 'measured_at'"),
            schema=schema,
        )


def load_residual_scale_file(path: Path) -> ResidualScale:
    """FAILS LOUDLY, same doctrine as `load_dose_band_file`."""
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read residual scale file {path}: {exc}") from exc
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return ResidualScale.from_json(blob)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed residual scale — {exc}") from exc


def _mean_ratio(target: Sequence[float], reference: Sequence[float]) -> tuple[float, list[float]]:
    """`rescale()`'s own statistic: the MEAN of the per-site target/reference
    ratio (not the ratio of the two means). Factored out so the cross-model
    bound below is computed by the identical arithmetic the band uses."""
    ratios = [float(t) / float(r) for r, t in zip(reference, target)]
    mean = sum(ratios) / len(ratios)
    if mean <= 0:
        raise ValueError("mean per-site ratio is non-positive")
    return mean, ratios


def cross_model_bound(
    reference_norm_ref: Sequence[float], target_norm_ref: Sequence[float],
    residual: ResidualScale,
) -> dict[str, Any]:
    """How far apart the absolute and relative hypotheses put this band.

    * ``absolute`` — what `rescale()` serves: equal audibility at equal
      ``alpha * norm_ref[s]``.
    * ``relative`` — equal audibility at equal ``alpha * norm_ref[s] /
      ||h[s]||``, i.e. at an equal FRACTION of the residual stream. Obtained by
      feeding `rescale()`'s statistic a target ruler pre-divided by the
      per-site residual-norm ratio.
    * ``disagreement`` — relative_scale / absolute_scale. 1.0 means the two
      hypotheses are the same transfer and the same-model validation covers
      this pair; far from 1.0 means it does not.

    ``conservative`` names whichever hypothesis puts the boundaries LOWER,
    because that is the one that calls a given alpha louder and therefore errs
    toward under-dosing rather than toward silent damage.
    """
    if not (len(reference_norm_ref) == len(target_norm_ref) == residual.n_sites):
        raise ValueError(
            f"cross-model bound needs one entry per site everywhere: "
            f"reference norm_ref {len(reference_norm_ref)}, target norm_ref "
            f"{len(target_norm_ref)}, residual scale {residual.n_sites}")
    abs_mean, abs_ratios = _mean_ratio(target_norm_ref, reference_norm_ref)
    h_ratios = residual.h_ratios()
    rel_target = [t / h for t, h in zip(target_norm_ref, h_ratios)]
    rel_mean, rel_ratios = _mean_ratio(rel_target, reference_norm_ref)
    abs_scale, rel_scale = 1.0 / abs_mean, 1.0 / rel_mean
    h_mean, _ = _mean_ratio(residual.target_h_norm, residual.reference_h_norm)
    disagreement = rel_scale / abs_scale
    return {
        "hypothesis_served": "absolute",
        "absolute": {
            "claim": "equal audibility at equal alpha * norm_ref[s]",
            "mean_ratio": round(abs_mean, 6), "scale": round(abs_scale, 6),
            "per_site_ratio": [round(r, 6) for r in abs_ratios],
        },
        "relative": {
            "claim": "equal audibility at equal alpha * norm_ref[s] / ||h[s]||",
            "mean_ratio": round(rel_mean, 6), "scale": round(rel_scale, 6),
            "per_site_ratio": [round(r, 6) for r in rel_ratios],
        },
        "residual_scale": {
            "mean_ratio_target_over_reference": round(h_mean, 6),
            "per_site_ratio": [round(r, 6) for r in h_ratios],
            "reference_model": residual.reference_model,
            "target_model": residual.target_model,
            "source": residual.source,
            "measured_at": residual.measured_at,
        },
        "disagreement": round(disagreement, 6),
        "conservative": "absolute" if abs_scale <= rel_scale else "relative",
        "served_is_conservative": abs_scale <= rel_scale,
        "note": (
            "the served band uses the ABSOLUTE hypothesis unchanged. This block "
            "is an error bar, not a correction: switching the served numbers to "
            "an unvalidated alternative is the failure this module exists to "
            "prevent. Between two maps of the SAME model the two hypotheses "
            "coincide, which is why the 2026-09-18 old-map -> w5-r32 validation "
            "says nothing about this factor."
        ),
    }


# ── DoseZone: one labelled alpha range ───────────────────────────────────────


@dataclass(frozen=True)
class DoseZone:
    """One labelled span of alpha. `hi_inclusive` matters at exactly one
    boundary in the classic 5-zone shape (subliminal's top, closed
    — see `five_zone_band`); every other edge belongs to the zone above it.
    Mirrors the zone rows of the UI's `zoneFor()`
    (`pleroma/serve/static/legacy_ui.html`) 1:1 on purpose, so the UI can render
    whatever `/info.dose_band.zones` sends without a local lookup table."""

    lo: float
    hi: float
    hi_inclusive: bool
    key: str
    label: str
    hint: str

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"zone {self.key!r} has hi <= lo ({self.hi} <= {self.lo})")

    def contains(self, alpha: float) -> bool:
        a = float(alpha)
        if a < self.lo:
            return False
        return a <= self.hi if self.hi_inclusive else a < self.hi

    def to_json(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "hi_inclusive": self.hi_inclusive,
                "key": self.key, "label": self.label, "hint": self.hint}

    @classmethod
    def from_json(cls, value: Any) -> DoseZone:
        blob = _obj(value, "dose zone")
        hi_inclusive = blob.get("hi_inclusive", False)
        if not isinstance(hi_inclusive, bool):
            raise ValueError("zone.hi_inclusive must be a boolean")
        hint = blob.get("hint", "")
        if not isinstance(hint, str):
            raise ValueError("zone.hint must be a string")
        return cls(
            lo=_num(blob.get("lo"), "zone.lo"), hi=_num(blob.get("hi"), "zone.hi"),
            hi_inclusive=hi_inclusive,
            key=_str(blob.get("key"), "zone.key"), label=_str(blob.get("label"), "zone.label"),
            hint=hint,
        )


ZONE_TEMPLATE: tuple[tuple[str, str, str], ...] = (
    ("inert", "inert", "below the floor — no measurable effect"),
    ("subliminal", "subliminal",
     "mechanically active, conversationally invisible"),
    ("threshold", "threshold", "the first detectable tilt"),
    ("audible", "audible", "the disposition's register is plainly present"),
    ("overdriven", "overdriven", "OVERDRIVEN — risk of word-salad damage"),
)


def five_zone_band(boundaries: Sequence[float], alpha_max: float) -> list[DoseZone]:
    """Build the classic 5-zone band — inert / subliminal / threshold /
    audible / overdriven — the shape the old map's measured band has, from 4
    numeric boundaries plus the top of the range.

    `boundaries` = (inert_hi, subliminal_hi, threshold_hi, audible_hi);
    `alpha_max` is overdriven's hi. subliminal's hi is CLOSED — matches the
    old table's `[0.125, 0.5]` convention (the top of the subliminal band is
    still safely inside it, not the first point of threshold) — every other
    zone's hi is open, so the zone above claims the exact boundary value.
    """
    if len(boundaries) != 4:
        raise ValueError(f"five_zone_band needs exactly 4 boundaries, got {len(boundaries)}")
    b0, b1, b2, b3 = (float(x) for x in boundaries)
    amax = float(alpha_max)
    if not (0.0 < b0 < b1 < b2 < b3 < amax):
        raise ValueError(
            "five_zone_band boundaries must be strictly increasing and below alpha_max: "
            f"0 < {b0} < {b1} < {b2} < {b3} < {amax} does not hold"
        )
    edges = [(0.0, b0, False), (b0, b1, True), (b1, b2, False), (b2, b3, False), (b3, amax, True)]
    return [
        DoseZone(lo=lo, hi=hi, hi_inclusive=inc, key=key, label=label, hint=hint)
        for (lo, hi, inc), (key, label, hint) in zip(edges, ZONE_TEMPLATE)
    ]


# ── DoseBand: a whole map's alpha->zone table, tiered ────────────────────────


@dataclass(frozen=True)
class DoseBand:
    """A complete alpha->zone table for one map, tagged with how it got
    there. `tier` is "measured" (attached from a real ladder) or "derived"
    (rescaled from another map's measured band) — "none" is not a DoseBand
    at all, it is the absence of one (`resolve_dose_band` returns `None`).

    `norm_ref` is REQUIRED even on a measured band whose own zones need no
    rescaling: it is what lets THIS band later serve as another map's
    `--dose-band-reference`, and every `loom_map_*.npz` already carries it,
    so there is no reason not to record it at attach time.
    """

    tier: str
    zones: tuple[DoseZone, ...]
    norm_ref: tuple[float, ...]
    source: str
    alpha_max: float
    map_fingerprint: str | None = None
    measured_at: str | None = None
    notes: str | None = None
    ladder: dict[str, Any] | None = None
    # The rescale receipt for a DERIVED band — the statistic
    # `rescale()` actually used, plus (when a `ResidualScale` was supplied) the
    # measured bound on the competing cross-model hypothesis. A derived band
    # asserts "audibility tracks absolute injected norm"; across model sizes
    # that is an assumption, and this field is where its size is written down
    # instead of being left as an unquantified worry. Never None on a band
    # produced by `rescale()`; None on a measured band, which derives nothing.
    derivation: dict[str, Any] | None = None
    schema: int = CURRENT_SCHEMA

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"dose band tier must be one of {TIERS}, got {self.tier!r}")
        if not self.zones:
            raise ValueError("dose band has no zones")
        zs = self.zones
        if abs(zs[0].lo - 0.0) > 1e-9:
            raise ValueError(f"dose band's first zone must start at alpha=0, starts at {zs[0].lo}")
        for i in range(len(zs) - 1):
            if abs(zs[i].hi - zs[i + 1].lo) > 1e-9:
                raise ValueError(
                    f"dose band zones are not contiguous: zone {zs[i].key!r} ends at "
                    f"{zs[i].hi} but zone {zs[i + 1].key!r} starts at {zs[i + 1].lo} — "
                    "every alpha from 0 to alpha_max must land in exactly one zone"
                )
        if abs(zs[-1].hi - self.alpha_max) > 1e-9:
            raise ValueError(
                f"dose band alpha_max ({self.alpha_max}) does not match the last zone's "
                f"hi ({zs[-1].hi})"
            )
        if not self.norm_ref:
            raise ValueError("dose band norm_ref must be a non-empty list")
        if any(n <= 0 for n in self.norm_ref):
            raise ValueError(f"dose band norm_ref must be strictly positive, got {self.norm_ref}")

    def zone_for(self, alpha: float) -> DoseZone:
        """Mirrors the UI's `zoneFor()`: first zone that contains
        alpha wins; an alpha outside [0, alpha_max] clamps to the nearest
        edge zone rather than raising — a slider one step past the end
        should not crash the gauge."""
        a = float(alpha)
        for z in self.zones:
            if z.contains(a):
                return z
        return self.zones[0] if a < self.zones[0].lo else self.zones[-1]

    def rescale(self, target_norm_ref: Sequence[float],
                target_map_fingerprint: str | None,
                residual_scale: ResidualScale | None = None) -> DoseBand:
        """Derive a provisional band for a DIFFERENT map by norm_ref ratio.

        ★ Why this is the whole feature (`docs/FINDINGS.md §2`; re-derive
        from `LoomMap.lever_of` if its ruler changes). `LoomMap.lever_of` rescales
        every predicted lever, per site, to `norm_ref[s]`, and `alpha`
        multiplies the result — so the vector actually injected at site `s`
        has norm `alpha * norm_ref[s]`. Per-site normalization buys
        comparability WITHIN a map (alpha means one fixed perturbation size
        no matter which future was picked); it buys nothing ACROSS maps,
        because `norm_ref` differs map to map. A ladder measured against
        w5-r32 (whose norm_ref runs ~2.1x the old map's) confirmed the
        natural hypothesis directly: audibility tracks ABSOLUTE perturbation
        magnitude, and w5-r32's measured thresholds landed at roughly HALF
        the old map's alphas — matching the norm_ref ratio, not "same alpha"
        (ratio 1).

        `scale = 1 / mean_s(target_norm_ref[s] / self.norm_ref[s])` — the
        MEAN of the per-site ratio target/reference, inverted, rather than the
        ratio of the two means. Validated on the old map -> w5-r32 pair: applied
        to the old map's band with no access to the w5-r32 ladder, this method
        predicted a subliminal ceiling of 0.236 (measured <=0.25) and, through
        `overdriven_ceiling`, a damage line of 0.708 (matched by a live wear at
        alpha=0.75 on w5-r32 producing visible word-salad) — one ladder rung of
        agreement on both checked numbers. See the module docstring for the
        spread check below and when it gets shakier.

        ★ `residual_scale` does NOT change a single number
        in the returned band. It attaches the measured bound on the competing
        RELATIVE hypothesis to `derivation` (see `cross_model_bound`), which is
        the only honest thing to do for a transfer across model sizes: the
        same-model validation above cannot distinguish absolute from relative,
        so a cross-model derived band that does not state the size of that
        ambiguity is claiming more than it knows. Pass it whenever the
        reference and target maps are from different models; omit it within one
        model, where the two hypotheses coincide.
        """
        if len(target_norm_ref) != len(self.norm_ref):
            raise ValueError(
                f"cannot derive a dose band: this band has {len(self.norm_ref)} sites, "
                f"the target map has {len(target_norm_ref)} — bands only transfer between "
                "maps with the same number of steering sites"
            )
        target_norm_ref = [float(x) for x in target_norm_ref]
        if any(t <= 0 for t in target_norm_ref):
            raise ValueError(
                f"target map's norm_ref must be strictly positive, got {target_norm_ref}")
        try:
            mean_ratio, ratios = _mean_ratio(target_norm_ref, self.norm_ref)
        except ValueError as exc:
            raise ValueError("mean per-site norm_ref ratio is non-positive — cannot derive "
                              f"a dose band ({exc})") from exc
        scale = 1.0 / mean_ratio
        spread = (max(ratios) - min(ratios)) / mean_ratio
        if spread > _SITE_RATIO_SPREAD_WARN:
            logger.warning(
                "dose band derivation: per-site norm_ref ratios disagree by %.0f%% "
                "(ratios target/reference=%s) — the mean-ratio rescale is a rougher "
                "approximation than usual for this map pair",
                spread * 100, [round(r, 4) for r in ratios],
            )
        new_zones = tuple(
            DoseZone(lo=round(z.lo * scale, 6), hi=round(z.hi * scale, 6),
                     hi_inclusive=z.hi_inclusive, key=z.key, label=z.label, hint=z.hint)
            for z in self.zones
        )
        derivation: dict[str, Any] = {
            "method": "per-site norm_ref-ratio rescale (mean of target/reference, inverted)",
            "reference_source": self.source,
            "reference_norm_ref": [float(x) for x in self.norm_ref],
            "reference_alpha_max": float(self.alpha_max),
            "mean_ratio": round(mean_ratio, 6),
            "scale": round(scale, 6),
            "per_site_ratio": [round(r, 6) for r in ratios],
            "per_site_spread": round(spread, 6),
            "per_site_spread_warn_threshold": _SITE_RATIO_SPREAD_WARN,
            "per_site_spread_exceeds_threshold": bool(spread > _SITE_RATIO_SPREAD_WARN),
            "cross_model": None,
        }
        source = (f"derived from {self.source!r} via per-site norm_ref-ratio rescale "
                  f"(mean ratio {mean_ratio:.4f}, x{scale:.4f}) — PROVISIONAL, not "
                  "itself measured on this map")
        if residual_scale is not None:
            bound = cross_model_bound(self.norm_ref, target_norm_ref, residual_scale)
            derivation["cross_model"] = bound
            source += (
                f". CROSS-MODEL transfer ({residual_scale.reference_model} -> "
                f"{residual_scale.target_model}): the competing relative-magnitude "
                f"hypothesis would put this band at x{bound['relative']['scale']:.4f} "
                f"instead, a factor of {bound['disagreement']:.4f}; the served "
                f"(absolute) numbers are the "
                f"{'CONSERVATIVE' if bound['served_is_conservative'] else '★ LESS CONSERVATIVE'} "
                "side of that ambiguity"
            )
        return DoseBand(
            tier="derived", zones=new_zones, norm_ref=tuple(target_norm_ref),
            source=source,
            alpha_max=round(self.alpha_max * scale, 6),
            map_fingerprint=target_map_fingerprint, measured_at=None, notes=self.notes,
            derivation=derivation, schema=self.schema,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "tier": self.tier, "source": self.source,
            "map_fingerprint": self.map_fingerprint, "norm_ref": list(self.norm_ref),
            "alpha_max": self.alpha_max, "measured_at": self.measured_at,
            "notes": self.notes, "ladder": self.ladder,
            "derivation": self.derivation,
            "zones": [z.to_json() for z in self.zones],
        }

    def to_info_json(self) -> dict[str, Any]:
        """The subset served on `GET /info` — small and JSON-safe. Omits
        `norm_ref` and `ladder` (internal/bulky provenance, not needed to
        render a gauge); the `source` string already names the scale factor
        for a derived band, so a viewer never has to reconstruct it.
        `lever_quality_caveat` is always the same constant string (see
        `LEVER_QUALITY_CAVEAT`) — a standing disclaimer, not per-band data.

        ★ `derivation` IS served, unlike `norm_ref`/`ladder`:
        it is small, and for a cross-model derived band it carries the band's
        own error bar. An operator who can see `tier: "derived"` but not how far
        the transfer reached, or that a competing hypothesis puts the boundaries
        somewhere else, cannot tell a 1.17x ambiguity from a 2x one — and 2x is
        the failure this module exists to prevent."""
        return {
            "tier": self.tier, "alpha_max": self.alpha_max, "source": self.source,
            "measured_at": self.measured_at, "notes": self.notes,
            "map_fingerprint": self.map_fingerprint,
            "derivation": self.derivation,
            "zones": [z.to_json() for z in self.zones],
            "lever_quality_caveat": LEVER_QUALITY_CAVEAT,
        }

    @classmethod
    def from_json(cls, value: Any) -> DoseBand:
        blob = _obj(value, "dose band")
        schema = blob.get("schema", 1)
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise ValueError(f"dose band 'schema' must be an integer, got {schema!r}")
        if schema > CURRENT_SCHEMA:
            raise ValueError(f"dose band schema {schema} is newer than this reader supports "
                              f"(schema {CURRENT_SCHEMA}) — upgrade pleroma")
        tier = _str(blob.get("tier"), "dose band 'tier'")
        zones_raw = blob.get("zones")
        if not isinstance(zones_raw, list) or not zones_raw:
            raise ValueError("dose band 'zones' must be a non-empty list")
        zones = tuple(DoseZone.from_json(z) for z in zones_raw)
        norm_ref = _floats(blob.get("norm_ref"), "dose band 'norm_ref'")
        source = _str(blob.get("source"), "dose band 'source'")
        alpha_max = _num(blob.get("alpha_max"), "dose band 'alpha_max'")
        ladder = blob.get("ladder")
        if ladder is not None and not isinstance(ladder, dict):
            raise ValueError("dose band 'ladder' must be an object or null")
        derivation = blob.get("derivation")
        if derivation is not None and not isinstance(derivation, dict):
            raise ValueError("dose band 'derivation' must be an object or null")
        return cls(
            tier=tier, zones=zones, norm_ref=tuple(norm_ref), source=source,
            alpha_max=alpha_max,
            map_fingerprint=_opt_str(blob.get("map_fingerprint"), "dose band 'map_fingerprint'"),
            measured_at=_opt_str(blob.get("measured_at"), "dose band 'measured_at'"),
            notes=_opt_str(blob.get("notes"), "dose band 'notes'"),
            ladder=ladder, derivation=derivation, schema=schema,
        )


NONE_INFO_JSON: dict[str, Any] = {
    "tier": "none", "alpha_max": None, "source": None, "measured_at": None,
    "notes": None, "map_fingerprint": None, "derivation": None, "zones": None,
    "lever_quality_caveat": None,
}


def dose_band_info_json(band: DoseBand | None) -> dict[str, Any]:
    """What `/info.dose_band` serves — `NONE_INFO_JSON` when there is
    nothing to report. The UI's job (`pleroma/serve/static/legacy_ui.html`) is to render
    `tier: "none"` as UNCALIBRATED and never invent numbers for it."""
    return band.to_info_json() if band is not None else dict(NONE_INFO_JSON)


# ── file IO ───────────────────────────────────────────────────────────────


def default_sidecar_path(map_path: Path) -> Path:
    """`<map>.doseband.json`, next to the map — the auto-discovered
    convention a server falls back to when `--dose-band` is not given."""
    map_path = Path(map_path)
    return map_path.parent / (map_path.name + ".doseband.json")


def load_dose_band_file(path: Path) -> DoseBand:
    """FAILS LOUDLY on anything malformed — unreadable, invalid JSON, or a
    JSON object that fails `DoseBand.from_json`'s validation. A dose band
    that cannot be trusted must never be half-applied; the caller (server
    startup or the CLI) is expected to let this exception surface, not
    swallow it into a degraded gauge."""
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read dose band file {path}: {exc}") from exc
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return DoseBand.from_json(blob)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed dose band — {exc}") from exc


def save_dose_band_file(band: DoseBand, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(band.to_json(), indent=2) + "\n")


# ── resolution: what a server actually serves ────────────────────────────


def resolve_dose_band(
    *, measured: DoseBand | None, reference: DoseBand | None,
    target_norm_ref: Sequence[float], target_map_fingerprint: str,
    residual_scale: ResidualScale | None = None,
) -> DoseBand | None:
    """The one function the server (and `python -m pleroma.dose.band show`) calls to decide
    what tier a map gets. Priority: a MEASURED band attached to this exact
    map, else a DERIVED band rescaled from a reference measured elsewhere,
    else `None` (tier "none" — uncalibrated). Never silently falls back to
    a band stamped for a different map: a fingerprint mismatch on `measured`
    is refused, not downgraded to derived or ignored, because that would
    hide a real misconfiguration (the wrong sidecar sitting next to the
    wrong map) behind a plausible-looking gauge.

    `residual_scale` is forwarded to `DoseBand.rescale` and only affects the
    DERIVED branch's `derivation.cross_model` provenance — never any served
    boundary. It is ignored when a measured band wins, which is correct: a
    band measured on this map derives nothing and needs no error bar on a
    transfer it did not make.
    """
    if measured is not None:
        match = same_map(measured.map_fingerprint, target_map_fingerprint)
        if match == "legacy":
            logger.warning(
                "dose band carries a LEGACY map fingerprint (%s): it matches the "
                "loaded map's v1 id, but v1 never covered the weights. "
                "Re-stamp the band with this map's v2 id (%s) once confirmed",
                measured.map_fingerprint, target_map_fingerprint)
        if match == "mismatch":
            raise ValueError(
                f"dose band is stamped for map fingerprint {measured.map_fingerprint} but "
                f"the server loaded {target_map_fingerprint} — refusing to serve a "
                "possibly-wrong-map band as MEASURED. If this band is meant as a rescale "
                "source for the loaded map, pass it as --dose-band-reference instead."
            )
        return measured
    if reference is not None:
        return reference.rescale(target_norm_ref, target_map_fingerprint,
                                 residual_scale=residual_scale)
    return None


# ── the overdriven ceiling: anchored on the reference's damage point, never
#    inferred from ladder accuracy (see the module docstring) ──────────────


def overdriven_ceiling(reference: DoseBand, target_norm_ref: Sequence[float]) -> float:
    """The alpha, on a map with `target_norm_ref`, that corresponds to
    `reference.alpha_max` — the reference map's own documented damage point
    (e.g. the old map's 1.5, where its replies turn to word-salad). Deliberately
    anchored on the reference's own top edge, not wherever its "overdriven" zone
    happens to start, because that top edge is the one number in the old
    doctrine with an unambiguous "don't" attached to it. Reuses
    `DoseBand.rescale`'s exact per-site-ratio method (so this number and a full
    `rescale()` of the same reference always agree) — a standalone entry point
    because the ceiling is wanted even when the caller is not deriving a whole
    band, e.g. `apply_overdriven_ceiling` capping a MEASURED band's top zone at
    attach time.
    """
    return reference.rescale(target_norm_ref, target_map_fingerprint=None).alpha_max


def apply_overdriven_ceiling(
    zones: Sequence[DoseZone], ceiling: float, alpha_max: float,
) -> tuple[list[DoseZone], bool]:
    """Cap `zones` at a damage ceiling derived from a REFERENCE map's known
    damage alpha — never inferred from this band's own ladder accuracy,
    which keeps climbing into damage instead of falling (module docstring;
    separating register content from coherence damage is a manual read, and
    the w5-r32 judges' own free-text
    cues at alpha 0.75-1.0 say "garbled details", "word-salad sentence",
    "cuts off mid-sentence" — high accuracy there is damage detection, not
    disposition legibility).

    Any zone extending past `ceiling` is truncated there; whatever alpha
    range remains up to `alpha_max` becomes a single "overdriven" zone
    (replacing anything the ladder itself called audible/threshold/etc past
    that point). Returns `(new_zones, sampled_into_damage)` — the second
    element is True iff a non-overdriven zone actually reached past
    `ceiling` before clamping, so the caller can warn: this ladder's own top
    rung(s) were measuring damage, not dose.
    """
    ceiling = float(ceiling)
    amax = float(alpha_max)
    if ceiling >= amax:
        return list(zones), False
    out: list[DoseZone] = []
    sampled_into_damage = False
    final_lo = ceiling
    for z in zones:
        if z.hi <= ceiling:
            out.append(z)
            continue
        sampled_into_damage = sampled_into_damage or (z.key != "overdriven")
        if z.lo < ceiling and z.key != "overdriven":
            # a non-overdriven zone straddles the ceiling: keep its part
            # below the line, and the new overdriven zone starts exactly
            # at the ceiling
            out.append(DoseZone(lo=z.lo, hi=ceiling, hi_inclusive=False,
                                 key=z.key, label=z.label, hint=z.hint))
        else:
            # already labelled "overdriven" (or starts at/after the
            # ceiling with nothing to preserve below it) — extend from its
            # own start instead of manufacturing a redundant second
            # "overdriven" zone right next to this one
            final_lo = z.lo
        break
    out.append(DoseZone(
        lo=final_lo, hi=amax, hi_inclusive=True, key="overdriven", label="overdriven",
        hint="OVERDRIVEN — at/above the damage ceiling derived from the reference map's "
             "own known word-salad point, not inferred from this ladder's accuracy (which "
             "keeps climbing into damage rather than falling)",
    ))
    return out, sampled_into_damage


# ── CLI ───────────────────────────────────────────────────────────────────


def _iso_today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _cmd_attach(args: argparse.Namespace) -> None:
    fp, _sites, norm_ref, _meta = load_map_identity(args.map)

    if args.zones_json is not None and args.boundaries is not None:
        raise SystemExit("attach: pass --zones-json OR --boundaries, not both")
    if args.zones_json is not None:
        zones_blob = json.loads(Path(args.zones_json).read_text())
        if not isinstance(zones_blob, list) or not zones_blob:
            raise SystemExit(f"--zones-json {args.zones_json} must contain a non-empty JSON list")
        zones = [DoseZone.from_json(z) for z in zones_blob]
        alpha_max = args.alpha_max if args.alpha_max is not None else zones[-1].hi
    elif args.boundaries is not None:
        if args.alpha_max is None:
            raise SystemExit("attach --boundaries also needs --alpha-max "
                              "(the top of the last zone)")
        boundaries = [float(x) for x in args.boundaries.split(",")]
        zones = five_zone_band(boundaries, args.alpha_max)
        alpha_max = args.alpha_max
    else:
        raise SystemExit("attach needs --zones-json or --boundaries")

    if args.overdriven_reference is not None:
        ref_band = load_dose_band_file(args.overdriven_reference)
        ceiling = overdriven_ceiling(ref_band, norm_ref)
        zones, sampled = apply_overdriven_ceiling(zones, ceiling, alpha_max)
        print(f"overdriven ceiling from {args.overdriven_reference}: {ceiling:.4f}")
        if sampled:
            print(f"WARNING: this ladder's own zones extended past the derived overdriven "
                  f"ceiling ({ceiling:.4f}) — the top rung(s) are likely measuring damage, "
                  "not dose; capped and relabelled 'overdriven'. Read the top rungs by hand: "
                  "register content is dose, coherence damage is not.")

    ladder = None
    if args.ladder_json is not None:
        ladder = json.loads(Path(args.ladder_json).read_text())
        if not isinstance(ladder, dict):
            raise SystemExit(f"--ladder-json {args.ladder_json} must contain a JSON object")

    band = DoseBand(
        tier="measured", zones=tuple(zones), norm_ref=tuple(float(x) for x in norm_ref),
        source=args.source, map_fingerprint=fp, alpha_max=float(alpha_max),
        measured_at=args.measured_at or _iso_today(), notes=args.notes, ladder=ladder,
    )
    out = args.out or default_sidecar_path(args.map)
    save_dose_band_file(band, out)
    print(f"wrote {out}  (tier=measured, map_fingerprint={fp}, "
          f"{len(band.zones)} zones, alpha_max={band.alpha_max})")


def _print_band(band: DoseBand, header: str) -> None:
    print(f"{header} — tier={band.tier}")
    print(f"  source: {band.source}")
    for z in band.zones:
        edge = "]" if z.hi_inclusive else ")"
        print(f"  [{z.lo:.4f}, {z.hi:.4f}{edge}  {z.key:<12} {z.hint}")
    d = band.derivation or {}
    cm = d.get("cross_model")
    if cm:
        print("  CROSS-MODEL bound "
              f"({cm['residual_scale']['reference_model']} -> "
              f"{cm['residual_scale']['target_model']}):")
        print(f"    absolute (SERVED) x{cm['absolute']['scale']:.4f}   "
              f"relative x{cm['relative']['scale']:.4f}   "
              f"disagreement {cm['disagreement']:.4f}x")
        print(f"    residual-stream scale target/reference "
              f"{cm['residual_scale']['mean_ratio_target_over_reference']:.4f}   "
              f"served side = {cm['conservative'] if cm['served_is_conservative'] else 'NOT the conservative one'}")
    elif band.tier == "derived":
        print("  (no cross-model bound attached — pass --residual-scale if the "
              "reference band was measured on a DIFFERENT model)")


def _resolve_for_cli(args: argparse.Namespace) -> tuple[str, Sequence[float], DoseBand | None]:
    fp, _sites, norm_ref, _meta = load_map_identity(args.map)
    band_path = args.dose_band or default_sidecar_path(args.map)
    measured = None
    if args.dose_band is not None:
        measured = load_dose_band_file(band_path)
    elif band_path.exists():
        measured = load_dose_band_file(band_path)
    reference = (load_dose_band_file(args.dose_band_reference)
                 if args.dose_band_reference else None)
    residual = (load_residual_scale_file(args.residual_scale)
                if getattr(args, "residual_scale", None) else None)
    resolved = resolve_dose_band(measured=measured, reference=reference,
                                 target_norm_ref=norm_ref, target_map_fingerprint=fp,
                                 residual_scale=residual)
    return fp, norm_ref, resolved


def _cmd_show(args: argparse.Namespace) -> None:
    fp, _norm_ref, resolved = _resolve_for_cli(args)
    if resolved is None:
        print(f"map {args.map}: fingerprint {fp} — NO dose band (tier=none, uncalibrated)")
        return
    _print_band(resolved, f"map {args.map}: fingerprint {fp}")


def _cmd_derive(args: argparse.Namespace) -> None:
    """Freeze a DERIVED band into a reviewable sidecar.

    ★ WHY THIS EXISTS. Without it a derived band is recomputed at every
    server startup from `--dose-band-reference`, so the numbers an operator
    reads off the gauge exist nowhere in git, cannot be diffed, and cannot
    carry provenance that `rescale()` does not
    itself compute — in particular the cross-model bound. `derive` runs the
    exact same `resolve_dose_band` the server runs and writes the result,
    tier and all, next to the map.

    It REFUSES to write a measured band (use `attach` for that) and it never
    relabels: what comes out is `tier: "derived"`, which is what the server
    will then serve from the sidecar, byte-identical to what it would have
    computed on the fly.
    """
    if args.dose_band_reference is None:
        raise SystemExit("derive needs --dose-band-reference (the measured band, "
                          "from ANOTHER map, to rescale from)")
    fp, norm_ref, resolved = _resolve_for_cli(args)
    if resolved is None:
        raise SystemExit("derive: nothing to derive — the reference band produced "
                          "no band for this map")
    if resolved.tier != "derived":
        raise SystemExit(
            f"derive: resolution produced a {resolved.tier!r} band, not a derived one — "
            "a MEASURED sidecar already exists for this map and `derive` will not "
            "overwrite a real ladder with a rescale. Remove or rename it first, or "
            "use `show` to inspect what is in force.")
    _print_band(resolved, f"map {args.map}: fingerprint {fp}")
    out = args.out or default_sidecar_path(args.map)
    if out.exists() and not args.force:
        raise SystemExit(f"derive: {out} already exists — pass --force to replace it")
    save_dose_band_file(resolved, out)
    print(f"wrote {out}  (tier=derived, map_fingerprint={fp}, "
          f"{len(resolved.zones)} zones, alpha_max={resolved.alpha_max})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    attach = sub.add_parser(
        "attach", help="write/attach a MEASURED dose band sidecar for a map")
    attach.add_argument(
        "--map", type=Path, required=True,
        help="the loom_map_*.npz to attach a band to (read-only — only its "
             "sites/norm_ref/mu_y/meta are read, no discriminants needed)")
    attach.add_argument("--out", type=Path, default=None,
                         help="sidecar path (default: <map>.doseband.json)")
    attach.add_argument("--source", required=True,
                         help="human-readable provenance — the ladder doc and date")
    attach.add_argument("--measured-at", default=None, help="ISO date (default: today, UTC)")
    attach.add_argument("--notes", default=None)
    attach.add_argument(
        "--boundaries", default=None,
        help="4 comma-separated alphas for the classic 5-zone shape "
             "(inert_hi,subliminal_hi,threshold_hi,audible_hi) — pairs with --alpha-max")
    attach.add_argument(
        "--alpha-max", type=float, default=None,
        help="top of the last zone (required with --boundaries; defaults to the "
             "last --zones-json zone's hi otherwise)")
    attach.add_argument(
        "--zones-json", type=Path, default=None,
        help="arbitrary zone list (JSON array of {lo,hi,hi_inclusive,key,label,hint}) "
             "— use instead of --boundaries when the ladder does not fit the 5-zone "
             "shape (e.g. a short ladder that only resolves 3 bands)")
    attach.add_argument(
        "--ladder-json", type=Path, default=None,
        help="optional raw ladder/report data, stored verbatim as audit provenance — "
             "NOT used to classify zones (separating register content from "
             "coherence damage is a manual read)")
    attach.add_argument(
        "--overdriven-reference", type=Path, default=None,
        help="a measured band from a map with a KNOWN damage point (e.g. the old map's "
             "documented alpha=1.5) — when given, its top edge is rescaled by norm_ref "
             "(overdriven_ceiling) and used to cap/relabel this band's own top zone(s), "
             "regardless of what the ladder's raw accuracy suggested. Recommended: ladder "
             "accuracy alone cannot tell 'audible-and-legible' from 'audible-because-it-"
             "broke' (see the module docstring)")
    attach.set_defaults(func=_cmd_attach)

    _RESIDUAL_HELP = (
        "a residual-scale JSON (see ResidualScale / dose_bands/"
        "residual_scale_3b_8b.json) giving both models' per-site median ||h||. "
        "REQUIRED FOR A CROSS-MODEL DERIVATION and ignored otherwise: it changes "
        "no served boundary, it attaches the measured bound on the competing "
        "relative-magnitude hypothesis so the band carries its own error bar")

    for name, helptext in (
        ("show", "print what a server would resolve for a map"),
        ("derive", "freeze the DERIVED band a server would compute into a "
                   "reviewable <map>.doseband.json sidecar (never 'measured')"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--map", type=Path, required=True)
        sp.add_argument("--dose-band", type=Path, default=None,
                        help="measured band file (default: <map>.doseband.json if present)")
        sp.add_argument(
            "--dose-band-reference", type=Path, default=None,
            help="a measured band from ANOTHER map, used to derive a provisional band "
                 "when no measured band exists for --map")
        sp.add_argument("--residual-scale", type=Path, default=None, help=_RESIDUAL_HELP)
        if name == "derive":
            sp.add_argument("--out", type=Path, default=None,
                            help="sidecar path (default: <map>.doseband.json)")
            sp.add_argument("--force", action="store_true",
                            help="replace an existing sidecar at --out")
        sp.set_defaults(func=_cmd_show if name == "show" else _cmd_derive)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
