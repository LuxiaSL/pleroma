"""Stage `fit`: read a registered v1a CV report (the ``<out>/v1a_report.json``
`pleroma.map.build.cv` writes) so its numbers cannot be misread.

| gate      | catalog | question |
|-----------|---------|----------|
| join      | K-01, small-corpus policy | did the CV see the fans the profile serves, and enough rows? |
| heldout   | E-01, E-02, E-03 | does HELD-OUT retrieval beat the zero-training shelf baseline and chance, at the served point? |
| ceiling   | E-01    | can this retrieval test discriminate at all, or is it saturated? |

`heldout` answers "is the map better than the shelf"; `ceiling` answers "does a
high top-1 mean anything here". At 70B both are needed: the map beats the shelf
(PASS) on a test whose shelf is already .909 (INCONCLUSIVE as a quality claim).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pleroma.config import ModelProfile
from pleroma.map.build.cv import MIN_CV_ROWS
from pleroma.validate._io import InputError, read_json
from pleroma.validate.result import GateResult, failed, guarded, inconclusive, passed

STAGE = "fit"
#: E-01: a zero-training baseline at or above this leaves no room to discriminate.
CEILING_SHELF_TOP1: float = 0.90
#: E-01: a map top-1 at or above this is saturated whatever the baseline.
CEILING_MAP_TOP1: float = 0.99
#: The registered test's own alpha (cv.py: PASS iff observed > 0 and p < .05).
ALPHA: float = 0.05


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class PrimaryPoint(_Loose):
    input: str
    target: str
    lam: float = Field(alias="lambda")
    rank: int


class ShelfBaseline(_Loose):
    retrieval_top1: float = Field(ge=0, le=1)


class RegisteredTest(_Loose):
    observed: float
    v1a_top1: float = Field(ge=0, le=1)
    shelf_top1: float = Field(ge=0, le=1)
    one_sided_p: float = Field(ge=0, le=1)
    verdict: str | None = None


class CVReport(_Loose):
    """The fields of the v1a CV report the gates read (others are kept)."""

    stage: str
    n_gens: int = Field(ge=0)
    n_fans: int = Field(ge=0)
    chance_top1: float = Field(gt=0, le=1)
    fan_source: str | None = None
    primary_point: PrimaryPoint
    baseline_shelf: ShelfBaseline
    registered_test: RegisteredTest
    grid: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_cv_report(path: str | Path) -> CVReport:
    blob = read_json(path, "CV report")
    try:
        report = CVReport.model_validate(blob)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise InputError(f"CV report is not a v1a_fit report: {loc}: {first['msg']} "
                         f"({exc.error_count()} problem(s))") from exc
    if report.stage != "v1a_fit":
        raise InputError(f"CV report stage is {report.stage!r}, expected 'v1a_fit'")
    return report


def _grid_key(lam: float, rank: int, inp: str, target: str) -> str:
    """The key `cv.py` writes a grid cell under."""
    return f"{inp}|{target}|lam{lam:g}|r{rank}"


def _served_point_matches(report: CVReport, profile: ModelProfile) -> bool:
    p = report.primary_point
    return (abs(p.lam - profile.map.lam) <= 1e-9 * max(1.0, abs(p.lam))
            and p.rank == profile.map.rank and p.target == profile.map.target)


@guarded(STAGE, "join")
def gate_join(report: CVReport, profile: ModelProfile) -> GateResult:
    """K-01 (fans dropped by the 2-means lever join) + the small-corpus policy
    (small corpora are reported, not refused)."""
    ev = dict(n_gens=report.n_gens, n_fans=report.n_fans, fan_source=report.fan_source,
              profile_fan_source=profile.map.fan_source, min_cv_rows=MIN_CV_ROWS)
    if report.fan_source is None:
        return inconclusive(STAGE, "join",
                            "CV report does not record its fan source: cannot tell whether "
                            "lone-outlier fans were dropped by the 2-means lever join (K-01)",
                            **ev)
    if report.fan_source != profile.map.fan_source:
        return failed(STAGE, "join",
                      f"CV joined fans via {report.fan_source!r} but the profile serves "
                      f"{profile.map.fan_source!r}: the CV did not see the served rows "
                      "(the 'levers' join drops every fan 2-means cannot split, K-01)", **ev)
    if report.n_gens < MIN_CV_ROWS:
        return inconclusive(STAGE, "join",
                            f"only {report.n_gens} rows joined (< {MIN_CV_ROWS}): held-out "
                            "retrieval is noisy at this size — read it, don't bank it", **ev)
    return passed(STAGE, "join",
                  f"CV joined {report.n_gens} gens in {report.n_fans} fans via "
                  f"{report.fan_source!r}, the source the profile serves", **ev)


@guarded(STAGE, "heldout")
def gate_heldout(report: CVReport, profile: ModelProfile) -> GateResult:
    """E-01/E-02/E-03: held-out (never in-sample) top-1 against the SHELF
    baseline and chance, via the report's own registered paired test."""
    t = report.registered_test
    p = report.primary_point
    ev: dict[str, Any] = dict(
        heldout_top1=t.v1a_top1, shelf_top1=t.shelf_top1, chance_top1=report.chance_top1,
        margin_vs_shelf=t.observed, one_sided_p=t.one_sided_p,
        report_verdict=t.verdict, registered_point={"lambda": p.lam, "rank": p.rank,
                                                    "input": p.input, "target": p.target},
        served_point={"lambda": profile.map.lam, "rank": profile.map.rank,
                      "target": profile.map.target})
    if not _served_point_matches(report, profile):
        cell = report.grid.get(_grid_key(profile.map.lam, profile.map.rank, p.input,
                                         profile.map.target))
        ev["served_point_grid_cell"] = cell
        return inconclusive(STAGE, "heldout",
                            f"the registered test is at λ={p.lam:g}, rank {p.rank}, target "
                            f"{p.target!r}, but the profile serves λ={profile.map.lam:g}, rank "
                            f"{profile.map.rank}, target {profile.map.target!r}: the test does "
                            "not describe the served map (its grid cell is descriptive only)",
                            **ev)
    if t.v1a_top1 <= report.chance_top1:
        return failed(STAGE, "heldout",
                      f"held-out top-1 {t.v1a_top1:.4f} is at or below chance "
                      f"{report.chance_top1:.4f}: the map retrieves nothing", **ev)
    if not (t.observed > 0 and t.one_sided_p < ALPHA):
        return failed(STAGE, "heldout",
                      f"map does not beat the zero-training shelf baseline: held-out top-1 "
                      f"{t.v1a_top1:.4f} vs shelf {t.shelf_top1:.4f} (margin {t.observed:+.4f}, "
                      f"one-sided p={t.one_sided_p:.3g})", **ev)
    if t.verdict is not None and t.verdict != "PASS":
        return inconclusive(STAGE, "heldout",
                            f"report's own verdict is {t.verdict!r} but its numbers pass "
                            "(margin > 0, p < .05): the report is internally inconsistent", **ev)
    return passed(STAGE, "heldout",
                  f"held-out top-1 {t.v1a_top1:.4f} beats the shelf baseline {t.shelf_top1:.4f} "
                  f"(margin {t.observed:+.4f}, p={t.one_sided_p:.3g}) and chance "
                  f"{report.chance_top1:.3f}", **ev)


@guarded(STAGE, "ceiling")
def gate_ceiling(report: CVReport) -> GateResult:
    """E-01: at 70B the shelf already scores .903 and the map .998 — "0.998!"
    says nothing about map quality. Flag it where it happens."""
    t = report.registered_test
    p = report.primary_point
    cell = report.grid.get(_grid_key(p.lam, p.rank, p.input, p.target)) or {}
    ev = dict(shelf_top1=t.shelf_top1, heldout_top1=t.v1a_top1,
              ceiling_shelf=CEILING_SHELF_TOP1, ceiling_map=CEILING_MAP_TOP1,
              heldout_pred_cos=cell.get("pred_cos_vs_own_targets"))
    if t.shelf_top1 >= CEILING_SHELF_TOP1:
        return inconclusive(STAGE, "ceiling",
                            f"retrieval test at ceiling: the zero-training shelf baseline "
                            f"already scores {t.shelf_top1:.3f} (>= {CEILING_SHELF_TOP1}), so a "
                            "high top-1 cannot rank maps — read held-out pred-cos "
                            "descriptively instead", **ev)
    if t.v1a_top1 >= CEILING_MAP_TOP1:
        return inconclusive(STAGE, "ceiling",
                            f"retrieval saturated: held-out top-1 {t.v1a_top1:.4f} "
                            f"(>= {CEILING_MAP_TOP1}) leaves no headroom to compare maps", **ev)
    return passed(STAGE, "ceiling",
                  f"retrieval test has headroom (shelf {t.shelf_top1:.3f}, map "
                  f"{t.v1a_top1:.3f}): top-1 can discriminate here", **ev)


def run_fit(cv_report: str | Path, profile: ModelProfile) -> list[GateResult]:
    try:
        report = load_cv_report(cv_report)
    except InputError as exc:
        return [inconclusive(STAGE, g, str(exc), path=str(cv_report))
                for g in ("join", "heldout", "ceiling")]
    return [gate_join(report, profile), gate_heldout(report, profile), gate_ceiling(report)]
