"""The auto-pick policies: the pure decision and the /info copy about them.

The picking-skill figures quoted here are fractions of an oracle's picking
skill, measured on 3B; docs/FINDINGS.md §8 is the public summary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pleroma.errors import CodedError, ErrorCode


AUTO_POLICIES = ("distinct", "loudest", "stay", "swerve", "gauge")

# `gauge` is the one policy here that cannot be computed from the draw alone:
# it needs a PROBE (steer toward each candidate, generate, extract, measure
# where the reply landed — pleroma.probe.orchestrator). /loom runs it only
# when the request opts in with "probe"; /probe runs it on an existing fan.
# Listed in AUTO_POLICIES so `auto.policy: "gauge"` is accepted and validated
# like any other, and refused with a USEFUL message when no probe has run.
PROBE_ONLY_POLICIES: tuple[str, ...] = ("gauge",)


def pick_auto(
    policy: str, scores: Mapping[int, Mapping[str, Any]], live: Sequence[int]
) -> tuple[int, str]:
    """The auto-select policies' pure decision: which candidate index to
    wear, and the EFFECTIVE policy actually used (the caller's banner must
    say which policy ran; no current policy falls back, so it always equals
    `policy`, but callers read the returned one). `scores` is the full
    per-candidate score dict /loom already computed; this only picks from it.

    Raises ValueError for an unknown policy, no live candidate at all, or a
    policy whose required score nobody has (e.g. `stay`/`swerve` need
    `cos_to_worn`, which needs something worn). Raises CodedError
    (``GAUGE_NEEDS_PROBE``) for `gauge` on a fan with no probe scores.
    """
    if policy not in AUTO_POLICIES:
        raise ValueError(f"auto.policy must be one of {AUTO_POLICIES}")
    if not live:
        raise ValueError("auto-select: no harvested future to pick from")

    def by(key: str, sign: float) -> int:
        vals = [(sign * scores[i].get(key, float("-inf")), i) for i in live]
        best = max(vals)
        if best[0] == float("-inf"):
            raise ValueError(f"auto policy needs score {key!r} "
                             "(is anything worn?)")
        return best[1]

    effective = policy
    if policy == "distinct":
        pick = by("distinct", +1.0)
    elif policy == "loudest":
        pick = by("loudness", +1.0)
    elif policy == "stay":
        pick = by("cos_to_worn", +1.0)
    elif policy == "swerve":
        pick = by("cos_to_worn", -1.0)
    else:  # gauge — the adopted pick policy, and the only PROBED one
        # Never falls back. A probe that did not run, or ran and scored
        # nothing, is a refusal with instructions — silently serving a
        # different policy under gauge's name would fake the gauge from
        # static geometry, which cannot stand in for a probe.
        if not any("gauge" in scores[i] and scores[i]["gauge"] is not None
                   for i in live):
            raise CodedError(
                ErrorCode.GAUGE_NEEDS_PROBE,
                "auto policy 'gauge' needs probe scores and this fan has "
                "none. Run the probe first: POST /probe, or pass "
                '"probe": true on this /loom call. (The gauge is PROBE-'
                "derived — adopted by measurement, docs/FINDINGS.md §8 — and cannot be read "
                "off the draw.)")
        vals = [(scores[i]["gauge"], i) for i in live
                if scores[i].get("gauge") is not None]
        pick = max(vals, key=lambda t: (t[0], -t[1]))[1]
    return pick, effective


# ── AUTO-POLICY PROVENANCE ────────────────────────────────────────────────────
#
# A bare list of policy KEYS on /info would not say which of them is backed by
# a measurement, and an operator reading a list of five keys would reasonably
# assume every one is equally good. They are not: their measured picking
# skill ranges from 71% of an oracle's down to unmeasured.
#
# Every entry carries an honest `status` labelled "estimated, uncertified"
# (a picking-skill estimate, not a certified measurement), and
# `AUTO_POLICIES_UNAVAILABLE` names what is missing and why. These are
# DESCRIPTIONS ONLY — `pick_auto` above is the behaviour; nothing here can
# alter a pick.
AUTO_POLICY_INFO: tuple[dict[str, Any], ...] = (
    {
        "key": "distinct", "needs_wear": False,
        "description": "highest mean cosine distance to the rest of the fan, "
                       "in corpus-standardized signature space.",
        "status": "measured — 33% of the ORACLE's picking skill "
                  "(harvest .1875 vs floor .1102). THIRD of the three measured "
                  "policies wired here: `gauge` probes at 71% and "
                  "`loudest` on v1a is free at 43%. There is no setting "
                  "in which `distinct` is the best available choice on this "
                  "server. Estimated, uncertified.",
        "cost": "free — read off the draw.",
    },
    {
        "key": "loudest", "needs_wear": False,
        "description": "largest mean RAW map-output norm before the ruler "
                       "flattens it — the map's own prediction of how far this "
                       "future wants to move.",
        "status": "★ MEASURED ON v1a: 43% of the ORACLE's "
                  "picking skill — the BEST FREE POLICY here, against 12% on "
                  "the wide map, where it fell BELOW random on `window`. "
                  "On that same conversation v1a's score harvests +.268 "
                  "against a .321 oracle. ★ THE DEFAULT RECOMMENDATION when "
                  "you are not paying for a probe: `gauge` measures 71% "
                  "but costs seconds per turn, and 43% of the skill for 0 s is "
                  "the better trade for most turns. Caveat: the harvest ground "
                  "truth is the judge's Δnr measured under OLD-MAP wear, so this "
                  "scores the PICKER, not the pairing of this picker with "
                  "v1a's own steering; n=3 conversations / 48 cells, "
                  "descriptive. Estimated, uncertified.",
        "cost": "free — read off the draw.",
    },
    {
        "key": "stay", "needs_wear": True,
        "description": "highest cosine to the code already worn — deepen the "
                       "bend you are in.",
        "status": "not measured as a pick policy; it is a steering intent, not "
                  "a quality bar. Estimated, uncertified.",
        "cost": "free — read off the draw.",
    },
    {
        "key": "swerve", "needs_wear": True,
        "description": "lowest cosine to the code already worn — leave the "
                       "bend you are in.",
        "status": "not measured as a pick policy; it is a steering intent, not "
                  "a quality bar. Estimated, uncertified.",
        "cost": "free — read off the draw.",
    },
    {
        "key": "gauge", "needs_wear": False, "needs_probe": True,
        "description": "★ the gauge, the PROBE. For each candidate: wear its "
                       "contrastive code, generate a short continuation, "
                       "extract that reply's v3 signature, and measure the "
                       "normalized rank of the target future among the fan by "
                       "COSINE distance in z space. Score = mean base nr − "
                       "mean steered nr (the judge's Δnr construction with "
                       "geometry in place of the judge). Ranks by that score.",
        "status": "★ THE POLICY THIS PROJECT ADOPTED BY MEASUREMENT (docs/FINDINGS.md §8): "
                  "71% of the ORACLE's picking skill (pooled harvest "
                  ".2731 vs a .1102 random floor) against `distinct`'s 33% and "
                  "wide-map `loudness`'s 12%. On `window` it picked the "
                  "oracle's EXACT top-4 (regret 0.000) where the static "
                  "policies picked below random. ★ But `loudest` on v1a buys "
                  "43% of that skill for FREE, so the probe's "
                  "real margin here is the remaining ~27 points, not 71. "
                  "★ As an INSTRUMENT this same gauge FAILED its registered "
                  "test (docs/FINDINGS.md §8: it missed the AUC bar by .0016) — it picks, it does "
                  "not judge; the judge remains the instrument and the human "
                  "still chooses. Estimated, uncertified.",
        "cost": "★ NOT FREE — it generates and harvests k more spans. "
                "MEASURED on 3B at k=8 reps=1: 7.9 s "
                "generate + 10.6 s harvest = 18.6 s, ON TOP of a 15.1 s draw "
                "(33.7 s for the turn). reps ride in one batched generate per "
                "candidate, so they are near-free to generate and cost a full "
                "span each to harvest. /info's `probe.cost_estimate` prices "
                "your exact settings. Opt-in only: no default turn is ever "
                "silently expensive.",
        "opt_in": 'POST /probe, or pass "probe": true on /loom.',
    },
)

# Nothing is unavailable: every measured policy is wired above. The key stays
# on /info as an empty list rather than an absent field, because clients read
# it and an absent key is ambiguous where an empty one is not.
AUTO_POLICIES_UNAVAILABLE: tuple[dict[str, Any], ...] = ()
