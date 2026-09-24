"""The probe orchestrator's pure decision layer — the probed gauge as a pick
policy.

House pattern (test_loom_detach_wear, test_loom_dose_policy): everything that
decides WHICH candidate gets worn is torch-free and model-free, so it is
pinned on a laptop. The model-touching half lives in loom_serve.run_probe and
is exercised by the live smoke, not here.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from pleroma.probe import orchestrator as lp
from pleroma.serve import legacy as ls
from pleroma.probe.freegauge import cosine_dist, gauge_nr


# ══ 1. the definition is IMPORTED, not re-implemented ═══════════════════════


def test_the_gauge_is_freegauge_calibrations_own_function() -> None:
    """★ The live policy must not be able to drift from the calibrated one by
    a paraphrase. The calibration ran these exact callables; so does the
    server."""
    import pleroma.probe.freegauge as fg

    assert lp.gauge_nr is fg.gauge_nr
    assert lp.cosine_dist is fg.cosine_dist
    assert lp.cell_delta is fg.cell_delta


def test_gauge_nr_is_cosine_in_z_space_and_normalized() -> None:
    """The gauge's definition (normalized rank of the target future among the
    fan, by cosine distance in z space), spelled out on a fan we can reason
    about by hand."""
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    # a vector sitting exactly on future 0 ranks it nearest -> nr 0
    assert gauge_nr([1.0, 0.0], fan, 0, cosine_dist) == 0.0
    # and ranks the antipode last -> nr 1
    assert gauge_nr([1.0, 0.0], fan, 2, cosine_dist) == 1.0
    # cosine, not Euclid: magnitude must not change the ranking
    assert gauge_nr([9.0, 0.0], fan, 0, cosine_dist) == 0.0


# ══ 2. the contrastive code — the construction the calibration scored ══════


def test_contrastive_code_is_own_minus_mean_of_the_rest() -> None:
    codes = [[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]]
    assert lp.contrastive_code(codes, 0) == [3.0, -1.5]
    assert lp.contrastive_code(codes, 2) == [-1.5, -1.5]


def test_contrastive_codes_of_a_fan_sum_to_zero_per_dimension() -> None:
    """A sanity property of the construction: it is a centering."""
    codes = [[1.0, 2.0], [3.0, -1.0], [-2.0, 5.0], [0.5, 0.5]]
    got = [lp.contrastive_code(codes, i) for i in range(len(codes))]
    for d in range(2):
        assert sum(g[d] for g in got) == pytest.approx(0.0)


def test_contrastive_code_refuses_a_fan_of_one_and_a_ragged_fan() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        lp.contrastive_code([[1.0]], 0)
    with pytest.raises(ValueError, match="width mismatch"):
        lp.contrastive_code([[1.0, 2.0], [1.0]], 0)
    with pytest.raises(ValueError, match="outside a fan"):
        lp.contrastive_code([[1.0], [2.0]], 5)


# ══ 3. the base term, and the one claim the default rests on ═══════════════


def test_leave_one_out_base_offset_is_CONSTANT_across_targets() -> None:
    """★ THE LOAD-BEARING CLAIM for base='fan'.

    A fan member sits in its own reference set at distance 0, so it takes rank
    0 and pushes every other target up by exactly one slot. The module claims
    that is a CONSTANT +1/(k-1) identical for every target, which is why it
    cancels in a RANKING (and why the receipt refuses to call these Δnr).
    Verified here against a leave-one-out rank computed without the self row.
    """
    fan = [[1.0, 0.1], [0.4, 1.0], [-1.0, 0.3], [0.2, -1.0], [0.9, 0.9]]
    k = len(fan)
    for t in range(k):
        for j in range(k):
            if j == t:
                continue
            with_self = gauge_nr(fan[j], fan, t, cosine_dist)
            others = [fan[m] for m in range(k) if m != j]
            # rank of t among the k-1 others, on the same /(k-1) scale
            t_in_others = [m for m in range(k) if m != j].index(t)
            without_self = gauge_nr(fan[j], others, t_in_others, cosine_dist)
            without_self *= (k - 2) / (k - 1)  # renormalize to the k-1 denom
            assert with_self - without_self == pytest.approx(1.0 / (k - 1))


def test_fan_base_refuses_a_fan_too_small_to_leave_one_out() -> None:
    with pytest.raises(ValueError, match="k>=3"):
        lp.base_nr_fan([[1.0, 0.0], [0.0, 1.0]], 0)


def test_fresh_base_uses_the_supplied_unworn_replies() -> None:
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    got = lp.base_nr_fresh(fan, [[1.0, 0.0], [0.0, 1.0]], 0)
    assert got == [0.0, 0.5]


# ══ 4. scoring and ranking ══════════════════════════════════════════════════


FAN: list[list[float]] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]


def test_a_reply_that_lands_on_its_target_outscores_one_that_does_not() -> None:
    """The whole policy in one assertion: steering that WORKED ranks first."""
    steered: dict[int, list[list[float]]] = {
        0: [[1.0, 0.0]],       # landed exactly on future 0 -> nr 0, big cell
        1: [[0.0, -1.0]],      # landed on the OPPOSITE of future 1 -> nr 1
        2: [[-1.0, 0.0]],      # landed on future 2
        3: [[0.0, 1.0]],       # landed opposite future 3
    }
    gauges = lp.score_candidates(FAN, steered, base_mode="fan")
    by = {g.index: g for g in gauges}
    assert by[0].cell is not None and by[1].cell is not None
    assert by[0].cell > by[1].cell
    assert by[2].cell > by[3].cell
    order = lp.rank_by_gauge(gauges)
    assert set(order[:2]) == {0, 2}
    assert lp.pick_gauge(gauges) in (0, 2)


def test_higher_cell_wins_and_ties_break_by_index() -> None:
    gauges = [
        lp.CandidateGauge(index=2, cell=0.5),
        lp.CandidateGauge(index=0, cell=0.5),
        lp.CandidateGauge(index=1, cell=0.9),
    ]
    assert lp.rank_by_gauge(gauges) == [1, 0, 2]
    assert lp.pick_gauge(gauges) == 1


def test_an_unscorable_candidate_is_listed_last_and_never_picked() -> None:
    """A dead probe row is visible, not hidden — and must never outrank a
    candidate we actually measured."""
    steered: dict[int, list[list[float]]] = {0: [[1.0, 0.0]], 2: [[-1.0, 0.0]]}
    gauges = lp.score_candidates(FAN, steered, base_mode="fan")
    by = {g.index: g for g in gauges}
    assert by[1].cell is None and by[1].note
    assert by[3].cell is None
    order = lp.rank_by_gauge(gauges)
    assert set(order[-2:]) == {1, 3}
    assert lp.pick_gauge(gauges) in (0, 2)


def test_reps_are_averaged_not_maximized() -> None:
    """cell_delta means: mean base − MEAN steered. One lucky rep must not
    carry a candidate."""
    lucky = lp.score_candidates(FAN, {0: [[1.0, 0.0], [0.0, -1.0]]},
                                base_mode="fan")[0]
    steady = lp.score_candidates(FAN, {0: [[1.0, 0.0], [1.0, 0.0]]},
                                 base_mode="fan")[0]
    assert lucky.cell is not None and steady.cell is not None
    assert steady.cell > lucky.cell
    assert lucky.n_reps_scored == 2


def test_the_probe_refuses_rather_than_falling_back_when_nothing_scored() -> None:
    """★ Silently serving another policy under gauge's name would pass static
    geometry off as the probe's measurement; the probe refuses instead."""
    gauges = lp.score_candidates(FAN, {}, base_mode="fan")
    assert all(g.cell is None for g in gauges)
    with pytest.raises(ValueError, match="refusing to pick"):
        lp.pick_gauge(gauges)


def test_fresh_base_requires_base_rows_and_is_absolute_comparable() -> None:
    with pytest.raises(ValueError, match="requires base_z"):
        lp.score_candidates(FAN, {0: [[1.0, 0.0]]}, base_mode="fresh")
    got = lp.score_candidates(FAN, {0: [[1.0, 0.0]]}, base_mode="fresh",
                              base_z=[[0.0, 1.0]])
    assert got[0].cell is not None


def test_score_candidates_rejects_an_unknown_base_mode() -> None:
    with pytest.raises(ValueError, match="base must be one of"):
        lp.score_candidates(FAN, {0: [[1.0, 0.0]]}, base_mode="vibes")


# ══ 5. the cost estimate — what the operator is shown before committing ════


def test_cost_grows_with_k_and_with_reps_and_is_in_seconds() -> None:
    a = lp.estimate_cost_s(8, 1)
    b = lp.estimate_cost_s(16, 1)
    c = lp.estimate_cost_s(8, 3)
    assert b["total_s"] > a["total_s"]
    assert c["total_s"] > a["total_s"]
    assert a["total_s"] == pytest.approx(a["generate_s"] + a["harvest_s"])
    assert a["n_spans_harvested"] == 8
    assert c["n_spans_harvested"] == 24


def test_reps_cost_harvest_but_not_generation() -> None:
    """The batching fact the design rests on: reps ride in one generate call
    per candidate, and cost a full span each at harvest — which is where 90%
    of a loom turn lives."""
    a, c = lp.estimate_cost_s(8, 1), lp.estimate_cost_s(8, 3)
    assert c["generate_s"] == pytest.approx(a["generate_s"])
    # exact on the span counts; the seconds are reported rounded to 0.1
    assert c["n_spans_harvested"] == 3 * a["n_spans_harvested"]
    assert c["harvest_s"] == pytest.approx(3 * a["harvest_s"], abs=0.15)


def test_a_fresh_base_costs_more_than_a_fan_base() -> None:
    fan = lp.estimate_cost_s(8, 1, base_mode="fan")
    fresh = lp.estimate_cost_s(8, 1, base_mode="fresh", n_base=3)
    assert fresh["total_s"] > fan["total_s"]
    assert fan["n_spans_harvested"] == 8
    assert fresh["n_spans_harvested"] == 11


def test_cost_estimate_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        lp.estimate_cost_s(1, 1)
    with pytest.raises(ValueError):
        lp.estimate_cost_s(8, 0)
    with pytest.raises(ValueError, match="base must be one of"):
        lp.estimate_cost_s(8, 1, base_mode="vibes")


# ══ 6. the wiring into pick_auto ════════════════════════════════════════════


def test_gauge_is_a_selectable_policy_and_the_tables_agree() -> None:
    assert "gauge" in ls.AUTO_POLICIES
    assert tuple(p["key"] for p in ls.AUTO_POLICY_INFO) == ls.AUTO_POLICIES
    assert ls.AUTO_POLICIES_UNAVAILABLE == ()


def test_pick_auto_gauge_picks_the_highest_measured_cell() -> None:
    scores: dict[int, dict[str, Any]] = {
        0: {"gauge": 0.10, "distinct": 0.9},
        1: {"gauge": 0.42, "distinct": 0.1},
        2: {"gauge": 0.30, "distinct": 0.5},
    }
    pick, effective = ls.pick_auto("gauge", scores, [0, 1, 2])
    assert (pick, effective) == (1, "gauge")
    # and it is genuinely a different judgment from the free policies
    assert ls.pick_auto("distinct", scores, [0, 1, 2])[0] == 0


def test_pick_auto_gauge_ignores_unscored_candidates() -> None:
    scores: dict[int, dict[str, Any]] = {
        0: {"gauge": None}, 1: {"gauge": 0.2}, 2: {},
    }
    assert ls.pick_auto("gauge", scores, [0, 1, 2])[0] == 1


def test_pick_auto_gauge_refuses_with_instructions_when_unprobed() -> None:
    """★ It must not quietly become `distinct` because nobody ran a probe."""
    scores: dict[int, dict[str, Any]] = {0: {"distinct": 0.9}, 1: {"distinct": 0.2}}
    with pytest.raises(ValueError) as exc:
        ls.pick_auto("gauge", scores, [0, 1])
    assert "/probe" in str(exc.value)
    assert "docs/FINDINGS.md §8" in str(exc.value)


def test_every_other_policy_is_byte_identical_in_behaviour() -> None:
    """Additive-only: adding `gauge` must not have moved any existing pick."""
    scores: dict[int, dict[str, Any]] = {
        0: {"distinct": 0.9, "loudness": 1.0, "camp": 0, "cos_to_worn": 0.8,
            "gauge": 0.01},
        1: {"distinct": 0.2, "loudness": 3.0, "camp": 1, "cos_to_worn": 0.1,
            "gauge": 0.99},
        2: {"distinct": 0.5, "loudness": 2.0, "camp": 0, "cos_to_worn": 0.4,
            "gauge": 0.50},
    }
    live = [0, 1, 2]
    assert ls.pick_auto("distinct", scores, live) == (0, "distinct")
    assert ls.pick_auto("loudest", scores, live) == (1, "loudest")
    assert ls.pick_auto("stay", scores, live) == (0, "stay")
    assert ls.pick_auto("swerve", scores, live) == (1, "swerve")


def test_an_unknown_policy_is_still_refused() -> None:
    with pytest.raises(ValueError, match="auto.policy must be one of"):
        ls.pick_auto("gauge-M2", {0: {"gauge": 0.1}}, [0])


# ══ 7. the honesty surface ══════════════════════════════════════════════════


def test_the_gauge_entry_carries_54s_share_53s_refusal_and_a_price() -> None:
    gauge = next(p for p in ls.AUTO_POLICY_INFO if p["key"] == "gauge")
    for token in ("docs/FINDINGS.md §8", "71%", "AUC bar", "uncertified", "43%"):
        assert token in gauge["status"], token
    assert "NOT FREE" in gauge["cost"]
    assert gauge["needs_probe"] is True


def test_probe_receipt_label_refuses_to_call_a_fan_base_a_delta_nr() -> None:
    """base='fan' is a ranking statistic. The module docstring says so; this
    pins that the distinction is machine-readable, not only prose."""
    assert lp.DEFAULT_BASE_MODE == "fan"
    assert "fresh" in lp.BASE_MODES
    assert not math.isnan(lp.PER_SPAN_HARVEST_S)
    assert lp.PER_SPAN_HARVEST_S > 0 and lp.PER_GEN_S > 0


# ══ 8. the resolution report — can the probe tell them apart? ══════════════


def test_resolution_is_unanswerable_at_one_rep_and_says_so() -> None:
    """★ With one reading per candidate there is no way to separate noise
    from signal. The report must refuse rather than report a number."""
    r = lp.resolution_report(lp.score_candidates(
        FAN, {i: [FAN[i]] for i in range(len(FAN))}, base_mode="fan"))
    assert r["resolved"] is None
    assert "reps>=2" in r["why"]


def test_resolution_detects_pure_noise_as_unresolved() -> None:
    """Candidates whose readings are indistinguishable must not be reported
    as an ordering the operator can trust."""
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    noisy = {i: [fan[(i + 1) % 4], fan[(i + 3) % 4], fan[i], fan[(i + 2) % 4]]
             for i in range(4)}
    r = lp.resolution_report(lp.score_candidates(fan, noisy, base_mode="fan"))
    assert r["resolved"] is False
    assert r["within_candidate_sd_nr"] > 0


def test_resolution_reports_resolved_when_candidates_are_consistent() -> None:
    """Every rep of a candidate agreeing = no within-candidate noise, so any
    between-candidate spread is real and the ordering IS resolved."""
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    clean = {i: [fan[i], fan[i], fan[i]] for i in range(4)}
    r = lp.resolution_report(lp.score_candidates(fan, clean, base_mode="fan"))
    assert r["resolved"] is True
    assert r["within_candidate_sd_nr"] == pytest.approx(0.0)
    # floored at 1: "0 readings" is not an answer to "how many do I need"
    assert r["reps_needed_to_resolve"] == 1


def test_a_fan_that_did_not_fork_is_named_as_such_not_blamed_on_noise() -> None:
    """★ Zero spread AND zero noise is not a noisy measurement — it is a fan
    with nothing in it to pick between. Saying "noise" there
    would send the operator to buy reps that cannot help."""
    g = [lp.CandidateGauge(index=0, cell=0.5, steered_nr=[0.1, 0.1]),
         lp.CandidateGauge(index=1, cell=0.5, steered_nr=[0.1, 0.1])]
    r = lp.resolution_report(g)
    assert r["resolved"] is False
    assert "did not fork" in r["why"]
    assert "docs/FINDINGS.md §11" in r["why"]
    assert r["reps_needed_to_resolve"] is None


def test_resolution_carries_the_fields_the_operator_needs() -> None:
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    mixed = {0: [fan[0], fan[0]], 1: [fan[1], fan[2]],
             2: [fan[2], fan[2]], 3: [fan[3], fan[0]]}
    r = lp.resolution_report(lp.score_candidates(fan, mixed, base_mode="fan"))
    for key in ("resolved", "reps", "within_candidate_sd_nr",
                "between_candidate_sd_cell", "estimated_signal_sd",
                "reps_needed_to_resolve", "why"):
        assert key in r, key
    assert r["reps"] == 2


# ══ 9. the /loom request guard — the default turn must stay free ═══════════


def test_request_probe_is_off_by_default_and_on_explicit_false() -> None:
    """★ The standing rule: no draw becomes expensive unless it was asked
    for. A free policy captures 43% of ORACLE picking skill
    (docs/FINDINGS.md section 8), so a silently-probed turn
    is a real cost imposed for no stated reason."""
    assert ls.request_probe({}) is None
    assert ls.request_probe({"probe": None}) is None
    assert ls.request_probe({"probe": False}) is None


def test_request_probe_accepts_true_and_an_empty_object_as_defaults() -> None:
    """★ `{}` is falsy in Python. A bare `if blob.get("probe")` would skip a
    request that plainly asked to probe with defaults, and the caller would
    then get gauge's "no probe scores" refusal instead of their probe."""
    assert ls.request_probe({"probe": True}) == {}
    assert ls.request_probe({"probe": {}}) == {}


def test_request_probe_passes_settings_through() -> None:
    cfg = {"reps": 3, "base": "fresh", "alpha": 0.4}
    assert ls.request_probe({"probe": cfg}) == cfg


def test_request_probe_refuses_a_wrong_type_loudly() -> None:
    """Same contract as request_detach_wear: a truthy string like "false"
    must not quietly cost the operator half a minute."""
    for bad in ("false", "true", 1, 0, 1.5, []):
        with pytest.raises(ValueError, match="probe must be omitted"):
            ls.request_probe({"probe": bad})


# ══ 10. regressions a correctness review found ══════════════════════════════


def test_resolution_excludes_one_rep_candidates_from_BOTH_terms() -> None:
    """★ REGRESSION. A candidate that survived one rep carries the full
    per-rep noise in its cell, but the correction only subtracts
    within^2/reps with reps>=2. Mixing them under-subtracts the noise and
    reports `resolved: true` on pure noise — simulated at ~62-75% for mixed
    rep counts before this fix. Both terms must use the same subset."""
    fan = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    steered = {0: [fan[0], fan[0]], 1: [fan[1], fan[1]],
               2: [fan[3]], 3: [fan[0]]}           # 2 and 3 have ONE rep
    g = lp.score_candidates(fan, steered, base_mode="fan")
    r = lp.resolution_report(g)
    assert r["n_candidates_used"] == 2
    assert r["n_candidates_excluded_for_reps"] == 2


def test_resolution_applies_a_margin_so_pure_noise_is_rarely_resolved() -> None:
    """★ REGRESSION. `between^2 > within^2/reps` is itself a noisy test and
    came out positive ~13% of the time on pure noise. A calibrated margin
    (Z=3.0) brings that down; this pins that the margin is actually applied
    by constructing a fan whose spread is exactly at the uncorrected bar."""
    import random
    rnd = random.Random(7)
    gs = []
    for i in range(8):
        reps = [rnd.gauss(0.5, 0.35) for _ in range(2)]
        m = sum(reps) / len(reps)
        gs.append(lp.CandidateGauge(index=i, cell=0.5 - m, steered_nr=reps,
                                    base_nr_mean=0.5, steered_nr_mean=m,
                                    n_reps_scored=2))
    r = lp.resolution_report(gs)
    # pure noise: no true signal exists, so this must NOT come back resolved
    assert r["resolved"] is False
    assert r["estimated_signal_sd"] == 0.0


def test_cost_estimate_publishes_n_base_so_a_client_can_reproduce_it() -> None:
    """The UI rescales the estimate for the k on screen; it needs n_base to
    do that correctly for a fresh base instead of assuming the default."""
    fan = lp.estimate_cost_s(8, 1, base_mode="fan")
    fresh = lp.estimate_cost_s(8, 1, base_mode="fresh", n_base=3)
    assert fan["n_base"] == 0
    assert fresh["n_base"] == 3
    assert fresh["n_spans_harvested"] == 8 + 3


def test_prefix_fingerprint_catches_a_moved_conversation() -> None:
    """★ REGRESSION. /probe must compare the whole prefix, not only the
    contemplated `text`: /chat, /undo, /edit, /truncate and /reroll all mutate
    the loom history WITHOUT clearing the fan — so the same text over a moved
    conversation would rank probe replies against futures of a different
    prefix."""
    h1 = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    h2 = h1 + [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    t = "the contemplated turn"
    assert ls.prefix_fingerprint(h1, t) == ls.prefix_fingerprint(list(h1), t)
    assert ls.prefix_fingerprint(h1, t) != ls.prefix_fingerprint(h2, t)
    assert ls.prefix_fingerprint(h1, t) != ls.prefix_fingerprint(h1, t + "!")
    # role and content must not be confusable by concatenation
    assert (ls.prefix_fingerprint([{"role": "user", "content": "ab"}], t)
            != ls.prefix_fingerprint([{"role": "usera", "content": "b"}], t))


def test_info_never_dies_on_a_k_the_estimator_refuses() -> None:
    """★ REGRESSION. /info is what the UI loads first and do_GET has NO
    exception handler, so anything that can raise in the payload takes the
    whole page down. --default-k is an unvalidated int."""
    blob = ls.build_info_payload(
        map_path="/m.npz", map_meta={}, sites=[8], branches=["base", "loom"],
        default_k=1, future_tokens=96, detach_wear_default=False,
        n_sessions=0, harvest_worker=None, restored_sessions=0,
        persistence={}, auto_policies=ls.AUTO_POLICY_INFO, loudness_ref=2.68,
        dose_band=None)
    assert blob["probe"]["cost_estimate"]["unavailable"] is True
    assert "k" in blob["probe"]["cost_estimate"]["why"]
