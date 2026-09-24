"""Contract tests for the unworn-draw protocol (`detach_wear` on /loom).

Background: /loom has always drawn its K futures UNDER whatever is currently
worn, which means every draw after the first in a worn session reads
signatures of an already-bent forward pass. `detach_wear` is the opt-in
alternative: detach the steering for the draw only, keep the worn code intact
for scoring, re-attach it (nothing to re-attach, really — see below) for
everything after.

The GPU half of ``loom_serve`` (do_loom, attach, draw_batch, the model
itself) lives behind deferred imports inside ``main()``, same as the rest of
this module — see test_loom_persistence.py's header. The detach_wear DECISION
was deliberately kept out of that closure and pulled up to module level
(``resolve_draw_wear``, ``attach_for_draw``, ``request_detach_wear``,
``drawn_under_wear_field``, ``cos_to_worn``, ``pick_auto``) for exactly this
reason: the thing that decides whether a draw is steered must be testable on
a laptop with numpy + pytest, no model required. What is asserted here:

  * the steering is genuinely detached for the draw — `attach_for_draw` never
    calls its `attach_fn` at all when the wear is detached (no hook reaches
    the model, not a hook that happens to write zeros);
  * the worn code SURVIVES detaching — `resolve_draw_wear` never mutates or
    blanks `worn`, and `cos_to_worn` / `pick_auto` (stay, swerve) still work
    from it while the draw itself is unbent;
  * wear is restored (i.e. hooks are removed) after both a normal draw and
    one that raises — `attach_for_draw` is a strict try/finally;
  * `drawn_under_wear` / `request_detach_wear` report the truth in both
    modes, and refuse a body that sent something other than a boolean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pleroma.serve import legacy as ls


# ── fakes standing in for the model-attached hooks ──────────────────────────


class FakeHandle:
    """Stands in for the object attach_residual_write() returns — the only
    contract that matters here is `.remove()`."""

    def __init__(self, log: list[str], tag: str) -> None:
        self.log = log
        self.tag = tag
        self.removed = False

    def remove(self) -> None:
        self.removed = True
        self.log.append(f"removed:{self.tag}")


def make_attach_fn(log: list[str]) -> Any:
    """A fake attach() — records the call and hands back handles that log
    their own removal, exactly like the real per-site ResidualWriteSpec
    handles do."""

    def attach_fn(vectors: np.ndarray, alpha: float) -> list[FakeHandle]:
        log.append(f"attach:alpha={alpha}:sites={len(vectors)}")
        return [FakeHandle(log, f"site{i}") for i in range(len(vectors))]

    return attach_fn


def make_worn(alpha: float = 0.35, loom_id: str = "abc-000") -> dict[str, Any]:
    rng = np.random.default_rng(20260918)
    vectors = rng.normal(size=(3, 8))
    return {"index": 1, "alpha": alpha, "loom_id": loom_id, "vectors": vectors,
            "code": [0.1] * 8, "loom_dir": "/x/loom_000",
            "map_fingerprint": "fp0000000000abcd"}


# ── resolve_draw_wear: the decision itself ──────────────────────────────────


def test_nothing_worn_never_draws_under_wear() -> None:
    assert ls.resolve_draw_wear(None, detach_wear=False) is None
    assert ls.resolve_draw_wear(None, detach_wear=True) is None


def test_worn_and_not_detached_draws_under_the_same_object() -> None:
    worn = make_worn()
    draw_worn = ls.resolve_draw_wear(worn, detach_wear=False)
    assert draw_worn is worn  # identity: never copied, never mutated


def test_worn_and_detached_resolves_to_none_but_worn_is_untouched() -> None:
    worn = make_worn()
    before = dict(worn)
    draw_worn = ls.resolve_draw_wear(worn, detach_wear=True)
    assert draw_worn is None
    assert worn == before                     # nothing blanked, nothing popped
    assert worn["vectors"] is before["vectors"]  # same array object, not a copy


# ── attach_for_draw: the mechanism ───────────────────────────────────────────


def test_attach_for_draw_genuinely_detaches_no_hook_reaches_the_model() -> None:
    """detach_wear=True upstream (draw_worn=None): attach_fn must NEVER be
    called — the forward pass sees no hook, not a hook that writes zeros."""
    log: list[str] = []
    attach_fn = make_attach_fn(log)
    result = ls.attach_for_draw(None, attach_fn, lambda: "K-futures-drawn")
    assert result == "K-futures-drawn"
    assert log == []  # attach_fn never invoked


def test_attach_for_draw_attaches_when_not_detached() -> None:
    log: list[str] = []
    attach_fn = make_attach_fn(log)
    worn = make_worn(alpha=0.5)
    result = ls.attach_for_draw(worn, attach_fn, lambda: "drawn")
    assert result == "drawn"
    assert log[0] == "attach:alpha=0.5:sites=3"
    # every handle removed afterwards
    assert log[1:] == ["removed:site0", "removed:site1", "removed:site2"]


def test_wear_is_restored_after_a_normal_draw() -> None:
    log: list[str] = []
    handles_seen: list[FakeHandle] = []

    def attach_fn(vectors: np.ndarray, alpha: float) -> list[FakeHandle]:
        hs = [FakeHandle(log, f"s{i}") for i in range(len(vectors))]
        handles_seen.extend(hs)
        return hs

    ls.attach_for_draw(make_worn(), attach_fn, lambda: None)
    assert handles_seen and all(h.removed for h in handles_seen)


def test_wear_is_restored_after_a_draw_that_raises() -> None:
    """The exact requirement: a failed draw cannot silently leave a session
    unsteered — because the wear was never touched, and everything attached
    for the draw gets removed regardless of the exception."""
    log: list[str] = []
    handles_seen: list[FakeHandle] = []

    def attach_fn(vectors: np.ndarray, alpha: float) -> list[FakeHandle]:
        hs = [FakeHandle(log, f"s{i}") for i in range(len(vectors))]
        handles_seen.extend(hs)
        return hs

    def boom() -> None:
        raise RuntimeError("generate() blew up mid-draw")

    worn = make_worn()
    before = dict(worn)
    with pytest.raises(RuntimeError, match="blew up"):
        ls.attach_for_draw(worn, attach_fn, boom)
    assert handles_seen and all(h.removed for h in handles_seen)
    assert worn == before  # the session's wear itself was never at risk


def test_detached_draw_that_raises_still_calls_nothing_and_propagates() -> None:
    log: list[str] = []
    attach_fn = make_attach_fn(log)

    def boom() -> None:
        raise RuntimeError("blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        ls.attach_for_draw(None, attach_fn, boom)
    assert log == []  # nothing was ever attached, so there is nothing to undo


# ── the worn code survives: cos_to_worn / stay / swerve while detached ──────


def test_cos_to_worn_uses_the_retained_worn_code_not_the_draw_state() -> None:
    """cos_to_worn is always computed from `worn` (never `draw_worn`) in
    do_loom — this test proves the function's own contract: fed the SAME
    worn vectors, it gives the same answer whether or not a draw around it
    was detached (it never sees `detach_wear` at all)."""
    rng = np.random.default_rng(1)
    worn_vectors = rng.normal(size=(3, 8))
    lever = worn_vectors + rng.normal(scale=0.01, size=(3, 8))  # near-identical
    score = ls.cos_to_worn(lever, worn_vectors)
    assert 0.9 < score <= 1.0
    # Detaching a draw never touches `worn`, so a session that ran with
    # detach_wear=True feeds cos_to_worn the exact same worn_vectors it
    # would have under wear.
    detached_worn = ls.resolve_draw_wear(
        {"vectors": worn_vectors, "alpha": 0.35}, detach_wear=True
    )
    assert detached_worn is None  # nothing attached for the draw...
    same_score = ls.cos_to_worn(lever, worn_vectors)  # ...but scoring is untouched
    assert same_score == score


def _scored(cands: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return cands


def test_stay_and_swerve_still_pick_correctly_while_detached() -> None:
    """Simulates what do_loom's scoring loop produces: cos_to_worn scores
    computed from the retained `worn` object, regardless of whether the draw
    that produced these candidate levers was itself detached. stay/swerve
    must still pick the max/min exactly as under wear."""
    scores = _scored({
        0: {"cos_to_worn": 0.92, "distinct": 0.10},
        1: {"cos_to_worn": 0.40, "distinct": 0.55},
        2: {"cos_to_worn": 0.71, "distinct": 0.30},
    })
    live = [0, 1, 2]
    stay_pick, stay_eff = ls.pick_auto("stay", scores, live)
    assert (stay_pick, stay_eff) == (0, "stay")
    swerve_pick, swerve_eff = ls.pick_auto("swerve", scores, live)
    assert (swerve_pick, swerve_eff) == (1, "swerve")


def test_stay_swerve_raise_when_nothing_is_scored_cos_to_worn() -> None:
    """If nothing was worn at all (so no candidate carries cos_to_worn),
    stay/swerve must refuse — same whether or not detach_wear was set,
    since detach_wear only matters when something IS worn."""
    scores = {0: {"distinct": 0.1}, 1: {"distinct": 0.2}}
    with pytest.raises(ValueError, match="needs score 'cos_to_worn'"):
        ls.pick_auto("stay", scores, [0, 1])


def test_pick_auto_unknown_policy_and_empty_live_raise() -> None:
    with pytest.raises(ValueError, match="auto.policy must be one of"):
        ls.pick_auto("nonsense", {0: {}}, [0])
    with pytest.raises(ValueError, match="no harvested future"):
        ls.pick_auto("distinct", {0: {"distinct": 0.1}}, [])


def test_pick_auto_refuses_the_retired_minority_policy() -> None:
    """`minority` consumed the 2-means camps, which the server does not serve;
    asking for it is refused."""
    scores = {0: {"distinct": 0.4}, 1: {"distinct": 0.7}}
    with pytest.raises(ValueError):
        ls.pick_auto("minority", scores, [0, 1])
    assert "minority" not in ls.AUTO_POLICIES


# ── drawn_under_wear: reports the truth ──────────────────────────────────────


def test_drawn_under_wear_is_none_when_nothing_worn() -> None:
    assert ls.drawn_under_wear_field(None) is None


def test_drawn_under_wear_reports_the_wear_when_attached() -> None:
    worn = make_worn(alpha=0.4, loom_id="xyz-002")
    field = ls.drawn_under_wear_field(worn)
    assert field == {"index": 1, "alpha": 0.4, "loom_id": "xyz-002"}


def test_drawn_under_wear_is_false_ish_when_detached_even_though_worn() -> None:
    """The exact honesty requirement: something IS worn, but this draw did
    not use it — drawn_under_wear must not claim otherwise."""
    worn = make_worn()
    draw_worn = ls.resolve_draw_wear(worn, detach_wear=True)
    field = ls.drawn_under_wear_field(draw_worn)
    assert not field  # None — falsy, never the worn metadata
    # meanwhile the session's actual worn state is unaffected and still true:
    assert worn["index"] == 1 and worn["alpha"] == pytest.approx(0.35)


@pytest.mark.parametrize("detach_wear", [False, True])
def test_drawn_under_wear_end_to_end_for_both_modes(detach_wear: bool) -> None:
    """One assembly, both modes: the field always matches what attach_for_draw
    actually attached, never merely what is worn."""
    log: list[str] = []
    attach_fn = make_attach_fn(log)
    worn = make_worn()
    draw_worn = ls.resolve_draw_wear(worn, detach_wear=detach_wear)
    ls.attach_for_draw(draw_worn, attach_fn, lambda: None)
    field = ls.drawn_under_wear_field(draw_worn)
    if detach_wear:
        assert field is None
        assert log == []
    else:
        assert field is not None and field["index"] == worn["index"]
        assert log and log[0].startswith("attach:")


# ── request_detach_wear: the per-request / server-default resolution ────────


def test_request_detach_wear_defaults_to_server_flag_when_body_omits_it() -> None:
    assert ls.request_detach_wear({}, server_default=False) is False
    assert ls.request_detach_wear({}, server_default=True) is True


def test_request_detach_wear_per_request_overrides_server_default() -> None:
    assert ls.request_detach_wear({"detach_wear": True}, server_default=False) is True
    assert ls.request_detach_wear({"detach_wear": False}, server_default=True) is False


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, 1.0, [], {}, "yes"])
def test_request_detach_wear_refuses_non_boolean(bad: Any) -> None:
    with pytest.raises(ValueError, match="detach_wear must be a boolean"):
        ls.request_detach_wear({"detach_wear": bad}, server_default=False)


# ── LoomSnapshot persists detach_wear honestly ──────────────────────────────


def test_loom_snapshot_round_trips_detach_wear(tmp_path: Any) -> None:
    store = ls.SessionStore(tmp_path)
    assert store.ensure_dir()
    sess = ls.LoomSession()
    sess.looms = [
        ls.LoomSnapshot(loom_id="s-000", k=6, horizon=192, created_at=1.0,
                        detach_wear=True),
        ls.LoomSnapshot(loom_id="s-001", k=6, horizon=192, created_at=2.0,
                        detach_wear=False),
    ]
    store.save("s", sess)
    back = store.restore_one("s", ls.State())
    assert back.looms[0].detach_wear is True
    assert back.looms[1].detach_wear is False


def test_loom_snapshot_predating_detach_wear_reads_as_false() -> None:
    """An older snapshot with no 'detach_wear' key at all can only mean the
    one behaviour that existed before this field did: drawn under wear."""
    snap = ls.LoomSnapshot.from_json(
        {"loom_id": "old-000", "k": 6, "horizon": 192, "created_at": 1.0}
    )
    assert snap.detach_wear is False
