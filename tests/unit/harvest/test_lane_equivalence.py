"""pleroma.harvest.lane_equivalence — ruled lane classes, and the one-lane check."""

from __future__ import annotations

import json

import pytest

from pleroma.harvest.lane_equivalence import (
    DEFAULT_REGISTRY,
    LaneEquivalence,
    LaneEquivalenceError,
    load_equivalence,
)

REF, NEW, OTHER = "lane-ref", "lane-new", "lane-other"


def _registry(**over) -> dict:
    cls = {"name": "model/lane-v1", "ruled_by": "test",
           "members": [{"lane_id": REF, "verdict": "reference"},
                       {"lane_id": NEW, "verdict": "EXACT", "evidence": "r.json"}]}
    cls.update(over)
    return {"classes": [cls]}


def test_members_of_one_class_are_one_lane() -> None:
    eq = LaneEquivalence.model_validate(_registry())
    assert eq.same_lane(REF, NEW) and eq.canonical(NEW) == "model/lane-v1"
    assert not eq.same_lane(REF, OTHER) and eq.canonical(OTHER) == OTHER
    assert eq.require_one_lane([REF, NEW, NEW]) == "model/lane-v1"
    with pytest.raises(LaneEquivalenceError, match="2 different lanes"):
        eq.require_one_lane([REF, OTHER])


def test_an_unregistered_single_lane_passes_as_itself() -> None:
    assert LaneEquivalence().require_one_lane([OTHER, OTHER]) == OTHER


@pytest.mark.parametrize("members, match", [
    ([{"lane_id": REF, "verdict": "EXACT", "evidence": "r"}], "exactly one reference"),
    ([{"lane_id": REF, "verdict": "reference"}, {"lane_id": NEW, "verdict": "PASS"}],
     "no evidence"),
    ([{"lane_id": REF, "verdict": "reference"}, {"lane_id": NEW, "verdict": "FAIL",
                                                 "evidence": "r"}], "verdict"),
])
def test_a_malformed_class_is_refused(members, match) -> None:
    with pytest.raises(ValueError, match=match):
        LaneEquivalence.model_validate(_registry(members=members))


def test_classes_must_be_disjoint() -> None:
    reg = _registry()
    reg["classes"].append({"name": "b", "ruled_by": "t",
                           "members": [{"lane_id": NEW, "verdict": "reference"}]})
    with pytest.raises(ValueError, match="in both"):
        LaneEquivalence.model_validate(reg)


def test_load_missing_is_empty_and_bad_is_refused(tmp_path) -> None:
    assert load_equivalence(tmp_path / "none.json").classes == ()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"classes": [{"name": "x"}]}))
    with pytest.raises(LaneEquivalenceError, match="unusable"):
        load_equivalence(bad)


def test_the_repo_registry_is_valid_and_its_evidence_exists() -> None:
    if not DEFAULT_REGISTRY.exists():
        pytest.skip("no registry in this checkout")
    eq = load_equivalence()
    root = DEFAULT_REGISTRY.parents[1]
    for cls in eq.classes:
        for m in cls.members:
            if m.evidence:
                assert (root / m.evidence).exists(), m.evidence
