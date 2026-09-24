"""Which harvest lane_ids count as ONE lane: a ruled, evidence-backed registry.

A ``lane_id`` hashes the lane's source files, so any change to the harvest code
(a new anamnesis pin, a move of the lane's source tree) mints a new id even when
every feature comes out bit-identical. Rows from different lane_ids must not
feed one map UNLESS someone measured that the lanes agree and ruled them the
same lane. This registry is that record: each class names its members, and each
member carries the parity report that admitted it and who ruled it.

    from pleroma.harvest.lane_equivalence import load_equivalence
    eq = load_equivalence()                 # reads DEFAULT_REGISTRY
    eq.same_lane(a, b)                      # True for the same id or one class
    eq.canonical(lane_id)                   # the class name, or the id itself

A missing registry file means no equivalences: every lane_id is its own lane.
A member may be admitted only on an ``EXACT`` or ``PASS`` parity verdict.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "profiles" / "lane_equivalence.json"


class LaneEquivalenceError(ValueError):
    """The registry is malformed, or lanes that were never ruled equal were mixed."""


class LaneMember(BaseModel):
    """One lane_id admitted to a class, and the evidence it was admitted on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_id: str = Field(min_length=1)
    anamnesis_commit: str | None = None
    #: the parity report (repo-relative) that compared this lane to the class's
    #: reference member; None only for the reference member itself
    evidence: str | None = None
    verdict: Literal["reference", "EXACT", "PASS"]
    note: str = ""


class LaneClass(BaseModel):
    """lane_ids ruled to be one lane semantically."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    ruled_by: str = Field(min_length=1)
    note: str = ""
    members: tuple[LaneMember, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_reference_and_evidence(self) -> LaneClass:
        refs = [m for m in self.members if m.verdict == "reference"]
        if len(refs) != 1:
            raise ValueError(f"lane class {self.name!r} needs exactly one reference member, "
                             f"has {len(refs)}")
        missing = [m.lane_id for m in self.members if m.verdict != "reference" and not m.evidence]
        if missing:
            raise ValueError(f"lane class {self.name!r}: members {missing} carry a parity "
                             "verdict but no evidence path")
        return self


class LaneEquivalence(BaseModel):
    """The registry: disjoint classes of equivalent lane_ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classes: tuple[LaneClass, ...] = ()

    @model_validator(mode="after")
    def _disjoint(self) -> LaneEquivalence:
        seen: dict[str, str] = {}
        for cls in self.classes:
            for m in cls.members:
                if m.lane_id in seen:
                    raise ValueError(f"lane_id {m.lane_id} is in both {seen[m.lane_id]!r} "
                                     f"and {cls.name!r}")
                seen[m.lane_id] = cls.name
        return self

    def class_of(self, lane_id: str) -> LaneClass | None:
        for cls in self.classes:
            if any(m.lane_id == lane_id for m in cls.members):
                return cls
        return None

    def canonical(self, lane_id: str) -> str:
        """The class name for a registered lane_id; the lane_id itself otherwise."""
        cls = self.class_of(lane_id)
        return cls.name if cls is not None else lane_id

    def same_lane(self, a: str, b: str) -> bool:
        return a == b or self.canonical(a) == self.canonical(b)

    def require_one_lane(self, lane_ids: Iterable[str], *, what: str = "these rows") -> str:
        """The one canonical lane ``lane_ids`` share, or raise naming them."""
        ids = sorted({str(x) for x in lane_ids})
        canon = sorted({self.canonical(x) for x in ids})
        if len(canon) != 1:
            raise LaneEquivalenceError(
                f"{what} come from {len(canon)} different lanes {canon} (lane_ids {ids}). "
                "Everything feeding one map must come from one lane; lanes measured to "
                "agree can be ruled equivalent in the lane-equivalence registry "
                "(pleroma.harvest.lane_equivalence.DEFAULT_REGISTRY).")
        return canon[0]


def load_equivalence(path: Path | str | None = None) -> LaneEquivalence:
    """Read the registry; a missing file is an empty registry (no equivalences)."""
    p = Path(path) if path is not None else DEFAULT_REGISTRY
    if not p.exists():
        return LaneEquivalence()
    try:
        return LaneEquivalence.model_validate(json.loads(p.read_text()))
    except (ValueError, TypeError) as exc:
        raise LaneEquivalenceError(f"unusable lane equivalence registry {p}: {exc}") from exc
