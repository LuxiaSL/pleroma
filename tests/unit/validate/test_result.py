"""GateResult / BatteryReport: typed, JSON-safe, never a traceback."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from pleroma.validate.result import (
    BatteryReport,
    GateResult,
    failed,
    guarded,
    inconclusive,
    passed,
)


def test_line_format() -> None:
    r = failed("dose", "band-span", "band measured under continuation, served span uniform")
    assert r.line() == ("FAIL          dose/band-span — band measured under continuation, "
                        "served span uniform")


def test_evidence_is_coerced_to_plain_json() -> None:
    r = passed("fit", "heldout", "ok", a=np.float32(0.5), b=np.arange(3), c=float("nan"),
               d=(1, 2), e=np.bool_(True))
    assert r.evidence == {"a": 0.5, "b": [0, 1, 2], "c": None, "d": [1, 2], "e": True}
    json.dumps(r.model_dump())


def test_verdict_is_a_closed_set_and_reason_required() -> None:
    with pytest.raises(ValidationError):
        GateResult(gate="x", stage="y", verdict="MAYBE", reason="r")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GateResult(gate="x", stage="y", verdict="PASS", reason="")


def test_guarded_turns_a_crash_into_a_named_inconclusive() -> None:
    @guarded("probe", "resolution")
    def boom() -> GateResult:
        raise RuntimeError("disk on fire")

    r = boom()
    assert r.verdict == "INCONCLUSIVE"
    assert "RuntimeError: disk on fire" in r.reason


def test_exit_code_only_on_fail() -> None:
    rep = BatteryReport(profile="p", results=[passed("a", "b", "ok"),
                                              inconclusive("a", "c", "cannot tell")])
    assert rep.exit_code == 0
    rep.results.append(failed("a", "d", "broken"))
    assert rep.exit_code == 1
    blob = rep.to_json()
    assert blob["counts"] == {"PASS": 1, "FAIL": 1, "INCONCLUSIVE": 1}
    assert blob["exit_code"] == 1
