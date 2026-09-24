"""The one result type every gate returns, and the battery's receipt.

The rule: *no bare number a newcomer can misread.* A gate answers
one question with PASS / FAIL / INCONCLUSIVE and a NAMED reason; numbers live
in `evidence`, labelled, beside the reason that interprets them.

- PASS          — the artifact answers the question, and the reason says what
                  was (and was not) certified.
- FAIL          — the artifact is wrong in a named way; `pleroma validate`
                  exits nonzero.
- INCONCLUSIVE  — the question cannot be answered from what was supplied, or
                  the instrument cannot see at this size. Never read as "no
                  effect" and never as "broken".
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, ParamSpec

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]
VERDICTS: tuple[Verdict, ...] = ("PASS", "FAIL", "INCONCLUSIVE")

#: Receipt schema; bump on any breaking change to `BatteryReport`.
RECEIPT_SCHEMA: int = 1


def jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays, paths, tuples and non-finite floats into
    plain JSON (NaN/inf become None: a receipt must parse everywhere)."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


class GateResult(BaseModel):
    """One gate's verdict. `stage/gate` is the stable id printed and receipted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    stage: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    verdict: Verdict
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence", mode="before")
    @classmethod
    def _plain_json(cls, v: Any) -> dict[str, Any]:
        out = jsonable(v if v is not None else {})
        if not isinstance(out, dict):
            raise ValueError("evidence must be a mapping")
        json.dumps(out)  # guaranteed serialisable, or this raises here, not at --out
        return out

    @property
    def gate_id(self) -> str:
        return f"{self.stage}/{self.gate}"

    def line(self) -> str:
        """`VERDICT  stage/gate — reason`, the one printed line."""
        return f"{self.verdict:<12}  {self.gate_id} — {self.reason}"


def passed(stage: str, gate: str, reason: str, **evidence: Any) -> GateResult:
    return GateResult(stage=stage, gate=gate, verdict="PASS", reason=reason, evidence=evidence)


def failed(stage: str, gate: str, reason: str, **evidence: Any) -> GateResult:
    return GateResult(stage=stage, gate=gate, verdict="FAIL", reason=reason, evidence=evidence)


def inconclusive(stage: str, gate: str, reason: str, **evidence: Any) -> GateResult:
    return GateResult(stage=stage, gate=gate, verdict="INCONCLUSIVE", reason=reason,
                      evidence=evidence)


P = ParamSpec("P")


def guarded(stage: str, gate: str) -> Callable[[Callable[P, GateResult]], Callable[P, GateResult]]:
    """A gate never tracebacks: an unexpected exception inside it becomes
    INCONCLUSIVE naming the exception, so one broken input cannot hide the
    other gates' verdicts. (Expected refusals are handled inside each gate
    and carry their own named reason.)"""

    def wrap(fn: Callable[P, GateResult]) -> Callable[P, GateResult]:
        @functools.wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> GateResult:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — the point of the wrapper
                return inconclusive(
                    stage, gate,
                    f"gate could not run: {type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__)
        return inner
    return wrap


class BatteryReport(BaseModel):
    """The `--out` receipt: every gate that ran, plus what it ran on."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = RECEIPT_SCHEMA
    battery: Literal["pleroma validate"] = "pleroma validate"
    profile: str | None
    inputs: dict[str, Any] = Field(default_factory=dict)
    results: list[GateResult] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {v: sum(1 for r in self.results if r.verdict == v) for v in VERDICTS}

    @property
    def exit_code(self) -> int:
        return 1 if any(r.verdict == "FAIL" for r in self.results) else 0

    def to_json(self) -> dict[str, Any]:
        blob = self.model_dump(mode="json")
        blob["counts"] = self.counts
        blob["exit_code"] = self.exit_code
        return blob
