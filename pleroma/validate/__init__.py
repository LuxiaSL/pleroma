"""`pleroma validate` — the tool flags its own false results.

Known ways this pipeline produces a plausible wrong number, as stage gates.
Every gate takes ALREADY-PRODUCED artifacts (paths, JSON, arrays) — no GPU, no
network, no API calls — and returns a `GateResult`: PASS / FAIL / INCONCLUSIVE
with a named reason a newcomer cannot misread, and the numbers as labelled
evidence beside it.

Input policy (the same in every gate):

- an input that is ABSENT or unparseable as evidence → INCONCLUSIVE, naming it;
- the artifact UNDER TEST present but malformed (a map the loader would refuse,
  a band the server would refuse) → FAIL, since that is the finding;
- an unexpected exception inside a gate → INCONCLUSIVE naming it (`guarded`),
  never a traceback.

Each failure mode carries a short, stable code (`D-02`, `K-01`, ...) that a
gate's reason cites; each gate's docstring states its failure mode in words,
and each stage module's table lists the codes it guards.

Stages and gates:

    format   template, stops, pad-eos, empty-visitor
    fit      join, heldout, ceiling
    export   map-structure, map-load, export-report
    dose     band, band-span, damage
    eval     length-null, positive-control
    probe    resolution

Run: ``python -m pleroma.validate --profile profiles/llama31-8b-instruct.toml [--cv-report X]
[--map Y] [--band Z] ...`` (see ``--help``).
"""

from pleroma.validate.battery import BatteryInputs, run_battery
from pleroma.validate.result import BatteryReport, GateResult, Verdict

__all__ = ["BatteryInputs", "BatteryReport", "GateResult", "Verdict", "run_battery"]
