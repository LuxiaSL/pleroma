"""`python -m pleroma.validate` — run every gate that has inputs.

    python -m pleroma.validate --profile profiles/llama31-8b-instruct.toml \\
        --cv-report <cv-out>/v1a_report.json \\
        --band <map>.npz.doseband.json \\
        --out validate-8b.json

Prints one line per gate, ``VERDICT  stage/gate — reason``, then a count line.
Exit status: 1 if any gate FAILs, else 0 (INCONCLUSIVE is not a failure — it
is a statement that the question cannot be answered from these inputs).
`main(argv)` is importable so `pleroma validate` can delegate to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pleroma.validate.battery import BatteryInputs, run_battery


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pleroma validate",
        description="Stage gates from the failure catalog. Every gate reads already-"
                    "produced artifacts (no GPU, no network) and prints PASS / FAIL / "
                    "INCONCLUSIVE with a named reason.")
    ap.add_argument("--profile", type=Path, required=True, help="ModelProfile TOML")
    g = ap.add_argument_group("format")
    g.add_argument("--tokenizer", type=Path,
                   help="LOCAL tokenizer dir (chat mode's template check; never downloads)")
    g = ap.add_argument_group("fit")
    g.add_argument("--cv-report", type=Path, help="v1a_fit report (v1a_report.json)")
    g = ap.add_argument_group("export")
    g.add_argument("--map", type=Path, help="loom map npz")
    g.add_argument("--discriminants", type=Path,
                   help="the map's discriminants npz (enables the full LoomMap load check)")
    g.add_argument("--export-report", type=Path, help="v1a_export report json")
    g = ap.add_argument_group("dose")
    g.add_argument("--band", type=Path,
                   help="dose band json (default with --map: <map>.doseband.json)")
    g.add_argument("--replies", type=Path,
                   help="steered replies (JSON list / {'replies': [...]} / JSONL) for the "
                        "damage detector")
    g.add_argument("--base-replies", type=Path,
                   help="unsteered replies, same format (attributes loops to the dose or not)")
    g = ap.add_argument_group("eval")
    g.add_argument("--length-null", type=Path,
                   help="judged + length-only per-conversation nr json (see pleroma.validate.evals)")
    g.add_argument("--positive-control", type=Path,
                   help="judge scores on a known-positive control + its own floor arm")
    g = ap.add_argument_group("probe")
    g.add_argument("--probe-receipt", type=Path, help="a /probe receipt (with 'resolution')")
    ap.add_argument("--out", type=Path, help="write the JSON receipt here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fields = {k: v for k, v in vars(args).items() if k != "out"}
    report = run_battery(BatteryInputs(**fields))
    for r in report.results:
        print(r.line())
    c = report.counts
    print(f"-- {c['PASS']} PASS, {c['FAIL']} FAIL, {c['INCONCLUSIVE']} INCONCLUSIVE"
          f" (profile {report.profile or 'NOT LOADED'})")
    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report.to_json(), indent=2) + "\n")
            print(f"receipt: {args.out}")
        except OSError as exc:
            print(f"pleroma validate: could not write receipt {args.out}: {exc}",
                  file=sys.stderr)
            return 2
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
