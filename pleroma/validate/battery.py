"""Run every gate that has inputs; collect one `BatteryReport`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from pleroma.config import ModelProfile, load_profile
from pleroma.validate import dose, evals, export, fit, probe
from pleroma.validate import format as fmt
from pleroma.validate._io import sha256_file
from pleroma.validate.result import BatteryReport, GateResult, failed, inconclusive


class BatteryInputs(BaseModel):
    """Already-produced artifacts. Every field but `profile` is optional; a gate
    runs only when its inputs are present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Path
    tokenizer: Path | None = None
    cv_report: Path | None = None
    map: Path | None = None
    discriminants: Path | None = None
    export_report: Path | None = None
    band: Path | None = None
    replies: Path | None = None
    base_replies: Path | None = None
    length_null: Path | None = None
    positive_control: Path | None = None
    probe_receipt: Path | None = None

    def receipt(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if value is None:
                continue
            p = Path(value)
            out[name] = {"path": str(p),
                         "sha256": sha256_file(p) if p.is_file() else None}
        return out


def load_profile_gated(path: Path) -> tuple[ModelProfile | None, list[GateResult]]:
    """The profile, or a FAIL naming why it does not load. When the TOML parses
    but fails validation, the pad/eos check still runs on its raw values (a
    pad == eos profile is refused by validation — say which rule it broke)."""
    try:
        return load_profile(path), []
    except FileNotFoundError as exc:
        return None, [inconclusive("format", "profile", str(exc), path=str(path))]
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
                         for e in exc.errors()[:5])
        results = [failed("format", "profile", f"profile does not validate: {msgs}",
                          path=str(path), n_errors=exc.error_count())]
        try:
            raw = tomllib.loads(Path(path).read_text())
            arch = raw["model"]["arch"]
            results.append(fmt.check_pad_eos(arch.get("pad_token_id"),
                                             arch.get("eos_token_ids", []),
                                             raw.get("format", {}).get("stops", [])))
        except (KeyError, TypeError, OSError, tomllib.TOMLDecodeError, ValueError):
            pass
        return None, results
    except ValueError as exc:
        return None, [failed("format", "profile", f"profile unreadable: {exc}", path=str(path))]


def load_tokenizer(path: Path) -> tuple[Any | None, str | None]:
    """A LOCAL tokenizer (never downloads), or the named reason it did not load."""
    if not Path(path).exists():
        return None, f"tokenizer directory not found: {path}"
    try:
        from transformers import AutoTokenizer  # heavy; only when asked for
    except ImportError:
        return None, "transformers is not installed: cannot load --tokenizer"
    try:
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True), None
    except Exception as exc:  # noqa: BLE001 — any loader failure is the reason
        return None, f"tokenizer failed to load from {path}: {type(exc).__name__}: {exc}"


def run_battery(inputs: BatteryInputs) -> BatteryReport:
    profile, results = load_profile_gated(inputs.profile)
    report = BatteryReport(profile=profile.name if profile else None,
                           inputs=inputs.receipt(), results=list(results))
    if profile is None:
        return report

    tok, tok_err = (load_tokenizer(inputs.tokenizer) if inputs.tokenizer
                    else (None, None))
    fmt_results = fmt.run_format(profile, tok)
    if tok_err is not None:
        fmt_results = [inconclusive("format", "template", tok_err)
                       if r.gate == "template" and r.verdict != "PASS" else r
                       for r in fmt_results]
    report.results.extend(fmt_results)
    if inputs.cv_report is not None:
        report.results.extend(fit.run_fit(inputs.cv_report, profile))
    report.results.extend(export.run_export(profile, inputs.map, inputs.discriminants,
                                            inputs.export_report))
    report.results.extend(dose.run_dose(profile, inputs.band, inputs.map, inputs.replies,
                                        inputs.base_replies))
    report.results.extend(evals.run_eval(inputs.length_null, inputs.positive_control))
    report.results.extend(probe.run_probe(inputs.probe_receipt))
    return report
