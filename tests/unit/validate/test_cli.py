"""`python -m pleroma.validate`: one line per gate, a receipt, exit 1 only on FAIL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pleroma.validate.__main__ import main

from .conftest import TOY, write_json

ROOT = Path(__file__).resolve().parents[3]


def toml_profile(tmp_path: Path, **arch: object) -> Path:
    import copy

    blob = copy.deepcopy(TOY)
    blob["model"]["arch"].update(arch)

    def val(v: object) -> str:
        return json.dumps(v)  # TOML-compatible for str/int/float/list-of-scalars

    lines: list[str] = []

    def emit(prefix: str, d: dict) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        lines.extend(f"{k} = {val(v)}" for k, v in scalars.items() if v is not None)
        for k, v in d.items():
            if isinstance(v, dict):
                emit(f"{prefix}.{k}" if prefix else k, v)

    emit("", blob)
    p = tmp_path / "toy.toml"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_clean_profile_exits_zero_and_writes_a_receipt(tmp_path: Path, capsys) -> None:
    out = tmp_path / "receipt.json"
    assert main(["--profile", str(toml_profile(tmp_path)), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("PASS") and "format/template — " in line for line in lines)
    blob = json.loads(out.read_text())
    assert blob["battery"] == "pleroma validate" and blob["exit_code"] == 0
    assert blob["inputs"]["profile"]["sha256"]
    assert {r["gate"] for r in blob["results"]} == {"template", "stops", "pad-eos",
                                                    "empty-visitor"}


def test_a_fail_exits_nonzero(tmp_path: Path, capsys) -> None:
    band = write_json(tmp_path / "b.json", {"tier": "measured", "zones": []})
    rc = main(["--profile", str(toml_profile(tmp_path)), "--band", str(band)])
    assert rc == 1
    assert "FAIL          dose/band — dose band is malformed" in capsys.readouterr().out


def test_pad_equal_eos_profile_is_named_not_a_traceback(tmp_path: Path, capsys) -> None:
    rc = main(["--profile", str(toml_profile(tmp_path, pad_token_id=2))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL          format/profile — profile does not validate" in out
    assert "FAIL          format/pad-eos — pad id 2 is also an eos id" in out


def test_missing_inputs_are_inconclusive_not_tracebacks(tmp_path: Path, capsys) -> None:
    rc = main(["--profile", str(toml_profile(tmp_path)),
               "--cv-report", str(tmp_path / "nope.json"),
               "--map", str(tmp_path / "nope.npz"),
               "--length-null", str(tmp_path / "nope2.json"),
               "--probe-receipt", str(tmp_path / "nope3.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Traceback" not in out
    assert "INCONCLUSIVE  fit/heldout — CV report not found" in out
    assert "INCONCLUSIVE  export/map-structure — map file not found" in out


def test_module_entry_point_runs(tmp_path: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "pleroma.validate", "--profile",
                           str(ROOT / "tests" / "fixtures" / "profiles" / "modelc-format-70b.toml")],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "PASS          format/stops" in proc.stdout
