"""The build stages of the golden path, driven by a profile + a deployment.

    pleroma shelf  --profile P --deploy D [--out X] [--dry-run]   # the wide map
    pleroma fit    --profile P --deploy D [--out X] [--dry-run]   # registered CV
    pleroma export --profile P --deploy D [--out X] [--dry-run]   # the served map

Each stage renders the argv for its build module from the profile (operating
point: λ, rank, target, fan source) and the deployment's `[build]` block (the
paths). It runs the module in a child process of this interpreter and then
writes a RECEIPT beside the output. The receipt holds the exact command,
every input's sha256 (with any pin checked BEFORE running), the output's
sha256, the git commit, the times and the exit code, so no build output
exists without the command that made it.

Stage order for a new install: shelf -> fit (needs the shelf map as its
frozen baseline) -> export (needs the fit's report for its held-out figure;
it does not need the shelf map: the standardisers are computed in-build).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pleroma.config.profile import BuildInputs, Deployment, ModelProfile, PinnedPath

StageName = Literal["shelf", "fit", "export"]
STAGES: tuple[StageName, ...] = ("shelf", "fit", "export")

_MODULES: dict[StageName, str] = {
    "shelf": "pleroma.map.build.wide",
    "fit": "pleroma.map.build.cv",
    "export": "pleroma.map.build.export",
}


class StageError(ValueError):
    """The stage cannot be rendered or its inputs fail their pins."""


class InputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    path: str
    sha256: str | None
    pinned: str | None = None


class Receipt(BaseModel):
    """What ran, on what, producing what. Written as `<out>.receipt.json`."""

    model_config = ConfigDict(extra="forbid")
    stage: StageName
    profile: str
    module: str
    argv: list[str]
    inputs: list[InputRecord]
    output: str
    output_sha256: str | None = None
    git_commit: str | None = None
    started: str
    finished: str | None = None
    returncode: int | None = None
    notes: list[str] = Field(default_factory=list)


def sha256_path(path: Path) -> str | None:
    """sha256 of a file, or None for a directory or a missing path."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _need_build(deploy: Deployment) -> BuildInputs:
    if deploy.build is None:
        raise StageError("the deployment has no [build] block: pairs_dir, levers, "
                         "hiddens, bins and out_dir are where the build stages read "
                         "and write (add a [build] block to the deployment TOML)")
    return deploy.build


def default_output(stage: StageName, profile: ModelProfile, build: BuildInputs) -> Path:
    r = profile.map.rank
    return {
        "shelf": build.out_dir / f"shelf_map_{profile.name}_r{r}.npz",
        "fit": build.out_dir / f"cv_{profile.name}",
        "export": build.out_dir / f"loom_map_{profile.name}_v1a_r{r}.npz",
    }[stage]


def render(stage: StageName, profile: ModelProfile, deploy: Deployment,
           out: Path | None = None, device: str = "cuda",
           extra: list[str] | None = None,
           ) -> tuple[list[str], list[tuple[str, PinnedPath | Path]], Path]:
    """(module argv, the inputs to pin, the output path) for one stage."""
    b = _need_build(deploy)
    m = profile.map
    out = Path(out) if out is not None else default_output(stage, profile, b)
    hiddens = [str(h.path) for h in b.hiddens]
    inputs: list[tuple[str, PinnedPath | Path]] = [
        ("pairs_dir", b.pairs_dir), ("levers", b.levers), ("bins", b.bins),
        *[(f"hiddens[{i}]", h) for i, h in enumerate(b.hiddens)],
    ]
    common = ["--pairs-dir", str(b.pairs_dir), "--levers", str(b.levers.path),
              "--bins", str(b.bins.path)]
    if stage == "shelf":
        argv = [*common, "--discriminants", str(deploy.discriminants.path),
                "--norm-ref-bank", str(b.levers.path), "--lam", f"{m.lam:g}",
                "--rank", str(m.rank), "--out", str(out)]
        inputs = [i for i in inputs if not i[0].startswith("hiddens")]
        inputs.append(("discriminants", deploy.discriminants))
    elif stage == "fit":
        if b.shelf_map is None:
            raise StageError("fit needs [build.shelf_map], the registered CV's frozen "
                             "baseline: run `pleroma shelf` first and pin its output")
        argv = [*common, "--hiddens", *hiddens, "--shelf-map", str(b.shelf_map.path),
                "--fan-source", m.fan_source, "--device", device, "--out-dir", str(out)]
        inputs.append(("shelf_map", b.shelf_map))
    else:
        argv = [*common, "--hiddens", *hiddens,
                "--discriminants", str(deploy.discriminants.path),
                "--fan-source", m.fan_source, "--lam", f"{m.lam:g}", "--rank", str(m.rank),
                "--target", m.target, "--device", device, "--out", str(out)]
        inputs.append(("discriminants", deploy.discriminants))
        if b.cv_report is not None:
            argv += ["--heldout-report", str(b.cv_report.path)]
            inputs.append(("cv_report", b.cv_report))
    # `extra` is the escape hatch (e.g. an off-point export's --selfcheck-floor);
    # it lands in the receipt's argv like everything else
    return [sys.executable, "-u", "-m", _MODULES[stage], *argv, *(extra or [])], inputs, out


def pin_inputs(inputs: list[tuple[str, PinnedPath | Path]]) -> list[InputRecord]:
    """Hash every input and refuse on a pin mismatch or a missing path."""
    records: list[InputRecord] = []
    for role, item in inputs:
        path = item.path if isinstance(item, PinnedPath) else Path(item)
        pinned = item.sha256 if isinstance(item, PinnedPath) else None
        if not Path(path).exists():
            raise StageError(f"{role}: {path} does not exist")
        sha = sha256_path(path)
        if pinned is not None and sha != pinned:
            raise StageError(f"{role}: {path} has sha256 {sha}, the deployment pins "
                             f"{pinned} — refusing to build on a different input")
        records.append(InputRecord(role=role, path=str(path), sha256=sha, pinned=pinned))
    return records


def receipt_path(out: Path) -> Path:
    out = Path(out)
    return (out / "receipt.json") if out.suffix == "" else out.with_name(out.name + ".receipt.json")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_stage(stage: StageName, profile: ModelProfile, deploy: Deployment, *,
              out: Path | None = None, device: str = "cuda", dry_run: bool = False,
              extra: list[str] | None = None) -> int:
    """Render, pin, run, and write the receipt. Returns the child's exit code."""
    argv, inputs, out_path = render(stage, profile, deploy, out, device, extra)
    if dry_run:
        print(" ".join(argv))
        return 0
    records = pin_inputs(inputs)
    receipt = Receipt(stage=stage, profile=profile.name, module=_MODULES[stage], argv=argv,
                      inputs=records, output=str(out_path), git_commit=_git_commit(),
                      started=_now())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(argv, check=False).returncode
    receipt.finished = _now()
    receipt.returncode = rc
    if rc == 0:
        receipt.output_sha256 = sha256_path(out_path)
        if out_path.is_dir():
            report = out_path / "v1a_report.json"
            receipt.notes.append(f"report: {report} sha256 {sha256_path(report)}")
    rp = receipt_path(out_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2) + "\n")
    print(f"{stage}: rc={rc}; receipt {rp}")
    return rc
