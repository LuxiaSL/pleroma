"""`python -m pleroma` — the golden-path entry point.

    python -m pleroma serve  --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml
    python -m pleroma worker --profile ... --deploy ... [--dry-run]
    python -m pleroma profile show profiles/llama31-8b-instruct.toml
    python -m pleroma shelf|fit|export --profile ... --deploy ... [--out X] [--dry-run]
    python -m pleroma validate --profile ... [--cv-report R] [--map M] [--band B] ...

`serve` and `worker` launch the server/worker with EVERY value taken from
the profile + deployment (pleroma.config.legacy renders the argv;
`tests/unit/test_cli.py` pins the rendered command). `--dry-run` prints the
exact command instead of running it, so every launch can be recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

from pleroma.config import load_deployment, load_profile
from pleroma.config.legacy import legacy_env, legacy_serve_argv, legacy_worker_argv
from pleroma.stages import STAGES, StageError, run_stage


def _load_pair(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    try:
        return load_profile(args.profile), load_deployment(args.deploy)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise SystemExit(f"pleroma: {exc}") from exc


def _launch(argv: list[str], dry_run: bool, env: dict[str, str] | None = None) -> int:
    cmd = [sys.executable, "-u", *argv]
    extra = dict(env or {})
    if dry_run:
        prefix = [f"{k}={v}" for k, v in sorted(extra.items())]
        print(" ".join([*(shlex.quote(p) for p in prefix), shlex.join(cmd)]))
        return 0
    os.execvpe(cmd[0], cmd, {**os.environ, **extra})  # replaces this process
    return 0  # pragma: no cover


def _cmd_serve(args: argparse.Namespace) -> int:
    profile, deploy = _load_pair(args)
    try:
        argv = legacy_serve_argv(profile, deploy)
    except ValueError as exc:
        raise SystemExit(f"pleroma serve: {exc}") from exc
    return _launch(argv, args.dry_run, legacy_env(profile))


def _cmd_worker(args: argparse.Namespace) -> int:
    profile, deploy = _load_pair(args)
    try:
        argv = legacy_worker_argv(profile, deploy)
    except ValueError as exc:
        raise SystemExit(f"pleroma worker: {exc}") from exc
    return _launch(argv, args.dry_run, legacy_env(profile))


def _cmd_stage(args: argparse.Namespace) -> int:
    profile, deploy = _load_pair(args)
    try:
        return run_stage(args.command, profile, deploy, out=args.out,
                         device=args.device, dry_run=args.dry_run,
                         extra=[a for a in args.extra if a != "--"])
    except StageError as exc:
        raise SystemExit(f"pleroma {args.command}: {exc}") from exc


def _cmd_profile_show(args: argparse.Namespace) -> int:
    try:
        profile = load_profile(args.path)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise SystemExit(f"pleroma: {exc}") from exc
    print(json.dumps(profile.model_dump(mode="json"), indent=2))
    return 0


def _validate_argv(argv: list[str]) -> list[str]:
    """`pleroma validate --deploy D`: fill the battery's inputs from the
    deployment (map, discriminants, band, CV report) unless given explicitly."""
    if "--deploy" not in argv:
        return argv
    i = argv.index("--deploy")
    if i + 1 >= len(argv):
        raise SystemExit("pleroma validate: --deploy needs a path")
    try:
        deploy = load_deployment(argv[i + 1])
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise SystemExit(f"pleroma: {exc}") from exc
    rest = argv[:i] + argv[i + 2:]
    fill: list[tuple[str, Path | None]] = [
        ("--map", deploy.map.path),
        ("--discriminants", deploy.discriminants.path),
        ("--band", deploy.dose_band.path if deploy.dose_band else None),
        ("--cv-report", deploy.build.cv_report.path
         if deploy.build and deploy.build.cv_report else None),
    ]
    if "--profile" not in rest:
        # the deployment names its profile: the repo's profiles/<name>.toml,
        # else a profiles/ dir beside the deployment's own directory
        name = f"{deploy.profile}.toml"
        candidates = [Path(__file__).resolve().parents[1] / "profiles" / name,
                      Path(argv[i + 1]).resolve().parent.parent / "profiles" / name]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise SystemExit(f"pleroma validate: the deployment names profile "
                             f"{deploy.profile!r}, found at none of "
                             f"{[str(c) for c in candidates]}; pass --profile")
        fill.insert(0, ("--profile", found))
    for flag, path in fill:
        if path is not None and flag not in rest:
            rest += [flag, str(path)]
    return rest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pleroma", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, what in (("serve", _cmd_serve, "the loom server"),
                           ("worker", _cmd_worker, "the harvest worker")):
        p = sub.add_parser(name, help=f"launch {what} from a profile + deployment")
        p.add_argument("--profile", type=Path, required=True, help="profiles/<model>.toml")
        p.add_argument("--deploy", type=Path, required=True, help="this install's deploy.toml")
        p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
        p.set_defaults(fn=fn)
    for name, what in zip(STAGES, ("fit the shelf (wide) map: the CV's frozen baseline",
                                   "the registered CV (held-out retrieval vs the shelf)",
                                   "export the served map (stats computed in-build)")):
        p = sub.add_parser(name, help=what)
        p.add_argument("--profile", type=Path, required=True, help="profiles/<model>.toml")
        p.add_argument("--deploy", type=Path, required=True,
                       help="this install's deploy.toml (needs a [build] block)")
        p.add_argument("--out", type=Path, default=None,
                       help="output path (default: [build].out_dir / a profile-named file)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
        p.add_argument("extra", nargs=argparse.REMAINDER,
                       help="after `--`: extra flags passed to the build module verbatim "
                            "(recorded in the receipt)")
        p.set_defaults(fn=_cmd_stage)
    prof = sub.add_parser("profile", help="inspect a profile")
    psub = prof.add_subparsers(dest="profile_command", required=True)
    show = psub.add_parser("show", help="validate a profile and print it resolved")
    show.add_argument("path", type=Path)
    show.set_defaults(fn=_cmd_profile_show)
    # `validate` owns its own parser (pleroma.validate.__main__); registered
    # here for --help, dispatched in main() before this parser runs.
    sub.add_parser("validate", add_help=False,
                   help="stage gates: PASS / FAIL / INCONCLUSIVE with a named reason")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["validate"]:
        from pleroma.validate.__main__ import main as validate_main
        return int(validate_main(_validate_argv(argv[1:])))
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
