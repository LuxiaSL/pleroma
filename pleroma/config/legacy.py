"""Profile + Deployment -> argv (and env) for the LEGACY entry points.

A profile launches the flag-driven entry points `pleroma.serve.legacy` and
`pleroma.harvest.worker` directly, with the profile's anamnesis registry file
(if any) exported on `ANAMNESIS_MODELS` (`legacy_env`). That is how a preset
anamnesis does not ship resolves in the launched process AND its children: an
environment variable is inherited, an in-process registration is not. Every
value the legacy CLI would otherwise take from its own defaults is passed
EXPLICITLY, so no stage falls back to a code default that disagrees with the
profile.
"""

from __future__ import annotations

from collections.abc import Mapping

from pleroma.config.anamnesis_registry import launch_env
from pleroma.config.profile import Deployment, ModelProfile, PromptMode


def _model_path(profile: ModelProfile, deploy: Deployment) -> str:
    return str(deploy.model_path) if deploy.model_path else profile.model.model_id


def _check_pair(profile: ModelProfile, deploy: Deployment) -> None:
    if deploy.profile != profile.name:
        raise ValueError(
            f"deployment is for profile {deploy.profile!r}, not {profile.name!r}")


def _module(profile: ModelProfile, role: str) -> list[str]:
    del profile  # every preset resolves through the registry (legacy_env)
    return ["-m", "pleroma.serve.legacy" if role == "serve"
            else "pleroma.harvest.worker"]


def legacy_env(profile: ModelProfile, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment ADDITIONS for a launched server/worker: the profile's
    anamnesis registry on ``ANAMNESIS_MODELS`` (empty when it names none)."""
    return launch_env(profile.model.anamnesis_registry, base)


def legacy_serve_argv(profile: ModelProfile, deploy: Deployment) -> list[str]:
    """Arguments after `python -u` for the legacy loom server."""
    _check_pair(profile, deploy)
    argv = [
        *_module(profile, "serve"),
        "--model-path", _model_path(profile, deploy),
        "--preset", profile.model.preset_name,
        "--model-dtype", profile.model.dtype,
        "--prompt-mode", profile.format.mode.value,
        "--temperature", repr(profile.sampling.temperature),
        "--top-p", repr(profile.sampling.top_p),
        "--map", str(deploy.map.path),
        "--discriminants", str(deploy.discriminants.path),
        "--calib-dir", str(deploy.calib_dir),
        "--work-dir", str(deploy.work_dir),
        "--future-tokens", str(profile.lengths.future_tokens),
        "--max-new-tokens", str(profile.lengths.reply_tokens),
        "--default-k", str(profile.lengths.default_k),
        "--max-turns", str(profile.lengths.max_turns),
        "--dose-policy-default", profile.dose.policy.value,
        "--wear-lever-default", profile.steer.lever_kind.value,
        "--host", deploy.host,
        "--port", str(deploy.port),
    ]
    if profile.format.mode is PromptMode.MODELC and profile.format.header:
        argv += ["--modelc-header", profile.format.header]
    if profile.format.mode is PromptMode.CHAT and profile.format.date_string:
        argv += ["--date-string", profile.format.date_string]
    if deploy.dose_band is not None:
        argv += ["--dose-band", str(deploy.dose_band.path)]
    if deploy.worker is not None:
        argv += ["--harvest-worker", deploy.worker.address,
                 "--harvest-worker-timeout", f"{deploy.worker.timeout_s:g}"]
    if deploy.allow_lan:
        argv.append("--allow-lan")
    return argv


def legacy_worker_argv(profile: ModelProfile, deploy: Deployment) -> list[str]:
    """Arguments after `python -u` for the legacy harvest worker."""
    _check_pair(profile, deploy)
    if deploy.worker is None:
        raise ValueError("deployment has no [worker] section")
    w = deploy.worker
    host, port = w.address.rsplit(":", 1)
    argv = [
        *_module(profile, "worker"),
        "--model-path", _model_path(profile, deploy),
        "--preset", profile.model.preset_name,
        "--model-dtype", profile.model.dtype,
        "--harvest-lane", w.lane,
        "--map", str(deploy.map.path),
        "--calib-dir", str(deploy.calib_dir),
        "--discriminants", str(deploy.discriminants.path),
        "--device", w.device,
        "--bins-layers", ",".join(str(x) for x in profile.map.bins.layers),
        "--bins-n-bins", str(profile.map.bins.n_bins),
        "--bins-resolution", profile.map.bins.resolution,
        "--host", host,
        "--port", port,
    ]
    if w.bins_device:
        argv += ["--bins-device", w.bins_device]
    return argv
