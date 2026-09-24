"""The ONLY door from the dose/steer contract tests to the code under test.

When a primitive moves inside ``pleroma/`` (``pleroma.dose.band``,
``pleroma.dose.policy``, ``pleroma.steer.hf``, ``pleroma.steer.vllm``), edit
THIS file and nothing else.

Three kinds of target live here:

1. **Importable names** — ``pleroma.dose.band`` (band loading, zones,
   cross-model transfer) and ``pleroma.dose.policy`` (flat/predicted, clamp,
   effective alpha). Plain imports.

2. **The ONE steering implementation** — ``pleroma.steer``:
   :data:`steer_hf` (HF residual-write pre-hooks) and :data:`steer_vllm`
   (the vLLM pre-hook on ``(positions, hidden)``; plain torch, no vllm).
   Every vLLM dose lane installs it; none carries its own injection closure.

3. **The loom's WEAR hook** — ``pleroma.serve.runtime.HFRuntime.attach``,
   bound to the map's sites (:func:`loom_attach`). The loom is the only HF
   steering caller, and the map/lever site convention is canonical. The tests
   bind the REAL method to a toy model, so they exercise the production
   WIRING (which sites, which span), not a re-typed copy of it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# ── dose band (pleroma.dose.band) ────────────────────────────────────────────
from pleroma.dose.band import (  # noqa: F401 — re-exported for the tests
    LEVER_QUALITY_CAVEAT,
    NONE_INFO_JSON,
    DoseBand,
    DoseZone,
    ResidualScale,
    apply_overdriven_ceiling,
    cross_model_bound,
    default_sidecar_path,
    dose_band_info_json,
    five_zone_band,
    load_dose_band_file,
    load_residual_scale_file,
    overdriven_ceiling,
    resolve_dose_band,
)

# ── the loom's dose policy (pleroma.dose.policy) ────────────────────────────
from pleroma.dose.policy import (  # noqa: F401
    DEFAULT_DOSE_POLICY,
    DOSE_POLICIES,
    DOSE_SCALE_MAX,
    DOSE_SCALE_MIN,
    DoseScale,
    dose_scale_for,
    effective_alpha_json,
    fan_mean_raw_norms,
    flat_dose_scale,
    resolve_dose_policy,
)
from pleroma.levers.kind import fan_wear_fields  # noqa: F401

# ── the shared HF write primitive pleroma.steer.hf is built on ───────────────
from anamnesis.extraction.model_loader import (  # noqa: F401
    ResidualWriteSpec,
    attach_residual_write,
    decoder_layers,
)

# ── pleroma.steer: the ONE steering implementation (HF + vLLM) ───────────────
from pleroma.steer import (  # noqa: F401
    TIME_K,
    Injection,
    InjectionSpan,
    SiteError,
    SpanError,
    SteerDTypeError,
    SteerError,
    TimeProfile,
    VectorShapeError,
    span_receipt,
)
from pleroma.steer import hf as steer_hf  # noqa: F401
from pleroma.steer import vllm as steer_vllm  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── artifacts ────────────────────────────────────────────────────────────────
#: public copies of banked bands (tests/fixtures/README.md).
DOSE_BANDS_DIR = REPO_ROOT / "tests" / "fixtures" / "dose_bands"
LOOM_UI = REPO_ROOT / "pleroma" / "serve" / "static" / "legacy_ui.html"


# ── vLLM lanes (pleroma.steer.vllm) ──────────────────────────────────────────


def vllm_state(*, vecs: Mapping[int, torch.Tensor] | None = None, prompt_len: int = 0,
               span: InjectionSpan | str,
               time_profile: TimeProfile | str = TimeProfile.CONSTANT,
               time_k: float = TIME_K) -> Any:
    """A pleroma.steer.vllm state armed with `vecs` / `prompt_len`."""
    st = steer_vllm.VllmInjectionState(span=span, time_profile=time_profile, time_k=time_k)
    st.wear(dict(vecs or {}), prompt_len)
    return st


def vllm_hook(state: Any, site: int) -> Callable[..., Any]:
    """The pre-hook installed on layers[site]: `hook(module, args, kwargs)`."""
    return steer_vllm.SitePreHook(state, site)


# ── HF attach closures ───────────────────────────────────────────────────────

#: lane -> (file, call signature, site convention, injection span). The loom
#: is the only HF caller left (it delegates to pleroma.steer.hf.attach).
HF_ATTACH_LANES: Mapping[str, dict[str, str]] = {
    "loom_serve": {"sig": "attach(vectors, alpha)", "sites": "map", "span": "uniform"},
}


def hf_attach(lane: str, model: Any, sites: Sequence[int]) -> Callable[..., list[Any]]:
    """The lane's `attach` bound to `model` and `sites`.

    `sites` is whatever the lane iterates in production (`loom_map.sites`
    for the loom). This binds the REAL `pleroma.serve.runtime.HFRuntime.attach`
    to the toy model (the tokenizer is only consulted for `chat_template`,
    which attach never reads)."""
    if lane not in HF_ATTACH_LANES:
        raise ValueError(f"unknown HF lane {lane!r}")
    from pleroma.serve.runtime import HFRuntime

    site_list = [int(s) for s in sites]
    rt = HFRuntime(
        torch=torch, model=model, tok=SimpleNamespace(chat_template=None),
        device="cpu", eos_ids=[0], pad_id=0, hidden_dim=0, prompt_mode="chat",
        modelc_header="", stop_strings=[], temperature=1.0, top_p=0.95,
        sites=site_list, steer_hf=steer_hf, injection_cls=Injection,
        span_cls=InjectionSpan)
    return rt.attach


def loom_attach(model: Any, sites: Sequence[int]) -> Callable[[np.ndarray, float], list[Any]]:
    """The loom's WEAR hook (pleroma.serve.runtime.HFRuntime.attach), the
    thing users feel."""
    return hf_attach("loom_serve", model, sites)


# ── the UI's dose functions (LOOM_UI), for server->UI contract checks ────────


def ui_js_slice(start_marker: str, end_marker: str) -> str:
    """A verbatim slice of the UI's script, `start_marker` inclusive up
    to `end_marker` exclusive (same technique as the legacy UI tests)."""
    src = LOOM_UI.read_text(encoding="utf-8")
    a = src.index(start_marker)
    b = src.index(end_marker, a)
    return src[a:b]
