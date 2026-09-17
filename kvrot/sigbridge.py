"""exp11 bridge: kv-rotation cache surgery → anamnesis-pl v3 signature extraction.

The one new capability of exp11 (design: notes/design-signature-preservation-2026-07-10.md
§bridge): teacher-force a frozen continuation against an *injected* KV cache (full, rotated,
recomputed, or naively-evicted) while running the anamnesis instrument, and produce a
``RawGenerationData`` byte-compatible with ``anamnesis.extraction.replay_extract`` so that
``compute_features_v2_from_data`` yields the standard v3 signature.

Ownership: this module lives in kv-rotation and IMPORTS anamnesis-pl (read-only dependency,
local checkout on PYTHONPATH — see :func:`ensure_anamnesis`). It never modifies it.

Alignment contract (mirrors replay_extract):
    The cache holds the context (``cache_len`` entries). The continuation tokens
    g_0..g_{N-1} are forwarded against it; entry ``i`` (i = 0..N-2) is the model state AT
    g_i, whose logits predict g_{i+1}. T = N-1 per-step entries. Attention rows span the
    ``cache_len + i + 1`` visible columns — the cache columns are exactly the *surgered*
    cache, which is the on-thesis T2.5 surface.

Positions: ``position_offset`` is the absolute RoPE position of g_0. FULL/ROT/REC use
``position_offset == cache_len``; the NAIVE condition (evict WITHOUT re-rotation) keeps
survivors at their ORIGINAL positions, so the continuation starts at survivor-max+1 while
the cache is physically compacted to K entries (``position_offset > cache_len``). The
positional-means correction indexes by ``prompt_length + t`` inside the extractor, so for
NAIVE we remap the means table (:func:`remap_positional_means`) — in practice the 3B
calibration covers 712 positions and every exp11 context is longer, so both paths clamp to
the same final row and the correction cancels in Δ; the remap keeps short/synthetic (test)
cases exact too.

Correctness gate (:func:`correctness_gate`): injected-FULL-cache replay of the continuation
must reproduce ``replay_extract``'s use_cache=False full-sequence features to rtol 1e-4 —
proving the injection path is not itself a signature perturbation. Must pass before any
cell runs.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

if TYPE_CHECKING:  # imported lazily at runtime (anamnesis-pl is a PYTHONPATH dependency)
    from anamnesis.extraction.model_loader import LoadedModel
    from anamnesis.extraction.state_extractor import ExtractionResult, RawGenerationData

F32 = NDArray[np.float32]

#: Where to find the anamnesis package if it is not already importable: the
#: repo root (the vendored copy) first, then a sibling checkout.
DEFAULT_ANAMNESIS_PATHS = (
    ".",
    "../anamnesis_exps/pipeline",
)


def ensure_anamnesis(explicit_path: str | None = None) -> str:
    """Make ``anamnesis`` importable; returns the path used.

    Resolution order: already importable → ``explicit_path`` → $ANAMNESIS_PL_PATH →
    :data:`DEFAULT_ANAMNESIS_PATHS`. Raises ImportError with instructions otherwise.
    """
    try:
        import anamnesis  # noqa: F401

        return getattr(anamnesis, "__file__", "<installed>") or "<installed>"
    except ImportError:
        pass
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if os.environ.get("ANAMNESIS_PL_PATH"):
        candidates.append(os.environ["ANAMNESIS_PL_PATH"])
    candidates.extend(DEFAULT_ANAMNESIS_PATHS)
    for cand in candidates:
        p = Path(cand).expanduser()
        if (p / "anamnesis" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            try:
                import anamnesis  # noqa: F401

                return str(p)
            except ImportError:
                sys.path.remove(str(p))
    raise ImportError(
        "anamnesis-pl not importable. Pass --anamnesis-path / set ANAMNESIS_PL_PATH to the "
        f"pipeline checkout (tried: {candidates})."
    )


# ── v3 configs (single source of truth for exp11; mirrors run_replay_extraction) ──


def v3_extraction_config(preset_name: str = "3b"):
    """The ExtractionConfig used for the v3 signature space (3B preset by default)."""
    from anamnesis.config import MODEL_PRESETS, ExtractionConfig

    preset = MODEL_PRESETS[preset_name]
    return ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=True,
    )


def v3_family_config(preset_name: str = "3b"):
    """The FeaturePipelineConfig producing the 2,713-dim v3 FULL feature space.

    Matches anamnesis.scripts.run_replay_extraction: baseline tiers + residual
    trajectory + attention_flow + gate + per_head; temporal_dynamics and the
    contrastive/value/qk addons OFF. This is the space factor_directions_3b.npz
    (FULL_W / FULL_names) was fit in.
    """
    from anamnesis.config import MODEL_PRESETS, FeaturePipelineConfig

    preset = MODEL_PRESETS[preset_name]
    return FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,
        enable_per_head=True,
        enable_stft=True,
        enable_contrastive_projection=False,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )


# ── Calibration (read-only artifacts; NEVER refit) ─────────────────────────────


@dataclass
class CalibrationBundle:
    """3B calibration artifacts, loaded read-only.

    NOTE (layout deviation from the design note): the spec points at
    ``anamnesis_exps/outputs/calibration/`` but that directory only holds the 8B
    calibration (``llama31_8b/``). The 3B artifacts live where the 3B ModelPreset
    points: ``anamnesis_exps/phase_0/outputs/calibration/`` (positional_means.npz +
    pca_model.pkl, legacy pooled-PCA format). exp11 stages them into a flat dir.
    """

    positional_means: F32          # [num_layers+1, max_pos, hidden]
    pca_components: F32 | None     # [n_components, hidden] (legacy pooled) or per-layer dict
    pca_mean: F32 | None           # [hidden] or per-layer dict
    source_dir: str = ""


def load_3b_calibration(calib_dir: Path | str, *, enable_tier3: bool = True) -> CalibrationBundle:
    """Load positional means + fitted T3 PCA from a calibration directory.

    Accepts both the legacy pooled dict pickle ({"components", "mean"}) and the C5
    per-layer format; mirrors anamnesis.scripts.run_replay_extraction._load_calibration
    but fails loudly instead of warning (exp11 must never run with silently-missing
    calibration).
    """
    calib_dir = Path(calib_dir).expanduser()
    pm_path = calib_dir / "positional_means.npz"
    if not pm_path.exists():
        raise FileNotFoundError(f"positional_means.npz not found in {calib_dir}")
    with np.load(pm_path) as pm_npz:
        positional_means = pm_npz["positional_means"].astype(np.float32)

    pca_components: Any = None
    pca_mean: Any = None
    if enable_tier3:
        pca_path = calib_dir / "pca_model.pkl"
        if not pca_path.exists():
            raise FileNotFoundError(f"pca_model.pkl not found in {calib_dir} (tier3 enabled)")
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        if isinstance(pca, dict):
            vals = list(pca.values())
            if vals and isinstance(vals[0], dict) and "components" in vals[0]:
                # C5 per-layer format
                pca_components = {
                    int(k): np.asarray(v["components"], dtype=np.float32) for k, v in pca.items()
                }
                pca_mean = {int(k): np.asarray(v["mean"], dtype=np.float32) for k, v in pca.items()}
            else:
                pca_components = np.asarray(pca["components"], dtype=np.float32)
                pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        else:  # fitted sklearn PCA object
            pca_components = np.asarray(pca.components_, dtype=np.float32)
            pca_mean = np.asarray(pca.mean_, dtype=np.float32)
    return CalibrationBundle(
        positional_means=positional_means,
        pca_components=pca_components,
        pca_mean=pca_mean,
        source_dir=str(calib_dir),
    )


# ── Mode-discriminant directions (metric 5b, exploratory) ──────────────────────


@dataclass
class FactorDirections:
    """Banked LDA mode-discriminant directions (factor_directions_3b.npz)."""

    space: str                     # "FULL" (2713-dim v3) or "HAND" (1463-dim hand families)
    labels: list[str]              # the 5 calibration modes
    W: NDArray[np.float64]         # [4, D] discriminant rows
    mean: NDArray[np.float64]      # [D] standardization mean (cancels in deltas)
    scale: NDArray[np.float64]     # [D] standardization scale
    names: list[str]               # [D] feature names — alignment contract
    centroids: dict[str, NDArray[np.float64]] = field(default_factory=dict)  # mode → [4]


def load_factor_directions(path: Path | str, space: str = "FULL") -> FactorDirections:
    if space not in ("FULL", "HAND"):
        raise ValueError(f"space must be FULL or HAND, got {space!r}")
    with np.load(Path(path).expanduser(), allow_pickle=True) as fd:
        labels = [str(x) for x in fd["labels"]]
        W = np.asarray(fd[f"{space}_W"], dtype=np.float64)
        mean = np.asarray(fd[f"{space}_mean"], dtype=np.float64)
        scale = np.asarray(fd[f"{space}_scale"], dtype=np.float64)
        names = [str(x) for x in fd[f"{space}_names"]]
        centroids = {
            m: np.asarray(fd[f"{space}_centroid_{m}"], dtype=np.float64)
            for m in labels
            if f"{space}_centroid_{m}" in fd
        }
    if W.shape[1] != len(names) or mean.shape[0] != len(names) or scale.shape[0] != len(names):
        raise ValueError(
            f"factor directions shape mismatch: W {W.shape}, names {len(names)}"
        )
    return FactorDirections(
        space=space, labels=labels, W=W, mean=mean, scale=scale, names=names, centroids=centroids
    )


@dataclass
class ModeProjection:
    """Result of projecting a signature delta into the mode-discriminant subspace."""

    mode_fraction: float           # ‖P_U Δz‖² / ‖Δz‖²  (U = orthonormalized rows of W)
    proj_w: list[float]            # raw W @ Δz (centroid-comparable 4-vector)
    delta_z_norm: float
    centroid_cosines: dict[str, float]  # cos(proj_w, centroid_m − centroid_mean)


def mode_projection(
    delta: F32, feature_names: list[str], fd: FactorDirections
) -> ModeProjection:
    """Metric 5b: Δz = Δ ⊘ scale, projected onto orthonormalized rows of W.

    HARD ASSERT (spec, amendment 1): the feature-name list must equal the banked
    ``names`` exactly, order included — otherwise the projection is meaningless.
    """
    if list(feature_names) != list(fd.names):
        n_a, n_b = len(feature_names), len(fd.names)
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(feature_names, fd.names)) if a != b), None
        )
        raise AssertionError(
            f"feature-name alignment failed vs {fd.space}_names: lengths {n_a} vs {n_b}, "
            f"first mismatch at index {first_bad}"
            + (
                f" ({feature_names[first_bad]!r} vs {fd.names[first_bad]!r})"
                if first_bad is not None
                else ""
            )
        )
    scale = np.where(np.abs(fd.scale) < 1e-12, 1.0, fd.scale)
    dz = np.asarray(delta, dtype=np.float64) / scale
    dz_norm = float(np.linalg.norm(dz))
    q, _ = np.linalg.qr(fd.W.T)            # [D, 4] orthonormal basis of the row space
    proj_orth = q.T @ dz                   # [4]
    frac = float((proj_orth @ proj_orth) / max(dz_norm**2, 1e-24))
    proj_w = fd.W @ dz                     # [4] in the same coordinates as the centroids
    cosines: dict[str, float] = {}
    if fd.centroids:
        cmean = np.mean(np.stack(list(fd.centroids.values())), axis=0)
        pn = np.linalg.norm(proj_w)
        for m, c in fd.centroids.items():
            cc = c - cmean
            denom = pn * np.linalg.norm(cc)
            cosines[m] = float((proj_w @ cc) / denom) if denom > 1e-12 else 0.0
    return ModeProjection(
        mode_fraction=frac,
        proj_w=[float(x) for x in proj_w],
        delta_z_norm=dz_norm,
        centroid_cosines=cosines,
    )


# ── Positional-means remap (NAIVE condition) ────────────────────────────────────


def remap_positional_means(
    positional_means: F32 | None, cache_len: int, position_offset: int, n_steps: int
) -> F32 | None:
    """Adjust the means table so extractor lookups (index ``cache_len + t``) return the
    means of the TRUE absolute positions (``position_offset + t``).

    Identity when ``position_offset == cache_len`` (FULL/ROT/REC) or when every index in
    play clamps to the table's final row (all realistic exp11 contexts). Only the NAIVE
    condition (survivors keep original positions ⇒ offset > cache_len) with a short cache
    actually needs the copy.
    """
    if positional_means is None or position_offset == cache_len:
        return positional_means
    if position_offset < cache_len:
        raise ValueError(
            f"position_offset {position_offset} < cache_len {cache_len} — exp11 conditions "
            "never place the continuation before the end of the cache"
        )
    max_pos = positional_means.shape[1]
    if cache_len >= max_pos - 1:
        return positional_means  # every lookup clamps to the final row on both paths
    remapped = positional_means.copy()
    for t in range(n_steps):
        dst = cache_len + t
        if dst >= max_pos:
            break
        src = min(position_offset + t, max_pos - 1)
        remapped[:, dst, :] = positional_means[:, src, :]
    return remapped


# ── The bridge ──────────────────────────────────────────────────────────────────


def _cache_length(past_key_values) -> int:
    """Sequence length of an HF cache (transformers 4.x/5.x tolerant)."""
    get_len = getattr(past_key_values, "get_seq_length", None)
    if callable(get_len):
        return int(get_len())
    layers = getattr(past_key_values, "layers", None)
    if layers:
        return int(layers[0].keys.shape[-2])
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache:
        return int(key_cache[0].shape[-2])
    raise TypeError(f"cannot determine cache length for {type(past_key_values)!r}")


def _slice_hook_batched(
    captures: dict[int, list[Tensor]], t_steps: int
) -> dict[int, list[F32]]:
    """Single-capture hook dicts ([1, heads, N, dim]) → {layer: T × [heads, dim]}."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in captures.items():
        if not tensors:
            continue
        if len(tensors) != 1:
            raise RuntimeError(
                f"expected 1 hook capture per layer for a batched forward, got "
                f"{len(tensors)} at layer {layer_idx} (stale hook state?)"
            )
        rows = tensors[0][0, :, :t_steps, :].float().cpu().numpy()  # [heads, T, dim]
        result[int(layer_idx)] = [rows[:, i, :] for i in range(t_steps)]
    return result


def _slice_hook_incremental(
    captures: dict[int, list[Tensor]], t_steps: int
) -> dict[int, list[F32]]:
    """Per-step hook dicts (N captures of [1, heads, 1, dim]) → {layer: T × [heads, dim]}."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in captures.items():
        if not tensors:
            continue
        if len(tensors) < t_steps:
            raise RuntimeError(
                f"layer {layer_idx}: {len(tensors)} per-step captures < T={t_steps}"
            )
        result[int(layer_idx)] = [
            tensors[i][0, :, 0, :].float().cpu().numpy() for i in range(t_steps)
        ]
    return result


def _slice_gate_batched(
    captures: dict[int, list[Tensor]], t_steps: int
) -> dict[int, list[F32]] | None:
    """Gate hook dicts ([1, N, intermediate]) → {layer: T × [intermediate]}."""
    result: dict[int, list[F32]] = {}
    for layer_idx, tensors in captures.items():
        if not tensors:
            continue
        if len(tensors) == 1:  # batched forward
            rows = tensors[0][0, :t_steps, :].float().cpu().numpy()
            result[int(layer_idx)] = [rows[i] for i in range(t_steps)]
        else:  # incremental
            result[int(layer_idx)] = [
                tensors[i][0, 0, :].float().cpu().numpy() for i in range(t_steps)
            ]
    return result or None


@torch.no_grad()
def replay_extract_cached(
    loaded: "LoadedModel",
    past_key_values,
    cont_ids: Tensor | list[int] | NDArray,
    position_offset: int,
    positional_means: F32 | None = None,
    *,
    incremental: bool = False,
) -> "RawGenerationData":
    """Teacher-force ``cont_ids`` against an injected cache; extract per-step states.

    Parameters
    ----------
    loaded : anamnesis LoadedModel
        Loaded with eager attention and the v3 hook surface (k_proj + gate_proj).
    past_key_values : transformers Cache (e.g. from kvrot.snapshot.to_hf_dynamic_cache)
        CONSUMED: the forward appends the continuation keys to it. Pass a fresh cache
        per call (rebuild from the KVSnapshot — snapshot tensors are not mutated).
    cont_ids : [1, N] / [N] token ids
        The frozen continuation (N >= 2; T = N-1 per-step entries are banked).
    position_offset : int
        Absolute RoPE position of the first continuation token. == cache length for
        FULL/ROT/REC; survivor-max+1 (> cache length) for NAIVE.
    positional_means : optional calibration means (remapped internally for NAIVE).
    incremental : bool
        False (default): ONE batched forward over all N tokens — the standard bridge.
        True: N single-token forwards extending the cache — the decode-path twin used
        for the path floor (metric 3b). Identical output layout.

    Returns
    -------
    RawGenerationData with prompt_length = cache length (attention-column semantics:
    the extractor's prompt/generated split lands exactly on the cache/continuation
    boundary, which is the surgered-cache readout T2.5 needs).
    """
    from anamnesis.extraction.state_extractor import RawGenerationData

    device = next(loaded.model.parameters()).device
    if isinstance(cont_ids, torch.Tensor):
        ids = cont_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(np.asarray(cont_ids), dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[0] != 1:
        raise ValueError(f"replay_extract_cached expects batch=1, got {ids.shape[0]}")

    n = int(ids.shape[1])
    if n < 2:
        raise ValueError(f"need >= 2 continuation tokens for a per-step transition, got {n}")
    t_steps = n - 1

    cache_len = _cache_length(past_key_values)
    if cache_len <= 0:
        raise ValueError("injected cache is empty — prefill the context first")
    if position_offset < cache_len:
        raise ValueError(
            f"position_offset {position_offset} < cache_len {cache_len} (see docstring)"
        )
    pm_eff = remap_positional_means(positional_means, cache_len, position_offset, t_steps)

    loaded.clear_hook_state()
    loaded.enable_hooks()
    try:
        if not incremental:
            out = loaded.model(
                ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                position_ids=torch.arange(
                    position_offset, position_offset + n, device=device
                ).unsqueeze(0),
                cache_position=torch.arange(cache_len, cache_len + n, device=device),
            )
            loaded.flush_hooks_to_cpu()

            hs = out.hidden_states  # tuple(num_layers+1) of [1, N, hidden]
            n_hs_layers = len(hs)
            hs_rows = [hs[l][0, :t_steps].float().cpu().numpy() for l in range(n_hs_layers)]
            hidden_states: list[F32] = [
                np.stack([hs_rows[l][i] for l in range(n_hs_layers)]) for i in range(t_steps)
            ]

            att = out.attentions  # tuple(num_layers) of [1, H, N, cache_len + N]
            if att is None or len(att) == 0:
                raise RuntimeError(
                    "forward returned no attentions — model must be loaded with "
                    "attn_implementation='eager'"
                )
            exp_cols = cache_len + n
            if int(att[0].shape[-1]) != exp_cols:
                raise RuntimeError(
                    f"attention column count {int(att[0].shape[-1])} != cache_len + N "
                    f"({exp_cols}) — cache injection did not take effect as expected"
                )
            att_rows = [a[0, :, :t_steps, :].float().cpu().numpy() for a in att]  # [H, T, C+N]
            attentions: list[F32] = [
                np.stack([att_rows[l][:, i, : cache_len + i + 1] for l in range(len(att))])
                for i in range(t_steps)
            ]

            logits_rows = out.logits[0, :t_steps].float().cpu().numpy()
            logits: list[F32] = [logits_rows[i] for i in range(t_steps)]
            del out
            pre_rope_keys = _slice_hook_batched(loaded.hook_state.pre_rope_keys, t_steps)
            gate_activations = _slice_gate_batched(loaded.hook_state.gate_activations, t_steps)
        else:
            hidden_states = []
            attentions = []
            logits = []
            n_hs_layers = None
            for t in range(n):
                out = loaded.model(
                    ids[:, t : t + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                    position_ids=torch.tensor([[position_offset + t]], device=device),
                    cache_position=torch.tensor([cache_len + t], device=device),
                )
                past_key_values = out.past_key_values
                if t < t_steps:
                    hs = out.hidden_states
                    n_hs_layers = len(hs)
                    hidden_states.append(
                        np.stack(
                            [hs[l][0, 0].float().cpu().numpy() for l in range(n_hs_layers)]
                        )
                    )
                    att = out.attentions
                    if att is None or len(att) == 0:
                        raise RuntimeError(
                            "forward returned no attentions — need eager attention"
                        )
                    exp_cols = cache_len + t + 1
                    if int(att[0].shape[-1]) != exp_cols:
                        raise RuntimeError(
                            f"step {t}: attention columns {int(att[0].shape[-1])} != "
                            f"{exp_cols}"
                        )
                    attentions.append(
                        np.stack([a[0, :, 0, :].float().cpu().numpy() for a in att])
                    )
                    logits.append(out.logits[0, -1].float().cpu().numpy())
                del out
            loaded.flush_hooks_to_cpu()
            pre_rope_keys = _slice_hook_incremental(loaded.hook_state.pre_rope_keys, t_steps)
            gate_activations = _slice_gate_batched(loaded.hook_state.gate_activations, t_steps)
    finally:
        loaded.clear_hook_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chosen_token_ids = ids[0, 1:n].float().cpu().numpy()  # [T]

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_token_ids,
        pre_rope_keys=pre_rope_keys or {},
        prompt_length=cache_len,
        positional_means=pm_eff,
        gate_activations=gate_activations,
        v_proj_values=None,
        queries=None,
    )


# ── Signature computation + alignment ────────────────────────────────────────────


def compute_signature(
    raw_data: "RawGenerationData",
    extraction_config,
    family_config,
    calib: CalibrationBundle,
) -> "ExtractionResult":
    """v3 signature from bridge output (thin wrapper; keeps configs consistent)."""
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data

    return compute_features_v2_from_data(
        raw_data, extraction_config, family_config, calib.pca_components, calib.pca_mean
    )


def assert_feature_alignment(reference_names: list[str], names: list[str], label: str) -> None:
    """Hard assert (spec confound #4): identical feature-name lists, order included."""
    if list(reference_names) == list(names):
        return
    first_bad = next(
        (i for i, (a, b) in enumerate(zip(reference_names, names)) if a != b), None
    )
    raise AssertionError(
        f"[{label}] feature-name alignment failed: lengths {len(reference_names)} vs "
        f"{len(names)}, first mismatch at "
        + (
            f"index {first_bad} ({reference_names[first_bad]!r} vs {names[first_bad]!r})"
            if first_bad is not None
            else "length"
        )
    )


# ── Correctness gate ─────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    passed: bool
    names_equal: bool
    n_features: int
    n_violations: int
    max_abs_diff: float
    max_rel_diff: float
    rtol: float
    atol: float
    worst: list[tuple[str, float, float]] = field(default_factory=list)  # (name, ref, bridge)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"correctness gate: {status} — {self.n_features} features, "
            f"{self.n_violations} outside rtol={self.rtol:g}/atol={self.atol:g}, "
            f"max_abs={self.max_abs_diff:.3e}, max_rel={self.max_rel_diff:.3e}, "
            f"names_equal={self.names_equal}"
        ]
        for name, r, b in self.worst[:10]:
            lines.append(f"    worst: {name}: ref={r:.6g} bridge={b:.6g}")
        return "\n".join(lines)


@torch.no_grad()
def correctness_gate(
    loaded: "LoadedModel",
    ctx_ids: Tensor,
    cont_ids: Tensor,
    extraction_config,
    family_config,
    calib: CalibrationBundle,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> GateResult:
    """Spec gate: injected-FULL-cache replay ≡ replay_extract(use_cache=False).

    ``ctx_ids`` [1, P] is the context, ``cont_ids`` [1, N] the frozen continuation.
    Both paths produce the full v3 signature; equal within rtol proves the injection
    path (explicit position_ids/cache_position + DynamicCache) is not itself a
    signature perturbation. MUST pass before any cell runs.
    """
    from transformers import DynamicCache

    from anamnesis.extraction.replay_extract import replay_extract

    device = next(loaded.model.parameters()).device
    ctx_ids = ctx_ids.to(device=device, dtype=torch.long)
    cont_ids = cont_ids.to(device=device, dtype=torch.long)
    p_len = int(ctx_ids.shape[1])

    # Reference path: one use_cache=False forward over [ctx + cont].
    full_ids = torch.cat([ctx_ids, cont_ids], dim=1)
    ref_raw = replay_extract(
        loaded, full_ids[0], p_len, positional_means=calib.positional_means
    )
    ref = compute_signature(ref_raw, extraction_config, family_config, calib)
    del ref_raw

    # Bridge path: prefill (hooks disabled) → injected-cache replay of the continuation.
    loaded.disable_hooks()
    try:
        out = loaded.model(ctx_ids, past_key_values=DynamicCache(), use_cache=True)
        cache = out.past_key_values
        del out
    finally:
        loaded.enable_hooks()
    bridge_raw = replay_extract_cached(
        loaded, cache, cont_ids, p_len, positional_means=calib.positional_means
    )
    del cache
    bridge = compute_signature(bridge_raw, extraction_config, family_config, calib)
    del bridge_raw

    names_equal = list(ref.feature_names) == list(bridge.feature_names)
    a = np.asarray(ref.features, dtype=np.float64)
    b = np.asarray(bridge.features, dtype=np.float64)
    if a.shape != b.shape:
        return GateResult(
            passed=False, names_equal=names_equal, n_features=len(a), n_violations=len(a),
            max_abs_diff=float("inf"), max_rel_diff=float("inf"), rtol=rtol, atol=atol,
        )
    abs_diff = np.abs(a - b)
    ok = np.isclose(a, b, rtol=rtol, atol=atol)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = abs_diff / np.maximum(np.abs(a), 1e-12)
    order = np.argsort(-abs_diff)
    worst = [
        (str(ref.feature_names[i]), float(a[i]), float(b[i]))
        for i in order[:10]
        if not ok[i]
    ]
    return GateResult(
        passed=bool(names_equal and ok.all()),
        names_equal=names_equal,
        n_features=int(a.shape[0]),
        n_violations=int((~ok).sum()),
        max_abs_diff=float(abs_diff.max()) if len(a) else 0.0,
        max_rel_diff=float(np.nanmax(rel)) if len(a) else 0.0,
        rtol=rtol,
        atol=atol,
        worst=worst,
    )
