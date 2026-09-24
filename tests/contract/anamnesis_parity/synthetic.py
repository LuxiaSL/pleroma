"""Synthetic capture + the frozen v3 feature-space config, shared by the CPU
parity test and the one-shot generator that banked ``v3_vendored_reference.json``.

The reference fixture is the output of the anamnesis snapshot the shipped
artifacts were fit with, on the inputs this module builds. Changing anything
in this module invalidates the fixture — the inputs are part of the contract.

The config mirrors the v3 family flags ``pleroma.harvest.replay.build_configs``
sets, on a toy 12-layer geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

L, H, NH, NKV, HD, V, P = 12, 64, 8, 4, 8, 300, 20
SAMPLED = [0, 3, 6, 9, 11]
PCA_LAYERS = [3, 6, 9]
TRAJ = [2, 5, 8, 11]
CONTR = [6, 9]
INTER = 96

# (tag, generated tokens, positional-calibration rows).
#   covered     — calibration covers every position
#   short_calib — positions run past the table (the clamp path)
#   tiny        — a 4-token generation
CASES: tuple[tuple[str, int, int], ...] = (
    ("covered", 40, 80),
    ("short_calib", 40, 30),
    ("tiny", 4, 80),
)


def make_raw(raw_cls: Any, T: int, pm_rows: int, seed: int = 7) -> Any:
    """A deterministic RawGenerationData with every field the v3 families read."""
    rng = np.random.RandomState(seed)
    hs = [rng.randn(L + 1, H).astype(np.float32) for _ in range(T)]
    att = []
    for t in range(T):
        a = rng.rand(L, NH, P + t + 1).astype(np.float32)
        a /= a.sum(axis=2, keepdims=True)
        att.append(a)
    logits = [rng.randn(V).astype(np.float32) for _ in range(T)]
    chosen = rng.randint(0, V, size=T).astype(np.float32)
    keys = {l: [rng.randn(NKV, HD).astype(np.float32) for _ in range(T)] for l in SAMPLED}
    gates = {l: [rng.randn(INTER).astype(np.float32) for _ in range(T)] for l in SAMPLED}
    vals = {l: [rng.randn(NKV, HD).astype(np.float32) for _ in range(T)] for l in range(L)}
    qs = {l: [rng.randn(NH, HD).astype(np.float32) for _ in range(T)] for l in range(L)}
    pm = rng.randn(L + 1, pm_rows, H).astype(np.float32) * 0.1
    return raw_cls(
        hidden_states=hs, attentions=att, logits=logits, chosen_token_ids=chosen,
        pre_rope_keys=keys, prompt_length=P, positional_means=pm,
        gate_activations=gates, v_proj_values=vals, queries=qs,
    )


def pca_fit() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(1)
    components = rng.randn(10, H).astype(np.float32)
    mean = rng.randn(H).astype(np.float32) * 0.1
    return components, mean


FAMILY_FLAGS: dict[str, Any] = dict(
    enable_residual_trajectory=True,
    enable_attention_flow=True,
    enable_gate_features=True,
    enable_temporal_dynamics=False,
    enable_per_head=True,
    enable_stft=True,
    enable_contrastive_projection=False,
    trajectory_layers=TRAJ,
    contrastive_layers=CONTR,
)
EXTRACTION_COMMON: dict[str, Any] = dict(
    sampled_layers=SAMPLED, pca_layers=PCA_LAYERS,
    early_layer_cutoff=3, late_layer_cutoff=9,
)


def results_to_doc(results: dict[str, Any], provenance: str) -> dict[str, Any]:
    """{tag: FeatureResult-like} -> the fixture's JSON layout (one name list and
    one block-slice list, which every case must share; per-case values)."""
    names = {tag: [str(n) for n in r.feature_names] for tag, r in results.items()}
    slices = {}
    for tag, r in results.items():
        bounds = getattr(r, "block_slices", None) or getattr(r, "tier_slices")
        slices[tag] = [f"{k}:{v[0]}:{v[1]}" for k, v in bounds.items()]
    first = next(iter(results))
    if any(n != names[first] for n in names.values()) or any(
            sl != slices[first] for sl in slices.values()):
        raise ValueError("cases disagree on feature names / block slices")
    return {
        "provenance": provenance,
        "feature_names": names[first],
        "block_slices": slices[first],
        "features": {tag: [float(x) for x in np.asarray(r.features, dtype=np.float64)]
                     for tag, r in results.items()},
    }
