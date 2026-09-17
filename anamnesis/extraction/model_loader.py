"""Model loading and pre-RoPE hook management.

Architecture-agnostic — all model specifics come from ModelConfig.

Responsibilities:
  - Load model + tokenizer with eager attention (required for attn weights)
  - Register forward hooks on k_proj linear layers to capture pre-RoPE keys
  - Provide hook lifecycle management (register, collect, clear, remove)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from anamnesis.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """Mutable storage for hook captures across generation steps.

    Tensors are kept on GPU during generation to avoid per-step synchronous
    CPU transfers. Call flush_to_cpu() after generation completes and before
    accessing tensors via get_generation_keys() / get_generation_gates().
    """

    # layer_idx → list of tensors, one per generation step
    # Each tensor shape: [1, num_kv_heads, seq_len_at_step, head_dim]
    #   - During prefill: seq_len_at_step = prompt_length
    #   - During generation: seq_len_at_step = 1
    pre_rope_keys: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of tensors, one per generation step
    # Each tensor shape: [1, seq_len_at_step, intermediate_size]
    #   - During prefill: seq_len_at_step = prompt_length
    #   - During generation: seq_len_at_step = 1
    # This is the gate_proj output BEFORE SiLU activation.
    # Apply SiLU in feature computation to get actual gate values.
    gate_activations: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of v_proj outputs (values), reshaped [1, num_kv_heads, seq_len, head_dim].
    # Captured at the same point as keys (post-projection, GQA layout). For replay (single
    # forward) each list holds one full-sequence tensor; for generate, one per step.
    v_proj_values: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of q_proj outputs (queries, PRE-RoPE), reshaped
    # [1, num_attention_heads, seq_len, head_dim]. Position-free query content;
    # re-apply RoPE offline (with banked positions) for post-RoPE QK geometry.
    queries: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # layer_idx → list of o_proj outputs (attention-block output, per-token
    # contribution the attention sublayer ADDS to the residual stream),
    # shape [1, seq_len, hidden_dim]. The attention-output surface (vmb Stage A):
    # with block-boundary hidden states banked, mlp_out is derivable as
    # resid_{l+1} − resid_l − attn_out, so this one capture closes two census cells.
    attn_outputs: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))

    # ── MoE expert routing (vmb arm A7, M6 DeepSeek-V2-Lite) — router allocation + branch norms.
    # layer_idx → per-step [batch, seq, n_routed_experts] DENSE pre-topk softmax (recomputed in the
    # MoE-module pre-hook from gate.weight, since DeepseekV2Moe bypasses gate.forward). Prefill = step 0.
    router_dist: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    # layer_idx → per-step [n_tok] L2 norm of the shared-expert branch output (shared_mass numerator).
    router_shared_norm: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    # layer_idx → per-step [n_tok] L2 norm of the routed-expert sum output (shared_mass denominator term).
    router_routed_norm: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    # ── v2.1 enrichment (xrt geometry + magnitude rungs) ──
    # layer_idx → per-step [batch, seq] L2 norm of the DENSE router logits (pre-softmax; the routing
    # commitment magnitude the softmax normalizes away). Feeds xrt_logit_norm (v2.1 magnitude). MoE only.
    router_logit_norm: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    # layer_idx → per-step [n_tok, hidden] shared / routed branch OUTPUT vectors. TRANSIENT — used only
    # to derive the per-token shared_routed cosine at conversion; NEVER persisted to raw (no disk cost).
    router_shared_vec: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    router_routed_vec: dict[int, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    # NB c_KV (MLA keys-analog) is captured INTO pre_rope_keys as [batch, 1, seq, kv_lora_rank] — it
    # reuses the whole keys pipeline (num_kv_heads=1, head_dim=512); no separate field needed.

    enabled: bool = True
    _on_cpu: bool = field(default=False, repr=False)

    def flush_to_cpu(self) -> None:
        """Batch-transfer all captured tensors from GPU to CPU.

        Replaces ~N*layers synchronous per-step transfers with one batch
        transfer per layer. Must be called after generation completes and
        before accessing tensors via get_generation_keys/gates.
        """
        if self._on_cpu:
            return
        for layer_idx in list(self.pre_rope_keys.keys()):
            tensors = self.pre_rope_keys[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.pre_rope_keys[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.gate_activations.keys()):
            tensors = self.gate_activations[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.gate_activations[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.v_proj_values.keys()):
            tensors = self.v_proj_values[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.v_proj_values[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.queries.keys()):
            tensors = self.queries[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.queries[layer_idx] = [t.cpu() for t in tensors]
        for layer_idx in list(self.attn_outputs.keys()):
            tensors = self.attn_outputs[layer_idx]
            if tensors and tensors[0].is_cuda:
                self.attn_outputs[layer_idx] = [t.cpu() for t in tensors]
        for store in (self.router_dist, self.router_shared_norm, self.router_routed_norm,
                      self.router_logit_norm, self.router_shared_vec, self.router_routed_vec):
            for layer_idx in list(store.keys()):
                tensors = store[layer_idx]
                if tensors and tensors[0].is_cuda:
                    store[layer_idx] = [t.cpu() for t in tensors]
        self._on_cpu = True

    def clear(self) -> None:
        """Release all captured tensors."""
        for tensors in self.pre_rope_keys.values():
            del tensors[:]
        self.pre_rope_keys.clear()
        for tensors in self.gate_activations.values():
            del tensors[:]
        self.gate_activations.clear()
        for tensors in self.v_proj_values.values():
            del tensors[:]
        self.v_proj_values.clear()
        for tensors in self.queries.values():
            del tensors[:]
        self.queries.clear()
        for tensors in self.attn_outputs.values():
            del tensors[:]
        self.attn_outputs.clear()
        for store in (self.router_dist, self.router_shared_norm, self.router_routed_norm,
                      self.router_logit_norm, self.router_shared_vec, self.router_routed_vec):
            for tensors in store.values():
                del tensors[:]
            store.clear()
        self._on_cpu = False

    def get_generation_keys(self, layer_idx: int) -> list[Tensor]:
        """Return captured keys for a layer (excluding prefill step 0)."""
        all_keys = self.pre_rope_keys.get(layer_idx, [])
        if len(all_keys) <= 1:
            return []
        # Skip index 0 (prefill), return generation steps only
        return all_keys[1:]

    def get_generation_gates(self, layer_idx: int) -> list[Tensor]:
        """Return captured gate activations for a layer (excluding prefill step 0)."""
        all_gates = self.gate_activations.get(layer_idx, [])
        if len(all_gates) <= 1:
            return []
        return all_gates[1:]


def decoder_layers(model):
    """Decoder layer list across architectures.

    Llama/Qwen/OLMo-2: `model.model.layers`. Gemma-3 (vmb M5) ships as
    Gemma3ForConditionalGeneration — a multimodal wrapper whose text decoder
    nests under `language_model`; hook paths must resolve through it.
    (M5 onboarding audit 2026-07-12; GPU validation pending — journal W10.)
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner.layers
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__} — extend "
        "decoder_layers() for this architecture"
    )


def _make_k_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a k_proj linear layer.

    k_proj output shape: [batch, seq_len, num_kv_heads * head_dim]
    We reshape to [batch, num_kv_heads, seq_len, head_dim] and keep on GPU.
    Call hook_state.flush_to_cpu() after generation to batch-transfer.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_kv_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.pre_rope_keys[layer_idx].append(reshaped)

    return hook_fn


def _make_gate_proj_hook(
    layer_idx: int,
    hook_state: HookState,
) -> Any:
    """Create a forward hook for a gate_proj linear layer (SwiGLU gate).

    gate_proj output shape: [batch, seq_len, intermediate_size]
    Kept on GPU; call hook_state.flush_to_cpu() after generation.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        hook_state.gate_activations[layer_idx].append(output.detach())

    return hook_fn


def _make_v_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_kv_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a v_proj linear layer (attention values).

    v_proj output shape: [batch, seq_len, num_kv_heads * head_dim] (GQA, same
    layout as k_proj). Reshaped to [batch, num_kv_heads, seq_len, head_dim] and
    kept on GPU. Call hook_state.flush_to_cpu() after the forward.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_kv_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.v_proj_values[layer_idx].append(reshaped)

    return hook_fn


def _make_q_proj_hook(
    layer_idx: int,
    hook_state: HookState,
    num_attention_heads: int,
    head_dim: int,
) -> Any:
    """Create a forward hook for a q_proj linear layer (PRE-RoPE queries).

    q_proj output shape: [batch, seq_len, num_attention_heads * head_dim].
    Reshaped to [batch, num_attention_heads, seq_len, head_dim] and kept on GPU.
    Captured pre-RoPE (mirrors the pre-RoPE key design); re-apply RoPE offline
    with banked positions for post-RoPE QK-space geometry.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        batch, seq_len, _ = output.shape
        reshaped = output.detach().reshape(batch, seq_len, num_attention_heads, head_dim)
        reshaped = reshaped.transpose(1, 2)
        hook_state.queries[layer_idx].append(reshaped)

    return hook_fn


def _make_o_proj_hook(
    layer_idx: int,
    hook_state: HookState,
) -> Any:
    """Create a forward hook for an o_proj linear layer (attention output).

    o_proj output shape: [batch, seq_len, hidden_dim] — the attention sublayer's
    additive contribution to the residual stream (pre residual-add). Kept on GPU;
    call hook_state.flush_to_cpu() after the forward.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        hook_state.attn_outputs[layer_idx].append(output.detach())

    return hook_fn


# ── MLA + MoE capture hooks (vmb arm A7, M6 DeepSeek-V2-Lite) ───────────────────
# Wired against the VERIFIED native transformers deepseek_v2 module tree (audit
# 2026-07-18, job b0713ba4d5d3). See HOOK-AUDIT-PLAN-M6-dsv2lite §3.

def _make_ckv_hook(
    layer_idx: int,
    hook_state: HookState,
    kv_lora_rank: int,
) -> Any:
    """Forward hook on `self_attn.kv_a_proj_with_mqa` — captures the c_KV latent.

    Output shape [batch, seq, kv_lora_rank + qk_rope_head_dim]; the first
    kv_lora_rank dims = c_KV (position-free compressed KV latent, the MLA
    keys-source analog). Stored INTO pre_rope_keys as [batch, 1, seq, kv_lora_rank]
    (num_kv_heads=1, head_dim=kv_lora_rank) so it rides the existing keys pipeline
    unchanged. k_pe (the trailing qk_rope dims) is position-carrying → dropped here.
    """

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        c_kv = output.detach()[..., :kv_lora_rank]           # [batch, seq, kv_lora_rank]
        batch, seq_len, d = c_kv.shape
        reshaped = c_kv.reshape(batch, seq_len, 1, d).transpose(1, 2)  # [batch, 1, seq, d]
        hook_state.pre_rope_keys[layer_idx].append(reshaped)

    return hook_fn


def _make_moe_router_prehook(
    layer_idx: int,
    hook_state: HookState,
) -> Any:
    """forward_PRE_hook on the MoE module (DeepseekV2Moe) — captures the dense router dist.

    DeepseekV2Moe.forward computes `F.linear(h.float(), self.gate.weight.float())`
    directly and NEVER calls gate.forward, so a forward hook on `mlp.gate` never
    fires. We recompute the dense pre-topk softmax here from the module's own
    gate.weight and the input hidden states (args[0]). Banked reading =
    `pretopk_softmax_dense` (HOOK-AUDIT-PLAN §R).
    """

    def pre_hook(module: nn.Module, args: tuple[Any, ...]) -> None:
        if not hook_state.enabled:
            return
        h = args[0]
        logits = torch.nn.functional.linear(
            h.to(torch.float32), module.gate.weight.to(torch.float32))
        dist = logits.softmax(dim=-1)                        # [batch, seq, n_routed_experts]
        hook_state.router_dist[layer_idx].append(dist.detach())
        # v2.1 magnitude rung: ‖router_logits‖ per token (pre-softmax commitment scale).
        hook_state.router_logit_norm[layer_idx].append(
            logits.norm(dim=-1).detach())                    # [batch, seq]

    return pre_hook


def _make_branch_norm_hook(
    layer_idx: int,
    hook_state: HookState,
    which: str,
) -> Any:
    """Forward hook on `mlp.shared_experts` / `mlp.experts` — per-token output L2 norm.

    which="shared" → shared-branch output [batch, seq, hidden]; which="routed" →
    routed-expert sum output [n_tok, hidden]. Both flattened to [n_tok] norms;
    paired downstream into shared_mass = shared / (shared + routed).
    """
    dst = hook_state.router_shared_norm if which == "shared" else hook_state.router_routed_norm
    vdst = hook_state.router_shared_vec if which == "shared" else hook_state.router_routed_vec

    def hook_fn(module: nn.Module, args: tuple[Any, ...], output: Tensor) -> None:
        if not hook_state.enabled:
            return
        o = output.detach()
        flat = o.reshape(-1, o.shape[-1])                    # [n_tok, hidden]
        dst[layer_idx].append(flat.norm(dim=-1))             # [n_tok] (shared_mass — v1, unchanged)
        # v2.1 geometry rung: keep the branch output VECTOR (transient) to derive the per-token
        # shared_routed cosine at conversion. Same reshape/order as the norm → position-aligned.
        vdst[layer_idx].append(flat)                         # [n_tok, hidden]

    return hook_fn


# ── Activation-WRITE path (vmb battery arm A5 / A5-inv) ─────────────────────────
#
# Read hooks above OBSERVE the forward pass; the write path PERTURBS it: a
# forward_pre_hook on a decoder layer adds alpha * unit(vector) to the residual
# stream entering that layer, at chosen sequence positions. Used for steering
# during replay of a fixed continuation (matched-token channel) and free-gen
# dose ladders. alpha=0 must reproduce the unperturbed forward exactly.


@dataclass
class ResidualWriteSpec:
    """One residual-stream injection: add `alpha * vector/||vector||` to the
    hidden states entering decoder layer `layer_idx`.

    start_pos/end_pos select ABSOLUTE sequence positions (end exclusive;
    None = unbounded). For steering over generated tokens only, set
    start_pos = prompt_length. Positional gating reads the layer's
    `cache_position` kwarg (absolute positions, present on every HF call path
    in transformers 5.x), so ONE spec is valid under single-forward replay AND
    incremental generate() (per-step seq_len=1 with KV cache) — no per-step
    toggling. If cache_position is absent the hook falls back to local-index
    slicing (correct only for full-sequence forwards) and RAISES on an
    ambiguous incremental step rather than silently mis-injecting.

    start_pos is read at every hook call, so a caller may mutate it per
    generation (prompt lengths vary) on a hook registered once.
    """

    layer_idx: int
    vector: Tensor              # [hidden_dim]; normalized at registration
    alpha: float
    start_pos: int | None = None
    end_pos: int | None = None
    normalize: bool = True


def _make_residual_write_pre_hook(
    spec: ResidualWriteSpec,
    enabled_flag: dict[str, bool],
    stats: dict[str, Any] | None = None,
) -> Any:
    """forward_pre_hook (with_kwargs) injecting spec into the layer's input hidden states.

    The decoder layer receives hidden_states as args[0] (HF Llama-class layers).
    Returns modified (args, kwargs). Positional gating uses the `cache_position`
    kwarg (absolute positions) when present — valid under prefill, incremental
    seq_len=1 steps, and single-forward replay alike. Injection tensor is
    cast/moved lazily to the input's dtype/device once, then cached on the spec
    closure (cache is keyed by alpha too, so per-cell alpha mutation is safe).
    `stats` (optional) accumulates calls/positions-injected/saw-cache-position
    for smoke assertions.
    """
    cache: dict[str, Tensor] = {}

    def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not enabled_flag["enabled"] or spec.alpha == 0.0:
            return args, kwargs
        cache_position = kwargs.get("cache_position")
        if not args or not isinstance(args[0], Tensor):
            # Defensive: some architectures pass hidden_states as a kwarg.
            hs = kwargs.get("hidden_states")
            if hs is None:
                logger.warning("residual write hook: no hidden_states found; skipping injection")
                return args, kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = _inject(hs, cache_position)
            return args, kwargs
        new_args = (_inject(args[0], cache_position),) + tuple(args[1:])
        return new_args, kwargs

    def _inject(hs: Tensor, cache_position: Tensor | None) -> Tensor:
        # id(spec.vector) in the key: callers may swap the vector tensor on a live
        # spec (pilot gates iterate vectors); a swapped tensor must never reuse a
        # stale cached delta.
        key = f"{hs.device}_{hs.dtype}_{spec.alpha}_{id(spec.vector)}"
        if key not in cache:
            v = spec.vector.detach().to(device=hs.device, dtype=torch.float32)
            if spec.normalize:
                v = v / v.norm().clamp_min(1e-12)
            cache[key] = (spec.alpha * v).to(dtype=hs.dtype)
        delta = cache[key]
        bounded = spec.start_pos is not None or spec.end_pos is not None
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["saw_cache_position"] = stats.get("saw_cache_position", False) or (
                cache_position is not None
            )
        if cache_position is not None:
            # Absolute-position gating: one semantics for prefill, incremental
            # decoding, and single-forward replay.
            s = spec.start_pos if spec.start_pos is not None else 0
            e = spec.end_pos if spec.end_pos is not None else int(cache_position.max().item()) + 1
            # mask must live on the hidden-states device — cache_position can be on a different
            # device (multi-GPU / CPU cache_position), which crashes out[:, mask, :] with
            # "indices should be ... on the same device as the indexed tensor".
            mask = ((cache_position >= s) & (cache_position < e)).to(hs.device)  # [seq_len] bool
            n_inj = int(mask.sum().item())
            if stats is not None:
                stats["positions"] = stats.get("positions", 0) + n_inj
            if n_inj == 0:
                return hs
            if n_inj == hs.shape[1]:
                return hs + delta
            out = hs.clone()
            out[:, mask, :] = out[:, mask, :] + delta
            return out
        # Fallback: local-index slicing — correct ONLY for full-sequence forwards.
        if bounded and hs.shape[1] == 1:
            raise RuntimeError(
                "residual write hook: positionally-bounded spec on a seq_len=1 "
                "forward without cache_position — cannot determine the absolute "
                "position; refusing to silently mis-inject (incremental decoding "
                "requires the cache_position kwarg)"
            )
        s = spec.start_pos if spec.start_pos is not None else 0
        e = spec.end_pos if spec.end_pos is not None else hs.shape[1]
        s = max(0, min(s, hs.shape[1]))
        e = max(s, min(e, hs.shape[1]))
        if stats is not None:
            stats["positions"] = stats.get("positions", 0) + (e - s)
        if s == 0 and e == hs.shape[1]:
            return hs + delta
        out = hs.clone()
        out[:, s:e, :] = out[:, s:e, :] + delta
        return out

    return pre_hook


@dataclass
class ResidualWriteHandle:
    """Removable handle for a registered residual write; also supports toggling.

    `stats` accumulates {calls, positions, saw_cache_position} across forwards
    (reset via reset_stats) — smoke tests assert injected-position counts against
    expectation. `spec.start_pos`/`spec.alpha` may be mutated between generations
    on a live handle (read per hook call)."""

    spec: ResidualWriteSpec
    _handle: Any
    _enabled_flag: dict[str, bool]
    stats: dict[str, Any] = field(default_factory=dict)

    def disable(self) -> None:
        self._enabled_flag["enabled"] = False

    def enable(self) -> None:
        self._enabled_flag["enabled"] = True

    def remove(self) -> None:
        self._handle.remove()

    def reset_stats(self) -> None:
        self.stats.clear()


def attach_residual_write(model: Any, spec: ResidualWriteSpec) -> ResidualWriteHandle:
    """Register a residual-write injection on a BARE HF model (no LoadedModel needed).

    Validates layer_idx/hidden_dim against model.config. Used by generation-side
    scripts (run_gen_tokens) that never construct a LoadedModel; the LoadedModel
    method delegates here so both paths share one implementation.
    """
    layers = decoder_layers(model)
    n_layers = len(layers)
    # wrapper-aware: Gemma3ForConditionalGeneration's Gemma3Config nests dims under
    # text_config (no top-level hidden_size); Llama/Qwen/OLMo expose it directly.
    _cfg = model.config
    hidden_dim = int(getattr(_cfg, "hidden_size", None)
                     or getattr(getattr(_cfg, "text_config", None), "hidden_size"))
    if not 0 <= spec.layer_idx < n_layers:
        raise ValueError(
            f"residual write layer_idx {spec.layer_idx} out of range (model has {n_layers} layers)"
        )
    if spec.vector.numel() != hidden_dim:
        raise ValueError(
            f"residual write vector has {spec.vector.numel()} elements, expected hidden_dim={hidden_dim}"
        )
    enabled_flag = {"enabled": True}
    stats: dict[str, Any] = {}
    handle = layers[spec.layer_idx].register_forward_pre_hook(
        _make_residual_write_pre_hook(spec, enabled_flag, stats), with_kwargs=True
    )
    logger.info(
        f"Residual write registered: layer {spec.layer_idx}, alpha={spec.alpha}, "
        f"pos=[{spec.start_pos},{spec.end_pos})"
    )
    return ResidualWriteHandle(spec=spec, _handle=handle, _enabled_flag=enabled_flag, stats=stats)


# ── A7: MoE routing perturbation (vmb arm A7, M6 DeepSeek-V2-Lite class) ─────────
#
# Perturbs the ACTUAL routing computation (the recording pre-hook only OBSERVES it).
# DeepseekV2Moe.forward: router_logits = F.linear(h, gate.weight) → route_tokens_to_experts
# (softmax + torch.topk(k=self.top_k)) → experts(idx, w) + shared_experts(residuals). Injection
# points (verified transformers 5.3.0): top_k is `self.top_k = config.num_experts_per_tok` cached
# at init → patch the instance attr; router-noise + expert drops wrap route_tokens_to_experts;
# shared ablation is a forward hook on shared_experts. All perturbations are seed-deterministic
# (seed combined with layer idx) so a teacher-forced replay of a perturbed cell is itself
# bitwise-reproducible — the A7 paired contrast stays clean (SPEC-A7-BLOCK §2c).


@dataclass
class MoEPerturbSpec:
    """One A7 routing perturbation applied to every MoE layer for a cell.

    mode:
      'topk'          — set the effective num_experts_per_tok to `top_k` (cached-attr path).
      'noise'         — add seeded N(0, eps·sigma_logit[L]) to router_logits BEFORE softmax+topk.
      'shared_ablate' — zero the shared-experts branch output (§2c PAIR 1).
      'routed_ablate' — zero the routed-experts branch output (§2c PAIR 1; full-branch complement to
                        shared_ablate — the pilot says the two branches are mass-comparable).
      'drop_topm'     — zero the `m` largest-weight routed experts per token (targeting: removes mass).
      'drop_randm'    — zero `m` seeded-random selected experts per token (targeting control, m fixed).
    """

    mode: str
    top_k: int | None = None                          # 'topk'
    eps: float | None = None                          # 'noise'
    sigma_logit: dict[int, float] | None = None       # 'noise' — per-layer router-logit σ (from pilot)
    m: int | None = None                              # drops
    seed: int = 0


@dataclass
class MoEPerturbHandle:
    """Removable handle for an installed MoE perturbation (restores every patched module)."""

    spec: MoEPerturbSpec
    _restores: list[Any]

    def remove(self) -> None:
        for r in self._restores:
            r()
        self._restores = []


def _moe_modules(model: Any) -> list[tuple[int, Any]]:
    """(layer_idx, moe_module) for every MoE layer (those exposing route_tokens_to_experts)."""
    out: list[tuple[int, Any]] = []
    for i, layer in enumerate(decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "route_tokens_to_experts"):
            out.append((i, mlp))
    return out


def attach_moe_perturbation(model: Any, spec: MoEPerturbSpec) -> MoEPerturbHandle:
    """Install an A7 routing perturbation on every MoE layer of a bare HF model. Context-managed:
    call handle.remove() to restore the original routing exactly (alpha=0/identity reproduces)."""
    mods = _moe_modules(model)
    if not mods:
        raise ValueError("attach_moe_perturbation: model has no MoE layers (route_tokens_to_experts)")
    restores: list[Any] = []

    for l_idx, mlp in mods:
        if spec.mode == "topk":
            if spec.top_k is None:
                raise ValueError("MoEPerturbSpec mode 'topk' requires top_k")
            orig_k = mlp.top_k
            mlp.top_k = int(spec.top_k)
            restores.append(lambda m=mlp, k=orig_k: setattr(m, "top_k", k))

        elif spec.mode == "noise":
            if spec.eps is None or spec.sigma_logit is None:
                raise ValueError("MoEPerturbSpec mode 'noise' requires eps + sigma_logit")
            sig = float(spec.sigma_logit.get(l_idx, spec.sigma_logit.get(str(l_idx), 0.0)))
            orig = mlp.route_tokens_to_experts
            seed = (int(spec.seed) * 1000003) ^ (l_idx + 1)
            eps = float(spec.eps)

            def _noised(router_logits, _orig=orig, _sig=sig, _eps=eps, _seed=seed):
                if _sig > 0.0 and _eps > 0.0:
                    g = torch.Generator(device=router_logits.device)
                    g.manual_seed(_seed)
                    noise = torch.randn(router_logits.shape, generator=g,
                                        device=router_logits.device,
                                        dtype=router_logits.dtype) * (_eps * _sig)
                    router_logits = router_logits + noise
                return _orig(router_logits)

            mlp.route_tokens_to_experts = _noised
            restores.append(lambda m=mlp: m.__dict__.pop("route_tokens_to_experts", None))

        elif spec.mode in ("drop_topm", "drop_randm"):
            if spec.m is None:
                raise ValueError(f"MoEPerturbSpec mode {spec.mode!r} requires m")
            orig = mlp.route_tokens_to_experts
            mmode = spec.mode
            mval = int(spec.m)
            seed = (int(spec.seed) * 1000003) ^ (l_idx + 1)

            def _dropped(router_logits, _orig=orig, _mode=mmode, _m=mval, _seed=seed):
                topk_idx, topk_w = _orig(router_logits)     # [N, k]
                k = topk_w.shape[-1]
                mm = min(_m, k)
                if mm <= 0:
                    return topk_idx, topk_w
                w = topk_w.clone()
                if _mode == "drop_topm":
                    drop = torch.topk(topk_w, k=mm, dim=-1, sorted=False).indices
                else:  # drop_randm — seeded per-row random positions (same every replay)
                    g = torch.Generator(device=topk_w.device)
                    g.manual_seed(_seed)
                    rand = torch.rand(topk_w.shape, generator=g, device=topk_w.device)
                    drop = torch.topk(rand, k=mm, dim=-1, sorted=False).indices
                w.scatter_(-1, drop, 0.0)
                return topk_idx, w

            mlp.route_tokens_to_experts = _dropped
            restores.append(lambda m=mlp: m.__dict__.pop("route_tokens_to_experts", None))

        elif spec.mode == "shared_ablate":
            h = mlp.shared_experts.register_forward_hook(lambda mod, a, o: o * 0.0)
            restores.append(h.remove)

        elif spec.mode == "routed_ablate":
            # zero the routed-experts branch output: MoE.forward does experts(...) + shared_experts;
            # a forward hook on mlp.experts → ×0 leaves only the shared branch (§2c PAIR 1 complement).
            h = mlp.experts.register_forward_hook(lambda mod, a, o: o * 0.0)
            restores.append(h.remove)

        else:
            raise ValueError(f"unknown MoEPerturbSpec mode {spec.mode!r}")

    logger.info(f"A7 MoE perturbation installed: mode={spec.mode} on {len(mods)} MoE layers")
    return MoEPerturbHandle(spec=spec, _restores=restores)


@dataclass
class LoadedModel:
    """Bundle of model, tokenizer, hooks, and their state."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hook_state: HookState
    hook_handles: list[torch.utils.hooks.RemovableHook]
    config: ModelConfig

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("All hooks removed")

    def clear_hook_state(self) -> None:
        """Clear captured hook data without removing hooks."""
        self.hook_state.clear()

    def flush_hooks_to_cpu(self) -> None:
        """Batch-transfer hook tensors from GPU to CPU after generation."""
        self.hook_state.flush_to_cpu()

    def disable_hooks(self) -> None:
        """Temporarily disable hook capture (e.g. during calibration)."""
        self.hook_state.enabled = False

    def enable_hooks(self) -> None:
        """Re-enable hook capture."""
        self.hook_state.enabled = True

    def add_residual_write(self, spec: ResidualWriteSpec) -> ResidualWriteHandle:
        """Register an activation-WRITE injection (forward_pre_hook) on a decoder layer.

        Returns a handle with enable()/disable()/remove(). The caller owns the
        handle's lifecycle — writes are NOT tracked in hook_handles so read-hook
        teardown (remove_hooks) can never silently leave a perturbation armed,
        and vice versa. alpha=0 (or a disabled handle) must reproduce the
        unperturbed forward bit-for-bit (no-op path returns args unchanged).
        """
        return attach_residual_write(self.model, spec)


def load_model(
    config: ModelConfig | None = None,
    sampled_layers: list[int] | None = None,
    register_gate_hooks: bool = False,
    key_layers: list[int] | None = None,
    value_layers: list[int] | None = None,
    query_layers: list[int] | None = None,
    attn_output_layers: list[int] | None = None,
    adapter_path: str | None = None,
) -> LoadedModel:
    """Load model, tokenizer, and register hooks.

    adapter_path: optional PEFT/LoRA adapter dir — merged (merge_and_unload)
    BEFORE any hook registration, so hooks attach to the merged modules
    (A6 checkpoint replay; PEFT wraps target modules, so post-hook merging
    would strand hooks on replaced modules).

    Args:
        config: Model configuration. Defaults to ModelConfig().
        sampled_layers: Default layer set for gate_proj hooks and (unless
            overridden by key_layers) k_proj hooks. Defaults to
            {0, 7, 14, 18, 21, 24, 27}.
        register_gate_hooks: If True, also register hooks on gate_proj
            (SwiGLU MLP gate) at sampled layers. Captures pre-SiLU gate
            activations for gate sparsity/diversity features.
        key_layers: Layers for k_proj (pre-RoPE key) hooks. Defaults to
            sampled_layers (backward compatible). Pass all layers for the
            full v3 capture surface.
        value_layers: Layers for v_proj (value) hooks. Default None = no value
            capture. Pass all layers to bank the OV-circuit value surface.
        query_layers: Layers for q_proj (pre-RoPE query) hooks. Default None =
            no query capture. Pass sampled layers for offline QK-space geometry.
        attn_output_layers: Layers for o_proj (attention-output) hooks. Default
            None = no capture. Pass all layers for the vmb battery capture
            surface (attention-output cell; mlp_out derivable with hidden states).

    Returns:
        LoadedModel with everything wired up.
    """
    if config is None:
        config = ModelConfig()
    if sampled_layers is None:
        sampled_layers = [0, 7, 14, 18, 21, 24, 27]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    logger.info(f"Loading model: {config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )
    model.eval()

    if adapter_path is not None:
        from peft import PeftModel

        logger.info(f"Merging PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Resolve which layers get which hooks ──
    # Backward compatible: keys default to sampled_layers; values/queries are off
    # unless explicitly requested (the full v3 replay surface passes all/sampled).
    if key_layers is None:
        key_layers = list(sampled_layers)

    def _valid(layers: list[int]) -> list[int]:
        out: list[int] = []
        for i in layers:
            if 0 <= i < config.num_layers:
                out.append(i)
            else:
                logger.warning(
                    f"Skipping invalid layer index {i} (model has {config.num_layers} layers)"
                )
        return out

    key_layers = _valid(key_layers)
    value_layers = _valid(value_layers or [])
    query_layers = _valid(query_layers or [])
    attn_output_layers = _valid(attn_output_layers or [])

    hook_state = HookState()
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    is_dsv2 = "deepseek_v2" in str(getattr(getattr(model, "config", None), "model_type", ""))
    gate_hook_count = 0
    router_hook_count = 0

    if is_dsv2:
        # ── MLA + MoE capture (vmb arm A7, M6 DeepSeek-V2-Lite). No k_proj/v_proj (MLA); q_proj is
        # 192-d/head (not wave-1) → keys via c_KV latent, values/queries skipped. HOOK-AUDIT-PLAN §5. ──
        first_k_dense = int(getattr(model.config, "first_k_dense_replace", 0))
        kv_lora_rank = int(getattr(model.config, "kv_lora_rank", 512))
        dl = decoder_layers(model)
        for layer_idx in key_layers:                          # c_KV keys-analog (position-free latent)
            handle = dl[layer_idx].self_attn.kv_a_proj_with_mqa.register_forward_hook(
                _make_ckv_hook(layer_idx=layer_idx, hook_state=hook_state, kv_lora_rank=kv_lora_rank))
            hook_handles.append(handle)
        for layer_idx in attn_output_layers:                  # o_proj (attention-output surface)
            handle = dl[layer_idx].self_attn.o_proj.register_forward_hook(
                _make_o_proj_hook(layer_idx=layer_idx, hook_state=hook_state))
            hook_handles.append(handle)
        moe_layers = [l for l in _valid(list(sampled_layers)) if l >= first_k_dense]
        if register_gate_hooks:                               # SwiGLU gate = MoE shared-expert branch ONLY
            # The dense L0 gate_proj is BOTH a different substrate AND a different width (intermediate_size
            # 10944 vs moe_intermediate*n_shared 2816) — mixing it into gate_features' cross-layer cosine
            # crashes on the shape mismatch. Capture the uniform shared-branch gate on MoE layers only.
            for layer_idx in moe_layers:
                handle = dl[layer_idx].mlp.shared_experts.gate_proj.register_forward_hook(
                    _make_gate_proj_hook(layer_idx=layer_idx, hook_state=hook_state))
                hook_handles.append(handle)
                gate_hook_count += 1
        for layer_idx in moe_layers:                          # router dist (pre-hook) + shared/routed branch norms
            mlp = dl[layer_idx].mlp
            hook_handles.append(mlp.register_forward_pre_hook(
                _make_moe_router_prehook(layer_idx=layer_idx, hook_state=hook_state)))
            hook_handles.append(mlp.shared_experts.register_forward_hook(
                _make_branch_norm_hook(layer_idx=layer_idx, hook_state=hook_state, which="shared")))
            hook_handles.append(mlp.experts.register_forward_hook(
                _make_branch_norm_hook(layer_idx=layer_idx, hook_state=hook_state, which="routed")))
            router_hook_count += 1
        logger.info(
            f"Model loaded (deepseek_v2/MLA+MoE). Hooks — c_KV×{len(key_layers)}, "
            f"o_proj×{len(attn_output_layers)}, gate(shared)×{gate_hook_count}, router+branch×{router_hook_count}"
        )
    else:
        # ── Llama-class path (k_proj / v_proj / q_proj / o_proj / gate_proj) ──
        for layer_idx in key_layers:                          # Pre-RoPE keys (k_proj)
            k_proj = decoder_layers(model)[layer_idx].self_attn.k_proj
            handle = k_proj.register_forward_hook(_make_k_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state,
                num_kv_heads=config.num_kv_heads, head_dim=config.head_dim))
            hook_handles.append(handle)
        for layer_idx in value_layers:                        # Values (v_proj) — OV-circuit surface
            v_proj = decoder_layers(model)[layer_idx].self_attn.v_proj
            handle = v_proj.register_forward_hook(_make_v_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state,
                num_kv_heads=config.num_kv_heads, head_dim=config.head_dim))
            hook_handles.append(handle)
        for layer_idx in query_layers:                        # Pre-RoPE queries (q_proj)
            q_proj = decoder_layers(model)[layer_idx].self_attn.q_proj
            handle = q_proj.register_forward_hook(_make_q_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state,
                num_attention_heads=config.num_attention_heads, head_dim=config.head_dim))
            hook_handles.append(handle)
        for layer_idx in attn_output_layers:                  # Attention output (o_proj)
            o_proj = decoder_layers(model)[layer_idx].self_attn.o_proj
            handle = o_proj.register_forward_hook(_make_o_proj_hook(
                layer_idx=layer_idx, hook_state=hook_state))
            hook_handles.append(handle)
        if register_gate_hooks:                               # SwiGLU gate_proj (sampled layers)
            for layer_idx in _valid(list(sampled_layers)):
                gate_proj = decoder_layers(model)[layer_idx].mlp.gate_proj
                handle = gate_proj.register_forward_hook(_make_gate_proj_hook(
                    layer_idx=layer_idx, hook_state=hook_state))
                hook_handles.append(handle)
                gate_hook_count += 1
        logger.info(
            f"Model loaded. Hooks — k_proj×{len(key_layers)}, v_proj×{len(value_layers)}, "
            f"q_proj×{len(query_layers)}, o_proj×{len(attn_output_layers)}, gate_proj×{gate_hook_count}"
        )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hook_state=hook_state,
        hook_handles=hook_handles,
        config=config,
    )
