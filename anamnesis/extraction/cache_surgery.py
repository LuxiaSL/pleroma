"""KV-cache surgery for ARM A4 (state-surgery): eviction / re-rotation / recompute.

Vendored from kv-rotation `src/kvrot/{rope,snapshot,eviction}.py` (exp11 machinery,
bit-exact rotation verified there) with battery-specific additions:
  - inv_freq built FROM THE MODEL CONFIG with a hard gate on unknown RoPE schemes
    (addendum 12h: the RoPE-config verification gate on M3/M4 conditional inclusion —
    fires at load time, never after a confusing result);
  - middle-region keep geometry (A4 block: sink + recent protected, contiguous
    middle block evicted).

RoPE background (kvrot docstring, condensed): HF caches store keys ALREADY rotated
at their original positions. Moving a token from position p to p' left-multiplies
its key by R(p'-p); exact iff inv_freq is position-independent (standard RoPE and
static llama3 rescaling; NOT dynamic-NTK — the homomorphism assert catches that).
Values are position-free and never touched.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ── RoPE (vendored, exact) ──────────────────────────────────────────────────────

def default_inv_freq(head_dim: int, theta: float, *, device=None) -> Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (idx / head_dim))


def llama3_inv_freq(
    head_dim: int, theta: float, *, factor: float, low_freq_factor: float,
    high_freq_factor: float, original_max_position_embeddings: int, device=None,
) -> Tensor:
    """llama3 static frequency rescaling (mirrors transformers' _compute_llama3_parameters)."""
    inv_freq = default_inv_freq(head_dim, theta, device=device)
    old_ctx = original_max_position_embeddings
    low_freq_wavelen = old_ctx / low_freq_factor
    high_freq_wavelen = old_ctx / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    return torch.where(is_medium, smoothed, inv_freq_llama)


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(delta_pos: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    freqs = torch.outer(delta_pos.to(torch.float32), inv_freq.to(torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def reindex_keys(key: Tensor, delta_pos: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotate already-rotated keys by delta_pos (per token). [..., S, head_dim]."""
    if key.shape[-2] != delta_pos.shape[0]:
        raise ValueError(f"delta_pos length {delta_pos.shape[0]} != key seq dim {key.shape[-2]}")
    cos, sin = _cos_sin(delta_pos.to(key.device), inv_freq.to(key.device))
    while cos.dim() < key.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    kf = key.float()
    return (kf * cos + rotate_half(kf) * sin).to(key.dtype)


def assert_rotation_homomorphism(inv_freq: Tensor, *, atol: float = 1e-5) -> None:
    """R(a)·R(b) == R(a+b) — what makes re-rotation exact. Fails loudly on dynamic-NTK."""
    d = inv_freq.shape[0] * 2
    k = torch.randn(1, 1, 3, d, dtype=torch.float32)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    lhs = reindex_keys(reindex_keys(k, a, inv_freq), b, inv_freq)
    rhs = reindex_keys(k, a + b, inv_freq)
    if not torch.allclose(lhs, rhs, atol=atol):
        raise AssertionError(
            "RoPE frequencies are not a rotation homomorphism — re-rotation would be "
            f"inexact. Max err={(lhs - rhs).abs().max().item():.3e}"
        )


def inv_freq_from_config(config) -> Tensor:
    """Build inv_freq from an HF model config, with the 12h RoPE gate.

    Reads EACH model's rope_theta/rope_scaling (never assumes Llama's values).
    Supported: no scaling (Qwen2.5/OLMo-2 class) and static 'llama3' rescaling.
    Anything else raises — the gate fires BEFORE any surgery, not after a
    confusing result.
    """
    # Multimodal wrappers (e.g. Gemma3ForConditionalGeneration's Gemma3Config) nest the
    # transformer/RoPE params under .text_config; unwrap so hidden_size/rope_theta resolve
    # (no-op for the flat Llama/Qwen/OLMo configs, which have no text_config). 2026-07-18.
    tc = getattr(config, "text_config", None)
    if tc is not None and getattr(tc, "num_attention_heads", None) is not None:
        config = tc
    # Gemma-3 dual RoPE: global layers rotate with rope_theta (1e6), sliding layers with
    # rope_local_base_freq (1e4). A single inv_freq table cannot be silently right for both
    # (the 14e anti-silent-default rule) — surface it so the ROTATE surgery kind is scoped
    # to global-theta only; naive/recompute (which need no reindex) are unaffected.
    local_base = getattr(config, "rope_local_base_freq", None)
    if local_base is not None:
        logger.warning(
            f"dual-RoPE config (rope_local_base_freq={local_base}): this table is the GLOBAL "
            "theta only. ROTATE re-rotation is correct for global layers; sliding layers use a "
            "different base — use naive/recompute surgery, or extend reindex to per-layer inv_freq.")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    # transformers 5.x moved the RoPE params INTO the rope_scaling / rope_parameters dict; the
    # top-level `rope_theta` attribute may be absent (the 14e bug: getattr's 10000.0 default fired
    # silently under tf 5.3, so llama3 rescaling ran on theta=10000 not 500000). Read theta from
    # EITHER layout, and RAISE if it is findable nowhere — never default (silent defaults are how
    # this bug happened).
    scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
    theta = getattr(config, "rope_theta", None)
    if theta is None and isinstance(scaling, dict):
        theta = scaling.get("rope_theta", scaling.get("theta"))
    if theta is None:
        raise ValueError(
            "RoPE gate FAILED (14e): rope_theta not found at config top-level NOR inside "
            f"rope_scaling/rope_parameters ({scaling!r}) — refusing to default to 10000 "
            "(the silent default that corrupted every battery ROTATE row). Locate theta or extend.")
    theta = float(theta)
    if scaling is None:
        logger.info(f"RoPE gate: standard RoPE theta={theta} head_dim={head_dim}")
        inv = default_inv_freq(head_dim, theta)
    else:
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type == "llama3":
            logger.info(f"RoPE gate: llama3 static rescaling theta={theta} {scaling}")
            inv = llama3_inv_freq(
                head_dim, theta,
                factor=float(scaling["factor"]),
                low_freq_factor=float(scaling["low_freq_factor"]),
                high_freq_factor=float(scaling["high_freq_factor"]),
                original_max_position_embeddings=int(scaling["original_max_position_embeddings"]),
            )
        elif rope_type in ("default", "linear") and float(scaling.get("factor", 1.0)) == 1.0:
            inv = default_inv_freq(head_dim, theta)
        else:
            raise ValueError(
                f"RoPE gate FAILED: unsupported rope_scaling {scaling!r} — re-rotation "
                "exactness not established for this scheme (12h gate; extend deliberately)"
            )
    assert_rotation_homomorphism(inv)
    return inv


def _live_inv_freq(model) -> Tensor | None:
    """The model's OWN inv_freq buffer — the frequencies the runtime actually rotates with
    (exp11's `_find_inv_freq` pattern, via named_buffers)."""
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq"):
            return buf.detach().float().cpu()
    return None


def operative_inv_freq(model) -> Tensor:
    """The inv_freq table surgery MUST rotate with, value-gated (14e).

    The 12h RoPE gate was SCHEME-level (llama3 vs NTK) + a homomorphism check — but ANY
    static frequency table is a homomorphism, so a wrong-theta table passed silently. This
    gate is VALUE-level: locate the LIVE buffer, assert it equals the config reconstruction
    to rtol=1e-6, and return the LIVE buffer as the operative table (the reconstruction is now
    only the cross-check). Abort on mismatch — never rotate with frequencies the runtime doesn't.
    """
    computed = inv_freq_from_config(model.config)
    live = _live_inv_freq(model)
    if live is None:
        logger.warning("14e value gate: no live inv_freq buffer on the model — falling back to "
                       "config reconstruction (the gate cannot run; ensure rotary buffers materialize)")
        return computed
    live = live.to(computed.dtype)
    if not torch.allclose(live, computed, rtol=1e-6, atol=1e-8):
        max_abs = float((live - computed).abs().max())
        max_rel = float(((live - computed).abs() / live.abs().clamp_min(1e-12)).max())
        raise ValueError(
            f"RoPE VALUE gate FAILED (14e): live inv_freq buffer != config reconstruction "
            f"(max|Δ|={max_abs:.3e}, max rel={max_rel:.1f}×). The runtime rotates with different "
            "frequencies than surgery would apply — this is exactly the ROTATE-corruption bug. Abort.")
    logger.info(f"14e value gate PASS: live inv_freq == config reconstruction "
                f"(n={live.numel()}, rtol=1e-6); using the LIVE buffer as operative")
    return live


# ── KVSnapshot + surgery (vendored) ─────────────────────────────────────────────

@dataclass
class KVSnapshot:
    """Engine-agnostic cache view; surgery is pure tensor math (CPU-testable)."""

    keys: list[Tensor]       # per layer [B, kv_heads, S, head_dim]
    values: list[Tensor]     # per layer [B, kv_heads, S, head_dim]
    positions: list[Tensor]  # per layer [S] absolute positions (long)

    def __post_init__(self) -> None:
        if not (len(self.values) == len(self.positions) == len(self.keys)):
            raise ValueError("KVSnapshot: per-layer list lengths must match")
        for li in range(len(self.keys)):
            if self.keys[li].shape[-2] != self.positions[li].shape[0]:
                raise ValueError(f"layer {li}: positions length != key seq len")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def seq_len(self, layer: int = 0) -> int:
        return int(self.keys[layer].shape[-2])

    @property
    def next_position(self) -> int:
        return int(max(int(p.max().item()) for p in self.positions)) + 1


def evict(snapshot: KVSnapshot, keep_indices: Tensor) -> KVSnapshot:
    """Drop all but keep_indices along the sequence dim (positions carried unchanged)."""
    keys, values, positions = [], [], []
    for li in range(snapshot.num_layers):
        idx = keep_indices.to(snapshot.keys[li].device)
        keys.append(snapshot.keys[li].index_select(-2, idx))
        values.append(snapshot.values[li].index_select(-2, idx))
        positions.append(snapshot.positions[li].to(idx.device).index_select(0, idx))
    return KVSnapshot(keys=keys, values=values, positions=positions)


def reindex(snapshot: KVSnapshot, new_positions: Tensor, inv_freq: Tensor) -> KVSnapshot:
    """Re-rotate keys so each token sits at its new position (exact). Values untouched."""
    keys, positions = [], []
    for li in range(snapshot.num_layers):
        new_p = new_positions.to(snapshot.positions[li].device)
        if new_p.shape[0] != snapshot.positions[li].shape[0]:
            raise ValueError(f"layer {li}: new_positions length != current seq len")
        delta = (new_p - snapshot.positions[li]).to(torch.float32)
        keys.append(reindex_keys(snapshot.keys[li], delta, inv_freq))
        positions.append(new_p.to(snapshot.positions[li].dtype))
    return KVSnapshot(keys=keys, values=[v for v in snapshot.values], positions=positions)


def from_hf_cache(past_key_values, *, positions: Tensor) -> KVSnapshot:
    keys, values = _extract_kv(past_key_values)
    return KVSnapshot(
        keys=list(keys), values=list(values),
        positions=[positions.to(keys[i].device) for i in range(len(keys))],
    )


def _extract_kv(past_key_values):
    if hasattr(past_key_values, "layers"):  # transformers >= 5
        layers = [ly for ly in past_key_values.layers if getattr(ly, "keys", None) is not None]
        return [ly.keys for ly in layers], [ly.values for ly in layers]
    if hasattr(past_key_values, "key_cache"):  # transformers < 5
        return list(past_key_values.key_cache), list(past_key_values.value_cache)
    if isinstance(past_key_values, (list, tuple)):
        return [l[0] for l in past_key_values], [l[1] for l in past_key_values]
    raise TypeError(f"unsupported cache type: {type(past_key_values)!r}")


def to_hf_dynamic_cache(snapshot: KVSnapshot):
    from transformers import DynamicCache

    cache = DynamicCache()
    for li in range(snapshot.num_layers):
        cache.update(snapshot.keys[li], snapshot.values[li], li)
    return cache


# ── A4 battery eviction geometry ────────────────────────────────────────────────

def middle_region_keep(
    context_len: int, evict_frac: float, *, num_sinks: int = 4, recent_protect: int = 32,
) -> Tensor:
    """Keep-indices for the A4 cells: evict ONE contiguous middle block of
    round(evict_frac * context_len) tokens, centered in the evictable region
    [num_sinks, context_len - recent_protect); sinks + recent tail always kept.
    """
    n_evict = int(round(evict_frac * context_len))
    lo, hi = num_sinks, context_len - recent_protect
    if n_evict <= 0:
        return torch.arange(context_len, dtype=torch.long)
    if hi - lo < n_evict:
        raise ValueError(
            f"cannot evict {n_evict} of {context_len} tokens with sinks={num_sinks}, "
            f"recent={recent_protect} protected (evictable={hi - lo})"
        )
    center = (lo + hi) // 2
    start = max(lo, min(center - n_evict // 2, hi - n_evict))
    evicted = set(range(start, start + n_evict))
    keep = [i for i in range(context_len) if i not in evicted]
    return torch.tensor(keep, dtype=torch.long)


# ── A4b dialogue eviction geometry (turn-aligned oldest-first) ───────────────────
# Ported from kv-rotation `src/kvrot/chat.py` (turn_token_spans / oldest_turns_to_evict /
# turn_keep_indices — exp11's dialogue substrate). ARM-A4b §Ops: turn-aligned oldest-first
# eviction, protect system + last 2 turns + N sinks. Pure tensor/int math (CPU-testable),
# matching this module's design constraint. kv-rotation owns the model/chat substrate; this
# is the anamnesis-side instrument port (pointers, not shared state — cross-project rule).


@dataclass(frozen=True)
class TurnSpan:
    """One rendered turn's half-open token span [start, end) in the full context."""

    index: int   # position in the message list
    role: str    # "system" | "user" | "assistant"
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


class TemplateNotPrefixStableError(RuntimeError):
    """The chat template rewrote earlier tokens when a turn was appended — turn spans
    (recovered by incremental-prefix rendering) would be misaligned. Fails loudly."""


def _render_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(ids, "keys") and "input_ids" in ids:      # transformers 5.x BatchEncoding
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):                 # batched form
        ids = ids[0]
    return list(ids)


def turn_token_spans(tokenizer, messages: list[dict], *,
                     add_generation_prompt: bool = False) -> tuple[Tensor, list["TurnSpan"]]:
    """Render messages and recover per-turn token spans → (context_ids [1,T], spans).

    Span i covers everything the template emitted for message i (headers/separators
    included). With add_generation_prompt, the trailing generation header is NOT part of
    any span. Raises TemplateNotPrefixStableError if the template rewrites history."""
    if not messages:
        raise ValueError("messages must be non-empty")
    spans: list[TurnSpan] = []
    prev: list[int] = []
    for i in range(1, len(messages) + 1):
        gen = add_generation_prompt and i == len(messages)
        cur = _render_ids(tokenizer, messages[:i], add_generation_prompt=gen)
        boundary = len(prev)
        if gen:
            bare = _render_ids(tokenizer, messages[:i], add_generation_prompt=False)
            if bare != cur[: len(bare)]:
                raise TemplateNotPrefixStableError(
                    "generation prompt is not a pure suffix of the rendered conversation")
            end = len(bare)
        else:
            end = len(cur)
        if prev != cur[:boundary]:
            raise TemplateNotPrefixStableError(
                f"rendering messages[:{i}] rewrote tokens emitted for messages[:{i - 1}]")
        spans.append(TurnSpan(index=i - 1, role=str(messages[i - 1].get("role", "?")),
                              start=boundary, end=end))
        prev = cur
    return torch.tensor([prev], dtype=torch.long), spans


def oldest_turns_to_evict(spans: list["TurnSpan"], *, target_tokens: int,
                          protect_roles: tuple[str, ...] = ("system",),
                          protect_last: int = 2) -> list[int]:
    """Whole turns to evict, oldest first, until target_tokens are freed; protect_roles
    turns and the final protect_last turns are never evicted. May return fewer than
    requested (protections leave nothing else) — the caller reports 'unreachable', per
    ARM-A4b §Cells (never silently clipped)."""
    if target_tokens <= 0:
        return []
    cutoff = max(0, len(spans) - max(protect_last, 0))
    chosen: list[int] = []
    freed = 0
    for sp in spans:
        if sp.index >= cutoff:
            break
        if sp.role in protect_roles:
            continue
        chosen.append(sp.index)
        freed += sp.n_tokens
        if freed >= target_tokens:
            break
    return chosen


def turn_keep_indices(spans: list["TurnSpan"], evict_turns: list[int], seq_len: int, *,
                      num_sink_tokens: int = 4) -> Tensor:
    """Keep-index set for turn-aligned eviction (ascending LongTensor). Every token outside
    the evicted turns is kept; the first num_sink_tokens are kept unconditionally (sinks);
    trailing tokens past the last span (e.g. a generation prompt) are kept."""
    last_end = max((sp.end for sp in spans), default=0)
    if seq_len < last_end:
        raise ValueError(f"seq_len {seq_len} shorter than final span end {last_end}")
    evict = set(evict_turns)
    unknown = evict - {sp.index for sp in spans}
    if unknown:
        raise ValueError(f"evict_turns reference unknown turn indices: {sorted(unknown)}")
    keep = torch.ones(seq_len, dtype=torch.bool)
    for sp in spans:
        if sp.index in evict:
            keep[sp.start: sp.end] = False
    if num_sink_tokens > 0:
        keep[: min(num_sink_tokens, seq_len)] = True
    return keep.nonzero(as_tuple=False).squeeze(-1)
