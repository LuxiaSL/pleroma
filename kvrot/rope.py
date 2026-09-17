"""Tier 0 — exact RoPE re-rotation of stored (pre-rotated) keys.

In vLLM/SGLang *and* in HF transformers, RoPE is applied to keys **before** they are
written to the KV cache (the attention kernel reads keys as-is). So the cached key for
a token at absolute position ``p`` is ``R(p) k`` where ``R(p)`` is the block-diagonal
rotary rotation by angle ``p * inv_freq_i`` in each 2-D channel pair.

To move that token to a new position ``p'`` we must left-multiply the stored key by
``R(p' - p)``. This is exact **iff** the per-dimension angle is linear in position, i.e.
``inv_freq`` does not depend on position — which holds for standard RoPE and for the
*static* llama3 frequency rescaling (it rescales ``inv_freq`` per-dimension, not per
position). It does **not** hold for dynamic-NTK schemes where the base changes with the
running sequence length; we guard against that in :func:`assert_rotation_homomorphism`.

Values are position-free and are never touched.

The rotation here is the *positional* part only. It is bit-exact. The residual
*content* drift (survivor KV still encodes the evicted tokens) is what the measurement
harness quantifies — that is the approximate part, and it is intentionally separate.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def default_inv_freq(head_dim: int, theta: float, *, device=None) -> Tensor:
    """Standard RoPE inverse frequencies, shape ``[head_dim // 2]`` (float32)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (idx / head_dim))


def llama3_inv_freq(
    head_dim: int,
    theta: float,
    *,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device=None,
) -> Tensor:
    """llama3 ('rope_type': 'llama3') static frequency rescaling.

    Mirrors transformers' ``_compute_llama3_parameters``. Static per-dimension, so the
    R(p)-is-linear-in-p property still holds and re-rotation stays exact.
    """
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
    """HF convention: pair dim ``i`` with ``i + d/2`` (not adjacent dims)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(delta_pos: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    """cos/sin for an angle ``delta_pos[:, None] * inv_freq[None, :]``, shape ``[S, head_dim]``."""
    freqs = torch.outer(delta_pos.to(torch.float32), inv_freq.to(torch.float32))  # [S, d/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, d]
    return emb.cos(), emb.sin()


def reindex_keys(key: Tensor, delta_pos: Tensor, inv_freq: Tensor) -> Tensor:
    """Apply an additional RoPE rotation of ``delta_pos`` (per token) to already-rotated keys.

    Args:
        key: ``[..., S, head_dim]`` (e.g. ``[batch, kv_heads, S, head_dim]``), already
            rotated at its original positions.
        delta_pos: ``[S]`` = ``new_pos - old_pos`` per token (negative shifts earlier).
        inv_freq: ``[head_dim // 2]`` — the *same* frequencies the model used at storage.

    Returns:
        Keys re-rotated to their new positions, same shape/dtype as ``key``.
    """
    if key.shape[-2] != delta_pos.shape[0]:
        raise ValueError(
            f"delta_pos length {delta_pos.shape[0]} != key seq dim {key.shape[-2]}"
        )
    cos, sin = _cos_sin(delta_pos.to(key.device), inv_freq.to(key.device))  # [S, d]
    while cos.dim() < key.dim():  # broadcast over batch / heads
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    # Compute in float32 for numerical fidelity, then cast back.
    kf = key.float()
    out = kf * cos + rotate_half(kf) * sin
    return out.to(key.dtype)


def assert_rotation_homomorphism(inv_freq: Tensor, *, atol: float = 1e-5) -> None:
    """Sanity check that R(a)·R(b) == R(a+b) for these frequencies.

    This is what makes ``reindex_keys`` exact. It holds for static RoPE/llama3 and
    would fail loudly for a frequency table that secretly depends on position.
    """
    d = inv_freq.shape[0] * 2
    k = torch.randn(1, 1, 3, d, dtype=torch.float32)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    lhs = reindex_keys(reindex_keys(k, a, inv_freq), b, inv_freq)
    rhs = reindex_keys(k, a + b, inv_freq)
    if not torch.allclose(lhs, rhs, atol=atol):
        raise AssertionError(
            "RoPE frequencies are not a rotation homomorphism — re-rotation would be "
            "inexact (dynamic-NTK scaling?). Max err="
            f"{(lhs - rhs).abs().max().item():.3e}"
        )
