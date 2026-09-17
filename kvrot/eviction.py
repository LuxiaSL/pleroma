"""Eviction policies → (keep-indices, new contiguous positions).

Phase 1 implements the Tier-0/1 policies needed to demonstrate the core result:

* ``none``        — keep everything (control).
* ``oldest``      — drop the first ``evict_count`` tokens, **including sinks**. This is
                    the naive rolling-window baseline that StreamingLLM shows is
                    catastrophic (ppl 5.4 → 5158). We keep it precisely to reproduce
                    that failure as a sanity check.
* ``sink_window`` — keep ``num_sink_tokens`` leading sinks, drop the next
                    ``evict_count`` tokens, keep the rest. The Tier-1 recipe.

Tier-2 importance-aware eviction (H2O/SnapKV-style) plugs in here later by returning a
different keep-index set from attention statistics.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import EvictionSpec


def compute_keep_indices(seq_len: int, spec: EvictionSpec) -> Tensor:
    """Indices (LongTensor, ascending) of tokens to retain for a sequence of ``seq_len``."""
    if spec.evict_count < 0:
        raise ValueError("evict_count must be >= 0")

    if spec.policy == "none" or spec.evict_count == 0:
        return torch.arange(seq_len, dtype=torch.long)

    if spec.policy == "oldest":
        ev = min(spec.evict_count, seq_len)
        return torch.arange(ev, seq_len, dtype=torch.long)

    if spec.policy == "sink_window":
        sinks = min(spec.num_sink_tokens, seq_len)
        start = sinks + spec.evict_count
        if start >= seq_len:
            # Nothing left to keep past the evicted block; keep only the sinks.
            return torch.arange(sinks, dtype=torch.long)
        return torch.cat(
            [torch.arange(sinks, dtype=torch.long), torch.arange(start, seq_len, dtype=torch.long)]
        )

    raise ValueError(f"unknown eviction policy {spec.policy!r}")


def importance_keep_indices(
    importance: Tensor,
    *,
    num_sink_tokens: int,
    recent_window: int,
    budget: int,
) -> Tensor:
    """Tier-2 importance-aware keep set: always keep ``num_sink_tokens`` sinks + the last
    ``recent_window`` tokens, then fill up to ``budget`` with the highest-importance
    remaining tokens (H2O/SnapKV-style heavy hitters). Returns ascending indices.

    ``importance`` is a per-token score (e.g. accumulated attention received). Unlike the
    contiguous ``sink_window`` policy, this can keep a salient token from deep in the
    middle and drop a low-value recent-ish one — eviction by importance, not by age.
    """
    c = int(importance.shape[0])
    if budget >= c:
        return torch.arange(c, dtype=torch.long)
    device = importance.device
    keep = torch.zeros(c, dtype=torch.bool, device=device)
    s = min(max(num_sink_tokens, 0), c)
    r = min(max(recent_window, 0), c)
    if s:
        keep[:s] = True
    if r:
        keep[c - r :] = True
    remaining = budget - int(keep.sum().item())
    if remaining > 0:
        cand = (~keep).nonzero(as_tuple=False).squeeze(-1)
        if cand.numel() > 0:
            k = min(remaining, int(cand.numel()))
            top = cand[torch.topk(importance[cand], k).indices]
            keep[top] = True
    return keep.nonzero(as_tuple=False).squeeze(-1).sort().values


def new_positions_for(
    keep_indices: Tensor, old_positions: Tensor, *, recompact: bool
) -> Tensor:
    """New absolute positions for the kept tokens.

    With ``recompact`` (StreamingLLM-style), survivors are renumbered to a contiguous
    ``0..K-1`` so all relative distances (esp. sink↔window) stay inside the trained
    range. Without it, original positions are preserved (leaves a positional gap).
    """
    if recompact:
        return torch.arange(keep_indices.shape[0], dtype=old_positions.dtype, device=old_positions.device)
    return old_positions.index_select(0, keep_indices.to(old_positions.device))
