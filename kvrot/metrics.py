"""North-star metric: behavioural drift of a rotated cache vs. the *full-context* cache.

``exact-vs-full`` is operationalised as the per-step KL divergence between the
next-token distribution produced from the full cache (reference) and from the rotated
cache, plus top-1 agreement. KL is measured in nats; 0 ⇒ behaviourally identical.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import Tensor


def stepwise_kl(ref_logits: Tensor, other_logits: Tensor) -> Tensor:
    """KL(p_ref ‖ p_other) per step. Logits ``[steps, vocab]`` → ``[steps]`` (nats)."""
    if ref_logits.shape != other_logits.shape:
        raise ValueError(f"shape mismatch: {ref_logits.shape} vs {other_logits.shape}")
    ref_logp = torch.log_softmax(ref_logits.float(), dim=-1)
    oth_logp = torch.log_softmax(other_logits.float(), dim=-1)
    return (ref_logp.exp() * (ref_logp - oth_logp)).sum(dim=-1)


def top1_agreement(ref_logits: Tensor, other_logits: Tensor) -> Tensor:
    """1.0 where argmax matches, else 0.0. Logits ``[steps, vocab]`` → ``[steps]``."""
    return (ref_logits.argmax(-1) == other_logits.argmax(-1)).to(torch.float32)


class DriftReport(BaseModel):
    """Summary of behavioural drift for one rotation variant vs. the full-context run."""

    label: str = ""
    n_steps: int
    mean_kl: float
    max_kl: float
    p95_kl: float
    top1_agreement: float = Field(description="fraction of steps whose argmax matches the reference")
    per_step_kl: list[float] = Field(default_factory=list)

    @classmethod
    def from_logits(
        cls, ref_logits: Tensor, other_logits: Tensor, *, label: str = ""
    ) -> "DriftReport":
        kl = stepwise_kl(ref_logits, other_logits)
        agree = top1_agreement(ref_logits, other_logits)
        n = int(kl.shape[0])
        p95 = torch.quantile(kl, 0.95).item() if n > 1 else float(kl.max().item())
        return cls(
            label=label,
            n_steps=n,
            mean_kl=float(kl.mean().item()),
            max_kl=float(kl.max().item()),
            p95_kl=float(p95),
            top1_agreement=float(agree.mean().item()),
            per_step_kl=[float(x) for x in kl.tolist()],
        )

    def one_line(self) -> str:
        return (
            f"{self.label:<28} steps={self.n_steps:>3} "
            f"mean_KL={self.mean_kl:.4e} p95={self.p95_kl:.4e} "
            f"max={self.max_kl:.4e} top1={self.top1_agreement:6.1%}"
        )


def kv_deviation(reused_key: Tensor, true_key: Tensor) -> Tensor:
    """Per-token KV deviation ‖reused − true‖₂ (Tier-3 recompute selector, CacheBlend's HKVD).

    Inputs ``[..., S, head_dim]`` → per-token deviation ``[S]`` (mean over heads/dims).
    """
    diff = (reused_key.float() - true_key.float()).norm(dim=-1)  # [..., S]
    reduce_dims = tuple(range(diff.dim() - 1))
    return diff.mean(dim=reduce_dims) if reduce_dims else diff
