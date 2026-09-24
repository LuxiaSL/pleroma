"""A tiny CPU residual stack shaped like an HF decoder, for the steering pins.

Just enough structure for ``anamnesis.extraction.model_loader``'s
``attach_residual_write``: ``model.config.hidden_size``, ``model.model.layers``
(a ModuleList), and layers that take ``hidden_states`` positionally plus a
``cache_position`` kwarg (the HF call convention the write hook gates on).

Each layer records the hidden states it RECEIVED, so a test can read exactly
what a forward-pre-hook wrote into the residual stream at that site. Not code
under test — a fixture.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

HIDDEN = 8
N_LAYERS = 6


class RecordingLayer(nn.Module):
    """``out = in + tanh(W in)``; remembers every input it was handed."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.seen: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor, cache_position: torch.Tensor | None = None,
                **kwargs: object) -> torch.Tensor:
        self.seen.append(hidden_states.detach().clone())
        return hidden_states + torch.tanh(self.lin(hidden_states))


class _Inner(nn.Module):
    def __init__(self, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([RecordingLayer(hidden) for _ in range(n_layers)])


class ToyDecoder(nn.Module):
    """``forward(h0, cache_position)`` runs the stack and returns the last
    hidden state. Deterministic weights (seeded)."""

    def __init__(self, hidden: int = HIDDEN, n_layers: int = N_LAYERS, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=n_layers)
        self.model = _Inner(hidden, n_layers)
        with torch.no_grad():
            for layer in self.model.layers:
                layer.lin.weight.copy_(torch.randn(hidden, hidden, generator=g) * 0.3)
                layer.lin.bias.copy_(torch.randn(hidden, generator=g) * 0.1)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.layers

    def reset(self) -> None:
        for layer in self.layers:
            layer.seen.clear()

    @torch.no_grad()
    def forward(self, h: torch.Tensor, cache_position: torch.Tensor | None = None) -> torch.Tensor:
        self.reset()
        for layer in self.layers:
            h = layer(h, cache_position=cache_position)
        return h

    def inputs(self) -> list[torch.Tensor]:
        """Per-layer input of the LAST forward (``hidden_states[i]`` in HF terms)."""
        return [layer.seen[-1] for layer in self.layers]


def prefill_input(seq: int, hidden: int = HIDDEN, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, hidden, generator=g), torch.arange(seq)


def step_input(pos: int, hidden: int = HIDDEN, seed: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """One incremental decode step at ABSOLUTE position ``pos`` (seq_len 1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, hidden, generator=g), torch.tensor([pos])
