"""Exp B1 torch side: g, the differentiable injector, and the teacher-forced NLL.

**This module imports torch at module scope and is therefore imported LAZILY**, from
inside ``main()`` of ``train_g`` / ``eval_g``. That is the house deferred-import rule
kept honest: the two entry points stay ``--help``-able on a laptop, and the torch
code they share lives in one file instead of being duplicated (and drifting) across
training and evaluation, where a divergence between the two would silently change
what the headline contrast means.

── g ──────────────────────────────────────────────────────────────────────────
``SignatureToBias``: LayerNorm(D) -> Linear(D, hidden) -> GELU -> Linear(hidden,
n_layers * hidden_dim), reshaped to ``[B, n_layers, hidden_dim]``. The FINAL layer
is zero-initialised, weight and bias, so an untrained g emits exactly zero bias and
the run starts at a bit-identical no-op against the alpha=0 arm. Any improvement is
then something training did, not something initialisation did.

── injection ─────────────────────────────────────────────────────────────────
``TrainableResidualInjector`` registers a forward hook per selected decoder layer
and adds ``bias[b, slot]`` to that layer's output hidden states at CONTINUATION
positions only, per sample (batch entries have different ``prompt_length``, so the
gate is a ``[B, S]`` mask, not a scalar bound).

Why not ``attach_residual_write``: anamnesis'
``model_loader.ResidualWriteSpec``/``attach_residual_write`` (model_loader.py:391-570)
takes a FIXED tensor, ``.detach()``es it inside the hook, normalises it and scales
by a scalar alpha. All three are right for a steering *dose ladder* and all three
are wrong here — B1 needs the gradient to reach g through the frozen model's graph,
and needs the bias magnitude to be part of what g learns. We reuse its *semantics*
(add to the residual stream, gate on absolute position, alpha=0 reproduces the
unperturbed forward exactly) and none of its code.

Two deliberate differences from that reference, both load-bearing:

  - It is a forward hook on the layer's OUTPUT, not a forward_pre_hook on its input.
    For an HF Llama-class decoder layer the output IS the post-MLP residual stream,
    so hooking the output of layer ``i`` is the same injection site as a pre-hook on
    layer ``i+1``. ``--inject-layers 7,14,18,21`` therefore perturbs what layers
    8/15/19/22 read. That is a labelling convention, stated here so nobody reads the
    layer list as "the input of layer 7".
  - It gates on the ``[B, S]`` mask built from the teacher-forced forward's own
    shape rather than on ``cache_position``. B1 only ever does full-sequence
    teacher-forced forwards; a ``seq_len`` that disagrees with the mask means
    somebody used a KV cache, and the hook RAISES rather than mis-injecting.

── the forward ───────────────────────────────────────────────────────────────
We call ``model.model(...)`` (the decoder stack, whose output is already through
the final RMSNorm) and apply ``model.lm_head`` ONLY at the positions being scored.
Running the full ``lm_head`` over a [8, 450] batch would materialise a
[8, 450, 128256] logits tensor — and, under autograd, its log-softmax too — for the
sake of a few dozen scored positions. Gathering first is the difference between
~3 GB and ~30 MB of logits.

Gradient note: the frozen model's parameters all have ``requires_grad=False`` and
the embedding output does not require grad, so autograd saves nothing for the layers
BELOW the first injection site. Peak activation memory is set by
``(n_layers - min(inject_layers))``, not by the whole stack.

Attention: ``attn_implementation="sdpa"``. B1 reads logits only and never asks for
attention weights, so the eager path B0 stage 2 needed (which materialises the full
attention matrix for capture) buys nothing here and costs a lot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from pleroma.build_pairs import PairRecord

logger = logging.getLogger("expB1.gnet")

#: torch dtypes selectable from the CLI.
DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ── g ──────────────────────────────────────────────────────────────────────────


class SignatureToBias(nn.Module):
    """z-signature [B, D] -> per-layer additive residual bias [B, n_layers, H].

    Zero-initialised output layer: ``forward`` returns exact zeros before any
    optimiser step, which makes "g at init" and "alpha=0" the same forward.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or hidden <= 0 or n_layers <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"SignatureToBias needs positive dims, got in_dim={in_dim}, "
                f"hidden={hidden}, n_layers={n_layers}, hidden_dim={hidden_dim}"
            )
        self.in_dim = int(in_dim)
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_layers * hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.in_dim:
            raise ValueError(f"g expects [B, {self.in_dim}], got {tuple(z.shape)}")
        h = self.act(self.fc1(self.norm(z)))
        return self.fc2(h).view(z.shape[0], self.n_layers, self.hidden_dim)

    @torch.no_grad()
    def bias_norms(self, z: Tensor) -> list[float]:
        """Mean L2 norm of the emitted bias per injected layer (a training diagnostic)."""
        bias = self.forward(z)
        return [float(bias[:, i].norm(dim=-1).mean()) for i in range(self.n_layers)]


# ── injection ──────────────────────────────────────────────────────────────────


def decoder_layers(model: Any) -> Any:
    """Decoder layer list, wrapper-aware (mirrors model_loader.decoder_layers:166)."""
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


class TrainableResidualInjector:
    """Adds g's live bias to selected decoder layers' outputs, gradients intact.

    Usage per forward::

        injector.set_bias(bias, mask)   # bias [B, n_inject, H]; mask [B, S] bool
        out = model.model(input_ids=..., attention_mask=..., use_cache=False)
        injector.clear()                # -> the next forward is the alpha=0 arm

    ``clear()`` leaves the hooks registered but makes them exact no-ops: the
    alpha=0 arm is then the same code path with nothing added, which is a stronger
    control than removing and re-adding hooks between arms.
    """

    def __init__(self, model: Any, layer_indices: Sequence[int]) -> None:
        layers = decoder_layers(model)
        n_layers = len(layers)
        if not layer_indices:
            raise ValueError("no injection layers given")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"duplicate injection layers: {list(layer_indices)}")
        for idx in layer_indices:
            if not 0 <= idx < n_layers:
                raise ValueError(
                    f"injection layer {idx} out of range (model has {n_layers} layers)"
                )
        self.layer_indices: list[int] = [int(i) for i in layer_indices]
        self._bias: Tensor | None = None
        self._mask: Tensor | None = None
        self._handles: list[Any] = []
        self.stats: dict[str, Any] = {"calls": 0, "positions": 0}
        for slot, idx in enumerate(self.layer_indices):
            self._handles.append(
                layers[idx].register_forward_hook(self._make_hook(slot, idx))
            )
        logger.info(
            "TrainableResidualInjector: hooks on decoder layers %s (output side; "
            "equivalently the input of layers %s)",
            self.layer_indices, [i + 1 for i in self.layer_indices],
        )

    # -- bias plumbing ---------------------------------------------------------

    def set_bias(self, bias: Tensor, mask: Tensor) -> None:
        """Arm the hooks for ONE forward. ``bias`` [B, n_inject, H], ``mask`` [B, S]."""
        if bias.ndim != 3 or bias.shape[1] != len(self.layer_indices):
            raise ValueError(
                f"bias must be [B, {len(self.layer_indices)}, H], got {tuple(bias.shape)}"
            )
        if mask.ndim != 2 or mask.shape[0] != bias.shape[0]:
            raise ValueError(
                f"mask must be [B={bias.shape[0]}, S], got {tuple(mask.shape)}"
            )
        self._bias = bias
        self._mask = mask.to(dtype=bias.dtype)

    def clear(self) -> None:
        """Disarm: subsequent forwards are the unperturbed (alpha=0) model."""
        self._bias = None
        self._mask = None

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "positions": 0}

    def __enter__(self) -> "TrainableResidualInjector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    # -- the hook --------------------------------------------------------------

    def _make_hook(self, slot: int, layer_idx: int) -> Any:
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
            bias, mask = self._bias, self._mask
            if bias is None or mask is None:
                return output  # alpha=0: untouched, no clone, no cast
            hidden, rest, was_tuple = _split_layer_output(output, layer_idx)
            if hidden.shape[0] != mask.shape[0] or hidden.shape[1] != mask.shape[1]:
                raise RuntimeError(
                    f"layer {layer_idx}: hidden states {tuple(hidden.shape[:2])} do not "
                    f"match the position mask {tuple(mask.shape)} — B1 injects on "
                    "full-sequence teacher-forced forwards only (a seq_len of 1 means a "
                    "KV cache is in play, which this hook cannot position-gate)"
                )
            # [B, 1, H] * [B, S, 1] -> [B, S, H]; differentiable in `bias`, so the
            # gradient reaches g through every layer above this one.
            delta = bias[:, slot].unsqueeze(1) * mask.unsqueeze(-1)
            self.stats["calls"] += 1
            self.stats["positions"] += int(mask.sum().item())
            new_hidden = hidden + delta.to(dtype=hidden.dtype)
            return (new_hidden, *rest) if was_tuple else new_hidden

        return hook


def _split_layer_output(output: Any, layer_idx: int) -> tuple[Tensor, tuple[Any, ...], bool]:
    """Unpack a decoder layer's return into (hidden_states, extras, was_tuple).

    transformers 4.x LlamaDecoderLayer returns ``(hidden_states, *optional)``;
    5.x returns the bare tensor. Both are handled rather than assumed, because the
    node's transformers version is not this laptop's and a silent
    ``tuple + Tensor`` would be a TypeError three frames away from the cause.
    """
    if isinstance(output, Tensor):
        return output, (), False
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0], tuple(output[1:]), True
    raise TypeError(
        f"layer {layer_idx} returned {type(output).__name__}, which is neither a "
        "Tensor nor a tuple starting with one — extend _split_layer_output()"
    )


# ── the frozen teacher ─────────────────────────────────────────────────────────


def load_frozen_model(model_path: str, dtype: torch.dtype, device: str) -> Any:
    """Load the teacher frozen and in eval mode, with SDPA attention.

    Frozen means ``requires_grad_(False)`` on every parameter AND ``eval()``: B1
    trains g only, and a stray dropout/train-mode difference between arms would
    show up as a fake effect in exactly the contrast we care about.
    """
    from transformers import AutoModelForCausalLM

    logger.info("loading frozen teacher %s (%s, sdpa)", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable:
        raise RuntimeError(f"teacher is not frozen: {n_trainable} trainable parameters")
    return model


def decoder_stack(model: Any) -> Any:
    """The module that returns ``last_hidden_state`` (i.e. everything but lm_head).

    For Llama/Qwen/OLMo ``model.model``; for a multimodal wrapper the text decoder
    nests one level deeper. Resolved rather than assumed so that calling the stack
    directly (to skip the full-vocab lm_head) does not quietly become architecture
    -specific.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    lm = getattr(inner, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None:
        lm_inner = getattr(lm, "model", lm)
        if hasattr(lm_inner, "layers"):
            return lm_inner
    raise AttributeError(
        f"cannot locate the decoder stack on {type(model).__name__} — extend "
        "decoder_stack() for this architecture"
    )


def model_hidden_dim(model: Any) -> int:
    """hidden_size, wrapper-aware (Gemma-class configs nest it under text_config)."""
    cfg = model.config
    dim = getattr(cfg, "hidden_size", None) or getattr(
        getattr(cfg, "text_config", None), "hidden_size", None
    )
    if not dim:
        raise AttributeError(f"cannot read hidden_size off {type(cfg).__name__}")
    return int(dim)


# ── batches ────────────────────────────────────────────────────────────────────


@dataclass
class TrajectoryBatch:
    """One right-padded batch of teacher-forced trajectories.

    ``fork_rows`` / ``fork_cols`` index the PREDICTOR position (absolute
    ``prompt_length + p - 1``); ``fork_targets`` is the realized token at
    continuation position ``p`` (absolute ``prompt_length + p``). Keeping them as
    flat parallel index tensors means the lm_head is applied to exactly the scored
    positions and nothing else.
    """

    pair_keys: list[str]
    input_ids: Tensor  # [B, S] int64
    attention_mask: Tensor  # [B, S] int64
    continuation_mask: Tensor  # [B, S] bool — absolute pos >= prompt_length, non-pad
    z: Tensor  # [B, D] float32
    fork_rows: Tensor  # [n_fork] int64
    fork_cols: Tensor  # [n_fork] int64
    fork_targets: Tensor  # [n_fork] int64
    fork_weights: Tensor  # [n_fork] float32 — 1/(B * n_fork_in_sample)
    all_rows: Tensor
    all_cols: Tensor
    all_targets: Tensor
    all_weights: Tensor
    n_truncated: int

    def to(self, device: str) -> "TrajectoryBatch":
        moved = {
            k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in vars(self).items()
        }
        return TrajectoryBatch(**moved)

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])


def load_sequence(pair: PairRecord) -> tuple[np.ndarray, int]:
    """(input_ids, prompt_length) from the gen's banked ``fork_series`` npz."""
    path = pair.fork_series_abs
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — train/eval need the banked input_ids; point "
            "--run-dirs at the B0 output dir that holds them"
        )
    with np.load(path) as npz:
        for key in ("input_ids", "prompt_length"):
            if key not in npz:
                raise KeyError(f"{path}: no {key!r} (has {list(npz.files)})")
        input_ids = np.asarray(npz["input_ids"], dtype=np.int64)
        prompt_length = int(np.asarray(npz["prompt_length"]).reshape(-1)[0])
    if not 0 < prompt_length < input_ids.shape[0]:
        raise ValueError(
            f"{path}: prompt_length {prompt_length} out of range for "
            f"{input_ids.shape[0]} tokens"
        )
    if prompt_length != pair.prompt_length:
        raise ValueError(
            f"{path}: prompt_length {prompt_length} != manifest {pair.prompt_length} — "
            "the pairs dir and the run dir describe different runs"
        )
    return input_ids, prompt_length


def resolve_run_dirs(pairs: Iterable[PairRecord], run_dirs: Sequence[Path]) -> dict[str, str]:
    """Map each pair's recorded run_dir onto one of ``--run-dirs``, by directory name.

    The manifest records absolute paths from the machine that BUILT it; the node
    that trains has the same run dirs under a different root. Matching on the
    directory name keeps the manifest portable without making the mapping implicit.
    """
    by_name = {Path(d).expanduser().resolve().name: str(Path(d).expanduser().resolve())
               for d in run_dirs}
    if len(by_name) != len(run_dirs):
        raise ValueError(f"--run-dirs have duplicate basenames: {[str(d) for d in run_dirs]}")
    mapping: dict[str, str] = {}
    for pair in pairs:
        if pair.run_label in mapping:
            continue
        if pair.run_label not in by_name:
            raise ValueError(
                f"pairs reference run dir {pair.run_label!r} which is not among "
                f"--run-dirs {sorted(by_name)}"
            )
        mapping[pair.run_label] = by_name[pair.run_label]
    return mapping


def rebase(pair: PairRecord, mapping: dict[str, str]) -> PairRecord:
    """Return ``pair`` with ``run_dir`` pointing at this machine's copy."""
    target = mapping.get(pair.run_label)
    if target is None or target == pair.run_dir:
        return pair
    return PairRecord(**{**vars(pair), "run_dir": target})


def build_batch(
    pairs: Sequence[PairRecord],
    z_rows: np.ndarray,
    pad_token_id: int,
    max_seq_len: int,
    include_all_positions: bool,
) -> TrajectoryBatch:
    """Right-pad a list of pairs into one teacher-forced batch.

    Truncation to ``max_seq_len`` drops fork positions that fall past the cut and
    COUNTS them (``n_truncated``); a pair with no fork position left is a caller
    error, not a silently empty loss term — ``prepare_pairs`` filters those out
    before batching so the count here is a tripwire.
    """
    if len(pairs) != z_rows.shape[0]:
        raise ValueError(f"{len(pairs)} pairs but {z_rows.shape[0]} z rows")
    if not pairs:
        raise ValueError("refusing to build an empty batch")

    seqs: list[np.ndarray] = []
    plens: list[int] = []
    n_truncated = 0
    for pair in pairs:
        ids, plen = load_sequence(pair)
        if ids.shape[0] > max_seq_len:
            ids = ids[:max_seq_len]
            n_truncated += 1
        seqs.append(ids)
        plens.append(plen)

    batch = len(seqs)
    seq_len = max(int(s.shape[0]) for s in seqs)
    input_ids = np.full((batch, seq_len), pad_token_id, dtype=np.int64)
    attention = np.zeros((batch, seq_len), dtype=np.int64)
    cont_mask = np.zeros((batch, seq_len), dtype=bool)

    fork_rows: list[int] = []
    fork_cols: list[int] = []
    fork_targets: list[int] = []
    fork_weights: list[float] = []
    all_rows: list[int] = []
    all_cols: list[int] = []
    all_targets: list[int] = []
    all_weights: list[float] = []

    for b, (pair, ids, plen) in enumerate(zip(pairs, seqs, plens)):
        n = int(ids.shape[0])
        input_ids[b, :n] = ids
        attention[b, :n] = 1
        cont_mask[b, plen:n] = True

        usable = [p for p in pair.fork_positions if 0 < plen + p < n]
        if not usable:
            raise ValueError(
                f"{pair.pair_key}: no fork position survives a {max_seq_len}-token cut "
                f"(positions {pair.fork_positions}, prompt_length {plen}, len {n})"
            )
        w = 1.0 / (batch * len(usable))
        for p in usable:
            fork_rows.append(b)
            fork_cols.append(plen + p - 1)  # the distribution that produced position p
            fork_targets.append(int(ids[plen + p]))
            fork_weights.append(w)

        if include_all_positions:
            cont_positions = list(range(plen, n))
            if cont_positions:
                wa = 1.0 / (batch * len(cont_positions))
                for abs_pos in cont_positions:
                    all_rows.append(b)
                    all_cols.append(abs_pos - 1)
                    all_targets.append(int(ids[abs_pos]))
                    all_weights.append(wa)

    def _t(values: list[Any], dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(values, dtype=dtype)

    return TrajectoryBatch(
        pair_keys=[p.pair_key for p in pairs],
        input_ids=torch.from_numpy(input_ids),
        attention_mask=torch.from_numpy(attention),
        continuation_mask=torch.from_numpy(cont_mask),
        z=torch.from_numpy(np.ascontiguousarray(z_rows, dtype=np.float32)),
        fork_rows=_t(fork_rows, torch.int64),
        fork_cols=_t(fork_cols, torch.int64),
        fork_targets=_t(fork_targets, torch.int64),
        fork_weights=_t(fork_weights, torch.float32),
        all_rows=_t(all_rows, torch.int64),
        all_cols=_t(all_cols, torch.int64),
        all_targets=_t(all_targets, torch.int64),
        all_weights=_t(all_weights, torch.float32),
        n_truncated=n_truncated,
    )


# ── the scored forward ─────────────────────────────────────────────────────────


def _gathered_nll(
    model: Any,
    hidden: Tensor,
    rows: Tensor,
    cols: Tensor,
    targets: Tensor,
    chunk: int,
) -> Tensor:
    """Per-position NLL at (rows, cols), lm_head applied to those positions only.

    Chunked so a ``--loss-all-positions`` batch (thousands of positions x a 128k
    vocab) does not materialise one enormous logits tensor at peak.
    """
    if rows.numel() == 0:
        return hidden.new_zeros((0,), dtype=torch.float32)
    outs: list[Tensor] = []
    for start in range(0, int(rows.numel()), chunk):
        stop = min(start + chunk, int(rows.numel()))
        sel = hidden[rows[start:stop], cols[start:stop]]  # [n, H]
        logits = model.lm_head(sel).float()
        outs.append(
            torch.nn.functional.cross_entropy(
                logits, targets[start:stop], reduction="none"
            )
        )
    return torch.cat(outs)


@dataclass
class ForwardResult:
    """What one arm's forward produced, in float32 and detached where noted."""

    loss: Tensor  # scalar, grad-carrying when g was armed
    fork_nll: Tensor  # [n_fork] float32, per scored position (detached)
    fork_rows: Tensor  # [n_fork] int64 — which batch entry each NLL belongs to
    mean_fork_nll: float
    mean_all_nll: float | None


def run_arm(
    model: Any,
    injector: TrainableResidualInjector | None,
    batch: TrajectoryBatch,
    bias: Tensor | None,
    all_positions_weight: float,
    lm_chunk: int,
) -> ForwardResult:
    """One teacher-forced forward + the fork NLL, with or without g's bias.

    ``bias is None`` (or ``injector is None``) is the alpha=0 arm: the hooks are
    disarmed and the forward is the frozen model's own.
    """
    if injector is not None:
        if bias is None:
            injector.clear()
        else:
            injector.set_bias(bias, batch.continuation_mask)
    try:
        out = decoder_stack(model)(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state  # already through the final norm
        fork_nll = _gathered_nll(
            model, hidden, batch.fork_rows, batch.fork_cols, batch.fork_targets, lm_chunk
        )
        loss = (fork_nll * batch.fork_weights).sum()
        mean_all: float | None = None
        if all_positions_weight > 0.0 and batch.all_rows.numel():
            all_nll = _gathered_nll(
                model, hidden, batch.all_rows, batch.all_cols, batch.all_targets, lm_chunk
            )
            loss = loss + all_positions_weight * (all_nll * batch.all_weights).sum()
            mean_all = float(all_nll.mean().detach())
    finally:
        if injector is not None:
            injector.clear()
    return ForwardResult(
        loss=loss,
        fork_nll=fork_nll.detach(),
        fork_rows=batch.fork_rows,
        mean_fork_nll=float(fork_nll.mean().detach()),
        mean_all_nll=mean_all,
    )


def per_sample_means(fork_nll: Tensor, fork_rows: Tensor, batch_size: int) -> list[float]:
    """Mean fork NLL per batch entry — the paired unit every B1 contrast is built on."""
    nll = fork_nll.detach().float().cpu()
    rows = fork_rows.detach().cpu()
    sums = torch.zeros(batch_size, dtype=torch.float64)
    counts = torch.zeros(batch_size, dtype=torch.float64)
    sums.index_add_(0, rows, nll.double())
    counts.index_add_(0, rows, torch.ones_like(nll, dtype=torch.float64))
    if bool((counts == 0).any()):
        raise ValueError("a batch entry contributed no scored position")
    return [float(v) for v in (sums / counts)]


def autocast_context(device: str, enabled: bool) -> Any:
    """bf16 autocast on CUDA when asked for; a no-op context otherwise."""
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def resolve_pad_token_id(model: Any, model_path: str) -> int:
    """A pad id for right-padding. Masked out everywhere, so any valid id will do.

    Config first (no tokenizer load), tokenizer second, and 0 only as a last resort
    with a warning — never silently, because a pad id outside the vocab is an
    index error deep inside the embedding.
    """
    cfg = model.config
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(cfg, attr, None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            return int(value)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        value = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if value is not None:
            return int(value)
    except Exception as exc:  # noqa: BLE001 — the fallback below is still valid
        logger.warning("could not read a pad id from the tokenizer (%s)", exc)
    logger.warning("no pad/eos token id found; padding with 0 (masked out anyway)")
    return 0
