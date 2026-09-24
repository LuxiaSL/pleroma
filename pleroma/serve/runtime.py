"""The GPU half of the loom: model + tokenizer, prompt rendering, sampling,
the steering hook, and the zero-vector gate.

``HFRuntime`` carries ``render``, ``_draw_inner``/``draw``,
``_draw_batch_inner``/``draw_batch``, ``trim_to_eos``, ``attach`` and
``finish_reply``. Torch, transformers and the steering hook are imported
inside ``load_runtime`` — never at module scope — so this module stays
laptop-importable.

``Runtime`` is the Protocol the routes use, so a test can drive every route
with a fake runtime and no model (tests/unit/serve/).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

from pleroma.format import modelc
from pleroma.format.prompt import render_prompt_ids
from pleroma.format.trim import split_modelc, trim_generated_ids
from pleroma.map.loom_map import LoomMap
from pleroma.serve.draws import loom_turn_seed

logger = logging.getLogger("loom_serve")


class Runtime(Protocol):
    """What the routes need from the model side. ``HFRuntime`` is the real one."""

    prompt_mode: str
    modelc_header: str
    stop_strings: list[str]
    eos_ids: list[int]
    pad_id: int
    chat_template_present: bool

    def render(self, messages: list[dict[str, str]]) -> Any: ...
    def draw(self, input_ids: Any, seed: int, max_new: int) -> list[int]: ...
    def draw_batch(self, input_ids: Any, batch: int, seed: int,
                   max_new: int) -> list[list[int]]: ...
    def trim_to_eos(self, row: list[int], plen: int) -> list[int]: ...
    def attach(self, vectors: np.ndarray, alpha: float) -> list[Any]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def finish_reply(self, decoded: str) -> tuple[str, dict[str, str]]: ...


class HFRuntime:
    """The loaded HF model + tokenizer under the prompt mode this server booted in.

    Built by ``load_runtime``; holds the model, the tokenizer, the sampling
    settings and the single generation thread every draw runs on.
    """

    def __init__(self, *, torch: Any, model: Any, tok: Any, device: str,
                 eos_ids: list[int], pad_id: int, hidden_dim: int,
                 prompt_mode: str, modelc_header: str, stop_strings: list[str],
                 temperature: float, top_p: float, sites: Sequence[int],
                 steer_hf: Any, injection_cls: Any, span_cls: Any,
                 date_string: str | None = None) -> None:
        self._torch = torch
        self.model = model
        self.tok = tok
        self.device = device
        self.eos_ids = eos_ids
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.prompt_mode = prompt_mode
        self.modelc_header = modelc_header
        #: chat mode: pinned into templates that take a date (Llama 3.x's
        #: "Today Date"). None keeps the template's own default, byte for byte.
        self.date_string = date_string
        self.stop_strings = stop_strings
        self.temperature = temperature
        self.top_p = top_p
        self.sites = tuple(sites)
        self.chat_template_present = tok.chat_template is not None
        # Stop STRINGS exist only in modelc mode. chat and raw pass none at all,
        # so their generate() call carries no stopping criteria beyond eos.
        self.gen_stop_kwargs: dict[str, Any] = (
            {"stop_strings": stop_strings, "tokenizer": tok} if stop_strings else {})
        self._steer_hf = steer_hf
        self._injection_cls = injection_cls
        self._span_cls = span_cls
        self.gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")

    def render(self, messages: list[dict[str, str]]) -> Any:
        """Prompt ids for one turn, under the prompt mode this server booted in.

        ★ The 'chat' branch is kept literal. It is not routed through
        render_prompt_ids and not "unified" with the other two: every banked
        number was measured through this exact ``apply_chat_template`` call,
        and the cheapest possible guarantee that it still means the same thing
        is that it is still the same call (``test_loom_prompt_mode.py`` pins
        its token ids). The mode branch is in FRONT of it, never inside it.
        """
        torch = self._torch
        if self.prompt_mode == "chat":
            extra = {} if self.date_string is None else {"date_string": self.date_string}
            result = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", **extra
            )
            ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
            return ids.to(self.device)
        row = render_prompt_ids(self.prompt_mode, self.tok, messages, self.modelc_header)
        return torch.tensor([row], dtype=torch.long, device=self.device)

    def _draw_inner(self, input_ids: Any, seed: int, max_new: int) -> list[int]:
        torch = self._torch
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
                **self.gen_stop_kwargs,
            )
        return [int(x) for x in out[0].tolist()]

    def draw(self, input_ids: Any, seed: int, max_new: int) -> list[int]:
        return self.gen_executor.submit(self._draw_inner, input_ids, seed, max_new).result()

    def _draw_batch_inner(self, input_ids: Any, batch: int, seed: int,
                          max_new: int) -> list[list[int]]:
        """K continuations of the SAME prefix, ONE generate() call, batch=K.

        No padding question arises (repeat of an identical prefix). ONE seed
        for the whole call: that trades away per-future reproducibility (one
        future cannot be redrawn alone) and keeps batch reproducibility (the
        set of K futures reproduces from (seed, k)).
        """
        torch = self._torch
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        batch_ids = input_ids.repeat(batch, 1)
        with torch.no_grad():
            out = self.model.generate(
                batch_ids,
                attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
                **self.gen_stop_kwargs,
            )
        return [[int(x) for x in row] for row in out.tolist()]

    def draw_batch(self, input_ids: Any, batch: int, seed: int,
                   max_new: int) -> list[list[int]]:
        return self.gen_executor.submit(
            self._draw_batch_inner, input_ids, batch, seed, max_new
        ).result()

    def trim_to_eos(self, row: list[int], plen: int) -> list[int]:
        """A batched row's generated slice, trimmed to the first eos (inclusive).

        generate() pad-fills rows that finish early — tokens no batch-of-1
        draw would have produced; feeding the untrimmed tail to the harvest
        would replay a doctored trajectory. The FIRST eos-id token is the true
        stopping point even when pad_id == eos_ids[0] on this stack.
        """
        # ★ ALSO strips a trailing run of pad_id when pad is not an eos id.
        # Model C pads with 128004 and ends with 128001, so a row finished by
        # a stop STRING is pad-filled to the batch max and would be replayed
        # into the harvest as if generated (see
        # pleroma.format.trim.trim_generated_ids). When pad_id is an eos id
        # (the 8B/3B stacks) this is a plain first-eos cut.
        return trim_generated_ids(row[plen:], self.eos_ids, self.pad_id)

    def attach(self, vectors: np.ndarray, alpha: float) -> list[Any]:
        # WEAR: the ONE steering hook (pleroma.steer.hf), UNIFORM span — every
        # position, prompt included. All-or-none.
        return self._steer_hf.attach(self.model, self._injection_cls(
            sites=tuple(self.sites), vectors=vectors, alpha=float(alpha),
            span=self._span_cls.UNIFORM))

    def decode(self, ids: Sequence[int]) -> str:
        """``tok.decode(ids, skip_special_tokens=True).strip()`` — the one
        decode every route uses."""
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def finish_reply(self, decoded: str) -> tuple[str, dict[str, str]]:
        """(reply, extra message keys) for one decoded chat reply.

        chat/raw: the decoded text, untouched.
        modelc: the trimmed reply, with the dreamed remainder and the tail kind
        as SEPARATE keys. Untrimmed, the reply keeps HF's matched stop, so
        history would carry a literal ``\\n\\n**User:**`` and the NEXT turn's
        document would render an empty visitor block
        (``**Model C:** ...\\n\\n**User:**\\n\\n**User:** next``).
        """
        if self.prompt_mode != "modelc":
            return decoded, {}
        sp = split_modelc(decoded)
        return sp.reply, {"dream": sp.dream, "tail_kind": sp.tail_kind.value}

    def gate(self, seed: int, selfcheck_tokens: int, n_sites: int) -> bool:
        """★ The boot gate: a zero-vector wear must be byte-identical to no
        hooks. Logs the verdict; the caller exits 3 on False."""
        # ── gate: zero-vector wear must be byte-identical to no hooks ─────────
        probe_ids = self.render([{"role": "user", "content": "say hello in one sentence"}])
        gate_seed = loom_turn_seed(seed, "__gate__", "base", 0)
        plain = self.draw(probe_ids, gate_seed, selfcheck_tokens)
        zeros = np.zeros((n_sites, self.hidden_dim), dtype=np.float32)
        handles = self.attach(zeros, 1.0)
        try:
            zeroed = self.draw(probe_ids, gate_seed, selfcheck_tokens)
        finally:
            for h in handles:
                h.remove()
        if plain != zeroed:
            logger.error("GATE FAILED — zero-vector wear changed sampled tokens; aborting")
            return False
        logger.info("GATE PASSED — %d tokens byte-identical under no-hook vs zero-wear",
                    len(plain) - int(probe_ids.shape[1]))
        return True


def load_runtime(args: argparse.Namespace, loom_map: LoomMap) -> HFRuntime | int:
    """Import torch, load tokenizer + model, apply the prompt-mode boot guard.

    Returns the runtime, or exit code 2 when the anamnesis checkout is not
    the pinned one (unless ``PLEROMA_ALLOW_ANAMNESIS_MISMATCH=1``), for a
    chat-mode tokenizer with no template, or for a model whose hidden size is
    not the map's.
    """
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(1)
    from pleroma.config.anamnesis_registry import resolve_preset
    from pleroma.harvest.anamnesis_seam import AnamnesisPinError, verify_anamnesis_pin
    from pleroma.steer import Injection, InjectionSpan
    from pleroma.steer import hf as steer_hf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pleroma.model.layers import model_hidden_dim
    from pleroma.model.load import DTYPES

    # The steering hook and every harvest are anamnesis' — refuse to serve on
    # an instrument other than the pinned submodule (override:
    # PLEROMA_ALLOW_ANAMNESIS_MISMATCH=1).
    try:
        verify_anamnesis_pin()
    except AnamnesisPinError as exc:
        logger.error("REFUSING TO SERVE: %s", exc)
        return 2
    preset = resolve_preset(args.preset)
    eos_ids = list(preset.eos_token_ids)
    device = str(args.device)
    dtype = DTYPES[str(args.model_dtype)]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    # ── the boot guard, CONDITIONAL on the prompt mode ───────────────────────
    # Refusing every template-less tokenizer would lock the loom out of model C
    # (document-trained, no `chat_template` key at all) and out of raw
    # continuation on every model. Chat mode requires one: rendering a chat
    # with no template silently falls back to a format nobody chose.
    if args.prompt_mode == "chat" and tok.chat_template is None:
        logger.error(
            "%s has no chat template, and prompt-mode 'chat' requires one. "
            "This model is not chat-templated — run --prompt-mode raw (bare "
            "document continuation) or --prompt-mode modelc (model C's "
            "document format) instead.", args.model_path)
        return 2
    if args.prompt_mode != "chat" and tok.chat_template is not None:
        logger.warning(
            "prompt-mode %r: this tokenizer DOES carry a chat template and it "
            "is being bypassed on purpose. Every banked number was "
            "measured under 'chat'; fans drawn here are a different cell.",
            args.prompt_mode)
    prompt_mode = str(args.prompt_mode)
    modelc_header = (modelc.resolve_header(str(args.modelc_header))
                     if prompt_mode == "modelc" else "")
    # Stop STRINGS exist only in modelc mode. chat and raw pass none at all.
    stop_strings: list[str] = (list(modelc.STOPS)
                               if prompt_mode == "modelc" and args.modelc_stops
                               else [])
    if prompt_mode == "modelc":
        logger.info(
            "prompt-mode modelc: header=%r -> %r | sampling T=%.3f top_p=%.3f | "
            "stops=%s", args.modelc_header, modelc_header,
            float(args.temperature), float(args.top_p),
            stop_strings or "NONE (--no-modelc-stops)")
    else:
        logger.info("prompt-mode %s: sampling T=%.3f top_p=%.3f",
                    prompt_mode, float(args.temperature), float(args.top_p))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(device).eval().requires_grad_(False)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    hidden_dim = model_hidden_dim(model)
    if hidden_dim != loom_map.hidden:
        logger.error("map hidden %d != model %d", loom_map.hidden, hidden_dim)
        return 2
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
    return HFRuntime(
        torch=torch, model=model, tok=tok, device=device, eos_ids=eos_ids,
        pad_id=pad_id, hidden_dim=hidden_dim, prompt_mode=prompt_mode,
        modelc_header=modelc_header, stop_strings=stop_strings,
        date_string=getattr(args, "date_string", None),
        temperature=float(args.temperature), top_p=float(args.top_p),
        sites=loom_map.sites, steer_hf=steer_hf, injection_cls=Injection,
        span_cls=InjectionSpan)
