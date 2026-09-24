"""Bank each gen's MEAN hidden state at the four inject sites (the lever bank's raw material).

GPU. One teacher-forced forward per banked trajectory, reading
``output_hidden_states=True`` (NOT attention — SDPA is fine and fast here), and
keeping, per site, the mean over the CONTINUATION positions only. 4 x 3072
float32 per gen; the whole 1,312-gen corpus is ~64 MB.

This is the raw material for the contrast levers (`pleroma.map.build.levers`): a
lever is ``mean_h(cluster A) - mean_h(cluster B)`` at a site, and the only thing that has
to be banked to build one is each gen's own per-site mean.

── the site convention (READ THIS BEFORE CHANGING --sites) ────────────────────

``--sites`` are decoder-layer **INPUT** indices — i.e. exactly the ``layer_idx``
that anamnesis' ``model_loader.attach_residual_write`` (a forward_PRE_hook on
that decoder layer) addresses. The default ``8,15,19,22`` is therefore the SAME
four residual-stream sites as an ``--inject-layers 7,14,18,21`` spelling, which
names each site by the layer whose OUTPUT it is (a forward hook on the layer
before). One site, two spellings, off by one:

    attach/site index  s   ==   inject-layer index  s - 1

Never convert between them with a literal +1 scattered through the code; a
converted list is stated once, where it is used.

The third spelling is the one this stage reads:
``BaseModelOutputWithPast.hidden_states[s]``. transformers appends the residual
stream to ``all_hidden_states`` BEFORE calling decoder layer ``s`` (the Llama
modeling code in transformers 4.51.3, and the same shape in 5.x), and appends
the final-normed stream once more after the loop, so the tuple has
``n_layers + 1`` entries and

    hidden_states[s]  IS  the input of decoder layer s   (for s < n_layers)

which is the site a residual write at ``layer_idx=s`` perturbs. That claim is
not taken on trust from a version number: ``verify_site_indexing()`` runs one
probe forward with a capturing forward_PRE_hook on every requested site and
asserts BYTE EQUALITY between what the pre-hook saw and ``hidden_states[s]``,
on the installed transformers, before a single gen is banked. It also asserts
``hidden_states[s] != hidden_states[s + 1]`` so the equality cannot pass
vacuously. A silent off-by-one here would put every lever one layer away from
the site it is injected at, and no downstream number could detect it.

── what is banked ─────────────────────────────────────────────────────────────

One ``<out-dir>/mean_hiddens.npz`` per invocation (a "gen-dir group"), holding every gen
from every ``--gen-dirs`` entry:

    means              [n_gens, n_sites, hidden_dim] float32
    generation_ids     [n_gens] int64
    prompt_ids / prompt_classes / source_dir_labels / corpus_keys   [n_gens] str
    seed_idxs          [n_gens] int64
    n_positions        [n_gens] int64 — continuation tokens actually averaged
    sites              [n_sites] int64

``corpus_keys`` is the join key downstream: a gen dir (``pilot_gen``) and its
replay dir (``pilot_replay``) name the same corpus, and the lever build has to
put the hidden means banked from the former next to the signatures banked in
the latter. ``corpus_key()`` strips the role token so both read ``pilot``.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.map.build.hiddens \\
        --gen-dirs outputs/pilot_gen outputs/wave2_gen \\
        --out-dir outputs/b15_hiddens_orig \\
        --model-path <hf-id-or-local-dir> --preset 3b
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import (house rule; this stage is GPU-bound).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pleroma.harvest.replay import load_gen_records, resolve_gen_records_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB15.bank")

#: Decoder-layer INPUT indices (== attach_residual_write layer_idx). The same four
#: residual-stream sites as the OUTPUT-side spelling 7,14,18,21.
DEFAULT_SITES: str = "8,15,19,22"

#: Directory-role tokens stripped by ``corpus_key`` — the words that distinguish a
#: stage-1 gen dir from its stage-2 replay dir without naming a different corpus.
ROLE_TOKENS: frozenset[str] = frozenset({"gen", "gens", "records", "replay", "replays"})


def parse_sites(spec: str) -> list[int]:
    """Parse ``--sites``: comma-separated decoder-layer INPUT indices, ascending.

    Order is normalised (and duplicates refused) so that the ``means`` axis is the
    same whichever order the flag was written in — every downstream file indexes
    sites positionally.
    """
    raw = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not raw:
        raise ValueError("--sites is empty")
    try:
        sites = [int(s) for s in raw]
    except ValueError as exc:
        raise ValueError(f"--sites must be integers, got {spec!r}") from exc
    if any(s < 0 for s in sites):
        raise ValueError(f"--sites must be >= 0, got {sites}")
    if len(set(sites)) != len(sites):
        raise ValueError(f"duplicate site in --sites: {sites}")
    return sorted(sites)


def corpus_key(name: str) -> str:
    """``pilot_gen`` / ``pilot_replay`` -> ``pilot``; ``repl_gen_a`` -> ``repl_a``.

    The join key between hidden means (banked off the stage-1 gen dirs) and
    signatures (banked in the stage-2 replay dirs). Role tokens are dropped, every
    other token is kept in order, so ``repl_gen_a`` and ``repl_replay_a`` agree
    while ``repl_*_a`` and ``repl_*_b`` stay distinct. A name made ENTIRELY of role
    tokens keeps its own basename rather than collapsing to the empty string.
    """
    parts = [p for p in str(name).strip().strip("/").split("/") if p]
    base = parts[-1] if parts else str(name)
    tokens = [t for t in base.split("_") if t]
    kept = [t for t in tokens if t.lower() not in ROLE_TOKENS]
    return "_".join(kept) if kept else base


# ── the work list ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenSpec:
    """One banked trajectory, as stage 1 wrote it plus where it came from."""

    generation_id: int
    source_dir_label: str
    corpus_key: str
    prompt_id: str
    prompt_class: str
    seed_idx: int
    prompt_length: int
    input_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.corpus_key}#{self.generation_id:04d}"

    @property
    def n_continuation(self) -> int:
        return len(self.input_ids) - self.prompt_length


def load_gen_specs(gen_dirs: Sequence[Path], limit: int = 0) -> list[GenSpec]:
    """Every stage-1 record under ``gen_dirs``, in (corpus_key, generation_id) order.

    Reads records through ``run_replay_b0.load_gen_records`` — the same six-field
    contract stage 2 enforces — so a record this stage accepts is one the rest of
    the pipeline already accepted. Duplicate ``(corpus_key, generation_id)`` pairs
    are a hard error: they would silently collapse two gens into one npz row.
    """
    specs: list[GenSpec] = []
    seen: set[str] = set()
    for gen_dir in gen_dirs:
        path = Path(gen_dir).expanduser()
        label = path.resolve().name
        key = corpus_key(label)
        records = load_gen_records(resolve_gen_records_dir(path))
        for rec in records:
            ids = [int(x) for x in rec["input_ids"]]
            plen = int(rec["prompt_length"])
            if not 0 < plen < len(ids):
                raise ValueError(
                    f"{label} gen_{int(rec['generation_id']):04d}: prompt_length {plen} "
                    f"out of range for {len(ids)} input_ids"
                )
            spec = GenSpec(
                generation_id=int(rec["generation_id"]),
                source_dir_label=label,
                corpus_key=key,
                prompt_id=str(rec["prompt_id"]),
                prompt_class=str(rec["prompt_class"]),
                seed_idx=int(rec["seed_idx"]),
                prompt_length=plen,
                input_ids=tuple(ids),
            )
            if spec.key in seen:
                raise ValueError(
                    f"duplicate gen key {spec.key} — two --gen-dirs map to corpus "
                    f"{key!r} and share generation ids; give them distinct names"
                )
            seen.add(spec.key)
            specs.append(spec)
        logger.info("%s -> corpus %r: %d records", path, key, len(records))
    if not specs:
        raise ValueError("no gen records found under --gen-dirs; refusing to run")
    specs.sort(key=lambda s: (s.corpus_key, s.generation_id))
    return specs[:limit] if limit else specs


def continuation_masks(
    prompt_lengths: Sequence[int], lengths: Sequence[int], seq_len: int
) -> NDArray[np.bool_]:
    """[B, seq_len] bool: True at absolute positions ``prompt_length <= p < len``.

    The position set every map-building stage means by "the continuation":
    absolute positions >= prompt_length, the same set a continuation-only
    injection gates on with ``start_pos = prompt_length``. Padding is
    right-hand and excluded. Raises ValueError when the two length lists
    differ in size or a row violates ``0 < prompt_length < length <= seq_len``.
    """
    if len(prompt_lengths) != len(lengths):
        raise ValueError(f"{len(prompt_lengths)} prompt lengths but {len(lengths)} lengths")
    mask = np.zeros((len(lengths), int(seq_len)), dtype=bool)
    for b, (plen, n) in enumerate(zip(prompt_lengths, lengths)):
        if not 0 < int(plen) < int(n) <= int(seq_len):
            raise ValueError(
                f"row {b}: prompt_length {plen}, length {n}, seq_len {seq_len} — "
                "need 0 < prompt_length < length <= seq_len"
            )
        mask[b, int(plen) : int(n)] = True
    return mask


def chunks(items: Sequence[Any], size: int) -> list[list[Any]]:
    """Fixed-size chunks in the given order (deterministic batching)."""
    if size <= 0:
        raise ValueError(f"batch size must be > 0, got {size}")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


# ── the GPU half ───────────────────────────────────────────────────────────────


def verify_site_indexing(
    torch: Any,
    stack: Any,
    layers: Any,
    sites: Sequence[int],
    input_ids: Any,
    attention_mask: Any,
) -> dict[str, Any]:
    """GATE: ``hidden_states[s]`` is byte-identical to the input of decoder layer s.

    Proved rather than cited: a capturing ``register_forward_pre_hook`` on each
    requested site records ``args[0]`` (what ``attach_residual_write`` would add
    its delta to) during one probe forward that also asks
    for ``output_hidden_states=True``. The two must be the same tensor values.

    Also asserts the tuple length is ``n_layers + 1`` and that consecutive entries
    DIFFER — an all-equal stack would make the equality above vacuous.
    """
    captured: dict[int, Any] = {}

    def make_hook(site: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            tensor = args[0] if args else kwargs.get("hidden_states")
            if tensor is None:
                raise RuntimeError(
                    f"site {site}: the decoder layer received no hidden_states in "
                    "args[0] nor kwargs — extend the capture hook for this architecture"
                )
            captured[site] = tensor.detach().clone()
            return None

        return hook

    handles = [layers[s].register_forward_pre_hook(make_hook(s), with_kwargs=True) for s in sites]
    try:
        with torch.no_grad():
            out = stack(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "the decoder stack returned no hidden_states even with "
            "output_hidden_states=True — this stage cannot bank site means"
        )
    n_layers = len(layers)
    if len(hidden_states) != n_layers + 1:
        raise RuntimeError(
            f"hidden_states has {len(hidden_states)} entries for a {n_layers}-layer "
            "model; expected n_layers + 1 (one per layer INPUT plus the final norm). "
            "The site<->index correspondence this stage relies on does not hold here."
        )
    for site in sites:
        if site not in captured:
            raise RuntimeError(f"site {site}: the capture hook never fired")
        if not torch.equal(captured[site].float(), hidden_states[site].float()):
            raise RuntimeError(
                f"SITE INDEXING GATE FAILED at site {site}: hidden_states[{site}] is "
                f"NOT the tensor decoder layer {site} received. Every lever would be "
                "built at a different residual-stream site than it is injected at."
            )
        if torch.equal(hidden_states[site].float(), hidden_states[site + 1].float()):
            raise RuntimeError(
                f"SITE INDEXING GATE is VACUOUS at site {site}: hidden_states[{site}] "
                f"== hidden_states[{site + 1}], so equality proves nothing."
            )
    logger.info(
        "SITE INDEXING GATE PASSED — hidden_states[s] is byte-identical to the input "
        "of decoder layer s for s in %s (%d-layer model, %d hidden_states entries)",
        list(sites), n_layers, len(hidden_states),
    )
    return {
        "status": "passed",
        "sites": [int(s) for s in sites],
        "n_layers": int(n_layers),
        "n_hidden_states": int(len(hidden_states)),
        "claim": (
            "hidden_states[s] IS the input of decoder layer s (== the site "
            "attach_residual_write(layer_idx=s) perturbs); verified by byte equality "
            "against a capturing forward_pre_hook on this machine's transformers"
        ),
    }


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gen-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="stage-1 out-dirs (or their gen_records/ subdirs); all land in ONE npz",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b", help="anamnesis registry preset")
    parser.add_argument(
        "--sites",
        default=DEFAULT_SITES,
        help=(
            "decoder-layer INPUT indices == attach_residual_write layer_idx; the "
            f"default {DEFAULT_SITES} is the same four residual-stream sites as the "
            "--inject-layers spelling 7,14,18,21 (which names them by the layer "
            "whose OUTPUT they are)"
        ),
    )
    parser.add_argument("--batch", type=int, default=8, help="gens per forward")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="cap gens (0 = all)")
    parser.add_argument(
        "--model-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch <= 0 or args.max_seq_len <= 0:
        logger.error("--batch and --max-seq-len must be > 0")
        return 2
    if args.limit < 0:
        logger.error("--limit must be >= 0, got %s", args.limit)
        return 2

    try:
        sites = parse_sites(str(args.sites))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        specs = load_gen_specs(list(args.gen_dirs), int(args.limit))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load stage-1 records: %s", exc)
        return 2
    logger.info(
        "%d gens over %d corpora (%s), %d prompts; sites %s",
        len(specs),
        len({s.corpus_key for s in specs}),
        ", ".join(sorted({s.corpus_key for s in specs})),
        len({s.prompt_id for s in specs}),
        sites,
    )

    # Deferred so --help works on any machine, with or without torch.
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA rebuilds a host-side graph for every unseen kv_len
    torch.set_num_threads(1)

    from pleroma.model.layers import decoder_layers, decoder_stack, model_hidden_dim
    from pleroma.model.load import DTYPES, load_frozen_model, resolve_pad_token_id

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("--device %s but torch.cuda.is_available() is False", device)
        return 2

    preset_layers: int | None = None
    try:
        from anamnesis.config import UnknownPresetError

        from pleroma.config.anamnesis_registry import resolve_preset

        try:
            preset_layers = int(resolve_preset(args.preset).num_layers)
        except UnknownPresetError as exc:
            logger.error("%s", exc)
            return 2
    except ImportError:
        logger.warning("anamnesis not importable; skipping the preset cross-check")

    try:
        model = load_frozen_model(args.model_path, DTYPES[args.model_dtype], device)
    except Exception as exc:  # noqa: BLE001 — loading is the commonest node-side failure
        logger.exception("could not load the model: %s", exc)
        return 1

    layers = decoder_layers(model)
    stack = decoder_stack(model)
    hidden_dim = model_hidden_dim(model)
    n_model_layers = len(layers)
    if preset_layers is not None and preset_layers != n_model_layers:
        logger.error(
            "preset %s says %d layers but %s has %d — refusing to run on a mismatched "
            "preset", args.preset, preset_layers, args.model_path, n_model_layers,
        )
        return 2
    bad = [s for s in sites if s >= n_model_layers]
    if bad:
        logger.error(
            "--sites %s are decoder-layer INPUT indices; %s are out of range for a "
            "%d-layer model", sites, bad, n_model_layers,
        )
        return 2
    if max(sites) == n_model_layers - 1:
        logger.info(
            "site %d is the input of the LAST decoder layer — valid for a pre-hook, "
            "but note there is no input-side site above it", max(sites),
        )

    pad_id = resolve_pad_token_id(model, str(args.model_path))

    # ── the site-indexing gate, before a single gen is banked ─────────────────
    probe = specs[0]
    probe_ids = torch.tensor(
        [list(probe.input_ids)[: int(args.max_seq_len)]], dtype=torch.long, device=device
    )
    try:
        gate = verify_site_indexing(
            torch, stack, layers, sites, probe_ids, torch.ones_like(probe_ids)
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 3

    # ── the run ───────────────────────────────────────────────────────────────
    means = np.zeros((len(specs), len(sites), hidden_dim), dtype=np.float32)
    n_positions = np.zeros(len(specs), dtype=np.int64)
    banked = np.zeros(len(specs), dtype=bool)
    failures: list[str] = []
    n_truncated = 0
    started = time.time()

    index_of = {spec.key: i for i, spec in enumerate(specs)}
    batches = chunks(specs, int(args.batch))
    for bi, group in enumerate(batches):
        try:
            seqs = [list(s.input_ids)[: int(args.max_seq_len)] for s in group]
            n_truncated += sum(
                1 for s, seq in zip(group, seqs) if len(seq) < len(s.input_ids)
            )
            lengths = [len(s) for s in seqs]
            seq_len = max(lengths)
            plens = [s.prompt_length for s in group]
            mask_np = continuation_masks(plens, lengths, seq_len)

            ids = np.full((len(group), seq_len), pad_id, dtype=np.int64)
            attn = np.zeros((len(group), seq_len), dtype=np.int64)
            for b, seq in enumerate(seqs):
                ids[b, : len(seq)] = seq
                attn[b, : len(seq)] = 1

            input_ids = torch.from_numpy(ids).to(device)
            attention_mask = torch.from_numpy(attn).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            counts = mask.sum(dim=1).to(torch.float32)  # [B]
            if bool((counts <= 0).any()):
                raise ValueError("a batch entry has no continuation position")

            with torch.no_grad():
                out = stack(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden_states = out.hidden_states
                weights = mask.to(torch.float32).unsqueeze(-1)  # [B, S, 1]
                for si, site in enumerate(sites):
                    hs = hidden_states[site].to(torch.float32)
                    summed = (hs * weights).sum(dim=1)  # [B, H]
                    site_means = (summed / counts.unsqueeze(-1)).cpu().numpy()
                    for b, spec in enumerate(group):
                        means[index_of[spec.key], si] = site_means[b]
            for b, spec in enumerate(group):
                row = index_of[spec.key]
                n_positions[row] = int(mask_np[b].sum())
                banked[row] = True
            del out, hidden_states
        except Exception as exc:  # noqa: BLE001 — a dead batch is recorded, never skipped
            keys = [s.key for s in group]
            logger.exception("batch %d %s failed", bi, keys)
            failures.append(f"{keys}: {type(exc).__name__}: {exc}")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        if (bi + 1) % 10 == 0 or bi == 0:
            done = int(banked.sum())
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "batch %d/%d — %d/%d gens banked, %.0fs (%.1f gen/s, ETA %.0fs)",
                bi + 1, len(batches), done, len(specs), elapsed, rate,
                (len(specs) - done) / rate if rate > 0 else 0.0,
            )

    kept = np.nonzero(banked)[0]
    if kept.size == 0:
        logger.error("no gen was banked; refusing to write an empty npz")
        return 1
    if kept.size < len(specs):
        logger.warning("%d/%d gens failed and are absent from the npz",
                       len(specs) - kept.size, len(specs))

    ordered = [specs[i] for i in kept]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_dir / "mean_hiddens.npz"
    np.savez_compressed(
        out_npz,
        means=means[kept],
        generation_ids=np.asarray([s.generation_id for s in ordered], dtype=np.int64),
        prompt_ids=np.asarray([s.prompt_id for s in ordered], dtype=np.str_),
        prompt_classes=np.asarray([s.prompt_class for s in ordered], dtype=np.str_),
        seed_idxs=np.asarray([s.seed_idx for s in ordered], dtype=np.int64),
        prompt_lengths=np.asarray([s.prompt_length for s in ordered], dtype=np.int64),
        n_positions=n_positions[kept],
        source_dir_labels=np.asarray([s.source_dir_label for s in ordered], dtype=np.str_),
        corpus_keys=np.asarray([s.corpus_key for s in ordered], dtype=np.str_),
        sites=np.asarray(sites, dtype=np.int64),
    )

    meta = {
        "stage": "expB15_bank_mean_hiddens",
        "model_id": str(args.model_path),
        "preset": str(args.preset),
        "model_dtype": str(args.model_dtype),
        "attn_implementation": "sdpa",
        "gen_dirs": [str(d) for d in args.gen_dirs],
        "corpus_keys": sorted({s.corpus_key for s in ordered}),
        "sites": sites,
        "site_convention": (
            "decoder-layer INPUT indices == attach_residual_write layer_idx == "
            "hidden_states index; equivalently the OUTPUT of layers "
            f"{[s - 1 for s in sites]} (B1's --inject-layers spelling)"
        ),
        "site_indexing_gate": gate,
        "hidden_dim": int(hidden_dim),
        "n_model_layers": int(n_model_layers),
        "batch": int(args.batch),
        "max_seq_len": int(args.max_seq_len),
        "n_truncated": int(n_truncated),
        "n_specs": len(specs),
        "n_banked": int(kept.size),
        "positions": "absolute >= prompt_length (continuation only), padding excluded",
        "reduction": "mean over continuation positions, accumulated in float32",
        "failures": failures,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out_dir / "bank_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "DONE — %d/%d gens x %d sites x %d dims (%d truncated, %d failed batches) in "
        "%.1fs -> %s",
        kept.size, len(specs), len(sites), hidden_dim, n_truncated, len(failures),
        time.time() - started, out_npz,
    )
    logger.info("NEXT: python -m pleroma.map.build.levers --hiddens %s "
                "--run-dirs <replay dirs>", out_npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
