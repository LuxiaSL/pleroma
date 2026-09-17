"""Build + validate a replay manifest for a banked run.

To replay-extract we need the full realized token sequence [prompt + g_0..g_{N-1}].
The banked data does NOT store it — only:
  - chosen_ids = g_1..g_{N-1}     (raw_tensors npz; the FIRST generated token is skipped)
  - generated_text               (metadata; decoded with skip_special_tokens=True)
  - system_prompt / user_prompt / prompt_length / num_generated_tokens   (metadata)

Reconstruction per gen (maximal fidelity — only g_0 is "recovered", the rest are the
exact banked tokens):
  - prompt_ids = chat_template(system, user)   (must MATCH banked prompt_length, else the
        tokenizer/chat-template drifted vs the original extraction → flag)
  - retok = encode(generated_text, add_special_tokens=False)
  - g_0 = retok[0]; STRONG validation: retok == [g_0] + chosen_ids(minus trailing EOS)
        → confirms g_0 and that chosen is the canonical tokenization of the text
  - full_gen_ids = [g_0] + chosen_ids   (banked tokens kept exactly, incl. trailing EOS)
  - input_ids = prompt_ids + full_gen_ids

Gens that fail any check are flagged with a reason (NOT silently dropped). The manifest
is JSON: {gen_id: {input_ids, prompt_length, n_gen, ok, reason}}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Known Llama 3.x end-of-sequence ids (3.2-3B / 3.1-8B Instruct both ship this set).
DEFAULT_EOS_IDS = (128001, 128008, 128009)


def _load_chosen_ids(
    raw_dir: Path, gen_id: int, chosen_map: dict[int, Any] | None = None,
) -> np.ndarray | None:
    """Load banked chosen_ids (= g_1..g_{N-1}) for one generation.

    Prefers an explicit chosen_map (a pre-extracted bundle — used when raw_tensors
    aren't co-located, e.g. building the manifest on the node before extraction),
    falling back to the run's raw_tensors npz.
    """
    if chosen_map is not None and gen_id in chosen_map:
        return np.asarray(chosen_map[gen_id]).astype(np.int64)
    p = raw_dir / f"gen_{gen_id:03d}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        if "chosen_ids" not in z.files:
            return None
        return z["chosen_ids"].astype(np.int64)


def build_prompt_ids(tokenizer: Any, system_prompt: str, user_prompt: str) -> list[int]:
    """Replicate generation_runner.format_prompt's chat template exactly (ids only)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors=None,
    )
    # tf returns either a flat list[int] or a (Batch)Encoding mapping with "input_ids".
    # BatchEncoding subclasses UserDict → it is NOT a `dict` instance; check for the key.
    if isinstance(result, (list, tuple)):
        ids = list(result)
    else:
        ids = result["input_ids"]
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]  # un-batch
    return [int(x) for x in ids]


def reconstruct_one(
    tokenizer: Any,
    gen_meta: dict[str, Any],
    chosen_ids: np.ndarray,
    eos_ids: set[int],
) -> tuple[list[int] | None, int, bool, str]:
    """Reconstruct + validate the full token sequence for one generation.

    Returns (input_ids | None, prompt_length, ok, reason).
    """
    system = gen_meta.get("system_prompt", "")
    user = gen_meta.get("user_prompt", "")
    gen_text = gen_meta.get("generated_text", "")
    meta_plen = int(gen_meta["prompt_length"])
    meta_ngen = int(gen_meta.get("num_generated_tokens", -1))

    prompt_ids = build_prompt_ids(tokenizer, system, user)
    if len(prompt_ids) != meta_plen:
        return (None, len(prompt_ids), False,
                f"prompt_len {len(prompt_ids)} != banked {meta_plen} (chat-template/tokenizer drift)")

    chosen = [int(x) for x in chosen_ids.tolist()]
    if not gen_text:
        return None, meta_plen, False, "empty generated_text"

    retok = [int(x) for x in tokenizer.encode(gen_text, add_special_tokens=False)]
    if len(retok) < 1:
        return None, meta_plen, False, "empty re-tokenization"

    # chosen may carry trailing EOS that skip-special decode dropped from gen_text.
    chosen_core = list(chosen)
    while chosen_core and chosen_core[-1] in eos_ids:
        chosen_core = chosen_core[:-1]

    g0 = retok[0]
    full_gen_ids = [g0] + chosen  # g_0 recovered + g_1..g_{N-1} exact banked (incl EOS)

    # PRIMARY: strict round-trip (retok == [g_0] + chosen_core).
    strict_ok = (len(retok) - 1 == len(chosen_core) and retok[1:] == chosen_core)
    if strict_ok:
        reason = "ok"
    else:
        # FALLBACK: a downstream tokenization ambiguity can make retok != [g_0]+chosen_core while
        # g_0 itself is still correct. Accept iff [g_0]+banked-chosen decodes back to the exact
        # generated_text — g_1..g_{N-1} remain the exact banked tokens; only g_0 is "recovered".
        if tokenizer.decode(full_gen_ids, skip_special_tokens=True) != gen_text:
            return (None, meta_plen, False,
                    f"round-trip mismatch (retok-1={len(retok) - 1}, chosen_core={len(chosen_core)})")
        reason = "ok-decode-fallback"

    if meta_ngen >= 0 and len(full_gen_ids) != meta_ngen:
        return (None, meta_plen, False, f"n_gen {len(full_gen_ids)} != banked {meta_ngen}")

    input_ids = prompt_ids + full_gen_ids
    return input_ids, meta_plen, True, reason


def build_replay_manifest(
    run_dir: Path,
    tokenizer: Any,
    eos_ids: set[int] | None = None,
    chosen_ids_map: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Build the replay manifest for all banked generations in a run.

    Parameters
    ----------
    run_dir : Path
        Run directory containing metadata.json + raw_tensors/.
    tokenizer : transformers tokenizer
        Must be the model's tokenizer (same chat template as the original extraction).
    eos_ids : set[int], optional
        EOS token ids to strip from the chosen-ids tail (default: Llama 3.x set).

    Returns
    -------
    dict with keys: "entries" {gen_id: {...}}, "n_ok", "n_flagged", "flagged" [reasons].
    """
    eos = set(eos_ids) if eos_ids is not None else set(DEFAULT_EOS_IDS)
    raw_dir = run_dir / "raw_tensors"
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta

    entries: dict[str, Any] = {}
    flagged: list[dict[str, Any]] = []
    n_ok = 0
    for g in gens:
        gid = int(g["generation_id"])
        chosen = _load_chosen_ids(raw_dir, gid, chosen_ids_map)
        if chosen is None:
            flagged.append({"gen_id": gid, "reason": "no raw_tensors / chosen_ids"})
            continue
        input_ids, plen, ok, reason = reconstruct_one(tokenizer, g, chosen, eos)
        if ok and input_ids is not None:
            entries[str(gid)] = {
                "input_ids": input_ids,
                "prompt_length": plen,
                "n_gen": len(input_ids) - plen,
            }
            n_ok += 1
        else:
            flagged.append({"gen_id": gid, "reason": reason})

    return {
        "entries": entries,
        "n_ok": n_ok,
        "n_flagged": len(flagged),
        "flagged": flagged,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build + validate a replay manifest for a run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir (has metadata.json + raw_tensors/)")
    parser.add_argument("--model-path", type=str, required=True, help="Model/tokenizer path or HF id")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--eos-ids", type=int, nargs="+", default=None, help="Override EOS ids")
    parser.add_argument("--chosen-ids", type=Path, default=None,
                        help="JSON {gen_id: [ids]} bundle of banked chosen_ids "
                             "(fallback when raw_tensors are not co-located on this host)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    chosen_ids_map: dict[int, Any] | None = None
    if args.chosen_ids is not None:
        with open(args.chosen_ids) as f:
            raw_map = json.load(f)
        chosen_ids_map = {int(k): v for k, v in raw_map.items()}
        logger.info(f"Loaded chosen_ids bundle: {len(chosen_ids_map)} gens")

    manifest = build_replay_manifest(
        args.run_dir, tokenizer,
        eos_ids=set(args.eos_ids) if args.eos_ids else None,
        chosen_ids_map=chosen_ids_map,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f)

    logger.info(
        f"Manifest: {manifest['n_ok']} ok, {manifest['n_flagged']} flagged → {args.out}"
    )
    if manifest["flagged"]:
        logger.warning("Flagged gens (reasons):")
        # group reasons
        from collections import Counter
        reasons = Counter(x["reason"].split("(")[0].strip() for x in manifest["flagged"])
        for r, c in reasons.most_common():
            logger.warning(f"  {c:4d}× {r}")
        sample = manifest["flagged"][:10]
        logger.warning(f"  e.g.: {[(x['gen_id'], x['reason']) for x in sample]}")


if __name__ == "__main__":
    main()
