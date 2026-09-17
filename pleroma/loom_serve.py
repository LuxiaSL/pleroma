"""LOOM v0 — the model looms over its own futures and bends toward one you pick.

The loop, live, end to end (INTENT: "a signature of trajectory i steering a
fresh run IS a future influencing a present"):

    /chat            talk to either branch (base | loom), duet conventions
    /loom            from the LOOM branch's current context: sample K futures,
                     harvest each future's v3 signature + bins-B features
                     (subprocess run_replay_b0 + extract_bins — the SAME frozen
                     instruments every banked corpus went through), push each
                     through the fitted map g (fit_loom_map.npz) -> K candidate
                     levers, norm-matched per site to the bank's alpha=1 scale
    /wear            pick candidate i at dose alpha: the loom branch now wears
                     that future's code on every subsequent turn
    /unwear /reset /info

Inheritances from duet_serve, deliberate and unchanged: loopback-only bind;
re-prefill every turn, no KV across requests; UNIFORM injection (start_pos=0);
one persistent generation thread (cuDNN-SDPA off, the 2026-09-16 forensics);
byte-identity gate before the socket opens; per-(session, branch, turn) seeds.

v0 honesty, in the transcript's own words: the map was trained on single bare
prompts' continuations, and here it reads futures of a multi-turn chat context
— off-distribution for g, exploratory instrument, same caveat duet_serve
carries. The measured receipts live in RESULTS-AMPLIFIER-2026-09-17.md.

Usage (from the repo root, inside the project venv):
    python -m pleroma.loom_serve \\
        --map data/loom/loom_map_3b.npz \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --calib-dir data/calibration/3b \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --work-dir outputs/loom/sessions --port 8767
Remote GPU box?  ssh -L 8767:localhost:8767 <host>  and browse localhost:8767.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.duet_serve import (
    build_messages,
    require_loopback,
    trim_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("loom_serve")

BRANCHES = ("base", "loom")


def loom_turn_seed(seed: int, session: str, branch: str, turn: int, draw: int = 0) -> int:
    raw = f"loomv0_{seed}_{session}_{branch}_{turn}_{draw}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def harvest_via_worker(
    worker_url: str, loom_dir: Path, connect_timeout: float, total_timeout: float
) -> dict[str, Any]:
    """POST /harvest to a running harvest_worker.py. Raises on ANY problem
    (unreachable, non-ok body, timeout) — the caller's job is to catch that
    and fall back to the subprocess pipeline, never to half-trust a worker
    response.
    """
    req = urllib.request.Request(
        worker_url.rstrip("/") + "/harvest",
        data=json.dumps({"loom_dir": str(loom_dir)}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=connect_timeout + total_timeout) as resp:
        blob = json.loads(resp.read())
    if not blob.get("ok"):
        raise RuntimeError(f"harvest worker reported failure: {blob}")
    return blob


# ── the map, loaded once, numpy only ──────────────────────────────────────────


class LoomMap:
    """fit_loom_map.npz: the full raw-signature -> lever inference pipeline."""

    def __init__(self, path: Path, discriminants: Path):
        with np.load(path, allow_pickle=True) as npz:
            # Two storage forms: full W [in, out] (large), or the rank-r SVD
            # factors W_U/W_S/W_Vt (small — what ships; W is rank-truncated at
            # fit time so the factored product IS W up to float roundoff).
            self._factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None
            if "W" in npz.files:
                self.W = np.asarray(npz["W"], dtype=np.float64)
                self._factors = None
            elif {"W_U", "W_S", "W_Vt"} <= set(npz.files):
                u_ = np.asarray(npz["W_U"], dtype=np.float64)
                s_ = np.asarray(npz["W_S"], dtype=np.float64)
                vt_ = np.asarray(npz["W_Vt"], dtype=np.float64)
                self.W = (u_ * s_) @ vt_
                self._factors = (u_, s_, vt_)
            else:
                raise KeyError(
                    f"{path} has neither 'W' nor 'W_U/W_S/W_Vt' (has {npz.files})"
                )
            self.mu_in = np.asarray(npz["mu_in"], dtype=np.float64)
            self.mu_y = np.asarray(npz["mu_y"], dtype=np.float64)
            self.v3_mu = np.asarray(npz["v3_mu"], dtype=np.float64)
            self.v3_sd = np.asarray(npz["v3_sd"], dtype=np.float64)
            self.v3_dead = np.asarray(npz["v3_dead"], dtype=bool)
            self.bins_mu = np.asarray(npz["bins_mu"], dtype=np.float64)
            self.bins_sd = np.asarray(npz["bins_sd"], dtype=np.float64)
            self.bins_dead = np.asarray(npz["bins_dead"], dtype=bool)
            self.sites = [int(x) for x in npz["sites"]]
            self.norm_ref = np.asarray(npz["norm_ref"], dtype=np.float64)
            self.meta = json.loads(str(npz["meta"]))
        with np.load(discriminants, allow_pickle=True) as npz:
            self.full_mean = np.asarray(npz["FULL_mean"], dtype=np.float64)
            full_scale = np.asarray(npz["FULL_scale"], dtype=np.float64)
        sha = hashlib.sha256(Path(discriminants).read_bytes()).hexdigest()
        if sha != self.meta["discriminants_sha256"]:
            raise ValueError(
                "discriminants sha mismatch: the map was fitted in a different "
                "frozen space than the one handed to the server"
            )
        self.degenerate = full_scale < 1e-12
        self.full_scale = np.where(self.degenerate, 1.0, full_scale)
        self.n_sites = len(self.sites)
        self.hidden = self.mu_y.size // self.n_sites
        # The CODE basis: W = U S Vᵀ (already rank-truncated). A candidate's
        # 8 coordinates in span(V) ARE the code — the entire content of what
        # the map says, before expansion to the 12,288-d lever.
        self.rank = int(self.meta["rank"])
        if self._factors is not None:
            self.Vt = self._factors[2][: self.rank]
        else:
            _, _, vt = np.linalg.svd(self.W, full_matrices=False)
            self.Vt = vt[: self.rank]

    def z_corpus_of(self, signature: np.ndarray) -> np.ndarray:
        """Raw v3 signature -> corpus-standardized z (the training pipeline's
        stage 1+2 — also the space basin_check-style spread readings live in)."""
        zf = (signature - self.full_mean) / self.full_scale
        zf[self.degenerate] = 0.0
        zc = (zf - self.v3_mu) / np.where(self.v3_dead, 1.0, self.v3_sd)
        zc[self.v3_dead] = 0.0
        return zc

    def lever_of(
        self, signature: np.ndarray, bins_row: np.ndarray
    ) -> tuple[np.ndarray, list[float]]:
        """Raw v3 signature + raw bins row -> (norm-matched lever [S, hidden],
        the map's RAW per-site output norms before matching — the 'loudness'
        of the code as the map spoke it, advisory only)."""
        zc = self.z_corpus_of(signature)
        bs = (bins_row - self.bins_mu) / np.where(self.bins_dead, 1.0, self.bins_sd)
        bs[self.bins_dead] = 0.0
        x = np.concatenate([zc, bs])
        flat = (x - self.mu_in) @ self.W + self.mu_y
        code = [round(float(c), 4) for c in self.Vt @ (flat - self.mu_y)]
        lever = flat.reshape(self.n_sites, self.hidden)
        out = np.empty_like(lever)
        raw_norms: list[float] = []
        for s in range(self.n_sites):
            n = float(np.linalg.norm(lever[s]))
            if n <= 0:
                raise ValueError(f"map emitted a zero row at site {self.sites[s]}")
            raw_norms.append(n)
            out[s] = lever[s] * (self.norm_ref[s] / n)
        return out, raw_norms, code


def two_means(unit_rows: np.ndarray, iters: int = 25) -> tuple[np.ndarray, float]:
    """Tiny 2-means on unit rows (cosine geometry) — basin_check's own move,
    turned on the K futures themselves. Returns (labels, separation) where
    separation = between-centroid distance / mean within-camp distance."""
    n = unit_rows.shape[0]
    d = 1.0 - unit_rows @ unit_rows.T
    a, b = np.unravel_index(np.argmax(d), d.shape)
    cents = np.stack([unit_rows[a], unit_rows[b]])
    labels = np.zeros(n, dtype=int)
    for it in range(iters):
        new = np.argmax(unit_rows @ cents.T, axis=1)
        if it > 0 and (new == labels).all():
            break
        labels = new
        for c in (0, 1):
            if (labels == c).any():
                m = unit_rows[labels == c].mean(axis=0)
                nm = float(np.linalg.norm(m))
                if nm > 0:
                    cents[c] = m / nm
    within: list[float] = []
    for c in (0, 1):
        rows = unit_rows[labels == c]
        if len(rows) >= 2:
            dc = 1.0 - rows @ rows.T
            within.append(float(dc[np.triu_indices(len(rows), 1)].mean()))
    between = float(1.0 - cents[0] @ cents[1])
    sep = between / max(float(np.mean(within)) if within else 1e-9, 1e-9)
    return labels, sep


AUTO_POLICIES = ("distinct", "loudest", "minority", "stay", "swerve")


# ── session state ──────────────────────────────────────────────────────────────


@dataclass
class LoomSession:
    histories: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {b: [] for b in BRANCHES}
    )
    candidates: list[dict[str, Any]] = field(default_factory=list)  # last /loom
    worn: dict[str, Any] | None = None  # {"index", "alpha", "loom_id", "vectors"}
    n_looms: int = 0
    loom_id: str | None = None          # id of the draw `candidates` belongs to
    last_futures: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, Any] | None = None  # live during /loom
    reroll_n: dict[str, int] = field(default_factory=dict)  # per-branch nonce

    def worn_public(self) -> dict[str, Any] | None:
        if not self.worn:
            return None
        return {
            "index": self.worn["index"], "alpha": self.worn["alpha"],
            "loom_id": self.worn.get("loom_id"),
            "per_site_norms_at_alpha1": [
                round(float(np.linalg.norm(r)), 3) for r in self.worn["vectors"]
            ],
            "code": self.worn.get("code"),
        }


class State:
    def __init__(self) -> None:
        self.sessions: dict[str, LoomSession] = {}
        self.lock = threading.Lock()

    def get(self, session: str) -> LoomSession:
        with self.lock:
            return self.sessions.setdefault(session, LoomSession())


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--discriminants", type=Path, required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument(
        "--kvrot-path", default=".",
        help="dir containing kvrot/sigbridge.py (default: the repo root — "
             "the vendored copy)",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320, help="chat replies")
    parser.add_argument("--future-tokens", type=int, default=192, help="per future")
    parser.add_argument("--default-k", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=12, help="history window")
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selfcheck-tokens", type=int, default=32)
    parser.add_argument(
        "--harvest-worker", default="",
        help=(
            "host:port of a running harvest_worker.py. When set, /loom tries "
            "POSTing to it first and only falls back to the original two-"
            "subprocess pipeline if the worker is unreachable or reports an "
            "error. Empty (default) = subprocess pipeline only."
        ),
    )
    parser.add_argument("--harvest-worker-connect-timeout", type=float, default=3.0)
    parser.add_argument("--harvest-worker-timeout", type=float, default=120.0)
    parser.add_argument("--atlas-report", default=None,
                        help="atlas_report.json for GET /atlas (default: "
                             "outputs/loom/atlas/atlas_report.json if present)")
    parser.add_argument(
        "--allow-lan", action="store_true",
        help="bind beyond loopback (e.g. 0.0.0.0). The API has NO AUTH: only "
        "do this on a network you trust end to end (an authed VPN with a "
        "handful of trusted users is the intended case); loopback-only "
        "remains the DEFAULT.",
    )
    args = parser.parse_args()

    if args.allow_lan:
        host = str(args.host)
        logger.warning("binding %s — visible to the whole network, NO AUTH "
                       "(--allow-lan)", host)
    else:
        host = require_loopback(args.host)
    loom_map = LoomMap(args.map, args.discriminants)
    logger.info("map: %d-d in, sites %s, rank %s, alpha=1 norms %s",
                loom_map.mu_in.size, loom_map.sites, loom_map.meta["rank"],
                [round(float(x), 3) for x in loom_map.norm_ref])

    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(1)
    from anamnesis.config import MODEL_PRESETS
    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pleroma import g_net

    preset = MODEL_PRESETS[args.preset]
    eos_ids = list(preset.eos_token_ids)
    device = str(args.device)
    dtype = g_net.DTYPES[str(args.model_dtype)]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.chat_template is None:
        logger.error("%s has no chat template", args.model_path)
        return 2
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(device).eval().requires_grad_(False)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    hidden_dim = g_net.model_hidden_dim(model)
    if hidden_dim != loom_map.hidden:
        logger.error("map hidden %d != model %d", loom_map.hidden, hidden_dim)
        return 2
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]

    from concurrent.futures import ThreadPoolExecutor

    gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")

    def render(messages: list[dict[str, str]]) -> Any:
        result = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        return ids.to(device)

    def _draw_inner(input_ids: Any, seed: int, max_new: int) -> list[int]:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )
        return [int(x) for x in out[0].tolist()]

    def draw(input_ids: Any, seed: int, max_new: int) -> list[int]:
        return gen_executor.submit(_draw_inner, input_ids, seed, max_new).result()

    def _draw_batch_inner(input_ids: Any, batch: int, seed: int,
                          max_new: int) -> list[list[int]]:
        """K continuations of the SAME prefix, ONE generate() call, batch=K.

        No padding question arises (repeat of an identical prefix). ONE seed
        for the whole call — BENCH-LOOM-2026-09-17.md's determinism section
        records what that trades away (per-future reproducibility) and what
        it keeps (the batch as a whole reproduces from (seed, k)).
        """
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        batch_ids = input_ids.repeat(batch, 1)
        with torch.no_grad():
            out = model.generate(
                batch_ids,
                attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )
        return [[int(x) for x in row] for row in out.tolist()]

    def draw_batch(input_ids: Any, batch: int, seed: int,
                   max_new: int) -> list[list[int]]:
        return gen_executor.submit(
            _draw_batch_inner, input_ids, batch, seed, max_new
        ).result()

    def trim_to_eos(row: list[int], plen: int) -> list[int]:
        """A batched row's generated slice, trimmed to the first eos (inclusive).

        generate() pad-fills rows that finish early — tokens no batch-of-1
        draw would have produced; feeding the untrimmed tail to the harvest
        would replay a doctored trajectory. The FIRST eos-id token is the true
        stopping point even when pad_id == eos_ids[0] on this stack.
        """
        gen = row[plen:]
        eos_set = set(eos_ids)
        stop = next((t for t, tid in enumerate(gen) if tid in eos_set), None)
        return gen[: stop + 1] if stop is not None else gen

    def attach(vectors: np.ndarray, alpha: float) -> list[Any]:
        handles: list[Any] = []
        try:
            for row, site in zip(vectors, loom_map.sites, strict=True):
                handles.append(attach_residual_write(
                    model,
                    ResidualWriteSpec(
                        layer_idx=int(site),
                        vector=torch.from_numpy(
                            np.ascontiguousarray(row, dtype=np.float32)
                        ),
                        alpha=float(alpha),
                        start_pos=0, end_pos=None, normalize=False,
                    ),
                ))
        except Exception:
            for h in handles:
                h.remove()
            raise
        return handles

    # ── gate: zero-vector wear must be byte-identical to no hooks ─────────────
    probe_ids = render([{"role": "user", "content": "say hello in one sentence"}])
    gate_seed = loom_turn_seed(args.seed, "__gate__", "base", 0)
    plain = draw(probe_ids, gate_seed, args.selfcheck_tokens)
    zeros = np.zeros((loom_map.n_sites, hidden_dim), dtype=np.float32)
    handles = attach(zeros, 1.0)
    try:
        zeroed = draw(probe_ids, gate_seed, args.selfcheck_tokens)
    finally:
        for h in handles:
            h.remove()
    if plain != zeroed:
        logger.error("GATE FAILED — zero-vector wear changed sampled tokens; aborting")
        return 3
    logger.info("GATE PASSED — %d tokens byte-identical under no-hook vs zero-wear",
                len(plain) - int(probe_ids.shape[1]))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state = State()
    py = sys.executable
    repo = Path(__file__).resolve().parent.parent

    # ── the loom itself ────────────────────────────────────────────────────────
    def do_chat(session: str, branch: str, text: str) -> dict[str, Any]:
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        sess = state.get(session)
        history, dropped = trim_history(
            list(sess.histories[branch]), int(args.max_turns)
        )
        messages = build_messages(history, text, None)
        ids = render(messages)
        turn = sum(1 for m in sess.histories[branch] if m["role"] == "assistant")
        seed = loom_turn_seed(args.seed, session, branch, turn)
        worn = sess.worn if branch == "loom" else None
        handles = attach(worn["vectors"], worn["alpha"]) if worn else []
        t0 = time.time()
        try:
            full = draw(ids, seed, int(args.max_new_tokens))
        finally:
            for h in handles:
                h.remove()
        reply = tok.decode(full[int(ids.shape[1]):], skip_special_tokens=True).strip()
        sess.histories[branch].append({"role": "user", "content": text})
        sess.histories[branch].append({"role": "assistant", "content": reply})
        return {
            "session": session, "branch": branch, "reply": reply,
            "n_turns": turn + 1, "dropped_messages": dropped,
            "worn": sess.worn_public() if branch == "loom" else None,
            "elapsed_s": round(time.time() - t0, 1),
        }

    def do_loom(session: str, text: str, k: int, horizon: int,
                auto: dict[str, Any] | None = None) -> dict[str, Any]:
        """Futures = K candidate replies to the CONTEMPLATED next user turn.

        The turn is NOT committed to history here — /loom is a preview of what
        the conversation could become; /chat afterwards actually says it (worn
        or not). Crisp semantics, and the futures are exactly the object the
        map was trained to read: continuations of one shared prefix.
        """
        if k < 2:
            raise ValueError("k must be >= 2 (the extractor needs siblings)")
        if not text.strip():
            raise ValueError("/loom needs 'text' — the contemplated next user turn")
        sess = state.get(session)
        history, _ = trim_history(list(sess.histories["loom"]), int(args.max_turns))
        messages = build_messages(history, text, None)
        ids = render(messages)
        plen = int(ids.shape[1])
        turn = sum(1 for m in history if m["role"] == "assistant")
        sess_tag = hashlib.sha256(session.encode()).hexdigest()[:10]
        loom_id = f"{sess_tag}-{sess.n_looms:03d}"
        loom_dir = args.work_dir / sess_tag / f"loom_{sess.n_looms:03d}"
        rec_dir = loom_dir / "gen_records"
        rec_dir.mkdir(parents=True, exist_ok=True)
        worn = sess.worn  # futures are drawn under the CURRENT wear, if any —
        # the loom sees the futures of the model it currently is.
        futures: list[dict[str, Any]] = []
        t0 = time.time()
        sess.progress = {"stage": "generate", "done": 0, "total": k}
        # Batched K-draw (loom latency item 1): ONE generate() call, batch=k,
        # one seed for the whole batch (determinism trade recorded in BENCH).
        batch_seed = loom_turn_seed(args.seed, session, "loom", turn, draw=0)
        handles = attach(worn["vectors"], worn["alpha"]) if worn else []
        try:
            rows = draw_batch(ids, k, batch_seed, int(horizon))
        finally:
            for h in handles:
                h.remove()
        sess.progress = {"stage": "generate", "done": k, "total": k}
        for j, row in enumerate(rows):
            gen = trim_to_eos(row, plen)
            full = row[:plen] + gen
            text = tok.decode(gen, skip_special_tokens=True).strip()
            record = {
                "generation_id": j, "prompt_id": f"loom{sess.n_looms:03d}",
                "prompt_class": "loom_context", "prompt": "<chat context>",
                "prompt_idx": 0, "seed_idx": j, "seed": batch_seed,
                "system_prompt": None, "user_prompt": "<chat context>",
                "prompt_length": plen, "num_generated_tokens": len(gen),
                "generated_text": text, "input_ids": full,
                "sampling": {"temperature": float(args.temperature),
                             "top_p": float(args.top_p),
                             "max_new_tokens": int(horizon), "eos_ids": eos_ids,
                             "attn_implementation": "sdpa"},
                "model_id": str(args.model_path), "preset": str(args.preset),
            }
            (rec_dir / f"gen_{j:03d}.json").write_text(json.dumps(record))
            futures.append({"index": j, "text": text, "n_tokens": len(gen)})
        gen_s = time.time() - t0

        # Harvest: the same frozen instruments the corpora went through.
        # Persistent worker first (loom latency item 2); subprocess pipeline
        # is the unconditional FALLBACK.
        sess.progress = {"stage": "harvest", "done": 0, "total": 2}
        t1 = time.time()
        harvest_via = "subprocess"
        worker_detail: dict[str, Any] | None = None
        if args.harvest_worker:
            try:
                worker_detail = harvest_via_worker(
                    f"http://{args.harvest_worker}", loom_dir,
                    float(args.harvest_worker_connect_timeout),
                    float(args.harvest_worker_timeout),
                )
                harvest_via = "worker"
                sess.progress = {"stage": "harvest", "done": 2, "total": 2}
            except Exception as exc:  # noqa: BLE001 — any worker problem falls back
                logger.warning(
                    "harvest worker %s unavailable/failed (%s: %s) — falling "
                    "back to the subprocess pipeline", args.harvest_worker,
                    type(exc).__name__, exc,
                )
        if harvest_via == "subprocess":
            env_cmds = [
                [py, "-m", "pleroma.run_replay_b0", "--gen-dir", str(loom_dir),
                 "--out-dir", str(loom_dir), "--preset", str(args.preset),
                 "--model-path", str(args.model_path),
                 "--calib-dir", str(args.calib_dir),
                 "--discriminants", str(args.discriminants),
                 "--kvrot-path", str(args.kvrot_path), "--save-raw", "none"],
                [py, "-m", "pleroma.extract_bins", "--gen-dirs", str(loom_dir),
                 "--model-path", str(args.model_path), "--preset", str(args.preset),
                 "--out", str(loom_dir / "bins.npz")],
            ]
            for si, cmd in enumerate(env_cmds):
                proc = subprocess.run(cmd, cwd=str(repo), capture_output=True,
                                      text=True)
                if proc.returncode != 0:
                    tail = (proc.stdout + proc.stderr)[-800:]
                    raise RuntimeError(f"harvest step failed ({cmd[2]}): ...{tail}")
                sess.progress = {"stage": "harvest", "done": si + 1, "total": 2}
        harvest_s = time.time() - t1

        with np.load(loom_dir / "bins.npz", allow_pickle=True) as bnpz:
            bins_feat = np.asarray(bnpz["features"], dtype=np.float64)
            bins_gid = [int(x) for x in bnpz["generation_id"]]
        bins_of = {g: i for i, g in enumerate(bins_gid)}
        candidates = []
        z_rows: list[np.ndarray | None] = []
        for f in futures:
            j = f["index"]
            sig_path = loom_dir / "signatures" / f"gen_{j:03d}.npz"
            note = None
            raw_norms: list[float] | None = None
            code8: list[float] | None = None
            z_row: np.ndarray | None = None
            try:
                with np.load(sig_path, allow_pickle=True) as snpz:
                    sig = np.asarray(snpz["features"], dtype=np.float64)
                brow = bins_feat[bins_of[j]]
                if not np.isfinite(brow).all():
                    raise ValueError("bins row is non-finite")
                lever, raw_norms, code8 = loom_map.lever_of(sig, brow)
                z_row = loom_map.z_corpus_of(sig)
            except Exception as exc:  # noqa: BLE001 — a dead future is listed, not hidden
                note = f"{type(exc).__name__}: {exc}"
                lever = None
            candidates.append({"index": j, "lever": lever, "note": note,
                               "raw_norms": raw_norms, "code": code8})
            z_rows.append(z_row)

        # ── advisory scores (Luxia 2026-09-17): the spread of paths, read in
        # signature space — basin_check's original control, aimed at the K
        # futures of ONE prefix, where within-fan cosine is exactly the right
        # instrument (unlike cos-to-contrast-vector, which the pull tests
        # dethroned as a quality bar).
        scores: dict[int, dict[str, Any]] = {f["index"]: {} for f in futures}
        spread_info: dict[str, Any] | None = None
        ok = [i for i, z in enumerate(z_rows) if z is not None]
        if len(ok) >= 2:
            unit = np.stack([z_rows[i] / np.linalg.norm(z_rows[i]) for i in ok])
            dist = 1.0 - unit @ unit.T
            for a, i in enumerate(ok):
                scores[i]["distinct"] = round(
                    float(np.mean([dist[a, b] for b in range(len(ok)) if b != a])), 4
                )
            spread_info = {
                "mean_pairwise_distance": round(
                    float(dist[np.triu_indices(len(ok), 1)].mean()), 4),
                "n_scored": len(ok),
            }
            if len(ok) >= 3:
                labels, sep = two_means(unit)
                for a, i in enumerate(ok):
                    scores[i]["camp"] = int(labels[a])
                spread_info["camps"] = [int((labels == 0).sum()),
                                        int((labels == 1).sum())]
                spread_info["separation"] = round(sep, 3)
                spread_info["camp_labels"] = (
                    "per-draw only — 2-means labels are not identities across "
                    "looms; camp 0 this draw is not camp 0 next draw"
                )
            spread_info["loudness_ref"] = round(float(np.mean(loom_map.norm_ref)), 3)
        for c in candidates:
            i = c["index"]
            if c["raw_norms"] is not None:
                scores[i]["loudness"] = round(float(np.mean(c["raw_norms"])), 3)
            if worn and c["lever"] is not None:
                pc = [float(np.dot(x_, y_) /
                            (np.linalg.norm(x_) * np.linalg.norm(y_)))
                      for x_, y_ in zip(c["lever"], worn["vectors"])]
                scores[i]["cos_to_worn"] = round(float(np.mean(pc)), 3)
        sess.candidates = candidates
        sess.loom_id = loom_id
        sess.n_looms += 1
        futures_public = [
            {"index": f["index"], "n_tokens": f["n_tokens"], "text": f["text"],
             "harvested": candidates[i]["lever"] is not None,
             "note": candidates[i]["note"], "scores": scores[f["index"]],
             "code": candidates[i]["code"]}
            for i, f in enumerate(futures)
        ]
        sess.last_futures = futures_public

        auto_selected: dict[str, Any] | None = None
        if auto:
            policy = str(auto.get("policy") or "")
            if policy not in AUTO_POLICIES:
                raise ValueError(f"auto.policy must be one of {AUTO_POLICIES}")
            alpha = float(auto.get("alpha", 0.35))
            live = [i for i in ok if candidates[i]["lever"] is not None]
            if not live:
                raise ValueError("auto-select: no harvested future to pick from")

            def by(key: str, sign: float) -> int:
                vals = [(sign * scores[i].get(key, float("-inf")), i) for i in live]
                best = max(vals)
                if best[0] == float("-inf"):
                    raise ValueError(f"auto policy needs score {key!r} "
                                     "(is anything worn?)")
                return best[1]

            effective = policy
            if policy == "distinct":
                pick = by("distinct", +1.0)
            elif policy == "loudest":
                pick = by("loudness", +1.0)
            elif policy == "minority":
                camps = [scores[i].get("camp") for i in live]
                if any(c is None for c in camps):
                    pick = by("distinct", +1.0)  # fewer than 3 scored
                    effective = "distinct"       # honest banner (UI friction #3)
                else:
                    minority = 0 if camps.count(0) <= camps.count(1) else 1
                    members = [i for i in live if scores[i]["camp"] == minority]
                    pick = max(members,
                               key=lambda i: scores[i].get("distinct", 0.0))
            elif policy == "stay":
                pick = by("cos_to_worn", +1.0)
            else:  # swerve
                pick = by("cos_to_worn", -1.0)
            sess.worn = {"index": pick, "alpha": alpha, "loom_id": loom_id,
                         "vectors": candidates[pick]["lever"],
                         "code": candidates[pick].get("code")}
            auto_selected = {"policy": policy, "effective_policy": effective,
                             "index": pick, "alpha": alpha,
                             "scores": scores[pick]}

        return {
            "session": session, "loom_id": loom_id, "n_futures": k,
            "horizon": int(horizon),
            "drawn_under_wear": (None if not worn
                                 else {"index": worn["index"], "alpha": worn["alpha"],
                                       "loom_id": worn.get("loom_id")}),
            "futures": futures_public,
            "spread": spread_info,
            "auto_selected": auto_selected,
            "worn": sess.worn_public(),
            "timing_s": {"generate": round(gen_s, 1), "harvest": round(harvest_s, 1)},
            "harvest_via": harvest_via,
            "harvest_worker_detail": worker_detail,
            "loom_dir": str(loom_dir),
        }

    def do_undo(session: str, branch: str) -> dict[str, Any]:
        """Pop the last user/assistant exchange. The popped turn is RETURNED
        (never silently discarded) so the UI can offer redo-by-resend."""
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        h = state.get(session).histories[branch]
        if len(h) < 2 or h[-1]["role"] != "assistant" or h[-2]["role"] != "user":
            raise ValueError("nothing to undo — history does not end in a "
                             "user/assistant exchange")
        popped_a = h.pop()
        popped_u = h.pop()
        return {"session": session, "branch": branch,
                "popped": {"user": popped_u["content"],
                           "assistant": popped_a["content"]},
                "n_turns": sum(1 for m in h if m["role"] == "assistant")}

    def do_truncate(session: str, branch: str, keep_turns: int) -> dict[str, Any]:
        """Cut a branch back to its first keep_turns exchanges (deep edit's
        first half: truncate to just before turn i, then /chat the new text)."""
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        if keep_turns < 0:
            raise ValueError("keep_turns must be >= 0")
        h = state.get(session).histories[branch]
        state.get(session).histories[branch] = h[: keep_turns * 2]
        return {"session": session, "branch": branch, "n_turns": keep_turns,
                "dropped_messages": max(len(h) - keep_turns * 2, 0)}

    def do_reroll(session: str, branch: str) -> dict[str, Any]:
        """Redraw the last assistant reply: same user turn, fresh seed (a
        per-branch nonce keys it), current wear. The replaced reply is
        returned, not destroyed silently."""
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        sess = state.get(session)
        h = sess.histories[branch]
        if len(h) < 2 or h[-1]["role"] != "assistant":
            raise ValueError("nothing to reroll — no assistant reply at the end")
        old = h.pop()
        user_text = h.pop()["content"]
        sess.reroll_n[branch] = sess.reroll_n.get(branch, 0) + 1
        nonce = 100000 + sess.reroll_n[branch]
        history, dropped = trim_history(list(h), int(args.max_turns))
        messages = build_messages(history, user_text, None)
        ids = render(messages)
        turn = sum(1 for m in h if m["role"] == "assistant")
        seed = loom_turn_seed(args.seed, session, branch, turn, draw=nonce)
        worn = sess.worn if branch == "loom" else None
        handles = attach(worn["vectors"], worn["alpha"]) if worn else []
        t0 = time.time()
        try:
            full = draw(ids, seed, int(args.max_new_tokens))
        finally:
            for hd in handles:
                hd.remove()
        reply = tok.decode(full[int(ids.shape[1]):], skip_special_tokens=True).strip()
        h.append({"role": "user", "content": user_text})
        h.append({"role": "assistant", "content": reply})
        return {"session": session, "branch": branch, "reply": reply,
                "replaced": old["content"], "reroll_n": sess.reroll_n[branch],
                "n_turns": turn + 1, "dropped_messages": dropped,
                "worn": sess.worn_public() if branch == "loom" else None,
                "elapsed_s": round(time.time() - t0, 1)}

    def do_edit(session: str, branch: str, text: str) -> dict[str, Any]:
        """Replace the LAST user turn with new text and regenerate (undo+chat,
        atomically). Deeper edits: /truncate then /chat from the client."""
        undo = do_undo(session, branch)
        out = do_chat(session, branch, text)
        out["replaced_user"] = undo["popped"]["user"]
        out["replaced_assistant"] = undo["popped"]["assistant"]
        return out

    def do_wear(session: str, index: int, alpha: float,
                loom_id: str | None) -> dict[str, Any]:
        sess = state.get(session)
        if not sess.candidates:
            raise ValueError("no candidates — /loom first")
        if loom_id is not None and loom_id != sess.loom_id:
            raise ValueError(
                f"loom_id {loom_id!r} is not the latest draw ({sess.loom_id!r}) — "
                "another tab or turn superseded it; /loom again"
            )
        match = [c for c in sess.candidates if c["index"] == index]
        if not match or match[0]["lever"] is None:
            raise ValueError(f"candidate {index} unavailable "
                             f"({match[0]['note'] if match else 'no such index'})")
        sess.worn = {"index": index, "alpha": float(alpha),
                     "loom_id": sess.loom_id, "vectors": match[0]["lever"],
                     "code": match[0].get("code")}
        return {"session": session, "worn": sess.worn_public(),
                "sites": loom_map.sites}

    # ── HTTP ───────────────────────────────────────────────────────────────────
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            logger.info("http %s", fmt % a)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self) -> Any:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(self.path)
            if parsed.path == "/atlas":
                ap = Path(__file__).resolve().parent.parent / \
                    "outputs/loom/atlas/atlas_report.json"
                if args.atlas_report is not None:
                    ap = Path(args.atlas_report)
                if not ap.exists():
                    self._send(404, {"error": "no atlas built yet"})
                    return
                self._send(200, json.loads(ap.read_text()))
                return
            if parsed.path == "/state":
                q = parse_qs(parsed.query)
                session = (q.get("session") or [""])[0].strip()
                if not session:
                    self._send(400, {"error": "missing session"})
                    return
                sess = state.get(session)
                self._send(200, {
                    "session": session,
                    "histories": sess.histories,
                    "worn": sess.worn_public(),
                    "loom_id": sess.loom_id,
                    "futures": sess.last_futures,
                    "n_looms": sess.n_looms,
                    "loom_in_progress": sess.progress,
                })
                return
            if parsed.path == "/loom/progress":
                q = parse_qs(parsed.query)
                session = (q.get("session") or [""])[0].strip()
                sess = state.get(session) if session else None
                self._send(200, {"session": session,
                                 "progress": sess.progress if sess else None})
                return
            if self.path in ("/", "/ui"):
                # Served from disk on EVERY request, deliberately: the UI is
                # being designed iteratively and a page reload should pick up
                # the new file without a server restart.
                ui = Path(__file__).resolve().parent / "loom_ui.html"
                if not ui.exists():
                    self._send(404, {"error": "loom_ui.html not present yet — "
                                              "the UI is being designed"})
                    return
                body = ui.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/info":
                self._send(200, {
                    "map": str(args.map), "map_meta": loom_map.meta,
                    "sites": loom_map.sites, "branches": list(BRANCHES),
                    "default_k": int(args.default_k),
                    "future_tokens": int(args.future_tokens),
                    "n_sessions": len(state.sessions),
                    "harvest_worker": args.harvest_worker or None,
                    "auto_policies": [
                        {"key": "distinct", "needs_wear": False},
                        {"key": "loudest", "needs_wear": False},
                        {"key": "minority", "needs_wear": False},
                        {"key": "stay", "needs_wear": True},
                        {"key": "swerve", "needs_wear": True},
                    ],
                    "loudness_ref": round(float(np.mean(loom_map.norm_ref)), 3),
                })
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                blob = self._json()
                session = str(blob.get("session") or "").strip()
                if not session:
                    raise ValueError("missing 'session'")
                if self.path == "/chat":
                    out = do_chat(session, str(blob.get("branch") or "loom"),
                                  str(blob.get("text") or ""))
                elif self.path == "/loom":
                    try:
                        out = do_loom(session, str(blob.get("text") or ""),
                                      int(blob.get("k") or args.default_k),
                                      int(blob.get("horizon") or args.future_tokens),
                                      auto=(blob.get("auto")
                                            if isinstance(blob.get("auto"), dict)
                                            else None))
                    finally:
                        state.get(session).progress = None
                elif self.path == "/wear":
                    out = do_wear(session, int(blob["index"]),
                                  float(blob.get("alpha", 0.5)),
                                  (str(blob["loom_id"])
                                   if blob.get("loom_id") else None))
                elif self.path == "/undo":
                    out = do_undo(session, str(blob.get("branch") or "loom"))
                elif self.path == "/reroll":
                    out = do_reroll(session, str(blob.get("branch") or "loom"))
                elif self.path == "/edit":
                    out = do_edit(session, str(blob.get("branch") or "loom"),
                                  str(blob.get("text") or ""))
                elif self.path == "/truncate":
                    out = do_truncate(session, str(blob.get("branch") or "loom"),
                                      int(blob["keep_turns"]))
                elif self.path == "/unwear":
                    state.get(session).worn = None
                    out = {"session": session, "worn": None}
                elif self.path == "/reset":
                    # Scope: clears BOTH branch histories, unwears, and forgets
                    # candidates — the session starts over entirely.
                    state.sessions.pop(session, None)
                    out = {"session": session, "cleared": True, "worn": None}
                else:
                    self._send(404, {"error": "unknown path"})
                    return
                self._send(200, out)
            except (ValueError, KeyError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
                logger.exception("request failed")
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer((host, int(args.port)), Handler)
    logger.info("LOOM v0 listening on %s:%d — tunnel: ssh -L %d:localhost:%d <host>",
                host, args.port, args.port, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
