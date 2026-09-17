"""Exp B1 stage 5a: talk to the fitted model. Node GPU, long-running HTTP server.

``steer_generate`` + ``feltshift_bundle`` give Luxia a blinded READING test: two
frozen continuations, call which one moved. This is the other half of her criterion
and the harder one — an actual conversation. Two branches of the same frozen 3B
answer the same user messages:

    base     the model as it ships
    fitted   the model wearing ONE source trajectory's member code as a persistent
             additive residual bias (``alpha * g(z_i)``, injected every forward)

Each branch keeps its OWN chat history. They are sent identical user turns and
nothing else is shared, so after turn one they are genuinely different
conversations — the fitted branch's turn-3 answer is conditioned on the fitted
branch's turn-2 answer, which is the point. This is not a matched-pair experiment
and is not analysed as one.

**RNG is deliberately NOT shared between branches.** ``steer_generate`` keys its
seed on (source, rep) and not on the arm, because there the two arms continue the
SAME prompt and a paired draw isolates the bias. Here the two branches have already
diverged in their inputs by turn two, so a shared seed would buy nothing real while
suggesting a comparability the transcript does not have. Seeds are keyed on
(seed, session, branch, turn): reproducible, per-branch, independent.

── the injection, and what re-prefill buys ───────────────────────────────────

The fitted branch attaches ``attach_residual_write`` at ``attach_layers_for(...)``
— the same +1 shift ``steer_generate`` documents and ``test_injection_site.py``
proves: g trains on decoder layer i's OUTPUT (``g_net.py:178``, a forward hook),
``attach_residual_write`` is a forward_PRE_hook on a layer's INPUT
(``model_loader.py:562``), so biases trained at 7,14,18,21 attach at 8,15,19,22.

Two deliberate differences from ``steer_generate``:

  * ``start_pos=0, end_pos=None`` — UNIFORM over every position, not
    continuation-only. Under re-prefill this is the only self-consistent choice. A
    ``start_pos=prompt_length`` spec would mean turn 3 re-encodes turn 1's tokens
    with a DIFFERENT injection pattern than turn 1 saw, so the conversation's own
    history would shift under the model between turns. Uniform injection makes the
    whole context, history included, live in one perturbed regime.

  * **No KV cache survives a request.** Every turn rebuilds the full token sequence
    from the branch's message list and re-prefills. That is a little wasteful (3B,
    chat-sized contexts — it is not the bottleneck) and it removes the failure this
    server would otherwise be one bug away from: a KV cache built under one hook
    state being consumed under another, e.g. the base branch decoding against keys
    and values that were computed with the bias applied. Within a single
    ``generate()`` the cache is used normally and ``cache_position`` gating keeps
    prefill and decode consistent (``model_loader.py:469-487``); across requests
    there is no shared state at all.

**Honest caveat, stated here because the transcripts will outlive this docstring:**
g was trained to bias the CONTINUATION positions of a single bare prompt. Applying
it uniformly across a multi-turn chat template is off-distribution for g. That makes
this an exploratory instrument — a way to meet the fitted model — and not the
measured readout. ``basin_readout`` is the measured readout.

── the gate ──────────────────────────────────────────────────────────────────

Same not-a-perturbation gate as ``steer_generate``, in this server's exact
configuration (uniform positions, the real alpha): one seed drawn with no hooks,
with ``alpha=0``, and with a ZERO vector at the real alpha. All three must produce
byte-identical token ids, and the zero-vector pass must report ``cache_position``
gating firing on every position. Failure aborts before the socket is opened.

── the wire ──────────────────────────────────────────────────────────────────

Stdlib ``http.server`` + ``json``, bound to LOOPBACK ONLY — the host may be shared, and a
0.0.0.0 bind would put a GPU and a chat endpoint on the department network. The
host is validated as a loopback address and the server refuses anything else.
Reach it from a laptop with an SSH tunnel, never by binding wider::

    ssh -L 8765:localhost:8765 <host>

  POST /chat   {"session": str, "branch": "base"|"fitted", "text": str}
               -> {"reply": str, "n_turns": int, "branch": ..., "session": ...}
  POST /reset  {"session": str} -> {"cleared": bool, "session": ...}
  GET  /info   -> the run's configuration + n_sessions

``/info`` exposes which pair the fitted branch wears. That is deliberate: BLINDING
IS THE CLIENT'S JOB (``duet_chat --mode blind``), and a server that lied to its own
operator would be a worse instrument than one that does not.

Every exchange is appended to a JSONL under ``--log-dir`` as it happens. The
transcript is data; a conversation that only exists in a terminal scrollback is a
conversation that did not happen.

Usage (from the repo root, inside the project venv)::

    python -m pleroma.duet_serve \\
        --pairs-dir outputs/b1_pairs_v2 \\
        --source-pair-key wave2_replay:117 \\
        --g-checkpoint outputs/b1_v3/g_state.pt \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --log-dir logs/duet --port 8765
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import (house rule).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import ipaddress
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from pleroma.build_pairs import PairRecord, load_pairs
from pleroma.steer_generate import (
    attach_layers_for,
    read_train_meta,
    reconcile_g_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("expB1.duet.serve")

#: The two conversations. ``base`` never attaches a hook.
BRANCHES: tuple[str, ...] = ("base", "fitted")

#: Largest JSON body accepted, so a stray upload cannot balloon the process.
MAX_BODY_BYTES = 1 << 20

#: Largest single user message, in characters.
MAX_TEXT_CHARS = 32_000


class GenerateFn(Protocol):
    """The model layer, as the HTTP layer sees it.

    Everything above this line is torch-free and unit-testable; the real
    implementation closes over the loaded model in ``main()``, and the tests pass a
    deterministic fake. Keeping the seam here is why the endpoints can be exercised
    against a real ``HTTPServer`` without a GPU.
    """

    def __call__(
        self, messages: list[dict[str, str]], branch: str, seed: int
    ) -> tuple[str, dict[str, Any]]:
        """(reply_text, diagnostics) for one turn of one branch."""


# ── pure state and pure functions ──────────────────────────────────────────────


def derive_turn_seed(seed: int, session: str, branch: str, turn: int) -> int:
    """Deterministic RNG seed for one (session, branch, turn).

    ``branch`` is IN the key on purpose — see the module docstring: the two branches
    are separate conversations, not a matched pair, and pretending otherwise by
    sharing a draw would be a comparability claim the transcript cannot support.
    """
    if seed < 0:
        raise ValueError(f"--seed must be >= 0, got {seed}")
    if turn < 0:
        raise ValueError(f"turn must be >= 0, got {turn}")
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}; known branches are {list(BRANCHES)}")
    raw = f"expB1duet_{seed}_{session}_{branch}_{turn}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def trim_history(
    messages: list[dict[str, str]], max_turns: int
) -> tuple[list[dict[str, str]], int]:
    """Keep the last ``max_turns`` user/assistant exchanges (0 = unbounded).

    Returns (messages, n_dropped_messages). A long duet would otherwise grow its
    re-prefill without limit until the context or the GPU says no; dropping the
    OLDEST exchanges keeps the conversation alive and is reported per turn rather
    than done silently. A leading system message, if any, is always preserved.
    """
    if max_turns <= 0:
        return list(messages), 0
    system = [m for m in messages[:1] if m.get("role") == "system"]
    body = messages[len(system):]
    keep = max_turns * 2
    if len(body) <= keep:
        return list(messages), 0
    dropped = body[: len(body) - keep]
    return system + body[len(body) - keep:], len(dropped)


def build_messages(
    history: list[dict[str, str]], text: str, system_prompt: str | None
) -> list[dict[str, str]]:
    """The message list handed to the chat template for ONE turn.

    ``history`` is the branch's own alternating record; ``text`` is the new user
    turn. The system prompt is prepended only when the caller configured one — B0
    and B1 prompts are BARE by design (``gen_forks``' whole premise: anything that
    tells the model how to think pins the basin we are watching it choose), so the
    default is no system message at all.
    """
    if not text.strip():
        raise ValueError("refusing to send an empty user turn")
    out: list[dict[str, str]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    out.extend(history)
    out.append({"role": "user", "content": text})
    return out


@dataclass
class SessionStore:
    """Per-(session, branch) message histories. Pure dicts, no model, no I/O."""

    sessions: dict[str, dict[str, list[dict[str, str]]]] = field(default_factory=dict)

    def history(self, session: str, branch: str) -> list[dict[str, str]]:
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        return self.sessions.setdefault(
            session, {b: [] for b in BRANCHES}
        ).setdefault(branch, [])

    def record(self, session: str, branch: str, user: str, reply: str) -> int:
        """Append one exchange; returns the branch's completed-turn count."""
        history = self.history(session, branch)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        return self.n_turns(session, branch)

    def n_turns(self, session: str, branch: str) -> int:
        return sum(
            1 for m in self.history(session, branch) if m.get("role") == "assistant"
        )

    def reset(self, session: str) -> bool:
        """Clear BOTH branches of one session. False if it was not there."""
        existed = session in self.sessions
        self.sessions[session] = {b: [] for b in BRANCHES}
        return existed

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


@dataclass
class ServeConfig:
    """What ``/info`` reports and what every log line is stamped with."""

    source_pair_key: str
    prompt_id: str
    prompt_class: str
    source_split: str
    alpha: float
    inject_layers: list[int]
    attach_layers: list[int]
    model_path: str
    g_checkpoint: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    system_prompt: str | None
    max_history_turns: int
    bias_l2: list[float] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "source_pair_key": self.source_pair_key,
            "prompt_id": self.prompt_id,
            "prompt_class": self.prompt_class,
            "source_split": self.source_split,
            "alpha": self.alpha,
            "inject_layers": list(self.inject_layers),
            "attach_layers": list(self.attach_layers),
            "bias_l2": list(self.bias_l2),
            "model_path": self.model_path,
            "g_checkpoint": self.g_checkpoint,
            "branches": list(BRANCHES),
            "sampling": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
            },
            "seed": self.seed,
            "system_prompt": self.system_prompt,
            "max_history_turns": self.max_history_turns,
        }


class ChatError(ValueError):
    """A bad request, reported as 400 rather than 500."""


def parse_chat_request(blob: Any) -> tuple[str, str, str]:
    """(session, branch, text) from a /chat body, or ChatError."""
    if not isinstance(blob, dict):
        raise ChatError("body must be a JSON object")
    session = blob.get("session")
    branch = blob.get("branch")
    text = blob.get("text")
    if not isinstance(session, str) or not session.strip():
        raise ChatError("'session' must be a non-empty string")
    if branch not in BRANCHES:
        raise ChatError(f"'branch' must be one of {list(BRANCHES)}, got {branch!r}")
    if not isinstance(text, str) or not text.strip():
        raise ChatError("'text' must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise ChatError(f"'text' is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")
    return session.strip(), branch, text


def parse_reset_request(blob: Any) -> str:
    """The session id from a /reset body, or ChatError."""
    if not isinstance(blob, dict):
        raise ChatError("body must be a JSON object")
    session = blob.get("session")
    if not isinstance(session, str) or not session.strip():
        raise ChatError("'session' must be a non-empty string")
    return session.strip()


def require_loopback(host: str) -> str:
    """Refuse any bind address that is not loopback. the host may be shared.

    A 0.0.0.0 bind would expose a GPU-backed chat endpoint with no auth to whatever
    can route to the node. The tunnel (``ssh -L 8765:localhost:8765 <host>``) is the
    supported way to reach this from a laptop, and it needs nothing wider.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        if host == "localhost":
            return "127.0.0.1"
        raise ValueError(
            f"--host {host!r} is not an IP address; this server binds loopback only "
            "(use an ssh -L tunnel to reach it from elsewhere)"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            f"--host {host} is not a loopback address. the host may be shared: binding wider "
            "would publish an unauthenticated GPU chat endpoint. Use "
            "'ssh -L 8765:localhost:8765 <host>' instead."
        )
    return host


# ── the exchange, model layer injected ─────────────────────────────────────────


@dataclass
class DuetState:
    """Everything the HTTP layer touches. Assembled by main(), faked by the tests."""

    config: ServeConfig
    store: SessionStore
    generate: GenerateFn
    log_path: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    log_lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, row: dict[str, Any]) -> None:
        """Append one JSONL row. A logging failure never kills a reply.

        Its own lock, not the generation lock: a reply can run to several KB, which
        is past the size a concurrent O_APPEND write is atomic at, and two
        half-interleaved JSON lines would corrupt the transcript exactly when the
        conversation got interesting. Separate from ``lock`` so a slow disk never
        holds up the GPU.
        """
        if self.log_path is None:
            return
        try:
            with self.log_lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # noqa: BLE001 — the conversation outranks its log
            logger.warning("could not append to %s: %s", self.log_path, exc)

    def exchange(self, session: str, branch: str, text: str) -> dict[str, Any]:
        """One turn: render, generate, record, log. Serialized by ``lock``.

        The lock is held across hook attach / generate / detach because the hooks
        live on the shared model object: two concurrent fitted requests, or a base
        request overlapping a fitted one, would read each other's injection state.
        ``ThreadingHTTPServer`` keeps the socket layer responsive; the GPU work is
        one-at-a-time by construction.
        """
        with self.lock:
            turn = self.store.n_turns(session, branch)
            history = self.store.history(session, branch)
            messages = build_messages(history, text, self.config.system_prompt)
            messages, n_dropped = trim_history(
                messages, self.config.max_history_turns
            )
            seed = derive_turn_seed(self.config.seed, session, branch, turn)
            started = time.time()
            reply, diagnostics = self.generate(messages, branch, seed)
            elapsed = time.time() - started
            n_turns = self.store.record(session, branch, text, reply)

        row = {
            "session": session,
            "branch": branch,
            "turn": turn,
            "user": text,
            "reply": reply,
            "seed": seed,
            "elapsed_s": round(elapsed, 3),
            "n_history_messages_dropped": n_dropped,
            "source_pair_key": self.config.source_pair_key,
            "alpha": self.config.alpha if branch == "fitted" else 0.0,
            **diagnostics,
        }
        self.log(row)
        if n_dropped:
            logger.warning(
                "session %s/%s: dropped %d old history message(s) at --max-history-turns %d",
                session, branch, n_dropped, self.config.max_history_turns,
            )
        logger.info(
            "%s/%s turn %d: %d chars in, %d chars out, %.1fs",
            session, branch, turn, len(text), len(reply), elapsed,
        )
        return {
            "reply": reply,
            "n_turns": n_turns,
            "branch": branch,
            "session": session,
            "turn": turn,
            "elapsed_s": round(elapsed, 3),
        }


def make_handler(state: DuetState) -> type[BaseHTTPRequestHandler]:
    """The request handler class, closed over one ``DuetState``."""

    class DuetHandler(BaseHTTPRequestHandler):
        server_version = "expB1-duet/1.0"
        protocol_version = "HTTP/1.1"

        # -- plumbing ------------------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:
            # BaseHTTPRequestHandler writes to stderr by default, which would
            # interleave with the real log. Route it through the module logger at
            # DEBUG instead of discarding it.
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                raise ChatError("bad Content-Length header") from exc
            if length <= 0:
                raise ChatError("empty body")
            if length > MAX_BODY_BYTES:
                raise ChatError(f"body is {length} bytes; the limit is {MAX_BODY_BYTES}")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ChatError(f"body is not valid JSON: {exc}") from exc

        # -- routes --------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            if self.path.rstrip("/") in ("/info", ""):
                payload = dict(state.config.info())
                payload["n_sessions"] = state.store.n_sessions
                self._send(200, payload)
                return
            self._send(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            route = self.path.rstrip("/")
            try:
                if route == "/chat":
                    session, branch, text = parse_chat_request(self._read_json())
                    self._send(200, state.exchange(session, branch, text))
                    return
                if route == "/reset":
                    session = parse_reset_request(self._read_json())
                    existed = state.store.reset(session)
                    logger.info("session %s reset (existed: %s)", session, existed)
                    self._send(200, {"cleared": existed, "session": session})
                    return
                self._send(404, {"error": f"no such endpoint: {self.path}"})
            except ChatError as exc:
                self._send(400, {"error": str(exc)})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the server
                logger.exception("request to %s failed", self.path)
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return DuetHandler


def resolve_source(pairs: list[PairRecord], pair_key: str) -> PairRecord:
    """The source pair, or a ValueError that actually helps.

    An unknown key is the commonest way to start this server wrong, and "KeyError"
    is a useless thing to read at that moment — so the message carries a handful of
    real val_seed keys to copy.
    """
    for pair in pairs:
        if pair.pair_key == pair_key:
            return pair
    suggestions = sorted(p.pair_key for p in pairs if p.split == "val_seed")[:8]
    if not suggestions:
        suggestions = sorted(p.pair_key for p in pairs)[:8]
    raise ValueError(
        f"--source-pair-key {pair_key!r} is not in this pairs dir. "
        f"Valid keys look like: {', '.join(suggestions)}"
        + (" …" if len(pairs) > len(suggestions) else "")
    )


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument(
        "--source-pair-key",
        required=True,
        help="which trajectory's member code the fitted branch wears",
    )
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--lever-npz", type=Path, default=None,
        help="Build D: wear a TRUE cluster-contrast lever (build_levers npz) "
             "instead of the g member code; requires --lever-group",
    )
    parser.add_argument(
        "--lever-group", default=None,
        help="which lever to wear, as 'prompt_id|wave' (e.g. 'if5|orig')",
    )
    parser.add_argument(
        "--lever-sign", type=float, default=1.0,
        help="+1 = toward cluster A, -1 = toward cluster B",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b", help="anamnesis MODEL_PRESETS key")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="LOOPBACK ONLY; reach it with 'ssh -L 8765:localhost:8765 <host>'",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="scales g's bias; 1.0 = exactly what g emits (0 makes fitted == base)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--seed", type=int, default=0, help="base of the per-(session, branch, turn) seed"
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="default none — B0/B1 prompts are bare by design (gen_forks' premise)",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=24,
        help="exchanges kept per branch before the oldest are dropped (0 = unbounded)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="JSONL transcript dir (default: none)"
    )
    parser.add_argument(
        "--selfcheck-prompt",
        default="Say a few words about anything that interests you.",
        help="the user turn the zero-bias byte-identity gate is drawn from",
    )
    parser.add_argument("--selfcheck-tokens", type=int, default=32)
    parser.add_argument(
        "--model-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.lever_npz is None) != (args.lever_group is None):
        parser.error("--lever-npz and --lever-group must be given together")

    if not 0 < args.top_p <= 1:
        logger.error("--top-p must be in (0, 1], got %s", args.top_p)
        return 2
    if args.temperature <= 0:
        logger.error("--temperature must be > 0 (do_sample=True), got %s", args.temperature)
        return 2
    if args.max_new_tokens <= 0 or args.selfcheck_tokens <= 0:
        logger.error("--max-new-tokens and --selfcheck-tokens must be > 0")
        return 2
    if args.seed < 0:
        logger.error("--seed must be >= 0, got %s", args.seed)
        return 2
    if not 0 < args.port < 65536:
        logger.error("--port must be in (0, 65536), got %s", args.port)
        return 2
    try:
        host = require_loopback(str(args.host))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if args.alpha == 0.0:
        logger.warning(
            "--alpha 0 makes the fitted branch identical to the base branch — that is "
            "a valid null duet, but say so in the transcript."
        )

    try:
        loaded = load_pairs(args.pairs_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error("could not load --pairs-dir %s: %s", args.pairs_dir, exc)
        return 2
    try:
        source = resolve_source(loaded.pairs, str(args.source_pair_key))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "fitted branch wears %s (prompt %s / %s, split %s)",
        source.pair_key, source.prompt_id, source.prompt_class, source.split,
    )

    # Deferred so --help works on any machine, with or without the node env.
    import numpy as np
    import torch

    torch.set_num_threads(1)
    # B200 / torch 2.11 (forensics 2026-09-16): SDPA prefers the cuDNN attention
    # backend on sm_100, and cuDNN builds a graph on the HOST for every unseen
    # (q,k,v,mask) shape — ~340 ms each, cached PER THREAD. Decode grows kv_len by
    # one per token, so the first pass over any unseen length range runs ~3 tok/s,
    # and a chat turn whose history has grown is always an unseen range. Measured:
    # 24 fresh kv_len values cost ~400 ms each with cuDNN on, 0.25 ms off; a
    # 512-token duet turn went 183 s -> 7.1 s. Flash/mem-efficient stay enabled
    # (both exact). Process-global; must run before the first generate. Note:
    # a different kernel means tokens are not bitwise-comparable to pre-fix
    # transcripts; the base/fitted byte-identity gate is unaffected (same kernel
    # both branches).
    torch.backends.cuda.enable_cudnn_sdp(False)

    from anamnesis.config import MODEL_PRESETS
    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pleroma import g_net

    if args.preset not in MODEL_PRESETS:
        logger.error("unknown preset %r; have %s", args.preset, sorted(MODEL_PRESETS))
        return 2
    preset = MODEL_PRESETS[args.preset]
    eos_ids = list(preset.eos_token_ids)
    if not eos_ids:
        logger.error("preset %r declares no eos ids", args.preset)
        return 2

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("--device %s but torch.cuda.is_available() is False", device)
        return 2

    # ── g ─────────────────────────────────────────────────────────────────────
    try:
        checkpoint = torch.load(args.g_checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("could not read --g-checkpoint %s: %s", args.g_checkpoint, exc)
        return 2
    arch = checkpoint.get("arch")
    if not arch:
        logger.error("%s has no 'arch' block — not a train_g checkpoint", args.g_checkpoint)
        return 2
    if int(arch["in_dim"]) != loaded.feature_dim:
        logger.error(
            "g was trained on %d-d signatures but the pairs dir has %d — different spaces",
            arch["in_dim"], loaded.feature_dim,
        )
        return 2
    ckpt_names = checkpoint.get("feature_names")
    if ckpt_names and list(ckpt_names) != loaded.feature_names:
        logger.error(
            "g's training feature_names differ from this pairs dir's — refusing to "
            "wear a code from another feature space"
        )
        return 2
    try:
        train_meta, train_meta_path = read_train_meta(args.g_checkpoint)
        inject_layers, g_hidden = reconcile_g_config(arch, train_meta, train_meta_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("g config unusable: %s", exc)
        return 2

    # ── the model, loaded for generation ──────────────────────────────────────
    dtype = g_net.DTYPES[str(args.model_dtype)]
    logger.info("loading %s (sdpa, %s)", args.model_path, dtype)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path)
        if tok.chat_template is None:
            logger.error(
                "%s has no chat template — a duet IS a chat; refusing to fall back to "
                "raw concatenation", args.model_path,
            )
            return 2
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        # use_cache is fine and wanted: it is the WITHIN-request prefill/decode cache.
        # No cache ever crosses a request — see the module docstring.
        model.config.use_cache = True
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("could not load the model: %s", exc)
        return 1

    hidden_dim = g_net.model_hidden_dim(model)
    n_model_layers = len(g_net.decoder_layers(model))
    if int(arch["hidden_dim"]) != hidden_dim:
        logger.error(
            "g emits %d-d biases but the model's hidden_dim is %d",
            arch["hidden_dim"], hidden_dim,
        )
        return 2
    if int(preset.num_layers) != n_model_layers:
        logger.error(
            "preset %s says %d layers but %s has %d", args.preset, preset.num_layers,
            args.model_path, n_model_layers,
        )
        return 2
    try:
        attach_layers = attach_layers_for(inject_layers, n_model_layers)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "INJECTION SITE: g trained on the OUTPUT of layers %s == the INPUT of layers "
        "%s; attach_residual_write is a forward_PRE_hook, so attaching at %s (+1).",
        inject_layers, attach_layers, attach_layers,
    )

    g = g_net.SignatureToBias(
        in_dim=int(arch["in_dim"]),
        hidden=g_hidden,
        n_layers=int(arch["n_layers"]),
        hidden_dim=hidden_dim,
    )
    g.load_state_dict(checkpoint["state_dict"])
    g.to(device=device, dtype=torch.float32).eval()
    g.requires_grad_(False)

    # The member code, computed ONCE. It is a fixed vector for the life of the
    # server: the fitted branch wears the same code in turn 1 and in turn 40.
    z = torch.from_numpy(
        np.ascontiguousarray(loaded.z[source.row].reshape(1, -1).astype(np.float32))
    ).to(device)
    with torch.no_grad():
        bias = g(z)
    bias_vectors = [bias[0, i].detach().to(torch.float32).clone() for i in range(bias.shape[1])]
    bias_l2 = [float(v.norm()) for v in bias_vectors]
    logger.info("member code from %s: per-layer bias L2 %s (alpha %.3f)",
                source.pair_key, [round(v, 4) for v in bias_l2], args.alpha)

    if args.lever_npz is not None:
        # Build D (2026-09-16): wear a TRUE cluster-contrast lever instead of the
        # g output. Build B measured these pulling free generation ±30-50pp of
        # basin membership; the g path above still loads (kept for the tested
        # startup path) but its bias is REPLACED here.
        lev = np.load(Path(args.lever_npz).expanduser(), allow_pickle=True)
        group_keys = [
            f"{p}|{w}" for p, w in zip(lev["group_prompt_ids"], lev["group_waves"])
        ]
        if args.lever_group not in group_keys:
            logger.error("--lever-group %r not in levers npz; available: %s",
                         args.lever_group, group_keys)
            return 2
        lever_sites = [int(s) for s in lev["sites"]]
        if lever_sites != list(attach_layers):
            logger.error("lever sites %s != duet attach layers %s — refusing a "
                         "silent site mismatch", lever_sites, list(attach_layers))
            return 2
        gi = group_keys.index(args.lever_group)
        sign = float(args.lever_sign)
        bias_vectors = [
            torch.from_numpy((sign * lev["levers"][gi, i]).astype(np.float32)).clone()
            for i in range(len(lever_sites))
        ]
        bias_l2 = [float(v.norm()) for v in bias_vectors]
        logger.warning(
            "LEVER MODE: fitted branch wears %s (sign %+g, alpha %.3f) — raw "
            "cluster-contrast lever, per-site L2 %s; the g member code above is "
            "NOT in use", args.lever_group, sign, args.alpha,
            [round(v, 4) for v in bias_l2],
        )

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]

    def attach(vectors: list[Any], alpha: float) -> list[Any]:
        """Attach the member code UNIFORMLY (start_pos=0) at the +1-shifted layers."""
        handles: list[Any] = []
        try:
            for vector, layer_idx in zip(vectors, attach_layers, strict=True):
                handles.append(
                    attach_residual_write(
                        model,
                        ResidualWriteSpec(
                            layer_idx=layer_idx,
                            vector=vector,
                            alpha=alpha,
                            # Uniform over the whole context: under re-prefill this is
                            # the only choice that keeps a turn's history in the same
                            # regime it was generated in.
                            start_pos=0,
                            end_pos=None,
                            # g's magnitude is part of what it learned; alpha is the
                            # operator's dose on top of that, not a renormalisation.
                            normalize=False,
                        ),
                    )
                )
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        return handles

    def render(messages: list[dict[str, str]]) -> Any:
        result = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
        return ids.to(device)

    # All GPU generation runs on ONE persistent worker thread. Empirical
    # (bench_order 2026-09-16): this stack pays a ~40s warm-up that is
    # THREAD-scoped — the same generate call runs at ~3 tok/s in a fresh thread
    # and ~70 tok/s in a warmed one. ThreadingHTTPServer spawns a new thread per
    # request, so without this every request paid the cold path forever. The
    # startup gate below runs through the same executor, pre-warming the thread.
    from concurrent.futures import ThreadPoolExecutor

    gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")

    def draw(input_ids: Any, seed: int, max_new: int) -> list[int]:
        return gen_executor.submit(_draw_inner, input_ids, seed, max_new).result()

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

    # ── the not-a-perturbation gate, in THIS server's configuration ───────────
    probe_ids = render([{"role": "user", "content": str(args.selfcheck_prompt)}])
    probe_len = int(probe_ids.shape[1])
    probe_seed = derive_turn_seed(int(args.seed), "__selfcheck__", "base", 0)
    n_probe = int(args.selfcheck_tokens)
    selfcheck: dict[str, Any] = {"seed": probe_seed, "n_tokens_requested": n_probe}
    try:
        plain = draw(probe_ids, probe_seed, n_probe)
        zeros = [torch.zeros(hidden_dim, dtype=torch.float32) for _ in attach_layers]

        handles = attach(zeros, 0.0)
        try:
            alpha_zero = draw(probe_ids, probe_seed, n_probe)
        finally:
            for handle in handles:
                handle.remove()

        handles = attach(zeros, float(args.alpha) or 1.0)
        try:
            vector_zero = draw(probe_ids, probe_seed, n_probe)
            stats = [dict(h.stats) for h in handles]
        finally:
            for handle in handles:
                handle.remove()
    except Exception as exc:  # noqa: BLE001 — the gate is allowed to fail loudly
        logger.exception("self-check could not run: %s", exc)
        return 3

    selfcheck["n_tokens_drawn"] = len(plain) - probe_len
    selfcheck["alpha_zero_identical"] = plain == alpha_zero
    selfcheck["zero_vector_identical"] = plain == vector_zero
    saw_cache_position = all(bool(s.get("saw_cache_position")) for s in stats)
    injected = [int(s.get("positions", 0)) for s in stats]
    selfcheck["saw_cache_position"] = saw_cache_position
    selfcheck["positions_injected"] = injected
    if not selfcheck["alpha_zero_identical"] or not selfcheck["zero_vector_identical"]:
        logger.error(
            "SELF-CHECK FAILED — a zero bias changed the sampled tokens (alpha=0: %s, "
            "zero-vector: %s). The injection path is a perturbation in its own right, "
            "so the fitted branch would differ from base for reasons that are not the "
            "member code. Refusing to open the socket.",
            selfcheck["alpha_zero_identical"], selfcheck["zero_vector_identical"],
        )
        return 3
    # Uniform injection means every position of prompt AND continuation is hit.
    expected = probe_len + selfcheck["n_tokens_drawn"]
    if not saw_cache_position or not all(n > 0 for n in injected):
        logger.error(
            "SELF-CHECK FAILED — hooks were byte-neutral but never injected: "
            "saw_cache_position=%s, positions=%s. A gate that never fires would make "
            "the fitted branch a second base branch.", saw_cache_position, injected,
        )
        return 3
    if any(n != expected for n in injected):
        logger.warning(
            "self-check: injected %s positions per layer, expected %d (prompt %d + "
            "%d new). Not fatal, but the uniform-position claim is worth re-reading.",
            injected, expected, probe_len, selfcheck["n_tokens_drawn"],
        )
    logger.info(
        "SELF-CHECK PASSED — %d tokens byte-identical under no-hook / alpha=0 / "
        "zero-vector; uniform gating fired at %s positions per layer",
        selfcheck["n_tokens_drawn"], injected,
    )

    # (2026-09-16 forensics: the earlier warm-up draw and keepalive that lived
    # here were removed — both were cargo-cult mitigations for what turned out to
    # be cuDNN SDPA host-side graph builds per unseen kv_len, fixed for real by
    # enable_cudnn_sdp(False) at load time. A keepalive at a cached length builds
    # nothing; a warm-up only pre-built lengths real turns never reuse.)

    # attach_residual_write logs one INFO line per registration and the fitted branch
    # re-attaches every turn; quieted only after the gate, and only to WARNING.
    from anamnesis.extraction import model_loader as _model_loader

    _model_loader.logger.setLevel(logging.WARNING)

    # ── the model layer the HTTP layer sees ───────────────────────────────────
    def generate(
        messages: list[dict[str, str]], branch: str, seed: int
    ) -> tuple[str, dict[str, Any]]:
        input_ids = render(messages)
        prompt_len = int(input_ids.shape[1])
        handles: list[Any] = []
        if branch == "fitted":
            handles = attach(bias_vectors, float(args.alpha))
        try:
            full = draw(input_ids, seed, int(args.max_new_tokens))
        finally:
            # ALWAYS: a handle left registered would silently make the next base
            # turn a fitted one.
            for handle in handles:
                handle.remove()
        new_ids = full[prompt_len:]
        return tok.decode(new_ids, skip_special_tokens=True).strip(), {
            "n_prompt_tokens": prompt_len,
            "n_new_tokens": len(new_ids),
            "n_messages": len(messages),
        }

    log_path: Path | None = None
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"duet_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        logger.info("exchange log -> %s", log_path)

    config = ServeConfig(
        source_pair_key=source.pair_key,
        prompt_id=source.prompt_id,
        prompt_class=source.prompt_class,
        source_split=source.split,
        alpha=float(args.alpha),
        inject_layers=list(inject_layers),
        attach_layers=list(attach_layers),
        model_path=str(args.model_path),
        g_checkpoint=str(args.g_checkpoint),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        system_prompt=args.system_prompt,
        max_history_turns=int(args.max_history_turns),
        bias_l2=bias_l2,
    )
    state = DuetState(
        config=config, store=SessionStore(), generate=generate, log_path=log_path
    )
    if log_path is not None:
        state.log({"event": "serve_start", "config": config.info(), "selfcheck": selfcheck})

    server = ThreadingHTTPServer((host, int(args.port)), make_handler(state))
    logger.info("duet server on http://%s:%d — loopback only", host, args.port)
    logger.info(
        "from a laptop: ssh -L %d:localhost:%d <host>, then "
        "python -m pleroma.duet_chat --port %d", args.port, args.port, args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        server.server_close()
        if log_path is not None:
            state.log({"event": "serve_stop", "n_sessions": state.store.n_sessions})
        logger.info("duet server stopped after %d session(s)", state.store.n_sessions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
