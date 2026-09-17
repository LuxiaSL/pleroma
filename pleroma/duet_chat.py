"""Exp B1 stage 5b: the duet client. Local, pure stdlib (urllib + readline).

You type once; both branches answer. ``duet_serve`` on the node holds one frozen 3B
and two conversations — ``base`` (as it ships) and ``fitted`` (wearing a source
trajectory's member code as a persistent residual bias) — each with its own history.
This is the terminal you meet them in.

    open mode    replies are labelled [BASE] and [FITTED]. For getting a feel for
                 what the code does at all, and for the first read.

    blind mode   at session start the two branches are shuffled onto the letters A
                 and B; replies are labelled [A] and [B] and NOTHING else in the
                 session says which is which. The key is written to
                 ``<transcript>.key.json`` IMMEDIATELY — before the first reply is
                 printed — so the answer exists on disk from the start and cannot be
                 back-fitted to what you decided. ``/reveal`` unblinds when you are
                 done, and appends the key to the transcript so the call and the
                 answer live in one file.

Blinding is the client's job by design. The server answers ``/info`` honestly about
which pair the fitted branch wears (it does not know the letters), because a server
that lied to its own operator would be a worse instrument than one that does not.

The transcript is appended AS IT HAPPENS, not at exit: a crash, a dropped tunnel or
a Ctrl-C mid-session leaves everything said so far on disk. The transcript is the
data; the scrollback is not.

Connect through an SSH tunnel — ``duet_serve`` binds loopback only::

    ssh -L 8765:localhost:8765 <host>     # in another terminal, leave it running
    python -m pleroma.duet_chat --mode blind

Commands inside the session: ``/reset`` (clear both branches), ``/reveal`` (unblind),
``/info`` (what the server is serving), ``/help``, ``/quit``.
"""

from __future__ import annotations

import os

# Pin CPU thread pools BEFORE any numeric import (house rule; nothing here needs
# them, and the pins are cheap insurance against an import pulling BLAS in).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Must match ``duet_serve.BRANCHES``. Imported rather than restated would drag the
#: server module (and its pairs-manifest import) into a client that needs neither;
#: the two are asserted equal by the tests instead.
BRANCHES: tuple[str, ...] = ("base", "fitted")

#: Letters the blind mode assigns, in display order.
LETTERS: tuple[str, ...] = ("A", "B")

#: Printed whenever the socket is not there, because it is nearly always this.
TUNNEL_HINT = (
    "Could not reach the duet server at {url}.\n"
    "  duet_serve binds LOOPBACK ONLY on the node, so you need the tunnel:\n"
    "      ssh -L {port}:localhost:{port} <host>\n"
    "  Leave that running in another terminal, check the server is up there, "
    "then retry."
)

HELP = """commands:
  /reset    clear both branches' histories on the server (same session id)
  /reveal   print the blind key (and append it to the transcript)
  /info     what the server is serving
  /help     this
  /quit     leave (the transcript is already written)
"""


class DuetError(RuntimeError):
    """The server answered, but not with a reply."""


class DuetUnreachable(RuntimeError):
    """The server did not answer at all — almost always a missing tunnel."""


# ── blinding ───────────────────────────────────────────────────────────────────


def assign_letters(seed: int) -> dict[str, str]:
    """{branch: letter} for one blind session. Deterministic in ``seed``.

    The shuffle is over the BRANCHES, so both assignments are reachable and neither
    is the default. Recorded in the key file with the seed, so a session can be
    reconstructed exactly from the key alone.
    """
    order = list(BRANCHES)
    random.Random(seed).shuffle(order)
    return {branch: letter for branch, letter in zip(order, LETTERS, strict=True)}


def display_order(mode: str, letters: dict[str, str] | None) -> list[tuple[str, str]]:
    """[(branch, label)] in the order replies are PRINTED.

    In blind mode the order is by LETTER (A then B), never by branch — printing
    base-then-fitted under letter labels would leak the whole mapping on line one.
    In open mode it is base then fitted, which is the reading order that makes
    sense when you already know.
    """
    if mode == "blind":
        if not letters:
            raise ValueError("blind mode needs a letter assignment")
        by_letter = {letter: branch for branch, letter in letters.items()}
        return [(by_letter[letter], f"[{letter}]") for letter in LETTERS]
    return [(branch, f"[{branch.upper()}]") for branch in BRANCHES]


def key_payload(
    session: str, seed: int, letters: dict[str, str], transcript: Path
) -> dict[str, Any]:
    """The contents of ``<transcript>.key.json``."""
    return {
        "stage": "expB1_duet_chat",
        "note": (
            "'by_letter' maps what you were shown to which branch produced it. "
            "'fitted' wore the source trajectory's member code; 'base' is the model "
            "as it ships. Written before the first reply was printed."
        ),
        "session": session,
        "seed": seed,
        "mode": "blind",
        "letters": dict(letters),
        "by_letter": {letter: branch for branch, letter in letters.items()},
        "transcript": str(transcript),
    }


def render_key_markdown(payload: dict[str, Any]) -> str:
    """The unblinding, as it is appended to the transcript."""
    by_letter = payload["by_letter"]
    lines = ["", "---", "", "## KEY REVEALED", ""]
    lines.extend(f"- **{letter}** = `{by_letter[letter]}`" for letter in LETTERS)
    lines.append("")
    lines.append(f"(session `{payload['session']}`, seed `{payload['seed']}`)")
    lines.append("")
    return "\n".join(lines)


# ── transcript ─────────────────────────────────────────────────────────────────


def render_header(session: str, mode: str, port: int, info: dict[str, Any] | None) -> str:
    """The transcript preamble. In blind mode it must not name the mapping."""
    lines = [
        f"# Duet transcript — session `{session}`",
        "",
        f"- mode: **{mode}**",
        f"- server: `http://127.0.0.1:{port}`",
        f"- started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if mode == "blind":
        lines.append(
            "- **BLINDED** — which of A/B is the fitted branch is in the `.key.json` "
            "beside this file. Do not open it before you have called the session."
        )
    if info:
        lines.append(f"- fitted branch wears: `{info.get('source_pair_key')}` "
                     f"(prompt `{info.get('prompt_id')}` / {info.get('prompt_class')}), "
                     f"alpha {info.get('alpha')}")
        lines.append(f"- inject layers {info.get('inject_layers')} "
                     f"-> attached at {info.get('attach_layers')}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def render_exchange(turn: int, user: str, replies: Sequence[tuple[str, str]]) -> str:
    """One markdown block: the user turn, then each labelled reply, in given order."""
    lines = [f"## Turn {turn}", "", "**You**", ""]
    lines.extend(f"> {line}" for line in user.splitlines() or [""])
    lines.append("")
    for label, reply in replies:
        lines.append(f"**{label}**")
        lines.append("")
        lines.append(reply if reply.strip() else "_(empty reply)_")
        lines.append("")
    return "\n".join(lines)


@dataclass
class Transcript:
    """Append-as-you-go markdown. Never buffers a session in memory to the end."""

    path: Path

    def append(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")


# ── the wire ───────────────────────────────────────────────────────────────────


@dataclass
class DuetClient:
    """Thin urllib client. One method per endpoint, errors typed, no retries.

    No retry loop on purpose: a duet turn is a GPU generation the operator is
    waiting on, and silently re-sending it would double-append to the server's
    history for that branch.
    """

    base_url: str
    timeout: float = 600.0

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(route, json.dumps(payload).encode("utf-8"))

    def _request(self, route: str, data: bytes | None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{route}"
        request = urllib.request.Request(
            url, data=data, method="POST" if data is not None else "GET"
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                message = detail
            raise DuetError(f"{route} -> HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DuetUnreachable(str(exc)) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise DuetError(f"{route} -> body is not JSON: {body[:200]}") from exc

    def chat(self, session: str, branch: str, text: str) -> dict[str, Any]:
        return self._post("/chat", {"session": session, "branch": branch, "text": text})

    def reset(self, session: str) -> dict[str, Any]:
        return self._post("/reset", {"session": session})

    def info(self) -> dict[str, Any]:
        return self._request("/info", None)


# ── the session ────────────────────────────────────────────────────────────────


def _print_unreachable(url: str, port: int, exc: Exception) -> None:
    print(TUNNEL_HINT.format(url=url, port=port), file=sys.stderr)
    print(f"  (underlying error: {exc})", file=sys.stderr)


def run_session(
    client: DuetClient,
    transcript: Transcript,
    session: str,
    mode: str,
    port: int,
    letters: dict[str, str] | None,
    key: dict[str, Any] | None,
    read_line: Any,
) -> int:
    """The conversation loop. ``read_line`` is injected so tests can drive it.

    Returns a process exit code. Every exit path — /quit, EOF, Ctrl-C, a dead
    server — leaves the transcript complete up to the last finished exchange,
    because each exchange was appended when it happened.
    """
    order = display_order(mode, letters)
    turn = 0
    revealed = False
    while True:
        try:
            line = read_line("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line is None:
            break
        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            command = text.split()[0].lower()
            if command in ("/quit", "/exit"):
                break
            if command == "/help":
                print(HELP)
                continue
            if command == "/reset":
                try:
                    client.reset(session)
                except DuetUnreachable as exc:
                    _print_unreachable(client.base_url, port, exc)
                    return 1
                except DuetError as exc:
                    print(f"server: {exc}", file=sys.stderr)
                    continue
                turn = 0
                transcript.append("\n---\n\n_(histories reset — both branches start over)_\n")
                print("both branches reset.")
                continue
            if command == "/info":
                try:
                    print(json.dumps(client.info(), indent=2))
                except DuetUnreachable as exc:
                    _print_unreachable(client.base_url, port, exc)
                    return 1
                except DuetError as exc:
                    print(f"server: {exc}", file=sys.stderr)
                continue
            if command == "/reveal":
                if mode != "blind" or key is None:
                    print("not a blind session — replies were already labelled.")
                    continue
                print(json.dumps(key["by_letter"], indent=2))
                if not revealed:
                    transcript.append(render_key_markdown(key))
                    revealed = True
                continue
            print(f"unknown command {command}. {HELP}")
            continue

        turn += 1
        replies: list[tuple[str, str]] = []
        try:
            for branch, label in order:
                result = client.chat(session, branch, text)
                replies.append((label, str(result.get("reply", ""))))
        except DuetUnreachable as exc:
            _print_unreachable(client.base_url, port, exc)
            if replies:
                transcript.append(render_exchange(turn, text, replies))
                transcript.append("\n_(the server became unreachable mid-turn)_\n")
            return 1
        except DuetError as exc:
            print(f"server: {exc}", file=sys.stderr)
            if replies:
                transcript.append(render_exchange(turn, text, replies))
                transcript.append(f"\n_(turn incomplete: {exc})_\n")
            turn -= 1
            continue

        # Written BEFORE it is printed: if the terminal dies between the two, the
        # disk is the one that was right.
        transcript.append(render_exchange(turn, text, replies))
        for label, reply in replies:
            print(f"\n{label}\n{reply}\n")
    return 0


def main() -> int:  # noqa: C901 — one linear procedure, sectioned for readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        default="open",
        choices=["open", "blind"],
        help="open = replies labelled [BASE]/[FITTED]; blind = [A]/[B] with a key file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--session", default=None, help="server-side session id (default: random)"
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="markdown transcript, appended as the session happens",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="blind letter assignment (default: random, and recorded in the key)",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request seconds")
    args = parser.parse_args()

    if not 0 < args.port < 65536:
        print(f"--port must be in (0, 65536), got {args.port}", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print(f"--timeout must be > 0, got {args.timeout}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = str(args.session) if args.session else f"duet-{uuid.uuid4().hex[:8]}"
    transcript_path = args.transcript or Path(f"./duet_transcript_{stamp}.md")
    transcript = Transcript(transcript_path)

    base_url = f"http://{args.host}:{args.port}"
    client = DuetClient(base_url=base_url, timeout=float(args.timeout))

    try:
        info = client.info()
    except DuetUnreachable as exc:
        _print_unreachable(base_url, int(args.port), exc)
        return 1
    except DuetError as exc:
        print(f"server answered but not usefully: {exc}", file=sys.stderr)
        return 1

    letters: dict[str, str] | None = None
    key: dict[str, Any] | None = None
    if args.mode == "blind":
        seed = int(args.seed) if args.seed is not None else random.randrange(2**31)
        letters = assign_letters(seed)
        key = key_payload(session, seed, letters, transcript_path)
        key_path = Path(str(transcript_path) + ".key.json")
        # IMMEDIATELY, before a single reply is printed: the answer must be on disk
        # from the start, or the test is "did I later decide I had known".
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(json.dumps(key, indent=2))
        print(f"BLIND session. Key already written to {key_path} — do not open it yet.")

    transcript.append(render_header(session, str(args.mode), int(args.port), info))
    print(
        f"duet session {session} ({args.mode}) against {base_url}\n"
        f"fitted branch wears {info.get('source_pair_key')} "
        f"(prompt {info.get('prompt_id')} / {info.get('prompt_class')}), "
        f"alpha {info.get('alpha')}\n"
        f"transcript -> {transcript_path}\n"
        f"{HELP}"
    )

    # readline gives line editing and history to input(); its absence is survivable
    # and not worth failing a conversation over.
    try:
        import readline  # noqa: F401  (imported for its side effect on input())
    except ImportError:  # pragma: no cover — present on every platform we run on
        pass

    try:
        code = run_session(
            client, transcript, session, str(args.mode), int(args.port), letters, key,
            input,
        )
    except KeyboardInterrupt:
        print()
        code = 0
    transcript.append(
        f"\n---\n\n_(session ended {time.strftime('%Y-%m-%d %H:%M:%S')})_\n"
    )
    print(f"transcript -> {transcript_path}")
    if args.mode == "blind":
        print(f"key        -> {str(transcript_path)}.key.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
