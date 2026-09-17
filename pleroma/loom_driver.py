"""LOOM A/B driver — run the loop end to end under different auto policies.

One fixed user script, N arms. Every arm gets the SAME user turns and the
SAME per-turn fan measurement (/loom is called every turn in every arm — it
commits nothing, so the fan is a free instrument reading); arms differ ONLY
in what happens to the fan: nothing (arm "none"), or auto-wear under a
policy at a dose. stay/swerve need something worn, so their first turn seeds
with "distinct" (recorded per turn as the server's effective_policy).

Emits one JSONL per arm (every /loom and /chat response, verbatim) and a
summary.json: per-turn spread, camps, worn evolution, pick scores — the raw
material for "what falls out of the loop under policy X".

Runs ON THE NODE against localhost (the /loom calls are ~14s each):
    python -m pleroma.loom_driver \\
        --server http://127.0.0.1:8767 \\
        --script pleroma/loomlab_script.json \\
        --arms none distinct:0.35 minority:0.35 loudest:0.35 stay:0.35 swerve:0.35 \\
        --run-id lab001 --k 6 --horizon 192 --out outputs/loomlab/lab001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("loom_driver")


def post(server: str, path: str, body: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{server}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_arm(spec: str) -> dict[str, Any]:
    """'none' or 'policy:alpha' (e.g. 'distinct:0.35')."""
    if spec == "none":
        return {"name": "none", "policy": None, "alpha": None}
    if ":" not in spec:
        raise ValueError(f"arm {spec!r} is not 'none' or 'policy:alpha'")
    policy, alpha = spec.split(":", 1)
    return {"name": spec.replace(":", "_a"), "policy": policy, "alpha": float(alpha)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://127.0.0.1:8767")
    parser.add_argument("--script", type=Path, required=True,
                        help="JSON: {\"turns\": [str, ...]}")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    turns = [str(t) for t in json.loads(args.script.read_text())["turns"]]
    arms = [parse_arm(a) for a in args.arms]
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run_id": args.run_id, "k": args.k,
                               "horizon": args.horizon, "n_turns": len(turns),
                               "script": str(args.script), "arms": {}}

    for arm in arms:
        session = f"{args.run_id}-{arm['name']}"
        post(args.server, "/reset", {"session": session})
        rows: list[dict[str, Any]] = []
        arm_track: list[dict[str, Any]] = []
        t_arm = time.time()
        for ti, text in enumerate(turns):
            body: dict[str, Any] = {"session": session, "text": text,
                                    "k": int(args.k), "horizon": int(args.horizon)}
            if arm["policy"] is not None:
                # stay/swerve need wear; seed turn 0 with distinct. The server
                # reports effective_policy either way — the record stays honest.
                policy = arm["policy"]
                if policy in ("stay", "swerve") and ti == 0:
                    policy = "distinct"
                body["auto"] = {"policy": policy, "alpha": arm["alpha"]}
            loom = post(args.server, "/loom", body)
            rows.append({"event": "loom", "turn": ti, **loom})
            chat = post(args.server, "/chat",
                        {"session": session, "branch": "loom", "text": text})
            rows.append({"event": "chat", "turn": ti, **chat})
            worn = chat.get("worn")
            arm_track.append({
                "turn": ti,
                "spread": loom.get("spread"),
                "auto_selected": loom.get("auto_selected"),
                "worn": worn,
                "reply_chars": len(chat.get("reply", "")),
                "loom_s": loom.get("timing_s"),
            })
            logger.info("[%s] turn %d: spread %.3f worn %s",
                        arm["name"], ti,
                        (loom.get("spread") or {}).get("mean_pairwise_distance", -1),
                        None if not worn else f"#{worn['index']}@{worn['alpha']}")
        out_path = args.out / f"arm_{arm['name']}.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in rows))
        summary["arms"][arm["name"]] = {
            "session": session, "track": arm_track,
            "elapsed_s": round(time.time() - t_arm, 1),
        }
        logger.info("arm %s done in %.0fs -> %s", arm["name"],
                    time.time() - t_arm, out_path)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("DONE -> %s", args.out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
