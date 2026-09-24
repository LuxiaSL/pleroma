"""The loom server's command line: the parser and the sampling-default resolution.

The flag set is a stable interface: the argv ``pleroma/config/legacy.py``
renders from a profile must parse here, so a flag is added, never renamed.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pleroma.dose.policy import (
    DEFAULT_DOSE_POLICY,
    DOSE_POLICIES,
    DOSE_SCALE_MAX,
    DOSE_SCALE_MIN,
)
from pleroma.format.modelc import (
    DEFAULT_HEADER as MODELC_DEFAULT_HEADER,
    HEADERS as MODELC_HEADERS,
    TEMPERATURE as MODELC_TEMPERATURE,
    TOP_P as MODELC_TOP_P,
)
from pleroma.format.prompt import PROMPT_MODES
from pleroma.levers.kind import DEFAULT_LEVER_KIND, LEVER_KINDS

logger = logging.getLogger("loom_serve")

#: The parser description: the first line of the server's docstring.
DESCRIPTION = ("LOOM v0 — the model looms over its own futures and bends toward "
               "one you pick.")


def build_parser() -> argparse.ArgumentParser:
    """The loom server's argparse parser (39 flags + the 3 API flags
    --cors-origin, --api-token, --ui-dir)."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--discriminants", type=Path, required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preset", default="3b")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--seed", type=int, default=20260917)
    # ★ DEFAULT None, resolved below. The resolved value for chat and raw is
    # 1.0 / 0.95, the sampling every chat-mode result was measured at. Only
    # --prompt-mode modelc resolves differently (1.0 / 0.98, the format spec's
    # standing house rule), and only when the operator said nothing.
    parser.add_argument("--temperature", type=float, default=None,
                        help="default 1.0 (all modes)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="default 0.95; 0.98 under --prompt-mode modelc, "
                             "model C's format rule (emphatically 0.98, not "
                             "0.95 — 'keeps the novel tail'). An explicit "
                             "--top-p always wins.")
    parser.add_argument(
        "--prompt-mode", choices=list(PROMPT_MODES), default="chat",
        help="how a turn becomes a prompt. 'chat' (DEFAULT) is "
             "apply_chat_template, add_generation_prompt — the render every "
             "banked result was measured through. 'raw' is bare "
             "document continuation, no template — the framing lever worth "
             "Jaccard .2467 -> .1015 and on_task .582 -> .366 on 8B instruct "
             "(docs/FINDINGS.md §11). 'modelc' "
             "assembles model C's conversation document "
             "(pleroma.format.modelc). See pleroma.serve's module "
             "docstring, PROMPT MODES, for how raw joins history.",
    )
    parser.add_argument(
        "--date-string", default=None,
        help="chat mode: the date pinned into chat templates that take one "
             "(Llama 3.x 'Today Date'). Set it to the date the map's corpus "
             "was generated under; unset keeps the template's own default.",
    )
    parser.add_argument(
        "--modelc-header", default=MODELC_DEFAULT_HEADER,
        help="prompt-mode modelc only: one of "
             + "|".join(sorted(MODELC_HEADERS))
             + ", or any literal custom header line (the spec's "
               "'keeper/custom headers follow the same shape', including the "
               "training-attested '... about {topic}.' suffix form). "
               "'bare' omits the header and begins at the bridge line. "
               f"Default {MODELC_DEFAULT_HEADER!r}.",
    )
    parser.add_argument(
        "--no-modelc-stops", dest="modelc_stops", action="store_false",
        help="prompt-mode modelc only: do NOT pass the format spec's stop "
             "strings. The format's own advice for multi-post threads in one "
             "generation. Stops are ON by default in modelc mode and are "
             "never passed in chat or raw mode.",
    )
    parser.set_defaults(modelc_stops=True)
    parser.add_argument("--max-new-tokens", type=int, default=320, help="chat replies")
    parser.add_argument("--future-tokens", type=int, default=192, help="per future")
    parser.add_argument("--default-k", type=int, default=6)
    parser.add_argument(
        "--detach-wear-default", dest="detach_wear_default", action="store_true",
        default=False,
        help=(
            "default 'detach_wear' to true for every /loom draw that does "
            "not set it explicitly in the request body — draw the K futures "
            "with the current wear's hook detached (the worn code is kept "
            "for scores.cos_to_worn and stay/swerve; only the generation "
            "itself runs unbent — see resolve_draw_wear). Per-request "
            "'detach_wear' in the POST body always overrides this. DEFAULT: "
            "false — futures drawn UNDER wear."
        ),
    )
    parser.add_argument(
        "--lever-bank", type=Path, default=None,
        help="a build_levers levers.npz to make wearable via POST /wear_code "
             "(atlas landmarks and named bank groups). Optional; without it "
             "/wear_code returns 400 and everything else is unchanged.")
    parser.add_argument("--max-turns", type=int, default=12, help="history window")
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selfcheck-tokens", type=int, default=32)
    parser.add_argument(
        "--harvest-worker", default="",
        help=(
            "host:port of a running harvest worker (pleroma.harvest.worker). "
            "When set, /loom tries "
            "POSTing to it first and only falls back to the two-"
            "subprocess pipeline if the worker is unreachable or reports an "
            "error. Empty (default) = subprocess pipeline only."
        ),
    )
    parser.add_argument("--harvest-worker-connect-timeout", type=float, default=3.0)
    parser.add_argument("--harvest-worker-timeout", type=float, default=120.0)
    parser.add_argument(
        "--harvest-workers", default="",
        help=(
            "PARALLEL HARVEST (measured ~3.8x): comma-separated host:port "
            "list of TWO OR MORE running harvest workers (pleroma.harvest.worker). "
            "/loom shards "
            "the draw across them and merges, which takes a k=8 turn from ~15 s "
            # `%%`: argparse %-formats every help string, so a bare "95% of"
            # reads as the conversion spec "% o" and `--help` dies with
            # "TypeError: %o format: an integer is required, not dict".
            "toward ~5.5 s. Harvest is 88-95%% of a loom turn, so this is the whole "
            "latency story. Strictly additive and strictly degrading: a worker that "
            "dies costs its shard (re-dispatched once), then the pool falls back to "
            "--harvest-worker and then to the two-subprocess pipeline. Empty "
            "(default) = no pool. One entry is refused — use "
            "--harvest-worker for that."
        ),
    )
    parser.add_argument(
        "--restore", dest="restore", action="store_true", default=True,
        help="reload persisted sessions from <work-dir>/sessions/ at startup "
             "(DEFAULT).",
    )
    parser.add_argument(
        "--no-restore", dest="restore", action="store_false",
        help="start with no sessions in memory. Snapshots on disk are left "
             "alone; the first write to one of them moves it aside as "
             "<name>.json.orphaned-<ts> rather than clobbering it.",
    )
    parser.add_argument(
        "--persist", dest="persist", action="store_true", default=True,
        help="write a session snapshot after every mutating request (DEFAULT).",
    )
    parser.add_argument(
        "--no-persist", dest="persist", action="store_false",
        help="never write snapshots (benchmarks, throwaway runs). Restore, if "
             "on, still reads what is already there.",
    )
    parser.add_argument("--atlas-report", default=None,
                        help="atlas_report.json for GET /atlas (default: "
                             "outputs/loom/atlas/atlas_report.json if present)")
    parser.add_argument(
        "--dose-band", type=Path, default=None,
        help=(
            "a MEASURED dose-band sidecar JSON for --map (see pleroma.dose.band; "
            "GET /info.dose_band.tier reports what was resolved). Default: "
            "<map>.doseband.json next to the map, loaded IF PRESENT. An "
            "explicit path here that is missing or malformed is a hard "
            "failure at startup — a band that cannot be trusted must never "
            "be half-applied; an absent DEFAULT path is fine (no band yet) "
            "and the server serves tier='none' (UNCALIBRATED)."
        ),
    )
    parser.add_argument(
        "--dose-band-reference", type=Path, default=None,
        help=(
            "a MEASURED dose-band sidecar JSON from a DIFFERENT map, used to "
            "DERIVE a provisional band for --map by rescaling on norm_ref "
            "(pleroma.dose.band.DoseBand.rescale). "
            "Only used when --dose-band resolves to nothing. NOT "
            "auto-discovered: which map's band to treat as a reference is a "
            "judgement call, so it must be named explicitly, e.g. "
            "tests/fixtures/dose_bands/old_map_measured.json."
        ),
    )
    parser.add_argument(
        "--dose-band-residual-scale", type=Path, default=None,
        help=(
            "a residual-scale JSON (pleroma.dose.band.ResidualScale; see "
            "tests/fixtures/dose_bands/residual_scale_3b_8b.json) giving the "
            "per-site median ||h|| of the reference band's model AND of this "
            "map's model. PASS IT WHENEVER --dose-band-reference WAS MEASURED "
            "ON A DIFFERENT MODEL. It changes NO served boundary: it attaches "
            "the measured bound on the competing relative-magnitude hypothesis "
            "to /info.dose_band.derivation.cross_model, so an operator can see "
            "how far apart the two hypotheses put the band (3B->8B: 1.165x, "
            "with the served numbers on the conservative side) instead of "
            "having to take a cross-model rescale on faith. Ignored when a "
            "MEASURED band wins — that band derived nothing."
        ),
    )
    parser.add_argument(
        "--dose-policy-default", choices=list(DOSE_POLICIES),
        default=DEFAULT_DOSE_POLICY,
        help=(
            "the server-wide default `dose_policy` for POST /wear and for "
            "/loom's auto-select. 'flat' (the DEFAULT) is the incumbent ruler "
            "— every candidate injected at the map's own norm_ref, the "
            "ruler every banked result used. 'predicted' divides "
            "by the FAN's mean raw norm instead of the candidate's own, so "
            "the map's within-fan magnitude prediction survives to the model "
            "(v1a predicts within-fan magnitude at r=.757); the fan's "
            "average dose is unchanged so alpha keeps its meaning, and the "
            "per-site scale is clamped to "
            f"[{DOSE_SCALE_MIN}, {DOSE_SCALE_MAX}] with every clamp reported. "
            "UNVALIDATED behaviourally. A request body's own 'dose_policy' "
            "always overrides this. /wear_code is NEVER predicted-dosed (no "
            "fan, no mean) — it stays flat regardless of this flag and "
            "refuses an explicit request for 'predicted'."
        ),
    )
    parser.add_argument(
        "--wear-lever-default", choices=list(LEVER_KINDS),
        default=DEFAULT_LEVER_KIND,
        help=(
            "the server-wide default `lever_kind` for POST /wear and for "
            "/loom's auto-select. 'absolute' (the DEFAULT) wears W applied to "
            "the candidate's own input row — the lever every banked result "
            "wore. 'contrast' wears W applied to the "
            "candidate's row minus the mean of the fan's OTHER valid rows — "
            "the one-vs-rest object v1a was fit on and /probe steers with "
            "(norm-matched to the same ruler, mu_y not added back). "
            "UNVALIDATED behaviourally on a fan wear. A request body's own "
            "'lever_kind' always overrides this."
        ),
    )
    parser.add_argument(
        "--allow-lan", action="store_true",
        help="bind beyond loopback (e.g. 0.0.0.0). The API has NO AUTH unless "
        "--api-token / PLEROMA_API_TOKEN is set: only do this on a network you "
        "trust end to end (an authed VPN with a handful of trusted users is the "
        "intended case); loopback-only remains the DEFAULT.",
    )
    # ── the versioned HTTP API (docs/API.md) ─────────────────────────────────
    parser.add_argument(
        "--cors-origin", dest="cors_origin", action="append", default=None,
        metavar="ORIGIN",
        help="allow a browser app at ORIGIN (scheme://host[:port], e.g. "
             "http://localhost:5173) to call /api/v1/* cross-origin. "
             "Repeatable. DEFAULT: none — same-origin only. "
             "'*' is refused. The un-prefixed legacy routes never answer "
             "cross-origin.",
    )
    parser.add_argument(
        "--api-token", default=None, metavar="TOKEN",
        help="require 'Authorization: Bearer TOKEN' on /api/v1/* (and, once "
             "--cors-origin or --allow-lan is set, on the legacy routes too — "
             "docs/API.md). Prefer the environment variable PLEROMA_API_TOKEN: "
             "a flag is visible to every user in `ps`. DEFAULT: no token.",
    )
    parser.add_argument(
        "--ui-dir", type=Path, default=None,
        help="serve this directory's index.html at / and /ui (and its static "
             "files beneath it) instead of the legacy UI page — e.g. "
             "a built ui/dist. DEFAULT: pleroma/serve/static/legacy_ui.html.",
    )
    return parser


def resolve_sampling_defaults(args: argparse.Namespace) -> None:
    """Fill ``--temperature``/``--top-p`` from the prompt mode, IN PLACE, and
    warn about modelc-only flags set outside modelc mode."""
    # ── sampling defaults, resolved against the prompt mode ──────────────────
    _modelc = args.prompt_mode == "modelc"
    if args.temperature is None:
        args.temperature = MODELC_TEMPERATURE if _modelc else 1.0
    if args.top_p is None:
        args.top_p = MODELC_TOP_P if _modelc else 0.95
    if not _modelc and args.modelc_header != MODELC_DEFAULT_HEADER:
        logger.warning("--modelc-header is set but --prompt-mode is %r — it has "
                       "no effect outside modelc mode", args.prompt_mode)
    if not _modelc and not args.modelc_stops:
        logger.warning("--no-modelc-stops is set but --prompt-mode is %r — "
                       "stop strings are never used outside modelc mode anyway",
                       args.prompt_mode)
    if _modelc and args.modelc_header not in MODELC_HEADERS:
        logger.info("--modelc-header %r is not a named header — taking it as a "
                    "literal custom header line", args.modelc_header)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (``sys.argv[1:]`` when None — ``loom_modelc`` rewrites
    ``sys.argv`` and relies on that) and resolve the sampling defaults."""
    args = build_parser().parse_args(argv)
    resolve_sampling_defaults(args)
    return args
