"""A stand-in loom server for rendering the legacy loom page without a GPU.

WHY THIS EXISTS. `pleroma/serve/static/legacy_ui.html` is thousands of lines
of copy and arithmetic about what the map means. Without a stand-in server the
only way to see it is to point a browser at a real server on a GPU host, so a
page that still says "v0" and "the map is rank-8" over a v1a server fails
nothing.

This module serves the REAL html file alongside canned `/info`, `/atlas`,
`/state` and `/loom` payloads shaped exactly like `loom_serve`'s, so a
headless browser can render the page and the DOM can be asserted on. The
default fixtures reproduce a live v1a configuration on purpose, including its
trap: a v1a rank-64 map served next to a stale rank-8 atlas
that `GET /atlas` falls back to when no `--atlas-report` is passed.

It never touches the real server. `serve()` binds an ephemeral port on
127.0.0.1 and the caller shuts it down.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

UI_PATH = Path(__file__).resolve().parents[4] / "pleroma" / "serve" / "static" / "legacy_ui.html"

# ── the live v1a configuration, as /info reports it ────────────────────────
V1A_INFO: dict[str, Any] = {
    "map": "/x/v1a_spot/loom_map_v1a_r64.npz",
    "map_meta": {
        "rank": 64,
        "lam": 10000.0,
        "n_fans": 1067,
        "prereg_token": "v1a-fit-001",
        "stage": "v1a_export - the deployment refit of the registered v1a fit",
    },
    "sites": [8, 15, 19, 22],
    "branches": ["loom", "base"],
    "default_k": 8,
    "future_tokens": 96,
    "detach_wear_default": False,
    "n_sessions": 3,
    "harvest_worker": "127.0.0.1:8768",
    "restored_sessions": 0,
    "persistence": {"work_dir": "outputs/loom/sessions_v1a"},
    "auto_policies": [
        {"key": "distinct", "needs_wear": False,
         "description": "furthest from the rest of the fan in signature space",
         "status": "measured, weak - 33% of ORACLE picking skill (docs/FINDINGS.md §8)"},
        {"key": "loudest", "needs_wear": False,
         "description": "largest raw map output norm",
         "status": "43% of ORACLE on v1a (docs/FINDINGS.md §8). estimated, uncertified"},
        {"key": "stay", "needs_wear": True, "description": "closest to the worn code",
         "status": "a steering intent, not a quality bar"},
        {"key": "swerve", "needs_wear": True, "description": "furthest from the worn code",
         "status": "a steering intent, not a quality bar"},
        # The PROBED policy: the one auto policy that costs the operator time,
        # advertised here as the live server advertises it.
        {"key": "gauge", "needs_wear": False, "needs_probe": True,
         "description": "the probe gauge: wear each candidate's code, generate a "
                        "short reply, measure the normalized rank of the target future "
                        "among the fan by cosine distance in z space",
         "status": "THE POLICY THIS PROJECT ADOPTED BY MEASUREMENT (docs/FINDINGS.md §8): 71% of "
                   "ORACLE picking skill. But 63 bought 43% of that for free with "
                   "loudest, so the real margin is ~27 points. 53 REFUSED this same "
                   "gauge as an INSTRUMENT. Estimated, uncertified.",
         "cost": "NOT FREE - it generates and harvests k more spans",
         "opt_in": 'POST /probe, or pass "probe": true on /loom'},
    ],
    "auto_policies_unavailable": [],
    # ── the probe, as /info advertises it ──────────────────────────────────
    # `cost_estimate` is the server's own price for ITS default k/reps, with the
    # per-gen and per-span basis published so a client can re-price the settings
    # on screen. The UI must use this and never a hardcoded latency.
    "probe": {
        "available": True,
        "endpoint": 'POST /probe (or "probe": true on /loom)',
        "policy_key": "gauge",
        "base_modes": ["fan", "fresh"],
        "default_base": "fan",
        "default_reps": 2,
        "cost_estimate": {
            "k": 8, "reps": 2, "base": "fan", "n_base": 0,
            "n_spans_harvested": 16,
            "generate_s": 7.9, "harvest_s": 21.3, "total_s": 29.2,
            "basis": {"per_gen_s": 0.99, "per_span_harvest_s": 1.33,
                      "measured": "B200, serial harvest worker"},
            "note": "ON TOP of the /loom draw itself. The harvest term dominates.",
        },
        "definition": "the probe gauge = normalized rank of the target future among the "
                      "same-pass futures by cosine distance in z space; cell = mean "
                      "base nr minus mean steered nr.",
        "label": "estimated, uncertified - 53 refused this gauge as an INSTRUMENT "
                 "and 54 adopted it only as a pick POLICY.",
    },
    "loudness_ref": 2.682,
    "dose_band": {
        "tier": "derived",
        "source": "rescaled from w5_r32_measured.json on the norm_ref ratio",
        "alpha_max": 1.016,
        "zones": [
            {"lo": 0.0, "hi": 0.127, "hi_inclusive": True, "key": "subliminal",
             "label": "subliminal", "hint": "below anything anyone has detected"},
            {"lo": 0.127, "hi": 0.508, "hi_inclusive": False, "key": "threshold",
             "label": "threshold", "hint": "detectable if you are looking"},
            {"lo": 0.508, "hi": 0.719, "hi_inclusive": False, "key": "audible",
             "label": "audible", "hint": "conversationally audible"},
            {"lo": 0.719, "hi": 1.016, "hi_inclusive": True, "key": "overdriven",
             "label": "overdriven", "hint": "word-salad risk"},
        ],
    },
    "ruler": {
        "norm_ref_in_force": [1.022, 2.168, 3.161, 4.379],
        "norm_ref_in_force_mean": 2.6825,
        "which": "norm_ref = WIDE MAP's ruler (decision 1)",
        "alternatives": {
            "norm_ref_v1a_own": {"norm_ref": [0.461, 0.98, 1.423, 1.989],
                                 "mean": 1.21325, "in_force": False},
        },
        "note": "alpha injects alpha x norm_ref_in_force[s] per site.",
    },
    # ── the dose policies, in the shape the live server publishes ──────────
    # ★ NOTE THE TRAP, and it is the live server's: `flat` carries
    # `"default": true` (a static statement about what the module ships with)
    # while the RUNNING default is `predicted`, set by --dose-policy-default.
    # A UI that believed the per-entry flag would describe a server it is not
    # talking to. `dose_policy_default` is the only
    # authority, and this fixture reproduces the disagreement on purpose.
    "dose_policies": [
        {"key": "flat", "default": True, "needs_fan": False,
         "description": "every candidate injected at the map's own norm_ref - "
                        "identical per-site magnitude for all of them.",
         "status": "the incumbent. Every dose band, ladder and published alpha in "
                   "this project was measured under it."},
        {"key": "predicted", "default": False, "needs_fan": True,
         "description": "divide by the FAN's mean raw norm instead of the "
                        "candidate's own, so a candidate the map predicts will "
                        "move further is injected proportionally harder. Per-site "
                        "scale clamped to [0.5, 2.0]; every clamp is reported.",
         "status": "UNVALIDATED - estimated, uncertified. No behavioural result "
                   "exists for this policy. It also breaks the 1:1 between alpha "
                   "and the dose band.",
         "unavailable_for": "/wear_code - a bank group, a random direction or a "
                            "bare code has no fan, so there is no mean to divide "
                            "by. Asking for it there is refused, never silently "
                            "flattened."},
    ],
    # the server runs with --dose-policy-default predicted
    "dose_policy_default": "predicted",
    "dose_scale_clamp": [0.5, 2.0],
}

# ── lever_kind ────────────────────────────────────────────────────────────
# What a server built with the lever_kind toggle ADDS to /info, in
# loom_serve.build_info_payload's exact shape. Deliberately NOT merged into
# V1A_INFO: that dict is the live server as captured, and the captured server
# has no lever_kind toggle. A test opts in with
# `Fixture(info={**V1A_INFO, **LEVER_KIND_INFO_FIELDS})`.
LEVER_KIND_INFO_FIELDS: dict[str, Any] = {
    "lever_kinds": [
        {"key": "absolute", "needs_fan": False, "default": True,
         "description": "W applied to the candidate's OWN input row.",
         "status": "the default: the ruler alone, applied unchanged."},
        {"key": "contrast", "needs_fan": True, "default": False,
         "description": "W applied to x_i minus the mean of the OTHER valid "
                        "members of the draw.",
         "status": "UNVALIDATED behaviourally on a fan wear.",
         "unavailable_for": "a draw with fewer than 2 harvested candidates."},
    ],
    "lever_kind_default": "absolute",
}

# ── a real /probe receipt, captured from a live 8B server ─────────────────
# k=4 reps=2 base=fan. It is the
# UNRESOLVED case on purpose: estimated_signal_sd 0.0, reps_needed_to_resolve
# null, absolute_comparable false. That is what the fans this server can
# currently draw actually return, and it is the case the UI must not launder.
LIVE_PROBE_RECEIPT: dict[str, Any] = {
    "loom_id": "stub-loom-1", "k": 4, "reps": 2, "alpha": 0.35,
    "horizon": 64, "base": "fan", "n_base": 0, "n_spans_harvested": 8,
    "candidates": [
        {"index": 0, "gauge": 0.1667, "gauge_base_nr": 0.6667,
         "gauge_steered_nr": 0.5, "gauge_reps": 2, "note": None},
        {"index": 1, "gauge": 0.3889, "gauge_base_nr": 0.5556,
         "gauge_steered_nr": 0.1667, "gauge_reps": 2, "note": None},
        {"index": 2, "gauge": -0.0556, "gauge_base_nr": 0.7778,
         "gauge_steered_nr": 0.8333, "gauge_reps": 2, "note": None},
        {"index": 3, "gauge": -0.1667, "gauge_base_nr": 0.6667,
         "gauge_steered_nr": 0.8333, "gauge_reps": 2, "note": None},
    ],
    "ranking": [1, 0, 2, 3],
    "pick": 1,
    "dead_rows": [],
    "session_worn_during_draw": False,
    "clean_regime": True,
    "resolution": {
        "resolved": False,
        "reps": 2,
        "n_candidates_used": 4,
        "n_candidates_excluded_for_reps": 0,
        "within_candidate_sd_nr": 0.2357,
        "between_candidate_sd_cell": 0.2464,
        "estimated_signal_sd": 0.0,
        "noise_share_of_cell_variance_at_this_reps": 0.458,
        "reps_needed_to_resolve": None,
        "why": "the between-candidate spread is entirely explained by measurement "
               "noise - these candidates are not distinguishable by the gauge at "
               "ANY rep count this loom can afford",
    },
    "timing_s": {"generate": 4.5, "harvest": 3.4, "total": 7.9},
    "harvest_via": "pool",
    "probe_dir": "/x/.../probe_1ff1605a90-000_1790060409",
    "absolute_comparable": False,
    "label": "the probe gauge, ESTIMATED AND UNCERTIFIED (docs/FINDINGS.md §8). This is a PICK policy "
             "(54, 71% of ORACLE), not a verdict: 53 refused this same gauge as "
             "an instrument. fable remains the instrument; you still choose. "
             "base='fan' (leave-one-out) makes these RANKING statistics only - "
             "the constant self-distance offset cancels in the order but NOT in "
             "the value, so do not read them as dnr.",
}

# the same probe with a RESOLVED verdict, for the other arm of the render tests
RESOLVED_RESOLUTION: dict[str, Any] = {
    "resolved": True,
    "reps": 2,
    "n_candidates_used": 4,
    "n_candidates_excluded_for_reps": 0,
    "within_candidate_sd_nr": 0.299,
    "between_candidate_sd_cell": 0.361,
    "estimated_signal_sd": 0.2003,
    "noise_share_of_cell_variance_at_this_reps": 0.343,
    "reps_needed_to_resolve": 2,
    "why": "the ordering is resolved at this rep count",
}

# the refusal pick_auto raises for auto.policy='gauge' on an unprobed fan. It
# is a 400 with NO futures in the body — the draw is lost, which is exactly why
# the UI needs a probe button that does not require a redraw.
GAUGE_REFUSAL = (
    "auto policy 'gauge' needs probe scores and this fan has none. Run the "
    'probe first: POST /probe, or pass "probe": true on this /loom call. '
    "(the gauge is PROBE-derived - docs/FINDINGS.md §8 - and cannot be read off the draw.)"
)

# ── the atlas GET /atlas actually falls back to on the node: w4-era, rank 8 ───
STALE_RANK8_ATLAS: dict[str, Any] = {
    "rank": 8,
    "n_groups": 138,
    "in_space_energy": 0.41,
    "singular_values": [9.1, 7.4, 6.2, 5.5, 4.9, 4.1, 3.6, 3.0],
    "axes": [
        {"axis": i, "sv": 9.1 - i, "coord_std": 0.4,
         "quantiles": [-0.62, -0.21, 0.0, 0.2, 0.61],
         "class_eta2": 0.02, "twin_consistency_r": 0.31,
         "high_extreme": ["cc1|orig", "if5|orig"], "low_extreme": ["dd2|repl"],
         "class_means": {"convergent_control": 0.1, "if_then": -0.08}}
        for i in range(8)
    ],
    "groups": [
        {"id": "cc1|orig", "class": "convergent_control",
         "coords": [0.3271, -0.0607, 0.2039, 0.0301, -0.0068, -0.3762, -0.0042, -0.2853]},
        {"id": "if5|orig", "class": "if_then",
         "coords": [-0.21, 0.42, -0.11, 0.05, 0.19, 0.02, -0.3, 0.14]},
    ],
}


def _code(seed: int, rank: int = 64) -> list[float]:
    """A deterministic pseudo-code of the map's own width."""
    out: list[float] = []
    x = seed * 7919 + 13
    for _ in range(rank):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(round((x / 2147483648.0 - 0.5) * 1.4, 4))
    return out


class Fixture(BaseModel):
    """What this stub serves. Every field mirrors a real endpoint's shape."""

    model_config = {"extra": "forbid"}

    info: dict[str, Any] = Field(default_factory=lambda: dict(V1A_INFO))
    atlas: dict[str, Any] | None = Field(
        default_factory=lambda: dict(STALE_RANK8_ATLAS),
        description="None serves a 404, which is the no-atlas case.",
    )
    k: int = 8
    rank: int = 64
    # ── the rig's pool probe ───────────────────────────────────────────────
    pool: dict[str, Any] | None = Field(
        default=None,
        description="GET /pool's body. None serves a 404 — an older server with "
                    "no pool probe.",
    )
    # ── probe fixtures ─────────────────────────────────────────────────────
    probe_resolution: dict[str, Any] | None = Field(
        default=None,
        description="Override LIVE_PROBE_RECEIPT's resolution block. None keeps "
                    "the live UNRESOLVED one, which is the honest default.",
    )
    probe_absolute: bool = Field(
        default=False,
        description="absolute_comparable. False (base='fan') is the server's "
                    "default and means the cells are ranking statistics only.",
    )
    probe_fails: bool = Field(
        default=False,
        description="POST /probe 500s and a /loom rider degrades to "
                    "{error, note} inside a 200 — loom_serve's real behaviour.",
    )
    probe_clean_regime: bool = True
    state_probe: dict[str, Any] | None = Field(
        default=None,
        description="What GET /state reports as its `probe`. None with "
                    "state_gauge_scores=True is the restored-snapshot case.",
    )
    state_gauge_scores: bool = Field(
        default=False,
        description="GET /state serves futures carrying `gauge` with no receipt, "
                    "which is what a persistence restore actually looks like.",
    )
    dose_policy_refuses: bool = Field(
        default=False,
        description="POST /wear 400s on an explicit dose_policy, the way the "
                    "server does when there is no fan mean to divide by.",
    )

    # ── restored-session fixtures ─────────────────────────────────────────
    histories: dict[str, list[dict[str, str]]] | None = Field(
        default=None,
        description="What GET /state reports as `histories`. The real server "
                    "sends {role, content} and NOTHING about per-turn wear, "
                    "which is the whole reason a restored turn must read "
                    "'wear unknown' in the what-changed panel.",
    )
    worn: dict[str, Any] | None = Field(
        default=None,
        description="What GET /state reports as `worn`. None is the "
                    "nothing-worn case, which the page must stay correct in.",
    )

    unscored_candidate: int | None = Field(
        default=None,
        description="A candidate the probe could not score: its `gauge` comes "
                    "back as an explicit null. Number(null) === 0, so this is "
                    "the fixture for the bug that rendered it as a measured "
                    "'+0.000'.",
    )

    def probe_receipt(self, *, reps: int = 2, base: str = "fan") -> dict[str, Any]:
        """The receipt POST /probe answers with, shaped exactly like the real one.

        The cells are the captured live ones, cycled out to this fixture's k so
        every candidate in the fan is scored (except `unscored_candidate`, which
        is deliberately null). `ranking` is descending by cell, the way
        loom_probe.rank_by_gauge orders it, and `pick` is its head.
        """
        r = json.loads(json.dumps(LIVE_PROBE_RECEIPT))
        live = [c["gauge"] for c in LIVE_PROBE_RECEIPT["candidates"]]
        cands: list[dict[str, Any]] = []
        for i in range(self.k):
            if self.unscored_candidate is not None and i == self.unscored_candidate:
                cands.append({"index": i, "gauge": None, "gauge_base_nr": None,
                              "gauge_steered_nr": None, "gauge_reps": 0,
                              "note": "no usable probe reply for this candidate"})
                continue
            g = round(live[i % len(live)] - 0.011 * (i // len(live)), 4)
            cands.append({"index": i, "gauge": g,
                          "gauge_base_nr": r["candidates"][i % len(live)]["gauge_base_nr"],
                          "gauge_steered_nr": r["candidates"][i % len(live)]["gauge_steered_nr"],
                          "gauge_reps": reps, "note": None})
        scored = [c for c in cands if c["gauge"] is not None]
        order = sorted(scored, key=lambda c: (-float(c["gauge"]), int(c["index"])))
        r["k"] = self.k
        r["reps"] = reps
        r["base"] = base
        r["absolute_comparable"] = bool(self.probe_absolute) or base == "fresh"
        r["clean_regime"] = bool(self.probe_clean_regime)
        r["session_worn_during_draw"] = not bool(self.probe_clean_regime)
        r["candidates"] = cands
        r["ranking"] = [int(c["index"]) for c in order]
        r["pick"] = int(order[0]["index"]) if order else None
        r["n_spans_harvested"] = len(scored) * reps
        if self.unscored_candidate is not None:
            r["dead_rows"] = [f"gen_{self.unscored_candidate:03d} "
                              f"(steered {self.unscored_candidate}): "
                              "FileNotFoundError: signatures/gen missing"]
        if self.probe_resolution is not None:
            r["resolution"] = dict(self.probe_resolution)
        r["resolution"]["reps"] = reps
        r["resolution"]["n_candidates_used"] = len(scored)
        return r

    def probed_futures(self, receipt: dict[str, Any]) -> list[dict[str, Any]]:
        """`futures` with the receipt's gauge cells written in, as /probe returns.

        A candidate the probe could not score gets an explicit `None` — the exact
        value a loose `Number(x)` guard renders as a measured "+0.000".
        """
        cells = {int(c["index"]): c["gauge"] for c in receipt["candidates"]}
        rows = self.futures()
        for row in rows:
            row["scores"]["gauge"] = cells.get(int(row["index"]), None)
        return rows

    def futures(self) -> list[dict[str, Any]]:
        # loudnesses chosen to span a measured loudness range (0.87-1.42x)
        loud = [1.447, 1.465, 2.336, 1.388, 1.603, 1.396, 1.695, 1.435][: self.k]
        mean = sum(loud) / len(loud)
        rows: list[dict[str, Any]] = []
        for i, lv in enumerate(loud):
            scores: dict[str, Any] = {
                "distinct": round(0.36 + 0.02 * i, 4),
                "loudness": lv,
                "predicted_dose_scale": round(lv / mean, 3),
                # lever_kind: the CONTRAST lever's own scale —
                # deliberately far from the absolute one, so a readout using
                # the wrong key is visible.
                "predicted_dose_scale_contrast": round(0.62 + 0.11 * i, 3),
            }
            rows.append({
                "index": i, "n_tokens": 96, "harvested": True, "note": None,
                "text": f"future {i}: a continuation that goes its own way.",
                "scores": scores, "code": _code(i, self.rank),
            })
        return rows

    def spread(self) -> dict[str, Any]:
        n = self.k
        camp_b = max(1, n // 3)
        return {
            "mean_pairwise_distance": 0.4414,
            "n_scored": n,
            "camps": [n - camp_b, camp_b],
            "separation": 0.612,
            "camp_labels": ("per-draw only - 2-means labels are not identities "
                            "across looms; camp 0 this draw is not camp 0 next draw"),
            "loudness_ref": 2.682,
        }


def make_handler(fx: Fixture, extra_head: str,
                 log: list[str] | None = None,
                 bodies: list[tuple[str, dict[str, Any]]] | None = None,
                 ) -> type[BaseHTTPRequestHandler]:
    """`log`, when given, records every request path the page actually makes.

    "The page never fetches /atlas" is only assertable if something counts. A
    test
    that merely checks the bank cells are absent would also pass against a
    page that fetches the atlas and then throws it away — which leaves the
    footgun loaded.
    """
    ui = UI_PATH.read_text(encoding="utf-8")
    if extra_head:
        # the harness script is APPENDED to the served bytes; the file on disk
        # is never modified, so what is asserted on is the real page plus a
        # driver that only clicks things a person could click.
        ui = ui.replace("</body>", extra_head + "\n</body>")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:  # keep pytest output clean
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if log is not None:
                log.append("GET " + path)
            if path in ("/", "/index.html"):
                self._send(200, ui.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/info":
                self._json(200, fx.info)
            elif path == "/atlas":
                if fx.atlas is None:
                    self._json(404, {"error": "no atlas built yet"})
                else:
                    self._json(200, fx.atlas)
            elif path == "/state":
                body: dict[str, Any] = {
                    "session": "luxia",
                    "histories": fx.histories or {"loom": [], "base": []},
                    "worn": fx.worn, "loom_id": None, "futures": [],
                    "n_looms": 0, "loom_in_progress": None,
                }
                # The restored-snapshot case. Gauge scores
                # persist; the receipt does NOT, and the server says so on
                # `probe_note`. The UI must not present those ranks as measured.
                if fx.state_gauge_scores:
                    body["loom_id"] = "stub-loom-1"
                    body["futures"] = fx.probed_futures(fx.probe_receipt())
                    body["probe"] = fx.state_probe
                    body["probe_note"] = (
                        "these gauge scores were restored from a snapshot; the probe "
                        "receipt (including its `resolution` - whether the ordering "
                        "was distinguishable from noise at all) is NOT persisted. "
                        "Re-probe before trusting them.")
                    body["n_looms"] = 1
                elif fx.state_probe is not None:
                    body["probe"] = fx.state_probe
                self._json(200, body)
            elif path == "/sessions":
                self._json(200, {"sessions": []})
            elif path == "/pool":
                if fx.pool is None:
                    self._json(404, {"error": "unknown path"})
                else:
                    self._json(200, fx.pool)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                blob: dict[str, Any] = json.loads(raw or b"{}")
            except ValueError:
                blob = {}
            if not isinstance(blob, dict):
                blob = {}
            if bodies is not None:  # what the page SENT
                bodies.append((path, json.loads(json.dumps(blob))))
            if path == "/loom":
                self._loom(blob)
            elif path == "/probe":
                self._probe(blob)
            elif path == "/wear":
                self._wear(blob)
            elif path == "/unwear":
                self._json(200, {"session": "luxia", "worn": None})
            else:
                self._json(404, {"error": "not found"})

        # ── /loom, now with the probe rider and the gauge refusal ───────────
        def _loom(self, blob: dict[str, Any]) -> None:
            auto = blob.get("auto") if isinstance(blob.get("auto"), dict) else None
            probe_cfg = blob.get("probe")
            wants_probe = probe_cfg is True or isinstance(probe_cfg, dict)
            # pick_auto's real refusal: a 400 with the server's instructions and
            # NO futures in the body. Never a silent substitution.
            if auto and str(auto.get("policy") or "") == "gauge" and not wants_probe:
                self._json(400, {"error": GAUGE_REFUSAL})
                return
            out: dict[str, Any] = {
                "loom_id": "stub-loom-1", "k": fx.k, "horizon": 96,
                "futures": fx.futures(), "spread": fx.spread(),
                "timing_s": {"generate": 1.4, "harvest": 7.8},
                "drawn_under_wear": None, "auto": None, "probe": None,
            }
            if wants_probe:
                cfg = probe_cfg if isinstance(probe_cfg, dict) else {}
                if fx.probe_fails:
                    # loom_serve degrades a failed probe INSIDE a 200 rather than
                    # throwing away a paid-for draw.
                    out["probe"] = {
                        "error": "ValueError: this fan has a candidate with no z vector",
                        "note": "the PROBE failed; the draw above is unaffected and "
                                "every free policy still works. auto.policy='gauge' "
                                "will refuse until a probe succeeds - POST /probe to "
                                "retry without redrawing.",
                    }
                else:
                    receipt = fx.probe_receipt(
                        reps=int(cfg.get("reps") or 2),
                        base=str(cfg.get("base") or "fan"))
                    out["probe"] = receipt
                    out["futures"] = fx.probed_futures(receipt)
            self._json(200, out)

        def _probe(self, blob: dict[str, Any]) -> None:
            if not str(blob.get("text") or "").strip():
                self._json(400, {"error": "/probe needs 'text' - the same "
                                          "contemplated user turn the draw was "
                                          "made against"})
                return
            if fx.probe_fails:
                self._json(400, {"error": "ValueError: this fan has a candidate "
                                          "with no z vector (an unharvested "
                                          "future) - the probe ranks against the "
                                          "WHOLE fan. /loom again."})
                return
            receipt = fx.probe_receipt(
                reps=int(blob.get("reps") or 2),
                base=str(blob.get("base") or "fan"))
            self._json(200, {"session": "luxia", "probe": receipt,
                             "futures": fx.probed_futures(receipt),
                             "worn": None})

        def _wear(self, blob: dict[str, Any]) -> None:
            pol = blob.get("dose_policy")
            server_default = str(fx.info.get("dose_policy_default") or "flat")
            effective = str(pol) if isinstance(pol, str) and pol else server_default
            if fx.dose_policy_refuses and isinstance(pol, str) and pol:
                # the real refusal text, so the UI's surfacing can be asserted
                self._json(400, {"error": (
                    "dose_policy='predicted' needs the fan mean raw norms from "
                    "the draw and this session has none, so this candidate "
                    "cannot be predicted-dosed. Use dose_policy='flat', or "
                    "/loom again first.")})
                return
            idx = int(blob.get("index") or 0)
            alpha = float(blob.get("alpha") or 0.5)
            rows = fx.futures()
            row = next((r for r in rows if int(r["index"]) == idx), rows[0])
            # lever_kind: echo the kind the way the real server does;
            # null against an /info that never published the toggle.
            lk = blob.get("lever_kind") or fx.info.get("lever_kind_default")
            base_scale = float(row["scores"]["predicted_dose_scale_contrast"]
                               if lk == "contrast"
                               else row["scores"]["predicted_dose_scale"])
            if effective == "flat":
                scale = [1.0] * 4
                note = "flat: the incumbent ruler - every candidate at norm_ref."
                eff_note = ("dose_policy=flat: effective alpha IS alpha; read the "
                            "dose band at alpha.")
            else:
                # each site clamped independently, and deliberately SPREAD: the
                # mean can read nominal while one site runs hotter, which is the
                # bug the zone read at alphaWorst exists to catch.
                scale = [round(min(2.0, max(0.5, base_scale * f)), 6)
                         for f in (1.0, 1.04, 1.08, 1.12)]
                note = ("predicted: per-site norm_ref x (this candidate's raw norm "
                        "/ the fan's mean raw norm). UNVALIDATED - estimated, "
                        "uncertified.")
                eff_note = ("dose_policy=predicted: read the dose band at "
                            "alpha_effective_per_site, not at alpha.")
            per = [round(alpha * s, 6) for s in scale]
            mean_scale = round(sum(scale) / len(scale), 6)
            dose = {
                "policy": effective, "scale": scale, "scale_raw": scale,
                "scale_mean": mean_scale,
                "fan_mean_raw_norms": None if effective == "flat" else [1.0] * 4,
                "candidate_raw_norms": None if effective == "flat" else scale,
                "clamped": [], "clamp_range": [0.5, 2.0], "note": note,
            }
            self._json(200, {
                "session": "luxia",
                "worn": {"index": idx, "alpha": alpha, "loom_id": "stub-loom-1",
                         "code": row["code"],
                         "per_site_norms_at_alpha1": [0.81, 1.8, 2.6, 3.62],
                         "dose": dose,
                         **({"lever_kind": lk} if lk else {})},
                "sites": fx.info.get("sites"),
                "dose": dose,
                "effective_alpha": {
                    "alpha": alpha,
                    "alpha_effective_mean": round(alpha * mean_scale, 6),
                    "alpha_effective_per_site": per,
                    "alpha_effective_range": [min(per), max(per)],
                    "note": eff_note,
                },
            })

    return Handler


def serve(fx: Fixture | None = None, extra_head: str = "") -> tuple[ThreadingHTTPServer, str]:
    """Start the stub on an ephemeral loopback port. Returns (server, base_url).

    The returned server carries `.request_log`, a list of every request path
    the page made — so a test can assert on what was NOT fetched.
    """
    fx = fx or Fixture()
    log: list[str] = []
    bodies: list[tuple[str, dict[str, Any]]] = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              make_handler(fx, extra_head, log, bodies))
    srv.request_log = log  # type: ignore[attr-defined]
    srv.post_bodies = bodies  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"
