"""POST /probe — the gauge probe on the session's current fan (and the probe
/loom runs when a request opts in).

``run_probe``/``do_probe`` read their inputs from the ``ServeContext``
(``self.*``) and the model runtime (``rt.*``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.probe import orchestrator as loom_probe
from pleroma.format.chat import build_messages, trim_history
from pleroma.serve.draws import attach_for_draw, loom_turn_seed, prefix_fingerprint
from pleroma.serve.errors import ApiError, ErrorCode
from pleroma.serve.models import ProbeSpec
from pleroma.serve.routes._base import RouteBase
from pleroma.serve.session import LoomSession

logger = logging.getLogger("loom_serve")


class ProbeRoutes(RouteBase):

    def run_probe(self, session: str, sess: LoomSession, ids: Any,
                  cfg: Mapping[str, Any]) -> dict[str, Any]:
        """THE PROBE. Measure the gauge on the session's current fan.

        For each candidate i: wear i's CONTRASTIVE code (code_i − mean of the
        rest — the construction the gauge's calibration scored), generate `reps`
        short continuations of the SAME contemplated prefix, harvest their v3
        signatures through the SAME frozen path the draw used, and measure
        where each reply landed among the fan by cosine distance in z space.

        ★ Token ids go to the harvest directly — no text round-trip. A
        decode/re-encode round-trip drifts the code by a mean Euclidean 0.549,
        so skipping it makes a live probe's regime CLEANER than the
        calibration's, which went through text.

        ★ ONE CONFOUND, NAMED RATHER THAN HIDDEN. The steered arm attaches
        the candidate's contrastive lever and NOTHING ELSE — the session's
        own wear is not re-attached. But the futures it is scored against
        were drawn under that wear (unless the draw set detach_wear), and
        under base="fan" those same futures ARE the base arm. So on a session
        that is already wearing something, the cell conflates "the contrastive
        code was added" with "the session's wear was removed", and credits
        all of it to the former. The probe is only clean on an UNWORN session
        or a detached draw. It is not silently corrected here — correcting it
        would be a different estimator than the one that was calibrated — so
        the receipt reports the regime it ran in (`session_worn_during_draw`)
        and the log warns.

        Returns the probe receipt. Side effect, deliberate and narrow: the
        gauge score is written into `scores[i]["gauge"]` on the session's
        candidates/futures so `auto.policy: "gauge"`, /state and the UI all
        read one number computed once.
        """
        rt = self.runtime
        if not sess.candidates:
            raise ApiError(ErrorCode.NO_CANDIDATES, "no candidates — /loom first")
        cands = sorted(sess.candidates, key=lambda c: int(c["index"]))
        codes = [c.get("code") for c in cands]
        if any(c is None for c in codes):
            raise ValueError(
                "this fan has an unharvested candidate with no code — the "
                "probe needs the whole fan to build a contrastive code. "
                "/loom again.")
        spec = ProbeSpec.from_mapping(cfg)
        alpha, reps, horizon = spec.alpha, spec.reps, spec.horizon
        base_mode, n_base = str(spec.base), spec.n_base
        if base_mode not in loom_probe.BASE_MODES:
            raise ValueError(f"probe.base must be one of {loom_probe.BASE_MODES}")
        if reps < 1 or horizon < 8 or n_base < 1:
            raise ValueError("probe needs reps>=1, horizon>=8, n_base>=1")

        k = len(cands)
        plen = int(ids.shape[1])
        # Was the fan drawn under a wear the probe replies will not carry?
        # `drawn_under_wear` on the loom record is the truth of what was
        # attached for THAT draw (never merely what is worn now).
        rec = next((r for r in reversed(sess.looms)
                    if r.loom_id == sess.loom_id), None)
        worn_at_draw = bool(
            sess.worn_vectors() is not None
            and not (rec.detach_wear if rec is not None else False))
        if worn_at_draw:
            logger.warning(
                "PROBE session=%s loom=%s: the fan was drawn UNDER WEAR but "
                "the probe replies carry only each candidate's contrastive "
                "code — the gauge cell mixes 'code added' with 'wear "
                "removed'. Reported as clean_regime=false; /unwear or draw "
                "with detach_wear=true for a clean probe.", session,
                sess.loom_id)
        # Never reuse a probe dir (an overwrite silently destroys a banked
        # harvest, which is why mkdir here has no exist_ok). One wall
        # second is too coarse on its own — two probes of the same fan inside
        # a second would collide — so a counter walks until a free name.
        base_dir = Path(str(sess.last_loom_dir or ".")).parent
        stamp = int(time.time())
        rec_dir = None
        for attempt in range(1000):
            probe_dir = base_dir / (
                f"probe_{sess.loom_id or 'x'}_{stamp}"
                + (f"_{attempt}" if attempt else ""))
            try:
                rec_dir = probe_dir / "gen_records"
                rec_dir.mkdir(parents=True)  # refuses if it exists
                break
            except FileExistsError:
                rec_dir = None
        if rec_dir is None:
            raise RuntimeError(
                f"could not create a fresh probe dir under {base_dir} after "
                "1000 attempts — refusing to reuse one")

        t0 = time.time()
        sess.progress = {"stage": "probe_generate", "done": 0, "total": k}
        gid = 0
        origin: dict[int, dict[str, Any]] = {}

        def bank(rows: list[list[int]], kind: str, cand: int | None) -> None:
            nonlocal gid
            for rep, row in enumerate(rows):
                gen = rt.trim_to_eos(row, plen)
                full = row[:plen] + gen
                (rec_dir / f"gen_{gid:03d}.json").write_text(json.dumps({
                    "generation_id": gid, "prompt_id": f"probe{cand if cand is not None else 'B'}",
                    "prompt_class": "loom_probe", "prompt": "<chat context>",
                    "prompt_idx": 0, "seed_idx": gid, "seed": None,
                    "system_prompt": None, "user_prompt": "<chat context>",
                    "prompt_length": plen, "num_generated_tokens": len(gen),
                    "generated_text": rt.decode(gen),
                    "input_ids": full,
                    "sampling": {"temperature": float(self.args.temperature),
                                 "top_p": float(self.args.top_p),
                                 "max_new_tokens": int(horizon),
                                 "eos_ids": rt.eos_ids,
                                 "attn_implementation": "sdpa"},
                    "model_id": str(self.args.model_path), "preset": str(self.args.preset),
                }))
                origin[gid] = {"kind": kind, "candidate": cand, "rep": rep}
                gid += 1

        # ── the steered arm: one BATCHED generate per candidate. reps ride in
        # the batch under one hook, so reps are near-free on the generate side
        # and cost a full span each on the harvest side — which is where ~90%
        # of a loom turn's time goes.
        for c in cands:
            i = int(c["index"])
            ccode = loom_probe.contrastive_code(
                [list(map(float, x or [])) for x in codes], i)
            lever, _raw = self.loom_map.lever_from_code(ccode, differential=True)
            # The seed must MOVE between probes of the same fan, or
            # "resolution says I need more reps, probe again" redraws the
            # identical batch and accumulates nothing. loom_id + the probe
            # counter make each probe its own draw.
            seed = loom_turn_seed(self.args.seed, f"{session}|{sess.loom_id}",
                                  f"probe{sess.n_probes}", i, draw=1)
            rows = attach_for_draw(
                {"vectors": lever, "alpha": alpha}, rt.attach,
                lambda: rt.draw_batch(ids, reps, seed, horizon))
            bank(rows, "steered", i)
            sess.progress = {"stage": "probe_generate", "done": i + 1, "total": k}

        # ── the base arm, when asked for: genuinely unworn, probe-horizon,
        # one batched call. attach_for_draw(None, ...) attaches NO hook at all
        # — absence of a hook, not a zeroed alpha (see its docstring).
        if base_mode == "fresh":
            seed = loom_turn_seed(self.args.seed, f"{session}|{sess.loom_id}",
                                  f"probe_base{sess.n_probes}", 0, draw=1)
            rows = attach_for_draw(
                None, rt.attach, lambda: rt.draw_batch(ids, n_base, seed, horizon))
            bank(rows, "base", None)
        gen_s = time.time() - t0

        # gen id -> [candidate, rep], so the UI can re-walk each
        # candidate once per rep as its replies land. Base arm -> null.
        sess.progress_origin = [
            [origin[g]["candidate"], int(origin[g]["rep"])] for g in sorted(origin)]
        harvest_via, worker_detail, harvest_s = self.harvester.run(
            probe_dir, sess, stage="probe_harvest")

        # ── read the probe replies back into z space, the SAME call the draw
        # made on the futures (LoomMap.z_corpus_of). Note the probe needs NO
        # bins row: z_corpus_of reads the signature alone, so a probe span is
        # cheaper than a draw span by the bins term.
        steered_z: dict[int, list[list[float]]] = {}
        base_z: list[list[float]] = []
        dead: list[str] = []
        for g, org in sorted(origin.items()):
            try:
                with np.load(probe_dir / "signatures" / f"gen_{g:03d}.npz",
                             allow_pickle=True) as snpz:
                    sig = np.asarray(snpz["features"], dtype=np.float64)
                z = [float(x) for x in self.loom_map.z_corpus_of(sig)]
            except Exception as exc:  # noqa: BLE001 — a dead probe row is listed
                dead.append(f"gen_{g:03d} ({org['kind']} "
                            f"{org['candidate']}): {type(exc).__name__}: {exc}")
                continue
            if org["kind"] == "base":
                base_z.append(z)
            else:
                steered_z.setdefault(int(org["candidate"]), []).append(z)

        future_z = [[float(x) for x in (c.get("z") if c.get("z") is not None
                                        else [])] for c in cands]
        if any(len(z) == 0 for z in future_z):
            raise ValueError(
                "this fan has a candidate with no z vector (an unharvested "
                "future) — the probe ranks against the WHOLE fan. /loom again.")

        gauges = loom_probe.score_candidates(
            future_z, steered_z, base_mode=base_mode,
            base_z=base_z if base_mode == "fresh" else None)
        order = loom_probe.rank_by_gauge(gauges)
        by_index = {g.index: g for g in gauges}

        # one number, computed once, read by auto-pick, /state and the UI
        for c in cands:
            i = int(c["index"])
            g = by_index.get(i)
            c["gauge"] = None if g is None or g.cell is None else round(g.cell, 4)
        for f in sess.last_futures:
            g = by_index.get(int(f["index"]))
            f.setdefault("scores", {})["gauge"] = (
                None if g is None or g.cell is None else round(g.cell, 4))

        total_s = gen_s + harvest_s
        receipt = {
            "loom_id": sess.loom_id, "k": k, "reps": reps, "alpha": alpha,
            "horizon": horizon, "base": base_mode,
            "n_base": n_base if base_mode == "fresh" else 0,
            "n_spans_harvested": len(origin),
            "candidates": [by_index[i].to_json() for i in sorted(by_index)],
            "ranking": order,
            "pick": (loom_probe.pick_gauge(gauges)
                     if any(g.cell is not None for g in gauges) else None),
            "dead_rows": dead,
            # ★ The regime this probe actually ran in — see the confound in
            # run_probe's docstring. True means the fan it scored against was
            # drawn under a wear the probe replies did NOT carry, so the cell
            # mixes "code added" with "wear removed". Clean = False.
            "session_worn_during_draw": bool(worn_at_draw),
            "clean_regime": not bool(worn_at_draw),
            # ★ Can this probe tell these candidates apart? Answered from the
            # probe's own reps, on THIS fan — not from a prior. Needs reps>=2.
            "resolution": loom_probe.resolution_report(gauges),
            "timing_s": {"generate": round(gen_s, 1),
                         "harvest": round(harvest_s, 1),
                         "total": round(total_s, 1)},
            "harvest_via": harvest_via, "harvest_worker_detail": worker_detail,
            "probe_dir": str(probe_dir),
            "absolute_comparable": base_mode == "fresh",
            "label": (
                "the gauge, ESTIMATED AND UNCERTIFIED. This is a PICK "
                "policy (71% of an oracle's picking skill), not a verdict: this "
                "same gauge failed its registered test as an instrument. The "
                "judge remains the instrument; you still choose."
                + ("" if base_mode == "fresh" else
                   " base='fan' (leave-one-out) makes these RANKING statistics "
                   "only — the constant self-distance offset cancels in the "
                   "order but NOT in the value, so do not read them as Δnr.")),
        }
        sess.last_probe = receipt
        sess.n_probes += 1
        logger.info("PROBE session=%s loom=%s k=%d reps=%d base=%s -> pick=%s "
                    "in %.1fs (gen %.1f, harvest %.1f)", session, sess.loom_id,
                    k, reps, base_mode, receipt["pick"], total_s, gen_s, harvest_s)
        return receipt

    def do_probe(self, session: str, text: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
        """POST /probe — run the gauge probe on this session's current fan.

        `text` must be the SAME contemplated user turn the draw was made
        against: the probe compares steered replies to that prefix against
        futures of that prefix, and a different prefix would silently compare
        two unrelated things. It is checked, not trusted.
        """
        sess = self.state.get(session)
        if not sess.candidates:
            raise ApiError(ErrorCode.NO_CANDIDATES, "no candidates — /loom first")
        if not text.strip():
            raise ValueError("/probe needs 'text' — the same contemplated user "
                             "turn the draw was made against")
        history, _ = trim_history(list(sess.histories["loom"]), int(self.args.max_turns))
        # ★ The WHOLE prefix is checked, not just the contemplated turn: the
        # conversation behind the fan can have moved since the draw (/chat,
        # /undo, /edit, /truncate, /reroll all mutate the loom history and
        # none of them clear `candidates`). Probing against a prefix the
        # futures did not come from compares two unrelated things and looks
        # perfectly healthy doing it.
        fp = prefix_fingerprint(history, text)
        if sess.last_loom_prefix_fp is not None and fp != sess.last_loom_prefix_fp:
            raise ApiError(
                ErrorCode.PROBE_PREFIX_MISMATCH,
                "/probe's prefix does not match the draw's. The probe ranks "
                "steered replies against THIS prefix's futures, so the "
                "contemplated turn AND the conversation behind it must be the "
                "ones the fan was drawn from"
                + (" — the text differs."
                   if sess.last_loom_text is not None
                   and text != sess.last_loom_text
                   else " — the text matches, so the CONVERSATION moved "
                        "(a /chat, /undo, /edit, /truncate or /reroll since "
                        "the draw).")
                + " /loom again.")
        ids = self.runtime.render(build_messages(history, text, None))
        t0 = time.time()
        receipt = self.run_probe(session, sess, ids, cfg)
        # `wear` is optional and must be an object; a bare `true` would sail
        # past a truthiness gate and die on True.get -> 500. Every other
        # optional-object field on this server is type-checked.
        wear = cfg.get("wear")
        if wear is not None and not isinstance(wear, Mapping):
            raise ValueError(
                "probe 'wear' must be an object of wear settings, e.g. "
                '{"alpha": 0.45} — omit it to probe without wearing')
        if wear and receipt["pick"] is not None:
            worn = self.do_wear(session, int(receipt["pick"]),
                           float(wear.get("alpha", cfg.get("alpha", 0.35))),
                           sess.loom_id, wear.get("dose_policy"),
                           wear.get("lever_kind"))
            receipt["worn"] = worn["worn"]
        receipt["wall_s"] = round(time.time() - t0, 1)
        return {"session": session, "probe": receipt,
                "futures": sess.last_futures, "worn": sess.worn_public()}
