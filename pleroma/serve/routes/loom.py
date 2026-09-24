"""POST /loom — draw K futures of the contemplated turn, harvest, score, and
(optionally) probe and auto-wear.

``do_loom`` reads its inputs from the ``ServeContext`` (``self.*``) and the
model runtime (``rt.*``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pleroma.dose.policy import (
    dose_scale_for,
    effective_alpha_json,
    fan_mean_raw_norms,
    resolve_dose_policy,
)
from pleroma.format.chat import build_messages, trim_history
from pleroma.format.trim import split_modelc
from pleroma.levers.kind import attach_fan_contrast, fan_wear_fields, resolve_lever_kind
from pleroma.serve.draws import (
    BINS_PROMPT_FLOOR,
    attach_for_draw,
    cos_to_worn,
    drawn_under_wear_field,
    loom_turn_seed,
    next_loom_index,
    prefix_fingerprint,
    resolve_draw_wear,
)
from pleroma.serve.modelc_fields import modelc_public_fields
from pleroma.serve.models import AutoSpec
from pleroma.serve.persistence import MAX_PERSISTED_LOOMS, LoomSnapshot
from pleroma.serve.policies import pick_auto
from pleroma.serve.routes._base import RouteBase

logger = logging.getLogger("loom_serve")


class LoomRoutes(RouteBase):

    def do_loom(self, session: str, text: str, k: int, horizon: int,
                auto: dict[str, Any] | None = None,
                detach_wear: bool = False,
                probe: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Futures = K candidate replies to the CONTEMPLATED next user turn.

        The turn is NOT committed to history here — /loom is a preview of what
        the conversation could become; /chat afterwards actually says it (worn
        or not). Crisp semantics, and the futures are exactly the object the
        map was trained to read: continuations of one shared prefix.

        detach_wear (opt-in, default False — see resolve_draw_wear/
        attach_for_draw above): when true and something is worn, the K
        futures are generated with the wear's hook NOT attached — an unbent
        forward pass. The worn code itself is untouched: scores.cos_to_worn
        and the stay/swerve auto-policies below still compare every future
        against it, exactly as when the draw is under wear. Tradeoff, stated
        honestly: a future previewed unbent may read as slightly less
        predictive of what the RE-WORN generation actually produces, since
        that generation runs bent and this preview did not.
        """
        rt = self.runtime
        if k < 2:
            raise ValueError("k must be >= 2 (the extractor needs siblings)")
        if not text.strip():
            raise ValueError("/loom needs 'text' — the contemplated next user turn")
        sess = self.state.get(session)
        # `text` is REBOUND inside the banking loop below (it becomes each
        # future's decoded text). Capture the contemplated turn first — /probe
        # checks against it, and reading the loop's leftover would hand it the
        # last future's words instead of the user's.
        contemplated = text
        history, _ = trim_history(list(sess.histories["loom"]), int(self.args.max_turns))
        messages = build_messages(history, text, None)
        ids = rt.render(messages)
        plen = int(ids.shape[1])
        if plen < BINS_PROMPT_FLOOR:
            logger.warning(
                "prompt is %d tokens, below the bins harvest's n_bins floor of %d — "
                "the bins-B half of the harvest WILL fail for every future "
                "('prompt_length %d < n_bins %d'), so this fan comes back "
                "TEXT-ONLY: no map, no candidate levers, no spread, /wear "
                "impossible. Lengthen the prompt (in chat mode the template "
                "adds ~34 tokens, which keeps prompts above this floor).",
                plen, BINS_PROMPT_FLOOR, plen, BINS_PROMPT_FLOOR)
        turn = sum(1 for m in history if m["role"] == "assistant")
        sess_tag = hashlib.sha256(session.encode()).hexdigest()[:10]
        sess_dir = self.args.work_dir / sess_tag
        idx = next_loom_index(sess_dir, sess.n_looms)
        loom_id = f"{sess_tag}-{idx:03d}"
        loom_dir = sess_dir / f"loom_{idx:03d}"
        rec_dir = loom_dir / "gen_records"
        rec_dir.mkdir(parents=True)  # refuses if it exists — never overwrite
        # Futures are drawn under the CURRENT wear, if any — the loom sees the
        # futures of the model it currently is. A restored-but-inert wear (no
        # vectors) counts as no wear here, exactly as it does in /chat.
        # `worn` is kept for the REST of this function (scores.cos_to_worn,
        # stay/swerve) regardless of detach_wear; `draw_worn` is the narrower
        # question of what gets ATTACHED for the generation below. See
        # resolve_draw_wear's docstring for why these must stay two things.
        worn_vectors = sess.worn_vectors()
        worn = sess.worn if worn_vectors is not None else None
        draw_worn = resolve_draw_wear(worn, detach_wear)
        futures: list[dict[str, Any]] = []
        t0 = time.time()
        sess.progress = {"stage": "generate", "done": 0, "total": k}
        # Batched K-draw: ONE generate() call, batch=k, one seed for the whole
        # batch — (seed, k) reproduces the set of k futures together; a single
        # future is not reproducible on its own (see Runtime._draw_batch_inner).
        batch_seed = loom_turn_seed(self.args.seed, session, "loom", turn, draw=0)
        rows = attach_for_draw(
            draw_worn, rt.attach, lambda: rt.draw_batch(ids, k, batch_seed, int(horizon))
        )
        sess.progress = {"stage": "generate", "done": k, "total": k}
        for j, row in enumerate(rows):
            gen = rt.trim_to_eos(row, plen)
            full = row[:plen] + gen
            text = rt.decode(gen)
            msplit = split_modelc(text) if rt.prompt_mode == "modelc" else None
            record = {
                "generation_id": j, "prompt_id": f"loom{sess.n_looms:03d}",
                "prompt_class": "loom_context", "prompt": "<chat context>",
                "prompt_idx": 0, "seed_idx": j, "seed": batch_seed,
                "system_prompt": None, "user_prompt": "<chat context>",
                "prompt_length": plen, "num_generated_tokens": len(gen),
                "generated_text": text, "input_ids": full,
                "sampling": {"temperature": float(self.args.temperature),
                             "top_p": float(self.args.top_p),
                             "max_new_tokens": int(horizon), "eos_ids": rt.eos_ids,
                             "attn_implementation": "sdpa"},
                "model_id": str(self.args.model_path), "preset": str(self.args.preset),
                # A banked gen record must say which framing
                # produced it. The harvest never reads this (it replays
                # input_ids), but a fan on disk that cannot name its own cell
                # is unreadable six weeks later.
                "prompt_mode": rt.prompt_mode,
                **({"modelc_header": rt.modelc_header,
                    "stop_strings": list(rt.stop_strings) or None,
                    # generated_text stays the full decode; these are its
                    # split. Length reads use modelc_reply, never
                    # generated_text.
                    "modelc_reply": msplit.reply,
                    "modelc_dream": msplit.dream,
                    "modelc_tail_kind": msplit.tail_kind.value}
                   if msplit is not None else {}),
            }
            (rec_dir / f"gen_{j:03d}.json").write_text(json.dumps(record))
            fut: dict[str, Any] = {"index": j, "text": text, "n_tokens": len(gen)}
            if rt.prompt_mode == "modelc":
                # "Dream mode": the model imagining the visitor's turn inside
                # its own generation. Documented and expected, not
                # degeneration — so it is KEPT, as its own field, and never
                # filtered (model C's format documents it; see
                # pleroma.format.modelc). `text` becomes the trimmed reply;
                # the gen_record on disk keeps the full decode.
                assert msplit is not None
                fut["text"] = msplit.reply
                fut.update(modelc_public_fields(msplit))
            futures.append(fut)
        gen_s = time.time() - t0

        # Harvest: the same frozen instruments the corpora went through.
        harvest_via, worker_detail, harvest_s = self.harvester.run(loom_dir, sess)

        with np.load(loom_dir / "bins.npz", allow_pickle=True) as bnpz:
            bins_feat = np.asarray(bnpz["features"], dtype=np.float64)
            bins_gid = [int(x) for x in bnpz["generation_id"]]
        bins_of = {g: i for i, g in enumerate(bins_gid)}
        candidates = []
        z_rows: list[np.ndarray | None] = []
        x_rows: list[np.ndarray | None] = []  # the map's input rows (lever_kind)
        for f in futures:
            j = f["index"]
            sig_path = loom_dir / "signatures" / f"gen_{j:03d}.npz"
            note = None
            raw_norms: list[float] | None = None
            code8: list[float] | None = None
            z_row: np.ndarray | None = None
            x_row: np.ndarray | None = None
            try:
                with np.load(sig_path, allow_pickle=True) as snpz:
                    sig = np.asarray(snpz["features"], dtype=np.float64)
                brow = bins_feat[bins_of[j]]
                if not np.isfinite(brow).all():
                    raise ValueError("bins row is non-finite")
                # == loom_map.lever_of(sig, brow), bit for bit (it is literally
                # this composition); split so the SAME x feeds the contrast.
                x_in = self.loom_map.input_of(sig, brow)
                lever, raw_norms, code8 = self.loom_map.lever_of_input(x_in)
                x_row = x_in
                z_row = self.loom_map.z_corpus_of(sig)
            except Exception as exc:  # noqa: BLE001 — a dead future is listed, not hidden
                note = f"{type(exc).__name__}: {exc}"
                lever = None
            # `z`: the candidate's
            # corpus-standardized signature, kept on the candidate so /probe
            # can rank a probe reply against THIS fan without re-reading the
            # loom dir. Same lifetime as `lever` — replaced wholesale by every
            # /loom, never persisted (it is a big array, and the snapshot is
            # deliberately jq-readable; see SessionSnapshot).
            candidates.append({"index": j, "lever": lever, "note": note,
                               "raw_norms": raw_norms, "code": code8,
                               "z": z_row})
            z_rows.append(z_row)
            x_rows.append(x_row)

        # ── the fan-contrast lever (lever_kind). Beside the
        # absolute one, never instead of it: every candidate keeps its
        # absolute `lever`/`raw_norms`/`code` and gains
        # `contrast_*`. Needs >= 2 harvested members; otherwise contrast is
        # unavailable for this draw and says so. Never fatal to the draw.
        fan_mean_contrast: list[float] | None = None
        fan_contrast_note: str | None = None
        try:
            fan_mean_contrast, fan_contrast_note = attach_fan_contrast(
                candidates, x_rows, self.loom_map.contrast_lever_of)
        except Exception as exc:  # noqa: BLE001 — the absolute draw outranks it
            fan_contrast_note = f"{type(exc).__name__}: {exc}"
            for c in candidates:
                c.update({"contrast_lever": None, "contrast_raw_norms": None,
                          "contrast_code": None, "contrast_peers": None,
                          "contrast_note": fan_contrast_note})
        if fan_contrast_note:
            logger.info("loom %s: lever_kind='contrast' unavailable (%s)",
                        loom_id, fan_contrast_note)

        # ── advisory scores: the spread of paths, read in signature space,
        # aimed at the K futures of ONE prefix, where within-fan cosine is
        # exactly the right instrument (unlike cos-to-contrast-vector, which
        # does not work as a quality bar).
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
            # No 2-means camps / separation: v1a has no partition, and a
            # signature-space 2-means split of a fan tracks nothing a reader
            # sees (its minority size vs a reader's: Spearman -0.04, n=16).
            spread_info["loudness_ref"] = round(float(np.mean(self.loom_map.norm_ref)), 3)
        # ── the fan's own magnitude scale (predicted
        # dosing). Computed HERE and only here: the loom is the one place
        # that knows all K candidates at once, and `dose_policy: "predicted"`
        # is defined against THIS draw's mean. A failure to compute it is
        # logged and leaves the fan mean None — predicted dosing then refuses
        # at /wear rather than falling back to a borrowed number.
        fan_mean: list[float] | None = None
        try:
            fan_mean = fan_mean_raw_norms(candidates)
        except ValueError as exc:
            logger.warning(
                "loom %s: could not compute the fan mean raw norms (%s) — "
                "dose_policy='predicted' will refuse for this draw; 'flat' "
                "is unaffected", loom_id, exc)
        for c in candidates:
            i = c["index"]
            if c["raw_norms"] is not None:
                scores[i]["loudness"] = round(float(np.mean(c["raw_norms"])), 3)
                # ADVISORY, estimated/uncertified: what
                # dose_policy='predicted' WOULD multiply this candidate's
                # injected magnitude by, clamp included. Shown so the choice
                # is visible before the wear, never used by pick_auto.
                if fan_mean is not None:
                    try:
                        scores[i]["predicted_dose_scale"] = round(
                            dose_scale_for(
                                policy="predicted", raw_norms=c["raw_norms"],
                                fan_mean=fan_mean,
                                n_sites=self.loom_map.n_sites).scale_mean, 3)
                    except ValueError as exc:  # a malformed row, not fatal
                        logger.warning("loom %s candidate %d: no predicted "
                                       "dose scale (%s)", loom_id, i, exc)
            # The same advisory for the
            # CONTRAST lever — its own raw norms over the fan's mean CONTRAST
            # raw norms. Same kind on both sides of the ratio, or it is a
            # number in no unit. Absent when the draw has no contrast.
            if (c.get("contrast_raw_norms") is not None
                    and fan_mean_contrast is not None):
                try:
                    scores[i]["predicted_dose_scale_contrast"] = round(
                        dose_scale_for(
                            policy="predicted",
                            raw_norms=c["contrast_raw_norms"],
                            fan_mean=fan_mean_contrast,
                            n_sites=self.loom_map.n_sites).scale_mean, 3)
                except ValueError as exc:
                    logger.warning("loom %s candidate %d: no contrast "
                                   "predicted dose scale (%s)", loom_id, i, exc)
            if worn and c["lever"] is not None:
                scores[i]["cos_to_worn"] = cos_to_worn(c["lever"], worn["vectors"])
        sess.candidates = candidates
        sess.fan_mean_raw_norms = fan_mean
        sess.fan_mean_raw_norms_contrast = fan_mean_contrast
        sess.fan_contrast_note = fan_contrast_note
        sess.loom_id = loom_id
        sess.last_loom_dir = str(loom_dir)
        sess.last_loom_text = contemplated
        sess.last_loom_prefix_fp = prefix_fingerprint(history, contemplated)
        sess.last_probe = None  # a new fan has not been probed
        sess.n_probes = 0
        sess.n_looms += 1
        futures_public = [
            {"index": f["index"], "n_tokens": f["n_tokens"], "text": f["text"],
             "harvested": candidates[i]["lever"] is not None,
             "note": candidates[i]["note"], "scores": scores[f["index"]],
             "code": candidates[i]["code"],
             # The contrast lever's code, null when this draw has no
             # contrast. `code` above is the absolute lever's.
             "code_contrast": candidates[i].get("contrast_code"),
             **{key: f[key] for key in ("dream", "tail_kind",
                                        "modelc_dream_blocks", "reply_words")
                if key in f}}
            for i, f in enumerate(futures)
        ]
        sess.last_futures = futures_public
        record = LoomSnapshot(
            loom_id=loom_id, k=int(k), horizon=int(horizon), created_at=t0,
            loom_dir=str(loom_dir), spread=spread_info,
            futures=[LoomSnapshot.future_meta(f) for f in futures_public],
            detach_wear=bool(detach_wear),
        )
        sess.looms.append(record)
        if len(sess.looms) > MAX_PERSISTED_LOOMS:
            del sess.looms[:-MAX_PERSISTED_LOOMS]

        # ── the probe (opt-in). Runs AFTER the fan is
        # scored and BEFORE auto-select, so `auto.policy: "gauge"` has real
        # measured scores to pick on within this one call. Never runs unless
        # asked: the default turn stays free, because the free `loudest`
        # policy already captures 43% of an oracle's picking skill.
        # run_probe writes each gauge into the very score dicts `scores`,
        # `futures_public` and `sess.last_futures` all share, so there is one
        # number and nothing to copy back.
        # `probe` arrives already normalised by request_probe(): None for no
        # probe, else a Mapping of settings (possibly empty = defaults). The
        # empty-dict case is why this is not a bare `if probe:`.
        probe_receipt: dict[str, Any] | None = None
        if probe is not None:
            # ★ A PROBE FAILURE MUST NOT DESTROY THE DRAW. /loom's standing
            # contract is that a dead future is LISTED with a note and the
            # fan still comes back. run_probe raises on several real
            # conditions (an unharvested candidate with no code or no z, a
            # harvest failure, a probe-dir collision) — and by this point the
            # 15 s draw is already done and `sess.candidates` already
            # replaced, so letting it propagate would turn a paid-for fan
            # into a 400 with no `futures` in the body and send the client to
            # /state to recover it. The probe degrades to an error inside the
            # response instead.
            try:
                probe_receipt = self.run_probe(session, sess, ids, probe)
                # The loom record's future_meta was snapshotted BEFORE the
                # probe ran, so without this refresh looms[].futures[].scores
                # would lack `gauge` while looms[].auto_selected.scores (a
                # shallow copy of a live dict) carried it — one snapshot
                # giving two answers to "was this fan probed?".
                record.futures = [LoomSnapshot.future_meta(f)
                                  for f in futures_public]
            except Exception as exc:  # noqa: BLE001 — the draw outranks the probe
                logger.exception("probe failed for loom %s — serving the draw "
                                 "without it", loom_id)
                probe_receipt = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "the PROBE failed; the draw above is unaffected "
                            "and every free policy still works. "
                            "auto.policy='gauge' will refuse until a probe "
                            "succeeds — POST /probe to retry without redrawing.",
                }

        auto_selected: dict[str, Any] | None = None
        if auto:
            spec = AutoSpec.from_mapping(auto)
            policy, alpha = spec.policy, spec.alpha
            live = [i for i in ok if candidates[i]["lever"] is not None]
            # stay/swerve rank on cos_to_worn, which is filled in above from
            # `worn` (never `draw_worn`) — auto-select works identically
            # whether this draw was made under wear or with it detached.
            pick, effective = pick_auto(policy, scores, live)
            # The auto-wear takes the SAME dose policy a manual /wear would —
            # the body's own `auto.dose_policy` if it sent one, else the
            # server default. A refusal here (predicted with no fan mean)
            # raises out of /loom as a 400 rather than quietly wearing flat.
            # And the SAME lever_kind — `auto.lever_kind`
            # if sent, else --wear-lever-default. `contrast` on a draw with no
            # contrast raises out of /loom as a 400, never wears absolute.
            wear_fields, dose = fan_wear_fields(
                candidates[pick],
                lever_kind=resolve_lever_kind(spec.lever_kind,
                                              str(self.args.wear_lever_default)),
                dose_policy=resolve_dose_policy(spec.dose_policy,
                                                str(self.args.dose_policy_default)),
                fan_mean=fan_mean, fan_mean_contrast=fan_mean_contrast,
                n_sites=self.loom_map.n_sites)
            sess.worn = {"index": pick, "alpha": alpha, "loom_id": loom_id,
                         "vectors": wear_fields["vectors"],
                         "code": wear_fields["code"],
                         "loom_dir": str(loom_dir), "map_fingerprint": self.map_fp,
                         "dose": wear_fields["dose"],
                         "lever_kind": wear_fields["lever_kind"],
                         **({"contrast_peers": wear_fields["contrast_peers"]}
                            if "contrast_peers" in wear_fields else {})}
            if dose.was_clamped:
                logger.warning(
                    "AUTO-WEAR session=%s loom=%s index=%d: dose scale "
                    "CLAMPED %s (policy=%s) — the wear is injected at the "
                    "clamp, not at the map's raw prediction",
                    session, loom_id, pick, list(dose.clamped), dose.policy)
            auto_selected = {"policy": policy, "effective_policy": effective,
                             "index": pick, "alpha": alpha,
                             "scores": scores[pick],
                             "dose": dose.to_json(),
                             "lever_kind": wear_fields["lever_kind"],
                             "effective_alpha": effective_alpha_json(alpha, dose)}
            record.auto_selected = dict(auto_selected)
            record.worn_index, record.worn_alpha = pick, alpha

        return {
            "session": session, "loom_id": loom_id, "n_futures": k,
            "horizon": int(horizon),
            "prompt_mode": rt.prompt_mode,
            "prompt_length": plen,
            # ★ A prompt shorter than the bins harvest's n_bins has NO bins-B
            # features, so the map cannot run and the fan comes back TEXT-ONLY
            # (no scores, no spread, no fan_mean_raw_norms, /wear impossible).
            # The chat template pads ~34 tokens onto every prompt, so only raw
            # and modelc prompts reach this floor in practice. Published per
            # draw so the caller does not have to infer it from a null.
            "bins_prompt_floor": BINS_PROMPT_FLOOR,
            "bins_feasible": plen >= BINS_PROMPT_FLOOR,
            "detach_wear": bool(detach_wear),
            # The TRUTH of what generated these futures — never what merely
            # happens to be worn. None both when nothing is worn and when
            # detach_wear detached it; `worn` below (the session's actual,
            # untouched wear) plus `detach_wear` above disambiguate the two.
            "drawn_under_wear": drawn_under_wear_field(draw_worn),
            "futures": futures_public,
            "spread": spread_info,
            # The divisor dose_policy='predicted' uses for
            # THIS draw, published so a receipt can be checked by hand. None
            # = nothing harvested, and predicted dosing will refuse.
            "fan_mean_raw_norms": (
                None if fan_mean is None
                else [round(float(x), 6) for x in fan_mean]),
            # The same divisor for the contrast
            # levers, and — when null — why this draw has no contrast.
            "fan_mean_raw_norms_contrast": (
                None if fan_mean_contrast is None
                else [round(float(x), 6) for x in fan_mean_contrast]),
            "contrast_note": fan_contrast_note,
            "auto_selected": auto_selected,
            # Null unless this request opted into the probe.
            # Its own timing is NOT folded into timing_s below — a caller must
            # be able to see what the draw cost and what the probe cost apart.
            "probe": probe_receipt,
            "worn": sess.worn_public(),
            "timing_s": {"generate": round(gen_s, 1), "harvest": round(harvest_s, 1)},
            "harvest_via": harvest_via,
            "harvest_worker_detail": worker_detail,
            "loom_dir": str(loom_dir),
        }
