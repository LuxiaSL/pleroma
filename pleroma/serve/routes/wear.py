"""POST /wear (a candidate of the current fan) and POST /wear_code (a lever
that belongs to no draw — ★ a dead path, kept: the UI does not call it, and it
stays because its ``code`` mode is the only way to wear a code from another
conversation; see ``pleroma.serve.atlas``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from pleroma.dose.policy import (
    DEFAULT_DOSE_POLICY,
    effective_alpha_json,
    flat_dose_scale,
    resolve_dose_policy,
)
from pleroma.levers.kind import DEFAULT_LEVER_KIND, fan_wear_fields, resolve_lever_kind
from pleroma.serve.atlas import (
    atlas_report_path,
    check_atlas_matches_bank,
    resolve_atlas_landmark,
)
from pleroma.serve.errors import ApiError, ErrorCode
from pleroma.serve.routes._base import RouteBase

logger = logging.getLogger("loom_serve")


class WearRoutes(RouteBase):

    def do_wear(self, session: str, index: int, alpha: float,
                loom_id: str | None,
                dose_policy: Any = None,
                lever_kind: Any = None) -> dict[str, Any]:
        """Wear candidate `index` of the session's current fan at dose `alpha`.

        ★ `dose_policy` (opt-in, default from
        --dose-policy-default which itself defaults to 'flat'):

          flat       — the incumbent. The candidate's lever is worn exactly as
                       `lever_of` flattened it, per-site norm = norm_ref[s].
                       BYTE-IDENTICAL to the unscaled lever: `apply()`
                       returns the same array object rather than multiplying
                       by 1.0.
          predicted  — the lever is rescaled per site by (this candidate's raw
                       map-output norm / the fan's mean raw norm), clamped to
                       [DOSE_SCALE_MIN, DOSE_SCALE_MAX). A candidate at the fan
                       average is unchanged; a louder-predicted one is injected
                       harder. The fan's average dose is unchanged, so alpha
                       keeps its calibrated meaning — but a SINGLE candidate's
                       effective alpha is alpha x its scale, which is what the
                       response's `effective_alpha` block reports and what the
                       dose band must be read at.

        The fan mean comes from THIS session's current draw (`sess.
        fan_mean_raw_norms`, replaced wholesale by every /loom) and is never
        borrowed, never recomputed from a different set, and never invented:
        no fan mean means `predicted` refuses.

        ★ `lever_kind` (default from --wear-lever-default
        which itself defaults to 'absolute'):

          absolute   — the candidate's own lever (`lever_of`): the vectors,
                       code and dose the draw computed, unmodified.
          contrast   — W·(x_i − mean of the other valid members): the
                       one-vs-rest object v1a was fit on. Refuses when the
                       draw has no contrast (fewer than 2 harvested).
                       `predicted` dosing then divides by the CONTRAST raw
                       norms and the fan's mean CONTRAST raw norms.
        """
        sess = self.state.get(session)
        if not sess.candidates:
            raise ApiError(ErrorCode.NO_CANDIDATES, "no candidates — /loom first")
        if loom_id is not None and loom_id != sess.loom_id:
            raise ApiError(
                ErrorCode.STALE_LOOM,
                f"loom_id {loom_id!r} is not the latest draw ({sess.loom_id!r}) — "
                "another tab or turn superseded it; /loom again"
            )
        match = [c for c in sess.candidates if c["index"] == index]
        if not match or match[0]["lever"] is None:
            raise ValueError(f"candidate {index} unavailable "
                             f"({match[0]['note'] if match else 'no such index'})")
        policy = resolve_dose_policy(dose_policy, str(self.args.dose_policy_default))
        kind = resolve_lever_kind(lever_kind, str(self.args.wear_lever_default))
        wear_fields, dose = fan_wear_fields(
            match[0], lever_kind=kind, dose_policy=policy,
            fan_mean=sess.fan_mean_raw_norms,
            fan_mean_contrast=sess.fan_mean_raw_norms_contrast,
            n_sites=self.loom_map.n_sites)
        if dose.was_clamped:
            logger.warning(
                "WEAR session=%s index=%d: dose scale CLAMPED %s (policy=%s, "
                "raw %s) — injected at the clamp, NOT at the map's raw "
                "prediction", session, index, list(dose.clamped), dose.policy,
                [round(x, 4) for x in dose.scale_raw])
        elif dose.policy != "flat":
            logger.info("WEAR session=%s index=%d policy=%s scale=%s alpha=%.3f",
                        session, index, dose.policy,
                        [round(x, 4) for x in dose.scale], float(alpha))
        if kind != DEFAULT_LEVER_KIND:
            logger.info("WEAR session=%s index=%d lever_kind=%s peers=%s",
                        session, index, kind, wear_fields.get("contrast_peers"))
        sess.worn = {"index": index, "alpha": float(alpha),
                     "loom_id": sess.loom_id,
                     "vectors": wear_fields["vectors"],
                     "code": wear_fields["code"],
                     "loom_dir": sess.last_loom_dir,
                     "map_fingerprint": self.map_fp,
                     "dose": wear_fields["dose"],
                     "lever_kind": wear_fields["lever_kind"],
                     **({"contrast_peers": wear_fields["contrast_peers"]}
                        if "contrast_peers" in wear_fields else {})}
        for record in sess.looms:  # the loom log remembers what got worn
            if record.loom_id == sess.loom_id:
                record.worn_index, record.worn_alpha = index, float(alpha)
        return {"session": session, "worn": sess.worn_public(),
                "sites": self.loom_map.sites,
                "dose": dose.to_json(),
                "effective_alpha": effective_alpha_json(float(alpha), dose),
                "lever_kind": kind}

    def do_wear_code(self, session: str, alpha: float,
                     group: str | None, axis: int | None,
                     pole: str | None, rank: int,
                     random_seed: int | None = None,
                     code: list[float] | None = None,
                     code_kind: str = "absolute",
                     dose_policy: Any = None) -> dict[str, Any]:
        """Wear a lever that does NOT come from the current fan — a banked
        landmark, a matched-norm random direction, or an explicit code.

        Four addressing modes:
            {"group": "w5_og01155|orig"}                -> that group's lever
            {"axis": 0, "pole": "low"|"high", "rank": 0} -> atlas extreme
            {"random_seed": 7}                           -> matched-norm random control
            {"code": [8 floats], "code_kind":
             "absolute"|"differential"}                  -> explicit code wear

        ★ It needs NO /loom first and no candidates, which is the point: a fan
        is not guaranteed to hold the contrast you want (its futures can share
        one manner and differ only in content). `loom_id` is null and `index`
        is -1 by design — this
        wear does not belong to a draw — and `source` records what was worn so
        the session snapshot stays auditable after a restart.

        ★ The `code` mode (transplant + fan-centered wear) is
        how a stored code from ANOTHER conversation's draw gets worn here
        (portability test), and how a client-computed
        `code_sel − mean(code_rest)` gets worn as the zero-training contrastive
        probe (`code_kind: "differential"` — mu_y cancels in the difference and
        is not added back). It needs only the map, not the lever bank.

        The lever is renormalised to the map's own per-site `norm_ref`, so alpha
        means the SAME THING here as for a /wear on a fan candidate. Skipping
        that would read alpha on the wrong ruler — the same alpha can differ
        2.139x in injected norm between two maps (docs/FINDINGS.md §2).

        ★ DOSE POLICY. This endpoint is ALWAYS flat, and
        that is not an oversight. `dose_policy: "predicted"` is defined as
        "divide by the FAN's mean raw norm instead of your own" — a bank
        group, a random direction and a bare code have no fan, so there is no
        mean and any number put in its place would be fabricated. An explicit
        request for 'predicted' here is REFUSED with that reason; the
        server-wide --dose-policy-default is deliberately NOT applied, because
        a flag meant for fan wears must not turn every landmark wear into a
        400. The flat dose receipt is still banked, so a /wear_code wear and a
        /wear wear carry the same shaped record.
        """
        if dose_policy is not None:
            requested = resolve_dose_policy(dose_policy, DEFAULT_DOSE_POLICY)
            if requested != "flat":
                raise ApiError(
                    ErrorCode.DOSE_POLICY_UNAVAILABLE,
                    f"dose_policy={requested!r} is not available on "
                    "/wear_code: it needs the mean raw norm of a FAN and this "
                    "wear has no fan (a bank group, a random direction or a "
                    "bare code belongs to no draw). Refusing to invent a fan "
                    "mean. Use dose_policy='flat' here, or /loom then /wear "
                    "if you want predicted dosing.")
        flat_dose = flat_dose_scale(self.loom_map.n_sites).to_json()
        modes = [m for m, on in (("group", group is not None),
                                 ("axis", axis is not None),
                                 ("random_seed", random_seed is not None),
                                 ("code", code is not None)) if on]
        if len(modes) > 1:
            raise ValueError(
                f"pass exactly ONE of 'group', 'axis'+'pole', 'random_seed' "
                f"or 'code'; got {modes}")
        if code is not None:
            if code_kind not in ("absolute", "differential"):
                raise ValueError(
                    "code_kind must be 'absolute' (a stored fan code, worn "
                    "as the candidate's own lever) or 'differential' (a "
                    "centered/contrastive code; mu_y is not added back)")
            vectors, raw_norms = self.loom_map.lever_from_code(
                code, differential=(code_kind == "differential"))
            code8 = [round(float(x), 4) for x in
                     np.asarray(code, dtype=np.float64).ravel()]
            sess = self.state.get(session)
            sess.worn = {
                "index": -1, "alpha": float(alpha), "loom_id": None,
                "vectors": vectors, "code": code8,
                "loom_dir": None, "map_fingerprint": self.map_fp,
                "source": f"explicit_code:{code_kind}",
                "dose": flat_dose,
            }
            logger.info("WEAR_CODE session=%s EXPLICIT %s alpha=%.3f "
                        "code_norm=%.4f raw_norms=%s",
                        session, code_kind, float(alpha),
                        float(np.linalg.norm(code8)),
                        [round(x, 3) for x in raw_norms])
            return {
                "session": session, "worn": sess.worn_public(),
                "sites": self.loom_map.sites,
                "source": {
                    "kind": "explicit_code", "code_kind": code_kind,
                    "code": code8,
                    "code_norm": round(float(np.linalg.norm(code8)), 4),
                    "raw_per_site_norms": [round(float(x), 4)
                                           for x in raw_norms],
                    "norm_matched_to": [round(float(x), 4)
                                        for x in self.loom_map.norm_ref],
                    "note": ("explicit-code wear: lever = code @ Vt"
                             + (" + mu_y" if code_kind == "absolute"
                                else " (differential: mu_y cancels)")
                             + ", per-site norm-matched AFTER any centering "
                               "(alpha is an absolute per-site norm: "
                               "docs/FINDINGS.md §2)"),
                },
            }
        if self.lever_bank is None:
            raise ValueError(
                "no lever bank loaded — restart the server with --lever-bank "
                "<levers.npz> to enable /wear_code's bank modes (the 'code' "
                "mode needs only the map)")
        if random_seed is not None:
            vectors, raw_norms = self.lever_bank.random_vectors(int(random_seed))
            sess = self.state.get(session)
            sess.worn = {
                "index": -1, "alpha": float(alpha), "loom_id": None,
                "vectors": vectors, "code": None,
                "loom_dir": None, "map_fingerprint": self.map_fp,
                "source": f"random:{int(random_seed)}",
                "dose": flat_dose,
            }
            logger.info("WEAR_CODE session=%s RANDOM seed=%d alpha=%.3f",
                        session, int(random_seed), float(alpha))
            return {
                "session": session, "worn": sess.worn_public(),
                "sites": self.loom_map.sites,
                "source": {
                    "kind": "random_direction", "group": None,
                    "random_seed": int(random_seed), "bank": self.lever_bank.path,
                    "raw_per_site_norms": [round(float(x), 4) for x in raw_norms],
                    "norm_matched_to": [round(float(x), 4)
                                        for x in self.lever_bank.norm_ref],
                    "note": ("random-direction control: matched per-site injected norm, "
                             "no basin. Matched on NORM, not damage — sound here "
                             "because the decoy design makes damage uninformative "
                             "about which future a reply resembles."),
                },
            }
        if group is None:
            if axis is None or pole is None:
                raise ValueError(
                    "/wear_code needs {'group': label}, "
                    "{'axis': n, 'pole': 'low'|'high'} or {'random_seed': n}")
            ap = atlas_report_path(self.args.atlas_report)
            if not ap.exists():
                raise ValueError(f"no atlas report at {ap} — pass 'group' "
                                 "directly, or build the atlas")
            atlas_blob = json.loads(ap.read_text())
            # ★ The fallback path holds a STALE w4 atlas whose ids
            # are not this bank's labels. Fail with the real reason, not with a
            # missing-group message that blames the group.
            check_atlas_matches_bank(atlas_blob, self.lever_bank, str(ap))
            group = resolve_atlas_landmark(atlas_blob, int(axis), str(pole),
                                           int(rank))
        vectors, raw_norms = self.lever_bank.vectors_for(group)
        gi = self.lever_bank.index_of[group]
        sess = self.state.get(session)
        sess.worn = {
            "index": -1, "alpha": float(alpha), "loom_id": None,
            "vectors": vectors, "code": None,
            "loom_dir": None, "map_fingerprint": self.map_fp,
            "source": f"bank:{group}",
            "dose": flat_dose,
        }
        logger.info("WEAR_CODE session=%s group=%s alpha=%.3f raw_norms=%s",
                    session, group, float(alpha),
                    [round(x, 3) for x in raw_norms])
        return {
            "session": session, "worn": sess.worn_public(),
            "sites": self.loom_map.sites,
            "source": {
                "kind": "bank_group", "group": group,
                "prompt_class": self.lever_bank.classes[gi],
                "bank": self.lever_bank.path,
                "axis": None if axis is None else int(axis),
                "pole": pole, "rank": int(rank),
                "raw_per_site_norms": [round(float(x), 4) for x in raw_norms],
                "norm_matched_to": [round(float(x), 4)
                                    for x in self.lever_bank.norm_ref],
            },
        }
