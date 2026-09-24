"""Stage `dose`: is α calibrated for THIS map under THIS span, and do steered
replies show damage?

| gate        | catalog | question |
|-------------|---------|----------|
| band        | D-02, U-02 | is there a MEASURED band, stamped for this map and ruler? |
| band-span   | C7-SPAN (injection span), D-02 | was the band measured under the injection span the profile serves? |
| damage      | D-01, D-05, F-06 | do steered replies loop (degenerate repetition)? |

A band records its span as `injection_span` — top-level, or inside a
`span_receipt` / `receipt` / `ladder` object — using the exact field
`pleroma.steer.span_receipt` writes into every steered-output receipt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.config import ModelProfile, PromptMode
from pleroma.dose.band import (
    DoseBand,
    default_sidecar_path,
    load_dose_band_file,
    load_map_identity,
)
from pleroma.format.trim import modelc_reply
from pleroma.map.identity import same_map
from pleroma.stats.diversity import repetition_rate, words
from pleroma.steer.spec import SpanError, as_span
from pleroma.validate._io import InputError, read_json, read_json_or_jsonl, require_mapping
from pleroma.validate.result import GateResult, failed, guarded, inconclusive, passed

STAGE = "dose"
#: A band measured before this date carries no span record, and the vLLM dose
#: lanes of that period wrote continuation-only.
SPAN_RECORDING_SINCE: str = "2026-09-23"
#: Where a band may carry its span receipt.
_SPAN_CONTAINERS: tuple[str, ...] = ("span_receipt", "receipt", "ladder")

#: A reply loops when this share of its word 4-grams are repeats. Normal prose
#: sits near 0; "…and so on, and so on" compounding (F-06) is far above.
LOOP_REPETITION_4GRAM: float = 0.20
#: Replies shorter than this have too few 4-grams to call a loop.
MIN_WORDS_FOR_LOOP: int = 12
#: FAIL when at least this share of scorable steered replies loop.
MAX_LOOP_SHARE: float = 0.10


# ── band ─────────────────────────────────────────────────────────────────────


def _resolve_band_path(band: str | Path | None, map_path: str | Path | None) -> Path | None:
    if band is not None:
        return Path(band)
    if map_path is not None:
        return default_sidecar_path(Path(map_path))
    return None


def _load_band(path: Path) -> tuple[DoseBand, dict[str, Any]]:
    """(validated band, raw JSON). Raises InputError when absent, ValueError
    when present but malformed (the server would refuse it)."""
    if not path.is_file():
        raise InputError(f"no dose band at {path}: α is uncalibrated for this map (tier none)")
    band = load_dose_band_file(path)  # ValueError on anything malformed
    raw = require_mapping(read_json(path, "dose band"), "dose band")
    return band, raw


@guarded(STAGE, "band")
def gate_band(band_path: str | Path | None, map_path: str | Path | None) -> GateResult:
    """D-02: α is not portable. A band must be MEASURED, and stamped for the
    served map (fingerprint) and its ruler (norm_ref)."""
    path = _resolve_band_path(band_path, map_path)
    if path is None:
        return inconclusive(STAGE, "band", "no dose band and no map supplied")
    try:
        band, _ = _load_band(path)
    except InputError as exc:
        return inconclusive(STAGE, "band", str(exc), path=str(path))
    except ValueError as exc:
        return failed(STAGE, "band", f"dose band is malformed, and the server refuses it: {exc}",
                      path=str(path))
    ev: dict[str, Any] = dict(path=str(path), tier=band.tier, source=band.source,
                              band_map_fingerprint=band.map_fingerprint,
                              measured_at=band.measured_at, norm_ref=list(band.norm_ref),
                              zones=[z.key for z in band.zones])
    if map_path is not None:
        mp = Path(map_path)
        if not mp.is_file():
            ev["map_note"] = f"map not found at {mp}: fingerprint unchecked"
        else:
            try:
                fp, _sites, norm_ref, _meta = load_map_identity(mp)
            except (KeyError, ValueError, OSError, EOFError) as exc:
                return inconclusive(STAGE, "band", f"map unreadable, fingerprint unchecked: {exc}",
                                    **ev)
            ev.update(map_fingerprint=fp, map_norm_ref=np.asarray(norm_ref).tolist())
            if band.map_fingerprint is None:
                return inconclusive(STAGE, "band",
                                    "band carries no map fingerprint: cannot show it was "
                                    "measured on this map", **ev)
            match = same_map(band.map_fingerprint, fp)
            ev["fingerprint_match"] = match
            if match == "legacy":
                return inconclusive(
                    STAGE, "band",
                    "band carries a LEGACY (v1) map fingerprint: it matches this map's "
                    "meta/sites/ruler but v1 never covered the weights, so it cannot "
                    "show the band was measured on THIS map — re-stamp it with the v2 "
                    "id once confirmed", **ev)
            if match == "mismatch":
                return failed(STAGE, "band",
                              f"band is stamped for map {band.map_fingerprint}, the map is {fp}: "
                              "its zones describe another map's α", **ev)
            if (len(band.norm_ref) != np.asarray(norm_ref).size
                    or not np.allclose(band.norm_ref, norm_ref, rtol=1e-6)):
                return failed(STAGE, "band",
                              "band was measured under a different ruler (norm_ref) than the "
                              "map wears: α means a different absolute norm", **ev)
    if band.tier != "measured":
        return inconclusive(STAGE, "band",
                            f"band is {band.tier.upper()} (rescaled from another map's ladder), "
                            "not measured on this map: its zones are provisional", **ev)
    return passed(STAGE, "band",
                  "measured band" + (", stamped for this map and its ruler"
                                     if "map_fingerprint" in ev else
                                     " (no map supplied: fingerprint unchecked)"), **ev)


def recorded_span(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """(span value, where it was found) from a band's raw JSON."""
    if raw.get("injection_span") is not None:
        return str(raw["injection_span"]), "injection_span"
    for key in _SPAN_CONTAINERS:
        box = raw.get(key)
        if isinstance(box, dict) and box.get("injection_span") is not None:
            return str(box["injection_span"]), f"{key}.injection_span"
    return None, None


@guarded(STAGE, "band-span")
def gate_band_span(band_path: str | Path | None, map_path: str | Path | None,
                   profile: ModelProfile) -> GateResult:
    """The loom injects UNIFORM (`docs/FINDINGS.md §4`); the 70B band
    was measured continuation-only. A band measured under another span, or recording none,
    does not describe the served dose."""
    served = profile.steer.injection_span.value
    path = _resolve_band_path(band_path, map_path)
    if path is None:
        return inconclusive(STAGE, "band-span", "no dose band and no map supplied",
                            served_span=served)
    try:
        band, raw = _load_band(path)
    except InputError as exc:
        return inconclusive(STAGE, "band-span", str(exc), path=str(path), served_span=served)
    except ValueError as exc:
        return failed(STAGE, "band-span", f"dose band is malformed: {exc}", path=str(path),
                      served_span=served)
    value, where = recorded_span(raw)
    ev: dict[str, Any] = dict(path=str(path), served_span=served, band_span=value,
                              recorded_at=where, measured_at=band.measured_at)
    if value is None:
        predates = bool(band.measured_at) and str(band.measured_at) < SPAN_RECORDING_SINCE
        on_vllm = "vllm" in f"{band.source} {band.notes or ''}".lower()
        if predates and on_vllm and served != "continuation":
            # Not a guess: before span recording, the vLLM dose lanes gated
            # positions >= prompt_len, so an unrecorded vLLM band is continuation.
            ev.update(band_span="continuation", recorded_at=None,
                      span_inferred_from="measured_at predates span recording and the "
                                         "band's source names the vLLM lane")
            return failed(STAGE, "band-span",
                          f"band measured under continuation (a vLLM dose lane before "
                          f"{SPAN_RECORDING_SINCE}, which injected only at positions >= "
                          f"prompt_len), served span {served}: re-measure the band under "
                          f"{served}", **ev)
        reason = (f"band records no injection span: cannot tell whether it was measured "
                  f"under the served span ({served})")
        if predates:
            reason += f"; it predates span recording ({band.measured_at})"
        return inconclusive(STAGE, "band-span", reason, **ev)
    try:
        span = as_span(value).value
    except SpanError as exc:
        return failed(STAGE, "band-span", f"band's recorded span is not a known span: {exc}",
                      **ev)
    if span != served:
        return failed(STAGE, "band-span",
                      f"band measured under {span}, served span {served}: its zones describe "
                      "a different intervention — re-measure under the served span", **ev)
    return passed(STAGE, "band-span", f"band measured under the served span ({served})", **ev)


# ── damage ───────────────────────────────────────────────────────────────────


def load_replies(path: str | Path, what: str = "replies") -> list[str]:
    """A JSON list of strings, `{"replies": [...]}`, or JSON Lines of strings
    or objects carrying `reply` / `text`."""
    blob = read_json_or_jsonl(path, what)
    if isinstance(blob, dict):
        blob = blob.get("replies")
        if blob is None:
            raise InputError(f"{what}: JSON object has no 'replies' list")
    if not isinstance(blob, list):
        raise InputError(f"{what} must be a list, got {type(blob).__name__}")
    out: list[str] = []
    for i, row in enumerate(blob):
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, dict) and isinstance(row.get("reply", row.get("text")), str):
            out.append(row.get("reply", row.get("text")))
        else:
            raise InputError(f"{what}[{i}] is neither a string nor an object with "
                             "'reply'/'text'")
    return out


def loop_scores(replies: Sequence[str], *, modelc_format: bool) -> list[float | None]:
    """4-gram repetition per reply (None = too short to call). Model C output
    is scored on its trimmed reply only (a dreamed turn is not the reply)."""
    scores: list[float | None] = []
    for text in replies:
        body = modelc_reply(text) if modelc_format else text
        ws = words(body)
        scores.append(repetition_rate(ws, 4) if len(ws) >= MIN_WORDS_FOR_LOOP else None)
    return scores


def _loop_share(scores: Sequence[float | None]) -> tuple[int, int, float | None]:
    scorable = [s for s in scores if s is not None]
    looped = sum(1 for s in scorable if s >= LOOP_REPETITION_4GRAM)
    return looped, len(scorable), (looped / len(scorable) if scorable else None)


@guarded(STAGE, "damage")
def gate_damage(replies: Sequence[str], profile: ModelProfile,
                base_replies: Sequence[str] | None = None) -> GateResult:
    """D-01 (damage before audibility), F-06 (compounding loops). A loop
    detector only: repetition is BLIND to word salad (D-05), so a PASS is not
    a fluency certificate."""
    mc = profile.format.mode is PromptMode.MODELC
    scores = loop_scores(replies, modelc_format=mc)
    looped, n, share = _loop_share(scores)
    ev: dict[str, Any] = dict(
        n_replies=len(replies), n_scorable=n, n_looped=looped, loop_share=share,
        threshold_repetition_4gram=LOOP_REPETITION_4GRAM, max_loop_share=MAX_LOOP_SHARE,
        min_words=MIN_WORDS_FOR_LOOP,
        looped_indices=[i for i, s in enumerate(scores)
                        if s is not None and s >= LOOP_REPETITION_4GRAM][:50])
    if n == 0:
        return inconclusive(STAGE, "damage",
                            f"no reply has >= {MIN_WORDS_FOR_LOOP} words: too short to call "
                            "a loop", **ev)
    if base_replies is not None:
        b_looped, b_n, b_share = _loop_share(loop_scores(base_replies, modelc_format=mc))
        ev.update(base_n_scorable=b_n, base_n_looped=b_looped, base_loop_share=b_share)
    else:
        b_share = None
    if share is not None and share >= MAX_LOOP_SHARE:
        if b_share is not None and b_share >= MAX_LOOP_SHARE:
            return inconclusive(STAGE, "damage",
                                f"{looped}/{n} steered replies loop, but so do {ev['base_n_looped']}"
                                f"/{ev['base_n_scorable']} unsteered ones: the loops are not "
                                "attributable to the dose", **ev)
        return failed(STAGE, "damage",
                      f"{looped}/{n} steered replies loop (4-gram repetition >= "
                      f"{LOOP_REPETITION_4GRAM}): the dose is damaging output — a judge "
                      "'detecting' it is reading breakage (D-01)", **ev)
    return passed(STAGE, "damage",
                  f"{looped}/{n} replies loop (< {MAX_LOOP_SHARE:.0%}); repetition is blind "
                  "to word salad, so this is a loop check, not a fluency certificate (D-05)",
                  **ev)


def run_dose(profile: ModelProfile, band: str | Path | None, map_path: str | Path | None,
             replies: str | Path | None, base_replies: str | Path | None) -> list[GateResult]:
    out: list[GateResult] = []
    if band is not None or map_path is not None:
        out.append(gate_band(band, map_path))
        out.append(gate_band_span(band, map_path, profile))
    if replies is not None:
        try:
            steered = load_replies(replies, "replies")
            base = load_replies(base_replies, "base replies") if base_replies else None
        except InputError as exc:
            out.append(inconclusive(STAGE, "damage", str(exc)))
        else:
            out.append(gate_damage(steered, profile, base))
    return out
