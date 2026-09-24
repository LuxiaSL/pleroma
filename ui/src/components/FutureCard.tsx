import { useState } from "preact/hooks";
import type { Future, WornPublic } from "../api/types";
import { fmt, snippet, words } from "../lib/format";
import { atClamp, type Pull } from "../lib/pull";

export interface FutureCardProps {
  future: Future;
  slot: number;
  horizon: number | null;
  pull: Pull | null;
  clamp: readonly number[] | null;
  selected: boolean;
  wornHere: boolean;
  lone: boolean;
  worn: WornPublic | null;
  onSelect: (slot: number) => void;
}

const PULL_TIP =
  "PULL — the map's own prediction of how far this future wants to move, as a multiple of this fan's mean " +
  "(1.00 = average for this draw). When the server reports predicted_dose_scale this IS that number: what " +
  "dose_policy 'predicted' would multiply the wear by. Estimated, uncertified.";
const DISTINCT_TIP =
  "DISTINCT — mean cosine distance from this future's SIGNATURE to the rest of the fan (~0–2). Signature " +
  "space, not the map's output and not what a reader sees.";
const LOUD_TIP =
  "‖raw‖ — the same quantity as pull in absolute units: the map's raw output norm before norm-matching, " +
  "against the ruler in force (never compare a bare alpha across rulers).";
const GAUGE_TIP =
  "GAUGE-M2 — measured by steering: this future's code was worn, a short reply generated, and its rank read. " +
  "Probe-derived; null = unscorable.";

/** One future of the fan: its text, its scores, and whether it is the one worn. */
export function FutureCard(p: FutureCardProps) {
  const [open, setOpen] = useState(false);
  const f = p.future;
  const s = f.scores;
  const cut = p.horizon !== null && f.n_tokens >= p.horizon;
  const w = words(f.text);
  const clamped = atClamp(p.pull, p.clamp);
  const long = f.text.length > 280;
  const cls = `card${p.selected ? " selected" : ""}${p.wornHere ? " worn" : ""}${f.harvested ? "" : " dead"}`;
  return (
    <article class={cls} data-testid={`future-card-${f.index}`} aria-label={`future #${f.index}`}>
      <div class="head">
        <span class="idx">#{f.index}</span>
        {p.wornHere && <span class="tag worn">worn</span>}
        {p.lone && (
          <span class="tag lone" title="the most solitary future in code space (1 − max cos to any fan-mate)">
            alone
          </span>
        )}
        {!f.harvested && (
          <span class="tag" title={f.note ?? ""}>
            not harvested
          </span>
        )}
        {cut ? (
          <span
            class="tag"
            title="hit the horizon: a cut span has no length of its own — its word count is a rate, not a length the model chose"
          >
            cut · {w} w / {f.n_tokens} tok
          </span>
        ) : (
          <span class="tag" title="stopped on its own before the horizon">
            {w} w
          </span>
        )}
        <span class="spacer" />
        <button
          type="button"
          class="tiny selbtn"
          aria-pressed={p.selected ? "true" : "false"}
          disabled={!f.harvested}
          onClick={() => p.onSelect(p.slot)}
          data-testid={`select-${f.index}`}
        >
          {p.selected ? "selected" : "select"}
        </button>
      </div>
      {p.pull && (
        <div class="pullbar" title={PULL_TIP} aria-hidden="true">
          <span style={{ width: `${Math.max(2, Math.min(100, (p.pull.value / 2) * 100))}%` }} />
        </div>
      )}
      <div class="scores">
        <span title={PULL_TIP}>
          pull <b>×{fmt(p.pull?.value ?? null, 2)}</b>
          {clamped && (
            <span class="alarm" title="at the dose-scale clamp: the map asked for more than the clamp allows">
              {" "}
              ⚠ clamp
            </span>
          )}
        </span>
        <span title={DISTINCT_TIP}>
          distinct <b>{fmt(s.distinct, 3)}</b>
        </span>
        <span title={LOUD_TIP}>
          ‖raw‖ <b>{fmt(s.loudness, 3)}</b>
        </span>
        {s.gauge !== undefined && s.gauge !== null && (
          <span title={GAUGE_TIP}>
            gauge <b>{fmt(s.gauge, 3)}</b>
          </span>
        )}
        {p.worn && s.cos_to_worn !== undefined && s.cos_to_worn !== null && (
          <span title="cosine of this future's code to the code worn when the fan was drawn">
            cos→worn <b>{fmt(s.cos_to_worn, 3)}</b>
          </span>
        )}
      </div>
      <div class="text">{open || !long ? f.text : snippet(f.text, 280)}</div>
      {long && (
        <button
          type="button"
          class="tiny"
          aria-expanded={open ? "true" : "false"}
          onClick={() => setOpen(!open)}
          style={{ justifySelf: "start" }}
        >
          {open ? "less" : "more"}
        </button>
      )}
      {f.note && f.harvested && <div class="note">{f.note}</div>}
    </article>
  );
}
