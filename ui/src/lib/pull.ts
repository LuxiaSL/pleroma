/**
 * PULL — the map's own prediction of how far a future wants to move, as a
 * multiple of its fan's mean (1.00 = average for this draw).
 *
 * ONE number per future, shared by the card and the 3-D view (the old UI
 * learned that a plot and a card stating two different pulls for one future
 * is a fault). Resolution order, as in loom_ui.html's fanPull():
 *   1. `scores.predicted_dose_scale` from the server — exactly what
 *      dose_policy "predicted" would multiply the wear by (clamped to the
 *      server's range, so a value AT the clamp is flagged);
 *   2. `scores.loudness` over the fan's mean loudness;
 *   3. (3-D view only) ‖centred code‖ over the fan's mean ‖centred code‖.
 */
import type { Future, FutureScores } from "../api/types";

export type PullSource = "server" | "fan-loudness" | "code";

export interface Pull {
  value: number;
  from: PullSource;
}

export function fanLoudnessRef(futures: readonly Future[]): number | null {
  const vs = futures.map((f) => Number(f.scores?.loudness)).filter((v) => Number.isFinite(v) && v > 0);
  if (vs.length < 2) return null;
  return vs.reduce((a, b) => a + b, 0) / vs.length;
}

export function fanPull(scores: FutureScores | null | undefined, loudnessRef: number | null): Pull | null {
  const server = Number(scores?.predicted_dose_scale);
  if (scores?.predicted_dose_scale != null && Number.isFinite(server) && server > 0) {
    return { value: server, from: "server" };
  }
  const l = Number(scores?.loudness);
  if (loudnessRef && scores?.loudness != null && Number.isFinite(l) && l > 0) {
    return { value: l / loudnessRef, from: "fan-loudness" };
  }
  return null;
}

/** True when a server pull sits at (or beyond) the dose-scale clamp. */
export function atClamp(pull: Pull | null, clamp: readonly number[] | null | undefined): boolean {
  if (pull?.from !== "server" || !clamp || clamp.length < 2) return false;
  const lo = clamp[0] as number;
  const hi = clamp[1] as number;
  return pull.value <= lo + 1e-4 || pull.value >= hi - 1e-4;
}
