/**
 * The dose band: which zone an alpha falls in, read from /info.dose_band.
 *
 * ★ A map with no band is UNCALIBRATED at every alpha — never fall back to
 * another map's zones: α is measured on each map's own ruler (norm_ref), and
 * two maps of one model can differ by 2× at the same α (docs/FINDINGS.md §2).
 * An absent `alpha_max` is absent, not 0: reading it as 0 would pin every wear
 * at α = 0.
 */
import type { DoseBandInfo, WornPublic } from "../api/types";

export interface DoseZone {
  lo: number;
  hi: number;
  hi_inclusive: boolean;
  key: string;
  label: string;
  hint: string;
}

export const UNCALIBRATED_ZONE: DoseZone = {
  lo: 0,
  hi: Number.POSITIVE_INFINITY,
  hi_inclusive: true,
  key: "uncalibrated",
  label: "uncalibrated",
  hint:
    "no dose band for this map — every alpha is unclassified. Doses do not transfer between maps; " +
    "do not carry over a number that felt right on a different map.",
};

export const DEFAULT_ALPHA_MAX = 1.5;

function num(v: unknown): number | null {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** The band's zones, validated; rows missing a bound are dropped, not guessed. */
export function doseZones(band: DoseBandInfo | null | undefined): DoseZone[] {
  const rows = band?.zones;
  if (!Array.isArray(rows)) return [];
  const out: DoseZone[] = [];
  for (const r of rows) {
    const lo = num(r.lo);
    const hi = num(r.hi);
    if (lo === null || hi === null || typeof r.key !== "string") continue;
    out.push({
      lo,
      hi,
      hi_inclusive: r.hi_inclusive === true,
      key: r.key,
      label: typeof r.label === "string" ? r.label : r.key,
      hint: typeof r.hint === "string" ? r.hint : "",
    });
  }
  return out.sort((a, b) => a.lo - b.lo);
}

export function alphaMax(band: DoseBandInfo | null | undefined): number {
  const v = band?.alpha_max;
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : DEFAULT_ALPHA_MAX;
}

/** First match wins, so a closed upper bound keeps its own zone. */
export function zoneFor(alpha: number, zones: readonly DoseZone[]): DoseZone {
  if (!zones.length) return UNCALIBRATED_ZONE;
  const v = Number.isFinite(alpha) ? alpha : 0;
  for (const z of zones) {
    if (v < z.lo) continue;
    if (z.hi_inclusive ? v <= z.hi : v < z.hi) return z;
  }
  return v < (zones[0] as DoseZone).lo ? (zones[0] as DoseZone) : (zones[zones.length - 1] as DoseZone);
}

/** The middle of the band's threshold zone — a sane starting knob — or null. */
export function bandStartAlpha(zones: readonly DoseZone[]): number | null {
  const z = zones.find((q) => q.key === "threshold");
  if (!z || !Number.isFinite(z.hi) || z.hi <= z.lo) return null;
  return Math.round((z.lo + z.hi) / 2 / 0.005) * 0.005;
}

export interface EffectiveDose {
  alpha: number;
  /** alpha at the loudest site — what the ZONE is read at */
  alphaWorst: number;
  scale: number;
  scaled: boolean;
  clamped: boolean;
  policy: string;
}

/** What a worn code's alpha means once its dose receipt is in (old UI's effectiveAlpha). */
export function effectiveDose(w: Pick<WornPublic, "alpha" | "dose"> | null | undefined): EffectiveDose {
  const a = Number(w?.alpha ?? 0);
  const d = w?.dose;
  const s = d && d.policy !== "flat" ? Number(d.scale_mean) : Number.NaN;
  if (!d || !Number.isFinite(s) || s <= 0) {
    return { alpha: a, alphaWorst: a, scale: 1, scaled: false, clamped: false, policy: d?.policy ?? "flat" };
  }
  const per = d.scale.map(Number).filter((v) => Number.isFinite(v) && v > 0);
  const worst = per.length ? Math.max(...per) : s;
  return {
    alpha: a * s,
    alphaWorst: a * worst,
    scale: s,
    scaled: Math.abs(s - 1) > 1e-6,
    clamped: d.clamped.length > 0,
    policy: d.policy,
  };
}
