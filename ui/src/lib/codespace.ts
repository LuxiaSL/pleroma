/**
 * The fan as ONE object in code space — the data behind the 3-D view.
 *
 * What a wear injects is the lever Vtᵀ·(c_i − mean_{j≠i} c_j): k/(k−1) × the
 * FAN-CENTRED code. A map's Vt rows are orthonormal (max|Vt·Vtᵀ − I| ≈ 1e-12,
 * as an SVD's right singular vectors are), so norms and cosines of centred codes
 * ARE norms and cosines of the levers. Nothing here reads an axis as
 * meaningful: everything is a norm, a cosine, or a projection.
 *
 *   position  the fan's own top-3 principal directions of its centred codes.
 *             The basis is arbitrary; the geometry is not. All axes share one
 *             scale. The share of spread the view holds is reported.
 *   size      pull (lib/pull.ts): the card's number, so plot and card agree.
 *   thread    to the nearest fan-mate BY COSINE IN FULL SPACE — a thread may
 *             cross the picture; that is the view being a projection.
 *   alone     the most solitary future (1 − max cos to any fan-mate).
 */
import type { Future } from "../api/types";
import { cosine, norm, pcaProject } from "./pca";
import { fanLoudnessRef, fanPull, type PullSource } from "./pull";

export type CodeKind = "code" | "code_contrast";

export interface CodePoint {
  /** position of the future in the `futures` array */
  slot: number;
  /** the future's own index (what /wear takes) */
  index: number;
  /** coordinates on the top principal directions (z = 0 in a 2-D fan) */
  pos: [number, number, number];
  pull: number;
  pullFrom: PullSource;
  /** ‖centred code‖ / mean over the fan — always computed, for the hover */
  codePull: number;
  /** slot of the nearest fan-mate by full-space cosine */
  nn: number;
  nnCos: number;
  /** 1 − nnCos */
  solitude: number;
}

export interface CodeSpace {
  kind: CodeKind;
  points: CodePoint[];
  /** 3 when the fan spans ≥3 dimensions, else 2 */
  dims: 2 | 3;
  /** the code rank R (64 on v1a) */
  rank: number;
  /** share of the fan's total spread each shown axis holds */
  axisShare: number[];
  /** sum of axisShare */
  viewShare: number;
  /** slot of the most solitary future */
  lone: number;
  /** futures in the draw that could not be placed (not harvested / no code) */
  unplaced: number;
}

export type CodeSpaceResult = { ok: true; space: CodeSpace } | { ok: false; reason: string; placed: number };

export function codeOf(f: Future, kind: CodeKind): number[] | null {
  const c = kind === "code" ? f.code : f.code_contrast;
  return Array.isArray(c) && c.length > 0 ? c.map(Number) : null;
}

export function buildCodeSpace(futures: readonly Future[], kind: CodeKind = "code"): CodeSpaceResult {
  const items: { slot: number; index: number; c: number[] }[] = [];
  futures.forEach((f, slot) => {
    const c = f.harvested === true ? codeOf(f, kind) : null;
    if (c) items.push({ slot, index: f.index, c });
  });
  const k = items.length;
  const unplaced = futures.length - k;
  if (k < 3) {
    return {
      ok: false,
      placed: k,
      reason:
        k === 0
          ? `no harvested future carries a \`${kind}\` — nothing to place`
          : `${k} harvested future${k === 1 ? "" : "s"} with a \`${kind}\` — a fan needs at least 3 to have a shape`,
    };
  }
  const R = items[0]?.c.length ?? 0;
  if (items.some((it) => it.c.length !== R || it.c.some((x) => !Number.isFinite(x)))) {
    return {
      ok: false,
      placed: k,
      reason: `the fan's ${kind}s are ragged or non-finite — refusing to draw them`,
    };
  }
  // centred data has rank ≤ k−1, so a 3-future fan is a plane
  const dims: 2 | 3 = k >= 4 ? 3 : 2;
  const proj = pcaProject(
    items.map((it) => it.c),
    3,
  );
  const norms = proj.centred.map((x) => norm(x));
  const meanNorm = norms.reduce((s, v) => s + v, 0) / k;
  if (!(meanNorm > 1e-12)) {
    return { ok: false, placed: k, reason: `every ${kind} in this fan is identical — no spread to show` };
  }
  const ref = fanLoudnessRef(futures);
  const points: CodePoint[] = items.map((it, r) => {
    let best = Number.NEGATIVE_INFINITY;
    let nn = -1;
    for (let q = 0; q < k; q++) {
      if (q === r) continue;
      const cs = cosine(proj.centred[r] as number[], proj.centred[q] as number[]);
      if (cs > best) {
        best = cs;
        nn = q;
      }
    }
    const codePull = (norms[r] as number) / meanNorm;
    const fp = fanPull(futures[it.slot]?.scores, ref);
    const c = proj.coords[r] as number[];
    return {
      slot: it.slot,
      index: it.index,
      pos: [c[0] ?? 0, c[1] ?? 0, dims === 3 ? (c[2] ?? 0) : 0],
      pull: fp ? fp.value : codePull,
      pullFrom: fp ? fp.from : "code",
      codePull,
      nn: (items[nn] as { slot: number }).slot,
      nnCos: best,
      solitude: 1 - best,
    };
  });
  let lone = 0;
  points.forEach((p, i) => {
    if (p.solitude > (points[lone] as CodePoint).solitude) lone = i;
  });
  const axisShare = proj.share.slice(0, dims);
  return {
    ok: true,
    space: {
      kind,
      points,
      dims,
      rank: R,
      axisShare,
      viewShare: axisShare.reduce((s, v) => s + v, 0),
      lone: (points[lone] as CodePoint).slot,
      unplaced,
    },
  };
}
