import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Future, LoomResponse } from "../../src/api/types";
import { buildCodeSpace } from "../../src/lib/codespace";
import { cosine, jacobiEig, norm, pcaProject } from "../../src/lib/pca";

/** Deterministic PRNG (mulberry32). */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A random orthonormal R×R matrix (Gram–Schmidt on gaussian-ish columns). */
function randomRotation(R: number, seed: number): number[][] {
  const r = rng(seed);
  const cols: number[][] = [];
  while (cols.length < R) {
    let v = Array.from({ length: R }, () => r() - 0.5);
    for (const c of cols) {
      const d = v.reduce((s, x, i) => s + x * (c[i] as number), 0);
      v = v.map((x, i) => x - d * (c[i] as number));
    }
    const n = norm(v);
    if (n > 1e-8) cols.push(v.map((x) => x / n));
  }
  return cols;
}

const dist = (a: readonly number[], b: readonly number[]) =>
  Math.hypot(...a.map((x, i) => x - (b[i] as number)));

describe("jacobiEig", () => {
  it("diagonalises a known symmetric matrix", () => {
    // eigenvalues of [[2,1],[1,2]] are 1 and 3
    const { values, vectors } = jacobiEig([
      [2, 1],
      [1, 2],
    ]);
    expect([...values].sort()).toEqual([expect.closeTo(1, 10), expect.closeTo(3, 10)]);
    // A v = λ v for each column
    for (let c = 0; c < 2; c++) {
      const v0 = vectors[0]?.[c] as number;
      const v1 = vectors[1]?.[c] as number;
      expect(2 * v0 + v1).toBeCloseTo((values[c] as number) * v0, 10);
      expect(v0 + 2 * v1).toBeCloseTo((values[c] as number) * v1, 10);
    }
  });
});

describe("pcaProject — known answers", () => {
  // six points on three orthogonal axes with variances 18 : 8 : 2 (sum of squares)
  const pts = [
    [3, 0, 0, 0],
    [-3, 0, 0, 0],
    [0, 2, 0, 0],
    [0, -2, 0, 0],
    [0, 0, 1, 0],
    [0, 0, -1, 0],
  ];

  it("recovers the principal variances and coordinates", () => {
    const p = pcaProject(pts, 3);
    expect(p.eigenvalues.slice(0, 3)).toEqual([
      expect.closeTo(18, 9),
      expect.closeTo(8, 9),
      expect.closeTo(2, 9),
    ]);
    expect(p.share).toEqual([
      expect.closeTo(18 / 28, 9),
      expect.closeTo(8 / 28, 9),
      expect.closeTo(2 / 28, 9),
    ]);
    // each point lands on exactly one axis, at its known distance
    expect(Math.abs(p.coords[0]?.[0] as number)).toBeCloseTo(3, 9);
    expect(Math.abs(p.coords[2]?.[1] as number)).toBeCloseTo(2, 9);
    expect(Math.abs(p.coords[4]?.[2] as number)).toBeCloseTo(1, 9);
    expect(Math.abs(p.coords[0]?.[1] as number)).toBeCloseTo(0, 9);
  });

  it("is rotation-invariant: rank-3 data in 64-d keeps every pairwise distance", () => {
    const R = 64;
    const Q = randomRotation(R, 7);
    const r = rng(11);
    const k = 9;
    const low = Array.from({ length: k }, () => [4 * (r() - 0.5), 2 * (r() - 0.5), r() - 0.5]);
    // embed along three orthonormal directions, plus an offset (PCA must centre it away)
    const hi = low.map((l) =>
      Array.from(
        { length: R },
        (_, j) =>
          (l[0] as number) * (Q[0]?.[j] as number) +
          (l[1] as number) * (Q[1]?.[j] as number) +
          (l[2] as number) * (Q[2]?.[j] as number) +
          5,
      ),
    );
    const p = pcaProject(hi, 3);
    for (let a = 0; a < k; a++)
      for (let b = a + 1; b < k; b++) {
        expect(dist(p.coords[a] as number[], p.coords[b] as number[])).toBeCloseTo(
          dist(hi[a] as number[], hi[b] as number[]),
          8,
        );
      }
    expect(p.share.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 9);
  });

  it("agrees with the covariance-matrix route (XᵀX eigenvectors), up to sign", () => {
    const r = rng(3);
    const k = 8;
    const R = 5;
    const X = Array.from({ length: k }, () => Array.from({ length: R }, () => r() * 2 - 1));
    const p = pcaProject(X, 3);
    const C = p.centred;
    const cov = Array.from({ length: R }, (_, i) =>
      Array.from({ length: R }, (_, j) =>
        C.reduce((s, row) => s + (row[i] as number) * (row[j] as number), 0),
      ),
    );
    const eg = jacobiEig(cov);
    const order = eg.values
      .map((_, i) => i)
      .sort((a, b) => (eg.values[b] as number) - (eg.values[a] as number));
    for (let d = 0; d < 3; d++) {
      const col = order[d] as number;
      const dir = eg.vectors.map((row) => row[col] as number);
      const proj = C.map((row) => row.reduce((s, x, j) => s + x * (dir[j] as number), 0));
      const ours = p.coords.map((c) => c[d] as number);
      const sign = Math.sign(proj.reduce((s, v, i) => s + v * (ours[i] as number), 0)) || 1;
      proj.forEach((v, i) => {
        expect(ours[i]).toBeCloseTo(sign * v, 8);
      });
    }
  });

  it("fixes the basis sign deterministically (positive third moment)", () => {
    const p = pcaProject(
      pts.map((q) => q.map((x) => -x * 1.0001 + 0.1 * (q[0] as number))),
      3,
    );
    for (let d = 0; d < 3; d++)
      expect(p.coords.reduce((s, c) => s + (c[d] as number) ** 3, 0)).toBeGreaterThanOrEqual(0);
  });

  it("refuses ragged and non-finite input", () => {
    expect(() => pcaProject([[1, 2], [3]], 2)).toThrow(/ragged/);
    expect(() => pcaProject([[1, Number.NaN]], 2)).toThrow(/non-finite/);
  });
});

describe("buildCodeSpace", () => {
  const fut = (index: number, code: number[] | null, harvested = true): Future => ({
    index,
    n_tokens: 10,
    text: `future ${index}`,
    harvested,
    note: null,
    scores: {},
    code,
  });

  it("refuses fewer than 3 harvested futures, with a reason", () => {
    const r = buildCodeSpace([fut(0, [1, 0]), fut(1, [0, 1]), fut(2, [1, 1], false)]);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.placed).toBe(2);
      expect(r.reason).toMatch(/at least 3/);
    }
  });

  it("shows a 3-future fan flat (centred rank ≤ 2)", () => {
    const r = buildCodeSpace([fut(0, [1, 0, 0]), fut(1, [0, 1, 0]), fut(2, [0, 0, 1])]);
    expect(r.ok && r.space.dims).toBe(2);
    if (r.ok) for (const p of r.space.points) expect(p.pos[2]).toBe(0);
  });

  it("threads to the nearest fan-mate by FULL-space cosine and names the loner", () => {
    const r = buildCodeSpace([
      fut(0, [1, 0.1, 0, 0]),
      fut(1, [1, 0, 0.1, 0]),
      fut(2, [-1, 0, 0, 0.1]),
      fut(3, [0, 0, 0, -3]),
    ]);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const byIndex = new Map(r.space.points.map((p) => [p.index, p]));
    expect(byIndex.get(0)?.nn).toBe(1);
    expect(byIndex.get(1)?.nn).toBe(0);
    // mean-centred pulls average to 1 by construction
    const mean = r.space.points.reduce((s, p) => s + p.codePull, 0) / r.space.points.length;
    expect(mean).toBeCloseTo(1, 9);
    expect(r.space.points.every((p) => p.pullFrom === "code")).toBe(true);
  });

  it("uses the server's pull when the scores carry one", () => {
    const fs = [fut(0, [1, 0, 0]), fut(1, [0, 1, 0]), fut(2, [0, 0, 1]), fut(3, [1, 1, 1])];
    (fs[0] as Future).scores = { predicted_dose_scale: 1.7 };
    const r = buildCodeSpace(fs);
    expect(r.ok && r.space.points[0]?.pull).toBe(1.7);
    expect(r.ok && r.space.points[0]?.pullFrom).toBe("server");
  });

  it("projects the recorded k=16 rank-64 fan", () => {
    const loom = JSON.parse(
      readFileSync(new URL("../../fixtures/loom_k16.json", import.meta.url), "utf-8"),
    ) as LoomResponse;
    const r = buildCodeSpace(loom.futures);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.space.rank).toBe(64);
    expect(r.space.dims).toBe(3);
    expect(r.space.points).toHaveLength(loom.futures.filter((f) => f.harvested && f.code).length);
    expect(r.space.viewShare).toBeGreaterThan(0);
    expect(r.space.viewShare).toBeLessThanOrEqual(1 + 1e-9);
    // same fan, same picture: deterministic
    const again = buildCodeSpace(loom.futures);
    expect(again.ok && again.space.points.map((p) => p.pos)).toEqual(r.space.points.map((p) => p.pos));
    // cosine sanity
    expect(cosine([1, 0], [0, 1])).toBe(0);
  });
});
