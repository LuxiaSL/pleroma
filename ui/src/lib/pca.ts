/**
 * Principal-component projection for small point sets (a fan: k ≤ ~24 points
 * in rank-R code space, R = 64 on v1a).
 *
 * Ported from loom_ui.html's `jacobiEig` + `codeObjects`: centre the points,
 * eigendecompose the k×k GRAM matrix G = X Xᵀ (cyclic Jacobi — O(k³) per
 * sweep, nothing at k ≤ 24), and read coordinates as V[:,d]·√λ_d. That is
 * exactly the projection of each centred point onto the d-th principal
 * direction of the covariance XᵀX, without ever forming the R×R matrix.
 */

export interface Eigen {
  /** eigenvalues, in Jacobi's (unsorted) order */
  values: number[];
  /** eigenvectors as COLUMNS: vectors[row][col] */
  vectors: number[][];
}

/** Symmetric eigendecomposition by cyclic Jacobi rotations. Input is not modified. */
export function jacobiEig(A: readonly (readonly number[])[], maxSweeps = 80): Eigen {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const V: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n).fill(0);
    row[i] = 1;
    V.push(row);
  }
  const at = (m: number[][], i: number, j: number): number => (m[i] as number[])[j] as number;
  const set = (m: number[][], i: number, j: number, v: number): void => {
    (m[i] as number[])[j] = v;
  };
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += at(a, i, j) ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = at(a, p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const th = (at(a, q, q) - at(a, p, p)) / (2 * apq);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = at(a, k, p);
          const akq = at(a, k, q);
          set(a, k, p, c * akp - s * akq);
          set(a, k, q, s * akp + c * akq);
        }
        for (let k = 0; k < n; k++) {
          const apk = at(a, p, k);
          const aqk = at(a, q, k);
          set(a, p, k, c * apk - s * aqk);
          set(a, q, k, s * apk + c * aqk);
        }
        for (let k = 0; k < n; k++) {
          const vkp = at(V, k, p);
          const vkq = at(V, k, q);
          set(V, k, p, c * vkp - s * vkq);
          set(V, k, q, s * vkp + c * vkq);
        }
      }
    }
  }
  return { values: a.map((r, i) => r[i] as number), vectors: V };
}

export interface Projection {
  /** per point, its coordinates on the top `dims` principal directions */
  coords: number[][];
  /** the centred points (full space) */
  centred: number[][];
  /** Gram matrix of the centred points */
  gram: number[][];
  /** all eigenvalues of the Gram matrix, sorted descending, clipped at 0 */
  eigenvalues: number[];
  /** the share of total variance each returned dimension holds */
  share: number[];
  /** the column mean that was subtracted */
  mean: number[];
}

/**
 * Centre `points` (k × R) and project onto the top `dims` principal
 * directions. The basis sign is arbitrary; it is fixed deterministically
 * (positive third moment per axis) so re-rendering the SAME fan never mirrors.
 * Dimensions beyond the data's rank come back as zeros with share 0.
 */
export function pcaProject(points: readonly (readonly number[])[], dims: number): Projection {
  const k = points.length;
  if (k === 0) throw new Error("pcaProject: no points");
  const R = (points[0] as readonly number[]).length;
  if (points.some((p) => p.length !== R)) throw new Error("pcaProject: ragged points");
  if (points.some((p) => p.some((x) => !Number.isFinite(x))))
    throw new Error("pcaProject: non-finite coordinate");

  const mean = new Array<number>(R).fill(0);
  for (const p of points) for (let j = 0; j < R; j++) mean[j] = (mean[j] as number) + (p[j] as number) / k;
  const centred = points.map((p) => p.map((x, j) => x - (mean[j] as number)));
  const gram = centred.map((a) =>
    centred.map((b) => {
      let s = 0;
      for (let j = 0; j < R; j++) s += (a[j] as number) * (b[j] as number);
      return s;
    }),
  );
  const eg = jacobiEig(gram);
  const order = eg.values
    .map((_, i) => i)
    .sort((p, q) => (eg.values[q] as number) - (eg.values[p] as number));
  const eigenvalues = order.map((i) => Math.max(0, eg.values[i] as number));
  const total = eigenvalues.reduce((s, v) => s + v, 0);
  const coords = centred.map((_, r) =>
    Array.from({ length: dims }, (_unused, d) => {
      const col = order[d];
      if (col === undefined) return 0;
      const lam = eigenvalues[d] as number;
      return ((eg.vectors[r] as number[])[col] as number) * Math.sqrt(lam);
    }),
  );
  for (let d = 0; d < dims; d++) {
    let skew = 0;
    for (const c of coords) skew += (c[d] as number) ** 3;
    if (skew < 0) for (const c of coords) c[d] = -(c[d] as number);
  }
  const share = Array.from({ length: dims }, (_, d) => (total > 0 ? (eigenvalues[d] ?? 0) / total : 0));
  return { coords, centred, gram, eigenvalues, share, mean };
}

export function norm(v: readonly number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d > 1e-12 ? dot / d : 0;
}
