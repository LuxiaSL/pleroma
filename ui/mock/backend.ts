/**
 * A mock /api/v1 for running the UI with no GPU and no loom server.
 *
 * Serves ui/fixtures/ — RECORDED live responses (the 8B v1a loom, rank-64
 * map, measured dose band) converted to the v1 contract by
 * scripts/build_fixtures.py — behind a small stateful session model, so the
 * whole slice is drivable: sessions, /state, /chat, /loom with a replayed
 * /loom/progress, /wear, /unwear, /restore.
 *
 * What is REAL here: every future's text, code, and scores; the dose band;
 * /info. What is NOT: chat replies (canned, and say so), which futures a new
 * draw returns (the recorded fan, re-indexed and cut to k), and timing (the
 * progress SHAPE is the server's — generate 0→k in one jump, then harvest
 * branch by branch — but the durations are illustrative).
 *
 * Error bodies are `{error, code}` with the server's own codes, and the
 * refusals the slice can hit (stale_loom, no_candidates, no_session,
 * session_required, unauthorized) use the server's semantics.
 */
import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type {
  ChatResponse,
  Future,
  HistoryMessage,
  InfoResponse,
  LoomProgress,
  LoomResponse,
  ProgressResponse,
  RestoreResponse,
  SessionRow,
  SessionsResponse,
  StateResponse,
  UnwearResponse,
  WearResponse,
  WornPublic,
} from "../src/api/types.ts";

const FIXTURES = resolve(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

function fixture<T>(name: string): T {
  return JSON.parse(readFileSync(resolve(FIXTURES, name), "utf-8")) as T;
}

export interface MockOptions {
  /** how long a /loom takes, seconds (the progress replay is spread over it) */
  loomSeconds?: number;
  /** /chat latency, ms */
  chatMs?: number;
  /** when set, /api/v1 requires `Authorization: Bearer <token>` */
  token?: string | null;
}

interface MockSession {
  name: string;
  histories: { loom: HistoryMessage[]; base: HistoryMessage[] };
  worn: WornPublic | null;
  loomId: string | null;
  futures: Future[];
  nLooms: number;
  /** the recorded fan new draws are cut from */
  source: LoomResponse;
  progress: { t0: number; timeline: [number, LoomProgress][] } | null;
  modified: number;
  inMemory: boolean;
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

const clone = <T>(v: T): T => structuredClone(v);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function hex(n: number): string {
  let s = "";
  for (let i = 0; i < n; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

/** The server's progress SHAPE (ui_proto/fixture_server.py SlowLoom, read off loom_serve). */
function progressTimeline(seconds: number, k: number): [number, LoomProgress][] {
  const gen = 0.18 * seconds;
  const har = 0.82 * seconds;
  const order = Array.from({ length: k }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = (i * 7919 + 13) % (i + 1);
    [order[i], order[j]] = [order[j] as number, order[i] as number];
  }
  const tl: [number, LoomProgress][] = [
    [0, { stage: "generate", done: 0, total: k }],
    [gen, { stage: "harvest", done: 0, total: 2, gens: { n_gens: k, harvested: [] } }],
  ];
  for (let j = 0; j < k; j++) {
    tl.push([
      gen + (har * (j + 1)) / (k + 1),
      { stage: "harvest", done: 0, total: 2, gens: { n_gens: k, harvested: order.slice(0, j + 1) } },
    ]);
  }
  tl.push([seconds, { stage: "harvest", done: 2, total: 2, gens: { n_gens: k, harvested: order } }]);
  return tl;
}

export class MockBackend {
  readonly info: InfoResponse;
  readonly sessions = new Map<string, MockSession>();
  private readonly wearTemplate: WearResponse;
  readonly opts: Required<MockOptions>;

  constructor(opts: MockOptions = {}) {
    this.opts = { loomSeconds: opts.loomSeconds ?? 4, chatMs: opts.chatMs ?? 350, token: opts.token ?? null };
    this.info = fixture<InfoResponse>("info.json");
    this.info.api = { ...this.info.api, auth_required: Boolean(this.opts.token) };
    this.wearTemplate = fixture<WearResponse>("wear.json");
    this.reset();
  }

  /** Back to the three recorded sessions (the e2e calls this between tests). */
  reset(): void {
    this.sessions.clear();
    const k16 = fixture<LoomResponse>("loom_k16.json");
    const worn = fixture<LoomResponse>("loom_worn.json");
    this.seed("lx-speak-k16", fixture<StateResponse>("state_drawn.json"), k16, true);
    this.seed("uifix-phaseb", fixture<StateResponse>("state_worn.json"), worn, true);
    // on disk only: exercises the picker's "restore" path
    this.seed("archived-8b-demo", fixture<StateResponse>("state_worn.json"), worn, false);
  }

  private seed(name: string, st: StateResponse, source: LoomResponse, inMemory: boolean): void {
    this.sessions.set(name, {
      name,
      histories: { loom: clone(st.histories.loom ?? []), base: clone(st.histories.base ?? []) },
      worn: clone(st.worn),
      loomId: st.loom_id,
      futures: clone(st.futures),
      nLooms: st.n_looms,
      source,
      progress: null,
      modified: Date.now() / 1000 - (inMemory ? 60 : 86_400),
      inMemory,
    });
  }

  private live(name: string): MockSession | undefined {
    const s = this.sessions.get(name);
    return s?.inMemory ? s : undefined;
  }

  /** POSTs create the session, as the real server does. */
  private touch(name: string): MockSession {
    let s = this.live(name);
    if (!s) {
      if (this.sessions.has(name)) {
        // an on-disk session written to without /restore: the server moves the
        // old snapshot aside and starts fresh — mirror that, simply
        this.sessions.delete(name);
      }
      const src = fixture<LoomResponse>("loom_k16.json");
      s = {
        name,
        histories: { loom: [], base: [] },
        worn: null,
        loomId: null,
        futures: [],
        nLooms: 0,
        source: src,
        progress: null,
        modified: Date.now() / 1000,
        inMemory: true,
      };
      this.sessions.set(name, s);
    }
    s.modified = Date.now() / 1000;
    return s;
  }

  private stateOf(s: MockSession): StateResponse {
    return {
      session: s.name,
      histories: clone(s.histories),
      worn: clone(s.worn),
      loom_id: s.loomId,
      futures: clone(s.futures),
      n_looms: s.nLooms,
      loom_in_progress: this.progressNow(s),
      probe: null,
      probe_note: null,
    };
  }

  private progressNow(s: MockSession): LoomProgress | null {
    if (!s.progress) return null;
    const el = (Date.now() - s.progress.t0) / 1000;
    let cur: LoomProgress | null = null;
    for (const [at, p] of s.progress.timeline) if (at <= el) cur = p;
    return cur ? clone(cur) : null;
  }

  private row(s: MockSession): SessionRow {
    return {
      session: s.name,
      file: `sessions/${s.name}.json`,
      ok: true,
      in_memory: s.inMemory,
      n_turns: {
        loom: s.histories.loom.filter((m) => m.role === "user").length,
        base: s.histories.base.filter((m) => m.role === "user").length,
      },
      n_messages: s.histories.loom.length + s.histories.base.length,
      n_looms: s.nLooms,
      loom_id: s.loomId,
      worn: clone(s.worn),
      modified_unix: s.modified,
      modified_iso: new Date(s.modified * 1000).toISOString(),
      size_bytes: JSON.stringify(s.histories).length + JSON.stringify(s.futures).length,
      error: null,
    };
  }

  // ── routes ──────────────────────────────────────────────────────────────

  async handle(
    method: string,
    path: string,
    query: URLSearchParams,
    body: Record<string, unknown>,
  ): Promise<unknown> {
    if (method === "GET") {
      switch (path) {
        case "/info":
          return this.info;
        case "/sessions": {
          const rows = [...this.sessions.values()].map((s) => this.row(s));
          const r: SessionsResponse = {
            sessions: rows,
            n_sessions: rows.length,
            persistence: {
              enabled: true,
              dir: "mock/sessions",
              writes: 0,
              write_failures: 0,
              restored_sessions: 2,
              failed_sessions: 0,
              last_error: null,
            },
          };
          return r;
        }
        case "/state": {
          const name = query.get("session") ?? "";
          if (!name) throw new HttpError(400, "session_required", "session required");
          const s = this.live(name);
          if (!s) {
            throw new HttpError(
              404,
              "no_session",
              `no session '${name}' (GET /api/v1/state never creates one — POST to it first)`,
            );
          }
          return this.stateOf(s);
        }
        case "/loom/progress": {
          const name = query.get("session") ?? "";
          const s = name ? this.live(name) : undefined;
          const r: ProgressResponse = { session: name, progress: s ? this.progressNow(s) : null };
          return r;
        }
        default:
          throw new HttpError(404, "not_found", `unknown path ${path}`);
      }
    }
    if (method !== "POST") throw new HttpError(404, "not_found", `unknown path ${path}`);
    const name = typeof body.session === "string" ? body.session.trim() : "";
    if (!name) throw new HttpError(400, "session_required", "session required");
    switch (path) {
      case "/chat":
        return this.chat(name, body);
      case "/loom":
        return this.loom(name, body);
      case "/wear":
        return this.wear(name, body);
      case "/unwear": {
        const s = this.touch(name);
        s.worn = null;
        const r: UnwearResponse = { session: name, worn: null };
        return r;
      }
      case "/restore":
        return this.restore(name);
      default:
        throw new HttpError(404, "not_found", `unknown path ${path}`);
    }
  }

  private async chat(name: string, body: Record<string, unknown>): Promise<ChatResponse> {
    const branch = body.branch ?? "loom";
    if (branch !== "loom" && branch !== "base") {
      throw new HttpError(400, "unknown_branch", `unknown branch ${JSON.stringify(branch)}`);
    }
    const text = typeof body.text === "string" ? body.text : "";
    const s = this.touch(name);
    const t0 = Date.now();
    await sleep(this.opts.chatMs);
    const bent =
      branch === "loom" && s.worn ? ` under code #${s.worn.index} at α ${s.worn.alpha.toFixed(3)}` : "";
    const reply =
      `[mock backend — no model ran] A canned ${branch}-branch reply${bent} to: “${text.slice(0, 120)}”. ` +
      "Point the UI at a live loom server to get a real one.";
    s.histories[branch].push({ role: "user", content: text }, { role: "assistant", content: reply });
    return {
      session: name,
      branch,
      reply,
      n_turns: s.histories[branch].filter((m) => m.role === "user").length,
      dropped_messages: 0,
      worn: branch === "loom" ? clone(s.worn) : null,
      elapsed_s: Math.round((Date.now() - t0) / 10) / 100,
    };
  }

  private async loom(name: string, body: Record<string, unknown>): Promise<LoomResponse> {
    if (typeof body.text !== "string") throw new HttpError(400, "bad_request", "'text'");
    const s = this.touch(name);
    const src = s.source;
    const kReq = body.k === undefined || body.k === null ? this.info.default_k : Number(body.k);
    if (!Number.isInteger(kReq) || kReq < 2)
      throw new HttpError(400, "bad_request", `k must be an integer ≥ 2, got ${String(body.k)}`);
    const k = Math.min(kReq, src.futures.length);
    const seconds = this.opts.loomSeconds;
    s.progress = { t0: Date.now(), timeline: progressTimeline(seconds, k) };
    const detach = body.detach_wear === true;
    const drawnUnder =
      s.worn && !detach ? { index: s.worn.index, alpha: s.worn.alpha, loom_id: s.worn.loom_id } : null;
    try {
      await sleep(seconds * 1000 + 150);
    } finally {
      s.progress = null;
    }
    const futures = clone(src.futures.slice(0, k)).map((f, i) => ({ ...f, index: i }));
    s.nLooms += 1;
    s.loomId = `${hex(10)}-${String(s.nLooms - 1).padStart(3, "0")}`;
    s.futures = futures;
    const horizon = body.horizon === undefined || body.horizon === null ? src.horizon : Number(body.horizon);
    return {
      ...clone(src),
      session: name,
      loom_id: s.loomId,
      n_futures: k,
      horizon,
      detach_wear: detach,
      drawn_under_wear: drawnUnder,
      futures,
      worn: clone(s.worn),
      timing_s: { generate: +(0.18 * seconds).toFixed(2), harvest: +(0.82 * seconds).toFixed(2) },
      harvest_via: "mock",
      harvest_worker_detail: null,
      loom_dir: `mock/looms/${s.loomId}`,
      auto_selected: null,
      probe: null,
    };
  }

  private wear(name: string, body: Record<string, unknown>): WearResponse {
    const s = this.touch(name);
    if (!s.futures.length || !s.loomId) {
      throw new HttpError(400, "no_candidates", "no candidates — /loom first");
    }
    const loomId = body.loom_id;
    if (typeof loomId === "string" && loomId !== s.loomId) {
      throw new HttpError(
        400,
        "stale_loom",
        `loom_id '${loomId}' is not the latest draw ('${s.loomId}') — another tab or turn superseded it; /loom again`,
      );
    }
    const index = Number(body.index);
    const f = s.futures.find((x) => x.index === index);
    if (!f)
      throw new HttpError(400, "bad_request", `'index' ${String(body.index)} is not a future of this draw`);
    if (!f.harvested || !f.code) {
      throw new HttpError(400, "bad_request", `future ${index} was not harvested — nothing to wear`);
    }
    const alpha = body.alpha === undefined ? 0.5 : Number(body.alpha);
    if (!Number.isFinite(alpha) || alpha < 0)
      throw new HttpError(400, "bad_request", `alpha must be ≥ 0, got ${String(body.alpha)}`);
    const t = clone(this.wearTemplate);
    // the recorded receipt belongs to another future: rescale its per-site
    // shape so its mean is THIS future's predicted_dose_scale (what the
    // server's `predicted` policy would apply), keeping card and receipt consistent
    const pds = f.scores.predicted_dose_scale;
    if (t.dose.policy === "predicted" && typeof pds === "number" && pds > 0 && t.dose.scale_mean > 0) {
      const r = pds / t.dose.scale_mean;
      t.dose.scale = t.dose.scale.map((x) => +(x * r).toFixed(6));
      t.dose.scale_raw = t.dose.scale_raw.map((x) => +(x * r).toFixed(6));
      t.dose.scale_mean = pds;
      t.dose.clamped = [];
    }
    const scale = t.dose.scale;
    const worn: WornPublic = {
      ...t.worn,
      dose: clone(t.dose),
      index,
      alpha,
      loom_id: s.loomId,
      code: clone(f.code),
      active: true,
      lever_kind: t.lever_kind,
    };
    s.worn = worn;
    const per = scale.map((x) => +(alpha * x).toFixed(6));
    return {
      ...t,
      session: name,
      worn,
      effective_alpha: {
        ...t.effective_alpha,
        alpha,
        alpha_effective_mean: +(alpha * t.dose.scale_mean).toFixed(6),
        alpha_effective_per_site: per,
        alpha_effective_range: [Math.min(...per), Math.max(...per)],
      },
    };
  }

  private restore(name: string): RestoreResponse {
    const s = this.sessions.get(name);
    if (!s) throw new HttpError(400, "no_snapshot", `no readable snapshot for session '${name}'`);
    s.inMemory = true;
    return {
      session: name,
      restored: true,
      file: `sessions/${name}.json`,
      n_turns: {
        loom: s.histories.loom.filter((m) => m.role === "user").length,
        base: s.histories.base.filter((m) => m.role === "user").length,
      },
      n_looms: s.nLooms,
      loom_id: s.loomId,
      worn: clone(s.worn),
      futures: clone(s.futures),
    };
  }

  // ── the connect middleware ──────────────────────────────────────────────

  middleware(prefix = "/api/v1") {
    return (req: IncomingMessage, res: ServerResponse, next: () => void): void => {
      const url = new URL(req.url ?? "/", "http://mock.invalid");
      if (url.pathname === "/__mock/reset" && req.method === "POST") {
        this.reset();
        res.statusCode = 204;
        res.end();
        return;
      }
      if (!url.pathname.startsWith(`${prefix}/`)) {
        next();
        return;
      }
      const path = url.pathname.slice(prefix.length);
      const send = (status: number, obj: unknown) => {
        const text = JSON.stringify(obj);
        res.statusCode = status;
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-store");
        res.end(text);
      };
      const token = this.opts.token;
      if (token && req.headers.authorization !== `Bearer ${token}`) {
        res.setHeader("WWW-Authenticate", "Bearer");
        send(401, { error: "missing or invalid bearer token", code: "unauthorized" });
        return;
      }
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        let body: Record<string, unknown> = {};
        if (chunks.length) {
          try {
            const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
              body = parsed as Record<string, unknown>;
          } catch {
            send(400, { error: "request body is not JSON", code: "bad_request" });
            return;
          }
        }
        this.handle(req.method ?? "GET", path, url.searchParams, body)
          .then((r) => send(200, r))
          .catch((err: unknown) => {
            if (err instanceof HttpError) send(err.status, { error: err.message, code: err.code });
            else
              send(500, {
                error: `${err instanceof Error ? err.name : "Error"}: ${String(err)}`,
                code: "internal",
              });
          });
      });
    };
  }
}
