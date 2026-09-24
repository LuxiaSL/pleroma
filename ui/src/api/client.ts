/**
 * The thin, typed client for the loom's /api/v1 (docs/API.md).
 *
 * - base URL + bearer token from config (never a hardcoded host; "" = same origin)
 * - every non-2xx `{error, code}` becomes a typed ApiError
 * - every request has an AbortController deadline, and honours a caller's signal
 * - `connect()` reads /info and refuses a server that does not speak API v1
 */
import { ApiError, errorFromResponse } from "./errors";
import { API_PREFIX, API_VERSION, SCHEMA_SHA256 } from "./schema-meta";
import type {
  Branch,
  ChatResponse,
  InfoResponse,
  JsonBody,
  LoomResponse,
  OkJson,
  Ops,
  ProgressResponse,
  RestoreResponse,
  SessionsResponse,
  StateResponse,
  UnwearResponse,
  WearResponse,
} from "./types";

export interface ClientConfig {
  /** Origin (and optional path) of the loom server; "" means this page's origin. */
  baseUrl: string;
  /** Bearer token, when the server runs with --api-token / PLEROMA_API_TOKEN. */
  token: string | null;
  /** Default per-request deadline. */
  timeoutMs?: number;
  /** Injected for tests; defaults to the global fetch. */
  fetch?: typeof fetch;
}

export interface RequestOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

/** Deadlines, per the server's documented latencies. /loom is 30–120 s on the
 *  live 8B rig (k=16 measured at 28.4 s); the old UI allows 420 s. */
export const TIMEOUTS = {
  default: 15_000,
  chat: 180_000,
  loom: 420_000,
  wear: 60_000,
  progress: 10_000,
} as const;

export interface ConnectResult {
  info: InfoResponse;
  /** False when the server was built from a different (still v1) contract. */
  schemaMatches: boolean;
  serverSchemaSha: string;
}

export function joinUrl(baseUrl: string, path: string): string {
  const base = baseUrl.trim().replace(/\/+$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export class LoomClient {
  readonly baseUrl: string;
  private readonly token: string | null;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(cfg: ClientConfig) {
    this.baseUrl = cfg.baseUrl;
    this.token = cfg.token?.trim() ? cfg.token.trim() : null;
    this.timeoutMs = cfg.timeoutMs ?? TIMEOUTS.default;
    if (cfg.fetch) {
      this.fetchImpl = cfg.fetch;
    } else {
      if (typeof globalThis.fetch !== "function") throw new Error("no fetch implementation available");
      // a wrapper, so the global fetch is always called with the right `this`
      this.fetchImpl = (input, init) => globalThis.fetch(input, init);
    }
  }

  /** The low-level call. `path` is the route WITHOUT the /api/v1 prefix. */
  async request<T>(
    method: "GET" | "POST",
    path: string,
    opts: RequestOptions & { query?: Record<string, string>; body?: unknown } = {},
  ): Promise<T> {
    const qs = opts.query ? `?${new URLSearchParams(opts.query).toString()}` : "";
    const url = joinUrl(this.baseUrl, `${API_PREFIX}${path}${qs}`);
    const route = `${method} ${API_PREFIX}${path}`;
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    if (opts.body !== undefined) headers["Content-Type"] = "application/json";

    const timeoutMs = opts.timeoutMs ?? this.timeoutMs;
    const ctl = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctl.abort();
    }, timeoutMs);
    const onCallerAbort = () => ctl.abort();
    if (opts.signal) {
      if (opts.signal.aborted) ctl.abort();
      else opts.signal.addEventListener("abort", onCallerAbort, { once: true });
    }

    let res: Response;
    try {
      const init: RequestInit = { method, headers, signal: ctl.signal };
      if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
      res = await this.fetchImpl(url, init);
    } catch (cause) {
      if (timedOut) {
        throw new ApiError({
          kind: "timeout",
          route,
          cause,
          message: `${route} gave no answer within ${Math.round(timeoutMs / 1000)} s — the request was abandoned (the server may still finish it)`,
        });
      }
      if (ctl.signal.aborted) {
        throw new ApiError({ kind: "aborted", route, cause, message: `${route} was cancelled` });
      }
      const why = cause instanceof Error ? cause.message : String(cause);
      throw new ApiError({
        kind: "network",
        route,
        cause,
        message: `${route} could not reach the server (${why})`,
      });
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onCallerAbort);
    }

    let text: string;
    try {
      text = await res.text();
    } catch (cause) {
      throw new ApiError({
        kind: "network",
        route,
        status: res.status,
        cause,
        message: `${route}: the connection dropped while reading the answer`,
      });
    }
    if (!res.ok) throw errorFromResponse(res.status, res.statusText, text, route);
    try {
      return JSON.parse(text) as T;
    } catch (cause) {
      throw new ApiError({
        kind: "bad_response",
        route,
        status: res.status,
        cause,
        message: `${route} answered ${res.status} with a body that is not JSON (is this a loom server?)`,
      });
    }
  }

  // ── typed routes ────────────────────────────────────────────────────────

  info(o?: RequestOptions): Promise<InfoResponse> {
    return this.request<OkJson<Ops["getInfo"]>>("GET", "/info", { ...o });
  }
  sessions(o?: RequestOptions): Promise<SessionsResponse> {
    return this.request<OkJson<Ops["getSessions"]>>("GET", "/sessions", { ...o });
  }
  /** v1 never creates the session: an unknown one is 404 `no_session`. */
  state(session: string, o?: RequestOptions): Promise<StateResponse> {
    return this.request<OkJson<Ops["getState"]>>("GET", "/state", { ...o, query: { session } });
  }
  progress(session: string, o?: RequestOptions): Promise<ProgressResponse> {
    return this.request<OkJson<Ops["getLoomProgress"]>>("GET", "/loom/progress", {
      timeoutMs: TIMEOUTS.progress,
      ...o,
      query: { session },
    });
  }
  chat(body: JsonBody<Ops["postChat"]> & { branch: Branch }, o?: RequestOptions): Promise<ChatResponse> {
    return this.request<OkJson<Ops["postChat"]>>("POST", "/chat", { timeoutMs: TIMEOUTS.chat, ...o, body });
  }
  loom(body: JsonBody<Ops["postLoom"]>, o?: RequestOptions): Promise<LoomResponse> {
    return this.request<OkJson<Ops["postLoom"]>>("POST", "/loom", { timeoutMs: TIMEOUTS.loom, ...o, body });
  }
  wear(body: JsonBody<Ops["postWear"]>, o?: RequestOptions): Promise<WearResponse> {
    return this.request<OkJson<Ops["postWear"]>>("POST", "/wear", { timeoutMs: TIMEOUTS.wear, ...o, body });
  }
  unwear(session: string, o?: RequestOptions): Promise<UnwearResponse> {
    return this.request<OkJson<Ops["postUnwear"]>>("POST", "/unwear", {
      timeoutMs: TIMEOUTS.wear,
      ...o,
      body: { session },
    });
  }
  restore(session: string, o?: RequestOptions): Promise<RestoreResponse> {
    return this.request<OkJson<Ops["postRestore"]>>("POST", "/restore", { ...o, body: { session } });
  }

  /**
   * Read /info and check the API version. Throws `incompatible` for a server
   * that predates /api/v1 or speaks a different major version; returns
   * `schemaMatches: false` (not an error — v1 is additive) when the server was
   * built from a different contract than this client was generated from.
   */
  async connect(o?: RequestOptions): Promise<ConnectResult> {
    let info: InfoResponse;
    try {
      info = await this.info(o);
    } catch (err) {
      if (err instanceof ApiError && err.kind === "server" && err.status === 404) {
        throw new ApiError({
          kind: "incompatible",
          status: 404,
          route: err.route,
          rawCode: err.rawCode,
          message:
            `${this.describeTarget()} has no ${API_PREFIX}/info — it is a server without the versioned ` +
            "API, or not a loom server. This UI needs /api/v1; the single-file loom UI works without it.",
        });
      }
      throw err;
    }
    const api = (info as Partial<InfoResponse>).api;
    if (!api || typeof api !== "object") {
      throw new ApiError({
        kind: "incompatible",
        route: `GET ${API_PREFIX}/info`,
        message: `${this.describeTarget()} answered /info without an \`api\` block — not a /api/v1 server.`,
      });
    }
    if (String(api.version) !== API_VERSION) {
      throw new ApiError({
        kind: "incompatible",
        route: `GET ${API_PREFIX}/info`,
        message:
          `incompatible server: it speaks API v${String(api.version)}, this UI was built for v${API_VERSION}. ` +
          "Rebuild the UI from the server's api-schema.json (npm run gen:api) or run a matching server.",
      });
    }
    return { info, schemaMatches: api.schema_sha256 === SCHEMA_SHA256, serverSchemaSha: api.schema_sha256 };
  }

  private describeTarget(): string {
    return this.baseUrl ? this.baseUrl : "this page's server";
  }
}
