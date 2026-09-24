import { describe, expect, it } from "vitest";
import { joinUrl, LoomClient } from "../../src/api/client";
import { ApiError, describeError, ERROR_CODES, errorFromResponse } from "../../src/api/errors";
import { SCHEMA_SHA256 } from "../../src/api/schema-meta";
import type { InfoResponse } from "../../src/api/types";
import { fastBackend, json, mockFetch } from "./helpers";

type Call = { url: string; init: RequestInit | undefined };

function client(
  respond: (url: string, init?: RequestInit) => Response | Promise<Response>,
  token: string | null = null,
) {
  const calls: Call[] = [];
  const c = new LoomClient({
    baseUrl: "http://loom.test:8767/",
    token,
    fetch: async (input, init) => {
      calls.push({ url: String(input), init });
      return respond(String(input), init);
    },
  });
  return { c, calls };
}

async function caught(p: Promise<unknown>): Promise<ApiError> {
  try {
    await p;
  } catch (err) {
    expect(err).toBeInstanceOf(ApiError);
    return err as ApiError;
  }
  throw new Error("expected a rejection");
}

describe("ApiError mapping", () => {
  it("maps {error, code} to a typed error, message verbatim", async () => {
    const sentence =
      "loom_id 'a-000' is not the latest draw ('b-001') — another tab or turn superseded it; /loom again";
    const { c } = client(() => json(400, { error: sentence, code: "stale_loom" }));
    const e = await caught(c.wear({ session: "s", index: 1, alpha: 0.5, loom_id: "a-000" }));
    expect(e.kind).toBe("server");
    expect(e.status).toBe(400);
    expect(e.code).toBe("stale_loom");
    expect(e.is("stale_loom")).toBe(true);
    expect(e.message).toBe(sentence);
    expect(e.route).toBe("POST /api/v1/wear");
  });

  it("keeps an unknown code raw and leaves `code` null (treat it as its status)", async () => {
    const { c } = client(() => json(409, { error: "new refusal", code: "some_future_code" }));
    const e = await caught(c.info());
    expect(e.code).toBeNull();
    expect(e.rawCode).toBe("some_future_code");
    expect(e.status).toBe(409);
  });

  it("describes a non-JSON error body by its status line", () => {
    const e = errorFromResponse(
      502,
      "Bad Gateway",
      "<html><body>upstream down</body></html>",
      "GET /api/v1/info",
    );
    expect(e.message).toContain("HTTP 502 Bad Gateway");
    expect(e.message).toContain("upstream down");
    expect(e.code).toBeNull();
  });

  it("network failure → kind network", async () => {
    const { c } = client(() => {
      throw new TypeError("Failed to fetch");
    });
    const e = await caught(c.sessions());
    expect(e.kind).toBe("network");
    expect(e.status).toBe(0);
    expect(describeError(e)).toContain("--cors-origin");
  });

  it("deadline → kind timeout", async () => {
    const c = new LoomClient({
      baseUrl: "",
      token: null,
      timeoutMs: 20,
      fetch: (_i, init) =>
        new Promise((_r, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    });
    const e = await caught(c.info());
    expect(e.kind).toBe("timeout");
    expect(e.message).toContain("no answer within");
  });

  it("caller abort → kind aborted", async () => {
    const ctl = new AbortController();
    const c = new LoomClient({
      baseUrl: "",
      token: null,
      fetch: (_i, init) =>
        new Promise((_r, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    });
    const p = c.loom({ session: "s", text: "t" }, { signal: ctl.signal });
    ctl.abort();
    expect((await caught(p)).kind).toBe("aborted");
  });

  it("2xx that is not JSON → kind bad_response", async () => {
    const { c } = client(() => new Response("<!doctype html>", { status: 200 }));
    expect((await caught(c.info())).kind).toBe("bad_response");
  });

  it("401 unauthorized hints at the token", async () => {
    const { c } = client(() => json(401, { error: "missing or invalid bearer token", code: "unauthorized" }));
    const e = await caught(c.sessions());
    expect(e.is("unauthorized")).toBe(true);
    expect(describeError(e)).toContain("token");
  });

  it("ERROR_CODES has no duplicates", () => {
    expect(new Set(ERROR_CODES).size).toBe(ERROR_CODES.length);
  });
});

describe("requests", () => {
  it("joins base + prefix + route, encodes the query, sends the bearer token", async () => {
    const { c, calls } = client(() => json(200, { session: "a b", progress: null }), "tok123");
    await c.progress("a b");
    expect(calls[0]?.url).toBe("http://loom.test:8767/api/v1/loom/progress?session=a+b");
    const h = calls[0]?.init?.headers as Record<string, string>;
    expect(h.Authorization).toBe("Bearer tok123");
    expect(h["Content-Type"]).toBeUndefined();
  });

  it("sends no Authorization header without a token; JSON body on POST", async () => {
    const { c, calls } = client(() => json(200, { session: "s", worn: null }));
    await c.unwear("s");
    const h = calls[0]?.init?.headers as Record<string, string>;
    expect(h.Authorization).toBeUndefined();
    expect(h["Content-Type"]).toBe("application/json");
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({ session: "s" });
  });

  it("same-origin by default", () => {
    expect(joinUrl("", "/api/v1/info")).toBe("/api/v1/info");
    expect(joinUrl("http://h:1///", "/x")).toBe("http://h:1/x");
  });
});

describe("connect()", () => {
  const backend = fastBackend();
  const info = (patch: Partial<InfoResponse["api"]>): InfoResponse => ({
    ...backend.info,
    api: { ...backend.info.api, ...patch },
  });

  it("accepts a v1 server and reports schema agreement", async () => {
    const c = new LoomClient({ baseUrl: "", token: null, fetch: mockFetch(backend) });
    const r = await c.connect();
    expect(r.info.api.version).toBe("1");
    expect(r.schemaMatches).toBe(backend.info.api.schema_sha256 === SCHEMA_SHA256);
  });

  it("flags a different contract without refusing it (v1 is additive)", async () => {
    const { c } = client(() => json(200, info({ schema_sha256: "0".repeat(64) })));
    const r = await c.connect();
    expect(r.schemaMatches).toBe(false);
  });

  it("refuses another major version with a clear message", async () => {
    const { c } = client(() => json(200, info({ version: "2" as "1" })));
    const e = await caught(c.connect());
    expect(e.kind).toBe("incompatible");
    expect(e.message).toMatch(/speaks API v2, this UI was built for v1/);
  });

  it("refuses a server with no /api/v1 (404) as incompatible", async () => {
    const { c } = client(() => json(404, { error: "unknown path", code: "not_found" }));
    const e = await caught(c.connect());
    expect(e.kind).toBe("incompatible");
    expect(e.message).toMatch(/without the versioned/);
  });

  it("refuses an /info with no api block", async () => {
    const { api: _drop, ...legacy } = backend.info;
    const { c } = client(() => json(200, legacy));
    expect((await caught(c.connect())).kind).toBe("incompatible");
  });
});
