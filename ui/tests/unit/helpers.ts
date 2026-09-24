import { MockBackend, type MockOptions } from "../../mock/backend";

/** A fetch that answers from a MockBackend in-process (no HTTP), with the
 *  same status/body/code semantics as its middleware. */
export function mockFetch(backend: MockBackend): typeof fetch {
  return async (input, init) => {
    const url = new URL(String(input), "http://mock.invalid");
    const prefix = "/api/v1";
    const method = init?.method ?? "GET";
    const signal = init?.signal;
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const token = backend.opts.token;
    const headers = (init?.headers ?? {}) as Record<string, string>;
    if (token && headers.Authorization !== `Bearer ${token}`) {
      return json(401, { error: "missing or invalid bearer token", code: "unauthorized" });
    }
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
    const run = backend.handle(method, url.pathname.slice(prefix.length), url.searchParams, body);
    const aborted = new Promise<never>((_, reject) => {
      signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {
        once: true,
      });
    });
    try {
      return json(200, await Promise.race([run, aborted]));
    } catch (err) {
      if (err instanceof DOMException) throw err;
      const e = err as { status?: number; code?: string; message?: string };
      if (typeof e.status === "number") return json(e.status, { error: e.message, code: e.code });
      return json(500, { error: String(err), code: "internal" });
    }
  };
}

export function json(status: number, obj: unknown): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
}

export function fastBackend(opts: MockOptions = {}): MockBackend {
  return new MockBackend({ loomSeconds: 0.12, chatMs: 0, ...opts });
}

export async function until(pred: () => boolean, ms = 3000): Promise<void> {
  const t0 = Date.now();
  while (!pred()) {
    if (Date.now() - t0 > ms) throw new Error("until: timed out");
    await new Promise((r) => setTimeout(r, 5));
  }
}
