/**
 * The mock backend must speak the SAME contract as the server: every body it
 * sends is validated here against pleroma/serve/api-schema.json (OpenAPI 3.1
 * = JSON Schema 2020-12), and every error body against ErrorBody. A mock that
 * drifts from the schema would let the UI grow against shapes no server sends.
 * (The Python suite separately checks ui/fixtures/ against the pydantic models.)
 */
import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";
import { fastBackend } from "./helpers";

const schemaDoc = JSON.parse(
  readFileSync(new URL("../../../pleroma/serve/api-schema.json", import.meta.url), "utf-8"),
) as Record<string, unknown>;

const ajv = new Ajv2020({ strict: false, allErrors: true });
ajv.addSchema({ ...schemaDoc, $id: "loom-api" });

function check(model: string, body: unknown): void {
  const validate = ajv.getSchema(`loom-api#/components/schemas/${model}`);
  if (!validate) throw new Error(`no schema ${model}`);
  const ok = validate(JSON.parse(JSON.stringify(body)));
  if (!ok) throw new Error(`${model}: ${ajv.errorsText(validate.errors, { separator: "\n" })}`);
}

async function refusal(p: Promise<unknown>): Promise<{ status: number; body: unknown }> {
  try {
    await p;
  } catch (err) {
    const e = err as { status: number; code: string; message: string };
    return { status: e.status, body: { error: e.message, code: e.code } };
  }
  throw new Error("expected a refusal");
}

describe("mock backend ⊂ /api/v1 contract", () => {
  it("every success body fits its response model", async () => {
    const b = fastBackend();
    const q = (s: string) => new URLSearchParams({ session: s });
    check("InfoResponse", await b.handle("GET", "/info", q(""), {}));
    check("SessionsResponse", await b.handle("GET", "/sessions", q(""), {}));
    check("StateResponse", await b.handle("GET", "/state", q("uifix-phaseb"), {}));
    check("StateResponse", await b.handle("GET", "/state", q("lx-speak-k16"), {}));
    check("ChatResponse", await b.handle("POST", "/chat", q(""), { session: "n", text: "hi" }));
    const loom = b.handle("POST", "/loom", q(""), { session: "n", text: "hi", k: 5 });
    await new Promise((r) => setTimeout(r, 60));
    check("ProgressResponse", await b.handle("GET", "/loom/progress", q("n"), {}));
    const drawn = (await loom) as { loom_id: string };
    check("LoomResponse", drawn);
    check(
      "WearResponse",
      await b.handle("POST", "/wear", q(""), { session: "n", index: 1, alpha: 0.3, loom_id: drawn.loom_id }),
    );
    check("StateResponse", await b.handle("GET", "/state", q("n"), {}));
    check("UnwearResponse", await b.handle("POST", "/unwear", q(""), { session: "n" }));
    check("RestoreResponse", await b.handle("POST", "/restore", q(""), { session: "archived-8b-demo" }));
    check("ProgressResponse", await b.handle("GET", "/loom/progress", q("nobody"), {}));
  });

  it("every refusal is an ErrorBody with the server's code and status", async () => {
    const b = fastBackend();
    const q = new URLSearchParams({ session: "nobody" });
    const cases: [Promise<unknown>, number, string][] = [
      [b.handle("GET", "/state", q, {}), 404, "no_session"],
      [b.handle("POST", "/chat", q, {}), 400, "session_required"],
      [b.handle("POST", "/wear", q, { session: "fresh", index: 0 }), 400, "no_candidates"],
      [b.handle("POST", "/chat", q, { session: "s", branch: "sideways", text: "x" }), 400, "unknown_branch"],
      [b.handle("POST", "/restore", q, { session: "never" }), 400, "no_snapshot"],
      [b.handle("GET", "/nope", q, {}), 404, "not_found"],
    ];
    for (const [p, status, code] of cases) {
      const r = await refusal(p);
      expect(r.status).toBe(status);
      check("ErrorBody", r.body);
      expect((r.body as { code: string }).code).toBe(code);
    }
  });
});
