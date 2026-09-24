/**
 * Every failure the client can produce, as ONE typed error.
 *
 * The server's non-2xx body is `{error, code}` (docs/API.md "Errors"): `error`
 * is written for the operator and is shown VERBATIM; `code` is the stable,
 * machine-readable reason. The versioning policy says a client must treat an
 * UNKNOWN code as its HTTP status — so `code` is only set when it is one this
 * client was generated with, and `rawCode` keeps whatever the server said.
 */
import type { ErrorCode } from "./types";

/** Every code in the generated `ErrorCode` enum, at runtime. The type-level
 *  check below fails `npm run typecheck` when the schema grows a code this
 *  list lacks — the python suite runs typecheck, so schema drift is caught. */
export const ERROR_CODES = [
  "bad_request",
  "not_found",
  "session_required",
  "no_session",
  "unknown_branch",
  "stale_loom",
  "no_candidates",
  "gauge_needs_probe",
  "lever_kind_invalid",
  "lever_kind_unavailable",
  "dose_policy_invalid",
  "dose_policy_unavailable",
  "probe_prefix_mismatch",
  "no_snapshot",
  "unauthorized",
  "forbidden_origin",
  "internal",
] as const satisfies readonly ErrorCode[];

type Missing = Exclude<ErrorCode, (typeof ERROR_CODES)[number]>;
// If this line errors, the schema gained an ErrorCode: add it to ERROR_CODES.
const _exhaustive: [Missing] extends [never] ? true : { missing: Missing } = true;
void _exhaustive;

const KNOWN = new Set<string>(ERROR_CODES);

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && KNOWN.has(value);
}

/**
 * What went wrong, by layer:
 *  - `server`       the server answered non-2xx (status + its own sentence)
 *  - `network`      no answer at all (refused, DNS, CORS blocked, offline)
 *  - `timeout`      our AbortController deadline fired
 *  - `aborted`      the caller cancelled
 *  - `bad_response` a 2xx whose body was not JSON
 *  - `incompatible` the server does not speak the API version this client was built for
 */
export type ApiErrorKind = "server" | "network" | "timeout" | "aborted" | "bad_response" | "incompatible";

export interface ApiErrorInit {
  kind: ApiErrorKind;
  message: string;
  status?: number;
  rawCode?: string | null;
  route?: string;
  cause?: unknown;
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  /** HTTP status; 0 when there was no HTTP answer. */
  readonly status: number;
  /** The server's code when this client knows it, else null (read `status`). */
  readonly code: ErrorCode | null;
  /** Whatever `code` the server sent, known or not. */
  readonly rawCode: string | null;
  /** `METHOD /path` of the request, for the operator. */
  readonly route: string;

  constructor(init: ApiErrorInit) {
    super(init.message, init.cause === undefined ? undefined : { cause: init.cause });
    this.name = "ApiError";
    this.kind = init.kind;
    this.status = init.status ?? 0;
    this.rawCode = init.rawCode ?? null;
    this.code = isErrorCode(this.rawCode) ? this.rawCode : null;
    this.route = init.route ?? "";
  }

  /** True when the server refused with exactly this code. */
  is(code: ErrorCode): boolean {
    return this.code === code;
  }
}

/**
 * Map a non-2xx response to an ApiError. The body is read defensively: a
 * proxy's HTML 502 or an empty body still yields a useful message.
 */
export function errorFromResponse(
  status: number,
  statusText: string,
  bodyText: string,
  route: string,
): ApiError {
  let message = "";
  let rawCode: string | null = null;
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (parsed && typeof parsed === "object") {
      const rec = parsed as Record<string, unknown>;
      if (typeof rec.error === "string") message = rec.error;
      if (typeof rec.code === "string") rawCode = rec.code;
    }
  } catch {
    // not JSON — fall through to the status line
  }
  if (!message) {
    const snippet = bodyText.trim().replace(/\s+/g, " ").slice(0, 160);
    message = `HTTP ${status}${statusText ? ` ${statusText}` : ""}${snippet ? ` — ${snippet}` : ""}`;
  }
  return new ApiError({ kind: "server", status, rawCode, message, route });
}

/** One line for a toast: what failed and what to do about the common cases. */
export function describeError(err: unknown): string {
  if (!(err instanceof ApiError)) {
    return err instanceof Error ? err.message : String(err);
  }
  switch (err.kind) {
    case "timeout":
    case "aborted":
    case "incompatible":
    case "bad_response":
      return err.message;
    case "network":
      return `${err.message} — is the server running, and (cross-origin) started with --cors-origin for this page?`;
    case "server":
      if (err.code === "unauthorized") return `${err.message} — set the API token in settings.`;
      if (err.code === "forbidden_origin")
        return `${err.message} — start the server with --cors-origin ${globalThis.location?.origin ?? "<this origin>"}.`;
      return err.message;
  }
}
