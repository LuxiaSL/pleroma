/**
 * Where the API is and how to authenticate, resolved in this order:
 *
 *   1. the settings panel (persisted in this browser's localStorage)
 *   2. build/dev env: VITE_LOOM_API_BASE, VITE_LOOM_API_TOKEN (.env.local)
 *   3. default: same origin, no token
 *
 * Nothing here names a host. "" means "the origin that served this page",
 * which is right both for `--ui-dir ui/dist` and for the Vite dev proxy.
 */

export interface ConnectionSettings {
  baseUrl: string;
  token: string;
}

const STORAGE_KEY = "loom-ui.connection.v1";

function envString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export function envDefaults(): ConnectionSettings {
  const env = import.meta.env as Record<string, unknown>;
  return { baseUrl: envString(env.VITE_LOOM_API_BASE), token: envString(env.VITE_LOOM_API_TOKEN) };
}

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null; // private mode, sandboxed iframe, or no DOM
  }
}

export function loadSettings(): ConnectionSettings {
  const defaults = envDefaults();
  const raw = storage()?.getItem(STORAGE_KEY);
  if (!raw) return defaults;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return defaults;
    const rec = parsed as Record<string, unknown>;
    return {
      baseUrl: typeof rec.baseUrl === "string" ? rec.baseUrl.trim() : defaults.baseUrl,
      token: typeof rec.token === "string" ? rec.token.trim() : defaults.token,
    };
  } catch {
    return defaults;
  }
}

export function saveSettings(s: ConnectionSettings): void {
  try {
    storage()?.setItem(STORAGE_KEY, JSON.stringify({ baseUrl: s.baseUrl.trim(), token: s.token.trim() }));
  } catch {
    // storage full or blocked: settings still apply for this page load
  }
}

/** Validate a base URL typed by the operator. "" is valid (same origin). */
export function validateBaseUrl(value: string): string | null {
  const v = value.trim();
  if (!v) return null;
  try {
    const u = new URL(v);
    if (u.protocol !== "http:" && u.protocol !== "https:") return "must be an http(s) URL";
    if (u.search || u.hash) return "no query string or fragment";
    return null;
  } catch {
    return "not a URL — e.g. http://127.0.0.1:8767, or leave empty for this page's origin";
  }
}

const SESSION_KEY = "loom-ui.session.v1";

export function loadLastSession(): string | null {
  const v = storage()?.getItem(SESSION_KEY);
  return v?.trim() ? v.trim() : null;
}

export function saveLastSession(name: string | null): void {
  try {
    if (name) storage()?.setItem(SESSION_KEY, name);
    else storage()?.removeItem(SESSION_KEY);
  } catch {
    // ignore
  }
}
