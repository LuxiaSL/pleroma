import { useState } from "preact/hooks";
import { saveSettings, validateBaseUrl } from "../config";
import { useApp, useController } from "./context";

/** Where the loom server is and how to authenticate. Shown until connected, and from the header. */
export function ConnectPanel({ onDone }: { onDone?: () => void }) {
  const ctl = useController();
  const conn = useApp((s) => s.conn);
  const settings = useApp((s) => s.settings);
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl);
  const [token, setToken] = useState(settings.token);
  const urlErr = validateBaseUrl(baseUrl);
  const busy = conn.status === "connecting";

  const submit = async (e: Event) => {
    e.preventDefault();
    if (urlErr) return;
    const next = { baseUrl: baseUrl.trim(), token: token.trim() };
    saveSettings(next);
    const ok = await ctl.connect(next);
    if (ok) onDone?.();
  };

  return (
    <section class="center">
      <form class="panel connect" onSubmit={submit} aria-labelledby="connect-h">
        <h2 id="connect-h" class="lbl" style={{ margin: 0 }}>
          connect to a loom server
        </h2>
        <p class="note" style={{ margin: 0 }}>
          This UI talks to the loom's versioned API (<span class="mono">/api/v1</span>). Leave the server
          empty to use the origin that served this page — right for <span class="mono">--ui-dir ui/dist</span>{" "}
          and for the dev proxy. A different origin must be allowed by the server with{" "}
          <span class="mono">--cors-origin {globalThis.location?.origin ?? ""}</span>.
        </p>
        <div class="grid">
          <label class="lbl" for="base-url">
            server
          </label>
          <input
            id="base-url"
            type="url"
            placeholder="(same origin)"
            value={baseUrl}
            onInput={(e) => setBaseUrl(e.currentTarget.value)}
            aria-invalid={urlErr ? "true" : "false"}
            aria-describedby={urlErr ? "base-url-err" : undefined}
            spellcheck={false}
            autocomplete="off"
          />
          <label class="lbl" for="api-token">
            token
          </label>
          <input
            id="api-token"
            type="password"
            placeholder="(none — only if the server has --api-token / PLEROMA_API_TOKEN)"
            value={token}
            onInput={(e) => setToken(e.currentTarget.value)}
            autocomplete="off"
          />
        </div>
        {urlErr && (
          <div id="base-url-err" class="alarm note">
            {urlErr}
          </div>
        )}
        <p class="note faint" style={{ margin: 0 }}>
          The token is kept in this browser's localStorage. It is sent only as{" "}
          <span class="mono">Authorization: Bearer</span> to the server above.
        </p>
        <div class="row">
          <button type="submit" class="primary" disabled={busy || !!urlErr} data-testid="connect-btn">
            {busy ? "connecting…" : "connect"}
          </button>
          {onDone && conn.status === "connected" && (
            <button type="button" onClick={onDone}>
              close
            </button>
          )}
        </div>
        {conn.status === "failed" && conn.error && (
          <div class="errbox" role="alert" data-testid="connect-error">
            {conn.error}
          </div>
        )}
      </form>
    </section>
  );
}
