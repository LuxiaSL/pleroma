import { useState } from "preact/hooks";
import type { SessionRow, WornPublic } from "../api/types";
import { fmt } from "../lib/format";
import { validateSessionName } from "../store/app";
import { useApp, useController } from "./context";

function ago(unix: number): string {
  const s = Math.max(0, Date.now() / 1000 - unix);
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129_600) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86_400)} d ago`;
}

function wornText(w: SessionRow["worn"]): string {
  if (!w) return "—";
  const alpha = (w as WornPublic).alpha;
  const inert = "active" in w && w.active === false;
  return `#${w.index} α ${fmt(alpha, 3)}${inert ? " (inert)" : ""}`;
}

/**
 * Every session the server knows — in memory, or only as a snapshot on disk
 * (GET /api/v1/sessions). The old single-file UI had no picker at all: it
 * sent whatever name was typed in the header and the legacy /state CREATED it.
 */
export function SessionPicker() {
  const ctl = useController();
  const sessions = useApp((s) => s.sessions);
  const [name, setName] = useState("");
  const nameErr = name ? validateSessionName(name) : null;

  return (
    <section class="center">
      <div class="panel picker">
        <div class="row">
          <h2 class="lbl" style={{ margin: 0 }}>
            sessions on this server
          </h2>
          <span class="spacer" />
          <button
            type="button"
            class="tiny"
            onClick={() => void ctl.refreshSessions()}
            disabled={sessions.loading}
          >
            {sessions.loading ? "reading…" : "refresh"}
          </button>
        </div>
        <p class="note" style={{ margin: 0 }}>
          A session is the server's tag for two conversations — <b>loom</b> (bends under a worn code) and{" "}
          <b>base</b> (the unbent control, a separate history) — plus its latest fan of futures. Opening one
          never creates it; a new name is created by its first SEND or LOOM.
        </p>
        {sessions.error && (
          <div class="errbox" role="alert">
            {sessions.error}
          </div>
        )}
        <div class="tablewrap">
          <table data-testid="session-table">
            <thead>
              <tr>
                <th scope="col">session</th>
                <th scope="col">where</th>
                <th scope="col">turns loom / base</th>
                <th scope="col" class="opt">
                  looms
                </th>
                <th scope="col" class="opt">
                  worn
                </th>
                <th scope="col" class="opt">
                  modified
                </th>
                <th scope="col">
                  <span class="sr-only">action</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {sessions.rows.length === 0 && !sessions.loading && (
                <tr>
                  <td colSpan={7} class="faint">
                    no sessions yet — name one below
                  </td>
                </tr>
              )}
              {sessions.rows.map((r) => (
                <tr key={r.session}>
                  <td class="name">{r.session}</td>
                  <td>
                    {!r.ok ? (
                      <span class="alarm" title={r.error ?? ""}>
                        unreadable
                      </span>
                    ) : r.in_memory ? (
                      "in memory"
                    ) : (
                      <span class="faint">on disk only</span>
                    )}
                  </td>
                  <td>
                    {r.n_turns.loom ?? 0} / {r.n_turns.base ?? 0}
                  </td>
                  <td class="opt">{r.n_looms}</td>
                  <td class="opt">{wornText(r.worn)}</td>
                  <td class="opt" title={r.modified_iso}>
                    {ago(r.modified_unix)}
                  </td>
                  <td>
                    {r.ok && r.in_memory && (
                      <button
                        type="button"
                        class="tiny primary"
                        onClick={() => void ctl.openSession(r.session)}
                      >
                        open
                      </button>
                    )}
                    {r.ok && !r.in_memory && (
                      <button
                        type="button"
                        class="tiny"
                        title="POST /restore — load the snapshot into the server's memory, then open it"
                        onClick={() => void ctl.restoreSession(r.session)}
                      >
                        restore
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <form
          class="row"
          onSubmit={(e) => {
            e.preventDefault();
            if (!nameErr && name.trim()) void ctl.openSession(name);
          }}
        >
          <label class="field">
            <span class="lbl">open or start</span>
            <input
              type="text"
              value={name}
              placeholder="session name"
              spellcheck={false}
              autocomplete="off"
              onInput={(e) => setName(e.currentTarget.value)}
              aria-invalid={nameErr ? "true" : "false"}
              data-testid="new-session-input"
            />
          </label>
          <button type="submit" disabled={!!nameErr || !name.trim()}>
            open
          </button>
          {nameErr && <span class="alarm note">{nameErr}</span>}
        </form>
      </div>
    </section>
  );
}
