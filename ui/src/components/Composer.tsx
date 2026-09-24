import { useState } from "preact/hooks";
import type { Branch } from "../api/types";
import { useApp, useController } from "./context";

function clampInt(v: string, lo: number, hi: number, dflt: number): number {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : dflt;
}

/**
 * SEND commits a turn. LOOM only contemplates it — K futures are drawn for
 * that text, none of them spoken, and the composer keeps the text so the
 * operator can wear a future and then SEND the same turn under it.
 */
export function Composer() {
  const ctl = useController();
  const info = useApp((s) => s.conn.info);
  const view = useApp((s) => s.view);
  const busy = useApp((s) => s.busy);
  const looming = useApp((s) => s.loom.running);
  const mirror = useApp((s) => s.mirror);
  const freshDraw = useApp((s) => s.freshDraw);
  const worn = useApp((s) => s.worn);
  const [text, setText] = useState("");
  const [k, setK] = useState(String(info?.default_k ?? 6));
  const [horizon, setHorizon] = useState(String(info?.future_tokens ?? 192));

  const target: Branch = view === "base" ? "base" : "loom";
  const blocked = !!busy || looming;
  const empty = !text.trim();

  const send = async () => {
    if (empty || blocked) return;
    const ok = await ctl.send(text, target);
    if (ok) setText("");
  };
  const loom = async () => {
    if (empty || blocked) return;
    const kk = clampInt(k, 2, 16, info?.default_k ?? 6);
    const hh = clampInt(horizon, 16, 1024, info?.future_tokens ?? 192);
    setK(String(kk));
    setHorizon(String(hh));
    await ctl.drawLoom(text, kk, hh);
  };

  return (
    <div class="composer">
      <label class="sr-only" for="composer">
        message
      </label>
      <textarea
        id="composer"
        data-testid="composer"
        value={text}
        placeholder="Type the next thing you might say. SEND commits it. LOOM only contemplates it — K futures are drawn, none of them spoken."
        onInput={(e) => setText(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            void (e.shiftKey ? loom() : send());
          }
        }}
      />
      <div class="row">
        <button
          type="button"
          class="primary"
          disabled={empty || blocked}
          onClick={() => void send()}
          data-testid="send-btn"
        >
          {busy === "chat" ? "sending…" : `send → ${target}`}
        </button>
        <button
          type="button"
          class="weft"
          disabled={empty || blocked}
          onClick={() => void loom()}
          data-testid="loom-btn"
        >
          {looming ? "drawing…" : "loom ↯"}
        </button>
        <label class="field" title="number of futures to draw">
          <span class="lbl">k</span>
          <input
            type="number"
            min={2}
            max={16}
            step={1}
            value={k}
            onInput={(e) => setK(e.currentTarget.value)}
            data-testid="k-input"
          />
        </label>
        <label class="field" title="tokens of future to generate per thread">
          <span class="lbl">horizon</span>
          <input
            type="number"
            min={16}
            max={1024}
            step={16}
            value={horizon}
            onInput={(e) => setHorizon(e.currentTarget.value)}
          />
        </label>
      </div>
      <div class="row">
        <label class="field" title="also post the same message to the unbent control branch">
          <input
            type="checkbox"
            checked={mirror}
            disabled={target === "base"}
            onChange={(e) => ctl.store.set({ mirror: e.currentTarget.checked })}
          />
          <span class="lbl">mirror to base</span>
        </label>
        <label
          class="field"
          title="take the current wear off before each draw — its imprint is already in the text it wrote. If the unwear fails, the draw runs with the wear detached, never under it."
        >
          <input
            type="checkbox"
            checked={freshDraw}
            onChange={(e) => ctl.store.set({ freshDraw: e.currentTarget.checked })}
          />
          <span class="lbl">fresh draw</span>
        </label>
        {worn && !freshDraw && (
          <span class="hint weft-ink">the next fan is drawn UNDER code #{worn.index}</span>
        )}
        <span class="spacer" />
        <span class="hint">⌘/ctrl+↵ send · ⌘/ctrl+⇧+↵ loom</span>
      </div>
    </div>
  );
}
