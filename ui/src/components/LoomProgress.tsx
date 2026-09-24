import { useEffect, useState } from "preact/hooks";
import type { LoomProgress } from "../api/types";
import { useApp, useController } from "./context";

/**
 * A determinate wait, driven by /loom/progress and drawing ONLY what it
 * reports. The server's shape: `generate` goes 0/k → k/k in one jump (one
 * batched generate call); `harvest` lands branch by branch in completion
 * order (`gens.harvested`). A server that reports no `gens` gets a plain bar.
 */
export function LoomProgressView() {
  const ctl = useController();
  const loom = useApp((s) => s.loom);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const p: LoomProgress | null = loom.progress;
  const k = p?.gens?.n_gens ?? loom.plan?.k ?? 0;
  const harvested = new Set(p?.gens?.harvested ?? []);
  const elapsed = loom.startedAt ? Math.max(0, (now - loom.startedAt) / 1000) : 0;

  let label: string;
  let frac: number;
  if (!p) {
    label = "waiting for the server to report progress…";
    frac = 0;
  } else if (p.stage === "generate") {
    label = `generating ${p.total} futures (one batched call) — ${p.done}/${p.total}`;
    frac = 0.18 * (p.total ? p.done / p.total : 0);
  } else if (p.stage === "harvest") {
    const n = p.gens ? p.gens.harvested.length : null;
    label =
      n !== null
        ? `harvesting codes — ${n}/${p.gens?.n_gens ?? k} branches landed`
        : `harvesting codes — stage ${p.done}/${p.total}`;
    frac = 0.18 + 0.82 * (n !== null && k ? n / k : p.total ? p.done / p.total : 0);
  } else {
    label = `${p.stage} — ${p.done}/${p.total}`;
    frac = p.total ? p.done / p.total : 0;
  }
  const generated = p ? p.stage !== "generate" || p.done >= p.total : false;

  return (
    <div class="progress panel" data-testid="loom-progress" aria-live="polite">
      <div class="row">
        <span class="lbl">drawing</span>
        <span class="mono" data-testid="loom-stage">
          {label}
        </span>
        <span class="spacer" />
        <span class="mono faint">{elapsed.toFixed(0)} s</span>
      </div>
      <div
        class="bar"
        role="progressbar"
        aria-label="loom progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(frac * 100)}
      >
        <span style={{ width: `${Math.round(frac * 100)}%` }} />
      </div>
      {k > 0 && (
        <div class="slots">
          {Array.from({ length: k }, (_, i) => (
            <span
              key={i}
              class={`slot${harvested.has(i) ? " done" : generated ? " gen" : ""}`}
              title={
                harvested.has(i) ? `#${i} harvested` : generated ? `#${i} generated, harvesting` : `#${i}`
              }
            >
              {i}
            </span>
          ))}
        </div>
      )}
      <div class="row">
        {loom.text && (
          <span class="note">for: “{loom.text.length > 90 ? `${loom.text.slice(0, 90)}…` : loom.text}”</span>
        )}
        <span class="spacer" />
        <button
          type="button"
          class="tiny"
          onClick={() => ctl.cancelLoom()}
          title="stop waiting here — the server still finishes the draw, and its futures are the session's next /state"
        >
          stop waiting
        </button>
      </div>
    </div>
  );
}
