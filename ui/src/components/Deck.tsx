import { lazy, Suspense } from "preact/compat";
import { useMemo, useState } from "preact/hooks";
import { buildCodeSpace } from "../lib/codespace";
import { fmt, seconds } from "../lib/format";
import { fanLoudnessRef, fanPull } from "../lib/pull";
import { isWornHere } from "../lib/wear";
import type { FanMeta } from "../store/app";
import { useApp, useController } from "./context";
import { FutureCard } from "./FutureCard";
import { LoomProgressView } from "./LoomProgress";

// three.js is ~600 kB: load it only when the view is opened
const CodeSpace3D = lazy(() => import("../views/CodeSpace3D"));

function deckStat(fan: FanMeta | null, n: number, running: boolean): string {
  if (running) return "drawing…";
  if (!fan || n === 0) return "idle · no futures drawn";
  const t = fan.timing_s;
  const total = t.generate + t.harvest;
  const timing = Number.isFinite(total)
    ? ` · drawn in ${seconds(total)} (generate ${seconds(t.generate)} + harvest ${seconds(t.harvest)})`
    : "";
  return `k=${n} · horizon ${fan.horizon}${timing}`;
}

export function Deck() {
  const ctl = useController();
  const futures = useApp((s) => s.futures);
  const fan = useApp((s) => s.fan);
  const running = useApp((s) => s.loom.running);
  const selected = useApp((s) => s.selected);
  const worn = useApp((s) => s.worn);
  const codeKind = useApp((s) => s.codeKind);
  const clamp = useApp((s) => s.conn.info?.dose_scale_clamp ?? null);
  const [showSpace, setShowSpace] = useState(true);

  const ref = useMemo(() => fanLoudnessRef(futures), [futures]);
  const space = useMemo(() => buildCodeSpace(futures, codeKind), [futures, codeKind]);
  const lone = space.ok ? space.space.lone : null;
  const hasContrast = futures.some((f) => Array.isArray(f.code_contrast) && f.code_contrast.length > 0);

  return (
    <>
      <div class="colhead">
        <span class="lbl">loom deck</span>
        <span class="lbl" data-testid="deck-stat">
          {deckStat(fan, futures.length, running)}
        </span>
        <span class="spacer" />
        <button
          type="button"
          class="tiny"
          aria-pressed={showSpace ? "true" : "false"}
          onClick={() => setShowSpace(!showSpace)}
          title="the fan as objects in its own code space — where each future sits, how hard it pulls, who is alone"
        >
          code space
        </button>
      </div>
      <div class="deck-body">
        {running && <LoomProgressView />}
        {fan?.stale && futures.length > 0 && (
          <div class="stale" role="status">
            a turn was committed after this fan was drawn — these futures answer an earlier context. A wear
            still applies; LOOM again for futures of the conversation as it now stands.
          </div>
        )}
        {fan?.drawnUnderWear && (
          <div class="note weft-ink">
            drawn UNDER code #{fan.drawnUnderWear.index} at α {fmt(fan.drawnUnderWear.alpha, 3)} — every
            future here carries that bend
          </div>
        )}
        {fan?.text && !running && (
          <div class="note">
            contemplated, not sent: “{fan.text.length > 140 ? `${fan.text.slice(0, 140)}…` : fan.text}”
          </div>
        )}
        {showSpace && futures.length > 0 && !running && (
          <div>
            <div class="row" style={{ marginBottom: "6px" }}>
              <span class="lbl">code space</span>
              <fieldset class="seg">
                <legend class="sr-only">which code</legend>
                <button
                  type="button"
                  class="tiny"
                  aria-pressed={codeKind === "code" ? "true" : "false"}
                  onClick={() => ctl.store.set({ codeKind: "code" })}
                  title="each future's own rank-R code"
                >
                  code
                </button>
                <button
                  type="button"
                  class="tiny"
                  aria-pressed={codeKind === "code_contrast" ? "true" : "false"}
                  disabled={!hasContrast}
                  onClick={() => ctl.store.set({ codeKind: "code_contrast" })}
                  title={
                    hasContrast
                      ? "the one-vs-rest contrast code"
                      : "this draw carries no code_contrast (a draw recorded without one)"
                  }
                >
                  contrast
                </button>
              </fieldset>
            </div>
            <Suspense
              fallback={
                <div class="codespace">
                  <div class="msgbox">loading the 3-D view…</div>
                </div>
              }
            >
              <CodeSpace3D
                result={space}
                futures={futures}
                worn={worn}
                fan={fan}
                selected={selected}
                onSelect={(slot) => ctl.select(slot)}
              />
            </Suspense>
          </div>
        )}
        {futures.length === 0 && !running && (
          <div class="empty">
            No futures drawn. Type a turn and press LOOM: K futures are drawn for it, none of them spoken.
            Pick one and WEAR it to bend the loom branch toward it.
          </div>
        )}
        <div class="cards" data-testid="cards">
          {futures.map((f, slot) => (
            <FutureCard
              key={`${fan?.loomId ?? "x"}-${f.index}`}
              future={f}
              slot={slot}
              horizon={fan?.horizon ?? null}
              pull={fanPull(f.scores, ref)}
              clamp={clamp}
              selected={selected === slot}
              wornHere={isWornHere(worn, f, fan?.loomId)}
              lone={lone === slot}
              worn={worn}
              onSelect={(sl) => ctl.select(sl)}
            />
          ))}
        </div>
      </div>
    </>
  );
}
