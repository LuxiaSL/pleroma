import { useEffect, useRef } from "preact/hooks";
import type { Branch } from "../api/types";
import { fmt, seconds } from "../lib/format";
import type { Msg } from "../store/app";
import { useApp, useController } from "./context";

const PANE_NOTE: Record<Branch, string> = {
  loom: "the branch that bends",
  base: "unbent control · separate history",
};

function Message({ m }: { m: Msg }) {
  if (m.role === "event") return <div class="msg event">{m.content}</div>;
  const bent = m.role === "assistant" && m.worn && m.worn.active;
  return (
    <div
      class={`msg ${m.role}${bent ? " bent" : ""}${m.pending ? " pending" : ""}`}
      aria-busy={m.pending ? "true" : "false"}
    >
      {m.pending ? "…waiting on the model" : m.content}
      {m.role === "assistant" && !m.pending && (bent || m.elapsed_s !== undefined) && (
        <div class="meta">
          {bent && m.worn && (
            <span class="weft-ink">
              bent · #{m.worn.index} α {fmt(m.worn.alpha, 3)}
            </span>
          )}
          {m.elapsed_s !== undefined && <span>{seconds(m.elapsed_s)}</span>}
        </div>
      )}
    </div>
  );
}

function Pane({ branch }: { branch: Branch }) {
  const msgs = useApp((s) => s.histories[branch]);
  const sessionStatus = useApp((s) => s.sessionStatus);
  const scroller = useRef<HTMLDivElement>(null);
  const last = msgs[msgs.length - 1];
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs.length, last?.pending]);
  return (
    <div class="pane" data-branch={branch}>
      <div class="pane-tag">
        <span>{branch}</span>
        <span class="faint">{PANE_NOTE[branch]}</span>
      </div>
      <div
        class="scroll"
        ref={scroller}
        role="log"
        aria-label={`${branch} transcript`}
        aria-live="polite"
        data-testid={`transcript-${branch}`}
        // biome-ignore lint/a11y/noNoninteractiveTabindex: a scrollable log must be reachable by keyboard
        tabIndex={0}
      >
        {msgs.length === 0 && (
          <div class="empty">
            {sessionStatus === "new"
              ? "a new session — nothing on the server yet. The first SEND or LOOM creates it."
              : `no turns on the ${branch} branch yet`}
          </div>
        )}
        {msgs.map((m) => (
          <Message key={m.id} m={m} />
        ))}
      </div>
    </div>
  );
}

export function Transcript() {
  const ctl = useController();
  const view = useApp((s) => s.view);
  const loomTurns = useApp((s) => s.histories.loom.filter((m) => m.role === "user").length);
  const baseTurns = useApp((s) => s.histories.base.filter((m) => m.role === "user").length);
  const views: (Branch | "split")[] = ["loom", "base", "split"];
  return (
    <>
      <div class="colhead">
        <span class="lbl">branch</span>
        <fieldset class="seg">
          <legend class="sr-only">branch view</legend>
          {views.map((v) => (
            <button
              key={v}
              type="button"
              class="tiny"
              aria-pressed={view === v ? "true" : "false"}
              onClick={() => ctl.store.set({ view: v })}
            >
              {v}
            </button>
          ))}
        </fieldset>
        <span class="spacer" />
        <span class="lbl" data-testid="turn-count">
          turns loom {loomTurns} · base {baseTurns}
        </span>
      </div>
      <div class={`panes${view === "split" ? " split" : ""}`}>
        {(view === "loom" || view === "split") && <Pane branch="loom" />}
        {(view === "base" || view === "split") && <Pane branch="base" />}
      </div>
    </>
  );
}
