/**
 * THE FAN'S CODE SPACE, in 3-D. Each harvested future is a body placed on the
 * fan's own top-3 principal directions of its centred codes (lib/codespace.ts),
 * sized by pull, threaded to its nearest fan-mate by full-space cosine; the
 * worn future glows in the weft colour. Hover names a body; click selects its
 * card (the cards stay the keyboard path — this view is a picture of them).
 */
import { useEffect, useRef, useState } from "preact/hooks";
import type { Future, WornPublic } from "../api/types";
import type { CodeSpaceResult } from "../lib/codespace";
import { fmt, pct, snippet } from "../lib/format";
import { isWornHere } from "../lib/wear";
import type { FanMeta } from "../store/app";
import { prefersReducedMotion } from "../theme";
import { CodeSpaceScene, readColors } from "./codeSpaceScene";

export interface CodeSpace3DProps {
  result: CodeSpaceResult;
  futures: readonly Future[];
  worn: WornPublic | null;
  fan: FanMeta | null;
  selected: number | null;
  onSelect: (slot: number) => void;
}

const PULL_FROM: Record<string, string> = {
  server: "server's predicted_dose_scale",
  "fan-loudness": "‖raw‖ over the fan mean",
  code: "‖centred code‖ over the fan mean",
};

export default function CodeSpace3D(props: CodeSpace3DProps) {
  const { result, futures, worn, fan, selected, onSelect } = props;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<CodeSpaceScene | null>(null);
  const [glError, setGlError] = useState<string | null>(null);
  const [hover, setHover] = useState<{ slot: number; x: number; y: number } | null>(null);
  const [labels, setLabels] = useState<{ slot: number; x: number; y: number; visible: boolean }[]>([]);
  const down = useRef<{ x: number; y: number } | null>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const wornSlot = futures.findIndex((f) => isWornHere(worn, f, fan?.loomId));
  const highlight = { selectedSlot: selected, wornSlot: wornSlot >= 0 ? wornSlot : null };

  // create / dispose the scene with the canvas
  useEffect(() => {
    const cv = canvas.current;
    const box = wrap.current;
    if (!cv || !box || !result.ok) return;
    let scene: CodeSpaceScene;
    try {
      scene = new CodeSpaceScene(cv, { reducedMotion: prefersReducedMotion(), colors: readColors() });
    } catch (err) {
      setGlError(
        `WebGL is unavailable here (${err instanceof Error ? err.message : String(err)}) — the cards below carry the same numbers.`,
      );
      return;
    }
    sceneRef.current = scene;
    let lastLabels = "";
    scene.onRender = (pts) => {
      const key = pts.map((p) => `${p.slot}:${p.x.toFixed(0)},${p.y.toFixed(0)}`).join("|");
      if (key !== lastLabels) {
        lastLabels = key;
        setLabels(pts);
      }
    };
    const ro = new ResizeObserver(() => scene.resize(box.clientWidth, box.clientHeight));
    ro.observe(box);
    scene.resize(box.clientWidth, box.clientHeight);
    const recolor = () => scene.setColors(readColors());
    const mq = globalThis.matchMedia?.("(prefers-color-scheme: light)");
    mq?.addEventListener("change", recolor);
    document.addEventListener("loom-theme", recolor);
    return () => {
      ro.disconnect();
      mq?.removeEventListener("change", recolor);
      document.removeEventListener("loom-theme", recolor);
      scene.dispose();
      sceneRef.current = null;
    };
  }, [result.ok]);

  // new data → rebuild
  useEffect(() => {
    if (result.ok) sceneRef.current?.setData(result.space, highlight);
  }, [result]);

  // selection / wear changes → restyle only
  useEffect(() => {
    sceneRef.current?.setHighlight(highlight);
  }, [highlight.selectedSlot, highlight.wornSlot]);

  if (!result.ok) {
    return (
      <div class="codespace" data-testid="codespace">
        <div class="msgbox" data-testid="codespace-msg">
          {result.reason}. The cards below still carry every score.
        </div>
      </div>
    );
  }
  const space = result.space;
  const byslot = new Map(space.points.map((p) => [p.slot, p]));
  const hp = hover ? byslot.get(hover.slot) : undefined;
  const hf = hover ? futures[hover.slot] : undefined;

  const local = (e: PointerEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const summary =
    `${space.points.length} futures in ${space.dims}-D; the view holds ${pct(space.viewShare)} of the fan's spread. ` +
    `Most solitary: #${futures[space.lone]?.index ?? "?"}.` +
    (wornSlot >= 0 ? ` Worn: #${futures[wornSlot]?.index}.` : "");

  return (
    <div
      class="codespace"
      ref={wrap}
      data-testid="codespace"
      data-dims={space.dims}
      role="img"
      aria-label={`code space: ${summary}`}
    >
      {glError ? (
        <div class="msgbox">{glError}</div>
      ) : (
        <canvas
          ref={canvas}
          data-testid="codespace-canvas"
          onPointerDown={(e) => {
            down.current = local(e);
          }}
          onPointerMove={(e) => {
            const p = local(e);
            const slot = sceneRef.current?.pick(p.x, p.y) ?? null;
            setHover(slot === null ? null : { slot, ...p });
          }}
          onPointerLeave={() => setHover(null)}
          onPointerUp={(e) => {
            const p = local(e);
            // a click, not the end of an orbit drag
            const d = down.current;
            if (d && Math.hypot(p.x - d.x, p.y - d.y) < 5) {
              const slot = sceneRef.current?.pick(p.x, p.y) ?? null;
              if (slot !== null) onSelectRef.current(slot);
            }
            down.current = null;
          }}
        />
      )}
      {labels.map((l) =>
        l.visible ? (
          <span
            key={l.slot}
            class="mono"
            aria-hidden="true"
            style={{
              position: "absolute",
              left: `${l.x + 8}px`,
              top: `${l.y - 16}px`,
              fontSize: "10px",
              pointerEvents: "none",
              color:
                l.slot === wornSlot ? "var(--weft)" : l.slot === selected ? "var(--ink)" : "var(--ink-faint)",
            }}
          >
            #{futures[l.slot]?.index}
            {l.slot === space.lone ? " alone" : ""}
          </span>
        ) : null,
      )}
      {hover && hp && hf && (
        <div
          class="tip"
          style={{
            left: `${Math.min(hover.x + 14, (wrap.current?.clientWidth ?? 400) - 240)}px`,
            top: `${hover.y + 12}px`,
          }}
          data-testid="codespace-tip"
        >
          <div class="mono">
            #{hf.index} · pull ×{fmt(hp.pull, 2)} · alone {fmt(hp.solitude, 2)}
            {hover.slot === wornSlot ? " · WORN" : ""}
          </div>
          <div class="faint" style={{ fontSize: "10.5px" }}>
            pull from {PULL_FROM[hp.pullFrom]}; nearest #{futures[hp.nn]?.index} (cos {fmt(hp.nnCos, 2)})
          </div>
          <div>{snippet(hf.text, 180)}</div>
        </div>
      )}
      <div class="legend">
        position: the fan's own top-{space.dims} principal directions of its centred{" "}
        {space.kind === "code" ? "codes" : "contrast codes"} (rank {space.rank}; axes hold{" "}
        {space.axisShare.map((s) => pct(s)).join(" / ")} = {pct(space.viewShare)} of the spread — the basis is
        arbitrary, the geometry is not) · size: pull · thread: nearest fan-mate by cosine in full space
        {space.dims === 2 ? " · 3 futures span a plane, so this fan is shown flat" : ""}
        {space.unplaced ? ` · ${space.unplaced} unharvested not placed` : ""}
      </div>
    </div>
  );
}
