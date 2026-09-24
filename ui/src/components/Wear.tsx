import { useMemo } from "preact/hooks";
import { alphaMax, type DoseZone, doseZones, effectiveDose, zoneFor } from "../lib/dose";
import { fmt } from "../lib/format";
import { fanLoudnessRef, fanPull } from "../lib/pull";
import { useApp, useController } from "./context";

function zoneStyle(z: DoseZone): Record<string, string> {
  return { background: `var(--z-${z.key}, var(--z-audible))` };
}

/** The top bar: what is worn right now, as the server last reported it. */
export function WearBar() {
  const ctl = useController();
  const worn = useApp((s) => s.worn);
  const busy = useApp((s) => s.busy);
  const running = useApp((s) => s.loom.running);
  const session = useApp((s) => s.session);
  const band = useApp((s) => s.conn.info?.dose_band ?? null);
  const zones = useMemo(() => doseZones(band), [band]);
  if (!session) return null;
  if (!worn) {
    return (
      <div class="wearbar" aria-live="polite" data-testid="wearbar">
        <span class="lamp" aria-hidden="true" />
        <span>NO CODE WORN — conversation running straight</span>
      </div>
    );
  }
  if (!worn.active) {
    // never claim a bend the server is not applying
    return (
      <div class="wearbar" aria-live="polite" data-testid="wearbar">
        <span class="lamp" aria-hidden="true" />
        <span>
          <span class="alarm">INERT — </span>member code #{worn.index} was worn before a restart but could NOT
          be rebuilt. The conversation is running STRAIGHT. Re-loom to wear it again.
        </span>
      </div>
    );
  }
  const eff = effectiveDose(worn);
  const z = zoneFor(eff.alphaWorst, zones);
  return (
    <div class="wearbar on" aria-live="polite" data-testid="wearbar">
      <span class="lamp" aria-hidden="true" />
      <span>
        WEARING <span class="big">member code #{worn.index}</span>
        {worn.lever_kind && <span class="dim"> · {worn.lever_kind}</span>} · dose α{" "}
        <span class="big">{fmt(worn.alpha, 3)}</span>{" "}
        <span
          class="zone-chip"
          style={zoneStyle(z)}
          title={
            eff.scaled ? `zone read at the LOUDEST site, α ${fmt(eff.alphaWorst, 3)} — not the mean` : z.hint
          }
        >
          {z.label}
        </span>
        {eff.scaled && (
          <span class={eff.clamped ? "alarm" : "dim"}>
            {" "}
            ×{fmt(eff.scale, 3)} ⇒ α_eff {fmt(eff.alpha, 3)} (loudest site {fmt(eff.alphaWorst, 3)})
            {eff.clamped ? " ⚠CLAMPED" : ""}
          </span>
        )}
      </span>
      <span class="spacer" />
      <button type="button" class="tiny" disabled={!!busy || running} onClick={() => void ctl.unwear()}>
        unwear
      </button>
    </div>
  );
}

/**
 * THE DOSE KNOB. α means an absolute per-site injected norm of α × the
 * ruler in force; the zones are the map's own measured band, and the tier
 * (measured / derived / uncalibrated) is always shown beside the zone it
 * governs.
 */
export function WearControls() {
  const ctl = useController();
  const info = useApp((s) => s.conn.info);
  const alpha = useApp((s) => s.alpha);
  const selected = useApp((s) => s.selected);
  const futures = useApp((s) => s.futures);
  const worn = useApp((s) => s.worn);
  const lastWear = useApp((s) => s.lastWear);
  const busy = useApp((s) => s.busy);
  const running = useApp((s) => s.loom.running);
  const band = info?.dose_band ?? null;
  const zones = useMemo(() => doseZones(band), [band]);
  const amax = alphaMax(band);
  const f = selected === null ? null : (futures[selected] ?? null);
  const predicted = info?.dose_policy_default === "predicted";
  const pull = f ? fanPull(f.scores, fanLoudnessRef(futures)) : null;
  const serverScale = pull && pull.from === "server" ? pull.value : null;
  // under `predicted` the wear's mean injected magnitude is α × the future's scale
  const est = predicted && serverScale !== null ? alpha * serverScale : null;
  const z = zoneFor(est ?? alpha, zones);
  const tier = band?.tier ?? "none";

  let why: string;
  if (!f)
    why = futures.length ? "select a harvested future to arm the dose" : "no fan drawn — LOOM a turn first";
  else if (!f.harvested) why = `#${f.index} was not harvested — nothing to wear`;
  else why = `wear #${f.index} at α ${fmt(alpha, 3)} — the loom branch bends toward it from the next SEND`;

  return (
    <section class="gauge" aria-label="dose">
      <div class="row">
        <span class="lbl">dose α</span>
        <span class="readout" data-testid="alpha-readout">
          {fmt(alpha, 3)}
        </span>
        <span class="zone-chip" style={zoneStyle(z)} title={z.hint} data-testid="zone">
          {z.label}
        </span>
        <span class={`tier-badge ${tier}`} title={band?.source ?? "no dose band for this map"}>
          {tier === "none" ? "uncalibrated" : tier}
        </span>
        {est !== null && (
          <span
            class="mono dim"
            style={{ fontSize: "11px" }}
            title="an estimate from the card's pull (mean scale); the server's receipt replaces it after WEAR, and the zone is then read at the LOUDEST site"
          >
            {info?.dose_policy_default}: ×{fmt(serverScale, 3)} ⇒ α_eff ≈ {fmt(est, 3)}
          </span>
        )}
      </div>
      <div class="track">
        <div class="bands" aria-hidden="true">
          {zones.length ? (
            zones.map((zz) => {
              const hi = Math.min(zz.hi, amax);
              const w = Math.max(0, ((hi - Math.min(zz.lo, amax)) / amax) * 100);
              return (
                <span
                  key={zz.key}
                  class="band"
                  style={{ ...zoneStyle(zz), width: `${w}%` }}
                  title={zz.label}
                />
              );
            })
          ) : (
            <span class="band" style={{ background: "var(--z-uncalibrated)", width: "100%" }} />
          )}
        </div>
        <input
          type="range"
          min={0}
          max={amax}
          step={0.005}
          value={alpha}
          aria-label="dose alpha"
          aria-valuetext={`${fmt(alpha, 3)} — ${z.label}`}
          onInput={(e) => ctl.setAlpha(Number(e.currentTarget.value))}
          data-testid="alpha-slider"
        />
      </div>
      <div class="row">
        <button
          type="button"
          class="weft"
          disabled={!f?.harvested || !!busy || running}
          onClick={() => void ctl.wear()}
          data-testid="wear-btn"
        >
          {busy === "wear" ? "wearing…" : f ? `wear #${f.index}` : "wear"}
        </button>
        <button
          type="button"
          disabled={!worn || !!busy || running}
          onClick={() => void ctl.unwear()}
          data-testid="unwear-btn"
        >
          unwear
        </button>
        <span class="note mono" style={{ fontSize: "11px" }}>
          {why}
        </span>
      </div>
      {lastWear && worn && lastWear.worn.index === worn.index && (
        <div class="receipt" data-testid="wear-receipt">
          <span>
            receipt · sites {lastWear.sites.join(",")} · lever {lastWear.lever_kind} · dose_policy{" "}
            {lastWear.dose.policy}
            {lastWear.dose.clamped.length ? (
              <span class="alarm"> · {lastWear.dose.clamped.length} site(s) CLAMPED</span>
            ) : null}
          </span>
          <span>
            α {fmt(lastWear.effective_alpha.alpha, 3)} ⇒ effective mean{" "}
            {fmt(lastWear.effective_alpha.alpha_effective_mean, 3)}, per site [
            {lastWear.effective_alpha.alpha_effective_per_site.map((v) => fmt(v, 3)).join(", ")}]
          </span>
        </div>
      )}
    </section>
  );
}
