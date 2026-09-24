import { useState } from "preact/hooks";
import { applyTheme, loadTheme, nextTheme, type ThemeMode } from "../theme";
import { useApp, useController } from "./context";

/** The rig: what map the server loaded, how we are connected, which session. */
export function Header({ onSettings }: { onSettings: () => void }) {
  const ctl = useController();
  const conn = useApp((s) => s.conn);
  const session = useApp((s) => s.session);
  const baseUrl = useApp((s) => s.settings.baseUrl);
  const [theme, setTheme] = useState<ThemeMode>(loadTheme);
  const info = conn.info;
  const meta = info?.map_meta ?? {};
  const rank = typeof meta.rank === "number" ? meta.rank : null;
  const token = typeof meta.prereg_token === "string" ? meta.prereg_token : null;
  const tier = info?.dose_band.tier ?? null;

  return (
    <header class="rig">
      {/* the version is the MAP the server loaded, read from /info — never a literal */}
      <div class="brand">
        LOOM
        <span class="ver" title={info ? info.map : "waiting on /info"}>
          {info ? `${token ?? "map"}${rank ? ` · rank ${rank}` : ""}` : "…"}
        </span>
      </div>
      <div class="cells">
        <span class="cell">
          <span class="k">server</span>
          <span
            class={`dot ${conn.status === "connected" ? "ok" : conn.status === "connecting" ? "busy" : conn.status === "failed" ? "bad" : ""}`}
            aria-hidden="true"
          />
          <span data-testid="conn-status">
            {conn.status === "connected" ? baseUrl || "same origin" : conn.status}
          </span>
        </span>
        {info && (
          <>
            <span class="cell">
              <span class="k">api</span>v{info.api.version}
              {conn.schemaMatches === false && (
                <span
                  class="alarm"
                  title="the server's schema_sha256 differs from the one this UI was generated from"
                >
                  {" "}
                  · schema differs
                </span>
              )}
            </span>
            <span class="cell">
              <span class="k">prompt</span>
              {info.prompt_mode}
            </span>
            <span class="cell">
              <span class="k">sites</span>
              {info.sites.join(",")}
            </span>
            <span class="cell">
              <span class="k">dose</span>
              {info.dose_policy_default}{" "}
              <span class={`tier-badge ${tier ?? ""}`}>
                {tier === "none" ? "uncalibrated" : (tier ?? "?")}
              </span>
            </span>
          </>
        )}
      </div>
      <span class="spacer" />
      {session && (
        <span class="cell mono" style={{ fontSize: "12px" }}>
          <span class="lbl">sess </span>
          <span data-testid="session-name">{session}</span>{" "}
          <button type="button" class="tiny" onClick={() => ctl.closeSession()}>
            sessions
          </button>
        </span>
      )}
      <button type="button" class="tiny" onClick={onSettings} aria-label="connection settings">
        settings
      </button>
      <button
        type="button"
        class="tiny"
        title="cycle theme: auto / dark / light"
        onClick={() => {
          const n = nextTheme(theme);
          applyTheme(n);
          setTheme(n);
        }}
      >
        {theme}
      </button>
    </header>
  );
}
