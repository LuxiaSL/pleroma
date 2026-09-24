/** auto → dark → light, persisted per browser; "auto" follows prefers-color-scheme. */
export type ThemeMode = "auto" | "dark" | "light";
const KEY = "loom-ui.theme";

export function loadTheme(): ThemeMode {
  try {
    const v = globalThis.localStorage?.getItem(KEY);
    return v === "dark" || v === "light" ? v : "auto";
  } catch {
    return "auto";
  }
}

export function applyTheme(mode: ThemeMode): void {
  document.documentElement.dataset.theme = mode;
  try {
    globalThis.localStorage?.setItem(KEY, mode);
  } catch {
    // ignore
  }
  document.dispatchEvent(new CustomEvent("loom-theme", { detail: mode }));
}

export function nextTheme(mode: ThemeMode): ThemeMode {
  return mode === "auto" ? "dark" : mode === "dark" ? "light" : "auto";
}

export function prefersReducedMotion(): boolean {
  return globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}
