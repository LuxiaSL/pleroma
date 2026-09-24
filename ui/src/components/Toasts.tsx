import { useEffect } from "preact/hooks";
import type { Toast } from "../store/app";
import { useApp, useController } from "./context";

function ToastItem({ t }: { t: Toast }) {
  const ctl = useController();
  useEffect(() => {
    // errors stay until dismissed; info fades
    if (t.tone !== "info") return;
    const h = setTimeout(() => ctl.dismissToast(t.id), 7000);
    return () => clearTimeout(h);
  }, [ctl, t.id, t.tone]);
  return (
    <div class={`toast ${t.tone}`} role={t.tone === "error" ? "alert" : "status"} data-testid="toast">
      <div class="row">
        <span class="t">{t.title}</span>
        <span class="spacer" />
        <button
          type="button"
          class="tiny"
          onClick={() => ctl.dismissToast(t.id)}
          aria-label={`dismiss ${t.title}`}
        >
          ✕
        </button>
      </div>
      <div>{t.body}</div>
    </div>
  );
}

export function Toasts() {
  const toasts = useApp((s) => s.toasts);
  return (
    <div class="toasts" aria-live="assertive">
      {toasts.map((t) => (
        <ToastItem key={t.id} t={t} />
      ))}
    </div>
  );
}
