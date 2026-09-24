import { useEffect, useState } from "preact/hooks";
import { saveLastSession } from "../config";
import { Composer } from "./Composer";
import { ConnectPanel } from "./ConnectPanel";
import { useApp } from "./context";
import { Deck } from "./Deck";
import { Header } from "./Header";
import { SessionPicker } from "./SessionPicker";
import { Toasts } from "./Toasts";
import { Transcript } from "./Transcript";
import { WearBar, WearControls } from "./Wear";

function Workspace() {
  const status = useApp((s) => s.sessionStatus);
  const err = useApp((s) => s.sessionError);
  if (status === "loading") return <div class="center empty">reading the session…</div>;
  if (status === "error") {
    return (
      <div class="center">
        <div class="errbox" role="alert">
          {err}
        </div>
      </div>
    );
  }
  return (
    <main class="work">
      <section class="col" aria-label="conversation">
        <Transcript />
        <Composer />
      </section>
      <section class="col" aria-label="loom deck">
        <Deck />
        <span class="spacer" />
        <WearControls />
      </section>
    </main>
  );
}

export function App() {
  const connected = useApp((s) => s.conn.status === "connected");
  const session = useApp((s) => s.session);
  const [settingsOpen, setSettingsOpen] = useState(false);
  useEffect(() => saveLastSession(session), [session]);
  return (
    <>
      <a class="sr-only" href="#composer">
        skip to composer
      </a>
      <Header onSettings={() => setSettingsOpen(!settingsOpen)} />
      <WearBar />
      {!connected || settingsOpen ? (
        <ConnectPanel {...(connected ? { onDone: () => setSettingsOpen(false) } : {})} />
      ) : session ? (
        <Workspace />
      ) : (
        <SessionPicker />
      )}
      <Toasts />
    </>
  );
}
