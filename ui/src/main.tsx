import { render } from "preact";
import { App } from "./components/App";
import { AppContext } from "./components/context";
import { loadLastSession, loadSettings } from "./config";
import { AppController } from "./store/app";
import { applyTheme, loadTheme } from "./theme";
import "./styles/theme.css";
import "./styles/app.css";

applyTheme(loadTheme());

const controller = AppController.create(loadSettings());
const root = document.getElementById("app");
if (!root) throw new Error("index.html has no #app");

render(
  <AppContext.Provider value={controller}>
    <App />
  </AppContext.Provider>,
  root,
);

// connect on load with the saved settings (default: same origin, no token);
// reopen the last session if the server still has it
void controller.connect().then(async (ok) => {
  const last = loadLastSession();
  if (ok && last) {
    await controller.openSession(last);
    // a remembered session that is gone is not worth an empty workspace
    if (controller.state.sessionStatus === "new" || controller.state.sessionStatus === "error")
      controller.closeSession();
  }
});

// debugging handle (and what the e2e reads the store through)
(globalThis as unknown as { __loom?: AppController }).__loom = controller;
