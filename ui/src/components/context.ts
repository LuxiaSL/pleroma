import { createContext } from "preact";
import { useContext } from "preact/hooks";
import type { AppController, AppState } from "../store/app";
import { useStore } from "../store/useStore";

export const AppContext = createContext<AppController | null>(null);

export function useController(): AppController {
  const c = useContext(AppContext);
  if (!c) throw new Error("useController outside <AppContext.Provider>");
  return c;
}

/** Select a slice of app state; re-renders only when it changes. */
export function useApp<T>(select: (s: AppState) => T): T {
  return useStore(useController().store, select);
}
