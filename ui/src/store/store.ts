/**
 * A tiny typed store: one immutable state object, shallow-merged updates,
 * synchronous subscribers. No framework in here — views subscribe through
 * `useStore` (./useStore.ts); the controller and the tests use it directly.
 */

export type Listener<S> = (state: S, prev: S) => void;
export type Updater<S> = Partial<S> | ((state: S) => Partial<S>);

export interface Store<S extends object> {
  get(): S;
  set(update: Updater<S>): void;
  subscribe(fn: Listener<S>): () => void;
}

export function createStore<S extends object>(initial: S): Store<S> {
  let state = initial;
  const listeners = new Set<Listener<S>>();
  return {
    get: () => state,
    set(update) {
      const patch = typeof update === "function" ? update(state) : update;
      let changed = false;
      for (const key of Object.keys(patch) as (keyof S)[]) {
        if (!Object.is(state[key], patch[key])) {
          changed = true;
          break;
        }
      }
      if (!changed) return;
      const prev = state;
      state = { ...state, ...patch };
      for (const fn of [...listeners]) {
        try {
          fn(state, prev);
        } catch (err) {
          // one broken view must not stop the others from rendering
          console.error("store listener failed", err);
        }
      }
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
