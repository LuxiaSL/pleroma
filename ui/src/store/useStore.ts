import { useEffect, useRef, useState } from "preact/hooks";
import type { Store } from "./store";

/**
 * Subscribe a component to a slice of a store. Re-renders only when the
 * selected value changes (Object.is), so a progress tick does not repaint
 * the transcript.
 */
export function useStore<S extends object, T>(store: Store<S>, select: (s: S) => T): T {
  const selectRef = useRef(select);
  selectRef.current = select;
  const [value, setValue] = useState<T>(() => select(store.get()));
  useEffect(() => {
    // catch an update that landed between render and subscribe
    setValue(() => selectRef.current(store.get()));
    return store.subscribe((s) => {
      const next = selectRef.current(s);
      setValue((cur) => (Object.is(cur, next) ? cur : next));
    });
  }, [store]);
  return value;
}
