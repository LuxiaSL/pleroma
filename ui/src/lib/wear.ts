import type { Future, WornPublic } from "../api/types";

/** Is `f` the future worn right now — same index AND (when the wear says) the same draw? */
export function isWornHere(worn: WornPublic | null, f: Future, loomId: string | null | undefined): boolean {
  if (!worn?.active || worn.index !== f.index) return false;
  return worn.loom_id === null || worn.loom_id === loomId;
}
