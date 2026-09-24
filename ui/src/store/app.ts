/**
 * The vertical slice's state and its verbs: connect → pick a session → talk
 * on either branch → draw a loom (with live progress) → wear / unwear.
 *
 * Framework-free on purpose: the views render `AppState` and call the
 * controller; the unit tests drive the controller against a fake client.
 * Every server answer is adopted from the TYPED response — the UI never
 * claims a state (a wear, a draw) the server did not report.
 */
import { LoomClient } from "../api/client";
import { ApiError, describeError } from "../api/errors";
import type {
  Branch,
  Future,
  HistoryMessage,
  InfoResponse,
  LoomBody,
  LoomProgress,
  LoomResponse,
  SessionRow,
  StateResponse,
  WearResponse,
  WornPublic,
} from "../api/types";
import type { ConnectionSettings } from "../config";
import type { CodeKind } from "../lib/codespace";
import { bandStartAlpha, doseZones, effectiveDose, zoneFor } from "../lib/dose";
import { fmt } from "../lib/format";
import { createStore, type Store } from "./store";

export type ConnStatus = "idle" | "connecting" | "connected" | "failed";

/** A transcript line: a server history message, or a local event line (`event`). */
export interface Msg {
  id: number;
  role: HistoryMessage["role"] | "event";
  content: string;
  dream?: string | null;
  /** assistant replies on the loom branch: the wear they ran under (from /chat's `worn`) */
  worn?: WornPublic | null;
  elapsed_s?: number;
  pending?: boolean;
}

export interface Toast {
  id: number;
  title: string;
  body: string;
  tone: "error" | "info";
}

export interface FanMeta {
  loomId: string;
  n_futures: number;
  horizon: number;
  timing_s: LoomResponse["timing_s"];
  drawnUnderWear: LoomResponse["drawn_under_wear"];
  detachWear: boolean;
  spread: LoomResponse["spread"];
  harvestVia: string;
  /** the contemplated turn the fan was drawn for (never committed) */
  text: string | null;
  /** true once a turn was committed after the draw: the futures answer an earlier context */
  stale: boolean;
}

export interface LoomRun {
  running: boolean;
  startedAt: number | null;
  progress: LoomProgress | null;
  plan: { k: number; horizon: number } | null;
  text: string | null;
}

export type SessionStatus = "none" | "loading" | "ready" | "new" | "error";

export interface AppState {
  settings: ConnectionSettings;
  conn: {
    status: ConnStatus;
    info: InfoResponse | null;
    error: string | null;
    schemaMatches: boolean | null;
  };
  sessions: { rows: SessionRow[]; loading: boolean; error: string | null };
  session: string | null;
  sessionStatus: SessionStatus;
  sessionError: string | null;
  histories: Record<Branch, Msg[]>;
  view: Branch | "split";
  mirror: boolean;
  worn: WornPublic | null;
  lastWear: WearResponse | null;
  futures: Future[];
  fan: FanMeta | null;
  loom: LoomRun;
  busy: null | "chat" | "wear" | "unwear" | "session";
  /** slot (position in `futures`) of the selected future */
  selected: number | null;
  alpha: number;
  freshDraw: boolean;
  codeKind: CodeKind;
  toasts: Toast[];
}

export const IDLE_LOOM: LoomRun = { running: false, startedAt: null, progress: null, plan: null, text: null };

export function initialState(settings: ConnectionSettings): AppState {
  return {
    settings,
    conn: { status: "idle", info: null, error: null, schemaMatches: null },
    sessions: { rows: [], loading: false, error: null },
    session: null,
    sessionStatus: "none",
    sessionError: null,
    histories: { loom: [], base: [] },
    view: "loom",
    mirror: false,
    worn: null,
    lastWear: null,
    futures: [],
    fan: null,
    loom: IDLE_LOOM,
    busy: null,
    selected: null,
    alpha: 0.5,
    freshDraw: true,
    codeKind: "code",
    toasts: [],
  };
}

let nextId = 1;
const newId = (): number => nextId++;

export function historyToMsgs(h: readonly HistoryMessage[] | undefined): Msg[] {
  return (h ?? []).map((m) => ({ id: newId(), role: m.role, content: m.content, dream: m.dream ?? null }));
}

/** A session name the server will accept as a tag (the API requires minLength 1). */
export function validateSessionName(name: string): string | null {
  const v = name.trim();
  if (!v) return "a session needs a name";
  if (v.length > 120) return "keep it under 120 characters";
  if (/[\s/\\]/.test(v)) return "no spaces or slashes — it becomes a snapshot filename on the server";
  return null;
}

export interface ControllerOptions {
  makeClient?: (s: ConnectionSettings) => LoomClient;
  /** /loom/progress poll period */
  pollMs?: number;
  now?: () => number;
}

export class AppController {
  readonly store: Store<AppState>;
  private client: LoomClient | null = null;
  private readonly makeClient: (s: ConnectionSettings) => LoomClient;
  private readonly pollMs: number;
  private readonly now: () => number;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private loomAbort: AbortController | null = null;
  /** bumps on every session switch so late answers for the old session are dropped */
  private epoch = 0;

  constructor(store: Store<AppState>, opts: ControllerOptions = {}) {
    this.store = store;
    this.makeClient =
      opts.makeClient ?? ((s) => new LoomClient({ baseUrl: s.baseUrl, token: s.token || null }));
    this.pollMs = opts.pollMs ?? 1000;
    this.now = opts.now ?? (() => Date.now());
  }

  static create(settings: ConnectionSettings, opts?: ControllerOptions): AppController {
    return new AppController(createStore(initialState(settings)), opts);
  }

  get state(): AppState {
    return this.store.get();
  }

  // ── toasts ────────────────────────────────────────────────────────────────

  toast(title: string, body: string, tone: Toast["tone"] = "error"): number {
    const id = newId();
    this.store.set((s) => ({ toasts: [...s.toasts.slice(-4), { id, title, body, tone }] }));
    return id;
  }

  dismissToast(id: number): void {
    this.store.set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  }

  private fail(title: string, err: unknown): void {
    this.toast(title, describeError(err));
  }

  private event(branch: Branch, content: string): void {
    this.store.set((s) => ({
      histories: {
        ...s.histories,
        [branch]: [...s.histories[branch], { id: newId(), role: "event", content }],
      },
    }));
  }

  // ── connection ────────────────────────────────────────────────────────────

  async connect(settings?: ConnectionSettings): Promise<boolean> {
    const cfg = settings ?? this.state.settings;
    this.store.set({
      settings: cfg,
      conn: { status: "connecting", info: null, error: null, schemaMatches: null },
    });
    let client: LoomClient;
    try {
      client = this.makeClient(cfg);
    } catch (err) {
      this.store.set({
        conn: { status: "failed", info: null, error: describeError(err), schemaMatches: null },
      });
      return false;
    }
    try {
      const res = await client.connect();
      this.client = client;
      const zones = doseZones(res.info.dose_band);
      const start = bandStartAlpha(zones);
      this.store.set({
        conn: { status: "connected", info: res.info, error: null, schemaMatches: res.schemaMatches },
        alpha: start ?? this.state.alpha,
      });
      void this.refreshSessions();
      return true;
    } catch (err) {
      this.client = null;
      this.store.set({
        conn: { status: "failed", info: null, error: describeError(err), schemaMatches: null },
      });
      return false;
    }
  }

  private api(): LoomClient {
    if (!this.client)
      throw new ApiError({ kind: "network", message: "not connected — connect to a server first" });
    return this.client;
  }

  async refreshSessions(): Promise<void> {
    this.store.set((s) => ({ sessions: { ...s.sessions, loading: true, error: null } }));
    try {
      const r = await this.api().sessions();
      const rows = [...r.sessions].sort((a, b) => b.modified_unix - a.modified_unix);
      this.store.set({ sessions: { rows, loading: false, error: null } });
    } catch (err) {
      this.store.set((s) => ({ sessions: { ...s.sessions, loading: false, error: describeError(err) } }));
    }
  }

  // ── sessions ──────────────────────────────────────────────────────────────

  private resetSessionView(name: string | null, status: SessionStatus): void {
    this.stopPolling();
    this.loomAbort?.abort();
    this.loomAbort = null;
    this.store.set({
      session: name,
      sessionStatus: status,
      sessionError: null,
      histories: { loom: [], base: [] },
      worn: null,
      lastWear: null,
      futures: [],
      fan: null,
      loom: IDLE_LOOM,
      selected: null,
      busy: null,
    });
  }

  /** Adopt a /state (or /restore-then-/state) answer wholesale. */
  adoptState(r: StateResponse): void {
    const fan: FanMeta | null = r.loom_id
      ? {
          loomId: r.loom_id,
          n_futures: r.futures.length,
          horizon: Math.max(0, ...r.futures.map((f) => f.n_tokens)),
          timing_s: { generate: Number.NaN, harvest: Number.NaN },
          drawnUnderWear: null,
          detachWear: false,
          spread: null,
          harvestVia: "restored from /state",
          text: null,
          stale: false,
        }
      : null;
    this.store.set({
      sessionStatus: "ready",
      histories: { loom: historyToMsgs(r.histories.loom), base: historyToMsgs(r.histories.base) },
      worn: r.worn,
      futures: r.futures,
      fan,
      selected: null,
      loom: r.loom_in_progress
        ? { running: true, startedAt: this.now(), progress: r.loom_in_progress, plan: null, text: null }
        : IDLE_LOOM,
    });
    if (r.loom_in_progress) this.startPolling(true);
  }

  /** Open a session: /state if the server has it in memory; a NEW session otherwise. */
  async openSession(name: string): Promise<void> {
    const bad = validateSessionName(name);
    if (bad) {
      this.toast("session", bad);
      return;
    }
    const tag = name.trim();
    const epoch = ++this.epoch;
    this.resetSessionView(tag, "loading");
    try {
      const r = await this.api().state(tag);
      if (epoch !== this.epoch) return;
      this.adoptState(r);
    } catch (err) {
      if (epoch !== this.epoch) return;
      if (err instanceof ApiError && err.is("no_session")) {
        // v1 never creates on read; the first SEND or LOOM will.
        this.store.set({ sessionStatus: "new" });
        return;
      }
      this.store.set({ sessionStatus: "error", sessionError: describeError(err) });
    }
  }

  /** Load an on-disk snapshot into memory (POST /restore), then open it. */
  async restoreSession(name: string): Promise<void> {
    const epoch = ++this.epoch;
    this.resetSessionView(name, "loading");
    try {
      await this.api().restore(name);
      if (epoch !== this.epoch) return;
      const r = await this.api().state(name);
      if (epoch !== this.epoch) return;
      this.adoptState(r);
      void this.refreshSessions();
    } catch (err) {
      if (epoch !== this.epoch) return;
      this.store.set({ sessionStatus: "error", sessionError: describeError(err) });
    }
  }

  closeSession(): void {
    this.epoch++;
    this.resetSessionView(null, "none");
    void this.refreshSessions();
  }

  // ── conversation ──────────────────────────────────────────────────────────

  /** Commit a user turn on `branch` (and on base too when mirroring). True on success. */
  async send(text: string, branch: Branch = "loom"): Promise<boolean> {
    const session = this.state.session;
    const t = text.trim();
    if (!session || !t || this.state.busy || this.state.loom.running) return false;
    const epoch = this.epoch;
    const branches: Branch[] = branch === "loom" && this.state.mirror ? ["loom", "base"] : [branch];
    this.store.set({ busy: "chat" });
    let ok = true;
    try {
      for (const b of branches) {
        const userMsg: Msg = { id: newId(), role: "user", content: t };
        const pending: Msg = { id: newId(), role: "assistant", content: "", pending: true };
        this.store.set((s) => ({
          histories: { ...s.histories, [b]: [...s.histories[b], userMsg, pending] },
        }));
        try {
          const r = await this.api().chat({ session, branch: b, text: t });
          if (epoch !== this.epoch) return false;
          const reply: Msg = {
            id: pending.id,
            role: "assistant",
            content: r.reply,
            dream: r.dream ?? null,
            worn: b === "loom" ? r.worn : null,
            elapsed_s: r.elapsed_s,
          };
          this.store.set((s) => ({
            histories: { ...s.histories, [b]: s.histories[b].map((m) => (m.id === pending.id ? reply : m)) },
            sessionStatus: "ready",
            worn: b === "loom" ? r.worn : s.worn,
            fan: s.fan ? { ...s.fan, stale: true } : s.fan,
          }));
          if (r.dropped_messages > 0) {
            this.event(
              b,
              `the history window dropped ${r.dropped_messages} oldest message(s) from the prompt`,
            );
          }
        } catch (err) {
          if (epoch !== this.epoch) return false;
          this.store.set((s) => ({
            histories: {
              ...s.histories,
              [b]: s.histories[b].filter((m) => m.id !== userMsg.id && m.id !== pending.id),
            },
          }));
          this.fail(`/chat (${b}) failed`, err);
          ok = false;
          break;
        }
      }
    } finally {
      if (epoch === this.epoch) this.store.set({ busy: null });
    }
    return ok;
  }

  // ── the loom ──────────────────────────────────────────────────────────────

  private startPolling(foreign = false): void {
    this.stopPolling();
    const session = this.state.session;
    if (!session) return;
    const epoch = this.epoch;
    const tick = async (): Promise<void> => {
      if (epoch !== this.epoch || !this.state.loom.running) return;
      try {
        const r = await this.api().progress(session);
        if (epoch !== this.epoch || !this.state.loom.running) return;
        if (r.progress) {
          this.store.set((s) => ({ loom: { ...s.loom, progress: r.progress } }));
        } else if (foreign) {
          // a draw someone else started has finished: pick up its futures
          this.store.set({ loom: IDLE_LOOM });
          const st = await this.api().state(session);
          if (epoch === this.epoch) this.adoptState(st);
          return;
        }
      } catch {
        // progress is a nicety; a failed poll never breaks the draw itself
      }
      if (epoch === this.epoch && this.state.loom.running) this.pollTimer = setTimeout(tick, this.pollMs);
    };
    this.pollTimer = setTimeout(tick, 0);
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }

  cancelLoom(): void {
    this.loomAbort?.abort();
  }

  /**
   * Draw K futures for a contemplated turn (nothing is committed). With
   * "fresh draw" on, a current wear comes off first — its imprint is already
   * in the text it wrote; if that unwear fails the draw runs with the wear
   * DETACHED instead, never under it by accident (the old UI's rule).
   */
  async drawLoom(text: string, k: number, horizon: number): Promise<boolean> {
    const session = this.state.session;
    const t = text.trim();
    if (!session || !t || this.state.busy || this.state.loom.running) return false;
    const epoch = this.epoch;
    const kk = Math.max(2, Math.min(32, Math.round(k)));
    const hh = Math.max(16, Math.min(1024, Math.round(horizon)));

    let detachWear = false;
    const worn = this.state.worn;
    if (worn && this.state.freshDraw) {
      try {
        const u = await this.api().unwear(session);
        if (epoch !== this.epoch) return false;
        this.store.set({ worn: u.worn, lastWear: null });
        this.event(
          "loom",
          `↯ unwore code #${worn.index} before drawing — its imprint stays in the text it wrote`,
        );
      } catch (err) {
        detachWear = true;
        this.toast(
          "fresh draw",
          `/unwear failed (${describeError(err)}) — drawing with the wear detached instead`,
          "info",
        );
      }
    }

    const ctl = new AbortController();
    this.loomAbort = ctl;
    this.store.set({
      loom: { running: true, startedAt: this.now(), progress: null, plan: { k: kk, horizon: hh }, text: t },
      futures: [],
      fan: null,
      selected: null,
    });
    this.startPolling();
    try {
      const body: LoomBody = { session, text: t, k: kk, horizon: hh };
      if (detachWear) body.detach_wear = true;
      const r = await this.api().loom(body, { signal: ctl.signal });
      if (epoch !== this.epoch) return false;
      this.adoptLoom(r, t);
      if (r.n_futures !== r.futures.length) {
        this.toast("loom", `n_futures=${r.n_futures} but ${r.futures.length} futures returned`, "info");
      }
      return true;
    } catch (err) {
      if (epoch !== this.epoch) return false;
      this.store.set({ loom: IDLE_LOOM });
      this.fail("/loom failed", err);
      return false;
    } finally {
      if (this.loomAbort === ctl) this.loomAbort = null;
      if (epoch === this.epoch) this.stopPolling();
    }
  }

  adoptLoom(r: LoomResponse, text: string | null): void {
    this.store.set({
      sessionStatus: "ready",
      loom: IDLE_LOOM,
      futures: r.futures,
      worn: r.worn,
      selected: null,
      fan: {
        loomId: r.loom_id,
        n_futures: r.n_futures,
        horizon: r.horizon,
        timing_s: r.timing_s,
        drawnUnderWear: r.drawn_under_wear,
        detachWear: r.detach_wear,
        spread: r.spread,
        harvestVia: r.harvest_via,
        text,
        stale: false,
      },
    });
  }

  select(slot: number | null): void {
    if (slot !== null && (slot < 0 || slot >= this.state.futures.length)) return;
    this.store.set({ selected: slot === this.state.selected ? null : slot });
  }

  setAlpha(alpha: number): void {
    if (Number.isFinite(alpha) && alpha >= 0) this.store.set({ alpha });
  }

  // ── wear ──────────────────────────────────────────────────────────────────

  /** Wear the selected (or given) future at the knob's alpha, scoped to this draw. */
  async wear(slot: number | null = this.state.selected, alpha: number = this.state.alpha): Promise<boolean> {
    const session = this.state.session;
    const f = slot === null ? undefined : this.state.futures[slot];
    if (!session || !f || this.state.busy || this.state.loom.running) return false;
    if (!f.harvested) {
      this.toast("wear", `future #${f.index} was not harvested — it has no code to wear`);
      return false;
    }
    const epoch = this.epoch;
    this.store.set({ busy: "wear" });
    try {
      const loomId = this.state.fan?.loomId ?? null;
      const r = await this.api().wear({
        session,
        index: f.index,
        alpha,
        ...(loomId ? { loom_id: loomId } : {}),
      });
      if (epoch !== this.epoch) return false;
      this.store.set({ worn: r.worn, lastWear: r });
      this.event("loom", wearLine(r, this.state.conn.info));
      return true;
    } catch (err) {
      if (epoch !== this.epoch) return false;
      this.fail("/wear failed", err);
      if (err instanceof ApiError && (err.is("stale_loom") || err.is("no_candidates"))) {
        this.toast("wear", "this draw is no longer the session's latest — re-reading server state", "info");
        const st = await this.api()
          .state(session)
          .catch(() => null);
        if (st && epoch === this.epoch) this.adoptState(st);
      }
      return false;
    } finally {
      if (epoch === this.epoch) this.store.set({ busy: null });
    }
  }

  async unwear(): Promise<boolean> {
    const session = this.state.session;
    if (!session || this.state.busy || this.state.loom.running) return false;
    const had = this.state.worn;
    const epoch = this.epoch;
    this.store.set({ busy: "unwear" });
    try {
      const r = await this.api().unwear(session);
      if (epoch !== this.epoch) return false;
      this.store.set({ worn: r.worn, lastWear: null });
      if (had) this.event("loom", `↯ unwore code #${had.index} — conversation running straight again`);
      return true;
    } catch (err) {
      if (epoch === this.epoch) this.fail("/unwear failed", err);
      return false;
    } finally {
      if (epoch === this.epoch) this.store.set({ busy: null });
    }
  }
}

/** The transcript line a wear leaves, worded like loom_ui.html's. */
export function wearLine(r: WearResponse, info: InfoResponse | null): string {
  const w = r.worn;
  const eff = effectiveDose(w);
  const zone = zoneFor(eff.alphaWorst, doseZones(info?.dose_band));
  const scaled = eff.scaled
    ? ` (dose_policy ${eff.policy}: ×${fmt(eff.scale, 3)} ⇒ effective α ${fmt(eff.alpha, 3)}` +
      `${Math.abs(eff.alphaWorst - eff.alpha) > 1e-6 ? `, loudest site ${fmt(eff.alphaWorst, 3)}` : ""}` +
      `${eff.clamped ? ", CLAMPED" : ""})`
    : "";
  const kind = w.lever_kind ? ` (${w.lever_kind})` : "";
  return `↯ wore member code #${w.index}${kind} at α ${fmt(w.alpha, 3)}${scaled} — ${zone.label}.`;
}
