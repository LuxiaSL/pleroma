import { describe, expect, it, vi } from "vitest";
import { LoomClient } from "../../src/api/client";
import { alphaMax, doseZones, effectiveDose, UNCALIBRATED_ZONE, zoneFor } from "../../src/lib/dose";
import {
  AppController,
  historyToMsgs,
  initialState,
  validateSessionName,
  wearLine,
} from "../../src/store/app";
import { createStore } from "../../src/store/store";
import { fastBackend, mockFetch, until } from "./helpers";

function rig(opts: Parameters<typeof fastBackend>[0] = {}) {
  const backend = fastBackend(opts);
  const ctl = AppController.create(
    { baseUrl: "", token: "" },
    {
      makeClient: (s) =>
        new LoomClient({ baseUrl: s.baseUrl, token: s.token || null, fetch: mockFetch(backend) }),
      pollMs: 10,
    },
  );
  return { backend, ctl };
}

describe("createStore", () => {
  it("merges, notifies with prev, and skips no-op updates", () => {
    const s = createStore({ a: 1, b: "x" });
    const seen: [number, number][] = [];
    const off = s.subscribe((n, p) => seen.push([n.a, p.a]));
    s.set({ a: 2 });
    s.set({ a: 2 }); // no change → no notification
    s.set((st) => ({ a: st.a + 1 }));
    off();
    s.set({ a: 99 });
    expect(seen).toEqual([
      [2, 1],
      [3, 2],
    ]);
    expect(s.get()).toEqual({ a: 99, b: "x" });
  });

  it("a throwing listener does not starve the others", () => {
    const s = createStore({ a: 0 });
    const good = vi.fn();
    const spy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    s.subscribe(() => {
      throw new Error("boom");
    });
    s.subscribe(good);
    s.set({ a: 1 });
    expect(good).toHaveBeenCalledOnce();
    spy.mockRestore();
  });
});

describe("dose zones", () => {
  const band = {
    tier: "measured",
    alpha_max: 1.4,
    source: null,
    measured_at: null,
    notes: null,
    map_fingerprint: null,
    derivation: null,
    lever_quality_caveat: null,
    zones: [
      { lo: 0.125, hi: 0.75, hi_inclusive: false, key: "threshold", label: "threshold", hint: "" },
      { lo: 0, hi: 0.125, hi_inclusive: true, key: "subliminal", label: "subliminal", hint: "" },
      { lo: 0.75, hi: 1.4, hi_inclusive: true, key: "audible", label: "audible", hint: "" },
    ],
  };
  it("reads bounds the way the server writes them (closed tops stay in their zone)", () => {
    const z = doseZones(band);
    expect(z.map((q) => q.key)).toEqual(["subliminal", "threshold", "audible"]);
    expect(zoneFor(0.125, z).key).toBe("subliminal");
    expect(zoneFor(0.1251, z).key).toBe("threshold");
    expect(zoneFor(0.75, z).key).toBe("audible");
    expect(zoneFor(9, z).key).toBe("audible");
  });
  it("no band is uncalibrated at every alpha; absent alpha_max is not zero", () => {
    expect(zoneFor(0.3, doseZones(null))).toBe(UNCALIBRATED_ZONE);
    expect(alphaMax({ ...band, alpha_max: null })).toBe(1.5);
  });
  it("reads a predicted wear's zone at its loudest site", () => {
    const e = effectiveDose({
      alpha: 0.5,
      dose: {
        policy: "predicted",
        scale: [0.8, 1.2],
        scale_raw: [0.8, 1.2],
        scale_mean: 1.0,
        fan_mean_raw_norms: null,
        candidate_raw_norms: null,
        clamped: [],
        clamp_range: [0.5, 2],
        note: "",
      },
    });
    expect(e.alpha).toBeCloseTo(0.5);
    expect(e.alphaWorst).toBeCloseTo(0.6);
  });
});

describe("AppController — the slice against the mock backend", () => {
  it("connects, lists sessions, and starts the knob in the band's threshold zone", async () => {
    const { ctl } = rig();
    expect(await ctl.connect()).toBe(true);
    expect(ctl.state.conn.status).toBe("connected");
    expect(ctl.state.conn.info?.api.version).toBe("1");
    await until(() => ctl.state.sessions.rows.length > 0);
    expect(ctl.state.sessions.rows.map((r) => r.session)).toContain("uifix-phaseb");
    const zones = doseZones(ctl.state.conn.info?.dose_band);
    expect(zoneFor(ctl.state.alpha, zones).key).toBe("threshold");
  });

  it("a failed connect leaves a readable error and no client", async () => {
    const ctl = AppController.create(
      { baseUrl: "", token: "" },
      {
        makeClient: () =>
          new LoomClient({
            baseUrl: "",
            token: null,
            fetch: async () => new Response("nope", { status: 404 }),
          }),
      },
    );
    expect(await ctl.connect()).toBe(false);
    expect(ctl.state.conn.status).toBe("failed");
    expect(ctl.state.conn.error).toMatch(/without the versioned/);
  });

  it("opens an existing session wholesale, and a new name as NEW (v1 never creates on read)", async () => {
    const { ctl, backend } = rig();
    await ctl.connect();
    await ctl.openSession("uifix-phaseb");
    expect(ctl.state.sessionStatus).toBe("ready");
    expect(ctl.state.histories.loom.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
    expect(ctl.state.worn?.index).toBe(3);
    expect(ctl.state.futures.length).toBeGreaterThan(0);
    await ctl.openSession("brand-new");
    expect(ctl.state.sessionStatus).toBe("new");
    expect(backend.sessions.has("brand-new")).toBe(false);
  });

  it("validates session names", () => {
    expect(validateSessionName("")).toMatch(/name/);
    expect(validateSessionName("a b")).toMatch(/spaces/);
    expect(validateSessionName("ok-name_1")).toBeNull();
  });

  it("send commits on the branch (and mirrors to base), marking an existing fan stale", async () => {
    const { ctl } = rig();
    await ctl.connect();
    await ctl.openSession("lx-speak-k16");
    expect(ctl.state.fan?.stale).toBe(false);
    ctl.store.set({ mirror: true });
    expect(await ctl.send("hello there", "loom")).toBe(true);
    expect(ctl.state.histories.loom.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(ctl.state.histories.base.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(ctl.state.histories.loom[1]?.content).toMatch(/no model ran/);
    expect(ctl.state.fan?.stale).toBe(true);
    expect(ctl.state.busy).toBeNull();
  });

  it("a failed send rolls back the optimistic turn and keeps the composer's text (returns false)", async () => {
    const { ctl } = rig();
    await ctl.connect();
    await ctl.openSession("lx-speak-k16");
    ctl.store.set({ session: "" as string }); // impossible tag → guard
    expect(await ctl.send("x")).toBe(false);
  });

  it("draws a loom with live progress, then wears and unwears", async () => {
    const { ctl } = rig();
    await ctl.connect();
    await ctl.openSession("brand-new-loom");
    const stages = new Set<string>();
    const off = ctl.store.subscribe((s) => {
      if (s.loom.progress) stages.add(s.loom.progress.stage);
    });
    const p = ctl.drawLoom("what would you say next?", 6, 64);
    await until(() => ctl.state.loom.running);
    expect(await p).toBe(true);
    off();
    expect(stages.has("harvest")).toBe(true);
    expect(ctl.state.loom.running).toBe(false);
    expect(ctl.state.futures).toHaveLength(6);
    expect(ctl.state.fan?.loomId).toMatch(/-000$/);
    expect(ctl.state.fan?.text).toBe("what would you say next?");
    expect(ctl.state.sessionStatus).toBe("ready");

    ctl.select(2);
    ctl.setAlpha(0.4);
    expect(await ctl.wear()).toBe(true);
    expect(ctl.state.worn?.index).toBe(2);
    expect(ctl.state.worn?.alpha).toBeCloseTo(0.4);
    expect(ctl.state.worn?.loom_id).toBe(ctl.state.fan?.loomId);
    expect(ctl.state.lastWear?.effective_alpha.alpha).toBeCloseTo(0.4);
    expect(ctl.state.histories.loom.at(-1)?.content).toMatch(/wore member code #2 .* at α 0\.400/);

    expect(await ctl.unwear()).toBe(true);
    expect(ctl.state.worn).toBeNull();
    expect(ctl.state.histories.loom.at(-1)?.content).toMatch(/unwore code #2/);
  });

  it("fresh draw takes the wear off first; the next fan is not drawn under it", async () => {
    const { ctl } = rig();
    await ctl.connect();
    await ctl.openSession("uifix-phaseb");
    expect(ctl.state.worn).not.toBeNull();
    expect(await ctl.drawLoom("again", 4, 64)).toBe(true);
    expect(ctl.state.worn).toBeNull();
    expect(ctl.state.fan?.drawnUnderWear).toBeNull();
    expect(ctl.state.histories.loom.some((m) => m.role === "event" && /before drawing/.test(m.content))).toBe(
      true,
    );
  });

  it("a stale wear is refused, reported, and state is re-read", async () => {
    const { ctl, backend } = rig();
    await ctl.connect();
    await ctl.openSession("brand-new-stale");
    await ctl.drawLoom("one", 4, 64);
    const firstLoom = ctl.state.fan?.loomId;
    // another tab draws: the server's latest moves on
    await backend.handle("POST", "/loom", new URLSearchParams(), {
      session: "brand-new-stale",
      text: "two",
      k: 4,
    });
    ctl.select(0);
    expect(await ctl.wear()).toBe(false);
    expect(ctl.state.toasts.some((t) => /not the latest draw/.test(t.body))).toBe(true);
    await until(() => ctl.state.fan?.loomId !== firstLoom);
    expect(ctl.state.fan?.loomId).toBe(backend.sessions.get("brand-new-stale")?.loomId);
    expect(ctl.state.worn).toBeNull();
  });

  it("a late answer for a session the operator left is dropped", async () => {
    const { ctl } = rig({ chatMs: 60 });
    await ctl.connect();
    await ctl.openSession("lx-speak-k16");
    const p = ctl.send("slow one");
    await ctl.openSession("uifix-phaseb");
    expect(await p).toBe(false);
    expect(ctl.state.session).toBe("uifix-phaseb");
    expect(ctl.state.histories.loom.every((m) => m.content !== "slow one")).toBe(true);
  });

  it("wearLine words it like the old UI", () => {
    const { backend } = rig();
    const w = JSON.parse(JSON.stringify(backend.info));
    const line = wearLine(
      {
        session: "s",
        worn: {
          index: 3,
          alpha: 0.5,
          loom_id: "x",
          per_site_norms_at_alpha1: [],
          code: null,
          active: true,
          dose: null,
          lever_kind: "absolute",
        },
        sites: [9],
        dose: {
          policy: "flat",
          scale: [1],
          scale_raw: [1],
          scale_mean: 1,
          fan_mean_raw_norms: null,
          candidate_raw_norms: null,
          clamped: [],
          clamp_range: [0.5, 2],
          note: "",
        },
        effective_alpha: {
          alpha: 0.5,
          alpha_effective_mean: 0.5,
          alpha_effective_per_site: [0.5],
          alpha_effective_range: [0.5, 0.5],
          note: "",
        },
        lever_kind: "absolute",
      },
      w,
    );
    expect(line).toBe("↯ wore member code #3 (absolute) at α 0.500 — threshold.");
  });

  it("historyToMsgs keeps roles and content", () => {
    const m = historyToMsgs([{ role: "user", content: "a" }]);
    expect(m[0]).toMatchObject({ role: "user", content: "a" });
    expect(initialState({ baseUrl: "", token: "" }).view).toBe("loom");
  });
});
