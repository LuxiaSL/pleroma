# ui/ — the loom's standalone front-end

A Vite + TypeScript (strict) + Preact app that drives the loom server over its
versioned API (`/api/v1`, see `../docs/API.md`). It is hosted apart from the
inference server — or served by it with `--ui-dir ui/dist` — and grows toward
parity with the single-file UI (`pleroma/serve/static/legacy_ui.html`), which
stays served and working unchanged until then.

## Run it

```sh
cd ui
npm ci

npm run dev:mock     # no GPU, no server: /api/v1 answered from ui/fixtures/ → http://localhost:5173
npm run dev          # proxy /api/v1 → $PLEROMA_API_TARGET (default http://127.0.0.1:8767)
```

**Against a live loom.** Either:

1. **Dev server + proxy** (recommended while developing). Tunnel or run the
   loom on `127.0.0.1:8767`, then `npm run dev`. The proxy strips the browser's
   `Origin`, so the server needs no `--cors-origin`; if the server has a token,
   export `PLEROMA_API_TOKEN` in the shell running `npm run dev` and the proxy
   adds the bearer header (the token never reaches the browser).
   `PLEROMA_API_TARGET=http://127.0.0.1:8795 npm run dev` points it elsewhere.
2. **Same origin, built.** `npm run build`, then start the server with
   `--ui-dir <abs path>/ui/dist`. `python -m pleroma serve` renders its argv
   from the profile + deployment only and has no field for the API flags yet,
   so take its command and append them:
   ```sh
   CMD=$(python -m pleroma serve --profile P --deploy D --dry-run)
   $CMD --ui-dir "$PWD/ui/dist"
   ```
   Open the server's own URL; leave the settings panel's *server* empty.
3. **Cross-origin** (the UI hosted anywhere else). Append
   `--cors-origin http://localhost:5173` (the UI's exact origin) the same way,
   with a token in the environment (`export PLEROMA_API_TOKEN=$(openssl rand
   -hex 16)` — the environment variable, not the flag, which `ps` shows), then
   put the server URL and token in the UI's *settings* panel. Note that a token plus
   `--cors-origin` also closes the legacy routes, so `loom_ui.html` stops
   working against that server (docs/API.md, "Auth and CORS").

Settings resolve as: settings panel (this browser's localStorage) → env
(`VITE_LOOM_API_BASE`, `VITE_LOOM_API_TOKEN` in `ui/.env.local`, gitignored) →
same origin, no token. No host is hardcoded.

## Scripts

| script | what |
|---|---|
| `dev` / `dev:mock` | dev server, proxied to a loom / served from fixtures (`MOCK_LOOM_SECONDS`, `MOCK_API_TOKEN` tune the mock) |
| `build` | typecheck, then `dist/` (relative asset URLs, so `/` and `/ui` both work) |
| `typecheck` | `tsc` over the app (`tsconfig.json`) and the node side: configs, mock, tests (`tsconfig.node.json`) |
| `gen:api` | regenerate `src/api/schema.d.ts` + `schema-meta.ts` from `../pleroma/serve/api-schema.json` |
| `check:api` | fail if those two files are stale (writes nothing) |
| `test` | vitest: client error mapping, PCA vs known answers, store/controller against the mock, mock ⊂ schema |
| `e2e` | Playwright over the mock, desktop (dark) + 400 px (light); `UI_SHOTS_DIR=… npm run e2e` keeps screenshots |
| `lint` / `format` | Biome |

The Python suite runs `check:api` and `typecheck` too (`tests/unit/test_ui_frontend.py`,
skipped without node or `ui/node_modules`), and validates `ui/fixtures/`
against the pydantic response models in strict mode — so a server-side contract
change that the UI or its mock no longer fits fails `pytest`.

## Architecture

```
src/api/        schema.d.ts, schema-meta.ts   GENERATED (gen:api) — never edit
                types.ts                      named aliases into the generated types
                errors.ts                     ApiError: {error, code} → kind/status/code; unknown codes kept raw
                client.ts                     LoomClient: base URL + bearer, deadlines, connect() version check
src/store/      store.ts, useStore.ts         a tiny typed store + a Preact subscription hook
                app.ts                        AppState + AppController: the slice's verbs (framework-free)
src/lib/        pca.ts, codespace.ts          Jacobi PCA of a fan's centred codes → 3-D placement
                pull.ts, dose.ts, wear.ts     pull, dose zones / effective alpha, "is this the worn one"
src/components/                               Preact views; read state via useApp(selector)
src/views/      CodeSpace3D.tsx, codeSpaceScene.ts   the three.js view (lazy-loaded chunk)
mock/backend.ts                               the mock /api/v1 (a Vite middleware; also used in-process by tests)
fixtures/                                     recorded live responses, fitted to the v1 models
scripts/        gen-api.mjs, build_fixtures.py
```

**Why Preact (+ a hand-rolled store), not vanilla or a big framework.** The old
page is 7k lines of imperative DOM, and most of its bugs were a view and the
state disagreeing. JSX with keyed diffing makes the deck/transcript
declarative for ~4 kB; there is no compiler magic and no runtime to learn. The
store is 40 lines and framework-free, so the controller (every API verb) is
unit-tested without a DOM; three.js lives in a plain class a component owns.

**The typed client.** `openapi-typescript` turns the committed OpenAPI
document into `schema.d.ts`; `client.ts` binds each route to its operation's
request/response types (`OkJson<Ops["postWear"]>`). `connect()` reads
`/info`: a server without `/api/v1` or with another major version is refused
with a message saying so; a different `schema_sha256` (same major version) is
flagged in the header, not refused — v1 is additive. `ERROR_CODES` mirrors the
generated enum, and a type-level check fails `typecheck` if the schema grows a
code the list lacks.

**The mock.** Every future's text/code/scores, the dose band and `/info` are
real (recorded from the live 8B v1a loom, rank-64 map, measured band). Chat
replies are canned and say so; a new draw is the recorded fan re-indexed and
cut to *k*; `/loom/progress` replays the server's shape (generate 0→k in one
jump, then harvest branch by branch). `tests/unit/mock-contract.test.ts`
validates its bodies against `api-schema.json`. Refresh the fixtures from the
recordings with `python ui/scripts/build_fixtures.py` (it fits them to the
models, redacts private paths, and checks them strictly).

## Adding a view

1. Put pure logic in `src/lib/` with a vitest test beside the others.
2. If it needs a new route, the route is already typed: add a method to
   `LoomClient` (one line, using `OkJson`/`JsonBody` of the operation) and a
   verb to `AppController` that adopts the typed answer into `AppState`.
3. Write the component under `src/components/` (or `src/views/` for canvas /
   three.js work — lazy-load anything heavy with `lazy(() => import(...))`),
   reading state with `useApp((s) => s.slice)`; select stable references.
4. Colours come from the CSS tokens in `src/styles/theme.css` (canvas code
   reads them with `getComputedStyle` and listens for the `loom-theme` event);
   honour `prefers-reduced-motion`.
5. Extend the mock if the route is new, and add an e2e step.

## The first slice vs the old UI

In: connect (with version/contract check) · a session picker (the old UI has
none) incl. restoring on-disk snapshots · loom/base/split transcripts ·
composer (send, mirror to base, fresh draw, k, horizon, ⌘↵ / ⌘⇧↵) · LOOM with
live `/loom/progress` (stage, per-branch harvest slots, stop waiting) · the
deck (pull, distinct, ‖raw‖, gauge, cos→worn, cut-at-horizon, alone) · dose knob
over the map's measured zones with tier badge and the predicted-dose estimate ·
wear/unwear with the server's receipt · the wear bar (incl. INERT) · the code
space in 3-D · light/dark/auto · keyboard focus · 400 px.

Not yet (see the porting order in the handoff / PR): auto-wear policies, the
gauge probe row, dose-policy and lever-kind controls, reroll/edit/undo/truncate,
"what changed" in split view, fan spread strip, the provenance inspector, the
pool/rig telemetry, reset.
