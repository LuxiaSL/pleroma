# The loom HTTP API — contract (v1)

The loom inference server (`python -m pleroma.serve`, `pleroma/serve/`) speaks a
typed, versioned JSON API for clients on **another origin** (the `ui/`
Vite/TypeScript app) and keeps its original un-prefixed routes for the
single-file `pleroma/serve/static/legacy_ui.html`, which must keep working unchanged.

Where this document and the code disagree, the code wins. The machine-readable
contract is `pleroma/serve/api-schema.json` (OpenAPI 3.1), also served live at
`GET /api/v1/schema`. The route table is `pleroma/serve/api.py`; the response
models are `pleroma/serve/responses.py`; the request (wire) models are the
`*Body` classes in `pleroma/serve/models.py`.

## Routes

Every route answers at `/api/v1<path>`. The un-prefixed `<path>` is a **legacy
alias** for the single-file UI, with the same behaviour plus the error `code`, the
parsed-path matching, and GET error handling below. Query strings are ignored for routing
everywhere.

| method | path (`/api/v1` +) | request | response model | mutating | notes |
|---|---|---|---|---|---|
| GET | `/info` | — | `InfoResponse` | | gains `api` (below) |
| GET | `/state?session=` | query | `StateResponse` | | **v1 never creates** the session: unknown → 404 `no_session`. The alias creates it, as always |
| GET | `/sessions` | — | `SessionsResponse` | | |
| GET | `/loom/progress?session=` | query | `ProgressResponse` | | v1 never creates (unknown → `progress: null`); the alias creates, as before |
| GET | `/pool` | — | `PoolResponse` | | 2 s cache |
| GET | `/atlas` | — | open object | | dead path, `deprecated` |
| GET | `/schema` | — | open object (OpenAPI 3.1) | | **v1 only**, no alias |
| POST | `/chat` | `ChatBody` | `ChatResponse` | ✓ | |
| POST | `/loom` | `LoomBody` | `LoomResponse` | ✓ | `auto: AutoSpec`, `probe: bool \| ProbeSpec` |
| POST | `/probe` | `ProbeBody` | `ProbeResponse` | ✓ | |
| POST | `/wear` | `WearBody` | `WearResponse` | ✓ | |
| POST | `/wear_code` | `WearCodeBody` | `WearCodeResponse` | ✓ | dead path, `deprecated` |
| POST | `/undo` | `BranchBody` | `UndoResponse` | ✓ | |
| POST | `/reroll` | `BranchBody` | `RerollResponse` | ✓ | |
| POST | `/edit` | `ChatBody` | `EditResponse` | ✓ | |
| POST | `/truncate` | `TruncateBody` | `TruncateResponse` | ✓ | |
| POST | `/unwear` | `SessionBody` | `UnwearResponse` | ✓ | |
| POST | `/reset` | `SessionBody` | `ResetResponse` | ✓ | |
| POST | `/restore` | `SessionBody` | `RestoreResponse` | | |

"Mutating" = a session snapshot is written after a successful request. Every
POST body needs `session`; a missing one is 400 `session_required` (checked
before the path, as it always was). `GET /`, `GET /ui` serve the UI (below)
and are not API routes.

**Request typing.** The server parses bodies with the coercion-exact
`from_blob` models, which are more lenient than the wire schema (`"k": "6"`
works, as it always has). Clients should send the schema's canonical types.
`auto` and the probe settings are read into `AutoSpec`/`ProbeSpec` at the same
point, with the same coercions, as before — so a malformed `auto.alpha` is
still refused after the draw, and a bad in-`/loom` probe setting still yields
`probe: {error, note}` beside a surviving draw.

**Response typing.** Each 200 body is validated against its model and then
sent **as the handler built it** — key order and number formatting are
byte-for-byte what they were. In production a body that does not fit its model
is logged (`CONTRACT VIOLATION`) and sent anyway: a paid-for GPU draw and an
already-persisted mutation are never turned into a 500 by a typing slip. With
`PLEROMA_API_STRICT=1` (the test suite's setting) a misfit is a 500, and so is
any difference between the model's dump and the body. Sub-shapes that are
genuinely open are typed `dict[str, Any]` and say so in the schema: `map_meta`,
the probe receipt's `resolution`, `harvest_worker_detail`, `/wear_code`'s
`source`, the dose band's `derivation`/`zones`/`notes`, `probe.cost_estimate`,
the `/info` policy/kind tables beyond `key`, and the `/atlas` report.

## `/info.api`

```json
"api": {"version": "1", "prefix": "/api/v1",
        "schema_sha256": "<sha256 of the schema's canonical JSON>",
        "auth_required": false, "legacy_aliases_open": true}
```

`schema_sha256` is over `json.dumps(schema, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`, UTF-8 encoded
(`pleroma.serve.schema.schema_sha256`). A client generated from `api-schema.json` can compare
it at startup to detect a server built from a different contract.

## Errors

Every non-2xx JSON body is

```json
{"error": "<the refusal, written for the operator — show it verbatim>",
 "code": "<machine-readable, below>"}
```

`error` is unchanged from the pre-API server (clients that read only `error`
see no difference). Status mapping: an `ApiError` carries its own status;
any other `ValueError`/`KeyError` → 400 (`KeyError` reads `"'index'"`);
anything else → 500 `"Type: message"`. GET now has the same handling as POST
(before, a throwing GET dropped the connection with no body).

| code | status | meaning |
|---|---|---|
| `bad_request` | 400 | malformed/refused, no more specific code |
| `not_found` | 404 | no route here; or `/atlas`, the UI, a static file is absent |
| `session_required` | 400 | no `session` in the body/query |
| `no_session` | 404 | `GET /api/v1/state` for a session not in memory |
| `unknown_branch` | 400 | `branch` is not `base`/`loom` |
| `stale_loom` | 400 | `/wear`'s `loom_id` is no longer the session's latest draw |
| `no_candidates` | 400 | `/wear` or `/probe` before any `/loom` |
| `gauge_needs_probe` | 400 | `auto.policy: "gauge"` on an unprobed fan |
| `lever_kind_invalid` | 400 | `lever_kind` is not `absolute`/`contrast` |
| `lever_kind_unavailable` | 400 | `contrast` on a draw with no contrast |
| `dose_policy_invalid` | 400 | `dose_policy` is not `flat`/`predicted` |
| `dose_policy_unavailable` | 400 | `predicted` with no fan mean / raw norms, or on `/wear_code` |
| `probe_prefix_mismatch` | 400 | `/probe`'s prefix is not the fan's |
| `no_snapshot` | 400 | `/restore` with no readable snapshot |
| `unauthorized` | 401 | missing/invalid bearer token (`WWW-Authenticate: Bearer`) |
| `forbidden_origin` | 403 | a browser Origin this server does not allow; any refused preflight |
| `internal` | 500 | unexpected; logged with a traceback |

Codes are raised where the refusal is decided (`pleroma.errors.CodedError`, a
`ValueError` subclass, in domain code; `pleroma.serve.errors.ApiError` in the
server). They replace the sentence regexes `loom_ui.html` uses (`/lever_kind/`,
`/dose_policy/`, `/gauge/`+`/probe/`, "not the latest draw"); the sentences are
unchanged so those regexes still match.

## Auth and CORS

Configured by three flags (all default off = the server behaves exactly as it
did for same-origin clients):

- `--cors-origin ORIGIN` (repeatable; `scheme://host[:port]`; `*` refused)
- `--api-token TOKEN`, or the environment variable **`PLEROMA_API_TOKEN`**
  (prefer it: a flag is visible to every user in `ps`). The flag wins.
- `--allow-lan` (pre-existing; binds beyond loopback)

**The rule** (`ApiSettings` in `pleroma/serve/api.py`):

1. **`/api/v1/*` requires `Authorization: Bearer <token>`** whenever a token is
   configured. Compared in constant time; never logged (the access log is the
   request line only).
2. **CORS is `/api/v1` only, and only for a configured origin.** For an allowed
   `Origin`, every response (errors included) carries
   `Access-Control-Allow-Origin: <origin>` and `Vary: Origin`; `OPTIONS` is a
   `204` with `Access-Control-Allow-Methods: GET, POST, OPTIONS`,
   `Access-Control-Allow-Headers: Content-Type, Authorization`,
   `Access-Control-Max-Age: 600`, and `Access-Control-Allow-Private-Network:
   true` when the preflight asks for it (Chrome's Private Network Access, for a
   public-origin page calling a loopback server). Any other preflight, and any
   request with a foreign `Origin`, is a 403 `forbidden_origin` with no CORS
   headers. Same-origin browser requests (whose `Origin` matches `Host`) and
   clients that send no `Origin` (curl, scripts) are unaffected.
3. **The legacy aliases are same-origin only**: never CORS headers, and a
   foreign browser `Origin` is refused (403). This also closes the
   cross-site `text/plain` POST ("CSRF") that any web page open in the
   operator's browser could otherwise send to them.
4. **The legacy aliases stay unauthenticated only while that is today's
   exposure**: no token configured, or a token on a loopback-only server with
   no `--cors-origin`. Once the server is opened to other origins
   (`--cors-origin`) or other hosts (`--allow-lan`), a configured token closes
   the aliases too (401 without it) — otherwise they would be an
   unauthenticated side door around the token. In that mode `loom_ui.html`
   (which sends no token) loads but its API calls fail; use the new UI.
5. **The served UI** (`/`, `/ui`, `--ui-dir` assets) never needs a token — a
   browser navigation cannot send one, and it is static.
6. The loopback bind guard (`pleroma.net.require_loopback`, `--allow-lan`) is
   unchanged. A token without `--cors-origin`/`--allow-lan` still binds
   loopback only; the ssh tunnel remains the way in from a laptop.

Startup logs one line with the resolved policy, and warns when the server is
opened (`--cors-origin` or `--allow-lan`) with no token.

| token | `--cors-origin` | `--allow-lan` | `/api/v1` | legacy aliases | old UI works |
|---|---|---|---|---|---|
| — | — | — | open, same-origin | open, same-origin | ✓ |
| — | ✓ | any | open, + CORS for the origin (warned) | open, same-origin | ✓ |
| — | — | ✓ | open (warned) | open, same-origin | ✓ |
| ✓ | — | — | bearer | open, same-origin | ✓ |
| ✓ | ✓ | any | bearer, + CORS | **bearer** | ✗ |
| ✓ | — | ✓ | bearer | **bearer** | ✗ |

## Serving a UI: `--ui-dir`

`--ui-dir DIR` serves `DIR/index.html` at `/` and `/ui` (read per request) and
any file beneath `DIR` at its path (e.g. `/assets/app.js`, with a correct
`Content-Type`), `Cache-Control: no-store`. API paths win over same-named
files. Refused (404): `..`/`.` segments (also percent-encoded), dot-files, NUL,
backslashes, and anything that resolves (symlinks followed) outside `DIR`.
Without the flag, `/` and `/ui` serve `pleroma/serve/static/legacy_ui.html` exactly as
before and no other file is served. There is no SPA fallback (an unknown path
is a 404, not `index.html`). The in-repo client for this is `ui/` (build with
`npm run build`, serve `ui/dist`; its assets use relative URLs so `/` and `/ui`
both work — see `ui/README.md`).

## Versioning policy

- `/api/v1` is **additive only**: new routes, new optional request fields, new
  response fields, and new error codes may appear; nothing published is removed,
  renamed, retyped, or given a new meaning. Clients must ignore unknown response
  fields and treat an unknown `code` as its HTTP status.
- A breaking change is `/api/v2`, served beside v1 until the clients move.
  `/info.api.version` names the newest version the server speaks.
- The legacy aliases are frozen to what `loom_ui.html` needs. They will be
  retired only when that file is, and not before a release note.
- Every change to a model or route changes `api-schema.json`; the schema
  freshness test makes that visible in review.

## Regenerating the schema

```sh
python -m pleroma.serve.schema > pleroma/serve/api-schema.json
python -m pleroma.serve.schema --check      # exit 1 if the committed copy is stale
```

`tests/unit/serve/test_api.py::test_committed_schema_is_fresh` fails when the
committed file is stale. The document is deterministic (no clock, host, or
server configuration in it), so the same code always produces the same bytes.
The `ui/` front-end generates its TypeScript client from the committed file
(`cd ui && npm run gen:api`); `npm run check:api` fails when it is stale, and
`tests/unit/test_ui_frontend.py` runs that check — and compares the client's
baked-in `schema_sha256` with the server's — from the Python suite.

## Examples

```sh
export PLEROMA_API_TOKEN=$(openssl rand -hex 16)
python -m pleroma.serve ... --cors-origin http://localhost:5173

curl -s -H "Authorization: Bearer $PLEROMA_API_TOKEN" localhost:8767/api/v1/info | jq .api
curl -s -H "Authorization: Bearer $PLEROMA_API_TOKEN" -H 'Content-Type: application/json' \
     -d '{"session":"s1","text":"what next?","k":6}' localhost:8767/api/v1/loom | jq .loom_id
curl -s localhost:8767/api/v1/state?session=nope    # 401, then with a token: 404 no_session
```
