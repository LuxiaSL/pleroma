# pleroma

*swarm unto itself.*

A research instrument for **looming**: talk to a model, have it sample K
possible futures of the conversation, read each future's *computational
manner* — not its text, but a signature of how the model computed it —
compressed to an 8-number code, and then **wear** a chosen future's code as a
residual-stream steering vector, bending the present toward that future.

```
 you type a contemplated turn
        │
        ▼
   /loom ── sample K futures ── harvest each one's path signature
        │                        (teacher-forced replay through frozen
        │                         extraction instruments: 2,713-d v3
        │                         signature + 420-d temporal bins)
        ▼
   rank-8 linear map g: signature ──► steering lever
        │                             (4 decoder-input sites)
        ▼
   K candidate codes, scored (distinctness / camp / loudness)
        │
        ▼
   /wear future i at dose α ──► every later turn is generated
                                 under that future's disposition
```

The whole loop runs live behind a small HTTP server with a full web panel:
futures deck, dose gauge with measured dose bands, 8-axis code fingerprints,
and a constellation view of the code space with 276 banked disposition
landmarks. Conversation verbs (`/edit`, `/reroll`, `/undo`, `/truncate`), a
parallel un-steered control branch, and auto-loom policies
(`distinct`/`loudest`/`minority`/`stay`/`swerve`) are all built in.

## What's in the box

- **`pleroma/`** — the pipeline and the server. `loom_serve.py` is the
  instrument; everything else builds or audits its inputs.
- **`anamnesis/`, `kvrot/`** — vendored snapshots of the frozen extraction
  instruments (path-signature feature stack + the v3 feature-space config).
  See `VENDOR.md` for provenance and pinning.
- **`data/`** — everything needed to run the shipped **Llama-3.2-3B-Instruct**
  loom immediately: the fitted rank-8 map (`loom_map_3b.npz`, sha-pinned to
  its feature space), the code atlas, the 138-disposition lever bank, and
  calibration (one 172MB file fetched by `data/fetch_calibration.sh`).

## Quick start

See [QUICKSTART.md](QUICKSTART.md). Short version: one CUDA GPU, one venv,
one fetch script, two processes, then open `http://localhost:8767`. There is
also a **new-model lane** — every stage from calibration to a fitted map can
be rebuilt from scratch for a different model.

[docs/LOOM-GUIDE.md](docs/LOOM-GUIDE.md) is the operator's manual: the full
HTTP API, the dose doctrine, recipes, and what to expect from steering.

## What this is, epistemically

A **v0 research instrument**, shared so others can play, poke, and break it —
not a polished product and not a paper. Honest notes:

- Measured on a 3B model. The map's code captures ~26% of a true
  disposition's energy — a narrow but causally potent channel.
- Steering transfers **disposition and register, not content**. Pick a future
  for a specific claim and you'll get its manner, not the claim.
- Doses are measured: α ∈ [0.125, 0.5] steers without being conversationally
  visible, α ≈ 1 is audible, α = 1.5 damages generation. The UI's gauge
  shows these bands.
- The map is trained on single-prompt continuations and deployed on
  multi-turn chat — deliberately off-distribution; same-context recovery is
  its deployment regime.
- The quantitative receipts (dose ladders with API-judged psychophysics,
  pull tests, fold-leak corrections, atlas gauge analysis) live in a private
  research ledger. If you want them, ask.

## Layout

| path | what |
|---|---|
| `pleroma/loom_serve.py` | the server + web UI (`loom_ui.html`) |
| `pleroma/harvest_worker.py` | optional persistent harvest process (faster /loom) |
| `pleroma/build_calibration.py` → `fit_loom_map.py` | the new-model lane, in pipeline order (see QUICKSTART) |
| `pleroma/build_code_atlas.py` | renders the rank-8 code space + landmark bank |
| `data/` | shipped 3B artifacts (map, atlas, levers, calibration) |
| `VENDOR.md` | provenance of the frozen extraction instruments |

## License

No license has been granted yet — all rights reserved for now. If you want
to use this beyond reading and running it locally, open an issue.
