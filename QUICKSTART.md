# Quickstart: the 8B golden path

This guide goes from a fresh clone to wearing a future on
Llama-3.1-8B-Instruct. Run every command **from the repo root**. The example
deployment's paths are relative to it.

Steps 1–5 need no GPU. Step 6 (serving) needs one; see the README's
requirements for how much, and which of those figures are estimates.

## 1. Clone and install

```sh
git clone --recurse-submodules https://github.com/LuxiaSL/pleroma
cd pleroma

uv venv --python 3.12
source .venv/bin/activate
# If the default torch wheel does not match your CUDA driver, install torch
# first from the matching index, e.g.:
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"
```

`pip install -e ".[dev]"` inside a plain `python -m venv` works too.

**anamnesis** (the signature instrument) comes with the install. It is
pinned to one commit in `pyproject.toml`, and every entry point refuses to
start unless the imported anamnesis *is* that commit and is unmodified.
`vendor/anamnesis` is the same commit as a git submodule, for reading or
working offline. To run against the submodule instead of the installed copy,
put it first on the path:

```sh
export PYTHONPATH="$PWD/vendor/anamnesis"
```

The pin check still applies (it reads the submodule's git HEAD). Do not
upgrade anamnesis casually. A changed extractor silently invalidates every
fitted map.

Check the install (CPU only; takes a few minutes):

```sh
python -m pytest -q
pleroma --help
```

## 2. Get the model

Llama-3.1-8B-Instruct is a gated repo. Accept Meta's license on Hugging
Face, log in, then download it to a local directory:

```sh
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct
# older huggingface_hub: huggingface-cli download ... --local-dir ...
```

## 3. Fetch the 8B kit

The kit is everything the loom needs beside the weights. Each file is a
release asset pinned by sha256 in `kit-manifest.json`:

| role | file | what it is |
|---|---|---|
| map | `loom_map_v1a_r64.npz` | the fitted v1a map (rank-64 codes → levers at layers 9/17/21/25), with its ruler `norm_ref` |
| dose_band | `loom_map_v1a_r64.npz.doseband.json` | the map's measured dose band, stamped with the map's fingerprint, ruler and span |
| discriminants | `factor_directions_8b.npz` | the frozen feature space the map was fit in (the map refuses any other) |
| calibration | `calibration/positional_means.npz`, `calibration/pca_model.pkl` | the model's anamnesis calibration |
| cv_report, export_report | `reports/*.json` | the registered CV and the export's self-check, for `pleroma validate` |

```sh
python tools/fetch_kit.py --list     # what the manifest names, and the sha256 of each asset
python tools/fetch_kit.py            # download into kit/llama31-8b-instruct/, verifying each file
python tools/fetch_kit.py --verify   # re-check what is on disk; downloads nothing
```

The fetcher's rules:

- Each file is streamed to `<dest>.part`, hashed, and moved into place only
  if its sha256 matches the manifest.
- On a mismatch it deletes the partial file and exits non-zero.
- It never overwrites an existing file whose hash is wrong unless you pass
  `--force`.
- It refuses a manifest with an unfilled hash.
- `pca_model.pkl` is a pickle. Load it only after it verified.

## 4. Write your deployment

A **profile** (`profiles/llama31-8b-instruct.toml`) describes the model and
the method:

- architecture, layer geometry, prompt format, sampling, lengths;
- map sites and operating point, injection span, lever kind, dose policy.

It ships with the repo. A **deployment** describes this machine: where the
files are and how the loom is served. Start from the example:

```sh
cp examples/deploy-llama31-8b.toml deploy.toml
python -m pleroma profile show profiles/llama31-8b-instruct.toml   # validate + print the profile
```

| field | meaning |
|---|---|
| `profile` | the profile's `name`; a mismatched pair is refused |
| `model_path` | the local weights directory (overrides the profile's `model_id`) |
| `calib_dir` | the kit's calibration directory. Always the calibration the map was fit with, never mixed with another |
| `work_dir` | where the server keeps one JSON snapshot per session. One per server |
| `host`, `port`, `allow_lan` | loopback `127.0.0.1:8767` by default. Set `allow_lan = true` only on a network you trust end to end |
| `[map]`, `[discriminants]`, `[dose_band]` | the kit's files. Optional `sha256 = "…"` pins, checked by `pleroma validate`; copy them from `fetch_kit.py --list` |
| `[worker]` `address` | the harvest worker's loopback address. The server calls it; you never do |
| `[worker]` `lane` | `replay`, the frozen v3 capture surface every 8B artifact was built with. `gpu` is not qualified for 8B |
| `[worker]` `device`, `bins_device` | the worker holds the model twice. Put the bins copy on a second GPU if one card cannot hold both |
| `timeout_s` | how long the server waits for one harvest |

Delete the `[worker]` block to run without a worker. Every `/loom` then
starts cold harvest subprocesses, which is slower.

## 5. Validate before you serve

```sh
pleroma validate --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml \
    --tokenizer models/Llama-3.1-8B-Instruct \
    --cv-report kit/llama31-8b-instruct/reports/v1a_report.json \
    --export-report kit/llama31-8b-instruct/reports/loom_map_v1a_r64_export_report.json \
    --out runs/validate-8b.json
```

`--deploy` fills in the map, discriminants and band. The command prints one
line per gate:

```
VERDICT  stage/gate — reason
```

The numbers each verdict rests on are in the JSON receipt (`--out`). Exit
status is 1 if any gate FAILs.

- **`format/*`**: the chat template renders deterministically, with the
  date pinned. Turns end on the model's eos ids. A pad token cannot be
  harvested as text.
- **`fit/heldout`**: held-out retrieval beats the shelf baseline and
  chance.
  - **`fit/ceiling`**: says whether that test can discriminate at all.
  - **`fit/join`**: says whether the CV saw the fans the profile serves.
- **`export/*`**: the map file has the profile's sites, rank and hidden
  size. The canonical loader accepts it with its discriminants. Its export
  report carries a held-out figure for *this* file.
- **`dose/band`**: the band is measured and stamped for this map's
  fingerprint and ruler.
- **`dose/band-span`**: the band was measured under the injection span the
  loom serves (uniform). If it FAILs, the band's zones describe a different
  (weaker) intervention than the one the loom applies. Read the gauge
  accordingly.
- **INCONCLUSIVE** means the inputs cannot answer the question. It does not
  mean "fine". `dose/damage` needs steered replies (`--replies`,
  `--base-replies`). `eval/*` needs judged artifacts. `probe/resolution`
  needs a `/probe` receipt. You produce all of these later.

## 6. Serve

Start two processes, each in its own terminal. Start the worker first.

```sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# print the exact command without running it; keep it with your notes
pleroma worker --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml --dry-run

CUBLAS_WORKSPACE_CONFIG=:16:8 \
pleroma worker --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml
```

```sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 \
pleroma serve --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml
```

- Every value comes from the profile and the deployment, and nothing falls
  back to a code default.
- `--dry-run` prints the command. Keep that printout: an unrecorded launch
  has cost a reconstruction before.
- The server loads the model on `cuda` (the first visible device), so
  choose the card with `CUDA_VISIBLE_DEVICES`.

Check the server is up:

```sh
curl -s localhost:8767/api/v1/info | python -m json.tool
```

`/info` describes the running server: its map, the ruler in force, the dose
band and its tier, the pick policies, and persistence. Where it disagrees
with a document, `/info` is right.

## 7. The UI

There are two front-ends:

- **The server's own page**: open <http://localhost:8767/>. This is the
  complete legacy panel: futures deck, dose gauge, auto-policies, probe,
  and conversation verbs.
- **The standalone UI** (`ui/`; Vite + TypeScript + three.js, including a
  3-D code space). It is still growing toward parity (`ui/README.md` lists
  what it has).

```sh
cd ui && npm ci
npm run dev:mock     # no GPU, no server: recorded 8B responses, http://localhost:5173
npm run dev          # proxies /api/v1 to http://127.0.0.1:8767
```

To have the loom serve the built UI itself, append `--ui-dir` to the printed
command. The profile/deploy pair has no field for it yet.

```sh
(cd ui && npm run build)
CMD=$(pleroma serve --profile profiles/llama31-8b-instruct.toml --deploy deploy.toml --dry-run)
$CMD --ui-dir "$PWD/ui/dist"
```

For a UI hosted on another origin, add `--cors-origin <its origin>` and set
`PLEROMA_API_TOKEN` in the server's environment. That combination also puts
the legacy routes behind the token.

## 8. First loom, first wear

In either UI:

1. Type a turn and press **Loom** instead of Send.
2. Read the K futures, their codes and scores, then pick one.
3. Wear it mid-threshold (α ≈ 0.675 on the 8B band, the default the UI
   proposes) and Send. Below 0.35 a blind reader rarely sees it, which is
   not the same as it having no effect; from 1.0 some directions break.
4. Send the same turn to the **base** branch (the untouched control) and
   compare the two replies.

The same loop over the API:

```sh
S=first-loom; B=localhost:8767/api/v1; J='content-type: application/json'
T='"Something small went wrong this morning and I keep thinking about it."'

curl -s $B/loom -H "$J" -d "{\"session\":\"$S\",\"text\":$T,\"k\":6}" > fan.json
python -c "import json; f=json.load(open('fan.json')); print(f['loom_id']); [print(x['index'], x['text'][:100].replace(chr(10),' ')) for x in f['futures']]"

LOOM_ID=$(python -c "import json; print(json.load(open('fan.json'))['loom_id'])")
curl -s $B/wear -H "$J" -d "{\"session\":\"$S\",\"index\":2,\"alpha\":0.35,\"loom_id\":\"$LOOM_ID\"}" | python -m json.tool

curl -s $B/chat -H "$J" -d "{\"session\":\"$S\",\"branch\":\"loom\",\"text\":$T}" | python -c "import json,sys; print(json.load(sys.stdin)['reply'])"
curl -s $B/chat -H "$J" -d "{\"session\":\"$S\",\"branch\":\"base\",\"text\":$T}" | python -c "import json,sys; print(json.load(sys.stdin)['reply'])"
```

What to expect:

- **Wearing carries manner, not content**: register, stance, the shape of
  the reply. Pick a future for one striking sentence and you get its
  manner, not the sentence.
- **Loom early.** A fan collapses as a conversation commits.
- **Prompts that leave the kind of reply open fork more.** A question that
  names its own dichotomy ("is it X or Y?") pins the reply and flattens the
  fan (FINDINGS §11).
- **Read the dose at the effective α.** The profile's dose policy is
  `predicted`: louder futures get proportionally more dose. The `/wear`
  response reports `effective_alpha`; read the band at that value, not at
  the α you sent.
- **Doses do not transfer between maps.** Never carry an α over from
  another map (FINDINGS §2).
- **`/reset` clears both branches** and overwrites the session snapshot. It
  is not an undo; `/undo` is.

The full route list is the OpenAPI document `pleroma/serve/api-schema.json`,
also served at `GET /api/v1/schema`.

---

## Build your own map

This is an outline of the pipeline that produced the kit, for another model
or other prompts. It takes hours of GPU time and a real corpus: the 8B map
came from 20,960 generations. Only the last three build stages and
`validate` are `pleroma` subcommands today. The earlier stages are being
folded into the package, and `pleroma --help` shows what currently exists.

1. **Profile.** Write `profiles/<model>.toml`:
   - the architecture, straight from the checkpoint's `config.json`;
   - the anamnesis preset and its layer geometry;
   - the steering sites and temporal-bins layers;
   - a pinned `format.date_string`.

   A preset that anamnesis does not ship goes in a registry file named by
   `[model] anamnesis_registry`. Check the profile with
   `pleroma profile show`.
2. **Calibrate.** Measure positional means and fit the residual PCA for
   *your* model:
   `python -m anamnesis.scripts.run_calibration --model <preset> --model-path <weights> --out-dir <dir>`.
   - Use one calibration, whole, for every later stage.
   - Never splice it with another model's.
   - The harvest lane refuses a span that reaches past the last position
     the calibration filled, so cover the conversation lengths you serve.
     (The 8B kit's covers 2048.)
3. **Fans.** Generate K seeded continuations per prompt (8B: 8 per prompt),
   in the format you will serve and with the date pinned. How much the fans
   fork sets the ceiling for everything after (FINDINGS §11). Screen your
   prompts for it.
4. **Harvest.**
   - Replay every generation through the frozen v3 capture surface.
   - Freeze the new model's feature-name manifest; it becomes your
     `discriminants`.
   - Extract the temporal bins.
   - Bank each generation's mean hidden state at the steering sites.
5. **Pairs and levers.** Join signatures to hidden states by fan. This
   gives the training pairs and the lever bank whose norms set the ruler.
6. **`pleroma shelf`** fits the wide map. It is the registered CV's frozen
   baseline.
7. **`pleroma fit`** runs the registered CV: held-out, within-fan retrieval
   against the shelf, grouped by prompt.
8. **`pleroma export`** writes the served map, with its standardisers
   computed in-build and its held-out figure taken from step 7's report.

Stages 6–8 read a `[build]` block in the deployment, and every input can be
sha-pinned:

```toml
[build]
pairs_dir = "runs/mymodel/pairs"
out_dir   = "runs/mymodel/golden"
levers    = { path = "runs/mymodel/levers/levers.npz" }
hiddens   = [ { path = "runs/mymodel/hiddens/mean_hiddens.npz" } ]
bins      = { path = "runs/mymodel/bins/bins.npz" }
shelf_map = { path = "runs/mymodel/golden/shelf_map.npz" }   # after `pleroma shelf`
cv_report = { path = "runs/mymodel/golden/v1a_report.json" } # after `pleroma fit`
```

```sh
pleroma shelf  --profile profiles/<model>.toml --deploy deploy.toml --dry-run   # then without --dry-run
pleroma fit    --profile profiles/<model>.toml --deploy deploy.toml
pleroma export --profile profiles/<model>.toml --deploy deploy.toml
```

Each stage writes `<out>.receipt.json` beside its output. The receipt holds
the exact command, every input's sha256 (pins are checked *before* the
stage runs), the output's sha256, the git commit, the times and the exit
code.

9. **Band.** Measure a dose band for the new map, under the injection span
   you serve (uniform), with a blind judged ladder read against its own
   α = 0 catch floor. Until then, a band rescaled from another map by the
   `norm_ref` ratio is `derived`, and marked provisional everywhere it
   appears. A map with no band is served as "uncalibrated".
10. **Validate.** Run `pleroma validate` with everything you produced: the
    CV report, the map and its discriminants, the export report and the
    band. Once you have them, add steered replies, judged artifacts and
    probe receipts. Read every INCONCLUSIVE as a question that is still
    open.
