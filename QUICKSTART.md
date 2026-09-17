# Quickstart

Two lanes. **Lane A** runs the shipped Llama-3.2-3B loom in minutes.
**Lane B** rebuilds every stage for a new model.

Throughout: run everything **from the repo root** — the vendored
`anamnesis/` and `kvrot/` packages import via the working directory, and all
data paths in the examples are repo-relative.

## Lane A — "I have a GPU, put me in the loom"

Hardware: one CUDA GPU. 24GB is comfortable (the server and the optional
harvest worker each hold a 3B in fp16); 16GB works if you skip the worker.

```bash
git clone https://github.com/LuxiaSL/pleroma && cd pleroma

# 1. env (uv shown; plain python -m venv + pip install -e . works too)
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. the one data file too big for git (172MB, sha-verified)
bash data/fetch_calibration.sh

# 3. the model — HF gated repo, accept the Llama license first,
#    or point --model-path at any local copy of Llama-3.2-3B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct

# 4. (optional, recommended) the persistent harvest worker: /loom drops
#    from ~45s to ~14s. Loopback only. Run it in its own terminal.
python -m pleroma.harvest_worker \
    --preset 3b --model-path meta-llama/Llama-3.2-3B-Instruct \
    --calib-dir data/calibration/3b \
    --discriminants data/discriminants/factor_directions_3b.npz \
    --host 127.0.0.1 --port 8768

# 5. the loom server
python -m pleroma.loom_serve \
    --map data/loom/loom_map_3b.npz \
    --discriminants data/discriminants/factor_directions_3b.npz \
    --calib-dir data/calibration/3b \
    --model-path meta-llama/Llama-3.2-3B-Instruct --preset 3b \
    --atlas-report data/loom/atlas/atlas_report.json \
    --work-dir outputs/loom/sessions --port 8767 \
    --harvest-worker 127.0.0.1:8768        # omit if you skipped step 4
```

Open **http://localhost:8767**. Remote GPU box? `ssh -L 8767:localhost:8767
<host>` and browse the same URL. (The server binds loopback by default;
`--allow-lan` exists but the API has no auth — only use it on a network you
trust end to end.)

First loom, in the panel: type a message, hit **Loom** instead of Send, watch
K futures land with their codes and scores, pick one, wear it at α = 0.35,
then Send. Then read `docs/LOOM-GUIDE.md` — especially the dose table and
"what to expect", which will save you the two classic confusions (steering
carries *manner*, not content; loom early — the fan of futures collapses as a
conversation commits).

Sanity check without the browser:

```bash
curl -s localhost:8767/info | python -m json.tool | head -20
```

## Lane B — "I have a different model"

Every instrument stage is rebuildable. The chain, in order (each script's
docstring is the authoritative reference for its flags — they are written to
be read):

```bash
# 0. teach the config your model: add a preset to anamnesis/config.py
#    MODEL_PRESETS (~20 declarative lines: dims, layer picks, eos ids —
#    copy the "3b" entry and adjust; mid/late layers are the ones that matter)

# 1. calibration: positional means + tier-3 PCA (~50 generations, one GPU)
python -m pleroma.build_calibration --preset <p> --out-dir data/calibration/<p>

# 2. a forked-generation corpus: K seeded continuations per prompt
python -m pleroma.gen_forks --prompts-per-class 2 --seeds-per-prompt 16 \
    --out-dir outputs/<p>/gen --preset <p>

# 3. replay + signatures, FREEZING the new model's feature space
#    (writes the manifest that plays the role factor_directions_3b.npz
#    plays for the shipped 3B — use it as --discriminants ever after)
python -m pleroma.run_replay_b0 --gen-dir outputs/<p>/gen \
    --out-dir outputs/<p>/replay --preset <p> \
    --calib-dir data/calibration/<p> \
    --freeze-names-to data/discriminants/manifest_<p>.npz

# 4. temporal bins (the second feature family the map eats)
python -m pleroma.extract_bins --gen-dirs outputs/<p>/gen --out outputs/<p>/bins.npz \
    --preset <p>

# 5. training pairs (CPU, laptop-runnable)
python -m pleroma.build_pairs   # see docstring: replay dirs -> pairs dir

# 6. per-gen mean hidden states at the steering sites, then cluster-contrast
#    levers (the targets g regresses onto, and the α=1 norm reference)
python -m pleroma.bank_mean_hiddens --gen-dirs outputs/<p>/gen \
    --out-dir outputs/<p>/hiddens --preset <p>
python -m pleroma.build_levers  # see docstring: hiddens + pairs -> levers.npz

# 7. the map
python -m pleroma.fit_loom_map --pairs-dir outputs/<p>/pairs \
    --levers outputs/<p>/levers.npz --bins outputs/<p>/bins.npz \
    --discriminants data/discriminants/manifest_<p>.npz \
    --norm-ref-bank outputs/<p>/levers.npz --out outputs/<p>/loom_map.npz

# 8. the atlas (optional but worth it — it is the second percept)
python -m pleroma.build_code_atlas --map outputs/<p>/loom_map.npz \
    --levers outputs/<p>/levers.npz --out-dir outputs/<p>/atlas

# 9. serve (as lane A, with your artifacts)
```

Scale expectations, honestly: the shipped 3B map was fit on 138 lever groups
from ~2,100 trajectories; a first pass with 40-ish prompts × 16 seeds is
enough to see the loop move, not enough to trust transfer claims. Steering
sites and dose bands are model-specific — re-derive them (start at the
decoder-input sites at ~30/55/70/80% depth and titrate α from 0.25 up while
watching for degeneration).

Two hard-won implementation notes already baked into the code, so you don't
fight them: cuDNN SDPA is disabled before generation (per-shape host graph
compilation makes token-by-token decode pathological on some stacks), and
extraction runs eager attention (flash/SDPA silently drop the attention
weights the feature families need).
