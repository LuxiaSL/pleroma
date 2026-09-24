# pleroma

A research instrument for **looming**. You talk to a language model. At any
turn, the loom samples K possible next replies (a *fan* of futures). It reads
*how the model computed* each one: not the text, but a signature of the
forward pass. You pick a future and **wear** it. From then on its signature,
turned into a steering vector, is added to the model's residual stream, and
later replies lean toward that future's manner.

It is shared so others can run it, poke at it and break it. It is not a
product, and not a paper.

## What it does

```
 a turn you are considering
        │
        ▼
 /loom ─ sample K futures (the fan)
        │
        ▼
 harvest each future: replay it through the model with the frozen
 anamnesis instrument ─► signature z_i (v3 features + a temporal-bins block)
        │
        │   Δz_i = z_i − mean of the fan's other futures      (fan contrast)
        ▼
 v1a map (ridge, rank 64) ─► code c_i: 64 numbers
                          └► lever: one residual-stream vector per site
                             (layers 9, 17, 21, 25 on Llama-3.1-8B)
        │
        ▼
 /wear future i at dose α ─► each site's vector is rescaled to α · norm_ref[site]
                             (the map's ruler) and added at every position
                             (uniform span) on every later turn of the loom branch
```

- **The map (v1a).** The map's input is a future's signature minus the mean
  of its fan-mates. Its target is that future's mean hidden state minus the
  same leave-one-out mean. It is fit as a ridge regression (λ = 1e4) and cut
  to rank 64. It outputs a 64-number **code** and a **lever**, one vector per
  steering site.
- **Absolute vs contrast levers.** An *absolute* lever maps the future's own
  signature. A *contrast* lever maps the signature minus the mean of the
  other futures, which is the object v1a was fit on. The 8B profile serves
  `absolute`. `/wear` takes `lever_kind` to switch. Which one should be the
  default is an open question.
- **Dose α and the ruler.** α is not a unit you can carry between maps. On a
  given map, α = 1 injects exactly `norm_ref[s]` at site `s`, and `norm_ref`
  differs from map to map. Two maps of the same 3B model once differed by
  2.14× at the same α ([FINDINGS §2](docs/FINDINGS.md)).
  Each map therefore ships a **measured dose band**: which α is below blind
  detection, threshold or overdriven *on that map's ruler*. The band is
  stamped with the map's fingerprint and with the injection span it was
  measured under.
- **Injection span.** The span is uniform: the lever is added at every
  position, prompt included. This was a design decision backed by a modest
  measured advantage over continuation-only injection
  ([FINDINGS §4](docs/FINDINGS.md)).
- **Picking and the gauge.** Free pick policies score the fan in signature
  space. `loudest` is the largest map output; `distinct` is the future
  farthest from the rest. The **probe** (gauge) wears each candidate briefly
  and sees where the reply lands. It costs extra latency, and when it says
  `resolved` it certifies only its top pick. These are policies for choosing
  a future, not measurements of steering
  ([FINDINGS §8](docs/FINDINGS.md)).

Wearing carries **manner, not content**: register, stance, reply
architecture. Pick a future for one striking sentence and you get its manner,
not the sentence. Loom early: a fan collapses as a conversation commits.

## Status: what is measured, and what is not

A research instrument. Every number below has a receipt in
[docs/FINDINGS.md](docs/FINDINGS.md), and the 8B kit (below) is enough to
re-check some of them.

**Measured**

- On Llama-3.2-3B, wearing a screened future shifts a blind reader toward
  that future. This was a pre-registered test: p = 5e-5, 9 of 10
  conversations positive, at doses that cost no fluency.
- The v1a construction beats the zero-training shelf baseline on held-out,
  within-fan retrieval (top-1 of 8) on both models:
  - 3B: .9339 vs .6915;
  - Llama-3.1-8B-Instruct: .9503 vs .7300.
  - Both results are at the permutation floor, p = 5e-5.
- **On Llama-3.1-8B-Instruct, wearing a future's held-out lever pulls
  toward that future and not its fan-mates** (16 seeded fans, every member
  worn, within-fan permutation null):
  - teacher-forced likelihood, no judge: Δnr +.28 at α 0.35 and +.31 at
    1.0, 16/16 fans each, p = 5e-5;
  - a judge ranking the fan's futures against a set of worn replies, beyond
    what a length-only ranker gets from the same replies: +.10 at α 0.35
    (p = .008) and +.20 at 0.675 (p = 2e-4, 15/16 fans), with two judges
    agreeing;
  - a random direction at the same per-site norm does nothing on any
    readout. So the steering is there at doses a blind pairwise reader
    cannot see ([FINDINGS §3b](docs/FINDINGS.md)).
- On 8B, a three-judge blind ladder under the uniform span measured the
  dose band: α ≤ 0.35 is at the judges' false-positive floor, 0.75 clears
  it, and from 1.0 one of the two ladder directions breaks into word salad
  as it becomes obvious. A blind pairwise reader is the weak test of a
  contrastive lever; the band marks its edges
  ([FINDINGS §3](docs/FINDINGS.md)).

**Not measured, or open**

- The 8B steering eval is 16 fans at one sampling of each; the effect is
  large against its null, but how it varies by prompt class is not mapped.
- The 8B calibration's positions 555–2047 were fit on generations that ran
  past end-of-turn, not on multi-turn chat history; positions 0–554 are the
  ones the map was fit under. Harvests of long conversations use the former.
- `predicted` dosing is not behaviourally validated. It is the profile's
  default: it gives louder futures proportionally more dose, and every
  per-site clamp is reported. `absolute` vs `contrast` as the default lever
  is also open.
- The 8B profile's future and reply lengths are code defaults, not read
  from a live launch record; the profile says so. (Its sampling is verified
  and its chat date is pinned to the corpus's.)
- Codes are tied to their map. Exports store canonical singular-vector
  signs, and every code carries its map's fingerprint; a code from another
  map is refused rather than read as a coordinate in this one.

## `pleroma validate`: no bare number a newcomer can misread

Almost every wrong conclusion this project has had to retract came from a
number read without its control. `pleroma validate` reads artifacts that
already exist. It needs no GPU, no network and no API. For each gate it
prints PASS, FAIL or INCONCLUSIVE with a named reason; the numbers are
listed beside the verdict as evidence. INCONCLUSIVE means "these inputs
cannot answer the question", never "fine".

The confounds it exists to catch:

| gate | the trap |
|---|---|
| `eval/length-null` | Steering moves length, and judges read length: the rate of pairs called "different" rises from .35 to .96 as their length gap grows. A judged effect must beat the length-only ranker, paired by conversation. |
| `eval/positive-control` | A judge that cannot separate a known effect from its *own* false-positive floor makes a null meaningless. The floor is measured, never assumed to be 0.5. |
| `dose/band`, `dose/band-span` | A band must be *measured*, stamped for this map and ruler, and measured under the span the loom serves. |
| `dose/damage` | High-dose "effects" are mostly damage: a random direction at matched damage reproduces most of them. This gate is a loop detector only; a PASS is not a fluency certificate. |
| `fit/heldout`, `fit/ceiling` | Held-out (never in-sample) retrieval must beat the shelf baseline and chance. When the baseline is already near the top, a high top-1 says nothing, and `ceiling` says so. |
| `format/*`, `export/*` | chat template and date pinning, stop strings, pad ≠ eos, and that the map file is the map the profile describes |

## Requirements

- **Python** ≥ 3.11 and a CUDA GPU for serving and building. The test suite
  and `pleroma validate` run on CPU.
- **Llama-3.1-8B-Instruct** weights: a gated Hugging Face repo, so accept
  Meta's license first.
- **Node** ≥ 20.19, only for the standalone UI.

GPU memory. The 8B has so far run only on datacenter GPUs, and its peak
memory has not been measured. The figures below are **estimates** from the
weights and one measurement on the 3B:

- The bf16 weights are about 16 GB (8.0 B parameters × 2 bytes), and the
  loom holds several copies:
  - the server holds one copy;
  - the harvest worker holds two, one for the signature replay and one for
    the temporal bins;
  - extraction runs eager attention, which needs extra room on top.
- The one measurement: a 3B harvest worker holds 13.3 GiB. That is its two
  eager copies plus about 0.7 GiB each of overhead.

| setup | estimate |
|---|---|
| server + worker on one GPU | about 50 GB plus activations: an 80 GB card |
| server on one GPU, worker on another (`device`) | about 18 GB + 34 GB: e.g. a 24 GB + a 48 GB card |
| server, worker replay, worker bins on three GPUs (`device`, `bins_device`) | about 18 GB each: three 24 GB cards |

Without a worker, every `/loom` launches cold harvest subprocesses, and each
one loads the model again. That is slower, and its peak memory is also
unmeasured. Harvest is mostly CPU feature math, so CPU cores matter too.
Measured on 8B: a k = 8 draw took 12.3 s (3.1 s generating, 9.2 s
harvesting) with two pooled workers.

## Quick links

- [QUICKSTART.md](QUICKSTART.md): install, fetch the 8B kit, validate,
  serve, first loom and wear. Then build your own map.
- [docs/FINDINGS.md](docs/FINDINGS.md): what has been measured, with the
  evidence and what the public kit can reproduce.
- `ui/README.md`: the standalone front-end (Vite + TypeScript + three.js),
  including a mock backend that needs no GPU.
- `pleroma/serve/api-schema.json`: the HTTP API contract (OpenAPI 3.1),
  also served live at `GET /api/v1/schema`.
- [CONTRIBUTING.md](CONTRIBUTING.md): tests, and the validate philosophy.
- anamnesis, the signature instrument, lives at
  <https://github.com/LuxiaSL/anamnesis> (MIT). This repo pins it to one
  commit, in `pyproject.toml` and as the `vendor/anamnesis` submodule.

## License

MIT. See [LICENSE](LICENSE).
