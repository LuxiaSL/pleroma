# Findings

A curated summary of what this instrument has measured. It keeps only two
kinds of finding: those about the method in general, and those tied to the
public Llama-3.1-8B-Instruct kit. Each entry gives the claim, the evidence in
brief, the model it was measured on, and whether you can re-check it:

- **Kit**: the public 8B kit plus `pleroma validate` (no GPU) re-checks it.
- **8B model**: you need only the model and a GPU, not the kit.
- **Not reproducible here**: it needs corpora or judged runs that are not
  published.

Unless an entry says otherwise, the statistics are:

- pre-registered tests, one-sided;
- permutation or sign-flip nulls with 20,000 draws, whose smallest possible
  p-value is 5e-5;
- cross-validation grouped by prompt, so no prompt's seeds appear on both
  sides of a split.

"Judged" means blind two-alternative or ranking tasks read by LLM judges,
always against the judges' own measured false-positive floor.

---

## 1. The v1a construction beats the shelf baseline at 3B and at 8B

**Claim.** Retrieval improves a lot when the map is trained on the object it
serves:

- **input**: a future's signature minus the mean of its fan-mates (the mean
  leaves the future itself out);
- **target**: that future's mean hidden state at the steering sites, minus
  the same leave-one-out mean.

It improves on both model sizes tested.

**Evidence.** The registered test is held-out, within-fan retrieval: pick the
right future out of its fan of 8 (chance .125). It is paired by fan, compared
against the shelf map's zero-training projection, and run at the fixed
operating point (λ = 1e4, rank 64).

| model | shelf baseline | v1a held-out | margin | p |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | .6915 | .9339 | +.2424 | 5e-5 |
| Llama-3.1-8B-Instruct | .7300 | .9503 | +.2203 | 5e-5 |

- **The input carries its own share of the win.** On 3B, the fan-relative
  input alone adds 16 points at the fixed point (.773 → .934), and it beats
  the raw signature in all 36 paired grid cells.
- **The margins are about the same size at the two scales.** No test
  comparing them was registered.
- An earlier version of the fit code standardized the shelf baseline's input
  twice. That understated the baseline, which made the 8B margin look like
  +.2831. The table gives the corrected numbers. If a CV report shows a
  shelf of .6672 for the 8B, it came from that code.

**Kit.** The kit ships the 8B CV report. `pleroma validate --cv-report …`
reads it into `fit/heldout` and `fit/ceiling`. Refitting needs the fan
corpus, which is not published.

## 2. Dose α means nothing without its ruler

**Claim.** On a given map, α = 1 injects exactly the map's `norm_ref[s]` at
site `s`. `norm_ref` differs from map to map, so a bare α compared across
maps is meaningless. Compare absolute per-site injected norms instead.

**Evidence** (3B). Two maps of the same model, both at α = 1, gave these
per-site norms:

| map | site norms |
|---|---|
| older map | [0.491, 1.044, 1.511, 2.096] |
| newer map | [1.012, 2.184, 3.239, 4.560] |

- The newer map injects 2.139× more at the same α. The per-site ratios run
  from 2.06 to 2.18.
- Three independent sources give the same ratio: a snapshot from a live
  session, the map file itself, and a dose ladder's recorded `norm_ref`.
- The mistake this caused: for days, one map's "destructive" dose was just
  the other map's audible dose, read on the wrong ruler.

**Consequence in the code.** A dose band is data. It is stamped with its
map's fingerprint and `norm_ref`, and it is tiered:

- `measured`;
- `derived`: rescaled from another map's band by the mean `norm_ref` ratio,
  and always marked provisional;
- `none`: the UI says "uncalibrated".

A band never transfers silently. The 8B map's `norm_ref` is
[0.673, 1.556, 2.306, 3.289], about 0.68× the 3B's.

**Kit.** The kit's band carries the fingerprint and `norm_ref`. `pleroma
validate` checks the stamp against the map (`dose/band`).

## 3. The 8B dose band, and why a blind reader is the weak test

**Claim.** On the kit map's ruler, under the uniform span the loom serves:

| α | zone | what it means |
|---|---|---|
| ≤ 0.35 | below blind detection | a blind pairwise reader is at its false-positive floor |
| 0.35 – 1.0 | threshold | blind readers start to catch it, lever-dependently |
| ≥ 1.0 | overdriven | measured damage for some directions |

"Below blind detection" is not "no effect". A v1a lever is contrastive: it
points at what makes one future distinct *within its fan*, and it mostly
moves manner (register, stance, reply architecture), which a reader comparing
two independent draws rarely sees. Whether a dose *steers* is a fan-relative
question (§3b and §5: does the reply move toward the worn future more than
toward its fan-mates?). This ladder answers a narrower one: when does the
change become obvious to a naive reader, and when does it start to break the
text? The band is kept for those two edges.

**Evidence.**

- **Design**: 2 levers (the v1a map's held-out predictions for 2 seeded
  members, on the kit map's ruler) × 8 blind probes × α ∈ {0, 0.125, 0.35,
  0.75, 1.0, 1.25} × 6 reps = 576 pairs, injected uniformly. Three judges
  (Claude Sonnet 5, Opus 5, Opus 5.5) read every pair blind: 1,704
  judgments.
- **Pooled**, each rung read against the ladder's own α = 0 catch floor
  (.56, CI [.49, .64]; one confidently-wrong call in 176):

  | α | accuracy | CI95 | p vs the floor |
  |---|---|---|---|
  | 0.125 | .49 | [.42, .56] | .15 |
  | 0.35 | .52 | [.44, .60] | .45 |
  | 0.75 | .69 | [.62, .75] | .011 |
  | 1.0 | .78 | [.72, .83] | < 1e-4 |
  | 1.25 | .91 | [.87, .94] | < 1e-4 |

- **Split by lever, the top of the curve is damage.** One lever becomes
  plainly detectable (.94 at 1.0, 1.00 at 1.25) exactly as the strongest
  judge starts citing incoherence (65% of its judgments at 1.0, 90% at 1.25;
  the α = 0 rate is 6%). The other stays coherent and first reads audible
  (.81) at 1.25. **No α on this map is both audible and safe for both
  levers.** So the overdriven edge is measured here, lever by lever, not
  derived from another map.

**Read with.**

- **Two levers.** Direction matters as much as dose at the top of the band;
  two directions show that, but do not map how often each kind occurs.
- **An earlier 8B ladder** (continuation-only span, v0-map directions, two
  judges) read .52 / .61 / .81 at 0.125 / 0.35 / 0.75. The uniform re-measure
  reads lower at the same α. The two differ in span, direction source and
  judge set at once, and pairwise draw variance is large at these doses, so
  the ladder does not say which of those moved it.
- **About a quarter of judge calls abstained.** Every rung is read against
  the catch floor, not against 0.5, for this reason.

**Kit.** The kit ships the band. Its per-rung tallies, per-lever accuracies,
damage-cue rates and span are in the file, and `pleroma validate` reads its
stamp and span. Re-running the ladder needs paid judge calls.

## 3b. The 8B map steers, measured fan-relatively — below blind detection too

**Claim.** On Llama-3.1-8B-Instruct, wearing member j's lever pulls what the
model does toward future j, and not toward j's fan-mates. It does so at
α = 0.35, where the blind ladder (§3) is at its false-positive floor. A random
direction at the same per-site norm does nothing.

**Design.** 16 fans (seeded, unscreened, five prompt classes) × all 8 members.
Each member's lever is the map's HELD-OUT prediction (5-fold by prompt: a
conversation the map was not fit on), put on the kit ruler and injected
uniformly on the fan's own prompt. Every member is worn equally, so under "a
lever does not pull toward its own future" the member labels are exchangeable
within a fan: the null permutes which future each wear is prescribed to (20k
permutations). Δnr = 0.5 − the mean normalized rank of the prescribed future
(0 = no pull, 0.5 = always first). All arms of a fan share one sampling seed.

**Evidence.**

| readout | α 0.35 | α 0.675 | α 1.0 | random, α 0.675 |
|---|---|---|---|---|
| likelihood (teacher-forced, no judge) | **+.282** (16/16) | **+.278** (16/16) | **+.309** (16/16) | +.015 (p .30) |
| judged, Opus 5.5 | +.169 | +.296 | +.275 | −.041 |
| judged − length-only, Opus 5.5 | **+.105** (p .008, 12/16) | **+.200** (p 2e-4, 15/16) | **+.212** (p 2e-4, 15/16) | −.011 |
| judged, Fable 5 | | +.282 | | +.006 |
| judged − length-only, Fable 5 | | **+.186** (p 4e-4, 14/16) | | |
| **length-matched (±15%)**, Opus 5.5 | **+.125** | **+.243** | **+.236** | −.085 |
| length-only ranker, same matching | +.031 | +.050 | +.035 | −.040 |
| **length-matched (±15%)**, Fable 5 | | **+.256** | | −.030 |

Every v1a cell's permutation p is 5e-5 (the floor) unless given.

- **Likelihood**: log p of each future's exact sampled tokens, worn vs unworn,
  per token; the fan's futures ranked by that lift. Deterministic: no judge
  and no draw variance.
- **Judged**: a judge sees the user turn, the 8 futures (lettered, shuffled
  once per conversation) and the SET of 3 replies drawn under one wear, and
  ranks all 8 futures by resemblance in how they are built, not what they
  are about. It never sees the arm, dose or target.
- **Length is part of the pull, and not most of it.** Reply length moves
  toward the worn future's own length (Spearman +.25 to +.34 against the
  target's length, p ≤ .005; the random direction −.04), so length tracking is
  itself lever-specific steering. To see what the judge reads beyond it, the
  **length-matched** readout ranks the target only among fan-mates within ±15%
  (or ±25%) of its length, with the same permutation null re-deriving the
  window for every permuted target. A length-only ranker put through the same
  matching still leaks +.03 to +.05 (replies track length finely); the judges
  read +.13 to +.26, 3–5× that. Every matched v1a judged cell has p = 2e-4.
- **Two judges agree.** Per-set Spearman between Opus 5.5 and Fable 5 is .73
  (Fable against itself was .83 in the 3B readout, a cross-family judge .79).

**Read with.**

- **The pull saturates early.** Δnr barely rises from 0.35 to 1.0, while
  every future grows less likely (−0.05 → −0.43 nats per token) and the band
  finds damage from 1.0. More dose buys loudness and breakage, not direction.
- **16 fans, one draw set each.** The effect is large against its null; its
  spread across prompt classes is not mapped.
- 26 Fable calls were refused by the API (a false-positive policy category,
  15 of them on one fan) and are missing from its column. Opus 5.5 ranked
  every set.

**Kit.** The statistic (`pleroma.stats.fanrank`) and the set judge's
registered instruction text (`pleroma.judge.prompts.SET_RANK_INSTRUCTIONS`)
ship. The pipeline that picks fans, draws worn sets and judges them is not yet
packaged as a `pleroma` command. The judged readout cost about $9 with
Opus 5.5.

## 4. Uniform injection modestly beats continuation-only

**Claim.** Adding the lever at every position, prompt included, gives more
lift than adding it only to generated positions. The difference is small.

**Evidence** (3B). A judge-free screen: 14,400 generations, all at matched
absolute injected norm (unmatched comparisons measure magnitude and call it
mechanism). Uniform injection was ahead:

- in every time profile;
- at every scale ≥ 0.15;
- by +.02 to +.04 of lift. For example, at scale 0.4 with a constant
  profile: +.181 vs +.139.

Time-varying profiles (step, ramp) were worse than constant. A later control
correction (§6) weakened the same screen's high-dose headline. The uniform
vs continuation-only comparison itself is at matched norm and stands.

**Not reproducible here.** The screen's lever banks are not published.

## 5. On-trunk steering exists, and the fan-centred construction reproduces it

**Claim.** In the same conversation that produced the fan ("on-trunk"),
wearing a screened future at sane doses shifts a blind reader's choice
toward that future.

**Evidence** (3B).

- **The registered test**: 10 conversations, doses 0.35 and 0.50 on that
  map's ruler, 480 judged replies. It is a one-sided, stratified permutation
  test against each conversation's own floor.
  - T = +23.3 (null p99 = +12.3), p = 5e-5.
  - 9 of 10 conversations lifted positive.
  - Fluency was free: in 5 of 10 conversations the steered replies were
    *more* fluent than the unsteered ones.
- **Fan-centred (contrast) wear passes the same test** with zero retraining:
  T = +26.2, p = 5e-5. It sits at parity with the original construction,
  not above it: mean lift +.244 vs +.209, paired p = .15.

**Not reproducible here** as registered. It needs the judged battery.
**8B model**: the loom makes the same kind of comparison by hand. Wear a
future, send the same turn to the `base` branch, and read the pair.

## 6. Controls must be matched on damage, not on norm

**Claim.**

- At a fixed injected norm, how much a lever damages fluency depends
  heavily on its direction.
- A random vector at the same *norm* is therefore not a control.
- At high dose, most apparent "capture" is damage.

**Evidence** (3B).

- At matched injected norm, the fluency cost ranged from 0.92 to 7.46 nats
  by direction alone.
- In the original screen, the lever at scale 1.0 cost +5.69 nats and its
  norm-matched random control +0.14. That is a 40× gap.
- Matched on *damage* instead, a random vector reproduces about 89% of the
  high-dose lift. What survives is direction-specific but small, and it
  sits at low dose.

This is why `dose/damage` exists. It is also why high-dose effects are
never quoted here as steering.

**Not reproducible here.** It needs the lever banks.

## 7. Judges read length and form

**Claim.** Steering moves length, and LLM judges asked "are these
different?" answer mostly from length, form and breakage. Content counts for
little. Every judged steering effect must therefore be read beside a
length-only ranker.

**Evidence** (3B). 13,440 banked pairwise verdicts, split by the pair's
relative length gap:

| length gap | P(judged different) |
|---|---|
| < 5% | .354 |
| 5–10% | .413 |
| 10–30% | .576 |
| 30–60% | .810 |
| > 60% | .956 |

- The judges' own rationales cite the reply's speech act about 70% of the
  time, flat across gaps.
- They cite length in a growing share of rationales as the gap grows.
- At high dose they cite breakage.
- In a judged dose corpus, breakage was the only cue that tracked dose; that
  corpus's ladder ran past the damage point.

**Consequence.** `eval/length-null` requires the judged Δ to beat the
length-only ranker (character counts), paired by conversation.
`eval/positive-control` requires the judge to see a known effect above its
own floor before a null means anything.

**Not reproducible here.** The verdicts are not published. The gates run on
your own judged artifacts.

## 8. The gauge is a policy; the judge is the instrument

**Claim.** The free pick policies and the probe choose futures reasonably
well. They are not measurements of steering.

**Evidence** (3B).

- As a picker, the probe ("gauge-M2": wear each candidate briefly and see
  where the reply lands) captures 71% of an oracle's picking skill.
- As a *measurement*, it FAILed its registered test.
- `loudest` on the v1a map captures 43%, for free; `distinct` captures 33%.
- The probe adds about 19 s at k = 8 and does not resolve cheaply. When it
  reports `resolved`, that certifies only its top pick.

**8B model.** The policies run live on the kit (`auto` on `/loom`, and
`/probe`). Grading them against an oracle needs judged fans.

## 9. Signature-space geometry does not predict what a reader sees

**Claim.** Choosing or gating on a feature-space quality number is the
documented way to choose wrong. The component of a signature that
replicates is not the component that acts.

**Evidence.**

- **3B**:
  - Hidden-space dispersion *anti*-tracks behavioural spread: ρ = −0.34 for
    the participation ratio, while signature distance tracks it at
    ρ = +0.64. A real fork is a big move along a few directions; a flat fan
    is small wiggles in many.
  - Fan-level spread summaries correlate with readers at about ρ = 0.
- **8B** (replicates across the size rung): the 8B's basin geometry is
  1.94–3.6× more spread than the 3B's, while its text is only 5–8% more
  diverse. So better retrieval at 8B (§1) is not evidence of better fans.

**Not reproducible here.** It needs the fan corpora and judged spreads.

## 10. The map is a denoiser, not a compressor

**Claim.** Most of one sampled future's measured displacement is noise. The
map keeps the part that replicates. This answers "why not wear the hidden
states directly?"

**Evidence** (3B, seed twins drawn on an unbiased sample). A single future's
displacement replicates across twins at only r ≈ .21. Split by the map's
shared basis:

| part of the displacement | replication |
|---|---|
| inside the basis | .62 |
| outside the basis | .17 |
| null | .064 |

**Not reproducible here.** It needs twin draws and hidden-state banks.

## 11. How much a fan forks is set by framing, not size or weights, and forking is ambiguity about the speech act

**Claim.**

- Chat framing flattens fans. Removing it makes instruct weights fork
  almost as much as base weights.
- Going from 3B to 8B barely helps.
- Prompts fork when the model cannot tell what *kind* of reply is wanted.

**Evidence.**

- **Framing vs weights vs size.** Surface diversity (mean pairwise Jaccard
  on a length-controlled window; lower means more forking), on matched
  Llama-3.1-8B base and instruct, 43 prompts, k = 16:
  - chat → raw continuation, on 8B instruct: .2467 → .1015;
  - framing effect d = +.145 (p = 1e-5);
  - weights effect d = −.008 (not significant);
  - so framing is about 18× the weights effect.
  - On a matched battery of 1,805 + 815 prompts, the 8B's chat fans overlap
    only 4.7–7.9% (relative) less than the 3B's.
- **Few-shot framing** (3B, judged, 10 prompts). It raises judged spread
  sharply (+.485, 10/10 prompts), but saturates it, and on-task replies
  collapse from .596 to .217. It trades "too flat" for "off the prompt".
  The useful operating point lies between chat and few-shot.
- **What makes a prompt fork** (3B, judged screens). The judges'
  discriminators move from format ("list vs prose") on flat fans to the
  speech act on forking ones. Naming a dichotomy ("is it X or Y?") *pins*
  the speech act and flattens the fan. The strongest single generator is a
  collision of registers.
- **Judged spread predicts steerability** (3B, ρ = +.83 over 9
  conversations). It is a property
  of the conversation, measurable before any steering: the intraclass
  correlation across redraws is .82.

**8B model.** The framing comparison is text-side and costs no judge calls.
Draw fans from the 8B in chat and raw framing and compare them.

## 12. What the readouts can and cannot resolve

**Claims.**

- **Map-vs-map comparisons are closed at affordable sample sizes.** At
  13 conversations × 12 replies, the smallest detectable difference is
  +.089. The v1a-minus-v0 difference measured on 3B, +.009, would need about
  1,181 conversations. So nothing here claims v1a *steers* better than v0.
  It retrieves better (§1).
- **Judged conclusions hold across judges down to a floor.** On one
  material, seven of eight judges reached the same conclusion, down to
  about .45 of the instrument's test-retest ceiling; it broke by about .40.
  Cheaper judges compress effect sizes, by up to 37% here.
- **Reasoning tokens, not headline price, set a judge's cost**: a 50×
  spread on identical inputs. Stage a small batch and read the usage before
  buying a run, and guard spend above the projection.

**Not reproducible here.** These are properties of judged runs.

## 13. Engineering facts that change how you read results

- **Codes are tied to one export.** A reproduction of a served export on a
  newer software stack produced the same map: the weights matched to
  2.6e-12. Yet half of its 64 singular-vector pairs had flipped sign. File
  shas and code orientation do not reproduce across stacks. Compare levers,
  not codes, across exports.
- **Codes are relational.** A code worn on a different conversation carries
  no content, and neither construction ports (3B, registered). A code
  means something only relative to the fan and conversation it came from.
- **Loom latency is mostly harvest.** On 3B, a k = 8 turn spent 1.5 s
  generating and 13.5 s harvesting, and 79% of the harvest was CPU feature
  math. A pool of persistent harvest workers is 3–4× faster. On 8B with two
  pooled workers, a k = 8 draw took 12.3 s: 3.1 s generating, 9.2 s
  harvesting.
- **Llama chat templates embed today's date** unless it is pinned. A prompt
  whose date drifts is a different prompt: the date lines were about 26 of
  40–57 prompt tokens in one calibration build. A profile pins it with
  `format.date_string`, and `format/template` (given `--tokenizer`) checks
  the render for wall-clock dependence.
