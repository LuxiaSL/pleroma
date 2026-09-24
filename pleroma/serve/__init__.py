"""LOOM v0 — the model looms over its own futures and bends toward one you pick.

The loop, live, end to end (the intent: "a signature of trajectory i steering
a fresh run IS a future influencing a present"):

    /chat            talk to either branch (base | loom), duet conventions
    /loom            from the LOOM branch's current context: sample K futures,
                     harvest each future's v3 signature + bins-B features
                     (``pleroma.harvest.replay`` + ``pleroma.harvest.bins`` —
                     the SAME frozen instruments every banked corpus went
                     through), push each through the fitted map g (the
                     ``--map`` npz) -> K candidate levers, norm-matched per
                     site to the bank's alpha=1 scale.
                     Drawn UNDER current wear by default; opt-in
                     detach_wear=true (or --detach-wear-default) draws with
                     the wear's hook detached instead, worn code retained for
                     scoring — the unworn-draw protocol, which keeps a fan
                     from being drawn under the very lever it will be
                     compared against
    /wear            pick candidate i at dose alpha: the loom branch wears
                     that future's code on every subsequent turn. Optional
                     "dose_policy": "flat" (the DEFAULT, the dose every
                     banked result was measured at) or
                     "predicted" (opt-in: divide by the FAN's mean raw norm
                     instead of the candidate's own, so the map's within-fan
                     magnitude prediction survives the ruler — v1a predicts
                     each candidate's magnitude within its fan at r=.757. The
                     fan's average dose is unchanged so alpha keeps its
                     meaning; per-site scale clamped to [0.5, 2.0] and every
                     clamp reported. BEHAVIOURALLY UNVALIDATED.)
                     Optional "lever_kind": "absolute" (the DEFAULT, W on
                     the candidate's own input row, the lever every banked
                     result wore) or "contrast" (W on the row
                     minus the mean of the fan's other valid rows — the
                     one-vs-rest object v1a was fit on; server default via
                     --wear-lever-default. BEHAVIOURALLY UNVALIDATED.)
    /probe           THE PROBE (opt-in, NOT free): run
                     the gauge probe on the current fan — wear each candidate's
                     CONTRASTIVE code, generate a short continuation, harvest
                     its v3 signature through the same frozen path, and score
                     by where the reply landed among the futures (normalized
                     rank by cosine distance in z space; cell = mean base nr −
                     mean steered nr). As a PICKER it captures 71% of an
                     oracle's picking skill vs `distinct`'s 33%, and it is the
                     one policy that needs a probe rather than the draw alone.
                     As an INSTRUMENT it failed its registered test: it picks,
                     it does not judge (docs/FINDINGS.md §8). Also reachable
                     as "probe": true on /loom, which makes
                     auto.policy="gauge" work in one call. Costs ~23 s at k=8
                     on top of the draw — /info prices your settings.
    /unwear /reset /info
    /sessions        GET: what is on disk (tags, turn counts, worn, mtime)
    /restore         POST: reload one session's snapshot from disk

Sessions are DURABLE: every mutating request writes a small
human-readable JSON snapshot (conversation turns, worn-lever metadata, loom
history) under <work-dir>/sessions/, atomically, and the server reloads them
on startup (--no-restore opts out). The big arrays are NOT in the snapshot —
they stay in the loom dirs, and a restored wear is rehydrated from the
future's banked signature when that dir is still there.

Conventions shared with the duet server this one grew from, deliberate:
loopback-only bind (``pleroma.net.require_loopback``); re-prefill every turn,
no KV across requests; UNIFORM injection (start_pos=0); one persistent
generation thread with cuDNN SDPA off (cuDNN SDPA builds a host-side graph
for every unseen sequence-length shape and caches it per thread); byte-identity
gate before the socket opens; per-(session, branch, turn) seeds.

v0 honesty: the map was trained on single bare prompts' continuations, and
here it reads futures of a multi-turn chat context — off-distribution for g,
an exploratory instrument, not a measured one.

PROMPT MODES (--prompt-mode {chat,raw,modelc}, default chat)
─────────────────────────────────────────────────────────────────────────────
Rendering through ``apply_chat_template`` alone would cost two things: the
server could not serve model C (whose tokenizer carries no ``chat_template``
key at all — it is document-trained), and it could not reach RAW
CONTINUATION on any model. The second is the expensive one. On 8B instruct,
chat -> raw moves mean pairwise fan Jaccard **0.2467 -> 0.1015** (paired
d = +0.1452, p = 1e-5, 43/43 prompts) — the single biggest fan-diversity
lever in the project, 8-13x the 3B->8B size lever (docs/FINDINGS.md §11).
It also has a bill: ``on_task`` **0.582 -> 0.366** (8B instruct), i.e. the
raw arm buys its spread partly by wandering off the prompt into unrelated
documents. Both numbers are the point of the flag; neither is hidden behind
it.

  chat    THE DEFAULT. ``apply_chat_template(messages,
          add_generation_prompt=True)``, one call. Every banked number was
          measured through this render, so this path is kept literal — not
          routed through the mode helpers — and ``test_loom_prompt_mode.py``
          asserts token-id identity against that literal call.

  raw     No template. The prompt is the conversation's accumulated plain text
          plus the new text, encoded with ``add_special_tokens=True`` (BOS
          once, nothing else) — the same raw render the .1015 was measured
          with.

          ★ HOW MULTI-TURN HISTORY IS JOINED, and why. Turn texts are
          concatenated in order with a BLANK LINE between them and NO role
          markers of any kind (``RAW_TURN_JOIN = "\\n\\n"``). The two honest
          alternatives were single-prefix-only (history disabled) and a
          role-marked scaffold; both were rejected, for stated reasons:

            - A role-marked join ("User: ... / Assistant: ...") is a template
              by another name. That scaffold is the FEWSHOT framing, which
              measures differently (jacc .099, on_task .14-.20). Shipping it
              under the name "raw" would make this server report one cell's
              label over another cell's behaviour, which is precisely the
              silent mis-measurement the flag exists to avoid.
            - An empty join ("") is not "no format" either. This server
              ``.strip()``s every banked turn before storing it, so "" would
              fuse the last word of one turn to the first of the next and
              produce token sequences no model ever generated — a worse
              invention than a paragraph break, and an invisible one.
            - Disabling history entirely would have been defensible, but it
              breaks the interaction this mode is FOR: put text in, draw
              continuations off it, wear a future, continue the same document.
              That loop needs the document to accumulate.
            - A blank line is the one join that is ubiquitous in pretraining
              as a neutral paragraph boundary and carries no role information.

          ★ AND THE HONEST BOUND ON IT: the .1015 was measured at ZERO
          history — a single bare prefix. At turn 0 this implementation is
          byte-identical to that render and the join never appears. Everything
          past turn 0 is an extrapolation this docstring NAMES rather than
          hides; the multi-turn raw regime has not been measured.

  modelc  Model C's document format, verbatim (``pleroma.format.modelc``):

              {header}

              Full conversation with Model C:

              **User:** {turn}

              **Model C:** {reply}

              **User:** {next turn}

              **Model C:**

          ending at ``**Model C:**`` with NO trailing space. Headers are
          selectable (--modelc-header stranger|claude|returning|reader|bare, or
          any literal custom line); the default is the stranger header.
          Sampling defaults become temperature 1.0 / top_p 0.98 — the
          format's standing house rule, emphatically 0.98 and not 0.95 —
          unless the operator passes --temperature/--top-p explicitly. Stops
          are the format's driven-request list. A ``**User:**`` block INSIDE a
          generation is documented "dream mode", the model imagining the
          visitor: it is real output, it is not degeneration, it is NOT
          filtered, and it is counted per future as ``modelc_dream_blocks``.

          ★ TAILS: HF ``generate(stop_strings=)`` KEEPS the matched stop, so
          an untrimmed stopped reply ends with a literal ``\n\n**User:**`` —
          which would be shown in the UI, counted as a word, and fed back into
          the next turn's document as an empty visitor block.
          ``pleroma.format.trim.split_modelc`` separates each generation into
          the trimmed reply (``text`` / ``content`` / ``reply``) and the
          dreamed remainder (``dream``, with ``tail_kind``); nothing is
          deleted, and lengths read the reply only. Batched rows finished by a
          stop string are not pad-filled into the harvest (``trim_to_eos``).

The boot guard is conditional: chat mode requires a chat template, the
other two do not. Nothing downstream of the draw is mode-aware — the harvest
(``pleroma.harvest.replay`` / ``pleroma.harvest.worker`` /
``pleroma.harvest.bins``) replays each future's SAVED ``input_ids``
teacher-forced and never re-renders a prompt, so the frozen instruments are
untouched by this flag.

★ THE BINS FLOOR. ``pleroma.harvest.bins.bin_edges`` REFUSES a prompt shorter
than ``n_bins`` (20) — "absence raises, never zero-fills". The chat template
adds ~34 tokens to every prompt, so in chat mode this floor is unreachable in
practice. In raw mode it is NOT: a 19-token prompt such as
"how would you choose to speak, if you could speak in any way you desireD?"
fails the bins half of the harvest for every future, and the fan comes back
TEXT-ONLY — replay/v3 signatures fine, bins-B features absent, therefore no
map output, no candidate levers, no spread, no ``fan_mean_raw_norms``, and
/wear impossible. The futures themselves are real and the diversity
measurement stands; the STEERING half of the loom simply is not available
below the floor. ``n_bins`` cannot be lowered to dodge it — the served map's
feature layout is bins20 — so the operational rule is: **a raw prompt must be
>= 20 tokens for the loom to steer.** /loom publishes ``prompt_length``,
``bins_prompt_floor`` and ``bins_feasible`` on every draw and logs a warning
below the floor, so this is never read as "the harvest broke".

Usage (from the repo root, inside the project venv):
    python -m pleroma.serve.legacy \\
        --map data/loom/loom_map_3b.npz \\
        --discriminants data/discriminants/factor_directions_3b.npz \\
        --calib-dir data/calibration/3b \\
        --model-path <hf-id-or-local-dir> --preset 3b \\
        --work-dir outputs/loom/sessions --port 8767
Remote GPU box?  ssh -L 8767:localhost:8767 <host>  and browse localhost:8767.

PACKAGE LAYOUT: the server is split by concern across this package —
``app`` (startup and ``main``), ``args``, ``runtime`` (model and render),
``session`` and ``persistence`` (state and snapshots), ``harvest_client``,
``http`` and ``api`` (routing), ``routes/`` (one module per endpoint family),
``models`` and ``responses`` (request and response shapes), ``schema``,
``info``, ``policies``, ``draws`` and ``atlas``. ``pleroma.serve.legacy`` is the flat ``loom_serve`` namespace:
it re-exports every name and delegates ``main()`` to
``pleroma.serve.app.main``. Importing this package is torch-free and has no
side effects; the thread-pool pins and logging setup run in the entry points
(``pleroma.serve.legacy``, ``__main__``).
"""

from __future__ import annotations
