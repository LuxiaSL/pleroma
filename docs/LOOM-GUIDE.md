# LOOM operator's guide

*For any operator, human or agent. The web panel at `/` exposes everything
below; this guide is the raw API and the measured doctrine underneath it.*

## 0. What this is, in three sentences

You converse with a frozen model (Llama-3.2-3B-Instruct as shipped) that can
preview K possible futures of the conversation, compress each future's
*computational manner* into an 8-number code (a rank-8 read of its internal
path signature — no labels anywhere in the pipeline), and then WEAR a chosen
future's code as a residual-stream steering vector, bending the present
toward that future. Wear carries disposition/register, not content: pick a
future for a detail and you get its *manner*, not the detail.

```bash
BASE=http://localhost:8767
curl -s $BASE/info | head -c 200   # JSON with "map" means you're in
```

## 1. The statefulness contract

**The server holds ALL conversation state, per session, in memory.** A
client never needs to carry transcripts; your persistent footprint is a
session name (pick your own prefix; anyone on the same server shares the
namespace) plus whatever notes you keep.

Resume anytime:

```bash
curl -s $BASE/state?session=$S     # full histories (both branches), worn code,
                                   # last fan with scores, any loom in progress
```

**The one hole, stated honestly:** state is in-memory — a server restart
loses it, and there is no history-restore endpoint yet. Cheap insurance:
after any turn you'd mind losing, `curl -s $BASE/state?session=$S >
snap.json`. If the server restarts, start a fresh session and paste a short
summary as your first turn.

## 2. Endpoints (POST JSON unless GET; errors are `{"error": str}` with 4xx/5xx)

### Conversing
| verb | body | returns |
|---|---|---|
| `/chat` | `{session, branch: "base"\|"loom", text}` | `{reply, n_turns, worn, elapsed_s, ...}` |
| `/reroll` | `{session, branch}` | like /chat + `{replaced, reroll_n}` — fresh seed, same user turn, UNDER current wear on loom |
| `/edit` | `{session, branch, text}` | like /chat + `{replaced_user, replaced_assistant}` — atomic replace-last-and-regen |
| `/undo` | `{session, branch}` | `{popped: {user, assistant}, n_turns}` — nothing destroyed silently |
| `/truncate` | `{session, branch, keep_turns}` | keep first N exchanges (deep-edit primitive) |
| `/reset` | `{session}` | clears BOTH branches, unwears, forgets candidates |

Two branches per session: `loom` (bendable) and `base` (untouched control —
send it the same turns for a counterfactual pair).

### Looming
| verb | body | returns |
|---|---|---|
| `/loom` | `{session, text, k=6, horizon=192, auto?}` | `{loom_id, futures: [...], spread, auto_selected?, worn, timing_s, harvest_via}` |
| `/wear` | `{session, index, alpha, loom_id}` | `{worn}` — always pass the loom_id from the draw you read |
| `/unwear` | `{session}` | `{worn: null}` |
| GET `/loom/progress?session=` | — | `{progress: {stage, done, total}\|null}` — poll ~1s during a loom |

`/loom` does NOT send the message — it previews K replies to your
*contemplated* next turn. `/chat` the same text afterwards to commit.
Futures are drawn UNDER current wear (`drawn_under_wear` says so).

Each future: `{index, text, n_tokens, harvested, note, scores, code}`.
- `scores.distinct` — mean signature-space cosine distance to the rest of the fan.
- `scores.camp` — 2-means camp over THIS fan (labels are per-draw only).
- `scores.loudness` — the map's raw output norm; judge against `spread.loudness_ref`.
- `scores.cos_to_worn` — lever similarity to the worn code (when worn).
- `code` — the 8 numbers that ARE what transfers (anonymous SVD axes; GET
  `/atlas` gives per-axis quantiles, extreme groups, and all banked
  landmark coordinates).
- `spread.mean_pairwise_distance` — the fan's forkedness. **It decays as a
  conversation commits (measured ~.5 → ~.16 over 8 turns): loom early, at
  forks; late looms buy paraphrases.**

### Auto-loom (the closed loop)
`auto: {policy, alpha}` on `/loom` makes the server pick AND wear:
- `distinct` (most its-own-path), `loudest`, `minority` (smaller camp; falls
  back to distinct under 3 scored — `auto_selected.effective_policy` tells
  the truth), `stay` / `swerve` (max/min similarity to worn; need something
  worn — seed turn 1 with distinct).
- GET `/info` advertises `auto_policies` with `needs_wear` flags.
- Known dynamics at α=0.35: `stay` locks onto itself (cos_to_worn → .9+),
  `swerve` churns; macro-effects want α ≥ ~0.7.

### Dose doctrine (measured on the shipped 3B, not vibes)
α ∈ [0.125, 0.5] subliminal (steers, conversationally invisible);
~0.75 threshold; 1.0 audible; **1.5 = word-salad damage — don't**.
Wear predicted codes raw; whitening/de-tail "repairs" only hurt.
On a new model, re-derive this table before trusting any of it.

## 3. Recipes

Minimal auto loop, subliminal, stateless from your side:
```bash
S=drift-$(date +%m%d); BASE=http://localhost:8767
say() { curl -s $BASE/loom -d "{\"session\":\"$S\",\"text\":$1,\"k\":6,\"auto\":{\"policy\":\"distinct\",\"alpha\":0.35}}" >/dev/null
        curl -s $BASE/chat -d "{\"session\":\"$S\",\"branch\":\"loom\",\"text\":$1}" | jq -r .reply; }
say '"hey — pick a thing you want to think about today."'
```

Manual curation: `/loom` without `auto`, read `futures[].text` + `scores` +
`code`, `/wear {index, alpha, loom_id}`, then `/chat`. Your judgment
outranks the scores — they are advisory.

Counterfactual pair: send every turn to BOTH branches; diff the replies.

A/B policy arms in bulk: `python -m pleroma.loom_driver` (see its docstring)
runs scripted multi-turn conversations under different auto policies with
/loom-every-turn as a free measurement.

## 4. Sharing one server

- One generation thread serves everyone: your /loom queues behind other
  users' turns and vice versa. One loom at a time; poll progress.
- The API has **no auth by design** — loopback bind is the default boundary.
  `--allow-lan` only on a network you trust end to end.
- Session names are the only namespacing: use a personal prefix, never touch
  a session you didn't create, never `/reset` one that isn't yours.

## 5. What to expect (so surprise is signal, not confusion)

The bend is front-loaded; the fan collapses as the conversation commits;
wear moves register and disposition, not specific claims. If you pick a
future for one striking phrase, expect the phrase to wash out and the
*manner* to arrive. That gap between what you aimed at and what changed is
the interesting object — log it when you feel it.
