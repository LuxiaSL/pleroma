# Contributing

Issues and pull requests are welcome, especially reproductions: runs of the
8B kit, `pleroma validate` receipts, and cases where a gate said PASS and
should not have.

## Running the tests

```sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough
uv pip install -e ".[dev]"
python -m pytest -q
```

- The suite needs no GPU, no network after install, and no API keys.
- Tests that build a model use a tiny random Llama on CPU.
- UI contract tests skip unless `ui/node_modules` exists. Run `npm ci` in
  `ui/` to include them.

The front-end has its own checks:

```sh
cd ui && npm ci
npm run check:api && npm run typecheck && npm test
npm run e2e        # Playwright, against the mock backend
```

`pleroma/serve/api-schema.json` is the API contract. If you change a route's
request or response, regenerate the schema and the UI's generated types
(`npm run gen:api`). The tests fail if either is stale.

## The documentation rule

These rules and their checkers come from
[anamnesis](https://github.com/LuxiaSL/anamnesis/blob/main/CONTRIBUTING.md),
this project's instrument. They apply to three surfaces:
- comments;
- docstrings;
- the message a `raise` or a logging call prints.

They do not apply to wire format: a fixture, a dict key, a status value, a
filename or a label that banked results carry. The program keys on that
content. Prose that a run writes for a human (a `note`, a `reason`, a report
caption) is read like a comment, so hold it to the same rules. The checkers
cannot see it, so it is a reviewer's catch.

### 1. State what is true now

A comment describes the code as it stands, not how it got there.
- No `previously`, `used to`, `changed in`, `we now` or `no longer`.
- No bare `TODO`, `FIXME` or `HACK`. A known gap is one of three things: a
  refusal the code makes, a test that pins the current behaviour, or an
  issue.
- **No dates**, not on an edit and not on evidence. Git records when a line
  changed, and a dated measurement belongs in the record that holds it.
  What the code needs is the standing fact it depends on, in the present
  tense.

### 2. Every referent must be reachable from this repository

A reader holding only this repository must be able to look up anything a
comment names. That means one of:
- a module path here;
- a file in this tree;
- a symbol this package defines;
- a public URL.

It never means any of these:
- a private tree or a planning document;
- a pre-registration or a `§` of a document that is not here;
- an arm code or a commit hash;
- "see the earlier discussion";
- a machine or a host.

Numbered sections of a document that *is* here are fine: `docs/FINDINGS.md §3b`
resolves. When the substance is short, state it. When it is long and public,
link it. When it is long and private, restate the one sentence this code
depends on.

A claim in a docstring must match the code under it. Describe what is.

## The gates

A pull request merges when these pass. Each one is a command you can run,
and CI runs all of them.

| gate | what it checks | command |
|---|---|---|
| tests | the suite | `python -m pytest -q` |
| G3 documentation | the two rules above | `python -m tools.check_timelessness --root pleroma tools tests` and `python -m tools.check_referents --root pleroma tools tests` |
| G4 import closure | every module is reachable from a test or an entry point | `python -m tools.check_import_closure --package pleroma --roots tests --allow-orphan pleroma.__main__ --allow-orphan pleroma.serve.__main__` |
| G2 test retention | a change that shrinks the suite shrinks the code by at least as much | `python -m tools.check_test_retention --base main` |

Each G3 rule is pinned by a corpus, `tests/unit/tools/test_gate_fixtures.py`,
not by a reader's judgement. The corpus lists the strings each checker must
catch and the prose it must leave alone. Closing a blind spot starts with
adding the string that slipped through.

Both pattern files carry a reason above every entry:
- `tools/timelessness_allowlist.txt`;
- `tools/referents_allowlist.txt`.

An entry claims that its reason is true *now*. When the reason expires, the
entry goes.

A gate may be waived only in writing, with the reason beside the receipt.
The gate itself is never rewritten to pass.

## The validate philosophy

Most of this project's retracted conclusions came from a number read
without its control:

- a judged effect that was really a length difference;
- a dose that meant something else on another map's ruler;
- a retrieval score on a test that was already saturated.

`pleroma validate` exists so the tool flags these itself. When you add a
measurement or a stage:

- **Give it a gate, not just a number.** A gate returns PASS, FAIL or
  INCONCLUSIVE with a named reason a newcomer cannot misread. The numbers
  go beside the verdict as labelled evidence.
- **Missing evidence is INCONCLUSIVE, never PASS.** An artifact that is
  present but malformed is a FAIL. An unexpected exception inside a gate is
  INCONCLUSIVE naming it, never a traceback.
- **Report the control next to the effect**:
  - the length-only ranker next to every judged Δ;
  - the judge's own measured floor, never 0.5;
  - a damage-matched random vector, not a norm-matched one;
  - held-out numbers, never in-sample.
- **Do not select on fit.** A better feature-space number is not evidence of
  better steering (`docs/FINDINGS.md` §9).
- **Carry the ruler.** Any dose you record names its map's fingerprint,
  its `norm_ref` and the injection span it was applied under.
- **No number without a receipt.** A number in a document must be findable
  in a file a reader can run or open.

## Pinned things

These are pinned on purpose. Change them only deliberately, and in their
own pull request:

- **anamnesis**: the pin lives in three places, which tests keep equal:
  - `pyproject.toml`;
  - `pleroma.harvest.anamnesis_seam.PINNED_COMMIT`;
  - the `vendor/anamnesis` submodule.

  Bumping it is a parity question: the CPU feature space must reproduce its
  reference fixture bit for bit.
- **The shipped profiles' values.**
- **The frozen v3 capture surface.** A published map is married to the
  exact feature space it was fit in, and it refuses any other.

## License

By contributing you agree that your contributions are licensed under the
MIT License (see `LICENSE`).
