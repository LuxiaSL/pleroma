# Vendored packages — provenance and deltas

Pleroma's harvest path runs every future through the **frozen instruments**
that produced all banked corpora: anamnesis' extraction stack and
kv-rotation's sigbridge (the v3 feature-space config). Both are vendored here
as snapshots so the repo is self-contained; neither is a submodule, and
neither should be upgraded casually — the shipped map/atlas/levers were fit
in exactly this space, and a changed extractor silently invalidates them
(run_replay_b0's GATE 0 will catch a changed feature *list*, not changed
feature *values*).

## anamnesis/ (trimmed snapshot)

- Source: the `anamnesis-pl` pipeline package (private), snapshot 2026-09-17,
  taken byte-for-byte from the copy the live loom server imported.
- Trim: `config.py`, `extraction/**`, `modes/`, `prompts/` only — the
  analysis and ops subpackages are not part of the harvest path and are not
  shipped.
- Upstream module shas at snapshot (sha256, pre-scrub):
  - `config.py` `ec4a87e2…270d09`
  - `extraction/feature_pipeline.py` `bfb27e09…b9280f`
  - `extraction/model_loader.py` `f1e3a714…6457a6`
  - `extraction/replay_extract.py` `0276ace2…60a5a1`
  - `extraction/raw_saver.py` `d17776df…438910afd5a` (truncated)
- Deltas from upstream (comment/cosmetic only, no behavior change):
  hostname references in comments genericized (`config.py`,
  `extraction/feature_pipeline.py`).
- License: MIT (upstream anamnesis).

## kvrot/ (trimmed snapshot)

- Source: the `kv-rotation` repo (private), `src/kvrot`, snapshot 2026-09-17;
  `sigbridge.py` sha256 `60e81c05…711d48f` upstream.
- Trim: `__init__.py`, `config.py`, `eviction.py`, `metrics.py`, `rope.py`,
  `snapshot.py`, `sigbridge.py` (the package init's closure plus sigbridge —
  the single source of truth for the v3 extraction/family config). The
  GPU harness modules are not shipped.
- Deltas from upstream: `sigbridge.DEFAULT_ANAMNESIS_PATHS` now prefers the
  repo root (the vendored `anamnesis/`) over private checkout paths, and two
  comments were genericized. No extraction semantics changed.

## Why the versions are pinned

`data/loom/loom_map_3b.npz` records the sha256 of
`data/discriminants/factor_directions_3b.npz` and refuses to serve against
any other file; the discriminants were fit in the space this exact extractor
emits. If you change the vendored extractor for a NEW model's space, that is
fine — you will be bootstrapping a fresh manifest with
`run_replay_b0 --freeze-names-to` anyway — but the shipped 3B artifacts stay
married to this snapshot.
