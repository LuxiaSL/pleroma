"""A banked lever file made wearable (the bank `pleroma.map.build.levers` writes; /wear_code)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pleroma.levers.ruler import apply_ruler

if TYPE_CHECKING:
    from pleroma.map.loom_map import LoomMap

logger = logging.getLogger("loom_serve")


class LeverBank:
    """A banked lever bank (the one `pleroma.map.build.levers` writes) made wearable by /wear_code.

    ★ WHY THIS EXISTS. `/wear` can only wear candidate i of the CURRENT fan, and
    whether a given fan carries the contrast an operator wants depends on the
    draw: a fan can come back with all its futures in one register. A banked
    lever comes from the atlas instead, where the extremes of a known axis are a
    GUARANTEED contrast with no seed luck. It is a second route to a contrast
    beside the fan pick, not a replacement for it.

    ★ THE RULER IS THE WHOLE POINT. A bank lever is renormalised per site to the
    map's own `norm_ref`, exactly as `LoomMap.lever_of` does for a fan candidate.
    Without that, alpha means one thing for a fan wear and another for a landmark
    wear: two maps' rulers can differ by a factor of about 2.1 per site, so the
    same alpha read on the wrong ruler is a different dose
    (docs/FINDINGS.md §2). Sites and hidden dim are checked against the live
    map and a mismatch REFUSES rather than injecting at the wrong layers.
    """

    def __init__(self, path: Path, loom_map: "LoomMap") -> None:
        blob = np.load(path, allow_pickle=True)
        for key in ("levers", "group_prompt_ids", "group_waves", "sites"):
            if key not in blob.files:
                raise ValueError(
                    f"{path}: lever bank is missing {key!r} "
                    f"(has {sorted(blob.files)}) — is this a lever bank from pleroma.map.build.levers?")
        self.levers = np.asarray(blob["levers"], dtype=np.float32)
        if self.levers.ndim != 3:
            raise ValueError(f"{path}: levers must be [G, S, hidden], "
                             f"got {self.levers.shape}")
        bank_sites = [int(x) for x in blob["sites"]]
        if bank_sites != list(loom_map.sites):
            raise ValueError(
                f"{path}: bank sites {bank_sites} != map sites "
                f"{list(loom_map.sites)} — refusing, a landmark wear would "
                "inject at the wrong layers")
        if self.levers.shape[1] != loom_map.n_sites or \
                self.levers.shape[2] != loom_map.hidden:
            raise ValueError(
                f"{path}: levers are {self.levers.shape[1]}x{self.levers.shape[2]}"
                f" but the map is {loom_map.n_sites}x{loom_map.hidden}")
        self.labels = [f"{p}|{w}" for p, w in
                       zip(blob["group_prompt_ids"], blob["group_waves"])]
        self.classes = ([str(x) for x in blob["group_prompt_classes"]]
                        if "group_prompt_classes" in blob.files
                        else [""] * len(self.labels))
        self.index_of = {lab: i for i, lab in enumerate(self.labels)}
        self.path = str(path)
        self.norm_ref = loom_map.norm_ref
        logger.info("LEVER BANK: %d groups from %s, sites %s — wearable via "
                    "/wear_code", len(self.labels), path, bank_sites)

    def vectors_for(self, label: str) -> tuple[np.ndarray, list[float]]:
        """Norm-matched lever for a group label, plus its RAW per-site norms.

        The raw norms are returned for the record: they are the bank lever's own
        loudness before the ruler is applied, the analogue of `lever_of`'s
        `raw_norms`, and the thing to look at if a landmark behaves oddly.
        """
        i = self.index_of.get(label)
        if i is None:
            raise ValueError(
                f"group {label!r} is not in {self.path} ({len(self.labels)} "
                "groups; labels look like 'w5_og01155|orig')")
        lever = np.asarray(self.levers[i], dtype=np.float64)
        return self._match_norms(lever, label)

    def random_vectors(self, seed: int) -> tuple[np.ndarray, list[float]]:
        """A matched-norm RANDOM direction: the control for a direction readout.

        ★ WHY IT LIVES HERE. The control asks whether a readout detects
        *direction* or merely *perturbation*: the random direction must read at
        chance, or the landmark result is void whatever its value. That needs a
        vector with the SAME per-site injected norm as a real landmark and no
        relationship to any basin, and it has to pass through the identical
        `norm_ref` renormalisation or the two are not comparable (the ruler
        mismatch above, in miniature).

        ★ On norm-matching vs damage-matching. When the question is CAPTURE
        MAGNITUDE, a norm-matched control is not a control: direction alone moves
        fluency cost from about 0.92 to 7.46 nats, so a norm-matched random
        direction understates damage. That does not bite here: a decoy design
        asks which of two futures one reply resembles, so damage is a property
        of the single reply and is identical on both sides of the comparison —
        **uninformative about the question being asked**. Norm-matching is
        therefore the right and sufficient match for this control's job, which is
        isolating direction from perturbation.

        Returns ``(per-site vectors, per-site norms)`` from `_match_norms`.
        """
        rng = np.random.default_rng(int(seed))
        draw = rng.standard_normal((self.levers.shape[1], self.levers.shape[2]))
        return self._match_norms(draw, f"random(seed={seed})")

    def _match_norms(self, lever: np.ndarray,
                     label: str) -> tuple[np.ndarray, list[float]]:
        """Scale each site row to the map's own `norm_ref`, reporting raw norms.

        The raw norms are returned for the record: they are the vector's own
        loudness before the ruler is applied, the analogue of `lever_of`'s
        `raw_norms`, and the thing to look at if a landmark behaves oddly.
        """
        out, raw = apply_ruler(lever, self.norm_ref, what=label)
        return out.astype(np.float32), raw


