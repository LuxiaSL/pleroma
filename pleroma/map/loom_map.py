"""The served map: raw signature -> input row -> lever (and code).

`pleroma.serve.legacy` re-exports it. Every per-site renormalisation goes
through `pleroma.levers.ruler`, the one ruler, so every lever kind agrees on
what alpha means and on how a near-zero or non-finite row is refused.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pleroma.levers.ruler import apply_ruler
from pleroma.map.identity import MapCode, fingerprint_v2, same_map, weights_digest
from pleroma.map.svd import is_canonical

#: Relative Frobenius gap allowed between a stored W and its stored factors
#: (both are float32 on disk, so exact equality is not available).
FACTOR_AGREEMENT_TOL: float = 1e-4


class LoomMap:
    """A fitted map file (the npz `pleroma.map.build.export` writes) plus its
    discriminants file: the full raw-signature -> lever inference pipeline.

    Raises KeyError when the map carries no weights, ValueError when its
    stored factors do not factor its stored W or when the discriminants file
    is not the one the map was fitted against (sha mismatch)."""

    def __init__(self, path: Path, discriminants: Path):
        with np.load(path, allow_pickle=True) as npz:
            # Identity over the STORED weights: see pleroma.map.identity.
            weight_arrays = {k: npz[k] for k in npz.files
                             if k in ("W", "W_U", "W_S", "W_Vt")}
            # (no weights at all is refused just below, with the format's
            # own message)
            self.weights_digest: str = (weights_digest(weight_arrays)
                                        if weight_arrays else "")
            # Two storage forms: full W [in, out] (large), or the rank-r SVD
            # factors W_U/W_S/W_Vt (small — what ships; W is rank-truncated at
            # fit time so the factored product IS W up to float roundoff).
            self._factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None
            has_factors = {"W_U", "W_S", "W_Vt"} <= set(npz.files)
            if "W" in npz.files:
                self.W = np.asarray(npz["W"], dtype=np.float64)
                self._factors = None
                if has_factors:
                    # ★ A map may ship BOTH. The stored factors are
                    # the fit's own SVD — the basis its codes were written in.
                    # Re-deriving an SVD from the (float32) W can flip a row's
                    # sign, and a banked code would then expand to the wrong
                    # lever. Keep W for the arithmetic; take the code basis
                    # from the factors, and refuse if they are not one W.
                    u_ = np.asarray(npz["W_U"], dtype=np.float64)
                    s_ = np.asarray(npz["W_S"], dtype=np.float64)
                    vt_ = np.asarray(npz["W_Vt"], dtype=np.float64)
                    gap = float(np.linalg.norm((u_ * s_) @ vt_ - self.W)
                                / max(np.linalg.norm(self.W), 1e-300))
                    if gap > FACTOR_AGREEMENT_TOL:
                        raise ValueError(
                            f"{path}: stored W_U/W_S/W_Vt do not factor the "
                            f"stored W (relative gap {gap:.2e} > "
                            f"{FACTOR_AGREEMENT_TOL:g}) — the map is internally "
                            "inconsistent; refusing to guess which is right")
                    # float32 on disk: re-orthonormalise the stored rows
                    # (QR, signs kept) so the code algebra is exact again in
                    # the STORED orientation.
                    q, r = np.linalg.qr(vt_.T)
                    vt_ = (q * np.where(np.diag(r) < 0, -1.0, 1.0)).T
                    self._factors = (u_, s_, vt_)
            elif has_factors:
                u_ = np.asarray(npz["W_U"], dtype=np.float64)
                s_ = np.asarray(npz["W_S"], dtype=np.float64)
                vt_ = np.asarray(npz["W_Vt"], dtype=np.float64)
                self.W = (u_ * s_) @ vt_
                self._factors = (u_, s_, vt_)
            else:
                raise KeyError(
                    f"{path} has neither 'W' nor 'W_U/W_S/W_Vt' (has {npz.files})"
                )
            self.mu_in = np.asarray(npz["mu_in"], dtype=np.float64)
            self.mu_y = np.asarray(npz["mu_y"], dtype=np.float64)
            self.v3_mu = np.asarray(npz["v3_mu"], dtype=np.float64)
            self.v3_sd = np.asarray(npz["v3_sd"], dtype=np.float64)
            self.v3_dead = np.asarray(npz["v3_dead"], dtype=bool)
            self.bins_mu = np.asarray(npz["bins_mu"], dtype=np.float64)
            self.bins_sd = np.asarray(npz["bins_sd"], dtype=np.float64)
            self.bins_dead = np.asarray(npz["bins_dead"], dtype=bool)
            self.sites = [int(x) for x in npz["sites"]]
            self.norm_ref = np.asarray(npz["norm_ref"], dtype=np.float64)
            self.meta = json.loads(str(npz["meta"]))
            # ── A map may SHIP MORE THAN ONE RULER and wear only one of them. v1a carries the WIDE map's
            # `norm_ref` by design — all three spot-check arms norm-matched to
            # one reference so that at equal alpha they wore identical per-site
            # magnitudes and only DIRECTION differed — and keeps its OWN ruler
            # beside it as `norm_ref_v1a_own` (mean 1.213 against the wide
            # map's 2.682: a factor of 2.2, the size of the ruler mismatch in
            # docs/FINDINGS.md §2). Which one is in force decides what
            # alpha MEANS, so both are read here and both are served on /info.
            # An operator who cannot tell which ruler is live cannot read a
            # dose band, and a dose gauge that is wrong is worse than no gauge
            # (pleroma.dose.band).
            #
            # These are LABELS: `self.norm_ref` is only ever
            # `npz["norm_ref"]`.
            self.norm_ref_alternatives: dict[str, list[float]] = {
                str(key): [float(x) for x in
                           np.asarray(npz[key], dtype=np.float64).ravel()]
                for key in npz.files
                if key.startswith("norm_ref_") and key != "norm_ref_which"
            }
            self.norm_ref_which: str | None = (
                str(npz["norm_ref_which"]) if "norm_ref_which" in npz.files
                else None
            )
        # These two arrays are the ONLY things the serving path reads out of the
        # discriminants file. `FULL_W`, `HAND_*`, `labels` and every centroid are
        # inert here — they matter only to the basin diagnostics
        # (pleroma.map.build.basin), which are colour, never a gate.
        # ★ And mean/scale are not load-bearing: composed with the corpus stage
        # they telescope away (measured — replacing the artifact with mean 0 /
        # scale 1 moves the z-matrix by row cosine 0.999999990821). What IS
        # load-bearing is `FULL_names` (the feature-name contract, checked by
        # Gate 0) and the sha pin just below, which marries this file to the
        # map's banked v3_mu/v3_sd. Do not "simplify" either away, and do not
        # drop --standardize corpus to compensate for the scale: without the
        # corpus stage the foreign FULL_scale leaves the inputs near-collinear
        # and the fit returns a null (the refusal in
        # pleroma.map.build.shelf.compute_shelf states the numbers).
        with np.load(discriminants, allow_pickle=True) as npz:
            self.full_mean = np.asarray(npz["FULL_mean"], dtype=np.float64)
            full_scale = np.asarray(npz["FULL_scale"], dtype=np.float64)
        sha = hashlib.sha256(Path(discriminants).read_bytes()).hexdigest()
        if sha != self.meta["discriminants_sha256"]:
            raise ValueError(
                "discriminants sha mismatch: the map was fitted in a different "
                "frozen space than the one handed to the server"
            )
        self.degenerate = full_scale < 1e-12
        self.full_scale = np.where(self.degenerate, 1.0, full_scale)
        self.n_sites = len(self.sites)
        self.hidden = self.mu_y.size // self.n_sites
        # The CODE basis: W = U S Vᵀ (already rank-truncated). A candidate's
        # 8 coordinates in span(V) ARE the code — the entire content of what
        # the map says, before expansion to the 12,288-d lever.
        self.rank = int(self.meta["rank"])
        if self._factors is not None:
            self.Vt = self._factors[2][: self.rank]
        else:
            _, _, vt = np.linalg.svd(self.W, full_matrices=False)
            self.Vt = vt[: self.rank]
        # A LABEL, never a transform: re-orienting a served map would re-aim
        # every code banked against it. A map exported without canonical
        # orientation keeps its export run's SVD signs, so its codes are tied
        # to that one export run (pleroma.map.svd).
        self.canonical_orientation: bool = is_canonical(self.Vt)
        # Fingerprint v2 (weights included, legacy v1 id embedded); the ONE id
        # stamped on wears, bands and codes made with this map.
        self.fingerprint: str = fingerprint_v2(
            self.sites, self.norm_ref, self.mu_y, self.meta, self.weights_digest)

    def z_corpus_of(self, signature: np.ndarray) -> np.ndarray:
        """Raw v3 signature -> corpus-standardized z (the training pipeline's
        stage 1+2 — also the space the basin check's spread readings live in)."""
        zf = (signature - self.full_mean) / self.full_scale
        zf[self.degenerate] = 0.0
        zc = (zf - self.v3_mu) / np.where(self.v3_dead, 1.0, self.v3_sd)
        zc[self.v3_dead] = 0.0
        return zc

    def input_of(self, signature: np.ndarray, bins_row: np.ndarray) -> np.ndarray:
        """Raw v3 signature + raw bins row -> the map's INPUT row `x` (corpus-z
        signature concatenated with the standardized bins row).

        ★ A separate step so the fan CONTRAST can be built from exactly the
        rows the absolute lever is built from. `lever_of` is pinned bit-for-bit
        against the unsplit arithmetic by
        test_loom_lever_kind.py::test_lever_of_is_bit_identical_after_the_split.
        """
        zc = self.z_corpus_of(signature)
        bs = (bins_row - self.bins_mu) / np.where(self.bins_dead, 1.0, self.bins_sd)
        bs[self.bins_dead] = 0.0
        return np.concatenate([zc, bs])

    def lever_of(
        self, signature: np.ndarray, bins_row: np.ndarray
    ) -> tuple[np.ndarray, list[float], list[float]]:
        """Raw v3 signature + raw bins row -> (norm-matched lever [S, hidden],
        the map's RAW per-site output norms before matching — the 'loudness'
        of the code as the map spoke it, advisory only — and the rank-r code).

        This is the `absolute` lever: W applied to the candidate's OWN input
        row. See `contrast_lever_of` for the fan-contrasted one."""
        return self.lever_of_input(self.input_of(signature, bins_row))

    def lever_of_input(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, list[float], list[float]]:
        """`lever_of` from an already-built input row (see `input_of`)."""
        flat = (x - self.mu_in) @ self.W + self.mu_y
        code = MapCode([round(float(c), 4) for c in self.Vt @ (flat - self.mu_y)],
                       self.fingerprint)
        lever = flat.reshape(self.n_sites, self.hidden)
        out, raw_norms = apply_ruler(lever, self.norm_ref, sites=self.sites,
                                     what="map output")
        return out, raw_norms, code

    def lever_from_code(self, code: Any, *, differential: bool,
                        map_fingerprint: str | None = None,
                        ) -> tuple[np.ndarray, list[float]]:
        """An explicit rank-r code -> (norm-matched lever [S, hidden], the RAW
        per-site norms before matching).

        ★ WHY. `code = Vt @ (flat − mu_y)` and Vt's rows are
        orthonormal, so `flat = code @ Vt + mu_y` reconstructs a candidate's
        lever EXACTLY — which makes a stored fan code from ANY prior draw
        wearable in any session (cross-conversation transplant), and makes a
        client-side `code_sel − mean(code_rest)` wearable as the zero-training
        contrastive probe. For a `differential` code, mu_y is NOT added back:
        it cancels in the subtraction, so the reconstructed lever is exactly
        the difference of the two absolute levers.

        The ruler applies unchanged (docs/FINDINGS.md §2): per-site renormalisation
        to the map's own norm_ref AFTER any centering, so alpha means the same
        thing here as for a /wear on a fan candidate. The RAW norms are
        returned for the record — for a differential code they are the
        map-space contrastive loudness, the one covariate the reachability
        triage found tracking readers.
        """
        c = np.asarray(code, dtype=np.float64).ravel()
        if c.size != self.rank:
            raise ValueError(
                f"code has {c.size} coordinates but the map's rank is "
                f"{self.rank} — is this code from a different map?")
        # ★ The size check above passed, so the ranks agree. A code carries
        # the map it came from (a `MapCode`, or the fingerprint banked beside a
        # JSON code), because a same-rank code from another map reconstructs
        # an unrelated direction (measured cos .068 to the intended lever).
        made_with = (code.map_fingerprint if isinstance(code, MapCode)
                     else map_fingerprint)
        if made_with is not None and same_map(made_with, self.fingerprint) == "mismatch":
            raise ValueError(
                f"code was made with map {made_with} but this map is "
                f"{self.fingerprint} — refusing to wear a foreign code")
        if not np.isfinite(c).all():
            raise ValueError("code has non-finite coordinates")
        flat = c @ self.Vt
        if not differential:
            flat = flat + self.mu_y
        lever = flat.reshape(self.n_sites, self.hidden)
        # a ~zero row of a DIFFERENTIAL code means the contrastive content is
        # a no-op at that site: refused by the ruler, never norm-matched up.
        return apply_ruler(lever, self.norm_ref, sites=self.sites, what="code")

    def contrast_lever_of(
        self, xs: Sequence[np.ndarray], pos: int
    ) -> tuple[np.ndarray, list[float], list[float]]:
        """The FAN-CONTRASTED lever for member `pos` of a fan whose valid input
        rows are `xs` (built by `input_of`): W applied to
        `x_pos − mean of the OTHER k−1 rows`.

        ★ WHY. v1a was FIT on this object — one-vs-rest member-minus-fan-mean
        displacement (`pleroma.map.build.join.fan_center`) — while `lever_of`
        applies W to the member's own row. Measured on six live 70B fans: cos(absolute,
        contrast) median 0.55, and the fan-common offset W·x̄ is median 0.87 of
        the absolute lever's norm. This is the lever the map was trained to
        emit.

        Conventions mirror `lever_from_code(..., differential=True)` EXACTLY,
        so a fan wear under `lever_kind: "contrast"` is the same object /probe
        and `/wear_code {code_kind: "differential"}` already wear:
          * mu_in cancels in the subtraction and mu_y is NOT added back;
          * the per-site ruler is applied AFTER the contrast,
            so alpha means one absolute per-site norm across both kinds;
          * a ~zero row (≤ 1e-9) refuses rather than norm-matching noise up to
            full loudness;
          * the code is `Vt @ flat`, rounded to 4 places like `lever_of`'s.
        Pinned equal to `lever_from_code(contrastive_code(codes, pos),
        differential=True)` by test_loom_lever_kind.py.

        Returns (norm-matched lever [S, hidden], RAW per-site norms, code).
        """
        rows = [np.asarray(x, dtype=np.float64).ravel() for x in xs]
        k = len(rows)
        if k < 2:
            raise ValueError(
                f"a fan contrast needs at least 2 valid members, got {k}")
        if not 0 <= int(pos) < k:
            raise ValueError(f"pos {pos} outside a fan of {k} valid members")
        widths = {r.size for r in rows}
        if len(widths) != 1:
            raise ValueError(
                f"fan input rows have mixed widths {sorted(widths)} — "
                "were they built by different maps?")
        pos = int(pos)
        rest = np.stack([rows[j] for j in range(k) if j != pos]).mean(axis=0)
        flat = (rows[pos] - rest) @ self.W
        code = [round(float(c), 4) for c in self.Vt @ flat]
        lever = flat.reshape(self.n_sites, self.hidden)
        # a ~zero row: this member does not differ from the rest of the fan
        out, raw_norms = apply_ruler(lever, self.norm_ref, sites=self.sites,
                                     what="fan contrast")
        return out, raw_norms, code


