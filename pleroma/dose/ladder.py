"""Dose-ladder primitives: probes, doses, per-reply seeds, opaque pair ids.

Pure, deterministic, torch-free — shared by every dose-ladder generator, so
the probes, seeds and pair ids a ladder uses do not depend on which lane
generated its replies.

── alpha = 0 IS A CATCH PAIR ──────────────────────────────────────────────────
At alpha 0 the "steered" arm still runs the WHOLE injection path with a
zero-scaled vector, so a catch pair is two draws from one distribution and there
is no correct answer. It measures the judge: how often a naive reader, told one
of these was modified, says so anyway. Hence :func:`parse_dose_alphas` ALLOWS 0.

── the seeds differ WITHIN a pair, on purpose ─────────────────────────────────
Keying the RNG on (source, rep) but not the arm would make base and steered
share a verbatim prefix and diverge mid-sentence — a cue a judge spots
instantly. So :func:`make_reply_seed` keys on ``(probe_id, alpha, rep, ARM)``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pleroma.levers.npz import group_label

F32 = NDArray[np.float32]

#: The two arms of every pair. ``base`` never attaches a hook.
ARMS: tuple[str, str] = ("base", "steered")

#: The HF ladder's historical default rung set (kept for reference/tests).
DEFAULT_ALPHAS: str = "0,0.125,0.25,0.5,1.0"


@dataclass(frozen=True)
class Probe:
    """One single-turn user message."""

    probe_id: str
    probe_class: str
    text: str


def load_probes(path: Path) -> list[Probe]:
    """Read a probes JSON, tolerating the ``prompts``/``prompt`` spelling too.

    Probe files come in two spellings: ``prompts``/``prompt`` and
    ``probes``/``text``. Both are accepted so a probe list in either spelling
    reads without a silent empty result.
    """
    blob = json.loads(Path(path).expanduser().read_text())
    rows = blob.get("probes") if isinstance(blob, dict) else blob
    if rows is None and isinstance(blob, dict):
        rows = blob.get("prompts")
    if not rows:
        raise ValueError(f"{path}: no 'probes' (or 'prompts') entries")
    probes: list[Probe] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("id") or row.get("probe_id") or "").strip()
        text = str(row.get("text") or row.get("prompt") or "").strip()
        if not pid or not text:
            raise ValueError(f"{path}: every probe needs an 'id' and a 'text' (got {row})")
        if pid in seen:
            raise ValueError(f"{path}: duplicate probe id {pid!r}")
        seen.add(pid)
        probes.append(
            Probe(probe_id=pid, probe_class=str(row.get("class") or "unclassified"), text=text)
        )
    return probes


def parse_dose_alphas(spec: str) -> list[float]:
    """Parse ``--alphas``, and unlike an eval's arm list ALLOW zero.

    Alpha 0 is a first-class rung — the catch pair — and it is a *steered* arm
    running the whole machinery at no strength. Refusing it would remove the
    only measurement of the judges themselves. Returned sorted; negatives and
    duplicates refused.
    """
    raw = [a.strip() for a in str(spec).split(",") if a.strip()]
    if not raw:
        raise ValueError("--alphas is empty")
    try:
        alphas = [float(a) for a in raw]
    except ValueError as exc:
        raise ValueError(f"--alphas must be numbers, got {spec!r}") from exc
    if any(a < 0 for a in alphas):
        raise ValueError(f"--alphas must be >= 0, got {alphas}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"duplicate dose in --alphas: {alphas}")
    return sorted(alphas)


def make_reply_seed(probe_id: str, alpha: float, rep: int, arm: str, salt: int) -> int:
    """The RNG seed for ONE reply — keyed on the arm as well as the rep.

    See the module docstring: keying the arm OUT would make the two replies of a
    pair share a verbatim prefix.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known arms are {list(ARMS)}")
    if rep < 0:
        raise ValueError(f"rep must be >= 0, got {rep}")
    if salt < 0:
        raise ValueError(f"--seed-offset must be >= 0, got {salt}")
    raw = f"expB15dose_{salt}_{probe_id}_{alpha:g}_{rep}_{arm}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def make_group_pair_id(group: str, probe_id: str, alpha: float, rep: int,
                       salt: int) -> str:
    """Opaque pair id that ALSO keys on the lever group.

    ★ WHY IT KEYS ON THE GROUP. A pair id hashed from (probe_id, alpha, rep,
    salt) alone gives every group of a multi-lever ladder byte-identical ids.
    The dose-band judge stage passes every group's pair file to ONE pair-judge
    call (``pleroma.judge.pair``), which dedupes on pair_id keeping the first
    occurrence: a 1,008-pair, three-lever ladder keyed that way is silently
    judged as 336 pairs from ONE lever.

    Still opaque, so the blind contract holds — a packet leak test greps for
    readable dose strings and a hash contains none.
    """
    digest = hashlib.sha256(
        f"expB15pairid_{salt}_{group}_{probe_id}_{alpha:g}_{rep}".encode())
    return f"pair_{digest.hexdigest()[:12]}"


def dose_vector(lever: NDArray[np.floating], alpha: float, sign: float) -> F32:
    """``alpha * sign * lever`` — the exact vector that will be written.

    alpha multiplies the RAW lever, so the sites keep their relative magnitudes
    and alpha 1 writes the whole contrast. ``sign`` picks which basin (+1 =
    cluster 0, the lever as banked). At alpha 0 this is exactly zero — the catch
    pair's vector.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if sign not in (1.0, -1.0):
        raise ValueError(f"--lever-sign must be +1.0 or -1.0, got {sign}")
    out = (float(alpha) * float(sign) * np.asarray(lever, dtype=np.float32)).astype(np.float32)
    if out.ndim != 2:
        raise ValueError(f"lever must be [n_sites, hidden_dim], got {out.shape}")
    return out


def find_group(bank: Any, label: str) -> int:
    """Index of ``prompt|wave`` in a levers npz, or a listing of what IS there.

    ``bank`` needs ``group_prompt_ids``, ``group_waves`` and ``n_groups``
    (a :class:`pleroma.levers.npz.LeverNpz`).
    """
    labels = [
        group_label(bank.group_prompt_ids[i], bank.group_waves[i])
        for i in range(bank.n_groups)
    ]
    if label not in labels:
        raise ValueError(
            f"--lever-group {label!r} is not in the levers npz "
            f"(it has {len(labels)}: {sorted(labels)[:8]}{'...' if len(labels) > 8 else ''})"
        )
    return labels.index(label)
