"""kvrot — accurate KV-cache rotation under prefix eviction.

CPU-importable core (rotation math, cache surgery, eviction, metrics). The real-model
harness lives in :mod:`kvrot.harness` and lazy-imports transformers, so importing this
package never requires the ``hf`` extra.
"""

from __future__ import annotations

from . import eviction, metrics, rope, snapshot
from .config import ArchSpec, EvictionSpec, RoPESpec
from .metrics import DriftReport
from .snapshot import KVSnapshot, evict, reindex

__all__ = [
    "rope",
    "snapshot",
    "eviction",
    "metrics",
    "ArchSpec",
    "RoPESpec",
    "EvictionSpec",
    "DriftReport",
    "KVSnapshot",
    "evict",
    "reindex",
]

__version__ = "0.1.0"
