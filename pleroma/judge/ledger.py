"""The CROSS-RUN spend ledger: a hard cap over many paid batches.

`pleroma.judge.pricing.SpendMeter` guards ONE invocation. This guards a
campaign (the 70B ladder + spot check ran under a hard $20 cap): every paid
batch is PROJECTED before it runs — `SpendLedger.check` refuses if spent +
projection would cross the cap — and RECORDED after it runs from the tool's own
spend receipt (a `SpendMeter.snapshot`, i.e. ACTUAL `usage` priced at
`pleroma.judge.pricing.PRICES`).

The on-disk JSON schema is stable, so every existing ledger loads as-is.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from pleroma.judge.pricing import PRICES, PRICES_AS_OF

DEFAULT_CAP_USD = 20.00


class LedgerError(ValueError):
    """A ledger or receipt file that cannot be read as what it claims to be."""


class LedgerEntry(BaseModel):
    what: str
    at: str
    spent_usd: float = Field(ge=0)
    calls: int = Field(ge=0)
    receipt: str
    tokens: dict[str, int] = Field(default_factory=dict)
    projected_usd: float | None = None


def _default_prices_note() -> str:
    pin, pout = PRICES["claude-sonnet-5"]
    return (f"pleroma.judge.pricing.PRICES — {PRICES_AS_OF}: claude-sonnet-5 "
            f"${pin:g} in / ${pout:g} out per MTok")


class SpendLedger(BaseModel):
    cap_usd: float = Field(default=DEFAULT_CAP_USD, gt=0)
    judge: str = "claude-sonnet-5 (API-default effort, adaptive thinking)"
    prices: str = Field(default_factory=_default_prices_note)
    entries: list[LedgerEntry] = Field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(e.spent_usd for e in self.entries), 6)

    @property
    def remaining(self) -> float:
        return round(self.cap_usd - self.total, 6)

    def check(self, projected_usd: float) -> bool:
        """True iff spent + projection stays within the cap."""
        if projected_usd < 0:
            raise LedgerError(f"projected spend must be >= 0, got {projected_usd}")
        return self.total + float(projected_usd) <= self.cap_usd

    def add_receipt(self, receipt_path: Path, what: str,
                    projected_usd: float | None = None) -> LedgerEntry:
        """Append one batch from its spend receipt (`SpendMeter.snapshot` JSON)."""
        try:
            rec: dict[str, Any] = json.loads(Path(receipt_path).read_text())
            entry = LedgerEntry(
                what=what, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                spent_usd=float(rec["spent_usd"]), calls=int(rec["calls"]),
                receipt=str(receipt_path),
                tokens={k: int(v) for k, v in (rec.get("tokens") or {}).items()},
                projected_usd=projected_usd)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise LedgerError(
                f"{receipt_path}: not a readable spend receipt (need spent_usd and "
                f"calls, as SpendMeter.snapshot writes them): {exc}") from exc
        self.entries.append(entry)
        return entry

    @classmethod
    def load(cls, path: Path) -> "SpendLedger":
        """The ledger at ``path``, or a fresh one if it does not exist yet."""
        if not path.exists():
            return cls()
        try:
            blob = json.loads(path.read_text())
            blob.pop("total_usd", None)
            blob.pop("remaining_usd", None)
            return cls.model_validate(blob)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LedgerError(f"{path}: not a spend ledger: {exc}") from exc

    def save(self, path: Path) -> None:
        blob = self.model_dump()
        blob["total_usd"] = self.total
        blob["remaining_usd"] = self.remaining
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=1) + "\n")
