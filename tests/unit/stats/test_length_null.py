"""The length-only null ranker, ported from `sitenorm_analyze` — pinned to the
banked number it produced, not just to its own definition."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pleroma.stats import length_null as L

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "length_null_sitenorm_8b.json"


def test_orders_by_absolute_gap_to_the_reply() -> None:
    # gaps to 100: |90|=10, |130|=30, |95|=5, |200|=100 -> order 2,0,1,3
    assert L.length_only_rank([90, 130, 95, 200], 100, 2) == 0.0
    assert L.length_only_rank([90, 130, 95, 200], 100, 1) == pytest.approx(2 / 3)
    assert L.length_only_rank([90, 130, 95, 200], 100, 3) == 1.0


def test_ties_break_on_the_lower_fan_index() -> None:
    assert L.length_only_rank([110, 90], 100, 0) == 0.0
    assert L.length_only_rank([110, 90], 100, 1) == 1.0


def test_refuses_a_ranking_of_one() -> None:
    with pytest.raises(ValueError):
        L.length_only_rank([5], 5, 0)


def _by_conv_from_fixture() -> dict[str, dict[str, list[float]]]:
    data = json.loads(FIXTURE.read_text())
    by_conv: dict[str, dict[str, list[float]]] = {}
    for row in data["rows"]:
        pid = row["prompt_id"]
        lo = L.length_only_rank(data["cand_chars"][pid], row["reply_chars"],
                                row["prescribed_index"])
        for key in row["keys"]:
            by_conv.setdefault(pid, {}).setdefault(key, []).append(lo)
    return by_conv


def test_reproduces_the_banked_8b_length_only_dnr() -> None:
    """★ The number the rule is stated in. The banked 8B sitenorm verdict at
    band 0.10, instrument `asfrozen`,
    LENGTH_ONLY_null_ranker: dnr_raw +0.2571 (12/13 positive),
    dnr_sitenorm +0.2053 (9/13). The manner judge on the same replies: +0.0848."""
    by_conv = _by_conv_from_fixture()
    raw = L.length_only_dnr(by_conv, "raw")
    sn = L.length_only_dnr(by_conv, "sitenorm")
    assert raw["n_conversations"] == 13
    assert round(raw["observed"], 4) == 0.2571
    assert raw["n_positive"] == 12
    assert round(sn["observed"], 4) == 0.2053
    assert sn["n_positive"] == 9


def test_length_unit_is_characters_not_words() -> None:
    assert json.loads(FIXTURE.read_text())["length_unit"].startswith("characters")
    assert L.chars("ab cd") == 5 and L.words("ab cd") == 2
