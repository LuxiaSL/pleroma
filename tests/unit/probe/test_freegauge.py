"""Tests for the free gauge's calibration math (the gauge core loom_probe uses)."""
from __future__ import annotations

import math
import pytest

from pleroma.probe.freegauge import (
    auc,
    cell_delta,
    cosine_dist,
    euclid,
    gauge_nr,
)


def test_gauge_nr_orders_by_distance() -> None:
    futures = [[0.0], [1.0], [2.0], [3.0]]
    assert gauge_nr([0.1], futures, 0, euclid) == 0.0
    assert gauge_nr([0.1], futures, 3, euclid) == 1.0
    assert gauge_nr([1.9], futures, 2, euclid) == 0.0


def test_gauge_nr_tie_breaks_by_index() -> None:
    futures = [[1.0], [1.0], [5.0]]
    # vec equidistant from futures 0 and 1 — index breaks the tie
    assert gauge_nr([1.0], futures, 0, euclid) == 0.0
    assert gauge_nr([1.0], futures, 1, euclid) == 0.5


def test_cell_delta_positive_when_steered_closer() -> None:
    assert cell_delta([0.8, 0.6], [0.2, 0.4]) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        cell_delta([], [0.1])


def test_auc_separates() -> None:
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_cosine_dist_basics() -> None:
    assert cosine_dist([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_dist([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)
    assert math.isclose(cosine_dist([1.0, 0.0], [-1.0, 0.0]), 2.0)
    with pytest.raises(ValueError):
        cosine_dist([0.0, 0.0], [1.0, 0.0])
