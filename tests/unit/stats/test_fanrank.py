"""pleroma.stats.fanrank — normalized ranks and the within-fan permutation test."""

from __future__ import annotations

import numpy as np
import pytest

from pleroma.stats.fanrank import fan_rank_test, normalized_ranks, nr_from_ranking


def test_normalized_ranks_order_and_ties() -> None:
    np.testing.assert_allclose(normalized_ranks([3.0, 1.0, 2.0]), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(normalized_ranks([3.0, 1.0, 2.0], higher_is_closer=False),
                               [1.0, 0.0, 0.5])
    np.testing.assert_allclose(normalized_ranks([1.0, 1.0, 0.0]), [0.25, 0.25, 1.0])
    with pytest.raises(ValueError):
        normalized_ranks([1.0])


def test_nr_from_ranking() -> None:
    np.testing.assert_allclose(nr_from_ranking([2, 0, 1], 3), [0.5, 1.0, 0.0])
    with pytest.raises(ValueError, match="permutation"):
        nr_from_ranking([0, 0, 1], 3)


def _perfect(k: int) -> np.ndarray:
    """Valid rank rows: wear j ranks future j first, the rest in index order."""
    rows = []
    for j in range(k):
        order = [j] + [i for i in range(k) if i != j]
        rows.append(nr_from_ranking(order, k))
    return np.stack(rows)


def test_perfect_steering_is_maximal_and_significant() -> None:
    res = fan_rank_test({f"f{i}": _perfect(8) for i in range(6)}, n_perms=2000)
    assert res.delta_nr == pytest.approx(0.5)
    assert res.p_one_sided < 0.001
    assert abs(res.null_mean) < 0.05


def test_pure_noise_is_null_and_calibrated() -> None:
    rng = np.random.default_rng(0)
    ps = []
    for rep in range(40):
        fans = {}
        for f in range(8):
            fans[f"f{f}"] = np.stack([normalized_ranks(rng.standard_normal(8)) for _ in range(8)])
        ps.append(fan_rank_test(fans, n_perms=400, seed=rep).p_one_sided)
    # roughly uniform: not piled up near 0
    assert np.mean(np.array(ps) < 0.05) < 0.2


def test_a_readout_that_favours_one_future_for_every_wear_is_not_steering() -> None:
    """Every wear pulls toward future 0 (a 'default' future): no label effect."""
    m = np.tile(normalized_ranks([9, 1, 2, 3, 4, 5, 6, 7]), (8, 1))
    res = fan_rank_test({f"f{i}": m for i in range(6)}, n_perms=2000)
    assert abs(res.delta_nr) < 1e-9 and res.p_one_sided > 0.3


def test_paired_form_subtracts_the_base() -> None:
    m = _perfect(4)
    base = np.full(4, 0.5)          # a base readout at chance for every future
    res = fan_rank_test({"a": m, "b": m}, base_by_fan={"a": base, "b": base}, n_perms=500)
    assert res.paired and res.delta_nr == pytest.approx(0.5)


def test_nan_cells_are_skipped_and_shapes_checked() -> None:
    m = _perfect(4)
    m[1, 1] = np.nan
    assert fan_rank_test({"a": m, "b": _perfect(4)}, n_perms=200).delta_nr == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fan_rank_test({"a": np.zeros((5, 4))})


def test_paired_fan_contrast_sign_flip() -> None:
    from pleroma.stats.fanrank import paired_fan_contrast
    a = {f"f{i}": 0.3 for i in range(12)}
    b = {f"f{i}": 0.1 for i in range(12)}
    res = paired_fan_contrast(a, b, n_perms=2000)
    assert res["contrast"] == pytest.approx(0.2) and res["p_one_sided"] < 0.001
    assert res["positive_fans"] == 12
    same = paired_fan_contrast(a, a, n_perms=500)
    assert same["contrast"] == 0 and same["p_one_sided"] > 0.5


def test_length_matched_test_sees_steering_that_length_cannot_explain() -> None:
    from pleroma.stats.fanrank import length_matched_test
    lengths = [100, 102, 98, 101, 300, 305, 298, 302]      # two length clusters
    ranks = {f"f{n}": {j: [j] + [i for i in range(8) if i != j] for j in range(8)}
             for n in range(6)}
    res = length_matched_test(ranks, {f: lengths for f in ranks}, tol=0.1, n_perms=500)
    assert res["delta_nr"] == pytest.approx(0.5) and res["p_one_sided"] < 0.01


def _length_ranker(lengths: list[int], reply_len: float) -> list[int]:
    return sorted(range(len(lengths)), key=lambda i: (abs(lengths[i] - reply_len), i))


def test_length_matched_test_and_a_length_ranker() -> None:
    """A length-only readout is null under matching when reply length tracks the
    target only loosely (noise wider than tol) — and is NOT when replies hit the
    target's length exactly. So the matched test needs the length ranker run
    through it as its own control; the analysis does that."""
    from pleroma.stats.fanrank import length_matched_test
    lengths = [100, 102, 98, 101, 300, 305, 298, 302]
    exact = {f"f{n}": {j: _length_ranker(lengths, lengths[j]) for j in range(8)}
             for n in range(6)}
    res = length_matched_test(exact, {f: lengths for f in exact}, tol=0.1, n_perms=500)
    assert res["p_one_sided"] < 0.01               # exact length tracking leaks through
    rng = np.random.default_rng(1)
    loose = {f"f{n}": {j: _length_ranker(lengths, lengths[j] * (1 + rng.normal(0, 0.3)))
                       for j in range(8)} for n in range(12)}
    res = length_matched_test(loose, {f: lengths for f in loose}, tol=0.1, n_perms=500)
    assert res["p_one_sided"] > 0.05
