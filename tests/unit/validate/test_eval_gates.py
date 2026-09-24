"""eval gates: J-01 (length-null), J-05/J-06 (positive-control)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pleroma.validate import evals

from .conftest import write_json


def ln_input(judged_gap: float, length_gap: float, n_conv: int = 10,
             noise: float = 0.02) -> evals.LengthNullInput:
    """Per conversation: base nr .5, steered nr .5 − gap (+ noise); Δnr = gap."""
    rng = np.random.default_rng(0)

    def arm(gap: float) -> dict[str, dict[str, list[float]]]:
        return {f"c{i}": {"base": [0.5, 0.5],
                          "steered": [0.5 - gap + float(rng.normal(0, noise)),
                                      0.5 - gap + float(rng.normal(0, noise))]}
                for i in range(n_conv)}

    return evals.LengthNullInput(arm="steered", base="base", judged=arm(judged_gap),
                                 length_only=arm(length_gap))


def test_length_null_pass_when_judge_beats_length() -> None:
    r = evals.gate_length_null(ln_input(0.25, 0.05))
    assert r.verdict == "PASS" and "beats the length-only ranker" in r.reason
    assert r.evidence["length_unit"] == "characters"


def test_length_null_fail_when_length_explains_it() -> None:
    r = evals.gate_length_null(ln_input(0.085, 0.257))
    assert r.verdict == "FAIL"
    assert "the length-only ranker explains the effect" in r.reason


def test_length_null_inconclusive_when_ci_spans_zero() -> None:
    r = evals.gate_length_null(ln_input(0.11, 0.10, noise=0.2))
    assert r.verdict == "INCONCLUSIVE" and "length is not ruled out" in r.reason


def test_length_null_inconclusive_at_small_n() -> None:
    r = evals.gate_length_null(ln_input(0.3, 0.0, n_conv=3))
    assert r.verdict == "INCONCLUSIVE" and "unpowered" in r.reason


def test_length_null_inconclusive_without_an_effect() -> None:
    r = evals.gate_length_null(ln_input(-0.1, -0.2))
    assert r.verdict == "INCONCLUSIVE" and "no effect for the length null" in r.reason


def test_positive_control_pass() -> None:
    d = evals.PositiveControlInput(positive=[1.0] * 18 + [0.0] * 2,
                                   floor=[1.0] * 9 + [0.0] * 11)
    r = evals.gate_positive_control(d)
    assert r.verdict == "PASS" and "separates the known-positive control" in r.reason


def test_positive_control_blind_judge_is_inconclusive_never_fail() -> None:
    d = evals.PositiveControlInput(positive=[1.0] * 10 + [0.0] * 10,
                                   floor=[1.0] * 10 + [0.0] * 10)
    r = evals.gate_positive_control(d)
    assert r.verdict == "INCONCLUSIVE"
    assert r.reason.startswith("judge cannot see effects of this size")
    assert "not 'no effect'" in r.reason


def test_positive_control_needs_its_own_floor() -> None:
    r = evals.gate_positive_control(evals.PositiveControlInput(positive=[1.0] * 20))
    assert r.verdict == "INCONCLUSIVE" and "never a fixed 0.5" in r.reason


def test_run_eval_names_malformed_inputs(tmp_path: Path) -> None:
    ln = write_json(tmp_path / "ln.json", {"arm": "steered", "judged": {}})
    pc = tmp_path / "missing.json"
    r1, r2 = evals.run_eval(ln, pc)
    assert r1.verdict == "INCONCLUSIVE" and "length-null input malformed" in r1.reason
    assert r2.verdict == "INCONCLUSIVE" and "not found" in r2.reason


def test_permutation_p_is_one_sided() -> None:
    assert evals.permutation_p_greater([1.0] * 10, [0.0] * 10, n_perms=2000) < 0.01
    assert evals.permutation_p_greater([0.0] * 10, [1.0] * 10, n_perms=2000) > 0.99
