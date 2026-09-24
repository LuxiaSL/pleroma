"""pleroma.dose.ladder — the dose-ladder primitives the vLLM lanes use.

Three properties
decide whether the blind that follows is a blind at all: alpha 0 is ACCEPTED
(the catch rung); pair ids are OPAQUE (and key on the lever group); the two
replies of a pair get DIFFERENT seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pleroma.dose.ladder import (
    ARMS,
    DEFAULT_ALPHAS,
    dose_vector,
    find_group,
    load_probes,
    make_group_pair_id,
    make_reply_seed,
    parse_dose_alphas,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBES_FILE = REPO_ROOT / "tests" / "fixtures" / "probes_synthetic.json"


class _Bank:
    """The slice of a LeverNpz ``find_group`` touches."""

    def __init__(self, labels: list[tuple[str, str]]) -> None:
        self.group_prompt_ids = [p for p, _ in labels]
        self.group_waves = [w for _, w in labels]
        self.n_groups = len(labels)


def test_alpha_zero_is_allowed_here_and_is_the_catch_rung() -> None:
    assert parse_dose_alphas(DEFAULT_ALPHAS) == [0.0, 0.125, 0.25, 0.5, 1.0]
    assert parse_dose_alphas("1.0,0") == [0.0, 1.0]
    with pytest.raises(ValueError, match=">= 0"):
        parse_dose_alphas("-0.5")
    with pytest.raises(ValueError, match="duplicate"):
        parse_dose_alphas("0,0")
    with pytest.raises(ValueError, match="empty"):
        parse_dose_alphas("  ,  ")
    with pytest.raises(ValueError, match="numbers"):
        parse_dose_alphas("0,x")


def test_the_catch_vector_is_exactly_zero_and_the_sign_flips() -> None:
    rng = np.random.default_rng(0)
    lever = rng.standard_normal((4, 16)).astype(np.float32)
    assert np.array_equal(dose_vector(lever, 0.0, 1.0), np.zeros_like(lever))
    assert np.allclose(dose_vector(lever, 0.5, 1.0), 0.5 * lever)
    assert np.allclose(dose_vector(lever, 0.5, -1.0), -0.5 * lever)
    assert dose_vector(lever, 0.5, 1.0).dtype == np.float32
    with pytest.raises(ValueError, match="lever-sign"):
        dose_vector(lever, 1.0, 0.5)
    with pytest.raises(ValueError, match=">= 0"):
        dose_vector(lever, -1.0, 1.0)
    with pytest.raises(ValueError, match="n_sites, hidden_dim"):
        dose_vector(lever[0], 1.0, 1.0)


def test_the_two_replies_of_a_pair_get_different_seeds() -> None:
    """The deliberate inversion: a shared seed is a verbatim-prefix cue leak."""
    seeds = {arm: make_reply_seed("topic_interest", 0.25, 1, arm, 3000) for arm in ARMS}
    assert seeds["base"] != seeds["steered"]
    assert all(0 <= s < 2**32 for s in seeds.values())
    assert seeds["base"] != make_reply_seed("topic_interest", 0.25, 2, "base", 3000)
    assert seeds["base"] != make_reply_seed("topic_interest", 0.5, 1, "base", 3000)
    assert seeds["base"] != make_reply_seed("dinner_plan", 0.25, 1, "base", 3000)
    assert seeds["base"] != make_reply_seed("topic_interest", 0.25, 1, "base", 3001)
    with pytest.raises(ValueError, match="unknown arm"):
        make_reply_seed("p", 0.25, 0, "fitted", 3000)
    with pytest.raises(ValueError, match="rep"):
        make_reply_seed("p", 0.25, -1, "base", 3000)
    with pytest.raises(ValueError, match="seed-offset"):
        make_reply_seed("p", 0.25, 0, "base", -1)


def test_reply_seed_golden_value() -> None:
    """Pinned: the seed recipe is part of every banked ladder's provenance."""
    import hashlib
    want = int(hashlib.sha256(b"expB15dose_3000_p_0.25_0_base").hexdigest()[:8], 16)
    assert make_reply_seed("p", 0.25, 0, "base", 3000) == want


def test_group_pair_ids_are_opaque_and_key_on_the_group() -> None:
    probes = load_probes(PROBES_FILE)
    alphas = parse_dose_alphas(DEFAULT_ALPHAS)
    ids = {make_group_pair_id(g, p.probe_id, a, r, 3000)
           for g in ("if5|orig", "mf1|orig") for p in probes for a in alphas for r in range(3)}
    # every group gets its own ids — the 1,008-judged-as-336 bug
    assert len(ids) == 2 * len(probes) * len(alphas) * 3
    for pair_id in ids:
        assert pair_id.startswith("pair_") and len(pair_id) == len("pair_") + 12
        lowered = pair_id.lower()
        for token in ("alpha", "0.125", "0.25", "rep", "if5", "catch", "steer", "orig"):
            assert token not in lowered
    assert make_group_pair_id("g", "p", 0.5, 1, 3000) == make_group_pair_id("g", "p", 0.5, 1, 3000)
    assert make_group_pair_id("g", "p", 0.5, 1, 3000) != make_group_pair_id("g", "p", 0.5, 1, 3001)
    import hashlib
    assert make_group_pair_id("g", "p", 0.5, 1, 3000) == \
        "pair_" + hashlib.sha256(b"expB15pairid_3000_g_p_0.5_1").hexdigest()[:12]


def test_load_probes_accepts_the_prompts_spelling_and_refuses_junk(tmp_path: Path) -> None:
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps({"prompts": [{"id": "a", "class": "home", "prompt": "hi"}]}))
    assert load_probes(alt)[0].text == "hi"

    dupe = tmp_path / "dupe.json"
    dupe.write_text(json.dumps({"probes": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]}))
    with pytest.raises(ValueError, match="duplicate probe id"):
        load_probes(dupe)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"probes": []}))
    with pytest.raises(ValueError, match="no 'probes'"):
        load_probes(empty)

    blank = tmp_path / "blank.json"
    blank.write_text(json.dumps({"probes": [{"id": "a", "text": "  "}]}))
    with pytest.raises(ValueError, match="'id' and a 'text'"):
        load_probes(blank)


def test_find_group_names_what_is_available() -> None:
    bank = _Bank([("if5", "orig"), ("mf1", "orig")])
    assert find_group(bank, "if5|orig") == 0
    assert find_group(bank, "mf1|orig") == 1
    with pytest.raises(ValueError, match="is not in the levers npz"):
        find_group(bank, "if5|repl")


def test_the_synthetic_probe_fixture_loads_and_is_tagged() -> None:
    probes = load_probes(PROBES_FILE)
    assert len(probes) == 8
    assert {p.probe_class for p in probes} == {"home", "mid", "away"}
    assert len({p.probe_id for p in probes}) == len(probes)
    assert sum(1 for p in probes if p.probe_class == "away") >= 3
