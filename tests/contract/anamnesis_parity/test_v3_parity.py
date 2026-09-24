"""The v3 CPU feature space through the installed anamnesis == the banked reference.

The shipped 3B/8B map, atlas and levers were fit in the space one anamnesis
snapshot emitted. The feature-list check only catches a changed feature LIST,
so this pins the VALUES: ``v3_vendored_reference.json`` is that snapshot's
output on the synthetic capture ``synthetic`` builds, and the anamnesis on the
path must reproduce it bit-for-bit, in the v3 configuration
``pleroma.harvest.replay.build_configs`` builds.

Three cases: calibration covering every position, positions running past the
calibration table (the clamp path), and a 4-token generation.

An anamnesis bump that fails this test has changed the instrument under the
shipped artifacts. That is not automatically wrong, but it is never silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.contract.anamnesis_parity import synthetic as s

FIXTURE = Path(__file__).with_name("v3_vendored_reference.json")


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def submodule_output() -> dict:
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
    from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
    from anamnesis.extraction.state_extractor import RawGenerationData

    ex = ExtractionConfig(**s.EXTRACTION_COMMON, enable_residual_pca=True)
    fam = FeaturePipelineConfig(include_core_blocks=True, **s.FAMILY_FLAGS)
    comps, mean = s.pca_fit()
    results = {
        tag: compute_features_with_families_from_data(
            s.make_raw(RawGenerationData, T, rows), ex, fam, comps, mean
        )
        for tag, T, rows in s.CASES
    }
    return s.results_to_doc(results, "submodule")


def test_the_submodule_is_what_got_imported() -> None:
    import anamnesis

    root = Path(__file__).resolve().parents[3] / "vendor" / "anamnesis"
    if not root.is_dir():
        pytest.skip("no vendor/anamnesis in this tree (anamnesis is an installed dependency)")
    assert Path(anamnesis.__file__).resolve().is_relative_to(root.resolve()), (
        f"anamnesis resolved to {anamnesis.__file__}, not the pinned submodule at {root}"
    )


def test_names_and_block_slices_match(reference, submodule_output) -> None:
    assert submodule_output["feature_names"] == reference["feature_names"]
    assert submodule_output["block_slices"] == reference["block_slices"]


@pytest.mark.parametrize("tag", [c[0] for c in s.CASES])
def test_values_match_bit_for_bit(tag: str, reference, submodule_output) -> None:
    got = np.asarray(submodule_output["features"][tag], dtype=np.float64)
    want = np.asarray(reference["features"][tag], dtype=np.float64)
    assert got.shape == want.shape
    # NaN-aware exact equality: NaN positions must coincide, every other value
    # must be identical to the last bit.
    np.testing.assert_array_equal(got, want)
