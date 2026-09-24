"""ModelProfile / Deployment: the shipped profiles load, and the validators
refuse the configurations that have bitten us."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from pleroma.config import (
    InjectionSpan,
    ModelProfile,
    load_profile,
)

ROOT = Path(__file__).resolve().parents[3]
PROFILES = sorted((ROOT / "profiles").glob("*.toml")) + sorted(
    (ROOT / "tests" / "fixtures" / "profiles").glob("*.toml"))


#: a model-C-format profile with public values (tests/fixtures/profiles)
MODELC_FORMAT = ROOT / "tests" / "fixtures" / "profiles" / "modelc-format-70b.toml"


def _raw(name: str) -> dict:
    path = MODELC_FORMAT if name == "modelc" else ROOT / "profiles" / name
    with path.open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_shipped_profile_validates(path: Path) -> None:
    """A profile in the repo that does not load is a broken default."""
    prof = load_profile(path)
    assert prof.name == path.stem


def test_profiles_default_to_uniform_injection() -> None:
    """Whole-sequence (uniform) injection is the design: it modestly beats
    continuation-only injection (docs/FINDINGS.md section 4)."""
    for path in PROFILES:
        assert load_profile(path).steer.injection_span is InjectionSpan.UNIFORM


def test_unknown_key_is_an_error_not_a_silent_default() -> None:
    raw = _raw("modelc")
    raw["lengths"]["futre_tokens"] = 999
    with pytest.raises(ValidationError, match="futre_tokens"):
        ModelProfile.model_validate(raw)


def test_pad_equal_to_eos_is_refused() -> None:
    raw = _raw("modelc")
    raw["model"]["arch"]["pad_token_id"] = 128001
    with pytest.raises(ValidationError, match="eos"):
        ModelProfile.model_validate(raw)


def test_site_outside_the_model_is_refused() -> None:
    raw = _raw("llama31-8b-instruct.toml")
    raw["map"]["sites"] = [9, 17, 21, 40]
    with pytest.raises(ValidationError, match="outside"):
        ModelProfile.model_validate(raw)


def test_unsorted_sites_are_refused() -> None:
    raw = _raw("modelc")
    raw["map"]["sites"] = [42, 22, 52, 62]
    with pytest.raises(ValidationError, match="increasing"):
        ModelProfile.model_validate(raw)


def test_modelc_without_stops_is_refused() -> None:
    raw = _raw("modelc")
    raw["format"]["stops"] = []
    with pytest.raises(ValidationError, match="stop"):
        ModelProfile.model_validate(raw)


def test_inconsistent_architecture_is_refused() -> None:
    raw = _raw("modelc")
    raw["model"]["arch"]["head_dim"] = 64
    with pytest.raises(ValidationError, match="hidden_dim"):
        ModelProfile.model_validate(raw)


def test_missing_profile_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        load_profile(tmp_path / "nope.toml")


