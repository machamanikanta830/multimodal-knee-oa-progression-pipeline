"""Synthetic tests for independent imaging-validation orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from imaging.independent_validation import (
    CROP_SIZE_MM,
    OUTPUT_SHAPE,
    TARGET_SPACING_MM,
    IndependentValidationError,
    _optional_float,
    _pooled_groups,
    _single_extracted_file,
)

pytestmark = pytest.mark.public_portable


def test_frozen_geometry_has_expected_pixel_shape() -> None:
    assert TARGET_SPACING_MM == 0.15
    assert CROP_SIZE_MM == 160.0
    assert OUTPUT_SHAPE == (1067, 1067)


@pytest.mark.parametrize("value", [None, "", "not-a-number"])
def test_optional_float_uses_identity_default_for_missing_values(value: object) -> None:
    assert _optional_float(value, 1.0) == 1.0


def test_small_heterogeneity_groups_are_pooled() -> None:
    values = pd.Series(["common"] * 5 + ["rare-a", "rare-b", None])

    pooled = _pooled_groups(values, minimum_size=5)

    assert pooled.tolist()[:5] == ["common"] * 5
    assert pooled.tolist()[5:] == ["<pooled groups n<5>"] * 3


def test_single_extracted_file_accepts_one_nested_file(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = nested / "image.dcm"
    expected.write_bytes(b"synthetic")

    assert _single_extracted_file(tmp_path) == expected


def test_single_extracted_file_rejects_ambiguous_contents(tmp_path: Path) -> None:
    (tmp_path / "first.dcm").write_bytes(b"synthetic")
    (tmp_path / "second.dcm").write_bytes(b"synthetic")

    with pytest.raises(IndependentValidationError, match="exactly one regular file"):
        _single_extracted_file(tmp_path)
