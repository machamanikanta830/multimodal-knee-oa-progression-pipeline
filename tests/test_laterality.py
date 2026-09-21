"""Synthetic evidence-state tests for bilateral laterality assignment."""

from __future__ import annotations

import pytest

from imaging.laterality import LateralityState, assess_laterality

pytestmark = pytest.mark.public_portable


def test_confident_marker_mapping_is_returned() -> None:
    result = assess_laterality(
        bilateral_acquisition=True,
        dicom_laterality="B",
        image_laterality=None,
        marker_mapping=("R", "L"),
    )

    assert result.state is LateralityState.CONFIDENT
    assert result.screen_left_anatomical_side == "R"
    assert result.screen_right_anatomical_side == "L"


def test_missing_mapping_evidence_remains_ambiguous() -> None:
    result = assess_laterality(
        bilateral_acquisition=True,
        dicom_laterality=None,
        image_laterality=None,
        marker_mapping=None,
    )

    assert result.state is LateralityState.AMBIGUOUS
    assert result.screen_left_anatomical_side is None
    assert result.screen_right_anatomical_side is None


def test_unilateral_tag_on_bilateral_image_is_conflicting_and_unmapped() -> None:
    result = assess_laterality(
        bilateral_acquisition=True,
        dicom_laterality="L",
        image_laterality=None,
        marker_mapping=("R", "L"),
    )

    assert result.state is LateralityState.CONFLICTING
    assert result.screen_left_anatomical_side is None
    assert result.screen_right_anatomical_side is None


def test_disagreeing_orientation_and_marker_mappings_are_conflicting() -> None:
    result = assess_laterality(
        bilateral_acquisition=True,
        dicom_laterality=None,
        image_laterality=None,
        marker_mapping=("R", "L"),
        orientation_mapping=("L", "R"),
    )

    assert result.state is LateralityState.CONFLICTING
    assert result.screen_left_anatomical_side is None
    assert result.screen_right_anatomical_side is None
