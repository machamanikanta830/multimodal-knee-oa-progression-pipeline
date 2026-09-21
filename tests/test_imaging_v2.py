"""Synthetic tests for the V2 localizer, methodology audit, and laterality hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imaging.laterality_v2 import LateralityV2State, assess_laterality_v2
from imaging.localization_v2 import LocalizationQC, localize_tibiofemoral_joint_v2
from imaging.methodology_v2 import MethodologyV2Error, _coverage_select
from imaging.preprocessing import PreprocessingError

pytestmark = pytest.mark.public_portable


def _synthetic_joint(rows: int = 360, columns: int = 260, joint_row: int = 205) -> np.ndarray:
    image = np.zeros((rows, columns), dtype=np.uint16)
    image[70 : joint_row - 8, 55:205] = 2600
    image[joint_row + 8 : 330, 45:215] = 2200
    return image


def test_v2_localizer_returns_coordinate_confidence_and_qc() -> None:
    result = localize_tibiofemoral_joint_v2(_synthetic_joint(), spacing_mm=0.15)

    assert 180 <= result.row <= 225
    assert 95 <= result.column <= 165
    assert 0 <= result.confidence <= 1
    assert result.qc_state in set(LocalizationQC)


def test_v2_localizer_fails_safely_on_constant_pixels() -> None:
    with pytest.raises(PreprocessingError, match="usable image contrast"):
        localize_tibiofemoral_joint_v2(np.zeros((300, 220), dtype=np.uint16))


def test_coverage_selection_is_deterministic_and_unique() -> None:
    frame = pd.DataFrame(
        {
            "acquisition_index": range(20),
            "manufacturer": ["A", "B"] * 10,
            "manufacturer_model_name": ["M1", "M2", "M3", "M4"] * 5,
            "original_row_spacing_mm": [0.1, 0.15, 0.2, 0.15] * 5,
            "dimension_family": ["1x1", "2x2"] * 10,
            "image_release_study": ["R1", "R2"] * 10,
            "xray_accept_qc": ["Y", "YD"] * 10,
            "qc_problem_signature": ["", "alignment"] * 10,
            "eligible_knee_count": [1, 2] * 10,
            "maximum_padding_mm": np.arange(20),
        }
    )

    first = _coverage_select(frame, 8, seed="synthetic-v2")
    second = _coverage_select(frame, 8, seed="synthetic-v2")

    assert first == second
    assert len(first) == len(set(first)) == 8


def test_coverage_selection_rejects_oversized_request() -> None:
    frame = pd.DataFrame({"acquisition_index": [1]})

    with pytest.raises(MethodologyV2Error, match="Not enough candidates"):
        _coverage_select(frame, 2, seed="synthetic-v2")


def test_unilateral_dicom_tag_does_not_override_primary_image_evidence() -> None:
    result = assess_laterality_v2(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        burned_in_marker_mapping=("R", "L"),
        dicom_laterality="L",
    )

    assert result.state is LateralityV2State.CONFIDENT
    assert result.screen_left_anatomical_side == "R"
    assert result.screen_right_anatomical_side == "L"
    assert result.unilateral_dicom_tag_artifact


def test_disagreeing_image_evidence_is_conflicting_and_unassigned() -> None:
    result = assess_laterality_v2(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        burned_in_marker_mapping=("R", "L"),
        anatomical_mapping=("L", "R"),
    )

    assert result.state is LateralityV2State.CONFLICTING
    assert result.screen_left_anatomical_side is None
    assert result.screen_right_anatomical_side is None


def test_screen_rule_requires_approval_and_exception_checks() -> None:
    unresolved = assess_laterality_v2(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        validated_screen_mapping=("R", "L"),
        screen_mapping_validation_approved=True,
        exception_detector_passed=False,
    )
    resolved = assess_laterality_v2(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        validated_screen_mapping=("R", "L"),
        screen_mapping_validation_approved=True,
        exception_detector_passed=True,
    )

    assert unresolved.state is LateralityV2State.AMBIGUOUS
    assert resolved.state is LateralityV2State.CONFIDENT
    assert resolved.evidence_source == "validated_oai_screen_position"


def test_unconfirmed_bilateral_geometry_never_receives_anatomical_side() -> None:
    result = assess_laterality_v2(
        bilateral_acquisition=False,
        midpoint_separation_clean=False,
        burned_in_marker_mapping=("R", "L"),
    )

    assert result.state is LateralityV2State.AMBIGUOUS
    assert result.screen_left_anatomical_side is None
