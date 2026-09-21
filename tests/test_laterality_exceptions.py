"""Synthetic tests for the bilateral-structure exception detector guarding Laterality V2."""

from __future__ import annotations

import numpy as np
import pytest

from imaging.laterality_exceptions import (
    MINIMUM_LIMB_SEPARATION_DEPTH,
    assess_bilateral_structure,
)
from imaging.laterality_v2 import LateralityV2State, assess_validated_oai_screen_rule

pytestmark = pytest.mark.public_portable


def _bilateral_acquisition(rows: int = 2048, columns: int = 2494) -> np.ndarray:
    """Build a coarse two-limb posteroanterior layout with background between the panels."""

    image = np.full((rows, columns), 140, dtype=np.uint16)
    midpoint = columns // 2
    rng = np.random.default_rng(11)
    for center in (midpoint // 2, midpoint + midpoint // 2):
        half = round(midpoint * 0.28)
        image[round(rows * 0.12) : round(rows * 0.90), center - half : center + half] = 2400
    texture = rng.integers(0, 200, size=image.shape, dtype=np.uint16)
    return np.where(image > 200, image + texture, image).astype(np.uint16)


def test_expected_bilateral_acquisition_passes_every_exception_check() -> None:
    check = assess_bilateral_structure(
        _bilateral_acquisition(), row_spacing_mm=0.17, column_spacing_mm=0.17
    )

    assert check.passed
    assert check.bilateral_structure_confirmed
    assert check.midpoint_separation_clean
    assert check.orientation_expected
    assert check.image_intact
    assert check.reasons == ()


def test_single_limb_acquisition_is_detected_as_non_bilateral() -> None:
    image = _bilateral_acquisition()
    image[:, image.shape[1] // 2 :] = 140

    check = assess_bilateral_structure(image, row_spacing_mm=0.17, column_spacing_mm=0.17)

    assert not check.passed
    assert not check.bilateral_structure_confirmed
    assert "one_screen_panel_lacks_anatomy" in check.reasons


def test_truncated_image_rows_are_detected_as_corruption() -> None:
    image = _bilateral_acquisition()
    image[round(image.shape[0] * 0.6) :] = 0

    check = assess_bilateral_structure(image, row_spacing_mm=0.17, column_spacing_mm=0.17)

    assert not check.passed
    assert not check.image_intact
    assert "truncated_image_rows" in check.reasons


def test_landscape_panels_are_detected_as_unexpected_orientation() -> None:
    check = assess_bilateral_structure(
        _bilateral_acquisition(rows=900, columns=4000),
        row_spacing_mm=0.17,
        column_spacing_mm=0.17,
    )

    assert not check.passed
    assert not check.orientation_expected
    assert "screen_panels_are_not_taller_than_wide" in check.reasons


def test_midpoint_dominant_anatomy_is_detected_as_incompatible_structure() -> None:
    rows, columns = 2048, 2494
    image = np.full((rows, columns), 140, dtype=np.uint16)
    midpoint = columns // 2
    image[round(rows * 0.12) : round(rows * 0.90), midpoint - 500 : midpoint + 500] = 2400

    check = assess_bilateral_structure(image, row_spacing_mm=0.17, column_spacing_mm=0.17)

    assert not check.passed
    assert not check.bilateral_structure_confirmed
    assert "two_limbs_do_not_separate_across_the_midpoint" in check.reasons
    assert check.limb_separation_depth < MINIMUM_LIMB_SEPARATION_DEPTH


def test_failed_exception_check_withholds_the_frozen_mapping() -> None:
    unresolved = assess_validated_oai_screen_rule(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        exception_detector_passed=False,
    )
    resolved = assess_validated_oai_screen_rule(
        bilateral_acquisition=True,
        midpoint_separation_clean=True,
        exception_detector_passed=True,
        dicom_laterality="R",
    )

    assert unresolved.state is LateralityV2State.AMBIGUOUS
    assert unresolved.screen_left_anatomical_side is None
    assert resolved.state is LateralityV2State.CONFIDENT
    assert resolved.screen_left_anatomical_side == "R"
    assert resolved.screen_right_anatomical_side == "L"
    assert resolved.unilateral_dicom_tag_artifact is True


def test_nonpositive_spacing_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        assess_bilateral_structure(
            _bilateral_acquisition(), row_spacing_mm=0.0, column_spacing_mm=0.17
        )
