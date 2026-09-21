"""Synthetic tests for pilot physical normalization and joint localization."""

from __future__ import annotations

import numpy as np
import pytest

from imaging.preprocessing import (
    PreprocessingError,
    crop_around_center,
    localize_bilateral_tibiofemoral_joints,
    localize_tibiofemoral_joint,
    physical_crop_shape,
    resample_to_spacing,
    robust_minmax,
    robust_zscore,
)

pytestmark = pytest.mark.public_portable


def test_resampling_uses_physical_spacing_and_preserves_uint16() -> None:
    image = np.arange(137 * 91, dtype=np.uint16).reshape(137, 91)

    output = resample_to_spacing(image, (0.2, 0.2), (0.1, 0.1))

    assert output.shape == (274, 182)
    assert output.dtype == np.uint16


def test_crop_boundary_padding_preserves_requested_geometry() -> None:
    image = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)

    crop, geometry = crop_around_center(image, (1, 1), (6, 6), padding_value=0)

    assert crop.shape == (6, 6)
    assert crop.dtype == image.dtype
    assert geometry.padding_top == 2
    assert geometry.padding_left == 2
    assert geometry.padding_bottom == 0
    assert geometry.padding_right == 0
    assert geometry.used_padding
    assert np.array_equal(crop[2:, 2:], image[:4, :4])


def test_robust_normalization_limits_outlier_influence() -> None:
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    image[-1, -1] = 1_000_000

    minmax = robust_minmax(image, lower_percentile=0, upper_percentile=95)
    zscore = robust_zscore(image, lower_percentile=0, upper_percentile=95)

    assert minmax.dtype == np.float32
    assert minmax.min() == 0
    assert minmax.max() == 1
    assert abs(float(zscore.mean())) < 1e-5
    assert abs(float(zscore.std()) - 1) < 1e-5


def test_projection_localizer_finds_synthetic_joint_gap() -> None:
    image = np.zeros((240, 180), dtype=np.uint16)
    image[55:115, 35:145] = 2400
    image[125:210, 30:150] = 2200

    localization = localize_tibiofemoral_joint(image)

    assert 108 <= localization.row <= 130
    assert 75 <= localization.column <= 105
    assert localization.robust_score > 0


def test_bilateral_localizer_uses_a_shared_row_without_assigning_side() -> None:
    screen_left = np.zeros((260, 170), dtype=np.uint16)
    screen_right = np.zeros((260, 190), dtype=np.uint16)
    screen_left[40:122, 35:145] = 2500
    screen_left[133:235, 25:155] = 2200
    screen_right[45:122, 45:155] = 2400
    screen_right[133:230, 35:165] = 2150
    screen_left[75:82, :] = 4095  # panel-specific distractor

    left, right = localize_bilateral_tibiofemoral_joints(screen_left, screen_right)

    assert left.row == right.row
    assert 115 <= left.row <= 140
    assert 65 <= left.column <= 110
    assert 75 <= right.column <= 120


def test_physical_crop_shape_is_derived_from_millimeters() -> None:
    assert physical_crop_shape((160.0, 160.0), (0.15, 0.15)) == (1067, 1067)


def test_localization_rejects_a_constant_image() -> None:
    image = np.zeros((240, 180), dtype=np.uint16)

    with pytest.raises(PreprocessingError, match="no usable contrast"):
        localize_tibiofemoral_joint(image)


def test_localization_fails_safely_for_unusable_geometry() -> None:
    image = np.zeros((32, 48), dtype=np.uint16)

    with pytest.raises(PreprocessingError, match="too small"):
        localize_tibiofemoral_joint(image)
