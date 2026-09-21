"""Acquisition-structure exception checks required by the frozen Laterality V2 policy.

The frozen policy maps ``screen_left`` to the anatomical right knee and ``screen_right`` to the
anatomical left knee only for acquisitions that match the validated OAI bilateral posteroanterior
fixed-flexion structure. This module supplies the image-evidence exception detector that the policy
depends on: it confirms bilateral structure, a clean midpoint separation, the expected gross
orientation, and an intact image, and otherwise reports why the validated rule does not apply.

The checks are deliberately permissive about normal anatomical and exposure variation and strict
only about gross structural anomalies. They read pixels and physical spacing only; they never read
identifiers and never assign an anatomical side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imaging.preprocessing import robust_minmax

MINIMUM_PANEL_HEIGHT_MM = 150.0
MINIMUM_PANEL_WIDTH_MM = 80.0
MINIMUM_PANEL_OCCUPANCY = 0.06
MINIMUM_OCCUPANCY_SYMMETRY = 0.30
MINIMUM_LIMB_SEPARATION_DEPTH = 0.10
MINIMUM_LIMB_PEAK_SEPARATION_MM = 60.0
MAXIMUM_VERTICAL_CENTROID_DISPARITY_MM = 60.0
MAXIMUM_FLAT_ROW_FRACTION = 0.60
MAXIMUM_EDGE_FLAT_RUN_FRACTION = 0.30
MAXIMUM_SATURATED_FRACTION = 0.60
PROFILE_SMOOTHING_MM = 10.0


@dataclass(frozen=True, slots=True)
class BilateralStructureCheck:
    """Image-evidence verdict on whether the validated OAI screen rule may be applied."""

    passed: bool
    bilateral_structure_confirmed: bool
    midpoint_separation_clean: bool
    orientation_expected: bool
    image_intact: bool
    screen_left_occupancy: float
    screen_right_occupancy: float
    occupancy_symmetry: float
    limb_separation_depth: float
    limb_peak_separation_mm: float
    vertical_centroid_disparity_mm: float
    panel_height_mm: float
    panel_width_mm: float
    panel_aspect_ratio: float
    flat_row_fraction: float
    edge_flat_run_fraction: float
    saturated_fraction: float
    reasons: tuple[str, ...]


def _occupancy_profile(normalized: np.ndarray, smoothing_pixels: int) -> np.ndarray:
    threshold = float(np.percentile(normalized, 60))
    profile = np.mean(normalized > threshold, axis=0).astype(np.float32)
    width = max(3, smoothing_pixels) | 1
    return np.convolve(profile, np.ones(width, dtype=np.float32) / width, mode="same")


def _vertical_centroid(normalized: np.ndarray) -> float:
    threshold = float(np.percentile(normalized, 60))
    weights = np.mean(normalized > threshold, axis=1).astype(np.float32)
    total = float(weights.sum())
    if total <= 0:
        return float(normalized.shape[0]) / 2
    return float(np.dot(np.arange(normalized.shape[0]), weights) / total)


def _edge_flat_run_fraction(flat_rows: np.ndarray) -> float:
    """Return the largest contiguous run of contrast-free rows that touches a horizontal edge.

    Collimated borders are normal and produce short edge runs. A run covering a large share of the
    detector height instead indicates a truncated or partially written image.
    """

    if not flat_rows.size:
        return 0.0
    leading = int(np.argmin(flat_rows)) if flat_rows[0] else 0
    trailing = int(np.argmin(flat_rows[::-1])) if flat_rows[-1] else 0
    if flat_rows.all():
        return 1.0
    return max(leading, trailing) / len(flat_rows)


def _limb_separation(profile: np.ndarray, midpoint: int) -> tuple[float, int]:
    """Measure how clearly two limbs separate along the horizontal occupancy profile.

    The strongest occupancy column in each half locates a limb. The depth of the valley between
    those two columns distinguishes a genuine two-limb acquisition, where background separates the
    knees, from a single central object that merely straddles the midpoint. Returned depth is the
    relative drop from the weaker limb peak to the valley, and the separation is in columns.
    """

    left = profile[:midpoint]
    right = profile[midpoint:]
    if not left.size or not right.size:
        return 0.0, 0
    left_column = int(np.argmax(left))
    right_column = midpoint + int(np.argmax(right))
    weaker_peak = min(float(left[left_column]), float(right[right_column - midpoint]))
    if weaker_peak <= 0:
        return 0.0, right_column - left_column
    valley = float(np.min(profile[left_column : right_column + 1]))
    return 1.0 - valley / weaker_peak, right_column - left_column


def _unusable(
    reasons: tuple[str, ...],
    *,
    midpoint_separation_clean: bool = False,
    image_intact: bool = False,
    panel_height_mm: float = 0.0,
    panel_width_mm: float = 0.0,
    panel_aspect_ratio: float = 0.0,
    flat_row_fraction: float = 1.0,
    edge_flat_run_fraction: float = 1.0,
    saturated_fraction: float = 0.0,
) -> BilateralStructureCheck:
    """Return a failing verdict for an acquisition that cannot be assessed any further."""

    return BilateralStructureCheck(
        passed=False,
        bilateral_structure_confirmed=False,
        midpoint_separation_clean=midpoint_separation_clean,
        orientation_expected=False,
        image_intact=image_intact,
        screen_left_occupancy=0.0,
        screen_right_occupancy=0.0,
        occupancy_symmetry=0.0,
        limb_separation_depth=0.0,
        limb_peak_separation_mm=0.0,
        vertical_centroid_disparity_mm=0.0,
        panel_height_mm=panel_height_mm,
        panel_width_mm=panel_width_mm,
        panel_aspect_ratio=panel_aspect_ratio,
        flat_row_fraction=flat_row_fraction,
        edge_flat_run_fraction=edge_flat_run_fraction,
        saturated_fraction=saturated_fraction,
        reasons=reasons,
    )


def assess_bilateral_structure(
    source: np.ndarray,
    *,
    row_spacing_mm: float,
    column_spacing_mm: float,
) -> BilateralStructureCheck:
    """Decide whether one acquisition matches the validated bilateral structure.

    ``passed`` is true only when bilateral structure, midpoint separation, gross orientation, and
    image integrity all hold. A false result is an exception for the frozen laterality rule, not a
    reversal of it: the caller withholds the anatomical side instead of guessing.
    """

    if source.ndim != 2:
        return _unusable(("pixel_array_is_not_two_dimensional",))
    if any(not np.isfinite(value) or value <= 0 for value in (row_spacing_mm, column_spacing_mm)):
        raise ValueError("Physical spacing must be finite and positive")

    rows, columns = source.shape
    midpoint = columns // 2
    reasons: list[str] = []

    values = np.asarray(source, dtype=np.float32)
    flat_rows = np.ptp(values, axis=1) <= 0
    flat_row_fraction = float(np.mean(flat_rows))
    edge_flat_run_fraction = _edge_flat_run_fraction(flat_rows)
    maximum = float(values.max())
    saturated_fraction = float(np.mean(values >= maximum)) if np.isfinite(maximum) else 1.0
    image_intact = True
    if columns < 2 or rows < 2 or float(np.ptp(values)) <= 0:
        image_intact = False
        reasons.append("no_usable_image_contrast")
    if flat_row_fraction > MAXIMUM_FLAT_ROW_FRACTION:
        image_intact = False
        reasons.append("flat_field_dominates_image")
    if edge_flat_run_fraction > MAXIMUM_EDGE_FLAT_RUN_FRACTION:
        image_intact = False
        reasons.append("truncated_image_rows")
    if saturated_fraction > MAXIMUM_SATURATED_FRACTION:
        image_intact = False
        reasons.append("saturated_or_corrupt_pixel_data")

    panel_height_mm = rows * row_spacing_mm
    panel_width_mm = midpoint * column_spacing_mm
    panel_aspect_ratio = panel_height_mm / panel_width_mm if panel_width_mm > 0 else 0.0

    midpoint_separation_clean = True
    if abs(midpoint - (columns - midpoint)) > 1:
        midpoint_separation_clean = False
        reasons.append("midpoint_split_does_not_divide_panels_evenly")
    if panel_height_mm < MINIMUM_PANEL_HEIGHT_MM or panel_width_mm < MINIMUM_PANEL_WIDTH_MM:
        midpoint_separation_clean = False
        reasons.append("panel_physical_extent_too_small_for_bilateral_structure")

    if not image_intact or not midpoint_separation_clean:
        return _unusable(
            tuple(reasons),
            midpoint_separation_clean=midpoint_separation_clean,
            image_intact=image_intact,
            panel_height_mm=panel_height_mm,
            panel_width_mm=panel_width_mm,
            panel_aspect_ratio=panel_aspect_ratio,
            flat_row_fraction=flat_row_fraction,
            edge_flat_run_fraction=edge_flat_run_fraction,
            saturated_fraction=saturated_fraction,
        )

    normalized = robust_minmax(source, lower_percentile=1.0, upper_percentile=99.0)
    left, right = normalized[:, :midpoint], normalized[:, midpoint:]
    left_occupancy = float(np.mean(left > float(np.percentile(normalized, 60))))
    right_occupancy = float(np.mean(right > float(np.percentile(normalized, 60))))
    strongest = max(left_occupancy, right_occupancy)
    occupancy_symmetry = min(left_occupancy, right_occupancy) / strongest if strongest > 0 else 0.0

    profile = _occupancy_profile(normalized, round(PROFILE_SMOOTHING_MM / column_spacing_mm))
    separation_depth, separation_columns = _limb_separation(profile, midpoint)
    limb_peak_separation_mm = separation_columns * column_spacing_mm

    bilateral_structure_confirmed = True
    if min(left_occupancy, right_occupancy) < MINIMUM_PANEL_OCCUPANCY:
        bilateral_structure_confirmed = False
        reasons.append("one_screen_panel_lacks_anatomy")
    if occupancy_symmetry < MINIMUM_OCCUPANCY_SYMMETRY:
        bilateral_structure_confirmed = False
        reasons.append("screen_panel_occupancy_is_grossly_asymmetric")
    if separation_depth < MINIMUM_LIMB_SEPARATION_DEPTH:
        bilateral_structure_confirmed = False
        reasons.append("two_limbs_do_not_separate_across_the_midpoint")
    if limb_peak_separation_mm < MINIMUM_LIMB_PEAK_SEPARATION_MM:
        bilateral_structure_confirmed = False
        reasons.append("limb_centers_are_too_close_for_a_bilateral_acquisition")

    disparity_mm = abs(_vertical_centroid(left) - _vertical_centroid(right)) * row_spacing_mm
    orientation_expected = True
    if panel_aspect_ratio < 1.0:
        orientation_expected = False
        reasons.append("screen_panels_are_not_taller_than_wide")
    if disparity_mm > MAXIMUM_VERTICAL_CENTROID_DISPARITY_MM:
        orientation_expected = False
        reasons.append("gross_vertical_disagreement_between_screen_panels")

    passed = bilateral_structure_confirmed and orientation_expected
    return BilateralStructureCheck(
        passed=passed,
        bilateral_structure_confirmed=bilateral_structure_confirmed,
        midpoint_separation_clean=midpoint_separation_clean,
        orientation_expected=orientation_expected,
        image_intact=image_intact,
        screen_left_occupancy=left_occupancy,
        screen_right_occupancy=right_occupancy,
        occupancy_symmetry=occupancy_symmetry,
        limb_separation_depth=separation_depth,
        limb_peak_separation_mm=limb_peak_separation_mm,
        vertical_centroid_disparity_mm=disparity_mm,
        panel_height_mm=panel_height_mm,
        panel_width_mm=panel_width_mm,
        panel_aspect_ratio=panel_aspect_ratio,
        flat_row_fraction=flat_row_fraction,
        edge_flat_run_fraction=edge_flat_run_fraction,
        saturated_fraction=saturated_fraction,
        reasons=tuple(reasons),
    )
