"""Physically normalized deterministic tibiofemoral joint localizer V2 candidate.

The localizer operates independently on each midpoint panel after resampling to a known physical
spacing. Bilateral agreement is used only to lower QC confidence, never to replace an independent
coordinate. This is an imaging-methodology component and does not use OA outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from imaging.preprocessing import PreprocessingError, robust_minmax


class LocalizationQC(str, Enum):
    """Automatic disposition for a proposed joint center."""

    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class JointLocalizationV2:
    """Joint candidate plus interpretable confidence and QC evidence."""

    row: int
    column: int
    confidence: float
    qc_state: LocalizationQC
    vertical_peak_z: float
    vertical_prominence_z: float
    joint_line_support: float
    row_fraction: float
    column_fraction: float
    pair_row_delta_mm: float | None = None
    bilateral_rescue_used: bool = False


def _smooth_physical(values: np.ndarray, millimeters: float, spacing_mm: float) -> np.ndarray:
    window = max(3, round(millimeters / spacing_mm)) | 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-8)


def _vertical_joint_profile(normalized: np.ndarray, spacing_mm: float) -> np.ndarray:
    rows, columns = normalized.shape
    # Peripheral burned-in labels and rectangular redaction masks were the dominant V2-r1
    # distractors. The tibiofemoral anatomy remains inside the central 60% of every development
    # panel, so exclude the periphery before forming the row score. A lower percentile also
    # requires an edge response to span a meaningful part of the knee rather than a small label.
    start = max(1, round(columns * 0.20))
    stop = min(columns - 1, round(columns * 0.80))
    central = normalized[:, start:stop]
    offset = max(2, round(2.0 / spacing_mm))
    gradient = np.zeros_like(central, dtype=np.float32)
    gradient[offset : rows - offset] = np.abs(central[2 * offset :] - central[: rows - 2 * offset])
    profile = np.percentile(gradient, 70, axis=1).astype(np.float32)
    return _smooth_physical(profile, 3.0, spacing_mm)


def _horizontal_joint_center(
    normalized: np.ndarray, joint_row: int, spacing_mm: float
) -> tuple[int, float]:
    rows, columns = normalized.shape
    # Estimate the center of the broad osseous foreground surrounding the joint. This is more
    # resistant to isolated metalwork and burned-in labels than taking a centroid of the strongest
    # local gradient responses.
    row_radius = max(4, round(30.0 / spacing_mm))
    row_start = max(0, joint_row - row_radius)
    row_stop = min(rows, joint_row + row_radius + 1)
    band = normalized[row_start:row_stop]
    if not band.size:
        return columns // 2, 0.0
    profile = np.percentile(band, 70, axis=0).astype(np.float32)
    profile = _smooth_physical(profile, 3.0, spacing_mm)
    search_start = max(1, round(columns * 0.08))
    search_stop = min(columns - 1, round(columns * 0.92))
    restricted = profile[search_start:search_stop]
    baseline = float(np.percentile(restricted, 45))
    weights = np.maximum(restricted - baseline, 0)
    if float(weights.sum()) <= 0:
        return columns // 2, 0.0
    coordinates = np.arange(search_start, search_stop)
    center = int(round(float(np.dot(coordinates, weights) / weights.sum())))
    support = float(np.mean(weights > 0))
    return center, support


def _localize_candidate(
    resampled_panel: np.ndarray,
    *,
    spacing_mm: float,
    candidate_center_row: int | None = None,
    candidate_radius_mm: float = 18.0,
) -> JointLocalizationV2:
    if resampled_panel.ndim != 2 or min(resampled_panel.shape) < 128:
        raise PreprocessingError("V2 localization requires a usable two-dimensional knee panel")
    if spacing_mm <= 0 or not np.isfinite(spacing_mm):
        raise ValueError("Physical spacing must be finite and positive")
    normalized = robust_minmax(resampled_panel, lower_percentile=1.0, upper_percentile=99.0)
    if float(np.ptp(normalized)) <= 0:
        raise PreprocessingError("V2 localization requires usable image contrast")
    rows, columns = normalized.shape
    profile = _vertical_joint_profile(normalized, spacing_mm)
    # Development QC showed that candidates above one third of the panel were collimation,
    # label, or shaft responses; all visually adequate joints were below this boundary. This is a
    # dimensionless anatomical field-of-view constraint, not a scanner-specific pixel coordinate.
    search_start = max(1, round(rows * 0.33))
    search_stop = min(rows - 1, round(rows * 0.82))
    if candidate_center_row is None:
        candidate_start, candidate_stop = search_start, search_stop
    else:
        radius = max(2, round(candidate_radius_mm / spacing_mm))
        candidate_start = max(search_start, candidate_center_row - radius)
        candidate_stop = min(search_stop, candidate_center_row + radius + 1)
    if candidate_stop <= candidate_start:
        raise PreprocessingError("V2 localization has no physically plausible candidate rows")
    joint_row = candidate_start + int(np.argmax(profile[candidate_start:candidate_stop]))
    standardized = _robust_z(profile[search_start:search_stop])
    local_index = joint_row - search_start
    peak_z = float(standardized[local_index])
    exclusion = max(2, round(15.0 / spacing_mm))
    competitor = standardized.copy()
    competitor[max(0, local_index - exclusion) : local_index + exclusion + 1] = -np.inf
    finite_competitor = competitor[np.isfinite(competitor)]
    second_z = float(np.max(finite_competitor)) if finite_competitor.size else 0.0
    prominence_z = peak_z - second_z
    joint_column, line_support = _horizontal_joint_center(normalized, joint_row, spacing_mm)
    row_fraction = joint_row / rows
    column_fraction = joint_column / columns

    confidence_components = (
        np.clip((peak_z - 2.0) / 6.0, 0, 1),
        np.clip((prominence_z + 0.5) / 3.0, 0, 1),
        np.clip(line_support / 0.45, 0, 1),
        np.clip(1.0 - abs(column_fraction - 0.5) / 0.38, 0, 1),
    )
    confidence = float(np.mean(confidence_components))
    if (
        peak_z >= 3.0
        and prominence_z >= -0.25
        and 0.12 <= column_fraction <= 0.88
        and 0.33 <= row_fraction <= 0.81
    ):
        state = LocalizationQC.PASS
    elif peak_z >= 1.5 and 0.08 <= column_fraction <= 0.92 and 0.32 <= row_fraction <= 0.82:
        state = LocalizationQC.BORDERLINE
    else:
        state = LocalizationQC.FAIL
    return JointLocalizationV2(
        row=joint_row,
        column=joint_column,
        confidence=confidence,
        qc_state=state,
        vertical_peak_z=peak_z,
        vertical_prominence_z=prominence_z,
        joint_line_support=line_support,
        row_fraction=row_fraction,
        column_fraction=column_fraction,
    )


def localize_tibiofemoral_joint_v2(
    resampled_panel: np.ndarray,
    *,
    spacing_mm: float = 0.15,
) -> JointLocalizationV2:
    """Locate a joint on one physically normalized panel and assign an automatic QC state."""

    return _localize_candidate(resampled_panel, spacing_mm=spacing_mm)


def localize_bilateral_tibiofemoral_joints_v2(
    screen_left_resampled: np.ndarray,
    screen_right_resampled: np.ndarray,
    *,
    spacing_mm: float = 0.15,
) -> tuple[JointLocalizationV2, JointLocalizationV2]:
    """Localize each panel independently and use bilateral row agreement as secondary QC."""

    left = localize_tibiofemoral_joint_v2(screen_left_resampled, spacing_mm=spacing_mm)
    right = localize_tibiofemoral_joint_v2(screen_right_resampled, spacing_mm=spacing_mm)
    initial_delta_mm = abs(left.row - right.row) * spacing_mm

    # When the two independent rows disagree, test two constrained alternatives. The
    # contralateral row only defines a physical search window: the replacement coordinate must
    # still be supported by the discrepant panel's own gradient and foreground evidence.
    if initial_delta_mm > 15.0:
        alternate_right = _localize_candidate(
            screen_right_resampled,
            spacing_mm=spacing_mm,
            candidate_center_row=left.row,
        )
        alternate_left = _localize_candidate(
            screen_left_resampled,
            spacing_mm=spacing_mm,
            candidate_center_row=right.row,
        )

        def pair_score(pair: tuple[JointLocalizationV2, JointLocalizationV2]) -> float:
            first, second = pair
            delta = abs(first.row - second.row) * spacing_mm
            evidence = (
                min(first.confidence, second.confidence)
                + 0.2 * (first.confidence + second.confidence) / 2
            )
            return float(evidence - min(delta, 60.0) / 120.0)

        primary_pair = (left, right)
        options = (primary_pair, (left, alternate_right), (alternate_left, right))
        eligible = [
            pair
            for pair in options
            if min(pair[0].vertical_peak_z, pair[1].vertical_peak_z) >= 1.5
            and 0.08 <= pair[0].column_fraction <= 0.92
            and 0.08 <= pair[1].column_fraction <= 0.92
        ]
        if eligible:
            selected = max(eligible, key=pair_score)
            if (
                selected is not primary_pair
                and pair_score(selected) >= pair_score(primary_pair) + 0.05
            ):
                left, right = (
                    replace(selected[0], bilateral_rescue_used=True),
                    replace(selected[1], bilateral_rescue_used=True),
                )

    delta_mm = abs(left.row - right.row) * spacing_mm

    def adjusted(value: JointLocalizationV2) -> JointLocalizationV2:
        state = value.qc_state
        confidence = value.confidence
        if delta_mm > 30.0:
            state = LocalizationQC.FAIL
            confidence *= 0.35
        elif delta_mm > 15.0 and state is LocalizationQC.PASS:
            state = LocalizationQC.BORDERLINE
            confidence *= 0.65
        return replace(
            value,
            confidence=float(np.clip(confidence, 0, 1)),
            qc_state=state,
            pair_row_delta_mm=delta_mm,
        )

    return adjusted(left), adjusted(right)
