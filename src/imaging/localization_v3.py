"""Safety-first deterministic tibiofemoral joint localization V3 candidate.

V3 separates candidate localization from anatomical plausibility validation. It operates on a
single midpoint panel after physical resampling and uses the panel position only to describe the
expected bilateral layout. It never assigns anatomical laterality and never uses scanner identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from imaging.preprocessing import PreprocessingError, robust_minmax


class QCState(str, Enum):
    """Three-state automatic disposition used by every V3 gate."""

    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class PanelLayoutQC:
    """Broad-panel structural checks performed before localization."""

    state: QCState
    physical_height_mm: float
    physical_width_mm: float
    aspect_ratio: float
    robust_contrast: float
    blank_block_fraction: float
    usable_width_fraction: float
    usable_height_fraction: float
    strongest_column_boundary_z: float
    strongest_row_boundary_z: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointBandCandidate:
    """One possible joint row and its panel-local evidence."""

    row: int
    column: int
    score: float
    vertical_peak_z: float
    vertical_prominence_z: float
    horizontal_support: float
    row_fraction: float
    column_fraction: float


@dataclass(frozen=True, slots=True)
class AnatomicalCropValidation:
    """Independent anatomical plausibility assessment of a proposed crop."""

    state: QCState
    score: float
    blank_block_fraction: float
    joint_band_edge_ratio: float
    superior_structure: float
    inferior_structure: float
    horizontal_joint_support: float
    central_structure_fraction: float
    collimation_dominance: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointLocalizationV3:
    """Selected center plus separate localizer, layout, and anatomy-gate evidence."""

    row: int
    column: int
    confidence: float
    qc_state: QCState
    localization_state: QCState
    layout: PanelLayoutQC
    anatomy: AnatomicalCropValidation
    candidate_count: int
    candidate_margin: float
    pair_row_delta_mm: float | None
    selected_candidate_score: float
    vertical_peak_z: float
    vertical_prominence_z: float
    horizontal_support: float
    row_fraction: float
    column_fraction: float


def _smooth(values: np.ndarray, width_pixels: int) -> np.ndarray:
    width = max(3, int(width_pixels)) | 1
    return np.convolve(values, np.ones(width, dtype=np.float32) / width, mode="same")


def _pixels(millimeters: float, spacing_mm: float, *, minimum: int = 1) -> int:
    return max(minimum, round(millimeters / spacing_mm))


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-8)


def _block_blank_fraction(normalized: np.ndarray, grid: int = 24) -> float:
    """Return the fraction of coarse blocks with negligible internal contrast."""

    rows, columns = normalized.shape
    grid_rows, grid_columns = min(grid, rows), min(grid, columns)
    row_stop = (rows // grid_rows) * grid_rows
    column_stop = (columns // grid_columns) * grid_columns
    blocks = (
        normalized[:row_stop, :column_stop]
        .reshape(grid_rows, row_stop // grid_rows, grid_columns, column_stop // grid_columns)
        .transpose(0, 2, 1, 3)
        .reshape(grid_rows, grid_columns, -1)
    )
    ranges = np.percentile(blocks, 90, axis=2) - np.percentile(blocks, 10, axis=2)
    return float(np.mean(ranges < 0.015))


def _extent_fraction(profile: np.ndarray) -> float:
    standardized = _robust_z(profile)
    active = np.flatnonzero(standardized > 0.75)
    if active.size < 2:
        return 0.0
    return float((active[-1] - active[0] + 1) / len(profile))


def assess_panel_layout(
    panel: np.ndarray,
    *,
    spacing_mm: float = 0.15,
) -> PanelLayoutQC:
    """Assess broad-panel geometry without proposing a joint coordinate."""

    if panel.ndim != 2 or min(panel.shape) < 128:
        raise PreprocessingError("V3 layout QC requires a usable two-dimensional panel")
    if spacing_mm <= 0 or not np.isfinite(spacing_mm):
        raise ValueError("Physical spacing must be finite and positive")
    normalized = robust_minmax(panel, lower_percentile=1.0, upper_percentile=99.0)
    contrast = float(np.ptp(normalized))
    if contrast <= 0:
        return PanelLayoutQC(
            state=QCState.FAIL,
            physical_height_mm=panel.shape[0] * spacing_mm,
            physical_width_mm=panel.shape[1] * spacing_mm,
            aspect_ratio=panel.shape[0] / panel.shape[1],
            robust_contrast=0.0,
            blank_block_fraction=1.0,
            usable_width_fraction=0.0,
            usable_height_fraction=0.0,
            strongest_column_boundary_z=0.0,
            strongest_row_boundary_z=0.0,
            reasons=("no_usable_contrast",),
        )
    rows, columns = normalized.shape
    gy = np.abs(np.diff(normalized, axis=0, prepend=normalized[:1]))
    gx = np.abs(np.diff(normalized, axis=1, prepend=normalized[:, :1]))
    row_profile = _smooth(np.percentile(gy, 75, axis=1), _pixels(3.0, spacing_mm))
    column_profile = _smooth(np.percentile(gx, 75, axis=0), _pixels(3.0, spacing_mm))
    row_z = _robust_z(row_profile)
    column_z = _robust_z(column_profile)
    physical_height = rows * spacing_mm
    physical_width = columns * spacing_mm
    aspect = physical_height / physical_width
    blank_fraction = _block_blank_fraction(normalized)
    usable_width = _extent_fraction(column_profile)
    usable_height = _extent_fraction(row_profile)
    strongest_column = float(np.max(column_z))
    strongest_row = float(np.max(row_z))

    fail_reasons: list[str] = []
    borderline_reasons: list[str] = []
    if not 0.85 <= aspect <= 3.6:
        fail_reasons.append("implausible_physical_aspect_ratio")
    if physical_height < 220 or physical_width < 115:
        fail_reasons.append("insufficient_physical_field_of_view")
    if blank_fraction > 0.985:
        fail_reasons.append("excessive_blank_background")
    if usable_width < 0.05 or usable_height < 0.05:
        fail_reasons.append("insufficient_anatomical_occupancy")
    if blank_fraction > 0.92:
        borderline_reasons.append("large_blank_regions")
    if usable_width < 0.10 or usable_height < 0.10:
        borderline_reasons.append("limited_anatomical_occupancy")
    if strongest_column > 100 or strongest_row > 100:
        borderline_reasons.append("dominant_exposure_or_collimation_boundary")
    if fail_reasons:
        state = QCState.FAIL
        reasons = tuple(fail_reasons + borderline_reasons)
    elif borderline_reasons:
        state = QCState.BORDERLINE
        reasons = tuple(borderline_reasons)
    else:
        state = QCState.PASS
        reasons = ()
    return PanelLayoutQC(
        state=state,
        physical_height_mm=physical_height,
        physical_width_mm=physical_width,
        aspect_ratio=aspect,
        robust_contrast=contrast,
        blank_block_fraction=blank_fraction,
        usable_width_fraction=usable_width,
        usable_height_fraction=usable_height,
        strongest_column_boundary_z=strongest_column,
        strongest_row_boundary_z=strongest_row,
        reasons=reasons,
    )


def _vertical_profile(
    normalized: np.ndarray,
    spacing_mm: float,
    panel_position: str,
) -> np.ndarray:
    rows, columns = normalized.shape
    if panel_position == "screen_left":
        column_start, column_stop = 0.25, 0.90
    elif panel_position == "screen_right":
        column_start, column_stop = 0.10, 0.75
    else:
        raise ValueError("panel_position must be screen_left or screen_right")
    start = max(1, round(columns * column_start))
    stop = min(columns - 1, round(columns * column_stop))
    central = normalized[:, start:stop]
    offset = _pixels(2.0, spacing_mm, minimum=2)
    gradient = np.zeros_like(central, dtype=np.float32)
    gradient[offset : rows - offset] = np.abs(central[2 * offset :] - central[: rows - 2 * offset])
    profile = np.percentile(gradient, 70, axis=1).astype(np.float32)
    return _smooth(profile, _pixels(3.0, spacing_mm))


def find_vertical_candidates(
    profile: np.ndarray,
    *,
    spacing_mm: float,
    maximum_candidates: int = 6,
    minimum_separation_mm: float = 18.0,
) -> list[tuple[int, float, float]]:
    """Return ranked, physically separated candidate rows as row/peak/prominence tuples."""

    if profile.ndim != 1 or len(profile) < 32:
        raise ValueError("A usable one-dimensional vertical profile is required")
    start = max(1, round(len(profile) * 0.30))
    stop = min(len(profile) - 1, round(len(profile) * 0.84))
    if stop <= start:
        raise ValueError("No plausible vertical candidate range remains")
    local = profile[start:stop]
    standardized = _robust_z(local)
    peak_indices = np.flatnonzero(
        (standardized >= np.roll(standardized, 1)) & (standardized >= np.roll(standardized, -1))
    )
    peak_indices = peak_indices[(peak_indices > 0) & (peak_indices < len(local) - 1)]
    if not peak_indices.size:
        peak_indices = np.asarray([int(np.argmax(standardized))])
    separation = _pixels(minimum_separation_mm, spacing_mm, minimum=2)
    ranked = sorted(
        peak_indices.tolist(), key=lambda index: float(standardized[index]), reverse=True
    )
    chosen: list[int] = []
    for index in ranked:
        if all(abs(index - previous) >= separation for previous in chosen):
            chosen.append(index)
        if len(chosen) == maximum_candidates:
            break
    results: list[tuple[int, float, float]] = []
    for index in chosen:
        competitor = standardized.copy()
        competitor[max(0, index - separation) : index + separation + 1] = -np.inf
        finite = competitor[np.isfinite(competitor)]
        second = float(np.max(finite)) if finite.size else 0.0
        results.append(
            (start + index, float(standardized[index]), float(standardized[index] - second))
        )
    return results


def _structural_horizontal_center(
    normalized: np.ndarray,
    *,
    row: int,
    spacing_mm: float,
    panel_position: str,
) -> tuple[int, float]:
    rows, columns = normalized.shape
    radius = _pixels(65.0, spacing_mm, minimum=8)
    band = normalized[max(1, row - radius) : min(rows - 1, row + radius + 1)]
    gx = np.abs(np.diff(band, axis=1, prepend=band[:, :1]))
    gy = np.abs(np.diff(band, axis=0, prepend=band[:1]))
    structure = np.percentile(gx + gy, 70, axis=0).astype(np.float32)
    structure = _smooth(structure, _pixels(5.0, spacing_mm))
    window = _pixels(70.0, spacing_mm, minimum=9)
    window_score = np.convolve(structure, np.ones(window, dtype=np.float32) / window, mode="same")
    start = max(window // 2, round(columns * 0.12))
    stop = min(columns - window // 2, round(columns * 0.88))
    if stop <= start:
        return columns // 2, 0.0
    restricted = window_score[start:stop]
    standardized = _robust_z(restricted)
    fractions = np.arange(start, stop) / columns
    expected = 0.59 if panel_position == "screen_left" else 0.41
    prior = np.exp(-0.5 * ((fractions - expected) / 0.20) ** 2)
    combined = standardized + 0.35 * prior
    local_index = int(np.argmax(combined))
    column = start + local_index
    support_threshold = float(np.percentile(structure[start:stop], 55))
    half = window // 2
    local_structure = structure[max(0, column - half) : min(columns, column + half + 1)]
    support = float(np.mean(local_structure > support_threshold)) if local_structure.size else 0.0
    return column, support


def _intensity_horizontal_center(
    normalized: np.ndarray,
    *,
    row: int,
    spacing_mm: float,
) -> tuple[int, float]:
    """Reproduce the broad V2 foreground centroid as one candidate, not a final answer."""

    rows, columns = normalized.shape
    radius = _pixels(30.0, spacing_mm, minimum=4)
    band = normalized[max(0, row - radius) : min(rows, row + radius + 1)]
    profile = np.percentile(band, 70, axis=0).astype(np.float32)
    profile = _smooth(profile, _pixels(3.0, spacing_mm))
    start, stop = max(1, round(columns * 0.08)), min(columns - 1, round(columns * 0.92))
    restricted = profile[start:stop]
    baseline = float(np.percentile(restricted, 45))
    weights = np.maximum(restricted - baseline, 0)
    if float(weights.sum()) <= 0:
        return columns // 2, 0.0
    coordinates = np.arange(start, stop)
    center = int(round(float(np.dot(coordinates, weights) / weights.sum())))
    return center, float(np.mean(weights > 0))


def _quick_anatomy_score(
    normalized: np.ndarray,
    *,
    row: int,
    column: int,
    spacing_mm: float,
) -> float:
    """Rank horizontal alternatives cheaply; the independent final validator remains decisive."""

    half = _pixels(80.0, spacing_mm)
    crop = normalized[
        max(0, row - half) : min(normalized.shape[0], row + half + 1),
        max(0, column - half) : min(normalized.shape[1], column + half + 1),
    ]
    if min(crop.shape, default=0) < 32:
        return 0.0
    stride = max(1, round(1.2 / spacing_mm))
    small = crop[::stride, ::stride]
    gx = np.abs(np.diff(small, axis=1, prepend=small[:, :1]))
    gy = np.abs(np.diff(small, axis=0, prepend=small[:1]))
    gradient = gx + gy
    center_row = min(small.shape[0] - 1, (row - max(0, row - half)) // stride)
    radius = _pixels(10.0, spacing_mm * stride)
    joint = gy[max(0, center_row - radius) : center_row + radius + 1]
    row_profile = np.percentile(gy, 70, axis=1)
    local_peak = float(np.max(row_profile[max(0, center_row - radius) : center_row + radius + 1]))
    joint_ratio = local_peak / max(float(np.median(row_profile)), 1e-6)
    threshold = float(np.percentile(gradient, 65))
    horizontal_support = float(np.mean(np.max(joint, axis=0) > threshold)) if joint.size else 0.0
    structure_energy = float(np.percentile(gradient, 85))
    energy_z = float(np.clip(structure_energy / max(float(np.median(gradient)), 1e-5), 0, 8))
    return float(
        0.45 * np.clip(joint_ratio / 3.0, 0, 1)
        + 0.35 * np.clip(horizontal_support / 0.70, 0, 1)
        + 0.20 * np.clip(energy_z / 5.0, 0, 1)
    )


def _candidate_list(
    panel: np.ndarray,
    *,
    spacing_mm: float,
    panel_position: str,
) -> list[JointBandCandidate]:
    normalized = robust_minmax(panel, lower_percentile=1.0, upper_percentile=99.0)
    profile = _vertical_profile(normalized, spacing_mm, panel_position)
    rows, columns = normalized.shape
    candidates: list[JointBandCandidate] = []
    for row, peak, prominence in find_vertical_candidates(profile, spacing_mm=spacing_mm):
        structural_column, structural_support = _structural_horizontal_center(
            normalized,
            row=row,
            spacing_mm=spacing_mm,
            panel_position=panel_position,
        )
        intensity_column, intensity_support = _intensity_horizontal_center(
            normalized,
            row=row,
            spacing_mm=spacing_mm,
        )
        expected_fraction = 0.59 if panel_position == "screen_left" else 0.41
        horizontal_options = {
            structural_column: structural_support,
            intensity_column: intensity_support,
            round(columns * expected_fraction): 0.50,
        }
        for column, support in horizontal_options.items():
            row_fraction = row / rows
            column_fraction = column / columns
            vertical_position = max(0.0, 1.0 - abs(row_fraction - 0.54) / 0.28)
            horizontal_position = float(
                np.exp(-0.5 * ((column_fraction - expected_fraction) / 0.20) ** 2)
            )
            anatomy_score = _quick_anatomy_score(
                normalized,
                row=row,
                column=column,
                spacing_mm=spacing_mm,
            )
            score = (
                0.32 * float(np.clip((peak - 1.5) / 8.0, 0, 1))
                + 0.12 * float(np.clip((prominence + 0.5) / 5.0, 0, 1))
                + 0.08 * float(np.clip(support / 0.50, 0, 1))
                + 0.10 * vertical_position
                + 0.12 * horizontal_position
                + 0.26 * anatomy_score
            )
            candidates.append(
                JointBandCandidate(
                    row=row,
                    column=column,
                    score=score,
                    vertical_peak_z=peak,
                    vertical_prominence_z=prominence,
                    horizontal_support=support,
                    row_fraction=row_fraction,
                    column_fraction=column_fraction,
                )
            )
    return sorted(candidates, key=lambda value: value.score, reverse=True)[:12]


def validate_anatomical_crop(
    panel: np.ndarray,
    *,
    center: tuple[int, int],
    spacing_mm: float = 0.15,
    crop_size_mm: float = 160.0,
) -> AnatomicalCropValidation:
    """Validate a proposed crop independently of the candidate-ranking score."""

    normalized = robust_minmax(panel, lower_percentile=1.0, upper_percentile=99.0)
    rows, columns = normalized.shape
    half = _pixels(crop_size_mm / 2, spacing_mm)
    row, column = center
    row_start, row_stop = max(0, row - half), min(rows, row + half + 1)
    col_start, col_stop = max(0, column - half), min(columns, column + half + 1)
    crop = normalized[row_start:row_stop, col_start:col_stop]
    if crop.size == 0:
        return AnatomicalCropValidation(
            QCState.FAIL, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ("empty_crop",)
        )
    # Evaluate at at most about 0.6 mm/pixel; all distances remain physical.
    stride = max(1, round(0.6 / spacing_mm))
    small = crop[::stride, ::stride]
    effective_spacing = spacing_mm * stride
    gy = np.abs(np.diff(small, axis=0, prepend=small[:1]))
    gx = np.abs(np.diff(small, axis=1, prepend=small[:, :1]))
    gradient = gx + gy
    small_center_row = min(small.shape[0] - 1, max(0, (row - row_start) // stride))
    small_center_col = min(small.shape[1] - 1, max(0, (column - col_start) // stride))
    central_half_width = _pixels(60.0, effective_spacing)
    x0 = max(0, small_center_col - central_half_width)
    x1 = min(small.shape[1], small_center_col + central_half_width + 1)
    joint_radius = _pixels(10.0, effective_spacing)
    joint = gy[
        max(0, small_center_row - joint_radius) : min(
            small.shape[0], small_center_row + joint_radius + 1
        ),
        x0:x1,
    ]
    row_profile = np.percentile(gy[:, x0:x1], 70, axis=1) if x1 > x0 else np.zeros(small.shape[0])
    local_joint = float(
        np.max(
            row_profile[
                max(0, small_center_row - joint_radius) : small_center_row + joint_radius + 1
            ]
        )
    )
    background_joint = float(np.median(row_profile))
    joint_ratio = local_joint / max(background_joint, 1e-6)
    threshold = float(np.percentile(gradient, 60))

    def region_structure(upper_mm: float, lower_mm: float) -> float:
        a = max(0, small_center_row + round(upper_mm / effective_spacing))
        b = min(small.shape[0], small_center_row + round(lower_mm / effective_spacing))
        region = gradient[a:b, x0:x1]
        return float(np.mean(region > threshold)) if region.size else 0.0

    superior = region_structure(-65.0, -15.0)
    inferior = region_structure(15.0, 65.0)
    horizontal_support = float(np.mean(np.max(joint, axis=0) > threshold)) if joint.size else 0.0
    central = gradient[:, x0:x1]
    central_structure = float(np.mean(central > threshold)) if central.size else 0.0
    blank_fraction = _block_blank_fraction(small, grid=16)
    column_profile = np.percentile(gx, 80, axis=0)
    collimation = float(np.max(_robust_z(column_profile))) if column_profile.size else 0.0

    fail_reasons: list[str] = []
    borderline_reasons: list[str] = []
    if superior < 0.12:
        fail_reasons.append("insufficient_superior_bone_structure")
    if inferior < 0.12:
        fail_reasons.append("insufficient_inferior_bone_structure")
    if horizontal_support < 0.35:
        fail_reasons.append("insufficient_horizontal_joint_support")
    if central_structure < 0.25:
        fail_reasons.append("shaft_or_background_dominant_crop")
    if blank_fraction > 0.88:
        fail_reasons.append("excessive_blank_background")
    if joint_ratio < 1.15:
        fail_reasons.append("no_plausible_joint_space_band")
    if superior < 0.18 or inferior < 0.18:
        borderline_reasons.append("limited_bone_coverage")
    if horizontal_support < 0.45 or central_structure < 0.32:
        borderline_reasons.append("limited_joint_anatomy_support")
    if blank_fraction > 0.76:
        borderline_reasons.append("large_blank_background")
    if collimation > 30:
        borderline_reasons.append("collimation_boundary_dominance")

    score = float(
        np.mean(
            [
                np.clip(superior / 0.40, 0, 1),
                np.clip(inferior / 0.40, 0, 1),
                np.clip(horizontal_support / 0.60, 0, 1),
                np.clip(central_structure / 0.40, 0, 1),
                np.clip(joint_ratio / 2.5, 0, 1),
                np.clip((0.90 - blank_fraction) / 0.50, 0, 1),
            ]
        )
    )
    if fail_reasons:
        state = QCState.FAIL
        reasons = tuple(fail_reasons + borderline_reasons)
    elif borderline_reasons:
        state = QCState.BORDERLINE
        reasons = tuple(borderline_reasons)
    else:
        state = QCState.PASS
        reasons = ()
    return AnatomicalCropValidation(
        state=state,
        score=score,
        blank_block_fraction=blank_fraction,
        joint_band_edge_ratio=joint_ratio,
        superior_structure=superior,
        inferior_structure=inferior,
        horizontal_joint_support=horizontal_support,
        central_structure_fraction=central_structure,
        collimation_dominance=collimation,
        reasons=reasons,
    )


def _combine_states(*states: QCState) -> QCState:
    if QCState.FAIL in states:
        return QCState.FAIL
    if QCState.BORDERLINE in states:
        return QCState.BORDERLINE
    return QCState.PASS


def _provisional_result(
    panel: np.ndarray,
    candidate: JointBandCandidate,
    candidates: list[JointBandCandidate],
    layout: PanelLayoutQC,
    *,
    spacing_mm: float,
    panel_position: str,
) -> JointLocalizationV3:
    anatomy = validate_anatomical_crop(
        panel,
        center=(candidate.row, candidate.column),
        spacing_mm=spacing_mm,
    )
    row_separation = _pixels(10.0, spacing_mm)
    column_separation = _pixels(20.0, spacing_mm)
    distinct_competitors = [
        value
        for value in candidates
        if abs(value.row - candidate.row) >= row_separation
        or abs(value.column - candidate.column) >= column_separation
    ]
    competitor_score = max((value.score for value in distinct_competitors), default=0.0)
    margin = candidate.score - competitor_score
    if candidate.vertical_peak_z < 1.5:
        local_state = QCState.FAIL
    elif margin < 0.02:
        local_state = QCState.BORDERLINE
    else:
        local_state = QCState.PASS
    if panel_position == "screen_left":
        if candidate.column_fraction < 0.30 or candidate.column_fraction > 0.84:
            local_state = QCState.FAIL
        elif candidate.column_fraction < 0.35 or candidate.column_fraction > 0.78:
            local_state = _combine_states(local_state, QCState.BORDERLINE)
    else:
        if candidate.column_fraction < 0.16 or candidate.column_fraction > 0.70:
            local_state = QCState.FAIL
        elif candidate.column_fraction < 0.22 or candidate.column_fraction > 0.65:
            local_state = _combine_states(local_state, QCState.BORDERLINE)
    state = _combine_states(layout.state, local_state, anatomy.state)
    confidence = float(
        np.clip(0.45 * candidate.score + 0.45 * anatomy.score + 0.10 * min(margin / 0.08, 1), 0, 1)
    )
    return JointLocalizationV3(
        row=candidate.row,
        column=candidate.column,
        confidence=confidence,
        qc_state=state,
        localization_state=local_state,
        layout=layout,
        anatomy=anatomy,
        candidate_count=len(candidates),
        candidate_margin=margin,
        pair_row_delta_mm=None,
        selected_candidate_score=candidate.score,
        vertical_peak_z=candidate.vertical_peak_z,
        vertical_prominence_z=candidate.vertical_prominence_z,
        horizontal_support=candidate.horizontal_support,
        row_fraction=candidate.row_fraction,
        column_fraction=candidate.column_fraction,
    )


def localize_tibiofemoral_joint_v3(
    panel: np.ndarray,
    *,
    panel_position: str,
    spacing_mm: float = 0.15,
) -> JointLocalizationV3:
    """Localize one panel, requiring both candidate and independent anatomy gates."""

    layout = assess_panel_layout(panel, spacing_mm=spacing_mm)
    if layout.state is QCState.FAIL:
        failed = AnatomicalCropValidation(
            QCState.FAIL, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ("layout_gate_failed",)
        )
        return JointLocalizationV3(
            panel.shape[0] // 2,
            panel.shape[1] // 2,
            0.0,
            QCState.FAIL,
            QCState.FAIL,
            layout,
            failed,
            0,
            0.0,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
        )
    candidates = _candidate_list(panel, spacing_mm=spacing_mm, panel_position=panel_position)
    if not candidates:
        raise PreprocessingError("V3 localization produced no candidate joint bands")
    return _provisional_result(
        panel,
        candidates[0],
        candidates,
        layout,
        spacing_mm=spacing_mm,
        panel_position=panel_position,
    )


def localize_bilateral_tibiofemoral_joints_v3(
    screen_left: np.ndarray,
    screen_right: np.ndarray,
    *,
    spacing_mm: float = 0.15,
) -> tuple[JointLocalizationV3, JointLocalizationV3]:
    """Rank bilateral candidate pairs, then apply independent anatomical validation."""

    layouts = {
        "screen_left": assess_panel_layout(screen_left, spacing_mm=spacing_mm),
        "screen_right": assess_panel_layout(screen_right, spacing_mm=spacing_mm),
    }
    panels = {"screen_left": screen_left, "screen_right": screen_right}
    if any(value.state is QCState.FAIL for value in layouts.values()):
        return (
            localize_tibiofemoral_joint_v3(
                screen_left, panel_position="screen_left", spacing_mm=spacing_mm
            ),
            localize_tibiofemoral_joint_v3(
                screen_right, panel_position="screen_right", spacing_mm=spacing_mm
            ),
        )
    candidates = {
        position: _candidate_list(panel, spacing_mm=spacing_mm, panel_position=position)
        for position, panel in panels.items()
    }
    pairs: list[tuple[float, JointBandCandidate, JointBandCandidate]] = []
    for left in candidates["screen_left"]:
        for right in candidates["screen_right"]:
            delta_mm = abs(left.row - right.row) * spacing_mm
            agreement = np.exp(-0.5 * (delta_mm / 10.0) ** 2)
            score = left.score + right.score + 0.40 * float(agreement)
            pairs.append((score, left, right))
    pairs.sort(key=lambda value: value[0], reverse=True)
    _, selected_left, selected_right = pairs[0]
    distinct_pairs = [
        value
        for value in pairs[1:]
        if abs(value[1].row - selected_left.row) * spacing_mm >= 10.0
        or abs(value[2].row - selected_right.row) * spacing_mm >= 10.0
        or abs(value[1].column - selected_left.column) * spacing_mm >= 20.0
        or abs(value[2].column - selected_right.column) * spacing_mm >= 20.0
    ]
    pair_margin = pairs[0][0] - (distinct_pairs[0][0] if distinct_pairs else 0.0)
    delta_mm = abs(selected_left.row - selected_right.row) * spacing_mm
    results: list[JointLocalizationV3] = []
    for position, selected in (
        ("screen_left", selected_left),
        ("screen_right", selected_right),
    ):
        result = _provisional_result(
            panels[position],
            selected,
            candidates[position],
            layouts[position],
            spacing_mm=spacing_mm,
            panel_position=position,
        )
        local_state = QCState.PASS if selected.vertical_peak_z >= 1.5 else QCState.FAIL
        if position == "screen_left":
            if selected.column_fraction < 0.30 or selected.column_fraction > 0.84:
                local_state = QCState.FAIL
            elif selected.column_fraction < 0.35 or selected.column_fraction > 0.78:
                local_state = _combine_states(local_state, QCState.BORDERLINE)
        else:
            if selected.column_fraction < 0.16 or selected.column_fraction > 0.70:
                local_state = QCState.FAIL
            elif selected.column_fraction < 0.22 or selected.column_fraction > 0.65:
                local_state = _combine_states(local_state, QCState.BORDERLINE)
        if delta_mm > 25:
            local_state = QCState.FAIL
        elif delta_mm > 15 or pair_margin < 0.005:
            local_state = _combine_states(local_state, QCState.BORDERLINE)
        state = _combine_states(result.layout.state, local_state, result.anatomy.state)
        results.append(
            replace(
                result,
                qc_state=state,
                localization_state=local_state,
                confidence=float(result.confidence * np.clip(pair_margin / 0.08, 0.35, 1.0)),
                candidate_margin=pair_margin,
                pair_row_delta_mm=delta_mm,
            )
        )
    return results[0], results[1]
