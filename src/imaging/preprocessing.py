"""Pilot-only physical normalization and deterministic joint-localization primitives.

These functions operate on in-memory arrays. They do not read identifiers, assign laterality,
download images, define cohorts, or create data partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class PreprocessingError(ValueError):
    """Raised when a pilot image cannot be processed deterministically."""


@dataclass(frozen=True, slots=True)
class JointLocalization:
    """Proposed tibiofemoral joint center in source-array coordinates."""

    row: int
    column: int
    score: float
    robust_score: float


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """Padding applied while producing a fixed-size crop."""

    requested_rows: int
    requested_columns: int
    padding_top: int
    padding_bottom: int
    padding_left: int
    padding_right: int

    @property
    def used_padding(self) -> bool:
        return any((self.padding_top, self.padding_bottom, self.padding_left, self.padding_right))


def robust_clip(
    array: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Clip finite pixels to robust bounds without changing source data."""

    if array.ndim != 2 or array.size == 0:
        raise PreprocessingError("Normalization requires a non-empty two-dimensional image")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must be ordered within 0..100")
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise PreprocessingError("Image contains no finite values")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    return np.clip(values, low, high)


def robust_minmax(
    array: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Return float32 values in 0..1 after robust clipping."""

    clipped = robust_clip(
        array,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    low = float(clipped.min())
    high = float(clipped.max())
    if high <= low:
        return np.zeros(clipped.shape, dtype=np.float32)
    return ((clipped - low) / (high - low)).astype(np.float32)


def robust_zscore(
    array: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Return a clipped float32 image with zero mean and unit standard deviation."""

    clipped = robust_clip(
        array,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    standard_deviation = float(clipped.std())
    if standard_deviation == 0:
        return np.zeros(clipped.shape, dtype=np.float32)
    return ((clipped - float(clipped.mean())) / standard_deviation).astype(np.float32)


def resample_to_spacing(
    array: np.ndarray,
    source_spacing: tuple[float, float],
    target_spacing: tuple[float, float],
) -> np.ndarray:
    """Resample a 2D image with bilinear interpolation while preserving its dtype."""

    if array.ndim != 2 or array.size == 0:
        raise PreprocessingError("Resampling requires a non-empty two-dimensional image")
    if len(source_spacing) != 2 or len(target_spacing) != 2:
        raise ValueError("Spacing must contain row and column values")
    if any(value <= 0 or not np.isfinite(value) for value in (*source_spacing, *target_spacing)):
        raise ValueError("Spacing values must be finite and positive")
    output_rows = max(1, round(array.shape[0] * source_spacing[0] / target_spacing[0]))
    output_columns = max(1, round(array.shape[1] * source_spacing[1] / target_spacing[1]))
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    resized = np.asarray(
        image.resize((output_columns, output_rows), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        return np.rint(np.clip(resized, limits.min, limits.max)).astype(array.dtype)
    return resized.astype(array.dtype, copy=False)


def crop_around_center(
    array: np.ndarray,
    center: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    padding_value: int | float = 0,
) -> tuple[np.ndarray, CropGeometry]:
    """Return an exact-size crop, padding rather than shifting an out-of-bounds request."""

    if array.ndim != 2 or array.size == 0:
        raise PreprocessingError("Cropping requires a non-empty two-dimensional image")
    output_rows, output_columns = output_shape
    if output_rows < 1 or output_columns < 1:
        raise ValueError("Crop dimensions must be positive")
    center_row, center_column = center
    top = center_row - output_rows // 2
    left = center_column - output_columns // 2
    bottom = top + output_rows
    right = left + output_columns
    source_top = max(top, 0)
    source_left = max(left, 0)
    source_bottom = min(bottom, array.shape[0])
    source_right = min(right, array.shape[1])
    padding_top = source_top - top
    padding_left = source_left - left
    padding_bottom = bottom - source_bottom
    padding_right = right - source_right
    output = np.full(output_shape, padding_value, dtype=array.dtype)
    destination_bottom = output_rows - padding_bottom
    destination_right = output_columns - padding_right
    if source_bottom > source_top and source_right > source_left:
        output[
            padding_top:destination_bottom,
            padding_left:destination_right,
        ] = array[source_top:source_bottom, source_left:source_right]
    geometry = CropGeometry(
        requested_rows=output_rows,
        requested_columns=output_columns,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        padding_right=padding_right,
    )
    return output, geometry


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _joint_space_projection(array: np.ndarray) -> np.ndarray:
    """Return a vertical dark-gap score for a broad knee panel.

    The score compares each row with brighter bands above and below at several physical-looking
    scales expressed as fractions of the image height. It is an image heuristic, not a clinical
    joint-space-width measurement.
    """
    normalized = robust_minmax(array, lower_percentile=1.0, upper_percentile=99.0)
    rows, columns = normalized.shape
    if rows < 64 or columns < 64:
        raise PreprocessingError("Image is too small for joint localization")
    if float(np.ptp(normalized)) <= 0:
        raise PreprocessingError("Image has no usable contrast for joint localization")
    horizontal_start = max(1, round(columns * 0.08))
    horizontal_stop = min(columns - 1, round(columns * 0.92))
    row_profile = np.percentile(normalized[:, horizontal_start:horizontal_stop], 80, axis=1)
    row_profile = _smooth(row_profile.astype(np.float32), max(3, round(rows * 0.003)))
    scale_scores: list[np.ndarray] = []
    for scale_fraction in (0.008, 0.012, 0.018, 0.025):
        offset = max(3, round(rows * scale_fraction))
        center = _smooth(row_profile, max(3, round(rows * 0.002)))
        score = np.zeros(rows, dtype=np.float32)
        score[offset : rows - offset] = (
            0.5 * (row_profile[: rows - 2 * offset] + row_profile[2 * offset :])
            - center[offset : rows - offset]
        )
        scale_scores.append(score)
    projection = np.max(np.stack(scale_scores), axis=0)
    foreground_fraction = np.mean(
        normalized[:, horizontal_start:horizontal_stop] > 0.35,
        axis=1,
    )
    foreground_fraction = _smooth(foreground_fraction, max(7, round(rows * 0.02)))
    projection *= np.clip(foreground_fraction / 0.25, 0, 1.5)
    search_start = max(1, round(rows * 0.25))
    search_stop = min(rows - 1, round(rows * 0.86))
    if search_stop <= search_start:
        raise PreprocessingError("Image geometry leaves no joint-localization search region")
    projection[:search_start] = -np.inf
    projection[search_stop:] = -np.inf
    return projection


def _robust_projection_score(projection: np.ndarray) -> np.ndarray:
    finite = np.isfinite(projection)
    values = projection[finite]
    if not values.size:
        raise PreprocessingError("Joint-localization projection contains no finite values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    standardized = np.full(projection.shape, -np.inf, dtype=np.float32)
    standardized[finite] = (values - median) / max(mad * 1.4826, 1e-8)
    return standardized


def _joint_column(array: np.ndarray, joint_row: int) -> int:
    normalized = robust_minmax(array, lower_percentile=1.0, upper_percentile=99.0)
    rows, columns = normalized.shape
    horizontal_start = max(1, round(columns * 0.08))
    horizontal_stop = min(columns - 1, round(columns * 0.92))

    half_band = max(8, round(rows * 0.045))
    band = normalized[max(0, joint_row - half_band) : min(rows, joint_row + half_band + 1)]
    column_profile = np.percentile(band, 70, axis=0)
    lower = float(np.percentile(column_profile, 45))
    weights = np.maximum(column_profile - lower, 0)
    weights[:horizontal_start] = 0
    weights[horizontal_stop:] = 0
    if float(weights.sum()) <= 0:
        return columns // 2
    return int(round(float(np.dot(np.arange(columns), weights) / weights.sum())))


def localize_tibiofemoral_joint(array: np.ndarray) -> JointLocalization:
    """Estimate one panel's joint center from a multi-scale joint-space projection.

    When a bilateral pair is available, :func:`localize_bilateral_tibiofemoral_joints` is
    preferred because it uses the shared acquisition row geometry. Every result still requires a
    visual QC state; the numerical score is not an acceptance rule.
    """

    projection = _joint_space_projection(array)
    standardized = _robust_projection_score(projection)
    joint_row = int(np.argmax(standardized))
    peak_score = float(projection[joint_row])
    if peak_score <= 0:
        raise PreprocessingError("No positive joint-space candidate was found")
    return JointLocalization(
        joint_row,
        _joint_column(array, joint_row),
        peak_score,
        float(standardized[joint_row]),
    )


def localize_bilateral_tibiofemoral_joints(
    screen_left: np.ndarray,
    screen_right: np.ndarray,
) -> tuple[JointLocalization, JointLocalization]:
    """Localize a shared joint row in two midpoint panels without assigning laterality.

    OAI fixed-flexion pilot knees share a detector row. Robustly standardized per-panel
    projections are summed so a marker, collimation edge, or low-contrast panel is less likely to
    dominate. The two horizontal centers remain panel-specific.
    """

    if screen_left.shape[0] != screen_right.shape[0]:
        raise PreprocessingError("Bilateral panels must share the same source row count")
    left_projection = _joint_space_projection(screen_left)
    right_projection = _joint_space_projection(screen_right)
    combined = _robust_projection_score(left_projection) + _robust_projection_score(
        right_projection
    )
    joint_row = int(np.argmax(combined))
    if left_projection[joint_row] <= 0 and right_projection[joint_row] <= 0:
        raise PreprocessingError("No positive paired joint-space candidate was found")
    peak_score = float(combined[joint_row])
    finite_values = combined[np.isfinite(combined)]
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    robust_score = (peak_score - median) / max(mad * 1.4826, 1e-8)
    return (
        JointLocalization(
            joint_row,
            _joint_column(screen_left, joint_row),
            float(left_projection[joint_row]),
            robust_score,
        ),
        JointLocalization(
            joint_row,
            _joint_column(screen_right, joint_row),
            float(right_projection[joint_row]),
            robust_score,
        ),
    )


def physical_crop_shape(
    physical_size_mm: tuple[float, float],
    target_spacing_mm: tuple[float, float],
) -> tuple[int, int]:
    """Convert a physical crop extent to a deterministic integer pixel shape."""

    if len(physical_size_mm) != 2 or len(target_spacing_mm) != 2:
        raise ValueError("Physical size and spacing must each contain two values")
    if any(value <= 0 or not np.isfinite(value) for value in physical_size_mm):
        raise ValueError("Physical crop dimensions must be finite and positive")
    if any(value <= 0 or not np.isfinite(value) for value in target_spacing_mm):
        raise ValueError("Target spacing must be finite and positive")
    return tuple(
        max(1, round(size / spacing))
        for size, spacing in zip(physical_size_mm, target_spacing_mm, strict=True)
    )


def _preview_image(array: np.ndarray) -> Image.Image:
    normalized = robust_minmax(array, lower_percentile=0.5, upper_percentile=99.5)
    return Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").convert("RGB")


def save_localization_preview(
    array: np.ndarray,
    localization: JointLocalization,
    crop_shape: tuple[int, int],
    destination: str | Path,
    *,
    pilot_index: int,
    panel_position: str,
    label_prefix: str = "Pilot",
    max_width: int = 900,
) -> None:
    """Save a de-identified panel preview with joint center and proposed crop box."""

    if panel_position not in {"screen_left", "screen_right"}:
        raise ValueError("Panel position must be screen_left or screen_right")
    image = _preview_image(array)
    scale = min(1.0, max_width / image.width)
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", (image.width, image.height + 28), color="black")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    center_x = round(localization.column * scale)
    center_y = round(localization.row * scale) + 28
    half_height = round(crop_shape[0] * scale / 2)
    half_width = round(crop_shape[1] * scale / 2)
    draw.rectangle(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        outline=(255, 220, 0),
        width=3,
    )
    draw.line((center_x - 14, center_y, center_x + 14, center_y), fill=(0, 255, 0), width=3)
    draw.line((center_x, center_y - 14, center_x, center_y + 14), fill=(0, 255, 0), width=3)
    label = panel_position.replace("_", "-")
    draw.text((8, 7), f"{label_prefix} {pilot_index:03d} | {label}", fill="white")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)


def save_crop_preview(
    array: np.ndarray,
    destination: str | Path,
    *,
    pilot_index: int,
    panel_position: str,
    label_prefix: str = "Pilot",
    preview_size: int = 512,
) -> None:
    """Save a de-identified square uint8 preview of a scientific crop."""

    if panel_position not in {"screen_left", "screen_right"}:
        raise ValueError("Panel position must be screen_left or screen_right")
    image = _preview_image(array).resize((preview_size, preview_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (preview_size, preview_size + 28), color="black")
    canvas.paste(image, (0, 28))
    label = panel_position.replace("_", "-")
    ImageDraw.Draw(canvas).text((8, 7), f"{label_prefix} {pilot_index:03d} | {label}", fill="white")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
