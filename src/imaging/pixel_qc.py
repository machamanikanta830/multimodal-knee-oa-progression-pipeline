"""Privacy-safe pixel helpers for the OAI imaging pilot.

These functions load pixels only from caller-provided local DICOM files. They do not download,
split cohorts, or serialize identifiers, paths, dates, private tags, or pixel arrays.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.errors import InvalidDicomError


class PixelInspectionError(ValueError):
    """Raised when pixel data cannot be inspected safely."""


@dataclass(frozen=True, slots=True)
class PixelMetadata:
    """Aggregate-safe properties of one decoded single-frame image."""

    rows: int
    columns: int
    source_min: float
    source_max: float
    rescaled_min: float
    rescaled_max: float
    photometric_interpretation: str
    display_inverted: bool
    midpoint_column: int
    left_half_columns: int
    right_half_columns: int
    pixel_bytes: int


def _single_value(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def split_bilateral_midpoint(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a two-dimensional image at its horizontal midpoint without dropping columns."""

    if array.ndim != 2:
        raise PixelInspectionError("Midpoint splitting requires a two-dimensional image")
    if array.shape[1] < 2:
        raise PixelInspectionError("Image is too narrow for midpoint splitting")
    midpoint = array.shape[1] // 2
    return array[:, :midpoint], array[:, midpoint:]


def normalize_for_display(
    array: np.ndarray,
    photometric_interpretation: str,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Return an 8-bit display image, including required MONOCHROME1 inversion."""

    if array.ndim != 2 or array.size == 0:
        raise PixelInspectionError("Display normalization requires a non-empty 2D image")
    interpretation = str(photometric_interpretation).upper()
    if interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
        raise PixelInspectionError("Only MONOCHROME1 or MONOCHROME2 is supported")
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise PixelInspectionError("Pixel array contains no finite values")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        output = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
        output = np.rint(scaled * 255).astype(np.uint8)
    return 255 - output if interpretation == "MONOCHROME1" else output


def inspect_pixels(path: str | Path) -> tuple[PixelMetadata, np.ndarray]:
    """Decode a single-frame DICOM and return aggregate-safe metadata plus display pixels."""

    try:
        dataset = pydicom.dcmread(path, force=False)
        frames = int(dataset.get("NumberOfFrames", 1))
        if frames != 1:
            raise PixelInspectionError("Pilot analysis requires exactly one frame per DICOM")
        source = np.asarray(dataset.pixel_array)
    except PixelInspectionError:
        raise
    except (OSError, InvalidDicomError, AttributeError, TypeError, ValueError) as error:
        raise PixelInspectionError("Input pixel data could not be decoded") from error
    if source.ndim != 2:
        raise PixelInspectionError("Pilot analysis requires a two-dimensional pixel array")
    interpretation = str(dataset.get("PhotometricInterpretation", "")).upper()
    slope = _single_value(dataset.get("RescaleSlope"), 1.0)
    intercept = _single_value(dataset.get("RescaleIntercept"), 0.0)
    rescaled = source.astype(np.float64) * slope + intercept
    left, right = split_bilateral_midpoint(source)
    display = normalize_for_display(rescaled, interpretation)
    metadata = PixelMetadata(
        rows=int(source.shape[0]),
        columns=int(source.shape[1]),
        source_min=float(source.min()),
        source_max=float(source.max()),
        rescaled_min=float(rescaled.min()),
        rescaled_max=float(rescaled.max()),
        photometric_interpretation=interpretation,
        display_inverted=interpretation == "MONOCHROME1",
        midpoint_column=int(source.shape[1] // 2),
        left_half_columns=int(left.shape[1]),
        right_half_columns=int(right.shape[1]),
        pixel_bytes=int(source.nbytes),
    )
    return metadata, display


def summarize_pixels(records: Sequence[PixelMetadata], *, failures: int = 0) -> dict[str, Any]:
    """Aggregate decoded-pixel facts without returning file-level values."""

    def distribution(values: Sequence[Any]) -> dict[str, int]:
        return {str(key): value for key, value in sorted(Counter(values).items())}

    pixel_bytes = [record.pixel_bytes for record in records]
    return {
        "files_considered": len(records) + failures,
        "decoded_single_frame_images": len(records),
        "pixel_decode_failures": failures,
        "dimensions": distribution([f"{r.rows}x{r.columns}" for r in records]),
        "photometric_interpretation": distribution(
            [record.photometric_interpretation for record in records]
        ),
        "display_inversion_required": sum(record.display_inverted for record in records),
        "source_intensity_minimum": min((record.source_min for record in records), default=None),
        "source_intensity_maximum": max((record.source_max for record in records), default=None),
        "rescaled_intensity_minimum": min(
            (record.rescaled_min for record in records), default=None
        ),
        "rescaled_intensity_maximum": max(
            (record.rescaled_max for record in records), default=None
        ),
        "pixel_array_bytes": {
            "total": sum(pixel_bytes),
            "median": statistics.median(pixel_bytes) if pixel_bytes else None,
            "mean": statistics.fmean(pixel_bytes) if pixel_bytes else None,
            "maximum": max(pixel_bytes, default=None),
        },
        "privacy": {
            "file_paths_serialized": False,
            "identity_fields_serialized": False,
            "dates_serialized": False,
            "pixel_arrays_serialized": False,
        },
    }


def pixel_metadata_as_dict(metadata: PixelMetadata) -> dict[str, Any]:
    """Serialize only the aggregate-safe fields defined by :class:`PixelMetadata`."""

    return asdict(metadata)


def save_qc_preview(
    display: np.ndarray,
    destination: str | Path,
    *,
    pilot_index: int,
    label_prefix: str = "Pilot",
    max_width: int = 1400,
) -> None:
    """Save a de-identified PNG preview with a midpoint overlay.

    The visible label is a sequential pilot index only. The image is intended exclusively for an
    ignored, access-controlled local QC directory.
    """

    if display.ndim != 2 or display.dtype != np.uint8:
        raise PixelInspectionError("QC preview input must be a two-dimensional uint8 array")
    if pilot_index < 1:
        raise ValueError("Pilot index must be positive")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(display, mode="L")
    if image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image.width, image.height + 28), color="black")
    canvas.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(canvas)
    midpoint = image.width // 2
    draw.line((midpoint, 28, midpoint, canvas.height - 1), fill=(255, 0, 255), width=3)
    draw.text((8, 7), f"{label_prefix} {pilot_index:03d} | midpoint", fill="white")
    canvas.save(target, format="PNG", optimize=True)


def save_contact_sheet(
    preview_paths: Sequence[str | Path],
    destination: str | Path,
    *,
    columns: int = 4,
    cell_width: int = 400,
) -> None:
    """Create a local contact sheet from de-identified previews."""

    if columns < 1 or cell_width < 1:
        raise ValueError("Contact-sheet geometry must be positive")
    paths = [Path(path) for path in preview_paths]
    if not paths:
        raise ValueError("At least one preview is required")
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            ratio = cell_width / source.width
            size = (cell_width, max(1, round(source.height * ratio)))
            thumbnails.append(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(32, 32, 32))
    for index, image in enumerate(thumbnails):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", optimize=True)
