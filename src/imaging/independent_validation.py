"""Run the frozen Milestone 5D preprocessing on an independent local image sample.

The module never downloads data, changes a cohort, creates a modeling split, or trains a model.
Per-acquisition artifacts use anonymous sequential indices and remain below Git-ignored data
directories. Anatomical laterality is not assigned by this automatic stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

from imaging.dicom_inspect import DicomMetadata, inspect_dicom, summarize_metadata
from imaging.pixel_qc import normalize_for_display, save_contact_sheet, split_bilateral_midpoint
from imaging.preprocessing import (
    PreprocessingError,
    crop_around_center,
    localize_bilateral_tibiofemoral_joints,
    physical_crop_shape,
    resample_to_spacing,
    save_crop_preview,
    save_localization_preview,
)

EXPECTED_ACQUISITIONS = 128
FULL_COHORT_ACQUISITIONS = 3621
TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
PANEL_POSITIONS = ("screen_left", "screen_right")
OUTPUT_SHAPE = physical_crop_shape(
    (CROP_SIZE_MM, CROP_SIZE_MM), (TARGET_SPACING_MM, TARGET_SPACING_MM)
)

DEFAULT_SELECTION_MANIFEST = Path(
    "data/processed/manifests/v00_xray_validation128_manifest.parquet"
)
DEFAULT_ARCHIVE_MANIFEST = Path(
    "data/interim/oai_images/validation128_validation/manifests/source_archives.csv"
)
DEFAULT_EXTRACTION_ROOT = Path("data/interim/oai_images/validation128_extracted")
DEFAULT_OUTPUT_ROOT = Path("data/interim/oai_images/validation128_validation")


class IndependentValidationError(ValueError):
    """Raised when the frozen validation cannot proceed without ambiguity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".csv", dir=target.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".json", dir=target.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _single_extracted_file(directory: Path) -> Path:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise IndependentValidationError(
            "Each anonymous extraction directory must contain exactly one regular file"
        )
    return files[0]


def _spacing(metadata: DicomMetadata) -> tuple[float, float]:
    value = metadata.pixel_spacing or metadata.imager_pixel_spacing
    if value is None or len(value) != 2:
        raise IndependentValidationError("A DICOM has no usable two-dimensional pixel spacing")
    spacing = (float(value[0]), float(value[1]))
    if any(not np.isfinite(item) or item <= 0 for item in spacing):
        raise IndependentValidationError("A DICOM has invalid pixel spacing")
    return spacing


def _optional_float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _preview(array: np.ndarray, photometric: str) -> Image.Image:
    display = normalize_for_display(array, photometric)
    return Image.fromarray(display, mode="L").convert("RGB")


def _pipeline_preview(
    source_display: np.ndarray,
    overlay_paths: dict[str, Path],
    crops: dict[str, np.ndarray],
    photometric: str,
    destination: Path,
    *,
    validation_index: int,
) -> None:
    bilateral = Image.fromarray(source_display, mode="L").convert("RGB")
    bilateral.thumbnail((1200, 520), Image.Resampling.LANCZOS)
    source_canvas = Image.new("RGB", (1200, bilateral.height + 28), "black")
    source_canvas.paste(bilateral, ((1200 - bilateral.width) // 2, 28))
    source_draw = ImageDraw.Draw(source_canvas)
    midpoint = 600
    source_draw.line((midpoint, 28, midpoint, source_canvas.height), fill="magenta", width=3)
    source_draw.text(
        (8, 7),
        f"Validation {validation_index:03d} | bilateral + midpoint",
        fill="white",
    )

    overlays: list[Image.Image] = []
    for position in PANEL_POSITIONS:
        with Image.open(overlay_paths[position]) as image:
            rendered = image.convert("RGB")
            rendered.thumbnail((590, 650), Image.Resampling.LANCZOS)
            overlays.append(rendered)
    overlay_height = max(image.height for image in overlays)
    overlay_canvas = Image.new("RGB", (1200, overlay_height), (20, 20, 20))
    overlay_canvas.paste(overlays[0], (0, 0))
    overlay_canvas.paste(overlays[1], (610, 0))

    crop_images = [_preview(crops[position], photometric) for position in PANEL_POSITIONS]
    for image in crop_images:
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    crop_canvas = Image.new("RGB", (1200, 528), "black")
    crop_canvas.paste(crop_images[0], (90, 28))
    crop_canvas.paste(crop_images[1], (610, 28))
    crop_draw = ImageDraw.Draw(crop_canvas)
    crop_draw.text((90, 7), "screen-left standardized crop", fill="white")
    crop_draw.text((610, 7), "screen-right standardized crop", fill="white")

    final = Image.new(
        "RGB",
        (1200, source_canvas.height + overlay_canvas.height + crop_canvas.height),
        (20, 20, 20),
    )
    offset = 0
    for image in (source_canvas, overlay_canvas, crop_canvas):
        final.paste(image, (0, offset))
        offset += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, format="PNG", optimize=True)


def _review_templates(output_root: Path, indices: list[int]) -> None:
    acquisition_path = output_root / "manifests/acquisition_review.csv"
    panel_path = output_root / "manifests/panel_review.csv"
    if not acquisition_path.exists():
        _atomic_csv(
            pd.DataFrame(
                {
                    "validation_index": indices,
                    "midpoint_state": "",
                    "marker_evidence": "",
                    "laterality_state": "",
                    "laterality_reason": "",
                    "screen_left_anatomical_side": "",
                    "screen_right_anatomical_side": "",
                    "review_basis": "",
                }
            ),
            acquisition_path,
        )
    if not panel_path.exists():
        _atomic_csv(
            pd.DataFrame(
                [
                    {
                        "validation_index": index,
                        "panel_position": position,
                        "localization_state": "",
                        "crop_adequacy_state": "",
                        "crop_boundary_issue": "",
                        "incomplete_medial_compartment": "",
                        "incomplete_lateral_compartment": "",
                        "insufficient_femoral_context": "",
                        "insufficient_tibial_context": "",
                        "review_basis": "",
                    }
                    for index in indices
                    for position in PANEL_POSITIONS
                ]
            ),
            panel_path,
        )


def prepare_validation(
    *,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    archive_manifest_path: Path = DEFAULT_ARCHIVE_MANIFEST,
    extraction_root: Path = DEFAULT_EXTRACTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run the frozen automatic preprocessing and create anonymous review artifacts."""

    selection = pd.read_parquet(selection_manifest_path)
    archives = pd.read_csv(archive_manifest_path)
    required_selection = {
        "image_file",
        "development_pilot_overlap",
        "baseline_visit",
        "bilateral_acquisition",
    }
    if not required_selection.issubset(selection.columns):
        raise IndependentValidationError("Selection manifest lacks required audit columns")
    if len(selection) != EXPECTED_ACQUISITIONS or selection["image_file"].nunique() != 128:
        raise IndependentValidationError("Selection must contain 128 unique acquisitions")
    if selection["development_pilot_overlap"].astype(bool).any():
        raise IndependentValidationError("Selection overlaps the development pilot")
    if not selection["baseline_visit"].eq("V00").all():
        raise IndependentValidationError("Selection contains a non-V00 acquisition")
    if not selection["bilateral_acquisition"].astype(bool).all():
        raise IndependentValidationError("Selection contains a non-bilateral acquisition")
    if len(archives) != EXPECTED_ACQUISITIONS or archives["validation_index"].nunique() != 128:
        raise IndependentValidationError("Archive manifest must contain 128 unique indices")

    linked = archives.merge(
        selection,
        left_on="associated_file_reference",
        right_on="image_file",
        how="left",
        validate="one_to_one",
    )
    if linked["image_file"].isna().any():
        raise IndependentValidationError("An extracted archive is not in the frozen selection")
    linked = linked.sort_values("validation_index", kind="stable")
    if linked["validation_index"].tolist() != list(range(1, EXPECTED_ACQUISITIONS + 1)):
        raise IndependentValidationError("Anonymous validation indices are not exactly 1..128")

    metadata_records: list[DicomMetadata] = []
    metadata_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    automatic_failures = 0
    dicom_paths: list[Path] = []
    stage_previews: list[Path] = []
    for row in linked.itertuples(index=False):
        index = int(row.validation_index)
        dicom_path = _single_extracted_file(extraction_root / f"acquisition_{index:03d}")
        dicom_paths.append(dicom_path)
        metadata = inspect_dicom(dicom_path)
        metadata_records.append(metadata)
        spacing = _spacing(metadata)
        metadata_rows.append(
            {
                "validation_index": index,
                "dicom_sha256": _sha256(dicom_path),
                **{
                    key: value
                    for key, value in asdict(metadata).items()
                    if key != "file_size_bytes"
                },
                "dicom_file_bytes": metadata.file_size_bytes,
            }
        )

        dataset = pydicom.dcmread(dicom_path, force=False)
        frames = int(dataset.get("NumberOfFrames", 1))
        if frames != 1:
            raise IndependentValidationError("A validation DICOM is not single-frame")
        source = np.asarray(dataset.pixel_array)
        if source.ndim != 2:
            raise IndependentValidationError("A validation DICOM is not two-dimensional")
        if source.dtype != np.dtype("uint16"):
            raise IndependentValidationError("Frozen uint16 processing cannot accept source dtype")
        photometric = str(dataset.get("PhotometricInterpretation", "")).upper()
        slope = _optional_float(dataset.get("RescaleSlope"), 1.0)
        intercept = _optional_float(dataset.get("RescaleIntercept"), 0.0)
        display = normalize_for_display(source.astype(np.float64) * slope + intercept, photometric)
        screen_left, screen_right = split_bilateral_midpoint(source)
        panels = {"screen_left": screen_left, "screen_right": screen_right}

        try:
            left_location, right_location = localize_bilateral_tibiofemoral_joints(
                screen_left, screen_right
            )
        except PreprocessingError:
            automatic_failures += 2
            for position, panel in panels.items():
                panel_rows.append(
                    {
                        "validation_index": index,
                        "panel_position": position,
                        "source_rows": panel.shape[0],
                        "source_columns": panel.shape[1],
                        "source_row_spacing_mm": spacing[0],
                        "source_column_spacing_mm": spacing[1],
                        "automatic_localization": "FAILED",
                        "crop_materialized": False,
                    }
                )
            continue
        locations = {"screen_left": left_location, "screen_right": right_location}
        overlays: dict[str, Path] = {}
        crops: dict[str, np.ndarray] = {}
        for position, panel in panels.items():
            location = locations[position]
            resampled = resample_to_spacing(panel, spacing, (TARGET_SPACING_MM, TARGET_SPACING_MM))
            center = (
                round(location.row * spacing[0] / TARGET_SPACING_MM),
                round(location.column * spacing[1] / TARGET_SPACING_MM),
            )
            crop, geometry = crop_around_center(
                resampled,
                center,
                OUTPUT_SHAPE,
                padding_value=int(np.percentile(resampled, 0.5)),
            )
            if crop.dtype != np.dtype("uint16"):
                raise IndependentValidationError("A standardized crop did not remain uint16")
            crop_dir = output_root / "crops" / f"acquisition_{index:03d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{position}_uint16.npy"
            np.save(crop_path, crop, allow_pickle=False)
            overlay_path = (
                output_root / "localization_overlays" / (f"validation_{index:03d}_{position}.png")
            )
            save_localization_preview(
                panel,
                location,
                physical_crop_shape((CROP_SIZE_MM, CROP_SIZE_MM), spacing),
                overlay_path,
                pilot_index=index,
                panel_position=position,
                label_prefix="Validation",
            )
            crop_preview = (
                output_root / "crop_previews" / (f"validation_{index:03d}_{position}.png")
            )
            save_crop_preview(
                crop,
                crop_preview,
                pilot_index=index,
                panel_position=position,
                label_prefix="Validation",
            )
            overlays[position] = overlay_path
            crops[position] = crop
            panel_rows.append(
                {
                    "validation_index": index,
                    "panel_position": position,
                    "source_rows": panel.shape[0],
                    "source_columns": panel.shape[1],
                    "source_row_spacing_mm": spacing[0],
                    "source_column_spacing_mm": spacing[1],
                    "joint_row": location.row,
                    "joint_column": location.column,
                    "projection_score": location.score,
                    "paired_projection_robust_score": location.robust_score,
                    "resampled_rows": resampled.shape[0],
                    "resampled_columns": resampled.shape[1],
                    "target_spacing_mm": TARGET_SPACING_MM,
                    "crop_size_mm": CROP_SIZE_MM,
                    "crop_rows": crop.shape[0],
                    "crop_columns": crop.shape[1],
                    "padding_top": geometry.padding_top,
                    "padding_bottom": geometry.padding_bottom,
                    "padding_left": geometry.padding_left,
                    "padding_right": geometry.padding_right,
                    "padding_used": geometry.used_padding,
                    "dtype": str(crop.dtype),
                    "automatic_localization": "SUCCEEDED",
                    "crop_materialized": True,
                }
            )
        stage_path = output_root / "stage_previews" / f"validation_{index:03d}.png"
        _pipeline_preview(
            display,
            overlays,
            crops,
            photometric,
            stage_path,
            validation_index=index,
        )
        stage_previews.append(stage_path)

    safe_metadata = pd.DataFrame(metadata_rows)
    panels = pd.DataFrame(panel_rows).sort_values(
        ["validation_index", "panel_position"], kind="stable"
    )
    _atomic_csv(safe_metadata, output_root / "manifests/dicom_metadata.csv")
    _atomic_csv(panels, output_root / "manifests/preprocessing_manifest.csv")
    _review_templates(output_root, linked["validation_index"].astype(int).tolist())

    contact_sheets: list[Path] = []
    for number, start in enumerate(range(0, len(stage_previews), 4), start=1):
        target = output_root / "contact_sheets" / f"validation_contact_{number:02d}.png"
        save_contact_sheet(stage_previews[start : start + 4], target, columns=2, cell_width=600)
        contact_sheets.append(target)

    crop_paths = list((output_root / "crops").rglob("*.npy"))
    crop_sizes = [path.stat().st_size for path in crop_paths]
    dicom_sizes = [path.stat().st_size for path in dicom_paths]
    summary = {
        "validation_acquisitions": EXPECTED_ACQUISITIONS,
        "development_pilot_overlap": 0,
        "valid_dicoms": len(metadata_records),
        "corrupt_dicoms": 0,
        "automatic_localization_succeeded_panels": int(
            panels["automatic_localization"].eq("SUCCEEDED").sum()
        ),
        "automatic_localization_failed_panels": automatic_failures,
        "standardized_crops": len(crop_paths),
        "standardized_shape": list(OUTPUT_SHAPE),
        "standardized_dtype": sorted(panels.get("dtype", pd.Series(dtype=str)).dropna().unique()),
        "target_spacing_mm": TARGET_SPACING_MM,
        "physical_crop_size_mm": CROP_SIZE_MM,
        "padding_used_panels": int(panels.get("padding_used", pd.Series(dtype=bool)).sum()),
        "dicom_metadata": summarize_metadata(metadata_records),
        "storage": {
            "extracted_dicom_total_bytes": sum(dicom_sizes),
            "extracted_dicom_median_bytes": statistics.median(dicom_sizes),
            "extracted_dicom_mean_bytes": statistics.fmean(dicom_sizes),
            "extracted_dicom_maximum_bytes": max(dicom_sizes),
            "uint16_crop_total_bytes": sum(crop_sizes),
            "uint16_crop_median_bytes": statistics.median(crop_sizes),
            "projected_full_cohort_extracted_dicom_bytes": round(
                statistics.fmean(dicom_sizes) * FULL_COHORT_ACQUISITIONS
            ),
            "projected_full_cohort_uint16_crop_bytes": round(
                statistics.fmean(crop_sizes) * FULL_COHORT_ACQUISITIONS * 2
            ),
            "projected_full_cohort_uint8_crop_bytes": OUTPUT_SHAPE[0]
            * OUTPUT_SHAPE[1]
            * FULL_COHORT_ACQUISITIONS
            * 2,
        },
        "contact_sheets": len(contact_sheets),
        "frozen_parameters": {
            "midpoint_split": True,
            "target_spacing_mm": TARGET_SPACING_MM,
            "interpolation": "bilinear",
            "physical_crop_mm": [CROP_SIZE_MM, CROP_SIZE_MM],
            "joint_localizer": "milestone_5c_paired_projection",
            "crop_dtype": "uint16",
            "future_model_normalization": "clip_0.5_99.5_then_minmax_0_1",
            "clahe": False,
        },
        "privacy": {
            "identifiers_in_aggregate_summary": False,
            "anonymous_sequential_preview_labels": True,
            "anatomical_side_assigned_automatically": False,
        },
    }
    _atomic_json(summary, output_root / "manifests/automatic_summary.json")
    return summary


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 3) if denominator else 0.0


def _pooled_groups(values: pd.Series, minimum_size: int = 5) -> pd.Series:
    text = values.fillna("<missing>").astype(str).replace("", "<missing>")
    counts = text.value_counts()
    rare = set(counts[counts < minimum_size].index)
    return text.map(lambda value: "<pooled groups n<5>" if value in rare else value)


def _heterogeneity_summary(acquisitions: pd.DataFrame, panels: pd.DataFrame) -> dict[str, Any]:
    dimensions = (
        acquisitions["rows"].astype(int).astype(str)
        + "x"
        + acquisitions["columns"].astype(int).astype(str)
    )
    groups = {
        "manufacturer": acquisitions["manufacturer"],
        "scanner_model": acquisitions["manufacturer_model_name"],
        "pixel_spacing_mm": acquisitions["source_spacing_mm"].map(lambda value: f"{value:g}"),
        "image_dimensions": dimensions,
        "image_release": acquisitions["image03_study"],
        "qc_acceptance": acquisitions["xray_accept_qc"],
        "qc_problem_signature": acquisitions["qc_problem_signature"],
    }
    output: dict[str, Any] = {}
    for group_name, raw_values in groups.items():
        grouped_values = _pooled_groups(raw_values)
        entries: list[dict[str, Any]] = []
        for value in sorted(grouped_values.unique()):
            indices = set(
                acquisitions.loc[grouped_values.eq(value), "validation_index"].astype(int)
            )
            acquisition_subset = acquisitions.loc[acquisitions["validation_index"].isin(indices)]
            panel_subset = panels.loc[panels["validation_index"].isin(indices)]
            midpoint_success = int(acquisition_subset["midpoint_state"].eq("CLEAN").sum())
            localization_success = int(panel_subset["localization_state"].eq("SUCCESS").sum())
            crop_success = int(panel_subset["crop_adequacy_state"].eq("ADEQUATE").sum())
            entries.append(
                {
                    "group": value,
                    "acquisitions": len(acquisition_subset),
                    "panels": len(panel_subset),
                    "midpoint_clean_percent": _percent(midpoint_success, len(acquisition_subset)),
                    "localization_success_percent": _percent(
                        localization_success, len(panel_subset)
                    ),
                    "crop_adequacy_percent": _percent(crop_success, len(panel_subset)),
                    "laterality_confident": int(
                        acquisition_subset["laterality_state"].eq("CONFIDENT").sum()
                    ),
                    "laterality_ambiguous": int(
                        acquisition_subset["laterality_state"].eq("AMBIGUOUS").sum()
                    ),
                    "laterality_conflicting": int(
                        acquisition_subset["laterality_state"].eq("CONFLICTING").sum()
                    ),
                }
            )
        output[group_name] = entries
    return output


def summarize_reviewed_validation(
    *,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    archive_manifest_path: Path = DEFAULT_ARCHIVE_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate local visual-review files and return aggregate acceptance results."""

    acquisitions_review = pd.read_csv(
        output_root / "manifests/acquisition_review.csv", keep_default_na=False
    )
    panels_review = pd.read_csv(output_root / "manifests/panel_review.csv", keep_default_na=False)
    preprocessing = pd.read_csv(output_root / "manifests/preprocessing_manifest.csv")
    metadata = pd.read_csv(output_root / "manifests/dicom_metadata.csv", keep_default_na=False)
    selection = pd.read_parquet(selection_manifest_path)
    archives = pd.read_csv(archive_manifest_path)
    extraction = json.loads(
        (output_root / "manifests/extraction_summary.json").read_text(encoding="utf-8")
    )

    expected_indices = set(range(1, EXPECTED_ACQUISITIONS + 1))
    if (
        set(acquisitions_review["validation_index"]) != expected_indices
        or acquisitions_review["validation_index"].duplicated().any()
    ):
        raise IndependentValidationError("Acquisition review must contain indices 1..128 once")
    expected_panel_keys = {
        (index, position) for index in expected_indices for position in PANEL_POSITIONS
    }
    actual_panel_keys = set(
        panels_review[["validation_index", "panel_position"]].itertuples(index=False, name=None)
    )
    if (
        actual_panel_keys != expected_panel_keys
        or panels_review.duplicated(["validation_index", "panel_position"]).any()
    ):
        raise IndependentValidationError("Panel review must contain both positions for 1..128")

    allowed_acquisition = {
        "midpoint_state": {"CLEAN", "BORDERLINE", "FAILED"},
        "laterality_state": {"CONFIDENT", "AMBIGUOUS", "CONFLICTING"},
    }
    allowed_panel = {
        "localization_state": {"SUCCESS", "BORDERLINE", "FAILED"},
        "crop_adequacy_state": {"ADEQUATE", "BORDERLINE", "INADEQUATE"},
    }
    for column, allowed in allowed_acquisition.items():
        if set(acquisitions_review[column]) - allowed:
            raise IndependentValidationError(f"Acquisition review has invalid {column}")
    for column, allowed in allowed_panel.items():
        if set(panels_review[column]) - allowed:
            raise IndependentValidationError(f"Panel review has invalid {column}")

    unresolved = acquisitions_review["laterality_state"].isin(["AMBIGUOUS", "CONFLICTING"])
    side_columns = ["screen_left_anatomical_side", "screen_right_anatomical_side"]
    if acquisitions_review.loc[unresolved, side_columns].ne("").any(axis=None):
        raise IndependentValidationError("Unresolved laterality rows contain an anatomical side")
    confident = acquisitions_review["laterality_state"].eq("CONFIDENT")
    confident_sides = acquisitions_review.loc[confident, side_columns]
    if (
        not confident_sides["screen_left_anatomical_side"].isin(["R", "L"]).all()
        or not confident_sides["screen_right_anatomical_side"].isin(["R", "L"]).all()
        or confident_sides["screen_left_anatomical_side"]
        .eq(confident_sides["screen_right_anatomical_side"])
        .any()
    ):
        raise IndependentValidationError("A confident mapping is incomplete or invalid")

    boolean_columns = [
        "crop_boundary_issue",
        "incomplete_medial_compartment",
        "incomplete_lateral_compartment",
        "insufficient_femoral_context",
        "insufficient_tibial_context",
    ]
    for column in boolean_columns:
        normalized = panels_review[column].map(lambda value: str(value).lower())
        if set(normalized) - {"true", "false"}:
            raise IndependentValidationError(f"Panel review has invalid Boolean field {column}")
        panels_review[column] = normalized.eq("true")

    panels = preprocessing.merge(
        panels_review,
        on=["validation_index", "panel_position"],
        how="inner",
        validate="one_to_one",
    )
    archive_link = archives[["validation_index", "associated_file_reference"]].merge(
        selection[
            [
                "image_file",
                "image03_study",
                "xray_accept_qc",
                "qc_problem_signature",
            ]
        ],
        left_on="associated_file_reference",
        right_on="image_file",
        how="left",
        validate="one_to_one",
    )
    safe_metadata = metadata[
        [
            "validation_index",
            "rows",
            "columns",
            "manufacturer",
            "manufacturer_model_name",
        ]
    ]
    acquisition = (
        acquisitions_review.merge(safe_metadata, on="validation_index", validate="one_to_one")
        .merge(
            archive_link[
                [
                    "validation_index",
                    "image03_study",
                    "xray_accept_qc",
                    "qc_problem_signature",
                ]
            ],
            on="validation_index",
            validate="one_to_one",
        )
        .merge(
            panels.groupby("validation_index", as_index=False)["source_row_spacing_mm"]
            .first()
            .rename(columns={"source_row_spacing_mm": "source_spacing_mm"}),
            on="validation_index",
            validate="one_to_one",
        )
    )

    midpoint_clean = int(acquisition["midpoint_state"].eq("CLEAN").sum())
    localization_success = int(panels["localization_state"].eq("SUCCESS").sum())
    crop_adequate = int(panels["crop_adequacy_state"].eq("ADEQUATE").sum())
    horizontal_padding = int(
        panels[["padding_left", "padding_right"]].fillna(0).sum(axis=1).gt(0).sum()
    )
    vertical_padding = int(
        panels[["padding_top", "padding_bottom"]].fillna(0).sum(axis=1).gt(0).sum()
    )
    standard_mapping = int(
        (
            confident
            & acquisitions_review["screen_left_anatomical_side"].eq("R")
            & acquisitions_review["screen_right_anatomical_side"].eq("L")
        ).sum()
    )
    reversed_mapping = int(
        (
            confident
            & acquisitions_review["screen_left_anatomical_side"].eq("L")
            & acquisitions_review["screen_right_anatomical_side"].eq("R")
        ).sum()
    )

    archive_sizes = archives["archive_bytes"].astype(int).tolist()
    dicom_sizes = metadata["dicom_file_bytes"].astype(int).tolist()
    crop_paths = list((output_root / "crops").rglob("*.npy"))
    crop_sizes = [path.stat().st_size for path in crop_paths]
    heterogeneity = _heterogeneity_summary(acquisition, panels)
    subgroup_preprocessing_failures = sum(
        entry["midpoint_clean_percent"] < 100
        or entry["localization_success_percent"] < 100
        or entry["crop_adequacy_percent"] < 100
        for entries in heterogeneity.values()
        for entry in entries
    )

    integrity_pass = (
        extraction["archives_extracted"] == EXPECTED_ACQUISITIONS
        and extraction["extraction_failures"] == 0
        and extraction["unsafe_members"] == 0
        and len(metadata) == EXPECTED_ACQUISITIONS
    )
    midpoint_pass = _percent(midpoint_clean, len(acquisition)) >= 99.0
    localization_pass = _percent(localization_success, len(panels)) >= 98.0
    crop_pass = _percent(crop_adequate, len(panels)) >= 98.0
    laterality_pass = not acquisitions_review.loc[unresolved, side_columns].ne("").any(axis=None)
    scanner_pass = subgroup_preprocessing_failures == 0

    summary = {
        "sample": {
            "acquisitions": len(acquisition),
            "panels": len(panels),
            "development_pilot_overlap": 0,
        },
        "integrity": {
            "archives_retrieved": extraction["archives_considered"],
            "archives_extracted": extraction["archives_extracted"],
            "unsafe_members": extraction["unsafe_members"],
            "extraction_failures": extraction["extraction_failures"],
            "valid_dicoms": len(metadata),
            "corrupt_dicoms": 0,
        },
        "midpoint": {
            "clean": midpoint_clean,
            "borderline": int(acquisition["midpoint_state"].eq("BORDERLINE").sum()),
            "failed": int(acquisition["midpoint_state"].eq("FAILED").sum()),
            "clean_percent": _percent(midpoint_clean, len(acquisition)),
        },
        "laterality": {
            "states": {
                key: int(value)
                for key, value in acquisition["laterality_state"].value_counts().items()
            },
            "unresolved_exceptions": int(unresolved.sum()),
            "confident_standard_mapping": standard_mapping,
            "confident_reversed_mapping": reversed_mapping,
            "unresolved_sides_forced": 0,
        },
        "localization": {
            "successful": localization_success,
            "borderline": int(panels["localization_state"].eq("BORDERLINE").sum()),
            "failed": int(panels["localization_state"].eq("FAILED").sum()),
            "success_percent": _percent(localization_success, len(panels)),
        },
        "crop_adequacy": {
            "adequate": crop_adequate,
            "borderline": int(panels["crop_adequacy_state"].eq("BORDERLINE").sum()),
            "inadequate": int(panels["crop_adequacy_state"].eq("INADEQUATE").sum()),
            "adequate_percent": _percent(crop_adequate, len(panels)),
            **{column: int(panels[column].sum()) for column in boolean_columns},
        },
        "padding": {
            "horizontal_panels": horizontal_padding,
            "vertical_panels": vertical_padding,
            "any_padding_panels": int(panels["padding_used"].fillna(False).sum()),
            "maximum_horizontal_padding_mm": round(
                float(panels[["padding_left", "padding_right"]].fillna(0).to_numpy().max())
                * TARGET_SPACING_MM,
                3,
            ),
            "maximum_vertical_padding_mm": round(
                float(panels[["padding_top", "padding_bottom"]].fillna(0).to_numpy().max())
                * TARGET_SPACING_MM,
                3,
            ),
        },
        "heterogeneity": heterogeneity,
        "storage": {
            "validation_archive_total_bytes": sum(archive_sizes),
            "archive_median_bytes": statistics.median(archive_sizes),
            "archive_mean_bytes": statistics.fmean(archive_sizes),
            "archive_maximum_bytes": max(archive_sizes),
            "extracted_dicom_total_bytes": sum(dicom_sizes),
            "dicom_median_bytes": statistics.median(dicom_sizes),
            "dicom_mean_bytes": statistics.fmean(dicom_sizes),
            "dicom_maximum_bytes": max(dicom_sizes),
            "uint16_crop_total_bytes": sum(crop_sizes),
            "uint16_crop_median_bytes": statistics.median(crop_sizes),
            "projected_3621_archive_bytes": round(
                statistics.fmean(archive_sizes) * FULL_COHORT_ACQUISITIONS
            ),
            "projected_3621_extracted_dicom_bytes": round(
                statistics.fmean(dicom_sizes) * FULL_COHORT_ACQUISITIONS
            ),
            "projected_3621_uint16_1067_crop_bytes": round(
                statistics.fmean(crop_sizes) * FULL_COHORT_ACQUISITIONS * 2
            ),
            "projected_3621_uint8_1067_crop_bytes": OUTPUT_SHAPE[0]
            * OUTPUT_SHAPE[1]
            * FULL_COHORT_ACQUISITIONS
            * 2,
            "projected_3621_uint8_512_cache_bytes": 512 * 512 * FULL_COHORT_ACQUISITIONS * 2,
        },
        "acceptance": {
            "archive_and_dicom_integrity": integrity_pass,
            "midpoint_at_least_99_percent": midpoint_pass,
            "localization_at_least_98_percent": localization_pass,
            "crop_adequacy_at_least_98_percent": crop_pass,
            "no_forced_uncertain_laterality": laterality_pass,
            "no_observed_systematic_subgroup_preprocessing_failure": scanner_pass,
            "all_predefined_criteria_met": all(
                (
                    integrity_pass,
                    midpoint_pass,
                    localization_pass,
                    crop_pass,
                    laterality_pass,
                    scanner_pass,
                )
            ),
        },
        "privacy": {
            "identifiers_in_aggregate_summary": False,
            "accessions_in_aggregate_summary": False,
            "paths_in_aggregate_summary": False,
        },
    }
    _atomic_json(summary, output_root / "manifests/reviewed_summary.json")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen preprocessing for the independent 128-acquisition sample."
    )
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--archive-manifest", type=Path, default=DEFAULT_ARCHIVE_MANIFEST)
    parser.add_argument("--extraction-root", type=Path, default=DEFAULT_EXTRACTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = prepare_validation(
        selection_manifest_path=args.selection_manifest,
        archive_manifest_path=args.archive_manifest,
        extraction_root=args.extraction_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
