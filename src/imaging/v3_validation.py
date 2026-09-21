"""Sample-only runner and anonymous QC outputs for localization V3."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

from imaging.archive_extract import safe_extract_archive
from imaging.localization_v3 import (
    JointLocalizationV3,
    QCState,
    localize_bilateral_tibiofemoral_joints_v3,
)
from imaging.pixel_qc import normalize_for_display, save_contact_sheet, split_bilateral_midpoint
from imaging.preprocessing import crop_around_center, resample_to_spacing, robust_minmax

TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
OUTPUT_SHAPE = (1067, 1067)
LOCALIZER_VERSION = "localization_v3_candidate_r1"
PANEL_POSITIONS = ("screen_left", "screen_right")
DEFAULT_ARCHIVE_ROOT = Path("data/raw/oai_images/full")
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/localization_v3_candidate")


class V3ValidationError(ValueError):
    """Raised when sample-only V3 processing cannot proceed safely."""


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=".parquet", dir=target.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
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


def _single_file(root: Path) -> Path:
    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise V3ValidationError("Sample archive does not contain exactly one DICOM candidate")
    return files[0]


def _serialize_result(result: JointLocalizationV3) -> dict[str, Any]:
    return {
        "joint_row_resampled": result.row,
        "joint_column_resampled": result.column,
        "confidence": result.confidence,
        "qc_state": result.qc_state.value,
        "localization_state": result.localization_state.value,
        "layout_state": result.layout.state.value,
        "layout_reasons": "|".join(result.layout.reasons),
        "layout_blank_block_fraction": result.layout.blank_block_fraction,
        "layout_usable_width_fraction": result.layout.usable_width_fraction,
        "layout_usable_height_fraction": result.layout.usable_height_fraction,
        "layout_column_boundary_z": result.layout.strongest_column_boundary_z,
        "layout_row_boundary_z": result.layout.strongest_row_boundary_z,
        "anatomy_state": result.anatomy.state.value,
        "anatomy_reasons": "|".join(result.anatomy.reasons),
        "anatomy_score": result.anatomy.score,
        "anatomy_blank_block_fraction": result.anatomy.blank_block_fraction,
        "joint_band_edge_ratio": result.anatomy.joint_band_edge_ratio,
        "superior_structure": result.anatomy.superior_structure,
        "inferior_structure": result.anatomy.inferior_structure,
        "horizontal_joint_support": result.anatomy.horizontal_joint_support,
        "central_structure_fraction": result.anatomy.central_structure_fraction,
        "collimation_dominance": result.anatomy.collimation_dominance,
        "candidate_count": result.candidate_count,
        "candidate_margin": result.candidate_margin,
        "pair_row_delta_mm": result.pair_row_delta_mm,
        "selected_candidate_score": result.selected_candidate_score,
        "vertical_peak_z": result.vertical_peak_z,
        "vertical_prominence_z": result.vertical_prominence_z,
        "horizontal_support": result.horizontal_support,
        "row_fraction": result.row_fraction,
        "column_fraction": result.column_fraction,
    }


def _render_preview(
    source: np.ndarray,
    *,
    photometric: str,
    spacing: tuple[float, float],
    results: dict[str, JointLocalizationV3],
    crops: dict[str, np.ndarray],
    sample_number: int,
    destination: Path,
) -> None:
    display = normalize_for_display(source, photometric)
    image = Image.fromarray(display, mode="L").convert("RGB")
    scale = min(1.0, 1200 / image.width)
    if scale < 1:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    header = 30
    original = Image.new("RGB", (1200, image.height + header), "black")
    x_offset = (1200 - image.width) // 2
    original.paste(image, (x_offset, header))
    draw = ImageDraw.Draw(original)
    midpoint = source.shape[1] // 2
    draw.line(
        (
            x_offset + round(midpoint * scale),
            header,
            x_offset + round(midpoint * scale),
            original.height,
        ),
        fill="magenta",
        width=3,
    )
    draw.text((8, 8), f"V3 anonymous QC {sample_number:03d}", fill="white")
    colors = {QCState.PASS: "lime", QCState.BORDERLINE: "yellow", QCState.FAIL: "red"}
    for position in PANEL_POSITIONS:
        result = results[position]
        panel_offset = 0 if position == "screen_left" else midpoint
        source_row = result.row * TARGET_SPACING_MM / spacing[0]
        source_column = result.column * TARGET_SPACING_MM / spacing[1]
        x = x_offset + round((panel_offset + source_column) * scale)
        y = header + round(source_row * scale)
        half_height = round((CROP_SIZE_MM / spacing[0] / 2) * scale)
        half_width = round((CROP_SIZE_MM / spacing[1] / 2) * scale)
        color = colors[result.qc_state]
        draw.rectangle(
            (x - half_width, y - half_height, x + half_width, y + half_height),
            outline=color,
            width=3,
        )
        draw.line((x - 12, y, x + 12, y), fill=color, width=3)
        draw.line((x, y - 12, x, y + 12), fill=color, width=3)

    crop_canvas = Image.new("RGB", (1200, 540), "black")
    crop_draw = ImageDraw.Draw(crop_canvas)
    for number, position in enumerate(PANEL_POSITIONS):
        crop_display = np.rint(
            robust_minmax(crops[position], lower_percentile=0.5, upper_percentile=99.5) * 255
        ).astype(np.uint8)
        crop_image = Image.fromarray(crop_display, mode="L").convert("RGB")
        crop_image = crop_image.resize((500, 500), Image.Resampling.LANCZOS)
        x = 90 if number == 0 else 610
        crop_canvas.paste(crop_image, (x, 34))
        result = results[position]
        crop_draw.text(
            (x, 10),
            f"{position.replace('_', '-')} | {result.qc_state.value} | anatomy {result.anatomy.state.value}",
            fill=colors[result.qc_state],
        )
    final = Image.new("RGB", (1200, original.height + crop_canvas.height), "black")
    final.paste(original, (0, 0))
    final.paste(crop_canvas, (0, original.height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, format="PNG", optimize=True)


def process_v3_sample(
    *,
    manifest_path: Path,
    sample_name: str,
    localizer_version: str = LOCALIZER_VERSION,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    output_base: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = Path("data/interim/oai_images/v3_work"),
    save_crops: bool = True,
) -> dict[str, Any]:
    """Run V3 on a development or sealed holdout manifest without altering source artifacts."""

    output_root = output_base / sample_name / localizer_version
    if output_root.exists():
        raise V3ValidationError("V3 sample output already exists and will not be overwritten")
    sample = pd.read_parquet(manifest_path).sort_values("acquisition_index", kind="stable")
    if sample["acquisition_index"].duplicated().any():
        raise V3ValidationError("V3 methodology manifest has duplicate acquisitions")
    if sample.get("selection_outcomes_read", pd.Series([True])).astype(bool).any():
        raise V3ValidationError("V3 methodology selection is not outcome-blind")
    work_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    previews: list[Path] = []
    for sample_number, acquisition in enumerate(sample.itertuples(index=False), start=1):
        label = f"sample_{sample_number:03d}"
        extraction = work_root / f"{sample_name}_{sample_number:03d}"
        if extraction.exists():
            raise V3ValidationError("A previous V3 temporary extraction exists")
        archive_path = archive_root / str(acquisition.associated_file_reference)
        try:
            contents = safe_extract_archive(archive_path, extraction)
            if contents.regular_files != 1:
                raise V3ValidationError("Unexpected V3 archive structure")
            dataset = pydicom.dcmread(_single_file(extraction), force=False)
            source = np.asarray(dataset.pixel_array)
            if source.ndim != 2 or source.dtype != np.dtype("uint16"):
                raise V3ValidationError("V3 source pixels are not single-frame uint16")
            spacing = (
                float(acquisition.original_row_spacing_mm),
                float(acquisition.original_column_spacing_mm),
            )
            panels = dict(zip(PANEL_POSITIONS, split_bilateral_midpoint(source), strict=True))
            resampled = {
                position: resample_to_spacing(
                    panel, spacing, (TARGET_SPACING_MM, TARGET_SPACING_MM)
                )
                for position, panel in panels.items()
            }
            left, right = localize_bilateral_tibiofemoral_joints_v3(
                resampled["screen_left"], resampled["screen_right"], spacing_mm=TARGET_SPACING_MM
            )
            results = {"screen_left": left, "screen_right": right}
            crops: dict[str, np.ndarray] = {}
            for position in PANEL_POSITIONS:
                result = results[position]
                padding_value = int(np.percentile(resampled[position], 0.5))
                crop, geometry = crop_around_center(
                    resampled[position],
                    (result.row, result.column),
                    OUTPUT_SHAPE,
                    padding_value=padding_value,
                )
                if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                    raise V3ValidationError("V3 candidate crop shape or dtype is invalid")
                crops[position] = crop
                crop_path = output_root / "crops" / label / f"{position}_uint16.npy"
                if save_crops:
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(crop_path, crop, allow_pickle=False)
                record = {
                    "sample_number": sample_number,
                    "acquisition_index": int(acquisition.acquisition_index),
                    "methodology_role": acquisition.methodology_role,
                    "panel_position": position,
                    "localizer_version": localizer_version,
                    **_serialize_result(result),
                    "padding_top": geometry.padding_top,
                    "padding_bottom": geometry.padding_bottom,
                    "padding_left": geometry.padding_left,
                    "padding_right": geometry.padding_right,
                    "vertical_padding_mm": (geometry.padding_top + geometry.padding_bottom)
                    * TARGET_SPACING_MM,
                    "horizontal_padding_mm": (geometry.padding_left + geometry.padding_right)
                    * TARGET_SPACING_MM,
                    "original_rows": int(source.shape[0]),
                    "original_columns": int(source.shape[1]),
                    "original_spacing_mm": spacing[0],
                    "manufacturer": acquisition.manufacturer,
                    "scanner_model": acquisition.manufacturer_model_name,
                    "image_release": acquisition.image_release_study,
                    "oai_qc_category": acquisition.xray_accept_qc,
                    "crop_path": crop_path.relative_to(output_base).as_posix()
                    if save_crops
                    else None,
                }
                records.append(record)
            preview = output_root / "qc/previews" / f"{label}.png"
            _render_preview(
                source,
                photometric=str(acquisition.photometric_interpretation),
                spacing=spacing,
                results=results,
                crops=crops,
                sample_number=sample_number,
                destination=preview,
            )
            previews.append(preview)
        finally:
            if extraction.exists():
                shutil.rmtree(extraction)

    frame = pd.DataFrame(records)
    _atomic_parquet(frame, output_root / "manifests/panel_results.parquet")
    for sheet_number, start in enumerate(range(0, len(previews), 16), start=1):
        save_contact_sheet(
            previews[start : start + 16],
            output_root / f"qc/contact_sheets/sheet_{sheet_number:03d}.png",
            columns=4,
            cell_width=300,
        )
    acquisition_states = (
        frame.assign(
            severity=frame["qc_state"].map(
                {QCState.PASS.value: 0, QCState.BORDERLINE.value: 1, QCState.FAIL.value: 2}
            )
        )
        .groupby("acquisition_index")["severity"]
        .max()
        .map({0: QCState.PASS.value, 1: QCState.BORDERLINE.value, 2: QCState.FAIL.value})
    )
    counts = acquisition_states.value_counts().to_dict()
    summary = {
        "sample_name": sample_name,
        "localizer_version": localizer_version,
        "acquisitions": len(sample),
        "panels": len(frame),
        "automatic_pass": int(counts.get(QCState.PASS.value, 0)),
        "automatic_borderline": int(counts.get(QCState.BORDERLINE.value, 0)),
        "automatic_fail": int(counts.get(QCState.FAIL.value, 0)),
        "temporary_files_remaining": sum(path.is_file() for path in work_root.rglob("*")),
        "selection_outcomes_read": False,
        "full_cohort_processed": False,
        "crops_saved": save_crops,
    }
    _atomic_json(summary, output_root / "manifests/automatic_summary.json")
    return summary
