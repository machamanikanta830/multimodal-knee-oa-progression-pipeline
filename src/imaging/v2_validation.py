"""Run and visually validate the versioned imaging-localization V2 candidate on samples only."""

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
from imaging.localization_v2 import LocalizationQC, localize_bilateral_tibiofemoral_joints_v2
from imaging.pixel_qc import normalize_for_display, save_contact_sheet, split_bilateral_midpoint
from imaging.preprocessing import crop_around_center, resample_to_spacing, robust_minmax

TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
OUTPUT_SHAPE = (1067, 1067)
LOCALIZER_VERSION = "v2_candidate_r3"
PANEL_POSITIONS = ("screen_left", "screen_right")

DEFAULT_ARCHIVE_ROOT = Path("data/raw/oai_images/full")
DEFAULT_V1_PANELS = Path("data/processed/manifests/v00_xray_full_standardized_images.parquet")
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v2_candidate")


class V2ValidationError(ValueError):
    """Raised when sample-only V2 processing cannot proceed safely."""


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
        raise V2ValidationError("Sample archive does not contain exactly one DICOM candidate")
    return files[0]


def _crop_path(output_root: Path, sample_label: str, panel_position: str) -> Path:
    return output_root / "crops" / sample_label / f"{panel_position}_uint16.npy"


def _render_comparison(
    source: np.ndarray,
    *,
    photometric: str,
    spacing: tuple[float, float],
    v1_rows: dict[str, pd.Series],
    v2: dict[str, dict[str, Any]],
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
    original.paste(image, ((1200 - image.width) // 2, header))
    draw = ImageDraw.Draw(original)
    x_offset = (1200 - image.width) // 2
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
    draw.text(
        (8, 8),
        f"V2 methodology {sample_number:03d} | red=V1, green=V2",
        fill="white",
    )
    for position in PANEL_POSITIONS:
        panel_offset = 0 if position == "screen_left" else midpoint
        old = v1_rows[position]
        old_x = x_offset + round((panel_offset + float(old["joint_column"])) * scale)
        old_y = header + round(float(old["joint_row"]) * scale)
        draw.line((old_x - 10, old_y, old_x + 10, old_y), fill="red", width=3)
        draw.line((old_x, old_y - 10, old_x, old_y + 10), fill="red", width=3)

        new = v2[position]
        source_row = new["joint_row_resampled"] * TARGET_SPACING_MM / spacing[0]
        source_column = new["joint_column_resampled"] * TARGET_SPACING_MM / spacing[1]
        new_x = x_offset + round((panel_offset + source_column) * scale)
        new_y = header + round(source_row * scale)
        half_height = round((CROP_SIZE_MM / spacing[0] / 2) * scale)
        half_width = round((CROP_SIZE_MM / spacing[1] / 2) * scale)
        draw.rectangle(
            (
                new_x - half_width,
                new_y - half_height,
                new_x + half_width,
                new_y + half_height,
            ),
            outline="lime",
            width=3,
        )
        draw.line((new_x - 12, new_y, new_x + 12, new_y), fill="lime", width=3)
        draw.line((new_x, new_y - 12, new_x, new_y + 12), fill="lime", width=3)

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
        state = v2[position]["qc_state"]
        crop_draw.text((x, 10), f"{position.replace('_', '-')} | {state}", fill="white")
    final = Image.new("RGB", (1200, original.height + crop_canvas.height), "black")
    final.paste(original, (0, 0))
    final.paste(crop_canvas, (0, original.height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, format="PNG", optimize=True)


def _render_laterality_preview(
    source: np.ndarray, photometric: str, sample_number: int, destination: Path
) -> None:
    display = normalize_for_display(source, photometric)
    image = Image.fromarray(display, mode="L").convert("RGB")
    image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1200, image.height + 30), "black")
    canvas.paste(image, ((1200 - image.width) // 2, 30))
    ImageDraw.Draw(canvas).text(
        (8, 8), f"Laterality audit {sample_number:03d} | screen panels only", fill="white"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def process_v2_sample(
    *,
    manifest_path: Path,
    sample_name: str,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    v1_panel_manifest_path: Path = DEFAULT_V1_PANELS,
    output_base: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = Path("data/interim/oai_images/v2_work"),
) -> dict[str, Any]:
    """Process one development or holdout manifest without modifying V1 or raw archives."""

    output_root = output_base / sample_name / LOCALIZER_VERSION
    if output_root.exists():
        raise V2ValidationError("V2 sample output already exists and will not be overwritten")
    sample = pd.read_parquet(manifest_path).sort_values("acquisition_index", kind="stable")
    if sample["acquisition_index"].duplicated().any():
        raise V2ValidationError("V2 methodology manifest has duplicate acquisitions")
    if sample.get("selection_outcomes_read", pd.Series([True])).astype(bool).any():
        raise V2ValidationError("V2 methodology selection is not outcome-blind")
    v1 = pd.read_parquet(v1_panel_manifest_path)
    v1 = v1.loc[v1["acquisition_index"].isin(sample["acquisition_index"])]
    if len(v1) != len(sample) * 2:
        raise V2ValidationError("V1 panel records do not cover the methodology sample")

    work_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    laterality_paths: list[Path] = []
    for sample_number, acquisition in enumerate(sample.itertuples(index=False), start=1):
        label = f"sample_{sample_number:03d}"
        extraction = work_root / f"{sample_name}_{sample_number:03d}"
        if extraction.exists():
            raise V2ValidationError("A previous V2 temporary extraction exists")
        archive_path = archive_root / str(acquisition.associated_file_reference)
        try:
            contents = safe_extract_archive(archive_path, extraction)
            if contents.regular_files != 1:
                raise V2ValidationError("Unexpected V2 archive structure")
            dicom_path = _single_file(extraction)
            dataset = pydicom.dcmread(dicom_path, force=False)
            source = np.asarray(dataset.pixel_array)
            if source.ndim != 2 or source.dtype != np.dtype("uint16"):
                raise V2ValidationError("V2 source pixels are not single-frame uint16")
            panels = dict(zip(PANEL_POSITIONS, split_bilateral_midpoint(source), strict=True))
            spacing = (
                float(acquisition.original_row_spacing_mm),
                float(acquisition.original_column_spacing_mm),
            )
            resampled = {
                position: resample_to_spacing(
                    panel,
                    spacing,
                    (TARGET_SPACING_MM, TARGET_SPACING_MM),
                )
                for position, panel in panels.items()
            }
            left, right = localize_bilateral_tibiofemoral_joints_v2(
                resampled["screen_left"], resampled["screen_right"]
            )
            locations = {"screen_left": left, "screen_right": right}
            crops: dict[str, np.ndarray] = {}
            panel_records: dict[str, dict[str, Any]] = {}
            for position in PANEL_POSITIONS:
                location = locations[position]
                padding_value = int(np.percentile(resampled[position], 0.5))
                crop, geometry = crop_around_center(
                    resampled[position],
                    (location.row, location.column),
                    OUTPUT_SHAPE,
                    padding_value=padding_value,
                )
                if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                    raise V2ValidationError("V2 candidate crop shape or dtype is invalid")
                crop_path = _crop_path(output_root, label, position)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(crop_path, crop, allow_pickle=False)
                crops[position] = crop
                panel_records[position] = {
                    "sample_number": sample_number,
                    "acquisition_index": int(acquisition.acquisition_index),
                    "methodology_role": acquisition.methodology_role,
                    "panel_position": position,
                    "localizer_version": LOCALIZER_VERSION,
                    "joint_row_resampled": location.row,
                    "joint_column_resampled": location.column,
                    "confidence": location.confidence,
                    "qc_state": location.qc_state.value,
                    "vertical_peak_z": location.vertical_peak_z,
                    "vertical_prominence_z": location.vertical_prominence_z,
                    "joint_line_support": location.joint_line_support,
                    "row_fraction": location.row_fraction,
                    "column_fraction": location.column_fraction,
                    "pair_row_delta_mm": location.pair_row_delta_mm,
                    "bilateral_rescue_used": location.bilateral_rescue_used,
                    "padding_top": geometry.padding_top,
                    "padding_bottom": geometry.padding_bottom,
                    "padding_left": geometry.padding_left,
                    "padding_right": geometry.padding_right,
                    "vertical_padding_mm": (geometry.padding_top + geometry.padding_bottom)
                    * TARGET_SPACING_MM,
                    "horizontal_padding_mm": (geometry.padding_left + geometry.padding_right)
                    * TARGET_SPACING_MM,
                    "crop_path": crop_path.relative_to(output_base).as_posix(),
                }
            old_rows = {
                position: v1.loc[
                    v1["acquisition_index"].eq(acquisition.acquisition_index)
                    & v1["screen_panel"].eq(position)
                ].iloc[0]
                for position in PANEL_POSITIONS
            }
            comparison = output_root / "qc/comparisons" / f"{label}.png"
            _render_comparison(
                source,
                photometric=str(acquisition.photometric_interpretation),
                spacing=spacing,
                v1_rows=old_rows,
                v2=panel_records,
                crops=crops,
                sample_number=sample_number,
                destination=comparison,
            )
            laterality_preview = output_root / "qc/laterality" / f"{label}.png"
            _render_laterality_preview(
                source,
                str(acquisition.photometric_interpretation),
                sample_number,
                laterality_preview,
            )
            preview_paths.append(comparison)
            laterality_paths.append(laterality_preview)
            intensity = np.asarray(source, dtype=np.float32)
            source_percentiles = np.percentile(intensity, [0.5, 1.0, 50.0, 99.0, 99.5])
            for position in PANEL_POSITIONS:
                record = panel_records[position]
                old = old_rows[position]
                record.update(
                    {
                        "v1_visual_state": getattr(acquisition, "localization_visual_state", None),
                        "v1_joint_row": int(old["joint_row"]),
                        "v1_joint_column": int(old["joint_column"]),
                        "v1_to_v2_vertical_delta_mm": abs(
                            record["joint_row_resampled"]
                            - round(int(old["joint_row"]) * spacing[0] / TARGET_SPACING_MM)
                        )
                        * TARGET_SPACING_MM,
                        "original_rows": int(source.shape[0]),
                        "original_columns": int(source.shape[1]),
                        "physical_fov_height_mm": source.shape[0] * spacing[0],
                        "physical_fov_width_mm": source.shape[1] * spacing[1],
                        "original_spacing_mm": spacing[0],
                        "manufacturer": acquisition.manufacturer,
                        "scanner_model": acquisition.manufacturer_model_name,
                        "image_release": acquisition.image_release_study,
                        "oai_qc_category": acquisition.xray_accept_qc,
                        "source_p005": float(source_percentiles[0]),
                        "source_p01": float(source_percentiles[1]),
                        "source_median": float(source_percentiles[2]),
                        "source_p99": float(source_percentiles[3]),
                        "source_p995": float(source_percentiles[4]),
                    }
                )
                records.append(record)
        finally:
            if extraction.exists():
                shutil.rmtree(extraction)

    frame = pd.DataFrame(records)
    _atomic_parquet(frame, output_root / "manifests/panel_results.parquet")
    for kind, paths in (("comparisons", preview_paths), ("laterality", laterality_paths)):
        for sheet_number, start in enumerate(range(0, len(paths), 4), start=1):
            save_contact_sheet(
                paths[start : start + 4],
                output_root / f"qc/{kind}_contact_sheets/sheet_{sheet_number:03d}.png",
                columns=2,
                cell_width=600,
            )
    counts = frame["qc_state"].value_counts().to_dict()
    summary = {
        "sample_name": sample_name,
        "localizer_version": LOCALIZER_VERSION,
        "acquisitions": len(sample),
        "panels": len(frame),
        "automatic_pass": int(counts.get(LocalizationQC.PASS.value, 0)),
        "automatic_borderline": int(counts.get(LocalizationQC.BORDERLINE.value, 0)),
        "automatic_fail": int(counts.get(LocalizationQC.FAIL.value, 0)),
        "comparison_previews": len(preview_paths),
        "laterality_previews": len(laterality_paths),
        "temporary_files_remaining": sum(path.is_file() for path in work_root.rglob("*")),
        "outcomes_read": False,
        "v1_overwritten": False,
    }
    _atomic_json(summary, output_root / "manifests/automatic_summary.json")
    return summary
