"""Materialize privacy-safe, local-only standardized crops for the imaging pilot.

The command consumes anonymous broad-panel arrays and review manifests. It never reads cohort
outcomes, downloads data, assigns participant splits, or changes source DICOMs. Anatomical side is
carried forward only when the separate laterality assessment is CONFIDENT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from imaging.pixel_qc import save_contact_sheet
from imaging.preprocessing import (
    PreprocessingError,
    crop_around_center,
    localize_bilateral_tibiofemoral_joints,
    physical_crop_shape,
    resample_to_spacing,
    robust_minmax,
    save_crop_preview,
    save_localization_preview,
)

DEFAULT_PANEL_ROOT = Path("data/interim/oai_images/pilot_preprocessed/broad_panels")
DEFAULT_PANEL_MANIFEST = Path(
    "data/interim/oai_images/pilot_preprocessed/manifests/broad_panel_manifest.csv"
)
DEFAULT_LATERALITY_MANIFEST = Path(
    "data/interim/oai_images/pilot_preprocessed/manifests/laterality_assessment.csv"
)
DEFAULT_LOCALIZATION_REVIEW = Path(
    "data/interim/oai_images/pilot_preprocessed/manifests/localization_review.csv"
)
DEFAULT_OUTPUT_ROOT = Path("data/interim/oai_images/pilot_preprocessed/standardized_v1")
PANEL_POSITIONS = ("screen_left", "screen_right")
REVIEW_STATES = {"SUCCESS", "BORDERLINE", "FAILED"}
FULL_ACQUISITION_COUNT = 3621


class PilotPreprocessingError(ValueError):
    """Raised when the local pilot inputs or review gates are inconsistent."""


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise PilotPreprocessingError(f"{label} is missing required columns: {sorted(missing)}")


def _load_inputs(
    panel_manifest_path: Path,
    laterality_manifest_path: Path,
    localization_review_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels = pd.read_csv(panel_manifest_path)
    laterality = pd.read_csv(laterality_manifest_path, keep_default_na=False)
    review = pd.read_csv(localization_review_path, keep_default_na=False)
    _require_columns(
        panels,
        {
            "pilot_index",
            "panel_position",
            "panel_rows",
            "panel_columns",
            "source_row_spacing_mm",
            "source_column_spacing_mm",
            "dtype",
        },
        "Panel manifest",
    )
    _require_columns(
        laterality,
        {
            "pilot_index",
            "state",
            "screen_left_anatomical_side",
            "screen_right_anatomical_side",
        },
        "Laterality manifest",
    )
    _require_columns(
        review,
        {"pilot_index", "panel_position", "localization_state"},
        "Localization review",
    )
    expected_panel_keys = pd.MultiIndex.from_product(
        [range(1, 33), PANEL_POSITIONS], names=["pilot_index", "panel_position"]
    )
    actual_panel_keys = pd.MultiIndex.from_frame(panels[["pilot_index", "panel_position"]])
    actual_review_keys = pd.MultiIndex.from_frame(review[["pilot_index", "panel_position"]])
    if not actual_panel_keys.is_unique or set(actual_panel_keys) != set(expected_panel_keys):
        raise PilotPreprocessingError("Panel manifest must contain exactly two positions for 32")
    if not actual_review_keys.is_unique or set(actual_review_keys) != set(expected_panel_keys):
        raise PilotPreprocessingError("Review manifest must contain exactly two positions for 32")
    if set(review["localization_state"]) - REVIEW_STATES:
        raise PilotPreprocessingError("Localization review contains an unsupported state")
    if laterality["pilot_index"].duplicated().any() or set(laterality["pilot_index"]) != set(
        range(1, 33)
    ):
        raise PilotPreprocessingError("Laterality manifest must contain 32 unique pilot indices")
    unresolved = laterality["state"].isin(["AMBIGUOUS", "CONFLICTING"])
    if (
        laterality.loc[
            unresolved,
            ["screen_left_anatomical_side", "screen_right_anatomical_side"],
        ]
        .ne("")
        .any(axis=None)
    ):
        raise PilotPreprocessingError("Unresolved laterality rows must not contain a mapping")
    return panels, laterality, review


def _uint8_preview(array: np.ndarray) -> Image.Image:
    normalized = robust_minmax(array, lower_percentile=0.5, upper_percentile=99.5)
    return Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").convert("RGB")


def _save_pipeline_preview(
    screen_left: np.ndarray,
    screen_right: np.ndarray,
    screen_left_overlay: Path,
    screen_right_overlay: Path,
    screen_left_crop: np.ndarray,
    screen_right_crop: np.ndarray,
    destination: Path,
    *,
    pilot_index: int,
) -> None:
    """Render source reconstruction, panel overlays, and crops with anonymous labels."""

    bilateral = _uint8_preview(np.concatenate([screen_left, screen_right], axis=1))
    bilateral.thumbnail((1200, 520), Image.Resampling.LANCZOS)
    bilateral_canvas = Image.new("RGB", (1200, bilateral.height + 28), "black")
    bilateral_canvas.paste(bilateral, ((1200 - bilateral.width) // 2, 28))
    bilateral_draw = ImageDraw.Draw(bilateral_canvas)
    midpoint = 600
    bilateral_draw.line((midpoint, 28, midpoint, bilateral_canvas.height), fill="magenta", width=3)
    bilateral_draw.text(
        (8, 7), f"Pilot {pilot_index:02d} | source-pixel bilateral reconstruction", fill="white"
    )

    overlay_images: list[Image.Image] = []
    for path in (screen_left_overlay, screen_right_overlay):
        with Image.open(path) as image:
            copy = image.convert("RGB")
            copy.thumbnail((590, 650), Image.Resampling.LANCZOS)
            overlay_images.append(copy)
    overlay_height = max(image.height for image in overlay_images)
    overlay_canvas = Image.new("RGB", (1200, overlay_height), (20, 20, 20))
    overlay_canvas.paste(overlay_images[0], (0, 0))
    overlay_canvas.paste(overlay_images[1], (610, 0))

    crop_images = [_uint8_preview(screen_left_crop), _uint8_preview(screen_right_crop)]
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
        (1200, bilateral_canvas.height + overlay_canvas.height + crop_canvas.height),
        (20, 20, 20),
    )
    offset = 0
    for image in (bilateral_canvas, overlay_canvas, crop_canvas):
        final.paste(image, (0, offset))
        offset += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, format="PNG", optimize=True)


def _side_for_panel(laterality_row: pd.Series, panel_position: str) -> str | None:
    if laterality_row["state"] != "CONFIDENT":
        return None
    value = laterality_row[f"{panel_position}_anatomical_side"]
    return str(value) if value in {"L", "R"} else None


def _spacing_summary(
    acquisition_spacings: list[float],
    candidates: tuple[float, ...],
    crop_sizes_mm: tuple[float, ...],
) -> list[dict[str, Any]]:
    values = np.asarray(acquisition_spacings, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for target in candidates:
        summaries.append(
            {
                "target_spacing_mm": target,
                "upsampling_count": int(np.sum(values > target)),
                "upsampling_percent": float(np.mean(values > target) * 100),
                "downsampling_count": int(np.sum(values < target)),
                "downsampling_percent": float(np.mean(values < target) * 100),
                "unchanged_count": int(np.sum(np.isclose(values, target))),
                "crop_pixels": {str(size): round(size / target) for size in crop_sizes_mm},
            }
        )
    return summaries


def prepare_pilot(
    *,
    panel_root: Path,
    panel_manifest_path: Path,
    laterality_manifest_path: Path,
    localization_review_path: Path,
    output_root: Path,
    target_spacing_mm: float = 0.15,
    crop_size_mm: float = 160.0,
) -> dict[str, Any]:
    """Create reviewed panel-position crops and return aggregate-only validation facts."""

    panels, laterality, review = _load_inputs(
        panel_manifest_path, laterality_manifest_path, localization_review_path
    )
    output_shape = physical_crop_shape(
        (crop_size_mm, crop_size_mm), (target_spacing_mm, target_spacing_mm)
    )
    records: list[dict[str, Any]] = []
    stage_previews: list[Path] = []
    for pilot_index in range(1, 33):
        panel_arrays = {
            position: np.load(panel_root / f"acquisition_{pilot_index:03d}" / f"{position}.npy")
            for position in PANEL_POSITIONS
        }
        manifest_rows = panels.loc[panels["pilot_index"].eq(pilot_index)].set_index(
            "panel_position"
        )
        for position, array in panel_arrays.items():
            row = manifest_rows.loc[position]
            if array.shape != (int(row["panel_rows"]), int(row["panel_columns"])):
                raise PilotPreprocessingError("Panel array geometry disagrees with its manifest")
            if str(array.dtype) != row["dtype"]:
                raise PilotPreprocessingError("Panel array dtype disagrees with its manifest")
        left_location, right_location = localize_bilateral_tibiofemoral_joints(
            panel_arrays["screen_left"], panel_arrays["screen_right"]
        )
        locations = {
            "screen_left": left_location,
            "screen_right": right_location,
        }
        laterality_row = laterality.loc[laterality["pilot_index"].eq(pilot_index)].iloc[0]
        review_rows = review.loc[review["pilot_index"].eq(pilot_index)].set_index("panel_position")
        processed_crops: dict[str, np.ndarray] = {}
        overlays: dict[str, Path] = {}
        for position in PANEL_POSITIONS:
            panel = panel_arrays[position]
            manifest_row = manifest_rows.loc[position]
            location = locations[position]
            source_spacing = (
                float(manifest_row["source_row_spacing_mm"]),
                float(manifest_row["source_column_spacing_mm"]),
            )
            review_state = str(review_rows.loc[position, "localization_state"])
            if review_state == "FAILED":
                records.append(
                    {
                        "pilot_index": pilot_index,
                        "panel_position": position,
                        "anatomical_side": _side_for_panel(laterality_row, position),
                        "laterality_state": laterality_row["state"],
                        "localization_state": review_state,
                        "crop_materialized": False,
                    }
                )
                continue
            resampled = resample_to_spacing(
                panel,
                source_spacing,
                (target_spacing_mm, target_spacing_mm),
            )
            resampled_center = (
                round(location.row * source_spacing[0] / target_spacing_mm),
                round(location.column * source_spacing[1] / target_spacing_mm),
            )
            padding_value = int(np.percentile(resampled, 0.5))
            crop, geometry = crop_around_center(
                resampled,
                resampled_center,
                output_shape,
                padding_value=padding_value,
            )
            crop_dir = output_root / "crops" / f"acquisition_{pilot_index:03d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{position}_uint16.npy"
            np.save(crop_path, crop, allow_pickle=False)
            preview_path = output_root / "crop_previews" / f"pilot_{pilot_index:03d}_{position}.png"
            save_crop_preview(
                crop,
                preview_path,
                pilot_index=pilot_index,
                panel_position=position,
            )
            source_crop_shape = physical_crop_shape((crop_size_mm, crop_size_mm), source_spacing)
            overlay_path = (
                output_root / "localization_overlays" / f"pilot_{pilot_index:03d}_{position}.png"
            )
            save_localization_preview(
                panel,
                location,
                source_crop_shape,
                overlay_path,
                pilot_index=pilot_index,
                panel_position=position,
            )
            overlays[position] = overlay_path
            processed_crops[position] = crop
            records.append(
                {
                    "pilot_index": pilot_index,
                    "panel_position": position,
                    "anatomical_side": _side_for_panel(laterality_row, position),
                    "laterality_state": laterality_row["state"],
                    "localization_state": review_state,
                    "joint_row": location.row,
                    "joint_column": location.column,
                    "paired_projection_robust_score": location.robust_score,
                    "source_rows": panel.shape[0],
                    "source_columns": panel.shape[1],
                    "source_row_spacing_mm": source_spacing[0],
                    "source_column_spacing_mm": source_spacing[1],
                    "resampled_rows": resampled.shape[0],
                    "resampled_columns": resampled.shape[1],
                    "target_spacing_mm": target_spacing_mm,
                    "crop_size_mm": crop_size_mm,
                    "crop_rows": crop.shape[0],
                    "crop_columns": crop.shape[1],
                    "padding_top": geometry.padding_top,
                    "padding_bottom": geometry.padding_bottom,
                    "padding_left": geometry.padding_left,
                    "padding_right": geometry.padding_right,
                    "padding_used": geometry.used_padding,
                    "dtype": str(crop.dtype),
                    "crop_materialized": True,
                }
            )
        if len(processed_crops) == 2:
            stage_path = output_root / "stage_previews" / f"pilot_{pilot_index:03d}.png"
            _save_pipeline_preview(
                panel_arrays["screen_left"],
                panel_arrays["screen_right"],
                overlays["screen_left"],
                overlays["screen_right"],
                processed_crops["screen_left"],
                processed_crops["screen_right"],
                stage_path,
                pilot_index=pilot_index,
            )
            stage_previews.append(stage_path)

    result = pd.DataFrame(records).sort_values(["pilot_index", "panel_position"])
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(manifest_dir / "preprocessing_manifest.csv", index=False)
    contact_sheet_paths: list[Path] = []
    for group_index, start in enumerate(range(0, len(stage_previews), 4), start=1):
        path = output_root / "contact_sheets" / f"pipeline_contact_{group_index:02d}.png"
        save_contact_sheet(
            stage_previews[start : start + 4],
            path,
            columns=2,
            cell_width=600,
        )
        contact_sheet_paths.append(path)

    crop_paths = list((output_root / "crops").rglob("*.npy"))
    crop_file_bytes = sum(path.stat().st_size for path in crop_paths)
    source_spacings = (
        panels.sort_values(["pilot_index", "panel_position"])
        .drop_duplicates("pilot_index")["source_row_spacing_mm"]
        .astype(float)
        .tolist()
    )
    materialized = result["crop_materialized"].fillna(False).astype(bool)
    summary = {
        "pilot_acquisitions": 32,
        "broad_panels": 64,
        "laterality_states": dict(Counter(laterality["state"])),
        "unresolved_laterality_acquisitions": int(
            laterality["state"].isin(["AMBIGUOUS", "CONFLICTING"]).sum()
        ),
        "localization_states": dict(Counter(review["localization_state"])),
        "standardized_crops": int(materialized.sum()),
        "standardized_shape": list(output_shape),
        "standardized_dtype": sorted(result.loc[materialized, "dtype"].dropna().unique()),
        "target_spacing_mm": target_spacing_mm,
        "physical_crop_size_mm": crop_size_mm,
        "padding_used_panels": int(result.loc[materialized, "padding_used"].sum()),
        "contact_sheets": len(contact_sheet_paths),
        "crop_file_bytes": crop_file_bytes,
        "projected_3621_acquisition_crop_file_bytes": round(
            crop_file_bytes / max(len(crop_paths), 1) * FULL_ACQUISITION_COUNT * 2
        ),
        "spacing_candidates": _spacing_summary(
            source_spacings, (0.15, 0.16, 0.17), (140.0, 160.0, 180.0)
        ),
        "privacy": {
            "participant_identifiers_serialized": False,
            "accessions_serialized": False,
            "dates_serialized": False,
            "source_paths_serialized": False,
            "labels_are_anonymous_sequential_indices": True,
        },
    }
    (manifest_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create reviewed local-only standardized crops for the 32-image pilot."
    )
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--panel-manifest", type=Path, default=DEFAULT_PANEL_MANIFEST)
    parser.add_argument("--laterality-manifest", type=Path, default=DEFAULT_LATERALITY_MANIFEST)
    parser.add_argument("--localization-review", type=Path, default=DEFAULT_LOCALIZATION_REVIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-spacing-mm", type=float, default=0.15)
    parser.add_argument("--crop-size-mm", type=float, default=160.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = prepare_pilot(
            panel_root=args.panel_root,
            panel_manifest_path=args.panel_manifest,
            laterality_manifest_path=args.laterality_manifest,
            localization_review_path=args.localization_review,
            output_root=args.output_root,
            target_spacing_mm=args.target_spacing_mm,
            crop_size_mm=args.crop_size_mm,
        )
    except (OSError, ValueError, PreprocessingError, pd.errors.ParserError) as error:
        raise SystemExit(f"Pilot preprocessing failed: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
