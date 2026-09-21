"""Apply the frozen Localization V3 and Laterality V2 policies to the full baseline X-ray set.

The module creates the versioned ``v3_frozen_full`` imaging dataset. It reads raw archives without
modifying them, never overwrites V1 or V2 artifacts, and never changes the analysis cohort, creates
a modeling split, or trains a model. Every identity-bearing manifest, real pixel array, preview, and
review artifact is written only below Git-ignored local data directories.

Frozen behaviour is imported, never re-implemented: candidate localization, the independent
anatomical validator, the three-state disposition, and the screen-to-anatomical mapping all come
from the approved modules and their recorded source hashes are re-verified before processing starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

from imaging import localization_v3
from imaging.archive_extract import UnsafeArchiveError, safe_extract_archive
from imaging.artifact_io import (
    atomic_json,
    atomic_parquet,
    atomic_png,
    hash_paths,
    hash_tree,
    json_default,
    sha256_file,
)
from imaging.dicom_inspect import DicomMetadata, inspect_dicom
from imaging.full_preprocessing import (
    VALIDATED_MAX_HORIZONTAL_PADDING_MM,
    VALIDATED_MAX_VERTICAL_PADDING_MM,
    verify_archive_hashes,
)
from imaging.laterality_exceptions import BilateralStructureCheck, assess_bilateral_structure
from imaging.laterality_v2 import LateralityV2State, assess_validated_oai_screen_rule
from imaging.localization_v3 import (
    JointLocalizationV3,
    QCState,
    localize_bilateral_tibiofemoral_joints_v3,
)
from imaging.methodology_v3 import verify_v2_preservation
from imaging.pixel_qc import normalize_for_display, split_bilateral_midpoint
from imaging.preprocessing import crop_around_center, resample_to_spacing, robust_minmax
from imaging.review_privacy import (
    BURNED_IN_MARGIN_MM,
    crop_rows_inside_margin,
    redact_horizontal_margins,
)

EXPECTED_ACQUISITIONS = 3621
EXPECTED_PANELS = EXPECTED_ACQUISITIONS * 2
DATASET_VERSION = "v3_frozen_full"
PREPROCESSING_VERSION = "oai-v00-xray-v3-frozen-full-v1"
LOCALIZER_VERSION = "localization_v3_candidate_r1"
LATERALITY_POLICY_VERSION = "laterality_v2_oai_frozen_v1"
TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
OUTPUT_SHAPE = (1067, 1067)
PANEL_POSITIONS = ("screen_left", "screen_right")
MIN_REPORTABLE_GROUP = 5
REVIEW_CANDIDATE_LIMIT = 4
MINIMUM_FREE_RESERVE_BYTES = 12 * 1024**3
REVIEW_ASSET_BYTES_ESTIMATE = 1_500_000
MANIFEST_AND_QC_RESERVE_BYTES = 128 * 1024**2
TEMPORARY_WORK_BYTES_PER_WORKER = 64 * 1024**2

# Representative V3 holdout disposition (192 acquisitions) used only as a compatibility reference.
REPRESENTATIVE_HOLDOUT_TOTAL = 192
REPRESENTATIVE_HOLDOUT_COUNTS = {
    QCState.PASS.value: 136,
    QCState.BORDERLINE.value: 55,
    QCState.FAIL.value: 1,
}
COMPATIBILITY_Z = 2.5758293035489004
SUBGROUP_FAIL_RATE_LIMIT = 0.05
SUBGROUP_PASS_RATE_FLOOR = 0.35

BILATERAL_PREVIEW_WIDTH = 1000
PANEL_OVERLAY_WIDTH = 640
CROP_PREVIEW_WIDTH = 480

DEFAULT_ARCHIVE_ROOT = Path("data/raw/oai_images/full")
DEFAULT_V1_ROOT = Path("data/processed/oai_images/full_v1")
DEFAULT_V2_ROOT = Path("data/processed/oai_images/v2_candidate")
DEFAULT_V3_METHODOLOGY_ROOT = Path("data/processed/oai_images/localization_v3_candidate")
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v3_frozen_full")
DEFAULT_WORK_ROOT = Path("data/interim/oai_images/v3_frozen_full_work")
DEFAULT_ACQUISITION_INPUT = Path(
    "data/processed/manifests/v00_xray_full_preprocessing_acquisitions.parquet"
)
DEFAULT_V1_PANEL_INPUT = Path("data/processed/manifests/v00_xray_full_standardized_images.parquet")
DEFAULT_IMAGE_MANIFEST = Path("data/processed/manifests/v00_xray_manifest.parquet")
DEFAULT_ARCHIVE_LEDGER = DEFAULT_V1_ROOT / "manifests/source_archives.parquet"
FROZEN_SOURCE_FILES = (
    Path("src/imaging/localization_v3.py"),
    Path("src/imaging/laterality_v2.py"),
    Path("src/imaging/laterality.py"),
    Path("src/imaging/preprocessing.py"),
    Path("src/imaging/pixel_qc.py"),
    Path("src/imaging/archive_extract.py"),
)
V3_FREEZE_RECORDS = (
    DEFAULT_V3_METHODOLOGY_ROOT / "audit/localization_v3_candidate_r1_freeze.json",
    DEFAULT_V3_METHODOLOGY_ROOT / "audit/localization_v3_candidate_r1_freeze_clarification.json",
    DEFAULT_V3_METHODOLOGY_ROOT / "audit/laterality_v2_oai_frozen_v1.json",
)
STATE_SEVERITY = {QCState.PASS.value: 0, QCState.BORDERLINE.value: 1, QCState.FAIL.value: 2}
SEVERITY_STATE = {value: key for key, value in STATE_SEVERITY.items()}
# OAI encodes the right knee as side 1 and the left knee as side 2.
ANATOMICAL_SIDE_TO_KNEE_SIDE_CODE = {"R": "1", "L": "2"}


class FrozenFullV3Error(ValueError):
    """Raised when the frozen full-cohort V3 workflow cannot proceed safely."""


# ---------------------------------------------------------------------------
# A. Preservation gate
# ---------------------------------------------------------------------------


def _freeze_record_source_hash() -> str:
    record = json.loads(V3_FREEZE_RECORDS[0].read_text(encoding="utf-8"))
    digest = str(record.get("source_sha256", ""))
    if len(digest) != 64:
        raise FrozenFullV3Error("The V3 freeze record does not carry a usable source hash")
    return digest


def _ledger_gate(
    label: str,
    frame: pd.DataFrame,
    ledger_path: Path,
) -> dict[str, Any]:
    """Create a preservation ledger on the first run and verify it on every later run."""

    if not ledger_path.is_file():
        atomic_parquet(frame, ledger_path)
        return {
            "artifact_group": label,
            "action": "ledger_created",
            "files": len(frame),
            "bytes": int(frame["bytes"].sum()) if len(frame) else 0,
            "all_unchanged": True,
            "added": 0,
            "removed": 0,
            "modified": 0,
        }
    previous = pd.read_parquet(ledger_path).set_index("artifact")
    current = frame.set_index("artifact")
    added = sorted(set(current.index) - set(previous.index))
    removed = sorted(set(previous.index) - set(current.index))
    shared = sorted(set(current.index) & set(previous.index))
    modified = [
        artifact
        for artifact in shared
        if current.loc[artifact, "sha256"] != previous.loc[artifact, "sha256"]
        or int(current.loc[artifact, "bytes"]) != int(previous.loc[artifact, "bytes"])
    ]
    return {
        "artifact_group": label,
        "action": "ledger_verified",
        "files": len(frame),
        "bytes": int(frame["bytes"].sum()) if len(frame) else 0,
        "all_unchanged": not (added or removed or modified),
        "added": len(added),
        "removed": len(removed),
        "modified": len(modified),
    }


def run_preservation_gate(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    archive_ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
    v1_root: Path = DEFAULT_V1_ROOT,
    v2_root: Path = DEFAULT_V2_ROOT,
    v3_methodology_root: Path = DEFAULT_V3_METHODOLOGY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    v1_manifest_paths: tuple[Path, ...] = (DEFAULT_ACQUISITION_INPUT, DEFAULT_V1_PANEL_INPUT),
    require_full_archive_set: bool = True,
    workers: int = 4,
) -> dict[str, Any]:
    """Verify and preserve every prior artifact before any V3 full-cohort pixel work.

    The gate re-hashes the raw archives, the V1 provisional artifacts, the V2 artifacts, the V3
    methodology and freeze records, and the frozen source files. It refuses to continue unless the
    frozen V3 source hash still equals the value recorded at freeze time.
    """

    audit_root = output_root / "audit"
    frozen_sources = hash_paths(
        [path for path in FROZEN_SOURCE_FILES if path.is_file()],
        relative_to=Path("."),
        workers=workers,
    )
    frozen_by_artifact = frozen_sources.set_index("artifact")["sha256"].to_dict()
    localizer_artifact = FROZEN_SOURCE_FILES[0].as_posix()
    recorded_hash = _freeze_record_source_hash()
    observed_hash = frozen_by_artifact.get(localizer_artifact)
    if observed_hash != recorded_hash:
        raise FrozenFullV3Error(
            "The frozen localization V3 source hash does not match its freeze record"
        )

    archives = verify_archive_hashes(archive_root=archive_root, ledger_path=archive_ledger_path)
    if require_full_archive_set and not archives["all_unchanged"]:
        raise FrozenFullV3Error("Raw archive verification failed; V3 reprocessing will not start")

    v1_paths = [path for path in v1_root.rglob("*") if path.is_file()]
    v1_paths += [path for path in v1_manifest_paths if path.is_file()]
    groups = [
        _ledger_gate(
            "frozen_source_files",
            frozen_sources,
            audit_root / "frozen_source_ledger.parquet",
        ),
        _ledger_gate(
            "v3_freeze_records",
            hash_paths(
                [path for path in V3_FREEZE_RECORDS if path.is_file()],
                relative_to=Path("."),
                workers=workers,
            ),
            audit_root / "v3_freeze_record_ledger.parquet",
        ),
        _ledger_gate(
            "v1_provisional_artifacts",
            hash_paths(v1_paths, relative_to=Path("."), workers=workers),
            audit_root / "v1_preservation_ledger.parquet",
        ),
        _ledger_gate(
            "v3_methodology_artifacts",
            hash_tree(v3_methodology_root, relative_to=Path("."), workers=workers),
            audit_root / "v3_methodology_ledger.parquet",
        ),
    ]
    v2 = verify_v2_preservation(v2_root=v2_root)
    unchanged = all(group["all_unchanged"] for group in groups) and bool(v2["all_unchanged"])
    summary = {
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "localizer_version": LOCALIZER_VERSION,
        "laterality_policy_version": LATERALITY_POLICY_VERSION,
        "frozen_localizer_source_sha256": recorded_hash,
        "frozen_localizer_source_hash_matches_freeze_record": True,
        "frozen_source_hashes": frozen_by_artifact,
        "raw_archives": archives,
        "v2_artifacts": {
            "artifact_group": "v2_artifacts",
            "action": "ledger_verified",
            "files": int(v2["files_checked"]),
            "hash_matches": int(v2["hash_matches"]),
            "all_unchanged": bool(v2["all_unchanged"]),
        },
        "artifact_groups": groups,
        "all_prior_artifacts_unchanged": unchanged,
        "prior_artifacts_modified_by_this_milestone": False,
        "gate_passed": bool(archives["all_unchanged"] and unchanged),
    }
    atomic_json(summary, audit_root / "preservation_gate.json")
    if not summary["gate_passed"]:
        raise FrozenFullV3Error("Preservation gate failed; prior artifacts are not intact")
    return summary


# ---------------------------------------------------------------------------
# B/C. Frozen per-acquisition preprocessing
# ---------------------------------------------------------------------------


def _crop_path(output_root: Path, acquisition_index: int, panel_position: str) -> Path:
    return (
        output_root
        / "crops"
        / f"acquisition_{acquisition_index:04d}"
        / f"{panel_position}_uint16.npy"
    )


def _review_root(output_root: Path, acquisition_index: int) -> Path:
    return output_root / "review/previews" / f"acquisition_{acquisition_index:04d}"


def _spacing(metadata: DicomMetadata) -> tuple[float, float]:
    value = metadata.pixel_spacing or metadata.imager_pixel_spacing
    if value is None or len(value) != 2:
        raise FrozenFullV3Error("A DICOM has no usable two-dimensional pixel spacing")
    spacing = (float(value[0]), float(value[1]))
    if any(not np.isfinite(item) or item <= 0 for item in spacing):
        raise FrozenFullV3Error("A DICOM has invalid pixel spacing")
    return spacing


def _single_extracted_file(directory: Path) -> Path:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise FrozenFullV3Error("An archive does not contain exactly one regular file")
    return files[0]


def _serialize_localization(result: JointLocalizationV3) -> dict[str, Any]:
    return {
        "joint_row_resampled": int(result.row),
        "joint_column_resampled": int(result.column),
        "localization_confidence": float(result.confidence),
        "qc_state": result.qc_state.value,
        "localization_state": result.localization_state.value,
        "layout_state": result.layout.state.value,
        "layout_reasons": "|".join(result.layout.reasons),
        "layout_physical_height_mm": float(result.layout.physical_height_mm),
        "layout_physical_width_mm": float(result.layout.physical_width_mm),
        "layout_blank_block_fraction": float(result.layout.blank_block_fraction),
        "layout_usable_width_fraction": float(result.layout.usable_width_fraction),
        "layout_usable_height_fraction": float(result.layout.usable_height_fraction),
        "anatomical_validator_state": result.anatomy.state.value,
        "anatomical_validator_reasons": "|".join(result.anatomy.reasons),
        "anatomical_validator_score": float(result.anatomy.score),
        "anatomy_blank_block_fraction": float(result.anatomy.blank_block_fraction),
        "joint_band_edge_ratio": float(result.anatomy.joint_band_edge_ratio),
        "superior_structure": float(result.anatomy.superior_structure),
        "inferior_structure": float(result.anatomy.inferior_structure),
        "horizontal_joint_support": float(result.anatomy.horizontal_joint_support),
        "central_structure_fraction": float(result.anatomy.central_structure_fraction),
        "collimation_dominance": float(result.anatomy.collimation_dominance),
        "candidate_count": int(result.candidate_count),
        "candidate_margin": float(result.candidate_margin),
        "pair_row_delta_mm": (
            None if result.pair_row_delta_mm is None else float(result.pair_row_delta_mm)
        ),
        "selected_candidate_score": float(result.selected_candidate_score),
        "vertical_peak_z": float(result.vertical_peak_z),
        "vertical_prominence_z": float(result.vertical_prominence_z),
        "horizontal_support": float(result.horizontal_support),
        "row_fraction": float(result.row_fraction),
        "column_fraction": float(result.column_fraction),
    }


def _serialize_structure(check: BilateralStructureCheck) -> dict[str, Any]:
    return {
        "structure_check_passed": bool(check.passed),
        "bilateral_structure_confirmed": bool(check.bilateral_structure_confirmed),
        "midpoint_separation_clean": bool(check.midpoint_separation_clean),
        "orientation_expected": bool(check.orientation_expected),
        "image_intact": bool(check.image_intact),
        "screen_left_occupancy": float(check.screen_left_occupancy),
        "screen_right_occupancy": float(check.screen_right_occupancy),
        "occupancy_symmetry": float(check.occupancy_symmetry),
        "limb_separation_depth": float(check.limb_separation_depth),
        "limb_peak_separation_mm": float(check.limb_peak_separation_mm),
        "vertical_centroid_disparity_mm": float(check.vertical_centroid_disparity_mm),
        "panel_aspect_ratio": float(check.panel_aspect_ratio),
        "flat_row_fraction": float(check.flat_row_fraction),
        "edge_flat_run_fraction": float(check.edge_flat_run_fraction),
        "saturated_fraction": float(check.saturated_fraction),
        "structure_check_reasons": "|".join(check.reasons),
    }


def _frozen_candidate_centers(
    panel: np.ndarray,
    *,
    panel_position: str,
    spacing_mm: float,
    limit: int = REVIEW_CANDIDATE_LIMIT,
) -> list[dict[str, float]]:
    """Read the frozen localizer's own ranked alternatives so a reviewer sees what it considered.

    The frozen V3 module is used read-only here. Recording its candidate list changes no threshold,
    no selected center, and no automatic state; it only supplies review context.
    """

    candidates = localization_v3._candidate_list(
        panel, spacing_mm=spacing_mm, panel_position=panel_position
    )
    return [
        {
            "row": int(candidate.row),
            "column": int(candidate.column),
            "score": round(float(candidate.score), 6),
        }
        for candidate in candidates[:limit]
    ]


def _display_scale(width: int, target_width: int) -> float:
    return min(1.0, target_width / width) if width > 0 else 1.0


def _render_bilateral_preview(
    source: np.ndarray,
    *,
    photometric: str,
    spacing: tuple[float, float],
    results: dict[str, JointLocalizationV3],
    acquisition_index: int,
    automatic_state: str,
    destination: Path,
) -> dict[str, Any]:
    """Render the anonymous bilateral acquisition with both proposed crops and the midpoint.

    The detector margins are masked before anything is drawn, so burned-in pixel text never reaches
    the rendered image while the localization geometry stays fully visible on top of the mask.
    """

    display = normalize_for_display(source, photometric)
    display, redaction = redact_horizontal_margins(display, row_spacing_mm=spacing[0], fill_value=0)
    image = Image.fromarray(display, mode="L").convert("RGB")
    scale = _display_scale(image.width, BILATERAL_PREVIEW_WIDTH)
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    header = 26
    canvas = Image.new("RGB", (image.width, image.height + header), "black")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    midpoint = source.shape[1] // 2
    midpoint_x = round(midpoint * scale)
    draw.line((midpoint_x, header, midpoint_x, canvas.height - 1), fill="magenta", width=2)
    draw.text(
        (6, 6),
        f"V3 case {acquisition_index:04d} | automatic {automatic_state}"
        f" | {redaction.margin_mm:g} mm detector margins masked",
        fill="white",
    )
    colors = {
        QCState.PASS.value: "lime",
        QCState.BORDERLINE.value: "yellow",
        QCState.FAIL.value: "red",
    }
    for position in PANEL_POSITIONS:
        result = results[position]
        panel_offset = 0 if position == "screen_left" else midpoint
        source_row = result.row * TARGET_SPACING_MM / spacing[0]
        source_column = result.column * TARGET_SPACING_MM / spacing[1]
        x = round((panel_offset + source_column) * scale)
        y = header + round(source_row * scale)
        half_height = round((CROP_SIZE_MM / spacing[0] / 2) * scale)
        half_width = round((CROP_SIZE_MM / spacing[1] / 2) * scale)
        color = colors[result.qc_state.value]
        draw.rectangle(
            (x - half_width, y - half_height, x + half_width, y + half_height),
            outline=color,
            width=2,
        )
        draw.line((x - 10, y, x + 10, y), fill=color, width=2)
        draw.line((x, y - 10, x, y + 10), fill=color, width=2)
    atomic_png(canvas, destination)
    return {
        "width": canvas.width,
        "height": canvas.height,
        "scale": scale,
        "margin_redaction_mm": redaction.margin_mm,
        "margin_redaction_rows": redaction.rows_redacted,
    }


def _render_panel_overlay(
    panel: np.ndarray,
    *,
    result: JointLocalizationV3,
    candidates: list[dict[str, float]],
    panel_position: str,
    destination: Path,
) -> dict[str, Any]:
    """Render one resampled panel with the selected center, crop box, and ranked alternatives.

    ``panel`` must already have its detector margins redacted, because this overlay shows the whole
    panel. The returned ``scale`` converts a click in the rendered overlay back to a resampled-panel
    coordinate, which is the coordinate space the frozen crop step consumes.
    """

    normalized = np.rint(
        robust_minmax(panel, lower_percentile=0.5, upper_percentile=99.5) * 255
    ).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L").convert("RGB")
    scale = _display_scale(image.width, PANEL_OVERLAY_WIDTH)
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    header = 22
    canvas = Image.new("RGB", (image.width, image.height + header), "black")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    half = round((CROP_SIZE_MM / TARGET_SPACING_MM / 2) * scale)
    for rank, candidate in enumerate(candidates[1:], start=2):
        x = round(candidate["column"] * scale)
        y = header + round(candidate["row"] * scale)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(120, 180, 255), width=2)
        draw.text((x + 7, y - 6), str(rank), fill=(120, 180, 255))
    x = round(result.column * scale)
    y = header + round(result.row * scale)
    draw.rectangle((x - half, y - half, x + half, y + half), outline=(255, 220, 0), width=2)
    draw.line((x - 12, y, x + 12, y), fill="lime", width=2)
    draw.line((x, y - 12, x, y + 12), fill="lime", width=2)
    draw.text(
        (6, 5),
        f"{panel_position.replace('_', '-')} | {result.qc_state.value}"
        f" | anatomy {result.anatomy.state.value}"
        f" | {BURNED_IN_MARGIN_MM:g} mm margins masked",
        fill="white",
    )
    atomic_png(canvas, destination)
    return {
        "width": canvas.width,
        "height": canvas.height,
        "header_pixels": header,
        "scale": scale,
        "panel_rows": int(panel.shape[0]),
        "panel_columns": int(panel.shape[1]),
    }


def _render_crop_preview(crop: np.ndarray, *, label: str, destination: Path) -> None:
    normalized = np.rint(
        robust_minmax(crop, lower_percentile=0.5, upper_percentile=99.5) * 255
    ).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L").convert("RGB")
    image = image.resize((CROP_PREVIEW_WIDTH, CROP_PREVIEW_WIDTH), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CROP_PREVIEW_WIDTH, CROP_PREVIEW_WIDTH + 22), "black")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), label, fill="white")
    atomic_png(canvas, destination)


def _failure_record(
    *,
    acquisition_index: int,
    category: str,
    stage: str,
    error: BaseException,
    linkage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "acquisition_index": acquisition_index,
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "category": category,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **linkage,
    }


def _acquisition_state(panel_records: list[dict[str, Any]]) -> str:
    severity = max(STATE_SEVERITY[record["qc_state"]] for record in panel_records)
    return SEVERITY_STATE[severity]


def _reason_codes(
    panel_records: list[dict[str, Any]],
    structure: dict[str, Any],
    laterality_state: str,
) -> tuple[str, ...]:
    codes: list[str] = []
    for record in panel_records:
        prefix = record["panel_position"]
        for field in ("layout_reasons", "anatomical_validator_reasons"):
            codes.extend(f"{prefix}:{code}" for code in record[field].split("|") if code)
        if record["localization_state"] != QCState.PASS.value:
            codes.append(f"{prefix}:localization_{record['localization_state'].lower()}")
    codes.extend(
        f"structure:{code}" for code in structure["structure_check_reasons"].split("|") if code
    )
    if laterality_state != LateralityV2State.CONFIDENT.value:
        codes.append(f"laterality:{laterality_state.lower()}")
    return tuple(dict.fromkeys(codes))


def _verify_existing_bundle(
    output_root: Path,
    acquisition_index: int,
    *,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every artifact required for an acquisition to count as COMPLETE."""

    bundle = output_root / "crops" / f"acquisition_{acquisition_index:04d}"
    record_path = output_root / "crops" / f"acquisition_{acquisition_index:04d}" / "record.json"
    if not record_path.is_file():
        raise FrozenFullV3Error("An existing V3 crop directory lacks its completion record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("preprocessing_version") != PREPROCESSING_VERSION
        or record.get("localizer_version") != LOCALIZER_VERSION
        or record.get("laterality_policy_version") != LATERALITY_POLICY_VERSION
    ):
        raise FrozenFullV3Error("An existing V3 crop bundle has incompatible provenance")
    expected_names = {"record.json", *(f"{position}_uint16.npy" for position in PANEL_POSITIONS)}
    actual_names = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    if actual_names != expected_names:
        raise FrozenFullV3Error("An existing V3 crop bundle has an incomplete file set")
    for position in PANEL_POSITIONS:
        stored = np.load(
            _crop_path(output_root, acquisition_index, position),
            mmap_mode="r",
            allow_pickle=False,
        )
        if stored.shape != OUTPUT_SHAPE or stored.dtype != np.dtype("uint16"):
            raise FrozenFullV3Error("An existing V3 crop fails shape or dtype validation")
    panels = record.get("panels")
    if not isinstance(panels, list) or len(panels) != len(PANEL_POSITIONS):
        raise FrozenFullV3Error("An existing V3 crop bundle lacks two panel provenance rows")
    positions = [panel.get("panel_position") for panel in panels]
    if set(positions) != set(PANEL_POSITIONS) or len(positions) != len(set(positions)):
        raise FrozenFullV3Error("An existing V3 crop bundle has invalid panel positions")
    states = [panel.get("qc_state") for panel in panels]
    if any(state not in STATE_SEVERITY for state in states):
        raise FrozenFullV3Error("An existing V3 crop bundle has an invalid automatic QC state")
    expected_state = SEVERITY_STATE[max(STATE_SEVERITY[state] for state in states)]
    if record.get("automatic_state") != expected_state:
        raise FrozenFullV3Error("An existing V3 crop bundle has inconsistent acquisition QC")
    laterality_state = record.get("laterality_state")
    sides = (
        record.get("screen_left_anatomical_side"),
        record.get("screen_right_anatomical_side"),
    )
    if laterality_state == LateralityV2State.CONFIDENT.value and sides != ("R", "L"):
        raise FrozenFullV3Error("A confident existing bundle violates frozen Laterality V2")
    if laterality_state in {
        LateralityV2State.AMBIGUOUS.value,
        LateralityV2State.CONFLICTING.value,
    } and sides != (None, None):
        raise FrozenFullV3Error("An unresolved existing bundle forces anatomical laterality")
    if laterality_state not in {state.value for state in LateralityV2State}:
        raise FrozenFullV3Error("An existing V3 crop bundle has an invalid laterality state")
    if not record.get("temporary_extraction_cleaned"):
        raise FrozenFullV3Error("An existing V3 crop bundle lacks cleanup confirmation")
    if task is not None:
        if int(record.get("acquisition_index", -1)) != acquisition_index:
            raise FrozenFullV3Error("An existing V3 crop bundle has the wrong acquisition index")
        expected_fields = {
            "participant_id": task["participant_id"],
            "source_participant_id": task["source_participant_id"],
            "accession_number": task["accession_number"],
            "associated_file_reference": task["associated_file_reference"],
            "source_archive_sha256": task["archive_sha256"],
            "source_archive_bytes": int(task["archive_bytes"]),
            "frozen_localizer_source_sha256": task["frozen_localizer_source_sha256"],
        }
        if any(str(record.get(key)) != str(value) for key, value in expected_fields.items()):
            raise FrozenFullV3Error("An existing V3 crop bundle has inconsistent linkage")
    for panel in panels:
        position = panel["panel_position"]
        expected = _crop_path(output_root, acquisition_index, position).resolve()
        relative = panel.get("crop_relative_path")
        if not relative or (output_root / str(relative)).resolve() != expected:
            raise FrozenFullV3Error("An existing V3 crop bundle has inconsistent crop provenance")
    review_root = _review_root(output_root, acquisition_index)
    if record.get("review_required"):
        assets = record.get("review_assets") or {}
        expected_review_paths: set[Path] = set()
        for key in (
            "bilateral_preview",
            "screen_left_overlay",
            "screen_left_crop_preview",
            "screen_right_overlay",
            "screen_right_crop_preview",
        ):
            relative = (assets.get(key) or {}).get("path")
            if not relative or not (output_root / str(relative)).is_file():
                raise FrozenFullV3Error("An existing V3 review bundle is incomplete")
            expected_review_paths.add((output_root / str(relative)).resolve())
        actual_review_paths = {path.resolve() for path in review_root.rglob("*") if path.is_file()}
        if actual_review_paths != expected_review_paths:
            raise FrozenFullV3Error("An existing V3 review bundle contains stale artifacts")
    elif review_root.exists():
        raise FrozenFullV3Error("A non-review acquisition has stale review artifacts")
    return record


def _verify_existing_failure(
    failure_path: Path,
    *,
    acquisition_index: int,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Validate an already-recorded terminal processing failure before resuming it."""

    record = json.loads(failure_path.read_text(encoding="utf-8"))
    if int(record.get("acquisition_index", -1)) != acquisition_index:
        raise FrozenFullV3Error("An existing failure record has the wrong acquisition index")
    if record.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise FrozenFullV3Error("An existing failure record has incompatible provenance")
    for field in (
        "participant_id",
        "source_participant_id",
        "accession_number",
        "associated_file_reference",
    ):
        if str(record.get(field)) != str(task[field]):
            raise FrozenFullV3Error("An existing failure record has inconsistent linkage")
    return record


def _remove_incomplete_acquisition_outputs(
    *,
    output_root: Path,
    work_root: Path,
    acquisition_index: int,
) -> None:
    """Remove only one acquisition's incomplete Milestone 5H artifacts before retrying it."""

    exact_paths = (
        output_root / "crops" / f"acquisition_{acquisition_index:04d}",
        output_root / "review/previews" / f"acquisition_{acquisition_index:04d}",
        work_root / f"acquisition_{acquisition_index:04d}",
    )
    for path in exact_paths:
        if path.exists():
            shutil.rmtree(path)
    pending_root = output_root / "pending_bundles"
    if pending_root.is_dir():
        for path in pending_root.glob(f"acquisition_{acquisition_index:04d}.*.pending"):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def audit_recovery_state(
    acquisitions: pd.DataFrame,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify each expected acquisition without changing any crop or temporary artifact."""

    tasks = _tasks(
        acquisitions,
        archive_root=DEFAULT_ARCHIVE_ROOT,
        output_root=output_root,
        work_root=work_root,
        frozen_hash=_freeze_record_source_hash(),
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        index = int(task["acquisition_index"])
        bundle = output_root / "crops" / f"acquisition_{index:04d}"
        failure_path = output_root / "failure_records" / f"acquisition_{index:04d}.json"
        extraction = work_root / f"acquisition_{index:04d}"
        review_root = _review_root(output_root, index)
        pending_root = output_root / "pending_bundles"
        pending = (
            list(pending_root.glob(f"acquisition_{index:04d}.*.pending"))
            if pending_root.is_dir()
            else []
        )
        reasons: list[str] = []
        if bundle.is_dir():
            try:
                _verify_existing_bundle(output_root, index, task=task)
                if extraction.exists():
                    reasons.append("temporary_extraction_remains")
                if pending:
                    reasons.append("pending_bundle_remains")
                status = "COMPLETE" if not reasons else "UNVERIFIED"
            except (FrozenFullV3Error, OSError, ValueError, json.JSONDecodeError) as error:
                status = "UNVERIFIED"
                reasons.append(type(error).__name__)
        elif failure_path.is_file():
            try:
                _verify_existing_failure(failure_path, acquisition_index=index, task=task)
                status = "FAILED"
            except (FrozenFullV3Error, OSError, ValueError, json.JSONDecodeError) as error:
                status = "UNVERIFIED"
                reasons.append(type(error).__name__)
        elif extraction.exists() or review_root.exists() or pending:
            status = "PARTIAL"
            if extraction.exists():
                reasons.append("temporary_extraction_remains")
            if review_root.exists():
                reasons.append("orphan_review_assets")
            if pending:
                reasons.append("pending_bundle_remains")
        else:
            status = "MISSING"
        rows.append(
            {
                "acquisition_index": index,
                "recovery_state": status,
                "reason_codes": "|".join(reasons),
                "crop_files_present": sum(
                    _crop_path(output_root, index, position).is_file()
                    for position in PANEL_POSITIONS
                ),
                "completion_record_present": (bundle / "record.json").is_file(),
                "failure_record_present": failure_path.is_file(),
                "review_directory_present": review_root.is_dir(),
                "temporary_extraction_present": extraction.is_dir(),
                "pending_bundle_count": len(pending),
            }
        )
    frame = pd.DataFrame(rows).sort_values("acquisition_index", kind="stable", ignore_index=True)
    counts = frame["recovery_state"].value_counts().to_dict()
    summary = {
        "expected_acquisitions": len(frame),
        "complete": int(counts.get("COMPLETE", 0)),
        "partial": int(counts.get("PARTIAL", 0)),
        "unverified": int(counts.get("UNVERIFIED", 0)),
        "missing": int(counts.get("MISSING", 0)),
        "failed": int(counts.get("FAILED", 0)),
        "resume_required": int((~frame["recovery_state"].eq("COMPLETE")).sum()),
        "verified_complete_plus_resume_required": len(frame),
        "duplicate_acquisition_indices": int(frame["acquisition_index"].duplicated().sum()),
        "audit_modified_existing_outputs": False,
    }
    if write:
        recovery_root = output_root / "recovery"
        atomic_parquet(frame, recovery_root / "recovery_manifest.parquet")
        atomic_json(summary, recovery_root / "recovery_summary.json")
    return frame, summary


def storage_preflight(
    recovery: pd.DataFrame,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Estimate remaining storage and enforce a reserve before starting pixel work."""

    remaining = int((~recovery["recovery_state"].eq("COMPLETE")).sum())
    crop_file_bytes = int(np.prod(OUTPUT_SHAPE) * np.dtype("uint16").itemsize + 128)
    expected_review = round(
        remaining
        * (REPRESENTATIVE_HOLDOUT_COUNTS[QCState.BORDERLINE.value] + 1)
        / REPRESENTATIVE_HOLDOUT_TOTAL
    )
    estimated_additional = (
        remaining * len(PANEL_POSITIONS) * crop_file_bytes
        + expected_review * REVIEW_ASSET_BYTES_ESTIMATE
        + MANIFEST_AND_QC_RESERVE_BYTES
        + max(1, workers) * TEMPORARY_WORK_BYTES_PER_WORKER
    )
    free = shutil.disk_usage(output_root.parent).free
    projected_free = free - estimated_additional
    summary = {
        "verified_complete_acquisitions": int(recovery["recovery_state"].eq("COMPLETE").sum()),
        "acquisitions_requiring_processing": remaining,
        "free_bytes_before_processing": free,
        "estimated_additional_bytes": estimated_additional,
        "projected_free_bytes_after_processing": projected_free,
        "required_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES,
        "preflight_passed": projected_free >= MINIMUM_FREE_RESERVE_BYTES,
    }
    if not summary["preflight_passed"]:
        raise FrozenFullV3Error("Insufficient disk reserve for the remaining frozen V3 outputs")
    return summary


def process_acquisition(task: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen V3 pipeline for one acquisition, resuming an already-finished bundle.

    The function performs the twelve frozen steps in order and removes the temporary extracted
    DICOM before the crop bundle is published. The raw archive is only ever read.
    """

    index = int(task["acquisition_index"])
    output_root = Path(task["output_root"])
    work_root = Path(task["work_root"])
    archive_root = Path(task["archive_root"])
    linkage = {
        "participant_id": task["participant_id"],
        "source_participant_id": task["source_participant_id"],
        "accession_number": task["accession_number"],
        "associated_file_reference": task["associated_file_reference"],
        "source_archive_sha256": task["archive_sha256"],
        "source_archive_bytes": int(task["archive_bytes"]),
    }
    bundle = output_root / "crops" / f"acquisition_{index:04d}"
    failure_path = output_root / "failure_records" / f"acquisition_{index:04d}.json"
    extraction = work_root / f"acquisition_{index:04d}"
    pending_root = output_root / "pending_bundles"
    pending = (
        list(pending_root.glob(f"acquisition_{index:04d}.*.pending"))
        if pending_root.is_dir()
        else []
    )
    stale_workspace = extraction.exists()
    if bundle.is_dir():
        try:
            _verify_existing_bundle(output_root, index, task=task)
            if extraction.exists() or pending:
                raise FrozenFullV3Error("A completed bundle has interrupted temporary artifacts")
            return {"acquisition_index": index, "status": "resumed_success"}
        except (FrozenFullV3Error, OSError, ValueError, json.JSONDecodeError):
            _remove_incomplete_acquisition_outputs(
                output_root=output_root,
                work_root=work_root,
                acquisition_index=index,
            )
            failure_path.unlink(missing_ok=True)
    elif failure_path.is_file():
        try:
            _verify_existing_failure(failure_path, acquisition_index=index, task=task)
            return {"acquisition_index": index, "status": "resumed_failure"}
        except (FrozenFullV3Error, OSError, ValueError, json.JSONDecodeError):
            failure_path.unlink(missing_ok=True)
            _remove_incomplete_acquisition_outputs(
                output_root=output_root,
                work_root=work_root,
                acquisition_index=index,
            )
    elif extraction.exists() or _review_root(output_root, index).exists() or pending:
        _remove_incomplete_acquisition_outputs(
            output_root=output_root,
            work_root=work_root,
            acquisition_index=index,
        )

    def failed(category: str, stage: str, error: BaseException) -> dict[str, Any]:
        record = _failure_record(
            acquisition_index=index,
            category=category,
            stage=stage,
            error=error,
            linkage=linkage,
        )
        atomic_json(record, failure_path)
        return {"acquisition_index": index, "status": "failed", "category": category}

    if stale_workspace:
        # The exact interrupted workspace was already removed above when it made the acquisition
        # PARTIAL. Keep the flag for provenance without deleting a second time.
        if extraction.exists():
            shutil.rmtree(extraction)
    archive_path = archive_root / task["archive_relative_path"]
    try:
        if not archive_path.is_file():
            raise FrozenFullV3Error("The expected source archive is missing")
        if archive_path.stat().st_size != int(task["archive_bytes"]):
            raise FrozenFullV3Error("The source archive size differs from package provenance")
        if sha256_file(archive_path) != str(task["archive_sha256"]):
            raise FrozenFullV3Error("The source archive hash differs from package provenance")
    except (OSError, ValueError, FrozenFullV3Error) as error:
        return failed("ARCHIVE_FAILURE", "verify_source_archive", error)
    try:
        contents = safe_extract_archive(archive_path, extraction)
    except (UnsafeArchiveError, FileExistsError, OSError, ValueError) as error:
        return failed("ARCHIVE_FAILURE", "validate_and_extract_archive", error)

    try:
        try:
            if contents.regular_files != 1:
                raise FrozenFullV3Error("Archive does not contain exactly one regular file")
            dicom_path = _single_extracted_file(extraction)
            metadata = inspect_dicom(dicom_path)
            dataset = pydicom.dcmread(dicom_path, force=False)
            if int(dataset.get("NumberOfFrames", 1)) != 1:
                raise FrozenFullV3Error("DICOM is not single-frame")
            source = np.asarray(dataset.pixel_array)
            if source.ndim != 2:
                raise FrozenFullV3Error("DICOM pixel array is not two-dimensional")
            if source.dtype != np.dtype("uint16"):
                raise FrozenFullV3Error("DICOM pixel array is not uint16")
            if metadata.rows != source.shape[0] or metadata.columns != source.shape[1]:
                raise FrozenFullV3Error("DICOM metadata and decoded dimensions disagree")
            spacing = _spacing(metadata)
            dicom_sha256 = sha256_file(dicom_path)
        except Exception as error:  # noqa: BLE001 - every stage failure is queued, never fatal
            return failed("DICOM_FAILURE", "verify_readable_dicom", error)

        try:
            screen_left, screen_right = split_bilateral_midpoint(source)
            if (
                screen_left.shape[0] != source.shape[0]
                or screen_right.shape[0] != source.shape[0]
                or screen_left.shape[1] + screen_right.shape[1] != source.shape[1]
            ):
                raise FrozenFullV3Error("Midpoint panels do not preserve source geometry")
            panels = {"screen_left": screen_left, "screen_right": screen_right}
        except Exception as error:  # noqa: BLE001
            return failed("MIDPOINT_FAILURE", "midpoint_bilateral_split", error)

        try:
            structure = assess_bilateral_structure(
                source,
                row_spacing_mm=spacing[0],
                column_spacing_mm=spacing[1],
            )
            laterality = assess_validated_oai_screen_rule(
                bilateral_acquisition=bool(task["bilateral_acquisition"])
                and structure.bilateral_structure_confirmed,
                midpoint_separation_clean=structure.midpoint_separation_clean,
                exception_detector_passed=structure.passed,
                dicom_laterality=metadata.laterality,
                image_laterality=metadata.image_laterality,
            )
        except Exception as error:  # noqa: BLE001
            return failed("LATERALITY_FAILURE", "frozen_laterality_v2_mapping", error)

        try:
            resampled = {
                position: resample_to_spacing(
                    panel, spacing, (TARGET_SPACING_MM, TARGET_SPACING_MM)
                )
                for position, panel in panels.items()
            }
        except Exception as error:  # noqa: BLE001
            return failed("RESAMPLE_FAILURE", "resample_to_target_spacing", error)

        try:
            left, right = localize_bilateral_tibiofemoral_joints_v3(
                resampled["screen_left"],
                resampled["screen_right"],
                spacing_mm=TARGET_SPACING_MM,
            )
            results = {"screen_left": left, "screen_right": right}
        except Exception as error:  # noqa: BLE001
            return failed("LOCALIZATION_FAILURE", "frozen_localization_v3", error)

        screen_sides = {
            "screen_left": laterality.screen_left_anatomical_side,
            "screen_right": laterality.screen_right_anatomical_side,
        }
        crops: dict[str, np.ndarray] = {}
        padding_values: dict[str, int] = {}
        panel_records: list[dict[str, Any]] = []
        try:
            for position in PANEL_POSITIONS:
                result = results[position]
                panel = resampled[position]
                padding_value = int(np.percentile(panel, 0.5))
                padding_values[position] = padding_value
                crop, geometry = crop_around_center(
                    panel,
                    (result.row, result.column),
                    OUTPUT_SHAPE,
                    padding_value=padding_value,
                )
                if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                    raise FrozenFullV3Error("Crop does not match the frozen output specification")
                crops[position] = crop
                horizontal = geometry.padding_left + geometry.padding_right
                vertical = geometry.padding_top + geometry.padding_bottom
                panel_records.append(
                    {
                        "panel_position": position,
                        "anatomical_side": screen_sides[position],
                        **_serialize_localization(result),
                        "crop_rows_from_detector_margin": crop_rows_inside_margin(
                            center_row=int(result.row),
                            crop_rows=OUTPUT_SHAPE[0],
                            panel_rows=int(panel.shape[0]),
                            row_spacing_mm=TARGET_SPACING_MM,
                        ),
                        "resampled_rows": int(panel.shape[0]),
                        "resampled_columns": int(panel.shape[1]),
                        "crop_padding_value": padding_value,
                        "padding_top": int(geometry.padding_top),
                        "padding_bottom": int(geometry.padding_bottom),
                        "padding_left": int(geometry.padding_left),
                        "padding_right": int(geometry.padding_right),
                        "horizontal_padding_mm": horizontal * TARGET_SPACING_MM,
                        "vertical_padding_mm": vertical * TARGET_SPACING_MM,
                        "crop_relative_path": _crop_path(output_root, index, position)
                        .relative_to(output_root)
                        .as_posix(),
                    }
                )
        except Exception as error:  # noqa: BLE001
            return failed("CROP_FAILURE", "frozen_160mm_crop", error)

        structure_record = _serialize_structure(structure)
        automatic_state = _acquisition_state(panel_records)
        laterality_resolved = laterality.state is LateralityV2State.CONFIDENT
        review_required = automatic_state != QCState.PASS.value or not laterality_resolved
        reason_codes = _reason_codes(panel_records, structure_record, laterality.state.value)

        review_assets: dict[str, Any] = {}
        if review_required:
            try:
                review_root = _review_root(output_root, index)
                bilateral = _render_bilateral_preview(
                    source,
                    photometric=str(metadata.photometric_interpretation),
                    spacing=spacing,
                    results=results,
                    acquisition_index=index,
                    automatic_state=automatic_state,
                    destination=review_root / "bilateral.png",
                )
                review_assets["bilateral_preview"] = {
                    "path": (review_root / "bilateral.png").relative_to(output_root).as_posix(),
                    **bilateral,
                }
                for position in PANEL_POSITIONS:
                    candidates = _frozen_candidate_centers(
                        resampled[position],
                        panel_position=position,
                        spacing_mm=TARGET_SPACING_MM,
                    )
                    # Rendered review assets come from a margin-redacted copy of the panel; the
                    # stored crop above keeps the frozen, unmodified pixels.
                    review_panel, _ = redact_horizontal_margins(
                        resampled[position],
                        row_spacing_mm=TARGET_SPACING_MM,
                        fill_value=padding_values[position],
                    )
                    overlay_path = review_root / f"overlay_{position}.png"
                    overlay = _render_panel_overlay(
                        review_panel,
                        result=results[position],
                        candidates=candidates,
                        panel_position=position,
                        destination=overlay_path,
                    )
                    review_crop, _ = crop_around_center(
                        review_panel,
                        (results[position].row, results[position].column),
                        OUTPUT_SHAPE,
                        padding_value=padding_values[position],
                    )
                    crop_preview_path = review_root / f"crop_{position}.png"
                    _render_crop_preview(
                        review_crop,
                        label=(
                            f"{position.replace('_', '-')} crop"
                            f" | {results[position].qc_state.value}"
                        ),
                        destination=crop_preview_path,
                    )
                    review_assets[f"{position}_overlay"] = {
                        "path": overlay_path.relative_to(output_root).as_posix(),
                        **overlay,
                    }
                    review_assets[f"{position}_crop_preview"] = {
                        "path": crop_preview_path.relative_to(output_root).as_posix()
                    }
                    review_assets[f"{position}_candidate_centers"] = candidates
            except Exception as error:  # noqa: BLE001
                return failed("REVIEW_ASSET_FAILURE", "render_review_assets", error)

        record: dict[str, Any] = {
            **linkage,
            "acquisition_index": index,
            "dataset_version": DATASET_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "localizer_version": LOCALIZER_VERSION,
            "frozen_localizer_source_sha256": task["frozen_localizer_source_sha256"],
            "laterality_policy_version": LATERALITY_POLICY_VERSION,
            "archive_members": contents.members,
            "archive_regular_files": contents.regular_files,
            "archive_directories": contents.directories,
            "archive_declared_file_bytes": contents.declared_file_bytes,
            "stale_workspace_reclaimed": stale_workspace,
            "dicom_sha256": dicom_sha256,
            "dicom_file_bytes": metadata.file_size_bytes,
            "original_rows": int(source.shape[0]),
            "original_columns": int(source.shape[1]),
            "original_row_spacing_mm": spacing[0],
            "original_column_spacing_mm": spacing[1],
            "target_spacing_mm": TARGET_SPACING_MM,
            "crop_size_mm": CROP_SIZE_MM,
            "crop_rows": OUTPUT_SHAPE[0],
            "crop_columns": OUTPUT_SHAPE[1],
            "crop_dtype": "uint16",
            "manufacturer": metadata.manufacturer,
            "manufacturer_model_name": metadata.manufacturer_model_name,
            "photometric_interpretation": metadata.photometric_interpretation,
            "bits_allocated": metadata.bits_allocated,
            "bits_stored": metadata.bits_stored,
            "pixel_representation": metadata.pixel_representation,
            "transfer_syntax_uid": metadata.transfer_syntax_uid,
            "compressed": metadata.compressed,
            "image_release_study": task["image_release_study"],
            "xray_accept_qc": task["xray_accept_qc"],
            "xray_alignment_problem": task["xray_alignment_problem"],
            "xray_centering_problem": task["xray_centering_problem"],
            "xray_incomplete_depiction": task["xray_incomplete_depiction"],
            "xray_positioning_problem": task["xray_positioning_problem"],
            "eligible_knee_sides": task["eligible_knee_sides"],
            "eligible_knee_count": int(task["eligible_knee_count"]),
            "midpoint_state": "SUCCESS",
            **structure_record,
            "laterality_state": laterality.state.value,
            "laterality_reason": laterality.reason,
            "laterality_evidence_source": laterality.evidence_source,
            "unilateral_dicom_tag_artifact": bool(laterality.unilateral_dicom_tag_artifact),
            "screen_left_anatomical_side": laterality.screen_left_anatomical_side,
            "screen_right_anatomical_side": laterality.screen_right_anatomical_side,
            "automatic_state": automatic_state,
            "review_required": review_required,
            "review_reason_codes": "|".join(reason_codes),
            "review_margin_redaction_mm": BURNED_IN_MARGIN_MM,
            "panels": panel_records,
            "review_assets": review_assets,
            "temporary_extraction_cleaned": False,
        }

        pending_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            os.path.join(pending_root, f"acquisition_{index:04d}.{os.getpid()}.pending")
        )
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            for position in PANEL_POSITIONS:
                np.save(temporary / f"{position}_uint16.npy", crops[position], allow_pickle=False)
                stored = np.load(temporary / f"{position}_uint16.npy", mmap_mode="r")
                if stored.shape != OUTPUT_SHAPE or stored.dtype != np.dtype("uint16"):
                    raise FrozenFullV3Error("Stored crop fails post-write validation")
            shutil.rmtree(extraction)
            if extraction.exists():
                raise FrozenFullV3Error("Temporary DICOM cleanup did not complete")
            record["temporary_extraction_cleaned"] = True
            atomic_json(record, temporary / "record.json")
            if bundle.exists():
                raise FrozenFullV3Error("A crop destination appeared during processing")
            bundle.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(bundle)
        except Exception as error:  # noqa: BLE001
            return failed("STORAGE_FAILURE", "publish_crop_bundle", error)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
    finally:
        if extraction.exists():
            shutil.rmtree(extraction, ignore_errors=True)
    return {
        "acquisition_index": index,
        "status": "processed",
        "automatic_state": automatic_state,
        "review_required": review_required,
    }


def _tasks(
    acquisitions: pd.DataFrame,
    *,
    archive_root: Path,
    output_root: Path,
    work_root: Path,
    frozen_hash: str,
) -> list[dict[str, Any]]:
    columns = [
        "acquisition_index",
        "participant_id",
        "source_participant_id",
        "accession_number",
        "associated_file_reference",
        "archive_relative_path",
        "archive_sha256",
        "archive_bytes",
        "image_release_study",
        "xray_accept_qc",
        "xray_alignment_problem",
        "xray_centering_problem",
        "xray_incomplete_depiction",
        "xray_positioning_problem",
        "eligible_knee_sides",
        "eligible_knee_count",
        "bilateral_acquisition",
    ]
    frame = acquisitions.reindex(columns=columns)
    # Missing OAI QC flags must reach the provenance record as null, not as a float NaN.
    records = frame.astype(object).where(frame.notna(), None).to_dict("records")
    for record in records:
        record["archive_root"] = str(archive_root)
        record["output_root"] = str(output_root)
        record["work_root"] = str(work_root)
        record["frozen_localizer_source_sha256"] = frozen_hash
    return records


def load_processing_inputs(
    *,
    acquisition_input_path: Path = DEFAULT_ACQUISITION_INPUT,
    archive_ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
    image_manifest_path: Path = DEFAULT_IMAGE_MANIFEST,
    expected_acquisitions: int = EXPECTED_ACQUISITIONS,
) -> pd.DataFrame:
    """Join the preserved archive ledger to the V1 acquisition linkage without altering either."""

    ledger = pd.read_parquet(archive_ledger_path)
    acquisitions = pd.read_parquet(acquisition_input_path)
    if len(ledger) != expected_acquisitions or len(acquisitions) != expected_acquisitions:
        raise FrozenFullV3Error("Archive ledger and acquisition linkage are not the full set")
    join_keys = ["acquisition_index", "associated_file_reference"]
    duplicated = [
        column
        for column in acquisitions.columns
        if column in ledger.columns and column not in join_keys
    ]
    merged = ledger.merge(
        acquisitions.drop(columns=duplicated),
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != expected_acquisitions:
        raise FrozenFullV3Error("Archive ledger does not align with the acquisition linkage")
    knees = pd.read_parquet(
        image_manifest_path, columns=["accession_number", "bilateral_acquisition"]
    )
    bilateral = knees.groupby("accession_number")["bilateral_acquisition"].all()
    merged["bilateral_acquisition"] = (
        merged["accession_number"].map(bilateral).fillna(False).astype(bool)
    )
    if not merged["bilateral_acquisition"].all():
        raise FrozenFullV3Error("The preserved image manifest contains a non-bilateral acquisition")
    return merged.sort_values("acquisition_index", kind="stable", ignore_index=True)


def run_full_reprocessing(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    acquisition_input_path: Path = DEFAULT_ACQUISITION_INPUT,
    archive_ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    workers: int | None = None,
    expected_acquisitions: int = EXPECTED_ACQUISITIONS,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Run or resume the frozen full-cohort V3 pass over every baseline acquisition."""

    gate_path = output_root / "audit/preservation_gate.json"
    if not gate_path.is_file():
        raise FrozenFullV3Error("The preservation gate must run before full V3 reprocessing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_passed"):
        raise FrozenFullV3Error("The recorded preservation gate did not pass")
    frozen_hash = str(gate["frozen_localizer_source_sha256"])
    recorded_source_hashes = gate.get("frozen_source_hashes") or {}
    for source in FROZEN_SOURCE_FILES:
        expected_hash = recorded_source_hashes.get(source.as_posix())
        if not expected_hash or sha256_file(source) != expected_hash:
            raise FrozenFullV3Error(f"Frozen source integrity failed for {source.name}")
    if recorded_source_hashes.get(FROZEN_SOURCE_FILES[0].as_posix()) != frozen_hash:
        raise FrozenFullV3Error("The frozen localization V3 source changed after the gate ran")

    acquisitions = load_processing_inputs(
        acquisition_input_path=acquisition_input_path,
        archive_ledger_path=archive_ledger_path,
        expected_acquisitions=expected_acquisitions,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = max(1, min(6, (os.cpu_count() or 2) - 2))
    recovery, recovery_summary = audit_recovery_state(
        acquisitions,
        output_root=output_root,
        work_root=work_root,
        write=True,
    )
    preflight = storage_preflight(
        recovery,
        output_root=output_root,
        workers=workers,
    )
    atomic_json(preflight, output_root / "recovery/storage_preflight.json")
    tasks = _tasks(
        acquisitions,
        archive_root=archive_root,
        output_root=output_root,
        work_root=work_root,
        frozen_hash=frozen_hash,
    )
    tallies = {
        "processed": 0,
        "resumed_success": 0,
        "resumed_failure": 0,
        "failed": 0,
    }
    completed = 0
    checkpoint_path = output_root / "manifests/processing_checkpoint.json"

    def checkpoint() -> None:
        atomic_json(
            {
                "expected_acquisitions": expected_acquisitions,
                "considered": completed,
                **tallies,
                "verified_complete_before_run": recovery_summary["complete"],
                "partial_before_run": recovery_summary["partial"],
                "unverified_before_run": recovery_summary["unverified"],
                "missing_before_run": recovery_summary["missing"],
                "checkpoint_is_authoritative": False,
                "authoritative_completion_evidence": "verified atomic acquisition bundles",
            },
            checkpoint_path,
        )

    checkpoint()
    if workers == 1:
        results = map(process_acquisition, tasks)
        for result in results:
            tallies[result["status"]] += 1
            completed += 1
            if completed % progress_every == 0 or completed == len(tasks):
                checkpoint()
                print(json.dumps({"considered": completed, **tallies}, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(process_acquisition, tasks, chunksize=4):
                tallies[result["status"]] += 1
                completed += 1
                if completed % progress_every == 0 or completed == len(tasks):
                    checkpoint()
                    print(
                        json.dumps({"considered": completed, **tallies}, sort_keys=True), flush=True
                    )

    successes = sorted((output_root / "crops").glob("acquisition_*/record.json"))
    failures = sorted((output_root / "failure_records").glob("acquisition_*.json"))
    pending = output_root / "pending_bundles"
    leftover_pending = (
        [path for path in pending.rglob("*") if path.is_file()] if pending.is_dir() else []
    )
    work_files = [path for path in work_root.rglob("*") if path.is_file()]
    result = {
        "expected_acquisitions": expected_acquisitions,
        "successful_acquisitions": len(successes),
        "failed_acquisitions": len(failures),
        "workers": workers,
        "verified_complete_before_run": recovery_summary["complete"],
        "partial_before_run": recovery_summary["partial"],
        "unverified_before_run": recovery_summary["unverified"],
        "missing_before_run": recovery_summary["missing"],
        "storage_preflight": preflight,
        **tallies,
        "temporary_extraction_files_remaining": len(work_files),
        "pending_bundle_files_remaining": len(leftover_pending),
        "raw_archives_modified": False,
    }
    atomic_json(result, output_root / "manifests/processing_run.json")
    return result


# ---------------------------------------------------------------------------
# D/E. Full manifest and manual-review queue
# ---------------------------------------------------------------------------


def _load_records(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "crops").glob("acquisition_*/record.json"))
    ]
    failures = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "failure_records").glob("acquisition_*.json"))
    ]
    return successes, failures


def _panel_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    eligible_sides = {
        value for value in str(record.get("eligible_knee_sides") or "").split(",") if value
    }
    rows: list[dict[str, Any]] = []
    for panel in record["panels"]:
        side = panel["anatomical_side"]
        knee_side_code = ANATOMICAL_SIDE_TO_KNEE_SIDE_CODE.get(side) if side else None
        if knee_side_code is None:
            linkage_status = "UNRESOLVED_LATERALITY"
            eligible: bool | None = None
        elif knee_side_code in eligible_sides:
            linkage_status = "RESOLVED_ELIGIBLE"
            eligible = True
        else:
            linkage_status = "RESOLVED_NOT_IN_ANALYSIS_COHORT"
            eligible = False
        rows.append(
            {
                "acquisition_index": record["acquisition_index"],
                "participant_id": record["participant_id"],
                "source_participant_id": record["source_participant_id"],
                "accession_number": record["accession_number"],
                "associated_file_reference": record["associated_file_reference"],
                "screen_panel": panel["panel_position"],
                "anatomical_side": side,
                "knee_side_code": knee_side_code,
                "participant_knee_linkage_status": linkage_status,
                "eligible_analysis_knee": eligible,
                "crop_relative_path": panel["crop_relative_path"],
                "dataset_version": record["dataset_version"],
                "preprocessing_version": record["preprocessing_version"],
                "localizer_version": record["localizer_version"],
                "frozen_localizer_source_sha256": record["frozen_localizer_source_sha256"],
                "laterality_policy_version": record["laterality_policy_version"],
                "laterality_state": record["laterality_state"],
                "laterality_reason": record["laterality_reason"],
                "laterality_evidence_source": record["laterality_evidence_source"],
                "unilateral_dicom_tag_artifact": record["unilateral_dicom_tag_artifact"],
                "acquisition_automatic_state": record["automatic_state"],
                "review_required": record["review_required"],
                "review_reason_codes": record["review_reason_codes"],
                "original_rows": record["original_rows"],
                "original_columns": record["original_columns"],
                "original_row_spacing_mm": record["original_row_spacing_mm"],
                "original_column_spacing_mm": record["original_column_spacing_mm"],
                "target_spacing_mm": record["target_spacing_mm"],
                "crop_size_mm": record["crop_size_mm"],
                "crop_rows": record["crop_rows"],
                "crop_columns": record["crop_columns"],
                "crop_dtype": record["crop_dtype"],
                "manufacturer": record["manufacturer"],
                "manufacturer_model_name": record["manufacturer_model_name"],
                "image_release_study": record["image_release_study"],
                "xray_accept_qc": record["xray_accept_qc"],
                "bilateral_structure_confirmed": record["bilateral_structure_confirmed"],
                "midpoint_separation_clean": record["midpoint_separation_clean"],
                "structure_check_passed": record["structure_check_passed"],
                "structure_check_reasons": record["structure_check_reasons"],
                **{
                    key: value
                    for key, value in panel.items()
                    if key
                    not in {
                        "panel_position",
                        "anatomical_side",
                        "crop_relative_path",
                    }
                },
            }
        )
    return rows


def _candidate_text(assets: dict[str, Any], position: str) -> str:
    candidates = assets.get(f"{position}_candidate_centers") or []
    return json.dumps(candidates, sort_keys=True)


def _review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queued = [record for record in records if record["review_required"]]
    queued.sort(key=lambda record: int(record["acquisition_index"]))
    rows: list[dict[str, Any]] = []
    for review_index, record in enumerate(queued, start=1):
        assets = record.get("review_assets") or {}
        panels = {panel["panel_position"]: panel for panel in record["panels"]}
        bilateral = assets.get("bilateral_preview") or {}
        row: dict[str, Any] = {
            "review_index": review_index,
            "acquisition_index": record["acquisition_index"],
            "participant_id": record["participant_id"],
            "source_participant_id": record["source_participant_id"],
            "accession_number": record["accession_number"],
            "associated_file_reference": record["associated_file_reference"],
            "dataset_version": record["dataset_version"],
            "localizer_version": record["localizer_version"],
            "laterality_policy_version": record["laterality_policy_version"],
            "automatic_state": record["automatic_state"],
            "review_reason_codes": record["review_reason_codes"],
            "laterality_state": record["laterality_state"],
            "laterality_reason": record["laterality_reason"],
            "structure_check_passed": record["structure_check_passed"],
            "structure_check_reasons": record["structure_check_reasons"],
            "bilateral_preview_path": bilateral.get("path"),
            "queue_reason_count": len(
                [code for code in str(record["review_reason_codes"]).split("|") if code]
            ),
        }
        for position in PANEL_POSITIONS:
            panel = panels[position]
            overlay = assets.get(f"{position}_overlay") or {}
            crop_preview = assets.get(f"{position}_crop_preview") or {}
            row.update(
                {
                    f"{position}_crop_path": panel["crop_relative_path"],
                    f"{position}_overlay_path": overlay.get("path"),
                    f"{position}_crop_preview_path": crop_preview.get("path"),
                    f"{position}_overlay_scale": overlay.get("scale"),
                    f"{position}_overlay_header_pixels": overlay.get("header_pixels"),
                    f"{position}_resampled_rows": panel["resampled_rows"],
                    f"{position}_resampled_columns": panel["resampled_columns"],
                    f"{position}_joint_row_resampled": panel["joint_row_resampled"],
                    f"{position}_joint_column_resampled": panel["joint_column_resampled"],
                    f"{position}_candidate_centers": _candidate_text(assets, position),
                    f"{position}_qc_state": panel["qc_state"],
                    f"{position}_localization_state": panel["localization_state"],
                    f"{position}_layout_state": panel["layout_state"],
                    f"{position}_anatomical_validator_state": panel["anatomical_validator_state"],
                    f"{position}_localization_confidence": panel["localization_confidence"],
                    f"{position}_horizontal_padding_mm": panel["horizontal_padding_mm"],
                    f"{position}_vertical_padding_mm": panel["vertical_padding_mm"],
                    f"{position}_anatomical_side": panel["anatomical_side"],
                }
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# G. Full-cohort QC
# ---------------------------------------------------------------------------


def _pooled_groups(values: pd.Series) -> pd.Series:
    text = values.fillna("<missing>").astype(str).replace("", "<missing>")
    counts = text.value_counts()
    small = set(counts[counts < MIN_REPORTABLE_GROUP].index)
    return text.map(lambda value: "<pooled groups n<5>" if value in small else value)


def _wilson_interval(successes: int, total: int, z: float = COMPATIBILITY_Z) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z
        * float(np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)))
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def _state_counts(states: pd.Series) -> dict[str, int]:
    counts = states.value_counts().to_dict()
    return {state: int(counts.get(state, 0)) for state in STATE_SEVERITY}


def _compatibility_report(states: pd.Series) -> dict[str, Any]:
    total = len(states)
    counts = _state_counts(states)
    entries: list[dict[str, Any]] = []
    for state, reference in REPRESENTATIVE_HOLDOUT_COUNTS.items():
        low, high = _wilson_interval(reference, REPRESENTATIVE_HOLDOUT_TOTAL)
        observed = counts[state] / total if total else 0.0
        entries.append(
            {
                "state": state,
                "full_cohort_acquisitions": counts[state],
                "full_cohort_percent": round(observed * 100, 3),
                "representative_estimate_percent": round(
                    reference / REPRESENTATIVE_HOLDOUT_TOTAL * 100, 3
                ),
                "representative_99_percent_interval": [
                    round(low * 100, 3),
                    round(high * 100, 3),
                ],
                "difference_percentage_points": round(
                    (observed - reference / REPRESENTATIVE_HOLDOUT_TOTAL) * 100, 3
                ),
                "within_representative_interval": bool(low <= observed <= high),
            }
        )
    return {
        "reference_sample": "V3 representative holdout (192 acquisitions)",
        "reference_interval_method": "Wilson 99% interval on the representative proportion",
        "exact_equality_required": False,
        "states": entries,
        "compatible_with_representative_estimate": all(
            entry["within_representative_interval"] for entry in entries
        ),
    }


def _state_distribution(
    acquisitions: pd.DataFrame, panels: pd.DataFrame, column: str, label: str
) -> list[dict[str, Any]]:
    pooled = _pooled_groups(acquisitions[column])
    entries: list[dict[str, Any]] = []
    for value in sorted(pooled.unique()):
        subset = acquisitions.loc[pooled.eq(value)]
        indices = set(subset["acquisition_index"].astype(int))
        panel_subset = panels.loc[panels["acquisition_index"].isin(indices)]
        counts = _state_counts(subset["automatic_state"])
        total = len(subset)
        entries.append(
            {
                "grouping": label,
                "group": value,
                "acquisitions": total,
                "panels": len(panel_subset),
                **{f"{state.lower()}_acquisitions": counts[state] for state in STATE_SEVERITY},
                **{
                    f"{state.lower()}_percent": round(counts[state] / total * 100, 3)
                    for state in STATE_SEVERITY
                },
                "review_required_acquisitions": int(subset["review_required"].sum()),
                "panels_with_horizontal_padding": int(
                    panel_subset["horizontal_padding_mm"].gt(0).sum()
                ),
                "panels_with_vertical_padding": int(
                    panel_subset["vertical_padding_mm"].gt(0).sum()
                ),
                "maximum_horizontal_padding_mm": round(
                    float(panel_subset["horizontal_padding_mm"].max() or 0.0), 3
                ),
                "maximum_vertical_padding_mm": round(
                    float(panel_subset["vertical_padding_mm"].max() or 0.0), 3
                ),
            }
        )
    return entries


def _flagged_subgroups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for entry in entries:
        reasons: list[str] = []
        if entry["fail_percent"] > SUBGROUP_FAIL_RATE_LIMIT * 100:
            reasons.append("elevated_automatic_fail_rate")
        if entry["pass_percent"] < SUBGROUP_PASS_RATE_FLOOR * 100:
            reasons.append("depressed_automatic_pass_rate")
        if reasons:
            flagged.append({**entry, "flag_reasons": reasons})
    return flagged


def _padding_statistics(panels: pd.DataFrame) -> dict[str, Any]:
    def describe(column: str) -> dict[str, Any]:
        values = panels[column].astype(float)
        padded = values[values > 0]
        return {
            "panels_with_padding": int((values > 0).sum()),
            "panels_with_padding_percent": round(float((values > 0).mean()) * 100, 3),
            "maximum_mm": round(float(values.max()), 3),
            "median_padded_mm": round(float(padded.median()), 3) if len(padded) else 0.0,
            "mean_padded_mm": round(float(padded.mean()), 3) if len(padded) else 0.0,
            "percentile_99_mm": round(float(np.percentile(values, 99)), 3),
        }

    return {
        "horizontal": describe("horizontal_padding_mm"),
        "vertical": describe("vertical_padding_mm"),
        "panels_with_any_padding": int(
            (panels["horizontal_padding_mm"].gt(0) | panels["vertical_padding_mm"].gt(0)).sum()
        ),
    }


def _directory_bytes(root: Path, pattern: str = "*") -> dict[str, int]:
    if not root.exists():
        return {"files": 0, "bytes": 0}
    paths = [path for path in root.rglob(pattern) if path.is_file()]
    return {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths)}


def finalize_frozen_full_v3(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    expected_acquisitions: int = EXPECTED_ACQUISITIONS,
) -> dict[str, Any]:
    """Write the full V3 manifest, the manual-review queue, and the full-cohort QC report."""

    records, failures = _load_records(output_root)
    if not records:
        raise FrozenFullV3Error("No successful V3 acquisitions are available to finalize")
    if len(records) + len(failures) != expected_acquisitions:
        raise FrozenFullV3Error("Full V3 processing is incomplete and cannot be finalized")
    record_indices = [int(record["acquisition_index"]) for record in records]
    failure_indices = [int(record["acquisition_index"]) for record in failures]
    if len(record_indices) != len(set(record_indices)) or len(failure_indices) != len(
        set(failure_indices)
    ):
        raise FrozenFullV3Error("Full V3 processing contains duplicate acquisition records")
    if set(record_indices).intersection(failure_indices):
        raise FrozenFullV3Error("An acquisition has both success and failure records")
    for index in record_indices:
        _verify_existing_bundle(output_root, index)
    pending_files = (
        [path for path in (output_root / "pending_bundles").rglob("*") if path.is_file()]
        if (output_root / "pending_bundles").exists()
        else []
    )
    if pending_files:
        raise FrozenFullV3Error("Pending crop bundles remain and finalization is unsafe")
    acquisitions = pd.DataFrame(
        [
            {key: value for key, value in record.items() if key not in {"panels", "review_assets"}}
            for record in records
        ]
    ).sort_values("acquisition_index", kind="stable", ignore_index=True)
    panels = pd.DataFrame([row for record in records for row in _panel_rows(record)]).sort_values(
        ["acquisition_index", "screen_panel"], kind="stable", ignore_index=True
    )
    if panels["crop_relative_path"].duplicated().any():
        raise FrozenFullV3Error("Panel crop paths are not unique")
    if len(panels) != 2 * len(acquisitions):
        raise FrozenFullV3Error("Each acquisition must contribute exactly two knee panels")

    manifest_root = output_root / "manifests"
    atomic_parquet(acquisitions, manifest_root / "v3_frozen_full_acquisitions.parquet")
    atomic_parquet(panels, manifest_root / "v3_frozen_full_panels.parquet")
    review_queue = pd.DataFrame(_review_rows(records))
    if not review_queue.empty:
        review_queue = review_queue.sort_values("review_index", kind="stable", ignore_index=True)
    atomic_parquet(review_queue, output_root / "review_queue.parquet")

    failure_frame = pd.DataFrame(failures) if failures else pd.DataFrame()
    if not failure_frame.empty:
        atomic_parquet(
            failure_frame.sort_values("acquisition_index", kind="stable", ignore_index=True),
            manifest_root / "geometric_preprocessing_failures.parquet",
        )

    exception_root = output_root / "exceptions"
    empty_failure = pd.DataFrame(
        columns=["acquisition_index", "category", "stage", "error_type", "error_message"]
    )
    failure_categories = {
        "archive_failure": "ARCHIVE_FAILURE",
        "dicom_failure": "DICOM_FAILURE",
        "midpoint_failure": "MIDPOINT_FAILURE",
        "resample_failure": "RESAMPLE_FAILURE",
        "localization_failure": "LOCALIZATION_FAILURE",
        "crop_failure": "CROP_FAILURE",
        "laterality_failure": "LATERALITY_FAILURE",
        "review_asset_failure": "REVIEW_ASSET_FAILURE",
        "storage_failure": "STORAGE_FAILURE",
    }
    for name, category in failure_categories.items():
        subset = (
            failure_frame.loc[failure_frame["category"].eq(category)]
            if not failure_frame.empty
            else empty_failure.copy()
        )
        atomic_parquet(subset, exception_root / f"{name}.parquet")
    automatic_review = acquisitions.loc[
        ~acquisitions["automatic_state"].eq(QCState.PASS.value)
    ].copy()
    atomic_parquet(automatic_review, exception_root / "automatic_localization_review.parquet")
    crop_inadequacy = panels.loc[
        ~panels["anatomical_validator_state"].eq(QCState.PASS.value)
    ].copy()
    atomic_parquet(crop_inadequacy, exception_root / "crop_inadequacy.parquet")
    excessive_padding = panels.loc[
        panels["horizontal_padding_mm"].gt(VALIDATED_MAX_HORIZONTAL_PADDING_MM)
        | panels["vertical_padding_mm"].gt(VALIDATED_MAX_VERTICAL_PADDING_MM)
    ].copy()
    atomic_parquet(excessive_padding, exception_root / "excessive_padding.parquet")
    atomic_parquet(
        acquisitions.loc[acquisitions["laterality_state"].eq(LateralityV2State.AMBIGUOUS.value)],
        exception_root / "laterality_ambiguity.parquet",
    )
    atomic_parquet(
        acquisitions.loc[acquisitions["laterality_state"].eq(LateralityV2State.CONFLICTING.value)],
        exception_root / "laterality_conflict.parquet",
    )

    states = acquisitions["automatic_state"]
    counts = _state_counts(states)
    laterality_counts = acquisitions["laterality_state"].value_counts().to_dict()
    groupings = {
        "manufacturer": "manufacturer",
        "scanner_model": "manufacturer_model_name",
        "image_release": "image_release_study",
        "oai_qc_category": "xray_accept_qc",
    }
    distributions: dict[str, list[dict[str, Any]]] = {}
    for label, column in groupings.items():
        distributions[label] = _state_distribution(acquisitions, panels, column, label)
    spacing_frame = acquisitions.assign(
        spacing_family=acquisitions["original_row_spacing_mm"].map(
            lambda value: f"{float(value):g}"
        ),
        dimension_family=(
            acquisitions["original_rows"].astype(int).astype(str)
            + "x"
            + acquisitions["original_columns"].astype(int).astype(str)
        ),
    )
    distributions["pixel_spacing_mm"] = _state_distribution(
        spacing_frame, panels, "spacing_family", "pixel_spacing_mm"
    )
    distributions["image_dimension_family"] = _state_distribution(
        spacing_frame, panels, "dimension_family", "image_dimension_family"
    )
    flagged = _flagged_subgroups(distributions["manufacturer"] + distributions["scanner_model"])

    crop_storage = _directory_bytes(output_root / "crops", "*_uint16.npy")
    review_preview_storage = _directory_bytes(output_root / "review/previews")
    manifest_storage = _directory_bytes(manifest_root)
    audit_storage = _directory_bytes(output_root / "audit")
    review_state_storage = _directory_bytes(output_root / "review", "*.jsonl")
    exception_storage = _directory_bytes(output_root / "exceptions")
    review_queue_bytes = (
        (output_root / "review_queue.parquet").stat().st_size
        if (output_root / "review_queue.parquet").is_file()
        else 0
    )
    archive_storage = _directory_bytes(archive_root, "*.tar.gz")
    work_files = (
        [path for path in work_root.rglob("*") if path.is_file()] if work_root.exists() else []
    )
    crop_sizes = [path.stat().st_size for path in (output_root / "crops").rglob("*_uint16.npy")]

    summary: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "localizer_version": LOCALIZER_VERSION,
        "laterality_policy_version": LATERALITY_POLICY_VERSION,
        "frozen_localizer_source_sha256": str(
            acquisitions["frozen_localizer_source_sha256"].iloc[0]
        ),
        "expected_acquisitions": expected_acquisitions,
        "acquisitions_processed": len(acquisitions),
        "knee_panels_produced": len(panels),
        "automatic_state": {
            "pass_acquisitions": counts[QCState.PASS.value],
            "borderline_acquisitions": counts[QCState.BORDERLINE.value],
            "fail_acquisitions": counts[QCState.FAIL.value],
            "pass_percent": round(counts[QCState.PASS.value] / len(acquisitions) * 100, 3),
            "borderline_percent": round(
                counts[QCState.BORDERLINE.value] / len(acquisitions) * 100, 3
            ),
            "fail_percent": round(counts[QCState.FAIL.value] / len(acquisitions) * 100, 3),
        },
        "panel_automatic_state": _state_counts(panels["qc_state"]),
        "panel_localization_state": _state_counts(panels["localization_state"]),
        "panel_layout_state": _state_counts(panels["layout_state"]),
        "panel_anatomical_validator_state": _state_counts(panels["anatomical_validator_state"]),
        "manual_review_queue": {
            "queued_acquisitions": int(len(review_queue)),
            "queued_percent": round(len(review_queue) / len(acquisitions) * 100, 3),
            "borderline_or_fail_acquisitions": counts[QCState.BORDERLINE.value]
            + counts[QCState.FAIL.value],
            "laterality_exception_acquisitions": int(
                (~acquisitions["laterality_state"].eq(LateralityV2State.CONFIDENT.value)).sum()
            ),
            "provisionally_accepted_pass_acquisitions": counts[QCState.PASS.value]
            - int(
                (
                    acquisitions["automatic_state"].eq(QCState.PASS.value)
                    & acquisitions["review_required"]
                ).sum()
            ),
            "adjudicated_in_this_milestone": 0,
        },
        "laterality": {
            "policy_mapping": {
                "screen_left": "anatomical_RIGHT",
                "screen_right": "anatomical_LEFT",
            },
            "confident_acquisitions": int(
                laterality_counts.get(LateralityV2State.CONFIDENT.value, 0)
            ),
            "ambiguous_acquisitions": int(
                laterality_counts.get(LateralityV2State.AMBIGUOUS.value, 0)
            ),
            "conflicting_acquisitions": int(
                laterality_counts.get(LateralityV2State.CONFLICTING.value, 0)
            ),
            "resolved_knee_panels": int(panels["anatomical_side"].notna().sum()),
            "structure_exception_acquisitions": int(
                (~acquisitions["structure_check_passed"]).sum()
            ),
            "unilateral_dicom_tag_artifact_acquisitions": int(
                acquisitions["unilateral_dicom_tag_artifact"].sum()
            ),
            "unilateral_dicom_tag_overrode_mapping": False,
            "uncertain_sides_forced": 0,
        },
        "localization_statistics": {
            "median_confidence": round(float(panels["localization_confidence"].median()), 4),
            "median_candidate_margin": round(float(panels["candidate_margin"].median()), 5),
            "median_anatomical_validator_score": round(
                float(panels["anatomical_validator_score"].median()), 4
            ),
            "median_candidate_count": float(panels["candidate_count"].median()),
            "maximum_pair_row_delta_mm": round(
                float(panels["pair_row_delta_mm"].max(skipna=True) or 0.0), 3
            ),
            "crop_rows": OUTPUT_SHAPE[0],
            "crop_columns": OUTPUT_SHAPE[1],
            "crop_dtype": "uint16",
            "target_spacing_mm": TARGET_SPACING_MM,
            "crop_size_mm": CROP_SIZE_MM,
            "crops_matching_frozen_specification": len(panels),
        },
        "padding": _padding_statistics(panels),
        "review_image_privacy": {
            "burned_in_pixel_text_observed": True,
            "redaction_applied_to": [
                "bilateral_preview",
                "panel_localization_overlays",
                "panel_crop_previews",
            ],
            "detector_margin_redacted_mm": BURNED_IN_MARGIN_MM,
            "redacted_edges": ["top", "bottom"],
            "deepest_observed_burned_in_structure_mm": 33.0,
            "stored_uint16_crops_redacted": False,
            "panels_whose_crop_includes_detector_margin": int(
                panels["crop_rows_from_detector_margin"].gt(0).sum()
            ),
            "maximum_crop_rows_from_detector_margin": int(
                panels["crop_rows_from_detector_margin"].max()
            ),
            "crop_rows_total": OUTPUT_SHAPE[0],
        },
        "state_distributions": distributions,
        "flagged_scanner_family_behavior": flagged,
        "representative_compatibility": _compatibility_report(states),
        "geometric_preprocessing_failures": {
            "total": len(failures),
            "by_category": (
                failure_frame["category"].value_counts().to_dict()
                if not failure_frame.empty
                else {}
            ),
        },
        "storage": {
            "raw_archives": {
                **archive_storage,
                "median_bytes": statistics.median(
                    [path.stat().st_size for path in archive_root.rglob("*.tar.gz")]
                )
                if archive_storage["files"]
                else 0,
            },
            "v3_uint16_crops": {
                **crop_storage,
                "median_bytes": statistics.median(crop_sizes) if crop_sizes else 0,
            },
            "v3_qc_overlays_and_previews": review_preview_storage,
            "manifests": {
                **manifest_storage,
                "review_queue_bytes": review_queue_bytes,
            },
            "exception_queues": exception_storage,
            "audit_ledgers": audit_storage,
            "review_adjudication_artifacts": review_state_storage,
            "redundant_uint8_or_float32_caches": 0,
        },
        "temporary_extraction": {
            "successful_extractions_cleaned": int(
                acquisitions["temporary_extraction_cleaned"].sum()
            ),
            "files_remaining": len(work_files),
            "bytes_remaining": sum(path.stat().st_size for path in work_files),
        },
        "frozen_parameters_changed": False,
        "v1_or_v2_artifacts_overwritten": False,
        "cohort_changed": False,
        "merged_into_analysis_cohort": False,
        "modeling_split_created": False,
        "model_training_performed": False,
        "manual_adjudication_performed": False,
        "ready_for_manual_qc_adjudication": bool(
            len(acquisitions) == expected_acquisitions
            and len(panels) == 2 * expected_acquisitions
            and not work_files
        ),
    }
    atomic_json(summary, manifest_root / "full_cohort_qc.json")
    return summary


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen Localization V3 and Laterality V2 policies to the full baseline "
            "bilateral X-ray set and build the manual-review queue."
        )
    )
    parser.add_argument(
        "stage", choices=("gate", "recover", "process", "finalize", "verify", "all")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes for the pixel pass; defaults to the machine size.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results: dict[str, Any] = {}
    if args.stage in {"gate", "all"}:
        results["preservation_gate"] = run_preservation_gate()
    if args.stage in {"recover", "all"}:
        acquisitions = load_processing_inputs()
        recovery, summary = audit_recovery_state(acquisitions, write=True)
        results["recovery"] = summary
        results["storage_preflight"] = storage_preflight(
            recovery,
            workers=args.workers or max(1, min(6, (os.cpu_count() or 2) - 2)),
        )
    if args.stage in {"process", "all"}:
        results["processing"] = run_full_reprocessing(workers=args.workers)
    if args.stage in {"finalize", "all"}:
        results["full_cohort_qc"] = finalize_frozen_full_v3()
    if args.stage in {"verify", "all"}:
        results["preservation_verification"] = run_preservation_gate()
    print(json.dumps(results, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
