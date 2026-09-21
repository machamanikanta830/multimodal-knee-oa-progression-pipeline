"""Incrementally apply the frozen Milestone 5D pipeline to full baseline OAI images.

All identity-bearing manifests, real pixels, previews, and exception records are written only below
Git-ignored data directories. The module never downloads images, changes the cohort, creates a
modeling split, or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

from imaging.archive_extract import ArchiveContents, UnsafeArchiveError, safe_extract_archive
from imaging.dicom_inspect import DicomMetadata, inspect_dicom
from imaging.laterality import LateralityState, assess_laterality
from imaging.pixel_qc import normalize_for_display, save_contact_sheet, split_bilateral_midpoint
from imaging.preprocessing import (
    CropGeometry,
    JointLocalization,
    crop_around_center,
    localize_bilateral_tibiofemoral_joints,
    resample_to_spacing,
)

EXPECTED_ACQUISITIONS = 3621
EXPECTED_PANELS = EXPECTED_ACQUISITIONS * 2
TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
OUTPUT_SHAPE = (1067, 1067)
PREPROCESSING_VERSION = "oai-v00-xray-frozen-5d-v1"
VALIDATED_MAX_HORIZONTAL_PADDING_MM = 34.95
VALIDATED_MAX_VERTICAL_PADDING_MM = 24.60
PANEL_POSITIONS = ("screen_left", "screen_right")
MIN_REPORTABLE_GROUP = 5

DEFAULT_REFERENCES = Path("data/processed/manifests/v00_xray_full_associated_files.txt")
DEFAULT_URLS = Path("data/processed/manifests/v00_xray_full_s3_links.txt")
DEFAULT_PACKAGE_METADATA = Path(
    os.environ.get(
        "OAI_NDA_PACKAGE_METADATA",
        "data/raw/nda_package_metadata/full_package_file_metadata.txt.gz",
    )
)
DEFAULT_ARCHIVE_ROOT = Path("data/raw/oai_images/full")
DEFAULT_IMAGE_MANIFEST = Path("data/processed/manifests/v00_xray_manifest.parquet")
DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/full_v1")
DEFAULT_ARCHIVE_LEDGER = DEFAULT_OUTPUT_ROOT / "manifests/source_archives.parquet"
DEFAULT_ACQUISITION_MANIFEST = Path(
    "data/processed/manifests/v00_xray_full_preprocessing_acquisitions.parquet"
)
DEFAULT_STANDARDIZED_MANIFEST = Path(
    "data/processed/manifests/v00_xray_full_standardized_images.parquet"
)


class FullPreprocessingError(ValueError):
    """Raised when the frozen full-cohort workflow cannot proceed safely."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".json", dir=target.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _package_records(
    references: list[str], package_metadata_path: Path
) -> dict[str, tuple[str, int]]:
    reference_set = set(references)
    matched: dict[str, tuple[str, int]] = {}
    ambiguous: set[str] = set()
    with gzip.open(package_metadata_path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            alias = (row.get("DOWNLOAD_ALIAS") or "").strip()
            if alias not in reference_set:
                continue
            if alias in matched:
                ambiguous.add(alias)
                continue
            url = (row.get("NDA_S3_URL") or "").strip()
            try:
                size = int(row.get("FILE_SIZE") or "")
            except ValueError as error:
                raise FullPreprocessingError(
                    "A selected package record has invalid size"
                ) from error
            if not url.startswith("s3://") or size < 0:
                raise FullPreprocessingError("A selected package record has invalid URL or size")
            matched[alias] = (url, size)
    missing = reference_set.difference(matched)
    if missing or ambiguous or len(matched) != EXPECTED_ACQUISITIONS:
        raise FullPreprocessingError(
            "Package metadata does not uniquely match all 3,621 selected references"
        )
    return matched


def audit_full_download(
    *,
    references_path: Path = DEFAULT_REFERENCES,
    urls_path: Path = DEFAULT_URLS,
    package_metadata_path: Path = DEFAULT_PACKAGE_METADATA,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
) -> dict[str, Any]:
    """Validate and hash every expected archive before pixel processing."""

    references = _read_nonempty_lines(references_path)
    urls = _read_nonempty_lines(urls_path)
    if len(references) != EXPECTED_ACQUISITIONS or len(set(references)) != len(references):
        raise FullPreprocessingError("Reference list is not exactly 3,621 unique records")
    if len(urls) != EXPECTED_ACQUISITIONS or len(set(urls)) != len(urls):
        raise FullPreprocessingError("URL list is not exactly 3,621 unique records")
    package = _package_records(references, package_metadata_path)
    if [package[reference][0] for reference in references] != urls:
        raise FullPreprocessingError("Authorized URL list disagrees with package metadata")

    local_files = [path for path in archive_root.rglob("*") if path.is_file()]
    partials = [path for path in local_files if path.name.endswith(".partial")]
    finals = [path for path in local_files if path.name.endswith(".tar.gz")]
    expected_paths = {archive_root / reference for reference in references}
    unrelated = set(local_files).difference(expected_paths)
    missing = expected_paths.difference(finals)
    size_mismatches = [
        path
        for path in finals
        if path in expected_paths
        and path.stat().st_size != package[path.relative_to(archive_root).as_posix()][1]
    ]
    if len(finals) != EXPECTED_ACQUISITIONS or partials or unrelated or missing or size_mismatches:
        raise FullPreprocessingError(
            "Full download gate failed: final, partial, unrelated, missing, or size checks differ"
        )

    def hash_record(item: tuple[int, str]) -> dict[str, Any]:
        index, reference = item
        path = archive_root / reference
        return {
            "acquisition_index": index,
            "associated_file_reference": reference,
            "archive_relative_path": path.relative_to(archive_root).as_posix(),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": _sha256(path),
            "expected_package_bytes": package[reference][1],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(hash_record, enumerate(references, start=1)))
    ledger = pd.DataFrame(records).sort_values("acquisition_index", kind="stable")
    duplicate_hashes = int(ledger["archive_sha256"].duplicated().sum())
    if duplicate_hashes:
        raise FullPreprocessingError("Downloaded archives contain duplicate content hashes")
    _atomic_parquet(ledger, ledger_path)
    sizes = ledger["archive_bytes"].astype(int).tolist()
    summary = {
        "expected_archives": EXPECTED_ACQUISITIONS,
        "present_archives": len(finals),
        "package_size_matches": len(finals) - len(size_mismatches),
        "partial_files": len(partials),
        "missing_archives": len(missing),
        "unrelated_files": len(unrelated),
        "duplicate_references": len(references) - len(set(references)),
        "duplicate_archive_hashes": duplicate_hashes,
        "archive_storage": {
            "total_bytes": sum(sizes),
            "minimum_bytes": min(sizes),
            "median_bytes": statistics.median(sizes),
            "mean_bytes": statistics.fmean(sizes),
            "maximum_bytes": max(sizes),
        },
        "download_gate_passed": True,
    }
    _atomic_json(summary, ledger_path.with_name("download_audit.json"))
    return summary


def verify_archive_hashes(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
) -> dict[str, int | bool]:
    """Re-hash source archives and compare them with the pre-processing ledger."""

    ledger = pd.read_parquet(ledger_path).sort_values("acquisition_index", kind="stable")

    def verify(row: Any) -> bool:
        path = archive_root / str(row.archive_relative_path)
        return (
            path.is_file()
            and path.stat().st_size == int(row.archive_bytes)
            and _sha256(path) == str(row.archive_sha256)
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        matches = list(pool.map(verify, ledger.itertuples(index=False)))
    return {
        "archives_checked": len(matches),
        "hash_matches": sum(matches),
        "all_unchanged": len(matches) == EXPECTED_ACQUISITIONS and all(matches),
    }


def _single_extracted_file(directory: Path) -> Path:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise FullPreprocessingError(
            "An acquisition archive does not contain exactly one regular file"
        )
    return files[0]


def _spacing(metadata: DicomMetadata) -> tuple[float, float]:
    value = metadata.pixel_spacing or metadata.imager_pixel_spacing
    if value is None or len(value) != 2:
        raise FullPreprocessingError("A DICOM has no usable two-dimensional pixel spacing")
    spacing = (float(value[0]), float(value[1]))
    if any(not np.isfinite(item) or item <= 0 for item in spacing):
        raise FullPreprocessingError("A DICOM has invalid pixel spacing")
    return spacing


def _single_group_value(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise FullPreprocessingError(f"An acquisition has inconsistent {column}")
    value = values.iloc[0]
    return None if pd.isna(value) else value


def _load_acquisition_linkage(image_manifest_path: Path, ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "participant_id",
        "source_participant_id",
        "knee_side_code",
        "baseline_visit",
        "read_project",
        "accession_number",
        "image_file",
        "image_description",
        "image_release_study",
        "xray_accept_qc",
        "xray_alignment_problem",
        "xray_centering_problem",
        "xray_incomplete_depiction",
        "xray_positioning_problem",
        "bilateral_acquisition",
    ]
    knees = pd.read_parquet(image_manifest_path, columns=columns)
    if knees.duplicated(["participant_id", "knee_side_code"]).any():
        raise FullPreprocessingError("Image manifest contains duplicate participant-knee keys")
    if not knees["baseline_visit"].eq("V00").all() or not knees["read_project"].eq("15").all():
        raise FullPreprocessingError("Image manifest is not uniformly V00 and read project 15")
    if not knees["bilateral_acquisition"].astype(bool).all():
        raise FullPreprocessingError("Image manifest contains a non-bilateral acquisition")

    rows: list[dict[str, Any]] = []
    copied = [
        "participant_id",
        "source_participant_id",
        "image_file",
        "image_description",
        "image_release_study",
        "xray_accept_qc",
        "xray_alignment_problem",
        "xray_centering_problem",
        "xray_incomplete_depiction",
        "xray_positioning_problem",
    ]
    for accession, group in knees.groupby("accession_number", sort=False, dropna=False):
        row = {"accession_number": accession}
        for column in copied:
            row[column] = _single_group_value(group, column)
        row["eligible_knee_sides"] = ",".join(sorted(group["knee_side_code"].astype(str)))
        row["eligible_knee_count"] = len(group)
        rows.append(row)
    acquisitions = pd.DataFrame(rows)
    linked = ledger.merge(
        acquisitions,
        left_on="associated_file_reference",
        right_on="image_file",
        how="left",
        validate="one_to_one",
    )
    if len(linked) != EXPECTED_ACQUISITIONS or linked["accession_number"].isna().any():
        raise FullPreprocessingError("Source archive ledger does not match the image manifest")
    return linked.sort_values("acquisition_index", kind="stable", ignore_index=True)


def _crop_path(output_root: Path, acquisition_index: int, panel_position: str) -> Path:
    return (
        output_root
        / "crops"
        / f"acquisition_{acquisition_index:04d}"
        / f"{panel_position}_uint16.npy"
    )


def _verify_completed_bundle(
    output_root: Path, acquisition_index: int, archive_sha256: str
) -> dict[str, Any]:
    directory = output_root / "crops" / f"acquisition_{acquisition_index:04d}"
    record_path = directory / "record.json"
    if not record_path.is_file():
        raise FullPreprocessingError("An existing crop directory lacks its completion record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("preprocessing_version") != PREPROCESSING_VERSION
        or record.get("source_archive_sha256") != archive_sha256
    ):
        raise FullPreprocessingError("An existing crop bundle has incompatible provenance")
    for position in PANEL_POSITIONS:
        crop = np.load(_crop_path(output_root, acquisition_index, position), mmap_mode="r")
        if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
            raise FullPreprocessingError("An existing crop bundle fails shape or dtype validation")
    return record


def _save_crop_bundle(
    output_root: Path,
    acquisition_index: int,
    crops: dict[str, np.ndarray],
    record: dict[str, Any],
) -> Path:
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    target = crop_root / f"acquisition_{acquisition_index:04d}"
    if target.exists():
        raise FullPreprocessingError(
            "A crop destination already exists and will not be overwritten"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=crop_root))
    try:
        for position in PANEL_POSITIONS:
            crop = crops[position]
            if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                raise FullPreprocessingError(
                    "Generated crop fails frozen shape or dtype validation"
                )
            np.save(temporary / f"{position}_uint16.npy", crop, allow_pickle=False)
            saved = np.load(temporary / f"{position}_uint16.npy", mmap_mode="r")
            if saved.shape != OUTPUT_SHAPE or saved.dtype != np.dtype("uint16"):
                raise FullPreprocessingError("Stored crop fails post-write validation")
        _atomic_json(record, temporary / "record.json")
        temporary.replace(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_failure(
    output_root: Path,
    *,
    acquisition_index: int,
    category: str,
    stage: str,
    error: Exception,
    linkage: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "acquisition_index": acquisition_index,
        "category": category,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "participant_id": linkage["participant_id"],
        "source_participant_id": linkage["source_participant_id"],
        "accession_number": linkage["accession_number"],
        "associated_file_reference": linkage["associated_file_reference"],
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    target = output_root / "failure_records" / f"acquisition_{acquisition_index:04d}.json"
    if not target.exists():
        _atomic_json(record, target)
    return record


def _location_record(
    location: JointLocalization, geometry: CropGeometry, position: str
) -> dict[str, Any]:
    horizontal_padding = geometry.padding_left + geometry.padding_right
    vertical_padding = geometry.padding_top + geometry.padding_bottom
    return {
        "panel_position": position,
        "joint_row": location.row,
        "joint_column": location.column,
        "projection_score": location.score,
        "paired_projection_robust_score": location.robust_score,
        "localization_state": "SUCCESS",
        "crop_adequacy_state": "TECHNICALLY_VALID",
        "padding_top": geometry.padding_top,
        "padding_bottom": geometry.padding_bottom,
        "padding_left": geometry.padding_left,
        "padding_right": geometry.padding_right,
        "horizontal_padding_mm": horizontal_padding * TARGET_SPACING_MM,
        "vertical_padding_mm": vertical_padding * TARGET_SPACING_MM,
        "excessive_padding": (
            horizontal_padding * TARGET_SPACING_MM > VALIDATED_MAX_HORIZONTAL_PADDING_MM
            or vertical_padding * TARGET_SPACING_MM > VALIDATED_MAX_VERTICAL_PADDING_MM
        ),
    }


def _process_acquisition(
    row: Any,
    *,
    archive_root: Path,
    output_root: Path,
    work_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    index = int(row.acquisition_index)
    archive_path = archive_root / str(row.archive_relative_path)
    final_crop_dir = output_root / "crops" / f"acquisition_{index:04d}"
    if final_crop_dir.exists():
        return _verify_completed_bundle(output_root, index, str(row.archive_sha256)), None, True
    failure_path = output_root / "failure_records" / f"acquisition_{index:04d}.json"
    if failure_path.exists():
        return None, json.loads(failure_path.read_text(encoding="utf-8")), True

    linkage = {
        "participant_id": row.participant_id,
        "source_participant_id": row.source_participant_id,
        "accession_number": row.accession_number,
        "associated_file_reference": row.associated_file_reference,
    }
    extraction = work_root / f"acquisition_{index:04d}"
    if extraction.exists():
        error = FullPreprocessingError("A prior unresolved extraction workspace exists")
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="DICOM_FAILURE",
                stage="preexisting_workspace",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    contents: ArchiveContents | None = None
    try:
        contents = safe_extract_archive(archive_path, extraction)
    except (UnsafeArchiveError, OSError, ValueError) as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="ARCHIVE_FAILURE",
                stage="safe_extraction",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    try:
        if contents.regular_files != 1:
            raise FullPreprocessingError("Archive does not contain exactly one regular file")
        dicom_path = _single_extracted_file(extraction)
        metadata = inspect_dicom(dicom_path)
        dataset = pydicom.dcmread(dicom_path, force=False)
        if int(dataset.get("NumberOfFrames", 1)) != 1:
            raise FullPreprocessingError("DICOM is not single-frame")
        source = np.asarray(dataset.pixel_array)
        if source.ndim != 2:
            raise FullPreprocessingError("DICOM pixel array is not two-dimensional")
        if source.dtype != np.dtype("uint16"):
            raise FullPreprocessingError("DICOM pixel array is not uint16")
        if metadata.rows != source.shape[0] or metadata.columns != source.shape[1]:
            raise FullPreprocessingError("DICOM metadata and decoded dimensions disagree")
        spacing = _spacing(metadata)
        dicom_sha256 = _sha256(dicom_path)
    except Exception as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="DICOM_FAILURE",
                stage="dicom_validation",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    try:
        screen_left, screen_right = split_bilateral_midpoint(source)
        if (
            screen_left.shape[0] != source.shape[0]
            or screen_right.shape[0] != source.shape[0]
            or screen_left.shape[1] + screen_right.shape[1] != source.shape[1]
        ):
            raise FullPreprocessingError("Midpoint panels do not preserve source geometry")
        panels = {"screen_left": screen_left, "screen_right": screen_right}
    except Exception as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="MIDPOINT_FAILURE",
                stage="midpoint_split",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    try:
        left_location, right_location = localize_bilateral_tibiofemoral_joints(
            screen_left, screen_right
        )
        locations = {"screen_left": left_location, "screen_right": right_location}
    except Exception as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="LOCALIZATION_FAILURE",
                stage="frozen_joint_localizer",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    # No frozen automatic marker reader exists. Convention alone is not sufficient evidence under
    # the approved policy, so unresolved acquisitions remain unlabeled for manual marker review.
    laterality = assess_laterality(
        bilateral_acquisition=True,
        dicom_laterality=metadata.laterality,
        image_laterality=metadata.image_laterality,
        marker_mapping=None,
        orientation_mapping=None,
        documented_convention_mapping=None,
    )
    crops: dict[str, np.ndarray] = {}
    panel_records: list[dict[str, Any]] = []
    try:
        for position in PANEL_POSITIONS:
            panel = panels[position]
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
            if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                raise FullPreprocessingError("Crop does not match frozen output specification")
            crops[position] = crop
            panel_records.append(_location_record(location, geometry, position))
    except Exception as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="CROP_INADEQUACY",
                stage="frozen_resample_and_crop",
                error=error,
                linkage=linkage,
            ),
            False,
        )

    screen_sides = {
        "screen_left": laterality.screen_left_anatomical_side,
        "screen_right": laterality.screen_right_anatomical_side,
    }
    for panel_record in panel_records:
        panel_record["anatomical_side"] = screen_sides[panel_record["panel_position"]]
        panel_record["crop_relative_path"] = (
            _crop_path(output_root, index, panel_record["panel_position"])
            .relative_to(output_root)
            .as_posix()
        )
    record = {
        **linkage,
        "acquisition_index": index,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_archive_sha256": str(row.archive_sha256),
        "source_archive_bytes": int(row.archive_bytes),
        "archive_members": contents.members,
        "archive_regular_files": contents.regular_files,
        "archive_directories": contents.directories,
        "archive_declared_file_bytes": contents.declared_file_bytes,
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
        "midpoint_state": "SUCCESS",
        "laterality_state": laterality.state.value,
        "laterality_reason": laterality.reason,
        "marker_evidence": "NOT_AUTOMATICALLY_REVIEWED",
        "screen_left_anatomical_side": laterality.screen_left_anatomical_side,
        "screen_right_anatomical_side": laterality.screen_right_anatomical_side,
        "eligible_knee_sides": row.eligible_knee_sides,
        "eligible_knee_count": int(row.eligible_knee_count),
        "manufacturer": metadata.manufacturer,
        "manufacturer_model_name": metadata.manufacturer_model_name,
        "photometric_interpretation": metadata.photometric_interpretation,
        "bits_allocated": metadata.bits_allocated,
        "bits_stored": metadata.bits_stored,
        "pixel_representation": metadata.pixel_representation,
        "transfer_syntax_uid": metadata.transfer_syntax_uid,
        "compressed": metadata.compressed,
        "image_release_study": row.image_release_study,
        "xray_accept_qc": row.xray_accept_qc,
        "xray_alignment_problem": row.xray_alignment_problem,
        "xray_centering_problem": row.xray_centering_problem,
        "xray_incomplete_depiction": row.xray_incomplete_depiction,
        "xray_positioning_problem": row.xray_positioning_problem,
        "panels": panel_records,
        "temporary_extraction_cleaned": False,
    }

    temporary_crop_root = output_root / "pending_crop_bundles"
    temporary_crop_root.mkdir(parents=True, exist_ok=True)
    temporary_bundle = Path(
        tempfile.mkdtemp(prefix=f"acquisition_{index:04d}.", dir=temporary_crop_root)
    )
    try:
        for position in PANEL_POSITIONS:
            np.save(
                temporary_bundle / f"{position}_uint16.npy", crops[position], allow_pickle=False
            )
            saved = np.load(temporary_bundle / f"{position}_uint16.npy", mmap_mode="r")
            if saved.shape != OUTPUT_SHAPE or saved.dtype != np.dtype("uint16"):
                raise FullPreprocessingError("Temporary crop fails post-write validation")
        shutil.rmtree(extraction)
        if extraction.exists():
            raise FullPreprocessingError("Temporary DICOM cleanup did not complete")
        record["temporary_extraction_cleaned"] = True
        _atomic_json(record, temporary_bundle / "record.json")
        destination = output_root / "crops" / f"acquisition_{index:04d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FullPreprocessingError("A crop destination appeared during processing")
        temporary_bundle.replace(destination)
    except Exception as error:
        return (
            None,
            _write_failure(
                output_root,
                acquisition_index=index,
                category="CROP_INADEQUACY",
                stage="crop_storage_or_cleanup",
                error=error,
                linkage=linkage,
            ),
            False,
        )
    finally:
        if temporary_bundle.exists():
            shutil.rmtree(temporary_bundle)
    return record, None, False


def _load_result_records(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "crops").glob("acquisition_*/record.json"))
    ]
    failures = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "failure_records").glob("acquisition_*.json"))
    ]
    return successes, failures


def run_full_preprocessing(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    image_manifest_path: Path = DEFAULT_IMAGE_MANIFEST,
    ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    work_root: Path = Path("data/interim/oai_images/full_work"),
) -> dict[str, Any]:
    """Run or safely resume incremental frozen preprocessing."""

    ledger = pd.read_parquet(ledger_path)
    if len(ledger) != EXPECTED_ACQUISITIONS:
        raise FullPreprocessingError("Source archive ledger is not complete")
    acquisitions = _load_acquisition_linkage(image_manifest_path, ledger)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    processed_now = 0
    resumed_existing = 0
    failures_now = 0
    for number, row in enumerate(acquisitions.itertuples(index=False), start=1):
        record, failure, resumed = _process_acquisition(
            row,
            archive_root=archive_root,
            output_root=output_root,
            work_root=work_root,
        )
        if resumed:
            resumed_existing += 1
        elif record is not None:
            processed_now += 1
        elif failure is not None:
            failures_now += 1
        if number % 25 == 0 or number == EXPECTED_ACQUISITIONS:
            successes, failures = _load_result_records(output_root)
            print(
                json.dumps(
                    {
                        "acquisitions_considered": number,
                        "successful_bundles": len(successes),
                        "failure_records": len(failures),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    successes, failures = _load_result_records(output_root)
    result = {
        "expected_acquisitions": EXPECTED_ACQUISITIONS,
        "successful_acquisitions": len(successes),
        "failed_acquisitions": len(failures),
        "processed_this_run": processed_now,
        "resumed_existing_results": resumed_existing,
        "failures_this_run": failures_now,
        "work_files_remaining": sum(path.is_file() for path in work_root.rglob("*")),
    }
    _atomic_json(result, output_root / "manifests/processing_run.json")
    return result


def _pooled_groups(values: pd.Series) -> pd.Series:
    text = values.fillna("<missing>").astype(str).replace("", "<missing>")
    counts = text.value_counts()
    small = set(counts[counts < MIN_REPORTABLE_GROUP].index)
    return text.map(lambda value: "<pooled groups n<5>" if value in small else value)


def _heterogeneity_summary(
    acquisitions: pd.DataFrame, panels: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "manufacturer": acquisitions["manufacturer"],
        "scanner_model": acquisitions["manufacturer_model_name"],
        "pixel_spacing_mm": acquisitions["original_row_spacing_mm"].map(
            lambda value: f"{float(value):g}"
        ),
        "image_dimensions": (
            acquisitions["original_rows"].astype(int).astype(str)
            + "x"
            + acquisitions["original_columns"].astype(int).astype(str)
        ),
        "image_release": acquisitions["image_release_study"],
        "oai_qc_category": acquisitions["xray_accept_qc"],
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for name, raw in groups.items():
        pooled = _pooled_groups(raw)
        entries: list[dict[str, Any]] = []
        for value in sorted(pooled.unique()):
            indices = set(acquisitions.loc[pooled.eq(value), "acquisition_index"].astype(int))
            panel_subset = panels.loc[panels["acquisition_index"].isin(indices)]
            entries.append(
                {
                    "group": value,
                    "acquisitions": len(indices),
                    "panels": len(panel_subset),
                    "midpoint_success_percent": 100.0,
                    "localization_success_percent": round(
                        panel_subset["localization_state"].eq("SUCCESS").mean() * 100, 3
                    ),
                    "technical_crop_valid_percent": round(
                        panel_subset["crop_adequacy_state"].eq("TECHNICALLY_VALID").mean() * 100,
                        3,
                    ),
                    "excessive_padding_panels": int(panel_subset["excessive_padding"].sum()),
                }
            )
        result[name] = entries
    return result


def _exception_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def finalize_full_preprocessing(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    acquisition_manifest_path: Path = DEFAULT_ACQUISITION_MANIFEST,
    standardized_manifest_path: Path = DEFAULT_STANDARDIZED_MANIFEST,
    work_root: Path = Path("data/interim/oai_images/full_work"),
    ledger_path: Path = DEFAULT_ARCHIVE_LEDGER,
) -> dict[str, Any]:
    """Create local manifests, exception queues, and aggregate QC after processing."""

    success_records, failure_records = _load_result_records(output_root)
    acquisition_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for record in success_records:
        acquisition_rows.append({key: value for key, value in record.items() if key != "panels"})
        eligible_sides = set(str(record["eligible_knee_sides"]).split(","))
        for panel in record["panels"]:
            side = panel["anatomical_side"]
            if side is None:
                linkage_status = "UNRESOLVED_LATERALITY"
                eligible_analysis_knee: bool | None = None
            elif side in eligible_sides:
                linkage_status = "RESOLVED_ELIGIBLE"
                eligible_analysis_knee = True
            else:
                linkage_status = "RESOLVED_NOT_IN_ANALYSIS_COHORT"
                eligible_analysis_knee = False
            panel_rows.append(
                {
                    "acquisition_index": record["acquisition_index"],
                    "participant_id": record["participant_id"],
                    "source_participant_id": record["source_participant_id"],
                    "accession_number": record["accession_number"],
                    "associated_file_reference": record["associated_file_reference"],
                    "screen_panel": panel["panel_position"],
                    "anatomical_side": side,
                    "participant_knee_linkage_status": linkage_status,
                    "eligible_analysis_knee": eligible_analysis_knee,
                    "standardized_crop_path": panel["crop_relative_path"],
                    "original_rows": record["original_rows"],
                    "original_columns": record["original_columns"],
                    "original_row_spacing_mm": record["original_row_spacing_mm"],
                    "original_column_spacing_mm": record["original_column_spacing_mm"],
                    "target_spacing_mm": record["target_spacing_mm"],
                    "crop_size_mm": record["crop_size_mm"],
                    "crop_rows": record["crop_rows"],
                    "crop_columns": record["crop_columns"],
                    "crop_dtype": record["crop_dtype"],
                    "joint_row": panel["joint_row"],
                    "joint_column": panel["joint_column"],
                    "projection_score": panel["projection_score"],
                    "paired_projection_robust_score": panel["paired_projection_robust_score"],
                    "localization_state": panel["localization_state"],
                    "crop_adequacy_state": panel["crop_adequacy_state"],
                    "padding_top": panel["padding_top"],
                    "padding_bottom": panel["padding_bottom"],
                    "padding_left": panel["padding_left"],
                    "padding_right": panel["padding_right"],
                    "horizontal_padding_mm": panel["horizontal_padding_mm"],
                    "vertical_padding_mm": panel["vertical_padding_mm"],
                    "excessive_padding": panel["excessive_padding"],
                    "laterality_state": record["laterality_state"],
                    "laterality_reason": record["laterality_reason"],
                    "preprocessing_version": record["preprocessing_version"],
                }
            )
    acquisitions = pd.DataFrame(acquisition_rows)
    panels = pd.DataFrame(panel_rows)
    failures = pd.DataFrame(failure_records)
    _atomic_parquet(acquisitions, acquisition_manifest_path)
    _atomic_parquet(panels, standardized_manifest_path)

    exception_root = output_root / "exceptions"
    base_failure_columns = [
        "acquisition_index",
        "category",
        "stage",
        "error_type",
        "error_message",
    ]
    failure_names = {
        "ARCHIVE_FAILURE": "archive_failure.csv",
        "DICOM_FAILURE": "dicom_failure.csv",
        "MIDPOINT_FAILURE": "midpoint_failure.csv",
        "LOCALIZATION_FAILURE": "localization_failure.csv",
        "CROP_INADEQUACY": "crop_inadequacy.csv",
    }
    for category, filename in failure_names.items():
        rows = (
            failures.loc[failures["category"].eq(category)].to_dict("records")
            if not failures.empty
            else []
        )
        _atomic_csv(_exception_frame(rows, base_failure_columns), exception_root / filename)

    padding_columns = [
        "acquisition_index",
        "screen_panel",
        "horizontal_padding_mm",
        "vertical_padding_mm",
    ]
    padding_rows = panels.loc[panels["excessive_padding"], padding_columns].to_dict("records")
    _atomic_csv(
        _exception_frame(padding_rows, padding_columns), exception_root / "excessive_padding.csv"
    )
    laterality_columns = ["acquisition_index", "laterality_state", "laterality_reason"]
    for state, filename in (
        (LateralityState.AMBIGUOUS.value, "laterality_ambiguity.csv"),
        (LateralityState.CONFLICTING.value, "laterality_conflict.csv"),
    ):
        rows = acquisitions.loc[
            acquisitions["laterality_state"].eq(state), laterality_columns
        ].to_dict("records")
        _atomic_csv(_exception_frame(rows, laterality_columns), exception_root / filename)

    archive_storage = json.loads(
        ledger_path.with_name("download_audit.json").read_text(encoding="utf-8")
    )["archive_storage"]
    crop_paths = list((output_root / "crops").glob("acquisition_*/*_uint16.npy"))
    crop_bytes = sum(path.stat().st_size for path in crop_paths)
    work_files = [path for path in work_root.rglob("*") if path.is_file()]
    work_bytes = sum(path.stat().st_size for path in work_files)
    laterality_counts = acquisitions["laterality_state"].value_counts().to_dict()
    resolved_panels = int(panels["anatomical_side"].notna().sum())
    resolved_cohort_knees = int(panels["eligible_analysis_knee"].eq(True).sum())
    any_horizontal = int(panels["horizontal_padding_mm"].gt(0).sum())
    any_vertical = int(panels["vertical_padding_mm"].gt(0).sum())
    heterogeneity = _heterogeneity_summary(acquisitions, panels) if not acquisitions.empty else {}
    subgroup_failures = sum(
        entry["midpoint_success_percent"] < 99.0
        or entry["localization_success_percent"] < 98.0
        or entry["technical_crop_valid_percent"] < 98.0
        for entries in heterogeneity.values()
        for entry in entries
    )
    manual_review_indices = set(
        acquisitions.loc[
            ~acquisitions["laterality_state"].eq(LateralityState.CONFIDENT.value),
            "acquisition_index",
        ].astype(int)
    )
    manual_review_indices.update(
        panels.loc[panels["excessive_padding"], "acquisition_index"].astype(int)
    )
    if not failures.empty:
        manual_review_indices.update(failures["acquisition_index"].astype(int))

    summary = {
        "expected_acquisitions": EXPECTED_ACQUISITIONS,
        "successful_acquisitions": len(acquisitions),
        "failed_acquisitions": len(failure_records),
        "valid_dicoms": len(acquisitions)
        + sum(
            record.get("stage") not in {"safe_extraction", "dicom_validation"}
            for record in failure_records
        ),
        "midpoint": {
            "successful": int(acquisitions["midpoint_state"].eq("SUCCESS").sum()),
            "failed": sum(
                record.get("category") == "MIDPOINT_FAILURE" for record in failure_records
            ),
        },
        "localization": {
            "successful_panels": int(panels["localization_state"].eq("SUCCESS").sum()),
            "borderline_panels": 0,
            "failed_acquisitions": sum(
                record.get("category") == "LOCALIZATION_FAILURE" for record in failure_records
            ),
        },
        "crop": {
            "technically_valid_panels": int(
                panels["crop_adequacy_state"].eq("TECHNICALLY_VALID").sum()
            ),
            "inadequate_acquisitions": sum(
                record.get("category") == "CROP_INADEQUACY" for record in failure_records
            ),
            "anatomical_adequacy_requires_visual_qc": True,
        },
        "padding": {
            "horizontal_panels": any_horizontal,
            "vertical_panels": any_vertical,
            "maximum_horizontal_mm": float(panels["horizontal_padding_mm"].max()),
            "maximum_vertical_mm": float(panels["vertical_padding_mm"].max()),
            "outside_independent_validation_envelope_panels": int(
                panels["excessive_padding"].sum()
            ),
        },
        "laterality": {
            "confident_acquisitions": int(
                laterality_counts.get(LateralityState.CONFIDENT.value, 0)
            ),
            "ambiguous_acquisitions": int(
                laterality_counts.get(LateralityState.AMBIGUOUS.value, 0)
            ),
            "conflicting_acquisitions": int(
                laterality_counts.get(LateralityState.CONFLICTING.value, 0)
            ),
            "resolved_panel_crops": resolved_panels,
            "resolved_analysis_cohort_knee_crops": resolved_cohort_knees,
            "manual_review_acquisitions": len(manual_review_indices),
            "automatic_marker_reader_available": False,
            "uncertain_sides_forced": 0,
        },
        "temporary_extraction": {
            "dicom_bytes_processed": int(acquisitions["dicom_file_bytes"].sum()),
            "successful_extractions_cleaned": int(
                acquisitions["temporary_extraction_cleaned"].sum()
            ),
            "files_remaining": len(work_files),
            "bytes_remaining": work_bytes,
        },
        "storage": {
            "raw_archives": archive_storage,
            "standardized_uint16_crop_bytes": crop_bytes,
            "crop_files": len(crop_paths),
            "optional_uint8_1067_estimate_bytes": OUTPUT_SHAPE[0]
            * OUTPUT_SHAPE[1]
            * len(crop_paths),
            "optional_uint8_512_estimate_bytes": 512 * 512 * len(crop_paths),
        },
        "heterogeneity": heterogeneity,
        "systematic_preprocessing_failure_groups": subgroup_failures,
        "frozen_parameters_unchanged": True,
        "model_training_performed": False,
        "modeling_split_created": False,
        "cohort_changed": False,
    }
    _atomic_json(summary, output_root / "manifests/final_summary.json")
    manifest_files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and "crops" not in path.relative_to(output_root).parts
    ] + [acquisition_manifest_path, standardized_manifest_path]
    summary["storage"]["manifest_and_qc_bytes"] = sum(
        path.stat().st_size for path in set(manifest_files) if path.exists()
    )
    _atomic_json(summary, output_root / "manifests/final_summary.json")
    return summary


def _qc_indices(records: list[dict[str, Any]], sample_size: int) -> list[int]:
    rows = []
    for record in records:
        maximum_padding = max(
            max(panel["horizontal_padding_mm"], panel["vertical_padding_mm"])
            for panel in record["panels"]
        )
        rows.append(
            {
                "acquisition_index": int(record["acquisition_index"]),
                "manufacturer": record["manufacturer"],
                "scanner_model": record["manufacturer_model_name"],
                "spacing": record["original_row_spacing_mm"],
                "dimensions": f"{record['original_rows']}x{record['original_columns']}",
                "pixel_area": int(record["original_rows"]) * int(record["original_columns"]),
                "image_release": record["image_release_study"],
                "qc_category": record["xray_accept_qc"],
                "laterality_state": record["laterality_state"],
                "maximum_padding": maximum_padding,
            }
        )
    frame = pd.DataFrame(rows).sort_values("acquisition_index", kind="stable")
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected and len(selected) < sample_size:
            selected.append(index)

    for index in frame.nlargest(min(16, len(frame)), "maximum_padding")["acquisition_index"]:
        add(int(index))
    for column in (
        "manufacturer",
        "scanner_model",
        "spacing",
        "image_release",
        "qc_category",
        "laterality_state",
    ):
        values = frame[column].fillna("<missing>").astype(str)
        for value in sorted(values.unique()):
            add(int(frame.loc[values.eq(value), "acquisition_index"].iloc[0]))
    dimension_bands = pd.qcut(frame["pixel_area"], q=min(8, len(frame)), duplicates="drop")
    for _, group in frame.groupby(dimension_bands, observed=True):
        add(int(group.iloc[len(group) // 2]["acquisition_index"]))
    if len(frame):
        row_bands = pd.qcut(frame["acquisition_index"], q=min(8, len(frame)), duplicates="drop")
        for _, group in frame.groupby(row_bands, observed=True):
            add(int(group.iloc[len(group) // 2]["acquisition_index"]))
    for index in np.linspace(1, EXPECTED_ACQUISITIONS, num=sample_size, dtype=int):
        add(int(index))
    return selected


def _render_qc_stage(
    source: np.ndarray,
    record: dict[str, Any],
    output_root: Path,
    destination: Path,
    *,
    qc_number: int,
) -> None:
    photometric = str(record["photometric_interpretation"])
    display = normalize_for_display(source, photometric)
    original = Image.fromarray(display, mode="L").convert("RGB")
    scale = min(1.0, 1200 / original.width)
    if scale < 1:
        original = original.resize(
            (1200, max(1, round(original.height * scale))), Image.Resampling.LANCZOS
        )
    header = 28
    original_canvas = Image.new("RGB", (1200, original.height + header), "black")
    original_canvas.paste(original, ((1200 - original.width) // 2, header))
    draw = ImageDraw.Draw(original_canvas)
    x_offset = (1200 - original.width) // 2
    midpoint = source.shape[1] // 2
    midpoint_x = x_offset + round(midpoint * scale)
    draw.line(
        (midpoint_x, header, midpoint_x, original_canvas.height - 1),
        fill="magenta",
        width=3,
    )
    half_height = round(CROP_SIZE_MM / record["original_row_spacing_mm"] / 2)
    half_width = round(CROP_SIZE_MM / record["original_column_spacing_mm"] / 2)
    for panel in record["panels"]:
        panel_offset = 0 if panel["panel_position"] == "screen_left" else midpoint
        center_x = x_offset + round((panel_offset + panel["joint_column"]) * scale)
        center_y = header + round(panel["joint_row"] * scale)
        draw.rectangle(
            (
                center_x - round(half_width * scale),
                center_y - round(half_height * scale),
                center_x + round(half_width * scale),
                center_y + round(half_height * scale),
            ),
            outline=(255, 220, 0),
            width=3,
        )
        draw.line((center_x - 12, center_y, center_x + 12, center_y), fill="lime", width=3)
        draw.line((center_x, center_y - 12, center_x, center_y + 12), fill="lime", width=3)
    draw.text((8, 7), f"Full QC {qc_number:03d} | frozen geometry", fill="white")

    crop_canvas = Image.new("RGB", (1200, 540), "black")
    crop_draw = ImageDraw.Draw(crop_canvas)
    for number, position in enumerate(PANEL_POSITIONS):
        crop = np.load(_crop_path(output_root, int(record["acquisition_index"]), position))
        crop_display = normalize_for_display(crop, photometric)
        image = Image.fromarray(crop_display, mode="L").convert("RGB")
        image = image.resize((500, 500), Image.Resampling.LANCZOS)
        x = 90 if number == 0 else 610
        crop_canvas.paste(image, (x, 32))
        crop_draw.text((x, 8), position.replace("_", "-"), fill="white")
    final = Image.new(
        "RGB", (1200, original_canvas.height + crop_canvas.height), color=(20, 20, 20)
    )
    final.paste(original_canvas, (0, 0))
    final.paste(crop_canvas, (0, original_canvas.height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, format="PNG", optimize=True)


def generate_full_qc(
    *,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sample_size: int = 64,
    qc_work_root: Path = Path("data/interim/oai_images/full_qc_work"),
) -> dict[str, Any]:
    """Generate anonymous representative QC previews without retaining extracted DICOMs."""

    records, failures = _load_result_records(output_root)
    if failures:
        raise FullPreprocessingError(
            "Representative QC generation requires review of preprocessing failures first"
        )
    if len(records) != EXPECTED_ACQUISITIONS:
        raise FullPreprocessingError("Full successful preprocessing is required before QC sampling")
    record_by_index = {int(record["acquisition_index"]): record for record in records}
    indices = _qc_indices(records, sample_size)
    qc_work_root.mkdir(parents=True, exist_ok=True)
    preview_root = output_root / "qc/stage_previews"
    previews: list[Path] = []
    selection_rows: list[dict[str, Any]] = []
    for qc_number, index in enumerate(indices, start=1):
        record = record_by_index[index]
        extraction = qc_work_root / f"qc_{qc_number:03d}"
        archive_path = archive_root / record["associated_file_reference"]
        contents = safe_extract_archive(archive_path, extraction)
        try:
            if contents.regular_files != 1:
                raise FullPreprocessingError("QC archive structure changed")
            dicom_path = _single_extracted_file(extraction)
            dataset = pydicom.dcmread(dicom_path, force=False)
            source = np.asarray(dataset.pixel_array)
            if source.ndim != 2 or source.dtype != np.dtype("uint16"):
                raise FullPreprocessingError("QC DICOM pixel structure changed")
            preview = preview_root / f"qc_{qc_number:03d}.png"
            _render_qc_stage(source, record, output_root, preview, qc_number=qc_number)
            previews.append(preview)
            selection_rows.append(
                {
                    "qc_number": qc_number,
                    "acquisition_index": index,
                    "manufacturer": record["manufacturer"],
                    "scanner_model": record["manufacturer_model_name"],
                    "spacing_mm": record["original_row_spacing_mm"],
                    "dimensions": f"{record['original_rows']}x{record['original_columns']}",
                    "image_release": record["image_release_study"],
                    "qc_category": record["xray_accept_qc"],
                    "laterality_state": record["laterality_state"],
                    "maximum_padding_mm": max(
                        max(panel["horizontal_padding_mm"], panel["vertical_padding_mm"])
                        for panel in record["panels"]
                    ),
                }
            )
        finally:
            if extraction.exists():
                shutil.rmtree(extraction)
    _atomic_parquet(pd.DataFrame(selection_rows), output_root / "qc/qc_selection_manifest.parquet")
    contacts: list[Path] = []
    for number, start in enumerate(range(0, len(previews), 4), start=1):
        target = output_root / "qc/contact_sheets" / f"full_qc_contact_{number:02d}.png"
        save_contact_sheet(previews[start : start + 4], target, columns=2, cell_width=600)
        contacts.append(target)
    result = {
        "anonymous_stage_previews": len(previews),
        "contact_sheets": len(contacts),
        "temporary_qc_files_remaining": sum(path.is_file() for path in qc_work_root.rglob("*")),
        "rendered_identifiers": False,
        "rendered_accessions": False,
        "rendered_dates": False,
        "rendered_paths": False,
    }
    _atomic_json(result, output_root / "qc/qc_generation_summary.json")
    return result


def apply_visual_qc_review(
    *,
    review_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    standardized_manifest_path: Path = DEFAULT_STANDARDIZED_MANIFEST,
) -> dict[str, Any]:
    """Validate and record an anonymous review of the representative QC sample.

    The review augments local manifests and exception queues. It never changes a crop or a frozen
    preprocessing parameter. Results from this deliberately stress-oriented sample are reported as
    diagnostic rates, not as full-cohort prevalence estimates.
    """

    selection_path = output_root / "qc/qc_selection_manifest.parquet"
    summary_path = output_root / "manifests/final_summary.json"
    selection = pd.read_parquet(selection_path)
    review = pd.read_csv(review_path)
    required = {
        "qc_number",
        "localization_visual_state",
        "crop_visual_state",
        "reason_code",
    }
    if set(review.columns) != required:
        raise FullPreprocessingError("Visual QC review columns do not match the expected schema")
    if review["qc_number"].duplicated().any() or selection["qc_number"].duplicated().any():
        raise FullPreprocessingError("Visual QC review contains duplicate anonymous labels")
    if set(review["qc_number"]) != set(selection["qc_number"]):
        raise FullPreprocessingError("Visual QC review does not cover the selected sample exactly")

    allowed_localization = {"SUCCESS", "BORDERLINE", "FAILED"}
    allowed_crop = {"ADEQUATE", "BORDERLINE", "INADEQUATE"}
    if not set(review["localization_visual_state"]).issubset(allowed_localization):
        raise FullPreprocessingError("Visual QC review has an invalid localization state")
    if not set(review["crop_visual_state"]).issubset(allowed_crop):
        raise FullPreprocessingError("Visual QC review has an invalid crop state")
    expected_pairs = {
        ("SUCCESS", "ADEQUATE"),
        ("BORDERLINE", "BORDERLINE"),
        ("FAILED", "INADEQUATE"),
    }
    actual_pairs = set(
        review[["localization_visual_state", "crop_visual_state"]].itertuples(
            index=False, name=None
        )
    )
    if not actual_pairs.issubset(expected_pairs):
        raise FullPreprocessingError("Visual localization and crop states are inconsistent")

    merged = selection.merge(review, on="qc_number", how="inner", validate="one_to_one")
    panels = pd.read_parquet(standardized_manifest_path)
    excessive_by_acquisition = panels.groupby("acquisition_index")["excessive_padding"].any()
    merged["outside_validation_padding_envelope"] = (
        merged["acquisition_index"].map(excessive_by_acquisition).fillna(False).astype(bool)
    )
    _atomic_parquet(merged, output_root / "qc/visual_review_manifest.parquet")

    localization_map = merged.set_index("acquisition_index")["localization_visual_state"].to_dict()
    crop_map = merged.set_index("acquisition_index")["crop_visual_state"].to_dict()
    panels["visual_qc_reviewed"] = panels["acquisition_index"].isin(localization_map)
    panels["visual_localization_state"] = panels["acquisition_index"].map(localization_map)
    panels["visual_crop_adequacy_state"] = panels["acquisition_index"].map(crop_map)
    _atomic_parquet(panels, standardized_manifest_path)

    exception_root = output_root / "exceptions"
    failure_rows = merged.loc[
        merged["localization_visual_state"].eq("FAILED"), ["acquisition_index"]
    ].copy()
    failure_rows = failure_rows.assign(
        category="LOCALIZATION_FAILURE",
        stage="representative_visual_qc",
        error_type="VisualLocalizationFailure",
        error_message="Frozen localizer did not center the tibiofemoral joint",
    )
    _atomic_csv(failure_rows, exception_root / "localization_failure.csv")

    crop_failure_rows = merged.loc[
        merged["crop_visual_state"].eq("INADEQUATE"), ["acquisition_index"]
    ].copy()
    crop_failure_rows = crop_failure_rows.assign(
        category="CROP_INADEQUACY",
        stage="representative_visual_qc",
        error_type="VisualCropInadequacy",
        error_message="Frozen crop did not retain adequate tibiofemoral anatomy",
    )
    _atomic_csv(crop_failure_rows, exception_root / "crop_inadequacy.csv")

    borderline_columns = ["acquisition_index", "reason_code"]
    _atomic_csv(
        merged.loc[merged["localization_visual_state"].eq("BORDERLINE"), borderline_columns],
        exception_root / "localization_borderline.csv",
    )
    _atomic_csv(
        merged.loc[merged["crop_visual_state"].eq("BORDERLINE"), borderline_columns],
        exception_root / "crop_borderline.csv",
    )

    def counts_and_rates(column: str) -> dict[str, Any]:
        counts = merged[column].value_counts().to_dict()
        return {
            "success_or_adequate": int(counts.get("SUCCESS", counts.get("ADEQUATE", 0))),
            "borderline": int(counts.get("BORDERLINE", 0)),
            "failed_or_inadequate": int(counts.get("FAILED", counts.get("INADEQUATE", 0))),
            "success_or_adequate_percent": round(
                counts.get("SUCCESS", counts.get("ADEQUATE", 0)) / len(merged) * 100,
                3,
            ),
            "borderline_percent": round(counts.get("BORDERLINE", 0) / len(merged) * 100, 3),
            "failed_or_inadequate_percent": round(
                counts.get("FAILED", counts.get("INADEQUATE", 0)) / len(merged) * 100,
                3,
            ),
        }

    visual_groups: dict[str, list[dict[str, Any]]] = {}
    for name, column in {
        "manufacturer": "manufacturer",
        "scanner_model": "scanner_model",
        "pixel_spacing_mm": "spacing_mm",
        "image_dimensions": "dimensions",
        "image_release": "image_release",
        "oai_qc_category": "qc_category",
    }.items():
        pooled = _pooled_groups(merged[column])
        entries: list[dict[str, Any]] = []
        for value in sorted(pooled.unique()):
            subset = merged.loc[pooled.eq(value)]
            entries.append(
                {
                    "group": value,
                    "reviewed_acquisitions": len(subset),
                    "successful": int(subset["localization_visual_state"].eq("SUCCESS").sum()),
                    "borderline": int(subset["localization_visual_state"].eq("BORDERLINE").sum()),
                    "failed": int(subset["localization_visual_state"].eq("FAILED").sum()),
                }
            )
        visual_groups[name] = entries

    padding_groups = []
    for flagged, subset in merged.groupby("outside_validation_padding_envelope", sort=False):
        padding_groups.append(
            {
                "outside_validation_padding_envelope": bool(flagged),
                "reviewed_acquisitions": len(subset),
                "successful": int(subset["localization_visual_state"].eq("SUCCESS").sum()),
                "borderline": int(subset["localization_visual_state"].eq("BORDERLINE").sum()),
                "failed": int(subset["localization_visual_state"].eq("FAILED").sum()),
            }
        )

    visual_result = {
        "sample_size_acquisitions": len(merged),
        "sample_design": "representative stress sample enriched for maximum padding",
        "rates_are_not_full_cohort_prevalence_estimates": True,
        "localization": counts_and_rates("localization_visual_state"),
        "crop_adequacy": counts_and_rates("crop_visual_state"),
        "padding_strata": padding_groups,
        "heterogeneity": visual_groups,
        "materially_worse_than_independent_validation": bool(
            merged["localization_visual_state"].eq("SUCCESS").mean() < 0.98
            or merged["crop_visual_state"].eq("ADEQUATE").mean() < 0.98
        ),
        "frozen_parameters_changed": False,
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["representative_visual_qc"] = visual_result
    summary["full_set_status"] = "STOPPED_FOR_HUMAN_REVIEW"
    summary["ready_for_analysis_cohort_linkage"] = False
    summary["systematic_preprocessing_failure_groups"] = (
        "See representative_visual_qc; high-padding failures require human review"
    )
    manifest_files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and "crops" not in path.relative_to(output_root).parts
    ] + [DEFAULT_ACQUISITION_MANIFEST, standardized_manifest_path]
    summary["storage"]["manifest_and_qc_bytes"] = sum(
        path.stat().st_size for path in set(manifest_files) if path.exists()
    )
    _atomic_json(summary, summary_path)
    return visual_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or incrementally preprocess the full frozen baseline X-ray set."
    )
    parser.add_argument(
        "stage",
        choices=("audit", "process", "finalize", "qc", "verify-hashes", "all"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results: dict[str, Any] = {}
    if args.stage in {"audit", "all"}:
        results["audit"] = audit_full_download()
    if args.stage in {"process", "all"}:
        results["processing"] = run_full_preprocessing()
    if args.stage in {"finalize", "all"}:
        results["summary"] = finalize_full_preprocessing()
    if args.stage in {"qc", "all"}:
        results["qc"] = generate_full_qc()
    if args.stage in {"verify-hashes", "all"}:
        results["source_hashes"] = verify_archive_hashes()
    print(json.dumps(results, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
