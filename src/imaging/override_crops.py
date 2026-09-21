"""Regeneration of single knee panels from human-supplied tibiofemoral joint centers.

A reviewer who chooses ``NEEDS_CENTER_OVERRIDE`` supplies a corrected joint center in
resampled-panel coordinates. This module rebuilds only that panel's crop using the frozen downstream
specification: 0.15 mm/pixel resampling, a 160 x 160 mm physical crop, a 1067 x 1067 uint16 output,
and the same low-percentile padding policy as the automatic pass.

Localization V3 is never re-run and no threshold is touched: the human center replaces only the
coordinate the frozen crop step consumes. The automatic crop is never modified or deleted. Corrected
crops are written to a separate ``overrides`` tree as human-adjudicated derivatives, so the frozen
automatic dataset and the adjudicated dataset remain independently auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom

from imaging.archive_extract import UnsafeArchiveError, safe_extract_archive
from imaging.artifact_io import atomic_json, atomic_parquet, sha256_file
from imaging.dicom_inspect import inspect_dicom
from imaging.frozen_full_v3 import (
    FROZEN_SOURCE_FILES,
    LATERALITY_POLICY_VERSION,
    LOCALIZER_VERSION,
    PREPROCESSING_VERSION,
    V3_FREEZE_RECORDS,
)
from imaging.pixel_qc import split_bilateral_midpoint
from imaging.preprocessing import crop_around_center, resample_to_spacing
from imaging.v3_review_ui import (
    ADJUDICATION_SCHEMA_VERSION,
    PANEL_POSITIONS,
    ReviewSession,
)

OVERRIDE_CROP_VERSION = "v3_frozen_full_override_crop_v1"
TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
OUTPUT_SHAPE = (1067, 1067)
PADDING_PERCENTILE = 0.5

DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v3_frozen_full")
DEFAULT_ARCHIVE_ROOT = Path("data/raw/oai_images/full")
DEFAULT_WORK_ROOT = Path("data/interim/oai_images/v3_frozen_full_override_work")
DEFAULT_ACQUISITION_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests/v3_frozen_full_acquisitions.parquet"


class OverrideCropError(ValueError):
    """Raised when a corrected crop cannot be produced safely."""


def _authoritative_localizer_hash() -> str:
    """Return the authoritative source hash of Localization V3, verified against freeze record."""
    freeze_record_path = Path(V3_FREEZE_RECORDS[0])
    try:
        freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record contains malformed JSON"
        ) from error
    except OSError as error:
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record cannot be read"
        ) from error

    if not isinstance(freeze_record, dict):
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record has an unexpected structure"
        )
    if "source_sha256" not in freeze_record:
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record is missing source_sha256"
        )
    approved_hash = freeze_record["source_sha256"]
    if approved_hash is None or approved_hash == "":
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record has an empty source_sha256"
        )
    if (
        not isinstance(approved_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", approved_hash) is None
    ):
        raise OverrideCropError(
            "The authoritative Localization V3 freeze record has an invalid source_sha256"
        )

    source_path = Path(FROZEN_SOURCE_FILES[0])
    try:
        source_hash = sha256_file(source_path)
    except OSError as error:
        raise OverrideCropError("The frozen Localization V3 source cannot be read") from error

    if source_hash != approved_hash:
        raise OverrideCropError(
            "Current Localization V3 source hash does not match approved freeze record"
        )
    return source_hash


def _override_bundle(output_root: Path, acquisition_index: int) -> Path:
    return output_root / "overrides" / f"acquisition_{acquisition_index:04d}"


def _automatic_crop_path(output_root: Path, acquisition_index: int, position: str) -> Path:
    return output_root / "crops" / f"acquisition_{acquisition_index:04d}" / f"{position}_uint16.npy"


def pending_overrides(session: ReviewSession) -> list[dict[str, Any]]:
    """List every panel whose latest decision asks for a human-specified joint center.

    Superseded decisions are ignored, so a reviewer who revises a case does not leave an orphaned
    corrected crop request behind.
    """

    requests: list[dict[str, Any]] = []
    for review_index, record in sorted(session.latest_decisions().items()):
        for position in PANEL_POSITIONS:
            if record[f"{position}_decision"] != "NEEDS_CENTER_OVERRIDE":
                continue
            row = record[f"{position}_center_override_row"]
            column = record[f"{position}_center_override_column"]
            if row is None or column is None:
                raise OverrideCropError("An override decision is missing its corrected center")
            requests.append(
                {
                    "review_index": int(review_index),
                    "acquisition_index": int(record["acquisition_index"]),
                    "panel_position": position,
                    "manual_center_row": int(row),
                    "manual_center_column": int(column),
                    "automatic_center_row": int(record[f"{position}_automatic_center_row"]),
                    "automatic_center_column": int(record[f"{position}_automatic_center_column"]),
                    "automatic_panel_state": str(record[f"{position}_automatic_state"]),
                    "automatic_acquisition_state": str(record["automatic_state"]),
                    "decision": "NEEDS_CENTER_OVERRIDE",
                    "adjudication_sequence": int(record["sequence"]),
                    "adjudication_recorded_at_utc": str(record["recorded_at_utc"]),
                }
            )
    return requests


def _resampled_panels(
    *,
    archive_root: Path,
    archive_reference: str,
    work_root: Path,
    acquisition_index: int,
) -> dict[str, np.ndarray]:
    """Rebuild both resampled panels with the frozen steps, then discard the extracted DICOM."""

    extraction = work_root / f"acquisition_{acquisition_index:04d}"
    if extraction.exists():
        shutil.rmtree(extraction, ignore_errors=True)
    try:
        safe_extract_archive(archive_root / archive_reference, extraction)
    except (UnsafeArchiveError, FileExistsError, OSError, ValueError) as error:
        shutil.rmtree(extraction, ignore_errors=True)
        raise OverrideCropError("The raw archive could not be read for a corrected crop") from error
    try:
        files = [path for path in extraction.rglob("*") if path.is_file()]
        if len(files) != 1:
            raise OverrideCropError("An archive does not contain exactly one regular file")
        metadata = inspect_dicom(files[0])
        source = np.asarray(pydicom.dcmread(files[0], force=False).pixel_array)
        if source.ndim != 2 or source.dtype != np.dtype("uint16"):
            raise OverrideCropError("A DICOM pixel array is not a two-dimensional uint16 image")
        spacing_values = metadata.pixel_spacing or metadata.imager_pixel_spacing
        if spacing_values is None or len(spacing_values) != 2:
            raise OverrideCropError("A DICOM has no usable two-dimensional pixel spacing")
        spacing = (float(spacing_values[0]), float(spacing_values[1]))
        if any(not np.isfinite(value) or value <= 0 for value in spacing):
            raise OverrideCropError("A DICOM has invalid pixel spacing")
        screen_left, screen_right = split_bilateral_midpoint(source)
        return {
            position: resample_to_spacing(panel, spacing, (TARGET_SPACING_MM, TARGET_SPACING_MM))
            for position, panel in (
                ("screen_left", screen_left),
                ("screen_right", screen_right),
            )
        }
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def _publish(bundle: Path, position: str, crop: np.ndarray, record: dict[str, Any]) -> Path:
    """Write one corrected crop and its record so a reader never sees a partial bundle."""

    bundle.mkdir(parents=True, exist_ok=True)
    target = bundle / f"{position}_uint16.npy"
    temporary = bundle / f".{position}_uint16.{os.getpid()}.npy"
    try:
        np.save(temporary, crop, allow_pickle=False)
        stored = np.load(temporary, mmap_mode="r")
        if stored.shape != OUTPUT_SHAPE or stored.dtype != np.dtype("uint16"):
            raise OverrideCropError("A corrected crop failed post-write validation")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(record, bundle / f"{position}_record.json")
    return target


def verify_existing_override(
    bundle: Path,
    request: dict[str, Any],
    *,
    known_sequences: set[int],
    output_root: Path | None = None,
    session: ReviewSession | None = None,
    automatic_record: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Decide whether an existing corrected crop may be reused, failing closed on any mismatch.

    Returns the stored record when it is proven to be the exact crop the active adjudication asks
    for and every scientifically relevant property matches independently derived authoritative
    state. Normal reuse is allowed only for the current effective adjudication sequence; a crop
    belonging to a superseded revision fails closed and is never silently reused or rebuilt.
    Any mismatch or missing file raises OverrideCropError rather than quietly reusing or
    overwriting a crop whose provenance cannot be verified.
    """

    if output_root is None:
        output_root = bundle.parent.parent
    output_root = Path(output_root)
    bundle = Path(bundle)

    position = str(request["panel_position"])
    acquisition_index = int(request["acquisition_index"])
    review_index = int(request["review_index"])
    record_path = bundle / f"{position}_record.json"
    crop_path = bundle / f"{position}_uint16.npy"
    if not record_path.is_file():
        return None
    try:
        stored = json.loads(record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise OverrideCropError(
            "An existing corrected crop record is unreadable; refusing to reuse or overwrite it"
        ) from error

    # 1. Authoritative automatic completion record & panel
    if automatic_record is None:
        auto_record_path = (
            output_root / "crops" / f"acquisition_{acquisition_index:04d}" / "record.json"
        )
        if not auto_record_path.is_file():
            raise OverrideCropError(
                "Authoritative automatic completion record is missing; refusing to reuse override"
            )
        try:
            automatic_record = json.loads(auto_record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise OverrideCropError(
                "Authoritative automatic completion record is unreadable; refusing to reuse override"
            ) from error

    auto_panels = {
        panel.get("panel_position"): panel
        for panel in automatic_record.get("panels", [])
        if isinstance(panel, dict)
    }
    if position not in auto_panels:
        raise OverrideCropError(
            f"Authoritative automatic record has no panel record for {position}"
        )
    auto_panel = auto_panels[position]
    auto_row = auto_panel.get("joint_row_resampled")
    auto_column = auto_panel.get("joint_column_resampled")
    if auto_row is None or auto_column is None:
        raise OverrideCropError(
            "Authoritative automatic panel record is missing joint center coordinates"
        )
    auto_row = int(auto_row)
    auto_column = int(auto_column)

    # 2. Authoritative methodology versions & source hashes
    authoritative_localizer_hash = _authoritative_localizer_hash()
    auto_localizer_hash = automatic_record.get("frozen_localizer_source_sha256")
    if auto_localizer_hash and auto_localizer_hash != authoritative_localizer_hash:
        raise OverrideCropError(
            "Automatic record localizer hash does not match authoritative frozen localizer hash"
        )

    if stored.get("override_crop_version") != OVERRIDE_CROP_VERSION:
        raise OverrideCropError(
            "An existing corrected crop has an incompatible override crop version"
        )
    if stored.get("adjudication_schema_version") != ADJUDICATION_SCHEMA_VERSION:
        raise OverrideCropError(
            "An existing corrected crop has an incompatible adjudication schema version"
        )
    if stored.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise OverrideCropError(
            "An existing corrected crop has an incompatible preprocessing version"
        )
    if stored.get("localizer_version") != LOCALIZER_VERSION:
        raise OverrideCropError("An existing corrected crop has an incompatible localizer version")
    if stored.get("laterality_policy_version") != LATERALITY_POLICY_VERSION:
        raise OverrideCropError(
            "An existing corrected crop has an incompatible laterality policy version"
        )

    # Validate localizer source hash in stored record
    stored_localizer_hashes = []
    if "frozen_localizer_source_sha256" in stored:
        stored_localizer_hashes.append(stored.get("frozen_localizer_source_sha256"))
    if "localizer_hash" in stored:
        stored_localizer_hashes.append(stored.get("localizer_hash"))
    if not stored_localizer_hashes:
        raise OverrideCropError("An existing corrected crop has no recorded localizer hash")
    for h in stored_localizer_hashes:
        if h != authoritative_localizer_hash:
            raise OverrideCropError(
                "An existing corrected crop localizer hash does not match the approved frozen localizer"
            )

    # 3. Adjudication sequence & superseded revision check (fail closed on superseded revision)
    recorded_sequence = stored.get("adjudication_sequence")
    if not isinstance(recorded_sequence, int) or recorded_sequence not in known_sequences:
        raise OverrideCropError(
            "An existing corrected crop names an adjudication revision that is not in the log"
        )
    active_sequence = int(request["adjudication_sequence"])
    if recorded_sequence > active_sequence:
        raise OverrideCropError(
            "An existing corrected crop was built from a newer adjudication than the active one"
        )
    if recorded_sequence < active_sequence:
        raise OverrideCropError(
            "An existing corrected crop belongs to a superseded adjudication revision; active reuse is prohibited"
        )

    # 4. Panel identity
    if stored.get("panel_position") != position:
        raise OverrideCropError("An existing corrected crop record has a mismatched panel position")

    # 5. Automatic center
    if (
        stored.get("automatic_center_row") != auto_row
        or stored.get("automatic_center_column") != auto_column
        or stored.get("automatic_center_row") != request.get("automatic_center_row")
        or stored.get("automatic_center_column") != request.get("automatic_center_column")
    ):
        raise OverrideCropError(
            "An existing corrected crop disagrees with the authoritative automatic joint center"
        )

    # 6. Manual center
    if stored.get("manual_center_row") != request.get("manual_center_row") or stored.get(
        "manual_center_column"
    ) != request.get("manual_center_column"):
        raise OverrideCropError(
            "An existing corrected crop disagrees with its own adjudication revision"
        )

    # 7. Center shift
    expected_shift_rows = int(request["manual_center_row"]) - auto_row
    expected_shift_cols = int(request["manual_center_column"]) - auto_column
    if (
        stored.get("center_shift_rows") != expected_shift_rows
        or stored.get("center_shift_columns") != expected_shift_cols
    ):
        raise OverrideCropError(
            "An existing corrected crop recorded center shift does not match the authoritative shift"
        )

    # 8. Output path and acquisition bundle linkage
    expected_override_rel = f"overrides/acquisition_{acquisition_index:04d}/{position}_uint16.npy"
    if stored.get("override_crop_relative_path") != expected_override_rel:
        raise OverrideCropError(
            "An existing corrected crop relative path does not match the expected bundle path"
        )
    if stored.get("acquisition_index") != acquisition_index:
        raise OverrideCropError(
            "An existing corrected crop record has a mismatched acquisition index"
        )
    if stored.get("review_index") != review_index:
        raise OverrideCropError("An existing corrected crop record has a mismatched review index")
    if crop_path.resolve() != (output_root / expected_override_rel).resolve():
        raise OverrideCropError(
            "The existing corrected crop file path does not match the expected path"
        )

    # 9. Automatic crop file & SHA-256
    automatic_crop_path = _automatic_crop_path(output_root, acquisition_index, position)
    if not automatic_crop_path.is_file():
        raise OverrideCropError("Authoritative automatic crop file is missing")
    actual_auto_sha256 = sha256_file(automatic_crop_path)
    if stored.get("automatic_crop_sha256") != actual_auto_sha256:
        raise OverrideCropError(
            "An existing corrected crop records an automatic crop hash that does not match the actual automatic crop"
        )
    expected_auto_rel = automatic_crop_path.relative_to(output_root).as_posix()
    if stored.get("automatic_crop_relative_path") != expected_auto_rel:
        raise OverrideCropError(
            "An existing corrected crop record has wrong automatic crop relative path"
        )

    # 10. Corrected crop file, shape, dtype, and SHA-256
    if not crop_path.is_file():
        raise OverrideCropError("An existing corrected crop record has no crop file beside it")
    digest = stored.get("corrected_crop_sha256")
    if not isinstance(digest, str) or not digest:
        raise OverrideCropError("An existing corrected crop has no recorded content hash")
    array = np.load(crop_path, mmap_mode="r")
    if tuple(array.shape) != OUTPUT_SHAPE:
        raise OverrideCropError("An existing corrected crop has the wrong shape")
    if array.dtype != np.dtype("uint16"):
        raise OverrideCropError("An existing corrected crop has the wrong dtype")
    if sha256_file(crop_path) != digest:
        raise OverrideCropError("An existing corrected crop does not match its recorded hash")
    if stored.get("crop_rows") != OUTPUT_SHAPE[0] or stored.get("crop_columns") != OUTPUT_SHAPE[1]:
        raise OverrideCropError("An existing corrected crop record has wrong crop dimensions")
    if stored.get("crop_dtype") != "uint16":
        raise OverrideCropError("An existing corrected crop record has wrong crop dtype")
    if stored.get("target_spacing_mm") != TARGET_SPACING_MM:
        raise OverrideCropError("An existing corrected crop record has wrong target spacing")
    if stored.get("crop_size_mm") != CROP_SIZE_MM:
        raise OverrideCropError("An existing corrected crop record has wrong crop size")

    return stored


def regenerate_override_crops(
    *,
    session: ReviewSession | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    acquisition_manifest_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Rebuild every requested corrected crop and record its provenance.

    The pass is resumable: a corrected crop whose recorded manual center already matches the latest
    adjudication is left alone, and a revised center regenerates only the affected panel.
    """

    output_root = Path(output_root)
    if session is None:
        session = ReviewSession(output_root=output_root)
    manifest_path = acquisition_manifest_path or (
        output_root / "manifests/v3_frozen_full_acquisitions.parquet"
    )
    requests = pending_overrides(session)
    if not requests:
        summary = {
            "override_crop_version": OVERRIDE_CROP_VERSION,
            "adjudication_schema_version": ADJUDICATION_SCHEMA_VERSION,
            "requested_panels": 0,
            "regenerated_panels": 0,
            "reused_panels": 0,
            "failed_panels": 0,
            "automatic_crops_modified": False,
            "temporary_files_remaining": 0,
        }
        atomic_json(summary, output_root / "overrides/override_summary.json")
        return summary

    linkage = pd.read_parquet(
        manifest_path, columns=["acquisition_index", "associated_file_reference"]
    ).set_index("acquisition_index")["associated_file_reference"]
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    regenerated = 0
    reused = 0
    failures: list[dict[str, Any]] = []
    by_acquisition: dict[int, list[dict[str, Any]]] = {}
    for request in requests:
        by_acquisition.setdefault(request["acquisition_index"], []).append(request)

    authoritative_localizer_hash = _authoritative_localizer_hash()
    known_sequences = {int(record["sequence"]) for record in session.adjudications()}
    for acquisition_index in sorted(by_acquisition):
        panel_requests = by_acquisition[acquisition_index]
        bundle = _override_bundle(output_root, acquisition_index)
        auto_record_path = (
            output_root / "crops" / f"acquisition_{acquisition_index:04d}" / "record.json"
        )
        auto_record = None
        if auto_record_path.is_file():
            try:
                auto_record = json.loads(auto_record_path.read_text(encoding="utf-8"))
            except Exception:
                auto_record = None
        outstanding = []
        for request in panel_requests:
            if not force:
                previous = verify_existing_override(
                    bundle,
                    request,
                    known_sequences=known_sequences,
                    output_root=output_root,
                    session=session,
                    automatic_record=auto_record,
                )
                if previous is not None:
                    rows.append(previous)
                    reused += 1
                    continue
            outstanding.append(request)
        if not outstanding:
            continue

        if acquisition_index not in linkage.index:
            for request in outstanding:
                failures.append({**request, "error": "acquisition_not_in_manifest"})
            continue
        try:
            panels = _resampled_panels(
                archive_root=Path(archive_root),
                archive_reference=str(linkage.loc[acquisition_index]),
                work_root=work_root,
                acquisition_index=acquisition_index,
            )
        except OverrideCropError as error:
            for request in outstanding:
                failures.append({**request, "error": str(error)})
            continue

        for request in outstanding:
            position = request["panel_position"]
            panel = panels[position]
            center = (request["manual_center_row"], request["manual_center_column"])
            if not 0 <= center[0] < panel.shape[0] or not 0 <= center[1] < panel.shape[1]:
                failures.append({**request, "error": "manual_center_outside_resampled_panel"})
                continue
            padding_value = int(np.percentile(panel, PADDING_PERCENTILE))
            crop, geometry = crop_around_center(
                panel, center, OUTPUT_SHAPE, padding_value=padding_value
            )
            if crop.shape != OUTPUT_SHAPE or crop.dtype != np.dtype("uint16"):
                failures.append({**request, "error": "corrected_crop_failed_specification"})
                continue
            automatic_path = _automatic_crop_path(output_root, acquisition_index, position)
            record = {
                **request,
                "override_crop_version": OVERRIDE_CROP_VERSION,
                "adjudication_schema_version": ADJUDICATION_SCHEMA_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "localizer_version": LOCALIZER_VERSION,
                "frozen_localizer_source_sha256": authoritative_localizer_hash,
                "localizer_hash": authoritative_localizer_hash,
                "laterality_policy_version": LATERALITY_POLICY_VERSION,
                "target_spacing_mm": TARGET_SPACING_MM,
                "crop_size_mm": CROP_SIZE_MM,
                "crop_rows": OUTPUT_SHAPE[0],
                "crop_columns": OUTPUT_SHAPE[1],
                "crop_dtype": "uint16",
                "resampled_rows": int(panel.shape[0]),
                "resampled_columns": int(panel.shape[1]),
                "crop_padding_value": padding_value,
                "padding_top": int(geometry.padding_top),
                "padding_bottom": int(geometry.padding_bottom),
                "padding_left": int(geometry.padding_left),
                "padding_right": int(geometry.padding_right),
                "horizontal_padding_mm": (geometry.padding_left + geometry.padding_right)
                * TARGET_SPACING_MM,
                "vertical_padding_mm": (geometry.padding_top + geometry.padding_bottom)
                * TARGET_SPACING_MM,
                "center_shift_rows": request["manual_center_row"] - request["automatic_center_row"],
                "center_shift_columns": request["manual_center_column"]
                - request["automatic_center_column"],
                "automatic_crop_relative_path": automatic_path.relative_to(output_root).as_posix(),
                "automatic_crop_preserved": automatic_path.is_file(),
                "automatic_crop_sha256": (
                    sha256_file(automatic_path) if automatic_path.is_file() else None
                ),
            }
            target = _publish(bundle, position, crop, record)
            # The corrected crop gets its own content hash so a later pass can prove the bytes it
            # is about to reuse are the bytes this adjudication produced.
            record["override_crop_relative_path"] = target.relative_to(output_root).as_posix()
            record["corrected_crop_sha256"] = sha256_file(target)
            atomic_json(record, bundle / f"{position}_record.json")
            rows.append(record)
            regenerated += 1

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["acquisition_index", "panel_position"], kind="stable", ignore_index=True
        )
    atomic_parquet(frame, output_root / "overrides/override_crops.parquet")
    if failures:
        atomic_parquet(
            pd.DataFrame(failures).sort_values(
                ["acquisition_index", "panel_position"], kind="stable", ignore_index=True
            ),
            output_root / "overrides/override_failures.parquet",
        )
    remaining = [path for path in work_root.rglob("*") if path.is_file()]
    summary = {
        "override_crop_version": OVERRIDE_CROP_VERSION,
        "adjudication_schema_version": ADJUDICATION_SCHEMA_VERSION,
        "requested_panels": len(requests),
        "regenerated_panels": regenerated,
        "reused_panels": reused,
        "reuse_integrity_checks": [
            "crop_file_exists",
            "shape_1067x1067",
            "dtype_uint16",
            "corrected_crop_sha256",
            "active_adjudication_revision",
            "automatic_crop_sha256",
            "panel_identity",
            "automatic_center",
            "manual_center",
            "center_shift",
            "adjudication_schema_version",
            "preprocessing_version",
            "localizer_version_and_source_hash",
            "laterality_policy_version",
            "override_crop_version",
            "output_path_and_linkage",
        ],
        "failed_panels": len(failures),
        "acquisitions_touched": len(by_acquisition),
        "automatic_crops_modified": False,
        "automatic_crops_present": int(frame["automatic_crop_preserved"].sum())
        if not frame.empty
        else 0,
        "temporary_files_remaining": len(remaining),
    }
    atomic_json(summary, output_root / "overrides/override_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild knee crops from reviewer-supplied joint centers using the frozen downstream "
            "crop specification, leaving every automatic crop untouched."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild corrected crops even when a matching one already exists.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the pending corrected-center requests without touching any pixels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = ReviewSession(output_root=args.dataset_root)
    if args.list_only:
        print(json.dumps(pending_overrides(session), indent=2, sort_keys=True))
        return 0
    summary = regenerate_override_crops(
        session=session,
        output_root=args.dataset_root,
        archive_root=args.archive_root,
        work_root=args.work_root,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
