"""Prepare local-only NDA artifacts for the frozen full baseline imaging set.

The command validates the existing cohort-linked V00 image manifest against ``image03`` and writes
only ignored package-preparation files. It never queries NDA, downloads images, changes the cohort,
creates modeling splits, or preprocesses pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from imaging.pilot_qc import DEFAULT_ANALYSIS_COHORT, DEFAULT_IMAGE_MANIFEST
from imaging.validation_sample import (
    DEFAULT_IMAGE03,
    EXPECTED_DESCRIPTION,
    EXPECTED_FORMAT,
    EXPECTED_MODALITY,
    EXPECTED_SCAN_OBJECT,
    EXPECTED_SCAN_TYPE,
    ValidationSelectionError,
    build_validation_candidates,
    extract_image03_subset,
)

EXPECTED_ACQUISITIONS = 3621
DEFAULT_FULL_IMAGE03 = Path("data/processed/manifests/v00_xray_full_image03.txt")
DEFAULT_FULL_GUIDS = Path("data/processed/manifests/v00_xray_full_guids.txt")
DEFAULT_FULL_REFERENCES = Path("data/processed/manifests/v00_xray_full_associated_files.txt")


class FullAcquisitionError(ValueError):
    """Raised when the approved full imaging set cannot be reproduced exactly."""


def _atomic_bytes_write(chunks: list[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(lines: list[str], target: Path) -> None:
    _atomic_bytes_write([f"{line}\n".encode() for line in lines], target)


def validate_full_acquisition_set(
    acquisitions: pd.DataFrame,
    linked_knees: pd.DataFrame,
    *,
    expected_acquisitions: int = EXPECTED_ACQUISITIONS,
) -> dict[str, Any]:
    """Validate the approved acquisition set and return aggregate-only counts."""

    required_acquisition = {
        "accession_number",
        "participant_id",
        "nda_guid",
        "image_file",
        "baseline_visit",
        "read_project",
        "bilateral_acquisition",
        "image_description",
        "xray_exam_type",
        "image03_description",
        "image_file_format",
        "image03_file_format",
        "image_modality",
        "image03_modality",
        "image03_scan_type",
        "image03_scan_object",
        "image03_visit",
        "xrmeta_side_code",
    }
    required_knee = {"participant_id", "knee_side_code", "accession_number"}
    if missing := required_acquisition.difference(acquisitions.columns):
        raise FullAcquisitionError(f"Full acquisition fields are missing: {sorted(missing)}")
    if missing := required_knee.difference(linked_knees.columns):
        raise FullAcquisitionError(f"Linked-knee fields are missing: {sorted(missing)}")
    if len(acquisitions) != expected_acquisitions:
        raise FullAcquisitionError(
            f"Expected {expected_acquisitions} acquisitions, found {len(acquisitions)}"
        )
    if acquisitions["accession_number"].duplicated().any():
        raise FullAcquisitionError("Full imaging set contains duplicate acquisitions")
    if acquisitions["image_file"].duplicated().any():
        raise FullAcquisitionError("Full imaging set contains duplicate associated-file references")
    if acquisitions[["participant_id", "accession_number"]].duplicated().any():
        raise FullAcquisitionError("Full imaging set contains duplicate participant acquisitions")
    if acquisitions["nda_guid"].isna().any() or acquisitions["nda_guid"].astype(str).eq("").any():
        raise FullAcquisitionError("A full-set acquisition is missing its NDA GUID")
    if (
        not acquisitions["participant_id"]
        .astype(str)
        .eq(acquisitions["nda_guid"].astype(str))
        .all()
    ):
        raise FullAcquisitionError("Participant linkage disagrees with the NDA GUID")

    expected_values = {
        "baseline_visit": "V00",
        "read_project": "15",
        "image_description": EXPECTED_DESCRIPTION,
        "xray_exam_type": EXPECTED_DESCRIPTION,
        "image03_description": EXPECTED_DESCRIPTION,
        "image_file_format": EXPECTED_FORMAT,
        "image03_file_format": EXPECTED_FORMAT,
        "image_modality": EXPECTED_MODALITY,
        "image03_modality": EXPECTED_MODALITY,
        "image03_scan_type": EXPECTED_SCAN_TYPE,
        "image03_scan_object": EXPECTED_SCAN_OBJECT,
        "image03_visit": "V00",
        "xrmeta_side_code": "3",
    }
    for column, expected in expected_values.items():
        if not acquisitions[column].astype(str).eq(expected).all():
            raise FullAcquisitionError(f"Full-set {column} is not uniformly {expected}")
    if not acquisitions["bilateral_acquisition"].astype(bool).all():
        raise FullAcquisitionError("A full-set acquisition is not marked bilateral")
    references = acquisitions["image_file"].astype(str)
    if not references.str.endswith(".tar.gz").all():
        raise FullAcquisitionError("A full-set associated-file reference is not a .tar.gz archive")

    knee_keys = ["participant_id", "knee_side_code"]
    if linked_knees.duplicated(knee_keys).any():
        raise FullAcquisitionError("The linked analysis cohort contains duplicate knee keys")
    counts = linked_knees.groupby("accession_number", dropna=False).size()
    if not counts.between(1, 2).all() or set(counts.index) != set(acquisitions["accession_number"]):
        raise FullAcquisitionError("Acquisition-to-knee linkage is incomplete or exceeds two knees")
    participant_map = acquisitions[["accession_number", "participant_id"]]
    checked = linked_knees[["accession_number", "participant_id"]].merge(
        participant_map,
        on="accession_number",
        how="left",
        validate="many_to_one",
        suffixes=("_knee", "_acquisition"),
    )
    if (
        checked["participant_id_acquisition"].isna().any()
        or not checked["participant_id_knee"]
        .astype(str)
        .eq(checked["participant_id_acquisition"].astype(str))
        .all()
    ):
        raise FullAcquisitionError("A knee is not linked to its acquisition participant")

    one_knee = int(counts.eq(1).sum())
    two_knee = int(counts.eq(2).sum())
    return {
        "intended_acquisitions": len(acquisitions),
        "unique_acquisition_references": int(acquisitions["accession_number"].nunique()),
        "unique_associated_file_references": int(references.nunique()),
        "duplicate_associated_file_references": int(references.duplicated().sum()),
        "unique_participants": int(acquisitions["participant_id"].nunique()),
        "unique_nda_guids": int(acquisitions["nda_guid"].nunique()),
        "linked_participant_knees": len(linked_knees),
        "one_knee_acquisitions": one_knee,
        "two_knee_acquisitions": two_knee,
        "all_v00": True,
        "all_read_project_15": True,
        "all_bilateral_pa_fixed_flexion": True,
        "all_linked_to_analysis_cohort": True,
    }


def prepare_full_acquisition_artifacts(
    *,
    analysis_cohort_path: str | Path = DEFAULT_ANALYSIS_COHORT,
    image_manifest_path: str | Path = DEFAULT_IMAGE_MANIFEST,
    image03_path: str | Path = DEFAULT_IMAGE03,
    output_image03_path: str | Path = DEFAULT_FULL_IMAGE03,
    output_guids_path: str | Path = DEFAULT_FULL_GUIDS,
    output_references_path: str | Path = DEFAULT_FULL_REFERENCES,
) -> dict[str, Any]:
    """Create exact ignored NDA preparation artifacts and return aggregate validation."""

    try:
        acquisitions, linked_knees = build_validation_candidates(
            analysis_cohort_path, image_manifest_path, image03_path
        )
    except ValidationSelectionError as error:
        raise FullAcquisitionError(str(error)) from error
    summary = validate_full_acquisition_set(acquisitions, linked_knees)

    accession_set = set(acquisitions["accession_number"].astype(str))
    image03_chunks = extract_image03_subset(image03_path, accession_set)
    guids = sorted(set(acquisitions["nda_guid"].astype(str)))
    references = sorted(acquisitions["image_file"].astype(str))
    if len(image03_chunks) != EXPECTED_ACQUISITIONS + 2:
        raise FullAcquisitionError("Full image03 subset does not contain exactly 3,621 records")
    if len(references) != EXPECTED_ACQUISITIONS or len(set(references)) != len(references):
        raise FullAcquisitionError("Full associated-file output is not exactly 3,621 unique lines")
    if len(guids) != summary["unique_nda_guids"]:
        raise FullAcquisitionError("Full GUID output does not contain each unique GUID once")

    _atomic_bytes_write(image03_chunks, Path(output_image03_path))
    _atomic_text_write(guids, Path(output_guids_path))
    _atomic_text_write(references, Path(output_references_path))
    return {
        **summary,
        "image03_records_written": len(image03_chunks) - 2,
        "guid_lines_written": len(guids),
        "associated_file_lines_written": len(references),
        "output_contains_identifiers": True,
        "output_must_remain_git_ignored": True,
        "nda_package_instructions": {
            "collection": "Osteoarthritis Initiative (OAI)",
            "structure": "Image / image03",
            "guid_count": len(guids),
            "include_associated_data_files": True,
            "browser_automation_used": False,
        },
        "download_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare exact ignored NDA artifacts for 3,621 baseline X-ray acquisitions."
    )
    parser.add_argument("--analysis-cohort", type=Path, default=DEFAULT_ANALYSIS_COHORT)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--image03", type=Path, default=DEFAULT_IMAGE03)
    parser.add_argument("--output-image03", type=Path, default=DEFAULT_FULL_IMAGE03)
    parser.add_argument("--output-guids", type=Path, default=DEFAULT_FULL_GUIDS)
    parser.add_argument("--output-references", type=Path, default=DEFAULT_FULL_REFERENCES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_full_acquisition_artifacts(
        analysis_cohort_path=args.analysis_cohort,
        image_manifest_path=args.image_manifest,
        image03_path=args.image03,
        output_image03_path=args.output_image03,
        output_guids_path=args.output_guids,
        output_references_path=args.output_references,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
