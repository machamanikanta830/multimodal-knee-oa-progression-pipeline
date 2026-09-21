"""Synthetic-only tests for outcome-blind OAI X-ray pilot preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from imaging.pilot_qc import (
    ANALYSIS_INPUT_COLUMNS,
    build_acquisition_candidates,
    prepare_pilot,
    select_pilot_accessions,
    summarize_pilot,
)

pytestmark = pytest.mark.public_portable


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    analysis_rows = []
    manifest_rows = []
    participant_number = 0
    for grade in range(4):
        for knee_count in (1, 2):
            for replicate in range(5):
                participant_number += 1
                participant = f"P{participant_number:03d}"
                source_participant = f"S{participant_number:03d}"
                accession = f"ACCESSION-{grade}-{knee_count}-{replicate}"
                for knee_index in range(knee_count):
                    side = str(knee_index + 1)
                    analysis_rows.append(
                        {
                            "participant_id": participant,
                            "knee_side_code": side,
                            "baseline_kl": grade,
                            "composite_progression": bool(replicate % 2),
                        }
                    )
                    manifest_rows.append(
                        {
                            "participant_id": participant,
                            "source_participant_id": source_participant,
                            "knee_side_code": side,
                            "baseline_visit": "V00",
                            "read_project": "15",
                            "baseline_barcode": accession,
                            "accession_number": accession,
                            "image_file": f"associated/synthetic/{accession}/image/file.tar.gz",
                            "image_description": "Bilateral PA Fixed Flexion Knee",
                            "image_file_format": "DICOM",
                            "image_modality": "X-Ray",
                            "image_release_study": "Synthetic release",
                            "xrmeta_side_code": "3",
                            "xray_completed": "1",
                            "xray_accept_qc": ["Y", "YD", "NR", "Y", "YD"][replicate],
                            "xray_alignment_problem": "X" if replicate == 1 else pd.NA,
                            "xray_centering_problem": "X" if replicate == 2 else pd.NA,
                            "xray_incomplete_depiction": pd.NA,
                            "xray_exam_type": "Bilateral PA Fixed Flexion Knee",
                            "xray_positioning_problem": "X" if replicate == 3 else pd.NA,
                            "bilateral_acquisition": True,
                            "knees_sharing_accession": knee_count,
                        }
                    )
    analysis_path = tmp_path / "analysis.parquet"
    manifest_path = tmp_path / "manifest.parquet"
    metadata_path = tmp_path / "package_metadata.csv.gz"
    pd.DataFrame(analysis_rows).to_parquet(analysis_path, index=False)
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    pd.DataFrame(
        {
            "NDA_S3_URL": ["s3://synthetic-package/tabular.txt"],
            "DOWNLOAD_ALIAS": ["tabular.txt"],
        }
    ).to_csv(metadata_path, index=False, compression="gzip")
    return analysis_path, manifest_path, metadata_path


def test_selects_32_unique_balanced_accessions_without_outcome_columns(tmp_path: Path) -> None:
    analysis_path, manifest_path, _ = _write_synthetic_inputs(tmp_path)

    acquisitions, linked = build_acquisition_candidates(analysis_path, manifest_path)
    pilot = select_pilot_accessions(acquisitions)
    report = summarize_pilot(pilot, linked)

    assert ANALYSIS_INPUT_COLUMNS == ["participant_id", "knee_side_code", "baseline_kl"]
    assert len(pilot) == 32
    assert pilot["accession_number"].nunique() == 32
    assert report["acquisition_kl_stratum"] == {str(grade): 8 for grade in range(4)}
    assert report["eligible_knees_per_acquisition"] == {"1": 16, "2": 16}
    assert report["eligible_knees_represented"] == 48
    assert not pilot["selection_outcomes_read"].any()
    assert "composite_progression" not in pilot.columns


def test_pilot_selection_is_deterministic(tmp_path: Path) -> None:
    analysis_path, manifest_path, _ = _write_synthetic_inputs(tmp_path)
    acquisitions, _ = build_acquisition_candidates(analysis_path, manifest_path)

    first = select_pilot_accessions(acquisitions)
    second = select_pilot_accessions(acquisitions.sample(frac=1, random_state=7))

    assert first["accession_number"].tolist() == second["accession_number"].tolist()


def test_prepare_writes_local_manifest_and_reference_list_but_no_images(tmp_path: Path) -> None:
    analysis_path, manifest_path, metadata_path = _write_synthetic_inputs(tmp_path)
    pilot_path = tmp_path / "processed" / "pilot.parquet"
    references_path = tmp_path / "processed" / "references.txt"

    report = prepare_pilot(
        analysis_cohort_path=analysis_path,
        image_manifest_path=manifest_path,
        package_metadata_path=metadata_path,
        pilot_manifest_path=pilot_path,
        retrieval_references_path=references_path,
    )

    pilot = pd.read_parquet(pilot_path)
    references = references_path.read_text(encoding="utf-8").splitlines()
    assert len(pilot) == 32
    assert len(references) == 32
    assert len(set(references)) == 32
    assert report["images_downloaded_by_this_command"] is False
    assert report["dicom_qc"]["valid_dicoms"] == 0
    assert report["storage_estimate"]["available"] is False
    assert report["retrieval"]["pilot_reference_exact_matches"] == 0
    assert not report["retrieval"]["direct_retrieval_from_existing_package_supported"]
    serialized = json.dumps(report)
    assert "P001" not in serialized
    assert "ACCESSION-" not in serialized
