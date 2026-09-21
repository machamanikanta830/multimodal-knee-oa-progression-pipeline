"""Synthetic tests for full baseline NDA artifact preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from imaging.full_acquisition import FullAcquisitionError, validate_full_acquisition_set

pytestmark = pytest.mark.public_portable


def _acquisitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accession_number": ["A1", "A2"],
            "participant_id": ["GUID1", "GUID2"],
            "nda_guid": ["GUID1", "GUID2"],
            "image_file": ["synthetic/a.tar.gz", "synthetic/b.tar.gz"],
            "baseline_visit": ["V00", "V00"],
            "read_project": ["15", "15"],
            "bilateral_acquisition": [True, True],
            "image_description": ["Bilateral PA Fixed Flexion Knee"] * 2,
            "xray_exam_type": ["Bilateral PA Fixed Flexion Knee"] * 2,
            "image03_description": ["Bilateral PA Fixed Flexion Knee"] * 2,
            "image_file_format": ["DICOM", "DICOM"],
            "image03_file_format": ["DICOM", "DICOM"],
            "image_modality": ["X-Ray", "X-Ray"],
            "image03_modality": ["X-Ray", "X-Ray"],
            "image03_scan_type": ["X-Ray", "X-Ray"],
            "image03_scan_object": ["Live", "Live"],
            "image03_visit": ["V00", "V00"],
            "xrmeta_side_code": ["3", "3"],
        }
    )


def _knees() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["GUID1", "GUID1", "GUID2"],
            "knee_side_code": ["R", "L", "R"],
            "accession_number": ["A1", "A1", "A2"],
        }
    )


def test_full_set_validation_reports_acquisitions_guids_and_knees() -> None:
    result = validate_full_acquisition_set(_acquisitions(), _knees(), expected_acquisitions=2)

    assert result["intended_acquisitions"] == 2
    assert result["unique_nda_guids"] == 2
    assert result["linked_participant_knees"] == 3
    assert result["one_knee_acquisitions"] == 1
    assert result["two_knee_acquisitions"] == 1


def test_full_set_validation_rejects_duplicate_associated_files() -> None:
    acquisitions = _acquisitions()
    acquisitions.loc[1, "image_file"] = acquisitions.loc[0, "image_file"]

    with pytest.raises(FullAcquisitionError, match="duplicate associated-file"):
        validate_full_acquisition_set(acquisitions, _knees(), expected_acquisitions=2)


def test_full_set_validation_rejects_nonbaseline_record() -> None:
    acquisitions = _acquisitions()
    acquisitions.loc[1, "baseline_visit"] = "V01"

    with pytest.raises(FullAcquisitionError, match="baseline_visit"):
        validate_full_acquisition_set(acquisitions, _knees(), expected_acquisitions=2)


def test_full_set_validation_rejects_unlinked_knee() -> None:
    knees = _knees()
    knees.loc[2, "accession_number"] = "A3"

    with pytest.raises(FullAcquisitionError, match="linkage is incomplete"):
        validate_full_acquisition_set(_acquisitions(), knees, expected_acquisitions=2)
