"""Synthetic-only tests for privacy-safe DICOM inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from imaging.dicom_inspect import (
    DicomInspectionError,
    inspect_dicom,
    inspect_paths,
    metadata_as_dict,
)

pytestmark = pytest.mark.public_portable


def _write_synthetic_dicom(
    path: Path,
    *,
    rows: int | None = 64,
    columns: int | None = 96,
    frames: int | None = None,
    optional_tags: bool = True,
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.PatientName = "PRIVATE^PERSON"
    dataset.PatientID = "PRIVATE-ID"
    dataset.AccessionNumber = "PRIVATE-ACC"
    dataset.StudyDate = "20200101"
    if rows is not None:
        dataset.Rows = rows
    if columns is not None:
        dataset.Columns = columns
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.SamplesPerPixel = 1
    dataset.PixelRepresentation = 0
    dataset.PhotometricInterpretation = "MONOCHROME2"
    if frames is not None:
        dataset.NumberOfFrames = frames
    if optional_tags:
        dataset.PixelSpacing = [0.2, 0.3]
        dataset.ImagerPixelSpacing = [0.2, 0.3]
        dataset.ViewPosition = "PA"
        dataset.Laterality = "B"
        dataset.ImageLaterality = "B"
        dataset.PatientOrientation = ["L", "F"]
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.BodyPartExamined = "KNEE"
        dataset.PresentationLUTShape = "IDENTITY"
        dataset.RescaleSlope = "1.5"
        dataset.RescaleIntercept = "-10"
        dataset.WindowCenter = "2048"
        dataset.WindowWidth = "4096"
        dataset.Manufacturer = "SYNTHETIC"
        dataset.ManufacturerModelName = "TEST-SCANNER"
        dataset.BurnedInAnnotation = "NO"
    pixel_count = (rows or 1) * (columns or 1) * (frames or 1)
    dataset.PixelData = bytes(pixel_count * 2)
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_metadata_extraction_and_image_dimensions(tmp_path: Path) -> None:
    path = _write_synthetic_dicom(tmp_path / "synthetic.dcm")

    metadata = inspect_dicom(path)

    assert (metadata.rows, metadata.columns) == (64, 96)
    assert (metadata.bits_allocated, metadata.bits_stored) == (16, 12)
    assert metadata.photometric_interpretation == "MONOCHROME2"
    assert metadata.pixel_spacing == (0.2, 0.3)
    assert metadata.modality == "DX"
    assert metadata.view_position == "PA"
    assert metadata.laterality == "B"
    assert metadata.patient_orientation == ("L", "F")
    assert metadata.number_of_frames == 1
    assert not metadata.number_of_frames_tag_present
    assert not metadata.multi_frame
    assert metadata.compressed is False
    assert metadata.rescale_slope == 1.5
    assert metadata.rescale_intercept == -10.0
    assert metadata.window_center == (2048.0,)
    assert metadata.window_width == (4096.0,)
    assert metadata.manufacturer == "SYNTHETIC"
    assert metadata.manufacturer_model_name == "TEST-SCANNER"
    assert metadata.burned_in_annotation == "NO"


def test_missing_optional_tags_and_multiframe_detection(tmp_path: Path) -> None:
    missing = inspect_dicom(
        _write_synthetic_dicom(
            tmp_path / "missing.dcm", rows=None, columns=None, optional_tags=False
        )
    )
    multiframe = inspect_dicom(
        _write_synthetic_dicom(tmp_path / "multi.dcm", frames=3, optional_tags=False)
    )

    assert missing.rows is None
    assert missing.columns is None
    assert missing.pixel_spacing is None
    assert missing.view_position is None
    assert missing.laterality is None
    assert missing.window_center is None
    assert missing.manufacturer is None
    assert multiframe.number_of_frames_tag_present
    assert multiframe.number_of_frames == 3
    assert multiframe.multi_frame


def test_invalid_file_is_graceful_and_does_not_expose_path(tmp_path: Path) -> None:
    private_name = tmp_path / "PRIVATE-PARTICIPANT-NAME.txt"
    private_name.write_bytes(b"not a dicom")

    with pytest.raises(DicomInspectionError) as error:
        inspect_dicom(private_name)
    assert "PRIVATE-PARTICIPANT-NAME" not in str(error.value)

    summary = inspect_paths([private_name])
    assert summary["valid_dicoms"] == 0
    assert summary["invalid_or_non_dicom_files"] == 1
    assert "PRIVATE-PARTICIPANT-NAME" not in json.dumps(summary)


def test_serialization_excludes_identity_dates_accessions_paths_and_pixels(tmp_path: Path) -> None:
    path = _write_synthetic_dicom(tmp_path / "PRIVATE-FILENAME.dcm")

    payload = json.dumps(metadata_as_dict(inspect_dicom(path)))
    aggregate = json.dumps(inspect_paths([path]))

    for forbidden in [
        "PRIVATE^PERSON",
        "PRIVATE-ID",
        "PRIVATE-ACC",
        "20200101",
        "PRIVATE-FILENAME",
        "PatientName",
        "PatientID",
        "AccessionNumber",
        "StudyDate",
        "PixelData",
        "PRIVATE-CREATOR-CONTENT",
    ]:
        assert forbidden not in payload
        assert forbidden not in aggregate
    assert '"identity_fields_serialized": false' in aggregate
    assert '"pixel_values_loaded": false' in aggregate
