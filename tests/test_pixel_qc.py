"""Synthetic-only tests for pixel decoding and midpoint geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from imaging.pixel_qc import (
    PixelInspectionError,
    inspect_pixels,
    normalize_for_display,
    pixel_metadata_as_dict,
    split_bilateral_midpoint,
    summarize_pixels,
)

pytestmark = pytest.mark.public_portable


def _write_pixels(path: Path, array: np.ndarray, *, frames: int = 1) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientID = "PRIVATE-PATIENT"
    dataset.AccessionNumber = "PRIVATE-ACC"
    dataset.StudyDate = "20200101"
    dataset.Rows = array.shape[-2]
    dataset.Columns = array.shape[-1]
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.SamplesPerPixel = 1
    dataset.PixelRepresentation = 0
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.RescaleSlope = "2"
    dataset.RescaleIntercept = "-5"
    if frames != 1:
        dataset.NumberOfFrames = frames
    dataset.PixelData = np.asarray(array, dtype="<u2").tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_monochrome1_is_inverted_for_consistent_display() -> None:
    pixels = np.array([[0, 1], [2, 3]], dtype=np.uint16)

    monochrome2 = normalize_for_display(
        pixels, "MONOCHROME2", lower_percentile=0, upper_percentile=100
    )
    monochrome1 = normalize_for_display(
        pixels, "MONOCHROME1", lower_percentile=0, upper_percentile=100
    )

    assert np.array_equal(monochrome1, 255 - monochrome2)
    assert monochrome1[0, 0] > monochrome1[-1, -1]


def test_midpoint_split_preserves_every_column() -> None:
    even = np.arange(24).reshape(3, 8)
    odd = np.arange(21).reshape(3, 7)

    even_left, even_right = split_bilateral_midpoint(even)
    odd_left, odd_right = split_bilateral_midpoint(odd)

    assert even_left.shape == (3, 4)
    assert even_right.shape == (3, 4)
    assert odd_left.shape == (3, 3)
    assert odd_right.shape == (3, 4)
    assert even_left.dtype == even.dtype
    assert even_right.dtype == even.dtype
    assert np.array_equal(np.concatenate([even_left, even_right], axis=1), even)
    assert np.array_equal(np.concatenate([odd_left, odd_right], axis=1), odd)


def test_single_frame_pixel_metadata_and_privacy(tmp_path: Path) -> None:
    source = np.arange(24, dtype=np.uint16).reshape(4, 6)
    path = _write_pixels(tmp_path / "PRIVATE-FILENAME.dcm", source)

    metadata, display = inspect_pixels(path)
    payload = json.dumps(pixel_metadata_as_dict(metadata))
    aggregate = json.dumps(summarize_pixels([metadata]))

    assert display.shape == source.shape
    assert metadata.source_min == 0
    assert metadata.source_max == 23
    assert metadata.rescaled_min == -5
    assert metadata.rescaled_max == 41
    assert metadata.left_half_columns == 3
    assert metadata.right_half_columns == 3
    for forbidden in ["PRIVATE-FILENAME", "PRIVATE-PATIENT", "PRIVATE-ACC", "20200101"]:
        assert forbidden not in payload
        assert forbidden not in aggregate
    assert '"pixel_arrays_serialized": false' in aggregate


def test_rejects_multiframe_pixels(tmp_path: Path) -> None:
    source = np.arange(48, dtype=np.uint16).reshape(2, 4, 6)
    path = _write_pixels(tmp_path / "multi.dcm", source, frames=2)

    with pytest.raises(PixelInspectionError, match="exactly one frame"):
        inspect_pixels(path)
