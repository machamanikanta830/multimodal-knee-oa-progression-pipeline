"""Privacy-safe DICOM metadata inspection for the OAI imaging pilot.

The command-line interface reports aggregate technical metadata only. It never serializes input
paths, patient identity fields, accession values, study dates, or pixel values.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID

SAFE_DICOM_KEYWORDS = [
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PhotometricInterpretation",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "NumberOfFrames",
    "Modality",
    "ViewPosition",
    "Laterality",
    "ImageLaterality",
    "PatientOrientation",
    "ImageOrientationPatient",
    "BodyPartExamined",
    "SamplesPerPixel",
    "PixelRepresentation",
    "PresentationLUTShape",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "Manufacturer",
    "ManufacturerModelName",
    "BurnedInAnnotation",
]


class DicomInspectionError(ValueError):
    """Raised when a file cannot be safely parsed as DICOM."""


@dataclass(frozen=True, slots=True)
class DicomMetadata:
    """Allow-listed technical metadata with no identity, date, path, or pixel fields."""

    rows: int | None
    columns: int | None
    bits_allocated: int | None
    bits_stored: int | None
    photometric_interpretation: str | None
    pixel_spacing: tuple[float, ...] | None
    imager_pixel_spacing: tuple[float, ...] | None
    transfer_syntax_uid: str | None
    transfer_syntax_name: str | None
    compressed: bool | None
    number_of_frames: int
    number_of_frames_tag_present: bool
    multi_frame: bool
    modality: str | None
    view_position: str | None
    laterality: str | None
    image_laterality: str | None
    patient_orientation: tuple[str, ...] | None
    image_orientation_patient: tuple[float, ...] | None
    body_part_examined: str | None
    samples_per_pixel: int | None
    pixel_representation: int | None
    presentation_lut_shape: str | None
    rescale_slope: float | None
    rescale_intercept: float | None
    window_center: tuple[float, ...] | None
    window_width: tuple[float, ...] | None
    manufacturer: str | None
    manufacturer_model_name: str | None
    burned_in_annotation: str | None
    file_size_bytes: int


def _optional_int(dataset: pydicom.dataset.Dataset, keyword: str) -> int | None:
    value = dataset.get(keyword)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_text(dataset: pydicom.dataset.Dataset, keyword: str) -> str | None:
    value = dataset.get(keyword)
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _optional_float(dataset: pydicom.dataset.Dataset, keyword: str) -> float | None:
    value = dataset.get(keyword)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float_tuple(value: Any) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    values = value if isinstance(value, Iterable) and not isinstance(value, str) else [value]
    try:
        return tuple(float(item) for item in values)
    except (TypeError, ValueError, OverflowError):
        return None


def _text_tuple(value: Any) -> tuple[str, ...] | None:
    if value in (None, ""):
        return None
    values = value if isinstance(value, Iterable) and not isinstance(value, str) else [value]
    result = tuple(str(item).strip() for item in values if str(item).strip())
    return result or None


def _transfer_syntax(
    dataset: pydicom.dataset.Dataset,
) -> tuple[str | None, str | None, bool | None]:
    file_meta = getattr(dataset, "file_meta", None)
    value = file_meta.get("TransferSyntaxUID") if file_meta is not None else None
    if value in (None, ""):
        return None, None, None
    uid_text = str(value)
    try:
        uid = UID(uid_text)
        return uid_text, uid.name, bool(uid.is_compressed)
    except (TypeError, ValueError):
        return uid_text, None, None


def inspect_dicom(path: str | Path) -> DicomMetadata:
    """Read allow-listed technical tags without loading pixel data.

    Error messages intentionally omit the input path so callers can serialize them safely.
    """

    source = Path(path)
    try:
        size = source.stat().st_size
        dataset = pydicom.dcmread(
            source,
            stop_before_pixels=True,
            force=False,
            specific_tags=SAFE_DICOM_KEYWORDS,
        )
    except (OSError, InvalidDicomError, ValueError) as error:
        raise DicomInspectionError("Input could not be parsed as a valid DICOM file") from error

    transfer_uid, transfer_name, compressed = _transfer_syntax(dataset)
    frames_tag_present = dataset.get("NumberOfFrames") not in (None, "")
    frames = _optional_int(dataset, "NumberOfFrames") if frames_tag_present else 1
    if frames is None or frames < 1:
        frames = 1
    return DicomMetadata(
        rows=_optional_int(dataset, "Rows"),
        columns=_optional_int(dataset, "Columns"),
        bits_allocated=_optional_int(dataset, "BitsAllocated"),
        bits_stored=_optional_int(dataset, "BitsStored"),
        photometric_interpretation=_optional_text(dataset, "PhotometricInterpretation"),
        pixel_spacing=_float_tuple(dataset.get("PixelSpacing")),
        imager_pixel_spacing=_float_tuple(dataset.get("ImagerPixelSpacing")),
        transfer_syntax_uid=transfer_uid,
        transfer_syntax_name=transfer_name,
        compressed=compressed,
        number_of_frames=frames,
        number_of_frames_tag_present=frames_tag_present,
        multi_frame=frames > 1,
        modality=_optional_text(dataset, "Modality"),
        view_position=_optional_text(dataset, "ViewPosition"),
        laterality=_optional_text(dataset, "Laterality"),
        image_laterality=_optional_text(dataset, "ImageLaterality"),
        patient_orientation=_text_tuple(dataset.get("PatientOrientation")),
        image_orientation_patient=_float_tuple(dataset.get("ImageOrientationPatient")),
        body_part_examined=_optional_text(dataset, "BodyPartExamined"),
        samples_per_pixel=_optional_int(dataset, "SamplesPerPixel"),
        pixel_representation=_optional_int(dataset, "PixelRepresentation"),
        presentation_lut_shape=_optional_text(dataset, "PresentationLUTShape"),
        rescale_slope=_optional_float(dataset, "RescaleSlope"),
        rescale_intercept=_optional_float(dataset, "RescaleIntercept"),
        window_center=_float_tuple(dataset.get("WindowCenter")),
        window_width=_float_tuple(dataset.get("WindowWidth")),
        manufacturer=_optional_text(dataset, "Manufacturer"),
        manufacturer_model_name=_optional_text(dataset, "ManufacturerModelName"),
        burned_in_annotation=_optional_text(dataset, "BurnedInAnnotation"),
        file_size_bytes=size,
    )


def metadata_as_dict(metadata: DicomMetadata) -> dict[str, Any]:
    """Serialize only fields defined by the privacy-safe dataclass."""

    return asdict(metadata)


def _display(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, tuple):
        return "\\".join(str(item) for item in value)
    return str(value)


def summarize_metadata(
    records: Sequence[DicomMetadata], *, invalid_files: int = 0
) -> dict[str, Any]:
    """Aggregate safe metadata records without returning file-level entries."""

    sizes = [record.file_size_bytes for record in records]

    def distribution(attribute: str) -> dict[str, int]:
        return dict(
            sorted(Counter(_display(getattr(record, attribute)) for record in records).items())
        )

    size_summary: dict[str, int | float | None]
    if sizes:
        size_summary = {
            "total_bytes": sum(sizes),
            "mean_bytes": round(statistics.fmean(sizes), 3),
            "median_bytes": round(statistics.median(sizes), 3),
            "maximum_bytes": max(sizes),
        }
    else:
        size_summary = {
            "total_bytes": 0,
            "mean_bytes": None,
            "median_bytes": None,
            "maximum_bytes": None,
        }
    dimensions = Counter(
        f"{record.rows}x{record.columns}"
        if record.rows is not None and record.columns is not None
        else "<missing>"
        for record in records
    )
    return {
        "files_considered": len(records) + invalid_files,
        "valid_dicoms": len(records),
        "invalid_or_non_dicom_files": invalid_files,
        "multi_frame_dicoms": sum(record.multi_frame for record in records),
        "dimensions": dict(sorted(dimensions.items())),
        "rows": distribution("rows"),
        "columns": distribution("columns"),
        "bits_allocated": distribution("bits_allocated"),
        "bits_stored": distribution("bits_stored"),
        "photometric_interpretation": distribution("photometric_interpretation"),
        "pixel_spacing": distribution("pixel_spacing"),
        "imager_pixel_spacing": distribution("imager_pixel_spacing"),
        "transfer_syntax_uid": distribution("transfer_syntax_uid"),
        "transfer_syntax_name": distribution("transfer_syntax_name"),
        "compressed": distribution("compressed"),
        "number_of_frames": distribution("number_of_frames"),
        "modality": distribution("modality"),
        "view_position": distribution("view_position"),
        "laterality": distribution("laterality"),
        "image_laterality": distribution("image_laterality"),
        "patient_orientation": distribution("patient_orientation"),
        "image_orientation_patient": distribution("image_orientation_patient"),
        "body_part_examined": distribution("body_part_examined"),
        "pixel_representation": distribution("pixel_representation"),
        "rescale_slope": distribution("rescale_slope"),
        "rescale_intercept": distribution("rescale_intercept"),
        "window_center": distribution("window_center"),
        "window_width": distribution("window_width"),
        "manufacturer": distribution("manufacturer"),
        "manufacturer_model_name": distribution("manufacturer_model_name"),
        "burned_in_annotation": distribution("burned_in_annotation"),
        "file_sizes": size_summary,
        "privacy": {
            "file_paths_serialized": False,
            "identity_fields_serialized": False,
            "dates_serialized": False,
            "pixel_values_loaded": False,
        },
    }


def inspect_paths(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Inspect files and return only aggregate metadata and an invalid-file count."""

    records: list[DicomMetadata] = []
    invalid = 0
    for path in paths:
        try:
            records.append(inspect_dicom(path))
        except DicomInspectionError:
            invalid += 1
    return summarize_metadata(records, invalid_files=invalid)


def discover_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Resolve input files/directories without exposing their names in output."""

    discovered: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError("One or more DICOM inputs do not exist")
    return sorted(set(discovered))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect DICOM files and print aggregate privacy-safe technical metadata."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="DICOM file(s) or directories")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = inspect_paths(discover_files(args.inputs))
    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
