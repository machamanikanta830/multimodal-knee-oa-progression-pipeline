"""Prepare the local-only V00 X-ray pilot and optionally summarize downloaded DICOMs.

Selection reads baseline KL and image/linkage metadata only. It deliberately does not read any
outcome label, follow-up value, or post-baseline event field from the analysis cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from imaging.dicom_inspect import discover_files, inspect_paths

DEFAULT_ANALYSIS_COHORT = Path("data/processed/cohorts/analysis_cohort_v1.parquet")
DEFAULT_IMAGE_MANIFEST = Path("data/processed/manifests/v00_xray_manifest.parquet")
DEFAULT_PACKAGE_METADATA = Path("data/raw/oai/package_file_metadata_1248557.txt.gz")
DEFAULT_PILOT_MANIFEST = Path("data/processed/manifests/v00_xray_pilot_manifest.parquet")
DEFAULT_RETRIEVAL_REFERENCES = Path("data/processed/manifests/v00_xray_pilot_associated_files.txt")
DEFAULT_PILOT_SIZE = 32
SELECTION_SEED = "oai-v00-xray-pilot-v1"
FULL_ACQUISITION_COUNT = 3621

ANALYSIS_INPUT_COLUMNS = [
    "participant_id",
    "knee_side_code",
    "baseline_kl",
]
MANIFEST_INPUT_COLUMNS = [
    "participant_id",
    "source_participant_id",
    "knee_side_code",
    "baseline_visit",
    "read_project",
    "baseline_barcode",
    "accession_number",
    "image_file",
    "image_description",
    "image_file_format",
    "image_modality",
    "image_release_study",
    "xrmeta_side_code",
    "xray_completed",
    "xray_accept_qc",
    "xray_alignment_problem",
    "xray_centering_problem",
    "xray_incomplete_depiction",
    "xray_exam_type",
    "xray_positioning_problem",
    "bilateral_acquisition",
    "knees_sharing_accession",
]
QC_PROBLEM_COLUMNS = [
    "xray_alignment_problem",
    "xray_centering_problem",
    "xray_incomplete_depiction",
    "xray_positioning_problem",
]


class PilotSelectionError(ValueError):
    """Raised when the pilot cannot be selected without ambiguity or design drift."""


def _stable_hash(value: str, seed: str = SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _atomic_parquet_write(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=".parquet", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(lines: list[str], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", dir=target.parent, text=True)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(f"{line}\n" for line in lines)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _single_value(series: pd.Series, label: str) -> Any:
    values = series.dropna().unique()
    if len(values) == 0:
        return pd.NA
    if len(values) != 1:
        raise PilotSelectionError(f"An accession does not have exactly one {label}")
    return values[0]


def build_acquisition_candidates(
    analysis_cohort_path: str | Path,
    image_manifest_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return accession candidates and their knee-level baseline-only linkage."""

    analysis = pd.read_parquet(analysis_cohort_path, columns=ANALYSIS_INPUT_COLUMNS)
    manifest = pd.read_parquet(image_manifest_path, columns=MANIFEST_INPUT_COLUMNS)
    key = ["participant_id", "knee_side_code"]
    if analysis.duplicated(key).any() or manifest.duplicated(key).any():
        raise PilotSelectionError("Analysis cohort or image manifest has duplicate knee keys")
    linked = manifest.merge(analysis, on=key, how="inner", validate="one_to_one")
    if len(linked) != len(analysis) or len(linked) != len(manifest):
        raise PilotSelectionError("Analysis cohort and image manifest no longer match one-to-one")
    if not linked["baseline_visit"].eq("V00").all():
        raise PilotSelectionError("Pilot candidates include a non-V00 image")
    if not linked["read_project"].eq("15").all():
        raise PilotSelectionError("Pilot candidates include a non-15 read project")
    if not linked["baseline_kl"].isin([0, 1, 2, 3]).all():
        raise PilotSelectionError("Pilot candidates include baseline KL outside 0-3")
    if not linked["bilateral_acquisition"].all():
        raise PilotSelectionError("Pilot candidates include a non-bilateral acquisition")

    grouped = linked.groupby("accession_number", sort=False, dropna=False)
    if grouped["participant_id"].nunique().gt(1).any():
        raise PilotSelectionError("An accession maps to multiple participants")
    if grouped.size().gt(2).any():
        raise PilotSelectionError("An accession maps to more than two eligible knees")

    acquisition_rows: list[dict[str, Any]] = []
    copied_columns = [
        "participant_id",
        "source_participant_id",
        "baseline_barcode",
        "image_file",
        "image_description",
        "image_file_format",
        "image_modality",
        "image_release_study",
        "xrmeta_side_code",
        "xray_completed",
        "xray_accept_qc",
        *QC_PROBLEM_COLUMNS,
        "xray_exam_type",
    ]
    for accession, group in grouped:
        grades = sorted(int(value) for value in group["baseline_kl"])
        row = {
            "accession_number": accession,
            "eligible_knee_count": len(group),
            "eligible_knee_sides": ",".join(sorted(group["knee_side_code"])),
            "baseline_kl_grades": ",".join(str(value) for value in grades),
            "pilot_kl_stratum": max(grades),
        }
        for column in copied_columns:
            row[column] = _single_value(group[column], column)
        acquisition_rows.append(row)
    acquisitions = pd.DataFrame(acquisition_rows)
    acquisitions["qc_problem_count"] = acquisitions[QC_PROBLEM_COLUMNS].notna().sum(axis=1)
    acquisitions["selection_hash"] = acquisitions["accession_number"].map(
        lambda value: _stable_hash(str(value))
    )
    return acquisitions, linked


def _requirement_mask(frame: pd.DataFrame, column: str, value: str | None) -> pd.Series:
    return frame[column].notna() if value is None else frame[column].eq(value)


def _add_qc_challenge_coverage(
    selected: pd.DataFrame,
    acquisitions: pd.DataFrame,
) -> pd.DataFrame:
    """Make minimal same-stratum substitutions for documented QC-category coverage."""

    requirements: list[tuple[str, str | None]] = [
        ("xray_accept_qc", str(value))
        for value in sorted(acquisitions["xray_accept_qc"].dropna().unique())
    ]
    requirements.extend((column, None) for column in QC_PROBLEM_COLUMNS)
    enforced: list[tuple[str, str | None]] = []
    result = selected.copy()
    for column, value in requirements:
        candidates_exist = _requirement_mask(acquisitions, column, value).any()
        if not candidates_exist:
            continue
        if _requirement_mask(result, column, value).any():
            enforced.append((column, value))
            continue
        candidates = acquisitions.loc[
            _requirement_mask(acquisitions, column, value)
            & ~acquisitions["accession_number"].isin(result["accession_number"])
        ].sort_values("selection_hash", kind="stable")
        replaced = False
        for _, candidate in candidates.iterrows():
            same_stratum = result.loc[
                result["pilot_kl_stratum"].eq(candidate["pilot_kl_stratum"])
                & result["eligible_knee_count"].eq(candidate["eligible_knee_count"])
            ].sort_values("selection_hash", ascending=False, kind="stable")
            for victim_index in same_stratum.index:
                trial = pd.concat(
                    [result.drop(index=victim_index), candidate.to_frame().T],
                    ignore_index=True,
                )
                if all(_requirement_mask(trial, key, item).any() for key, item in enforced):
                    result = trial
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            raise PilotSelectionError(f"Could not preserve pilot strata while covering {column}")
        enforced.append((column, value))
    return result


def select_pilot_accessions(
    acquisitions: pd.DataFrame,
    *,
    pilot_size: int = DEFAULT_PILOT_SIZE,
) -> pd.DataFrame:
    """Select a deterministic, outcome-blind, structurally diverse accession pilot."""

    strata = [(grade, knees) for grade in range(4) for knees in (1, 2)]
    if pilot_size % len(strata):
        raise PilotSelectionError("Pilot size must be divisible by the eight KL/knee-count strata")
    per_stratum = pilot_size // len(strata)
    selected: list[pd.DataFrame] = []
    for grade, knees in strata:
        candidates = acquisitions.loc[
            acquisitions["pilot_kl_stratum"].eq(grade)
            & acquisitions["eligible_knee_count"].eq(knees)
        ].copy()
        if len(candidates) < per_stratum:
            raise PilotSelectionError(
                f"Insufficient candidates in KL {grade}, {knees}-knee stratum"
            )
        candidates = candidates.sort_values("selection_hash", kind="stable")
        selected.append(candidates.head(per_stratum))
    pilot = pd.concat(selected, ignore_index=True)
    pilot = _add_qc_challenge_coverage(pilot, acquisitions)
    if len(pilot) != pilot_size or pilot["accession_number"].nunique() != pilot_size:
        raise PilotSelectionError(
            "Pilot does not contain the requested number of unique accessions"
        )
    pilot["selection_seed"] = SELECTION_SEED
    pilot["selection_outcomes_read"] = False
    return pilot.drop(columns=["selection_hash"]).sort_values(
        ["pilot_kl_stratum", "eligible_knee_count", "accession_number"],
        kind="stable",
        ignore_index=True,
    )


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def summarize_pilot(pilot: pd.DataFrame, linked_knees: pd.DataFrame) -> dict[str, Any]:
    selected_knees = linked_knees.loc[
        linked_knees["accession_number"].isin(pilot["accession_number"])
    ]
    references = pilot["image_file"].astype("string")
    parsed = [urlparse(value) for value in references]
    return {
        "pilot_acquisitions": len(pilot),
        "pilot_participants": int(pilot["participant_id"].nunique()),
        "eligible_knees_represented": len(selected_knees),
        "acquisition_kl_stratum": _counts(pilot["pilot_kl_stratum"]),
        "eligible_knee_baseline_kl": _counts(selected_knees["baseline_kl"]),
        "eligible_knees_per_acquisition": _counts(pilot["eligible_knee_count"]),
        "xray_accept_qc": _counts(pilot["xray_accept_qc"]),
        "qc_problem_count": _counts(pilot["qc_problem_count"]),
        "qc_problem_flags": {
            column: int(pilot[column].notna().sum()) for column in QC_PROBLEM_COLUMNS
        },
        "image_release_study": _counts(pilot["image_release_study"]),
        "retrieval_references": {
            "unique_references": int(references.nunique()),
            "missing_references": int(references.isna().sum()),
            "scheme_counts": dict(Counter(item.scheme or "<relative>" for item in parsed)),
            "archive_suffix_counts": dict(
                Counter("".join(PurePosixPath(item.path).suffixes[-2:]) for item in parsed)
            ),
            "references_with_query_strings": sum(bool(item.query) for item in parsed),
        },
        "selection": {
            "outcomes_read": False,
            "seed": SELECTION_SEED,
            "rule": (
                "four acquisitions from each maximum-baseline-KL (0-3) by eligible-knee-count "
                "(one/two) stratum using a deterministic hash; make the minimum same-stratum "
                "substitutions needed to cover each observed acceptance code and problem-flag type"
            ),
        },
    }


def inspect_existing_package_retrieval(
    package_metadata_path: str | Path,
    pilot: pd.DataFrame,
) -> dict[str, Any]:
    """Check whether pilot image archives are entries in the existing NDA package metadata."""

    metadata = pd.read_csv(package_metadata_path, dtype="string")
    required = {"NDA_S3_URL", "DOWNLOAD_ALIAS"}
    if not required.issubset(metadata.columns):
        raise PilotSelectionError("NDA package metadata lacks expected download-reference columns")
    aliases = set(metadata["DOWNLOAD_ALIAS"].dropna())
    package_urls = set(metadata["NDA_S3_URL"].dropna())
    references = set(pilot["image_file"].dropna())
    basenames = {PurePosixPath(reference).name for reference in references}
    direct_matches = len(references.intersection(aliases | package_urls))
    basename_matches = len(basenames.intersection(aliases))
    archive_entries = int(metadata["DOWNLOAD_ALIAS"].str.endswith(".tar.gz", na=False).sum())
    downloadcmd_path = shutil.which("downloadcmd")
    return {
        "existing_package_entries": len(metadata),
        "existing_package_archive_entries": archive_entries,
        "pilot_reference_exact_matches": direct_matches,
        "pilot_archive_basename_matches": basename_matches,
        "downloadcmd_installed": downloadcmd_path is not None,
        "downloadcmd_help_inspected": False,
        "direct_retrieval_from_existing_package_supported": (
            direct_matches == len(references) and downloadcmd_path is not None
        ),
        "conclusion": (
            "The current package metadata does not include the pilot associated archives. "
            "Create an NDA package that includes the selected associated image files before "
            "attempting download with a locally installed and authenticated downloadcmd."
            if direct_matches != len(references)
            else "Pilot archive references are present in the package metadata."
        ),
    }


def estimate_storage_from_dicoms(
    dicom_summary: dict[str, Any],
    *,
    full_acquisition_count: int = FULL_ACQUISITION_COUNT,
) -> dict[str, Any]:
    """Estimate raw storage only when at least one pilot DICOM was actually inspected."""

    valid = int(dicom_summary["valid_dicoms"])
    mean_size = dicom_summary["file_sizes"]["mean_bytes"]
    if not valid or mean_size is None:
        return {
            "available": False,
            "reason": "No downloaded pilot DICOMs were available for an empirical estimate.",
            "raw_dicom_full_cohort_bytes": None,
            "extracted_panel_storage_bytes": None,
            "model_ready_cache_storage_bytes": None,
        }
    return {
        "available": True,
        "assumption": "one inspected DICOM file per unique bilateral acquisition",
        "raw_dicom_full_cohort_bytes": round(float(mean_size) * full_acquisition_count),
        "extracted_panel_storage_bytes": None,
        "model_ready_cache_storage_bytes": None,
        "panel_and_cache_reason": (
            "Panel extraction and cache encodings are not approved, so their storage is not estimated."
        ),
    }


def prepare_pilot(
    *,
    analysis_cohort_path: str | Path = DEFAULT_ANALYSIS_COHORT,
    image_manifest_path: str | Path = DEFAULT_IMAGE_MANIFEST,
    package_metadata_path: str | Path = DEFAULT_PACKAGE_METADATA,
    pilot_manifest_path: str | Path = DEFAULT_PILOT_MANIFEST,
    retrieval_references_path: str | Path = DEFAULT_RETRIEVAL_REFERENCES,
    pilot_size: int = DEFAULT_PILOT_SIZE,
    dicom_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Select and persist the ignored pilot plus aggregate-only preparation results."""

    acquisitions, linked_knees = build_acquisition_candidates(
        analysis_cohort_path, image_manifest_path
    )
    pilot = select_pilot_accessions(acquisitions, pilot_size=pilot_size)
    _atomic_parquet_write(pilot, Path(pilot_manifest_path))
    _atomic_text_write(sorted(pilot["image_file"].astype(str)), Path(retrieval_references_path))

    report = summarize_pilot(pilot, linked_knees)
    report["retrieval"] = inspect_existing_package_retrieval(package_metadata_path, pilot)
    if dicom_directory is None:
        dicom_summary = inspect_paths([])
        report["images_downloaded_by_this_command"] = False
    else:
        dicom_summary = inspect_paths(discover_files([dicom_directory]))
        report["images_downloaded_by_this_command"] = False
    report["dicom_qc"] = dicom_summary
    report["storage_estimate"] = estimate_storage_from_dicoms(dicom_summary)
    report["local_outputs"] = {
        "pilot_manifest": str(Path(pilot_manifest_path).resolve()),
        "retrieval_references": str(Path(retrieval_references_path).resolve()),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the deterministic baseline-only OAI X-ray pilot and print aggregate results. "
            "This command never downloads images."
        )
    )
    parser.add_argument("--analysis-cohort", type=Path, default=DEFAULT_ANALYSIS_COHORT)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--package-metadata", type=Path, default=DEFAULT_PACKAGE_METADATA)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--retrieval-references", type=Path, default=DEFAULT_RETRIEVAL_REFERENCES)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument(
        "--dicom-dir",
        type=Path,
        help="Optional existing local pilot DICOM directory to inspect; never downloads files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_pilot(
        analysis_cohort_path=args.analysis_cohort,
        image_manifest_path=args.image_manifest,
        package_metadata_path=args.package_metadata,
        pilot_manifest_path=args.pilot_manifest,
        retrieval_references_path=args.retrieval_references,
        pilot_size=args.pilot_size,
        dicom_directory=args.dicom_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
