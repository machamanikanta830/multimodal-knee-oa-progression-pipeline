"""Select the independent 128-acquisition imaging-preprocessing validation sample.

Only baseline acquisition metadata and baseline KL are read. The command creates local ignored
NDA preparation artifacts and checks for already-present archives; it never downloads files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from imaging.pilot_qc import (
    DEFAULT_ANALYSIS_COHORT,
    DEFAULT_IMAGE_MANIFEST,
    DEFAULT_PILOT_MANIFEST,
    QC_PROBLEM_COLUMNS,
    build_acquisition_candidates,
)

DEFAULT_IMAGE03 = Path("data/raw/oai/image03.txt")
DEFAULT_VALIDATION_MANIFEST = Path(
    "data/processed/manifests/v00_xray_validation128_manifest.parquet"
)
DEFAULT_VALIDATION_IMAGE03 = Path("data/processed/manifests/v00_xray_validation128_image03.txt")
DEFAULT_VALIDATION_GUIDS = Path("data/processed/manifests/v00_xray_validation128_guids.txt")
DEFAULT_VALIDATION_REFERENCES = Path(
    "data/processed/manifests/v00_xray_validation128_associated_files.txt"
)
DEFAULT_LOCAL_IMAGE_ROOT = Path("data/raw/oai_images")
VALIDATION_SIZE = 128
SELECTION_SEED = "oai-v00-xray-validation128-v1"
EXPECTED_DESCRIPTION = "Bilateral PA Fixed Flexion Knee"
EXPECTED_FORMAT = "DICOM"
EXPECTED_MODALITY = "X-Ray"
EXPECTED_SCAN_TYPE = "X-Ray"
EXPECTED_SCAN_OBJECT = "Live"
MIN_REPORTABLE_GROUP = 5

IMAGE03_COLUMNS = [
    "accession_number",
    "subjectkey",
    "src_subject_id",
    "image_file",
    "image_description",
    "scan_type",
    "scan_object",
    "image_file_format",
    "image_modality",
    "scanner_manufacturer_pd",
    "scanner_type_pd",
    "scanner_software_versions_pd",
    "image_extent1",
    "image_extent2",
    "image_unit1",
    "image_unit2",
    "image_resolution1",
    "image_resolution2",
    "study",
    "visit",
]

DIVERSITY_WEIGHTS = {
    "scanner_manufacturer": 4.0,
    "scanner_model": 4.0,
    "scanner_software": 1.0,
    "dimension_family": 2.0,
    "spacing_family": 4.0,
    "image_release_study": 3.0,
    "xray_accept_qc": 3.0,
    "qc_problem_signature": 3.0,
}


class ValidationSelectionError(ValueError):
    """Raised when an independent sample cannot be prepared without ambiguity."""


def _stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{value}".encode()).hexdigest()


def _category(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna("<missing>").replace("", "<missing>")


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


def _atomic_bytes_write(chunks: list[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text_write(lines: list[str], target: Path) -> None:
    _atomic_bytes_write([f"{line}\n".encode() for line in lines], target)


def read_image03_metadata(path: str | Path) -> pd.DataFrame:
    """Read only documented fields needed for baseline image selection and NDA preparation."""

    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=[1],
        usecols=IMAGE03_COLUMNS,
        dtype="string",
        low_memory=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    frame["accession_number"] = _category(frame["accession_number"])
    return frame


def build_validation_candidates(
    analysis_cohort_path: str | Path,
    image_manifest_path: str | Path,
    image03_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one row per eligible V00 acquisition using baseline-only fields."""

    acquisitions, linked_knees = build_acquisition_candidates(
        analysis_cohort_path, image_manifest_path
    )
    image03 = read_image03_metadata(image03_path)
    accessions = set(acquisitions["accession_number"].astype(str))
    image03 = image03.loc[image03["accession_number"].isin(accessions)].copy()
    if image03["accession_number"].duplicated().any() or len(image03) != len(acquisitions):
        raise ValidationSelectionError(
            "Baseline accessions do not map one-to-one to image03 metadata"
        )
    rename = {
        "subjectkey": "nda_guid",
        "src_subject_id": "image03_source_participant_id",
        "image_file": "image03_image_file",
        "image_description": "image03_description",
        "scan_type": "image03_scan_type",
        "scan_object": "image03_scan_object",
        "image_file_format": "image03_file_format",
        "image_modality": "image03_modality",
        "scanner_manufacturer_pd": "scanner_manufacturer",
        "scanner_type_pd": "scanner_model",
        "scanner_software_versions_pd": "scanner_software",
        "image_extent1": "recorded_extent_1",
        "image_extent2": "recorded_extent_2",
        "image_unit1": "recorded_unit_1",
        "image_unit2": "recorded_unit_2",
        "image_resolution1": "recorded_resolution_1",
        "image_resolution2": "recorded_resolution_2",
        "study": "image03_study",
        "visit": "image03_visit",
    }
    candidates = acquisitions.merge(
        image03.rename(columns=rename),
        on="accession_number",
        how="left",
        validate="one_to_one",
    )
    if candidates["nda_guid"].isna().any():
        raise ValidationSelectionError("A validation candidate is missing its NDA GUID")
    if not candidates["participant_id"].astype(str).eq(candidates["nda_guid"]).all():
        raise ValidationSelectionError("Participant linkage disagrees with image03 NDA GUID")
    if not candidates["image_file"].astype(str).eq(candidates["image03_image_file"]).all():
        raise ValidationSelectionError("Associated-file linkage disagrees with image03")
    expected_values = {
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
    }
    for column, expected in expected_values.items():
        if not candidates[column].astype(str).eq(expected).all():
            raise ValidationSelectionError(f"Candidate {column} is not uniformly {expected}")
    if not candidates["xrmeta_side_code"].astype(str).eq("3").all():
        raise ValidationSelectionError("Candidate X-ray metadata are not uniformly bilateral")

    candidates["baseline_visit"] = "V00"
    candidates["read_project"] = "15"
    candidates["bilateral_acquisition"] = True
    candidates["validation_kl_stratum"] = candidates["pilot_kl_stratum"].astype(int)
    candidates["dimension_family"] = (
        _category(candidates["recorded_extent_1"])
        + "x"
        + _category(candidates["recorded_extent_2"])
    )
    candidates["spacing_family"] = (
        _category(candidates["recorded_resolution_1"])
        + "x"
        + _category(candidates["recorded_resolution_2"])
        + " "
        + _category(candidates["recorded_unit_1"])
        + "/"
        + _category(candidates["recorded_unit_2"])
    )
    candidates["qc_problem_signature"] = candidates[QC_PROBLEM_COLUMNS].apply(
        lambda row: (
            "+".join(
                column.removeprefix("xray_").removesuffix("_problem")
                for column, value in row.items()
                if pd.notna(value) and str(value).strip()
            )
            or "none"
        ),
        axis=1,
    )
    for column in DIVERSITY_WEIGHTS:
        candidates[column] = _category(candidates[column])
    return candidates, linked_knees


def select_validation_acquisitions(
    candidates: pd.DataFrame,
    development_accessions: set[str],
    *,
    sample_size: int = VALIDATION_SIZE,
) -> pd.DataFrame:
    """Select a balanced, deterministic stress sample with greedy metadata diversity."""

    required = {
        "accession_number",
        "validation_kl_stratum",
        "eligible_knee_count",
        *DIVERSITY_WEIGHTS,
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValidationSelectionError(f"Candidate fields are missing: {sorted(missing)}")
    if candidates["accession_number"].duplicated().any():
        raise ValidationSelectionError("Validation candidates contain duplicate accessions")
    if sample_size % 8:
        raise ValidationSelectionError("Validation size must be divisible by eight core strata")
    pool = (
        candidates.loc[~candidates["accession_number"].astype(str).isin(development_accessions)]
        .copy()
        .reset_index(drop=True)
    )
    per_stratum = sample_size // 8
    strata = [(grade, knees) for grade in range(4) for knees in (1, 2)]
    for grade, knees in strata:
        count = int(
            (pool["validation_kl_stratum"].eq(grade) & pool["eligible_knee_count"].eq(knees)).sum()
        )
        if count < per_stratum:
            raise ValidationSelectionError(
                f"Insufficient independent candidates in KL {grade}, knee-count {knees}"
            )

    frequencies = {column: Counter(pool[column].astype(str)) for column in DIVERSITY_WEIGHTS}
    covered = {column: set() for column in DIVERSITY_WEIGHTS}
    pool["selection_hash"] = pool["accession_number"].astype(str).map(_stable_hash)
    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    for _ in range(per_stratum):
        for grade, knees in strata:
            eligible = pool.loc[
                pool["validation_kl_stratum"].eq(grade)
                & pool["eligible_knee_count"].eq(knees)
                & ~pool.index.isin(selected_index_set)
            ]
            ranked: list[tuple[float, float, str, int]] = []
            for index, row in eligible.iterrows():
                novelty = sum(
                    weight
                    for column, weight in DIVERSITY_WEIGHTS.items()
                    if str(row[column]) not in covered[column]
                )
                rarity = sum(
                    weight / frequencies[column][str(row[column])]
                    for column, weight in DIVERSITY_WEIGHTS.items()
                )
                ranked.append((-novelty, -rarity, str(row["selection_hash"]), int(index)))
            _, _, _, chosen_index = min(ranked)
            selected_indices.append(chosen_index)
            selected_index_set.add(chosen_index)
            for column in DIVERSITY_WEIGHTS:
                covered[column].add(str(pool.at[chosen_index, column]))

    selected = pool.loc[selected_indices].copy()
    if len(selected) != sample_size or selected["accession_number"].nunique() != sample_size:
        raise ValidationSelectionError(
            "Validation selection is not exactly 128 unique acquisitions"
        )
    if set(selected["accession_number"].astype(str)).intersection(development_accessions):
        raise ValidationSelectionError("Validation selection overlaps the development pilot")
    selected["selection_seed"] = SELECTION_SEED
    selected["selection_outcomes_read"] = False
    selected["development_pilot_overlap"] = False
    return selected.drop(columns="selection_hash").sort_values(
        ["validation_kl_stratum", "eligible_knee_count", "accession_number"],
        kind="stable",
        ignore_index=True,
    )


def extract_image03_subset(
    source_path: str | Path,
    selected_accessions: set[str],
) -> list[bytes]:
    """Return exact source lines for the header, dictionary, and selected image03 records."""

    source = Path(source_path)
    with source.open("rb") as handle:
        header_line = handle.readline()
        dictionary_line = handle.readline()
        if not header_line or not dictionary_line:
            raise ValidationSelectionError("image03 is missing its header or dictionary row")
        header = next(csv.reader([header_line.decode("utf-8-sig")], delimiter="\t"))
        try:
            accession_index = header.index("accession_number")
        except ValueError as error:
            raise ValidationSelectionError("image03 lacks accession_number") from error
        matched: dict[str, bytes] = {}
        for line in handle:
            row = next(csv.reader([line.decode("utf-8")], delimiter="\t"))
            if len(row) != len(header):
                raise ValidationSelectionError("image03 contains an unsupported multiline row")
            accession = row[accession_index]
            if accession not in selected_accessions:
                continue
            if accession in matched:
                raise ValidationSelectionError(
                    "A selected accession occurs more than once in image03"
                )
            matched[accession] = line
    missing = selected_accessions.difference(matched)
    if missing:
        raise ValidationSelectionError(
            f"image03 is missing {len(missing)} selected acquisition records"
        )
    return [header_line, dictionary_line, *(matched[key] for key in sorted(matched))]


def _safe_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("<missing>").value_counts(dropna=False)
    result = {
        str(key): int(value) for key, value in counts.items() if value >= MIN_REPORTABLE_GROUP
    }
    suppressed = int(counts[counts < MIN_REPORTABLE_GROUP].sum())
    if suppressed:
        result["<groups smaller than 5 pooled>"] = suppressed
    return result


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "nonmissing": int(len(values)),
        "minimum": float(values.min()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
    }


def summarize_selection(
    selected: pd.DataFrame,
    linked_knees: pd.DataFrame,
    development_accessions: set[str],
    local_image_root: Path,
) -> dict[str, Any]:
    """Return aggregate-only selection, heterogeneity, and local-availability results."""

    accessions = set(selected["accession_number"].astype(str))
    selected_knees = linked_knees.loc[linked_knees["accession_number"].astype(str).isin(accessions)]
    references = selected["image_file"].astype(str)
    expected_names = {PurePosixPath(reference).name for reference in references}
    local_archives = list(local_image_root.rglob("*.tar.gz")) if local_image_root.exists() else []
    matched_archives = [path for path in local_archives if path.name in expected_names]
    partials = list(local_image_root.rglob("*.partial")) if local_image_root.exists() else []
    return {
        "validation_acquisitions": len(selected),
        "validation_participants": int(selected["participant_id"].nunique()),
        "eligible_knees_represented": len(selected_knees),
        "development_pilot_overlap": len(accessions.intersection(development_accessions)),
        "unique_associated_file_references": int(references.nunique()),
        "duplicate_associated_file_references": int(references.duplicated().sum()),
        "unique_nda_guids": int(selected["nda_guid"].nunique()),
        "baseline_visit": _safe_counts(selected["baseline_visit"]),
        "bilateral_acquisitions": int(selected["bilateral_acquisition"].sum()),
        "acquisition_kl_stratum": _safe_counts(selected["validation_kl_stratum"]),
        "eligible_knees_per_acquisition": _safe_counts(selected["eligible_knee_count"]),
        "eligible_knee_baseline_kl": _safe_counts(selected_knees["baseline_kl"]),
        "heterogeneity": {
            "manufacturer": _safe_counts(selected["scanner_manufacturer"]),
            "scanner_model": _safe_counts(selected["scanner_model"]),
            "recorded_spacing": _safe_counts(selected["spacing_family"]),
            "recorded_extent_families": int(selected["dimension_family"].nunique()),
            "recorded_extent_1": _numeric_summary(selected["recorded_extent_1"]),
            "recorded_extent_2": _numeric_summary(selected["recorded_extent_2"]),
            "recorded_resolution_1": _numeric_summary(selected["recorded_resolution_1"]),
            "recorded_resolution_2": _numeric_summary(selected["recorded_resolution_2"]),
            "image_release": _safe_counts(selected["image_release_study"]),
            "xray_accept_qc": _safe_counts(selected["xray_accept_qc"]),
            "qc_problem_signatures": _safe_counts(selected["qc_problem_signature"]),
            "qc_problem_flags": {
                column: int(selected[column].notna().sum()) for column in QC_PROBLEM_COLUMNS
            },
        },
        "selection": {
            "seed": SELECTION_SEED,
            "outcomes_read": False,
            "rule": (
                "sixteen acquisitions per maximum-baseline-KL (0-3) by eligible-knee-count "
                "(one/two) stratum; within fixed round-robin quotas, greedily maximize weighted "
                "novelty and inverse-frequency coverage of baseline scanner, geometry, release, "
                "and QC metadata with deterministic hash tie-breaking"
            ),
        },
        "local_archive_status": {
            "expected": len(expected_names),
            "present": len(matched_archives),
            "missing": len(expected_names) - len({path.name for path in matched_archives}),
            "duplicate_local_matches": len(matched_archives)
            - len({path.name for path in matched_archives}),
            "partial_files_under_local_image_root": len(partials),
            "ready_for_preprocessing": len({path.name for path in matched_archives})
            == len(expected_names),
        },
        "nda_package_instructions": {
            "collection": "Osteoarthritis Initiative (OAI)",
            "structure": "Image / image03",
            "guid_count": int(selected["nda_guid"].nunique()),
            "include_associated_data_files": True,
            "browser_automation_used": False,
        },
    }


def prepare_validation_sample(
    *,
    analysis_cohort_path: str | Path = DEFAULT_ANALYSIS_COHORT,
    image_manifest_path: str | Path = DEFAULT_IMAGE_MANIFEST,
    development_manifest_path: str | Path = DEFAULT_PILOT_MANIFEST,
    image03_path: str | Path = DEFAULT_IMAGE03,
    output_manifest_path: str | Path = DEFAULT_VALIDATION_MANIFEST,
    output_image03_path: str | Path = DEFAULT_VALIDATION_IMAGE03,
    output_guids_path: str | Path = DEFAULT_VALIDATION_GUIDS,
    output_references_path: str | Path = DEFAULT_VALIDATION_REFERENCES,
    local_image_root: str | Path = DEFAULT_LOCAL_IMAGE_ROOT,
) -> dict[str, Any]:
    """Prepare independent-sample artifacts and stop short of any download."""

    development = pd.read_parquet(development_manifest_path, columns=["accession_number"])
    development_accessions = set(development["accession_number"].astype(str))
    if len(development_accessions) != 32:
        raise ValidationSelectionError("Development pilot is not exactly 32 unique acquisitions")
    candidates, linked_knees = build_validation_candidates(
        analysis_cohort_path, image_manifest_path, image03_path
    )
    selected = select_validation_acquisitions(candidates, development_accessions)
    selected_accessions = set(selected["accession_number"].astype(str))
    image03_chunks = extract_image03_subset(image03_path, selected_accessions)
    guids = sorted(selected["nda_guid"].astype(str))
    references = sorted(selected["image_file"].astype(str))
    if len(guids) != VALIDATION_SIZE or len(set(guids)) != VALIDATION_SIZE:
        raise ValidationSelectionError("Validation sample does not contain 128 unique NDA GUIDs")
    if len(references) != VALIDATION_SIZE or len(set(references)) != VALIDATION_SIZE:
        raise ValidationSelectionError("Validation associated-file references are not unique")
    if any(not reference.endswith(".tar.gz") for reference in references):
        raise ValidationSelectionError("A validation associated-file reference is not .tar.gz")

    _atomic_parquet_write(selected, Path(output_manifest_path))
    _atomic_bytes_write(image03_chunks, Path(output_image03_path))
    _atomic_text_write(guids, Path(output_guids_path))
    _atomic_text_write(references, Path(output_references_path))
    return summarize_selection(
        selected, linked_knees, development_accessions, Path(local_image_root)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the independent 128-acquisition validation sample and prepare NDA artifacts. "
            "This command never downloads images or reads future outcomes."
        )
    )
    parser.add_argument("--analysis-cohort", type=Path, default=DEFAULT_ANALYSIS_COHORT)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--development-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--image03", type=Path, default=DEFAULT_IMAGE03)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--output-image03", type=Path, default=DEFAULT_VALIDATION_IMAGE03)
    parser.add_argument("--output-guids", type=Path, default=DEFAULT_VALIDATION_GUIDS)
    parser.add_argument("--output-references", type=Path, default=DEFAULT_VALIDATION_REFERENCES)
    parser.add_argument("--local-image-root", type=Path, default=DEFAULT_LOCAL_IMAGE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_validation_sample(
        analysis_cohort_path=args.analysis_cohort,
        image_manifest_path=args.image_manifest,
        development_manifest_path=args.development_manifest,
        image03_path=args.image03,
        output_manifest_path=args.output_manifest,
        output_image03_path=args.output_image03,
        output_guids_path=args.output_guids,
        output_references_path=args.output_references,
        local_image_root=args.local_image_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
