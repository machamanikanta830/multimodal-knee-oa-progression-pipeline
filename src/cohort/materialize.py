"""Materialize the approved local-only v1 analysis cohort and X-ray manifest.

The command reads the immutable OAI package, validates the reviewed Milestone 3 counts, and
writes participant-level Parquet files only below the Git-ignored ``data/processed`` tree. Its
terminal output is aggregate-only and is safe to use when updating tracked documentation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from cohort.builder import (
    BASELINE_VISIT,
    FOLLOWUP_VISIT,
    MISSING_SENTINELS,
    READ_PROJECT,
    CohortBuild,
    CohortIntegrityError,
    _check_unique,
    _read_oai_table,
    _side_value,
    build_cohort,
)

DEFAULT_STUDY_CONFIG = Path("configs/study_v1.yaml")
DEFAULT_FEATURE_CONFIG = Path("configs/features_v1.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed")
COHORT_RELATIVE_PATH = Path("cohorts/analysis_cohort_v1.parquet")
MANIFEST_RELATIVE_PATH = Path("manifests/v00_xray_manifest.parquet")

IDENTIFIER_COLUMNS = ("participant_id", "source_participant_id")
RAW_PARTICIPANT_COLUMN_MAP = {"ageyears": "baseline_age", "sex": "baseline_sex"}
NONNUMERIC_PRIMARY_FEATURES = frozenset({"sex"})
ALLOWED_TABULAR_PRIMARY_SOURCES = frozenset(
    {
        "oai_koos_womac01.txt",
        "oai_enrollee01.txt",
        "oai_oarisk01.txt",
        "oai_physfunct01.txt",
    }
)
FORBIDDEN_PREDICTOR_VARIABLES = frozenset(
    {
        "subjectkey",
        "src_subject_id",
        "barcode",
        "accession_number",
        "image_file",
        "readprj",
        "lkdate",
        "rkdate",
        "lkblrp",
        "rkblrp",
        "lktlpr",
        "rktlpr",
        "lkrpcf",
        "rkrpcf",
    }
)


class MaterializationError(ValueError):
    """Raised when configuration, linkage, or reference-count validation fails."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML using only the Python standard library."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MaterializationError(f"Configuration must be a mapping: {path}")
    return value


def _validate_configs(study: dict[str, Any], features: dict[str, Any]) -> None:
    expected_study_values = {
        "status": "human_approved",
        "unit_of_analysis": "participant-knee",
        "baseline_visit": BASELINE_VISIT,
        "followup_visit": FOLLOWUP_VISIT,
        "radiographic_read_project": READ_PROJECT,
        "primary_outcome": "composite_progression",
        "master_analysis_population": "composite_analysis_eligible",
        "radiographic_sensitivity_population": "radiographic_analysis_eligible",
    }
    contradictions = {
        key: (study.get(key), expected)
        for key, expected in expected_study_values.items()
        if study.get(key) != expected
    }
    if contradictions:
        raise MaterializationError(f"Study configuration contradicts approved v1: {contradictions}")
    if study.get("allowed_baseline_kl_grades") != [0, 1, 2, 3]:
        raise MaterializationError("Approved baseline KL grades must be exactly 0, 1, 2, and 3")
    if study.get("kl_progression_minimum_grade_increase") != 1:
        raise MaterializationError("Approved KL progression threshold must remain >= 1")
    if features.get("outcome_driven_selection_performed") is not False:
        raise MaterializationError("Feature configuration must prohibit outcome-driven selection")

    primary = features.get("primary_features")
    if not isinstance(primary, list) or not primary:
        raise MaterializationError("Feature configuration has no primary feature panel")
    output_columns: list[str] = []
    for feature in primary:
        if feature.get("baseline_only") is not True:
            raise MaterializationError(f"Non-baseline primary feature: {feature.get('id')}")
        if feature.get("kind") == "tabular":
            if feature.get("source_file") not in ALLOWED_TABULAR_PRIMARY_SOURCES:
                raise MaterializationError(
                    f"Unapproved primary tabular source for {feature.get('id')}: "
                    f"{feature.get('source_file')}"
                )
            sources = feature.get("source_variables", {})
            forbidden = set(sources.values()).intersection(FORBIDDEN_PREDICTOR_VARIABLES)
            if forbidden:
                raise MaterializationError(
                    f"Identity/linkage/outcome variable configured as predictor: {sorted(forbidden)}"
                )
        output = feature.get("output_column")
        if output is not None:
            output_columns.append(output)
    if len(output_columns) != len(set(output_columns)):
        raise MaterializationError("Primary feature output columns are not unique")


def _numeric_feature(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").replace(list(MISSING_SENTINELS), pd.NA)
    return pd.to_numeric(cleaned, errors="coerce")


def _feature_value(records: pd.DataFrame, feature: dict[str, Any]) -> pd.Series:
    sources = feature["source_variables"]
    if "participant" in sources:
        source = RAW_PARTICIPANT_COLUMN_MAP.get(sources["participant"], sources["participant"])
        value = records[source]
    elif "left" in sources and "right" in sources:
        value = _side_value(records, records["side"], sources["left"], sources["right"])
    elif "knee" in sources:
        source = "baseline_kl" if sources["knee"] == "xrkl" else sources["knee"]
        value = records[source]
    else:
        raise MaterializationError(f"Unsupported source mapping for feature {feature.get('id')}")
    if feature["id"] in NONNUMERIC_PRIMARY_FEATURES:
        return value.astype("string")
    return _numeric_feature(value)


def _build_analysis_table(
    build: CohortBuild,
    study: dict[str, Any],
    features: dict[str, Any],
) -> pd.DataFrame:
    source = build.records.loc[build.records["composite_analysis_eligible"]].copy()
    _check_unique(source, ["subjectkey", "side"], "composite analysis population")
    if not source["visit_baseline"].eq(BASELINE_VISIT).all():
        raise MaterializationError("Master cohort contains a non-V00 baseline record")
    if not source["readprj"].eq(READ_PROJECT).all():
        raise MaterializationError("Master cohort contains a non-15 radiographic read project")
    radiographic_source = source["radiographic_analysis_eligible"]
    if not source.loc[radiographic_source, "visit_v06"].eq(FOLLOWUP_VISIT).all():
        raise MaterializationError("Radiographic population contains a non-V06 follow-up record")
    if not source["baseline_kl"].isin([0, 1, 2, 3]).all():
        raise MaterializationError("Master cohort baseline KL is outside the approved 0-3 range")

    expected_radiographic = pd.Series(pd.NA, index=source.index, dtype="boolean")
    expected_radiographic.loc[radiographic_source] = (
        source.loc[radiographic_source, "v06_kl"] - source.loc[radiographic_source, "baseline_kl"]
    ).ge(1)
    radiographic_mismatch = source["radiographic_kl_progression"].isna().ne(
        expected_radiographic.isna()
    ) | source["radiographic_kl_progression"].fillna(False).ne(expected_radiographic.fillna(False))
    expected_composite = expected_radiographic.fillna(False) | source["replacement_before_v06"]
    composite_mismatch = source["composite_progression"].astype(bool).ne(expected_composite)
    if radiographic_mismatch.any() or composite_mismatch.any():
        raise MaterializationError("Outcome labels disagree with the approved cohort-builder rules")

    analysis = pd.DataFrame(index=source.index)
    analysis["participant_id"] = source["subjectkey"]
    analysis["source_participant_id"] = source["src_subject_id"]
    analysis["knee_side_code"] = source["side"]
    analysis["knee_side_label"] = source["side"].map({"1": "right", "2": "left"})
    analysis["baseline_visit"] = study["baseline_visit"]
    analysis["followup_visit"] = study["followup_visit"]
    analysis["read_project"] = study["radiographic_read_project"]
    analysis["baseline_barcode"] = source["barcode_baseline"]

    analysis["baseline_kl"] = source["baseline_kl"].astype("Int64")
    analysis["baseline_jsn_medial"] = source["baseline_jsn_medial"]
    analysis["baseline_jsn_lateral"] = source["baseline_jsn_lateral"]

    primary_feature_columns: list[str] = []
    domain_feature_columns: dict[str, list[str]] = {
        "pro": [],
        "clinical": [],
        "physical_function": [],
    }
    for feature in features["primary_features"]:
        if feature["kind"] != "tabular":
            continue
        output = feature["output_column"]
        analysis[output] = _feature_value(source, feature)
        primary_feature_columns.append(output)
        domain_feature_columns[feature["domain"]].append(output)

    for variable in features.get("local_descriptive_or_design_variables", []):
        analysis[variable["output_column"]] = source[variable["source_variable"]].astype("string")

    analysis["radiographic_analysis_eligible"] = source["radiographic_analysis_eligible"].astype(
        bool
    )
    analysis["composite_analysis_eligible"] = True
    analysis["jsn_analysis_eligible"] = source["jsn_analysis_eligible"].astype(bool)
    analysis["radiographic_kl_progression"] = source["radiographic_kl_progression"].astype(
        "boolean"
    )
    analysis["composite_progression"] = source["composite_progression"].astype("boolean")
    analysis["jsn_progression"] = source["jsn_progression"].astype("boolean")
    analysis["replacement_before_v06"] = source["replacement_before_v06"].astype(bool)

    analysis["imaging_domain_complete"] = source["imaging_linkage_available"].astype(bool)
    analysis["pro_domain_complete"] = analysis[domain_feature_columns["pro"]].notna().all(axis=1)
    analysis["clinical_domain_complete"] = (
        analysis[domain_feature_columns["clinical"]].notna().all(axis=1)
    )
    analysis["physical_function_domain_complete"] = (
        analysis[domain_feature_columns["physical_function"]].notna().all(axis=1)
    )
    domain_masks = [
        "imaging_domain_complete",
        "pro_domain_complete",
        "clinical_domain_complete",
        "physical_function_domain_complete",
    ]
    analysis["available_domain_count"] = analysis[domain_masks].sum(axis=1).astype("int8")
    analysis["all_four_domains_complete"] = analysis["available_domain_count"].eq(4)
    analysis["baseline_kl_default_predictor"] = False

    analysis = analysis.reset_index(drop=True)
    _check_unique(
        analysis,
        ["participant_id", "knee_side_code"],
        "materialized analysis cohort",
    )
    if not analysis["knee_side_code"].isin(["1", "2"]).all():
        raise MaterializationError("Analysis cohort contains an invalid knee-side code")
    if analysis[primary_feature_columns].empty:
        raise MaterializationError("No primary tabular features were materialized")
    return analysis


def _build_image_manifest(
    package_dir: str | Path,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    xrmeta_columns = [
        "subjectkey",
        "src_subject_id",
        "interview_date",
        "visit",
        "side",
        "completed",
        "accept",
        "align",
        "center",
        "depict",
        "examtype",
        "xrqc_position",
        "barcode",
    ]
    image_columns = [
        "subjectkey",
        "src_subject_id",
        "interview_date",
        "image_file",
        "image_description",
        "experiment_id",
        "scan_type",
        "scan_object",
        "image_file_format",
        "image_modality",
        "study",
        "experiment_description",
        "visit",
        "accession_number",
    ]
    xrmeta = _read_oai_table(package_dir, "oai_xrmeta01.txt", xrmeta_columns)
    image_index = _read_oai_table(package_dir, "image03.txt", image_columns)
    needed = set(analysis["baseline_barcode"].dropna())
    xrmeta = xrmeta.loc[xrmeta["barcode"].isin(needed)].copy()
    image_index = image_index.loc[image_index["accession_number"].isin(needed)].copy()
    _check_unique(xrmeta, ["barcode"], "baseline oai_xrmeta01.txt linkage")
    _check_unique(image_index, ["accession_number"], "baseline image03.txt linkage")

    manifest = (
        analysis[
            [
                "participant_id",
                "source_participant_id",
                "knee_side_code",
                "knee_side_label",
                "baseline_visit",
                "read_project",
                "baseline_barcode",
            ]
        ]
        .merge(
            xrmeta.rename(
                columns={
                    "subjectkey": "xrmeta_participant_id",
                    "src_subject_id": "xrmeta_source_participant_id",
                    "interview_date": "xray_acquisition_date",
                    "visit": "xrmeta_visit",
                    "side": "xrmeta_side_code",
                    "completed": "xray_completed",
                    "accept": "xray_accept_qc",
                    "align": "xray_alignment_problem",
                    "center": "xray_centering_problem",
                    "depict": "xray_incomplete_depiction",
                    "examtype": "xray_exam_type",
                    "xrqc_position": "xray_positioning_problem",
                    "barcode": "baseline_barcode",
                }
            ),
            on="baseline_barcode",
            how="left",
            validate="many_to_one",
        )
        .merge(
            image_index.rename(
                columns={
                    "subjectkey": "image_participant_id",
                    "src_subject_id": "image_source_participant_id",
                    "interview_date": "image_index_acquisition_date",
                    "visit": "image_index_visit",
                    "study": "image_release_study",
                }
            ),
            left_on="baseline_barcode",
            right_on="accession_number",
            how="left",
            validate="many_to_one",
        )
    )

    required_links = [
        "xrmeta_participant_id",
        "image_participant_id",
        "accession_number",
        "image_file",
    ]
    if manifest[required_links].isna().any().any():
        missing = manifest[required_links].isna().sum().loc[lambda values: values.gt(0)]
        raise MaterializationError(f"Baseline image linkage is incomplete: {missing.to_dict()}")

    mismatch_checks = {
        "cohort versus X-ray metadata participant": manifest["participant_id"].ne(
            manifest["xrmeta_participant_id"]
        ),
        "cohort versus image index participant": manifest["participant_id"].ne(
            manifest["image_participant_id"]
        ),
        "cohort versus X-ray metadata source participant": manifest["source_participant_id"].ne(
            manifest["xrmeta_source_participant_id"]
        ),
        "cohort versus image index source participant": manifest["source_participant_id"].ne(
            manifest["image_source_participant_id"]
        ),
    }
    disagreements = {name: int(mask.sum()) for name, mask in mismatch_checks.items() if mask.any()}
    if disagreements:
        raise MaterializationError(f"Image linkage identifier disagreement: {disagreements}")
    if not manifest["xrmeta_visit"].eq(BASELINE_VISIT).all():
        raise MaterializationError("Linked X-ray metadata contain a non-baseline visit")
    if not manifest["image_index_visit"].eq(BASELINE_VISIT).all():
        raise MaterializationError("Linked image index records contain a non-baseline visit")

    manifest["bilateral_acquisition"] = manifest["xrmeta_side_code"].eq("3")
    manifest["knees_sharing_accession"] = (
        manifest.groupby("accession_number")["knee_side_code"].transform("size").astype("int8")
    )
    if manifest["knees_sharing_accession"].gt(2).any():
        raise MaterializationError("A baseline acquisition maps to more than two cohort knees")
    cross_participant = manifest.groupby("accession_number")["participant_id"].nunique().gt(1)
    if cross_participant.any():
        raise MaterializationError("A baseline acquisition maps to multiple participants")

    manifest = manifest.drop(
        columns=[
            "xrmeta_participant_id",
            "xrmeta_source_participant_id",
            "image_participant_id",
            "image_source_participant_id",
            "xrmeta_visit",
        ]
    ).reset_index(drop=True)
    _check_unique(
        manifest,
        ["participant_id", "knee_side_code"],
        "materialized baseline image manifest",
    )
    return manifest


def _population_summary(frame: pd.DataFrame, mask: pd.Series, label_column: str) -> dict[str, Any]:
    population = frame.loc[mask]
    labels = population[label_column].astype(bool)
    contributors = population.groupby("participant_id").size()
    events = int(labels.sum())
    return {
        "knees": len(population),
        "participants": int(population["participant_id"].nunique()),
        "participants_with_one_knee": int(contributors.eq(1).sum()),
        "participants_with_two_knees": int(contributors.eq(2).sum()),
        "events": events,
        "non_events": len(population) - events,
        "event_rate_percent": round(100 * events / len(population), 3),
    }


def _missingness_summary(
    frame: pd.DataFrame,
    mask: pd.Series,
    columns: list[str],
) -> dict[str, dict[str, int | float]]:
    population = frame.loc[mask]
    output: dict[str, dict[str, int | float]] = {}
    for column in columns:
        missing = int(population[column].isna().sum())
        output[column] = {
            "missing_knees": missing,
            "nonmissing_knees": len(population) - missing,
            "missing_percent": round(100 * missing / len(population), 3),
        }
    return output


def _domain_summary(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    population = frame.loc[mask]
    domain_columns = [
        "imaging_domain_complete",
        "pro_domain_complete",
        "clinical_domain_complete",
        "physical_function_domain_complete",
    ]
    output: dict[str, Any] = {}
    for column in domain_columns:
        selected = population[column]
        output[column] = {
            "knees": int(selected.sum()),
            "participants_with_at_least_one_knee": int(
                population.loc[selected, "participant_id"].nunique()
            ),
            "percent_knees": round(100 * selected.mean(), 3),
        }
    available = population["available_domain_count"]

    def pattern(mask: pd.Series) -> dict[str, int]:
        return {
            "knees": int(mask.sum()),
            "participants_with_at_least_one_knee": int(
                population.loc[mask, "participant_id"].nunique()
            ),
        }

    output["missing_domain_patterns"] = {
        "all_four_complete": pattern(available.eq(4)),
        "missing_exactly_one_domain": pattern(available.eq(3)),
        "missing_multiple_domains": pattern(available.le(2)),
    }
    output["available_domain_count_distribution_knees"] = {
        str(count): int(available.eq(count).sum()) for count in range(5)
    }
    return output


def _image_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    accession_sizes = manifest.groupby("accession_number").size()

    def counts(column: str) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in manifest[column].value_counts(dropna=False).items()
        }

    return {
        "manifest_rows": len(manifest),
        "unique_participants": int(manifest["participant_id"].nunique()),
        "unique_accessions": int(manifest["accession_number"].nunique()),
        "unique_image_files": int(manifest["image_file"].nunique()),
        "missing_image_files": int(manifest["image_file"].isna().sum()),
        "accessions_linked_to_one_knee": int(accession_sizes.eq(1).sum()),
        "accessions_linked_to_two_knees": int(accession_sizes.eq(2).sum()),
        "bilateral_acquisition_knees": int(manifest["bilateral_acquisition"].sum()),
        "exam_type_by_knee": counts("xray_exam_type"),
        "xrmeta_side_code_by_knee": counts("xrmeta_side_code"),
        "xray_completed_by_knee": counts("xray_completed"),
        "xray_accept_qc_by_knee": counts("xray_accept_qc"),
        "xray_alignment_problem_by_knee": counts("xray_alignment_problem"),
        "xray_centering_problem_by_knee": counts("xray_centering_problem"),
        "xray_incomplete_depiction_by_knee": counts("xray_incomplete_depiction"),
        "xray_positioning_problem_by_knee": counts("xray_positioning_problem"),
        "image_description_by_knee": counts("image_description"),
        "scan_type_by_knee": counts("scan_type"),
        "scan_object_by_knee": counts("scan_object"),
        "image_file_format_by_knee": counts("image_file_format"),
        "image_modality_by_knee": counts("image_modality"),
        "image_release_study_by_knee": counts("image_release_study"),
    }


def _validate_reference_counts(
    analysis: pd.DataFrame,
    study: dict[str, Any],
) -> None:
    references = study.get("reviewed_reference_counts", {})
    observed = {
        "composite_analysis_eligible": {
            "knees": len(analysis),
            "participants": int(analysis["participant_id"].nunique()),
        },
        "radiographic_analysis_eligible": {
            "knees": int(analysis["radiographic_analysis_eligible"].sum()),
            "participants": int(
                analysis.loc[analysis["radiographic_analysis_eligible"], "participant_id"].nunique()
            ),
        },
    }
    contradictions = {
        population: {"expected": references.get(population), "observed": counts}
        for population, counts in observed.items()
        if references.get(population) != counts
    }
    if contradictions:
        raise MaterializationError(
            f"Reviewed Milestone 3 population counts changed: {contradictions}"
        )


def build_analysis_frames(
    package_dir: str | Path,
    study_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and validate local participant-knee outputs without writing them."""

    _validate_configs(study_config, feature_config)
    try:
        build = build_cohort(package_dir, retain_source_values=True)
    except CohortIntegrityError as error:
        raise MaterializationError(str(error)) from error
    analysis = _build_analysis_table(build, study_config, feature_config)
    _validate_reference_counts(analysis, study_config)
    manifest = _build_image_manifest(package_dir, analysis)
    if len(manifest) != len(analysis):
        raise MaterializationError("Image manifest and analysis cohort row counts differ")

    feature_columns = [
        item["output_column"]
        for item in feature_config["primary_features"]
        if item["kind"] == "tabular"
    ]
    populations = {
        "composite": pd.Series(True, index=analysis.index),
        "radiographic": analysis["radiographic_analysis_eligible"],
    }
    report = {
        "study_config": study_config["config_name"],
        "feature_config": feature_config["config_name"],
        "analysis_unit": study_config["unit_of_analysis"],
        "population_summaries": {
            "composite": _population_summary(
                analysis,
                populations["composite"],
                "composite_progression",
            ),
            "radiographic": _population_summary(
                analysis,
                populations["radiographic"],
                "radiographic_kl_progression",
            ),
        },
        "replacement_contribution": {
            "replacement_events": int(analysis["replacement_before_v06"].sum()),
            "replacement_only_events": int(
                (
                    analysis["replacement_before_v06"]
                    & ~analysis["radiographic_kl_progression"].fillna(False)
                ).sum()
            ),
        },
        "per_variable_missingness": {
            name: _missingness_summary(analysis, mask, feature_columns)
            for name, mask in populations.items()
        },
        "domain_availability": {
            name: _domain_summary(analysis, mask) for name, mask in populations.items()
        },
        "image_manifest": _image_summary(manifest),
        "validation": {
            "unique_participant_knee_rows": True,
            "reference_counts_match": True,
            "baseline_only_predictors": True,
            "no_imputation": True,
            "no_split": True,
            "no_model_training": True,
            "one_manifest_row_per_cohort_knee": True,
        },
    }
    return analysis, manifest, report


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


def materialize_analysis_data(
    package_dir: str | Path,
    *,
    study_config_path: str | Path = DEFAULT_STUDY_CONFIG,
    feature_config_path: str | Path = DEFAULT_FEATURE_CONFIG,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Build validated frames, write ignored local Parquet files, and return aggregates."""

    study = load_config(study_config_path)
    features = load_config(feature_config_path)
    analysis, manifest, report = build_analysis_frames(package_dir, study, features)
    output = Path(output_directory)
    cohort_path = output / COHORT_RELATIVE_PATH
    manifest_path = output / MANIFEST_RELATIVE_PATH
    _atomic_parquet_write(analysis, cohort_path)
    _atomic_parquet_write(manifest, manifest_path)
    report["local_outputs"] = {
        "analysis_cohort": str(cohort_path.resolve()),
        "image_manifest": str(manifest_path.resolve()),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the approved v1 cohort and V00 X-ray manifest under an ignored local "
            "processed-data directory; print aggregate-only validation results."
        )
    )
    parser.add_argument("package_dir", type=Path, help="Downloaded, immutable OAI package")
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = materialize_analysis_data(
        args.package_dir,
        study_config_path=args.study_config,
        feature_config_path=args.feature_config,
        output_directory=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
