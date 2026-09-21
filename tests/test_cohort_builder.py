"""Synthetic-only tests for the approved Milestone 3 cohort builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from cohort.builder import CohortIntegrityError, build_cohort, main
from cohort.materialize import (
    MaterializationError,
    build_analysis_frames,
    load_config,
    materialize_analysis_data,
)

pytestmark = pytest.mark.public_portable


def _write_oai_table(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerow([f"Description for {column}" for column in columns])
        writer.writerows(rows)


def _synthetic_package(
    tmp_path: Path,
    *,
    duplicate_primary: bool = False,
    duplicate_image: bool = False,
) -> Path:
    package = tmp_path / "synthetic_oai"
    package.mkdir(parents=True)
    xray_columns = [
        "subjectkey",
        "src_subject_id",
        "visit",
        "readprj",
        "side",
        "xrkl",
        "xrjsl",
        "xrjsm",
        "barcode",
    ]
    xray_rows = [
        # Bilateral participant: one progressing knee and one stable knee.
        ["P1", "S1", "V00", "15", "1", "1", "0", "0", "B1_0"],
        ["P1", "S1", "V06", "15", "1", "2", "0", "1", "B1_6"],
        ["P1", "S1", "V00", "15", "2", "2", "0", "1", "B1_0"],
        ["P1", "S1", "V06", "15", "2", "2", "0", "1", "B1_6"],
        # KL decrease and a fractional JSN follow-up grade.
        ["P2", "S2", "V00", "15", "1", "3", "0", "0", "B2R0"],
        ["P2", "S2", "V06", "15", "1", "2", "0", "1.2", "B2R6"],
        ["P3", "S3", "V00", "15", "1", "4", "3", "3", "B3R0"],
        ["P3", "S3", "V06", "15", "1", "4", "3", "3", "B3R6"],
        # Replacement before planned V06, with no V06 radiograph.
        ["P4", "S4", "V00", "15", "1", "2", "0", "1", "B4R0"],
        # Replacement after planned V06, with no V06 radiograph.
        ["P5", "S5", "V00", "15", "1", "1", "0", "0", "B5R0"],
        # Missing baseline KL.
        ["P6", "S6", "V00", "15", "1", "", "0", "0", "B6R0"],
        ["P6", "S6", "V06", "15", "1", "1", "0", "0", "B6R6"],
        # Missing V06 KL.
        ["P7", "S7", "V00", "15", "1", "1", "0", "0", "B7R0"],
        ["P7", "S7", "V06", "15", "1", "", "0", "0", "B7R6"],
        # Replacement after actual V06 but before the nominal 48-month anniversary.
        ["P8", "S8", "V00", "15", "1", "1", "0", "0", "B8R0"],
        ["P8", "S8", "V06", "15", "1", "1", "0", "0", "B8R6"],
        # Replacement before actual V06 with a stable radiographic outcome.
        ["P9", "S9", "V00", "15", "1", "1", "0", "0", "B9R0"],
        ["P9", "S9", "V06", "15", "1", "1", "0", "0", "B9R6"],
        # A different read project is intentionally ignored.
        ["P1", "S1", "V00", "37", "1", "0", "0", "0", "OTHER"],
    ]
    if duplicate_primary:
        xray_rows.append(["P2", "S2", "V00", "15", "1", "3", "0", "0", "DUP"])
    _write_oai_table(package / "oai_kxrsemiquant01.txt", xray_columns, xray_rows)

    xray_dates = {
        "B8R6": "2023-12-01",
        "B9R6": "2024-01-15",
    }
    barcodes = sorted({row[-1] for row in xray_rows if row[-1] != "OTHER"})
    barcode_owner = {}
    for row in xray_rows:
        if row[-1] != "OTHER":
            barcode_owner.setdefault(row[-1], (row[0], row[1]))
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
    xrmeta_rows = []
    for barcode in barcodes:
        participant, source_participant = barcode_owner[barcode]
        visit = "V06" if barcode.endswith("6") else "V00"
        xrmeta_rows.append(
            [
                participant,
                source_participant,
                xray_dates.get(barcode, "2024-01-01" if visit == "V06" else "2020-01-01"),
                visit,
                "3",
                "1",
                "Y",
                "",
                "",
                "",
                "Bilateral PA Fixed Flexion Knee",
                "",
                barcode,
            ]
        )
    _write_oai_table(package / "oai_xrmeta01.txt", xrmeta_columns, xrmeta_rows)

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
    image_rows = []
    for barcode in barcodes:
        participant, source_participant = barcode_owner[barcode]
        visit = "V06" if barcode.endswith("6") else "V00"
        image_rows.append(
            [
                participant,
                source_participant,
                xray_dates.get(barcode, "2024-01-01" if visit == "V06" else "2020-01-01"),
                f"synthetic/{barcode}.dcm",
                "Bilateral PA Fixed Flexion Knee",
                f"EXP-{barcode}",
                "X-Ray",
                "Live",
                "DICOM",
                "X-Ray",
                "Synthetic image release",
                "",
                visit,
                barcode,
            ]
        )
    if duplicate_image:
        image_rows.append(image_rows[0].copy())
    _write_oai_table(
        package / "image03.txt",
        image_columns,
        image_rows,
    )

    outcome_columns = [
        "subjectkey",
        "src_subject_id",
        "visit",
        "lkdate",
        "rkdate",
        "lkblrp",
        "rkblrp",
        "lktlpr",
        "rktlpr",
        "lkrpcf",
        "rkrpcf",
    ]
    outcome_rows = []
    replacement_dates = {
        "P4": "2023-01-01",
        "P5": "2025-01-01",
        "P8": "2023-12-15",
        "P9": "2023-01-01",
    }
    for number in range(1, 10):
        participant = f"P{number}"
        replacement_date = replacement_dates.get(participant, "")
        outcome_rows.append(
            [
                participant,
                f"S{number}",
                "V99",
                "",
                replacement_date,
                "0",
                "0",
                "",
                "1" if replacement_date else "",
                "",
                "1" if replacement_date else "",
            ]
        )
    _write_oai_table(package / "oai_outcome01.txt", outcome_columns, outcome_rows)

    participant_rows = [
        [
            f"P{number}",
            f"S{number}",
            "V00",
            str(50 + number),
            "F",
            "1",
            "0",
            "1",
            "A",
        ]
        for number in range(1, 10)
    ]
    _write_oai_table(
        package / "oai_enrollee01.txt",
        [
            "subjectkey",
            "src_subject_id",
            "visit",
            "ageyears",
            "sex",
            "race",
            "ethnicity",
            "e_cohort",
            "site",
        ],
        participant_rows,
    )
    _write_oai_table(
        package / "oai_oarisk01.txt",
        ["subjectkey", "src_subject_id", "visit", "bmi", "ksurgl", "ksurgr", "famkr"],
        [[f"P{number}", f"S{number}", "V00", "25", "0", "0", "0"] for number in range(1, 10)],
    )

    pro_columns = [
        "subjectkey",
        "src_subject_id",
        "visit",
        "womac_pain_left",
        "womac_pain_right",
        "womac_stiffness_left",
        "womac_stiffness_right",
        "womac_disability_left",
        "womac_disability_right",
        "koos_lkpain",
        "koos_rkpain",
        "koos_lksymptoms",
        "koos_rksymptoms",
    ]
    pro_rows = []
    for number in range(1, 10):
        values = ["1"] * 10
        if number == 1:
            values = ["2", "12", "3", "7", "4", "14", "60", "90", "70", "95"]
        pro_rows.append([f"P{number}", f"S{number}", "V00", *values])
    _write_oai_table(package / "oai_koos_womac01.txt", pro_columns, pro_rows)

    function_columns = [
        "subjectkey",
        "src_subject_id",
        "visit",
        "w20mpace",
        "cstime1",
        "w400mcmp",
        "w400mtim",
        "lemaxf",
        "remaxf",
        "lfmaxf",
        "rfmaxf",
    ]
    function_rows = []
    for number in range(1, 10):
        values = ["1", "10", "1", "300", "100", "100", "80", "80"]
        if number == 1:
            values = ["1", "10", "1", "300", "100", "200", "80", "180"]
        if number == 4:
            values[5] = ""
        function_rows.append([f"P{number}", f"S{number}", "V00", *values])
    _write_oai_table(package / "oai_physfunct01.txt", function_columns, function_rows)
    return package


def _knee(build, participant: str, side: str = "1"):
    match = build.records.loc[
        build.records["subjectkey"].eq(participant) & build.records["side"].eq(side)
    ]
    assert len(match) == 1
    return match.iloc[0]


def test_kl_labels_ceiling_missingness_and_bilateral_linkage(tmp_path: Path) -> None:
    build = build_cohort(_synthetic_package(tmp_path))

    assert bool(_knee(build, "P1")["radiographic_kl_progression"])
    assert not bool(_knee(build, "P1", "2")["radiographic_kl_progression"])
    assert not bool(_knee(build, "P2")["radiographic_kl_progression"])
    assert not bool(_knee(build, "P3")["baseline_primary_eligible"])
    assert not bool(_knee(build, "P6")["baseline_kl_usable"])
    assert not bool(_knee(build, "P7")["radiographic_analysis_eligible"])

    summary = build.report.radiographic
    assert (summary.eligible_knees, summary.progression_events, summary.non_events) == (5, 1, 4)
    assert (summary.participants_with_one_knee, summary.participants_with_two_knees) == (3, 1)


def test_replacement_timing_and_composite_preserve_missing_v06(tmp_path: Path) -> None:
    build = build_cohort(_synthetic_package(tmp_path))

    before_without_xray = _knee(build, "P4")
    after_without_xray = _knee(build, "P5")
    after_actual_xray = _knee(build, "P8")
    before_actual_xray = _knee(build, "P9")
    assert bool(before_without_xray["replacement_before_v06"])
    assert bool(before_without_xray["composite_analysis_eligible"])
    assert bool(before_without_xray["composite_progression"])
    assert not bool(after_without_xray["replacement_before_v06"])
    assert not bool(after_without_xray["composite_analysis_eligible"])
    assert not bool(after_actual_xray["replacement_before_v06"])
    assert bool(before_actual_xray["replacement_before_v06"])
    assert not bool(before_actual_xray["radiographic_kl_progression"])
    assert bool(before_actual_xray["composite_progression"])

    composite = build.report.composite
    replacement = build.report.replacement
    assert (composite.eligible_knees, composite.progression_events, composite.non_events) == (
        6,
        3,
        3,
    )
    assert replacement.replacements_before_v06_boundary_baseline_eligible == 2
    assert replacement.pre_v06_replacements_without_usable_v06_kl == 1
    assert replacement.pre_v06_replacements_with_usable_v06_kl == 1
    assert replacement.replacement_and_kl_progression_events == 0
    assert replacement.replacement_only_composite_events == 2


def test_flow_jsn_partial_grade_and_domain_masks_do_not_filter(tmp_path: Path) -> None:
    build = build_cohort(_synthetic_package(tmp_path))

    assert [(step.stage, step.knees) for step in build.report.flow] == [
        ("READPRJ 15 V00 source records", 10),
        ("identifier/linkage-consistent participant-knees", 10),
        ("usable baseline KL grade", 9),
        ("baseline KL below 4", 8),
        ("baseline-primary eligible after replacement-at-baseline check", 8),
        ("usable V06 KL grade", 5),
        ("pre-V06 replacement branch", 2),
        ("radiographic-analysis eligible", 5),
        ("composite-analysis eligible", 6),
    ]
    assert bool(_knee(build, "P2")["jsn_progression"])
    assert build.report.jsn_secondary.progression_events == 2
    assert build.report.composite.eligible_knees == 6
    composite_availability = build.report.availability[1]
    assert composite_availability.physical_function_core_complete == 5
    assert bool(_knee(build, "P4")["composite_analysis_eligible"])
    assert not bool(_knee(build, "P4")["physical_function_core_complete"])


def test_duplicate_read_project_rows_fail_instead_of_being_silently_selected(
    tmp_path: Path,
) -> None:
    package = _synthetic_package(tmp_path, duplicate_primary=True)

    with pytest.raises(CohortIntegrityError, match="READPRJ 15 records are not unique"):
        build_cohort(package)


def test_cli_serializes_aggregates_without_synthetic_identifiers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _synthetic_package(tmp_path)

    assert main([str(package)]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["radiographic"]["eligible_knees"] == 5
    assert "P1" not in output
    assert "S1" not in output


def _synthetic_configs() -> tuple[dict, dict]:
    repository = Path(__file__).resolve().parents[1]
    study = load_config(repository / "configs/study_v1.yaml")
    features = load_config(repository / "configs/features_v1.yaml")
    study["reviewed_reference_counts"] = {
        "composite_analysis_eligible": {"knees": 6, "participants": 5},
        "radiographic_analysis_eligible": {"knees": 5, "participants": 4},
    }
    return study, features


def test_analysis_table_side_matching_populations_and_no_postbaseline_predictors(
    tmp_path: Path,
) -> None:
    study, features = _synthetic_configs()
    analysis, manifest, report = build_analysis_frames(
        _synthetic_package(tmp_path), study, features
    )

    assert len(analysis) == 6
    assert analysis["participant_id"].nunique() == 5
    assert not analysis.duplicated(["participant_id", "knee_side_code"]).any()
    assert analysis["radiographic_analysis_eligible"].sum() == 5
    assert report["population_summaries"]["composite"]["events"] == 3
    assert report["population_summaries"]["radiographic"]["events"] == 1
    assert analysis["baseline_visit"].eq("V00").all()
    assert analysis["followup_visit"].eq("V06").all()
    assert analysis["read_project"].eq("15").all()
    assert study["future_partitioning"]["required_grouping_level"] == "participant"
    assert study["future_partitioning"]["both_knees_must_share_partition"] is True
    assert study["future_partitioning"]["implemented_in_this_milestone"] is False

    configured_columns = {
        item["output_column"] for item in features["primary_features"] if item["kind"] == "tabular"
    }
    assert configured_columns.issubset(analysis.columns)
    assert str(analysis["composite_progression"].dtype) == "boolean"
    assert str(analysis["radiographic_kl_progression"].dtype) == "boolean"
    assert analysis["available_domain_count"].between(0, 4).all()

    p1_right = analysis.loc[
        analysis["participant_id"].eq("P1") & analysis["knee_side_code"].eq("1")
    ].iloc[0]
    p1_left = analysis.loc[
        analysis["participant_id"].eq("P1") & analysis["knee_side_code"].eq("2")
    ].iloc[0]
    assert p1_right["womac_pain"] == 12
    assert p1_left["womac_pain"] == 2
    assert p1_right["knee_extension_strength_n"] == 200
    assert p1_left["knee_extension_strength_n"] == 100

    replacement_only = analysis.loc[analysis["participant_id"].eq("P4")].iloc[0]
    assert bool(replacement_only["composite_progression"])
    assert not bool(replacement_only["radiographic_analysis_eligible"])
    assert pd.isna(replacement_only["radiographic_kl_progression"])
    assert not bool(replacement_only["physical_function_domain_complete"])

    forbidden_columns = {"v06_kl", "replacement_date", "v06_boundary_date", "v06_xray_date"}
    assert forbidden_columns.isdisjoint(analysis.columns)
    assert not analysis["baseline_kl_default_predictor"].any()

    p1_manifest = manifest.loc[manifest["participant_id"].eq("P1")]
    assert len(manifest) == len(analysis)
    assert len(p1_manifest) == 2
    assert p1_manifest["accession_number"].nunique() == 1
    assert p1_manifest["knees_sharing_accession"].eq(2).all()
    assert manifest["image_index_visit"].eq("V00").all()
    assert manifest["xray_exam_type"].eq("Bilateral PA Fixed Flexion Knee").all()


def test_materialization_writes_only_requested_local_outputs(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    study, features = _synthetic_configs()
    study_path = tmp_path / "study.yaml"
    feature_path = tmp_path / "features.yaml"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    feature_path.write_text(json.dumps(features), encoding="utf-8")
    output = tmp_path / "processed"

    report = materialize_analysis_data(
        package,
        study_config_path=study_path,
        feature_config_path=feature_path,
        output_directory=output,
    )

    cohort_path = output / "cohorts/analysis_cohort_v1.parquet"
    manifest_path = output / "manifests/v00_xray_manifest.parquet"
    assert cohort_path.is_file()
    assert manifest_path.is_file()
    assert pd.read_parquet(cohort_path).shape[0] == 6
    assert pd.read_parquet(manifest_path).shape[0] == 6
    serialized = json.dumps(report)
    assert "P1" not in serialized
    assert "S1" not in serialized


def test_changed_reference_counts_and_ambiguous_images_fail_closed(tmp_path: Path) -> None:
    study, features = _synthetic_configs()
    changed_study = json.loads(json.dumps(study))
    changed_study["reviewed_reference_counts"]["composite_analysis_eligible"]["knees"] = 7
    with pytest.raises(MaterializationError, match="population counts changed"):
        build_analysis_frames(_synthetic_package(tmp_path / "counts"), changed_study, features)

    with pytest.raises((CohortIntegrityError, MaterializationError), match="not unique"):
        build_analysis_frames(
            _synthetic_package(tmp_path / "images", duplicate_image=True), study, features
        )


def test_postbaseline_or_outcome_source_cannot_be_configured_as_predictor(tmp_path: Path) -> None:
    study, features = _synthetic_configs()
    leaking_features = json.loads(json.dumps(features))
    leaking_features["primary_features"].append(
        {
            "id": "replacement_date_leak",
            "domain": "clinical",
            "kind": "tabular",
            "source_file": "oai_outcome01.txt",
            "source_variables": {"participant": "rkdate"},
            "output_column": "replacement_date_leak",
            "baseline_only": True,
        }
    )

    with pytest.raises(MaterializationError, match="Unapproved primary tabular source"):
        build_analysis_frames(_synthetic_package(tmp_path), study, leaking_features)
