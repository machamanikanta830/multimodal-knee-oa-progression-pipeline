"""Synthetic-only tests for aggregate OAI cohort and outcome feasibility helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from cohort.feasibility import (
    KneeFeature,
    read_oai_dictionary,
    summarize_complete_cases,
    summarize_feature_missingness,
    summarize_horizon,
    summarize_project_visits,
)
from cohort.outcome_candidates import (
    summarize_jsn_progression,
    summarize_kl_progression,
    summarize_quantitative_jsw_change,
    summarize_replacement_interference,
)

pytestmark = pytest.mark.public_portable


XRAY_COLUMNS = [
    "subjectkey",
    "visit",
    "readprj",
    "side",
    "xrkl",
    "xrjsl",
    "xrjsm",
    "barcode",
]


def _write_oai_table(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
    descriptions: dict[str, str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerow(
            [(descriptions or {}).get(column, f"Description for {column}") for column in columns]
        )
        writer.writerows(rows)


def _synthetic_package(tmp_path: Path) -> Path:
    package = tmp_path / "synthetic_oai"
    package.mkdir()
    _write_oai_table(
        package / "oai_kxrsemiquant01.txt",
        XRAY_COLUMNS,
        [
            ["P1", "V00", "15", "1", "1", "0", "0", "B1"],
            ["P1", "V06", "15", "1", "2", "0", "1", "B2"],
            ["P1", "V00", "15", "2", "4", "3", "3", "B1"],
            ["P1", "V06", "15", "2", "4", "3", "3", "B2"],
            ["P2", "V00", "15", "1", "0", "0", "0", "B3"],
            ["P2", "V06", "15", "1", "0", "1", "0", "B4"],
            ["P3", "V00", "15", "1", "2", "0", "1", "B5"],
        ],
        descriptions={"xrkl": "Kellgren-Lawrence grade"},
    )
    _write_oai_table(
        package / "oai_kxrquantjsw01.txt",
        ["subjectkey", "visit", "readprj", "side", "mcmjsw"],
        [
            ["P1", "V00", "16", "1", "4.0"],
            ["P1", "V06", "16", "1", "3.0"],
            ["P2", "V00", "16", "1", "2.0"],
            ["P2", "V06", "16", "1", "2.0"],
        ],
    )
    _write_oai_table(
        package / "oai_outcome01.txt",
        ["subjectkey", "visit", "lkdays", "rkdays", "lkblrp", "rkblrp"],
        [
            ["P1", "V99", "", "1000", "0", "0"],
            ["P2", "V99", "", "", "0", "0"],
            ["P3", "V99", "", "1200", "0", "0"],
        ],
    )
    _write_oai_table(
        package / "features.txt",
        ["subjectkey", "visit", "age", "pain_left", "pain_right", "pace"],
        [
            ["P1", "V00", "60", "", "2", "1.1"],
            ["P2", "V00", "65", "1", "3", ""],
            ["P3", "V00", "70", "2", "2", "1.0"],
        ],
    )
    return package


def test_dictionary_row_is_read_as_documentation_and_skipped_from_counts(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)

    dictionary = read_oai_dictionary(package, "oai_kxrsemiquant01.txt", ["xrkl"])
    summary = summarize_horizon(package, "15", "V06")

    assert dictionary == {"xrkl": "Kellgren-Lawrence grade"}
    assert summary.baseline_knees == 4
    assert summary.followup_knees == 3
    assert summary.matched_knees == 3
    assert summary.matched_participants == 2
    assert summary.participants_with_one_eligible_knee == 1
    assert summary.participants_with_two_eligible_knees == 1
    assert summary.baseline_kl_distribution == {"0": 1, "1": 1, "2": 0, "3": 0, "4": 1}


def test_project_summary_does_not_require_image_tables_when_linkage_disabled(
    tmp_path: Path,
) -> None:
    package = _synthetic_package(tmp_path)

    summaries = summarize_project_visits(package, ["15"], include_barcode_linkage=False)

    assert [(summary.visit, summary.knees) for summary in summaries] == [("V00", 4), ("V06", 3)]
    assert all(summary.knees_linked_to_xrmeta is None for summary in summaries)


def test_project_summary_distinguishes_knee_records_from_bilateral_barcodes(
    tmp_path: Path,
) -> None:
    package = _synthetic_package(tmp_path)
    _write_oai_table(package / "oai_xrmeta01.txt", ["barcode"], [["B1"], ["B3"]])
    _write_oai_table(package / "image03.txt", ["accession_number"], [["B1"]])

    baseline = summarize_project_visits(package, ["15"])[0]

    assert baseline.knees == 4
    assert baseline.unique_barcodes == 3
    assert baseline.knees_linked_to_xrmeta == 3
    assert baseline.unique_barcodes_linked_to_xrmeta == 2
    assert baseline.knees_linked_to_image_index == 2
    assert baseline.unique_barcodes_linked_to_image_index == 1


def test_ordinal_progression_summaries_include_ceiling_impact(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)

    kl = summarize_kl_progression(package, "15", "V06")
    jsn = summarize_jsn_progression(package, "15", "V06")

    assert (kl.eligible_knees, kl.progression_events, kl.nonprogression_knees) == (3, 1, 2)
    assert kl.baseline_kl4_ceiling_cases == 1
    assert (kl.eligible_excluding_baseline_kl4, kl.events_excluding_baseline_kl4) == (2, 1)
    assert (jsn.eligible_knees, jsn.progression_events, jsn.nonprogression_knees) == (3, 2, 1)
    assert jsn.baseline_kl_distribution_progressors == {
        "0": 1,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    serialized = json.dumps([asdict(kl), asdict(jsn)])
    assert "P1" not in serialized
    assert "P2" not in serialized


def test_quantitative_jsw_reports_continuous_change_without_event_threshold(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)

    summary = summarize_quantitative_jsw_change(package, "16", "V06")

    assert summary.eligible_knees == 2
    assert summary.followup_minus_baseline_mean_mm == -0.5
    assert summary.followup_minus_baseline_median_mm == -0.5
    assert not hasattr(summary, "progression_events")


def test_feature_and_complete_case_results_are_aggregate_only(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)
    age = KneeFeature(name="age", source_file="features.txt", variable="age")
    pain = KneeFeature(
        name="pain",
        source_file="features.txt",
        left_variable="pain_left",
        right_variable="pain_right",
    )
    pace = KneeFeature(name="pace", source_file="features.txt", variable="pace")

    missingness = summarize_feature_missingness(package, "15", "V06", [age, pain, pace])
    complete = summarize_complete_cases(
        package,
        "15",
        "V06",
        {"pro": [pain], "all": [age, pain, pace]},
    )

    assert [(item.feature, item.missing_knees) for item in missingness] == [
        ("age", 0),
        ("pain", 1),
        ("pace", 1),
    ]
    assert [(item.panel, item.complete_knees) for item in complete] == [("pro", 2), ("all", 1)]
    serialized = json.dumps([asdict(item) for item in [*missingness, *complete]])
    assert "P1" not in serialized
    assert "P2" not in serialized


def test_replacement_timing_reports_matched_and_unmatched_baseline_knees(tmp_path: Path) -> None:
    package = _synthetic_package(tmp_path)

    summary = summarize_replacement_interference(package, "15", "V06")

    assert summary.baseline_knees == 4
    assert summary.replacements_by_scheduled_horizon == 2
    assert summary.replacements_by_horizon_with_followup_xray_match == 1
    assert summary.replacements_by_horizon_without_followup_xray_match == 1
