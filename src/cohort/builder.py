"""Construct the approved Milestone 3 OAI cohort in memory.

The public command-line interface prints aggregate JSON only. ``build_cohort`` returns an
in-memory participant-knee table so later pipeline stages can reuse and test the linkage logic,
but it never writes that table or prints participant identifiers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cohort.feasibility import _load_barcode_sets, _numeric, _read_oai_table

READ_PROJECT = "15"
BASELINE_VISIT = "V00"
FOLLOWUP_VISIT = "V06"
VALID_KL_GRADES = frozenset(range(5))
KNEE_KEY = ["subjectkey", "side"]
MISSING_SENTINELS = frozenset({"-9", "-5", "-2"})


class CohortIntegrityError(ValueError):
    """Raised when source ambiguity prevents deterministic participant-knee linkage."""


@dataclass(frozen=True, slots=True)
class FlowCount:
    """Aggregate count at one cohort-flow stage."""

    stage: str
    knees: int
    participants: int
    knees_excluded_from_previous: int | None


@dataclass(frozen=True, slots=True)
class IntegritySummary:
    """Aggregate-only identifier, key, and timing checks."""

    baseline_source_rows: int
    invalid_baseline_identifier_or_side_rows: int
    inconsistent_identifier_mapping_rows: int
    duplicate_read_project_rows: int
    v06_knees_without_v00_match: int
    baseline_barcodes_without_xray_metadata: int
    baseline_barcodes_without_image_index: int
    replacement_dates_not_after_baseline: int
    replacement_dates_with_unresolved_timing: int
    baseline_replacement_flags: int


@dataclass(frozen=True, slots=True)
class EndpointSummary:
    """Aggregate outcome and contributor counts for one analysis endpoint."""

    endpoint: str
    eligible_knees: int
    unique_participants: int
    participants_with_one_knee: int
    participants_with_two_knees: int
    progression_events: int
    non_events: int
    event_rate_percent: float
    baseline_kl_distribution: dict[str, int]
    event_baseline_kl_distribution: dict[str, int]
    non_event_baseline_kl_distribution: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReplacementSummary:
    """Aggregate replacement timing and contribution to the composite endpoint."""

    documented_postbaseline_replacements_starting_knees: int
    replacements_before_v06_boundary_starting_knees: int
    pre_v06_replacements_excluded_by_baseline_rules: int
    documented_postbaseline_replacements_baseline_eligible: int
    replacements_before_v06_boundary_baseline_eligible: int
    replacements_after_v06_boundary_baseline_eligible: int
    pre_v06_replacements_with_usable_v06_kl: int
    pre_v06_replacements_without_usable_v06_kl: int
    composite_events_with_replacement: int
    replacement_and_kl_progression_events: int
    replacement_only_composite_events: int


@dataclass(frozen=True, slots=True)
class AvailabilitySummary:
    """Baseline feature availability for one outcome-eligible population."""

    population: str
    eligible_knees: int
    imaging_linkage_available: int
    pro_core_complete: int
    clinical_core_complete: int
    physical_function_core_complete: int
    all_four_domains_complete: int
    component_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CohortReport:
    """All aggregate outputs retained from one cohort build."""

    package_directory: str
    read_project: str
    baseline_visit: str
    followup_visit: str
    replacement_boundary_rule: str
    flow: tuple[FlowCount, ...]
    exclusions: dict[str, int]
    integrity: IntegritySummary
    radiographic: EndpointSummary
    composite: EndpointSummary
    jsn_secondary: EndpointSummary
    replacement: ReplacementSummary
    availability: tuple[AvailabilitySummary, ...]


@dataclass(slots=True)
class CohortBuild:
    """In-memory participant-knee records plus their aggregate report.

    ``records`` contains participant identifiers and must not be persisted to tracked files.
    """

    records: pd.DataFrame = field(repr=False)
    report: CohortReport


def _usable_grade(series: pd.Series, valid_grades: frozenset[int]) -> pd.Series:
    numeric = _numeric(series)
    return numeric.notna() & numeric.isin(valid_grades)


def _usable_jsn(series: pd.Series) -> pd.Series:
    """Accept documented OARSI 0–3 values, including observed partial grades."""

    numeric = _numeric(series)
    return numeric.notna() & numeric.between(0, 3, inclusive="both")


def _present(series: pd.Series, sentinels: frozenset[str] = frozenset()) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.notna() & values.ne("") & ~values.isin(sentinels)


def _side_value(
    frame: pd.DataFrame,
    side: pd.Series,
    left_column: str,
    right_column: str,
) -> pd.Series:
    return frame[right_column].where(side.eq("1"), frame[left_column])


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _check_unique(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    if frame.duplicated(columns, keep=False).any():
        joined = " + ".join(columns)
        raise CohortIntegrityError(f"{source} is not unique on {joined}")


def _baseline_table(
    package_dir: str | Path,
    filename: str,
    columns: list[str],
) -> pd.DataFrame:
    requested = list(dict.fromkeys(["subjectkey", "src_subject_id", "visit", *columns]))
    table = _read_oai_table(package_dir, filename, requested)
    baseline = table.loc[table["visit"].eq(BASELINE_VISIT)].copy()
    _check_unique(baseline, ["subjectkey"], f"{filename} {BASELINE_VISIT}")
    return baseline


def _merge_subject_table(
    records: pd.DataFrame,
    table: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    """Join a participant table and reject cross-table identifier disagreement."""

    linked_source_id = "_linked_src_subject_id"
    table = table.rename(columns={"src_subject_id": linked_source_id})
    result = records.merge(table, on="subjectkey", how="left", validate="many_to_one")
    mismatch = (
        result["src_subject_id"].notna()
        & result[linked_source_id].notna()
        & result["src_subject_id"].ne(result[linked_source_id])
    )
    if mismatch.any():
        raise CohortIntegrityError(f"Identifier mapping disagrees between X-ray data and {source}")
    return result.drop(columns=[linked_source_id])


def _identifier_conflicts(project: pd.DataFrame) -> tuple[set[str], set[str]]:
    complete = project.dropna(subset=["subjectkey", "src_subject_id"])
    subject_conflicts = set(
        complete.groupby("subjectkey")["src_subject_id"].nunique().loc[lambda x: x.gt(1)].index
    )
    source_conflicts = set(
        complete.groupby("src_subject_id")["subjectkey"].nunique().loc[lambda x: x.gt(1)].index
    )
    return subject_conflicts, source_conflicts


def _load_primary_xrays(package_dir: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
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
    xray = _read_oai_table(package_dir, "oai_kxrsemiquant01.txt", columns)
    project = xray.loc[
        xray["readprj"].eq(READ_PROJECT) & xray["visit"].isin([BASELINE_VISIT, FOLLOWUP_VISIT])
    ].copy()
    duplicate = project.duplicated([*KNEE_KEY, "visit", "readprj"], keep=False)
    if duplicate.any():
        raise CohortIntegrityError(
            "READPRJ 15 records are not unique on participant + visit + knee"
        )

    subject_conflicts, source_conflicts = _identifier_conflicts(project)
    baseline_source = project.loc[project["visit"].eq(BASELINE_VISIT)].copy()
    missing_or_bad = (
        baseline_source["subjectkey"].isna()
        | baseline_source["src_subject_id"].isna()
        | ~baseline_source["side"].isin(["1", "2"])
    )
    inconsistent = baseline_source["subjectkey"].isin(subject_conflicts) | baseline_source[
        "src_subject_id"
    ].isin(source_conflicts)
    baseline = baseline_source.loc[~missing_or_bad & ~inconsistent].copy()

    followup = project.loc[project["visit"].eq(FOLLOWUP_VISIT)].copy()
    followup_valid = followup.loc[
        followup["subjectkey"].notna()
        & followup["src_subject_id"].notna()
        & followup["side"].isin(["1", "2"])
        & ~followup["subjectkey"].isin(subject_conflicts)
        & ~followup["src_subject_id"].isin(source_conflicts)
    ].copy()
    baseline_keys = pd.MultiIndex.from_frame(baseline[KNEE_KEY])
    followup_keys = pd.MultiIndex.from_frame(followup_valid[KNEE_KEY])
    unmatched_followup = int((~followup_keys.isin(baseline_keys)).sum())

    stats = {
        "baseline_source_rows": len(baseline_source),
        "baseline_source_participants": int(baseline_source["subjectkey"].nunique()),
        "invalid_baseline_identifier_or_side_rows": int(missing_or_bad.sum()),
        "inconsistent_identifier_mapping_rows": int((~missing_or_bad & inconsistent).sum()),
        "duplicate_read_project_rows": 0,
        "v06_knees_without_v00_match": unmatched_followup,
    }
    paired = baseline.merge(
        followup_valid.drop(columns=["src_subject_id", "readprj"]),
        on=KNEE_KEY,
        how="left",
        validate="one_to_one",
        suffixes=("_baseline", "_v06"),
    )
    return paired, stats


def _add_image_linkage(
    records: pd.DataFrame,
    package_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    xrmeta = _read_oai_table(
        package_dir,
        "oai_xrmeta01.txt",
        ["barcode", "interview_date"],
    )
    _check_unique(xrmeta, ["barcode"], "oai_xrmeta01.txt")
    xrmeta["xray_date"] = _parse_dates(xrmeta["interview_date"])
    metadata_dates = xrmeta.set_index("barcode")["xray_date"]
    _, image_barcodes = _load_barcode_sets(package_dir)

    result = records.copy()
    result["baseline_xray_date"] = result["barcode_baseline"].map(metadata_dates)
    result["v06_xray_date"] = result["barcode_v06"].map(metadata_dates)
    result["imaging_linkage_available"] = (
        result["barcode_baseline"].notna()
        & result["barcode_baseline"].isin(metadata_dates.index)
        & result["barcode_baseline"].isin(image_barcodes)
    )
    stats = {
        "baseline_barcodes_without_xray_metadata": int(
            (~result["barcode_baseline"].isin(metadata_dates.index)).sum()
        ),
        "baseline_barcodes_without_image_index": int(
            (~result["barcode_baseline"].isin(image_barcodes)).sum()
        ),
    }
    return result, stats


def _add_replacement_timing(
    records: pd.DataFrame,
    package_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
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
    outcome = _read_oai_table(package_dir, "oai_outcome01.txt", columns)
    outcome = outcome.loc[outcome["visit"].eq("V99")].copy()
    _check_unique(outcome, ["subjectkey"], "oai_outcome01.txt V99")
    result = _merge_subject_table(
        records,
        outcome.drop(columns=["visit"]),
        source="oai_outcome01.txt V99",
    )
    result["replacement_date"] = _parse_dates(
        _side_value(result, result["side"], "lkdate", "rkdate")
    )
    result["baseline_replacement_flag"] = _numeric(
        _side_value(result, result["side"], "lkblrp", "rkblrp")
    ).eq(1)
    result["replacement_type_documented"] = _present(
        _side_value(result, result["side"], "lktlpr", "rktlpr")
    )
    result["replacement_confirmation_documented"] = _present(
        _side_value(result, result["side"], "lkrpcf", "rkrpcf")
    )

    planned_boundary = result["baseline_xray_date"].map(
        lambda value: value + pd.DateOffset(months=48) if pd.notna(value) else pd.NaT
    )
    result["v06_boundary_date"] = result["v06_xray_date"].fillna(planned_boundary)
    has_replacement = result["replacement_date"].notna()
    after_baseline = has_replacement & result["replacement_date"].gt(result["baseline_xray_date"])
    timing_resolved = ~has_replacement | (
        result["baseline_xray_date"].notna() & result["v06_boundary_date"].notna()
    )
    result["documented_postbaseline_replacement"] = after_baseline
    result["replacement_before_v06"] = (
        after_baseline
        & timing_resolved
        & result["replacement_date"].le(result["v06_boundary_date"])
    )
    result["replacement_after_v06"] = (
        after_baseline
        & timing_resolved
        & result["replacement_date"].gt(result["v06_boundary_date"])
    )
    stats = {
        "replacement_dates_not_after_baseline": int(
            (has_replacement & result["baseline_xray_date"].notna() & ~after_baseline).sum()
        ),
        "replacement_dates_with_unresolved_timing": int((has_replacement & ~timing_resolved).sum()),
        "baseline_replacement_flags": int(result["baseline_replacement_flag"].sum()),
    }
    return result, stats


def _add_feature_masks(records: pd.DataFrame, package_dir: str | Path) -> pd.DataFrame:
    result = records.copy()

    pro_columns = [
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
    pro = _baseline_table(package_dir, "oai_koos_womac01.txt", pro_columns)
    result = _merge_subject_table(
        result,
        pro.drop(columns=["visit"]),
        source="oai_koos_womac01.txt V00",
    )
    pro_pairs = {
        "womac_pain_available": ("womac_pain_left", "womac_pain_right"),
        "womac_stiffness_available": ("womac_stiffness_left", "womac_stiffness_right"),
        "womac_disability_available": (
            "womac_disability_left",
            "womac_disability_right",
        ),
        "koos_pain_available": ("koos_lkpain", "koos_rkpain"),
        "koos_symptoms_available": ("koos_lksymptoms", "koos_rksymptoms"),
    }
    for mask, (left, right) in pro_pairs.items():
        result[mask] = _present(_side_value(result, result["side"], left, right))
    pro_masks = list(pro_pairs)
    result["pro_any_available"] = result[pro_masks].any(axis=1)
    result["pro_core_complete"] = result[pro_masks].all(axis=1)

    enrollee = _baseline_table(
        package_dir,
        "oai_enrollee01.txt",
        ["ageyears", "sex", "race", "ethnicity", "e_cohort", "site"],
    ).drop(columns=["visit"])
    enrollee = enrollee.rename(columns={"ageyears": "baseline_age", "sex": "baseline_sex"})
    result = _merge_subject_table(
        result,
        enrollee,
        source="oai_enrollee01.txt V00",
    )

    risk = _baseline_table(
        package_dir,
        "oai_oarisk01.txt",
        ["bmi", "ksurgl", "ksurgr", "famkr"],
    ).drop(columns=["visit"])
    result = _merge_subject_table(result, risk, source="oai_oarisk01.txt V00")
    result["age_available"] = _present(result["baseline_age"])
    result["sex_available"] = _present(result["baseline_sex"])
    result["bmi_available"] = _present(result["bmi"], MISSING_SENTINELS)
    result["prior_knee_surgery_available"] = _present(
        _side_value(result, result["side"], "ksurgl", "ksurgr")
    )
    result["family_replacement_history_available"] = _present(result["famkr"])
    result["clinical_core_complete"] = result[
        ["age_available", "sex_available", "bmi_available"]
    ].all(axis=1)
    result["clinical_with_history_complete"] = result[
        [
            "age_available",
            "sex_available",
            "bmi_available",
            "prior_knee_surgery_available",
            "family_replacement_history_available",
        ]
    ].all(axis=1)

    function_columns = [
        "w20mpace",
        "cstime1",
        "w400mcmp",
        "w400mtim",
        "lemaxf",
        "remaxf",
        "lfmaxf",
        "rfmaxf",
    ]
    function = _baseline_table(package_dir, "oai_physfunct01.txt", function_columns).drop(
        columns=["visit"]
    )
    result = _merge_subject_table(result, function, source="oai_physfunct01.txt V00")
    result["walk_20m_available"] = _present(result["w20mpace"])
    result["chair_stand_available"] = _present(result["cstime1"])
    result["walk_400m_status_available"] = _present(result["w400mcmp"])
    result["walk_400m_time_available"] = _present(result["w400mtim"])
    result["extension_strength_available"] = _present(
        _side_value(result, result["side"], "lemaxf", "remaxf")
    )
    result["flexion_strength_available"] = _present(
        _side_value(result, result["side"], "lfmaxf", "rfmaxf")
    )
    result["physical_function_any_available"] = result[
        [
            "walk_20m_available",
            "chair_stand_available",
            "extension_strength_available",
            "flexion_strength_available",
            "walk_400m_status_available",
            "walk_400m_time_available",
        ]
    ].any(axis=1)
    result["physical_function_core_complete"] = result[
        [
            "walk_20m_available",
            "chair_stand_available",
            "extension_strength_available",
            "flexion_strength_available",
        ]
    ].all(axis=1)
    return result


def _add_eligibility_and_labels(records: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    result["baseline_kl"] = _numeric(result["xrkl_baseline"])
    result["v06_kl"] = _numeric(result["xrkl_v06"])
    result["baseline_kl_usable"] = _usable_grade(result["xrkl_baseline"], VALID_KL_GRADES)
    result["v06_kl_usable"] = _usable_grade(result["xrkl_v06"], VALID_KL_GRADES)
    result["baseline_primary_eligible"] = (
        result["baseline_kl_usable"]
        & result["baseline_kl"].lt(4)
        & ~result["baseline_replacement_flag"]
    )
    result["radiographic_analysis_eligible"] = (
        result["baseline_primary_eligible"] & result["v06_kl_usable"]
    )
    result["radiographic_kl_progression"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    radiographic = result["radiographic_analysis_eligible"]
    result.loc[radiographic, "radiographic_kl_progression"] = (
        result.loc[radiographic, "v06_kl"] - result.loc[radiographic, "baseline_kl"]
    ).ge(1)

    result["composite_analysis_eligible"] = result["baseline_primary_eligible"] & (
        result["v06_kl_usable"] | result["replacement_before_v06"]
    )
    result["composite_progression"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    composite = result["composite_analysis_eligible"]
    radiographic_event = result["radiographic_kl_progression"].fillna(False)
    result.loc[composite, "composite_progression"] = (
        radiographic_event.loc[composite] | result.loc[composite, "replacement_before_v06"]
    )

    required_jsn = [
        ("xrjsm_baseline", "baseline_jsn_medial"),
        ("xrjsm_v06", "v06_jsn_medial"),
        ("xrjsl_baseline", "baseline_jsn_lateral"),
        ("xrjsl_v06", "v06_jsn_lateral"),
    ]
    jsn_complete = pd.Series(True, index=result.index)
    for source, target in required_jsn:
        result[target] = _numeric(result[source])
        jsn_complete &= _usable_jsn(result[source])
    result["jsn_analysis_eligible"] = result["baseline_primary_eligible"] & jsn_complete
    result["jsn_progression"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    jsn = result["jsn_analysis_eligible"]
    result.loc[jsn, "jsn_progression"] = (
        result.loc[jsn, "v06_jsn_medial"] - result.loc[jsn, "baseline_jsn_medial"]
    ).ge(1) | (result.loc[jsn, "v06_jsn_lateral"] - result.loc[jsn, "baseline_jsn_lateral"]).ge(1)
    return result


def _population_count(records: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    population = records.loc[mask]
    return len(population), int(population["subjectkey"].nunique())


def _grade_distribution(series: pd.Series) -> dict[str, int]:
    return {str(grade): int(series.eq(grade).sum()) for grade in range(5)}


def _endpoint_summary(
    records: pd.DataFrame,
    *,
    endpoint: str,
    eligible_column: str,
    label_column: str,
) -> EndpointSummary:
    eligible = records.loc[records[eligible_column]].copy()
    label = eligible[label_column].astype(bool)
    contributors = eligible.groupby("subjectkey").size()
    if contributors.gt(2).any():
        raise CohortIntegrityError("A participant contributes more than two knees")
    events = int(label.sum())
    return EndpointSummary(
        endpoint=endpoint,
        eligible_knees=len(eligible),
        unique_participants=int(eligible["subjectkey"].nunique()),
        participants_with_one_knee=int(contributors.eq(1).sum()),
        participants_with_two_knees=int(contributors.eq(2).sum()),
        progression_events=events,
        non_events=len(eligible) - events,
        event_rate_percent=round(100 * events / len(eligible), 3) if len(eligible) else 0.0,
        baseline_kl_distribution=_grade_distribution(eligible["baseline_kl"]),
        event_baseline_kl_distribution=_grade_distribution(eligible.loc[label, "baseline_kl"]),
        non_event_baseline_kl_distribution=_grade_distribution(eligible.loc[~label, "baseline_kl"]),
    )


def _availability_summary(
    records: pd.DataFrame,
    population: str,
    eligible_column: str,
) -> AvailabilitySummary:
    eligible = records.loc[records[eligible_column]]
    domain_columns = [
        "imaging_linkage_available",
        "pro_core_complete",
        "clinical_core_complete",
        "physical_function_core_complete",
    ]
    components = [
        "pro_any_available",
        "womac_pain_available",
        "womac_stiffness_available",
        "womac_disability_available",
        "koos_pain_available",
        "koos_symptoms_available",
        "age_available",
        "sex_available",
        "bmi_available",
        "prior_knee_surgery_available",
        "family_replacement_history_available",
        "clinical_with_history_complete",
        "physical_function_any_available",
        "walk_20m_available",
        "chair_stand_available",
        "extension_strength_available",
        "flexion_strength_available",
        "walk_400m_status_available",
        "walk_400m_time_available",
    ]
    all_domains = eligible[domain_columns].all(axis=1)
    return AvailabilitySummary(
        population=population,
        eligible_knees=len(eligible),
        imaging_linkage_available=int(eligible["imaging_linkage_available"].sum()),
        pro_core_complete=int(eligible["pro_core_complete"].sum()),
        clinical_core_complete=int(eligible["clinical_core_complete"].sum()),
        physical_function_core_complete=int(eligible["physical_function_core_complete"].sum()),
        all_four_domains_complete=int(all_domains.sum()),
        component_counts={column: int(eligible[column].sum()) for column in components},
    )


def _build_report(
    records: pd.DataFrame,
    package_dir: str | Path,
    source_stats: dict[str, int],
) -> CohortReport:
    identity = pd.Series(True, index=records.index)
    baseline_usable = records["baseline_kl_usable"]
    below_four = baseline_usable & records["baseline_kl"].lt(4)
    baseline_primary = records["baseline_primary_eligible"]
    radiographic = records["radiographic_analysis_eligible"]
    replacement = baseline_primary & records["replacement_before_v06"]
    composite = records["composite_analysis_eligible"]

    flow_masks = [
        ("identifier/linkage-consistent participant-knees", identity),
        ("usable baseline KL grade", baseline_usable),
        ("baseline KL below 4", below_four),
        ("baseline-primary eligible after replacement-at-baseline check", baseline_primary),
        ("usable V06 KL grade", radiographic),
        ("pre-V06 replacement branch", replacement),
        ("radiographic-analysis eligible", radiographic),
        ("composite-analysis eligible", composite),
    ]
    flow: list[FlowCount] = []
    flow.append(
        FlowCount(
            "READPRJ 15 V00 source records",
            source_stats["baseline_source_rows"],
            source_stats["baseline_source_participants"],
            None,
        )
    )
    previous_count: int | None = source_stats["baseline_source_rows"]
    for stage, mask in flow_masks:
        knees, participants = _population_count(records, mask)
        excluded = (
            None if stage in {"pre-V06 replacement branch", "composite-analysis eligible"} else 0
        )
        if excluded is not None and previous_count is not None:
            excluded = previous_count - knees
        flow.append(FlowCount(stage, knees, participants, excluded))
        if stage != "pre-V06 replacement branch":
            previous_count = knees

    radiographic_summary = _endpoint_summary(
        records,
        endpoint="radiographic KL progression",
        eligible_column="radiographic_analysis_eligible",
        label_column="radiographic_kl_progression",
    )
    composite_summary = _endpoint_summary(
        records,
        endpoint="KL progression or replacement before V06",
        eligible_column="composite_analysis_eligible",
        label_column="composite_progression",
    )
    jsn_summary = _endpoint_summary(
        records,
        endpoint="medial or lateral JSN progression",
        eligible_column="jsn_analysis_eligible",
        label_column="jsn_progression",
    )

    pre_v06 = baseline_primary & records["replacement_before_v06"]
    radiographic_event = records["radiographic_kl_progression"].fillna(False)
    replacement_summary = ReplacementSummary(
        documented_postbaseline_replacements_starting_knees=int(
            records["documented_postbaseline_replacement"].sum()
        ),
        replacements_before_v06_boundary_starting_knees=int(
            records["replacement_before_v06"].sum()
        ),
        pre_v06_replacements_excluded_by_baseline_rules=int(
            (records["replacement_before_v06"] & ~baseline_primary).sum()
        ),
        documented_postbaseline_replacements_baseline_eligible=int(
            (baseline_primary & records["documented_postbaseline_replacement"]).sum()
        ),
        replacements_before_v06_boundary_baseline_eligible=int(pre_v06.sum()),
        replacements_after_v06_boundary_baseline_eligible=int(
            (baseline_primary & records["replacement_after_v06"]).sum()
        ),
        pre_v06_replacements_with_usable_v06_kl=int((pre_v06 & records["v06_kl_usable"]).sum()),
        pre_v06_replacements_without_usable_v06_kl=int((pre_v06 & ~records["v06_kl_usable"]).sum()),
        composite_events_with_replacement=int((composite & pre_v06).sum()),
        replacement_and_kl_progression_events=int((composite & pre_v06 & radiographic_event).sum()),
        replacement_only_composite_events=int((composite & pre_v06 & ~radiographic_event).sum()),
    )

    exclusions = {
        "invalid_identifier_or_side": source_stats["invalid_baseline_identifier_or_side_rows"],
        "inconsistent_identifier_mapping": source_stats["inconsistent_identifier_mapping_rows"],
        "no_usable_baseline_kl": int((~baseline_usable).sum()),
        "baseline_kl_4": int((baseline_usable & records["baseline_kl"].eq(4)).sum()),
        "baseline_replacement_flag": int((below_four & records["baseline_replacement_flag"]).sum()),
        "no_usable_v06_kl_after_baseline_eligibility": int(
            (baseline_primary & ~records["v06_kl_usable"]).sum()
        ),
        "no_usable_v06_kl_but_preserved_for_composite_by_replacement": int(
            (baseline_primary & ~records["v06_kl_usable"] & pre_v06).sum()
        ),
        "no_v06_kl_and_no_pre_v06_replacement": int(
            (baseline_primary & ~records["v06_kl_usable"] & ~pre_v06).sum()
        ),
    }
    integrity = IntegritySummary(
        baseline_source_rows=source_stats["baseline_source_rows"],
        invalid_baseline_identifier_or_side_rows=source_stats[
            "invalid_baseline_identifier_or_side_rows"
        ],
        inconsistent_identifier_mapping_rows=source_stats["inconsistent_identifier_mapping_rows"],
        duplicate_read_project_rows=source_stats["duplicate_read_project_rows"],
        v06_knees_without_v00_match=source_stats["v06_knees_without_v00_match"],
        baseline_barcodes_without_xray_metadata=source_stats[
            "baseline_barcodes_without_xray_metadata"
        ],
        baseline_barcodes_without_image_index=source_stats["baseline_barcodes_without_image_index"],
        replacement_dates_not_after_baseline=source_stats["replacement_dates_not_after_baseline"],
        replacement_dates_with_unresolved_timing=source_stats[
            "replacement_dates_with_unresolved_timing"
        ],
        baseline_replacement_flags=source_stats["baseline_replacement_flags"],
    )
    return CohortReport(
        package_directory=str(Path(package_dir).expanduser().resolve()),
        read_project=READ_PROJECT,
        baseline_visit=BASELINE_VISIT,
        followup_visit=FOLLOWUP_VISIT,
        replacement_boundary_rule=(
            "actual V06 X-ray acquisition date when present; otherwise baseline X-ray "
            "+ 48 calendar months"
        ),
        flow=tuple(flow),
        exclusions=exclusions,
        integrity=integrity,
        radiographic=radiographic_summary,
        composite=composite_summary,
        jsn_secondary=jsn_summary,
        replacement=replacement_summary,
        availability=(
            _availability_summary(
                records, "radiographic-analysis eligible", "radiographic_analysis_eligible"
            ),
            _availability_summary(
                records, "composite-analysis eligible", "composite_analysis_eligible"
            ),
        ),
    )


def _drop_raw_feature_values(records: pd.DataFrame) -> pd.DataFrame:
    """Retain linkage, labels, and masks while dropping raw baseline feature values."""

    raw_columns = {
        "src_subject_id",
        "visit_baseline",
        "visit_v06",
        "xrkl_baseline",
        "xrkl_v06",
        "xrjsm_baseline",
        "xrjsm_v06",
        "xrjsl_baseline",
        "xrjsl_v06",
        "barcode_baseline",
        "barcode_v06",
        "lkdate",
        "rkdate",
        "lkblrp",
        "rkblrp",
        "lktlpr",
        "rktlpr",
        "lkrpcf",
        "rkrpcf",
        "baseline_age",
        "baseline_sex",
        "race",
        "ethnicity",
        "e_cohort",
        "site",
        "bmi",
        "ksurgl",
        "ksurgr",
        "famkr",
        "w20mpace",
        "cstime1",
        "w400mcmp",
        "w400mtim",
        "lemaxf",
        "remaxf",
        "lfmaxf",
        "rfmaxf",
    }
    raw_columns.update(
        column
        for column in records.columns
        if column.startswith(("womac_", "koos_")) and not column.endswith("_available")
    )
    return records.drop(columns=sorted(raw_columns.intersection(records.columns)))


def build_cohort(
    package_dir: str | Path,
    *,
    retain_source_values: bool = False,
) -> CohortBuild:
    """Build approved V00→V06 READPRJ-15 records in memory and aggregate them.

    Raw files are read but never modified. The returned participant-level key is required for
    bilateral linkage and later participant-grouped splitting; callers must not log or persist it
    to tracked artifacts. Set ``retain_source_values`` only for local, ignored analysis-data
    materialization. The aggregate command-line interface always keeps its safe default of
    dropping source feature values.
    """

    records, source_stats = _load_primary_xrays(package_dir)
    records, image_stats = _add_image_linkage(records, package_dir)
    source_stats.update(image_stats)
    records, replacement_stats = _add_replacement_timing(records, package_dir)
    source_stats.update(replacement_stats)
    records = _add_feature_masks(records, package_dir)
    records = _add_eligibility_and_labels(records)
    report = _build_report(records, package_dir, source_stats)
    output_records = records if retain_source_values else _drop_raw_feature_values(records)
    return CohortBuild(records=output_records, report=report)


def report_as_dict(report: CohortReport) -> dict[str, Any]:
    """Return a JSON-safe aggregate report."""

    return asdict(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the approved READPRJ-15 V00-to-V06 cohort in memory and print only "
            "aggregate counts."
        )
    )
    parser.add_argument("package_dir", type=Path, help="Downloaded OAI package directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build = build_cohort(args.package_dir)
    print(json.dumps(report_as_dict(build.report), indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
