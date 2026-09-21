"""Aggregate candidate radiographic outcome summaries for OAI feasibility review.

No function in this module constructs or persists a final cohort or participant-level outcome
table. Candidate labels are calculated only within one explicit ``READPRJ`` and returned as
aggregate counts and distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cohort.feasibility import (
    BASELINE_VISIT,
    KNEE_KEY,
    VISIT_MONTHS,
    _load_semquant_pair,
    _numeric,
    _read_oai_table,
)


@dataclass(frozen=True, slots=True)
class ProgressionSummary:
    """Aggregate event summary for one candidate ordinal progression definition."""

    definition: str
    read_project: str
    followup_visit: str
    elapsed_months: int
    eligible_knees: int
    eligible_participants: int
    progression_events: int
    nonprogression_knees: int
    event_rate_percent: float
    baseline_kl_missing: int
    baseline_kl_distribution_progressors: dict[str, int]
    baseline_kl_distribution_nonprogressors: dict[str, int]
    baseline_kl4_ceiling_cases: int
    eligible_excluding_baseline_kl4: int
    events_excluding_baseline_kl4: int
    nonprogression_excluding_baseline_kl4: int
    event_rate_excluding_baseline_kl4_percent: float


@dataclass(frozen=True, slots=True)
class QuantitativeChangeSummary:
    """Aggregate continuous change distribution for quantitative minimum JSW."""

    read_project: str
    followup_visit: str
    elapsed_months: int
    eligible_knees: int
    eligible_participants: int
    followup_minus_baseline_mean_mm: float
    followup_minus_baseline_sd_mm: float
    followup_minus_baseline_min_mm: float
    followup_minus_baseline_p05_mm: float
    followup_minus_baseline_p25_mm: float
    followup_minus_baseline_median_mm: float
    followup_minus_baseline_p75_mm: float
    followup_minus_baseline_p95_mm: float
    followup_minus_baseline_max_mm: float


@dataclass(frozen=True, slots=True)
class ReplacementHorizonSummary:
    """Aggregate follow-up-replacement timing among baseline project knees."""

    read_project: str
    followup_visit: str
    scheduled_horizon_days: int
    baseline_knees: int
    baseline_participants: int
    knees_with_documented_followup_replacement: int
    replacements_by_scheduled_horizon: int
    replacements_by_horizon_with_followup_xray_match: int
    replacements_by_horizon_without_followup_xray_match: int
    baseline_replacement_xray_flags: int


def _grade_distribution(series: pd.Series) -> dict[str, int]:
    numeric = _numeric(series)
    return {str(grade): int(numeric.eq(grade).sum()) for grade in range(5)}


def _progression_summary(
    eligible: pd.DataFrame,
    event: pd.Series,
    *,
    definition: str,
    read_project: str,
    followup_visit: str,
) -> ProgressionSummary:
    event = event.astype(bool)
    baseline_kl = _numeric(eligible["xrkl_baseline"])
    ceiling = baseline_kl.eq(4).fillna(False)
    retained = ~ceiling
    retained_events = event & retained
    retained_count = int(retained.sum())
    event_count = int(event.sum())
    retained_event_count = int(retained_events.sum())
    return ProgressionSummary(
        definition=definition,
        read_project=str(read_project),
        followup_visit=followup_visit,
        elapsed_months=VISIT_MONTHS[followup_visit],
        eligible_knees=len(eligible),
        eligible_participants=int(eligible["subjectkey"].nunique()),
        progression_events=event_count,
        nonprogression_knees=len(eligible) - event_count,
        event_rate_percent=(round(100 * event_count / len(eligible), 3) if len(eligible) else 0.0),
        baseline_kl_missing=int(baseline_kl.isna().sum()),
        baseline_kl_distribution_progressors=_grade_distribution(baseline_kl[event]),
        baseline_kl_distribution_nonprogressors=_grade_distribution(baseline_kl[~event]),
        baseline_kl4_ceiling_cases=int(ceiling.sum()),
        eligible_excluding_baseline_kl4=retained_count,
        events_excluding_baseline_kl4=retained_event_count,
        nonprogression_excluding_baseline_kl4=retained_count - retained_event_count,
        event_rate_excluding_baseline_kl4_percent=(
            round(100 * retained_event_count / retained_count, 3) if retained_count else 0.0
        ),
    )


def summarize_kl_progression(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> ProgressionSummary:
    """Summarize the candidate rule follow-up K-L minus baseline K-L >= 1."""

    _, _, paired = _load_semquant_pair(package_dir, str(read_project), followup_visit)
    baseline = _numeric(paired["xrkl_baseline"])
    followup = _numeric(paired["xrkl_followup"])
    eligible = paired.loc[baseline.notna() & followup.notna()].copy()
    event = _numeric(eligible["xrkl_followup"]) - _numeric(eligible["xrkl_baseline"]) >= 1
    return _progression_summary(
        eligible,
        event,
        definition="K-L grade increase >=1",
        read_project=str(read_project),
        followup_visit=followup_visit,
    )


def summarize_jsn_progression(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> ProgressionSummary:
    """Summarize >=1 increase in medial or lateral OARSI JSN within one read project."""

    _, _, paired = _load_semquant_pair(package_dir, str(read_project), followup_visit)
    required = [
        "xrjsm_baseline",
        "xrjsm_followup",
        "xrjsl_baseline",
        "xrjsl_followup",
    ]
    complete = (
        pd.concat([_numeric(paired[column]) for column in required], axis=1).notna().all(axis=1)
    )
    eligible = paired.loc[complete].copy()
    medial_change = _numeric(eligible["xrjsm_followup"]) - _numeric(eligible["xrjsm_baseline"])
    lateral_change = _numeric(eligible["xrjsl_followup"]) - _numeric(eligible["xrjsl_baseline"])
    event = medial_change.ge(1) | lateral_change.ge(1)
    return _progression_summary(
        eligible,
        event,
        definition="Medial or lateral OARSI JSN grade increase >=1",
        read_project=str(read_project),
        followup_visit=followup_visit,
    )


def summarize_quantitative_jsw_change(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> QuantitativeChangeSummary:
    """Describe continuous follow-up minus baseline minimum medial JSW without thresholding."""

    columns = ["subjectkey", "visit", "readprj", "side", "mcmjsw"]
    frame = _read_oai_table(package_dir, "oai_kxrquantjsw01.txt", columns)
    project = frame.loc[frame["readprj"].eq(str(read_project))]
    baseline = project.loc[project["visit"].eq(BASELINE_VISIT)]
    followup = project.loc[project["visit"].eq(followup_visit)]
    if (
        baseline.duplicated(["subjectkey", "side"]).any()
        or followup.duplicated(["subjectkey", "side"]).any()
    ):
        raise ValueError(
            "Quantitative JSW records are not unique within the requested project/visit"
        )
    paired = baseline.merge(
        followup,
        on=list(KNEE_KEY),
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_followup"),
    )
    baseline_value = _numeric(paired["mcmjsw_baseline"])
    followup_value = _numeric(paired["mcmjsw_followup"])
    eligible = paired.loc[baseline_value.notna() & followup_value.notna()].copy()
    change = _numeric(eligible["mcmjsw_followup"]) - _numeric(eligible["mcmjsw_baseline"])
    quantiles = change.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return QuantitativeChangeSummary(
        read_project=str(read_project),
        followup_visit=followup_visit,
        elapsed_months=VISIT_MONTHS[followup_visit],
        eligible_knees=len(eligible),
        eligible_participants=int(eligible["subjectkey"].nunique()),
        followup_minus_baseline_mean_mm=round(float(change.mean()), 4),
        followup_minus_baseline_sd_mm=round(float(change.std(ddof=1)), 4),
        followup_minus_baseline_min_mm=round(float(change.min()), 4),
        followup_minus_baseline_p05_mm=round(float(quantiles.loc[0.05]), 4),
        followup_minus_baseline_p25_mm=round(float(quantiles.loc[0.25]), 4),
        followup_minus_baseline_median_mm=round(float(quantiles.loc[0.5]), 4),
        followup_minus_baseline_p75_mm=round(float(quantiles.loc[0.75]), 4),
        followup_minus_baseline_p95_mm=round(float(quantiles.loc[0.95]), 4),
        followup_minus_baseline_max_mm=round(float(change.max()), 4),
    )


def summarize_replacement_interference(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> ReplacementHorizonSummary:
    """Count documented replacements relative to a scheduled horizon among baseline knees.

    The horizon cutoff uses scheduled months converted to 365.25-day years. It is a feasibility
    approximation, not a censoring rule; a future cohort must compare event and acquisition dates.
    """

    baseline, followup, _ = _load_semquant_pair(package_dir, str(read_project), followup_visit)
    followup_keys = followup.loc[:, ["subjectkey", "side"]].assign(has_followup=True)
    outcomes = _read_oai_table(
        package_dir,
        "oai_outcome01.txt",
        ["subjectkey", "visit", "lkdays", "rkdays", "lkblrp", "rkblrp"],
    )
    outcome = outcomes.loc[outcomes["visit"].eq("V99")].copy()
    if outcome.duplicated("subjectkey").any():
        raise ValueError("Outcome summary is not unique by subjectkey")
    knee = baseline.loc[:, ["subjectkey", "side"]].merge(
        outcome, on="subjectkey", how="left", validate="many_to_one"
    )
    knee = knee.merge(followup_keys, on=list(KNEE_KEY), how="left", validate="one_to_one")
    replacement_days = _numeric(knee["rkdays"]).where(
        knee["side"].eq("1"), _numeric(knee["lkdays"])
    )
    baseline_replacement = knee["rkblrp"].where(knee["side"].eq("1"), knee["lkblrp"])
    horizon_days = int(round(VISIT_MONTHS[followup_visit] * 365.25 / 12))
    by_horizon = replacement_days.notna() & replacement_days.le(horizon_days)
    has_followup = knee["has_followup"].fillna(False).astype(bool)
    return ReplacementHorizonSummary(
        read_project=str(read_project),
        followup_visit=followup_visit,
        scheduled_horizon_days=horizon_days,
        baseline_knees=len(knee),
        baseline_participants=int(knee["subjectkey"].nunique()),
        knees_with_documented_followup_replacement=int(replacement_days.notna().sum()),
        replacements_by_scheduled_horizon=int(by_horizon.sum()),
        replacements_by_horizon_with_followup_xray_match=int((by_horizon & has_followup).sum()),
        replacements_by_horizon_without_followup_xray_match=int((by_horizon & ~has_followup).sum()),
        baseline_replacement_xray_flags=int(
            pd.to_numeric(baseline_replacement, errors="coerce").fillna(0).ne(0).sum()
        ),
    )
