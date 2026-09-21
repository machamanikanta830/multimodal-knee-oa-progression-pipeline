"""Aggregate-only feasibility helpers for the downloaded OAI discrete package.

The functions in this module keep participant identifiers in memory only. Public results are
frozen dataclasses containing counts, percentages, distributions, and documented labels; no
participant-level table is returned or written. Raw files are opened read-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DICTIONARY_ROW_INDEX = 1
BASELINE_VISIT = "V00"
VISIT_MONTHS: Mapping[str, int] = {
    "V00": 0,
    "V01": 12,
    "V03": 24,
    "V05": 36,
    "V06": 48,
    "V08": 72,
    "V10": 96,
}
KNEE_KEY = ("subjectkey", "side")
XRAY_COLUMNS = (
    "subjectkey",
    "visit",
    "readprj",
    "side",
    "xrkl",
    "xrjsl",
    "xrjsm",
    "barcode",
)


@dataclass(frozen=True, slots=True)
class ProjectVisitSummary:
    """Aggregate availability for one semi-quantitative read project and visit."""

    read_project: str
    visit: str
    months: int
    knees: int
    participants: int
    kl_available: int
    medial_jsn_available: int
    lateral_jsn_available: int
    both_jsn_available: int
    knees_with_barcode: int
    unique_barcodes: int
    knees_linked_to_xrmeta: int | None
    knees_linked_to_image_index: int | None
    unique_barcodes_linked_to_xrmeta: int | None
    unique_barcodes_linked_to_image_index: int | None


@dataclass(frozen=True, slots=True)
class HorizonSummary:
    """Aggregate same-knee feasibility for one read project and follow-up visit."""

    read_project: str
    baseline_visit: str
    followup_visit: str
    elapsed_months: int
    baseline_knees: int
    baseline_participants: int
    followup_knees: int
    followup_participants: int
    matched_knees: int
    matched_participants: int
    baseline_knees_without_match: int
    attrition_percent: float
    kl_eligible_knees: int
    kl_eligible_participants: int
    participants_with_one_eligible_knee: int
    participants_with_two_eligible_knees: int
    baseline_kl_distribution: dict[str, int]
    followup_kl_distribution: dict[str, int]


@dataclass(frozen=True, slots=True)
class KneeFeature:
    """One baseline candidate represented at participant-knee level.

    Set ``variable`` for a non-lateral participant feature. Set both ``left_variable`` and
    ``right_variable`` for a side-specific feature. Missing sentinels are opt-in and must be
    supplied only when documented for that exact variable.
    """

    name: str
    source_file: str
    variable: str | None = None
    left_variable: str | None = None
    right_variable: str | None = None
    missing_sentinels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        is_participant = self.variable is not None
        is_lateral = self.left_variable is not None and self.right_variable is not None
        if is_participant == is_lateral:
            raise ValueError(
                "Provide either variable or both left_variable/right_variable, but not both"
            )


@dataclass(frozen=True, slots=True)
class FeatureMissingness:
    """Aggregate baseline missingness for a candidate within an eligible knee set."""

    feature: str
    eligible_knees: int
    nonmissing_knees: int
    missing_knees: int
    missing_percent: float
    eligible_participants: int
    participants_with_any_nonmissing_knee: int


@dataclass(frozen=True, slots=True)
class CompleteCaseSummary:
    """Strict complete-case impact for a named candidate feature panel."""

    panel: str
    eligible_knees: int
    complete_knees: int
    knees_lost: int
    retained_knee_percent: float
    eligible_participants: int
    participants_with_any_complete_knee: int
    participants_lost: int


def _package_path(package_dir: str | Path, filename: str) -> Path:
    package = Path(package_dir).expanduser().resolve()
    path = package / filename
    if not package.is_dir():
        raise NotADirectoryError(f"OAI package directory does not exist: {package}")
    if not path.is_file():
        raise FileNotFoundError(f"Required OAI table does not exist: {path}")
    return path


def _read_oai_table(
    package_dir: str | Path,
    filename: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Read selected columns while excluding the embedded second-line dictionary row."""

    frame = pd.read_csv(
        _package_path(package_dir, filename),
        sep="\t",
        usecols=list(dict.fromkeys(columns)),
        dtype="string",
        skiprows=[DICTIONARY_ROW_INDEX],
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    for column in frame.select_dtypes(include="string").columns:
        frame[column] = frame[column].str.strip().mask(lambda values: values.eq(""))
    return frame


def read_oai_dictionary(
    package_dir: str | Path,
    filename: str,
    columns: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return embedded field descriptions without reading participant records."""

    path = _package_path(package_dir, filename)
    dictionary = pd.read_csv(
        path,
        sep="\t",
        nrows=1,
        dtype="string",
        keep_default_na=False,
    )
    requested = list(dictionary.columns) if columns is None else list(columns)
    missing = sorted(set(requested).difference(dictionary.columns))
    if missing:
        raise ValueError(f"Columns absent from {filename}: {missing}")
    return {column: str(dictionary.at[0, column]) for column in requested}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _grade_distribution(series: pd.Series, valid_grades: Iterable[int]) -> dict[str, int]:
    numeric = _numeric(series)
    return {str(grade): int(numeric.eq(grade).sum()) for grade in valid_grades}


def _load_semquant(package_dir: str | Path) -> pd.DataFrame:
    frame = _read_oai_table(package_dir, "oai_kxrsemiquant01.txt", XRAY_COLUMNS)
    duplicate = frame.duplicated(["subjectkey", "visit", "side", "readprj"], keep=False)
    if duplicate.any():
        raise ValueError(
            "Semi-quantitative records are not unique on subjectkey + visit + side + readprj"
        )
    return frame


def _load_semquant_pair(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if followup_visit not in VISIT_MONTHS or followup_visit == BASELINE_VISIT:
        raise ValueError(f"Unsupported follow-up visit: {followup_visit}")
    xray = _load_semquant(package_dir)
    project = xray.loc[xray["readprj"].eq(str(read_project))]
    baseline = project.loc[project["visit"].eq(BASELINE_VISIT)].copy()
    followup = project.loc[project["visit"].eq(followup_visit)].copy()
    paired = baseline.merge(
        followup,
        on=list(KNEE_KEY),
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_followup"),
    )
    return baseline, followup, paired


def _load_barcode_sets(package_dir: str | Path) -> tuple[set[str], set[str]]:
    xrmeta = _read_oai_table(package_dir, "oai_xrmeta01.txt", ["barcode"])
    xrmeta_barcodes = set(xrmeta["barcode"].dropna())

    image_barcodes: set[str] = set()
    reader = pd.read_csv(
        _package_path(package_dir, "image03.txt"),
        sep="\t",
        usecols=["accession_number"],
        dtype="string",
        skiprows=[DICTIONARY_ROW_INDEX],
        keep_default_na=False,
        na_values=[""],
        chunksize=100_000,
    )
    for chunk in reader:
        values = chunk["accession_number"].str.strip()
        image_barcodes.update(values[values.ne("") & values.notna()].tolist())
    return xrmeta_barcodes, image_barcodes


def summarize_project_visits(
    package_dir: str | Path,
    read_projects: Sequence[str] = ("15", "37", "42"),
    *,
    include_barcode_linkage: bool = True,
) -> list[ProjectVisitSummary]:
    """Summarize visit-level availability for explicitly requested read projects."""

    xray = _load_semquant(package_dir)
    xrmeta_barcodes: set[str] = set()
    image_barcodes: set[str] = set()
    if include_barcode_linkage:
        xrmeta_barcodes, image_barcodes = _load_barcode_sets(package_dir)

    summaries: list[ProjectVisitSummary] = []
    for read_project in read_projects:
        project = xray.loc[xray["readprj"].eq(str(read_project))]
        for visit in VISIT_MONTHS:
            rows = project.loc[project["visit"].eq(visit)]
            if rows.empty:
                continue
            barcodes = set(rows["barcode"].dropna())
            summaries.append(
                ProjectVisitSummary(
                    read_project=str(read_project),
                    visit=visit,
                    months=VISIT_MONTHS[visit],
                    knees=len(rows),
                    participants=int(rows["subjectkey"].nunique()),
                    kl_available=int(_numeric(rows["xrkl"]).notna().sum()),
                    medial_jsn_available=int(_numeric(rows["xrjsm"]).notna().sum()),
                    lateral_jsn_available=int(_numeric(rows["xrjsl"]).notna().sum()),
                    both_jsn_available=int(
                        (_numeric(rows["xrjsm"]).notna() & _numeric(rows["xrjsl"]).notna()).sum()
                    ),
                    knees_with_barcode=int(rows["barcode"].notna().sum()),
                    unique_barcodes=len(barcodes),
                    knees_linked_to_xrmeta=(
                        int(rows["barcode"].isin(xrmeta_barcodes).sum())
                        if include_barcode_linkage
                        else None
                    ),
                    knees_linked_to_image_index=(
                        int(rows["barcode"].isin(image_barcodes).sum())
                        if include_barcode_linkage
                        else None
                    ),
                    unique_barcodes_linked_to_xrmeta=(
                        len(barcodes.intersection(xrmeta_barcodes))
                        if include_barcode_linkage
                        else None
                    ),
                    unique_barcodes_linked_to_image_index=(
                        len(barcodes.intersection(image_barcodes))
                        if include_barcode_linkage
                        else None
                    ),
                )
            )
    return summaries


def summarize_horizon(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> HorizonSummary:
    """Summarize same-project, same-participant-knee longitudinal KL feasibility."""

    baseline, followup, paired = _load_semquant_pair(package_dir, str(read_project), followup_visit)
    baseline_kl = _numeric(paired["xrkl_baseline"])
    followup_kl = _numeric(paired["xrkl_followup"])
    eligible = paired.loc[baseline_kl.notna() & followup_kl.notna()].copy()
    counts_per_participant = eligible.groupby("subjectkey", observed=True).size()
    matched_knees = len(paired)
    attrition = len(baseline) - matched_knees
    return HorizonSummary(
        read_project=str(read_project),
        baseline_visit=BASELINE_VISIT,
        followup_visit=followup_visit,
        elapsed_months=VISIT_MONTHS[followup_visit],
        baseline_knees=len(baseline),
        baseline_participants=int(baseline["subjectkey"].nunique()),
        followup_knees=len(followup),
        followup_participants=int(followup["subjectkey"].nunique()),
        matched_knees=matched_knees,
        matched_participants=int(paired["subjectkey"].nunique()),
        baseline_knees_without_match=attrition,
        attrition_percent=round(100 * attrition / len(baseline), 3) if len(baseline) else 0.0,
        kl_eligible_knees=len(eligible),
        kl_eligible_participants=int(eligible["subjectkey"].nunique()),
        participants_with_one_eligible_knee=int(counts_per_participant.eq(1).sum()),
        participants_with_two_eligible_knees=int(counts_per_participant.eq(2).sum()),
        baseline_kl_distribution=_grade_distribution(eligible["xrkl_baseline"], range(5)),
        followup_kl_distribution=_grade_distribution(eligible["xrkl_followup"], range(5)),
    )


def summarize_horizons(
    package_dir: str | Path,
    project_followups: Mapping[str, Sequence[str]],
) -> list[HorizonSummary]:
    """Summarize multiple explicitly supplied project/follow-up combinations."""

    return [
        summarize_horizon(package_dir, project, followup)
        for project, followups in project_followups.items()
        for followup in followups
    ]


def _eligible_knees(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
) -> pd.DataFrame:
    _, _, paired = _load_semquant_pair(package_dir, read_project, followup_visit)
    eligible = paired.loc[
        _numeric(paired["xrkl_baseline"]).notna() & _numeric(paired["xrkl_followup"]).notna(),
        ["subjectkey", "side"],
    ].copy()
    return eligible


def _load_feature_tables(
    package_dir: str | Path,
    features: Sequence[KneeFeature],
) -> dict[str, pd.DataFrame]:
    columns_by_file: dict[str, set[str]] = {}
    for feature in features:
        columns = columns_by_file.setdefault(feature.source_file, {"subjectkey", "visit"})
        if feature.variable is not None:
            columns.add(feature.variable)
        else:
            columns.update([feature.left_variable, feature.right_variable])

    tables: dict[str, pd.DataFrame] = {}
    for filename, columns in columns_by_file.items():
        table = _read_oai_table(package_dir, filename, sorted(columns))
        baseline = table.loc[table["visit"].eq(BASELINE_VISIT)].copy()
        if baseline.duplicated("subjectkey").any():
            raise ValueError(f"{filename} is not unique by subjectkey at {BASELINE_VISIT}")
        tables[filename] = baseline
    return tables


def _feature_availability(
    eligible: pd.DataFrame,
    feature: KneeFeature,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    merged = eligible.merge(baseline, on="subjectkey", how="left", validate="many_to_one")
    if feature.variable is not None:
        value = merged[feature.variable]
    else:
        value = merged[feature.right_variable].where(
            merged["side"].eq("1"), merged[feature.left_variable]
        )
        value = value.where(merged["side"].isin(["1", "2"]))
    if feature.missing_sentinels:
        value = value.mask(value.isin(feature.missing_sentinels))
    return merged[["subjectkey", "side"]].assign(**{feature.name: value.notna()})


def summarize_feature_missingness(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
    features: Sequence[KneeFeature],
) -> list[FeatureMissingness]:
    """Calculate knee-level baseline missingness in a KL-eligible reference population."""

    eligible = _eligible_knees(package_dir, read_project, followup_visit)
    tables = _load_feature_tables(package_dir, features)
    participant_count = int(eligible["subjectkey"].nunique())
    summaries: list[FeatureMissingness] = []
    for feature in features:
        availability = _feature_availability(eligible, feature, tables[feature.source_file])
        nonmissing = int(availability[feature.name].sum())
        participants_nonmissing = int(
            availability.loc[availability[feature.name], "subjectkey"].nunique()
        )
        missing = len(eligible) - nonmissing
        summaries.append(
            FeatureMissingness(
                feature=feature.name,
                eligible_knees=len(eligible),
                nonmissing_knees=nonmissing,
                missing_knees=missing,
                missing_percent=round(100 * missing / len(eligible), 3),
                eligible_participants=participant_count,
                participants_with_any_nonmissing_knee=participants_nonmissing,
            )
        )
    return summaries


def summarize_complete_cases(
    package_dir: str | Path,
    read_project: str,
    followup_visit: str,
    panels: Mapping[str, Sequence[KneeFeature]],
) -> list[CompleteCaseSummary]:
    """Quantify strict complete-case losses for named, explicitly supplied feature panels."""

    eligible = _eligible_knees(package_dir, read_project, followup_visit)
    all_features = [feature for features in panels.values() for feature in features]
    tables = _load_feature_tables(package_dir, all_features)
    participant_count = int(eligible["subjectkey"].nunique())
    summaries: list[CompleteCaseSummary] = []
    for panel, features in panels.items():
        if not features:
            raise ValueError(f"Feature panel {panel!r} is empty")
        availability = eligible.copy()
        for feature in features:
            feature_frame = _feature_availability(eligible, feature, tables[feature.source_file])
            availability = availability.merge(
                feature_frame,
                on=list(KNEE_KEY),
                how="left",
                validate="one_to_one",
            )
        complete_mask = availability[[feature.name for feature in features]].all(axis=1)
        complete_knees = int(complete_mask.sum())
        complete_participants = int(availability.loc[complete_mask, "subjectkey"].nunique())
        summaries.append(
            CompleteCaseSummary(
                panel=panel,
                eligible_knees=len(eligible),
                complete_knees=complete_knees,
                knees_lost=len(eligible) - complete_knees,
                retained_knee_percent=round(100 * complete_knees / len(eligible), 3),
                eligible_participants=participant_count,
                participants_with_any_complete_knee=complete_participants,
                participants_lost=participant_count - complete_participants,
            )
        )
    return summaries
