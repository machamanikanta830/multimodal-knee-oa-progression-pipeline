"""Integrate, audit and immutably freeze the approved participant-knee dataset (Milestone 6A).

Only the locked cohort and approved adjudicated imaging manifest supply row values. No cohort
builder is invoked, no images are generated, and no predictor processing is fitted. The freeze
marker is published last and is the entry point for future participant-grouped split generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cohort.materialize import _validate_configs, load_config
from imaging.artifact_io import atomic_json, atomic_parquet, hash_tree, sha256_file
from imaging.final_adjudicated import _aggregate, verify_automatic_tree
from imaging.frozen_full_v3 import DEFAULT_ARCHIVE_LEDGER, DEFAULT_ARCHIVE_ROOT
from imaging.v3_review_ui import ReviewSession

DATASET_VERSION = "final_multimodal_v1"
DEFAULT_DIRECTORY = Path("data/processed/multimodal/final_v1")
DEFAULT_COHORT = Path("data/processed/cohorts/analysis_cohort_v1.parquet")
DEFAULT_IMAGE_ROOT = Path("data/processed/oai_images/v3_frozen_full")
DEFAULT_IMAGING = DEFAULT_IMAGE_ROOT / "final_adjudicated_v1/imaging_manifest.parquet"
APPROVED_IMAGING_SHA = "cce476f5ecc2381232b026a9aec09dffe58db391116deadb7fddb2405aee0fa7"
KEYS = ("participant_id", "knee_side_code", "baseline_visit")
TARGETS = ("composite_progression", "radiographic_kl_progression", "jsn_progression")
CLINICAL = ("age_years", "sex", "bmi", "prior_knee_surgery", "family_knee_replacement_history")
PRO = ("womac_pain", "womac_stiffness", "womac_disability", "koos_pain", "koos_symptoms")
FUNCTION = (
    "walk_20m_pace_mps",
    "chair_stand_time_seconds",
    "knee_extension_strength_n",
    "knee_flexion_strength_n",
)
GROUPS = {
    "identifiers": (
        "participant_id",
        "source_participant_id",
        "knee_side_code",
        "knee_side_label",
        "baseline_visit",
    ),
    "targets": TARGETS,
    "target_support_metadata": (
        "followup_visit",
        "baseline_jsn_medial",
        "baseline_jsn_lateral",
        "radiographic_analysis_eligible",
        "composite_analysis_eligible",
        "jsn_analysis_eligible",
        "replacement_before_v06",
    ),
    "demographic_clinical": CLINICAL,
    "patient_reported": PRO,
    "physical_function": FUNCTION,
    "baseline_radiographic_severity": ("baseline_kl",),
    "imaging_input": ("effective_crop_relative_path",),
    "imaging_provenance_metadata": (
        "acquisition_index",
        "panel_position",
        "anatomical_side",
        "laterality_provenance",
        "effective_crop_sha256",
        "effective_provenance",
        "preprocessing_version",
        "localizer_version",
        "laterality_policy_version",
        "localizer_source_sha256",
        "laterality_source_sha256",
        "preprocessing_source_sha256",
        "final_manifest_version",
        "target_spacing_mm",
        "crop_size_mm",
        "crop_rows",
        "crop_columns",
        "crop_dtype",
    ),
    "non_predictor_metadata": (
        "read_project",
        "descriptive_race",
        "descriptive_ethnicity",
        "design_site",
        "design_oai_cohort",
    ),
}
OMITTED = (
    "baseline_barcode",
    "imaging_domain_complete",
    "pro_domain_complete",
    "clinical_domain_complete",
    "physical_function_domain_complete",
    "available_domain_count",
    "all_four_domains_complete",
    "baseline_kl_default_predictor",
)
COHORT_COLUMNS = tuple(
    c for g, columns in GROUPS.items() if not g.startswith("imaging") for c in columns
)
IMAGE_COLUMNS = (*GROUPS["imaging_input"], *GROUPS["imaging_provenance_metadata"])
TABULAR_A = (*CLINICAL, *PRO, *FUNCTION)
ALLOWED_PROVENANCE = ("AUTO_PASS", "HUMAN_ACCEPT", "HUMAN_OVERRIDE")
EXPECTED = {
    "rows": 6961,
    "participants": 3621,
    "outcomes": {
        "composite_progression": {"observed": 6961, "positive": 1071, "missing": 0},
        "radiographic_kl_progression": {"observed": 6851, "positive": 961, "missing": 110},
        "jsn_progression": {"observed": 6851, "positive": 734, "missing": 110},
        "replacement_before_v06": {"observed": 6961, "positive": 110, "missing": 0},
    },
    "provenance": {"AUTO_PASS": 5089, "HUMAN_ACCEPT": 1476, "HUMAN_OVERRIDE": 396},
}
ARTIFACTS = (
    "final_multimodal_dataset.parquet",
    "schema.json",
    "feature_groups.json",
    "missingness_audit.json",
    "dataset_integrity_audit.json",
)


class DatasetIntegrityError(ValueError):
    """Ambiguous, leaked, missing or changed inputs cannot be frozen."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetIntegrityError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Unexpected JSON structure")
    return value


def outcome_counts(frame: pd.DataFrame) -> dict:
    return {
        c: {
            "observed": int(frame[c].notna().sum()),
            "positive": int(frame[c].sum()),
            "missing": int(frame[c].isna().sum()),
        }
        for c in (*TARGETS, "replacement_before_v06")
    }


def validate_cohort(frame: pd.DataFrame, expected: dict | None) -> None:
    require(
        set(frame.columns) == set((*COHORT_COLUMNS, *OMITTED)),
        "Locked cohort has missing/unclassified/unexpected columns",
    )
    require(not frame[list(KEYS)].isna().any().any(), "Null cohort key")
    require(
        not frame.duplicated(["participant_id", "knee_side_code"]).any(), "Duplicate cohort key"
    )
    require(frame.knee_side_code.isin(["1", "2"]).all(), "Invalid cohort knee side")
    require(
        frame.knee_side_label.eq(frame.knee_side_code.map({"1": "right", "2": "left"})).all(),
        "Wrong cohort knee side label",
    )
    require(
        frame.baseline_visit.eq("V00").all() and frame.followup_visit.eq("V06").all(),
        "Unexpected visit mismatch",
    )
    require(frame.read_project.eq("15").all(), "Unexpected radiographic read project")
    require(frame.baseline_kl.isin([0, 1, 2, 3]).all(), "Invalid baseline KL or KL4 knee")
    require(
        not frame.baseline_kl_default_predictor.any(), "Baseline KL cannot be a default predictor"
    )
    require(not frame.source_participant_id.isna().any(), "Missing source participant identity")
    for left, right in (
        ("participant_id", "source_participant_id"),
        ("source_participant_id", "participant_id"),
    ):
        require(
            frame.groupby(left)[right].nunique().eq(1).all(),
            "Inconsistent participant identity mapping",
        )
    for c in (
        *TARGETS,
        "replacement_before_v06",
        "radiographic_analysis_eligible",
        "composite_analysis_eligible",
        "jsn_analysis_eligible",
    ):
        require(
            pd.api.types.is_bool_dtype(frame[c].dtype),
            "Outcome/support flags must retain Boolean semantics",
        )
    require(
        frame.composite_progression.notna().all() and frame.composite_analysis_eligible.all(),
        "Locked composite outcome unavailable",
    )
    for target, flag in (
        ("radiographic_kl_progression", "radiographic_analysis_eligible"),
        ("jsn_progression", "jsn_analysis_eligible"),
    ):
        require(
            frame[target].notna().eq(frame[flag]).all(), "Outcome availability/eligibility mismatch"
        )
    require(
        frame.loc[frame.radiographic_kl_progression.isna(), "replacement_before_v06"].all(),
        "Unavailable KL outcome lacks locked replacement branch",
    )
    require(
        frame.loc[frame.replacement_before_v06, "composite_progression"].all(),
        "Replacement branch is not a composite event",
    )
    for columns, marker in (
        (PRO, "pro_domain_complete"),
        (CLINICAL, "clinical_domain_complete"),
        (FUNCTION, "physical_function_domain_complete"),
    ):
        require(
            frame[list(columns)].notna().all(axis=1).eq(frame[marker]).all(),
            "Legacy domain completeness mismatch",
        )
    if expected is not None:
        require(
            len(frame) == expected["rows"]
            and frame.participant_id.nunique() == expected["participants"],
            "Locked cohort size mismatch",
        )
        require(
            outcome_counts(frame) == expected["outcomes"],
            "Locked outcome counts mismatch; STOP, do not redefine",
        )


def verify_images(frame: pd.DataFrame, root: Path) -> dict:
    require(
        frame.effective_provenance.isin(ALLOWED_PROVENANCE).all(),
        "Unresolved/excluded effective crop",
    )
    require(
        frame.imaging_included.eq(True).all()
        and frame.inclusion_status.eq("INCLUDED").all()
        and frame.exclusion_reason.isna().all(),
        "Unresolved/excluded imaging row",
    )
    require(
        not frame.effective_crop_relative_path.duplicated().any(),
        "Duplicate effective-crop mapping",
    )
    geometry = {
        "target_spacing_mm": 0.15,
        "crop_size_mm": 160.0,
        "crop_rows": 1067,
        "crop_columns": 1067,
        "crop_dtype": "uint16",
    }
    for c, value in geometry.items():
        require(frame[c].eq(value).all(), "Frozen image geometry mismatch")

    def verify(row):
        side = "R" if row.knee_side_code == "1" else "L"
        panel = "screen_left" if side == "R" else "screen_right"
        require(
            row.anatomical_side == side and row.panel_position == panel,
            "Wrong anatomical side/panel",
        )
        prefix = "overrides" if row.effective_provenance == "HUMAN_OVERRIDE" else "crops"
        reference = f"{prefix}/acquisition_{row.acquisition_index:04d}/{panel}_uint16.npy"
        require(
            row.effective_crop_relative_path == reference,
            "Effective crop acquisition/panel identity mismatch",
        )
        path = root / reference
        require(
            path.resolve().is_relative_to(root.resolve()) and path.is_file(),
            "Missing/unsafe effective image file",
        )
        require(
            isinstance(row.effective_crop_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", row.effective_crop_sha256) is not None,
            "Malformed image SHA",
        )
        require(sha256_file(path) == row.effective_crop_sha256, "Effective-crop SHA mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(
            array.shape == (1067, 1067) and array.dtype == np.dtype("uint16"),
            "Effective crop shape/dtype mismatch",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, frame.itertuples(index=False)))
    return {
        "images_checked": len(frame),
        "unique_effective_images": int(frame.effective_crop_relative_path.nunique()),
        "missing_files": 0,
        "sha_mismatches": 0,
        "side_mismatches": 0,
        "geometry_mismatches": 0,
    }


def integrate(
    cohort: pd.DataFrame, imaging: pd.DataFrame, root: Path, *, expected: dict | None = EXPECTED
) -> tuple[pd.DataFrame, dict]:
    validate_cohort(cohort, expected)
    required = {
        *KEYS,
        "source_participant_id",
        "knee_side_label",
        "imaging_included",
        "inclusion_status",
        "exclusion_reason",
        *IMAGE_COLUMNS,
    }
    require(required.issubset(imaging.columns), "Imaging manifest lacks required columns")
    require(not imaging[list(KEYS)].isna().any().any(), "Null imaging linkage key")
    require(
        not imaging.duplicated(["participant_id", "knee_side_code"]).any(), "Duplicate imaging key"
    )
    require(imaging.baseline_visit.eq("V00").all(), "Unexpected imaging visit mismatch")
    require(len(imaging) == len(cohort), "Missing/extra imaging rows")
    ordered = cohort[list(COHORT_COLUMNS)].copy()
    ordered["_row_order"] = range(len(cohort))
    selected = imaging[list(required)].copy()
    joined = ordered.merge(
        selected,
        on=list(KEYS),
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("", "_imaging"),
        indicator=True,
    )
    require(joined._merge.eq("both").all(), "Missing imaging link/participant mismatch")
    require(
        joined.source_participant_id.eq(joined.source_participant_id_imaging).all(),
        "Imaging participant mismatch",
    )
    require(
        joined.knee_side_label.eq(joined.knee_side_label_imaging).all(), "Imaging side mismatch"
    )
    joined = joined.sort_values("_row_order", ignore_index=True)
    image_audit = verify_images(joined, Path(root))
    final = joined[[*COHORT_COLUMNS, *IMAGE_COLUMNS]].copy()
    for c in COHORT_COLUMNS:
        final[c] = final[c].astype(cohort[c].dtype)
    for c in IMAGE_COLUMNS:
        final[c] = final[c].astype(imaging[c].dtype)
    pd.testing.assert_frame_equal(
        final[list(COHORT_COLUMNS)], cohort[list(COHORT_COLUMNS)].reset_index(drop=True)
    )
    require(len(final) == len(cohort), "Silent cohort row drop")
    provenance = {p: int(final.effective_provenance.eq(p).sum()) for p in ALLOWED_PROVENANCE}
    if expected is not None:
        require(provenance == expected["provenance"], "Approved imaging provenance counts mismatch")
    sizes = final.groupby("participant_id").size().value_counts()
    return final, {
        "rows": len(final),
        "participants": int(final.participant_id.nunique()),
        "columns": len(final.columns),
        "resolved_knees": len(final),
        "excluded_knees": 0,
        "silent_drops": 0,
        "duplicate_keys": 0,
        "missing_links": 0,
        "outcomes": outcome_counts(final),
        "provenance_counts": provenance,
        "left_right_distribution": final.knee_side_label.value_counts().to_dict(),
        "participant_knee_distribution": {
            "one_knee": int(sizes.get(1, 0)),
            "two_knees": int(sizes.get(2, 0)),
        },
        "image_audit": image_audit,
    }


def definitions(feature_config: dict) -> dict[str, dict]:
    primary = {
        f["output_column"]: f for f in feature_config["primary_features"] if f["kind"] == "tabular"
    }
    require(
        set(primary) == set(TABULAR_A),
        "Feature configuration is not the locked 14-column baseline panel",
    )
    require(
        all(f["baseline_only"] for f in primary.values()), "Post-baseline feature configuration"
    )
    optional = feature_config["optional_formulation_features"]
    require(
        len(optional) == 1
        and optional[0]["output_column"] == "baseline_kl"
        and optional[0]["baseline_only"]
        and optional[0]["default_predictor"] is False,
        "Baseline KL formulation configuration mismatch",
    )
    meanings = {
        "participant_id": "Internal participant identifier, mapped from NDA subjectkey; grouping key, never a predictor.",
        "source_participant_id": "Source-project participant identifier, mapped from src_subject_id; linkage only.",
        "knee_side_code": "Knee side: 1 RIGHT, 2 LEFT.",
        "knee_side_label": "Lowercase anatomical knee-side label.",
        "baseline_visit": "Locked baseline visit V00.",
        "followup_visit": "Planned endpoint visit V06, approximately 48 months; not evidence of a measured follow-up.",
        "composite_progression": "Locked primary label: KL increase >=1 OR qualifying replacement after baseline and on/before V06 boundary.",
        "radiographic_kl_progression": "Locked radiographic sensitivity label: project-15 V06 KL minus V00 KL >=1; unavailable without usable V06 KL.",
        "jsn_progression": "Locked secondary label: medial OR lateral project-15 JSN increase >=1 from V00 to V06; unavailable without eligible readings.",
        "replacement_before_v06": "Locked post-baseline qualifying replacement branch; outcome support, never a predictor.",
        "baseline_jsn_medial": "V00 medial OARSI JSN grade; secondary-outcome provenance only, not a predictor.",
        "baseline_jsn_lateral": "V00 lateral OARSI JSN grade; secondary-outcome provenance only, not a predictor.",
        "radiographic_analysis_eligible": "Locked radiographic KL analysis eligibility; reveals follow-up availability.",
        "jsn_analysis_eligible": "Locked JSN analysis eligibility; reveals follow-up availability.",
        "composite_analysis_eligible": "Locked composite endpoint eligibility; true for every retained knee.",
        "read_project": "Radiographic reading project 15 at baseline and follow-up; design metadata.",
        "descriptive_race": "Enrollee race category for description/subgroup evaluation only.",
        "descriptive_ethnicity": "Enrollee ethnicity category for description/subgroup evaluation only.",
        "design_site": "Enrollee study site for dataset/QC/design evaluation only.",
        "design_oai_cohort": "Enrollee OAI sampling subcohort for design evaluation only.",
        "effective_crop_relative_path": "Authoritative baseline pixel-loading reference relative to the frozen V3 root; path text is never encoded as a predictor.",
        "acquisition_index": "Frozen bilateral acquisition index; detailed linkage remains in the imaging manifest.",
        "panel_position": "Source screen panel: screen_left or screen_right; linkage only.",
        "anatomical_side": "Validated anatomical R/L side; not a disease predictor.",
        "laterality_provenance": "Frozen policy or explicit human exception mapping provenance; not a predictor.",
        "effective_crop_sha256": "SHA-256 of the effective uint16 NPY image artifact.",
        "effective_provenance": "AUTO_PASS, HUMAN_ACCEPT or HUMAN_OVERRIDE; QC provenance, never a predictor.",
        "target_spacing_mm": "Frozen resampled spacing in millimetres per pixel.",
        "crop_size_mm": "Frozen nominal square crop size in millimetres.",
        "crop_rows": "Frozen crop pixel row count.",
        "crop_columns": "Frozen crop pixel column count.",
        "crop_dtype": "Frozen crop pixel dtype uint16.",
    }
    for c in GROUPS["imaging_provenance_metadata"]:
        if c not in meanings:
            meanings[c] = (
                f"Authoritative imaging-manifest {c}: frozen implementation/version reproducibility metadata, never a predictor."
            )
    result = {}
    for group, columns in GROUPS.items():
        for c in columns:
            candidate = c in TABULAR_A or c == "baseline_kl"
            feature = primary.get(c, optional[0] if c == "baseline_kl" else None)
            description = meanings.get(c)
            if feature:
                description = f"{feature['display_name']}; {feature['units_or_coding']}."
            timing = (
                "V00"
                if candidate
                or c
                in (
                    *GROUPS["identifiers"],
                    "baseline_jsn_medial",
                    "baseline_jsn_lateral",
                    *GROUPS["non_predictor_metadata"],
                    "effective_crop_relative_path",
                )
                else (
                    "V00_to_V06_outcome_or_ascertainment"
                    if group in {"targets", "target_support_metadata"}
                    else "imaging_processing_adjudication_metadata"
                )
            )
            source = {
                "artifact": "analysis_cohort_v1.parquet"
                if c in COHORT_COLUMNS
                else "imaging_manifest.parquet",
                "evidence": "configs/features_v1.yaml; src/cohort/materialize.py"
                if feature
                else (
                    "docs/outcome-definition.md; src/cohort/builder.py"
                    if group in {"targets", "target_support_metadata"}
                    else "docs/feature-set-v1.md; src/imaging/final_adjudicated.py"
                ),
            }
            if feature:
                source.update(
                    {
                        "source_file": feature["source_file"],
                        "source_variables": feature["source_variables"],
                        "measurement_level": feature["measurement_level"],
                        "units_or_coding": feature["units_or_coding"],
                    }
                )
            result[c] = {
                "column_name": c,
                "group": group,
                "conceptual_description": description,
                "source": source,
                "domain": group,
                "timing": timing,
                "predictor_eligible": candidate,
                "default_predictor": c in TABULAR_A,
                "default_formulation_A_eligible": c in TABULAR_A,
                "formulation_B_eligible": candidate,
                "target": c in TARGETS,
                "leakage_sensitive": c not in TABULAR_A,
                "baseline_severity_feature": c == "baseline_kl",
                "input_kind": "image_loading_reference"
                if c == "effective_crop_relative_path"
                else ("tabular_predictor_candidate" if candidate else "non_predictor"),
            }
    return result


def feature_artifacts(frame: pd.DataFrame, feature_config: dict) -> tuple[dict, dict]:
    policy = definitions(feature_config)
    require(
        set(policy) == set(frame.columns),
        "Every dataset column must be classified; unclassified/leakage column",
    )
    schema = {
        "dataset_version": DATASET_VERSION,
        "columns": [
            {
                **policy[c],
                "dtype": str(frame[c].dtype),
                "missing_count": int(frame[c].isna().sum()),
                "missing_percentage": round(float(frame[c].isna().mean() * 100), 6),
            }
            for c in frame.columns
        ],
    }
    groups = {
        "dataset_version": DATASET_VERSION,
        "groups": {g: list(cols) for g, cols in GROUPS.items()},
        "formulations": {
            "A": {
                "tabular_predictors": list(TABULAR_A),
                "image_loading_references": list(GROUPS["imaging_input"]),
            },
            "B": {
                "tabular_predictors": [*TABULAR_A, "baseline_kl"],
                "image_loading_references": list(GROUPS["imaging_input"]),
            },
        },
        "image_reference_policy": "Load baseline pixels only. Paths/hashes/QC/laterality/adjudications are never encoded as disease predictors.",
        "default_formulation": "A",
        "split_group_column": "participant_id",
        "cohort_reconstruction_prohibited": True,
        "omitted_locked_cohort_columns": {
            c: "Linkage-only barcode or redundant legacy availability/formulation marker; preserved in the source cohort."
            for c in OMITTED
        },
    }
    validate_classification(frame, schema, groups, feature_config)
    return schema, groups


def validate_classification(
    frame: pd.DataFrame, schema: dict, groups: dict, feature_config: dict
) -> None:
    policy = definitions(feature_config)
    entries = schema["columns"]
    require(
        len(entries) == len(frame.columns)
        and {e["column_name"] for e in entries} == set(frame.columns),
        "Every column must be classified exactly once",
    )
    flattened = [c for columns in groups["groups"].values() for c in columns]
    require(
        len(flattened) == len(set(flattened)) == len(frame.columns)
        and set(flattened) == set(frame.columns),
        "Feature groups contain missing/duplicate/nonexistent columns",
    )
    require(
        groups["groups"] == {g: list(cols) for g, cols in GROUPS.items()},
        "Feature group classification changed",
    )
    require(
        groups.get("default_formulation") == "A"
        and groups.get("split_group_column") == "participant_id"
        and groups.get("cohort_reconstruction_prohibited") is True,
        "Default formulation/participant grouping policy changed",
    )
    for entry in entries:
        for field, expected in policy[entry["column_name"]].items():
            require(entry.get(field) == expected, "Leakage/formulation classification mismatch")
        require(entry["dtype"] == str(frame[entry["column_name"]].dtype), "Schema dtype drift")
        if entry["predictor_eligible"]:
            require(
                entry["timing"] == "V00" and not entry["target"],
                "Non-baseline/target predictor leakage",
            )
    require(
        groups["formulations"]["A"]["tabular_predictors"] == list(TABULAR_A)
        and groups["formulations"]["B"]["tabular_predictors"] == [*TABULAR_A, "baseline_kl"],
        "Formulation predictor leakage",
    )
    for formulation in ("A", "B"):
        require(
            groups["formulations"][formulation]["image_loading_references"]
            == list(GROUPS["imaging_input"]),
            "Imaging input/provenance leakage",
        )


def missingness_audit(frame: pd.DataFrame, schema: dict) -> dict:
    columns = {}
    for entry in schema["columns"]:
        c, missing = entry["column_name"], entry["missing_count"]
        outcome = c in ("radiographic_kl_progression", "jsn_progression")
        columns[c] = {
            "missing_count": missing,
            "missing_percentage": entry["missing_percentage"],
            "outcome_unavailable_by_study_design": missing if outcome else 0,
            "measurement_missing_reason_not_retained": missing if c in TABULAR_A else 0,
            "true_missing_measurement": None if missing and c in TABULAR_A else 0,
            "not_applicable": None if missing and c in TABULAR_A else 0,
            "structural_missingness": None if missing and c in TABULAR_A else 0,
            "interpretation": "Locked ineligible radiographic endpoint; never fill as non-event."
            if outcome
            else (
                "Blank, sentinel and failed numeric parse distinctions were collapsed by the locked materializer; exact reason is not recoverable here."
                if missing
                else "No missing values."
            ),
        }
    masks = {g: frame[list(cols)].notna().all(axis=1) for g, cols in GROUPS.items()}
    complete = {
        g: {
            "columns": len(GROUPS[g]),
            "complete_knees": int(mask.sum()),
            "incomplete_knees": int((~mask).sum()),
            "complete_percentage": round(float(mask.mean() * 100), 6),
            "participants_with_at_least_one_complete_knee": int(
                frame.loc[mask, "participant_id"].nunique()
            ),
        }
        for g, mask in masks.items()
    }
    modalities = ("imaging_input", "patient_reported", "demographic_clinical", "physical_function")
    patterns: dict[str, int] = {}
    for pattern in zip(*(masks[g] for g in modalities), strict=True):
        key = "|".join(
            f"{g}={'complete' if value else 'incomplete'}"
            for g, value in zip(modalities, pattern, strict=True)
        )
        patterns[key] = patterns.get(key, 0) + 1
    overlap = {a: {b: int((masks[a] & masks[b]).sum()) for b in modalities} for a in modalities}
    all_complete = pd.concat([masks[g] for g in modalities], axis=1).all(axis=1)
    ranges = {
        "womac_pain": (0, 20),
        "womac_stiffness": (0, 8),
        "womac_disability": (0, 68),
        "koos_pain": (0, 100),
        "koos_symptoms": (0, 100),
        "baseline_kl": (0, 3),
        "baseline_jsn_medial": (0, 3),
        "baseline_jsn_lateral": (0, 3),
        "prior_knee_surgery": (0, 1),
    }
    quality = {}
    for c in frame.columns:
        values = frame[c].dropna()
        if pd.api.types.is_numeric_dtype(frame[c].dtype) and not pd.api.types.is_bool_dtype(
            frame[c].dtype
        ):
            require(
                np.isfinite(values.to_numpy(dtype=float)).all(), "Non-finite numeric measurement"
            )
            quality[c] = {
                "observed": len(values),
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
            if c in ranges:
                low, high = ranges[c]
                violations = int((~values.between(low, high)).sum())
                quality[c].update({"documented_range": [low, high], "range_violations": violations})
                require(
                    violations == 0,
                    "Locked measurement outside documented range; do not silently repair",
                )
        elif c in (
            "sex",
            "family_knee_replacement_history",
            "descriptive_race",
            "descriptive_ethnicity",
            "design_site",
            "design_oai_cohort",
        ):
            quality[c] = {
                "observed_categories": {str(k): int(v) for k, v in values.value_counts().items()}
            }
    return {
        "rows": len(frame),
        "columns": columns,
        "feature_group_complete_cases": complete,
        "modality_order": list(modalities),
        "modality_overlap": overlap,
        "modality_complete_patterns": patterns,
        "all_four_modalities_complete_knees": int(all_complete.sum()),
        "all_four_modalities_complete_participants": int(
            frame.loc[all_complete, "participant_id"].nunique()
        ),
        "outcomes": outcome_counts(frame),
        "data_quality": quality,
        "missing_reason_limitations": "Baseline measurement missingness cannot be divided reliably into true missing/not-applicable/structural categories from the locked cohort. Null category counts mean unknown, not zero. Endpoint availability is known from locked eligibility flags.",
        "imputation_performed": False,
        "normalization_scaling_or_feature_selection_fitted": False,
        "complete_case_filtering_performed": False,
    }


def preservation_snapshot(baseline: dict, root: Path) -> dict:
    protected = {p: sha256_file(Path(p)) for p in baseline["protected_files"]}
    require(
        protected == baseline["protected_files"],
        "Protected source/adjudication/configuration SHA changed",
    )
    automatic = verify_automatic_tree(root)
    overrides = hash_tree(root / "overrides", relative_to=root, workers=8)
    raw = pd.read_parquet(DEFAULT_ARCHIVE_LEDGER)

    def archive_entry(row):
        p = DEFAULT_ARCHIVE_ROOT / row.archive_relative_path
        require(
            p.is_file() and p.stat().st_size == row.archive_bytes, "Raw archive missing/changed"
        )
        value = sha256_file(p)
        require(value == row.archive_sha256, "Raw archive SHA changed")
        return {"artifact": str(p), "bytes": p.stat().st_size, "sha256": value}

    with ThreadPoolExecutor(max_workers=8) as pool:
        archives = pd.DataFrame(pool.map(archive_entry, raw.itertuples(index=False)))
    require(
        set(DEFAULT_ARCHIVE_ROOT.rglob("*.tar.gz"))
        == {DEFAULT_ARCHIVE_ROOT / str(p) for p in raw.archive_relative_path},
        "Raw archive set changed",
    )
    tabular = hash_tree(Path("data/raw/oai"), relative_to=Path("."), workers=8)
    final = hash_tree(root / "final_adjudicated_v1", relative_to=root, workers=8)
    session = ReviewSession(
        queue_path=root / "review_queue.parquet",
        adjudication_path=root / "review/adjudications.jsonl",
        output_root=root,
    )
    latest = session.latest_decisions()
    snapshot = {
        "protected_files": protected,
        "review_progress": session.progress(),
        "history_sha256": digest(session.adjudications()),
        "laterality_state_sha256": digest([latest[i] for i in session.laterality_exceptions]),
    }
    for prefix, frame in (
        ("automatic", automatic),
        ("override", overrides),
        ("raw_archive", archives),
        ("raw_tabular", tabular),
        ("final_imaging", final),
    ):
        snapshot[f"{prefix}_aggregate_sha256"] = _aggregate(frame)
        count_key = "raw_archives" if prefix == "raw_archive" else f"{prefix}_files"
        snapshot[count_key] = len(frame)
        if prefix != "final_imaging":
            snapshot[f"{prefix}_bytes"] = int(frame.bytes.sum())
    require(snapshot == baseline, "Protected artifact preservation fingerprint mismatch")
    return snapshot


def publish_json(value: dict, path: Path) -> None:
    if path.exists():
        require(read_json(path) == value, "Existing artifact differs; refusing silent overwrite")
    else:
        atomic_json(value, path)


def publish_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
    else:
        atomic_parquet(frame, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)


def freeze_dataset(
    *,
    cohort_path: Path = DEFAULT_COHORT,
    imaging_path: Path = DEFAULT_IMAGING,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    directory: Path = DEFAULT_DIRECTORY,
    expected: dict | None = EXPECTED,
    approved_imaging_sha: str | None = APPROVED_IMAGING_SHA,
    preservation: dict | None = None,
) -> dict:
    cohort_path, imaging_path, image_root, directory = map(
        Path, (cohort_path, imaging_path, image_root, directory)
    )
    inputs = {str(p): sha256_file(p) for p in (cohort_path, imaging_path)}
    if approved_imaging_sha is not None:
        require(
            inputs[str(imaging_path)] == approved_imaging_sha,
            "Authoritative imaging manifest SHA mismatch",
        )
    if preservation is not None:
        print("Verifying protected pre-change fingerprints...", flush=True)
        preservation_snapshot(preservation, image_root)
    study, features = load_config("configs/study_v1.yaml"), load_config("configs/features_v1.yaml")
    _validate_configs(study, features)
    cohort, imaging = pd.read_parquet(cohort_path), pd.read_parquet(imaging_path)
    print("Joining locked knees and verifying every effective image...", flush=True)
    frame, audit = integrate(cohort, imaging, image_root, expected=expected)
    schema, groups = feature_artifacts(frame, features)
    missingness = missingness_audit(frame, schema)
    if preservation is not None:
        print("Rechecking adjudications, images, frozen sources and raw data...", flush=True)
        after = preservation_snapshot(preservation, image_root)
    else:
        after = None
    require(
        {p: sha256_file(p) for p in inputs} == inputs, "Source cohort/manifest changed during build"
    )
    audit.update(
        {
            "dataset_version": DATASET_VERSION,
            "source_sha256": inputs,
            "source_rows_and_order_preserved": True,
            "all_columns_classified": True,
            "leakage_audit_passed": True,
            "default_tabular_predictor_count": len(TABULAR_A),
            "formulation_B_tabular_predictor_count": len(TABULAR_A) + 1,
            "baseline_KL_default_predictor": False,
            "images_loaded_as_pixels_only": True,
            "preservation": after,
            "integrity_passed": True,
            "split_created": False,
            "preprocessing_fitted": False,
            "models_trained": False,
        }
    )
    path = directory / ARTIFACTS[0]
    publish_parquet(frame, path)
    audit["dataset_sha256"] = sha256_file(path)
    for value, name in (
        (schema, "schema.json"),
        (groups, "feature_groups.json"),
        (missingness, "missingness_audit.json"),
        (audit, "dataset_integrity_audit.json"),
    ):
        publish_json(value, directory / name)
    code_paths = (
        Path("src/multimodal/freeze.py"),
        Path("src/multimodal/__init__.py"),
        Path("src/cohort/builder.py"),
        Path("src/cohort/materialize.py"),
        Path("src/cohort/feasibility.py"),
        Path("src/imaging/artifact_io.py"),
        Path("src/imaging/final_adjudicated.py"),
        Path("configs/study_v1.yaml"),
        Path("configs/features_v1.yaml"),
    )
    marker = directory / "dataset_freeze.json"
    existing = read_json(marker) if marker.exists() else None
    frozen = {
        "dataset_version": DATASET_VERSION,
        "status": "FROZEN_AUTHORITATIVE",
        "created_at_utc": existing["created_at_utc"] if existing else datetime.now(UTC).isoformat(),
        "dataset_sha256": audit["dataset_sha256"],
        "cohort_source_sha256": inputs[str(cohort_path)],
        "imaging_manifest_sha256": inputs[str(imaging_path)],
        "source_artifacts": {
            "cohort": str(cohort_path),
            "imaging_manifest": str(imaging_path),
            "image_root": str(image_root),
        },
        "rows": len(frame),
        "participants": audit["participants"],
        "columns": len(frame.columns),
        "feature_group_definition_sha256": sha256_file(directory / "feature_groups.json"),
        "outcome_counts": audit["outcomes"],
        "provenance_counts": audit["provenance_counts"],
        "artifacts_sha256": {name: sha256_file(directory / name) for name in ARTIFACTS},
        "code_configuration_sha256": {str(p): sha256_file(p) for p in code_paths},
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("pandas", "numpy", "pyarrow")},
        },
        "future_split_group_column": "participant_id",
        "both_knees_share_partition": True,
        "sole_input_to_future_split_generation": True,
        "cohort_reconstruction_prohibited": True,
        "default_formulation": "A",
        "split_created": False,
        "preprocessing_fitted": False,
        "models_trained": False,
    }
    publish_json(frozen, marker)
    load_frozen_dataset(directory)
    return frozen


def load_frozen_dataset(directory: Path = DEFAULT_DIRECTORY) -> pd.DataFrame:
    """Verify the freeze, then read the sole future splitting input; never rebuild the cohort."""
    directory = Path(directory)
    frozen = read_json(directory / "dataset_freeze.json")
    require(
        frozen["status"] == "FROZEN_AUTHORITATIVE" and frozen["dataset_version"] == DATASET_VERSION,
        "Dataset is not authoritatively frozen",
    )
    require(
        frozen.get("future_split_group_column") == "participant_id"
        and frozen.get("both_knees_share_partition") is True
        and frozen.get("cohort_reconstruction_prohibited") is True
        and frozen.get("sole_input_to_future_split_generation") is True
        and frozen.get("default_formulation") == "A"
        and frozen.get("split_created") is False
        and frozen.get("preprocessing_fitted") is False
        and frozen.get("models_trained") is False,
        "Frozen future-input/leakage policy changed",
    )
    require(set(frozen["artifacts_sha256"]) == set(ARTIFACTS), "Incomplete frozen artifact set")
    for name, expected_sha in frozen["artifacts_sha256"].items():
        require(
            sha256_file(directory / name) == expected_sha, "Frozen dataset/artifact SHA mismatch"
        )
    require(
        frozen["dataset_sha256"] == frozen["artifacts_sha256"][ARTIFACTS[0]]
        and frozen["feature_group_definition_sha256"]
        == frozen["artifacts_sha256"]["feature_groups.json"],
        "Inconsistent frozen dataset/feature-group fingerprints",
    )
    require(
        sha256_file(Path("configs/features_v1.yaml"))
        == frozen["code_configuration_sha256"]["configs/features_v1.yaml"],
        "Frozen feature configuration changed",
    )
    frame = pd.read_parquet(directory / ARTIFACTS[0])
    require(
        len(frame) == frozen["rows"]
        and frame.participant_id.nunique() == frozen["participants"]
        and len(frame.columns) == frozen["columns"],
        "Frozen dataset dimensions mismatch",
    )
    require(
        not frame.duplicated(["participant_id", "knee_side_code"]).any(), "Frozen duplicate knee"
    )
    require(outcome_counts(frame) == frozen["outcome_counts"], "Frozen outcome counts mismatch")
    require(
        {p: int(frame.effective_provenance.eq(p).sum()) for p in ALLOWED_PROVENANCE}
        == frozen["provenance_counts"],
        "Frozen image provenance counts mismatch",
    )
    validate_classification(
        frame,
        read_json(directory / "schema.json"),
        read_json(directory / "feature_groups.json"),
        load_config("configs/features_v1.yaml"),
    )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args(argv)
    baseline = read_json(args.output_dir / "pre_change_preservation.json")
    result = freeze_dataset(directory=args.output_dir, preservation=baseline)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "code_configuration_sha256",
                    "runtime",
                    "artifacts_sha256",
                    "source_artifacts",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
