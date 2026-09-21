"""Final multimodal integration/freeze tests: synthetic cohorts and pixel files only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_cohort_builder import _synthetic_configs, _synthetic_package

import multimodal.freeze as module
from cohort.materialize import build_analysis_frames
from imaging.artifact_io import atomic_json, atomic_parquet, hash_tree, sha256_file
from multimodal.freeze import DatasetIntegrityError

pytestmark = pytest.mark.public_portable


@pytest.fixture
def synthetic(tmp_path):
    study, features = _synthetic_configs()
    cohort, _, _ = build_analysis_frames(_synthetic_package(tmp_path), study, features)
    root = tmp_path / "images"
    template = np.zeros((1067, 1067), dtype=np.uint16)
    rows = []
    participants = {p: i for i, p in enumerate(cohort.participant_id.unique(), start=1)}
    for i, c in enumerate(cohort.itertuples(index=False)):
        acquisition = participants[c.participant_id]
        panel = "screen_left" if c.knee_side_code == "1" else "screen_right"
        provenance = module.ALLOWED_PROVENANCE[i % 3]
        prefix = "overrides" if provenance == "HUMAN_OVERRIDE" else "crops"
        reference = f"{prefix}/acquisition_{acquisition:04d}/{panel}_uint16.npy"
        path = root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        template[0, 0] = i
        np.save(path, template, allow_pickle=False)
        r = {
            "participant_id": c.participant_id,
            "source_participant_id": c.source_participant_id,
            "knee_side_code": c.knee_side_code,
            "knee_side_label": c.knee_side_label,
            "baseline_visit": "V00",
            "acquisition_index": acquisition,
            "panel_position": panel,
            "anatomical_side": "R" if c.knee_side_code == "1" else "L",
            "laterality_provenance": "FROZEN_LATERALITY_V2",
            "effective_provenance": provenance,
            "effective_crop_relative_path": reference,
            "effective_crop_sha256": sha256_file(path),
            "imaging_included": True,
            "inclusion_status": "INCLUDED",
            "exclusion_reason": None,
            "target_spacing_mm": 0.15,
            "crop_size_mm": 160.0,
            "crop_rows": 1067,
            "crop_columns": 1067,
            "crop_dtype": "uint16",
        }
        for column in module.GROUPS["imaging_provenance_metadata"]:
            if column not in r:
                r[column] = "a" * 64 if column.endswith("sha256") else "synthetic_frozen_v1"
        rows.append(r)
    imaging = pd.DataFrame(rows)
    cohort_path, imaging_path = tmp_path / "cohort.parquet", tmp_path / "imaging.parquet"
    atomic_parquet(cohort, cohort_path)
    atomic_parquet(imaging, imaging_path)
    return {
        "cohort": cohort,
        "imaging": imaging,
        "cohort_path": cohort_path,
        "imaging_path": imaging_path,
        "root": root,
        "directory": tmp_path / "final_v1",
        "features": features,
    }


def integrate(state):
    return module.integrate(state["cohort"], state["imaging"], state["root"], expected=None)


def freeze(state):
    return module.freeze_dataset(
        cohort_path=state["cohort_path"],
        imaging_path=state["imaging_path"],
        image_root=state["root"],
        directory=state["directory"],
        expected=None,
        approved_imaging_sha=sha256_file(state["imaging_path"]),
    )


def test_one_row_per_knee_and_source_values_order_preserved(synthetic):
    final, audit = integrate(synthetic)
    assert len(final) == 6 and final.participant_id.nunique() == 5
    assert len(final.columns) == 54
    assert not final.duplicated(["participant_id", "knee_side_code"]).any()
    pd.testing.assert_frame_equal(
        final[list(module.COHORT_COLUMNS)], synthetic["cohort"][list(module.COHORT_COLUMNS)]
    )
    assert audit["silent_drops"] == audit["excluded_knees"] == 0
    assert final.radiographic_kl_progression.isna().sum() == 1


def test_exact_6961_row_3621_participant_synthetic_reconciliation(synthetic):
    base = synthetic["cohort"].iloc[[0]]
    pairs = [(p, code) for p in range(1, 3622) for code in (("1", "2") if p <= 3340 else ("2",))]
    cohort = pd.concat([base] * 6961, ignore_index=True)
    cohort["participant_id"] = pd.array([f"SYNTH_P{p}" for p, c in pairs], dtype="string")
    cohort["source_participant_id"] = pd.array([f"SYNTH_S{p}" for p, c in pairs], dtype="string")
    cohort["knee_side_code"] = pd.array([c for p, c in pairs], dtype="string")
    cohort["knee_side_label"] = cohort.knee_side_code.map({"1": "right", "2": "left"})
    image = pd.concat([synthetic["imaging"].iloc[[0]]] * 6961, ignore_index=True)
    template = synthetic["root"] / synthetic["imaging"].iloc[0].effective_crop_relative_path
    for column in ("participant_id", "source_participant_id", "knee_side_code", "knee_side_label"):
        image[column] = cohort[column]
    references = []
    for p, code in pairs:
        position = "screen_left" if code == "1" else "screen_right"
        ref = f"crops/acquisition_{p + 10000:04d}/{position}_uint16.npy"
        dest = synthetic["root"] / ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.hardlink_to(template)
        references.append(ref)
    image["acquisition_index"] = [p + 10000 for p, c in pairs]
    image["panel_position"] = ["screen_left" if c == "1" else "screen_right" for p, c in pairs]
    image["anatomical_side"] = ["R" if c == "1" else "L" for p, c in pairs]
    image["effective_crop_relative_path"] = references
    final, audit = module.integrate(cohort, image, synthetic["root"], expected=None)
    assert (
        audit["rows"] == audit["resolved_knees"] == audit["image_audit"]["images_checked"] == 6961
    )
    assert audit["participants"] == 3621
    assert audit["participant_knee_distribution"] == {"one_knee": 281, "two_knees": 3340}
    assert final[list(module.KEYS)].to_dict("records") == cohort[list(module.KEYS)].to_dict(
        "records"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_cohort",
        "duplicate_imaging",
        "missing_link",
        "wrong_side",
        "wrong_label",
        "wrong_participant",
        "wrong_source_participant",
        "wrong_visit",
        "excluded",
        "missing_image",
        "sha",
        "wrong_panel",
        "wrong_acquisition",
    ],
)
def test_linkage_and_artifact_mutations_fail_closed(synthetic, mutation):
    c, i = synthetic["cohort"], synthetic["imaging"]
    if mutation == "duplicate_cohort":
        synthetic["cohort"] = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    elif mutation == "duplicate_imaging":
        synthetic["imaging"] = pd.concat([i, i.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_link":
        synthetic["imaging"] = i.iloc[1:]
    elif mutation == "wrong_side":
        i.loc[0, "anatomical_side"] = "L"
    elif mutation == "wrong_label":
        i.loc[0, "knee_side_label"] = "left"
    elif mutation == "wrong_participant":
        i.loc[0, "participant_id"] = "WRONG"
    elif mutation == "wrong_source_participant":
        i.loc[0, "source_participant_id"] = "WRONG"
    elif mutation == "wrong_visit":
        i.loc[0, "baseline_visit"] = "V06"
    elif mutation == "excluded":
        i.loc[0, "effective_provenance"] = "UNRESOLVED"
    elif mutation == "missing_image":
        (synthetic["root"] / i.loc[0, "effective_crop_relative_path"]).unlink()
    elif mutation == "sha":
        i.loc[0, "effective_crop_sha256"] = "0" * 64
    elif mutation == "wrong_panel":
        i.loc[0, "panel_position"] = "screen_right"
    elif mutation == "wrong_acquisition":
        i.loc[0, "acquisition_index"] = 999
    with pytest.raises(DatasetIntegrityError):
        integrate(synthetic)


def test_leakage_groups_followup_targets_and_qc_are_never_predictors(synthetic):
    frame, _ = integrate(synthetic)
    schema, groups = module.feature_artifacts(frame, synthetic["features"])
    definitions = {e["column_name"]: e for e in schema["columns"]}
    assert len(definitions) == len(frame.columns) == 54
    for group in (
        "identifiers",
        "targets",
        "target_support_metadata",
        "imaging_provenance_metadata",
        "non_predictor_metadata",
        "imaging_input",
    ):
        for c in groups["groups"][group]:
            e = definitions[c]
            assert (
                not e["predictor_eligible"]
                and not e["default_formulation_A_eligible"]
                and not e["formulation_B_eligible"]
            )
    assert definitions["followup_visit"]["leakage_sensitive"]
    assert definitions["replacement_before_v06"]["leakage_sensitive"]
    assert groups["formulations"]["A"]["image_loading_references"] == [
        "effective_crop_relative_path"
    ]
    assert all(definitions[c]["timing"] == "V00" for c in module.TABULAR_A)


def test_baseline_KL_is_B_only_retained_severity(synthetic):
    frame, _ = integrate(synthetic)
    schema, groups = module.feature_artifacts(frame, synthetic["features"])
    e = next(e for e in schema["columns"] if e["column_name"] == "baseline_kl")
    assert e["baseline_severity_feature"] and e["predictor_eligible"]
    assert not e["default_predictor"] and not e["default_formulation_A_eligible"]
    assert e["formulation_B_eligible"]
    assert "baseline_kl" not in groups["formulations"]["A"]["tabular_predictors"]
    assert groups["formulations"]["B"]["tabular_predictors"] == [*module.TABULAR_A, "baseline_kl"]


@pytest.mark.parametrize(
    "mutation",
    [
        "target_as_predictor",
        "qc_as_predictor",
        "KL_in_A",
        "nonexistent_group_column",
        "unclassified",
        "new_followup_column",
    ],
)
def test_feature_classification_tampering_fails(synthetic, mutation):
    frame, _ = integrate(synthetic)
    schema, groups = module.feature_artifacts(frame, synthetic["features"])
    if mutation in {"target_as_predictor", "qc_as_predictor"}:
        c = "composite_progression" if mutation == "target_as_predictor" else "effective_provenance"
        next(e for e in schema["columns"] if e["column_name"] == c)["predictor_eligible"] = True
    elif mutation == "KL_in_A":
        groups["formulations"]["A"]["tabular_predictors"].append("baseline_kl")
    elif mutation == "nonexistent_group_column":
        groups["groups"]["patient_reported"].append("not_a_column")
    elif mutation == "unclassified":
        schema["columns"].pop()
    elif mutation == "new_followup_column":
        frame["v06_KL"] = 3
    with pytest.raises(DatasetIntegrityError):
        module.validate_classification(frame, schema, groups, synthetic["features"])


def test_unknown_followup_cohort_column_is_not_silently_discarded(synthetic):
    synthetic["cohort"]["v06_kl"] = 3
    with pytest.raises(DatasetIntegrityError, match="unclassified"):
        integrate(synthetic)


def test_missingness_keeps_design_unavailability_and_unknown_measurement_reason(synthetic):
    frame, _ = integrate(synthetic)
    schema, _ = module.feature_artifacts(frame, synthetic["features"])
    audit = module.missingness_audit(frame, schema)
    assert (
        audit["columns"]["radiographic_kl_progression"]["outcome_unavailable_by_study_design"] == 1
    )
    assert (
        audit["columns"]["knee_extension_strength_n"]["measurement_missing_reason_not_retained"]
        == 1
    )
    assert audit["columns"]["knee_extension_strength_n"]["not_applicable"] is None
    assert sum(audit["modality_complete_patterns"].values()) == len(frame)
    assert not audit["imputation_performed"] and not audit["complete_case_filtering_performed"]


def test_outcome_count_mismatch_stops_instead_of_redefining(synthetic):
    expected = {
        "rows": 6,
        "participants": 5,
        "outcomes": module.outcome_counts(synthetic["cohort"]),
        "provenance": {p: 2 for p in module.ALLOWED_PROVENANCE},
    }
    expected["outcomes"]["composite_progression"]["positive"] = 99
    with pytest.raises(DatasetIntegrityError, match="outcome counts mismatch"):
        module.integrate(
            synthetic["cohort"], synthetic["imaging"], synthetic["root"], expected=expected
        )


def test_freeze_is_idempotent_roundtrip_stable_and_preserves_sources_images(synthetic):
    before = {p: sha256_file(p) for p in (synthetic["cohort_path"], synthetic["imaging_path"])}
    images = hash_tree(synthetic["root"], relative_to=synthetic["root"])
    frozen = freeze(synthetic)
    artifacts = hash_tree(synthetic["directory"], relative_to=synthetic["directory"])
    assert freeze(synthetic) == frozen
    pd.testing.assert_frame_equal(
        artifacts, hash_tree(synthetic["directory"], relative_to=synthetic["directory"])
    )
    pd.testing.assert_frame_equal(
        images, hash_tree(synthetic["root"], relative_to=synthetic["root"])
    )
    assert {p: sha256_file(p) for p in before} == before
    frame = module.load_frozen_dataset(synthetic["directory"])
    expected, _ = integrate(synthetic)
    pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    assert frozen["sole_input_to_future_split_generation"] and frozen["both_knees_share_partition"]
    assert not frozen["split_created"] and not frozen["preprocessing_fitted"]


def test_frozen_artifact_sha_mutation_fails_loading(synthetic):
    freeze(synthetic)
    path = synthetic["directory"] / "feature_groups.json"
    data = module.read_json(path)
    data["formulations"]["A"]["tabular_predictors"].append("replacement_before_v06")
    atomic_json(data, path)
    with pytest.raises(DatasetIntegrityError, match="SHA mismatch"):
        module.load_frozen_dataset(synthetic["directory"])


def test_frozen_build_refuses_changed_data_without_overwrite(synthetic):
    freeze(synthetic)
    frozen_before = hash_tree(synthetic["directory"], relative_to=synthetic["directory"])
    cohort = pd.read_parquet(synthetic["cohort_path"])
    cohort.loc[0, "age_years"] += 1
    atomic_parquet(cohort, synthetic["cohort_path"])
    with pytest.raises((AssertionError, DatasetIntegrityError)):
        freeze(synthetic)
    pd.testing.assert_frame_equal(
        frozen_before, hash_tree(synthetic["directory"], relative_to=synthetic["directory"])
    )
