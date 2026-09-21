"""Authoritative participant split tests: synthetic tables/files only, no production writes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction

import pandas as pd
import pytest
from test_multimodal_freeze import freeze, synthetic  # noqa: F401

from imaging.artifact_io import atomic_json, sha256_file
from multimodal.freeze import DatasetIntegrityError, load_config, load_frozen_dataset
from multimodal.splits import (
    ARTIFACTS,
    DEFAULT_CONFIG,
    SPLITS,
    balance_audit,
    build_split,
    generate,
    load_frozen_split,
    objective,
    participant_strata,
    source_files,
    validate_manifests,
)

pytestmark = pytest.mark.public_portable


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG)


def make_frame(counts=None):
    """Sufficiently large, independently constructed synthetic primary-burden strata."""
    counts = counts or {(1, 0): 178, (1, 1): 103, (2, 0): 2538, (2, 1): 636, (2, 2): 166}
    rows = []
    participant = 0
    for (knees, events), n in counts.items():
        for _ in range(n):
            participant += 1
            for k in range(knees):
                code = str(k + 1)
                rows.append(
                    {
                        "participant_id": f"SYNTH_P{participant:05d}",
                        "knee_side_code": code,
                        "knee_side_label": "right" if code == "1" else "left",
                        "anatomical_side": "R" if code == "1" else "L",
                        "baseline_visit": "V00",
                        "composite_progression": k < events,
                        "radiographic_kl_progression": k < events,
                        "jsn_progression": k < events,
                        "replacement_before_v06": False,
                        "baseline_kl": participant % 4,
                        "effective_provenance": "AUTO_PASS",
                        "effective_crop_relative_path": f"synthetic/{participant}_{k}.npy",
                        "effective_crop_sha256": "a" * 64,
                        "age_years": 60.0,
                        "sex": "1",
                        "bmi": 25.0,
                        "prior_knee_surgery": 0.0,
                        "family_knee_replacement_history": "0",
                        "womac_pain": 0.0,
                        "womac_stiffness": 0.0,
                        "womac_disability": 0.0,
                        "koos_pain": 90.0,
                        "koos_symptoms": 90.0,
                        "walk_20m_pace_mps": 1.1,
                        "chair_stand_time_seconds": 10.0,
                        "knee_extension_strength_n": 100.0,
                        "knee_flexion_strength_n": 70.0,
                    }
                )
    frame = pd.DataFrame(rows)
    for c in (
        "participant_id",
        "knee_side_code",
        "knee_side_label",
        "anatomical_side",
        "baseline_visit",
    ):
        frame[c] = frame[c].astype("string")
    for c in (
        "composite_progression",
        "radiographic_kl_progression",
        "jsn_progression",
        "replacement_before_v06",
    ):
        frame[c] = frame[c].astype("boolean")
    return frame


@pytest.fixture
def small():
    return make_frame({(1, 0): 15, (1, 1): 10, (2, 0): 30, (2, 1): 15, (2, 2): 10})


def test_exact_6961_knee_3621_participant_reconciliation_and_balance(config):
    frame = make_frame()
    p, k, search = generate(frame, config)
    audit = balance_audit(frame, p, k, config)
    assert len(frame) == len(k) == 6961 and len(p) == 3621
    assert (
        not p.participant_id.duplicated().any()
        and not k.duplicated(["participant_id", "knee_side_code"]).any()
    )
    assert k.groupby("participant_id").split.nunique().eq(1).all()
    assert audit["integrity"]["participant_pairwise_overlap"] == {
        "TRAIN_VALIDATION": 0,
        "TRAIN_TEST": 0,
        "VALIDATION_TEST": 0,
    }
    assert search["participant_quotas"] == {"TRAIN": 2535, "VALIDATION": 543, "TEST": 543}
    assert search["candidate_count"] == 256 and not search["stratification"]["collapsed"]
    assert search["feasible_allocations_searched"] == search["feasible_allocation_count"]
    assert not audit["material_imbalance_flags"]
    for split, target in zip(SPLITS, (0.7, 0.15, 0.15), strict=True):
        s = audit["splits"][split]
        assert abs(s["participants"] / 3621 - target) <= 1 / 3621
        assert abs(s["knees"] / 6961 - target) < 0.005
        assert abs(s["primary_endpoint"]["prevalence"] - 1071 / 6961) < 0.02
        assert sum(s["participant_event_burden"].values()) == s["participants"]
        assert sum(s["participant_knee_contribution"].values()) == s["participants"]
    assert sum(s["primary_endpoint"]["positive"] for s in audit["splits"].values()) == 1071
    assert audit["integrity"]["knees_assigned_once"] == 6961


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_participant",
        "extra_participant",
        "duplicate_participant",
        "missing_knee",
        "extra_knee",
        "duplicate_knee",
        "wrong_side",
        "wrong_anatomical_side",
        "wrong_participant",
        "wrong_split",
        "row_order",
        "wrong_burden",
        "wrong_stratum",
        "wrong_visit",
    ],
)
def test_manifest_mutations_fail_closed(small, config, mutation):
    p, k, _ = generate(small, config)
    if mutation == "missing_participant":
        p = p.iloc[1:]
    elif mutation == "extra_participant":
        extra = p.iloc[[0]].copy()
        extra["participant_id"] = "EXTRA"
        p = pd.concat([p, extra], ignore_index=True)
    elif mutation == "duplicate_participant":
        p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_knee":
        k = k.iloc[1:]
    elif mutation == "extra_knee":
        extra = k.iloc[[0]].copy()
        extra["participant_id"] = "EXTRA"
        k = pd.concat([k, extra], ignore_index=True)
    elif mutation == "duplicate_knee":
        k = pd.concat([k, k.iloc[[0]]], ignore_index=True)
    elif mutation == "wrong_side":
        k.loc[0, "knee_side_code"] = "2"
    elif mutation == "wrong_anatomical_side":
        k.loc[0, "anatomical_side"] = "L"
    elif mutation == "wrong_participant":
        k.loc[0, "participant_id"] = "OTHER"
    elif mutation == "wrong_split":
        k.loc[0, "split"] = next(s for s in SPLITS if s != k.loc[0, "split"])
    elif mutation == "row_order":
        k = k.iloc[::-1].reset_index(drop=True)
    elif mutation == "wrong_burden":
        p.loc[0, "primary_event_knee_count"] = 2
    elif mutation == "wrong_stratum":
        p.loc[0, "stratification_group"] = "OTHER"
    elif mutation == "wrong_visit":
        k.loc[0, "baseline_visit"] = "V06"
    with pytest.raises((DatasetIntegrityError, AssertionError)):
        validate_manifests(small, p, k, config)


def test_sparse_strata_are_collapsed_without_predictors(config):
    frame = make_frame({(1, 0): 2, (1, 1): 1, (2, 0): 30, (2, 1): 30, (2, 2): 1})
    p, info = participant_strata(frame, 20)
    assert info["collapsed"] and info["selected_stage"] == "any_event"
    assert info["selected_strata"] == {"ANY0": 32, "ANY1": 32}
    assert len(p) == 64 and p.primary_event_knee_count.sum() == 33
    tiny = make_frame({(1, 0): 4, (2, 2): 1})
    _, collapsed = participant_strata(tiny, 20)
    assert collapsed["selected_stage"] == "all_participants"
    a, b, _ = generate(tiny, config)
    assert len(a) == 5 and len(b) == 6 and set(a.split) == set(SPLITS)


def test_deterministic_regeneration_preserves_source_and_ignores_descriptors(small, config):
    original = small.copy(deep=True)
    p, k, search = generate(small, config)
    changed = small.copy(deep=True)
    for c in ("baseline_kl", "age_years", "bmi", "radiographic_kl_progression", "jsn_progression"):
        changed[c] = pd.NA
    changed["effective_provenance"] = "HUMAN_OVERRIDE"
    changed["womac_pain"] = pd.NA
    p2, k2, search2 = generate(changed, config)
    pd.testing.assert_frame_equal(p, p2, check_exact=True)
    pd.testing.assert_frame_equal(k, k2, check_exact=True)
    assert search == search2
    pd.testing.assert_frame_equal(small, original, check_exact=True)
    shuffled = small.sample(frac=1, random_state=19).reset_index(drop=True)
    p3, _, _ = generate(shuffled, config)
    pd.testing.assert_frame_equal(p, p3, check_exact=True)


def test_search_configuration_cannot_admit_predictors_or_secondary_outcomes(small, config):
    config["selection_columns"].append("baseline_kl")
    with pytest.raises(DatasetIntegrityError, match="must not enter"):
        generate(small, config)


def test_incomplete_cases_retained_shared_A_B_split_and_audit_only_missingness(small, config):
    small.loc[:5, "bmi"] = pd.NA
    small.loc[6:9, "knee_extension_strength_n"] = pd.NA
    p, k, _ = generate(small, config)
    audit = balance_audit(small, p, k, config)
    assert len(k) == len(small)
    assert (
        sum(s["modality_availability"]["clinical_complete"] for s in audit["splits"].values())
        == len(small) - 6
    )
    assert (
        sum(
            s["modality_availability"]["physical_function_complete"]
            for s in audit["splits"].values()
        )
        == len(small) - 4
    )
    assert sum(
        s["secondary_outcomes_descriptive_only"]["radiographic_kl_progression"]["observed"]
        for s in audit["splits"].values()
    ) == len(small)
    assert "baseline_kl" not in config["selection_columns"]
    assert "formulation" not in p.columns and "formulation" not in k.columns


@pytest.fixture
def frozen_source(synthetic):  # noqa: F811
    freeze(synthetic)
    source = synthetic["directory"]
    output = source / "splits/v1"
    return source, output, sha256_file(source / "final_multimodal_dataset.parquet")


def run_build(state, **kwargs):
    source, output, sha = state
    return build_split(
        source=source,
        output=output,
        approved_sha=sha,
        expected=kwargs.pop("expected", None),
        **kwargs,
    )


def test_publish_rerun_hash_timestamp_dtype_stability_and_source_unchanged(frozen_source):
    source, output, sha = frozen_source
    before = source_files(source)
    marker = run_build(frozen_source)
    first = {p.name: (sha256_file(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert run_build(frozen_source) == marker
    assert first == {p.name: (sha256_file(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert before == source_files(source)
    p, k = load_frozen_split(output, source=source, approved_sha=sha, expected=None)
    frame = load_frozen_dataset(source)
    summary = json.loads((output / "split_summary.json").read_text())
    assert {"formulation_A", "formulation_B", "tabular", "image_only", "multimodal"}.issubset(
        summary["shared_by"]
    )
    assert len(k) == len(frame) and len(p) == frame.participant_id.nunique()
    for col in ("participant_id", "knee_side_code", "baseline_visit"):
        assert k[col].dtype == frame[col].dtype
    assert set(ARTIFACTS).issubset(first)


def test_changed_source_SHA_or_dimensions_fail_before_output(frozen_source):
    source, output, _ = frozen_source
    with pytest.raises(DatasetIntegrityError, match="Changed frozen dataset SHA"):
        build_split(source=source, output=output, approved_sha="0" * 64, expected=None)
    assert not output.exists()
    with pytest.raises(DatasetIntegrityError, match="row/participant/column count"):
        run_build(frozen_source, expected=(6961, 3621, 54))
    assert not output.exists()


def test_mutated_frozen_manifest_fails_and_is_not_overwritten(frozen_source):
    source, output, sha = frozen_source
    run_build(frozen_source)
    with (output / ARTIFACTS[0]).open("ab") as stream:
        stream.write(b"tamper")
    before = {p.name: sha256_file(p) for p in output.iterdir()}
    with pytest.raises(DatasetIntegrityError, match="artifact SHA mismatch"):
        load_frozen_split(output, source=source, approved_sha=sha, expected=None)
    with pytest.raises(DatasetIntegrityError, match="artifact SHA mismatch"):
        run_build(frozen_source)
    assert before == {p.name: sha256_file(p) for p in output.iterdir()}


def test_changed_seed_cannot_silently_replace_frozen_split(frozen_source, config, tmp_path):
    run_build(frozen_source)
    output = frozen_source[1]
    before = {p.name: sha256_file(p) for p in output.iterdir()}
    altered = deepcopy(config)
    altered["seed"] += "-altered"
    path = tmp_path / "altered_config.yaml"
    atomic_json(altered, path)
    with pytest.raises((DatasetIntegrityError, AssertionError)):
        run_build(frozen_source, config_path=path)
    assert before == {p.name: sha256_file(p) for p in output.iterdir()}


def test_wrong_freeze_metadata_fails(frozen_source):
    source, output, sha = frozen_source
    run_build(frozen_source)
    marker = json.loads((output / "split_freeze.json").read_text())
    marker["source_participant_count"] += 1
    atomic_json(marker, output / "split_freeze.json")
    with pytest.raises(DatasetIntegrityError, match="Source dimensions"):
        load_frozen_split(output, source=source, approved_sha=sha, expected=None)


@pytest.mark.parametrize("name", ["final_multimodal_dataset.parquet", "dataset_freeze.json"])
def test_actual_source_artifact_tamper_fails_before_publication(frozen_source, name):
    source, output, _ = frozen_source
    with (source / name).open("ab") as stream:
        stream.write(b"tampered_source")
    with pytest.raises((DatasetIntegrityError, json.JSONDecodeError)):
        run_build(frozen_source)
    assert not output.exists()


def test_predefined_objective_is_lexicographic_and_search_selects_exact_minimum(small, config):
    first = {
        "TRAIN": {"participants": 70, "knees": 133, "events": 50, "one_knee": 7, "two_knees": 63},
        "VALIDATION": {
            "participants": 15,
            "knees": 28,
            "events": 0,
            "one_knee": 2,
            "two_knees": 13,
        },
        "TEST": {"participants": 15, "knees": 29, "events": 0, "one_knee": 1, "two_knees": 14},
    }
    second = deepcopy(first)
    second["TRAIN"].update(knees=132, events=35)
    second["VALIDATION"].update(knees=29, events=7)
    second["TEST"]["events"] = 8
    assert objective(first, [70, 15, 15]) < objective(second, [70, 15, 15])
    _, _, search = generate(small, config)
    minimum = min(
        (tuple(Fraction(x) for x in row["objective_exact"]), row["candidate_index"])
        for row in search["candidates"]
    )
    assert minimum == (
        tuple(Fraction(x) for x in search["objective_exact"]),
        search["selected_candidate_index"],
    )


def test_concurrent_identical_builders_reuse_one_consistent_freeze(frozen_source):
    source, output, sha = frozen_source
    before = source_files(source)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_build(frozen_source), range(2)))
    assert results[0] == results[1]
    p, k = load_frozen_split(output, source=source, approved_sha=sha, expected=None)
    assert len(p) == 5 and len(k) == 6
    assert source_files(source) == before


def test_concurrent_conflicting_builder_cannot_replace_first_freeze(
    frozen_source, config, tmp_path
):
    altered = deepcopy(config)
    altered["seed"] += "-conflicting"
    path = tmp_path / "conflicting_config.yaml"
    atomic_json(altered, path)

    def attempt(config_path):
        try:
            return run_build(frozen_source, config_path=config_path)
        except (DatasetIntegrityError, AssertionError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (DEFAULT_CONFIG, path)))
    successful = [r for r in results if r is not None]
    assert len(successful) == 1
    source, output, sha = frozen_source
    p, _ = load_frozen_split(output, source=source, approved_sha=sha, expected=None)
    assert p.split_seed.eq(successful[0]["seed"]).all()
