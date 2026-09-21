"""Synthetic-only modeling tests; no TEST scoring and no production fitting or writes."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_multimodal_freeze import synthetic  # noqa: F401
from test_multimodal_splits import frozen_source, make_frame  # noqa: F401

import modeling.tabular as module
from imaging.artifact_io import atomic_json, sha256_file
from multimodal.freeze import DatasetIntegrityError, load_config
from multimodal.splits import generate

pytestmark = pytest.mark.public_portable


@pytest.fixture
def config():
    c = deepcopy(load_config(module.CONFIG))
    c["logistic_candidates"] = [{"C": 0.1}, {"C": 1.0}]
    c["catboost_candidates"] = [
        {"depth": 2, "learning_rate": 0.1, "iterations": 30, "l2_leaf_reg": 3.0}
    ]
    c["catboost_fixed"]["early_stopping_rounds"] = 5
    c["bootstrap_replicates"] = 10
    return c


@pytest.fixture
def development():
    f = make_frame({(2, 0): 70, (2, 1): 15, (2, 2): 15})
    rng = np.random.default_rng(17)
    for c in module.A:
        if c == "sex":
            f[c] = pd.array(rng.choice(["F", "M"], len(f)), dtype="string")
        elif c in module.CATEGORICAL:
            f[c] = pd.array(rng.integers(0, 2, len(f)), dtype="Int64")
        else:
            f[c] = f[c].astype(float) + rng.normal(size=len(f))
    f.loc[::7, "bmi"] = np.nan
    f.loc[::11, "family_knee_replacement_history"] = pd.NA
    p, k, _ = generate(f, load_config("configs/split_v1.yaml"))
    membership = p.set_index("participant_id").split.to_dict()
    parts = {}
    for name in ("TRAIN", "VALIDATION"):
        rows = (
            f.loc[
                k.split.eq(name).to_numpy(),
                [*module.IDENTITY, *module.A, "baseline_kl", module.TARGET],
            ]
            .copy()
            .reset_index(drop=True)
        )
        parts[name] = module.DevelopmentPartition(
            name,
            rows,
            membership,
            frozenset(rows[list(module.IDENTITY)].itertuples(index=False, name=None)),
        )
    data = module.DevelopmentData(
        parts["TRAIN"],
        parts["VALIDATION"],
        {key: "a" * 64 for key in [*module.APPROVED, "dataset_freeze", "split_freeze", "schema"]},
    )
    data.validate()
    return data


def fit(data, config, family="logistic", feature_set="formulation_A"):
    return module.fit_candidate(
        data,
        family,
        feature_set,
        config[f"{family}_candidates"][0],
        config,
        {"source_hashes": data.source_hashes},
    )


def test_formulation_contract_excludes_targets_followup_provenance_and_KL_in_A(development):
    assert len(module.A) == 14 and "baseline_kl" not in module.A
    assert module.FEATURE_SETS["formulation_B"] == (*module.A, "baseline_kl")
    for feature_set in module.FEATURE_SETS:
        columns = development.train.predictors(feature_set).columns
        assert not set(columns) & {
            "composite_progression",
            "followup_visit",
            "radiographic_kl_progression",
            "jsn_progression",
            "replacement_before_v06",
            "effective_provenance",
            "effective_crop_sha256",
            "adjudication_sequence",
        }
    with pytest.raises(DatasetIntegrityError):
        development.train.predictors("baseline_kl")


def test_train_only_imputation_scaling_and_categorical_encoding(development, config):
    train = development.train.frame
    val = development.validation.frame
    train.loc[0, "age_years"] = np.nan
    val["age_years"] = 1e6
    val["bmi"] = 1e7
    val["sex"] = "UNSEEN"
    model = fit(development, config)
    preprocessing = model.named_steps["preprocessing"]
    numeric = preprocessing.named_transformers_["numeric"]
    numeric_columns = [c for c in module.A if c not in module.CATEGORICAL]
    bmi_index = numeric_columns.index("bmi")
    median = float(train.bmi.median())
    assert numeric.named_steps["imputer"].statistics_[bmi_index] == median
    assert median != 0
    np.testing.assert_allclose(
        numeric.named_steps["scaler"].mean_[bmi_index], train.bmi.fillna(median).mean()
    )
    categorical = preprocessing.named_transformers_["categorical"]
    assert set(categorical.named_steps["encoder"].categories_[0]) == {"F", "M"}
    before = module.preprocessing_audit(model, development)
    with pytest.warns(UserWarning, match="unknown categories"):
        probability = module.predict_validation(model, development.validation)
    assert np.isfinite(probability).all() and before == module.preprocessing_audit(
        model, development
    )
    assert model.named_steps["types"].training_rows_ == len(train)


@pytest.mark.parametrize("native", [False, True])
def test_nullable_integer_categories_stable_with_or_without_missingness(native):
    columns = ("prior_knee_surgery", "family_knee_replacement_history")
    train = pd.DataFrame({c: pd.array([0, 1, pd.NA], dtype="Int64") for c in columns})
    validation = pd.DataFrame({c: pd.array([0, 1], dtype="Int64") for c in columns})
    original = train.copy(deep=True)
    adapter = module.TypedAdapter(columns, columns, native=native)
    converted = adapter.fit_transform(train)
    transformed = adapter.transform(validation)
    for c in columns:
        assert converted[c].iloc[:2].tolist() == transformed[c].tolist() == ["0", "1"]
        assert adapter.train_categories_[c] == ["0", "1"]
        assert converted[c].iloc[2] == "__MISSING__" if native else pd.isna(converted[c].iloc[2])
    pd.testing.assert_frame_equal(original, train, check_exact=True)


@pytest.mark.parametrize("family", ["logistic", "catboost"])
def test_deterministic_fit_and_validation_prediction_linkage(development, config, family):
    first = fit(development, config, family)
    second = fit(development, config, family)
    np.testing.assert_allclose(
        module.predict_validation(first, development.validation),
        module.predict_validation(second, development.validation),
        atol=1e-12,
        rtol=0,
    )
    assert first.named_steps["types"].training_rows_ == len(development.train.frame)
    if family == "catboost":
        assert first.named_steps["model"].get_all_params()["counter_calc_method"] == "SkipTest"
        converted = first.named_steps["types"].transform(
            development.train.predictors("formulation_A")
        )
        assert converted.bmi.isna().sum() == development.train.frame.bmi.isna().sum()
        assert "__MISSING__" in set(converted.family_knee_replacement_history)


@pytest.mark.parametrize("action", ["fit", "predict", "metrics", "bootstrap"])
def test_TEST_role_is_never_fit_transformed_or_scored(development, config, action):
    model = fit(development, config)
    wrong = deepcopy(development.validation)
    wrong.name = "TEST"
    with pytest.raises(DatasetIntegrityError):
        if action == "fit":
            module.fit_candidate(
                module.DevelopmentData(development.train, wrong, development.source_hashes),
                "logistic",
                "formulation_A",
                {"C": 1},
                config,
                {},
            )
        elif action == "predict":
            module.predict_validation(model, wrong)
        elif action == "metrics":
            module.metrics(wrong, np.full(len(wrong.frame), 0.2))
        else:
            list(module.cluster_bootstrap_indices(wrong, 10, 62027))
    assert not hasattr(development, "test")


def test_actual_TEST_participant_cannot_be_disguised_as_TRAIN(development, config):
    test_id = next(pid for pid, role in development.train.membership.items() if role == "TEST")
    development.train.frame.loc[0, "participant_id"] = test_id
    with pytest.raises(DatasetIntegrityError, match="outside authoritative"):
        fit(development, config)


def test_clustered_bootstrap_includes_both_knees_for_each_sampled_participant(development):
    validation = development.validation
    ids = validation.frame.participant_id.to_numpy()
    samples = list(module.cluster_bootstrap_indices(validation, 20, 62027))
    again = list(module.cluster_bootstrap_indices(validation, 20, 62027))
    for sample, second in zip(samples, again, strict=True):
        np.testing.assert_array_equal(sample, second)
        multiplicity = np.bincount(sample, minlength=len(ids))
        for pid in set(ids):
            assert np.unique(multiplicity[ids == pid]).size == 1
        assert sum(multiplicity[ids == pid][0] for pid in set(ids)) == len(set(ids))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_participant", "TEST_role"])
def test_validation_predictions_missing_duplicates_or_wrong_role_fail(development, mutation):
    predictions = development.validation.frame[[*module.IDENTITY, module.TARGET]].copy()
    predictions["predicted_probability"] = 0.2
    predictions["model_id"] = "synthetic"
    predictions["split"] = "VALIDATION"
    if mutation == "missing":
        predictions = predictions.iloc[1:]
    elif mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif mutation == "wrong_participant":
        predictions.loc[0, "participant_id"] = "WRONG"
    else:
        predictions.loc[0, "split"] = "TEST"
    with pytest.raises((DatasetIntegrityError, AssertionError)):
        module.validate_predictions(development.validation, predictions, {"synthetic"})


@pytest.mark.parametrize("key", ["dataset", "participants", "knees", "feature_groups"])
def test_exact_approved_source_SHAs_required(frozen_source, key):  # noqa: F811
    from test_multimodal_splits import run_build

    run_build(frozen_source)
    source, _, _ = frozen_source
    approved = {
        name: sha256_file(source / path)
        for name, path in {
            "dataset": "final_multimodal_dataset.parquet",
            "participants": "splits/v1/participant_split_manifest.parquet",
            "knees": "splits/v1/knee_split_manifest.parquet",
            "feature_groups": "feature_groups.json",
        }.items()
    }
    approved[key] = "0" * 64
    with pytest.raises(DatasetIntegrityError, match="SHA mismatch"):
        module.load_development(source, approved=approved, expected=None, outcome_counts=None)


def test_end_to_end_artifacts_hashes_rerun_and_no_TEST_predictions(
    development, config, tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    path = tmp_path / "predeclared.yaml"
    atomic_json(config, path)
    output = tmp_path / "run"
    before_train = development.train.frame.copy(deep=True)
    before_validation = development.validation.frame.copy(deep=True)
    marker = module.run(output=output, config_path=path, data=development)
    first = {
        str(p.relative_to(output)): (sha256_file(p), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert module.run(output=output, config_path=path, data=development) == marker
    assert first == {
        str(p.relative_to(output)): (sha256_file(p), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    pd.testing.assert_frame_equal(before_train, development.train.frame, check_exact=True)
    pd.testing.assert_frame_equal(before_validation, development.validation.frame, check_exact=True)
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    assert set(predictions.participant_id) == set(development.validation.frame.participant_id)
    assert predictions.split.eq("VALIDATION").all() and predictions.model_id.nunique() == 11
    assert len(predictions) == 11 * len(development.validation.frame)
    assert not list(output.rglob("*TEST*"))
    assert marker["test_scored"] is False and marker["default_formulation"] == "A"
    for name, h in marker["artifacts_sha256"].items():
        assert sha256_file(output / name) == h
    selected = json.loads((output / "selected_candidates.json").read_text())
    assert len(selected) == 10
    for candidate in selected.values():
        model = module.joblib.load(output / candidate["model_artifact"])
        assert type(model.named_steps["types"]).__module__ == "modeling.tabular"
        assert model.milestone_provenance_["source_hashes"] == development.source_hashes
    with (output / "validation_predictions.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(DatasetIntegrityError, match="artifact SHA mismatch"):
        module.run(output=output, config_path=path, data=development)


def test_config_cannot_enable_TEST_or_class_weighting(development, config):
    config["test_scoring_permitted"] = True
    with pytest.raises(DatasetIntegrityError):
        fit(development, config)


def test_command_line_delegates_to_importable_model_module(monkeypatch):
    import runpy

    calls = []
    monkeypatch.setattr(module, "main", lambda: calls.append("canonical") or 0)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exit_code:
            runpy.run_module("modeling.tabular", run_name="__main__")
    assert exit_code.value.code == 0 and calls == ["canonical"]


def test_null_reference_is_TRAIN_prevalence_and_calibration_is_diagnostic_only(development):
    probability = np.full(len(development.validation.frame), development.train.labels().mean())
    score = module.metrics(development.validation, probability)
    assert score["AUROC"] == 0.5
    assert score["AUPRC"] == pytest.approx(development.validation.labels().mean())
    assert score["calibration"]["slope"] is None
