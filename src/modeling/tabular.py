"""Milestone 6C: frozen-source, TRAIN-fitted, VALIDATION-only primary tabular baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from imaging.artifact_io import file_lock
from multimodal.freeze import (
    CLINICAL,
    DEFAULT_DIRECTORY,
    FUNCTION,
    PRO,
    load_config,
    load_frozen_dataset,
    publish_json,
    publish_parquet,
    read_json,
    require,
    sha256_file,
)
from multimodal.splits import (
    APPROVED_DATASET_SHA,
    load_frozen_split,
    source_files,
    verify_preservation,
)

OUTPUT = Path("data/processed/modeling/tabular/v1")
CONFIG = Path("configs/tabular_v1.yaml")
TARGET = "composite_progression"
IDENTITY = ("participant_id", "knee_side_code", "baseline_visit")
CATEGORICAL = ("sex", "prior_knee_surgery", "family_knee_replacement_history")
A = (*CLINICAL, *PRO, *FUNCTION)
FEATURE_SETS = {
    "demographic_clinical": CLINICAL,
    "patient_reported": PRO,
    "physical_function": FUNCTION,
    "formulation_A": A,
    "formulation_B": (*A, "baseline_kl"),
}
APPROVED = {
    "dataset": APPROVED_DATASET_SHA,
    "participants": "f2c4f4b0e3260f9a9f482acb88d5f53fdcad61bf66261f3d947ad3b7da3115ba",
    "knees": "4658b04a667a747c9bbc5422fa5ec7f516677b98714e46e3bbba7fc974502815",
    "feature_groups": "cde2f2dbb4eba356559daf4c854e3f955bac99daa5cf9c03afd2f6052bde9e28",
}


@dataclass
class DevelopmentPartition:
    name: str
    frame: pd.DataFrame
    membership: dict
    expected_keys: frozenset

    def validate(self) -> None:
        require(self.name in ("TRAIN", "VALIDATION"), "TEST is forbidden in modeling/evaluation")
        require(
            self.frame.participant_id.map(self.membership).eq(self.name).all(),
            "Participant outside authoritative development partition",
        )
        require(not self.frame.duplicated(list(IDENTITY)).any(), "Duplicate development knee")
        keys = frozenset(self.frame[list(IDENTITY)].itertuples(index=False, name=None))
        require(
            keys == self.expected_keys and len(keys) == len(self.frame),
            "Missing/extra/wrong development knee",
        )
        require(not self.frame[TARGET].isna().any(), "Unavailable frozen primary label")
        require(self.frame.baseline_visit.eq("V00").all(), "Unexpected development visit")

    def predictors(self, feature_set: str) -> pd.DataFrame:
        self.validate()
        require(feature_set in FEATURE_SETS, "Unapproved predictor feature set")
        return self.frame[list(FEATURE_SETS[feature_set])].copy()

    def labels(self) -> np.ndarray:
        self.validate()
        return self.frame[TARGET].to_numpy(dtype=np.int64)


@dataclass
class DevelopmentData:
    train: DevelopmentPartition
    validation: DevelopmentPartition
    source_hashes: dict

    def validate(self) -> None:
        self.train.validate()
        self.validation.validate()
        require(
            self.train.name == "TRAIN" and self.validation.name == "VALIDATION",
            "Wrong development role",
        )
        require(
            not set(self.train.frame.participant_id) & set(self.validation.frame.participant_id),
            "Participant leakage",
        )


def load_development(
    source: Path = DEFAULT_DIRECTORY,
    approved: dict = APPROVED,
    expected: tuple | None = (6961, 3621, 54),
    outcome_counts: tuple | None = (4873, 749, 1044, 161),
) -> DevelopmentData:
    source = Path(source)
    paths = {
        "dataset": source / "final_multimodal_dataset.parquet",
        "participants": source / "splits/v1/participant_split_manifest.parquet",
        "knees": source / "splits/v1/knee_split_manifest.parquet",
        "feature_groups": source / "feature_groups.json",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    require(hashes == approved, "Approved frozen dataset/split/feature-group SHA mismatch")
    participants, knees = load_frozen_split(
        source / "splits/v1", source=source, approved_sha=approved["dataset"], expected=expected
    )
    frame = load_frozen_dataset(source)
    groups = read_json(source / "feature_groups.json")
    schema = {e["column_name"]: e for e in read_json(source / "schema.json")["columns"]}
    require(
        groups["formulations"]["A"]["tabular_predictors"] == list(A)
        and groups["formulations"]["B"]["tabular_predictors"] == list((*A, "baseline_kl")),
        "Frozen feature formulations disagree",
    )
    for group in ("demographic_clinical", "patient_reported", "physical_function"):
        require(
            groups["groups"][group] == list(FEATURE_SETS[group]), "Frozen feature domain changed"
        )
    for c in (*A, "baseline_kl"):
        e = schema[c]
        require(
            e["timing"] == "V00"
            and e["predictor_eligible"]
            and not e["target"]
            and e["default_formulation_A_eligible"] == (c in A),
            "Predictor leakage/formulation mismatch",
        )
    membership = participants.set_index("participant_id").split.to_dict()
    partitions = {}
    for name in ("TRAIN", "VALIDATION"):
        # No TEST predictor/target matrix is constructed or returned.
        rows = (
            frame.loc[knees.split.eq(name).to_numpy(), [*IDENTITY, *A, "baseline_kl", TARGET]]
            .copy()
            .reset_index(drop=True)
        )
        partitions[name] = DevelopmentPartition(
            name,
            rows,
            membership,
            frozenset(rows[list(IDENTITY)].itertuples(index=False, name=None)),
        )
    if outcome_counts is not None:
        actual = tuple(
            v
            for name in ("TRAIN", "VALIDATION")
            for v in (len(partitions[name].frame), int(partitions[name].frame[TARGET].sum()))
        )
        require(actual == outcome_counts, "Frozen TRAIN/VALIDATION outcome census mismatch")
    hashes.update(
        {
            "dataset_freeze": sha256_file(source / "dataset_freeze.json"),
            "split_freeze": sha256_file(source / "splits/v1/split_freeze.json"),
            "schema": sha256_file(source / "schema.json"),
        }
    )
    data = DevelopmentData(partitions["TRAIN"], partitions["VALIDATION"], hashes)
    data.validate()
    return data


class TypedAdapter(TransformerMixin, BaseEstimator):
    """Deterministic dtype conversion; fitted category vocabulary is TRAIN-only audit metadata."""

    def __init__(self, columns: tuple, categorical: tuple, native: bool = False):
        self.columns = columns
        self.categorical = categorical
        self.native = native

    def fit(self, X, y=None):
        require(tuple(X.columns) == self.columns, "Predictor schema mismatch")
        self.train_categories_ = {
            c: sorted(str(v) for v in X[c].dropna().unique()) for c in self.categorical
        }
        self.training_rows_ = len(X)
        self.n_features_in_ = len(self.columns)
        return self

    def transform(self, X):
        require(tuple(X.columns) == self.columns, "Predictor schema mismatch")
        result = X.copy()
        for c in self.columns:
            if c in self.categorical:
                require(
                    not X[c].dropna().astype(str).eq("__MISSING__").any(),
                    "Reserved missing token collision",
                )
                result[c] = (
                    X[c]
                    .astype(object)
                    .map(
                        lambda v: (
                            "__MISSING__"
                            if self.native and pd.isna(v)
                            else np.nan
                            if pd.isna(v)
                            else str(v)
                        )
                    )
                    .astype(object)
                )
            else:
                result[c] = X[c].to_numpy(dtype=np.float64, na_value=np.nan)
                require(not np.isinf(result[c]).any(), "Nonfinite predictor")
        return result


def validate_config(config: dict) -> None:
    require(
        config["target"] == TARGET and config["default_formulation"] == "A",
        "Wrong target/formulation",
    )
    require(
        config["feature_sets"] == list(FEATURE_SETS)
        and config["categorical_columns"] == list(CATEGORICAL),
        "Feature definitions changed",
    )
    require(
        config["test_scoring_permitted"] is False and config["class_weight_sensitivity"] is False,
        "TEST/class-weight policy changed",
    )
    require(
        config["selection"]
        == [
            "validation_AUROC_descending",
            "validation_AUPRC_descending",
            "validation_Brier_ascending",
            "candidate_index_ascending",
        ],
        "Selection policy changed",
    )
    require(
        config["logistic_fixed"]["class_weight"] is None
        and config["logistic_fixed"]["l1_ratio"] == 0.0,
        "Not unweighted L2 logistic",
    )
    require(
        config["catboost_fixed"]["counter_calc_method"] == "SkipTest"
        and config["catboost_fixed"]["thread_count"] == 1
        and config["catboost_fixed"]["allow_writing_files"] is False,
        "Unsafe CatBoost preprocessing/determinism",
    )
    require(
        not {"class_weights", "auto_class_weights", "scale_pos_weight"}
        & set(config["catboost_fixed"]),
        "Weighted CatBoost prohibited",
    )
    require(
        1 <= len(config["logistic_candidates"]) <= 3
        and 1 <= len(config["catboost_candidates"]) <= 3,
        "Unbounded model search",
    )


def fit_candidate(
    data: DevelopmentData,
    family: str,
    feature_set: str,
    candidate: dict,
    config: dict,
    provenance: dict,
) -> Pipeline:
    data.validate()
    validate_config(config)
    X = data.train.predictors(feature_set)
    y = data.train.labels()
    require(np.unique(y).size == 2, "TRAIN must contain both primary classes")
    cols = FEATURE_SETS[feature_set]
    cats = tuple(c for c in cols if c in CATEGORICAL)
    nums = tuple(c for c in cols if c not in cats)
    adapter = TypedAdapter(cols, cats, native=family == "catboost")
    if family == "logistic":
        require(
            all(X[c].notna().any() for c in cols),
            "All-missing TRAIN predictor; cannot silently drop",
        )
        transformers = []
        if nums:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(nums),
                )
            )
        if cats:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    drop="first", handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    list(cats),
                )
            )
        model = Pipeline(
            [
                ("types", adapter),
                ("preprocessing", ColumnTransformer(transformers)),
                (
                    "model",
                    LogisticRegression(
                        **config["logistic_fixed"], **candidate, random_state=config["seed"]
                    ),
                ),
            ]
        )
        with threadpool_limits(limits=1):
            model.fit(X, y)
    else:
        require(family == "catboost", "Unapproved model family")
        model = Pipeline(
            [
                ("types", adapter),
                (
                    "model",
                    CatBoostClassifier(
                        **config["catboost_fixed"],
                        **candidate,
                        random_seed=config["seed"],
                        cat_features=list(cats),
                        metadata={"source_provenance": json.dumps(provenance, sort_keys=True)},
                    ),
                ),
            ]
        )
        converted = adapter.fit_transform(X, y)
        validation = adapter.transform(data.validation.predictors(feature_set))
        model.named_steps["model"].fit(
            converted, y, eval_set=(validation, data.validation.labels())
        )
    model.milestone_provenance_ = provenance
    model.feature_set_ = feature_set
    model.family_ = family
    return model


def predict_validation(model: Pipeline, partition: DevelopmentPartition) -> np.ndarray:
    require(partition.name == "VALIDATION", "Only VALIDATION can be scored in 6C")
    X = partition.predictors(model.feature_set_)
    with threadpool_limits(limits=1):
        probability = model.predict_proba(X)[:, 1]
    require(
        np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
        "Invalid probabilities",
    )
    return probability


def calibration(y: np.ndarray, probability: np.ndarray) -> dict:
    z = np.log(np.clip(probability, 1e-12, 1 - 1e-12) / np.clip(1 - probability, 1e-12, 1))
    if np.ptp(z) < 1e-10:
        return {
            "intercept": None,
            "slope": None,
            "status": "Constant prediction: joint intercept/slope not identifiable.",
        }
    design = np.column_stack([np.ones(len(z)), z])

    def loss(beta):
        linear = design @ beta
        return np.mean(np.logaddexp(0, linear) - y * linear)

    def gradient(beta):
        return design.T @ (expit(design @ beta) - y) / len(y)

    result = minimize(
        loss,
        np.array([0.0, 1.0]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    require(
        result.success or np.max(np.abs(gradient(result.x))) < 1e-6,
        "Calibration MLE did not converge",
    )
    return {
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
        "status": "Joint unpenalized validation diagnostic; never applied.",
    }


def metrics(partition: DevelopmentPartition, probability: np.ndarray) -> dict:
    require(partition.name == "VALIDATION", "TEST metrics forbidden")
    y = partition.labels()
    require(len(probability) == len(y) and np.unique(y).size == 2, "Invalid validation evaluation")
    return {
        "observations": len(y),
        "positive_events": int(y.sum()),
        "prevalence": float(y.mean()),
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "AUPRC_no_skill_reference": float(y.mean()),
        "calibration": calibration(y, probability),
    }


def cluster_bootstrap_indices(partition: DevelopmentPartition, replicates: int, seed: int):
    require(partition.name == "VALIDATION", "Bootstrap is VALIDATION-only")
    partition.validate()
    ids = partition.frame.participant_id.to_numpy()
    participants = sorted(set(ids))
    groups = [np.flatnonzero(ids == pid) for pid in participants]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.integers(len(groups), size=len(groups))
        yield np.concatenate([groups[i] for i in sampled])


def bootstrap(
    partition: DevelopmentPartition, probabilities: dict, config: dict
) -> tuple[pd.DataFrame, dict]:
    y = partition.labels()
    rows = []
    sample_hash = hashlib.sha256()
    for replicate, index in enumerate(
        cluster_bootstrap_indices(
            partition, config["bootstrap_replicates"], config["bootstrap_seed"]
        )
    ):
        sample_hash.update(index.astype("<i8").tobytes())
        actual = y[index]
        for name, probability in probabilities.items():
            p = probability[index]
            good = np.unique(actual).size == 2
            rows.append(
                {
                    "model_id": name,
                    "replicate": replicate,
                    "observations": len(index),
                    "events": int(actual.sum()),
                    "AUROC": float(roc_auc_score(actual, p)) if good else np.nan,
                    "AUPRC": float(average_precision_score(actual, p)) if good else np.nan,
                    "Brier": float(np.mean((actual - p) ** 2)),
                    "single_class_replicate": not good,
                }
            )
    frame = pd.DataFrame(rows)
    frame["model_id"] = frame.model_id.astype("string")
    intervals = {
        name: {
            metric: {
                "lower": float(values[metric].dropna().quantile(0.025)),
                "upper": float(values[metric].dropna().quantile(0.975)),
                "valid_replicates": int(values[metric].notna().sum()),
            }
            for metric in ("AUROC", "AUPRC", "Brier")
        }
        for name, values in frame.groupby("model_id", sort=True)
    }
    return frame, {
        "replicates": config["bootstrap_replicates"],
        "seed": config["bootstrap_seed"],
        "unit": "participant",
        "all_knees_in_sampled_cluster_included": True,
        "paired_draws_shared_by_all_models": True,
        "sample_indices_sha256": sample_hash.hexdigest(),
        "confidence_level": 0.95,
        "method": "percentile; conditional on fitted/selected development models",
        "intervals": intervals,
    }


def preprocessing_audit(model: Pipeline, data: DevelopmentData) -> dict:
    columns = FEATURE_SETS[model.feature_set_]
    cats = model.named_steps["types"].categorical
    entries = {
        c: {
            "TRAIN_missing": int(data.train.frame[c].isna().sum()),
            "VALIDATION_missing": int(data.validation.frame[c].isna().sum()),
        }
        for c in columns
    }
    if model.family_ == "logistic":
        preprocessing = model.named_steps["preprocessing"]
        for name, transform, cols in preprocessing.transformers_:
            if name == "remainder":
                continue
            for i, c in enumerate(cols):
                v = transform.named_steps["imputer"].statistics_[i]
                entries[c]["TRAIN_imputation_value"] = str(v) if c in cats else float(v)
                if name == "numeric":
                    scaler = transform.named_steps["scaler"]
                    entries[c].update(
                        TRAIN_scaler_mean=float(scaler.mean_[i]),
                        TRAIN_scaler_scale=float(scaler.scale_[i]),
                    )
                else:
                    entries[c]["TRAIN_encoding_categories"] = list(
                        transform.named_steps["encoder"].categories_[i]
                    )
    else:
        for c in columns:
            entries[c]["TRAIN_imputation_value"] = None
            entries[c]["missing_handling"] = (
                "__MISSING__ categorical token" if c in cats else "Native NaN"
            )
    return {
        "features": entries,
        "fit_partition": "TRAIN",
        "training_rows": len(data.train.frame),
        "adapter_TRAIN_categories": model.named_steps["types"].train_categories_,
        "validation_transform_refits": False,
    }


def experiments(data: DevelopmentData, config: dict, provenance: dict) -> dict:
    data.validate()
    validate_config(config)
    models, candidate_results, selected, probability, audits, interpretations = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for family in ("logistic", "catboost"):
        for feature_set in FEATURE_SETS:
            best = None
            for i, params in enumerate(config[f"{family}_candidates"]):
                name = f"{family}__{feature_set}__candidate_{i}"
                model = fit_candidate(
                    data,
                    family,
                    feature_set,
                    params,
                    config,
                    {
                        **provenance,
                        "model_id": name,
                        "features": list(FEATURE_SETS[feature_set]),
                        "candidate_index": i,
                        "candidate_parameters": params,
                        "fit_partition": "TRAIN",
                        "evaluation_partition": "VALIDATION",
                    },
                )
                p = predict_validation(model, data.validation)
                score = metrics(data.validation, p)
                models[name] = model
                candidate_results[name] = score
                key = (-score["AUROC"], -score["AUPRC"], score["Brier"], i)
                if best is None or key < best[0]:
                    best = key, name, p
            _, name, p = best
            model_id = f"{family}__{feature_set}"
            selected[model_id] = {
                "candidate_id": name,
                "candidate_index": models[name].milestone_provenance_["candidate_index"],
                "parameters": models[name].milestone_provenance_["candidate_parameters"],
                "features": list(FEATURE_SETS[feature_set]),
                "model_artifact": f"models/{name}.joblib",
                "metrics": candidate_results[name],
                "retained_trees": int(models[name].named_steps["model"].tree_count_)
                if family == "catboost"
                else None,
            }
            probability[model_id] = p
            audits[model_id] = preprocessing_audit(models[name], data)
            if family == "logistic":
                classifier = models[name].named_steps["model"]
                names = models[name].named_steps["preprocessing"].get_feature_names_out()
                interpretations[model_id] = {
                    "intercept": float(classifier.intercept_[0]),
                    "coefficients": {
                        c: float(v) for c, v in zip(names, classifier.coef_[0], strict=True)
                    },
                    "scale": "Numeric coefficients per TRAIN-imputed/scaled standard deviation; categorical coefficients relative to dropped first TRAIN category. Penalized descriptive coefficients, no inferential claims.",
                }
            else:
                interpretations[model_id] = {
                    "importance_type": "PredictionValuesChange",
                    "feature_importance": {
                        c: float(v)
                        for c, v in zip(
                            FEATURE_SETS[feature_set],
                            models[name].named_steps["model"].feature_importances_,
                            strict=True,
                        )
                    },
                }
            print(f"Selected {model_id}", flush=True)
    null_probability = float(data.train.labels().mean())
    probability["prevalence_reference"] = np.full(len(data.validation.frame), null_probability)
    null = {
        "TRAIN_prevalence_probability": null_probability,
        "metrics": metrics(data.validation, probability["prevalence_reference"]),
    }
    frames = []
    for name, p in probability.items():
        rows = data.validation.frame[[*IDENTITY, TARGET]].copy()
        rows["predicted_probability"] = p
        rows["model_id"] = pd.array([name] * len(rows), dtype="string")
        rows["split"] = pd.array(["VALIDATION"] * len(rows), dtype="string")
        frames.append(rows)
    predictions = pd.concat(frames, ignore_index=True)
    validate_predictions(data.validation, predictions, set(probability))
    print("Bootstrapping validation participant clusters...", flush=True)
    boot, intervals = bootstrap(data.validation, probability, config)
    curves = {}
    for name, p in probability.items():
        observed, predicted = calibration_curve(
            data.validation.labels(),
            p,
            n_bins=config["calibration_curve"]["bins"],
            strategy=config["calibration_curve"]["strategy"],
        )
        curves[name] = {
            "mean_predicted_probability": predicted.tolist(),
            "observed_event_fraction": observed.tolist(),
        }
    return {
        "models": models,
        "selected": selected,
        "candidate_metrics": candidate_results,
        "predictions": predictions,
        "bootstrap": boot,
        "intervals": intervals,
        "preprocessing": audits,
        "interpretability": interpretations,
        "calibration": curves,
        "null": null,
    }


def validate_predictions(
    validation: DevelopmentPartition, predictions: pd.DataFrame, model_ids: set
) -> None:
    require(validation.name == "VALIDATION", "TEST predictions forbidden")
    validation.validate()
    require(
        set(predictions.model_id) == model_ids and predictions.split.eq("VALIDATION").all(),
        "Wrong prediction roles/models",
    )
    require(
        not predictions.duplicated([*IDENTITY, "model_id"]).any(),
        "Duplicate validation predictions",
    )
    for _, group in predictions.groupby("model_id", sort=False):
        pd.testing.assert_frame_equal(
            group[[*IDENTITY, TARGET]].reset_index(drop=True),
            validation.frame[[*IDENTITY, TARGET]].reset_index(drop=True),
            check_exact=True,
        )


def preserve(baseline: dict) -> None:
    verify_preservation(baseline, DEFAULT_DIRECTORY)
    require(
        source_files(DEFAULT_DIRECTORY / "splits/v1") == baseline["split_files"],
        "Frozen split bundle changed",
    )


def verify_existing(output: Path, provenance: dict) -> dict | None:
    path = output / "run_freeze.json"
    if not path.exists():
        require(
            not (output / "models").exists(),
            "Unmarked partial fitted artifacts; refusing blind reuse",
        )
        return None
    frozen = read_json(path)
    require(
        frozen["provenance"] == provenance and frozen["test_scored"] is False,
        "Model configuration/source provenance changed",
    )
    for name, h in frozen["artifacts_sha256"].items():
        require(sha256_file(output / name) == h, "Fitted artifact SHA mismatch")
    return frozen


def model_artifact(model: Pipeline, path: Path, data: DevelopmentData) -> None:
    if path.exists():
        # Existing run hashes were verified before loading trusted local pickle artifacts.
        original = joblib.load(path)
        require(
            original.milestone_provenance_ == model.milestone_provenance_,
            "Fitted model provenance differs",
        )
        np.testing.assert_allclose(
            predict_validation(original, data.validation),
            predict_validation(model, data.validation),
            rtol=0,
            atol=1e-12,
        )
        require(
            preprocessing_audit(original, data) == preprocessing_audit(model, data),
            "Refitted preprocessing differs",
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".fitted_", suffix=".joblib", dir=path.parent)
        os.close(fd)
        temp = Path(temporary)
        try:
            joblib.dump(model, temp, compress=3)
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()


def plot_calibration(curves: dict, path: Path) -> None:
    if path.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(18, 7), layout="constrained")
    for ax, (name, values) in zip(
        axes.flat, [(n, v) for n, v in curves.items() if n != "prevalence_reference"], strict=True
    ):
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(values["mean_predicted_probability"], values["observed_event_fraction"], "o-")
        ax.set(
            title=name.replace("__", "\n"),
            xlabel="Mean predicted probability",
            ylabel="Observed event fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    figure.suptitle("VALIDATION calibration — development diagnostics only")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run(
    *,
    output: Path = OUTPUT,
    config_path: Path = CONFIG,
    data: DevelopmentData | None = None,
    preservation: dict | None = None,
) -> dict:
    output, config_path = Path(output), Path(config_path)
    data = data or load_development()
    config = load_config(config_path)
    validate_config(config)
    provenance = {
        "source_hashes": data.source_hashes,
        "model_configuration_sha256": sha256_file(config_path),
        "preprocessing_configuration_sha256": hashlib.sha256(
            json.dumps(config["preprocessing"], sort_keys=True).encode()
        ).hexdigest(),
        "training_code_sha256": sha256_file(Path(__file__)),
        "seed": config["seed"],
        "bootstrap_seed": config["bootstrap_seed"],
        "TRAIN_rows": len(data.train.frame),
        "VALIDATION_rows": len(data.validation.frame),
        "runtime": {
            n: version(n)
            for n in (
                "scikit-learn",
                "catboost",
                "numpy",
                "pandas",
                "scipy",
                "joblib",
                "matplotlib",
                "threadpoolctl",
            )
        },
    }
    with file_lock(output / "run_freeze.json"):
        existing = verify_existing(output, provenance)
        if preservation is not None:
            print("Verifying source preservation before fitting...", flush=True)
            preserve(preservation)
        result = experiments(data, config, provenance)
        if preservation is not None:
            print("Verifying source preservation after fitting...", flush=True)
            preserve(preservation)
        for name, model in result["models"].items():
            model_artifact(model, output / f"models/{name}.joblib", data)
        publish_parquet(result["predictions"], output / "validation_predictions.parquet")
        publish_parquet(result["bootstrap"], output / "bootstrap_metrics.parquet")
        for name, value in (
            ("selected_candidates", result["selected"]),
            ("candidate_metrics", result["candidate_metrics"]),
            ("preprocessing_audit", result["preprocessing"]),
            ("interpretability", result["interpretability"]),
            ("bootstrap_intervals", result["intervals"]),
            ("calibration_data", result["calibration"]),
            ("null_reference", result["null"]),
            ("feature_lists", {s: list(v) for s, v in FEATURE_SETS.items()}),
            ("predeclared_configuration", config),
        ):
            publish_json(value, output / f"{name}.json")
        plot_calibration(result["calibration"], output / "validation_calibration.png")
        require(
            not any("test_prediction" in p.name.lower() for p in output.rglob("*")),
            "TEST artifact forbidden",
        )
        artifacts = {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name not in ("run_freeze.json", "run_freeze.json.lock")
        }
        frozen = {
            "run_version": config["run_version"],
            "status": "FROZEN_DEVELOPMENT_BASELINES",
            "created_at_utc": existing["created_at_utc"]
            if existing
            else datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "artifacts_sha256": artifacts,
            "test_scored": False,
            "training_partition": "TRAIN",
            "evaluation_partition": "VALIDATION",
            "default_formulation": "A",
            "formulation_B_role": "Sensitivity only",
            "advance_to_later_comparison": ["logistic__formulation_A", "catboost__formulation_A"],
            "other_models_retained_for": "Domain ablations and explicit B sensitivity, not a final project model.",
            "preprocessing_fit_partition": "TRAIN",
            "source_preservation_passed": preservation is not None,
            "bootstrap_unit": "participant",
            "bootstrap_replicates": config["bootstrap_replicates"],
            "determinism": "CPU single-thread fits; refit predictions checked at absolute tolerance 1e-12. Native CatBoost model bytes may contain volatile timestamps/GUIDs; original validated artifacts are reused.",
            "no_class_weighting_resampling_or_threshold_selection": True,
        }
        publish_json(frozen, output / "run_freeze.json")
        verify_existing(output, provenance)
        return frozen


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    baseline = read_json(OUTPUT / "pre_change_preservation.json")
    result = run(preservation=baseline)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "test_scored",
                    "default_formulation",
                    "advance_to_later_comparison",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    # Keep fitted custom transformers importable, rather than pickled as __main__.
    from modeling.tabular import main as canonical_main

    raise SystemExit(canonical_main())
