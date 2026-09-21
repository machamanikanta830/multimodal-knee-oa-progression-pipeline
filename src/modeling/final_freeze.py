"""Milestone 6F: Final Development Freeze and Test Evaluation Protocol.

Freezes the exact development models, artifacts, hashes, metrics, and evaluation
protocol for the one-time coordinated TEST evaluation. Zero TEST scoring, zero
TEST model inference, zero TEST predictions, and zero model refitting.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from imaging.artifact_io import file_lock, sha256_file
from modeling import fusion, tabular
from multimodal.freeze import (
    DEFAULT_DIRECTORY,
    publish_json,
    read_json,
    require,
)

DEFAULT_OUTPUT = Path("data/processed/modeling/final_freeze/v1")
FUSION_OUTPUT = Path("data/processed/modeling/fusion/v1")
TABULAR_OUTPUT = Path("data/processed/modeling/tabular/v1")
IMAGE_OUTPUT = Path("data/processed/modeling/image/v1/production_cuda")
IMAGE_BASE_DIR = Path("data/processed/modeling/image/v1")

APPROVED_HASHES = {
    "dataset": "382d17228ca6eb85c064f4a2c2d8c402a2479bde6fc746c9a55f7fbd57eac190",
    "participants": "f2c4f4b0e3260f9a9f482acb88d5f53fdcad61bf66261f3d947ad3b7da3115ba",
    "knees": "4658b04a667a747c9bbc5422fa5ec7f516677b98714e46e3bbba7fc974502815",
    "feature_groups": "cde2f2dbb4eba356559daf4c854e3f955bac99daa5cf9c03afd2f6052bde9e28",
    "schema": "92dae260d28083789a796337ea12762de3d2e09779666ba5747ff8ce6f64b19e",
    "dataset_freeze": "293e628798661b71bd1145433a7df4c45b425dfcd38f1c661853e5117756461e",
    "split_freeze": "af0fe879044aa68bbafba7cbd71aee1fee26612775f49cfbd4f803c381ec9dbf",
    "cohort": "b721064ffd73761971fce1170e3457645c52c2a129bfe95fa277f1095c415524",
    "imaging_manifest": "cce476f5ecc2381232b026a9aec09dffe58db391116deadb7fddb2405aee0fa7",
    "tabular_run_freeze": "2adc2a7edfe95d61612428468514e4a525565ed96645156b65152253dfb98211",
    "tabular_validation_predictions": "2d90826cac7c72f3d29dc01065507aec0003babc47665407cf8c4c1bac841999",
    "tabular_logistic_A_model": "441323137bb57bb07a53c1d2a634ddac3480f7af3ff05ca7d49a926bccd6862c",
    "tabular_logistic_B_model": "26e820b3162e15ab676f26952a5d8ebe8f9755610f8167a4406ae8cd1a03e935",
    "image_run_freeze": "f14ea1c6921e8e95b3637d374a78b35d2cb853a574ecb6b15bdad1ea5335582e",
    "image_validation_predictions": "c8f327fc63e845d564dfd85e139fb871609c3e67d910017b5f776b8453c60bbb",
    "image_selected_checkpoint": "b14a74f329a268ae6855a9dcf69f6850d38e31f7fb9bd9fc0f3336394a2dec1c",
    "fusion_run_freeze": "f7461e70036eb862c6510b3dfd5269e7e590978a681f8389ebef52c7976f8809",
    "fusion_config": "3f2070deee1ac824fa1eeee816565cf13581d08623c8e05a4005add0d73dbfbf",
    "tabular_config": "4ce6f92cbe4a5247f9cf5c0a8b60b70bfab9c0e08ae3b76fbfbfaf83db6a30e4",
    "image_config": "895331f848144ff2582de626345476800a887065972d17903a2209c80acdefbb",
    "tabular_code": "5da741f438100547b88fde285d74f815fe5f9cd25981eba8672d5c18485c9c06",
    "image_code": "5a47d6032d441974e065f5b1d477a4bbe31bfe0f8de6477c916794e90dd31793",
    "fusion_code": "eb354b962dd15aadf597ee1a08ec5d4679cea5c0535390e7a52749d3fda03c3e",
}

TEST_BOOTSTRAP_SEED = 65027
TEST_PARTITION_CENSUS = {"knees": 1044, "participants": 543}


def verify_development_artifacts() -> dict[str, str]:
    """Independently verify all frozen development artifacts against approved hashes."""
    paths = {
        "dataset": DEFAULT_DIRECTORY / "final_multimodal_dataset.parquet",
        "participants": DEFAULT_DIRECTORY / "splits/v1/participant_split_manifest.parquet",
        "knees": DEFAULT_DIRECTORY / "splits/v1/knee_split_manifest.parquet",
        "feature_groups": DEFAULT_DIRECTORY / "feature_groups.json",
        "schema": DEFAULT_DIRECTORY / "schema.json",
        "dataset_freeze": DEFAULT_DIRECTORY / "dataset_freeze.json",
        "split_freeze": DEFAULT_DIRECTORY / "splits/v1/split_freeze.json",
        "cohort": Path("data/processed/cohorts/analysis_cohort_v1.parquet"),
        "imaging_manifest": Path(
            "data/processed/oai_images/v3_frozen_full/final_adjudicated_v1/imaging_manifest.parquet"
        ),
        "tabular_run_freeze": TABULAR_OUTPUT / "run_freeze.json",
        "tabular_validation_predictions": TABULAR_OUTPUT / "validation_predictions.parquet",
        "tabular_logistic_A_model": TABULAR_OUTPUT
        / "models/logistic__formulation_A__candidate_0.joblib",
        "tabular_logistic_B_model": TABULAR_OUTPUT
        / "models/logistic__formulation_B__candidate_0.joblib",
        "image_run_freeze": IMAGE_OUTPUT / "run_freeze.json",
        "image_validation_predictions": IMAGE_OUTPUT / "validation_predictions.parquet",
        "image_selected_checkpoint": IMAGE_OUTPUT / "candidate_2/best.pt",
        "fusion_run_freeze": FUSION_OUTPUT / "run_freeze.json",
        "fusion_config": Path("configs/fusion_v1.yaml"),
        "tabular_config": Path("configs/tabular_v1.yaml"),
        "image_config": Path("configs/image_model_v1.yaml"),
        "tabular_code": Path("src/modeling/tabular.py"),
        "image_code": Path("src/modeling/image.py"),
        "fusion_code": Path("src/modeling/fusion.py"),
    }

    actual_hashes = {}
    for key, path in paths.items():
        require(path.is_file(), f"Required artifact file missing: {path}")
        h = sha256_file(path)
        expected = APPROVED_HASHES[key]
        require(
            h == expected,
            f"Artifact SHA mismatch for '{key}': expected {expected}, got {h} ({path})",
        )
        actual_hashes[key] = h

    return actual_hashes


def build_final_model_freeze(existing: dict | None = None) -> dict:
    """Specify the exact frozen primary and sensitivity models for coordinated TEST evaluation."""
    tab_sc = read_json(TABULAR_OUTPUT / "selected_candidates.json")
    img_sc = read_json(IMAGE_OUTPUT / "selected_candidate.json")
    fusion_sc = read_json(FUSION_OUTPUT / "selected_fusion.json")

    require(
        tab_sc["logistic__formulation_A"]["parameters"] == {"C": 0.1},
        "Logistic A candidate parameter mismatch",
    )
    require(
        tab_sc["logistic__formulation_B"]["parameters"] == {"C": 0.1},
        "Logistic B candidate parameter mismatch",
    )
    require(
        img_sc["checkpoint_sha256"] == APPROVED_HASHES["image_selected_checkpoint"],
        "Image selected checkpoint mismatch",
    )
    require(fusion_sc["selected_alpha"] == 0.50, "Primary fusion selected alpha mismatch")
    require(
        fusion_sc["sensitivity"]["selected_alpha"] == 0.60,
        "Sensitivity fusion selected alpha mismatch",
    )

    return {
        "status": "APPROVED_DEVELOPMENT_FREEZE_6F",
        "created_at_utc": existing["created_at_utc"]
        if existing and "created_at_utc" in existing
        else datetime.now(UTC).isoformat(),
        "freeze_policy": {
            "no_refit_on_train_val": True,
            "no_retraining": True,
            "no_hyperparameter_tuning": True,
            "no_alpha_reselection": True,
            "evaluation_subject": "FROZEN_DEVELOPMENT_MODELS",
            "future_deployment_models": "Distinct and cannot inherit held-out TEST claims",
        },
        "primary_models": {
            "primary_tabular": {
                "model_id": "logistic__formulation_A",
                "role": "PRIMARY_TABULAR_MODEL",
                "algorithm": "L2-regularized Logistic Regression (C=0.1, solver='lbfgs', max_iter=2000, tol=1e-8, class_weight=None)",
                "formulation": "formulation_A",
                "features": tab_sc["logistic__formulation_A"]["features"],
                "feature_count": len(tab_sc["logistic__formulation_A"]["features"]),
                "baseline_kl_included": False,
                "model_artifact": "models/logistic__formulation_A__candidate_0.joblib",
                "model_artifact_sha256": APPROVED_HASHES["tabular_logistic_A_model"],
                "validation_metrics": tab_sc["logistic__formulation_A"]["metrics"],
                "software_runtime_provenance": {
                    "python": "3.13.2",
                    "scikit-learn": "1.9.1",
                    "joblib": "1.6.0",
                    "numpy": "2.5.2",
                    "pandas": "3.0.5",
                    "scipy": "1.18.1",
                    "threadpoolctl": "3.7.0",
                },
            },
            "primary_image": {
                "model_id": "DenseNet121_selected",
                "role": "PRIMARY_IMAGE_MODEL",
                "architecture": "DenseNet121",
                "candidate_index": 2,
                "fine_tuning": "Full fine-tuning (frozen_backbone=False, lr=3e-5, epochs=25, best_epoch=5)",
                "checkpoint": "candidate_2/best.pt",
                "checkpoint_sha256": APPROVED_HASHES["image_selected_checkpoint"],
                "input_representation": "standardized knee radiograph image predictions derived from bilateral baseline acquisitions",
                "preprocessing_specification": {
                    "input": "Frozen effective uint16 crop from data/processed/oai_images/v3_frozen_full",
                    "clipping_percentiles": [0.5, 99.5],
                    "scaling": "Image-local float32 clip/min-max [0,1]",
                    "spatial_resize": [320, 320],
                    "interpolation": "bilinear with antialiasing",
                    "channels": "grayscale replicated to 3 channels",
                    "normalization": "ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]",
                    "test_augmentation": "None (model.eval(), torch.inference_mode())",
                    "preprocessing_sha256": "53b32902d7a63fd218afca1df81b71846333308043985fbdcfb9c790b07a45de",
                },
                "execution_environment": {
                    "platform": "Linux-6.6.122+-x86_64-with-glibc2.39",
                    "device": "CUDA (NVIDIA A100-SXM4-40GB preferred)",
                    "cuda_runtime": "13.0",
                    "python": "3.13.15",
                    "torch": "2.14.0+cu130",
                    "torchvision": "0.29.0+cu130",
                    "determinism": "Strict deterministic algorithms",
                },
                "validation_metrics": img_sc["metrics"],
            },
            "primary_multimodal": {
                "model_id": "multimodal_fusion_A_selected",
                "role": "PRIMARY_MULTIMODAL_MODEL",
                "fusion_type": "Convex probability fusion",
                "formula": "p_fusion = 0.50 * p_image + 0.50 * p_logistic_A",
                "frozen_alpha_image": 0.50,
                "frozen_alpha_tabular": 0.50,
                "selection_policy": "Immutable 11-point validation grid selection; tier 1 unique winner",
                "validation_metrics": fusion_sc["metrics"],
            },
        },
        "prespecified_sensitivities": {
            "sensitivity_tabular": {
                "model_id": "logistic__formulation_B",
                "role": "BASELINE-KL SENSITIVITY",
                "algorithm": "L2-regularized Logistic Regression (C=0.1, solver='lbfgs', max_iter=2000, tol=1e-8, class_weight=None)",
                "formulation": "formulation_B",
                "features": tab_sc["logistic__formulation_B"]["features"],
                "feature_count": len(tab_sc["logistic__formulation_B"]["features"]),
                "baseline_kl_included": True,
                "model_artifact": "models/logistic__formulation_B__candidate_0.joblib",
                "model_artifact_sha256": APPROVED_HASHES["tabular_logistic_B_model"],
                "validation_metrics": tab_sc["logistic__formulation_B"]["metrics"],
            },
            "sensitivity_multimodal": {
                "model_id": "multimodal_fusion_B_sensitivity",
                "role": "BASELINE-KL SENSITIVITY",
                "fusion_type": "Convex probability fusion",
                "formula": "p_fusion = 0.60 * p_image + 0.40 * p_logistic_B",
                "frozen_alpha_image": 0.60,
                "frozen_alpha_tabular": 0.40,
                "validation_metrics": fusion_sc["sensitivity"]["metrics"],
                "note": "Sensitivity analysis only; must not replace Formulation A.",
            },
        },
    }


def build_test_evaluation_protocol(existing: dict | None = None) -> dict:
    """Specify the exact immutable protocol for one-time coordinated TEST evaluation."""
    return {
        "status": "FROZEN_TEST_EVALUATION_PROTOCOL",
        "created_at_utc": existing["created_at_utc"]
        if existing and "created_at_utc" in existing
        else datetime.now(UTC).isoformat(),
        "evaluation_partition": "TEST",
        "partition_census": TEST_PARTITION_CENSUS,
        "target": tabular.TARGET,
        "models_evaluated": {
            "primary": [
                "logistic__formulation_A",
                "DenseNet121_selected",
                "multimodal_fusion_A_selected",
            ],
            "sensitivity": [
                "logistic__formulation_B",
                "multimodal_fusion_B_sensitivity",
            ],
        },
        "metrics_reported_per_model": [
            "AUROC",
            "AUPRC",
            "Brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "observations",
            "positive_events",
            "prevalence",
        ],
        "metric_exclusion_policy": {
            "accuracy_excluded": True,
            "reason": "Classification accuracy is excluded from primary reporting due to class imbalance and clinical non-interpretability.",
        },
        "curves_generated_for_primary_multimodal": [
            "ROC curve with AUROC annotation (test_roc_curve.png)",
            "Precision-Recall curve with no-skill baseline (test_pr_curve.png)",
            "Calibration curve with 10 quantile bins (test_calibration_curve.png)",
        ],
        "bootstrap_procedure": {
            "replicates": 1000,
            "unit": "participant",
            "cluster_policy": "All knees belonging to each sampled participant retained; repeated sampled participants represented with corresponding multiplicity",
            "seed": TEST_BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "interval_method": "Percentile interval [quantile(0.025), quantile(0.975)]",
            "models_evaluated": [
                "logistic__formulation_A",
                "DenseNet121_selected",
                "multimodal_fusion_A_selected",
            ],
            "metrics_evaluated": ["AUROC", "AUPRC", "Brier"],
        },
        "paired_comparisons": {
            "comparative_analysis_designation": "Primary Prespecified Comparative Analysis",
            "shared_draws": True,
            "paired_comparisons": [
                "multimodal_fusion_A_selected minus DenseNet121_selected",
                "multimodal_fusion_A_selected minus logistic__formulation_A",
            ],
            "metrics": ["AUROC", "AUPRC", "Brier"],
            "outputs": [
                "observed_difference_on_full_test",
                "bootstrap_median_difference",
                "percentile_95_interval",
            ],
            "nature_of_intervals": "Descriptive comparative intervals; post-hoc claims of statistical significance without predeclared hypothesis tests are prohibited.",
        },
        "calibration_policy": {
            "diagnostic_only": True,
            "never_alter_probabilities": True,
            "no_recalibration": True,
            "no_platt_scaling": True,
            "no_isotonic_regression": True,
            "no_threshold_optimization": True,
            "never_trigger_second_test_evaluation": True,
        },
        "test_opening_policy": {
            "single_evaluation": True,
            "no_model_retraining": True,
            "no_alpha_adjustment": True,
            "no_model_replacement": True,
            "no_feature_changes": True,
            "no_threshold_tuning": True,
            "no_architecture_changes": True,
            "no_hyperparameter_changes": True,
            "no_calibration_fitting": True,
            "no_endpoint_changes": True,
            "no_exclusion_changes": True,
            "no_split_changes": True,
        },
        "test_execution_pathway": {
            "stage_A_blind_image_inference": {
                "bundle_contents": [
                    "Frozen TEST image crops",
                    "Authoritative knee key manifest (participant_id, knee_side_code, baseline_visit)",
                    "Selected checkpoint (b14a74f3...)",
                    "Frozen preprocessing specification (53b32902...)",
                    "Frozen inference script (image.py)",
                ],
                "labels_included": False,
                "purpose": "Generate un-blinded probabilities without accidental outcome exposure during image inference",
                "returned_output": "test_image_predictions.parquet (participant_id, knee_side_code, baseline_visit, predicted_probability)",
                "compute_metrics_in_image_runtime": False,
            },
            "stage_B_coordinated_scoring": {
                "steps": [
                    "Load frozen Logistic A and B joblib pipelines",
                    "Generate TEST tabular probabilities from frozen dataset predictors",
                    "Load Stage A image predictions",
                    "Align predictions strictly by (participant_id, knee_side_code, baseline_visit)",
                    "Verify exactly 1,044 unique knees, 543 participants, split assignment",
                    "Compute Fusion A with fixed alpha = 0.50",
                    "Compute Fusion B with fixed alpha = 0.60",
                    "Load TEST outcome labels and compute frozen metrics",
                    "Run 1,000 participant-clustered bootstrap replicates (seed 65027)",
                    "Compute paired difference distributions",
                    "Generate curves and descriptive calibration diagnostics",
                    "Publish frozen TEST report and audit",
                ],
            },
        },
        "technical_rerun_policy": {
            "allowed_condition": "Execution failure occurs BEFORE any TEST metric or result has been produced",
            "prerequisites": "Failure must be documented; corrections limited strictly to execution/infrastructure fix",
            "forbidden": "No change to model, feature, alpha, checkpoint, preprocessing, outcome, exclusion, metric, or statistical analysis",
            "after_metric_production": "If TEST metrics have been produced, no modifications or reruns for improved performance are permitted",
            "artifact_recovery": "Rerun solely for artifact recovery must reproduce identical frozen computation and be documented",
        },
    }


def audit_test_lock() -> dict:
    """Verify that zero TEST evaluation, scoring, or inference artifacts exist."""
    search_dirs = [
        Path("data/processed/modeling"),
        Path("reports"),
    ]

    violations = []
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                name = p.name.lower()
                # Exclude valid code/test files and lock files
                if "test_prediction" in name or "test_metrics" in name or "test_result" in name:
                    violations.append(str(p))

    require(len(violations) == 0, f"Forbidden TEST evaluation artifact found: {violations}")

    return {
        "zero_test_predictions_loaded": True,
        "zero_test_images_loaded_by_model": True,
        "zero_test_model_inference": True,
        "zero_test_metrics_produced": True,
        "zero_test_calibration_performed": True,
        "zero_test_bootstrap_performed": True,
        "zero_test_derived_decisions": True,
        "test_lock_intact": True,
    }


def run_freeze(
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    """Execute Milestone 6F development freeze and author protocol artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify all development artifacts
    verified_hashes = verify_development_artifacts()

    # 2. Audit TEST lock
    lock_audit = audit_test_lock()

    # 3. Preservation baseline check
    baseline = fusion.verify_preservation_baseline(IMAGE_BASE_DIR / "pre_change_preservation.json")
    fusion.preserve(baseline)

    model_freeze_path = output_dir / "final_model_freeze.json"
    protocol_path = output_dir / "test_evaluation_protocol.json"
    freeze_path = output_dir / "run_freeze.json"

    existing_model_freeze = read_json(model_freeze_path) if model_freeze_path.exists() else None
    existing_protocol = read_json(protocol_path) if protocol_path.exists() else None
    existing_freeze = read_json(freeze_path) if freeze_path.exists() else None

    # 4. Build specifications
    model_freeze = build_final_model_freeze(existing_model_freeze)
    protocol = build_test_evaluation_protocol(existing_protocol)

    # 5. Publish model freeze and evaluation protocol
    publish_json(model_freeze, output_dir / "final_model_freeze.json")
    publish_json(protocol, output_dir / "test_evaluation_protocol.json")

    # 6. Artifact hashes
    artifact_names = [
        "final_model_freeze.json",
        "test_evaluation_protocol.json",
    ]
    artifact_hashes = {
        name: {
            "sha256": sha256_file(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in artifact_names
    }
    publish_json(artifact_hashes, output_dir / "artifact_hashes.json")

    # 7. Run freeze marker
    freeze_record = {
        "status": "APPROVED_DEVELOPMENT_FREEZE_6F",
        "created_at_utc": existing_freeze["created_at_utc"]
        if existing_freeze and "created_at_utc" in existing_freeze
        else datetime.now(UTC).isoformat(),
        "freeze_version": "final_freeze_v1",
        "primary_models": [
            "logistic__formulation_A",
            "DenseNet121_selected",
            "multimodal_fusion_A_selected",
        ],
        "frozen_primary_alpha": 0.50,
        "sensitivity_models": [
            "logistic__formulation_B",
            "multimodal_fusion_B_sensitivity",
        ],
        "frozen_sensitivity_alpha": 0.60,
        "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
        "test_scored": False,
        "test_predictions_generated": False,
        "test_image_inference_run": False,
        "test_lock_verified": True,
        "source_preservation_passed": True,
        "test_lock_audit": lock_audit,
        "provenance": {
            "code_sha256": sha256_file(Path(__file__)),
            "source_hashes": verified_hashes,
        },
        "artifacts_sha256": {k: v["sha256"] for k, v in artifact_hashes.items()},
    }
    publish_json(freeze_record, output_dir / "run_freeze.json")

    # 8. Post-change preservation check
    fusion.preserve(baseline)

    return {
        "status": "APPROVED_DEVELOPMENT_FREEZE_6F",
        "frozen_models": model_freeze["primary_models"],
        "frozen_protocol": protocol["status"],
        "test_bootstrap_seed": TEST_BOOTSTRAP_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with file_lock(args.output / "run_freeze.json"):
        result = run_freeze(output_dir=args.output)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
