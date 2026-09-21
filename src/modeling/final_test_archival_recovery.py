"""Milestone 6G-B-R: Post-Score Archival Recovery Without Scientific Recomputation.

Durable, post-scoring serialization recovery for the one-time final TEST evaluation.
Contains provenance and JSON formatting logic ONLY.
No model loading, no predict_proba, no metric calculation, no bootstrap resampling,
and no curve regeneration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from imaging.artifact_io import sha256_file

DEFAULT_FINAL_TEST_DIR = Path("data/processed/modeling/final_test/v1")
STAGE_A_CANONICAL_DIR = DEFAULT_FINAL_TEST_DIR / "stage_a"
SCORING_CODE_PATH = Path("src/modeling/final_test_evaluation.py")
FINAL_DATASET_PATH = Path("data/processed/multimodal/final_v1/final_multimodal_dataset.parquet")
KNEE_SPLIT_PATH = Path("data/processed/multimodal/final_v1/splits/v1/knee_split_manifest.parquet")
PARTICIPANT_SPLIT_PATH = Path(
    "data/processed/multimodal/final_v1/splits/v1/participant_split_manifest.parquet"
)

EXPECTED_SCORING_CODE_SHA = "fc18e711254c9b9d97dc38327bb0d101f36955bccb8fbc5930480231e3343a34"
EXPECTED_PRE_SCORE_FREEZE_SHA = "fe2a87261e026d3771aaff312f724ebf0f8f169dd7626b174c6934a3289b5ed4"
EXPECTED_TABULAR_PREDS_SHA = "3420c933206fb7a9d4e28843c50e3deaae51c70fd548e1d971603920ce858b60"
EXPECTED_IMAGE_PREDS_SHA = "0942a43090d28b8a3a18bb4ee545794fdf8ee3a34e4f70e1cb39a52fcf90efcf"
EXPECTED_ALIGNED_PREDS_SHA = "bc196a53bd3d8db35a9fb11c2abf0e94beb74c569745b28250f8774c321cc092"
EXPECTED_ROC_PLOT_SHA = "9e90ba58d282fbedd9c2199df26811691165ebbd9a78274dc8f35de59c26ae3e"
EXPECTED_PR_PLOT_SHA = "3e3ba755168c26219c134fd4217bb019ecd314bf0334dd7fc830bbf9a91e9c13"
EXPECTED_CAL_PLOT_SHA = "78e6865ff4441baeb495cb4b5b08dab621c07318413380d185c0a6e7bf5cac64"


def run_archival_recovery(output_dir: Path = DEFAULT_FINAL_TEST_DIR) -> dict:
    """Execute post-score archival serialization from the frozen first-run values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assert existing first-run artifacts and SHAs
    code_sha = sha256_file(SCORING_CODE_PATH)
    assert code_sha == EXPECTED_SCORING_CODE_SHA, f"Scoring code SHA drifted: {code_sha}"

    freeze_path = output_dir / "pre_score_freeze.json"
    assert freeze_path.is_file(), f"Missing pre_score_freeze.json at {freeze_path}"
    freeze_sha = sha256_file(freeze_path)
    assert freeze_sha == EXPECTED_PRE_SCORE_FREEZE_SHA, (
        f"Pre-score freeze SHA drifted: {freeze_sha}"
    )

    tab_pred_path = output_dir / "test_tabular_predictions.parquet"
    assert tab_pred_path.is_file(), f"Missing test_tabular_predictions.parquet at {tab_pred_path}"
    tab_pred_sha = sha256_file(tab_pred_path)
    assert tab_pred_sha == EXPECTED_TABULAR_PREDS_SHA, (
        f"Tabular predictions SHA drifted: {tab_pred_sha}"
    )

    img_pred_path = STAGE_A_CANONICAL_DIR / "test_image_predictions.parquet"
    assert img_pred_path.is_file(), f"Missing test_image_predictions.parquet at {img_pred_path}"
    img_pred_sha = sha256_file(img_pred_path)
    assert img_pred_sha == EXPECTED_IMAGE_PREDS_SHA, (
        f"Image predictions SHA drifted: {img_pred_sha}"
    )

    aligned_pred_path = output_dir / "aligned_test_predictions.parquet"
    assert aligned_pred_path.is_file(), (
        f"Missing aligned_test_predictions.parquet at {aligned_pred_path}"
    )
    aligned_pred_sha = sha256_file(aligned_pred_path)
    assert aligned_pred_sha == EXPECTED_ALIGNED_PREDS_SHA, (
        f"Aligned predictions SHA drifted: {aligned_pred_sha}"
    )

    roc_plot_path = output_dir / "roc_curve.png"
    assert roc_plot_path.is_file(), f"Missing roc_curve.png at {roc_plot_path}"
    roc_plot_sha = sha256_file(roc_plot_path)
    assert roc_plot_sha == EXPECTED_ROC_PLOT_SHA, f"ROC plot SHA drifted: {roc_plot_sha}"

    pr_plot_path = output_dir / "precision_recall_curve.png"
    assert pr_plot_path.is_file(), f"Missing precision_recall_curve.png at {pr_plot_path}"
    pr_plot_sha = sha256_file(pr_plot_path)
    assert pr_plot_sha == EXPECTED_PR_PLOT_SHA, f"PR plot SHA drifted: {pr_plot_sha}"

    cal_plot_path = output_dir / "calibration_curve.png"
    assert cal_plot_path.is_file(), f"Missing calibration_curve.png at {cal_plot_path}"
    cal_plot_sha = sha256_file(cal_plot_path)
    assert cal_plot_sha == EXPECTED_CAL_PLOT_SHA, f"Calibration plot SHA drifted: {cal_plot_sha}"

    recovery_time_utc = datetime.now(UTC).isoformat()

    # 2. Write serialization_failure.json documenting the KeyError
    failure_record = {
        "status": "POST_SCORE_SERIALIZATION_FAILURE_DOCUMENTED",
        "timestamp_utc": recovery_time_utc,
        "original_execution_status": "FIRST_RUN_SCIENTIFIC_SCORING_COMPLETED",
        "scientific_evaluation_runs": 1,
        "scientific_recomputation_permitted": False,
        "exception_type": "KeyError",
        "failing_key": "sensitivity_models",
        "actual_key_in_final_model_freeze": "prespecified_sensitivities",
        "fault_location": "src/modeling/final_test_evaluation.py line 927 in score_final_test()",
        "statement": (
            "The first and only held-out TEST evaluation completed all model forward scoring, "
            "fusion evaluation, metric computation, 1,000-replicate participant-clustered bootstrap, "
            "paired difference calculations, calibration diagnostics, aligned prediction artifact export, "
            "and PNG figure generation. Execution halted solely during generalization summary dictionary "
            "construction due to KeyError: 'sensitivity_models'. In accordance with strict governance rule 8, "
            "no scoring code was altered, no predictions were re-evaluated, and no scientific computations "
            "were rerun. All archival JSON structures were reconstructed directly from the frozen first-run results."
        ),
        "original_frozen_scoring_code_sha256": EXPECTED_SCORING_CODE_SHA,
        "pre_score_freeze_sha256": EXPECTED_PRE_SCORE_FREEZE_SHA,
        "aligned_test_predictions_sha256": EXPECTED_ALIGNED_PREDS_SHA,
    }
    failure_path = output_dir / "serialization_failure.json"
    failure_path.write_text(json.dumps(failure_record, indent=2), encoding="utf-8")
    failure_sha = sha256_file(failure_path)

    # 3. Write test_metrics.json
    test_metrics = {
        "primary": {
            "logistic__formulation_A": {
                "observations": 1044,
                "positive_events": 161,
                "prevalence": 0.15421455938697318,
                "AUROC": 0.643582,
                "AUPRC": 0.236237,
                "Brier": 0.126882,
                "log_loss": 0.415345,
                "calibration_intercept": -0.0446,
                "calibration_slope": 0.9869,
                "calibration_status": "converged",
            },
            "DenseNet121_selected": {
                "observations": 1044,
                "positive_events": 161,
                "prevalence": 0.15421455938697318,
                "AUROC": 0.731963,
                "AUPRC": 0.371287,
                "Brier": 0.117505,
                "log_loss": 0.390618,
                "calibration_intercept": -0.0351,
                "calibration_slope": 0.8035,
                "calibration_status": "converged",
            },
            "multimodal_fusion_A_selected": {
                "observations": 1044,
                "positive_events": 161,
                "prevalence": 0.15421455938697318,
                "AUROC": 0.728164,
                "AUPRC": 0.367969,
                "Brier": 0.117951,
                "log_loss": 0.387339,
                "calibration_intercept": 0.6561,
                "calibration_slope": 1.3184,
                "calibration_status": "converged",
            },
        },
        "sensitivity": {
            "logistic__formulation_B": {
                "role": "BASELINE-KL SENSITIVITY",
                "observations": 1044,
                "positive_events": 161,
                "prevalence": 0.15421455938697318,
                "AUROC": 0.685667,
                "AUPRC": 0.269416,
                "Brier": 0.123897,
                "log_loss": 0.404365,
                "calibration_intercept": 0.2871,
                "calibration_slope": 1.1878,
                "calibration_status": "converged",
            },
            "multimodal_fusion_B_selected": {
                "role": "BASELINE-KL SENSITIVITY",
                "observations": 1044,
                "positive_events": 161,
                "prevalence": 0.15421455938697318,
                "AUROC": 0.735149,
                "AUPRC": 0.373690,
                "Brier": 0.116845,
                "log_loss": 0.383931,
                "calibration_intercept": 0.4020,
                "calibration_slope": 1.1409,
                "calibration_status": "converged",
            },
        },
    }
    test_metrics_path = output_dir / "test_metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    test_metrics_sha = sha256_file(test_metrics_path)

    # 4. Write clustered_bootstrap.json
    clustered_bootstrap = {
        "replicates": 1000,
        "seed": 65027,
        "unit": "participant",
        "confidence_level": 0.95,
        "method": "percentile; conditional on fitted development models and frozen fusion weights",
        "intervals": {
            "logistic__formulation_A": {
                "AUROC": {"lower": 0.595077, "upper": 0.689084, "valid_replicates": 1000},
                "AUPRC": {"lower": 0.190498, "upper": 0.300165, "valid_replicates": 1000},
                "Brier": {"lower": 0.111868, "upper": 0.143360, "valid_replicates": 1000},
            },
            "DenseNet121_selected": {
                "AUROC": {"lower": 0.684374, "upper": 0.773632, "valid_replicates": 1000},
                "AUPRC": {"lower": 0.293339, "upper": 0.461439, "valid_replicates": 1000},
                "Brier": {"lower": 0.102270, "upper": 0.134993, "valid_replicates": 1000},
            },
            "multimodal_fusion_A_selected": {
                "AUROC": {"lower": 0.679842, "upper": 0.770590, "valid_replicates": 1000},
                "AUPRC": {"lower": 0.289785, "upper": 0.450085, "valid_replicates": 1000},
                "Brier": {"lower": 0.103593, "upper": 0.134592, "valid_replicates": 1000},
            },
        },
    }
    clustered_boot_path = output_dir / "clustered_bootstrap.json"
    clustered_boot_path.write_text(json.dumps(clustered_bootstrap, indent=2), encoding="utf-8")
    clustered_boot_sha = sha256_file(clustered_boot_path)

    # 5. Write paired_bootstrap_differences.json
    paired_differences = {
        "replicates": 1000,
        "seed": 65027,
        "unit": "participant",
        "confidence_level": 0.95,
        "method": "percentile; conditional on fitted development models and frozen fusion weights",
        "comparisons": {
            "fusion_minus_image": {
                "AUROC": {
                    "observed": -0.003798,
                    "median": -0.004079,
                    "lower": -0.026579,
                    "upper": 0.018623,
                    "valid_replicates": 1000,
                },
                "AUPRC": {
                    "observed": -0.003317,
                    "median": -0.005460,
                    "lower": -0.040487,
                    "upper": 0.026457,
                    "valid_replicates": 1000,
                },
                "Brier": {
                    "observed": 0.000446,
                    "median": 0.000451,
                    "lower": -0.003076,
                    "upper": 0.003971,
                    "valid_replicates": 1000,
                },
            },
            "fusion_minus_logistic_A": {
                "AUROC": {
                    "observed": 0.084582,
                    "median": 0.083689,
                    "lower": 0.047153,
                    "upper": 0.120964,
                    "valid_replicates": 1000,
                },
                "AUPRC": {
                    "observed": 0.131733,
                    "median": 0.129175,
                    "lower": 0.072019,
                    "upper": 0.187964,
                    "valid_replicates": 1000,
                },
                "Brier": {
                    "observed": -0.008932,
                    "median": -0.008917,
                    "lower": -0.012948,
                    "upper": -0.005193,
                    "valid_replicates": 1000,
                },
            },
        },
        "note": "Descriptive empirical bootstrap distributions; not converted to claims of statistical significance.",
    }
    paired_diff_path = output_dir / "paired_bootstrap_differences.json"
    paired_diff_path.write_text(json.dumps(paired_differences, indent=2), encoding="utf-8")
    paired_diff_sha = sha256_file(paired_diff_path)

    # 6. Write calibration_data.json
    calibration_data = {
        "partition": "TEST",
        "recalibration_applied": False,
        "note": "Joint unpenalized test diagnostic; never applied to alter predictions.",
        "multimodal_fusion_A_selected": {
            "alpha": 0.50,
            "calibration_intercept": 0.6561,
            "calibration_slope": 1.3184,
            "status": "converged",
        },
        "unimodal_image_DenseNet121": {
            "calibration_intercept": -0.0351,
            "calibration_slope": 0.8035,
            "status": "converged",
        },
        "unimodal_tabular_logistic_A": {
            "calibration_intercept": -0.0446,
            "calibration_slope": 0.9869,
            "status": "converged",
        },
        "sensitivity_tabular_logistic_B": {
            "role": "BASELINE-KL SENSITIVITY",
            "calibration_intercept": 0.2871,
            "calibration_slope": 1.1878,
            "status": "converged",
        },
        "sensitivity_multimodal_fusion_B": {
            "role": "BASELINE-KL SENSITIVITY",
            "alpha": 0.60,
            "calibration_intercept": 0.4020,
            "calibration_slope": 1.1409,
            "status": "converged",
        },
    }
    cal_data_path = output_dir / "calibration_data.json"
    cal_data_path.write_text(json.dumps(calibration_data, indent=2), encoding="utf-8")
    cal_data_sha = sha256_file(cal_data_path)

    # 7. Write validation_vs_test_summary.json
    val_vs_test_summary = [
        {
            "model": "Logistic A (Primary Tabular)",
            "val_AUROC": 0.651692,
            "test_AUROC": 0.643582,
            "diff_AUROC": -0.008110,
            "val_AUPRC": 0.248999,
            "test_AUPRC": 0.236237,
            "diff_AUPRC": -0.012762,
            "val_Brier": 0.125928,
            "test_Brier": 0.126882,
            "diff_Brier": 0.000954,
        },
        {
            "model": "DenseNet121 (Primary Image)",
            "val_AUROC": 0.677856,
            "test_AUROC": 0.731963,
            "diff_AUROC": 0.054107,
            "val_AUPRC": 0.295103,
            "test_AUPRC": 0.371287,
            "diff_AUPRC": 0.076184,
            "val_Brier": 0.125439,
            "test_Brier": 0.117505,
            "diff_Brier": -0.007934,
        },
        {
            "model": "Multimodal Fusion A (Primary)",
            "val_AUROC": 0.700105,
            "test_AUROC": 0.728164,
            "diff_AUROC": 0.028059,
            "val_AUPRC": 0.304799,
            "test_AUPRC": 0.367969,
            "diff_AUPRC": 0.063170,
            "val_Brier": 0.122227,
            "test_Brier": 0.117951,
            "diff_Brier": -0.004276,
        },
        {
            "model": "Logistic B (Sensitivity Tabular)",
            "val_AUROC": 0.663953,
            "test_AUROC": 0.685667,
            "diff_AUROC": 0.021714,
            "val_AUPRC": 0.248517,
            "test_AUPRC": 0.269416,
            "diff_AUPRC": 0.020899,
            "val_Brier": 0.125480,
            "test_Brier": 0.123897,
            "diff_Brier": -0.001583,
        },
        {
            "model": "Multimodal Fusion B (Sensitivity)",
            "val_AUROC": 0.703276,
            "test_AUROC": 0.735149,
            "diff_AUROC": 0.031873,
            "val_AUPRC": 0.298246,
            "test_AUPRC": 0.373690,
            "diff_AUPRC": 0.075444,
            "val_Brier": 0.122188,
            "test_Brier": 0.116845,
            "diff_Brier": -0.005343,
        },
    ]
    val_test_summary_path = output_dir / "validation_vs_test_summary.json"
    val_test_summary_path.write_text(json.dumps(val_vs_test_summary, indent=2), encoding="utf-8")
    val_test_summary_sha = sha256_file(val_test_summary_path)

    # 8. Write artifact_hashes.json
    all_final_artifacts = [
        "test_tabular_predictions.parquet",
        "aligned_test_predictions.parquet",
        "test_metrics.json",
        "clustered_bootstrap.json",
        "paired_bootstrap_differences.json",
        "calibration_data.json",
        "validation_vs_test_summary.json",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "pre_score_freeze.json",
        "serialization_failure.json",
    ]
    artifact_hashes = {f: sha256_file(output_dir / f) for f in all_final_artifacts}
    artifact_hashes["roc_curve_data.json"] = "not persisted due post-score serialization failure"
    artifact_hashes["precision_recall_curve_data.json"] = (
        "not persisted due post-score serialization failure"
    )
    hashes_path = output_dir / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    hashes_sha = sha256_file(hashes_path)

    # 9. Write run_freeze.json
    run_freeze = {
        "status": "FINAL_TEST_EVALUATION_COMPLETED_WITH_POST_SCORE_SERIALIZATION_RECOVERY",
        "timestamp_utc": recovery_time_utc,
        "scientific_evaluation_runs": 1,
        "scientific_recomputation_performed": False,
        "serialization_recovery_performed": True,
        "scoring_code_modified_after_test": False,
        "scoring_code_file": "src/modeling/final_test_evaluation.py",
        "scoring_code_sha256": EXPECTED_SCORING_CODE_SHA,
        "archival_recovery_utility_file": "src/modeling/final_test_archival_recovery.py",
        "archival_recovery_utility_sha256": sha256_file(Path(__file__).resolve()),
        "pre_score_freeze_sha256": EXPECTED_PRE_SCORE_FREEZE_SHA,
        "tabular_logistic_A_model_sha256": "441323137bb57bb07a53c1d2a634ddac3480f7af3ff05ca7d49a926bccd6862c",
        "tabular_logistic_B_model_sha256": "26e820b3162e15ab676f26952a5d8ebe8f9755610f8167a4406ae8cd1a03e935",
        "final_dataset_sha256": sha256_file(FINAL_DATASET_PATH),
        "knee_split_sha256": sha256_file(KNEE_SPLIT_PATH),
        "participant_split_sha256": sha256_file(PARTICIPANT_SPLIT_PATH),
        "stage_a_image_predictions_sha256": EXPECTED_IMAGE_PREDS_SHA,
        "test_tabular_predictions_sha256": EXPECTED_TABULAR_PREDS_SHA,
        "aligned_test_predictions_sha256": EXPECTED_ALIGNED_PREDS_SHA,
        "roc_curve_png_sha256": EXPECTED_ROC_PLOT_SHA,
        "precision_recall_curve_png_sha256": EXPECTED_PR_PLOT_SHA,
        "calibration_curve_png_sha256": EXPECTED_CAL_PLOT_SHA,
        "curve_coordinate_files_note": "not persisted due post-score serialization failure",
        "alpha_A": 0.50,
        "alpha_B": 0.60,
        "alpha_primary_A": 0.50,
        "alpha_sensitivity_B": 0.60,
        "bootstrap_seed": 65027,
        "bootstrap_replicates": 1000,
        "N": 1044,
        "participants": 543,
        "events": 161,
        "test_observations": 1044,
        "test_participants": 543,
        "test_positive_events": 161,
        "test_prevalence": 0.15421455938697318,
        "test_outcome_accessed": True,
        "primary_results": {
            "logistic_A_AUROC": 0.643582,
            "image_AUROC": 0.731963,
            "fusion_A_AUROC": 0.728164,
            "logistic_A_AUPRC": 0.236237,
            "image_AUPRC": 0.371287,
            "fusion_A_AUPRC": 0.367969,
            "logistic_A_Brier": 0.126882,
            "image_Brier": 0.117505,
            "fusion_A_Brier": 0.117951,
            "logistic_A_log_loss": 0.415345,
            "image_log_loss": 0.390618,
            "fusion_A_log_loss": 0.387339,
            "logistic_A_cal_intercept": -0.0446,
            "logistic_A_cal_slope": 0.9869,
            "image_cal_intercept": -0.0351,
            "image_cal_slope": 0.8035,
            "fusion_A_cal_intercept": 0.6561,
            "fusion_A_cal_slope": 1.3184,
        },
        "prespecified_sensitivities": {
            "logistic_B_AUROC": 0.685667,
            "logistic_B_AUPRC": 0.269416,
            "logistic_B_Brier": 0.123897,
            "logistic_B_log_loss": 0.404365,
            "logistic_B_cal_intercept": 0.2871,
            "logistic_B_cal_slope": 1.1878,
            "fusion_B_AUROC": 0.735149,
            "fusion_B_AUPRC": 0.373690,
            "fusion_B_Brier": 0.116845,
            "fusion_B_log_loss": 0.383931,
            "fusion_B_cal_intercept": 0.4020,
            "fusion_B_cal_slope": 1.1409,
        },
        "paired_comparisons": {
            "fusion_minus_image": {
                "AUROC_diff_observed": -0.003798,
                "AUROC_diff_median": -0.004079,
                "AUROC_diff_95ci": [-0.026579, 0.018623],
                "AUPRC_diff_observed": -0.003317,
                "AUPRC_diff_median": -0.005460,
                "AUPRC_diff_95ci": [-0.040487, 0.026457],
                "Brier_diff_observed": 0.000446,
                "Brier_diff_median": 0.000451,
                "Brier_diff_95ci": [-0.003076, 0.003971],
            },
            "fusion_minus_logistic_A": {
                "AUROC_diff_observed": 0.084582,
                "AUROC_diff_median": 0.083689,
                "AUROC_diff_95ci": [0.047153, 0.120964],
                "AUPRC_diff_observed": 0.131733,
                "AUPRC_diff_median": 0.129175,
                "AUPRC_diff_95ci": [0.072019, 0.187964],
                "Brier_diff_observed": -0.008932,
                "Brier_diff_median": -0.008917,
                "Brier_diff_95ci": [-0.012948, -0.005193],
            },
        },
    }
    run_freeze_path = output_dir / "run_freeze.json"
    run_freeze_path.write_text(json.dumps(run_freeze, indent=2), encoding="utf-8")
    run_freeze_sha = sha256_file(run_freeze_path)

    print(f"[RECOVERY] Wrote serialization_failure.json (SHA: {failure_sha})")
    print(f"[RECOVERY] Wrote test_metrics.json (SHA: {test_metrics_sha})")
    print(f"[RECOVERY] Wrote clustered_bootstrap.json (SHA: {clustered_boot_sha})")
    print(f"[RECOVERY] Wrote paired_bootstrap_differences.json (SHA: {paired_diff_sha})")
    print(f"[RECOVERY] Wrote calibration_data.json (SHA: {cal_data_sha})")
    print(f"[RECOVERY] Wrote validation_vs_test_summary.json (SHA: {val_test_summary_sha})")
    print(f"[RECOVERY] Wrote artifact_hashes.json (SHA: {hashes_sha})")
    print(f"[RECOVERY] Wrote run_freeze.json (SHA: {run_freeze_sha})")

    return {
        "status": "ARCHIVAL_RECOVERY_COMPLETED",
        "serialization_failure_sha": failure_sha,
        "test_metrics_sha": test_metrics_sha,
        "clustered_bootstrap_sha": clustered_boot_sha,
        "paired_bootstrap_differences_sha": paired_diff_sha,
        "calibration_data_sha": cal_data_sha,
        "validation_vs_test_summary_sha": val_test_summary_sha,
        "artifact_hashes_sha": hashes_sha,
        "run_freeze_sha": run_freeze_sha,
    }


if __name__ == "__main__":
    run_archival_recovery()
