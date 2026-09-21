"""Milestone 6E: Leakage-Safe Multimodal Fusion Development.

Combines baseline tabular information with selected DenseNet121 X-ray predictions
using a predeclared, leakage-safe convex probability fusion on VALIDATION only.
Zero TEST exposure, no new predictive model fitting, no image inference, and no
stacking/meta-models fitted on in-sample predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from imaging.artifact_io import file_lock, sha256_file
from modeling import image, tabular
from multimodal.freeze import (
    DEFAULT_DIRECTORY,
    load_config,
    publish_json,
    publish_parquet,
    read_json,
    require,
)

DEFAULT_OUTPUT = Path("data/processed/modeling/fusion/v1")
CONFIG_PATH = Path("configs/fusion_v1.yaml")
TABULAR_OUTPUT = Path("data/processed/modeling/tabular/v1")
IMAGE_OUTPUT = Path("data/processed/modeling/image/v1/production_cuda")
IMAGE_BASE_DIR = Path("data/processed/modeling/image/v1")

APPROVED = {
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
    "image_run_freeze": "f14ea1c6921e8e95b3637d374a78b35d2cb853a574ecb6b15bdad1ea5335582e",
    "image_validation_predictions": "c8f327fc63e845d564dfd85e139fb871609c3e67d910017b5f776b8453c60bbb",
    "image_selected_checkpoint": "b14a74f329a268ae6855a9dcf69f6850d38e31f7fb9bd9fc0f3336394a2dec1c",
}

EXPECTED_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTION_HIERARCHY = [
    "validation_AUROC_descending",
    "validation_AUPRC_descending",
    "validation_Brier_ascending",
    "validation_log_loss_ascending",
    "distance_to_half_ascending",
    "alpha_ascending",
]


def validate_config(config: dict) -> None:
    """Ensure immutable predeclared parameters and strict policy compliance."""
    require(config.get("version") == "fusion_primary_v1", "Config version mismatch")
    require(config.get("target") == tabular.TARGET, "Target mismatch")
    require(
        config.get("primary_formulation") == "formulation_A",
        "Primary formulation must be formulation_A",
    )
    require(
        config.get("sensitivity_formulation") == "formulation_B",
        "Sensitivity formulation must be formulation_B",
    )
    models = config.get("models", {})
    require(
        models.get("image") == "DenseNet121_selected"
        and models.get("tabular_primary") == "logistic__formulation_A"
        and models.get("tabular_sensitivity") == "logistic__formulation_B",
        "Configured model identifiers changed",
    )
    grid = [round(float(a), 2) for a in config.get("alpha_grid", [])]
    require(grid == EXPECTED_GRID, "Alpha grid must match immutable predeclared 11-point grid")
    require(
        config.get("selection") == SELECTION_HIERARCHY,
        "Selection hierarchy changed",
    )
    require(config.get("test_scoring_permitted") is False, "TEST scoring is strictly forbidden")
    require(
        config.get("bootstrap_replicates") == 1000 and config.get("bootstrap_seed") == 64027,
        "Bootstrap replicates or seed changed",
    )


def load_and_align_validation_predictions(
    tabular_predictions_path: Path,
    image_predictions_path: Path,
    knee_split_manifest_path: Path,
) -> pd.DataFrame:
    """Verify input predictions and strictly align on participant_id + knee_side_code + baseline_visit."""
    require(tabular_predictions_path.is_file(), "Tabular validation predictions file missing")
    require(image_predictions_path.is_file(), "Image validation predictions file missing")
    require(knee_split_manifest_path.is_file(), "Knee split manifest file missing")

    # Authoritative VALIDATION knees
    knee_manifest = pd.read_parquet(knee_split_manifest_path)
    val_manifest = (
        knee_manifest[knee_manifest["split"] == "VALIDATION"].copy().reset_index(drop=True)
    )
    require(len(val_manifest) == 1044, "Authoritative VALIDATION knees must count exactly 1044")
    require(
        val_manifest["participant_id"].nunique() == 543,
        "Authoritative VALIDATION participants must count exactly 543",
    )
    keys = ["participant_id", "knee_side_code", "baseline_visit"]
    require(
        not val_manifest.duplicated(subset=keys).any(),
        "Duplicate knee key found in authoritative VALIDATION manifest",
    )
    manifest_keys = set(tuple(x) for x in val_manifest[keys].itertuples(index=False, name=None))

    # Tabular predictions verification & filtering
    tab_df = pd.read_parquet(tabular_predictions_path)
    require(
        "split" in tab_df.columns and (tab_df["split"] == "VALIDATION").all(),
        "Tabular predictions contain non-VALIDATION rows",
    )

    tab_a = tab_df[tab_df["model_id"] == "logistic__formulation_A"].copy().reset_index(drop=True)
    require(
        len(tab_a) == 1044,
        f"logistic__formulation_A must have exactly 1044 rows, got {len(tab_a)}",
    )
    require(
        tab_a["participant_id"].nunique() == 543,
        "logistic__formulation_A must have exactly 543 unique participants",
    )
    require(
        not tab_a.duplicated(subset=keys).any(),
        "logistic__formulation_A has duplicate keys",
    )
    require(
        set(tuple(x) for x in tab_a[keys].itertuples(index=False, name=None)) == manifest_keys,
        "logistic__formulation_A knee keys do not match authoritative VALIDATION manifest",
    )

    tab_b = tab_df[tab_df["model_id"] == "logistic__formulation_B"].copy().reset_index(drop=True)
    require(
        len(tab_b) == 1044,
        f"logistic__formulation_B must have exactly 1044 rows, got {len(tab_b)}",
    )
    require(
        tab_b["participant_id"].nunique() == 543,
        "logistic__formulation_B must have exactly 543 unique participants",
    )
    require(
        not tab_b.duplicated(subset=keys).any(),
        "logistic__formulation_B has duplicate keys",
    )
    require(
        set(tuple(x) for x in tab_b[keys].itertuples(index=False, name=None)) == manifest_keys,
        "logistic__formulation_B knee keys do not match authoritative VALIDATION manifest",
    )

    # Image predictions verification & filtering
    img_df = pd.read_parquet(image_predictions_path)
    require(
        "split" in img_df.columns and (img_df["split"] == "VALIDATION").all(),
        "Image predictions contain non-VALIDATION rows",
    )

    img_sel = img_df[img_df["model_id"] == "DenseNet121_selected"].copy().reset_index(drop=True)
    require(
        len(img_sel) == 1044,
        f"DenseNet121_selected must have exactly 1044 rows, got {len(img_sel)}",
    )
    require(
        img_sel["participant_id"].nunique() == 543,
        "DenseNet121_selected must have exactly 543 unique participants",
    )
    require(
        not img_sel.duplicated(subset=keys).any(),
        "DenseNet121_selected has duplicate keys",
    )
    require(
        set(tuple(x) for x in img_sel[keys].itertuples(index=False, name=None)) == manifest_keys,
        "DenseNet121_selected knee keys do not match authoritative VALIDATION manifest",
    )

    # Strict merge by key (order-invariant) starting from manifest
    aligned = val_manifest[keys + ["knee_side_label", "anatomical_side", "split"]].copy()

    aligned = aligned.merge(
        tab_a[keys + ["composite_progression", "predicted_probability"]].rename(
            columns={
                "predicted_probability": "predicted_probability_logistic_A",
                "composite_progression": "label_tab_a",
            }
        ),
        on=keys,
        how="inner",
    )
    require(len(aligned) == 1044, "Merge with tab_a produced unexpected row count")

    aligned = aligned.merge(
        tab_b[keys + ["composite_progression", "predicted_probability"]].rename(
            columns={
                "predicted_probability": "predicted_probability_logistic_B",
                "composite_progression": "label_tab_b",
            }
        ),
        on=keys,
        how="inner",
    )
    require(len(aligned) == 1044, "Merge with tab_b produced unexpected row count")

    aligned = aligned.merge(
        img_sel[keys + ["composite_progression", "predicted_probability"]].rename(
            columns={
                "predicted_probability": "predicted_probability_image",
                "composite_progression": "label_img",
            }
        ),
        on=keys,
        how="inner",
    )
    require(len(aligned) == 1044, "Merge with img_sel produced unexpected row count")

    # Verify label and participant agreement across all sources
    require(
        (aligned["label_tab_a"] == aligned["label_tab_b"]).all(),
        "Label disagreement between tabular formulation A and formulation B",
    )
    require(
        (aligned["label_tab_a"] == aligned["label_img"]).all(),
        "Label disagreement between tabular and image models",
    )

    aligned["composite_progression"] = aligned["label_tab_a"].astype(int)
    aligned = aligned.drop(columns=["label_tab_a", "label_tab_b", "label_img"])

    for c in aligned.columns:
        if aligned[c].dtype == object or pd.api.types.is_string_dtype(aligned[c].dtype):
            aligned[c] = aligned[c].astype("string")

    require(
        aligned["composite_progression"].sum() == 161,
        f"Total positive events must be exactly 161, got {aligned['composite_progression'].sum()}",
    )
    require(
        (aligned["split"] == "VALIDATION").all(),
        "All aligned rows must be assigned to VALIDATION split",
    )
    for col in (
        "predicted_probability_logistic_A",
        "predicted_probability_logistic_B",
        "predicted_probability_image",
    ):
        require(
            np.isfinite(aligned[col]).all() and ((aligned[col] >= 0) & (aligned[col] <= 1)).all(),
            f"Invalid probability values in {col}",
        )

    return aligned


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard validation metrics."""
    return {
        "observations": len(y_true),
        "positive_events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "AUPRC_no_skill_reference": float(y_true.mean()),
    }


def evaluate_grid(
    aligned_df: pd.DataFrame,
    alpha_grid: list[float],
    formulation: str = "A",
) -> pd.DataFrame:
    """Evaluate convex probability fusion across the frozen alpha grid."""
    y = aligned_df["composite_progression"].to_numpy(dtype=int)
    p_img = aligned_df["predicted_probability_image"].to_numpy(dtype=float)
    col_tab = f"predicted_probability_logistic_{formulation}"
    p_tab = aligned_df[col_tab].to_numpy(dtype=float)

    rows = []
    for alpha in alpha_grid:
        alpha_f = round(float(alpha), 2)
        p_fused = alpha_f * p_img + (1.0 - alpha_f) * p_tab
        m = compute_metrics(y, p_fused)
        dist_half = round(abs(alpha_f - 0.50), 6)
        rows.append(
            {
                "formulation": formulation,
                "alpha": alpha_f,
                "AUROC": m["AUROC"],
                "AUPRC": m["AUPRC"],
                "Brier": m["Brier"],
                "log_loss": m["log_loss"],
                "distance_to_half": dist_half,
            }
        )
    return pd.DataFrame(rows)


def select_alpha(grid_df: pd.DataFrame) -> dict:
    """Apply the deterministic 6-step tie-breaking selection rule."""
    sorted_df = grid_df.sort_values(
        by=["AUROC", "AUPRC", "Brier", "log_loss", "distance_to_half", "alpha"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    return sorted_df.iloc[0].to_dict()


def run_clustered_bootstrap(
    aligned_df: pd.DataFrame,
    selected_alpha_a: float,
    config: dict,
) -> tuple[dict, dict, pd.DataFrame]:
    """Execute 1,000 participant-clustered bootstrap replicates on frozen validation models and weight."""
    replicates = config["bootstrap_replicates"]
    seed = config["bootstrap_seed"]

    y = aligned_df["composite_progression"].to_numpy(dtype=int)
    p_img = aligned_df["predicted_probability_image"].to_numpy(dtype=float)
    p_tab_a = aligned_df["predicted_probability_logistic_A"].to_numpy(dtype=float)
    # Frozen selected alpha
    p_fusion = selected_alpha_a * p_img + (1.0 - selected_alpha_a) * p_tab_a

    participants = aligned_df["participant_id"].to_numpy()
    unique_participants = sorted(set(participants))
    cluster_map = {pid: np.flatnonzero(participants == pid) for pid in unique_participants}
    groups = [cluster_map[pid] for pid in unique_participants]

    rng = np.random.default_rng(seed)
    sample_hash = hashlib.sha256()

    models = {
        "multimodal_fusion_A_selected": p_fusion,
        "DenseNet121_selected": p_img,
        "logistic__formulation_A": p_tab_a,
    }

    metrics_records = []
    diff_img_records = {"AUROC": [], "AUPRC": [], "Brier": []}
    diff_tab_records = {"AUROC": [], "AUPRC": [], "Brier": []}

    for rep in range(replicates):
        sampled = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        sample_hash.update(idx.astype("<i8").tobytes())

        y_b = y[idx]
        good = np.unique(y_b).size == 2

        rep_metrics = {}
        for m_name, p_all in models.items():
            p_b = p_all[idx]
            auc = float(roc_auc_score(y_b, p_b)) if good else np.nan
            ap = float(average_precision_score(y_b, p_b)) if good else np.nan
            brier = float(np.mean((y_b - p_b) ** 2))
            rep_metrics[m_name] = {"AUROC": auc, "AUPRC": ap, "Brier": brier}
            metrics_records.append(
                {
                    "model_id": m_name,
                    "replicate": rep,
                    "observations": len(idx),
                    "events": int(y_b.sum()),
                    "AUROC": auc,
                    "AUPRC": ap,
                    "Brier": brier,
                    "single_class_replicate": not good,
                }
            )

        if good:
            for metric_name in ("AUROC", "AUPRC", "Brier"):
                f_val = rep_metrics["multimodal_fusion_A_selected"][metric_name]
                i_val = rep_metrics["DenseNet121_selected"][metric_name]
                t_val = rep_metrics["logistic__formulation_A"][metric_name]

                diff_img_records[metric_name].append(f_val - i_val)
                diff_tab_records[metric_name].append(f_val - t_val)

    bootstrap_df = pd.DataFrame(metrics_records)

    intervals = {}
    for m_name, group_df in bootstrap_df.groupby("model_id"):
        intervals[m_name] = {
            metric: {
                "lower": float(group_df[metric].dropna().quantile(0.025)),
                "upper": float(group_df[metric].dropna().quantile(0.975)),
                "valid_replicates": int(group_df[metric].notna().sum()),
            }
            for metric in ("AUROC", "AUPRC", "Brier")
        }

    clustered_bootstrap = {
        "replicates": replicates,
        "seed": seed,
        "unit": "participant",
        "all_knees_in_sampled_cluster_included": True,
        "paired_draws_shared_by_all_models": True,
        "sample_indices_sha256": sample_hash.hexdigest(),
        "confidence_level": 0.95,
        "method": "percentile; conditional on fitted/selected development models and frozen fusion weight",
        "frozen_selected_alpha": selected_alpha_a,
        "intervals": intervals,
    }

    paired_differences = {
        "replicates": replicates,
        "seed": seed,
        "unit": "participant",
        "paired_draws_shared_by_all_models": True,
        "confidence_level": 0.95,
        "method": "percentile; conditional on fitted/selected development models and frozen fusion weight",
        "frozen_selected_alpha": selected_alpha_a,
        "comparisons": {
            "fusion_minus_image": {
                metric: {
                    "median": float(np.median(diff_img_records[metric])),
                    "lower": float(np.quantile(diff_img_records[metric], 0.025)),
                    "upper": float(np.quantile(diff_img_records[metric], 0.975)),
                    "valid_replicates": len(diff_img_records[metric]),
                }
                for metric in ("AUROC", "AUPRC", "Brier")
            },
            "fusion_minus_logistic_A": {
                metric: {
                    "median": float(np.median(diff_tab_records[metric])),
                    "lower": float(np.quantile(diff_tab_records[metric], 0.025)),
                    "upper": float(np.quantile(diff_tab_records[metric], 0.975)),
                    "valid_replicates": len(diff_tab_records[metric]),
                }
                for metric in ("AUROC", "AUPRC", "Brier")
            },
        },
        "statistical_claim_note": "Descriptive empirical bootstrap distributions; not converted to claims of statistical significance.",
    }

    return clustered_bootstrap, paired_differences, bootstrap_df


def compute_and_plot_calibration(
    aligned_df: pd.DataFrame,
    selected_alpha_a: float,
    output_dir: Path,
) -> dict:
    """Compute descriptive quantile calibration curves and render calibration plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = aligned_df["composite_progression"].to_numpy(dtype=int)
    p_img = aligned_df["predicted_probability_image"].to_numpy(dtype=float)
    p_tab_a = aligned_df["predicted_probability_logistic_A"].to_numpy(dtype=float)
    p_fusion = selected_alpha_a * p_img + (1.0 - selected_alpha_a) * p_tab_a

    prob_true_f, prob_pred_f = calibration_curve(y, p_fusion, n_bins=10, strategy="quantile")
    prob_true_i, prob_pred_i = calibration_curve(y, p_img, n_bins=10, strategy="quantile")
    prob_true_t, prob_pred_t = calibration_curve(y, p_tab_a, n_bins=10, strategy="quantile")

    cal_f = tabular.calibration(y, p_fusion)
    cal_i = tabular.calibration(y, p_img)
    cal_t = tabular.calibration(y, p_tab_a)

    cal_data = {
        "partition": "VALIDATION",
        "recalibration_applied": False,
        "note": "Joint unpenalized validation diagnostic; never applied to alter predictions.",
        "multimodal_fusion_A_selected": {
            "alpha": selected_alpha_a,
            "mean_predicted_probability": prob_pred_f.tolist(),
            "observed_event_fraction": prob_true_f.tolist(),
            "calibration_intercept": cal_f["intercept"],
            "calibration_slope": cal_f["slope"],
        },
        "unimodal_image_DenseNet121": {
            "mean_predicted_probability": prob_pred_i.tolist(),
            "observed_event_fraction": prob_true_i.tolist(),
            "calibration_intercept": cal_i["intercept"],
            "calibration_slope": cal_i["slope"],
        },
        "unimodal_tabular_logistic_A": {
            "mean_predicted_probability": prob_pred_t.tolist(),
            "observed_event_fraction": prob_true_t.tolist(),
            "calibration_intercept": cal_t["intercept"],
            "calibration_slope": cal_t["slope"],
        },
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.plot(
        prob_pred_f,
        prob_true_f,
        "s-",
        color="#1f77b4",
        linewidth=2,
        label=f"Multimodal Fusion A (alpha={selected_alpha_a:.2f})",
    )
    ax.plot(
        prob_pred_i,
        prob_true_i,
        "o--",
        color="#ff7f0e",
        alpha=0.7,
        label="DenseNet121 Image (Candidate 2)",
    )
    ax.plot(
        prob_pred_t,
        prob_true_t,
        "^--",
        color="#2ca02c",
        alpha=0.7,
        label="Logistic Formulation A",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Event Fraction", fontsize=11)
    ax.set_title("Validation Calibration: Multimodal Fusion vs Unimodal Baselines", fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)

    return cal_data


def build_domain_ablation_summary(
    tabular_selected_path: Path,
    image_selected_path: Path,
    fusion_a_metrics: dict,
    fusion_a_alpha: float,
    fusion_b_metrics: dict,
    fusion_b_alpha: float,
) -> dict:
    """Compile validation domain ablation summary comparing unimodal domains and multimodal fusions."""
    tab_sc = read_json(tabular_selected_path)
    img_sc = read_json(image_selected_path)

    domains = [
        {
            "rank_order": 1,
            "domain": "Demographic & Clinical",
            "model": "logistic__demographic_clinical",
            "formulation_role": "Unimodal tabular domain baseline",
            "AUROC": tab_sc["logistic__demographic_clinical"]["metrics"]["AUROC"],
            "AUPRC": tab_sc["logistic__demographic_clinical"]["metrics"]["AUPRC"],
            "Brier": tab_sc["logistic__demographic_clinical"]["metrics"]["Brier"],
            "log_loss": tab_sc["logistic__demographic_clinical"]["metrics"]["log_loss"],
            "calibration_intercept": tab_sc["logistic__demographic_clinical"]["metrics"][
                "calibration"
            ]["intercept"],
            "calibration_slope": tab_sc["logistic__demographic_clinical"]["metrics"]["calibration"][
                "slope"
            ],
        },
        {
            "rank_order": 2,
            "domain": "Patient-Reported Outcomes (PRO)",
            "model": "logistic__patient_reported",
            "formulation_role": "Unimodal tabular domain baseline",
            "AUROC": tab_sc["logistic__patient_reported"]["metrics"]["AUROC"],
            "AUPRC": tab_sc["logistic__patient_reported"]["metrics"]["AUPRC"],
            "Brier": tab_sc["logistic__patient_reported"]["metrics"]["Brier"],
            "log_loss": tab_sc["logistic__patient_reported"]["metrics"]["log_loss"],
            "calibration_intercept": tab_sc["logistic__patient_reported"]["metrics"]["calibration"][
                "intercept"
            ],
            "calibration_slope": tab_sc["logistic__patient_reported"]["metrics"]["calibration"][
                "slope"
            ],
        },
        {
            "rank_order": 3,
            "domain": "Physical Function",
            "model": "logistic__physical_function",
            "formulation_role": "Unimodal tabular domain baseline",
            "AUROC": tab_sc["logistic__physical_function"]["metrics"]["AUROC"],
            "AUPRC": tab_sc["logistic__physical_function"]["metrics"]["AUPRC"],
            "Brier": tab_sc["logistic__physical_function"]["metrics"]["Brier"],
            "log_loss": tab_sc["logistic__physical_function"]["metrics"]["log_loss"],
            "calibration_intercept": tab_sc["logistic__physical_function"]["metrics"][
                "calibration"
            ]["intercept"],
            "calibration_slope": tab_sc["logistic__physical_function"]["metrics"]["calibration"][
                "slope"
            ],
        },
        {
            "rank_order": 4,
            "domain": "Tabular Formulation A",
            "model": "logistic__formulation_A",
            "formulation_role": "Primary approved tabular baseline",
            "AUROC": tab_sc["logistic__formulation_A"]["metrics"]["AUROC"],
            "AUPRC": tab_sc["logistic__formulation_A"]["metrics"]["AUPRC"],
            "Brier": tab_sc["logistic__formulation_A"]["metrics"]["Brier"],
            "log_loss": tab_sc["logistic__formulation_A"]["metrics"]["log_loss"],
            "calibration_intercept": tab_sc["logistic__formulation_A"]["metrics"]["calibration"][
                "intercept"
            ],
            "calibration_slope": tab_sc["logistic__formulation_A"]["metrics"]["calibration"][
                "slope"
            ],
        },
        {
            "rank_order": 5,
            "domain": "Image-only DenseNet121",
            "model": "DenseNet121_selected (Candidate 2)",
            "formulation_role": "Primary approved image baseline",
            "AUROC": img_sc["metrics"]["AUROC"],
            "AUPRC": img_sc["metrics"]["AUPRC"],
            "Brier": img_sc["metrics"]["Brier"],
            "log_loss": img_sc["metrics"]["log_loss"],
            "calibration_intercept": img_sc["metrics"]["calibration"]["intercept"],
            "calibration_slope": img_sc["metrics"]["calibration"]["slope"],
        },
        {
            "rank_order": 6,
            "domain": "Multimodal Fusion A",
            "model": f"Convex Fusion A (alpha={fusion_a_alpha:.2f})",
            "formulation_role": "PRIMARY MULTIMODAL FUSION",
            "AUROC": fusion_a_metrics["AUROC"],
            "AUPRC": fusion_a_metrics["AUPRC"],
            "Brier": fusion_a_metrics["Brier"],
            "log_loss": fusion_a_metrics["log_loss"],
            "calibration_intercept": fusion_a_metrics["calibration"]["intercept"],
            "calibration_slope": fusion_a_metrics["calibration"]["slope"],
        },
        {
            "rank_order": 7,
            "domain": "Tabular Formulation B Sensitivity",
            "model": "logistic__formulation_B",
            "formulation_role": "BASELINE-KL SENSITIVITY",
            "AUROC": tab_sc["logistic__formulation_B"]["metrics"]["AUROC"],
            "AUPRC": tab_sc["logistic__formulation_B"]["metrics"]["AUPRC"],
            "Brier": tab_sc["logistic__formulation_B"]["metrics"]["Brier"],
            "log_loss": tab_sc["logistic__formulation_B"]["metrics"]["log_loss"],
            "calibration_intercept": tab_sc["logistic__formulation_B"]["metrics"]["calibration"][
                "intercept"
            ],
            "calibration_slope": tab_sc["logistic__formulation_B"]["metrics"]["calibration"][
                "slope"
            ],
        },
        {
            "rank_order": 8,
            "domain": "Multimodal Fusion B Sensitivity",
            "model": f"Convex Fusion B (alpha={fusion_b_alpha:.2f})",
            "formulation_role": "BASELINE-KL SENSITIVITY",
            "AUROC": fusion_b_metrics["AUROC"],
            "AUPRC": fusion_b_metrics["AUPRC"],
            "Brier": fusion_b_metrics["Brier"],
            "log_loss": fusion_b_metrics["log_loss"],
            "calibration_intercept": fusion_b_metrics["calibration"]["intercept"],
            "calibration_slope": fusion_b_metrics["calibration"]["slope"],
        },
    ]

    return {
        "evaluation_partition": "VALIDATION",
        "observations": 1044,
        "positive_events": 161,
        "prevalence": 161 / 1044,
        "comparison_note": "Validation ranking alone does not establish definitive superiority; TEST remains strictly locked.",
        "domains": domains,
    }


def verify_preservation_baseline(image_preservation_path: Path) -> dict:
    """Verify recursive preservation of all protected upstream artifacts and return snapshot."""
    require(image_preservation_path.is_file(), "Image preservation baseline file missing")
    image_baseline = read_json(image_preservation_path)
    image.preserve(image_baseline)

    # Check explicit approved hashes
    explicit_paths = {
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
        "image_run_freeze": IMAGE_OUTPUT / "run_freeze.json",
        "image_validation_predictions": IMAGE_OUTPUT / "validation_predictions.parquet",
        "image_selected_checkpoint": IMAGE_OUTPUT / "candidate_2/best.pt",
    }
    actual_hashes = {key: sha256_file(path) for key, path in explicit_paths.items()}
    require(
        actual_hashes == APPROVED,
        f"Protected artifact SHA mismatch: expected {APPROVED}, got {actual_hashes}",
    )

    # Hash tree for image production files
    image_production_files = {
        str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(IMAGE_OUTPUT.rglob("*"))
        if p.is_file() and not p.name.endswith(".lock")
    }

    image_code_config = {
        str(p): sha256_file(p)
        for p in (
            Path("src/modeling/image.py"),
            Path("src/modeling/colab_bundle.py"),
            Path("configs/image_model_v1.yaml"),
        )
    }

    return {
        "image_preservation": image_baseline,
        "explicit_approved_hashes": actual_hashes,
        "image_production_files": image_production_files,
        "image_code_config": image_code_config,
    }


def preserve(baseline: dict) -> None:
    """Verify that no protected upstream artifact was modified during fusion execution."""
    image.preserve(baseline["image_preservation"])
    for key, expected_sha in APPROVED.items():
        require(
            baseline["explicit_approved_hashes"][key] == expected_sha,
            f"Explicit hash drifted for {key}",
        )
    current_image_files = {
        str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(IMAGE_OUTPUT.rglob("*"))
        if p.is_file() and not p.name.endswith(".lock")
    }
    require(
        current_image_files == baseline["image_production_files"],
        "Image production artifacts modified during fusion run",
    )
    current_code = {
        str(p): sha256_file(p)
        for p in (
            Path("src/modeling/image.py"),
            Path("src/modeling/colab_bundle.py"),
            Path("configs/image_model_v1.yaml"),
        )
    }
    require(
        current_code == baseline["image_code_config"],
        "Image code or configuration modified during fusion run",
    )


def run_fusion(
    config_path: Path = CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    """Execute complete Milestone 6E fusion pipeline with strict verification."""
    config = load_config(config_path)
    validate_config(config)

    # 1. Preservation baseline verification
    baseline = verify_preservation_baseline(IMAGE_BASE_DIR / "pre_change_preservation.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    publish_json(baseline, output_dir / "pre_change_preservation.json")

    # 2. Strict prediction alignment by key
    aligned_df = load_and_align_validation_predictions(
        tabular_predictions_path=TABULAR_OUTPUT / "validation_predictions.parquet",
        image_predictions_path=IMAGE_OUTPUT / "validation_predictions.parquet",
        knee_split_manifest_path=DEFAULT_DIRECTORY / "splits/v1/knee_split_manifest.parquet",
    )
    publish_parquet(aligned_df, output_dir / "aligned_validation_predictions.parquet")

    # 3. Grid evaluation
    grid_a = evaluate_grid(aligned_df, config["alpha_grid"], formulation="A")
    grid_b = evaluate_grid(aligned_df, config["alpha_grid"], formulation="B")
    combined_grid = pd.concat([grid_a, grid_b], ignore_index=True)
    combined_grid["formulation"] = combined_grid["formulation"].astype("string")
    publish_parquet(combined_grid, output_dir / "fusion_grid_results.parquet")

    # 4. Deterministic alpha selection
    selected_a = select_alpha(grid_a)
    selected_b = select_alpha(grid_b)

    alpha_a = selected_a["alpha"]
    alpha_b = selected_b["alpha"]

    y = aligned_df["composite_progression"].to_numpy(dtype=int)
    p_img = aligned_df["predicted_probability_image"].to_numpy(dtype=float)
    p_tab_a = aligned_df["predicted_probability_logistic_A"].to_numpy(dtype=float)
    p_tab_b = aligned_df["predicted_probability_logistic_B"].to_numpy(dtype=float)

    p_fused_a = alpha_a * p_img + (1.0 - alpha_a) * p_tab_a
    p_fused_b = alpha_b * p_img + (1.0 - alpha_b) * p_tab_b

    metrics_a = {
        **compute_metrics(y, p_fused_a),
        "calibration": tabular.calibration(y, p_fused_a),
    }
    metrics_b = {
        **compute_metrics(y, p_fused_b),
        "calibration": tabular.calibration(y, p_fused_b),
    }

    # Format selected primary validation predictions (1,044 rows)
    val_preds = pd.DataFrame(
        {
            "participant_id": aligned_df["participant_id"].astype("string"),
            "knee_side_code": aligned_df["knee_side_code"].astype("string"),
            "baseline_visit": aligned_df["baseline_visit"].astype("string"),
            "composite_progression": aligned_df["composite_progression"].astype(np.int64),
            "predicted_probability": p_fused_a.astype(np.float64),
            "model_id": pd.Series(
                ["multimodal_fusion_A_selected"] * len(aligned_df), dtype="string"
            ),
            "split": pd.Series(["VALIDATION"] * len(aligned_df), dtype="string"),
        }
    )
    publish_parquet(val_preds, output_dir / "validation_predictions.parquet")

    # 5. Participant-clustered bootstrap with frozen selected alpha
    clustered_bootstrap, paired_diffs, _ = run_clustered_bootstrap(aligned_df, alpha_a, config)
    publish_json(clustered_bootstrap, output_dir / "clustered_bootstrap.json")
    publish_json(paired_diffs, output_dir / "paired_bootstrap_differences.json")

    # 6. Descriptive calibration curve and plot
    cal_data = compute_and_plot_calibration(aligned_df, alpha_a, output_dir)
    publish_json(cal_data, output_dir / "calibration_data.json")

    # 7. Validation metrics record
    tab_sc = read_json(TABULAR_OUTPUT / "selected_candidates.json")
    img_sc = read_json(IMAGE_OUTPUT / "selected_candidate.json")
    val_metrics = {
        "primary_fusion_A": {
            "selected_alpha": alpha_a,
            "metrics": metrics_a,
        },
        "sensitivity_fusion_B": {
            "role": "BASELINE-KL SENSITIVITY",
            "selected_alpha": alpha_b,
            "metrics": metrics_b,
        },
        "unimodal_references": {
            "DenseNet121_selected": img_sc["metrics"],
            "logistic__formulation_A": tab_sc["logistic__formulation_A"]["metrics"],
            "logistic__formulation_B": tab_sc["logistic__formulation_B"]["metrics"],
        },
    }
    publish_json(val_metrics, output_dir / "validation_metrics.json")

    # 8. Domain ablation summary
    ablation_summary = build_domain_ablation_summary(
        tabular_selected_path=TABULAR_OUTPUT / "selected_candidates.json",
        image_selected_path=IMAGE_OUTPUT / "selected_candidate.json",
        fusion_a_metrics=metrics_a,
        fusion_a_alpha=alpha_a,
        fusion_b_metrics=metrics_b,
        fusion_b_alpha=alpha_b,
    )
    publish_json(ablation_summary, output_dir / "domain_ablation_summary.json")

    # 9. Selected fusion record
    selected_fusion = {
        "selected_model": "multimodal_fusion_A_selected",
        "primary_formulation": "formulation_A",
        "selected_alpha": alpha_a,
        "selection_rule": {
            "hierarchy": SELECTION_HIERARCHY,
            "grid": config["alpha_grid"],
            "outcome": f"alpha={alpha_a:.2f} selected based on highest VALIDATION AUROC ({metrics_a['AUROC']:.6f})",
        },
        "metrics": metrics_a,
        "sensitivity": {
            "role": "BASELINE-KL SENSITIVITY",
            "formulation": "formulation_B",
            "selected_alpha": alpha_b,
            "metrics": metrics_b,
            "note": "Formulation B contains baseline KL and remains sensitivity-only; does not replace Formulation A.",
        },
        "unimodal_comparisons": {
            "image_DenseNet121": img_sc["metrics"],
            "tabular_logistic_A": tab_sc["logistic__formulation_A"]["metrics"],
        },
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "code_sha256": sha256_file(Path(__file__)),
            "bootstrap_seed": config["bootstrap_seed"],
            "validation_knees": 1044,
            "validation_participants": 543,
            "validation_events": 161,
            "source_hashes": APPROVED,
        },
    }
    publish_json(selected_fusion, output_dir / "selected_fusion.json")

    # Publish config snapshot
    publish_json(config, output_dir / "fusion_v1.yaml")

    # 10. Post-change preservation check
    preserve(baseline)

    # 11. Artifact hashes and run freeze
    artifact_names = [
        "fusion_v1.yaml",
        "aligned_validation_predictions.parquet",
        "fusion_grid_results.parquet",
        "selected_fusion.json",
        "validation_predictions.parquet",
        "validation_metrics.json",
        "clustered_bootstrap.json",
        "paired_bootstrap_differences.json",
        "calibration_data.json",
        "calibration.png",
        "domain_ablation_summary.json",
        "pre_change_preservation.json",
    ]
    artifact_hashes = {
        name: {
            "sha256": sha256_file(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in artifact_names
    }
    publish_json(artifact_hashes, output_dir / "artifact_hashes.json")

    freeze_path = output_dir / "run_freeze.json"
    existing_freeze = read_json(freeze_path) if freeze_path.exists() else None

    freeze = {
        "status": "APPROVED_MILESTONE_6E",
        "created_at_utc": existing_freeze["created_at_utc"]
        if existing_freeze and "created_at_utc" in existing_freeze
        else datetime.now(UTC).isoformat(),
        "run_version": config["version"],
        "primary_formulation": "formulation_A",
        "sensitivity_formulation": "formulation_B",
        "selected_alpha_A": alpha_a,
        "selected_alpha_B": alpha_b,
        "primary_metrics": metrics_a,
        "sensitivity_metrics": metrics_b,
        "no_new_predictive_model_fitted": True,
        "no_oof_stacking_or_meta_model": True,
        "no_image_inference_run": True,
        "test_scored": False,
        "test_predictions_generated": False,
        "test_lock_verified": True,
        "source_preservation_passed": True,
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "code_sha256": sha256_file(Path(__file__)),
            "alpha_grid": config["alpha_grid"],
            "selected_alpha_primary": alpha_a,
            "selected_alpha_sensitivity": alpha_b,
            "bootstrap_seed": config["bootstrap_seed"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "validation_rows": 1044,
            "validation_participants": 543,
            "validation_events": 161,
            "source_hashes": APPROVED,
        },
        "artifacts_sha256": {k: v["sha256"] for k, v in artifact_hashes.items()},
    }
    publish_json(freeze, output_dir / "run_freeze.json")

    # Verify no TEST artifact exists anywhere in fusion output directory
    for path in output_dir.rglob("*"):
        require(
            "test" not in path.name.lower() or path.name.endswith(".lock"),
            "Forbidden TEST artifact detected in fusion directory",
        )

    return {
        "status": "APPROVED_MILESTONE_6E",
        "selected_alpha_A": alpha_a,
        "selected_alpha_B": alpha_b,
        "metrics_A": metrics_a,
        "metrics_B": metrics_b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with file_lock(args.output / "run_freeze.json"):
        result = run_fusion(config_path=args.config, output_dir=args.output)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
