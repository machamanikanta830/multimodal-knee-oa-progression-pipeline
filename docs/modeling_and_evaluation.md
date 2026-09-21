# Modeling Architecture & Evaluation Protocol

This document details the unimodal and multimodal model architectures, hyperparameter configurations, fusion strategies, and statistical evaluation protocols.

---

## 1. Unimodal Model Formulations

```mermaid
flowchart TD
    subgraph TabularPipeline [Tabular Modeling Pipeline]
        TAB_RAW[14 Non-Radiographic Baseline Features] --> TAB_IMPUTE[Train-Fitted Median Imputation\n(No Missingness Indicators)]

        TAB_IMPUTE --> TAB_SCALE[StandardScaler: Mean/Variance from TRAIN]
        TAB_SCALE --> TAB_LOGREG[L2-Penalized Logistic Regression\nFormulation A: Primary]
        TAB_SCALE --> TAB_LOGREG_B[L2-Penalized Logistic Regression\nFormulation B: + Baseline KL (Sensitivity)]
    end

    subgraph ImagePipeline [Imaging Modeling Pipeline]
        IMG_RAW[Standardized Knee Crop: 1067x1067 uint16] --> IMG_TRANSFORM[p0.5/p99.5 Clip -> [0,1] Scale ->\n320x320 Bilinear -> ImageNet Norm]
        IMG_TRANSFORM --> DENSENET[DenseNet121 Backbone\nImageNet-1K Pretrained]
        DENSENET --> CLASSIFIER[Linear Classifier Head -> Sigmoid]
    end

    subgraph FusionEngine [Late Probability Fusion Engine]
        TAB_LOGREG --> P_TAB[P_tabular]
        DENSENET --> P_IMG[P_image]
        P_TAB & P_IMG --> PROB_FUSION[Convex Combination:\nP_fusion = alpha * P_image + 1-alpha * P_tabular]
        PROB_FUSION --> FUSION_A[Primary Fusion A: alpha = 0.50]
        PROB_FUSION --> FUSION_B[Sensitivity Fusion B: alpha = 0.60]
    end
```

### A. Primary Tabular Model (Logistic Formulation A)
- **Model Family:** $L_2$-penalized Logistic Regression with lbfgs solver ($C=0.1$, `max_iter=2000`, `tol=1e-8`).
- **Predictor Set (14 features):** Non-radiographic baseline clinical, patient-reported, and physical-function attributes:
  1. `age_years` (Participant age at baseline)
  2. `sex` (Biological sex)
  3. `bmi` (Body mass index, $\text{kg/m}^2$)
  4. `prior_knee_surgery` (Documented history of prior index knee surgery)
  5. `family_knee_replacement_history` (Immediate family history of knee replacement)
  6. `womac_pain` (WOMAC pain subscale)
  7. `womac_stiffness` (WOMAC stiffness subscale)
  8. `womac_disability` (WOMAC physical function disability subscale)
  9. `koos_pain` (KOOS baseline pain subscale)
  10. `koos_symptoms` (KOOS symptoms subscale)
  11. `walk_20m_pace_mps` (20-meter walking pace test in meters/second)
  12. `chair_stand_time_seconds` (Repeated chair stand completion time in seconds)
  13. `knee_extension_strength_n` (Isometric knee extension strength in Newtons)
  14. `knee_flexion_strength_n` (Isometric knee flexion strength in Newtons)
- **Prespecified Exclusion:** **Baseline KL grade is excluded** from Formulation A to evaluate pure clinical, patient-reported, and physical performance signals independent of baseline radiographic structural damage.

### B. Secondary / Sensitivity Tabular Model (Logistic Formulation B)
- **Predictor Set (15 features):** Identical to Formulation A, plus `baseline_kl` ($0, 1, 2, 3$).
- **Role:** Evaluates whether adding baseline radiographic severity improves the tabular clinical baseline.

### C. Deep Learning Imaging Model (DenseNet121 Candidate 2)
- **Architecture:** Torchvision `DenseNet121` with pretrained `IMAGENET1K_V1` weights.
- **Input Dimension:** $320 \times 320 \times 3$ (bilinear interpolation, antialias=True).
- **Optimization:** AdamW optimizer ($\text{learning rate} = 3 \times 10^{-5}$, $\text{weight decay} = 10^{-4}$), unweighted binary cross-entropy loss (`BCEWithLogitsLoss`), batch size 2, AMP mixed precision.
- **Training Schedule:** Fine-tuning all layers for up to 25 epochs with early stopping (patience = 5 epochs on VALIDATION AUROC after minimum 3 epochs); checkpoint selection saved the best candidate on the hierarchy of descending VALIDATION AUROC, descending AUPRC, ascending Brier score, and ascending log loss.

---

## 2. Multimodal Probability Fusion Strategy

The pipeline implements late probability fusion via convex combination:
$$P_{\text{fusion}} = \alpha \cdot P_{\text{image}} + (1 - \alpha) \cdot P_{\text{tabular}}$$

- **Primary Fusion A:** Combines DenseNet121 with Logistic Formulation A. The fusion parameter $\alpha = 0.50$ was selected on the VALIDATION partition ($N=1,044$) to balance modalities equally.
- **Sensitivity Fusion B:** Combines DenseNet121 with Logistic Formulation B (incorporating baseline KL). The parameter $\alpha = 0.60$ was chosen on VALIDATION, reflecting slightly greater weighting on imaging.

---

## 3. Statistical Evaluation Protocol

### Metric Definitions
1. **Area Under the ROC Curve (AUROC):** Primary measure of ranking discrimination across all classification thresholds.
2. **Area Under the Precision-Recall Curve (AUPRC / Average Precision):** Primary metric for positive event discrimination under class imbalance ($15.4\%$ prevalence).
3. **Brier Score:** Mean squared error between predicted probabilities and observed binary outcomes:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
4. **Binary Log Loss (Cross-Entropy):** Penalizes confident but incorrect probability assignments.
5. **Calibration Diagnostics:** Unpenalized logistic recalibration fit:
   $$\text{logit}(y) = \beta_0 + \beta_1 \cdot \text{logit}(p)$$
   where $\beta_0$ represents calibration intercept (ideal $0$) and $\beta_1$ represents calibration slope (ideal $1$). Diagnostics are evaluated without altering predicted probabilities.

### Why Accuracy Was Rejected as a Primary Metric
In a cohort with $15.4\%$ event prevalence, a trivial majority-class classifier that predicts all knees to be non-progressors achieves **84.6% classification accuracy**, despite identifying zero progressing knees. Accuracy obscures clinical utility under class imbalance. AUROC, AUPRC, and Brier score provide robust, threshold-independent evaluations.

### Participant-Clustered Bootstrap Resampling
Because participants contribute up to two knees, knees from the same participant share shared genetic, biomechanical, and systemic factors:
- **Resampling Unit:** The **participant** (all knees of a sampled participant enter the bootstrap replicate together).
- **Parameters:** 1,000 bootstrap draws, fixed seed **65027**, 95% two-sided percentile intervals:
  $$\text{CI}_{95\%} = [\text{percentile}(2.5), \text{percentile}(97.5)]$$
- **Paired Comparisons:** For comparative differences ($\text{Model}_A - \text{Model}_B$), differences are calculated **within each bootstrap replicate**, preserving paired participant covariance.

---

## 4. Frozen Evaluation Governance

The evaluation followed a strict held-out execution gate under development embargo:
1. All tabular models, image checkpoints, and fusion parameters were finalized and cryptographically hashed in `final_model_freeze.json`.
2. A durable pre-score lock file (`pre_score_freeze.json`) was written to disk before TEST outcome access.
3. Scoring code (`src/modeling/final_test_evaluation.py`) verified input hashes before executing the single forward pass.
