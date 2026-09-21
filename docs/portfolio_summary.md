# Project Portfolio Summary: Multimodal Knee OA Progression Pipeline

An executive summary tailored for machine learning engineers, data scientists, research evaluators, and hiring managers.

---

## 1. Executive Summary

| Category | Details |
|---|---|
| **Problem Statement** | Predict structural knee osteoarthritis progression over ~48 months using baseline clinical, patient-reported, and radiographic data. |
| **Dataset & Scale** | The Osteoarthritis Initiative (OAI): **6,961 knees** across **3,621 unique participants** ($1,071$ progression events, $15.4\%$ prevalence). |
| **Data Modalities** | Bilateral digital plain radiographs, structured clinical risk factors, WOMAC/KOOS patient-reported outcomes, and objective physical function tests. |
| **Modeling Paradigms** | Deep convolutional vision network (**DenseNet121**), $L_2$-penalized Logistic Regression, and late probability fusion ($\alpha = 0.50$). |
| **Validation Protocol** | Participant-grouped partitioning (zero bilateral/longitudinal leakage), 1,000 participant-clustered bootstrap replicates, one-time held-out TEST evaluation under development embargo ($N = 1,044$ knees, $161$ events). |
| **Key Result** | Standalone image deep learning achieved **0.732 AUROC** / **0.371 AUPRC**. Multimodal Fusion A achieved **0.728 AUROC** / **0.368 AUPRC**, substantially improving over tabular clinical features (**0.644 AUROC**) and performing similarly to the image-only model. |

---

## 2. Engineering & Scientific Highlights

### 1. Robust Medical Image Engineering
- **DICOM Standardization:** Extracted raw bilateral knee radiographs and converted varied scanner pixel dimensions to a standardized spatial resolution of **0.150 mm/pixel**.
- **Anatomical Joint Centering:** Extracted standardized $160 \times 160\text{ mm}$ physical square joint crops ($1,067 \times 1,067$ uint16), standardizing physical pixel spacing and spatial scale.
- **Human-in-the-Loop Quality Control:** Designed a blinded human adjudication interface for 1,956 knee panels across 978 queued acquisitions; generated 422 center overrides to achieve complete anatomical resolution across all modeling knees.
- **Fail-Closed Runtime:** Model-time pipeline incorporates per-image percentile intensity clipping (p0.5/p99.5), min-max scaling to $[0, 1]$, bilinear spatial resize to $320 \times 320$, and ImageNet standardization.

### 2. Leakage-Aware Experimental Design
- **Participant-Level Grouping:** Bilateral knee pairs share systemic biology and environmental factors. Both knees of any participant were deterministically allocated to the same partition using SHA-256 candidate ranking, guaranteeing **zero participant overlap** across TRAIN ($4,873$ knees), VALIDATION ($1,044$ knees), and TEST ($1,044$ knees).
- **Train-Only Preprocessing:** Imputation statistics, scaling parameters, and feature encodings were fit strictly on the training partition and transformed downstream, preventing lookahead leakage.

### 3. Rigorous Clinical Endpoint Engineering
- Formulated an authoritative binary composite endpoint ($\text{composite\_progression} = \Delta\text{KL} \ge 1 \lor \text{replacement\_before\_v06}$) over ~48 months, capturing structural worsening and surgical knee replacement without unmodeled informative loss to follow-up.

### 4. Enterprise-Grade Provenance & Reproducibility
- Implemented a held-out execution gate: all models, checkpoints, feature rosters, and statistical code were locked with SHA-256 hashes prior to authorized final evaluation.
- Evaluated performance with **participant-clustered bootstrap resampling** (1,000 replicates, fixed seed), capturing true participant-level uncertainty.

---

## 3. Final Held-Out TEST Results Summary

Evaluated on $N = 1,044$ knees across $543$ participants with $161$ positive events ($15.42\%$ prevalence):

```text
========================================================================================
Model                              AUROC [95% CI]        AUPRC [95% CI]      Brier Score
========================================================================================
Logistic Formulation A (Tabular)   0.644 [0.595, 0.689]  0.236 [0.190, 0.300]   0.127
DenseNet121 Candidate 2 (Image)    0.732 [0.684, 0.774]  0.371 [0.293, 0.461]   0.118
Primary Multimodal Fusion A        0.728 [0.680, 0.771]  0.368 [0.290, 0.450]   0.118
========================================================================================
Paired Comparison                  Delta AUROC [95% CI]       Delta AUPRC [95% CI]
----------------------------------------------------------------------------------------
Fusion A vs. Logistic A            +0.085 [+0.047, +0.121]    +0.132 [+0.072, +0.188]
Fusion A vs. DenseNet121           -0.004 [-0.027, +0.019]    -0.003 [-0.040, +0.026]
========================================================================================
```

### Core Takeaway
Deep learning on baseline knee radiographs provides the strongest predictive signal for 48-month structural OA progression. Integrating clinical risk factors and patient-reported outcomes materially lifts performance above tabular baselines, but provides minimal incremental gain over high-resolution joint imaging alone.
