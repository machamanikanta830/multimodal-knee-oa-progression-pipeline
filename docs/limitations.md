# Scientific & Methodological Limitations

A transparent accounting of the scientific, computational, and clinical limitations of this study.

---

## 1. Single-Cohort Population & Generalizability

- **OAI Research Cohort:** The Osteoarthritis Initiative recruited individuals with symptomatic knee OA or recognized risk factors across four dedicated academic medical centers. Participants in this prospective cohort may differ systematically from unselected patients presenting in routine primary care or community orthopedic settings.
- **Absence of External Validation:** While the pipeline implements rigorous internal validation with a locked, participant-grouped held-out TEST cohort, it has not been validated on an independent external cohort (such as the Multicenter Osteoarthritis Study [MOST] or the Cohort Hip and Cohort Knee [CHECK] study). External generalizability across different imaging systems, patient demographics, and clinical protocols remains to be established.
- **Validation-to-TEST Discrimination Dynamics:** The image model's discrimination was higher on the held-out TEST partition (TEST AUROC = 0.731963) than on validation (VALIDATION AUROC = 0.677856; Delta = +0.054107). This difference may reflect partition-specific sampling variability. The pattern does not by itself establish a generalization gain and reinforces the need for external cohort validation. No known split or preprocessing leakage was identified in the project's existing leakage audits.

---

## 2. Research Pipeline Scope vs. Clinical Deployment

- **Non-Diagnostic Scope:** This repository is an academic research demonstration investigating prognostic modeling from multimodal baseline data. It is **not** a validated medical device, clinical decision support software, or diagnostic system.
- **Predictive Correlates, Not Causal Targets:** Identified prognostic features indicate statistical association with 48-month structural progression. They do not demonstrate causal pathology and should not be interpreted as actionable targets for therapeutic intervention without prospective interventional trials.

---

## 3. Imaging Modality & Feature Scope

- **Baseline Cross-Sectional Imaging Only:** The image branch evaluates only baseline (V00) plain radiographs. It does not integrate longitudinal imaging trajectories, 3D magnetic resonance imaging (MRI), or quantitative cartilage volume assessments.
- **Fixed-Flexion Projection Dependence:** Localization and crop standardization assume fixed-flexion knee radiography with a positioning frame (SynaFlexer). Performance on conventional non-standardized anteroposterior (AP) standing radiographs is uncharacterized.

---

## 4. Fusion Strategy & Multimodal Incremental Value

- **Late Probability Fusion Simplicity:** Multimodal integration was restricted to prespecified linear convex probability combination ($P_{\text{fusion}} = \alpha \cdot P_{\text{image}} + (1-\alpha) \cdot P_{\text{tabular}}$). Intermediate feature representations (e.g., cross-attention transformers, joint embedding spaces) were not evaluated in the primary frozen pipeline.
- **Absence of Multimodal Superiority over Image-Only:** On the held-out TEST partition, primary Fusion A ($\text{AUROC} = 0.728$) did not clearly outperform the unimodal image model ($\text{AUROC} = 0.732$; $\Delta = -0.0038$, 95% paired CI $[-0.0266, +0.0186]$). While fusion substantially improved upon tabular clinical features alone, baseline clinical factors provided minimal incremental discrimination once deep radiographic features were present.

---

## 5. Statistical & Calibration Dynamics

- **Class Imbalance:** With an empirical event rate of $15.4\%$ ($161$ progressing knees out of $1,044$ TEST knees), class imbalance affects precision and calibration. Classification accuracy is uninformative, and positive predictive value is constrained at high sensitivity.
- **Post-Fusion Calibration Shift:** While unimodal models exhibited calibration slopes near $1.0$ ($0.987$ for Logistic A; $0.804$ for DenseNet121), unpenalized probability mixing in Fusion A introduced a positive intercept ($+0.656$) and steeper slope ($1.318$). Formal Platt scaling or isotonic recalibration would be required before clinical risk communication.

---

## 6. Operational & Engineering Constraints

- **Human Quality Control Requirement:** Automatic joint center localization was not $100\%$ autonomous; $422$ of $1,956$ reviewed panels ($21.6\%$) required manual center adjustment. Full automation in production workflows would necessitate more robust landmark detection.
- **Access-Controlled Data:** Raw OAI DICOM files and participant-level derived clinical tables cannot be publicly distributed. End-to-end reproduction of data ingestion requires formal application and authorization through the Osteoarthritis Initiative.
