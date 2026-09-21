# Data Access, Governance & Participant Privacy

This document describes the data governance policies, participant privacy safeguards, and access procedures governing the Osteoarthritis Initiative (OAI) dataset.

---

## 1. Access-Controlled Research Data Status

The data analyzed in this research originate from the Osteoarthritis Initiative (OAI), a public-private partnership sponsored by the National Institutes of Health (National Institute of Arthritis and Musculoskeletal and Skin Diseases [NIAMS]).

- **Non-Public Redistribution Restrictions:** In accordance with OAI access and usage terms, raw OAI data files, imaging archives, and participant-level derived analytical files are **access-controlled** and cannot be redistributed publicly. This public repository is designed not to distribute the restricted participant-level OAI data used in the study. Repository content controls are distinct from external institutional data-use obligations.
- **Excluded Materials:** The following categories of data are maintained strictly in local storage and are excluded from public repository distribution:
  - Raw OAI DICOM image files and compressed archives (`.tar.gz`).
  - Raw OAI clinical, demographic, and laboratory tables (`.txt`).
  - Standardized anatomical radiograph crops (`.uint16` arrays and PNG panel views).
  - Participant-identifiable or participant-level derived datasets (`analysis_cohort_v1.parquet`, `final_multimodal_dataset.parquet`).
  - Participant-level split manifests (`participant_split_manifest.parquet`).
  - Prediction-level Parquet artifacts containing individual participant outputs (`aligned_test_predictions.parquet`, etc.).
  - Human review adjudication panels displaying participant radiographs.

---

## 2. Participant Privacy Safeguards

The pipeline enforces multiple levels of participant privacy protection:
1. **De-Identification Protocol:** The original OAI study applied de-identification protocols removing direct personal identifiers (names, addresses, telephone numbers, direct dates of birth) in accordance with clinical research ethics.
2. **Local Isolation:** All participant-level data reside under the `data/` directory hierarchy, which is strictly managed by local ignore rules.
3. **Aggregate-Only Public Documentation:** All tracked documentation, markdown reports, and summary artifacts report aggregate statistics only (e.g., sample sizes, mean values, metric distributions, and non-identifiable architecture diagrams). Zero participant identifiers or individual-level records are published.
4. **Blinded Adjudication:** Reviewers involved in human quality control observed masked anonymous bilateral context, coordinate overlays, candidate joint centers, QC evidence, and joint crops to perform localization and cropping quality control. Reviewers were strictly blinded to participant identifiers, clinic site locations, Kellgren-Lawrence grades, and clinical progression outcomes.

---

## 3. Obtaining Authorized Access to OAI Data

Researchers wishing to access the underlying OAI dataset can obtain authorized access directly from the coordinating entity:

1. **Access Portal:** Access requests and account registration are managed through the official OAI data repository:  
   [https://nda.nih.gov/oai/](https://nda.nih.gov/oai/) (National Institute of Mental Health Data Archive / OAI portal)
2. **User Requirements:** Prospective users must register for an account, agree to the OAI access and usage terms, and provide institutional research affiliation.
3. **Available Packages:**
   - Clinical datasets: Baseline (V00) through 96-month follow-up clinical and patient-reported tabular files.
   - Imaging datasets: Baseline bilateral fixed-flexion digital radiographs (Clinical Center Project readings and primary image archives).

---

## 4. Official OAI / NIAMS Attribution & Disclaimer

In adherence to OAI publication and dissemination guidelines, all scientific communications utilizing OAI resources acknowledge the consortium:

> The Osteoarthritis Initiative (OAI) is a public-private partnership comprised of five contracts (N01-AR-2-2258, N01-AR-2-2259, N01-AR-2-2260, N01-AR-2-2261, and N01-AR-2-2262) funded by the National Institutes of Health (NIH), a branch of the Department of Health and Human Services, and conducted by the OAI Study Investigators. Private funding partners include Merck Research Laboratories, Novartis Pharmaceuticals Corporation, GlaxoSmithKline, and Pfizer, Inc. Private partner funding is managed by the Foundation for the National Institutes of Health (FNIH). This repository and its research documentation were prepared using an OAI public-use dataset and do not necessarily reflect the opinions or views of the OAI investigators, the NIH, or the private funding partners.
