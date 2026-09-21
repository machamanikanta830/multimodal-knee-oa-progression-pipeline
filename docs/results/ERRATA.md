# Scientific Erratum: Multimodal Fusion B Validation Presentation

**Issue Identifier:** ERRATUM-2026-09-H01  
**Target Artifact:** `docs/results/validation_vs_test_summary.json`  
**Status:** Preserved Archival Recovered Artifact with Documentary Erratum  
**Impact on TEST Results:** ZERO (Held-out TEST evaluations unaffected)  
**Recomputation Performed:** ZERO (Transcribed from existing authoritative freeze)

---

## 1. Summary

An independent Codex audit confirmed that the archival artifact `docs/results/validation_vs_test_summary.json` contains a presentation of VALIDATION metrics for **Multimodal Fusion B (Sensitivity)** that does not correspond to the authoritative selected sensitivity model.

To maintain absolute provenance integrity under the project's research governance protocol, `validation_vs_test_summary.json` is **preserved byte-for-byte** as an archival recovered artifact (SHA-256: `0baed5db4ea8f20b44029956ba93042eb059b8e3c131e2e2df592ceefc15f447`).

This document provides the authoritative correction transcribed directly from the existing locked private development artifacts.

---

## 2. Discrepancy Analysis

| Field | Archival Presentation in `validation_vs_test_summary.json` | Authoritative Frozen Value (`selected_fusion.json` & `final_model_freeze.json`) | Discrepancy Note |
|---|---|---|---|
| **Selected $\alpha$ (Image Weight)** | Unspecified in table | **0.60** (60% image, 40% tabular) | Prespecified 11-point grid search winner |
| **Validation AUROC** | 0.703276 | **0.694892** (0.6948924825728214) | -0.008384 discrepancy in presentation |
| **Validation AUPRC** | 0.298246 | **0.299284** (0.29928367626042385) | +0.001038 discrepancy in presentation |
| **Validation Brier Score** | 0.122188 | **0.122701** (0.12270106933826550) | +0.000513 discrepancy in presentation |
| **Validation Log Loss** | N/A | **0.402195** (0.40219474908343694) | Unchanged |
| **Validation $N$ / Events** | 1,044 / 161 | **1,044 / 161** | Exact match |

---

## 3. Authoritative Source Artifacts

The authoritative values above are verified directly against the existing private frozen development ledgers established prior to held-out TEST evaluation:
1. `data/processed/modeling/fusion/v1/selected_fusion.json` (lines 68–88)
2. `data/processed/modeling/fusion/v1/validation_metrics.json`
3. `data/processed/modeling/final_freeze/v1/final_model_freeze.json` (lines 12–35)

---

## 4. Scope & Scientific Impact

1. **Held-Out TEST Evaluation Unaffected:**  
   The held-out TEST evaluation for Fusion B used the authoritative frozen alpha ($\alpha = 0.60$) and yielded TEST AUROC **0.735149**, TEST AUPRC **0.373690**, and TEST Brier **0.116845** across $N=1,044$ knees ($161$ events). These results are unaffected.
2. **Zero Metric Recomputation:**  
   No models were refit, no predictions were rerun, and no bootstrap intervals were recomputed. This erratum documents existing authoritative recorded values only.
3. **Sensitivity Role Preserved:**  
   Multimodal Fusion B incorporates baseline Kellgren-Lawrence grade (`baseline_kl`) and remains strictly a prespecified sensitivity analysis. Primary Multimodal Fusion A ($\alpha = 0.50$, combining DenseNet121 with Logistic Formulation A without baseline KL) remains the primary scientific model of this study.
