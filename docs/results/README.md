# Aggregate Scientific Results & Provenance

This directory contains the authoritative, public aggregate scientific results from the locked held-out TEST evaluation of the multimodal knee osteoarthritis progression pipeline.

## Provenance Statement

1. **One-Time Scientific Evaluation:** The locked held-out TEST cohort ($N = 1,044$ knees across $543$ participants) was evaluated exactly once during Milestone 6G-B under automated fail-closed governance.
2. **Archival Recovery Provenance:** Following forward inference and metric computation, an unhandled post-scoring serialization exception interrupted routine disk writes (`KeyError: 'sensitivity_models'`). Aggregate summaries were serialized during archival recovery from retained first-run recorded values. Participant-level frozen prediction parquets remain preserved privately in local research storage. Replicate-level bootstrap samples were not persisted to disk; only summary percentile intervals and median paired differences are publicly available.
3. **Integrity Verification:** Cryptographic SHA-256 digests prove the post-creation integrity of the archived artifacts, rather than independent recomputation of the original in-memory state.
4. **Zero Scientific Recomputation:** Zero model weights, predictions, metrics, or bootstrap iterations were recomputed, refit, or rerun.
5. **Data Privacy Safeguards:** Individual participant identifiers, row-level predictions, and raw imaging/tabular data are strictly excluded from public distribution in compliance with Osteoarthritis Initiative (OAI) data use agreements and NIH/NIAMS governance. The files in this directory contain aggregate cohort-level statistics only.

## Included Artifacts

| Filename | Description | SHA-256 Digest |
|---|---|---|
| `test_metrics.json` | Final primary and sensitivity performance metrics (AUROC, AUPRC, Brier, Log Loss, calibration parameters) on the held-out TEST partition. | `d001a21938e5406a440dd82ea883acdf1f088a6ebb3999743cd29f2e8ad63702` |
| `clustered_bootstrap.json` | 1,000 participant-clustered bootstrap percentile 95% confidence intervals (seed 65027). Summary intervals only. | `67d493173c2f50091c7e2ce22ec643d25afbd1ae87dddbd4c55d80eeee7064ec` |
| `paired_bootstrap_differences.json` | Paired bootstrap differences comparing Multimodal Fusion A against unimodal image and tabular baselines. | `ed0eef81461ebac4693bc85faecc74a31ae47262917bd88dadbdd2d7c993f0d0` |
| `calibration_data.json` | Diagnostic calibration slope and intercept parameters on held-out TEST data. | `b99c74e7bcc1cb6880d91acbb0ed369377db497b3feaf7ac9efabb7f50caca06` |
| `validation_vs_test_summary.json` | Archival comparative table of VALIDATION versus held-out TEST performance across candidate formulations. Preserved byte-for-byte; see ERRATA.md. | `0baed5db4ea8f20b44029956ba93042eb059b8e3c131e2e2df592ceefc15f447` |
| `ERRATA.md` | Authoritative scientific erratum documenting the preserved archival status of `validation_vs_test_summary.json` and transcribing frozen Fusion B validation metrics. | N/A (Documentation) |
| `errata.json` | Machine-readable documentary erratum record containing authoritative frozen Fusion B validation metrics. | N/A (Documentation) |
