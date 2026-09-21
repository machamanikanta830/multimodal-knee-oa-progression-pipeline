# Modeling Plan

## Status

Conceptual placeholder only. No models, encoders, feature pipelines, or data partitions are
implemented or trained in Milestone 0/1.

## Fixed concept

Future work will evaluate separate domain-specific models or encoders followed by late fusion.
The key experiment is a per-domain ablation comparing each eligible single domain with the
multimodal combination.

## Required comparison principles

- Use the same reviewed cohort, outcome, horizon, and participant-grouped evaluation framework
  for comparable model variants.
- Include meaningful non-ML or simple reference baselines before complex approaches.
- Report missingness and effective sample size for every model comparison.
- Prevent participant, contralateral-knee, visit, and temporal leakage.
- Assess discrimination, calibration, uncertainty, and clinically interpretable performance.
- Preserve reproducible provenance for inputs, configuration, code, and outputs.

## Deferred decisions

Human review is required before selecting:

- the outcome and prediction horizon;
- domain-specific inputs, representations, and architectures;
- eligibility for complete-case versus missingness-aware comparisons;
- development/validation/test design and participant grouping implementation;
- tuning strategy, evaluation metrics, confidence intervals, and subgroup analyses;
- calibration method and decision thresholds; and
- interpretation methods and the claims they can support.

Heavy modeling dependencies are intentionally absent from the initial environment. TensorFlow,
PyTorch, MONAI, XGBoost, MLflow, and similar packages should be considered only in a later,
reviewed milestone.
