# Milestone 6C: TRAIN-fitted tabular development baselines

The sole source is the approved frozen multimodal dataset and approved participant/knee split.
No independent cohort reconstruction, resplitting or model-specific complete-case filtering occurs.
TRAIN: 4,873 knees/749 events; VALIDATION: 1,044 knees/161 events. TEST is locked and never scored.
The loader verifies exact approved source hashes plus 6A/6B freeze catalogs before returning only
TRAIN and VALIDATION feature/primary-target matrices. TEST identity labels may be read for structural
membership protection; no TEST prediction, model metric, calibration, threshold or error analysis occurs.

## Feature/type contract

Exact existing predictors:

- Clinical/demographic: `age_years`, `sex`, `bmi`, `prior_knee_surgery`,
  `family_knee_replacement_history`.
- Patient reported: `womac_pain`, `womac_stiffness`, `womac_disability`, `koos_pain`, `koos_symptoms`.
- Physical function: `walk_20m_pace_mps`, `chair_stand_time_seconds`,
  `knee_extension_strength_n`, `knee_flexion_strength_n`.
- Formulation A (default): all fourteen above, **no KL**.
- Formulation B (sensitivity): those fourteen plus ordinal numeric `baseline_kl` (0–3).

Frozen dtypes are nullable integers/floats, except categorical `sex` (string). The documented binary
history/surgery fields are categoricals despite integer storage. TRAIN sex categories are F/M;
history/surgery codes are 0/1. Preserve source tokens, without inventing additional coding/semantics.
Convert nullable integer categoricals through object scalars before string conversion, so the same
code is represented identically whether or not a partition contains missing measurements.
Age/BMI, WOMAC/KOOS scores and physical measures are numeric; KL is ordered numeric in B only.
Targets, future availability, replacements, QC/provenance, identifiers and images are never predictors.
No KL-derived feature or KL missingness indicator enters A. Domain definitions are unchanged.

## Predeclared models and selection

The configuration in `configs/tabular_v1.yaml` is defined before any production fitting.
Fit both families separately for each of the five feature sets. All models are unweighted; no
class weighting, accuracy selection, resampling, class weights or threshold optimization is used.

L2 logistic regression: `C` in 0.1, 1, 10; lbfgs, l1_ratio=0, 2,000 maximum iterations, tolerance 1e-8.
Numeric TRAIN-median imputation followed by TRAIN StandardScaler; categorical TRAIN-most-frequent
imputation then TRAIN one-hot encoding (drop first, unknown categories safely ignored). The pipeline
couples deterministic dtype conversion, ColumnTransformer and estimator. All-missing TRAIN features
fail rather than disappearing. Validation transformation never refits. Unknown-category all-zero
encoding is the documented dropped-reference behavior, not evidence of a learned validation category.

CatBoost CPU/single-thread: (depth, learning rate, max iterations, L2) candidates
(4,.03,800,3), (4,.06,800,5), (6,.03,800,5). Native numeric NaN, no numeric imputation/scaling;
categorical missing values become a fixed `__MISSING__` string. Native categorical/border statistics
learn from TRAIN only, explicitly `counter_calc_method=SkipTest`. VALIDATION AUC selects the retained
iteration with 80-round early stopping. No global dataset imputation precedes either family.

See the official [CatBoost missing-value policy](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing)
and [scikit-learn unknown-category policy](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

Selection: validation AUROC descending, AUPRC descending, Brier ascending, then candidate order.
AUPRC uses sklearn average precision (step-integrated precision–recall, not trapezoidal PR area).
Also report log loss, prevalence/positive count and calibration intercept/slope. AUPRC no-skill
reference is VALIDATION prevalence. The sanity reference predicts TRAIN prevalence on VALIDATION.
Constant-reference joint calibration intercept/slope is not identifiable and is reported as null.

Calibration uses an unpenalized diagnostic logistic regression of validation outcomes against the
logit of predicted probabilities, with joint intercept and slope. This is assessment only; these
diagnostic coefficients are never applied to predictions. A quantile-binned validation curve is
reported/visualized. No final calibration model is fitted, selected or saved for deployment.

## Uncertainty, artifacts and determinism

Model seed: 62026; bootstrap seed: 62027. Candidate order is explicit. Paired 1,000-replicate
VALIDATION participant bootstrap samples participants with replacement and includes every eligible
knee each time its participant is drawn. Store replicate AUROC/AP/Brier and 95% percentile intervals.
Single-class replicates are flagged/excluded from AUROC/AP, not silently counted as valid. Intervals
are conditional on the fitted/selected development model and do not correct validation-selection
optimism or represent model-training uncertainty. No statistical-superiority claims are made.

Version-pinned optional `[modeling]` dependencies are binary-compatible with the inspected Python
3.13 Apple Silicon environment. Fixed-seed CPU single-thread refits are checked for validation
prediction equality at absolute tolerance 1e-12. CatBoost's native serialization may include
volatile fit timestamps/GUIDs; on rerun existing original fitted files are hash-verified and reused
after semantic preprocessing/prediction comparison. They are not needlessly rewritten.

```sh
.venv/bin/python -m modeling.tabular
```

Outputs: `data/processed/modeling/tabular/v1/`. Every candidate pipeline is saved under `models/`.
Every pipeline links dataset/split/group hashes, model/preprocessing configuration hashes, code,
seeds and TRAIN/VALIDATION counts. Selected candidate references, all candidate validation metrics,
selected validation predictions, missingness/TRAIN-imputation/scaling audits, bootstrap replicates/
intervals, calibration data/plot, feature lists, coefficient/native-importance summaries and run
freeze catalog are saved. Predictions contain only internal participant ID, side, visit, primary
label, probability and model ID—no raw imaging URLs or unnecessary identifiers.

Writers are serialized in the new modeling directory. Existing differing/corrupted frozen artifacts
fail closed; trusted original artifacts are reused. Full pre/post protected-source fingerprints
cover the original multimodal/split bundles, cohort, imaging manifest, adjudications, overrides,
automatic pixels/processing records, frozen imaging methodology and raw data. No source is modified.

Both all-tabular A family baselines advance as reference candidates for later imaging/multimodal
comparison. B is explicitly sensitivity-only even if better on validation; selected single domains
remain for ablations. This does not declare a final project model or scientifically superior domain.
Limited interpretation: standardized logistic coefficients and native PredictionValuesChange
CatBoost importances only; no significance/causal claims and no elaborate SHAP analysis.

The initial unapproved development run is retained at
`data/processed/modeling/tabular/v1_superseded_conversion_bug/`, not an authoritative model bundle.
It exposed a pandas nullable-integer mapping issue (0/1 became 0.0/1.0 only in partitions with
missing values). The adapter was corrected and both native/one-hot regression cases added before
refitting the identical predeclared search. No source dataset, split, feature or outcome was changed.
