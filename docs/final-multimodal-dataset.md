# Milestone 6A: frozen final multimodal dataset

The sole authoritative input to future split generation is
`data/processed/multimodal/final_v1/final_multimodal_dataset.parquet`, verified through its
`dataset_freeze.json`. Future experiments must not independently reconstruct the cohort. No splits,
models, imputers, encoders or scalers are created or fitted in this milestone.

## Source mapping

The locked cohort has 43 columns; the adjudicated imaging manifest has 59. Join exactly one-to-one on
`participant_id + knee_side_code + baseline_visit`, preserving cohort row order. Source participant
identity and side labels must agree. Code 1 is RIGHT/screen-left; code 2 is LEFT/screen-right. Baseline
is V00 and the planned outcome visit is V06 (approximately 48 months), under reading project 15.

Targets are copied without recomputation: `composite_progression` (primary),
`radiographic_kl_progression` (radiographic sensitivity), and `jsn_progression` (secondary).
`replacement_before_v06` and endpoint eligibility flags are outcome-supporting metadata only.
Raw V06 KL/JSN grades and replacement dates are not columns in the locked cohort and are not added.

All 14 baseline tabular candidates come from the existing `configs/features_v1.yaml`:

| Domain | Locked output columns | Existing baseline sources |
|---|---|---|
| Patient reported | `womac_pain`, `womac_stiffness`, `womac_disability`, `koos_pain`, `koos_symptoms` | Side-matched left/right WOMAC and KOOS in `oai_koos_womac01` |
| Demographic/clinical | `age_years`, `sex`, `bmi`, `prior_knee_surgery`, `family_knee_replacement_history` | `ageyears`, `sex`; `bmi`, side-matched `ksurgl/ksurgr`, `famkr` |
| Physical function | `walk_20m_pace_mps`, `chair_stand_time_seconds`, `knee_extension_strength_n`, `knee_flexion_strength_n` | `w20mpace`, `cstime1`, side-matched `lemaxf/remaxf` and `lfmaxf/rfmaxf` |

Exact source files, variables, units, levels, timing and descriptions are carried into `schema.json`
from the reviewed feature configuration and existing endpoint code/documentation. No new OAI variable
is introduced. Age, sex, BMI, family history, walk and chair measures are participant-level values
repeated across eligible knees, not newly interpreted as knee-specific measurements.

## Table and feature policy

The table has 54 columns: 35 retained cohort fields and 19 imaging fields. The source barcode and
seven redundant availability/formulation markers remain in the unchanged cohort rather than being
copied into the modeling table. Detailed human adjudication, corrected-center, automatic-crop and
technical audit fields remain in the unchanged imaging manifest, referenced by path/SHA in the freeze.

Formulation A is the default: 14 tabular candidates plus the final baseline pixels. Formulation B
adds `baseline_kl` (stored severity, `baseline_severity_feature=true`, `default_predictor=false`).
Neither formulation includes baseline JSN grades, targets, outcome availability, replacement history
after baseline, identifiers, race/ethnicity descriptive fields, sampling/site design, or image-QC
provenance as predictor values. Baseline JSN remains target-supporting provenance only.

`feature_groups.json` is the explicit loading contract. Its formulation lists separate
`tabular_predictors` from `image_loading_references`. The image-path column has predictor eligibility
flags **false**, because path strings must never become disease predictors; the baseline pixels it
loads are the imaging input in both formulations. No path, acquisition ID, SHA, provenance category,
laterality/adjudication flag or geometry metadata may be automatically encoded as a tabular feature.

Every table column belongs to exactly one of ten named groups and has a schema record with dtype,
description, source/domain, timing, predictor/formulation/target/leakage flags and missingness. Unknown
source or final columns fail closed rather than being silently promoted, discarded or deduplicated.

## Missingness and quality

No missing value is filled and no complete-case selection changes the cohort. Radiographic KL/JSN
labels remain unavailable for the 110 locked ineligible replacement-branch knees, not non-events.
The locked materializer collapsed baseline blanks, configured sentinels and unsuccessful numeric
parses into missing values; it did not retain per-cell missing-reason codes. Therefore their exact
division into true missing measurement, not applicable and structural missingness is **unknown**.
Audit category counts use JSON null to denote unknown, not zero, and never invent missing mechanisms.

The audit includes all-column and outcome missingness, each group's complete-case counts, modality
overlap/patterns, finite numeric checks, observed numerical ranges/categories and checks against ranges
explicitly documented in the existing sources. This is descriptive QA, not fitted preprocessing or
outcome-driven feature selection. Future missing-data treatment and encoding must be split-specific.

## Freeze and verification

```sh
.venv/bin/python -m multimodal.freeze
```

Production execution requires the pre-change preservation fingerprint stored in the output directory.
It verifies every cohort effective file for canonical acquisition/panel path, existence, SHA, side,
shape and dtype, and compares all protected fingerprints before and after construction: cohort,
manifest, queue/log/history/laterality, automatic pixels and records, override bundles, approved
imaging artifacts, frozen methodology/configuration, raw image archives and raw tabular files.
It does not invoke cohort reconstruction, imaging generation or review persistence.

Parquet readback must preserve exact values/dtypes and row ordering. Existing identical artifacts
are retained byte-for-byte; differences cause an error. JSON creation time is preserved on rerun.
The freeze marker is published last, with dataset/schema/groups/audit SHA values, authoritative
source hashes, source/code/configuration hashes, runtime versions and the participant grouping rule.

Use `multimodal.freeze.load_frozen_dataset()` to verify and read the frozen input without rebuilding
the cohort. All participant-level artifacts are local restricted research data, not public exports;
tracked documentation and command output are aggregate-only. No private URLs/archive GUIDs are added.

The future workflow is:

```text
frozen final multimodal dataset
    -> participant-grouped split manifest (both knees share partition)
    -> split-specific preprocessing
    -> models
```

This milestone stops at the dataset freeze.
