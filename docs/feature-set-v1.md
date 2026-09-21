# Baseline Feature Set v1

## Selection policy

`configs/features_v1.yaml` is the executable source of truth. Features were frozen using
scientific relevance, V00 availability, correct participant/knee measurement level, documented
side matching and coding, observed missingness, and leakage safety. No association with any
outcome label was calculated or used for selection.

## Primary panel

| Domain | Output | Exact OAI source | Level / side rule | Documented scale or unit | Composite missing |
|---|---|---|---|---|---:|
| Imaging | baseline radiograph | `oai_kxrsemiquant01.barcode` → `oai_xrmeta01.barcode` → `image03.accession_number` | participant-knee linked to bilateral acquisition | X-ray/DICOM metadata; pixels not downloaded | 0.000% linkage missing |
| PRO | `womac_pain` | `womac_pain_left`, `womac_pain_right` | knee; side matched | 0–20, higher worse | 0.000% |
| PRO | `womac_stiffness` | `womac_stiffness_left`, `womac_stiffness_right` | knee; side matched | 0–8, higher worse | 0.072% |
| PRO | `womac_disability` | `womac_disability_left`, `womac_disability_right` | knee; side matched | 0–68, higher worse | 0.330% |
| PRO | `koos_pain` | `koos_lkpain`, `koos_rkpain` | knee; side matched | 0–100, lower worse | 0.029% |
| PRO | `koos_symptoms` | `koos_lksymptoms`, `koos_rksymptoms` | knee; side matched | 0–100, lower worse | 0.000% |
| Clinical | `age_years` | `oai_enrollee01.ageyears` | participant; shared across knees | years | 0.000% |
| Clinical | `sex` | `oai_enrollee01.sex` | participant; shared across knees | documented categorical value | 0.000% |
| Clinical | `bmi` | `oai_oarisk01.bmi` | participant; shared across knees | kg/m²; documented sentinels missing | 0.057% |
| Clinical | `prior_knee_surgery` | `ksurgl`, `ksurgr` | knee; side matched | 0 no, 1 yes | 0.187% |
| Clinical | `family_knee_replacement_history` | `oai_oarisk01.famkr` | participant; shared across knees | documented yes/no item | 1.422% |
| Physical function | `walk_20m_pace_mps` | `oai_physfunct01.w20mpace` | participant; shared across knees | metres/second | 0.302% |
| Physical function | `chair_stand_time_seconds` | `oai_physfunct01.cstime1` | participant; shared across knees | seconds and hundredths | 4.540% |
| Physical function | `knee_extension_strength_n` | `lemaxf`, `remaxf` | knee; side matched | newtons | 10.386% |
| Physical function | `knee_flexion_strength_n` | `lfmaxf`, `rfmaxf` | knee; side matched | newtons | 10.458% |

The panel deliberately represents both knee-specific and participant-level constructs. A
participant-level value is repeated across that participant's eligible knees with its level
recorded in configuration; it is not misrepresented as a knee-specific measurement.

## Stored but not default predictors

- Baseline KL is retained for prespecified formulation B; formulation A without explicit KL is
  the default.
- Race and ethnicity are retained locally for cohort description and responsible subgroup
  evaluation, not as default biological predictors.
- Site and OAI cohort are retained for design and QC evaluation, not as default predictors.
- Baseline medial/lateral JSN values support structural outcome provenance but are excluded from
  the primary predictor panel because JSN change is a sensitivity endpoint.
- Identifiers, barcodes, accessions, file paths, read project, visits, eligibility masks, and all
  outcome fields are never predictor inputs.

## Explicit exclusions from the concise primary panel

- WOMAC total: redundant with the chosen subscales.
- KOOS quality of life: not lateral in the downloaded dictionary.
- KOOS sport/recreation: not lateral and about one-quarter missing at V00.
- 400-m walk completion/time: the time is conditional on protocol completion and requires joint
  handling; excluded pending protocol review.
- Chair-space calculation: derived from the selected chair-stand test.
- `injl`/`injr`: completely empty in this release; conditional age fields are not substitutes for
  an unavailable injury indicator.
- Accelerometry: present only after V00 in this package.
- Every post-baseline or V99 variable: outcome/leakage risk.

## Deferred preprocessing

No imputation, scaling, winsorization, categorization, image preprocessing, or feature reduction
has occurred. These operations must later be specified using development data only. Categorical
levels and missingness indicators, if used, must be learned/defined without test-set leakage.
