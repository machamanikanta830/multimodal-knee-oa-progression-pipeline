# Baseline Feature Candidates

## Scope

This inventory is limited to observed V00 variables documented in the downloaded NDA/OAI
tables. It is a feasibility list, not a selected feature set. No association with outcomes was
used. “V00 missing” is release-wide baseline missingness from Milestone 1; left/right percentages
are shown separately where the source has two fields.

The strict complete-case section then evaluates the same candidates within two transparent
longitudinal reference populations. Participant variables are replicated to eligible knees only
in memory; side-specific fields are aligned by documented `side` (1 right, 2 left).

## Imaging and structural candidates

| Exact field/source | Level and side | Units/coding | V00 availability | Leakage/review status | Retain for review? |
|---|---|---|---|---|---|
| Baseline radiograph reached through `oai_kxrsemiquant01.barcode` → `oai_xrmeta01.barcode` → `image03.accession_number`/`image_file` | Image linked to participant-knee; bilateral barcode can serve both sides | Identifiers/paths, not numeric predictors | Project 15: 8,921/8,921 baseline knee records link to metadata and image index; pixels not downloaded | Image content is a safe baseline domain; identifiers/paths must never be predictors | Yes, after image download/QC review |
| `xrkl` | Knee + side + project + visit | K-L 0–4 | 0% missing in project 15 V00 | Requires review: legitimate baseline severity but closely related to a K-L change target | Yes, special-case comparison |
| `xrjsm`, `xrjsl` | Knee + side + project + visit | OARSI JSN 0–3 | 0% missing in project 15 V00 | Requires review or exclude from predictors when JSN change is the target | Yes, outcome/sensitivity planning |
| `examtype`, `accept` in `oai_xrmeta01` | Image/visit | Documented series type and QC categories | Complete in the X-ray metadata table | Linkage/QC only; using acquisition/QC metadata as predictors requires bias review | Yes for QC, not as default predictors |

The package contains an image index, not local pixels, so it supports linkage feasibility but not
image-encoder feasibility or imaging QC yet.

## Patient-reported candidates

| Exact variable(s) | Level/side | Coding | V00 missing | Leakage/review status | Retain? |
|---|---|---|---:|---|---|
| `womac_pain_left`, `womac_pain_right` | Knee-specific left/right | 0–20, higher worse | 0.000% / 0.063% | Safe baseline candidate | Yes |
| `womac_stiffness_left`, `womac_stiffness_right` | Knee-specific left/right | 0–8, higher worse | 0.104% / 0.021% | Safe baseline candidate | Yes |
| `womac_disability_left`, `womac_disability_right` | Knee-specific left/right | 0–68, higher worse | 0.480% / 0.354% | Safe baseline candidate | Yes |
| `koos_lkpain`, `koos_rkpain` | Knee-specific left/right | 0–100, lower worse | 0.042% / 0.063% | Safe baseline candidate | Yes |
| `koos_lksymptoms`, `koos_rksymptoms` | Knee-specific left/right | 0–100, lower worse | 0.000% / 0.000% | Safe baseline candidate | Yes |
| `koos_qol` | Non-lateral in dictionary | 0–100, lower worse | 0.021% | Requires review before sharing across both knees | Yes |
| `koos_sports` | Non-lateral in dictionary | 0–100, lower worse | 25.313% | Safe timing, but missingness and non-lateral interpretation require review | Yes, expanded panel only |

`womac_total_left`/`womac_total_right` are available but are calculated sums of the listed WOMAC
domains. Including totals alongside their components would be redundant and needs a prespecified
representation choice.

## Clinical and demographic candidates

| Exact variable(s) | Level/side | Units/coding | V00 missing | Leakage/review status | Retain? |
|---|---|---|---:|---|---|
| `ageyears` | Participant | Years | 0.000% | Safe baseline candidate | Yes |
| `sex` | Participant | M/F/O/NR per NDA | 0.000% | Safe baseline candidate; subgroup/fairness interpretation required | Yes |
| `race`, `ethnicity` | Participant | NDA categories | 0.000% / 0.000% | Requires scientific, equity, and transportability justification | Yes for review, not automatic inclusion |
| `bmi` | Participant/visit | Body mass index; documented sentinels -9/-5/-2, none observed at V00 | 0.083% | Safe baseline candidate | Yes |
| `e_cohort` | Participant | 1 progression, 2 incidence, 3 non-exposed control | 0.000% | Requires review: baseline design variable that encodes sampling/risk enrichment | Yes for design adjustment review |
| `site` | Participant | Study site | 0.000% | Requires review: possible technical/site shortcut and transportability issue | Yes for design/QC review |
| `ksurgl`, `ksurgr` | Knee-specific left/right | 0 no, 1 yes; prior surgery/arthroscopy | 0.188% / 0.167% | Baseline-safe timing; broad surgery definition and confounding require review | Yes |
| `famkr` | Participant | Blood relative had knee replacement for arthritis; documented yes/no item | 1.314% | Safe baseline candidate | Yes |
| `injl`, `injr` | Knee-specific left/right | Direct injury history item | 100% / 100% | Unusable in this export; exclude | No |
| `injl1`, `injr1` | Knee-specific conditional age-at-first-injury | Age | 74.896% / 71.726% | Does not safely substitute for the empty direct injury indicator | Review only |

Baseline `ksurgcv` is well populated and identifies prior total/partial replacement at the
participant/either-knee level, but it cannot assign the affected side. Side-specific baseline
replacement and exclusion rules should instead use reviewed V99 baseline X-ray flags and other
documented evidence; this remains a cohort issue, not a default predictor.

## Objective physical-function candidates

| Exact variable(s) | Level/side | Units/coding | V00 missing | Leakage/review status | Retain? |
|---|---|---|---:|---|---|
| `w20mpace` | Participant/visit | m/s | 0.438% | Safe baseline candidate; shared across both knees | Yes |
| `cstime1` | Participant/visit | Seconds and hundredths | 5.150% | Safe baseline candidate; shared across both knees | Yes |
| `cspace` | Participant/visit | Stands/second | 5.046% | Derived from chair-stand test; choose time or pace, not both silently | Yes for review |
| `w400mcmp` | Participant/visit | 1 completed no stop; 2 completed with rests; 3 attempted/incomplete; 4 excluded; 5 other non-attempt | 2.043% | Requires handling as status, not continuous performance | Yes |
| `w400mtim` | Participant/visit | Seconds to 400 m or stopping point | 4.817% | Requires joint interpretation with completion status | Yes |
| `lemaxf`, `remaxf` | Knee-specific left/right | Maximum extension force, N | 12.156% / 12.052% | Safe baseline candidate; missingness/measurement protocol review | Yes |
| `lfmaxf`, `rfmaxf` | Knee-specific left/right | Maximum flexion force, N | 12.302% / 12.198% | Safe baseline candidate; missingness/measurement protocol review | Yes |

The physical-performance domain is plausible: walking and chair-stand measures are highly
populated, and side-specific strength is available for about 88% at baseline. Accelerometry is
OAI-linked but occurs only at V06/V08 and is excluded from the baseline design.

## Missingness in longitudinal reference populations

The project 15/V06 reference contains 7,014 K-L-eligible knees from 3,584 participants. The most
important knee-level baseline missingness within it is:

| Candidate | Missing knees | Missing |
|---|---:|---:|
| WOMAC pain | 0 | 0.000% |
| WOMAC stiffness | 5 | 0.071% |
| WOMAC disability | 21 | 0.299% |
| KOOS pain | 1 | 0.014% |
| KOOS symptoms | 0 | 0.000% |
| KOOS QoL | 2 | 0.029% |
| KOOS sports | 1,617 | 23.054% |
| BMI | 4 | 0.057% |
| Side-specific prior surgery | 14 | 0.200% |
| Family knee-replacement history | 102 | 1.454% |
| 20-m pace | 21 | 0.299% |
| Chair-stand time | 311 | 4.434% |
| 400-m completion status / time | 160 / 319 | 2.281% / 4.548% |
| Extension / flexion force | 725 / 730 | 10.336% / 10.408% |

### Strict complete-case impact

The expanded PRO panel contains all seven listed PRO candidates; clinical contains age, sex,
race, ethnicity, cohort, site, BMI, side-specific prior surgery, and family history; function
contains 20-m pace, chair time, 400-m status/time, and extension/flexion force. Imaging is complete
by construction in these K-L-eligible X-ray reference populations.

“Participants lost” means participants with no complete eligible knee; it does not require both
knees to be complete.

| Reference and panel | Eligible knees / participants | Complete knees | Knees lost | Participants with any complete knee | Participants lost |
|---|---|---:|---:|---:|---:|
| Project 15/V06 — imaging | 7,014 / 3,584 | 7,014 | 0 | 3,584 | 0 |
| Project 15/V06 — PRO expanded | 7,014 / 3,584 | 5,383 | 1,631 | 2,733 | 851 |
| Project 15/V06 — clinical expanded | 7,014 / 3,584 | 6,894 | 120 | 3,525 | 59 |
| Project 15/V06 — function expanded | 7,014 / 3,584 | 5,900 | 1,114 | 3,020 | 564 |
| Project 15/V06 — all expanded candidates | 7,014 / 3,584 | 4,469 | 2,545 | 2,272 | 1,312 |
| Project 37/V10 — all expanded candidates | 3,237 / 2,006 | 2,208 | 1,029 | 1,356 | 650 |

A transparent reduced sensitivity panel—PRO without KOOS sports; clinical without family history;
function limited to 20-m pace, chair time, and extension force—retains 5,980/7,014 project-15/V06
knees (85.258%) and 3,061/3,584 participants with at least one complete knee. This is only a
missingness sensitivity calculation, not a selected v1 feature set.

Strict all-expanded complete-case analysis would discard 36.285% of eligible project-15/V06
knees, driven mainly by KOOS sports and strength. It should not be adopted automatically.

## Missing-data strategies for later review

No imputation was performed. Later options include:

- a prespecified complete-case analysis with explicit effective sample sizes;
- median/mode imputation learned within development data only;
- model-native missing handling where technically appropriate;
- missing-indicator features when absence is meaningful and not a leakage proxy; and
- multiple imputation with a statistical model compatible with knee clustering and analysis goals.

All fitting parameters for any future imputation must be learned within participant-grouped
development partitions. Outcome availability and future visits must never enter imputation of
baseline predictors.
