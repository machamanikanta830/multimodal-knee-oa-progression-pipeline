# Data Dictionary

## Milestone 1 scope and evidence

This provisional dictionary contains only variables observed in the authorized package at
`data/raw/oai/`. Descriptions come from the embedded second-line NDA dictionary in each table and
were cross-checked against the official NDA structure API. Coding and units come from those
official definitions where available. Availability and missingness are observations from the
download, not general claims about every OAI release.

Evidence sources:

- embedded field-description row in each downloaded NDA table;
- [official NDA data-structure API](https://nda.nih.gov/api/datadictionary/v2/datastructure/oai_enrollee01)
  using the exact table short name;
- [OAI central image assessment guide](https://nda.nih.gov/static/docs/ImageAssessmentDataOverview.pdf);
- [OAI questionnaire schedule](https://stage-strapi-public-uploads.s3.amazonaws.com/QuestionnaireMeasures.pdf);
- [OAI examination schedule](https://stage-strapi-public-uploads.s3.amazonaws.com/ExamMeasures.pdf).

`V00 missing` is aggregate blank/sentinel missingness among downloaded V00 records. For X-ray
tables it is record-level, not participant-level. Empty structural fields are explicitly marked
unusable. No variable has been selected for a cohort or final outcome.

## Linkage and time

| Variable | Source | Documented meaning | Level | Coding/units | Observed availability | Missing/uncertainty |
|---|---|---|---|---|---|---|
| `subjectkey` | all tables | NDA Global Unique Identifier for research subject | Participant | GUID, `NDAR*` | All table visits | 0 blank; do not expose values |
| `src_subject_id` | all tables | Subject ID defined by the source project | Participant | String | All table visits | 0 blank; one-to-one with `subjectkey` within every table |
| `visit` | all tables | Visit name | Assessment | String | Structure-specific | Codes observed from V00 through V11 plus V99; mapping below |
| `side` | knee/image tables | Side | Knee/image | 0 neither; 1 right; 2 left; 3 bilateral; blank incomplete | X-ray assessment visits | Only 1/2 observed in assessment tables; 1/2/3 in X-ray metadata |
| `readprj` | X-ray assessment tables | Image assessment project | Knee/project | String | Project-specific | Required with participant, visit, and side to identify assessment records |
| `barcode` | X-ray assessment/metadata | Barcode of image analyzed | Image | String | X-ray visits | 184 blanks in quantitative JSW; otherwise complete in inspected X-ray tables |
| `accession_number` | `image03.txt` | Accession number of image analyzed | Image | String | Image visits | Complete and unique across 435,558 records; candidate link to `barcode` |

Observed/documented visit mapping:

| Code | Timepoint |
|---|---|
| `V00` | Baseline |
| `V01` | 12 months |
| `V02` | 18 months |
| `V03` | 24 months |
| `V04` | 30 months |
| `V05` | 36 months |
| `V06` | 48 months |
| `V07` | 60 months |
| `V08` | 72 months |
| `V09` | 84 months |
| `V10` | 96 months |
| `V11` | 108 months |
| `V99` | Derived outcome-summary table, not a clinical visit |

The V00/V01/V02/V04/V06/V08/V10 labels are also explicit in official NDA OAI study titles; the
full sequence is consistent with the visit coding documented in `oai_outcome01`. Later timepoints
and `V99` still require confirmation against the OAI visit-prefix guide before cohort code.

## Demographic and clinical candidates

| Variable | Source | Documented meaning | Level | Coding/units | Visits | V00 missing |
|---|---|---|---|---|---|---:|
| `ageyears` | `oai_enrollee01.txt` | Age in years | Participant | Integer years | V00 | 0.000% |
| `interview_age` | all tables | Age at assessment | Participant/visit | Integer months | All | 0% in key tables |
| `sex` | all tables | Sex at birth | Participant | M, F, O, NR | All | 0.000% in enrollee |
| `ethnicity` | `oai_enrollee01.txt` | Participant ethnicity | Participant | Documented categories | V00 | 0.000% |
| `race` | `oai_enrollee01.txt` | Participant race | Participant | Documented categories | V00 | 0.000% |
| `e_cohort` | `oai_enrollee01.txt` | OAI subcohort assignment | Participant | 1 progression; 2 incidence; 3 non-exposed control | V00 | 0.000% |
| `site` | `oai_enrollee01.txt` | Study site | Participant | String | V00 | 0.000% |
| `bmi` | `oai_oarisk01.txt` | Body mass index | Participant/visit | Float; API notes -9/-5/-2 as missing/not collected | V00–V11 | 0.083%; all four missing values were blank, not sentinel codes |
| `height_av` | `oai_oarisk01.txt` | Average height | Participant/visit | Millimetres | Selected visits | 1.751% |
| `weight_met` | `oai_oarisk01.txt` | Weight, metric | Participant/visit | Float; API notes 999/-999 as missing/no data | Selected visits | 0.083%; no documented sentinel observed |
| `ksurg` | `oai_oarisk01.txt` | Either-knee history of surgery, including arthroscopy/ligament repair/meniscectomy | Participant | 0 no; 1 yes | Baseline-focused | 0.104% |
| `ksurgl` | `oai_oarisk01.txt` | Left knee ever had surgery or arthroscopy | Left knee | 0 no; 1 yes | Baseline-focused | 0.188% |
| `ksurgr` | `oai_oarisk01.txt` | Right knee ever had surgery or arthroscopy | Right knee | 0 no; 1 yes | Baseline-focused | 0.167% |
| `famkr` | `oai_oarisk01.txt` | Blood relative had knee replacement for arthritis | Participant | Documented yes/no item | Baseline-focused | 1.314% |
| `injl` | `oai_oarisk01.txt` | Left knee ever injured enough to limit walking for at least two days | Left knee | 0 no; 1 yes | Defined but unpopulated | 100.000% — unusable in this export |
| `injr` | `oai_oarisk01.txt` | Right knee ever injured enough to limit walking for at least two days | Right knee | 0 no; 1 yes | Defined but unpopulated | 100.000% — unusable in this export |
| `injl1` | `oai_oarisk01.txt` | Age at first left-knee injury | Left knee | Age; conditional | Baseline-focused | 74.896%; not a direct injury indicator |
| `injr1` | `oai_oarisk01.txt` | Age at first right-knee injury | Right knee | Age; conditional | Baseline-focused | 71.726%; not a direct injury indicator |

Overall missingness for baseline-focused history variables is misleading because they are not
repeated at every visit; visit-specific missingness must be used. A scientifically defensible knee
injury-history indicator remains unresolved because the direct fields are empty.

## WOMAC, KOOS, and pain candidates

| Variable | Source | Documented meaning | Level | Coding/units | Visits | V00 missing |
|---|---|---|---|---|---|---:|
| `womac_pain_left` | `oai_koos_womac01.txt` | Calculated left-knee WOMAC pain | Left knee | 0–20; higher worse | V00–V10 | 0.000% |
| `womac_pain_right` | `oai_koos_womac01.txt` | Calculated right-knee WOMAC pain | Right knee | 0–20; higher worse | V00–V10 | 0.063% |
| `womac_stiffness_left` | same | Calculated left-knee WOMAC stiffness | Left knee | 0–8; higher worse | V00–V10 | 0.104% |
| `womac_stiffness_right` | same | Calculated right-knee WOMAC stiffness | Right knee | 0–8; higher worse | V00–V10 | 0.021% |
| `womac_disability_left` | same | Calculated left-knee WOMAC disability | Left knee | 0–68; higher worse | V00–V10 | 0.480% |
| `womac_disability_right` | same | Calculated right-knee WOMAC disability | Right knee | 0–68; higher worse | V00–V10 | 0.354% |
| `womac_total_left` | same | Calculated left-knee WOMAC total | Left knee | 0–96; higher worse | V00–V10 | 0.584% |
| `womac_total_right` | same | Calculated right-knee WOMAC total | Right knee | 0–96; higher worse | V00–V10 | 0.438% |
| `koos_lkpain` | same | Left-knee KOOS pain score | Left knee | 0–100; lower worse | V00–V10 | 0.042% |
| `koos_rkpain` | same | Right-knee KOOS pain score | Right knee | 0–100; lower worse | V00–V10 | 0.063% |
| `koos_lksymptoms` | same | Left-knee KOOS symptoms score | Left knee | 0–100; lower worse | V00–V10 | 0.000% |
| `koos_rksymptoms` | same | Right-knee KOOS symptoms score | Right knee | 0–100; lower worse | V00–V10 | 0.000% |
| `koos_qol` | same | KOOS quality-of-life score | Participant/non-lateral in dictionary | 0–100; lower worse | V00–V10 | 0.021% |
| `koos_sports` | same | KOOS sports/recreation function score | Participant/non-lateral in dictionary | 0–100; lower worse | V00–V10 | 25.313% |
| `kpnl12` | `oai_oapain01.txt` | Left knee pain/aching/stiffness in past 12 months | Left knee | 0 no; 1 yes | V00–V11 | 0.125% |
| `kpnr12` | `oai_oapain01.txt` | Right knee pain/aching/stiffness in past 12 months | Right knee | 0 no; 1 yes | V00–V11 | 0.125% |
| `p7lkacv` | `oai_oapain01.txt` | Left-knee 0–10 pain rating, past 7 days | Left knee | 0–10 | Defined but unpopulated | 100.000% — unusable in this export |
| `p7rkacv` | `oai_oapain01.txt` | Right-knee 0–10 pain rating, past 7 days | Right knee | 0–10 | Defined but unpopulated | 100.000% — unusable in this export |

## Objective physical-function candidates

| Variable | Source | Documented meaning | Level | Coding/units | Visits | V00 missing |
|---|---|---|---|---|---|---:|
| `w20mpace` | `oai_physfunct01.txt` | 20-metre walk pace | Participant/visit | m/s | V00, V01, V03, V05, V06, V08, V10 | 0.438% |
| `wlk20t1` | same | Trial-1 completion result | Participant/visit | 1 completed; 2 not attempted/unable; 3 attempted/incomplete | same | 0.313% |
| `w400mcmp` | same | 400-metre walk completion status | Participant/visit | Codes 1–5 documented | same table, not every visit administered | 2.043% |
| `w400mtim` | same | 400-metre total/stop time | Participant/visit | Seconds | same table, not every visit administered | 4.817% |
| `cstime1` | same | Repeated chair-stand trial-1 time | Participant/visit | Seconds and hundredths | same | 5.150% |
| `cspace` | same | Repeated chair-stand pace | Participant/visit | Stands/second | same | 5.046% |
| `lemaxf` | same | Maximum left-knee extension force | Left knee/visit | Newtons | selected function visits | 12.156% |
| `remaxf` | same | Maximum right-knee extension force | Right knee/visit | Newtons | selected function visits | 12.052% |
| `lfmaxf` | same | Maximum left-knee flexion force | Left knee/visit | Newtons | selected function visits | 12.302% |
| `rfmaxf` | same | Maximum right-knee flexion force | Right knee/visit | Newtons | selected function visits | 12.198% |
| `aacnt` | `oai_accelsummary01.txt` | Average daily physical-activity counts | Participant/visit | Counts | V06 and V08 only | No V00 records; 0% blank among 3,321 rows |
| `anvdays` | same | Number of valid accelerometry days | Participant/visit | Days | V06 and V08 only | No V00 records; 0% blank among 3,321 rows |

The walk/chair measures are participant-level performance tests; isometric force is side-specific.
The linked accelerometry table is genuine OAI data but cannot provide a baseline predictor because
it has no V00 records.

## Radiographic and imaging candidates

| Variable | Source | Documented meaning | Level | Coding/units | Visits | V00 missing |
|---|---|---|---|---|---|---:|
| `xrkl` | `oai_kxrsemiquant01.txt` | Kellgren–Lawrence grade | Knee/project/visit | 0–4 | V00, V01, V03, V05, V06, V08, V10 | 0.007% of 13,348 V00 records |
| `xrjsl` | same | OARSI lateral-compartment joint-space narrowing | Knee/project/visit | 0–3 | same | 0.007% of V00 records |
| `xrjsm` | same | OARSI medial-compartment joint-space narrowing | Knee/project/visit | 0–3 | same | 0.000% of V00 records |
| `mcmjsw` | `oai_kxrquantjsw01.txt` | Medial minimum joint-space width | Knee/project/visit | Millimetres | V00, V01, V03, V05, V06, V08, V10 | 0.956% of 6,801 V00 records |
| `mjswbb` | same | Reader judged medial compartment bone-on-bone | Knee/project/visit | 0 no; 1 yes | same | 5.852% of V00 records |
| `accept` | `oai_xrmeta01.txt` | X-ray QC rating | Image | NA/N/NR/Y/YD/P documented | X-ray visits | 0.000% |
| `examtype` | same | X-ray series type | Image | Documented series names | X-ray visits | 0.000% |
| `ffknee` | `oai_inventory01.txt` | Bilateral fixed-flexion X-ray available | Participant/visit | 0 no; 1 yes | Inventory visits | 0.000% at V00 |
| `image_file` | `image03.txt` | Data-file path | Image | File path | V00–V10 selected visits | 0.000%; paths are not copied here |
| `scan_type` | same | Scan type | Image | X-Ray or documented MR sequence category | V00–V10 selected visits | 0.000% |
| `image_modality` | same | Imaging modality | Image | MRI or X-Ray observed | V00–V10 selected visits | 0.000% |

The best-supported radiographic outcome candidates are `xrkl`, `xrjsm`, and `xrjsl`, with
`mcmjsw` as a quantitative alternative. This is schema discovery only: no progression definition,
project pooling rule, horizon, or primary outcome has been selected.

## Missing-value convention

Blank strings are the dominant observed representation of unavailable values in this NDA export.
The NDA API also defines field-specific sentinels (for example BMI -9/-5/-2 and weight 999/-999),
but none of those sentinels occurred in the downloaded BMI/weight fields. Codes such as `NA` can
be valid categories in other fields (`accept`) and must not be globally converted to missing.

Missingness code must therefore be variable-specific and evidence-backed. The project must not
apply a global list of numeric or text sentinels.

## Unresolved dictionary issues

- Several exact file headers are absent from the current NDA API definition (for example
  `kpnl30cv` and `kpnr30cv`); they remain undocumented here rather than being inferred.
- Direct left/right knee injury indicator fields exist in the structure but are completely empty.
- `V99` outcome fields are derived summaries and must not be adopted as a progression endpoint
  without separate scientific review.
- The relationship among projects 37 and 42 requires explicit handling under the official imaging
  guidance; no recoding was applied during inventory.
