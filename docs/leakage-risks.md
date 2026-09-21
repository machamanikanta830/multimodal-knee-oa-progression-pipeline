# Leakage Risks and Predictor Classification

## Governing boundary

For a V00 prediction task, a predictor is eligible only if it existed at or before the reviewed
baseline index time. A field can be temporally available and still require review because it
encodes study sampling, the outcome construct, site, treatment, or identity. These classifications
are provisional and apply to the proposed use, not to the intrinsic quality of an OAI variable.

## Classification

| Candidate or field group | Classification | Reason/control |
|---|---|---|
| V00 radiograph pixels, once downloaded and QC-approved | **SAFE BASELINE CANDIDATE** | Baseline image domain; acquisition window and QC must be fixed before use |
| V00 side-matched WOMAC pain/stiffness/disability and KOOS pain/symptoms | **SAFE BASELINE CANDIDATE** | Baseline, knee-specific PROs; no future data |
| V00 age, sex, BMI, and documented family history | **SAFE BASELINE CANDIDATE** | Baseline participant factors; demographic use still needs responsible subgroup interpretation |
| V00 20-m pace, chair stand, 400-m measures, and side-matched strength | **SAFE BASELINE CANDIDATE** | Baseline objective function; completion/missingness rules must be prespecified |
| V00 `xrkl` | **REQUIRES REVIEW** | Legitimate baseline severity but directly related to a K-L-change target and has ceiling effects |
| V00 `xrjsm`/`xrjsl` | **REQUIRES REVIEW** | Baseline structural severity; do not include silently when JSN change is the target |
| V00 `race`, `ethnicity` | **REQUIRES REVIEW** | Potentially relevant to inequity/transportability; not a causal biological risk proxy |
| V00 `e_cohort` | **REQUIRES REVIEW** | Encodes OAI sampling/risk enrichment and may create a shortcut |
| V00 `site` and acquisition/QC metadata | **REQUIRES REVIEW** | Possible center/technical shortcut; primarily appropriate for QC or design adjustment |
| V00 `ksurgl`/`ksurgr` and `ksurg` | **REQUIRES REVIEW** | Pre-baseline history is temporally safe, but broad procedures and confounding need definition |
| `koos_qol`, `koos_sports` | **REQUIRES REVIEW** | Non-lateral in the dictionary; replicating one value to both knees needs approval; sports is highly missing |
| `subjectkey`, `src_subject_id`, row IDs | **EXCLUDE / LEAKAGE RISK** | Identity/linkage keys only; can memorize participants and must never be model inputs |
| `barcode`, `accession_number`, `image_file`, filenames/paths | **EXCLUDE / LEAKAGE RISK** as predictors | Linkage/loading only; may encode participant, visit, or acquisition source |
| `readprj` | **EXCLUDE / LEAKAGE RISK** as a biological predictor | Required to define valid readings; also encodes selected assessment samples |
| Any V01–V11 PRO, clinical, function, surgery, or imaging measure | **EXCLUDE / LEAKAGE RISK** for V00 prediction | Collected after baseline; may reflect disease course or treatment |
| Follow-up `xrkl`, `xrjsm`, `xrjsl`, `mcmjsw` | **EXCLUDE / LEAKAGE RISK** | Direct outcome components |
| All V99 `oai_outcome01` derived summaries | **EXCLUDE / LEAKAGE RISK** | Encode future incidence, JSN progression, replacement, death, and follow-up availability |
| `lkdays`/`rkdays`, replacement type/confirmation, `ksrgl12`/`ksrgr12`, `krsl12`/`krsr12` | **EXCLUDE / LEAKAGE RISK** | Post-baseline surgery/replacement and timing |
| Accelerometry at V06/V08 | **EXCLUDE / LEAKAGE RISK** for V00 task | Genuine OAI-linked data, but future rather than baseline |
| Direct injury fields `injl`/`injr` | **EXCLUDE** | Completely empty in this export |
| Conditional age-at-injury `injl1`/`injr1` as a binary injury proxy | **EXCLUDE / REQUIRES NEW RULE** | Missingness cannot silently be converted into “no injury” |

## Specific leakage pathways

### Future and derived outcomes

The V99 table contains precomputed incident OA, JSN progression, last observed status, replacement,
and death variables. It can be used only for reviewed outcome/censoring validation—not as a
predictor, imputation input, feature-selection signal, or development-data filter that indirectly
reveals future status.

### Baseline/follow-up X-ray duplication

Outcome construction necessarily reads baseline and target X-ray grades. Feature assembly must
receive only the baseline image/approved baseline grades. File loaders should use explicit visit
and project assertions so a target radiograph, its barcode/path, or a later assessment cannot enter
the baseline image stream.

Multiple rows for the same participant + visit + side occur because of `readprj`. The valid key is
participant + visit + side + project. Silently choosing the first row, averaging projects, or using
one project at baseline and another at follow-up would create an invalid target and potentially
duplicate knees across examples.

### Bilateral participant leakage

Project 15/V06 contains 3,430 participants with two K-L-eligible knees. Random knee-level splitting
would place highly related contralateral knees—and identical participant-level predictors—on both
sides of a split. Future splits and cross-validation folds must be grouped by participant. No
split is implemented in Milestone 2.

### Surgery and informative follow-up

Replacement and post-baseline surgery can be consequences of disease severity and can also remove
native-knee radiographs from follow-up. Excluding replaced knees without a reviewed strategy can
condition on a future event; using post-baseline surgery as a feature directly leaks the course.
Replacement must be handled in outcome/censoring logic and sensitivity analyses, never predictor
construction.

### Missingness leakage

Missing indicators can encode postbaseline dropout, replacement, or target availability if they
are created from longitudinal data. Baseline missingness indicators may be considered only from
V00 measurements. Imputation and preprocessing must be learned inside future participant-grouped
development folds; outcome availability cannot determine baseline feature selection or filling.

## Baseline K-L as a special case

Two future formulations should be compared on the same reviewed cohort and participant-grouped
evaluation design:

### Formulation A — no explicit tabular baseline K-L

Inputs: baseline radiograph image + baseline PRO + baseline clinical/demographic + baseline
physical function. Baseline K-L remains necessary for outcome change calculation and ceiling
eligibility, but is not supplied as a tabular input. The image can still contain the morphology
that drives K-L; this formulation tests what the imaging representation learns without an explicit
reader-grade shortcut.

### Formulation B — add explicit baseline K-L

Inputs: everything in A plus side/project-matched V00 `xrkl`. This tests whether a clinically
interpretable reader grade adds information or stabilizes prediction. It also creates a strong
baseline-severity predictor and makes ceiling/grade-stratified evaluation essential.

Comparing A and B can quantify the incremental value of the tabular grade and whether multimodal
gains persist beyond known radiographic severity. The comparison must not change cohort membership,
outcome definition, horizon, or split. Neither formulation has been built or trained.

## Controls required before modeling

1. Freeze the approved baseline visit, target visit, `readprj`, and participant-knee key.
2. Add assertions that predictor tables contain V00 only and outcome tables contain only the
   explicit V00/target pair.
3. Maintain a predictor denylist for identifiers, V99 fields, and all postbaseline measurements.
4. Group every future partition and resampling fold by participant.
5. Record whether baseline K-L/JSN are predictor inputs, outcome-only fields, or exclusions.
6. Resolve replacement/censoring before creating labels or measuring complete-case loss.
7. Inspect image paths/metadata for encoded IDs and visit labels before any image pipeline.
