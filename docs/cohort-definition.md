# Cohort Definition

## Status

Milestone 4 reuses the human-approved v1 participant-knee cohort logic and materializes the
validated master cohort locally under Git-ignored `data/processed/`. Tracked reports remain
aggregate-only. It does not download images, impute values, split participants, or train a model.

## Fixed design

- Unit: participant-knee. Both eligible knees may be retained.
- Participant linkage key: `subjectkey`; knee side: `side` 1 right and 2 left.
- Baseline: V00.
- Primary follow-up: V06, approximately 48 months.
- Radiographic read project: `READPRJ` 15 only at both visits. No other project is pooled,
  transformed, or substituted.
- Any future train/validation/test partition must group on participant, keeping both knees from a
  participant in the same partition. No partition exists yet.

## Source and linkage rules

The starting population is the unique set of project-15 V00 semi-quantitative knee records in
`oai_kxrsemiquant01.txt`. A participant-knee is the pair `subjectkey + side`. Project-15 records
must be unique on `subjectkey + visit + side + readprj`; ambiguity raises an error rather than
selecting a row silently. The `subjectkey` ↔ `src_subject_id` relationship must be one-to-one, and
side must be documented as 1 or 2.

V06 radiographic values are joined on the same participant-knee under unchanged project 15.
X-ray acquisition dates are obtained through the semi-quantitative `barcode` linked to
`oai_xrmeta01.barcode`. Image-index availability is established by linking the same barcode to
`image03.accession_number`; paths and identifiers are not predictors.

## Sequential eligibility rules

1. Start with project-15 V00 participant-knee records.
2. Require usable participant identifiers, documented side 1/2, consistent identifier mapping,
   and an unambiguous project-specific key.
3. Require a numeric baseline `xrkl` in the documented 0–4 range.
4. Exclude baseline KL grade 4 because the approved primary radiographic outcome cannot increase
   beyond grade 4.
5. As a safety check, exclude a knee flagged in V99 as already replaced on the baseline OAI
   X-ray. No project-15 V00 knee was removed by this check in the downloaded release.
6. Radiographic-analysis eligibility additionally requires a usable project-15 V06 KL grade.
7. Composite-analysis eligibility requires either a usable project-15 V06 KL grade or a
   documented post-baseline knee replacement on or before the V06 boundary.

A replacement case is therefore retained for the composite endpoint even when its V06 KL grade
is missing. A knee with neither a usable V06 KL grade nor a qualifying replacement is not assigned
a composite label. Missing baseline features do not affect outcome eligibility at this milestone.

## Replacement timing

Knee-specific V99 `lkdate`/`rkdate` supplies the event date. The rule requires the replacement
date to be later than the linked baseline X-ray acquisition date. The upper boundary is:

- the actual linked V06 X-ray acquisition date when a V06 image exists; or
- the baseline X-ray acquisition date plus exactly 48 calendar months when no V06 image exists.

The upper comparison is inclusive. An actual V06 acquisition date takes precedence over a
nominal enrollment-day cutoff. Replacement type and confirmation fields are retained as
availability flags for later sensitivity/adjudication work, but a documented knee-specific date
is the implemented v1 event evidence. Post-baseline replacement information is outcome-only and
must never be used as an input feature.

## Baseline availability masks

Availability is reported, not used for complete-case selection:

- **Imaging linkage:** nonmissing baseline project-15 barcode linked to both X-ray metadata and
  the downloaded image index. This establishes linkage only; pixels have not been downloaded or
  quality-controlled.
- **PRO core complete:** side-matched WOMAC pain, stiffness, and disability plus KOOS pain and
  symptoms are all available. An any-PRO flag and component flags are also retained.
- **Clinical v1 complete:** age, sex, BMI, side-specific prior knee surgery, and family
  knee-replacement history are all available. This frozen v1 mask is intentionally stricter than
  the earlier Milestone 3 exploratory clinical core of age, sex, and BMI.
- **Physical-function core complete:** 20-m walk pace, chair-stand time, side-matched extension
  strength, and side-matched flexion strength are all available. The 400-m completion status/time
  and any-function flags are also quantified.

No imputation, feature-value transformation, or all-domain completeness requirement is applied.

## Implementation

Reusable outcome/eligibility code is in `src/cohort/builder.py`; local materialization is in
`src/cohort/materialize.py`. Both command-line interfaces serialize only aggregate reports:

```bash
PYTHONPATH=src python -m cohort.builder data/raw/oai
PYTHONPATH=src python -m cohort.materialize data/raw/oai --output-dir data/processed
```

The participant key is necessary for bilateral linkage and future grouped splitting, but must not
be logged, used as a predictor, or written to tracked outputs. The materialized participant-level
Parquet table is ignored by Git and remains local access-controlled data.

## Still outside scope

- No index-knee selection, exclusion based on baseline feature completeness, or image-QC rule.
- No replacement adjudication beyond documented knee-specific date and timing.
- No decision about statistical handling of two correlated knees.
- No imputation method, model, partition, or evaluation procedure.
