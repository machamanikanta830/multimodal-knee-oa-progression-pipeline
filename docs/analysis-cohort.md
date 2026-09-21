# Analysis-Ready Cohort v1

## Status and governance

Milestone 4 materializes the human-approved v1 cohort as a local-only Parquet table. The file
contains participant identifiers and other row-level OAI-derived data, so it is access-controlled
research data and is ignored by Git:

`data/processed/cohorts/analysis_cohort_v1.parquet`

This document contains aggregate counts and schema descriptions only. No row, participant ID,
barcode, accession, date, or file path from the OAI package is reproduced here. Raw OAI files were
read but not modified. No imputation, complete-case filtering, image download, data partition, or
model training was performed.

## Reproduction

From the repository root in the configured environment:

```bash
PYTHONPATH=src python -m cohort.materialize data/raw/oai --output-dir data/processed
```

The command reads `configs/study_v1.yaml` and `configs/features_v1.yaml`, validates the fixed
study rules, stops if the reviewed population counts change, writes the local cohort and image
manifest atomically, and prints aggregate JSON only.

## Population roles

The one-row-per-participant-knee composite population is the master table. The radiographic
population is a nested sensitivity subset identified by a Boolean mask; it is not a second file.

| Population | Knees | Participants | One-knee participants | Two-knee participants | Events | Non-events | Event rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Composite primary | 6,961 | 3,621 | 281 | 3,340 | 1,071 | 5,890 | 15.386% |
| Radiographic KL sensitivity | 6,851 | 3,583 | 315 | 3,268 | 961 | 5,890 | 14.027% |

The 110 additional composite knees are qualifying replacement-only events without usable V06 KL.
No qualifying replacement overlaps an observed KL-progression event in v1. These counts exactly
match the reviewed Milestone 3 references; a mismatch now raises an error before output is
accepted.

## Stored information

The local table has 43 columns in the following functional groups:

- linkage and design: participant identifiers, knee side, V00/V06 labels, `READPRJ` 15, and the
  baseline X-ray barcode;
- baseline structure: KL plus medial and lateral JSN values;
- primary baseline predictors: the 14 tabular variables frozen in `features_v1.yaml`;
- local descriptive/design fields: race, ethnicity, site, and OAI cohort, none marked as default
  predictors;
- outcome and eligibility fields: composite, radiographic KL, JSN, replacement, and analysis-mask
  columns;
- domain-availability fields and the explicit marker that baseline KL is not a default predictor.

The file does not store raw V06 grades, replacement dates, the V06 boundary date, or other
post-baseline measurements as predictor columns. Outcome labels and eligibility flags are retained
only because the file is an analysis table. Identifiers, barcodes, and design fields are linkage/QC
data and must never be supplied to a model.

## Per-variable missingness

Missingness was calculated after side matching and documented sentinel handling. It did not
change eligibility.

| Frozen baseline feature | Composite missing, n/N (%) | Radiographic missing, n/N (%) |
|---|---:|---:|
| WOMAC pain | 0/6,961 (0.000%) | 0/6,851 (0.000%) |
| WOMAC stiffness | 5/6,961 (0.072%) | 5/6,851 (0.073%) |
| WOMAC disability | 23/6,961 (0.330%) | 21/6,851 (0.307%) |
| KOOS pain | 2/6,961 (0.029%) | 1/6,851 (0.015%) |
| KOOS symptoms | 0/6,961 (0.000%) | 0/6,851 (0.000%) |
| Age | 0/6,961 (0.000%) | 0/6,851 (0.000%) |
| Sex | 0/6,961 (0.000%) | 0/6,851 (0.000%) |
| BMI | 4/6,961 (0.057%) | 4/6,851 (0.058%) |
| Prior knee surgery | 13/6,961 (0.187%) | 13/6,851 (0.190%) |
| Family knee-replacement history | 99/6,961 (1.422%) | 99/6,851 (1.445%) |
| 20-m walk pace | 21/6,961 (0.302%) | 19/6,851 (0.277%) |
| Chair-stand time | 316/6,961 (4.540%) | 301/6,851 (4.394%) |
| Knee-extension strength | 723/6,961 (10.386%) | 711/6,851 (10.378%) |
| Knee-flexion strength | 728/6,961 (10.458%) | 716/6,851 (10.451%) |

## Domain completeness

The v1 clinical mask requires all five frozen clinical features. This is intentionally stricter
than the Milestone 3 exploratory clinical core, which required only age, sex, and BMI.

| Domain complete | Composite knees (%) | Composite participants with at least one complete knee | Radiographic knees (%) | Radiographic participants with at least one complete knee |
|---|---:|---:|---:|---:|
| Imaging linkage | 6,961 (100.000%) | 3,621 | 6,851 (100.000%) | 3,583 |
| PRO | 6,932 (99.583%) | 3,610 | 6,824 (99.606%) | 3,573 |
| Clinical | 6,845 (98.334%) | 3,562 | 6,735 (98.307%) | 3,524 |
| Physical function | 5,949 (85.462%) | 3,102 | 5,862 (85.564%) | 3,073 |

| Missing-domain pattern | Composite knees | Composite participants with at least one knee in pattern | Radiographic knees | Radiographic participants with at least one knee in pattern |
|---|---:|---:|---:|---:|
| All four complete | 5,841 | 3,049 | 5,754 | 3,020 |
| Missing exactly one domain | 1,084 | 577 | 1,063 | 569 |
| Missing multiple domains | 36 | 22 | 34 | 21 |

The 36 composite knees missing multiple domains comprise 35 with two domains available and one
with only imaging available; none has zero domains available. Participant counts are nonexclusive
across row patterns because a bilateral participant's two knees can have different side-specific
availability. No complete-case cohort has been defined.

## Baseline KL formulations

- Formulation A, the default: do not add the reader-assigned baseline KL grade as an explicit
  tabular predictor; the radiograph remains the imaging input.
- Formulation B, prespecified comparison: add baseline KL explicitly as an ordinal/categorical
  predictor after human approval of encoding.

The table stores baseline KL so both formulations remain possible, but
`baseline_kl_default_predictor` is false for every row. No model has been built to compare them.

## Remaining review gates

Human review is still required for replacement adjudication sensitivity, acceptable X-ray QC
flags, image orientation/cropping, missing-data handling, categorical encoding, use of baseline KL
formulation B, dependence-aware evaluation for bilateral knees, and participant-grouped split
design. None is implemented here.
