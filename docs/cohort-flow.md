# Milestone 3 Aggregate Cohort Flow

## Scope

This report was generated from the read-only package under `data/raw/oai/` using the approved
V00→V06, `READPRJ`-15 participant-knee logic. It contains aggregate counts only: no participant
identifiers, barcodes, dates, or row examples. No participant-level derived cohort was written.

The replacement branch is not a sequential exclusion: it re-adds qualifying severe events that
lack V06 radiographic outcomes to the composite population.

## Flow

| Stage | Knees | Participants | Change/exclusion |
|---|---:|---:|---:|
| Project-15 V00 source records | 8,921 | 4,490 | starting population |
| Identifier/linkage-consistent participant-knees | 8,921 | 4,490 | 0 excluded |
| Usable baseline KL | 8,921 | 4,490 | 0 excluded |
| Baseline KL below 4 | 8,627 | 4,488 | 294 KL-4 knees excluded |
| Baseline-primary eligible after baseline-replacement check | 8,627 | 4,488 | 0 excluded |
| Usable project-15 V06 KL | 6,851 | 3,583 | 1,776 lack usable V06 KL |
| Qualifying pre-V06 replacement branch | 110 | 100 | branch from baseline-eligible knees |
| **Radiographic-analysis eligible** | **6,851** | **3,583** | requires V06 KL |
| **Composite-analysis eligible** | **6,961** | **3,621** | radiographic set + 110 replacement cases |

Of the 1,776 baseline-eligible knees without usable V06 KL, 110 are retained as qualifying
replacement events for the composite endpoint and 1,666 have neither V06 KL nor a qualifying
replacement.

## Endpoint summaries

| Endpoint | Knees | Participants | One-knee participants | Two-knee participants | Events | Non-events | Event rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Radiographic KL progression | 6,851 | 3,583 | 315 | 3,268 | 961 | 5,890 | 14.027% |
| Composite progression | 6,961 | 3,621 | 281 | 3,340 | 1,071 | 5,890 | 15.386% |
| Secondary JSN progression | 6,851 | 3,583 | 315 | 3,268 | 734 | 6,117 | 10.714% |

### Baseline KL distributions

| Population | KL 0 | KL 1 | KL 2 | KL 3 | KL 4 |
|---|---:|---:|---:|---:|---:|
| Radiographic eligible | 2,747 | 1,326 | 1,891 | 887 | 0 |
| Radiographic events | 226 | 262 | 282 | 191 | 0 |
| Radiographic non-events | 2,521 | 1,064 | 1,609 | 696 | 0 |
| Composite eligible | 2,751 | 1,331 | 1,933 | 946 | 0 |
| Composite events | 230 | 267 | 324 | 250 | 0 |
| Composite non-events | 2,521 | 1,064 | 1,609 | 696 | 0 |

## Replacement accounting

- 914 project-15 baseline knees have a documented post-baseline replacement date.
- 184 replacements occur by the implemented V06 boundary before applying baseline eligibility.
- 74 of those 184 knees have baseline KL 4 and are excluded by the approved ceiling rule.
- 110 baseline-eligible replacements occur by the boundary; all 110 lack usable V06 KL and add
  severe events to the composite population.
- Among baseline-eligible knees, 641 documented replacements occur after the V06 boundary.
- The composite has 110 replacement-only events and no observed overlap between qualifying
  replacement and radiographic KL progression.

The exact-date rule differs from the earlier Milestone 2 scheduled-day feasibility approximation.
Four knees that were inside the approximate 1,461-day enrollment cutoff had an actual V06
radiograph before replacement and are correctly classified as post-V06 by the final logic.

## Baseline domain availability

Counts below describe outcome-eligible knees and do not define complete-case cohorts.

| Availability mask | Radiographic, n/N (%) | Composite, n/N (%) |
|---|---:|---:|
| Imaging barcode linked to metadata and image index | 6,851/6,851 (100.000%) | 6,961/6,961 (100.000%) |
| PRO core complete | 6,824/6,851 (99.606%) | 6,932/6,961 (99.583%) |
| Clinical core complete | 6,847/6,851 (99.942%) | 6,957/6,961 (99.943%) |
| Physical-function core complete | 5,862/6,851 (85.564%) | 5,949/6,961 (85.462%) |
| All four domain masks complete | 5,846/6,851 (85.331%) | 5,933/6,961 (85.232%) |

The physical-function mask is the limiting core domain, but 100% of knees in both endpoint
populations have at least one objective function measure. Complete availability is not required
and no imputation has been performed.

## Aggregate integrity checks

- No project-15 V00 row has a missing identifier, inconsistent identifier mapping, or invalid
  knee side.
- No duplicate project-15 participant + visit + knee row was found.
- Three project-15 V06 knees have no matching project-15 V00 knee; they cannot enter a
  baseline-defined cohort.
- All 8,921 project-15 V00 knee records link by barcode to X-ray metadata and the image index.
- No project-15 baseline knee is flagged as already replaced on the baseline X-ray.
- All replacement dates used for these baseline knees are after baseline and have resolvable
  timing.

## Interpretation boundary

These are candidate analysis populations produced by fixed rules, not a fitted model or an
estimate of predictive performance. No feature selection, missing-data treatment, image download,
participant split, or model training occurred.
