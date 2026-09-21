# Milestone 5H — full-cohort frozen V3 imaging

## Scope and frozen specification

Milestone 5H applied the independently approved Localization V3 candidate and Laterality V2
policy to all 3,621 baseline bilateral acquisitions. The run retained the frozen 0.15 mm/pixel
spacing, bilinear interpolation, 160 × 160 mm physical crop, 1,067 × 1,067 output, and uint16
scientific representation. It did not materialize the proposed percentile/min-max model
normalization, change the cohort, create a modeling split, or train a model.

Laterality uses the OAI-specific validated mapping only after bilateral-structure exception checks:
screen-left maps to anatomical right and screen-right maps to anatomical left. Unilateral DICOM
laterality metadata is recorded as a metadata artifact and cannot override that mapping.

## Interrupted-run recovery

The interrupted implementation was audited before processing resumed. The repository contains no
Git commits, so local preservation ledgers—not Git history—were the available integrity baseline.
The audit found 996 complete acquisition bundles, six partial acquisitions, zero unverified
bundles, and 2,619 acquisitions with no V3 full-processing output. The six partial cases consisted
of temporary extraction residue; two also had orphaned review previews.

The resumed implementation verifies every existing bundle before skipping it. A complete bundle
requires two readable uint16 arrays of the frozen shape, a compatible provenance/QC record, a
valid automatic state, a Laterality V2-consistent mapping, complete review assets when required,
and no temporary extraction residue. Only the exact incomplete acquisition is reclaimed and
reprocessed. Crop bundles, JSON records, Parquet manifests, checkpoints, preview PNGs, and review
adjudications use atomic publication or append-only durable writes.

All 996 complete bundles were verified and skipped. The six interrupted acquisitions and the
2,619 untouched acquisitions were processed normally. No acquisition was counted complete from a
filename or directory count alone.

## Full-cohort result

| Measure | Aggregate result |
|---|---:|
| Expected acquisitions | 3,621 |
| Complete acquisitions | 3,621 |
| Resumed and verified without reprocessing | 996 |
| Processed in the recovery run | 2,625 |
| Knee-panel crops | 7,242 |
| Hard archive/DICOM/midpoint/resample/localization/crop failures | 0 |
| Acquisition PASS | 2,645 (73.046%) |
| Acquisition BORDERLINE | 932 (25.739%) |
| Acquisition FAIL | 44 (1.215%) |
| Manual-review queue | 978 (27.009%) |

The full-cohort automatic-state distribution falls within the prespecified 99% compatibility
intervals from the representative V3 holdout. That comparison is a distributional monitoring
check, not a substitute for review of BORDERLINE and FAIL cases.

At panel level, the final automatic state was 5,911 PASS, 1,278 BORDERLINE, and 53 FAIL. The
conceptually separate anatomical validator returned 6,707 PASS, 506 BORDERLINE, and 29 FAIL.
These are automatic safety-gate states; no claim of manual anatomical adequacy is made for the
unreviewed full cohort.

## Laterality and participant-knee linkage

Laterality was CONFIDENT for 3,618 acquisitions, AMBIGUOUS for three, and CONFLICTING for zero.
This yields 7,236 anatomically resolved panel crops and six unresolved panels. The unilateral
DICOM-tag artifact occurred in 446 acquisitions and overrode the OAI-specific mapping in zero.
No unresolved anatomical side was forced.

Of the 7,242 panel crops, 6,955 resolve to eligible analysis-knee rows, 281 are resolved
contralateral panels not in the analysis cohort, and six remain unlinked until the three
laterality exceptions are adjudicated. The imaging set is therefore materialized and linkage-ready
in structure, but it is not approved for analysis use until the review queue and laterality
exceptions are adjudicated.

## Padding and automatic exception queues

Horizontal padding occurred in 1,313 panels (18.130%): median 7.65 mm among padded panels, mean
7.636 mm, 99th percentile 16.95 mm across all panels, and maximum 43.05 mm. Vertical padding
occurred in 23 panels (0.318%): median 7.35 mm among padded panels, mean 10.180 mm, and maximum
23.40 mm. In total, 1,334 panels used padding on at least one axis.

One panel exceeded the prior independent-validation padding envelope. The crop-size and spacing
specification were not changed. The local exception tables contain 976 non-PASS acquisitions,
535 panels where the anatomical validator was not PASS, one excessive-padding panel, and three
laterality ambiguities. All hard-processing exception tables are empty.

## Acquisition heterogeneity

No manufacturer or scanner-model group met the prespecified systematic-failure alert. Review load
nevertheless varies materially and should guide queue ordering and reviewer workload planning.

| Aggregate group | Acquisitions | PASS / BORDERLINE / FAIL | Review required |
|---|---:|---:|---:|
| AGFA / ADC_51xx | 898 | 838 / 55 / 5 | 62 |
| Agfa-Gevaert / ADC_5146 | 1,116 | 681 / 415 / 20 | 435 |
| LS100 / Lumisys family | 679 | 360 / 310 / 9 | 319 |
| Swissray | 533 | 488 / 38 / 7 | 45 |
| Fuji | 111 | 83 / 27 / 1 | 28 |
| Manufacturer missing | 284 | 195 / 87 / 2 | 89 |

The highest review burdens occur in the 0.101, 0.102, and 0.15 mm/pixel groups and in the
2,320 × 2,828 family. The difficult 0.17 mm/pixel, 2,048 × 2,494 AGFA/ADC_51xx family has a much
lower review rate under V3 than under the unsafe V2 candidate. The `YD` acceptance group requires
review more often than `Y`; the small `NR` group has one automatic FAIL among 12 acquisitions.
These patterns are monitoring findings, not exclusion rules.

Small dimension families can show high percentages from one event. They must not be interpreted
as systematic failures without additional review. No scanner-specific coordinate rule was added.

## Storage and cleanup

| Artifact class | Actual size |
|---|---:|
| Source archives | 35.303 GB |
| Extracted DICOM bytes processed cumulatively | 59.245 GB |
| Standardized uint16 crop arrays | 16.491 GB |
| Review overlays and previews | 1.063 GB |
| Full V3 local output, including records and manifests | 17.584 GB |

All 3,621 successful temporary extractions were removed after verification. No extraction files or
pending crop bundles remain. Approximately 24.1 GB of filesystem free space remained at the final
check. No redundant full-cohort uint8 or float32 cache was created.

## Review interface and stopping point

The local review queue is deterministically ordered and resumable. It shows anonymous bilateral
overviews, proposed crops, localization overlays, automatic reason codes, and a visible explanation
of the 35 mm preview-only detector-margin masking used to hide burned-in text. It supports ACCEPT,
REJECT, and NEEDS_CENTER_OVERRIDE. A corrected center is stored as a separate adjudication and
does not overwrite the automatic result. Review records are tied to a queue fingerprint. The
reviewer-facing workflow, decision semantics, and launch command are documented in
[imaging-adjudication.md](imaging-adjudication.md).

No case was manually adjudicated in Milestone 5H. The review queue contains 978 acquisitions.
Full-cohort V3 processing is complete, but analysis-cohort linkage and model use remain blocked
until human review establishes how accepted crops and center overrides are materialized and the
three laterality exceptions are resolved.
