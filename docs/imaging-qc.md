# Imaging QC Policy and Validation Evidence

## Current status

This document records the pilot, independent-validation, and full-baseline QC evidence. It defines
no clinical image exclusion rule. Real DICOMs, arrays, identity-bearing manifests, exception
records, and previews remain local and Git-ignored; tracked outputs contain aggregate technical
facts only.

## Archive and extraction controls

- Match every downloaded archive to the authorized reference and package byte size before use.
- Require unique archive paths and content hashes and reject partial files.
- Extract each archive separately into a new destination.
- Reject absolute paths, parent traversal, links, special files, duplicate member targets, and any
  existing extraction destination.
- Re-hash the immutable archive set after analysis.

The pilot passed all controls: 32/32 expected archives, 32/32 size matches, no duplicates or
partials, and 32/32 successful extractions.

## Privacy-safe DICOM inspection

Only allow-listed technical fields are serialized: geometry, pixel encoding, physical spacing,
transfer syntax, frame count, rescale/window values, view/orientation/laterality fields, and
manufacturer/model. Identity fields, accession values, dates, source/member paths, private tags,
and pixel arrays are excluded from reports. Synthetic tests enforce this boundary.

## Display handling

Pixel decoding must require a two-dimensional single-frame array. Apply documented rescale slope
and intercept when present, use explicit identity defaults when absent, and invert only
`MONOCHROME1`. Pilot previews use robust percentile normalization for visual QC; this is not the
approved quantitative model normalization. All 32 pilot images are `MONOCHROME2` and need no
photometric inversion.

## Laterality evidence hierarchy

Do not equate pixel half with anatomical side. Review evidence in this order:

1. internally consistent bilateral DICOM orientation/laterality fields;
2. visible burned-in right/left markers;
3. a documented acquisition/display convention;
4. screen position only after the preceding evidence has validated the convention.

The official [OAI protocol](https://nda.nih.gov/static/docs/StudyDesignProtocolAndAppendices.pdf)
supports that both knees were acquired together but does not alone prove the stored display-side
mapping. In this pilot, visible markers consistently support screen-left = anatomical right and
screen-right = anatomical left where markers are present.

Classifications are:

- `CONFIDENT`: per-image marker/orientation evidence supports the mapping with no contradiction;
- `AMBIGUOUS`: per-image evidence is absent or insufficient;
- `CONFLICTING`: technical and visible evidence disagree.

Pilot result: 29 confident, 2 ambiguous, and 1 conflicting. The ambiguous/conflicting images stay in
an explicit review queue; no silent fallback is approved.

## Panel and geometry checks

The broad midpoint split is pilot-supported for geometric separation: 32/32 clean, 0 crossing the
midpoint, 0 with major half-image crop/asymmetry, and 0 with major rotation. It is not sufficient
for anatomical labeling. The Milestone 5C prototype:

- use midpoint only to form provisional broad panels;
- retains the `screen_left`/`screen_right` names and attaches side only after evidence validation;
- localizes one shared joint row from paired joint-space projection profiles and a separate
  horizontal center in each panel;
- resamples with bilinear interpolation at a candidate 0.15 mm/pixel before a candidate 160 mm
  square crop; and
- fails on unusable contrast/geometry rather than substituting a guessed center.

Visual overlay review classified 64/64 pilot panel localizations as successful, with 0 borderline,
0 failed, and 0 manual center overrides. All crops retain both compartments. Twenty-one use
horizontal padding (maximum 29.70 mm) to avoid crossing into the other midpoint half; none uses
vertical padding. These pilot counts do not constitute validated automatic acceptance thresholds.

## Physical scale

Pixel spacing varies twofold (0.100–0.200 mm/pixel), while horizontal physical field of view varies
from 348.00 to 468.34 mm. Matrix-only resizing would not preserve physical scale. At 0.15 mm/pixel,
11/32 acquisitions require upsampling, 9/32 require downsampling, and 12/32 remain unchanged. The
0.16 and 0.17 alternatives downsample 23/32. The 0.15 mm/pixel and 160 mm crop recommendations
remain human review items.

## Intensity prototype

Keep resampled scientific crops as uint16. For a later model input, the conservative proposal is
0.5th–99.5th percentile clipping followed by float32 min-max scaling to [0, 1]. Do not overwrite
the source, use the uint8 previews as scientific data, or apply CLAHE by default. Clipped z-score
normalization is available for comparison but is not the recommended default.

## Existing OAI QC metadata

Pilot visual usability was 25/25 for `YD`, 6/6 for `Y`, and 1/1 for `NR`. Every flagged challenge
case retained visible joint regions and clean midpoint separation: alignment 4/4, centering 1/1,
incomplete depiction 1/1, and positioning 3/3. This small, deliberately enriched pilot cannot show
that these codes are clinically interchangeable, and one or two examples cannot support an
exclusion. Preserve all flags for stratified QC and seek authoritative code semantics before a
policy decision.

## Human decisions still required

- Approve how ambiguous/conflicting laterality evidence is adjudicated.
- Approve whether the empirical marker-supported screen mapping may be used as a fallback and under
  what QC evidence threshold.
- Decide how the full-set shaft-localization cluster should be handled without tuning on the
  reviewed stress sample: versioned localizer revision, automatic confidence gate, manual review,
  or a combination.
- Define a new independent validation protocol for any revised locator and crop-acceptance rule.
- Decide how to identify and handle arthroplasty or other ineligible contralateral panels after
  applying knee-level cohort linkage.
- Define technical versus clinical usability and the role of OAI acceptance/problem flags.
- Decide whether provisional crops unaffected by a later algorithm revision can be retained or the
  complete cache must be regenerated under one version.

## Milestone 5H full V3 QC gate

All 3,621 acquisitions completed the frozen V3 technical pipeline, producing 7,242 conformant
uint16 panel crops with zero hard extraction, DICOM, midpoint, resampling, localization-execution,
or crop-write failures. Acquisition-level automatic states are 2,645 PASS, 932 BORDERLINE, and 44
FAIL. Every non-PASS acquisition and every laterality exception is present in the local review
queue; the union contains 978 acquisitions.

Laterality is confident for 3,618 acquisitions, ambiguous for three, and conflicting for zero.
The frozen OAI mapping resolves 7,236 panel sides and leaves six unresolved without forcing a
label. The 446 unilateral DICOM-tag artifacts never override the validated screen-position rule.

Horizontal padding occurs in 1,313 panels and vertical padding in 23; one panel exceeds the prior
validation envelope. No manufacturer or scanner-model group triggers the prespecified systematic
failure alert, although review burden is higher in the Agfa-Gevaert/ADC_5146 and LS100/Lumisys
families. These automatic-state patterns prioritize review and do not establish exclusion rules.

The anonymous local review interface is ready and its queue contains no adjudications. Until the
978 cases are reviewed and the three laterality exceptions are resolved, the full cache must not
be treated as analysis-ready solely because every output has the expected shape and type.

## Milestone 5D independent-validation result

Before inspecting an independent image, the project fixed these acceptance criteria: 100% archive
retrieval, zero unsafe members/corrupt DICOMs, at least 99% clean midpoint separation, at least 98%
successful joint localization, at least 98% adequate crops, no forced uncertain laterality, and no
systematic failure pattern in a sufficiently populated acquisition subgroup. Padding is acceptable
when anatomy remains complete.

The frozen method was applied without tuning to an independent 128-acquisition V00 stress sample
with zero development-pilot overlap. All 128 archives passed safe extraction and DICOM decoding.
Midpoint separation was clean in 128/128 acquisitions; localization and crop adequacy were each
successful in 256/256 panels. No crop lost a tibiofemoral compartment or adequate femoral/tibial
context. Padding occurred in 68 panels and was accepted only after anatomy was confirmed complete.

Laterality review yielded 113 confident, 2 ambiguous, and 13 conflicting acquisitions. All
confident acquisitions supported screen-left = anatomical right; no anatomical side was assigned
to the 15 exceptions. The conflicts cluster around a unilateral DICOM laterality value on bilateral
images, so DICOM laterality remains non-authoritative alone. No reportable scanner, spacing,
dimension, release, or QC subgroup showed a midpoint, localization, or crop failure pattern.

All predefined acceptance criteria passed. This validates the candidate geometry for human review;
it does not create an image exclusion rule or authorize silent laterality assignment. The local
exception queue and aggregate subgroup monitoring remain required for full-cohort operation.

## Milestone 5E full-baseline QC result

The source and incremental-processing controls succeeded for all 3,621 acquisitions: package-size
agreement, unique archive hashes, safe extraction, single readable DICOM structure, pixel decode,
two-panel output verification, temporary cleanup, and post-run archive hash agreement. Midpoint
splitting produced two technical panels for every acquisition.

Automated checks returned a center and valid 1,067 × 1,067 uint16 array for all 7,242 panels, but
these checks do not prove that the center is the tibiofemoral joint. Anonymous visual review of 64
acquisition pairs—deliberately spanning heterogeneity and enriched for maximum padding—found:

| Visual classification | Acquisitions | Stress-sample rate |
|---|---:|---:|
| Successful localization / adequate crop | 53 | 82.812% |
| Borderline | 2 | 3.125% |
| Failed localization / inadequate crop | 9 | 14.062% |

The nine failures localized non-joint shaft anatomy. Four occurred among the five reviewed
acquisitions beyond the Milestone 5D padding envelope. Seven occurred in the reportable
AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 subgroup; two occurred in the related
Agfa-Gevaert/ADC_5146 family. Failures appeared in both `Y` and `YD` OAI acceptance categories, so
neither category provides a sufficient automatic exclusion or acceptance rule. Because the sample
is stress-oriented, these counts demonstrate a real failure mode but do not estimate its
full-cohort prevalence.

This is materially worse than Milestone 5D and activates the stop-before-retuning requirement.
The frozen parameters and outputs were left unchanged, visual failures/borderlines were added to
local exception queues, and the full image cache is not approved for analysis-cohort linkage.

Full-set laterality classification is also intentionally unresolved: 0 acquisitions met the
confident evidence threshold, 3,175 are ambiguous, and 446 conflicting. DICOM evidence was not
promoted to anatomical side without an automated or human burned-in-marker assessment. No
uncertain side was forced, and all 3,621 acquisitions remain in the laterality review workflow.

## Milestone 5F independent V2 QC

The outcome-blind development and sealed-holdout design used 192 and 256 acquisitions respectively,
with zero overlap. V2 development crop adequacy was 189/192 (98.438%); all 3 visual failures were
automatically detected. The independent representative holdout yielded 126 `PASS`, 2 `BORDERLINE`,
0 `FAIL`, and 128/128 adequate crops. The stress holdout yielded 115 `PASS`, 0 `BORDERLINE`, 13
`FAIL`, and 115/128 adequate crops. Eleven of the 13 stress failures were incorrectly labeled
automatic `PASS`, so the candidate fails the safety gate even though the representative component
meets its numerical threshold.

Eleven holdout failures were in the reportable AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 family,
whose adequacy was 36/47 (76.596%). The observed V2 failure is principally an off-body horizontal
centroid rather than the V1 shaft-row mechanism. Do not use OAI acceptance category alone as an
exclusion rule and do not enlarge the crop to mask the localization error.

The independent laterality audit contained 128 acquisitions and covered all manufacturer, model,
spacing, release, acceptance, and reportable dimension groups. All 128 supported screen-left →
anatomical right, with no reverse mapping. The gated screen rule achieved 128/128 automatic
resolution and no observed incorrect assignments. Twenty audited images carried a unilateral DICOM
tag; image anatomy remained concordant with the screen rule. DICOM laterality therefore remains
supporting evidence only.

No full V2 processing is approved. Future QC must add a horizontal panel-layout/coverage gate, then
evaluate a newly versioned candidate on another untouched holdout before regenerating the full
cache.

## Milestone 5G V3 anatomical-safety QC

V3 adds three independently recorded decisions: panel-layout QC, multi-candidate localization, and
an anatomical plausibility validator. Final `PASS` is possible only when all three agree. The
validator checks superior/inferior bone coverage, tibiofemoral-band evidence, horizontal support,
central anatomy, blank fraction, and collimation dominance. Competing candidates with insufficient
score separation are queued rather than forced.

All 477 previously reviewed development acquisitions were examined, including all known V1/V2
failures and borderlines. V3 produced 477/477 adequate crops and 0 false passes. Its automatic
development distribution was 324 `PASS`, 146 `BORDERLINE`, and 7 `FAIL`.

Parameters were frozen before selecting or inspecting a new, disjoint 384-acquisition holdout.
Anonymous review found:

| Component | Adequate | Automatic PASS / BORDERLINE / FAIL | False PASS |
|---|---:|---:|---:|
| Representative (n=192) | 192 (100.000%) | 136 / 55 / 1 | 0 (0.000%) |
| Stress (n=192) | 191 (99.479%) | 154 / 36 / 2 | 0 (0.000%) |
| Combined (n=384) | 383 (99.740%) | 290 / 91 / 3 | 0 (0.000%) |

The one inadequate stress crop selected lower tibial anatomy and was automatically
`BORDERLINE`. It therefore demonstrates a residual localization failure but not an unsafe pass.
The reportable AGFA/ADC_51xx family had 148/149 adequate crops and 0 false passes, compared with
36/47 adequate and 11 unsafe passes under V2. No reportable acquisition subgroup contained an
unsafe accepted crop.

V3 meets the predefined independent criteria and is safe to freeze for human approval. Its
representative automatic coverage is conservative: approximately 29.2% are queued, projecting
about 1,056 manual-review acquisitions out of 3,621. Full reprocessing and manual adjudication are
separate approval steps and were not performed.
