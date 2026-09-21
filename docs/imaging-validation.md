# Independent Imaging Preprocessing Validation — Milestone 5D

## Predefined protocol and acceptance criteria

These criteria were recorded before inspecting or preprocessing any independent-validation image.
The validation sample must exclude all 32 development-pilot acquisitions. Selection may use only
baseline acquisition metadata and baseline KL 0–3 for imaging heterogeneity; it must not read
progression outcomes, future labels, or post-baseline events.

The frozen candidate pipeline is: bilateral midpoint broad-panel split; retain `screen_left` and
`screen_right` until laterality evidence is confident; use the marker-validated screen-left →
anatomical right and screen-right → anatomical left mapping only in confident cases; bilinear
resampling to 0.15 mm/pixel; the frozen paired tibiofemoral joint localizer; a 160 × 160 mm crop;
uint16 intermediate storage; and future 0.5th–99.5th percentile clipping followed by min-max
scaling to [0, 1]. CLAHE is not used.

Acceptance criteria are fixed for the first validation report:

1. Archive/DICOM integrity: 100% of expected archives retrieved, zero unsafe extraction members,
   and zero corrupt DICOMs.
2. Midpoint separation: at least 99% of bilateral acquisitions separate cleanly into two complete
   broad knee panels.
3. Joint localization: at least 98% of broad panels have successful automatic localization;
   borderline and failed cases are counted separately.
4. Crop adequacy: at least 98% of materialized crops retain both tibiofemoral compartments and
   adequate femoral/tibial context. Padding is not a failure when anatomy is preserved.
5. Laterality: never assign anatomical side when evidence is ambiguous or conflicting. Report
   confidence states without imposing a numerical threshold while the evidence mechanism remains
   under review.
6. Scanner heterogeneity: no sufficiently populated manufacturer/scanner, spacing, dimension,
   release, or QC subgroup may show a systematic preprocessing failure pattern.

If a criterion is missed, the frozen method will not be modified during this evaluation. Failures
will be reported first for human review. Parameter changes, if recommended later, require a new
version and a separate validation rather than retroactive tuning on this sample.

## Privacy and stopping rule

Acquisition-level manifests, GUIDs, laterality reviews, exception queues, pixels, and paths remain
local and Git-ignored. Tracked outputs contain aggregate counts only and no identifiers,
accessions, dates, raw paths, signed URLs, or S3 references.

If the selected associated archives are not already present locally, Milestone 5D stops after
creating the selection manifest, exact `image03` subset, and GUID query file. Human NDA package
creation must use the OAI collection, the Image / `image03` structure, all 128 GUIDs, and
“Include associated data files” enabled. No browser automation or image download is authorized.

## Validation outcome

The frozen pipeline met every predefined Milestone 5D acceptance criterion on the independent
sample. No setting was tuned before or during evaluation. The result supports the technical
feasibility of full-cohort preprocessing, subject to human approval of storage and the laterality
exception workflow.

## Independent sample selection

The selector first removes the 32 development-pilot acquisitions. It then allocates 16
acquisitions to each combination of acquisition-level maximum baseline KL 0–3 and one/two eligible
knees. Within those fixed stress-test quotas, a deterministic round-robin greedy procedure
maximizes weighted novelty and inverse-frequency coverage across recorded manufacturer, scanner
model/software, dimension pair, resolution pair, image release, acceptance category, and problem-
flag signature. A fixed seed breaks ties. This is an imaging heterogeneity sample, not a prevalence
estimate.

Validation checks passed for sample preparation:

- 128 unique V00 acquisitions from 128 participants, representing 192 eligible knees;
- 0 overlap with the 32-acquisition development pilot;
- 128 unique associated-file references and 128 unique NDA GUIDs;
- 128/128 documented as bilateral PA fixed-flexion knee X-rays;
- 32 acquisitions in each maximum-baseline-KL stratum and 64 each with one/two eligible knees;
- no progression or future outcome was read; the local manifest's audit flag is false throughout;
  and
- the copied `image03` header and embedded dictionary row are byte-identical to the raw source,
  followed by exactly the 128 selected records.

## Baseline acquisition heterogeneity

The sample covers all six observed manufacturer categories (including missing), all six scanner-
model categories (including missing), all six recorded software categories, all nine recorded
resolution families, all four image releases, all three acceptance categories, and ten problem-
flag signatures. It contains 117 distinct recorded dimension pairs.

Recorded first/second extents span 1,732–3,560 and 1,760–4,822 pixels, respectively. Recorded
resolution is square in this sample and spans 0.100–0.200 mm/pixel with median 0.101. Reportable
manufacturer groups contain 98, 16, and 5 acquisitions; nine acquisitions from smaller groups are
pooled. Reportable scanner-model groups contain 69, 35, 12, and 5 acquisitions; seven from smaller
groups are pooled. The two large image-release groups contain 83 and 41 acquisitions, with four
from smaller releases pooled. Acceptance categories contain 72 and 55 acquisitions, with one from
a smaller category pooled. All four available problem-flag types are represented; small strata are
not separately disclosed.

## Archive, DICOM, and pixel findings

All 128 expected archives were present and matched their authorized references and recorded byte
sizes. Safe extraction accepted 128 archives, rejected zero unsafe members, and had zero failures.
Each archive yielded one valid DICOM, for 128/128 valid and zero corrupt DICOMs.

All images are single-frame, unsigned, uncompressed `MONOCHROME2` DICOMs using Explicit VR Little
Endian. All allocate 16 bits; 125 store 16 bits and 3 store 12 bits. Pixel spacing spans
0.100–0.200 mm/pixel. Image matrices span 1,732–3,560 rows and 1,760–4,822 columns. The validation
set covers six manufacturer categories, six scanner-model categories, nine spacing families, 117
dimension families, four image releases, three acceptance categories, and ten problem-flag
signatures, pooling small groups in stratified reports.

## Geometry, localization, and crop review

Anonymous contact-sheet review covered all 128 acquisitions and 256 broad panels:

| Review outcome | Count | Rate |
|---|---:|---:|
| Clean midpoint separation | 128/128 acquisitions | 100% |
| Successful joint localization | 256/256 panels | 100% |
| Borderline localization | 0/256 panels | 0% |
| Failed localization | 0/256 panels | 0% |
| Adequate 160 mm crop | 256/256 panels | 100% |
| Borderline/inadequate crop | 0/256 panels | 0% |

No crop had a boundary issue, incomplete medial or lateral compartment, or insufficient femoral or
tibial context. Padding was used in 68/256 panels: 63 horizontally and 6 vertically, with one panel
using both. Maximum one-sided padding was 34.95 mm horizontally and 24.60 mm vertically. Padding
was not classified as failure because the relevant anatomy remained complete.

## Laterality evidence

Laterality review classified 113/128 acquisitions as confident, 2 as ambiguous, and 13 as
conflicting. All 113 confident images supported screen-left = anatomical right and screen-right =
anatomical left; none supported the reverse. No anatomical side was assigned to the 15 unresolved
acquisitions, which remain in a local ignored exception queue.

The conflicting group is concentrated in one scanner-manufacturer family: a unilateral DICOM
`Laterality` value appears on a documented bilateral image. This is a metadata-evidence conflict,
not a midpoint, localization, or crop failure. It confirms that the DICOM laterality tag cannot be
used alone and that exception handling remains mandatory.

## Heterogeneity analysis

Midpoint separation, localization, and crop adequacy were each 100% in every reportable and pooled
manufacturer, scanner-model, pixel-spacing, image-dimension, image-release, acceptance-code, and
QC-signature stratum. No acquisition subgroup showed a systematic preprocessing failure pattern.
Laterality conflicts clustered as described above, but the pipeline responded conservatively by
withholding anatomical labels.

## Empirical storage update

The 128 archives occupy 1.96 GB (1.82 GiB), with median 16.74 MB, mean 15.30 MB, and maximum 22.51
MB. Extracted DICOMs occupy 3.33 GB (3.10 GiB), with median 29.94 MB, mean 26.02 MB, and maximum
33.53 MB. The 256 uncompressed uint16 1,067 × 1,067 crops occupy 582.94 MB.

Extrapolation to 3,621 bilateral acquisitions gives approximately 55.39 GB (51.59 GiB) for raw
archives, 94.24 GB (87.76 GiB) for extracted DICOMs, 16.49 GB (15.36 GiB) for uint16 1,067-pixel
crops, 8.24 GB (7.68 GiB) for equivalent uint8 crops, or 1.90 GB (1.77 GiB) for a 512 × 512 uint8
cache. These estimates exclude filesystem/container overhead. The archive and DICOM estimates are
higher than the original 32-image pilot projections; the fixed-size crop estimate is unchanged.

## Acceptance decision and stopping point

| Criterion | Target | Independent result | Pass |
|---|---:|---:|---:|
| Expected archives retrieved | 100% | 128/128 | Yes |
| Unsafe members | 0 | 0 | Yes |
| Corrupt DICOMs | 0 | 0 | Yes |
| Clean midpoint separation | ≥99% | 100% | Yes |
| Automatic joint localization | ≥98% | 100% | Yes |
| Adequate 160 mm crops | ≥98% | 100% | Yes |
| No forced uncertain laterality | Required | 0 forced assignments | Yes |
| No systematic scanner failure | Required | None observed | Yes |

The researcher subsequently approved Milestone 5D and froze the geometric preprocessing settings
for full-baseline processing. The laterality exception queue remains mandatory and unresolved
images cannot be joined to side-specific cohort rows. Full image acquisition is still gated by NDA
package availability and separate approval before the large download. No full-cohort download,
model training, cohort change, or modeling split was performed during validation.
