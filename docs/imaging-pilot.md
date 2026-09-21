# Milestone 5 Imaging Pilot

## Scope and safety boundary

The outcome-blind pilot contains 32 baseline bilateral PA fixed-flexion acquisitions from 32
participants and represents 48 eligible knees. It was selected from baseline KL and image/QC
metadata only; no future outcome was read. All archives, extracted DICOMs, per-acquisition review
records, and preview pixels remain in Git-ignored local storage. No cohort, split, model, or
full-cohort download was created or changed.

The official [OAI protocol](https://nda.nih.gov/static/docs/StudyDesignProtocolAndAppendices.pdf)
states that the right and left knees are imaged together for the fixed-flexion acquisition. It does
not, by itself, establish a safe screen-left/screen-right labeling rule.

## Download and archive verification

- All 32 expected `.tar.gz` archives were present; no partial file remained.
- All 32 logical archive paths matched the requested pilot references and all 32 byte sizes matched
  the authorized package metadata.
- There were no duplicate paths or duplicate archive-content hashes.
- Raw archive size was 304,561,053 bytes (290.45 MiB) logically and 305,995,776 bytes allocated.
- Every archive had the same safe structure: one benign current-directory marker plus one regular
  file. There were no links, special files, traversal paths, duplicate member targets, or extraction
  failures.
- Extraction produced 32 separate acquisition directories and 32 regular files totaling
  539,790,814 bytes (514.78 MiB). Each file was a valid DICOM; there were no non-DICOM files.

The source archives were not modified. Extraction rejects absolute paths, parent traversal, links,
special files, duplicate targets, and existing output destinations.

## DICOM and pixel structure

All 32 DICOMs are unsigned, single-sample, single-frame 16-bit images with 16 stored bits. All use
Explicit VR Little Endian transfer syntax and none is compressed. Twenty-three record modality
`CR` and nine record `RG`. The body part is `KNEE` throughout; `ViewPosition` is `PA` in 22 and
missing in 10.

There are 15 dimension pairs. The most common are 2,320 × 2,828 pixels (9), 2,048 × 2,494 (8),
and 2,828 × 2,320 (3); the other 12 acquisitions each have a different pair. Rows span 1,712–3,537
and columns span 2,140–4,637.

Pixel decoding succeeded for all 32 images. The aggregate stored intensity range is 0–4,095. All
are `MONOCHROME2`, so no photometric inversion is required for a consistent light-bone display.
Rescale slope/intercept are explicitly 1/0 in 29 and absent in 3; no non-identity rescale was
observed. Window center/width are present in 12 and absent in 20, so a robust explicit display or
normalization policy is needed rather than reliance on those optional tags. Two DICOMs have a
non-standard character-set spelling that `pydicom` recovers with a warning; this did not affect
technical tags or pixel decoding.

## Laterality and orientation evidence

The DICOM laterality fields are insufficient on their own:

- `ImageLaterality` is missing in all 32.
- `Laterality` is missing in 30, bilateral in 1, and unilateral-left in 1 despite visible bilateral
  anatomy.
- `PatientOrientation` is `L\\F` in 12 and missing in 20.
- `ImageOrientationPatient` is missing in all 32.
- `BurnedInAnnotation` is missing in all 32 even though markers/annotations are visibly burned into
  many images; this tag must not be used as proof that annotations are absent.

Visual review found burned-in side-marker evidence consistent with screen-left = anatomical right
and screen-right = anatomical left in 30 acquisitions. The evidence classification is 29
`CONFIDENT`, 2 `AMBIGUOUS` because no usable per-image side marker or orientation tag was visible,
and 1 `CONFLICTING` because bilateral burned-in markers conflict with a unilateral DICOM laterality
tag. No image contradicted the marker-supported screen-position convention, but that empirical
pattern must not silently resolve the three exception cases.

## Midpoint and knee-localization feasibility

All 32 acquisitions contain one bilateral image. A horizontal midpoint placed one complete knee in
each half in all 32; no knee crossed the midpoint, no major half-image crop/asymmetry was observed,
and no 90°/180° rotation was present. Borders, black masking, collimation, calibration objects,
marker placement, exposure appearance, and vertical joint position nevertheless vary.

The three candidate strategies therefore have different roles:

1. A fixed midpoint alone separates broad panels in this pilot but cannot label anatomical side.
2. Midpoint separation plus validated laterality mapping is the preferred broad-panel strategy,
   with explicit rejection/manual review for missing or conflicting evidence.
3. A deterministic paired projection-profile locator was subsequently prototyped in Milestone 5C
   because field of view and joint position vary. It uses the shared bilateral detector row and a
   panel-specific horizontal center; no learned detector was built.

One acquisition visibly contains an arthroplasty in one panel. This does not change the pilot, but
future knee-level image materialization must apply the existing eligibility linkage before retaining
a panel and must not assume that both panels from every bilateral acquisition are eligible.

## Physical scale and scanner variation

All DICOMs provide square pixel spacing, but values range from 0.100 to 0.200 mm/pixel (median
0.150; mean 0.143). The distribution is: 0.100 (2), 0.101 (5), 0.102 (2), 0.150 (12), 0.157696
(2), 0.170 (8), and 0.200 (1). Estimated physical field of view spans 269.98–424.20 mm vertically
and 348.00–468.34 mm horizontally.

Five recorded manufacturer groups are represented: AGFA (8), Agfa-Gevaert AG (12), Fuji (1),
LS100 (9), and Swissray (2), with platform-associated differences in matrix and spacing. Directly
resizing every half-image to one pixel matrix would therefore encode different physical scales.
Milestone 5C compared 0.15, 0.16, and 0.17 mm/pixel and 140, 160, and 180 mm square crops. The
pilot-supported proposal is 0.15 mm/pixel followed by a 160 × 160 mm crop, with padding rather than
crossing the midpoint. Both values remain unapproved pending human review.

## Metadata QC flags versus visible usability

All 32 images were visually usable for pilot pixel and midpoint review: all 25 `YD`, all 6 `Y`, and
the single `NR` image. The four alignment-flagged, one centering-flagged, one incomplete-depiction-
flagged, and three positioning-flagged acquisitions also retained both visible joint regions and
clean midpoint separation.

These counts do not establish clinical-grade acceptability. The `NR` and single-category examples
are too sparse for exclusion inference, and the available dictionary still does not define all code
semantics adequately. No QC category is converted into an exclusion rule.

## Storage observations and projections

Pilot size statistics are:

| Representation | Minimum | Median | Mean | Maximum | Pilot total |
|---|---:|---:|---:|---:|---:|
| Archive | 2,936,246 B | 7,291,545 B | 9,517,533 B | 20,125,746 B | 304,561,053 B |
| Extracted DICOM | 7,534,746 B | 13,123,704 B | 16,868,463 B | 32,275,066 B | 539,790,814 B |

Mean-based projections for 3,621 bilateral acquisitions are approximate:

| Representation/assumption | Approximate storage |
|---|---:|
| Downloaded archives | 34.46 GB (32.10 GiB) |
| Extracted DICOMs | 61.08 GB (56.88 GiB) |
| Two broad midpoint panels, uncompressed source 16-bit pixels | 61.07 GB (56.87 GiB) |
| Two 512 × 512 panels, uint8 | 1.90 GB (1.77 GiB) |
| Two 512 × 512 panels, uint16 | 3.80 GB (3.54 GiB) |
| Two 512 × 512 panels, float32 | 7.59 GB (7.07 GiB) |
| Two 1,067 × 1,067 standardized crops, uint16 | 16.49 GB (15.36 GiB) |

The estimates exclude filesystem/container overhead and compression. The standardized-crop row is
a pilot projection for the proposed 0.15 mm/pixel, 160 mm configuration and is not an approved
full-cohort allocation.

## Recommendation and review gates

Milestone 5C completed that non-model prototype. All 64 paired-heuristic centers and 160 mm crops
passed visual pilot review, with no center override; 21 crops use horizontal padding and none uses
vertical padding. Do not download or preprocess the full cohort until the three laterality
exceptions, physical configuration, normalization, localization acceptance thresholds, and QC
policy are approved on a larger validation sample.

No full-cohort download, final preprocessing pipeline, image model, or data split was created.
