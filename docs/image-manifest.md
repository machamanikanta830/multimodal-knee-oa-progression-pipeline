# V00 Knee X-ray Manifest

## Purpose and location

The exact local-only manifest for the v1 composite cohort is:

`data/processed/manifests/v00_xray_manifest.parquet`

It contains identifiers, barcodes, accessions, dates, and OAI image paths and is therefore ignored
by Git. This tracked document reports aggregate structure only. No images were downloaded.

The linkage is deterministic:

`READPRJ`-15 V00 participant-knee → `oai_kxrsemiquant01.barcode` →
`oai_xrmeta01.barcode` → `image03.accession_number`.

Source tables are required to be unique at the barcode/accession level for the selected images,
and participant identifiers must agree across all three sources. Any ambiguity or mismatch raises
an error rather than selecting a candidate silently.

## Aggregate coverage and cardinality

- 6,961 manifest rows, exactly one per master-cohort participant-knee.
- 3,621 participants, 3,621 accessions, and 3,621 distinct indexed image files.
- No missing barcode, X-ray-metadata link, accession, or indexed image path.
- 281 accessions link to one eligible knee; 3,340 bilateral accessions link to both eligible knees.
- No accession maps across participants, no participant-knee has multiple candidate image-index
  rows, and no acquisition maps to more than two knees.

The one-accession-to-two-knees pattern is expected here because the recorded series is bilateral;
it must not be treated as a duplicate image error.

## View and image-index findings

All 6,961 knee rows have:

- X-ray metadata side code 3, documented as bilateral;
- exam type and image description `Bilateral PA Fixed Flexion Knee`;
- scan type and modality `X-Ray`;
- scan object `Live`;
- indexed format `DICOM`;
- completion code 1.

The indexed release-study distribution is 4,009 knee rows from OAI Image Release 0.C.1, 2,900
from 0.E.1, 36 from 0.C.2, and 16 from 0.E.2.

For v1, the recommended intended baseline view is therefore the documented bilateral PA
fixed-flexion knee acquisition linked to the exact project-15 baseline barcode. This is a
data-supported recommendation for human approval, not a pixel-level QC rule. The package has
image paths/metadata but not downloaded pixels, so DICOM laterality tags, image orientation,
panel cropping, and left/right extraction cannot yet be verified.

## Recorded QC fields

No QC exclusion was applied. Aggregate metadata values among the 6,961 knee rows are:

- acceptance: 2,059 `Y`, 4,878 `YD`, and 24 `NR`;
- alignment problem: 448 `X`, 6,513 missing;
- centering problem: 146 `X`, 6,815 missing;
- incomplete depiction: 6 `X`, 6,955 missing;
- positioning problem: 128 `X`, 6,833 missing.

The meanings and acceptability of `YD`, `NR`, `X`, and blank status must be approved from OAI
documentation and imaging expertise before an image-QC exclusion rule is created. These fields
remain metadata, not silently interpreted quality labels.

## Manifest schema roles

The 30 local columns cover participant/knee linkage, baseline/read-project constants, barcode,
X-ray and image-index acquisition dates, recorded QC fields, exact image metadata/path,
accession, and bilateral-sharing cardinality. Identifiers and file/linkage metadata must not be
used as predictors. The manifest exists only to support exact future authorized image retrieval
and QC.

## Milestone 5A pilot subset

A deterministic, outcome-blind 32-accession subset is now stored locally at
`data/processed/manifests/v00_xray_pilot_manifest.parquet`, with its associated-file references in
`data/processed/manifests/v00_xray_pilot_associated_files.txt`. Both are Git-ignored and contain
access-controlled row-level information.

The pilot balances acquisition-level maximum baseline KL 0–3 and one- versus two-eligible-knee
representation, then makes minimal within-stratum substitutions for QC-code/problem-flag coverage.
It contains 16 one-knee and 16 two-knee acquisitions and represents 48 eligible knees. Selection
does not read or use future progression outcomes.

The 32 image references are relative `.tar.gz` associated-file paths, but the existing tabular NDA
package metadata contains no associated image archives. No image was downloaded. Retrieval,
DICOM, laterality, orientation, QC, and storage findings are detailed in
[imaging-pilot.md](imaging-pilot.md).
