# Full Baseline Imaging Acquisition — Milestone 5E

## Scope and stopping decision

The full baseline set contains 3,621 approved V00 bilateral PA fixed-flexion acquisitions linked
to 6,961 participant-knee analysis rows. The independently validated Milestone 5D geometry was
applied without tuning: midpoint panel separation, 0.15 mm/pixel bilinear resampling, the frozen
paired localizer, a 160 × 160 mm crop, and 1,067 × 1,067 uint16 output. No normalization cache,
cohort change, split, or model was created.

Technical materialization completed, but representative visual QC found systematic anatomical
localization failures. Under the prespecified stopping rule, the image set is **not ready for
analysis-cohort linkage**. All crops are retained as provisional local artifacts, affected records
are queued for review, and no preprocessing parameter was changed.

## Download and source integrity

All 3,621 expected archives were present, matched their authorized package-record byte sizes, and
had unique references and content hashes. There were zero partial, missing, duplicate, unrelated,
or size-mismatched archives. The archive set occupies 35,303,240,852 bytes (32.88 GiB); archive
sizes range from 1,909,429 to 22,969,529 bytes, with median 7,095,688 and mean 9,749,583 bytes.

Preprocessing used a SHA-256 ledger captured before extraction. A complete post-processing
rehash matched all 3,621 entries, confirming that no source archive changed.

## Incremental extraction and DICOM validation

Each archive was processed in a separate temporary workspace. Extraction rejected unsafe member
types and path traversal, required the expected single-file acquisition structure, decoded the
DICOM and its pixel array, generated and verified both crop outputs, and only then removed the
temporary DICOM. All 3,621 archives passed; all 3,621 DICOMs were readable; and no unsafe member,
archive failure, DICOM failure, or persistent extracted file remained.

The processed DICOM volume was 59,244,822,952 bytes (55.18 GiB). This is cumulative input volume,
not persistent duplicate storage.

## Frozen preprocessing output

The midpoint operation completed for 3,621/3,621 acquisitions and generated 7,242 provisional
screen-panel crops. Automated shape/dtype checks passed for every crop: 1,067 × 1,067, uint16,
with the frozen physical geometry. The crop cache occupies 16,490,801,652 bytes (15.36 GiB).

Padding occurred in 2,412 panels horizontally and 48 vertically. Maximum padding was 40.20 mm
horizontally and 30.15 mm vertically. Twenty-five panels exceeded the maximum padding observed in
Milestone 5D and were queued for review. Padding is a geometric record, not by itself a clinical
exclusion.

## Representative visual QC and failure gate

Anonymous staged previews were generated for 64 acquisitions selected to span manufacturer,
scanner model, spacing, dimensions, image release, OAI QC category, laterality state, and the
largest observed padding. This is a stress-oriented diagnostic sample enriched for padding; its
rates are not estimates of full-cohort failure prevalence.

Although the localizer returned a numerical center for every panel, visual review classified the
bilateral acquisition pairs as follows:

| Visual result | Acquisition pairs | Rate in reviewed stress sample |
|---|---:|---:|
| Successful localization and adequate crop | 53 | 82.812% |
| Borderline localization/crop | 2 | 3.125% |
| Failed localization and inadequate crop | 9 | 14.062% |

The clear failures place the crop on non-joint femoral or tibial anatomy, so output-shape checks
alone are not an adequate localization QC criterion. Four of five reviewed acquisitions outside
the Milestone 5D padding envelope failed. Seven failures occurred among 18 reviewed acquisitions
in the AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 family; all nine failures in this review were
from the larger baseline image release represented in the sample. Two additional failures occurred
in the related Agfa-Gevaert/ADC_5146 family. Both `Y` and `YD` OAI acceptance categories contained
failures, so the existing category alone does not isolate them.

This materially underperforms the independent validation result and triggers the frozen-pipeline
stop. The existing crops must not be treated as analysis-ready, and the localizer must not be
changed until the failure pattern and a new validation plan receive human approval.

## Laterality

No frozen automated burned-in-marker reader exists. DICOM laterality/orientation evidence alone
is insufficient under the approved policy, so no screen panel was silently mapped to anatomical
side. Automatic evidence classification yielded 0 confident, 3,175 ambiguous, and 446 conflicting
acquisitions. Thus zero participant-knee crops currently have resolved anatomical laterality and
all 3,621 acquisitions require laterality review before side-specific linkage.

The local exception queues preserve ambiguity and conflict separately. The validated screen-left
to anatomical-right convention may only be applied after per-acquisition evidence is confident.

## Storage and retained artifacts

The retained uint16 crops use 16.49 GB. Local manifests, exception queues, anonymous staged
previews, and contact sheets use approximately 78.12 MB. No redundant full uint8 or float32 cache
was created. For reference only, equivalent uncompressed uint8 crops would require about 8.24 GB,
and a 512 × 512 uint8 cache about 1.90 GB.

All identity-bearing manifests, exception records, crop arrays, and real-image previews remain in
Git-ignored local storage. Tracked documentation contains aggregate results only.

## Human review required

- Decide whether the shaft-localization cluster warrants a versioned localizer revision, a
  prespecified automatic confidence gate, manual adjudication, or a combination.
- Define a fresh validation protocol before accepting any revised geometry; do not tune against
  the reviewed full-set sample and then report it as independent validation.
- Approve a scalable burned-in-marker/laterality adjudication workflow. Until then, no crop can be
  joined to an anatomical participant-knee row.
- Decide whether all provisional crops should be regenerated under a later approved version or
  whether strictly documented unaffected cases may be retained.

No modeling or analysis-cohort linkage may proceed from these provisional images.
