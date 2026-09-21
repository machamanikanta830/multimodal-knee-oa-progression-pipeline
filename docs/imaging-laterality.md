# Frozen OAI Bilateral-Radiograph Laterality V2

## Approval status

The human researcher approved this policy for freezing in Milestone 5G. Its scope is the validated
OAI bilateral PA fixed-flexion acquisition set/protocol, not radiology images in general. After the
acquisition passes bilateral-geometry and exception checks, `screen_left` maps to anatomical right
and `screen_right` maps to anatomical left. A unilateral DICOM laterality value on these documented
bilateral images cannot override the rule.

The independent audit found 128/128 concordant mappings; combined confidently interpretable
evidence across prior stages is 270/270, with no observed reversal and coverage of every available
manufacturer, scanner-model, spacing, release, and reportable dimension family. The policy retains
exceptions for an unexpected acquisition type, a non-bilateral image, gross orientation anomaly,
corruption/cropping, or future image evidence contradicting the rule. An exception with unresolved
evidence remains unlabeled.

## Evidence policy

The V2 hierarchy prioritizes:

1. reliable burned-in `R`/`L` image markers;
2. anatomical evidence, especially fibular/lateral-compartment orientation;
3. an empirically validated OAI screen-position rule with acquisition-level exception checks; and
4. DICOM laterality metadata as supporting evidence only.

A unilateral DICOM `Laterality` or `ImageLaterality` value on a confirmed bilateral acquisition is
recorded as a metadata artifact. It cannot establish side and cannot override stronger image
evidence. Anatomical side is withheld whenever primary image evidence disagrees, bilateral geometry
is not confirmed, or the exception detector does not pass.

## Independent audit design

An outcome-blind 128-acquisition sample was selected from the independent V2 holdout. It covers all
6 manufacturer categories, all 6 scanner-model categories, all 9 spacing families, all 4 image
releases, all 3 OAI acceptance categories, and all 6 dimension families large enough for aggregate
reporting. No OA outcome was used.

Manual review used image anatomy as the ground truth and used visible burned-in markers as
supporting evidence. All 128 acquisitions were confidently mapped as screen-left → anatomical
right and screen-right → anatomical left; no reverse mapping was observed. This is consistent with
all 29 confidently interpretable development-pilot images and all 113 confidently interpretable
images from the earlier independent validation, for 270 confident observations across the three
review waves and zero observed reversals.

## Scalable rule evaluation

The automatic V2 rule uses the audited screen mapping only after confirming bilateral midpoint
structure and passing explicit exception checks. On the independent manual audit it resolved
128/128 acquisitions (100% coverage), made 0 observed incorrect assignments, and achieved 100%
audited accuracy. Accuracy takes precedence over coverage; a failed exception check still returns
`AMBIGUOUS`, and genuinely disagreeing image evidence returns `CONFLICTING`.

Twenty of the 128 audited acquisitions carried the earlier unilateral-DICOM-tag condition. In every
case image anatomy supported the validated screen mapping. Across the unchanged full V1 manifest,
all 446 previously `CONFLICTING` acquisitions have exactly the reason code for a unilateral DICOM
tag on a bilateral image. Thus the prior 446 conflicts are explained by the conservative V1
metadata policy, not by 446 observed image-side reversals.

## Preservation and decision boundary

The laterality hierarchy is approved and frozen separately from localization. It has not been
applied to or materialized across all 3,621 acquisitions. The preserved V1 records therefore
remain unchanged at 0 `CONFIDENT`, 3,175 `AMBIGUOUS`, and 446 `CONFLICTING`, and none is newly linked
to participant-knee analysis rows in this milestone. Full automatic resolution and exception
counts must be measured during a future approved, versioned reprocessing run.
