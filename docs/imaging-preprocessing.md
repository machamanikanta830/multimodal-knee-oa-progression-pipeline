# Imaging Preprocessing Specification and Version Status

## Status and scope

Midpoint broad-panel separation, 0.15 mm/pixel bilinear resampling, a 160 × 160 mm crop, a
1,067 × 1,067 uint16 intermediate, and the proposed future percentile/min-max normalization remain
frozen. The original full-cohort localizer is preserved as `v1_provisional`, and
`v2_candidate_r3` is preserved but rejected. Milestone 5G evaluated the separately versioned
`localization_v3_candidate_r1` with an explicit anatomical safety gate. It met its independent
acceptance criteria and is recommended for human approval before versioned full-cohort
reprocessing. No cohort change, data split, or model is authorized.

## Separation of geometry and anatomical laterality

The source image is split at its horizontal midpoint into `screen_left` and `screen_right` broad
panels without resizing. The arrays remain uint16, and source dimensions and row/column pixel
spacing are recorded. All 64 halves retained exactly one complete knee in the preceding visual
review.

Panel position is not treated as anatomical side. The local laterality assessment carries DICOM
laterality/orientation evidence, burned-in marker evidence, and the visual pilot assessment in a
separate manifest. A screen-to-anatomy mapping is emitted only for 29 `CONFIDENT` acquisitions.
The 2 `AMBIGUOUS` and 1 `CONFLICTING` acquisitions have blank anatomical-side mappings and remain
in a three-acquisition exception queue.

## Deterministic joint localization

The prototype uses the paired bilateral geometry rather than independently selecting an edge in
each half:

1. Robustly scale each panel for localization only; this does not change the stored source panel.
2. Form an 80th-percentile horizontal intensity profile over the central panel width.
3. At four image-height scales, score a dark row lying between brighter superior and inferior
   bands. Weight the score by foreground support and search only the central 25%–86% of image
   height.
4. Robustly standardize and sum the two panel profiles, yielding one shared detector row for the
   bilateral pair.
5. Estimate a separate horizontal center in each panel using a bone-weighted intensity centroid
   near the selected row.

This is a localization heuristic, not a joint-space-width measurement. The method was visually
audited on all 64 panels: 64 successful, 0 borderline, and 0 clear failures. No manual center
override was required. The result includes the arthroplasty panel, where it centers the prosthetic
joint region. The pilot alone did not establish automated acceptance.

Milestone 5D subsequently tested the unchanged method on 256 independent panels: 256 were
successful, none was borderline, and none failed. Unusable contrast or geometry still produces an
explicit failure instead of a guessed center.

## Physical-spacing comparison

Upsampling means the source spacing is coarser than the target; downsampling means it is finer.
Counts are acquisition-level (n=32). Exact equality uses the recorded spacing.

| Target spacing | Upsampling | Downsampling | Unchanged | 140 mm crop | 160 mm crop | 180 mm crop |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 mm/pixel | 11 (34.375%) | 9 (28.125%) | 12 (37.500%) | 933 px | 1,067 px | 1,200 px |
| 0.16 mm/pixel | 9 (28.125%) | 23 (71.875%) | 0 | 875 px | 1,000 px | 1,125 px |
| 0.17 mm/pixel | 1 (3.125%) | 23 (71.875%) | 8 (25.000%) | 824 px | 941 px | 1,059 px |

The pilot recommendation is **0.15 mm/pixel**. It equals the observed median and most common
spacing, leaves 12 acquisitions unchanged, and balances upsampling and downsampling better than
0.16 or 0.17. Bilinear interpolation is used for this continuous-tone radiograph prototype.
Upsampling cannot create new anatomical information and downsampling smooths high-frequency
detail, so neither operation is treated as improving source resolution. This recommendation needs
human approval.

## Physical crop comparison

Candidate square crop extents were evaluated around the localized joint after 0.15 mm/pixel
resampling:

| Physical crop | Output matrix | Panels needing horizontal padding | Maximum padding |
|---:|---:|---:|---:|
| 140 × 140 mm | 933 × 933 | 11/64 | 19.65 mm |
| 160 × 160 mm | 1,067 × 1,067 | 21/64 | 29.70 mm |
| 180 × 180 mm | 1,200 × 1,200 | 36/64 | 39.75 mm |

No vertical padding is needed at 160 mm. All reviewed 160 mm crops retain both tibiofemoral
compartments and substantial distal-femoral/proximal-tibial context. Padding is deliberately used
instead of crossing the midpoint into the other knee. The pilot recommendation is **160 × 160
mm**, while 140 mm is the lower-storage alternative and 180 mm adds padding and surrounding
anatomy without an observed pilot benefit. A musculoskeletal imaging expert should approve the
anatomical context before this choice is frozen.

## Intensity handling

The scientific pilot artifact is the resampled uint16 crop; raw DICOM pixels are never overwritten.
The conservative proposed model-input transform is clipping to the panel/crop 0.5th and 99.5th
percentiles followed by float32 min-max scaling to [0, 1]. Across the 64 broad panels, the robust
range is a median 80.44% of the raw min-to-max range. Strictly below-bound pixels have a median of
0% because many backgrounds tie at the lower bound (mean 0.189%, maximum 0.500%); strictly
above-bound pixels have a median 0.493% (mean 0.416%, maximum 0.500%).

Clipped z-score normalization was also implemented and tested, but its unbounded values and
dependence on the amount of background make it a less transparent default. CLAHE was not adopted:
its local enhancement can amplify noise or apparent edges and is not needed to make this pilot
usable. The uint8 PNGs are display previews only and are not scientific source data.

## Local pilot products and storage

The ignored pilot product contains 64 uint16 crops at 1,067 × 1,067, 64 anonymous crop previews,
64 localization overlays, 32 staged acquisition previews, 8 contact sheets, and aggregate/local
manifests. The crop arrays occupy 145,734,784 bytes for this pilot.

For 3,621 acquisitions (7,242 panels), an uncompressed cache at the proposed shape would be
approximately 16.49 GB (15.36 GiB) as uint16, 8.24 GB (7.68 GiB) as uint8, or 32.98 GB (30.72 GiB)
as float32, before filesystem/container overhead. Extrapolating the compressed 512-pixel pilot PNG
previews gives approximately 1.02 GB (0.95 GiB), but PNG size is content-dependent.

## Approval and failure gates

- Approve a documented adjudication process for all pilot and independent-validation laterality
  exceptions.
- Keep 0.15 mm/pixel, 160 × 160 mm, bilinear interpolation, and padding behavior frozen. The
  percentile transform remains deferred until model-input construction.
- Treat the completed 128-acquisition independent review and its predefined thresholds as the
  validation evidence; retain subgroup monitoring during full-cohort materialization.
- Preserve explicit automatic success/failure behavior and visual exception review; the validated
  rates do not justify a silent fallback when a future image fails.
- Keep an exception queue for localization failure and never substitute the panel midpoint as a
  silent fallback.

The subsequent full-cohort run was authorized separately. Its provisional cache is not approved
for cohort linkage because full-set visual QC triggered the frozen-method failure gate described
below.

## Milestone 5D independent-validation result

The frozen pipeline was applied without tuning to 128 independent acquisitions and 256 broad knee
panels. Midpoint separation was clean in 128/128 acquisitions; localization succeeded in 256/256
panels; and all 256 proposed 160 mm crops retained both tibiofemoral compartments with adequate
femoral and tibial context. Padding was required in 68 panels but did not remove required anatomy.
No sufficiently populated scanner, spacing, dimension, release, or QC subgroup showed a geometric
preprocessing failure.

Laterality remained evidence-gated: 113 acquisitions were confident and all supported the pilot
screen mapping, while 2 were ambiguous and 13 conflicting. The 15 unresolved acquisitions remain
unlabeled in a local exception queue. The conflict pattern reinforces that a unilateral DICOM
laterality tag on a bilateral image is not authoritative by itself.

The independently validated settings are now frozen for full-cohort processing. The 0.5th–99.5th
percentile clipping plus [0, 1] min-max transform remains a proposed future model-input operation
and was not used to create a modeling dataset. Full image acquisition remains separately gated by
NDA package availability and explicit approval before a large download.

## Milestone 5E full-set outcome

The frozen method processed 3,621 baseline acquisitions into 7,242 technically valid uint16
arrays. Midpoint splitting and output shape/type validation succeeded throughout. No geometry or
intensity parameter was changed, and the future percentile/min-max normalization was not
materialized.

Automatic return status proved insufficient as anatomical QC. In a 64-acquisition anonymous
stress sample, visual review found 53 successful/adequate pairs, 2 borderline pairs, and 9 pairs
where the shared detector row localized non-joint shaft anatomy and yielded inadequate crops.
Four of five reviewed acquisitions outside the prior padding envelope failed. Seven failures were
in one reportable AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 subgroup. This result is materially
worse than the independent validation and invokes the prespecified stop-before-retuning rule.

The rates from this enriched QC sample are diagnostic, not full-cohort prevalence estimates. The
provisional crops remain local and unchanged; failed and borderline cases are queued. Any localizer
revision must receive a new version and an independently defined validation plan. Until then, the
full image set is not ready for analysis-cohort linkage.

Laterality also remains unresolved at scale. Because no frozen automatic burned-in-marker reader
exists and DICOM fields cannot establish side alone, the full run made zero anatomical-side
assignments: 3,175 acquisitions are ambiguous and 446 conflicting. This conservatism is required
by the evidence policy, not a geometric-processing failure.

## Milestone 5F candidate result

Milestone 5F preserved the full V1 cache and evaluated a separately versioned deterministic V2
candidate. V2 performs physical resampling before independent panel localization and returns an
explicit confidence/QC state. It repaired all 9 known V1 failures in development, but independent
stress validation exposed a new off-body horizontal-centroid failure in the reportable
AGFA/ADC_51xx 0.17 mm/pixel family. The representative holdout had 128/128 adequate acquisitions;
the stress holdout had 115/128 (89.844%), including 11 wrong crops incorrectly labeled `PASS`.
Consequently V2 is not approved for full-cohort reprocessing.

The 160 × 160 mm, 1,067 × 1,067 uint16 crop specification was not changed. Review of all 25 panels
outside the previous padding envelope found that inadequate crops followed localization errors,
not evidence that the physical crop is intrinsically inappropriate when centered correctly.

The laterality policy was revised independently: marker and anatomical image evidence outrank
DICOM metadata, and an audited screen-position rule may be used only with bilateral-geometry and
exception checks. A new stratified 128-acquisition manual audit found 128/128 mappings consistent
with screen-left → anatomical right, zero reversals, 100% automatic rule coverage, and zero observed
incorrect assignments. This rule has not been materialized across the preserved full V1 outputs.

## Milestone 5G anatomical-safety result

V3 separates panel-layout QC, multi-candidate localization, and anatomical crop validation. A
candidate cannot receive `PASS` unless all gates agree. The validator explicitly measures bone
coverage above and below the center, joint-band and horizontal anatomical support, foreground
occupancy, blank background, and exposure/collimation dominance. It is designed to reject shaft-
only and off-body crops rather than merely returning a coordinate.

All 477 methodology-development acquisitions were adequate after V3, including every known V1/V2
failure; automatic states were 324 `PASS`, 146 `BORDERLINE`, and 7 `FAIL`, with 0 false passes. The
new untouched holdout contained 192 representative and 192 stress acquisitions. Representative
adequacy was 192/192 and stress adequacy was 191/192. The one inadequate stress crop was labeled
`BORDERLINE`; no visually inadequate crop was labeled `PASS` in either component.

The difficult AGFA/ADC_51xx family improved from 36/47 adequate with 11 unsafe V2 passes to 148/149
adequate with no unsafe V3 pass in the new holdout. The remaining failure was explicitly queued.
No crop-size, spacing, interpolation, output-shape, or intensity-policy change was made.

The safety gain is conservative coverage. Representative automatic states project roughly 70.8%
`PASS`, 28.6% `BORDERLINE`, and 0.5% `FAIL`, or about 1,056 acquisitions requiring review if applied
to the full 3,621. Human approval of that review workflow is required before full reprocessing.

## Milestone 5H frozen full-cohort materialization

The approved frozen V3 method has now been applied to all 3,621 baseline bilateral acquisitions.
The interrupted first run was recovered without trusting incomplete directory counts: 996 bundles
were fully verified and skipped, six partial cases were reclaimed and reprocessed, and 2,619
previously untouched cases were processed. All 3,621 final bundles contain two verified 1,067 ×
1,067 uint16 crops, and no hard processing failure occurred.

Automatic acquisition states are 2,645 PASS, 932 BORDERLINE, and 44 FAIL. The 978-acquisition
manual-review queue is slightly smaller than the representative projection and is distributionally
compatible with it. No threshold, target spacing, crop size, interpolation rule, anatomical gate,
or normalization policy was changed. See [the full Milestone 5H report](imaging-full-v3.md) for
recovery controls, laterality/linkage counts, subgroup monitoring, padding, storage, and the review
stopping gate.
