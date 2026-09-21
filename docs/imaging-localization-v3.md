# Localization V3 Candidate and Anatomical Safety Gate

## Decision boundary

Milestone 5G addresses localization safety after `v2_candidate_r3` produced 11 anatomically
incorrect crops labeled automatic `PASS` in its independent stress holdout. It does not change the
analysis cohort, laterality policy, 0.15 mm/pixel target spacing, bilinear interpolation, 160 × 160
mm crop, 1,067 × 1,067 uint16 output, or proposed future intensity normalization. V1 and V2
outputs remain preserved and must not be used interchangeably with V3.

`localization_v3_candidate_r1` was developed on 477 previously reviewed acquisitions. Its source
hash and parameter record were frozen before a new 384-acquisition holdout was selected or opened.
The holdout contains 192 representative and 192 stress-stratified acquisitions, has no overlap
with any prior imaging-methodology sample, and was selected without OA progression outcomes.

## Failure diagnosis

The earlier failures were not explained by crop size. A correctly centered 160 mm crop retains the
required tibiofemoral anatomy. V1 could select a strong vertical response from femoral/tibial shaft,
collimation, or exposure-field boundaries. V2 improved the vertical search but its decisive
AGFA/ADC_51xx stress failures were usually caused by an off-body horizontal centroid despite a
plausible vertical row. In that 47-acquisition V2 subgroup, all 11 inadequate acquisitions were
unsafe automatic passes. Their median horizontal padding was 18.3 mm versus 0 mm among adequate
cases; vertical peak and prominence did not separate the two groups.

V3 therefore treats both axes as safety problems. It searches multiple vertical bands and multiple
horizontal centers, ranks bilateral pairs only with weak positional priors, and requires a separate
anatomical validator before `PASS`. It contains no scanner-name branch or scanner-specific pixel
coordinate.

## Stage 0: panel-layout QC

Panel checks run after physical resampling and before localization. They measure physical field of
view, aspect ratio, robust contrast, coarse blank/background fraction, usable foreground width and
height, and dominant row/column exposure boundaries. A clearly unusable panel returns `FAIL`; a
large blank region, limited occupancy, or dominant boundary returns `BORDERLINE`. A panel that
fails structural checks is not silently promoted by a later localization score.

## Stage 1: candidate localization

The localizer:

1. applies robust scaling only to a working localization array;
2. constructs a vertical gradient profile over a panel-position-aware central region;
3. searches the physically plausible 30%–84% height interval;
4. retains as many as six vertical peaks separated by at least 18 mm;
5. evaluates a structural sliding-window center, a broad intensity centroid, and the expected
   bilateral panel region as horizontal alternatives;
6. ranks candidate pairs using local gradients, horizontal support, physical position, and
   bilateral row consistency; and
7. returns `BORDERLINE` when distinct candidates remain too close in score, rather than forcing
   the top coordinate.

The screen-position prior is a broad geometric prior, not an anatomical-side assignment. It is not
used as a scanner-specific coordinate rule.

## Stage 2: independent anatomical validation

The selected 160 mm candidate crop is evaluated independently of its ranking score. The gate
requires superior and inferior bony structure, a plausible central joint-space response,
horizontal plateau/condylar support, central anatomical occupancy, and acceptable blank fraction.
It also detects dominant collimation boundaries. Missing superior or inferior bone, a shaft- or
background-dominant crop, absent joint-band evidence, or excessive blank area prevents `PASS`.

Final state is the worst of panel layout, candidate confidence, anatomical validation, and
bilateral-consistency state. A visually wrong crop labeled `PASS` is the primary safety failure;
`BORDERLINE` and `FAIL` are explicit review-queue outcomes.

## Development results

All 477 development acquisitions were reviewed using anonymous contact sheets. They include all
known V1 and V2 failures and borderlines plus prior representative successes. All 477 V3 crops
were anatomically adequate. Automatic disposition was 324 `PASS`, 146 `BORDERLINE`, and 7 `FAIL`.
There were 0 false passes (0.000%).

These results are methodology-development evidence and are not an independent performance claim.

## New independent holdout results

| Holdout component | Acquisitions | Adequate | Adequacy | False PASS |
|---|---:|---:|---:|---:|
| Representative | 192 | 192 | 100.000% | 0 (0.000%) |
| Stress-stratified | 192 | 191 | 99.479% | 0 (0.000%) |
| Combined | 384 | 383 | 99.740% | 0 (0.000%) |

Combined automatic disposition was 290 `PASS`, 91 `BORDERLINE`, and 3 `FAIL`. The single
inadequate crop selected lower tibial anatomy; it was automatically labeled `BORDERLINE` and is
therefore a safely detected review case, not a false pass.

The new holdout includes 149 acquisitions in the reportable AGFA/ADC_51xx, 0.17 mm/pixel,
2,048 × 2,494 family. V3 produced 148 adequate and 1 inadequate result in that family, with 0 false
passes; the inadequate result was queued. No reportable manufacturer, model, spacing, dimension,
image-release, or OAI acceptance-code subgroup contained an unsafe pass or a systematic pattern of
anatomically incorrect accepted crops.

The representative holdout projects 70.833% automatic `PASS`, 28.646% `BORDERLINE`, and 0.521%
`FAIL`. Applied to 3,621 acquisitions, that is approximately 2,565 automatic passes and 1,056
manual-review cases (about 1,037 borderline and 19 fail). This is a substantial review burden and
must not be reduced by loosening the safety gate without another version and independent test.

## Acceptance decision

V3 meets the predefined safety criteria: representative adequacy is at least 99%, stress adequacy
is at least 95%, both false-pass rates are below their limits, and no subgroup shows systematic
unsafe acceptance. The evidence supports freezing `localization_v3_candidate_r1` for human
approval and subsequent versioned full-cohort reprocessing. Full reprocessing has not occurred in
this milestone.

The remaining implementation decision is operational: approve resources and an adjudication
procedure for the projected manual queue. Any attempt to improve automatic coverage is a new
localizer version and requires a new untouched validation sample.
