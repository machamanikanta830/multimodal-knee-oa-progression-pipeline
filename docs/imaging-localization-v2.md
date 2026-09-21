# Imaging localization V2 candidate

## Scope and stopping rule

Milestone 5F developed and independently evaluated an outcome-blind imaging-methodology candidate.
The existing full-cohort outputs remain `v1_provisional`; all revised sample outputs are separately
versioned `v2_candidate`. No full-cohort V2 reprocessing, cohort change, modeling split, or model
training was performed.

The development sample contains 192 acquisitions: all 9 known V1 localization failures, both known
borderline cases, 53 reviewed successes, the earlier pilot, and additional heterogeneity coverage.
The sealed independent holdout contains 256 previously uninspected acquisitions split evenly into
representative and stress-stratified components. Development/holdout overlap is zero, and no OA
progression outcome was read for selection.

## V1 failure diagnosis

V1 localized a shared row on the two unresampled broad panels. Its broad vertical search allowed a
strong shaft, collimation edge, hardware, or exposure-field response to dominate the true
tibiofemoral joint response. The median V1 row fraction was 0.530 in visually successful cases and
0.735 in failures, consistent with selection of a lower shaft band in the dominant failure mode.
Horizontal positioning was a secondary contributor.

Seven of the 9 known V1 failures were in the AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 family;
the other 2 were in the related Agfa-Gevaert/ADC_5146 family. This implicates field of view and
profile scaling rather than a universal fixed pixel coordinate. The diagnosis supports physical
normalization and panel-independent evidence; it does not support scanner-specific coordinates.

## V2 candidate algorithm

The frozen candidate evaluated here:

1. resamples each broad panel to 0.15 mm/pixel before localization;
2. robustly scales the 1st–99th intensity percentiles for localization only;
3. calculates a physically smoothed vertical-gradient profile over the central 60% of columns;
4. restricts the joint-row search to a dimensionless plausible vertical band;
5. localizes each panel independently;
6. estimates the horizontal center from broad osseous foreground near the proposed row;
7. uses contralateral row agreement only as secondary evidence and permits a constrained rescue
   only when the affected panel retains independent evidence; and
8. returns a joint center, confidence, and `PASS`/`BORDERLINE`/`FAIL` state.

The scientific crop representation remains unchanged: 160 × 160 mm, 1,067 × 1,067 pixels,
uint16. Robust scaling is not written into the canonical crop.

## Development result

At panel level, automatic V2 results were 359 `PASS`, 19 `BORDERLINE`, and 6 `FAIL`. At acquisition
level they were 179, 10, and 3 respectively. Anonymous visual review found 189/192 successful,
adequate acquisitions and 3 failures, for 98.438% crop adequacy. Every visual failure was detected
as an automatic failure; no development failure was incorrectly labeled `PASS`.

V2 corrected all 9 known V1 failures and both V1 borderline cases. However, it produced 3 new
failures among the reviewed V1-success cases. This was the first indication that solving the V1
vertical-band failure did not establish robust horizontal localization.

## Frozen independent holdout result

The candidate settings were recorded before the holdout was opened and were not changed afterward.
Automatic acquisition states were 244 `PASS`, 10 `BORDERLINE`, and 2 `FAIL`. Visual adjudication
showed:

| Holdout component | PASS | BORDERLINE | FAIL | Adequate crops | Adequacy |
|---|---:|---:|---:|---:|---:|
| Representative | 126 | 2 | 0 | 128/128 | 100.000% |
| Stress-stratified | 115 | 0 | 13 | 115/128 | 89.844% |
| Combined | 241 | 2 | 13 | 243/256 | 94.922% |

The representative component met its ≥98% localization and crop criteria and had no clear failure.
The stress component missed its ≥95% adequacy criterion. More importantly, only 2 of its 13 visual
failures were automatically failed; 11 wrong crops were incorrectly labeled `PASS`. That violates
the safety priority that a detected review case is preferable to a wrong crop passed automatically.

## Holdout failure mechanism and clustering

Eleven failures cluster in the AGFA/ADC_51xx, 0.17 mm/pixel, 2,048 × 2,494 family. Only 36/47
acquisitions in that reportable holdout subgroup had adequate crops (76.596%). The 0.17 mm/pixel
family therefore also had 36/47 adequate acquisitions, whereas all other spacings combined had
207/209 (99.043%). The principal V2 holdout failure was a correct or near-correct joint row paired
with an off-body horizontal centroid. The foreground profile was attracted toward the peripheral
exposure/soft-tissue field, and the permissive column confidence rule did not recognize the
screen-panel geometry reversal. Two isolated vertical/localization failures occurred outside that
reportable scanner group.

Failures were concentrated in the baseline image release and `Y` acceptance group that contains
this scanner family; these labels do not independently establish an exclusion rule. There is no
evidence that changing the 160 mm crop size would repair the observed error.

## Padding review

Holdout padding occurred in 208/512 panels: 206 horizontally and 2 vertically. Maximum aggregate
padding was 30.60 mm horizontally and 14.40 mm vertically. Padding was not itself deemed failure
when both compartments and adequate femoral/tibial context remained visible.

All 25 panels outside the earlier validation padding envelope were re-reviewed across 13
acquisitions. V2 produced 16 adequate and 9 inadequate results; only 1 of the 9 inadequate panels
was automatically failed. Sixteen still required some padding, with maxima of 32.85 mm horizontal
and 17.25 mm vertical. The inadequate results were caused by incorrect localization, including one
clear collimation/shaft-band selection, rather than evidence that a correctly centered 160 mm crop
is intrinsically too large. The crop size remains unchanged pending a reliable localizer.

## Decision

`v2_candidate_r3` is **not safe to freeze or apply to all 3,621 acquisitions**. The deterministic
approach has an identifiable, potentially correctable horizontal-QC failure, so a learned landmark
detector is not justified yet. A new version should constrain panel geometry, detect a centroid on
the wrong side of the expected bilateral layout, and validate horizontal bone coverage. Any such
change must be made on development data and evaluated on another untouched holdout before
full-cohort reprocessing.
