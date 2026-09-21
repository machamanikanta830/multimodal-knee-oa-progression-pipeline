# Radiographic Outcome Options

## Status and rules used

These are candidate calculations for human review, not final labels. Every change calculation
matched `subjectkey + side` within one unchanged `readprj`. No projects were pooled or recoded.
Only aggregate results are retained.

The official OAI image-assessment guide supports comparing timepoints within one project and
warns against creating change from different projects. The downloaded dictionary documents
`xrkl` as K-L grade and `xrjsm`/`xrjsl` as medial/lateral OARSI JSN grades. The two prespecified
candidate rules evaluated here are:

1. follow-up `xrkl - baseline xrkl >= 1`;
2. follow-up minus baseline `>= 1` in either `xrjsm` or `xrjsl`.

These calculations do not resolve reader reliability, clinical importance, regression to the
mean, replacement handling, or whether a one-grade change should be confirmed at another visit.

## Candidate 1 — K-L increase of at least one grade

`K-L among progressors/non-progressors` is the baseline grade distribution in order
`0/1/2/3/4`. “After excluding K-L 4” reports eligible knees/events/event rate.

| Project | Horizon | Eligible | Events | Non-events | Rate | K-L among progressors | K-L among non-progressors | Baseline K-L 4 | After excluding K-L 4 |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| 15 | V01, 12 m | 8,361 | 520 | 7,841 | 6.219% | 125/148/158/89/0 | 3112/1365/2053/1051/260 | 260 | 8,101 / 520 / 6.419% |
| 15 | V03, 24 m | 7,856 | 653 | 7,203 | 8.312% | 153/180/196/124/0 | 2890/1260/1879/947/227 | 227 | 7,629 / 653 / 8.559% |
| 15 | V05, 36 m | 7,494 | 833 | 6,661 | 11.116% | 195/236/240/162/0 | 2733/1165/1730/839/194 | 194 | 7,300 / 833 / 11.411% |
| 15 | V06, 48 m | 7,014 | 961 | 6,053 | 13.701% | 226/262/282/191/0 | 2521/1064/1609/696/163 | 163 | 6,851 / 961 / 14.027% |
| 37 | V06, 48 m | 3,164 | 199 | 2,965 | 6.290% | 147/52/0/0/0 | 2116/846/3/0/0 | 0 | 3,164 / 199 / 6.290% |
| 37 | V08, 72 m | 3,121 | 497 | 2,624 | 15.924% | 360/133/3/1/0 | 1864/751/8/1/0 | 0 | 3,121 / 497 / 15.924% |
| 37 | V10, 96 m | 3,237 | 615 | 2,622 | 18.999% | 424/184/6/1/0 | 1897/714/9/1/1 | 1 | 3,236 / 615 / 19.005% |
| 42 | V06, 48 m | 469 | 246 | 223 | 52.452% | 69/157/19/1/0 | 1/6/156/60/0 | 0 | 469 / 246 / 52.452% |
| 42 | V08, 72 m | 446 | 261 | 185 | 58.520% | 67/153/31/10/0 | 1/4/134/46/0 | 0 | 446 / 261 / 58.520% |
| 42 | V10, 96 m | 419 | 244 | 175 | 58.234% | 61/136/34/13/0 | 1/1/128/45/0 | 0 | 419 / 244 / 58.234% |

Project 42's very different grade mix and event rates reinforce that it must not be treated as a
stand-alone interchangeable sample. The official guide's project 37/42 recoding statement exists,
but no recode has been applied.

## Candidate 2 — medial or lateral JSN increase of at least one grade

Eligibility requires nonmissing medial and lateral JSN at both timepoints. The baseline K-L
distributions are again ordered `0/1/2/3/4`.

| Project | Horizon | Eligible | Events | Non-events | Rate | K-L among progressors | K-L among non-progressors | Baseline K-L 4 | After excluding K-L 4 |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| 15 | V01, 12 m | 8,361 | 370 | 7,991 | 4.425% | 49/49/176/95/1 | 3188/1464/2035/1045/259 | 260 | 8,101 / 369 / 4.555% |
| 15 | V03, 24 m | 7,855 | 497 | 7,358 | 6.327% | 62/72/230/131/2 | 2981/1368/1844/940/225 | 227 | 7,628 / 495 / 6.489% |
| 15 | V05, 36 m | 7,494 | 633 | 6,861 | 8.447% | 82/107/275/168/1 | 2846/1294/1695/833/193 | 194 | 7,300 / 632 / 8.658% |
| 15 | V06, 48 m | 7,014 | 735 | 6,279 | 10.479% | 96/120/323/195/1 | 2651/1206/1568/692/162 | 163 | 6,851 / 734 / 10.714% |
| 37 | V06, 48 m | 3,164 | 49 | 3,115 | 1.549% | 31/18/0/0/0 | 2232/880/3/0/0 | 0 | 3,164 / 49 / 1.549% |
| 37 | V08, 72 m | 3,121 | 177 | 2,944 | 5.671% | 119/51/6/1/0 | 2105/833/5/1/0 | 0 | 3,121 / 177 / 5.671% |
| 37 | V10, 96 m | 3,237 | 276 | 2,961 | 8.526% | 169/98/8/1/0 | 2152/800/7/1/1 | 1 | 3,236 / 276 / 8.529% |
| 42 | V06, 48 m | 469 | 191 | 278 | 40.725% | 65/97/24/5/0 | 5/66/151/56/0 | 0 | 469 / 191 / 40.725% |
| 42 | V08, 72 m | 446 | 212 | 234 | 47.534% | 63/99/36/14/0 | 5/58/129/42/0 | 0 | 446 / 212 / 47.534% |
| 42 | V10, 96 m | 419 | 204 | 215 | 48.687% | 57/86/42/19/0 | 5/51/120/39/0 | 0 | 419 / 204 / 48.687% |

For project 15/V06, excluding 163 baseline K-L 4 knees removes one JSN event and changes the
rate from 10.479% to 10.714%. K-L 4 is a strict ceiling for K-L progression but not necessarily
for compartment-specific JSN; that scientific distinction needs review.

## Candidate 3 — combined structural endpoint

No general `K-L increase OR JSN increase` label was calculated. The official V99 dictionary does
contain derived incident-ROA fields such as `lxioan`/`rxioan` (incident K-L ≥2 with JSN) and
separate whole-grade JSN progression summaries, but those are not documentation for a generic
combined progression rule across all baseline grades. A combined endpoint remains unresolved and
must be specified and justified before code is added.

All V99 derived outcomes are excluded from predictors regardless of whether they later serve as
validation references.

## Quantitative minimum medial JSW

Project 16 is the documented longitudinal quantitative JSW project. Change is reported as
`follow-up mcmjsw - baseline mcmjsw` in millimetres; negative values are mathematical decreases.
No progression threshold is imposed.

| Horizon | Eligible knees | Participants | Mean (SD), mm | P05 | P25 | Median | P75 | P95 | Min / Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V01, 12 m | 5,777 | 3,252 | -0.0641 (0.5693) | -0.9682 | -0.2870 | -0.0220 | 0.2000 | 0.7322 | -5.800 / 3.430 |
| V03, 24 m | 5,429 | 3,071 | -0.1886 (0.6362) | -1.2486 | -0.4340 | -0.1130 | 0.1350 | 0.6256 | -5.151 / 6.035 |
| V05, 36 m | 5,199 | 2,967 | -0.3276 (0.7357) | -1.6212 | -0.5870 | -0.2240 | 0.0500 | 0.5660 | -5.400 / 9.300 |
| V06, 48 m | 4,954 | 2,839 | -0.3997 (0.7879) | -1.9270 | -0.6860 | -0.2580 | 0.0257 | 0.5933 | -5.800 / 3.542 |
| V08, 72 m | 3,521 | 2,148 | -0.4731 (0.8595) | -2.1300 | -0.8130 | -0.3390 | 0.0300 | 0.6330 | -5.927 / 2.832 |
| V10, 96 m | 3,637 | 2,234 | -0.6438 (1.0064) | -2.5850 | -1.0700 | -0.4320 | -0.0260 | 0.5976 | -7.027 / 2.634 |

The official guide notes that knees with end-stage radiographic OA or very narrow JSW at 48
months were not measured at V08/V10. Long-term quantitative-JSW completeness is therefore
outcome-related and cannot be treated as ordinary random attrition.

The observed extrema (for example +9.300 mm at V05 and -7.027 mm at V10) require measurement/QC
review before any analysis. They were neither removed nor corrected during feasibility work.

## Knee replacement and surgery

The package contains both baseline history and future outcomes:

- V00 `ksurg`/`ksurgl`/`ksurgr` describe prior knee surgery or arthroscopy and are well populated.
  V00 `ksurgcv` reports 4,724 no and 67 yes among 4,791 nonmissing participants for an earlier
  total/partial knee replacement. Direct V00 `krsl`, `krsr`, `sreplkr`, and `sreprkr` are empty.
- Post-baseline `ksrgl12`/`ksrgr12` describe surgery or arthroscopy since the prior visit and are
  populated at multiple follow-ups. `krsl12`/`krsr12` contain replacement reports at selected
  follow-ups. These are post-baseline events, not predictors.
- V99 `lkdays`/`rkdays` provide days from enrollment to follow-up replacement; 465 left and 470
  right knee dates are populated (935 knee events). `lktlpr`/`rktlpr` classify 903 events: 839
  total (code 1) and 64 partial (code 2). Confirmation, pre-operative diagnosis, prior/after
  visits, and pre/post X-ray visit fields are also available.
- V99 baseline-X-ray replacement flags identify 25 left and 38 right knees in the full enrollee
  table. None occur among project 15 baseline assessment knees.

Scheduled-horizon counts below use replacement days compared with 365.25-day scheduled-year
cutoffs. This is an aggregate feasibility approximation, not a future censoring implementation;
the definitive rule must use reviewed event and image-acquisition dates.

| Project/horizon | Baseline knees | Any documented follow-up replacement | Replacement by horizon | Also has target X-ray | No target X-ray |
|---|---:|---:|---:|---:|---:|
| 15/V01 | 8,921 | 914 | 23 | 2 | 21 |
| 15/V03 | 8,921 | 914 | 67 | 5 | 62 |
| 15/V05 | 8,921 | 914 | 120 | 2 | 118 |
| 15/V06 | 8,921 | 914 | 189 | 4 | 185 |
| 37/V10 | 3,349 | 74 | 7 | 0 | 7 |
| 42/V10 | 482 | 53 | 9 | 1 | 8 |

Replacement can remove the native joint from radiographic follow-up and create informative
missingness. Three defensible strategies remain open:

1. exclude/censor knees replaced before the target acquisition and analyze replacement
   separately;
2. treat confirmed total/partial replacement as structural progression in a composite outcome;
3. use a multi-state or competing-event framework that separates radiographic progression,
   replacement, loss to follow-up, and death.

No strategy has been selected. Treating replacement as progression would need adjudication rules,
timing, reason for surgery, and sensitivity analyses; simple exclusion risks informative bias.

## Recommendation for human review — not a final outcome

For a v1 project-15/V06 protocol, carry both **K-L increase ≥1** and **medial-or-lateral JSN
increase ≥1** into domain-expert review. K-L has 961 events among 7,014 knees (13.701%); JSN has
735 among 7,014 (10.479%). K-L is simple and broadly interpretable but has an explicit grade-4
ceiling. JSN is more directly related to joint-space loss but needs compartment and reliability
review. Quantitative JSW should remain a continuous sensitivity outcome until a validated
threshold is approved. Do not instantiate a combined endpoint yet.
