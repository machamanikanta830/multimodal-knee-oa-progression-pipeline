# Outcome Definition

## Approved v1 outcomes

All radiographic changes use the same participant-knee, V00 and V06, and `READPRJ` 15. Projects
are never pooled.

### A. Radiographic KL progression

For knees with usable baseline and V06 KL grades and baseline KL below 4:

```text
radiographic_kl_progression = (V06 xrkl - V00 xrkl) >= 1
```

A stable or lower V06 KL grade is a non-event; decreases are neither corrected nor recoded. The
label remains missing for radiographic-ineligible knees. This release yields 961 events among
6,851 eligible knees (14.027%).

### B. Composite progression

For baseline-eligible knees with either a usable V06 KL grade or a qualifying pre-V06
replacement:

```text
composite_progression = radiographic_kl_progression OR replacement_before_v06
```

When the V06 KL grade is unavailable but a qualifying replacement is documented, the composite
label is an event. A missing radiograph without a qualifying replacement does not become an
assumed non-event; the knee is composite-ineligible. This release yields 1,071 events among 6,961
eligible knees (15.386%). The 110 additional events are qualifying replacements without usable
V06 KL; no qualifying replacement overlaps a usable V06 KL record in this release.

Replacement timing uses the actual V06 radiograph date when present, otherwise exactly 48
calendar months after the baseline radiograph. See `cohort-definition.md` for the full rule.

## Secondary structural outcome

The approved secondary label is an increase of at least 1 in either medial or lateral OARSI JSN:

```text
jsn_progression =
    (V06 xrjsm - V00 xrjsm >= 1)
    OR (V06 xrjsl - V00 xrjsl >= 1)
```

Eligibility requires usable numeric `xrjsm` and `xrjsl` values in the documented 0–3 range at
both visits, unchanged project 15, and the same baseline rules as the KL analysis. V06 contains
observed partial grades such as 1.2 and 2.4; they are retained as documented in-range values, but
the event threshold remains an increase of at least one full grade. The result is 734 events among
6,851 knees (10.714%).

## Sensitivity information retained in memory

The logic separately retains:

- radiographic KL progression alone;
- qualifying replacement status and timing;
- radiographic eligibility;
- composite eligibility;
- replacement cases with and without V06 KL; and
- enough flags to later analyze KL progression while excluding replacement cases.

No post-baseline replacement field is part of a baseline feature mask or predictor panel.

## Not decided here

- Whether confirmation status or replacement type should restrict a later adjudicated
  sensitivity analysis.
- Whether KL change should be confirmed at another visit or handled with an ordinal model.
- How correlated bilateral outcomes will be handled statistically.
- Whether JSN is a secondary endpoint, sensitivity endpoint, or component of a later expanded
  composite beyond the approved rule above.
