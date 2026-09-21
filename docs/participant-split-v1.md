# Milestone 6B: sole authoritative participant split

The split is derived only from the frozen final multimodal dataset, never from independently
rebuilt cohorts. Its approved source SHA is
`382d17228ca6eb85c064f4a2c2d8c402a2479bde6fc746c9a55f7fbd57eac190`.
All 6,961 knees from 3,621 participants are assigned, including incomplete-modality rows.

## Inspection and predefined design

Observed participant strata, `(eligible knees, primary progressing knees)`, are:

| State | Participants |
|---|---:|
| (1, 0) | 178 |
| (1, 1) | 103 |
| (2, 0) | 2,538 |
| (2, 1) | 636 |
| (2, 2) | 166 |

The smallest observed stratum has 103 participants (about 15 in each holdout); none is sparse.
Keep all five. The predefined minimum stratum size is 20, giving approximately three participants
per 15% holdout. If needed in synthetic data, conservatively coarsen globally in this order:
exact eligible-knee/event-knee counts; eligible-knee count plus any-event indicator; any-event
indicator; one all-participant stratum. Each attempted stage and counts are recorded. Coarsening
never uses predictors or secondary endpoints; exceptionally rare events cannot be guaranteed
representation in every holdout. No coarsening is necessary for the authoritative dataset.

`configs/split_v1.yaml` establishes the design before candidate comparison. There was no project-wide
numeric ML seed convention. Following the repository's versioned SHA-based selection seeds, use
`oai-final-multimodal-participant-split-v1`. SHA-256 ranking avoids RNG/library-version dependence.

## Deterministic candidate search

Use 70/15/15 integer weights and Hamilton largest-remainder global participant quotas: TRAIN 2,535,
VALIDATION 543, TEST 543. Equal remainders are resolved in TRAIN, VALIDATION, TEST order.
Enumerate every feasible floor/ceil stratum allocation satisfying these exact global quotas.
Allocation enumeration is deterministic; 256 candidate indices cycle through these allocations.
Within each stratum, rank participants by SHA-256 of `seed|candidate_index|stratum|participant_id`,
with participant ID as the collision tie-break. Disjoint slices supply the split assignments.

Before comparing candidates, fix this lexicographic objective:

1. Participant overlap is a hard failure, not a tolerated balance trade-off.
2. Maximum, then summed absolute participant-share deviation from 70/15/15.
3. Maximum, then summed absolute knee-share deviation.
4. Maximum, then summed absolute primary knee-event prevalence deviation from the overall prevalence.
5. Maximum, then summed absolute primary-event share deviation from 70/15/15.
6. Maximum, then summed absolute split-share deviation separately among one-knee and two-knee participants.
7. Lowest candidate index breaks any remaining tie.

Comparisons use exact rational arithmetic. Only participant identity, eligible knee count and primary
event burden enter selection. When precise strata are retained, candidates sharing an allocation
also share objective values; choosing the first such candidate is intentional, not a search over
secondary characteristics. Candidate scores and selected index are recorded. No downstream model
results exist or are consulted. Side, baseline KL, image-QC provenance, missingness and secondary
outcomes are described only after selection; they are never optimized.

Material-imbalance flags use absolute differences: participant share 0.1 percentage points; knee
share 0.5 points; primary prevalence 2 points; primary-event share 2 points; one-/two-knee participant
share 3 points. Integer discreteness allows at least one participant/event or two knees of allocation
difference, and one event per split in prevalence. These flags are disclosed rather than initiating
unplanned repeated searches. Natural variation is expected. Exact global participant quotas and
zero leakage are separately enforced.

## Artifacts, immutability and use

Artifacts reside in `data/processed/multimodal/final_v1/splits/v1/`:

- `participant_split_manifest.parquet`: participant-sorted assignments, eligible-knee count,
  primary-event knee count, stratum, version, seed and selected candidate index.
- `knee_split_manifest.parquet`: one row in original frozen-dataset order, with authoritative
  participant/knee/visit/side identity, split and dataset row index.
- `split_summary.json`: configuration, stratification inspection, search scores, split counts and use policies.
- `split_balance_audit.json`: endpoint/contribution/side/KL/provenance/modality balance, absolute
  and relative deviations, missingness and linkage/leakage audit.
- `split_freeze.json`: source and output hashes, relevant code/config/runtime versions and lock policies.
- `pre_change_preservation.json`: protected-source fingerprint captured before implementation.
- `split_freeze.json.lock`: zero-byte writer lock scoped only to this new split directory.

```sh
.venv/bin/python -m multimodal.splits
```

Production execution checks the complete preservation fingerprints before and after construction,
including the entire unchanged 6A bundle, cohort, imaging manifest, adjudications, automatic crops
and processing records, overrides, frozen methodology, raw image archives and raw tabular files.
It invokes no review write, cohort builder or image generator. Parquet readback validates exact
values, dtypes and ordering. On rerun identical outputs are reused without rewriting; differing
or corrupted existing outputs fail closed. The freeze timestamp is retained. A deliberately different
split requires explicit authorization and a separately versioned output—not replacement of v1.
Concurrent split writers are serialized by the new split's lock sidecar, never by a review/data lock.

Use `multimodal.splits.load_frozen_split()` to validate and read memberships without independently
resplitting. This exact split applies to both formulations, all tabular/image/multimodal models,
ablations, calibration comparisons and interpretation. Model-specific eligibility may select rows
later, but must inherit these labels; it must never create new split assignments.
Stratification groups and event-burden columns are generation metadata, never predictor features.

TRAIN fits models and all train-fitted preprocessing. VALIDATION supports model/hyperparameter/
threshold selection, early stopping and scientifically appropriate calibration-model selection.
TEST is locked for final evaluation only; future test model performance must not inform development.

No modeling, imputation, encoders, scalers, image normalization, fitted statistics, feature selection,
class weights, resampling, augmentation or tensor generation occurs in 6B. All artifacts are local
restricted research data; reports/documentation are aggregate-only.
