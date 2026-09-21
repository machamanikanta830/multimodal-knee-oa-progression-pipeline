# Milestone 5J: final adjudicated imaging

This is imaging resolution only. It does not change the locked cohort, generate model splits, train
models, relocalize joints, change laterality policy, or write human adjudications.

## Authoritative outputs

Under `data/processed/oai_images/v3_frozen_full/final_adjudicated_v1/`:

- `pre_change_preservation.json`: independent pre-write review counts and protected SHA fingerprints.
- `imaging_manifest.parquet`: exactly one row, in original order, per locked analysis-cohort knee.
- `panel_resolution.parquet`: all 7,242 panels, including explicit out-of-cohort accounting.
- `override_linkage.parquet`: all requested corrected panels, including those outside the cohort.
- `override_materialization.json`: generation/reuse outcomes from the unchanged frozen generator.
- `integrity_audit.json`: full linkage, artifact, SHA, laterality, accounting and preservation audit.

Publication is complete only when the integrity audit reports `integrity_passed` and
`reconciliation_passed`. References in the manifests are relative to the V3 dataset root. These are
restricted local research artifacts, not public exports. Internal participant identifiers are retained
for linkage; raw archive GUIDs, accession numbers and private URLs are omitted from new manifests.

## Resolution rules

Unqueued automatic PASS uses the original crop (`AUTO_PASS`). Queued human ACCEPT uses the original
crop (`HUMAN_ACCEPT`). A latest-revision corrected-center verdict uses a separate corrected artifact
(`HUMAN_OVERRIDE`). Human REJECT and explicitly unresolved laterality have no effective image and an
explicit exclusion reason (`REJECTED` and `UNRESOLVED`). Missing or contradictory data is an error,
never a silently invented exclusion. All cohort rows are retained, including any excluded row.

Frozen OAI Laterality V2 maps screen-left to anatomical RIGHT (knee code 1), and screen-right to
anatomical LEFT (code 2). Each exceptional acquisition requires an explicit human mapping confirmation
or exclusion. Misleading unilateral DICOM metadata is not authoritative.

## Fail-closed execution and reuse

Run from the repository root with the project environment:

```sh
.venv/bin/python -m imaging.final_adjudicated --work-root data/interim/oai_images/milestone_5j_override_work
```

The staging root must have no pre-existing files. Only newly extracted staging data is cleaned by the
existing frozen override pipeline; no previous imaging or raw data is removed.

Before generation, the finalizer verifies existing preservation ledgers, the authoritative live-source
versus freeze hash, preprocessing and laterality sources, automatic pixels/records, queue/log identity,
revision chains, cohort linkage, corrected centers and original source-archive hashes. It never
recreates authoritative ledgers, uses `force`, changes frozen code, or overwrites automatic crops.

Corrected crops retain the frozen geometry: 0.15 mm/pixel, 160 mm square, 1067 x 1067, uint16, and the
existing padding rule. Existing corrected crops are reused only after the frozen provenance verifier
accepts them against the active revision. Invalid or superseded derivatives fail closed. Identical
Parquet outputs are retained byte-for-byte; differing existing outputs cause an error. Audit/status
JSON may refresh to describe a successful reuse run, without new adjudications or crop revisions.

Every included cohort file is checked for existence, shape, dtype and SHA. Automatic crops, processing
records, log, queue, cohort, configurations and frozen methodology are rechecked after materialization.
The audit reports effective provenance counts, explicit exclusions, zero-drop reconciliation, side and
participant distributions, and the all-panel/out-of-cohort reconciliation.
