# Milestone 5I — manual QC adjudication

Localization V3 and Laterality V2 are frozen. Manual adjudication is not another
algorithm-development stage: it records human judgement about crops the frozen pipeline already
produced, and it never modifies an automatic result, a threshold, or a frozen source file.

## Launching the review tool

```
cd /path/to/multimodal-knee-oa-progression-pipeline
.venv/bin/python -m imaging.v3_review_ui --port 8765 --open-browser
```

The tool binds only to `127.0.0.1` and refuses any other interface. It prints the review URL and
the current progress counters as JSON on startup, then serves until interrupted with `Ctrl+C`.
Progress can be inspected without starting the server:

```
.venv/bin/python -m imaging.v3_review_ui --progress-only
```

## What the reviewer sees

The queue holds 978 acquisitions: 932 BORDERLINE, 44 FAIL, and 2 automatic PASS acquisitions whose
laterality was not resolved. The 2,645 automatic PASS acquisitions are not queued, because they
already passed the independently validated frozen safety gate.

Each case shows the anonymous bilateral acquisition with both proposed 160 mm crop boxes, a
clickable localization overlay per screen panel carrying the frozen localizer's ranked alternative
centers, and both proposed crops. The sidebar lists the automatic acquisition state, per-panel QC,
layout and anatomical-validator states, localization confidence, the automatic joint center, padding
in millimetres, and the non-identifying reason codes.

Nothing identifying is rendered. Cases are labelled by their queue-local acquisition index only, and
every page is checked before it is served: if a rendered page would contain a participant
identifier, source participant identifier, accession number, or archive reference, rendering fails
rather than serving the page. The top and bottom 35 mm of every preview are masked because some
acquisitions carry burned-in text there; the masking is preview-only and the stored uint16 crops
retain their original pixels.

## Decisions are per panel

A bilateral acquisition can have one adequate knee and one inadequate knee, so each screen panel
takes its own decision and one problematic panel never forces the other knee to be rejected.

| Decision | Meaning |
|---|---|
| `ACCEPT` | the proposed joint localization and crop are anatomically adequate |
| `REJECT` | this panel cannot safely yield an adequate crop and stays excluded |
| `NEEDS_CENTER_OVERRIDE` | the automatic crop is wrong but a reliable tibiofemoral center can be specified by hand |

Both panels must carry a decision before a case is recorded, which keeps a half-reviewed
acquisition out of the log entirely rather than leaving it in an ambiguous state.

To specify a corrected center, click the tibiofemoral joint on that panel's overlay. The click draws
a magenta 160 mm crop box and crosshair so the proposed crop can be compared against the yellow
automatic box before anything is submitted, and it selects `NEEDS_CENTER_OVERRIDE` for that panel.
Clicking again moves the center; the per-panel clear button removes it.

The sidebar carries a standing legend, describing crop adequacy only and not clinical findings:

| Decision | Legend shown to the reviewer |
|---|---|
| ACCEPT | Joint is centered with adequate femoral and tibial context. |
| OVERRIDE | Joint is visible, but the proposed crop center is wrong. |
| REJECT | A reliable joint crop cannot be obtained. |

## Laterality exceptions

Three acquisitions (queue cases 226, 319, and 347) have unresolved anatomical sides because the
frozen policy's bilateral-structure checks did not confirm two separated limbs. They appear in the
ordinary queue with an extra required section, and are also collected at `/exceptions`.

Laterality is never forced. Each of these cases needs an explicit choice:

| Decision | Meaning |
|---|---|
| `CONFIRM_VALIDATED_MAPPING` | the structure supports the validated mapping, screen-left is the anatomical right knee and screen-right is the anatomical left knee |
| `MARK_UNRESOLVED_EXCLUDE` | the side cannot be established, so neither knee may join a side-specific cohort row |

A case with unresolved laterality cannot be recorded without this decision, and the decision is
refused for any acquisition the frozen policy already resolved.

## Persistence and resume

Every adjudication is appended as one JSON line to a Git-ignored log. Nothing is ever rewritten in
place, and the queue is ordered by a deterministic `review_index`, so an interrupted session resumes
at the first undecided case.

Each record stores the automatic panel state and the automatic joint center alongside the human
decision, so the automatic result stays readable next to the judgement made about it. Every record
also stores the SHA-256 fingerprint of the queue it was made against, so decisions cannot be
silently reinterpreted against a different queue. The header shows reviewed, remaining, and the
running ACCEPT, REJECT, and override totals; `/progress` returns the same counters as JSON.

### The write transaction

An in-memory lock only orders threads inside one interpreter, so two review servers sharing a log
could each read an empty log and each believe it was recording the first decision. Every write
therefore runs as one transaction under an `fcntl.flock` advisory lock held on a sidecar
`adjudications.jsonl.lock`:

1. acquire the exclusive file lock;
2. re-read the authoritative log from disk;
3. clear any interrupted final append, preserving it first;
4. re-verify the queue fingerprint and every record's schema;
5. verify the expected current or superseded state;
6. assign the next sequence;
7. append the fully prepared bytes, retrying a short write until complete;
8. `fsync`, then release the lock.

Because the read and the append are inside the same lock, a losing writer sees the winner's record
and receives a deterministic conflict. Nothing is appended and no earlier decision changes.

Each rendered page also carries the sequence it was rendered from as
`expected_current_sequence`. If the case moved on before the form was submitted — someone else
decided it, or revised it — the submission is refused with an explicit "nothing was recorded" page
rather than appending a decision that reinterprets work the reviewer never saw.

### Revisions

Re-deciding a case is refused unless the revision is explicit. A revision is appended as a new
record carrying `supersedes`, and the superseded record is retained in full.

### Crash recovery

A crash during an append can leave an unterminated record at end of file. On load, that single
terminal fragment is reported rather than raised, so prior decisions stay readable and the session
resumes. The next write copies the fragment into an append-only
`adjudications.jsonl.corrupt_tail` sidecar, with its bytes preserved base64-encoded, and then
shortens the log by exactly the fragment's length. Every previously valid record survives untouched;
leaving the fragment in place is not an option, because the next append would fuse onto it and
destroy a real record.

An unterminated record is never promoted to history even when it happens to parse, because a write
that lost its newline cannot be distinguished from one that stopped early, and a decision whose
durability was interrupted should not silently become authoritative.

Damage anywhere earlier in the log fails closed with `LogCorruptionError`. That includes a
newline-terminated line that does not parse, since a completed write cannot be explained by an
interrupted append and may indicate edited history. Arbitrary malformed JSON is never repaired.
`/progress` reports both any pending fragment and the number already quarantined.

## Final status precedence

`effective_panel_status()` resolves each queued knee panel under one explicit order:

1. the effective record for a case is its highest-sequence record, and every earlier record is
   superseded and reported as history only, so a superseded record can never become final again;
2. a panel's human decision comes only from that effective record;
3. the automatic V3 state stays in `automatic_acquisition_state` and `automatic_panel_state` and is
   never replaced by a human decision;
4. a laterality decision governs only `anatomical_side_resolved`, never the panel's crop verdict, so
   an unresolved side cannot silently reject a knee and a confirmed side cannot silently accept one.

Panel decisions map to `crop_source` as `HUMAN_ACCEPT`, `HUMAN_OVERRIDE`, or `REJECTED`, and an
undecided panel is `PENDING_REVIEW` rather than defaulted in either direction. Both panels of a case
are recorded in one transaction, so a revision revises the pair; panel independence lives in the
decisions themselves, not in separate records.

`--snapshot` writes this table to `review/effective_panel_status.parquet` beside the decision
snapshot.

## Corrected crops

Corrected crops are produced by a separate pass after review, never by the interactive tool:

```
.venv/bin/python -m imaging.override_crops --list-only
.venv/bin/python -m imaging.override_crops
```

Only the overridden panel is rebuilt. The pass re-derives that acquisition's resampled panels from
the raw archive with the frozen steps, then applies the frozen downstream specification at the human
center: 0.15 mm/pixel, a 160 x 160 mm physical crop, a 1067 x 1067 uint16 output, and the same
low-percentile padding policy. Localization V3 is not re-run; the human coordinate replaces only the
center the frozen crop step consumes.

Results are written to `overrides/acquisition_XXXX/` as human-adjudicated derivatives, separate from
the automatic `crops/` tree. Each corrected panel record holds the panel, the automatic center, the
manual center, the center shift, the decision, the corrected crop path, the padding actually applied,
the adjudication sequence it was built from, the adjudication schema version, the override crop
version, the frozen preprocessing, localizer and laterality policy versions, its own
`corrected_crop_sha256`, and the SHA-256 of the untouched automatic crop it supersedes.

The pass is resumable, and reuse is verified rather than assumed. Before an existing corrected crop
is reused, the pass checks that the crop file exists, that its shape is 1067 x 1067, that its dtype
is uint16, that its SHA-256 still matches the recorded `corrected_crop_sha256`, and that its
provenance names the adjudication revision currently in force. Any mismatch fails closed: the pass
refuses to reuse or overwrite a crop whose origin cannot be established, rather than silently
rebuilding over it.

A record naming a revision that is not in the log at all, or a revision newer than the active one,
is an error. A record naming a genuinely superseded earlier revision of that panel is not: a revised
center rebuilds only its own panel, which is the documented resume behaviour. A panel whose override
was superseded by a later ACCEPT or REJECT is not regenerated at all. `--force` rebuilds without
consulting the existing crop. Temporary extractions are removed even when a panel fails.

## Preservation during adjudication

The automatic crops carry a hash baseline recorded before the first decision, covering 7,242 panel
crops and 3,621 completion records totalling 16.517 GB:

```
.venv/bin/python -m imaging.adjudication_audit --strict
```

The first call created `audit/automatic_crop_ledger.parquet`; later calls verify it and report any
artifact added, removed, or modified. A corrected crop written to the `overrides` tree leaves the
baseline unchanged by construction, which is what allows a later finalization step to state that the
adjudicated manifest was built on the same pixels the automatic pass produced.

## Finalization is blocked

The final adjudicated imaging manifest is not built until all 978 queue entries and all three
laterality exceptions carry human decisions. The expected target is 6,961 participant-knee rows:
6,955 already-resolved eligible knee panels plus the 6 knee linkages held by the three laterality
exceptions. No modeling split is created in this milestone.
