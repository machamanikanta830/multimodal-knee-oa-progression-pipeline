"""Local manual-review workflow for the frozen V3 full-cohort crop queue.

The tool is a loopback-only web application built on the standard library, so it adds no dependency
and runs from the same environment as the rest of the pipeline. It shows the anonymous bilateral
acquisition, both proposed knee crops, the localization overlays with the frozen localizer's ranked
alternatives, and the automatic state and reason codes.

Decisions are recorded per screen panel, because one knee of a bilateral acquisition can be
adequate while the other is not. Each panel receives ACCEPT, REJECT, or NEEDS_CENTER_OVERRIDE, and
a corrected tibiofemoral center can be clicked directly on that panel's overlay. Acquisitions whose
laterality could not be resolved by the frozen policy additionally require an explicit laterality
decision, which is never inferred.

Adjudications are appended to a Git-ignored line-delimited log, never rewritten in place, so a
session survives interruption and cannot silently overwrite an earlier decision. The automatic V3
state, crop, joint center, confidence, and reason codes are read-only here: a human decision is
recorded alongside them and never replaces them.

No participant identifier, accession, date, or storage path is rendered in the interface, and the
preview assets it serves already have their burned-in detector margins masked.
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from imaging.artifact_io import (
    LogTail,
    append_bytes,
    atomic_parquet,
    file_lock,
    quarantine_log_tail,
    read_jsonl_checked,
    serialize_jsonl_record,
    sha256_file,
)
from imaging.review_privacy import BURNED_IN_MARGIN_MM

DEFAULT_OUTPUT_ROOT = Path("data/processed/oai_images/v3_frozen_full")
DEFAULT_QUEUE_PATH = DEFAULT_OUTPUT_ROOT / "review_queue.parquet"
DEFAULT_ADJUDICATION_PATH = DEFAULT_OUTPUT_ROOT / "review/adjudications.jsonl"
DEFAULT_SNAPSHOT_PATH = DEFAULT_OUTPUT_ROOT / "review/adjudications.parquet"
ADJUDICATION_SCHEMA_VERSION = "v3_frozen_full_adjudication_v2"
PANEL_DECISIONS = ("ACCEPT", "REJECT", "NEEDS_CENTER_OVERRIDE")
LATERALITY_DECISIONS = ("CONFIRM_VALIDATED_MAPPING", "MARK_UNRESOLVED_EXCLUDE")
CONFIDENT_LATERALITY = "CONFIDENT"
PANEL_POSITIONS = ("screen_left", "screen_right")
# Frozen downstream crop specification, restated here only to size the review crop-box preview.
TARGET_SPACING_MM = 0.15
CROP_SIZE_MM = 160.0
ASSET_COLUMNS = {
    "bilateral": "bilateral_preview_path",
    "screen_left_overlay": "screen_left_overlay_path",
    "screen_right_overlay": "screen_right_overlay_path",
    "screen_left_crop": "screen_left_crop_preview_path",
    "screen_right_crop": "screen_right_crop_preview_path",
}
# Columns that carry protected linkage and must never reach the rendered interface.
PROTECTED_COLUMNS = (
    "participant_id",
    "source_participant_id",
    "accession_number",
    "associated_file_reference",
)
MINIMUM_PROTECTED_VALUE_LENGTH = 4
MAXIMUM_NOTE_LENGTH = 500
# Provenance label recorded for the final manifest, kept separate from the automatic V3 state.
_CROP_SOURCE_BY_DECISION = {
    "ACCEPT": "HUMAN_ACCEPT",
    "NEEDS_CENTER_OVERRIDE": "HUMAN_OVERRIDE",
    "REJECT": "REJECTED",
}
# Plain-language reminder of what each decision means. Deliberately about crop adequacy only.
DECISION_LEGEND = (
    ("ACCEPT", "Joint is centered with adequate femoral and tibial context."),
    ("OVERRIDE", "Joint is visible, but the proposed crop center is wrong."),
    ("REJECT", "A reliable joint crop cannot be obtained."),
)


class ReviewToolError(ValueError):
    """Raised when a review action would be unsafe, ambiguous, or destructive."""


class ReviewConflictError(ReviewToolError):
    """Raised when a writer's view of a case is stale relative to the committed log.

    A conflict is always deterministic: the writer is told its expected state no longer holds and
    nothing is appended, so two reviewers or two servers can never both commit a first decision.
    """


@dataclass(frozen=True, slots=True)
class PanelDecision:
    """One reviewer decision about one screen panel, with an optional corrected joint center.

    Coordinates are row and column in the resampled panel, which is the coordinate space the frozen
    crop step consumes, so a corrected center can be applied without re-running localization.
    """

    panel_position: str
    decision: str
    override_row: int | None = None
    override_column: int | None = None

    @property
    def has_override(self) -> bool:
        return self.override_row is not None and self.override_column is not None


class ReviewSession:
    """Deterministically ordered, resumable adjudication state for the V3 review queue."""

    def __init__(
        self,
        *,
        queue_path: Path = DEFAULT_QUEUE_PATH,
        adjudication_path: Path = DEFAULT_ADJUDICATION_PATH,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        if not Path(queue_path).is_file():
            raise ReviewToolError("The V3 review queue does not exist yet")
        self.queue_path = Path(queue_path)
        self.adjudication_path = Path(adjudication_path)
        self.output_root = Path(output_root).resolve()
        frame = pd.read_parquet(self.queue_path)
        if "review_index" not in frame.columns:
            raise ReviewToolError("The review queue lacks its deterministic review_index")
        if frame["review_index"].duplicated().any():
            raise ReviewToolError("The review queue contains duplicate review indices")
        self.queue = frame.sort_values("review_index", kind="stable", ignore_index=True)
        self.queue_fingerprint = sha256_file(self.queue_path)
        self.order = [int(value) for value in self.queue["review_index"]]
        self.laterality_exceptions = [
            int(value)
            for value in self.queue.loc[
                self.queue["laterality_state"].ne(CONFIDENT_LATERALITY), "review_index"
            ]
        ]
        self._write_lock = threading.Lock()

    # -- queue access -------------------------------------------------------

    def row(self, review_index: int) -> pd.Series:
        matches = self.queue.loc[self.queue["review_index"].eq(review_index)]
        if matches.empty:
            raise ReviewToolError("The requested review case is not in the queue")
        return matches.iloc[0]

    def requires_laterality_decision(self, review_index: int) -> bool:
        """Return whether the frozen policy left this acquisition's anatomical sides unresolved."""

        return int(review_index) in self.laterality_exceptions

    def asset_path(self, review_index: int, name: str) -> Path:
        """Resolve one allowlisted preview asset, refusing anything outside the dataset root."""

        if name not in ASSET_COLUMNS:
            raise ReviewToolError("Unknown review asset name")
        relative = self.row(review_index).get(ASSET_COLUMNS[name])
        if relative is None or pd.isna(relative):
            raise ReviewToolError("The requested review asset was not rendered for this case")
        resolved = (self.output_root / str(relative)).resolve()
        if not resolved.is_relative_to(self.output_root) or resolved.suffix != ".png":
            raise ReviewToolError("The requested review asset is outside the dataset root")
        if not resolved.is_file():
            raise ReviewToolError("The requested review asset is missing on disk")
        return resolved

    # -- adjudication state -------------------------------------------------

    def _validate_log(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for record in records:
            if record.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
                raise ReviewToolError("An adjudication has an incompatible schema version")
            if record.get("queue_fingerprint") != self.queue_fingerprint:
                raise ReviewToolError("Adjudications belong to a different review queue")
            if int(record.get("review_index", -1)) not in self.order:
                raise ReviewToolError("An adjudication refers to an unknown review case")
            for position in PANEL_POSITIONS:
                if record.get(f"{position}_decision") not in PANEL_DECISIONS:
                    raise ReviewToolError("An adjudication contains an invalid panel decision")
        return sorted(
            records,
            key=lambda record: (int(record["review_index"]), int(record.get("sequence", 0))),
        )

    def load(self) -> tuple[list[dict[str, Any]], LogTail | None]:
        """Read the committed log, reporting an interrupted final append instead of raising."""

        records, tail = read_jsonl_checked(self.adjudication_path)
        return self._validate_log(records), tail

    def adjudications(self) -> list[dict[str, Any]]:
        """Return every recorded decision in queue order, then in recording order."""

        return self.load()[0]

    def recovery_status(self) -> dict[str, Any]:
        """Describe any interrupted final append and every fragment already quarantined."""

        _, tail = self.load()
        sidecar = Path(f"{self.adjudication_path}.corrupt_tail")
        quarantined, _ = read_jsonl_checked(sidecar)
        return {
            "interrupted_final_append_pending": tail is not None,
            "pending_fragment_bytes": 0 if tail is None else tail.fragment_bytes,
            "quarantined_fragments": len(quarantined),
            "quarantine_relative_name": sidecar.name if quarantined else None,
        }

    def latest_decisions(self) -> dict[int, dict[str, Any]]:
        """Return the effective record per case: the highest sequence wins, earlier ones stay.

        Records are sorted by ``(review_index, sequence)``, so the last record seen for a case is
        its newest revision. A superseded record can therefore never be promoted back to effective.
        """

        latest: dict[int, dict[str, Any]] = {}
        for record in self.adjudications():
            latest[int(record["review_index"])] = record
        return latest

    def revision_history(self, review_index: int) -> list[dict[str, Any]]:
        """Return every recorded revision for one case, oldest first, including superseded ones."""

        return [
            record
            for record in self.adjudications()
            if int(record["review_index"]) == int(review_index)
        ]

    def next_pending(self, *, after: int = 0) -> int | None:
        """Return the first undecided case at or after ``after``, wrapping to the queue start."""

        decided = set(self.latest_decisions())
        forward = [value for value in self.order if value >= after and value not in decided]
        if forward:
            return forward[0]
        remaining = [value for value in self.order if value not in decided]
        return remaining[0] if remaining else None

    def progress(self) -> dict[str, Any]:
        """Report reviewed, remaining, and per-decision counts at acquisition and panel level."""

        latest = self.latest_decisions()
        panel_counts = {decision: 0 for decision in PANEL_DECISIONS}
        laterality_counts = {decision: 0 for decision in LATERALITY_DECISIONS}
        for record in latest.values():
            for position in PANEL_POSITIONS:
                panel_counts[str(record[f"{position}_decision"])] += 1
            choice = record.get("laterality_decision")
            if choice in laterality_counts:
                laterality_counts[str(choice)] += 1
        decided_exceptions = [index for index in self.laterality_exceptions if index in latest]
        return {
            "queued_acquisitions": len(self.order),
            "reviewed_acquisitions": len(latest),
            "remaining_acquisitions": len(self.order) - len(latest),
            "reviewed_panels": len(latest) * len(PANEL_POSITIONS),
            "revisions_recorded": len(self.adjudications()) - len(latest),
            "panel_decisions": panel_counts,
            "laterality_exceptions": {
                "total": len(self.laterality_exceptions),
                "reviewed": len(decided_exceptions),
                "remaining": len(self.laterality_exceptions) - len(decided_exceptions),
                "decisions": laterality_counts,
            },
            "log_recovery": self.recovery_status(),
            "queue_fingerprint": self.queue_fingerprint,
        }

    def _validate_panels(
        self, row: pd.Series, panels: tuple[PanelDecision, ...]
    ) -> dict[str, PanelDecision]:
        by_position: dict[str, PanelDecision] = {}
        for panel in panels:
            if panel.panel_position not in PANEL_POSITIONS:
                raise ReviewToolError("A decision names an unknown screen panel")
            if panel.panel_position in by_position:
                raise ReviewToolError("A screen panel received two decisions")
            if panel.decision not in PANEL_DECISIONS:
                raise ReviewToolError(
                    "Each panel decision must be ACCEPT, REJECT, or NEEDS_CENTER_OVERRIDE"
                )
            if panel.decision == "NEEDS_CENTER_OVERRIDE":
                if not panel.has_override:
                    raise ReviewToolError(
                        "A center override requires a corrected center for that panel"
                    )
                rows = int(row[f"{panel.panel_position}_resampled_rows"])
                columns = int(row[f"{panel.panel_position}_resampled_columns"])
                if not 0 <= int(panel.override_row) < rows:
                    raise ReviewToolError("A corrected center falls outside the resampled panel")
                if not 0 <= int(panel.override_column) < columns:
                    raise ReviewToolError("A corrected center falls outside the resampled panel")
            elif panel.has_override:
                raise ReviewToolError(
                    "A corrected center is only allowed with NEEDS_CENTER_OVERRIDE"
                )
            by_position[panel.panel_position] = panel
        missing = [position for position in PANEL_POSITIONS if position not in by_position]
        if missing:
            raise ReviewToolError("Both screen panels need a decision before the case is recorded")
        return by_position

    def record_adjudication(
        self,
        review_index: int,
        panels: tuple[PanelDecision, ...],
        *,
        laterality_decision: str | None = None,
        note: str = "",
        allow_revision: bool = False,
        expected_current_sequence: int | None = None,
    ) -> dict[str, Any]:
        """Append one adjudication inside a cross-process transaction.

        ``expected_current_sequence`` is the sequence the caller believed was effective for this
        case, or ``0`` for a case it believed was undecided. Supplying it turns a lost race into a
        deterministic conflict instead of an append that silently reinterprets someone else's work.

        The transaction acquires an advisory file lock, re-reads the authoritative log from disk,
        clears any interrupted final append, re-verifies the queue fingerprint and the expected
        supersession state, assigns the next sequence, appends durably, and then releases the lock.
        """

        row = self.row(review_index)
        if len(note) > MAXIMUM_NOTE_LENGTH:
            raise ReviewToolError("Reviewer note is too long")
        by_position = self._validate_panels(row, panels)
        if self.requires_laterality_decision(review_index):
            if laterality_decision not in LATERALITY_DECISIONS:
                raise ReviewToolError(
                    "This acquisition has unresolved laterality and needs an explicit "
                    "laterality decision"
                )
        elif laterality_decision is not None:
            raise ReviewToolError(
                "A laterality decision applies only to an acquisition with unresolved laterality"
            )

        # The in-memory lock still orders threads of this server cheaply; the file lock is what
        # makes the transaction safe against a second process sharing the same log.
        with self._write_lock, file_lock(self.adjudication_path):
            records, tail = read_jsonl_checked(self.adjudication_path)
            recovered_fragment_bytes = None
            if tail is not None:
                quarantine_log_tail(self.adjudication_path, tail)
                recovered_fragment_bytes = tail.fragment_bytes
                records, tail = read_jsonl_checked(self.adjudication_path)
            adjudications = self._validate_log(records)
            effective: dict[int, dict[str, Any]] = {}
            for item in adjudications:
                effective[int(item["review_index"])] = item
            existing = effective.get(int(review_index))

            if existing is None and allow_revision:
                raise ReviewConflictError(
                    "This case has no decision to revise; another writer may have removed it"
                )
            if existing is not None and not allow_revision:
                raise ReviewConflictError(
                    "This case already has a decision; a revision must be explicit"
                )
            if expected_current_sequence is not None:
                observed = 0 if existing is None else int(existing["sequence"])
                if int(expected_current_sequence) != observed:
                    raise ReviewConflictError(
                        "This case changed since it was displayed; reload before deciding"
                    )

            record: dict[str, Any] = {
                "schema_version": ADJUDICATION_SCHEMA_VERSION,
                "review_index": int(review_index),
                "acquisition_index": int(row["acquisition_index"]),
                "automatic_state": str(row["automatic_state"]),
                "laterality_state": str(row["laterality_state"]),
                "laterality_decision": laterality_decision,
                "reviewer_note": note,
                "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "queue_fingerprint": self.queue_fingerprint,
                "sequence": len(adjudications) + 1,
                "supersedes": None if existing is None else int(existing["sequence"]),
                "recovered_fragment_bytes": recovered_fragment_bytes,
            }
            for position in PANEL_POSITIONS:
                panel = by_position[position]
                record[f"{position}_decision"] = panel.decision
                record[f"{position}_automatic_state"] = str(row[f"{position}_qc_state"])
                record[f"{position}_automatic_center_row"] = int(
                    row[f"{position}_joint_row_resampled"]
                )
                record[f"{position}_automatic_center_column"] = int(
                    row[f"{position}_joint_column_resampled"]
                )
                record[f"{position}_center_override_row"] = (
                    None if panel.override_row is None else int(panel.override_row)
                )
                record[f"{position}_center_override_column"] = (
                    None if panel.override_column is None else int(panel.override_column)
                )
            # Serialize fully before touching the log so the append is a single prepared write.
            append_bytes(serialize_jsonl_record(record), self.adjudication_path)
        return record

    def effective_panel_status(self) -> dict[tuple[int, str], dict[str, Any]]:
        """Resolve the final status of every queued knee panel under an explicit precedence.

        Precedence, in order:

        1. The effective record for a case is its highest-sequence record. Every earlier record is
           superseded and is reported as history only.
        2. A panel's human decision comes only from that effective record.
        3. The automatic V3 state stays in its own field and is never replaced by a human decision.
        4. A laterality decision governs only whether an anatomical side may be assigned. It never
           changes the panel's crop verdict, so an unresolved side cannot silently reject a knee and
           a confirmed side cannot silently accept one.
        """

        effective = self.latest_decisions()
        statuses: dict[tuple[int, str], dict[str, Any]] = {}
        for review_index in self.order:
            row = self.row(review_index)
            record = effective.get(review_index)
            history = self.revision_history(review_index)
            superseded = [
                int(item["sequence"])
                for item in history
                if record is not None and int(item["sequence"]) != int(record["sequence"])
            ]
            requires_laterality = self.requires_laterality_decision(review_index)
            laterality_decision = None if record is None else record.get("laterality_decision")
            if requires_laterality:
                side_resolved = laterality_decision == "CONFIRM_VALIDATED_MAPPING"
            else:
                side_resolved = True
            for position in PANEL_POSITIONS:
                decision = None if record is None else str(record[f"{position}_decision"])
                statuses[(review_index, position)] = {
                    "review_index": review_index,
                    "acquisition_index": int(row["acquisition_index"]),
                    "panel_position": position,
                    "automatic_acquisition_state": str(row["automatic_state"]),
                    "automatic_panel_state": str(row[f"{position}_qc_state"]),
                    "human_panel_decision": decision,
                    "crop_source": _CROP_SOURCE_BY_DECISION.get(decision, "PENDING_REVIEW"),
                    "adjudication_sequence": None if record is None else int(record["sequence"]),
                    "superseded_sequences": superseded,
                    "revisions_recorded": max(0, len(history) - 1),
                    "requires_laterality_decision": requires_laterality,
                    "laterality_decision": laterality_decision,
                    "anatomical_side_resolved": side_resolved,
                    "side_assignment_blocked": not side_resolved,
                    "review_complete": record is not None,
                }
        return statuses

    def snapshot(self, *, target: Path = DEFAULT_SNAPSHOT_PATH) -> Path:
        """Write the current latest-decision view as a deterministic local table."""

        latest = self.latest_decisions()
        rows = [latest[index] for index in self.order if index in latest]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("review_index", kind="stable", ignore_index=True)
        return atomic_parquet(frame, target)

    def status_snapshot(self, *, target: Path) -> Path:
        """Write the resolved per-panel precedence table for later finalization."""

        statuses = self.effective_panel_status()
        frame = pd.DataFrame([statuses[key] for key in sorted(statuses)])
        if not frame.empty:
            frame["superseded_sequences"] = frame["superseded_sequences"].map(
                lambda values: "|".join(str(value) for value in values)
            )
        return atomic_parquet(frame, target)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #14161a;
       color: #e9edf2; }
header { padding: 10px 16px; background: #1d2027; display: flex; gap: 18px; align-items: baseline;
         position: sticky; top: 0; flex-wrap: wrap; z-index: 5; }
h1 { font-size: 16px; margin: 0; }
.state { font-weight: 700; }
.PASS { color: #5ddc7a; } .BORDERLINE { color: #ffd23f; } .FAIL { color: #ff6b6b; }
main { padding: 16px; display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 16px;
       align-items: start; }
/* Without min-width the image row sets the column's floor and pushes the decision controls off
   screen, so the reviewer could not reach the buttons on a normal display. */
.panels { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; min-width: 0; }
figure { margin: 0; position: relative; min-width: 0; max-width: 100%; }
figcaption { font-size: 12px; color: #a9b3c0; padding: 4px 0; }
img { max-width: 100%; max-height: 76vh; width: auto; border: 1px solid #333a45; display: block; }
img.overlay { cursor: crosshair; }
img.bilateral { cursor: default; }
.marker { position: absolute; border: 2px solid #ff5cf0; pointer-events: none; display: none; }
.marker-cross { position: absolute; width: 22px; height: 22px; pointer-events: none;
                display: none; }
.marker-cross::before, .marker-cross::after { content: ""; position: absolute;
    background: #ff5cf0; }
.marker-cross::before { left: 10px; top: 0; width: 2px; height: 22px; }
.marker-cross::after { top: 10px; left: 0; height: 2px; width: 22px; }
aside { background: #1d2027; padding: 12px; border-radius: 6px; font-size: 13px;
        position: sticky; top: 44px; max-height: calc(100vh - 60px); overflow-y: auto; }
ul { padding-left: 18px; margin: 6px 0; }
fieldset { border: 1px solid #3a4250; border-radius: 5px; margin: 10px 0; padding: 8px 10px; }
legend { font-size: 12px; color: #a9b3c0; }
label.choice { display: block; padding: 3px 0; cursor: pointer; }
button { font-size: 14px; padding: 9px 12px; margin: 4px 0; width: 100%; border-radius: 5px;
         border: 1px solid #3a4250; background: #262b33; color: #e9edf2; cursor: pointer; }
button.primary { background: #2f6b3f; }
button:disabled { opacity: 0.6; cursor: wait; }
.rapid-help, .save-status { font-size: 12px; color: #a9b3c0; margin: 6px 0; }
.save-error { color: #ff6b6b; }
textarea { width: 100%; box-sizing: border-box; background: #262b33; color: #e9edf2;
           border: 1px solid #3a4250; border-radius: 4px; }
.override-state { font-family: ui-monospace, monospace; font-size: 12px; color: #ff9ef4; }
.notice { color: #ffd23f; margin: 6px 0; }
.exception { border-left: 3px solid #ff6b6b; padding-left: 8px; }
dl.legend { margin: 10px 0; padding: 8px 10px; background: #22262e; border-radius: 5px;
            font-size: 12px; }
dl.legend dt { font-weight: 700; color: #cbd5e2; }
dl.legend dd { margin: 0 0 6px 0; color: #a9b3c0; }
nav a { color: #8fb8ff; margin-right: 12px; }
table { border-collapse: collapse; font-size: 13px; }
td, th { border: 1px solid #3a4250; padding: 4px 8px; text-align: left; }
"""

_SCRIPT = """
const overrides = {};
let saving = false;
function panelInput(panel, decision) {
  return document.querySelector('input[name="' + panel + '_decision"][value="' + decision + '"]');
}
function markCenter(event) {
  if (saving) { return; }
  const image = event.currentTarget;
  const panel = image.dataset.panel;
  const scale = parseFloat(image.dataset.scale);
  const header = parseFloat(image.dataset.header);
  const rect = image.getBoundingClientRect();
  // clientWidth and clientLeft exclude the border, so the click maps to rendered pixels exactly.
  const factor = image.naturalWidth / image.clientWidth;
  const x = (event.clientX - rect.left - image.clientLeft) * factor;
  const y = (event.clientY - rect.top - image.clientTop) * factor;
  const column = Math.round(x / scale);
  const row = Math.round((y - header) / scale);
  if (row < 0) { return; }
  overrides[panel] = { row: row, column: column };
  const choice = panelInput(panel, 'NEEDS_CENTER_OVERRIDE');
  if (choice) { choice.checked = true; }
  render();
}
function drawMarker(panel) {
  const image = document.querySelector('img.overlay[data-panel="' + panel + '"]');
  const box = document.getElementById('marker-' + panel);
  const cross = document.getElementById('cross-' + panel);
  if (!image || !box || !cross) { return; }
  const chosen = overrides[panel];
  if (!chosen) { box.style.display = 'none'; cross.style.display = 'none'; return; }
  const scale = parseFloat(image.dataset.scale);
  const header = parseFloat(image.dataset.header);
  const cropPixels = parseFloat(image.dataset.crop);
  const display = image.clientWidth / image.naturalWidth;
  const x = chosen.column * scale * display + image.clientLeft;
  const y = (chosen.row * scale + header) * display + image.clientTop;
  const side = cropPixels * scale * display;
  box.style.left = (x - side / 2) + 'px';
  box.style.top = (y - side / 2) + 'px';
  box.style.width = side + 'px';
  box.style.height = side + 'px';
  box.style.display = 'block';
  cross.style.left = (x - 11) + 'px';
  cross.style.top = (y - 11) + 'px';
  cross.style.display = 'block';
}
function render() {
  const parts = Object.keys(overrides).sort().map(function (panel) {
    return panel.replace('_', '-') + ' row=' + overrides[panel].row +
           ' col=' + overrides[panel].column;
  });
  document.getElementById('override-state').textContent =
    parts.length ? parts.join('  |  ') : 'no corrected center selected';
  document.getElementById('overrides').value = JSON.stringify(overrides);
  ['screen_left', 'screen_right'].forEach(drawMarker);
}
function clearOverride(panel) {
  if (saving) { return; }
  delete overrides[panel];
  const choice = panelInput(panel, 'NEEDS_CENTER_OVERRIDE');
  if (choice && choice.checked) { choice.checked = false; }
  render();
}
function validateSubmit() {
  const missing = [];
  ['screen_left', 'screen_right'].forEach(function (panel) {
    const chosen = document.querySelector('input[name="' + panel + '_decision"]:checked');
    if (!chosen) { missing.push(panel.replace('_', '-')); }
    else if (chosen.value === 'NEEDS_CENTER_OVERRIDE' && !overrides[panel]) {
      missing.push(panel.replace('_', '-') + ' corrected center');
    }
  });
  const laterality = document.getElementById('laterality-required');
  if (laterality && laterality.value === '1' &&
      !document.querySelector('input[name="laterality_decision"]:checked')) {
    missing.push('laterality decision');
  }
  if (missing.length) {
    document.getElementById('override-state').textContent = 'still needed: ' + missing.join(', ');
    return false;
  }
  return true;
}
async function submitDecision(event) {
  event.preventDefault();
  if (saving || !validateSubmit()) { return; }
  const form = event.currentTarget;
  // Serialize before disabling controls. Every write still uses the existing /decision action,
  // including its expected sequence and explicit-revision fields; there is no keyboard endpoint.
  const payload = new URLSearchParams(new FormData(form));
  const controls = Array.from(form.elements);
  const disabled = controls.map(function (control) { return control.disabled; });
  const status = document.getElementById('save-status');
  const errorState = document.getElementById('save-error');
  saving = true;
  status.textContent = 'Saving...';
  errorState.textContent = '';
  controls.forEach(function (control) { control.disabled = true; });
  try {
    const response = await fetch(form.action, {
      method: form.method, body: payload, redirect: 'follow'
    });
    // The existing server redirects only after record_adjudication has committed successfully.
    // A conflict/error stays on this page, keeping its selections, center clicks and note.
    if (!response.ok || !response.redirected) {
      const documentText = await response.text();
      const errorDocument = new DOMParser().parseFromString(documentText, 'text/html');
      const message = errorDocument.querySelector('p');
      throw new Error(message ? message.textContent : 'The decision could not be saved.');
    }
    const destination = new URL(response.url);
    if (destination.origin !== window.location.origin ||
        !(/^\\/case\\/\\d+$/.test(destination.pathname) || destination.pathname === '/')) {
      throw new Error('The save response did not identify the next review case.');
    }
    // Keep writes disabled until navigation completes, even for a second fast keypress.
    window.location.assign(destination.pathname);
  } catch (error) {
    errorState.textContent = error.message || 'The decision could not be saved.';
    saving = false;
    status.textContent = '';
    controls.forEach(function (control, index) { control.disabled = disabled[index]; });
    errorState.scrollIntoView({block: 'nearest'});
  }
}
function clearUnsaved() {
  if (saving) { return; }
  // Reset to the HTML defaults: persisted decisions remain selected on a revisited page.
  document.getElementById('form').reset();
  Object.keys(overrides).forEach(function (panel) { delete overrides[panel]; });
  document.getElementById('save-error').textContent = '';
  render();
}
function rapidKey(event) {
  const focused = document.activeElement;
  const editable = function (element) {
    return element && (element.isContentEditable ||
      element.closest('textarea, input, select, [contenteditable]'));
  };
  if (saving || event.repeat || event.defaultPrevented || event.isComposing ||
      event.ctrlKey || event.altKey || event.metaKey ||
      editable(event.target) || editable(focused)) { return; }
  const key = event.key.toLowerCase();
  if (!['a', 'enter', 'n', 'p', 'escape'].includes(key)) { return; }
  event.preventDefault();
  const form = document.getElementById('form');
  if (key === 'a') {
    if (Object.keys(overrides).length ||
        document.querySelector('input[value="NEEDS_CENTER_OVERRIDE"]:checked')) {
      document.getElementById('override-state').textContent =
        'Clear the corrected-center selection or use the explicit save workflow.';
      return;
    }
    if (form.elements.expected_current_sequence.value !== '0') {
      document.getElementById('override-state').textContent =
        'This case is already recorded; use the explicit controls to revise it.';
      return;
    }
    if (document.getElementById('laterality-required').value === '1' &&
        !document.querySelector('input[name="laterality_decision"]:checked')) {
      document.getElementById('override-state').textContent = 'still needed: laterality decision';
      return;
    }
    ['screen_left', 'screen_right'].forEach(function (panel) {
      panelInput(panel, 'ACCEPT').checked = true;
    });
    render();
    form.requestSubmit();
  } else if (key === 'enter') {
    form.requestSubmit();
  } else if (key === 'n') {
    window.location.assign(form.dataset.nextUndecided);
  } else if (key === 'p') {
    const previous = document.getElementById('previous-case');
    if (previous) { window.location.assign(previous.getAttribute('href')); }
  } else if (key === 'escape') {
    clearUnsaved();
  }
}
window.addEventListener('load', function () {
  document.querySelectorAll('img.overlay').forEach(function (image) {
    image.addEventListener('click', markCenter);
  });
  window.addEventListener('resize', render);
  document.getElementById('form').addEventListener('submit', submitDecision);
  document.addEventListener('keydown', rapidKey);
  render();
});
"""


def _figure(source: str, caption: str, *, attributes: str = "", css_class: str = "") -> str:
    classes = f' class="{css_class}"' if css_class else ""
    return (
        f'<figure><img src="{html.escape(source)}"{classes} {attributes} alt="">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _overlay_figure(review_index: int, position: str, row: pd.Series) -> str:
    """Render one clickable overlay with the marker elements the crop-box preview needs."""

    scale = float(row.get(f"{position}_overlay_scale") or 0.0)
    header = float(row.get(f"{position}_overlay_header_pixels") or 0.0)
    crop_pixels = CROP_SIZE_MM / TARGET_SPACING_MM
    label = position.replace("_", "-")
    return (
        "<figure>"
        f'<img src="/asset?case={review_index}&name={position}_overlay" class="overlay" '
        f'data-panel="{position}" data-scale="{scale}" data-header="{header}" '
        f'data-crop="{crop_pixels}" alt="">'
        f'<div class="marker" id="marker-{position}"></div>'
        f'<div class="marker-cross" id="cross-{position}"></div>'
        f"<figcaption>{html.escape(label)} overlay - click to set a corrected center"
        "</figcaption></figure>"
    )


def _reason_items(codes: str) -> str:
    items = [code for code in str(codes or "").split("|") if code]
    if not items:
        return "<li>none recorded</li>"
    return "".join(f"<li>{html.escape(code)}</li>" for code in items)


def _candidate_items(text: str) -> str:
    try:
        candidates = json.loads(text) if text else []
    except json.JSONDecodeError:
        candidates = []
    if not candidates:
        return "<li>none recorded</li>"
    return "".join(
        "<li>rank {rank}: row {row}, column {column} (score {score:.4f})</li>".format(
            rank=rank, row=item["row"], column=item["column"], score=float(item["score"])
        )
        for rank, item in enumerate(candidates, start=1)
    )


def _panel_fieldset(position: str, row: pd.Series, existing: dict[str, Any] | None) -> str:
    """Render the three independent decision choices for one screen panel."""

    label = position.replace("_", "-")
    chosen = None if existing is None else str(existing.get(f"{position}_decision") or "")
    choices = ""
    for decision in PANEL_DECISIONS:
        checked = " checked" if decision == chosen else ""
        choices += (
            f'<label class="choice"><input type="radio" name="{position}_decision" '
            f'value="{decision}"{checked}> {decision}</label>'
        )
    side = row.get(f"{position}_anatomical_side")
    side_text = "unresolved" if side is None or pd.isna(side) else str(side)
    return (
        f"<fieldset><legend>{html.escape(label)} &mdash; automatic "
        f"{html.escape(str(row[f'{position}_qc_state']))}, anatomical side "
        f"{html.escape(side_text)}</legend>{choices}"
        f'<button type="button" onclick="clearOverride(\'{position}\')">'
        f"clear {html.escape(label)} corrected center</button></fieldset>"
    )


def _laterality_fieldset(row: pd.Series, existing: dict[str, Any] | None) -> str:
    """Render the explicit laterality section required for an unresolved acquisition."""

    chosen = None if existing is None else str(existing.get("laterality_decision") or "")
    choices = ""
    for decision in LATERALITY_DECISIONS:
        checked = " checked" if decision == chosen else ""
        choices += (
            f'<label class="choice"><input type="radio" name="laterality_decision" '
            f'value="{decision}"{checked}> {decision}</label>'
        )
    return (
        '<fieldset class="exception"><legend>Laterality exception &mdash; anatomical side was '
        "not resolved automatically</legend>"
        f"<p>Frozen policy state {html.escape(str(row['laterality_state']))}: "
        f"{html.escape(str(row['laterality_reason']))}.</p>"
        "<p>Structure evidence: "
        f"{html.escape(str(row['structure_check_reasons']) or 'none recorded')}.</p>"
        "<p>Confirm the validated screen mapping (screen-left is the anatomical right knee, "
        "screen-right is the anatomical left knee) only if the acquisition structure supports it. "
        "Otherwise mark it unresolved so neither knee is joined to a side-specific cohort row.</p>"
        f"{choices}</fieldset>"
    )


def _assert_anonymous(document: str, row: pd.Series) -> None:
    """Fail closed if a rendered page would contain protected linkage from the queue row."""

    for column in PROTECTED_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value)
        if len(text) >= MINIMUM_PROTECTED_VALUE_LENGTH and text in document:
            raise ReviewToolError("A rendered review page would expose protected linkage")


def render_case(session: ReviewSession, review_index: int) -> str:
    """Render one anonymous review case as a self-contained HTML page."""

    row = session.row(review_index)
    progress = session.progress()
    existing = session.latest_decisions().get(int(review_index))
    position_in_queue = session.order.index(int(review_index)) + 1
    state = str(row["automatic_state"])
    needs_laterality = session.requires_laterality_decision(review_index)

    panels_html = _figure(
        f"/asset?case={review_index}&name=bilateral",
        "anonymous bilateral acquisition with both proposed 160 mm crops",
        css_class="bilateral",
    )
    for position in PANEL_POSITIONS:
        overlay_scale = row.get(f"{position}_overlay_scale")
        if overlay_scale is not None and not pd.isna(overlay_scale):
            panels_html += _overlay_figure(review_index, position, row)
        panels_html += _figure(
            f"/asset?case={review_index}&name={position}_crop",
            f"{position.replace('_', '-')} proposed crop"
            f" ({html.escape(str(row[f'{position}_qc_state']))})",
        )

    panel_facts = "".join(
        "<li>{position}: automatic {qc}, layout {layout}, anatomy {anatomy}, "
        "confidence {confidence:.3f}, automatic center row {row}, column {column}, "
        "padding {horizontal:.2f} mm horizontal / {vertical:.2f} mm vertical</li>".format(
            position=html.escape(position.replace("_", "-")),
            qc=html.escape(str(row[f"{position}_qc_state"])),
            layout=html.escape(str(row[f"{position}_layout_state"])),
            anatomy=html.escape(str(row[f"{position}_anatomical_validator_state"])),
            confidence=float(row[f"{position}_localization_confidence"]),
            row=int(row[f"{position}_joint_row_resampled"]),
            column=int(row[f"{position}_joint_column_resampled"]),
            horizontal=float(row[f"{position}_horizontal_padding_mm"]),
            vertical=float(row[f"{position}_vertical_padding_mm"]),
        )
        for position in PANEL_POSITIONS
    )
    candidates = "".join(
        f"<li>{html.escape(position.replace('_', '-'))}<ul>"
        f"{_candidate_items(str(row.get(f'{position}_candidate_centers') or ''))}</ul></li>"
        for position in PANEL_POSITIONS
    )
    revision_notice = ""
    if existing is not None:
        summary = ", ".join(
            f"{position.replace('_', '-')} {existing[f'{position}_decision']}"
            for position in PANEL_POSITIONS
        )
        revision_notice = (
            f'<p class="notice">Already adjudicated ({html.escape(summary)}) as revision '
            f"{int(existing['sequence'])}. Submitting again records an explicit revision; the "
            "original entry is retained.</p>"
        )
    legend = "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(text)}</dd>"
        for name, text in DECISION_LEGEND
    )
    current_sequence = 0 if existing is None else int(existing["sequence"])
    previous_index = session.order[position_in_queue - 2] if position_in_queue > 1 else None
    next_index = (
        session.order[position_in_queue] if position_in_queue < len(session.order) else None
    )
    navigation = ""
    if previous_index is not None:
        navigation += f'<a id="previous-case" href="/case/{previous_index}">previous</a>'
    if next_index is not None:
        navigation += f'<a href="/case/{next_index}">next</a>'
    navigation += '<a href="/">next undecided</a><a href="/exceptions">laterality exceptions</a>'

    counts = progress["panel_decisions"]
    next_pending = session.next_pending(after=review_index + 1)
    rapid_next = "/" if next_pending is None else f"/case/{next_pending}"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>V3 crop review {review_index}</title>
<style>{_STYLE}</style><script>{_SCRIPT}</script></head>
<body>
<header>
  <h1>V3 crop review</h1>
  <span>case {position_in_queue} of {len(session.order)}</span>
  <span>automatic <span class="state {html.escape(state)}">{html.escape(state)}</span></span>
  <span>{progress["reviewed_acquisitions"]} reviewed,
        {progress["remaining_acquisitions"]} remaining</span>
  <span>panels: {counts["ACCEPT"]} accept, {counts["REJECT"]} reject,
        {counts["NEEDS_CENTER_OVERRIDE"]} override</span>
  <span id="save-status" class="save-status" role="status" aria-live="polite"></span>
  <nav>{navigation}</nav>
</header>
<main>
  <section class="panels">{panels_html}</section>
  <aside>
    <p>Anonymous case label {int(row["acquisition_index"]):04d}. No participant identifier,
       accession, date, or file path is shown.</p>
    <p class="notice">The top and bottom {BURNED_IN_MARGIN_MM:g} mm of every image are masked
       because some acquisitions carry burned-in text there. A masked band is expected and is not
       evidence of a truncated image. Masking affects previews only; stored uint16 crops retain
       their original pixels.</p>
    <p><strong>Laterality</strong>: {html.escape(str(row["laterality_state"]))}
       ({html.escape(str(row["laterality_reason"]))})</p>
    <p><strong>Automatic reason codes</strong></p>
    <ul>{_reason_items(str(row["review_reason_codes"]))}</ul>
    <p><strong>Panel evidence</strong></p>
    <ul>{panel_facts}</ul>
    <p><strong>Frozen localizer candidate centers</strong></p>
    <ul>{candidates}</ul>
    {revision_notice}
    <dl class="legend">{legend}</dl>
    <p class="rapid-help">Rapid review: A = accept both + save + next;<br>
       Enter = save selections + next; N = next undecided;<br>
       P = previous; Esc = clear unsaved selections.<br>
       Shortcuts pause while typing or saving.</p>
    <form id="form" method="post" action="/decision" data-next-undecided="{rapid_next}">
      <input type="hidden" name="review_index" value="{review_index}">
      <input type="hidden" name="overrides" id="overrides" value="{{}}">
      <input type="hidden" name="allow_revision"
             value="{"1" if existing is not None else "0"}">
      <input type="hidden" name="expected_current_sequence" value="{current_sequence}">
      <input type="hidden" id="laterality-required" value="{"1" if needs_laterality else "0"}">
      {_panel_fieldset("screen_left", row, existing)}
      {_panel_fieldset("screen_right", row, existing)}
      {_laterality_fieldset(row, existing) if needs_laterality else ""}
      <p class="override-state" id="override-state"></p>
      <label for="note">Reviewer note (optional)</label>
      <textarea id="note" name="note" rows="2"
                maxlength="{MAXIMUM_NOTE_LENGTH}"></textarea>
      <button type="submit" class="primary">record decision for both panels</button>
      <p id="save-error" class="save-error" role="alert"></p>
    </form>
  </aside>
</main>
</body></html>
"""
    _assert_anonymous(document, row)
    return document


def render_exception_index(session: ReviewSession) -> str:
    """Render the separate laterality-exception section as its own worklist."""

    latest = session.latest_decisions()
    rows = ""
    for index in session.laterality_exceptions:
        row = session.row(index)
        record = latest.get(index)
        decision = "not yet reviewed" if record is None else str(record["laterality_decision"])
        rows += (
            f'<tr><td><a href="/case/{index}">case {index}</a></td>'
            f"<td>{html.escape(str(row['automatic_state']))}</td>"
            f"<td>{html.escape(str(row['laterality_state']))}</td>"
            f"<td>{html.escape(str(row['structure_check_reasons']) or 'none recorded')}</td>"
            f"<td>{html.escape(decision)}</td></tr>"
        )
    if not rows:
        rows = '<tr><td colspan="5">No laterality exceptions in this queue.</td></tr>'
    progress = session.progress()["laterality_exceptions"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V3 laterality exceptions</title>
<style>{_STYLE}</style></head><body>
<header><h1>Laterality exceptions</h1>
  <span>{progress["reviewed"]} of {progress["total"]} reviewed</span>
  <nav><a href="/">back to queue</a></nav>
</header>
<main><aside>
  <p>The frozen Laterality V2 policy did not resolve anatomical sides for these acquisitions.
     Laterality is never forced: each needs an explicit structural review. Only a confirmed
     mapping allows the knees to join side-specific cohort rows.</p>
  <table><tr><th>case</th><th>automatic state</th><th>laterality</th>
    <th>structure evidence</th><th>decision</th></tr>{rows}</table>
</aside></main></body></html>
"""
    for index in session.laterality_exceptions:
        _assert_anonymous(document, session.row(index))
    return document


def _parse_panel_decisions(form: dict[str, list[str]]) -> tuple[PanelDecision, ...]:
    """Build panel decisions from the submitted form, pairing overrides with their panel."""

    payload = form.get("overrides", ["{}"])[0]
    try:
        parsed = json.loads(payload) if payload.strip() else {}
    except json.JSONDecodeError as error:
        raise ReviewToolError("Corrected centers were not submitted as valid JSON") from error
    if not isinstance(parsed, dict):
        raise ReviewToolError("Corrected centers must be keyed by screen panel")

    decisions: list[PanelDecision] = []
    for position in PANEL_POSITIONS:
        values = form.get(f"{position}_decision", [])
        if not values:
            raise ReviewToolError("Both screen panels need a decision before the case is recorded")
        decision = values[0]
        override = parsed.get(position)
        row: int | None = None
        column: int | None = None
        if decision == "NEEDS_CENTER_OVERRIDE":
            if not isinstance(override, dict):
                raise ReviewToolError(
                    "A center override requires a corrected center for that panel"
                )
            try:
                row = int(override["row"])
                column = int(override["column"])
            except (KeyError, TypeError, ValueError) as error:
                raise ReviewToolError(
                    "A corrected center is missing integer coordinates"
                ) from error
        decisions.append(
            PanelDecision(
                panel_position=position,
                decision=decision,
                override_row=row,
                override_column=column,
            )
        )
    return tuple(decisions)


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Loopback-only request handler bound to a single :class:`ReviewSession`."""

    session: ReviewSession
    server_version = "OAIV3Review/2.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _error(self, status: HTTPStatus, message: str) -> None:
        body = f'<!doctype html><p>{html.escape(message)}</p><p><a href="/">back</a></p>'
        self._send(status, body.encode("utf-8"), "text/html; charset=utf-8")

    def _conflict(self, review_index: int, message: str) -> None:
        """Report a lost race explicitly, so nothing is discarded without the reviewer knowing."""

        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            f"<title>Review conflict</title><style>{_STYLE}</style></head><body>"
            "<header><h1>Nothing was recorded</h1></header><main><aside>"
            f'<p class="notice">{html.escape(message)}</p>'
            "<p>Another reviewer or another review server committed a decision for this case "
            "first. Your submission was not written to the log and no previous decision was "
            "changed.</p>"
            f'<p><a href="/case/{int(review_index)}">reload this case</a> '
            '<a href="/">next undecided</a></p>'
            "</aside></main></body></html>"
        )
        self._send(HTTPStatus.CONFLICT, body.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                pending = self.session.next_pending()
                if pending is None:
                    self._send(
                        HTTPStatus.OK,
                        self._completion_page().encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                self._redirect(f"/case/{pending}")
                return
            if parsed.path == "/progress":
                body = json.dumps(self.session.progress(), indent=2, sort_keys=True)
                self._send(HTTPStatus.OK, body.encode("utf-8"), "application/json")
                return
            if parsed.path == "/exceptions":
                body = render_exception_index(self.session)
                self._send(HTTPStatus.OK, body.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/case/"):
                review_index = int(parsed.path.removeprefix("/case/"))
                body = render_case(self.session, review_index)
                self._send(HTTPStatus.OK, body.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/asset":
                review_index = int(query["case"][0])
                path = self.session.asset_path(review_index, query["name"][0])
                self._send(HTTPStatus.OK, path.read_bytes(), "image/png")
                return
        except ReviewToolError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
            return
        except (KeyError, IndexError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "The request was not understood")
            return
        self._error(HTTPStatus.NOT_FOUND, "Unknown review route")

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if urlparse(self.path).path != "/decision":
            self._error(HTTPStatus.NOT_FOUND, "Unknown review route")
            return
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        try:
            review_index = int(form["review_index"][0])
            laterality = form.get("laterality_decision", [None])[0]
            expected = form.get("expected_current_sequence", [None])[0]
            self.session.record_adjudication(
                review_index,
                _parse_panel_decisions(form),
                laterality_decision=laterality,
                note=form.get("note", [""])[0],
                allow_revision=form.get("allow_revision", ["0"])[0] == "1",
                expected_current_sequence=None if expected is None else int(expected),
            )
        except ReviewConflictError as error:
            self._conflict(review_index, str(error))
            return
        except ReviewToolError as error:
            self._error(HTTPStatus.CONFLICT, str(error))
            return
        except (KeyError, IndexError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "The decision was not understood")
            return
        pending = self.session.next_pending(after=review_index + 1)
        self._redirect("/" if pending is None else f"/case/{pending}")

    def _completion_page(self) -> str:
        progress = self.session.progress()
        return (
            '<!doctype html><html><head><meta charset="utf-8">'
            f"<title>V3 crop review</title><style>{_STYLE}</style></head><body>"
            "<header><h1>V3 crop review</h1>"
            '<nav><a href="/exceptions">laterality exceptions</a></nav></header>'
            "<main><aside>"
            f"<p>All {progress['queued_acquisitions']} queued acquisitions have a decision.</p>"
            f"<pre>{html.escape(json.dumps(progress, indent=2, sort_keys=True))}</pre>"
            "</aside></main></body></html>"
        )


def serve(
    *,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    adjudication_path: Path = DEFAULT_ADJUDICATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:  # pragma: no cover - interactive entry point
    """Serve the review tool on the loopback interface until interrupted."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ReviewToolError("The review server may bind only to a loopback address")
    session = ReviewSession(
        queue_path=queue_path,
        adjudication_path=adjudication_path,
        output_root=output_root,
    )
    handler = type("BoundReviewHandler", (ReviewRequestHandler,), {"session": session})
    with ThreadingHTTPServer((host, port), handler) as server:
        address = f"http://{host}:{server.server_port}/"
        print(json.dumps({"review_url": address, **session.progress()}, sort_keys=True), flush=True)
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(json.dumps(session.progress(), sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review frozen V3 knee crops locally and record per-panel decisions."
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--adjudications", type=Path, default=DEFAULT_ADJUDICATION_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Print adjudication progress and exit without starting the server.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Write the latest-decision table and exit without starting the server.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_only or args.snapshot:
        session = ReviewSession(
            queue_path=args.queue,
            adjudication_path=args.adjudications,
            output_root=args.dataset_root,
        )
        if args.snapshot:
            session.snapshot(target=args.dataset_root / "review/adjudications.parquet")
            session.status_snapshot(
                target=args.dataset_root / "review/effective_panel_status.parquet"
            )
        print(json.dumps(session.progress(), indent=2, sort_keys=True))
        return 0
    serve(  # pragma: no cover - interactive
        queue_path=args.queue,
        adjudication_path=args.adjudications,
        output_root=args.dataset_root,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
