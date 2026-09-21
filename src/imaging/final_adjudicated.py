"""Fail-closed Milestone 5J resolution of the locked analysis cohort's imaging.

Automatic artifacts and adjudications are read-only. Corrected pixels are delegated to the existing
frozen override pipeline, without force. This module validates authoritative inputs before that
pipeline writes anything, then verifies every derivative and every cohort row before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imaging.artifact_io import atomic_json, atomic_parquet, hash_tree, sha256_file
from imaging.frozen_full_v3 import (
    FROZEN_SOURCE_FILES,
    LATERALITY_POLICY_VERSION,
    LOCALIZER_VERSION,
    PREPROCESSING_VERSION,
    V3_FREEZE_RECORDS,
)
from imaging.override_crops import (
    CROP_SIZE_MM,
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WORK_ROOT,
    OUTPUT_SHAPE,
    OVERRIDE_CROP_VERSION,
    PADDING_PERCENTILE,
    TARGET_SPACING_MM,
    _authoritative_localizer_hash,
    pending_overrides,
    regenerate_override_crops,
    verify_existing_override,
)
from imaging.v3_review_ui import ADJUDICATION_SCHEMA_VERSION, PANEL_POSITIONS, ReviewSession

FINAL_MANIFEST_VERSION = "final_adjudicated_imaging_v1"
DEFAULT_COHORT = Path("data/processed/cohorts/analysis_cohort_v1.parquet")
FINAL_DIRECTORY = "final_adjudicated_v1"
PROVENANCES = ("AUTO_PASS", "HUMAN_ACCEPT", "HUMAN_OVERRIDE", "REJECTED", "UNRESOLVED")
SIDE_BY_PANEL = {"screen_left": ("R", "1"), "screen_right": ("L", "2")}
EXPECTED_REVIEW = {
    "queued_acquisitions": 978,
    "reviewed_acquisitions": 978,
    "remaining_acquisitions": 0,
    "reviewed_panels": 1956,
    "revisions_recorded": 8,
    "panel_decisions": {"ACCEPT": 1534, "REJECT": 0, "NEEDS_CENTER_OVERRIDE": 422},
    "laterality_exceptions": {
        "total": 3,
        "reviewed": 3,
        "remaining": 0,
        "decisions": {"CONFIRM_VALIDATED_MAPPING": 3, "MARK_UNRESOLVED_EXCLUDE": 0},
    },
}


class FinalImagingError(ValueError):
    """An input or effective artifact cannot be proven authoritative; nothing is silently repaired."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalImagingError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise FinalImagingError(f"Required JSON artifact cannot be read: {path.name}") from error
    _require(isinstance(value, dict), f"Unexpected JSON structure: {path.name}")
    return value


def _clean(value: Any) -> Any:
    return None if pd.isna(value) else value


def _same(left: Any, right: Any) -> bool:
    return _clean(left) == _clean(right)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _aggregate(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("artifact").itertuples(index=False):
        digest.update(row.artifact.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(row.sha256))
    return digest.hexdigest()


def _artifact(root: Path, reference: str) -> Path:
    reference_path = Path(reference)
    resolved = (root / reference_path).resolve()
    _require(
        not reference_path.is_absolute()
        and ".." not in reference_path.parts
        and resolved.is_relative_to(root.resolve()),
        "An artifact reference escapes its authoritative root",
    )
    _require(resolved.is_file(), "A required artifact file is missing")
    return resolved


def _verify_file_ledger(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"Authoritative preservation ledger missing: {path.name}")
    frame = pd.read_parquet(path)
    _require(not frame.artifact.duplicated().any(), "Duplicate artifact in preservation ledger")
    result = {}
    for row in frame.itertuples(index=False):
        artifact = Path(row.artifact)
        _require(artifact.is_file(), "A frozen methodology artifact is missing")
        _require(
            artifact.stat().st_size == row.bytes and sha256_file(artifact) == row.sha256,
            "Frozen methodology/source SHA mismatch",
        )
        result[row.artifact] = row.sha256
    return result


def verify_automatic_tree(root: Path) -> pd.DataFrame:
    """Verify, never create or update, the pre-human-review crop/record ledger."""
    path = root / "audit/automatic_crop_ledger.parquet"
    _require(path.is_file(), "Authoritative automatic crop ledger is missing")
    previous = pd.read_parquet(path).sort_values("artifact", ignore_index=True)
    current = hash_tree(root / "crops", relative_to=root, workers=8).sort_values(
        "artifact", ignore_index=True
    )
    _require(not previous.artifact.duplicated().any(), "Duplicate automatic ledger artifact")
    _require(
        previous.artifact.tolist() == current.artifact.tolist()
        and previous.sha256.tolist() == current.sha256.tolist()
        and previous.bytes.tolist() == current.bytes.tolist(),
        "Automatic crop or processing-record SHA mismatch/mutation",
    )
    return current


def _methodology(root: Path) -> dict[str, str]:
    approved_hash = _authoritative_localizer_hash()
    sources = _verify_file_ledger(root / "audit/frozen_source_ledger.parquet")
    freezes = _verify_file_ledger(root / "audit/v3_freeze_record_ledger.parquet")
    _require(
        set(sources) == {str(path) for path in FROZEN_SOURCE_FILES}
        and set(freezes) == {str(path) for path in V3_FREEZE_RECORDS},
        "Frozen methodology ledger has unexpected/missing members",
    )
    _require(sources[str(FROZEN_SOURCE_FILES[0])] == approved_hash, "Localizer approval mismatch")
    policy = _json(Path(V3_FREEZE_RECORDS[2]))
    _require(
        policy.get("policy_version") == LATERALITY_POLICY_VERSION
        and policy.get("status") == "APPROVED_FROZEN"
        and policy.get("mapping")
        == {"screen_left": "anatomical_RIGHT", "screen_right": "anatomical_LEFT"}
        and policy.get("dicom_unilateral_laterality_authoritative") is False,
        "Frozen laterality policy/mapping mismatch",
    )
    return {
        "localizer_source_sha256": approved_hash,
        "laterality_source_sha256": sources[str(FROZEN_SOURCE_FILES[1])],
        "preprocessing_source_sha256": sources[str(FROZEN_SOURCE_FILES[3])],
        "localizer_freeze_record_sha256": freezes[str(V3_FREEZE_RECORDS[0])],
        "laterality_freeze_record_sha256": freezes[str(V3_FREEZE_RECORDS[2])],
        "override_implementation_sha256": sha256_file(Path("src/imaging/override_crops.py")),
    }


def _validate_history(session: ReviewSession, records: dict[int, dict]) -> None:
    """Validate the whole revision chain and all identity/automatic snapshots without writing."""
    history = sorted(session.adjudications(), key=lambda item: item["sequence"])
    _require(
        [item["sequence"] for item in history] == list(range(1, len(history) + 1)),
        "Adjudication sequences are duplicated, missing or reordered",
    )
    predecessors: dict[int, int] = {}
    for decision in history:
        index = int(decision["review_index"])
        queue = session.row(index)
        acquisition = int(queue.acquisition_index)
        automatic = records[acquisition]
        _require(decision.get("supersedes") == predecessors.get(index), "Invalid revision chain")
        predecessors[index] = int(decision["sequence"])
        # The frozen adjudication schema stores acquisition_index and queue fingerprint, not
        # participant/accession fields. Those identities are proven through the pinned queue.
        for field in ("acquisition_index",):
            _require(
                _same(decision.get(field), automatic[field]),
                "Adjudication acquisition identity mismatch",
            )
        _require(
            decision["automatic_state"] == automatic["automatic_state"], "Adjudication QC mismatch"
        )
        _require(
            decision["laterality_state"] == automatic["laterality_state"],
            "Adjudication laterality mismatch",
        )
        if session.requires_laterality_decision(index):
            _require(
                decision.get("laterality_decision")
                in {"CONFIRM_VALIDATED_MAPPING", "MARK_UNRESOLVED_EXCLUDE"},
                "A laterality exception lacks an explicit adjudication",
            )
        else:
            _require(
                decision.get("laterality_decision") is None, "Unexpected laterality adjudication"
            )
        panels = {panel["panel_position"]: panel for panel in automatic["panels"]}
        for position in PANEL_POSITIONS:
            panel = panels[position]
            for suffix, field in (
                ("automatic_center_row", "joint_row_resampled"),
                ("automatic_center_column", "joint_column_resampled"),
                ("automatic_state", "qc_state"),
            ):
                _require(
                    _same(decision[f"{position}_{suffix}"], panel[field]),
                    "Adjudication panel/automatic center mismatch",
                )
            row = decision[f"{position}_center_override_row"]
            column = decision[f"{position}_center_override_column"]
            if decision[f"{position}_decision"] == "NEEDS_CENTER_OVERRIDE":
                _require(
                    type(row) is int
                    and type(column) is int
                    and 0 <= row < panel["resampled_rows"]
                    and 0 <= column < panel["resampled_columns"],
                    "Invalid corrected-center coordinates",
                )
            else:
                _require(
                    row is None and column is None,
                    "Corrected center attached to a non-override decision",
                )


def prepare_inputs(
    *,
    session: ReviewSession,
    output_root: Path,
    cohort_path: Path,
    expected_knees: int | None = 6961,
    expected_participants: int | None = 3621,
    expected_review: dict | None = EXPECTED_REVIEW,
) -> dict[str, Any]:
    """Read-only preflight: missing or contradictory linkage is an error, not an exclusion."""
    root = Path(output_root)
    progress = session.progress()
    _require(progress["remaining_acquisitions"] == 0, "Human review is incomplete")
    _require(
        not progress["log_recovery"]["interrupted_final_append_pending"]
        and progress["log_recovery"]["quarantined_fragments"] == 0,
        "Review log recovery is not clean",
    )
    if expected_review is not None:
        for field, value in expected_review.items():
            _require(progress[field] == value, f"Production review state mismatch: {field}")
    _require(
        sha256_file(session.queue_path) == session.queue_fingerprint,
        "Review queue fingerprint changed",
    )
    method = _methodology(root)
    automatic_tree = verify_automatic_tree(root)
    hashes = automatic_tree.set_index("artifact").sha256.to_dict()
    acquisitions_path = root / "manifests/v3_frozen_full_acquisitions.parquet"
    panels_path = root / "manifests/v3_frozen_full_panels.parquet"
    acquisitions = pd.read_parquet(acquisitions_path)
    panels = pd.read_parquet(panels_path)
    cohort = pd.read_parquet(cohort_path)
    keys = ["participant_id", "knee_side_code", "baseline_visit"]
    _require(not cohort[keys].isna().any().any(), "Missing cohort linkage key")
    _require(not cohort.duplicated(keys).any(), "Duplicate cohort linkage")
    _require(cohort.knee_side_code.isin(["1", "2"]).all(), "Invalid cohort knee side")
    _require(cohort.baseline_visit.eq("V00").all(), "Unexpected cohort baseline visit")
    if expected_knees is not None:
        _require(len(cohort) == expected_knees, "Locked cohort row count mismatch")
    if expected_participants is not None:
        _require(
            cohort.participant_id.nunique() == expected_participants,
            "Locked cohort participant count mismatch",
        )
    _require(not acquisitions.acquisition_index.duplicated().any(), "Duplicate acquisition linkage")
    _require(
        not acquisitions.participant_id.duplicated().any(),
        "Duplicate participant/acquisition linkage",
    )
    _require(
        not panels.duplicated(["acquisition_index", "screen_panel"]).any(),
        "Duplicate panel linkage",
    )
    _require(len(panels) == 2 * len(acquisitions), "Missing/extra automatic panels")
    _require(
        set(session.queue.acquisition_index)
        == set(acquisitions.loc[acquisitions.review_required, "acquisition_index"]),
        "Review queue and acquisition review-required flags disagree",
    )
    versions = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "localizer_version": LOCALIZER_VERSION,
        "laterality_policy_version": LATERALITY_POLICY_VERSION,
        "frozen_localizer_source_sha256": method["localizer_source_sha256"],
        "target_spacing_mm": TARGET_SPACING_MM,
        "crop_size_mm": CROP_SIZE_MM,
        "crop_rows": OUTPUT_SHAPE[0],
        "crop_columns": OUTPUT_SHAPE[1],
        "crop_dtype": "uint16",
    }
    records: dict[int, dict] = {}
    for acquisition in acquisitions.to_dict(orient="records"):
        index = int(acquisition["acquisition_index"])
        record = _json(_artifact(root, f"crops/acquisition_{index:04d}/record.json"))
        for field in acquisitions.columns:
            _require(
                field in record and _same(acquisition[field], record[field]),
                "Acquisition manifest/record provenance mismatch",
            )
        for field, value in versions.items():
            _require(record.get(field) == value, f"Automatic {field} provenance mismatch")
        automatic_panels = record.get("panels", [])
        _require(
            len(automatic_panels) == 2
            and {p.get("panel_position") for p in automatic_panels} == set(PANEL_POSITIONS),
            "Automatic panel identity mismatch",
        )
        for panel in automatic_panels:
            position = panel["panel_position"]
            matched = panels.loc[
                panels.acquisition_index.eq(index) & panels.screen_panel.eq(position)
            ]
            _require(len(matched) == 1, "Missing automatic panel linkage")
            tabular = matched.iloc[0]
            for field in (
                "participant_id",
                "source_participant_id",
                "accession_number",
                "associated_file_reference",
                *versions,
            ):
                _require(
                    _same(tabular[field], record[field]),
                    "Panel/acquisition preprocessing or identity provenance mismatch",
                )
            for field in panel:
                if field in panels.columns:
                    _require(
                        _same(tabular[field], panel[field]),
                        "Panel manifest/record provenance mismatch",
                    )
            side, code = SIDE_BY_PANEL[position]
            resolved = record["laterality_state"] == "CONFIDENT"
            _require(
                _same(panel.get("anatomical_side"), side if resolved else None),
                "Wrong anatomical side in automatic record",
            )
            _require(
                _same(tabular.knee_side_code, code if resolved else None),
                "Wrong anatomical side/knee code in panel manifest",
            )
            reference = f"crops/acquisition_{index:04d}/{position}_uint16.npy"
            _require(
                panel["crop_relative_path"] == reference and reference in hashes,
                "Automatic crop panel/path identity mismatch",
            )
            array = np.load(_artifact(root, reference), mmap_mode="r", allow_pickle=False)
            _require(
                array.shape == OUTPUT_SHAPE and array.dtype == np.dtype("uint16"),
                "Automatic crop shape/dtype mismatch",
            )
        records[index] = record
    _require(len(hashes) == 3 * len(records), "Extra/missing automatic processing records")
    by_participant = {record["participant_id"]: record for record in records.values()}
    _require(
        set(cohort.participant_id) == set(by_participant),
        "Missing cohort acquisition linkage or unexpected participant",
    )
    for row in cohort.itertuples(index=False):
        _require(
            row.source_participant_id
            == by_participant[row.participant_id]["source_participant_id"],
            "Cohort participant identity mismatch",
        )
        _require(
            row.knee_side_label == {"1": "right", "2": "left"}[row.knee_side_code],
            "Wrong cohort anatomical side label",
        )
    for queue in session.queue.to_dict(orient="records"):
        record = records[int(queue["acquisition_index"])]
        for field in (
            "participant_id",
            "source_participant_id",
            "accession_number",
            "associated_file_reference",
            "automatic_state",
            "laterality_state",
        ):
            _require(
                _same(queue[field], record[field]), "Review queue acquisition provenance mismatch"
            )
        for panel in record["panels"]:
            position = panel["panel_position"]
            for suffix, field in (
                ("joint_row_resampled", "joint_row_resampled"),
                ("joint_column_resampled", "joint_column_resampled"),
                ("qc_state", "qc_state"),
            ):
                _require(
                    _same(queue[f"{position}_{suffix}"], panel[field]),
                    "Review queue panel/center mismatch",
                )
    _validate_history(session, records)
    protected = {
        str(path): sha256_file(path)
        for path in [
            *FROZEN_SOURCE_FILES,
            *V3_FREEZE_RECORDS,
            Path("src/imaging/laterality_exceptions.py"),
            Path("src/imaging/override_crops.py"),
            Path("src/imaging/frozen_full_v3.py"),
            Path("configs/study_v1.yaml"),
            Path("configs/features_v1.yaml"),
            cohort_path,
            session.queue_path,
            acquisitions_path,
            panels_path,
        ]
    }
    latest = session.latest_decisions()
    snapshot = {
        "progress": progress,
        "log_sha256": sha256_file(session.adjudication_path),
        "history_sha256": _digest(session.adjudications()),
        "laterality_state_sha256": _digest([latest[i] for i in session.laterality_exceptions]),
        "history_records": len(session.adjudications()),
        "crop_count": int(automatic_tree.artifact.str.endswith("_uint16.npy").sum()),
        "crop_aggregate": _aggregate(
            automatic_tree[automatic_tree.artifact.str.endswith("_uint16.npy")]
        ),
        "record_count": len(records),
        "record_aggregate": _aggregate(
            automatic_tree[automatic_tree.artifact.str.endswith("record.json")]
        ),
        "protected_files": protected,
    }
    baseline_path = root / FINAL_DIRECTORY / "pre_change_preservation.json"
    if baseline_path.exists():
        baseline = _json(baseline_path)
        for field, value in snapshot.items():
            _require(baseline.get(field) == value, f"Pre-change preservation mismatch: {field}")
    return {
        "records": records,
        "cohort": cohort,
        "hashes": hashes,
        "method": method,
        "snapshot": snapshot,
    }


def validate_override_requests(
    inputs: dict, session: ReviewSession, root: Path, archive_root: Path
) -> None:
    """Prove original crop/archive and active centers BEFORE the frozen generator may publish."""
    known = {record["sequence"] for record in session.adjudications()}
    checked_archives = set()
    for request in pending_overrides(session):
        index, position = request["acquisition_index"], request["panel_position"]
        record = inputs["records"][index]
        bundle = root / "overrides" / f"acquisition_{index:04d}"
        stored = bundle / f"{position}_record.json"
        crop = bundle / f"{position}_uint16.npy"
        _require(
            stored.exists() == crop.exists(), "Orphaned override crop/record; refusing overwrite"
        )
        verify_existing_override(
            bundle,
            request,
            known_sequences=known,
            output_root=root,
            session=session,
            automatic_record=record,
        )
        if index not in checked_archives:
            archive = _artifact(archive_root, record["associated_file_reference"])
            _require(
                archive.stat().st_size == record["source_archive_bytes"]
                and sha256_file(archive) == record["source_archive_sha256"],
                "Original acquisition archive SHA/provenance mismatch",
            )
            checked_archives.add(index)


def resolve_panels(inputs: dict, session: ReviewSession, root: Path) -> pd.DataFrame:
    latest_by_acquisition = {
        int(decision["acquisition_index"]): decision
        for decision in session.latest_decisions().values()
    }
    requests = {
        (r["acquisition_index"], r["panel_position"]): r for r in pending_overrides(session)
    }
    history = session.adjudications()
    known = {record["sequence"] for record in history}
    revision_numbers = dict.fromkeys(session.order, -1)
    for revision in history:
        revision_numbers[revision["review_index"]] += 1
    rows = []
    for index, record in sorted(inputs["records"].items()):
        decision = latest_by_acquisition.get(index)
        _require(
            bool(record["review_required"]) == (decision is not None),
            "Missing/unexpected effective human decision",
        )
        side_resolved = record["laterality_state"] == "CONFIDENT" or (
            decision is not None and decision["laterality_decision"] == "CONFIRM_VALIDATED_MAPPING"
        )
        for panel in record["panels"]:
            position = panel["panel_position"]
            side, code = SIDE_BY_PANEL[position]
            human = None if decision is None else decision[f"{position}_decision"]
            automatic_ref = panel["crop_relative_path"]
            automatic_sha = inputs["hashes"][automatic_ref]
            override_ref = override_sha = override_version = None
            corrected_row = corrected_column = None
            if human == "NEEDS_CENTER_OVERRIDE":
                request = requests[(index, position)]
                verified = verify_existing_override(
                    root / "overrides" / f"acquisition_{index:04d}",
                    request,
                    known_sequences=known,
                    output_root=root,
                    session=session,
                    automatic_record=record,
                )
                _require(verified is not None, "Required override has not been materialized")
                _require(
                    verified["decision"] == human
                    and verified["automatic_crop_preserved"] is True
                    and verified["adjudication_recorded_at_utc"] == decision["recorded_at_utc"],
                    "Override adjudication provenance mismatch",
                )
                override_ref, override_sha = (
                    verified["override_crop_relative_path"],
                    verified["corrected_crop_sha256"],
                )
                corrected_row, corrected_column = (
                    request["manual_center_row"],
                    request["manual_center_column"],
                )
                override_version = OVERRIDE_CROP_VERSION
            if not side_resolved:
                provenance, reason = "UNRESOLVED", "HUMAN_LATERALITY_UNRESOLVED_EXCLUDE"
            elif human == "REJECT":
                provenance, reason = "REJECTED", "HUMAN_REJECTED_CROP"
            elif human == "NEEDS_CENTER_OVERRIDE":
                provenance, reason = "HUMAN_OVERRIDE", None
            elif human == "ACCEPT":
                provenance, reason = "HUMAN_ACCEPT", None
            else:
                _require(
                    record["automatic_state"] == panel["qc_state"] == "PASS",
                    "Unqueued panel is not AUTO PASS",
                )
                provenance, reason = "AUTO_PASS", None
            included = reason is None
            effective_ref = (
                (override_ref if provenance == "HUMAN_OVERRIDE" else automatic_ref)
                if included
                else None
            )
            effective_sha = (
                (override_sha if provenance == "HUMAN_OVERRIDE" else automatic_sha)
                if included
                else None
            )
            rows.append(
                {
                    "participant_id": record["participant_id"],
                    "source_participant_id": record["source_participant_id"],
                    "knee_side_code": code,
                    "knee_side_label": "right" if code == "1" else "left",
                    "baseline_visit": "V00",
                    "acquisition_index": index,
                    "panel_position": position,
                    "anatomical_side": side if side_resolved else None,
                    "laterality_provenance": "FROZEN_LATERALITY_V2"
                    if record["laterality_state"] == "CONFIDENT"
                    else (
                        "HUMAN_CONFIRMED_VALIDATED_MAPPING"
                        if side_resolved
                        else "HUMAN_UNRESOLVED_EXCLUDE"
                    ),
                    "automatic_laterality_state": record["laterality_state"],
                    "laterality_decision": None
                    if decision is None
                    else decision.get("laterality_decision"),
                    "automatic_crop_relative_path": automatic_ref,
                    "automatic_crop_sha256": automatic_sha,
                    "automatic_record_sha256": inputs["hashes"][
                        f"crops/acquisition_{index:04d}/record.json"
                    ],
                    "source_archive_sha256": record["source_archive_sha256"],
                    "source_dicom_sha256": record["dicom_sha256"],
                    "automatic_qc_state": panel["qc_state"],
                    "automatic_acquisition_state": record["automatic_state"],
                    "human_review_required": bool(record["review_required"]),
                    "human_decision": human,
                    "review_index": None if decision is None else decision["review_index"],
                    "adjudication_sequence": None if decision is None else decision["sequence"],
                    "adjudication_revision": None
                    if decision is None
                    else revision_numbers[decision["review_index"]],
                    "adjudication_supersedes": None if decision is None else decision["supersedes"],
                    "adjudication_recorded_at_utc": None
                    if decision is None
                    else decision["recorded_at_utc"],
                    "automatic_center_row": panel["joint_row_resampled"],
                    "automatic_center_column": panel["joint_column_resampled"],
                    "corrected_center_row": corrected_row,
                    "corrected_center_column": corrected_column,
                    "override_crop_relative_path": override_ref,
                    "override_crop_sha256": override_sha,
                    "effective_crop_relative_path": effective_ref,
                    "effective_crop_sha256": effective_sha,
                    "effective_provenance": provenance,
                    "imaging_included": included,
                    "inclusion_status": "INCLUDED" if included else "EXCLUDED",
                    "exclusion_reason": reason,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "localizer_version": LOCALIZER_VERSION,
                    "laterality_policy_version": LATERALITY_POLICY_VERSION,
                    **inputs["method"],
                    "override_crop_version": override_version,
                    "adjudication_schema_version": ADJUDICATION_SCHEMA_VERSION,
                    "adjudication_log_sha256": inputs["snapshot"]["log_sha256"],
                    "review_queue_sha256": session.queue_fingerprint,
                    "analysis_cohort_sha256": inputs["snapshot"]["protected_files"][
                        str(inputs["cohort_path"])
                    ],
                    "final_manifest_version": FINAL_MANIFEST_VERSION,
                    "target_spacing_mm": TARGET_SPACING_MM,
                    "crop_size_mm": CROP_SIZE_MM,
                    "crop_rows": OUTPUT_SHAPE[0],
                    "crop_columns": OUTPUT_SHAPE[1],
                    "crop_dtype": "uint16",
                    "padding_percentile": PADDING_PERCENTILE,
                }
            )
    frame = pd.DataFrame(rows)
    for column in (
        "review_index",
        "adjudication_sequence",
        "adjudication_revision",
        "adjudication_supersedes",
        "corrected_center_row",
        "corrected_center_column",
    ):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def reconcile_cohort(cohort: pd.DataFrame, panels: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant_id", "knee_side_code", "baseline_visit"]
    _require(not panels.duplicated(keys).any(), "Duplicate effective panel/cohort mapping")
    _require(not cohort.duplicated(keys).any(), "Duplicate cohort linkage")
    ordered = cohort[keys].copy()
    ordered.insert(0, "analysis_row_index", range(1, len(cohort) + 1))
    final = ordered.merge(
        panels, on=keys, how="left", validate="one_to_one", sort=False, indicator=True
    )
    _require(final._merge.eq("both").all(), "Missing cohort acquisition/panel linkage")
    final = final.drop(columns="_merge").sort_values("analysis_row_index", ignore_index=True)
    # A merge from nullable cohort strings can yield object keys, which Pandas 3 reads back
    # from Parquet as inferred strings. Keep the resolved panel's string dtype so exact
    # round-trip comparison works without weakening existing-artifact validation.
    for key in keys:
        final[key] = final[key].astype(panels[key].dtype)
    _require(len(final) == len(cohort), "Analysis rows silently dropped")
    return final


def audit_manifest(final: pd.DataFrame, cohort: pd.DataFrame, root: Path) -> dict:
    _require(len(final) == len(cohort), "Analysis rows silently dropped")
    keys = ["participant_id", "knee_side_code", "baseline_visit"]
    _require(
        not final.duplicated(keys).any()
        and list(final[keys].itertuples(index=False, name=None))
        == list(cohort[keys].itertuples(index=False, name=None)),
        "Final cohort linkage/order differs from the authoritative cohort",
    )
    _require(final.effective_provenance.isin(PROVENANCES).all(), "Unknown effective provenance")
    included = final.loc[final.imaging_included]
    excluded = final.loc[~final.imaging_included]
    _require(excluded.exclusion_reason.notna().all(), "Silent exclusion without explicit reason")
    _require(
        excluded.effective_provenance.isin(["REJECTED", "UNRESOLVED"]).all(),
        "Unauthorized exclusion",
    )
    _require(
        excluded.effective_crop_relative_path.isna().all(),
        "Excluded knee references an effective crop",
    )
    _require(included.exclusion_reason.isna().all(), "Included knee has an exclusion reason")
    _require(
        not included.effective_crop_relative_path.duplicated().any(),
        "Duplicate effective-crop mappings",
    )
    for row in final.itertuples(index=False):
        side, code = SIDE_BY_PANEL[row.panel_position]
        _require(row.knee_side_code == code, "Laterality mismatch between knee and screen panel")
        _require(
            _same(row.anatomical_side, side if row.effective_provenance != "UNRESOLVED" else None),
            "Wrong anatomical side",
        )
        if row.imaging_included:
            artifact = _artifact(root, row.effective_crop_relative_path)
            _require(
                sha256_file(artifact) == row.effective_crop_sha256, "Effective crop SHA mismatch"
            )
            array = np.load(artifact, mmap_mode="r", allow_pickle=False)
            _require(
                array.shape == OUTPUT_SHAPE and array.dtype == np.dtype("uint16"),
                "Effective crop shape/dtype mismatch",
            )
            expected_ref = (
                row.override_crop_relative_path
                if row.effective_provenance == "HUMAN_OVERRIDE"
                else row.automatic_crop_relative_path
            )
            _require(
                row.effective_crop_relative_path == expected_ref,
                "Effective crop provenance/path mismatch",
            )
    participant_sizes = included.groupby("participant_id").size().value_counts().to_dict()
    return {
        "total_analysis_knees": len(final),
        "unique_participants": int(final.participant_id.nunique()),
        "resolved_effective_crops": len(included),
        "explicitly_excluded_knees": len(excluded),
        "provenance_counts": {p: int(final.effective_provenance.eq(p).sum()) for p in PROVENANCES},
        "missing_acquisition_links": 0,
        "duplicate_effective_crop_mappings": 0,
        "laterality_mismatches": 0,
        "missing_artifact_files": 0,
        "sha_mismatches": 0,
        "override_provenance_failures": 0,
        "automatic_crop_mutation_count": 0,
        "analysis_rows_silently_dropped": 0,
        "effective_image_files_referenced": int(included.effective_crop_relative_path.nunique()),
        "left_right_distribution": final.knee_side_label.value_counts().to_dict(),
        "resolved_left_right_distribution": included.knee_side_label.value_counts().to_dict(),
        "participant_knee_distribution_after_resolution": {
            "one_knee": int(participant_sizes.get(1, 0)),
            "two_knees": int(participant_sizes.get(2, 0)),
            "no_resolved_knees": int(
                final.participant_id.nunique() - included.participant_id.nunique()
            ),
        },
        "reconciliation_passed": len(included) + len(excluded) == len(cohort),
        "integrity_passed": True,
    }


def _publish(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        except AssertionError as error:
            raise FinalImagingError(
                "Existing final artifact differs; refusing silent overwrite"
            ) from error
    else:
        atomic_parquet(frame, path)


def finalize_imaging(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    cohort_path: Path = DEFAULT_COHORT,
    session: ReviewSession | None = None,
    expected_knees: int | None = 6961,
    expected_participants: int | None = 3621,
    expected_review: dict | None = EXPECTED_REVIEW,
) -> dict:
    root, cohort_path = Path(output_root), Path(cohort_path)
    session = session or ReviewSession(
        queue_path=root / "review_queue.parquet",
        adjudication_path=root / "review/adjudications.jsonl",
        output_root=root,
    )
    print(
        "Verifying frozen inputs, automatic bytes, review history and cohort linkage...", flush=True
    )
    inputs = prepare_inputs(
        session=session,
        output_root=root,
        cohort_path=cohort_path,
        expected_knees=expected_knees,
        expected_participants=expected_participants,
        expected_review=expected_review,
    )
    inputs["cohort_path"] = cohort_path
    directory = root / FINAL_DIRECTORY
    validate_override_requests(inputs, session, root, Path(archive_root))
    _require(
        not any(path.is_file() for path in Path(work_root).rglob("*")),
        "Override staging contains pre-existing files; choose a new empty work root",
    )
    print(
        f"Materializing/verifying {len(pending_overrides(session))} frozen override crops...",
        flush=True,
    )
    materialization = regenerate_override_crops(
        session=session, output_root=root, archive_root=archive_root, work_root=work_root
    )
    _require(
        materialization["failed_panels"] == 0 and materialization["temporary_files_remaining"] == 0,
        "Override materialization failed",
    )
    panels = resolve_panels(inputs, session, root)
    final = reconcile_cohort(inputs["cohort"], panels)
    cohort_keys = set(
        zip(final.participant_id, final.knee_side_code, final.baseline_visit, strict=True)
    )
    panels["in_analysis_cohort"] = [
        key in cohort_keys
        for key in zip(
            panels.participant_id, panels.knee_side_code, panels.baseline_visit, strict=True
        )
    ]
    panels["cohort_exclusion_reason"] = panels.in_analysis_cohort.map(
        {True: None, False: "NOT_IN_LOCKED_ANALYSIS_COHORT"}
    )
    print(
        "Auditing every effective cohort crop and re-verifying protected automatic bytes...",
        flush=True,
    )
    audit = audit_manifest(final, inputs["cohort"], root)
    after = verify_automatic_tree(root)
    _require(
        _aggregate(after[after.artifact.str.endswith("_uint16.npy")])
        == inputs["snapshot"]["crop_aggregate"],
        "Automatic crops changed during materialization",
    )
    _require(
        _aggregate(after[after.artifact.str.endswith("record.json")])
        == inputs["snapshot"]["record_aggregate"],
        "Processing records changed during materialization",
    )
    _require(
        sha256_file(session.adjudication_path) == inputs["snapshot"]["log_sha256"],
        "Human adjudications changed during finalization",
    )
    for path, digest in inputs["snapshot"]["protected_files"].items():
        _require(
            sha256_file(path) == digest, "A protected source/configuration/cohort artifact changed"
        )
    _methodology(root)
    _publish(final, directory / "imaging_manifest.parquet")
    _publish(panels, directory / "panel_resolution.parquet")
    overrides = panels.loc[panels.human_decision.eq("NEEDS_CENTER_OVERRIDE")].reset_index(drop=True)
    _publish(overrides, directory / "override_linkage.parquet")
    audit.update(
        {
            "all_panel_count": len(panels),
            "panels_outside_analysis_cohort": int((~panels.in_analysis_cohort).sum()),
            "all_panel_provenance_counts": {
                p: int(panels.effective_provenance.eq(p).sum()) for p in PROVENANCES
            },
            "outside_cohort_provenance_counts": {
                p: int(panels.loc[~panels.in_analysis_cohort, "effective_provenance"].eq(p).sum())
                for p in PROVENANCES
            },
            "all_override_panels": len(overrides),
            "laterality_exception_review": session.progress()["laterality_exceptions"],
            "preservation": inputs["snapshot"],
            "manifest_sha256": sha256_file(directory / "imaging_manifest.parquet"),
            "panel_resolution_sha256": sha256_file(directory / "panel_resolution.parquet"),
            "override_linkage_sha256": sha256_file(directory / "override_linkage.parquet"),
            "finalizer_version": FINAL_MANIFEST_VERSION,
            "finalizer_source_sha256": sha256_file(Path(__file__)),
        }
    )
    atomic_json(materialization, directory / "override_materialization.json")
    atomic_json(audit, directory / "integrity_audit.json")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    args = parser.parse_args(argv)
    audit = finalize_imaging(
        output_root=args.dataset_root,
        archive_root=args.archive_root,
        work_root=args.work_root,
        cohort_path=args.cohort,
    )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "preservation"}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
