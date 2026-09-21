"""Tests for cross-process adjudication safety, crash recovery, and status precedence."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import EXCEPTION_CASE, build_review_queue

from imaging.artifact_io import (
    LogCorruptionError,
    append_bytes,
    read_jsonl_checked,
    serialize_jsonl_record,
)
from imaging.v3_review_ui import (
    PanelDecision,
    ReviewConflictError,
    ReviewSession,
)

pytestmark = pytest.mark.public_portable

# Each writer runs in its own interpreter, so an in-memory lock cannot be what makes this safe.
_WRITER = """
import json, sys, time
from pathlib import Path
from imaging.v3_review_ui import (
    PanelDecision, ReviewConflictError, ReviewSession, ReviewToolError,
)

queue, log, root, start, decision, use_token = sys.argv[1:7]
session = ReviewSession(
    queue_path=Path(queue), adjudication_path=Path(log), output_root=Path(root)
)
panels = (
    PanelDecision(panel_position="screen_left", decision=decision),
    PanelDecision(panel_position="screen_right", decision=decision),
)
deadline = float(start)
while time.time() < deadline:
    pass
try:
    record = session.record_adjudication(
        1, panels, expected_current_sequence=0 if use_token == "1" else None
    )
    print(json.dumps({"status": "committed", "sequence": record["sequence"]}))
except ReviewConflictError as error:
    print(json.dumps({"status": "conflict", "message": str(error)}))
except ReviewToolError as error:
    print(json.dumps({"status": "error", "message": str(error)}))
"""


def _session(tmp_path: Path) -> ReviewSession:
    queue_path = build_review_queue(tmp_path)
    return ReviewSession(
        queue_path=queue_path,
        adjudication_path=queue_path.parent / "review/adjudications.jsonl",
        output_root=queue_path.parent,
    )


def _both(decision: str) -> tuple[PanelDecision, ...]:
    return tuple(
        PanelDecision(panel_position=position, decision=decision)
        for position in ("screen_left", "screen_right")
    )


def _race(tmp_path: Path, *, writers: int = 5, use_token: bool) -> list[dict]:
    """Launch independent writer processes that all try to decide the same case at once."""

    queue_path = build_review_queue(tmp_path)
    log_path = queue_path.parent / "review/adjudications.jsonl"
    script = tmp_path / "writer.py"
    script.write_text(_WRITER, encoding="utf-8")

    start = time.time() + 2.0
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(queue_path),
                str(log_path),
                str(queue_path.parent),
                f"{start:.6f}",
                "ACCEPT" if index % 2 == 0 else "REJECT",
                "1" if use_token else "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(writers)
    ]
    outcomes = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr
        outcomes.append(json.loads(stdout.strip()))
    return outcomes


@pytest.mark.parametrize("use_token", [True, False])
def test_only_one_process_commits_a_first_decision(tmp_path: Path, use_token: bool) -> None:
    outcomes = _race(tmp_path, use_token=use_token)

    committed = [item for item in outcomes if item["status"] == "committed"]
    conflicts = [item for item in outcomes if item["status"] == "conflict"]
    assert len(committed) == 1
    assert len(conflicts) == len(outcomes) - 1
    assert [item["status"] for item in outcomes].count("error") == 0
    assert committed[0]["sequence"] == 1

    queue_path = tmp_path / "v3_frozen_full/review_queue.parquet"
    session = ReviewSession(
        queue_path=queue_path,
        adjudication_path=queue_path.parent / "review/adjudications.jsonl",
        output_root=queue_path.parent,
    )
    records, tail = session.load()

    assert tail is None
    assert len(records) == 1
    assert [record["sequence"] for record in records] == [1]
    assert records[0]["supersedes"] is None
    assert session.progress()["reviewed_acquisitions"] == 1
    assert session.next_pending() == 2


def test_a_stale_revision_token_is_a_deterministic_conflict(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    session.record_adjudication(1, _both("REJECT"), allow_revision=True)

    # A page rendered when revision 1 was current is now stale.
    with pytest.raises(ReviewConflictError, match="changed since it was displayed"):
        session.record_adjudication(
            1, _both("ACCEPT"), allow_revision=True, expected_current_sequence=1
        )

    assert [record["sequence"] for record in session.revision_history(1)] == [1, 2]
    assert session.latest_decisions()[1]["sequence"] == 2


def test_revising_a_case_that_has_no_decision_is_a_conflict(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewConflictError, match="no decision to revise"):
        session.record_adjudication(1, _both("ACCEPT"), allow_revision=True)

    assert session.adjudications() == []


def test_an_interrupted_final_append_is_recovered_and_preserved(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    session.record_adjudication(2, _both("REJECT"))
    fragment = b'{"schema_version": "v3_frozen_full_adjudication_v2", "review_index": 3, "seq'
    append_bytes(fragment, session.adjudication_path)

    records, tail = session.load()

    assert len(records) == 2
    assert tail is not None
    assert tail.fragment == fragment
    assert tail.reason == "interrupted_final_append"
    assert session.progress()["log_recovery"]["interrupted_final_append_pending"] is True
    assert session.next_pending() == EXCEPTION_CASE

    # The next real append clears the fragment, preserving it in a sidecar first.
    session.record_adjudication(
        EXCEPTION_CASE, _both("ACCEPT"), laterality_decision="MARK_UNRESOLVED_EXCLUDE"
    )
    recovered_records, recovered_tail = session.load()
    sidecar = Path(f"{session.adjudication_path}.corrupt_tail")
    quarantined, _ = read_jsonl_checked(sidecar)

    assert recovered_tail is None
    assert [record["review_index"] for record in recovered_records] == [1, 2, EXCEPTION_CASE]
    assert len(quarantined) == 1
    assert base64.b64decode(quarantined[0]["fragment_base64"]) == fragment
    assert quarantined[0]["reason"] == "interrupted_final_append"
    assert recovered_records[-1]["recovered_fragment_bytes"] == len(fragment)
    assert session.progress()["log_recovery"] == {
        "interrupted_final_append_pending": False,
        "pending_fragment_bytes": 0,
        "quarantined_fragments": 1,
        "quarantine_relative_name": sidecar.name,
    }


def test_a_short_write_of_a_whole_record_is_not_promoted_to_history(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    payload = serialize_jsonl_record(
        {
            "schema_version": "v3_frozen_full_adjudication_v2",
            "review_index": 2,
            "sequence": 2,
        }
    )
    # Everything except the terminating newline reached the disk.
    append_bytes(payload[:-1], session.adjudication_path)

    records, tail = session.load()

    assert len(records) == 1
    assert tail is not None
    assert tail.fragment == payload[:-1]


def test_corruption_before_the_final_record_fails_closed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    append_bytes(b"this is not a json record\n", session.adjudication_path)
    session_two = _session(tmp_path)
    del session_two

    with pytest.raises(LogCorruptionError, match="is not a terminal"):
        session.load()
    with pytest.raises(LogCorruptionError):
        session.record_adjudication(2, _both("ACCEPT"))


def test_corruption_in_the_middle_of_the_log_fails_closed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    append_bytes(b"{broken\n", session.adjudication_path)
    append_bytes(
        serialize_jsonl_record({"schema_version": "later", "review_index": 2}),
        session.adjudication_path,
    )

    with pytest.raises(LogCorruptionError, match="record 2"):
        session.load()


def test_a_recovered_log_keeps_its_earlier_records_byte_for_byte(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    before = session.adjudication_path.read_bytes()
    append_bytes(b'{"partial": ', session.adjudication_path)

    session.record_adjudication(2, _both("REJECT"))
    after = session.adjudication_path.read_bytes()

    assert after.startswith(before)


# ---------------------------------------------------------------------------
# Final status precedence
# ---------------------------------------------------------------------------


def test_an_initial_decision_is_effective_with_no_superseded_history(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))

    status = session.effective_panel_status()[(1, "screen_left")]

    assert status["human_panel_decision"] == "ACCEPT"
    assert status["crop_source"] == "HUMAN_ACCEPT"
    assert status["adjudication_sequence"] == 1
    assert status["superseded_sequences"] == []
    assert status["revisions_recorded"] == 0
    assert status["review_complete"] is True


def test_a_superseded_revision_never_becomes_final_again(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    session.record_adjudication(1, _both("REJECT"), allow_revision=True)
    session.record_adjudication(1, _both("ACCEPT"), allow_revision=True)

    status = session.effective_panel_status()[(1, "screen_right")]
    history = session.revision_history(1)

    assert status["adjudication_sequence"] == 3
    assert status["human_panel_decision"] == "ACCEPT"
    assert status["crop_source"] == "HUMAN_ACCEPT"
    assert status["superseded_sequences"] == [1, 2]
    assert status["revisions_recorded"] == 2
    # Old revisions remain traceable in order.
    assert [record["screen_right_decision"] for record in history] == [
        "ACCEPT",
        "REJECT",
        "ACCEPT",
    ]
    assert [record["supersedes"] for record in history] == [None, 1, 2]


def test_the_two_panels_resolve_independently(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(
        1,
        (
            PanelDecision(
                panel_position="screen_left",
                decision="NEEDS_CENTER_OVERRIDE",
                override_row=640,
                override_column=430,
            ),
            PanelDecision(panel_position="screen_right", decision="REJECT"),
        ),
    )

    statuses = session.effective_panel_status()

    assert statuses[(1, "screen_left")]["crop_source"] == "HUMAN_OVERRIDE"
    assert statuses[(1, "screen_right")]["crop_source"] == "REJECTED"


def test_the_automatic_state_is_preserved_beside_the_human_decision(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("REJECT"))

    status = session.effective_panel_status()[(1, "screen_left")]

    assert status["automatic_acquisition_state"] == "BORDERLINE"
    assert status["automatic_panel_state"] == "BORDERLINE"
    assert status["human_panel_decision"] == "REJECT"


def test_an_undecided_case_is_pending_rather_than_defaulted(tmp_path: Path) -> None:
    session = _session(tmp_path)

    status = session.effective_panel_status()[(2, "screen_left")]

    assert status["human_panel_decision"] is None
    assert status["crop_source"] == "PENDING_REVIEW"
    assert status["review_complete"] is False
    assert status["adjudication_sequence"] is None


@pytest.mark.parametrize(
    ("laterality", "side_resolved"),
    [("CONFIRM_VALIDATED_MAPPING", True), ("MARK_UNRESOLVED_EXCLUDE", False)],
)
def test_laterality_governs_side_assignment_but_not_the_crop_verdict(
    tmp_path: Path, laterality: str, side_resolved: bool
) -> None:
    session = _session(tmp_path)
    session.record_adjudication(EXCEPTION_CASE, _both("ACCEPT"), laterality_decision=laterality)

    status = session.effective_panel_status()[(EXCEPTION_CASE, "screen_left")]

    # The panel keeps its accepted crop verdict either way.
    assert status["human_panel_decision"] == "ACCEPT"
    assert status["crop_source"] == "HUMAN_ACCEPT"
    assert status["automatic_panel_state"] == "BORDERLINE"
    # Only the ability to assign an anatomical side changes.
    assert status["requires_laterality_decision"] is True
    assert status["laterality_decision"] == laterality
    assert status["anatomical_side_resolved"] is side_resolved
    assert status["side_assignment_blocked"] is not side_resolved


def test_a_resolved_case_needs_no_laterality_decision_to_assign_a_side(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))

    status = session.effective_panel_status()[(1, "screen_left")]

    assert status["requires_laterality_decision"] is False
    assert status["laterality_decision"] is None
    assert status["anatomical_side_resolved"] is True


def test_the_status_snapshot_is_written_in_deterministic_order(tmp_path: Path) -> None:
    import pandas as pd

    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    session.record_adjudication(1, _both("REJECT"), allow_revision=True)

    target = session.status_snapshot(target=tmp_path / "status.parquet")
    frame = pd.read_parquet(target)

    assert frame["review_index"].tolist() == [1, 1, 2, 2, 3, 3]
    assert frame["panel_position"].tolist()[:2] == ["screen_left", "screen_right"]
    assert frame.loc[0, "superseded_sequences"] == "1"
    assert frame.loc[0, "crop_source"] == "REJECTED"
