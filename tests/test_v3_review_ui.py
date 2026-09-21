"""Synthetic tests for the local V3 crop-review workflow."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest
from conftest import EXCEPTION_CASE, build_review_queue

from imaging.v3_review_ui import (
    DECISION_LEGEND,
    PROTECTED_COLUMNS,
    PanelDecision,
    ReviewConflictError,
    ReviewRequestHandler,
    ReviewSession,
    ReviewToolError,
    render_case,
    render_exception_index,
    serve,
)

pytestmark = pytest.mark.public_portable


def _queue(tmp_path: Path, *, cases: int = 3) -> Path:
    return build_review_queue(tmp_path, cases=cases)


def _session(tmp_path: Path, *, cases: int = 3) -> ReviewSession:
    queue_path = _queue(tmp_path, cases=cases)
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


def test_session_orders_the_queue_deterministically_and_reports_progress(tmp_path: Path) -> None:
    session = _session(tmp_path)

    assert session.order == [1, 2, 3]
    progress = session.progress()
    assert progress["queued_acquisitions"] == 3
    assert progress["remaining_acquisitions"] == 3
    assert progress["reviewed_acquisitions"] == 0
    assert progress["laterality_exceptions"]["total"] == 1
    assert session.next_pending() == 1


def test_the_two_panels_take_independent_decisions(tmp_path: Path) -> None:
    session = _session(tmp_path)

    record = session.record_adjudication(
        1,
        (
            PanelDecision(panel_position="screen_left", decision="ACCEPT"),
            PanelDecision(panel_position="screen_right", decision="REJECT"),
        ),
    )

    assert record["screen_left_decision"] == "ACCEPT"
    assert record["screen_right_decision"] == "REJECT"
    assert session.progress()["panel_decisions"] == {
        "ACCEPT": 1,
        "REJECT": 1,
        "NEEDS_CENTER_OVERRIDE": 0,
    }


def test_a_case_is_not_recorded_until_both_panels_have_a_decision(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="Both screen panels need a decision"):
        session.record_adjudication(
            1, (PanelDecision(panel_position="screen_left", decision="ACCEPT"),)
        )
    assert session.adjudications() == []


def test_decisions_are_appended_and_the_session_resumes_where_it_stopped(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    session.record_adjudication(2, _both("REJECT"), note="no adequate crop")

    resumed = ReviewSession(
        queue_path=session.queue_path,
        adjudication_path=session.adjudication_path,
        output_root=session.output_root,
    )

    assert resumed.next_pending() == 3
    assert resumed.progress()["reviewed_acquisitions"] == 2
    assert resumed.progress()["panel_decisions"] == {
        "ACCEPT": 2,
        "REJECT": 2,
        "NEEDS_CENTER_OVERRIDE": 0,
    }


def test_an_existing_decision_is_never_silently_overwritten(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))

    with pytest.raises(ReviewConflictError, match="revision must be explicit"):
        session.record_adjudication(1, _both("REJECT"))

    revision = session.record_adjudication(1, _both("REJECT"), allow_revision=True)
    records = session.adjudications()

    assert revision["supersedes"] == 1
    assert len(records) == 2
    assert [record["screen_left_decision"] for record in records] == ["ACCEPT", "REJECT"]
    assert session.progress()["revisions_recorded"] == 1
    assert session.progress()["reviewed_acquisitions"] == 1


def test_center_override_requires_in_bounds_coordinates(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="requires a corrected center"):
        session.record_adjudication(
            1,
            (
                PanelDecision(panel_position="screen_left", decision="NEEDS_CENTER_OVERRIDE"),
                PanelDecision(panel_position="screen_right", decision="ACCEPT"),
            ),
        )
    with pytest.raises(ReviewToolError, match="outside the resampled panel"):
        session.record_adjudication(
            1,
            (
                PanelDecision(
                    panel_position="screen_left",
                    decision="NEEDS_CENTER_OVERRIDE",
                    override_row=5000,
                    override_column=10,
                ),
                PanelDecision(panel_position="screen_right", decision="ACCEPT"),
            ),
        )

    record = session.record_adjudication(
        1,
        (
            PanelDecision(
                panel_position="screen_left",
                decision="NEEDS_CENTER_OVERRIDE",
                override_row=610,
                override_column=430,
            ),
            PanelDecision(panel_position="screen_right", decision="ACCEPT"),
        ),
    )

    assert record["screen_left_center_override_row"] == 610
    assert record["screen_left_center_override_column"] == 430
    assert record["screen_left_automatic_center_row"] == 600
    assert record["screen_left_automatic_center_column"] == 420
    assert record["screen_right_center_override_row"] is None


def test_corrected_centers_are_rejected_for_plain_decisions(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="only allowed with NEEDS_CENTER_OVERRIDE"):
        session.record_adjudication(
            1,
            (
                PanelDecision(
                    panel_position="screen_left",
                    decision="ACCEPT",
                    override_row=600,
                    override_column=420,
                ),
                PanelDecision(panel_position="screen_right", decision="ACCEPT"),
            ),
        )


def test_unknown_decision_and_unknown_panel_are_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="ACCEPT, REJECT"):
        session.record_adjudication(
            1,
            (
                PanelDecision(panel_position="screen_left", decision="MAYBE"),
                PanelDecision(panel_position="screen_right", decision="ACCEPT"),
            ),
        )
    with pytest.raises(ReviewToolError, match="unknown screen panel"):
        session.record_adjudication(
            1, (PanelDecision(panel_position="screen_middle", decision="ACCEPT"),)
        )


def test_unresolved_laterality_needs_an_explicit_decision(tmp_path: Path) -> None:
    session = _session(tmp_path)

    assert session.laterality_exceptions == [EXCEPTION_CASE]
    with pytest.raises(ReviewToolError, match="explicit\n?\\s*laterality decision"):
        session.record_adjudication(EXCEPTION_CASE, _both("ACCEPT"))

    record = session.record_adjudication(
        EXCEPTION_CASE,
        _both("ACCEPT"),
        laterality_decision="CONFIRM_VALIDATED_MAPPING",
    )

    assert record["laterality_decision"] == "CONFIRM_VALIDATED_MAPPING"
    assert session.progress()["laterality_exceptions"] == {
        "total": 1,
        "reviewed": 1,
        "remaining": 0,
        "decisions": {"CONFIRM_VALIDATED_MAPPING": 1, "MARK_UNRESOLVED_EXCLUDE": 0},
    }


def test_laterality_decision_is_refused_for_a_resolved_acquisition(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="only to an acquisition with unresolved"):
        session.record_adjudication(
            1, _both("ACCEPT"), laterality_decision="CONFIRM_VALIDATED_MAPPING"
        )


def test_an_unknown_laterality_decision_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ReviewToolError, match="laterality decision"):
        session.record_adjudication(
            EXCEPTION_CASE, _both("ACCEPT"), laterality_decision="ASSUME_LEFT"
        )


def test_exception_index_lists_only_unresolved_acquisitions(tmp_path: Path) -> None:
    session = _session(tmp_path)

    page = render_exception_index(session)

    assert f"/case/{EXCEPTION_CASE}" in page
    assert "MARK_UNRESOLVED_EXCLUDE" not in page or "not yet reviewed" in page
    assert "/case/1" not in page
    for column in PROTECTED_COLUMNS:
        assert str(session.row(EXCEPTION_CASE)[column]) not in page


def test_asset_resolution_is_restricted_to_rendered_previews(tmp_path: Path) -> None:
    session = _session(tmp_path)

    assert session.asset_path(1, "bilateral").is_file()
    assert session.asset_path(1, "screen_left_overlay").is_file()
    with pytest.raises(ReviewToolError, match="Unknown review asset"):
        session.asset_path(1, "../../etc/passwd")


def test_rendered_case_exposes_no_participant_identifier(tmp_path: Path) -> None:
    session = _session(tmp_path)
    row = session.row(1)

    page = render_case(session, 1)

    for column in PROTECTED_COLUMNS:
        assert str(row[column]) not in page
    assert "crops/acquisition_0101" not in page
    assert "BORDERLINE" in page
    assert "NEEDS_CENTER_OVERRIDE" in page
    assert "0101" in page
    for name, text in DECISION_LEGEND:
        assert name in page
        assert text in page
    assert 'name="expected_current_sequence"' in page
    assert 'value="0"' in page


def test_rendering_fails_closed_if_the_page_would_leak_linkage(tmp_path: Path) -> None:
    session = _session(tmp_path)
    # A reason code that smuggles the accession number must not be rendered silently.
    session.queue.loc[0, "review_reason_codes"] = str(session.row(1)["accession_number"])

    with pytest.raises(ReviewToolError, match="expose protected linkage"):
        render_case(session, 1)


def test_the_laterality_exception_case_renders_its_own_section(tmp_path: Path) -> None:
    session = _session(tmp_path)

    page = render_case(session, EXCEPTION_CASE)

    assert "Laterality exception" in page
    assert "CONFIRM_VALIDATED_MAPPING" in page
    assert "MARK_UNRESOLVED_EXCLUDE" in page
    assert "Laterality exception" not in render_case(session, 1)


def test_snapshot_writes_latest_decisions_in_queue_order(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(2, _both("ACCEPT"))
    session.record_adjudication(1, _both("REJECT"))

    target = session.snapshot(target=tmp_path / "snapshot.parquet")

    frame = pd.read_parquet(target)
    assert frame["review_index"].tolist() == [1, 2]
    assert frame["screen_left_decision"].tolist() == ["REJECT", "ACCEPT"]


def test_missing_queue_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ReviewToolError, match="review queue does not exist"):
        ReviewSession(queue_path=tmp_path / "absent.parquet")


def test_adjudications_from_a_different_queue_are_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    changed = pd.read_parquet(session.queue_path)
    changed.loc[0, "automatic_state"] = "FAIL"
    changed.to_parquet(session.queue_path, index=False)

    resumed = ReviewSession(
        queue_path=session.queue_path,
        adjudication_path=session.adjudication_path,
        output_root=session.output_root,
    )

    with pytest.raises(ReviewToolError, match="different review queue"):
        resumed.progress()


def test_review_server_refuses_a_non_loopback_bind(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)

    with pytest.raises(ReviewToolError, match="loopback"):
        serve(queue_path=queue_path, output_root=queue_path.parent, host="0.0.0.0")


def _serve(session: ReviewSession):
    handler = type("BoundHandler", (ReviewRequestHandler,), {"session": session})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_workflow_serves_a_case_and_records_a_clicked_center(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, thread = _serve(session)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("GET", "/case/1")
        page = connection.getresponse()
        page_body = page.read().decode("utf-8")

        connection.request("GET", "/asset?case=1&name=screen_left_overlay")
        asset = connection.getresponse()
        asset_body = asset.read()

        connection.request("GET", "/exceptions")
        exceptions = connection.getresponse()
        exceptions.read()

        payload = urlencode(
            {
                "review_index": "1",
                "screen_left_decision": "NEEDS_CENTER_OVERRIDE",
                "screen_right_decision": "ACCEPT",
                "overrides": json.dumps({"screen_left": {"row": 640, "column": 400}}),
                "note": "center is too low",
                "allow_revision": "0",
            }
        )
        connection.request(
            "POST",
            "/decision",
            body=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        decision = connection.getresponse()
        decision.read()

        connection.request("GET", "/progress")
        progress = json.loads(connection.getresponse().read())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    recorded = session.adjudications()
    assert page.status == 200
    assert "V3 crop review" in page_body
    assert asset.status == 200
    assert asset_body.startswith(b"\x89PNG")
    assert exceptions.status == 200
    assert decision.status == 303
    assert decision.getheader("Location") == "/case/2"
    assert progress["reviewed_acquisitions"] == 1
    assert recorded[0]["screen_left_decision"] == "NEEDS_CENTER_OVERRIDE"
    assert recorded[0]["screen_right_decision"] == "ACCEPT"
    assert recorded[0]["screen_left_center_override_row"] == 640


def test_http_rejects_a_duplicate_decision_with_a_conflict(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_adjudication(1, _both("ACCEPT"))
    server, thread = _serve(session)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request(
            "POST",
            "/decision",
            body=urlencode(
                {
                    "review_index": "1",
                    "screen_left_decision": "REJECT",
                    "screen_right_decision": "REJECT",
                    "allow_revision": "0",
                }
            ),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 409
    assert len(session.adjudications()) == 1
