"""Synthetic keyboard tests: real rendered script and existing HTTP transaction path.

The script harness uses Node >=18's built-in VM/fetch, without npm dependencies. No test uses
production queue/adjudication paths. Actual browser rendering is covered by the separate smoke test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import build_review_queue

from imaging.artifact_io import sha256_file
from imaging.v3_review_ui import (
    PanelDecision,
    ReviewRequestHandler,
    ReviewSession,
    render_case,
)

pytestmark = pytest.mark.public_portable


@pytest.fixture
def rapid_session(tmp_path: Path) -> ReviewSession:
    queue = build_review_queue(tmp_path, cases=4)
    return ReviewSession(
        queue_path=queue,
        adjudication_path=queue.parent / "review/adjudications.jsonl",
        output_root=queue.parent,
    )


def _accept(session: ReviewSession, index: int) -> None:
    session.record_adjudication(
        index,
        tuple(PanelDecision(position, "ACCEPT") for position in ("screen_left", "screen_right")),
    )


def _run(
    session: ReviewSession,
    actions: list[dict],
    *,
    index: int = 1,
    conflict: bool = False,
    server_error: bool = False,
) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Rapid-review script tests require Node >=18 (no npm packages)")
    page = render_case(session, index)
    if conflict:
        # A different writer commits after this page was rendered; its expected sequence is stale.
        _accept(session, index)
    posts = []

    class Handler(ReviewRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib signature
            posts.append(self.path)
            if server_error:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "synthetic persistence server error")
            else:
                super().do_POST()

    Handler.session = session
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [node, str(Path(__file__).with_name("rapid_review_harness.cjs"))],
            input=json.dumps(
                {
                    "html": page,
                    "base": f"http://127.0.0.1:{server.server_port}",
                    "actions": actions,
                    "delay": 20,
                }
            ),
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )
        observed = json.loads(result.stdout)
        observed["posts"] = posts
        return observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_accepts_both_in_one_existing_transaction_and_skips_decided_cases(
    rapid_session: ReviewSession,
) -> None:
    _accept(rapid_session, 2)
    previous = rapid_session.adjudication_path.read_bytes()
    result = _run(rapid_session, [{"key": "a"}])
    assert result["posts"] == ["/decision"]
    assert len(result["requests"]) == 1
    assert result["selected"] == ["ACCEPT", "ACCEPT"]
    assert result["overrides"] == {}
    assert result["navigations"] == ["/case/3"]
    assert result["requests"][0]["saving"] == "Saving..."
    assert result["requests"][0]["allDisabled"] is True
    assert rapid_session.adjudication_path.read_bytes().startswith(previous)
    records = rapid_session.adjudications()
    assert len(records) == 2
    new = next(record for record in records if record["review_index"] == 1)
    assert new["screen_left_decision"] == new["screen_right_decision"] == "ACCEPT"


def test_repeat_double_keypress_and_manual_submit_during_save_make_one_write(
    rapid_session: ReviewSession,
) -> None:
    result = _run(
        rapid_session,
        [
            {"key": "a"},
            {"key": "a", "repeat": True},
            {"key": "a"},
            {"key": "Enter"},
            {"submit": True},
            {"key": "n"},
            {"key": "Escape"},
        ],
    )
    assert result["posts"] == ["/decision"]
    assert len(rapid_session.adjudications()) == 1
    assert result["navigations"] == ["/case/2"]
    assert result["selected"] == ["ACCEPT", "ACCEPT"]


@pytest.mark.parametrize("focus", ["textarea", "input", "select", "editable", "editable-child"])
@pytest.mark.parametrize("key", ["a", "Enter", "n", "p", "Escape"])
def test_shortcuts_do_nothing_in_editable_fields(
    rapid_session: ReviewSession, focus: str, key: str
) -> None:
    result = _run(rapid_session, [{"note": "unsaved"}, {"key": key, "focus": focus}])
    assert result["posts"] == []
    assert result["navigations"] == []
    assert result["note"] == "unsaved"
    assert rapid_session.adjudications() == []


def test_a_does_not_replace_a_pending_corrected_center(rapid_session: ReviewSession) -> None:
    result = _run(rapid_session, [{"center": "screen_left"}, {"key": "a"}])
    assert result["posts"] == []
    assert result["overrides"] == {"screen_left": {"row": 610, "column": 430}}
    assert result["selected"] == ["NEEDS_CENTER_OVERRIDE", None]
    assert "Clear" in result["overrideState"]


def test_enter_saves_valid_selections_and_advances(rapid_session: ReviewSession) -> None:
    result = _run(
        rapid_session,
        [
            {"select": "screen_left", "value": "ACCEPT"},
            {"select": "screen_right", "value": "ACCEPT"},
            {"key": "Enter"},
        ],
    )
    assert result["posts"] == ["/decision"]
    assert result["navigations"] == ["/case/2"]
    assert len(rapid_session.adjudications()) == 1


def test_enter_requires_a_corrected_center(rapid_session: ReviewSession) -> None:
    result = _run(
        rapid_session,
        [
            {"select": "screen_left", "value": "NEEDS_CENTER_OVERRIDE"},
            {"select": "screen_right", "value": "ACCEPT"},
            {"key": "Enter"},
        ],
    )
    assert result["posts"] == []
    assert result["navigations"] == []
    assert "corrected center" in result["overrideState"]


@pytest.mark.parametrize("mode", ["conflict", "server_error"])
def test_persistence_failure_keeps_case_and_visible_state(
    rapid_session: ReviewSession, mode: str
) -> None:
    result = _run(rapid_session, [{"note": "retain me"}, {"key": "a"}], **{mode: True})
    assert result["posts"] == ["/decision"]
    assert result["navigations"] == []
    assert result["selected"] == ["ACCEPT", "ACCEPT"]
    assert result["note"] == "retain me"
    assert result["error"]
    assert result["status"] == ""
    assert result["disabled"] is False
    assert len(rapid_session.adjudications()) == (1 if mode == "conflict" else 0)


@pytest.mark.parametrize("key, expected", [("n", "/case/3"), ("p", "/case/1")])
def test_navigation_shortcuts_never_write(
    rapid_session: ReviewSession, key: str, expected: str
) -> None:
    result = _run(rapid_session, [{"key": key}], index=2)
    assert result["posts"] == []
    assert result["navigations"] == [expected]
    assert rapid_session.adjudications() == []


def test_escape_clears_unsaved_state_but_restores_persisted_decisions(
    rapid_session: ReviewSession,
) -> None:
    _accept(rapid_session, 1)
    original = rapid_session.adjudication_path.read_bytes()
    result = _run(
        rapid_session,
        [{"center": "screen_left"}, {"note": "unsaved change"}, {"key": "Escape"}],
    )
    assert result["posts"] == result["navigations"] == []
    assert result["selected"] == ["ACCEPT", "ACCEPT"]
    assert result["overrides"] == {}
    assert result["note"] == ""
    assert rapid_session.adjudication_path.read_bytes() == original


def test_escape_clears_a_clean_pages_unsaved_selections(rapid_session: ReviewSession) -> None:
    result = _run(
        rapid_session,
        [
            {"center": "screen_left"},
            {"select": "screen_right", "value": "ACCEPT"},
            {"key": "Escape"},
        ],
    )
    assert result["selected"] == [None, None]
    assert result["overrides"] == {}
    assert result["posts"] == []


def test_a_does_not_silently_revise_a_persisted_case(rapid_session: ReviewSession) -> None:
    _accept(rapid_session, 1)
    original = sha256_file(rapid_session.adjudication_path)
    result = _run(rapid_session, [{"key": "a"}])
    assert result["posts"] == []
    assert "already recorded" in result["overrideState"]
    assert sha256_file(rapid_session.adjudication_path) == original


@pytest.mark.parametrize("key", ["a", "Enter"])
def test_laterality_exception_cannot_be_bypassed(rapid_session: ReviewSession, key: str) -> None:
    result = _run(
        rapid_session,
        [
            {"select": "screen_left", "value": "ACCEPT"},
            {"select": "screen_right", "value": "ACCEPT"},
            {"key": key},
        ],
        index=3,
    )
    assert result["posts"] == result["navigations"] == []
    assert "laterality decision" in result["overrideState"]


def test_explicit_laterality_decision_allows_valid_keyboard_save(
    rapid_session: ReviewSession,
) -> None:
    result = _run(
        rapid_session,
        [{"laterality": "MARK_UNRESOLVED_EXCLUDE"}, {"key": "a"}],
        index=3,
    )
    assert result["posts"] == ["/decision"]
    assert result["navigations"] == ["/case/4"]
    assert rapid_session.adjudications()[0]["laterality_decision"] == "MARK_UNRESOLVED_EXCLUDE"


def test_manual_control_and_enter_use_the_same_submission_action(
    rapid_session: ReviewSession,
) -> None:
    result = _run(
        rapid_session,
        [
            {"center": "screen_left"},
            {"select": "screen_right", "value": "ACCEPT"},
            {"submit": True},
        ],
    )
    assert result["posts"] == ["/decision"]
    assert result["navigations"] == ["/case/2"]
    assert rapid_session.adjudications()[0]["screen_left_center_override_row"] == 610
