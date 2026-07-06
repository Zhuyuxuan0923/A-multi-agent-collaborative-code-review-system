"""Tests for the Code Review API server -- all 5 endpoints.

Uses FastAPI TestClient with a mocked TaskManager so no real LLM calls
or database operations are performed.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from study_agent.api.server import app


def _make_mock_tm(**overrides: object) -> MagicMock:
    """Build a MagicMock TaskManager with sensible defaults for every method
    the server calls. Individual tests can override specific return values.
    """
    tm = MagicMock()
    tm.submit_review.return_value = "abc123def456"
    tm.submit_pr_review.return_value = "pr001pr001aa"
    tm.get_task.return_value = {
        "id": "abc123def456",
        "status": "completed",
        "progress": None,
        "created_at": "2026-07-05T10:00:00Z",
        "error": None,
    }
    tm.get_report.return_value = {
        "task_id": "abc123def456",
        "report_md": "# Code Review Report\n\nScore: 8/10",
        "score": 8,
        "issues": [
            {
                "severity": "Warning",
                "category": "Performance",
                "title": "N+1 query",
                "line": 20,
                "suggestion": "Use JOIN instead",
            }
        ],
        "technologies": ["Python", "FastAPI"],
        "reviewed_by": ["reviewer", "researcher", "reporter"],
        "created_at": "2026-07-05T10:00:05Z",
        "latency_ms": 5000,
    }
    # Apply per-test overrides
    for attr, value in overrides.items():
        setattr(tm, attr, value)
    return tm


# ============================================================================
# GET /health
# ============================================================================


def test_health_returns_ok() -> None:
    """Health check endpoint always returns status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ============================================================================
# POST /api/review  (manual code submit)
# ============================================================================


def test_submit_review_returns_202() -> None:
    """Valid code submission returns 202 Accepted with task_id."""
    mock_tm = _make_mock_tm()
    mock_tm.submit_review.return_value = "abcd1234abcd"

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/review",
            json={"code": "def foo(): pass", "language": "python"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "abcd1234abcd"
    assert data["status"] == "queued"


def test_submit_review_passes_all_fields_to_task_manager() -> None:
    """All optional fields (github_token, pr_url) are forwarded to TaskManager."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        client.post(
            "/api/review",
            json={
                "code": "x = 1",
                "language": "python",
                "github_token": "ghp_test123",
                "pr_url": "https://github.com/a/b/pull/1",
            },
        )

    mock_tm.submit_review.assert_called_once_with(
        code="x = 1",
        language="python",
        github_token="ghp_test123",
        pr_url="https://github.com/a/b/pull/1",
    )


def test_submit_review_empty_code_returns_422() -> None:
    """Pydantic validation rejects empty code string."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post("/api/review", json={"code": ""})

    assert response.status_code == 422


def test_submit_review_missing_code_returns_422() -> None:
    """Missing required field 'code' returns 422."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post("/api/review", json={"language": "go"})

    assert response.status_code == 422


# ============================================================================
# POST /api/review/pr  (PR URL submit)
# ============================================================================


def test_submit_pr_review_returns_202() -> None:
    """Valid PR URL submission returns 202 with task_id."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/review/pr",
            json={
                "pr_url": "https://github.com/owner/repo/pull/42",
                "github_token": "ghp_token123",
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "pr001pr001aa"
    assert data["status"] == "queued"


def test_submit_pr_review_missing_token_returns_422() -> None:
    """Missing github_token returns 422 validation error."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/review/pr",
            json={"pr_url": "https://github.com/owner/repo/pull/42"},
        )

    assert response.status_code == 422


def test_submit_pr_review_failed_fetch_returns_400() -> None:
    """When PRDiffFetcher fails, the task is created as failed and 400 is returned."""
    mock_tm = _make_mock_tm()
    mock_tm.submit_pr_review.return_value = "failed123abc"
    # Simulate a task that was created but immediately failed
    mock_tm.get_task.return_value = {
        "id": "failed123abc",
        "status": "failed",
        "error": "Failed to fetch PR diff: GitHub API returned 404: Not Found",
        "created_at": "2026-07-05T10:00:00Z",
        "progress": None,
    }

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/review/pr",
            json={
                "pr_url": "https://github.com/fake/fake/pull/1",
                "github_token": "ghp_faketoken12345",
            },
        )

    assert response.status_code == 400
    assert "Failed to fetch PR diff" in response.json()["detail"]


# ============================================================================
# POST /api/webhook/github  (webhook receiver)
# ============================================================================


def test_webhook_pull_request_event_returns_202() -> None:
    """Valid pull_request event creates a review and returns 202."""
    mock_tm = _make_mock_tm()
    mock_tm.submit_pr_review.return_value = "webhook00001"

    payload = {
        "action": "opened",
        "pull_request": {
            "html_url": "https://github.com/owner/repo/pull/99",
            "title": "Add new feature",
        },
        "repository": {"full_name": "owner/repo"},
    }

    with (
        patch("study_agent.api.server.task_manager", mock_tm),
        patch.dict(os.environ, {"GITHUB_BOT_TOKEN": "ghp_bot_token"}),
    ):
        client = TestClient(app)
        response = client.post(
            "/api/webhook/github",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "webhook00001"
    assert data["status"] == "queued"
    assert data["event"] == "pull_request"


def test_webhook_with_event_header_but_no_pr_data() -> None:
    """pull_request event without PR data returns 200 with message."""
    mock_tm = _make_mock_tm()

    payload = {"action": "opened"}

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/webhook/github",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "no PR data" in data["message"]


def test_webhook_non_pr_event_returns_200_with_message() -> None:
    """push events (not PR) are ignored gracefully."""
    mock_tm = _make_mock_tm()

    payload = {"ref": "refs/heads/main", "commits": []}

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/webhook/github",
            json=payload,
            headers={"X-GitHub-Event": "push"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "ignored" in data["message"]
    assert data["event"] == "push"


def test_webhook_invalid_json_returns_400() -> None:
    """Malformed JSON body returns 400."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.post(
            "/api/webhook/github",
            content=b"not valid json {{{",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


# ============================================================================
# GET /api/task/{task_id}  (task status)
# ============================================================================


def test_get_task_returns_task_response() -> None:
    """Existing task returns its status fields."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.get("/api/task/abc123def456")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "abc123def456"
    assert data["status"] == "completed"
    assert data["created_at"] == "2026-07-05T10:00:00Z"


def test_get_task_not_found_returns_404() -> None:
    """Non-existent task returns 404."""
    mock_tm = _make_mock_tm()
    mock_tm.get_task.return_value = None

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.get("/api/task/nonexistent")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ============================================================================
# GET /api/report/{task_id}  (review report)
# ============================================================================


def test_get_report_returns_full_report() -> None:
    """Completed task returns the full structured report."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.get("/api/report/abc123def456")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "abc123def456"
    assert data["report_md"].startswith("# Code Review Report")
    assert data["score"] == 8
    assert len(data["issues"]) == 1
    assert data["issues"][0]["severity"] == "Warning"
    assert "Python" in data["technologies"]
    assert "reviewer" in data["reviewed_by"]


def test_get_report_task_not_found_returns_404() -> None:
    """Non-existent task returns 404."""
    mock_tm = _make_mock_tm()
    mock_tm.get_report.return_value = None
    mock_tm.get_task.return_value = None  # task doesn't exist at all

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.get("/api/report/nonexistent")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_report_not_ready_returns_404() -> None:
    """Task exists but review is not completed yet -- returns 404 with status info."""
    mock_tm = _make_mock_tm()
    mock_tm.get_report.return_value = None
    mock_tm.get_task.return_value = {
        "id": "running123ab",
        "status": "running",
        "progress": "reviewing",
        "created_at": "2026-07-05T10:00:00Z",
        "error": None,
    }

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        response = client.get("/api/report/running123ab")

    assert response.status_code == 404
    assert "Report not ready" in response.json()["detail"]


# ============================================================================
# Edge cases
# ============================================================================


def test_submit_review_passes_github_token_and_pr_url_as_none_by_default() -> None:
    """When github_token and pr_url are omitted, they are passed as None (defaults)."""
    mock_tm = _make_mock_tm()

    with patch("study_agent.api.server.task_manager", mock_tm):
        client = TestClient(app)
        client.post("/api/review", json={"code": "x = 1"})

    mock_tm.submit_review.assert_called_once_with(
        code="x = 1",
        language="python",
        github_token=None,
        pr_url=None,
    )
