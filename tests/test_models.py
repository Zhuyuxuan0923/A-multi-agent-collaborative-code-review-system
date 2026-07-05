# mypy: disable-error-code="no-untyped-def"
"""Pydantic model tests -- request and response schemas for the code review API.

Covers:
  - ReviewRequest validation (min_length, default language)
  - PRReviewRequest required fields
  - TaskResponse serialization
  - ReportResponse with empty/partial data
  - ErrorResponse defaults
  - All enums
"""

import pytest
from pydantic import ValidationError

from study_agent.api.models import (
    ErrorResponse,
    Issue,
    PRReviewRequest,
    ReportResponse,
    ReviewRequest,
    TaskProgress,
    TaskResponse,
    TaskStatus,
    WebhookPayload,
)

# ── Enum tests ─────────────────────────────────────────────


def test_task_status_enum_values():
    """TaskStatus has four expected values."""
    assert TaskStatus.QUEUED.value == "queued"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"


def test_task_progress_enum_values():
    """TaskProgress has four expected values."""
    assert TaskProgress.REVIEWING.value == "reviewing"
    assert TaskProgress.RESEARCHING.value == "researching"
    assert TaskProgress.REPORTING.value == "reporting"
    assert TaskProgress.POSTING_COMMENT.value == "posting_comment"


# ── ReviewRequest ──────────────────────────────────────────


def test_review_request_validates_code_min_length():
    """Empty code string raises ValidationError."""
    with pytest.raises(ValidationError):
        ReviewRequest(code="")


def test_review_request_validates_code_max_length():
    """Code exceeding max_length raises ValidationError."""
    with pytest.raises(ValidationError):
        ReviewRequest(code="x" * 100001)


def test_review_request_defaults_language_to_python():
    """language field defaults to 'python' when omitted."""
    req = ReviewRequest(code="print('hello')")
    assert req.language == "python"


def test_review_request_optional_fields_none_by_default():
    """github_token and pr_url default to None."""
    req = ReviewRequest(code="print('hello')")
    assert req.github_token is None
    assert req.pr_url is None


def test_review_request_accepts_all_fields():
    """All fields can be set explicitly."""
    req = ReviewRequest(
        code="def foo(): pass",
        language="javascript",
        github_token="ghp_xxx",
        pr_url="https://github.com/owner/repo/pull/1",
    )
    assert req.code == "def foo(): pass"
    assert req.language == "javascript"
    assert req.github_token == "ghp_xxx"
    assert req.pr_url == "https://github.com/owner/repo/pull/1"


# ── PRReviewRequest ────────────────────────────────────────


def test_pr_review_request_requires_pr_url():
    """Missing pr_url raises ValidationError."""
    with pytest.raises(ValidationError):
        PRReviewRequest(github_token="ghp_xxx")


def test_pr_review_request_requires_github_token():
    """Missing github_token raises ValidationError."""
    with pytest.raises(ValidationError):
        PRReviewRequest(pr_url="https://github.com/owner/repo/pull/1")


def test_pr_review_request_with_valid_data():
    """Both required fields provided -> valid."""
    req = PRReviewRequest(
        pr_url="https://github.com/owner/repo/pull/42",
        github_token="ghp_secret123",
    )
    assert req.pr_url == "https://github.com/owner/repo/pull/42"
    assert req.github_token == "ghp_secret123"


def test_pr_review_request_rejects_empty_pr_url():
    """Empty pr_url string raises ValidationError."""
    with pytest.raises(ValidationError):
        PRReviewRequest(pr_url="", github_token="ghp_xxx")


def test_pr_review_request_rejects_empty_github_token():
    """Empty github_token string raises ValidationError."""
    with pytest.raises(ValidationError):
        PRReviewRequest(pr_url="https://github.com/owner/repo/pull/1", github_token="")


# ── WebhookPayload ─────────────────────────────────────────


def test_webhook_payload_defaults():
    """WebhookPayload has sensible defaults for all fields."""
    payload = WebhookPayload()
    assert payload.action == ""
    assert payload.pull_request is None
    assert payload.repository is None


def test_webhook_payload_with_data():
    """WebhookPayload accepts full webhook data."""
    payload = WebhookPayload(
        action="opened",
        pull_request={"number": 1, "title": "Fix bug"},
        repository={"full_name": "owner/repo"},
    )
    assert payload.action == "opened"
    assert payload.pull_request == {"number": 1, "title": "Fix bug"}
    assert payload.repository == {"full_name": "owner/repo"}


# ── TaskResponse ───────────────────────────────────────────


def test_task_response_serializes_to_dict_with_correct_keys():
    """TaskResponse model_dump() has all expected keys."""
    resp = TaskResponse(
        task_id="abc-123",
        status=TaskStatus.QUEUED,
        created_at="2025-01-01T00:00:00Z",
    )
    data = resp.model_dump()
    expected_keys = {"task_id", "status", "progress", "created_at", "result", "error"}
    assert expected_keys.issubset(set(data.keys()))
    assert data["task_id"] == "abc-123"
    assert data["status"] == "queued"
    assert data["progress"] is None
    assert data["result"] is None
    assert data["error"] is None


def test_task_response_with_progress_and_error():
    """TaskResponse can carry progress and error fields."""
    resp = TaskResponse(
        task_id="t-1",
        status=TaskStatus.FAILED,
        progress=TaskProgress.REVIEWING,
        created_at="2025-06-01T12:00:00Z",
        error="GitHub API rate limit exceeded",
    )
    assert resp.status == TaskStatus.FAILED
    assert resp.progress == TaskProgress.REVIEWING
    assert resp.error == "GitHub API rate limit exceeded"


# ── Issue ──────────────────────────────────────────────────


def test_issue_minimal_creation():
    """Issue requires only severity, category, and title."""
    issue = Issue(severity="Warning", category="style", title="Line too long")
    assert issue.severity == "Warning"
    assert issue.category == "style"
    assert issue.title == "Line too long"
    assert issue.line is None
    assert issue.suggestion == ""


def test_issue_full_creation():
    """Issue with all fields populated."""
    issue = Issue(
        severity="Critical",
        category="security",
        title="SQL injection risk",
        line=42,
        suggestion="Use parameterized queries.",
    )
    assert issue.line == 42
    assert issue.suggestion == "Use parameterized queries."


# ── ReportResponse ─────────────────────────────────────────


def test_report_response_handles_empty_issues_list():
    """ReportResponse default issues list is empty."""
    report = ReportResponse(
        task_id="rpt-1",
        report_md="# Review\nNo issues found.",
        score=100,
        reviewed_by=["bot-01"],
        created_at="2025-01-01T00:00:00Z",
    )
    assert report.issues == []
    assert report.technologies == []
    assert report.latency_ms == 0


def test_report_response_with_issues_and_technologies():
    """ReportResponse carries issues and technologies lists."""
    issues = [
        Issue(severity="Warning", category="style", title="Long line"),
        Issue(severity="Suggestion", category="perf", title="Use list comprehension"),
    ]
    report = ReportResponse(
        task_id="rpt-2",
        report_md="# Review\nFound 2 issues.",
        score=75,
        issues=issues,
        technologies=["Python", "FastAPI"],
        reviewed_by=["bot-01", "bot-02"],
        created_at="2025-02-02T00:00:00Z",
        latency_ms=3200,
    )
    assert len(report.issues) == 2
    assert report.issues[0].severity == "Warning"
    assert report.technologies == ["Python", "FastAPI"]
    assert report.score == 75
    assert report.latency_ms == 3200


# ── ErrorResponse ──────────────────────────────────────────


def test_error_response_creates_with_default_task_id_none():
    """ErrorResponse task_id defaults to None."""
    err = ErrorResponse(detail="Something went wrong")
    assert err.detail == "Something went wrong"
    assert err.task_id is None


def test_error_response_with_task_id():
    """ErrorResponse can include a task_id for tracing."""
    err = ErrorResponse(detail="Timeout", task_id="task-42")
    assert err.detail == "Timeout"
    assert err.task_id == "task-42"


# ── model_dump / JSON round-trip ───────────────────────────


def test_task_response_json_round_trip():
    """TaskResponse can round-trip through JSON."""
    resp = TaskResponse(
        task_id="t-99",
        status=TaskStatus.COMPLETED,
        progress=TaskProgress.REPORTING,
        created_at="2025-03-03T00:00:00Z",
        result={"score": 88},
    )
    json_str = resp.model_dump_json()
    assert '"task_id":"t-99"' in json_str
    assert '"status":"completed"' in json_str
    assert '"progress":"reporting"' in json_str


def test_enums_are_strings_in_json():
    """Enum values serialize as their string values, not enum names."""
    req = PRReviewRequest(
        pr_url="https://github.com/owner/repo/pull/1",
        github_token="ghp_xxx",
    )
    data = req.model_dump()
    assert isinstance(data["pr_url"], str)
    assert isinstance(data["github_token"], str)
