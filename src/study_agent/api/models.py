# mypy: disable-error-code="misc"
"""Pydantic models for the code review API.

Defines all request/response schemas and enums used by the 5 API endpoints:
  - POST /review          (manual code submit)
  - POST /review/pr       (PR URL submit)
  - GET  /task/{task_id}  (task status)
  - GET  /report/{task_id} (review report)
  - POST /webhook         (GitHub webhook)
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """Lifecycle status of a review task."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskProgress(StrEnum):
    """Granular progress within a running task."""

    REVIEWING = "reviewing"
    RESEARCHING = "researching"
    REPORTING = "reporting"
    POSTING_COMMENT = "posting_comment"


# ── Request Models ─────────────────────────────────────────


class ReviewRequest(BaseModel):
    """Manual code submit -- user pastes code directly."""

    code: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Source code to review",
    )
    language: str = Field(
        default="python",
        description="Programming language",
    )
    github_token: str | None = Field(
        default=None,
        description="Optional: GitHub PAT for PR comment",
    )
    pr_url: str | None = Field(
        default=None,
        description="Optional: related PR URL",
    )


class PRReviewRequest(BaseModel):
    """PR URL submit -- system fetches diff from GitHub."""

    pr_url: str = Field(
        ...,
        min_length=1,
        description="GitHub PR URL",
    )
    github_token: str = Field(
        ...,
        min_length=1,
        description="GitHub Personal Access Token",
    )


class WebhookPayload(BaseModel):
    """GitHub webhook event (simplified)."""

    action: str = Field(
        default="",
        description="Event action: opened, synchronize, etc.",
    )
    pull_request: dict[str, Any] | None = Field(
        default=None,
        description="PR data from webhook",
    )
    repository: dict[str, Any] | None = Field(
        default=None,
        description="Repository data",
    )


# ── Response Models ────────────────────────────────────────


class Issue(BaseModel):
    """A single issue found during code review."""

    severity: str  # Critical / Warning / Suggestion
    category: str
    title: str
    line: int | None = None
    suggestion: str = ""


class TaskResponse(BaseModel):
    """Returned by GET /task/{task_id} -- current status of a review task."""

    task_id: str
    status: TaskStatus
    progress: TaskProgress | None = None
    created_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


class ReportResponse(BaseModel):
    """Returned by GET /report/{task_id} -- full review report."""

    task_id: str
    report_md: str
    score: int
    issues: list[Issue] = []
    technologies: list[str] = []
    reviewed_by: list[str] = []
    created_at: str
    latency_ms: int = 0


class ErrorResponse(BaseModel):
    """Standard error response for all endpoints."""

    detail: str
    task_id: str | None = None
