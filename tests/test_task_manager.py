"""Tests for TaskManager -- async task lifecycle + orchestrator integration.

Uses temporary file-based SQLite databases (not ``:memory:``) because TaskManager
opens a new connection for every operation. File-based DBs persist across connection
boundaries; ``:memory:`` would create a fresh database per connection.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from study_agent.api.task_manager import TaskManager, _hash_token

# -- Fixture helpers ----------------------------------------------------------


def _make_tm(tmp_path: Path) -> TaskManager:
    """Create a TaskManager backed by a temp-file DB (not :memory:)."""
    db = str(tmp_path / "test_reviews.db")
    return TaskManager(db_path=db)


# -- Fake data helpers --------------------------------------------------------


def _fake_review_result(
    score: int = 8,
    issues: list[dict[str, Any]] | None = None,
) -> tuple[object, ...]:
    """Build a mock return value for CodeReviewOrchestrator.review_with_intermediates."""
    from study_agent.agent.code_review_agents import ResearchResult, ReviewResult

    review = ReviewResult(
        summary="Code looks good overall.",
        score=score,
        issues=issues or [],
        raw_json='{"summary": "good", "score": 8, "issues": []}',
    )
    research = ResearchResult(
        technologies=["Python"],
        best_practices=[{"title": "Use type hints", "description": "PEP 484"}],
        common_pitfalls=[],
        recommendations=["Add tests"],
        references=["https://docs.python.org/3/"],
        raw_json='{"technologies": ["Python"]}',
    )
    report_md = "# Code Review Report\n\nOverall score: 8/10"
    return review, research, report_md


def _sample_issues() -> list[dict[str, Any]]:
    """Return a small set of sample issues for testing."""
    return [
        {
            "severity": "Critical",
            "category": "安全漏洞",
            "line": 10,
            "title": "SQL Injection",
            "description": "用户输入直接拼接到 SQL",
            "suggestion": "使用参数化查询",
        },
        {
            "severity": "Warning",
            "category": "性能问题",
            "line": 20,
            "title": "N+1 query",
            "description": "循环内执行查询",
            "suggestion": "使用 JOIN 批量加载",
        },
    ]


# ============================================================================
# Sync tests
# ============================================================================


class TestTaskManagerSync:
    """Tests that require no event loop (pure DB operations)."""

    def test_get_task_returns_none_for_missing(self, tmp_path: Path) -> None:
        """get_task returns None for a task_id that does not exist."""
        tm = _make_tm(tmp_path)
        assert tm.get_task("nonexistent-id") is None

    def test_get_report_returns_none_for_nonexistent(self, tmp_path: Path) -> None:
        """get_report returns None when the task does not exist."""
        tm = _make_tm(tmp_path)
        assert tm.get_report("nonexistent-id") is None

    def test_submit_pr_review_with_bad_url_creates_failed_task(self, tmp_path: Path) -> None:
        """Invalid PR URL creates a task with status 'failed' and an error message."""
        tm = _make_tm(tmp_path)
        task_id = tm.submit_pr_review("not-a-valid-github-url", "fake-token-12345")
        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "failed"
        assert "Invalid" in task.get("error", "")
        assert "not-a-valid-github-url" in task.get("error", "")


class TestHashToken:
    """Tests for the _hash_token helper."""

    def test_produces_consistent_hash(self) -> None:
        """Same input produces the same SHA256 hex digest."""
        h1 = _hash_token("my-token")
        h2 = _hash_token("my-token")
        assert h1 == h2

    def test_output_is_64_char_hex(self) -> None:
        """SHA256 hex digest is always 64 characters."""
        h = _hash_token("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_raw_token_not_in_hash(self) -> None:
        """The hash output must not contain the raw token string."""
        token = "secret-github-token-abc123"
        h = _hash_token(token)
        assert token not in h

    def test_different_tokens_produce_different_hashes(self) -> None:
        """Different inputs must produce different hashes."""
        h1 = _hash_token("token-a")
        h2 = _hash_token("token-b")
        assert h1 != h2


# ============================================================================
# Async tests
# ============================================================================


class TestTaskManagerAsync:
    """Async tests that call submit_review and exercise the background task flow.

    The orchestrator is always mocked so no real LLM calls are made.
    """

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_submit_review_creates_task_and_returns_12_char_id(self, tmp_path: Path) -> None:
        """submit_review creates a task in the DB and returns a 12-char hex id."""
        tm = _make_tm(tmp_path)
        with patch.object(
            tm.orchestrator, "review_with_intermediates", return_value=_fake_review_result()
        ):
            task_id = tm.submit_review(code="def foo(): pass", language="python")
            bg = tm._running_tasks.get(task_id)
            if bg:
                await bg

        assert len(task_id) == 12
        task = tm.get_task(task_id)
        assert task is not None
        assert task["code"] == "def foo(): pass"
        assert task["language"] == "python"
        assert task["type"] == "review"

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_review_completes_and_persists_results(self, tmp_path: Path) -> None:
        """A successful review updates status to completed and stores all results."""
        tm = _make_tm(tmp_path)
        fake_return = _fake_review_result(score=9, issues=_sample_issues())
        with patch.object(tm.orchestrator, "review_with_intermediates", return_value=fake_return):
            task_id = tm.submit_review(code="x = 1")
            bg = tm._running_tasks.get(task_id)
            if bg:
                await bg

        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"
        assert task["score"] == 9
        assert "# Code Review Report" in task.get("report_md", "")
        assert task["completed_at"] is not None

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_get_report_returns_full_data_for_completed_task(self, tmp_path: Path) -> None:
        """get_report returns structured report dict when task is completed."""
        tm = _make_tm(tmp_path)
        with patch.object(
            tm.orchestrator, "review_with_intermediates", return_value=_fake_review_result()
        ):
            task_id = tm.submit_review(code="print(1)")
            bg = tm._running_tasks.get(task_id)
            if bg:
                await bg

        report = tm.get_report(task_id)
        assert report is not None
        assert report["task_id"] == task_id
        assert "report_md" in report
        assert report["score"] == 8
        assert isinstance(report["issues"], list)
        assert "reviewer" in report["reviewed_by"]
        assert "researcher" in report["reviewed_by"]
        assert "reporter" in report["reviewed_by"]

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_get_report_returns_none_for_incomplete_task(self, tmp_path: Path) -> None:
        """get_report returns None when the task status is not 'completed'."""
        tm = _make_tm(tmp_path)

        # Use a threading.Event to block the mock inside asyncio.to_thread.
        # (The mock is a regular sync function running in a thread, so
        # asyncio.Event won't work across the thread boundary.)
        blocker = threading.Event()

        def slow_review(*args: object, **kwargs: object) -> tuple[object, ...]:
            blocker.wait()
            return _fake_review_result()

        with patch.object(tm.orchestrator, "review_with_intermediates", side_effect=slow_review):
            task_id = tm.submit_review(code="x = 1")
            # Give the background task a moment to start
            await asyncio.sleep(0.05)

        # Task should still be running (blocker not set yet)
        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] in ("queued", "running")
        assert tm.get_report(task_id) is None

        # Clean up: unblock and let the task finish
        blocker.set()
        bg = tm._running_tasks.get(task_id)
        if bg:
            await bg

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_review_error_sets_failed_status(self, tmp_path: Path) -> None:
        """If _run_review raises, the task is marked as 'failed' with the error."""
        tm = _make_tm(tmp_path)
        with patch.object(
            tm.orchestrator,
            "review_with_intermediates",
            side_effect=RuntimeError("LLM timeout"),
        ):
            task_id = tm.submit_review(code="x = 1")
            bg = tm._running_tasks.get(task_id)
            if bg:
                await bg

        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "failed"
        assert "LLM timeout" in task.get("error", "")

    @pytest.mark.asyncio  # type: ignore[untyped-decorator]
    async def test_submit_pr_review_with_successful_fetch_creates_background_task(
        self, tmp_path: Path
    ) -> None:
        """When PRDiffFetcher returns a diff successfully, a background task is spawned."""
        tm = _make_tm(tmp_path)
        fake_diff = MagicMock()
        fake_diff.diff_raw = "diff --git a/x.py b/x.py\n+print(1)\n-old"
        fake_diff.error = ""

        with (
            patch(
                "study_agent.api.task_manager.PRDiffFetcher.fetch", return_value=fake_diff
            ) as mock_fetch,
            patch.object(
                tm.orchestrator,
                "review_with_intermediates",
                return_value=_fake_review_result(),
            ),
        ):
            task_id = tm.submit_pr_review("https://github.com/owner/repo/pull/42", "fake-token")
            mock_fetch.assert_called_once_with(
                "https://github.com/owner/repo/pull/42", "fake-token"
            )
            bg = tm._running_tasks.get(task_id)
            if bg:
                await bg

        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"
        assert task["pr_url"] == "https://github.com/owner/repo/pull/42"
