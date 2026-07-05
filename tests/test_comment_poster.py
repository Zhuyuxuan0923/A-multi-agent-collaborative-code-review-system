"""Tests for CommentPoster -- post review summary as PR comment."""

from pytest_httpx import HTTPXMock

from study_agent.github.comment_poster import CommentPoster, CommentResult

PR_URL = "https://github.com/testuser/testrepo/pull/1"


class TestCommentResult:
    def test_defaults(self) -> None:
        """All fields should have sensible defaults."""
        result = CommentResult(success=False)
        assert result.success is False
        assert result.comment_url == ""
        assert result.error == ""


class TestBuildCommentBody:
    def test_build_comment_body_short(self) -> None:
        """Body includes score, issues_count, and report."""
        report = "## Findings\n\n- Issue 1: SQL injection\n- Issue 2: Missing error handling\n"
        body = CommentPoster.build_comment_body(report, score=7, issues_count=3)

        assert "Code Review by AI Agent" in body
        assert "Score: 7/10" in body
        assert "Issues found: 3" in body
        assert "---" in body
        assert report in body

    def test_build_comment_body_truncates_long_report(self) -> None:
        """Report > 65KB is truncated with note."""
        # Build a report that is definitely over 65,536 characters.
        long_line = "x" * 80 + "\n"
        chunks_needed = 66000 // len(long_line) + 10
        long_report = long_line * chunks_needed

        assert len(long_report) > 65536, "test data must exceed the limit"

        body = CommentPoster.build_comment_body(long_report, score=5, issues_count=10)

        assert len(body) <= 65536
        assert "(truncated)" in body
        assert "Code Review by AI Agent" in body  # header still present


class TestPostComment:
    def test_post_comment_success(self, httpx_mock: HTTPXMock) -> None:
        """POST succeeds, returns comment URL."""
        api_url = "https://api.github.com/repos/testuser/testrepo/issues/1/comments"
        httpx_mock.add_response(
            method="POST",
            url=api_url,
            json={"html_url": "https://github.com/testuser/testrepo/pull/1#issuecomment-123"},
            status_code=201,
        )

        report = "## Findings\n\n- Issue 1\n"
        result = CommentPoster.post(
            PR_URL, token="ghp_test", report_md=report, score=8, issues_count=1
        )

        assert result.success is True
        assert result.comment_url == "https://github.com/testuser/testrepo/pull/1#issuecomment-123"
        assert result.error == ""

    def test_post_comment_invalid_pr_url(self) -> None:
        """Invalid URL returns error without API call."""
        result = CommentPoster.post(
            "https://gitlab.com/user/repo/-/merge_requests/1",
            token="ghp_test",
            report_md="irrelevant",
            score=0,
            issues_count=0,
        )

        assert result.success is False
        assert result.comment_url == ""
        assert "Invalid PR URL" in result.error

    def test_post_comment_api_error(self, httpx_mock: HTTPXMock) -> None:
        """403 returns error, not crash."""
        api_url = "https://api.github.com/repos/testuser/testrepo/issues/1/comments"
        httpx_mock.add_response(
            method="POST",
            url=api_url,
            status_code=403,
            json={"message": "Resource not accessible by integration"},
        )

        report = "## Findings\n\n- Issue 1\n"
        result = CommentPoster.post(
            PR_URL, token="ghp_test", report_md=report, score=8, issues_count=1
        )

        assert result.success is False
        assert result.comment_url == ""
        assert "403" in result.error
