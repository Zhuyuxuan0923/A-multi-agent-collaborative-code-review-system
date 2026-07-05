"""CommentPoster -- post code review summary as a GitHub PR comment.

Parses a PR URL, builds a Markdown-formatted review comment body with truncation
support for the GitHub API's 65,536-character limit, and posts it via the
GitHub REST API.

Usage:
    result = CommentPoster.post(
        "https://github.com/owner/repo/pull/42",
        token="ghp_xxx",
        report_md="## Findings\n\n- SQL injection in login\n",
        score=6,
        issues_count=2,
    )
    if result.success:
        print(f"Comment posted: {result.comment_url}")
"""

from __future__ import annotations

import httpx

from study_agent.github.client import GitHubClient
from study_agent.github.diff_fetcher import PRDiffFetcher

MAX_COMMENT_LENGTH = 65536


class CommentResult:
    """Result of a comment-posting operation.

    Attributes:
        success: ``True`` when the comment was created successfully.
        comment_url: The ``html_url`` of the created comment (empty on failure).
        error: Human-readable error description (empty on success).
    """

    def __init__(
        self,
        success: bool,
        comment_url: str = "",
        error: str = "",
    ) -> None:
        self.success = success
        self.comment_url = comment_url
        self.error = error


class CommentPoster:
    """Posts a code-review summary as a comment on a GitHub Pull Request.

    All methods are classmethods -- no internal state, no instance needed.
    """

    @staticmethod
    def build_comment_body(report_md: str, score: int, issues_count: int) -> str:
        """Build a Markdown comment body for posting to a PR.

        Constructs a header with score and issue count, then appends the report.
        If the total exceeds ``MAX_COMMENT_LENGTH`` (GitHub's API limit), the
        report is truncated and a ``(truncated)`` note is appended.

        Args:
            report_md: The Markdown review report to include.
            score: Review score out of 10.
            issues_count: Number of issues found during review.

        Returns:
            A Markdown string suitable for posting as a GitHub PR comment.
        """
        header = (
            "## Code Review by AI Agent\n\n"
            f"**Score: {score}/10** | **Issues found: {issues_count}**\n\n"
            "---\n\n"
        )

        truncated_note = "\n\n*(truncated)*"
        available_for_report = MAX_COMMENT_LENGTH - len(header) - len(truncated_note)

        if len(report_md) <= available_for_report:
            return header + report_md

        # Truncate the report to fit within the limit.
        truncated_report = report_md[:available_for_report]
        return header + truncated_report + truncated_note

    @classmethod
    def post(
        cls,
        pr_url: str,
        token: str,
        report_md: str,
        score: int,
        issues_count: int,
    ) -> CommentResult:
        """Post a review summary comment on a GitHub Pull Request.

        Args:
            pr_url: Full PR URL, e.g.
                    ``https://github.com/owner/repo/pull/42``.
            token: GitHub Personal Access Token for authentication.
            report_md: The Markdown review report body.
            score: Review score out of 10.
            issues_count: Number of issues found.

        Returns:
            A ``CommentResult`` indicating success or failure.
            On success, ``comment_url`` holds the ``html_url`` of the comment.
        """
        parsed = PRDiffFetcher.parse_pr_url(pr_url)
        if parsed is None:
            return CommentResult(
                success=False,
                error=f"Invalid PR URL: {pr_url}",
            )

        owner, repo, pr_number = parsed
        body = cls.build_comment_body(report_md, score, issues_count)

        client = GitHubClient(token=token)
        try:
            path = f"/repos/{owner}/{repo}/issues/{pr_number}/comments"
            data = client.post(path, json_data={"body": body})
            return CommentResult(
                success=True,
                comment_url=data["html_url"],
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return CommentResult(
                success=False,
                error=f"GitHub API returned {status}: {exc.response.text[:200].strip()}",
            )
        finally:
            client.close()
