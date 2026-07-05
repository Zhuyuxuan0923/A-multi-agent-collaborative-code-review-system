"""PRDiffFetcher -- fetch and parse GitHub pull request diffs.

Retrieves the unified diff for a PR via the GitHub REST API and extracts
basic statistics: files changed, additions, and deletions.

Usage:
    result = PRDiffFetcher.fetch("https://github.com/owner/repo/pull/42", token="ghp_xxx")
    print(f"{result.files_changed} files, +{result.additions} -{result.deletions}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from study_agent.github.client import GitHubClient


@dataclass
class PRDiffResult:
    """The result of fetching and parsing a PR diff.

    Attributes:
        diff_raw: The raw unified diff text returned by the GitHub API.
        files_changed: Number of files touched in the PR.
        additions: Number of lines added (``+`` lines, excluding the
                   ``+++`` file markers).
        deletions: Number of lines removed (``-`` lines, excluding the
                   ``---`` file markers).
        error: Error description when the fetch fails. Empty string on success.
    """

    diff_raw: str = ""
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0
    error: str = ""


class PRDiffFetcher:
    """Fetches unified diffs for GitHub pull requests.

    All methods are classmethods -- no internal state, no instance needed.
    """

    # Matches https://github.com/<owner>/<repo>/pull/<number> with optional trailing /
    _URL_PATTERN = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?$")

    @classmethod
    def parse_pr_url(cls, pr_url: str) -> tuple[str, str, int] | None:
        """Extract owner, repo, and PR number from a GitHub PR URL.

        Args:
            pr_url: A GitHub pull request URL, e.g.
                    ``https://github.com/owner/repo/pull/42``.

        Returns:
            A ``(owner, repo, pr_number)`` tuple, or ``None`` if the URL
            does not match the expected pattern.
        """
        match = cls._URL_PATTERN.match(pr_url.strip())
        if not match:
            return None
        return (match.group(1), match.group(2), int(match.group(3)))

    @classmethod
    def fetch(cls, pr_url: str, token: str) -> PRDiffResult:
        """Fetch the unified diff for a GitHub pull request.

        Args:
            pr_url: Full PR URL, e.g. ``https://github.com/owner/repo/pull/42``.
            token: GitHub Personal Access Token for authentication.

        Returns:
            A ``PRDiffResult`` with the raw diff text and line-count statistics.
            On error the ``error`` field contains a human-readable description.
        """
        parsed = cls.parse_pr_url(pr_url)
        if parsed is None:
            return PRDiffResult(error=f"Invalid PR URL: {pr_url}")

        owner, repo, pr_number = parsed
        path = f"/repos/{owner}/{repo}/pulls/{pr_number}"

        client = GitHubClient(token=token)
        try:
            resp = client.get_raw(
                path,
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            if resp.status_code != 200:
                return PRDiffResult(
                    error=f"GitHub API returned {resp.status_code}: " f"{resp.text[:200].strip()}"
                )

            diff_text = resp.text

            # Count statistics from the unified diff.
            # "\n+" matches added lines, "\n+++" matches the file marker header.
            # "\n-" matches removed lines, "\n---" matches the file marker header.
            # Subtract these to exclude the marker lines from the counts.
            additions = diff_text.count("\n+") - diff_text.count("\n+++")
            deletions = diff_text.count("\n-") - diff_text.count("\n---")
            files_changed = diff_text.count("diff --git ")

            return PRDiffResult(
                diff_raw=diff_text,
                files_changed=files_changed,
                additions=max(0, additions),
                deletions=max(0, deletions),
            )
        except httpx.RequestError as exc:
            return PRDiffResult(error=f"Network error fetching PR diff: {exc}")
        finally:
            client.close()
