"""Tests for PRDiffFetcher -- fetch and parse GitHub PR diffs."""

from pytest_httpx import HTTPXMock

from study_agent.github.diff_fetcher import PRDiffFetcher, PRDiffResult

SAMPLE_DIFF = """diff --git a/app.py b/app.py
index 123..456 100644
--- a/app.py
+++ b/app.py
@@ -10,6 +10,8 @@ def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = " + user_id
+    query = "SELECT * FROM users WHERE id = ?"
+    cursor.execute(query, (user_id,))
"""

YOUR_REPO_URL = "https://github.com/testuser/testrepo/pull/1"


class TestPRDiffResult:
    def test_defaults(self) -> None:
        """All fields should have sensible defaults."""
        result = PRDiffResult()
        assert result.diff_raw == ""
        assert result.files_changed == 0
        assert result.additions == 0
        assert result.deletions == 0
        assert result.error == ""


class TestPRDiffFetcherParseUrl:
    def test_parse_pr_url(self) -> None:
        """Extract owner, repo, pr_number from valid URL."""
        result = PRDiffFetcher.parse_pr_url(YOUR_REPO_URL)
        assert result == ("testuser", "testrepo", 1)

    def test_parse_pr_url_with_trailing_slash(self) -> None:
        """Trailing slash should be handled."""
        result = PRDiffFetcher.parse_pr_url("https://github.com/testuser/testrepo/pull/1/")
        assert result == ("testuser", "testrepo", 1)

    def test_parse_pr_url_invalid(self) -> None:
        """Non-GitHub URL returns None."""
        assert PRDiffFetcher.parse_pr_url("https://gitlab.com/user/repo/-/merge_requests/1") is None
        assert PRDiffFetcher.parse_pr_url("not-a-url") is None
        assert PRDiffFetcher.parse_pr_url("") is None
        assert PRDiffFetcher.parse_pr_url("https://github.com/owner/pull/1") is None
        assert PRDiffFetcher.parse_pr_url("https://github.com/owner/repo/issues/1") is None


class TestPRDiffFetcherFetch:
    def test_fetch_diff(self, httpx_mock: HTTPXMock) -> None:
        """Mock GitHub API, verify diff_raw, files_changed, additions, deletions."""
        url = "https://api.github.com/repos/testuser/testrepo/pulls/1"
        httpx_mock.add_response(
            method="GET",
            url=url,
            text=SAMPLE_DIFF,
            status_code=200,
        )

        result = PRDiffFetcher.fetch(YOUR_REPO_URL, token="ghp_test")

        assert result.diff_raw == SAMPLE_DIFF
        assert result.files_changed == 1
        assert result.additions == 2
        assert result.deletions == 1
        assert result.error == ""

    def test_fetch_diff_pr_not_found(self, httpx_mock: HTTPXMock) -> None:
        """404 returns error string, not crash."""
        url = "https://api.github.com/repos/testuser/testrepo/pulls/999"
        httpx_mock.add_response(
            method="GET",
            url=url,
            status_code=404,
            json={"message": "Not Found"},
        )

        result = PRDiffFetcher.fetch(
            "https://github.com/testuser/testrepo/pull/999", token="ghp_test"
        )

        assert result.diff_raw == ""
        assert result.error != ""
        assert "404" in result.error

    def test_fetch_diff_invalid_url(self) -> None:
        """Invalid URL returns error, no API call made."""
        result = PRDiffFetcher.fetch(
            "https://gitlab.com/user/repo/-/merge_requests/1", token="ghp_test"
        )

        assert result.diff_raw == ""
        assert result.files_changed == 0
        assert result.additions == 0
        assert result.deletions == 0
        assert "Invalid PR URL" in result.error
