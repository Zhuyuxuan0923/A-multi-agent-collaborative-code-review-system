"""Tests for GitHubClient -- shared HTTP layer for GitHub API calls."""

import httpx
import pytest
from pytest_httpx import HTTPXMock

from study_agent.github.client import GitHubClient


class TestGitHubClient:
    def test_client_sets_auth_header(self, httpx_mock: HTTPXMock) -> None:
        """Verify Authorization header is sent when token is provided."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            json={"login": "testuser"},
        )

        client = GitHubClient(token="ghp_test123")
        client.get("/user")

        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["Authorization"] == "Bearer ghp_test123"

    def test_client_user_agent_header(self, httpx_mock: HTTPXMock) -> None:
        """Verify User-Agent header is always set."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            json={"login": "testuser"},
        )

        client = GitHubClient()
        client.get("/user")

        request = httpx_mock.get_request()
        assert request is not None
        assert "code-review-agent" in request.headers["User-Agent"]

    def test_client_base_url_default(self) -> None:
        """Verify default base_url is api.github.com."""
        client = GitHubClient()
        assert client.base_url == "https://api.github.com"

    def test_client_get_method(self, httpx_mock: HTTPXMock) -> None:
        """Verify get() returns parsed JSON."""
        expected = {"full_name": "python/cpython", "stargazers_count": 65000}
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/python/cpython",
            json=expected,
        )

        client = GitHubClient()
        result = client.get("/repos/python/cpython")

        assert result == expected

    def test_client_get_raises_on_error(self, httpx_mock: HTTPXMock) -> None:
        """Verify get() raises HTTPStatusError on 4xx."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/repos/nonexistent/repo",
            status_code=404,
            json={"message": "Not Found"},
        )

        client = GitHubClient()
        with pytest.raises(httpx.HTTPStatusError):
            client.get("/repos/nonexistent/repo")
