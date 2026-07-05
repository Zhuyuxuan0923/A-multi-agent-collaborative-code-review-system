"""GitHubClient -- shared HTTP layer for GitHub API calls.

Every request automatically receives Authorization (when a token is provided),
User-Agent, and Accept headers.
"""

from __future__ import annotations

from typing import Any

import httpx


class GitHubClient:
    """A thin wrapper around httpx.Client for GitHub REST API calls.

    Features:
      - Automatic Authorization header (Bearer token) when a token is supplied
      - Fixed User-Agent ("code-review-agent/0.1.0") on every request
      - Fixed Accept ("application/vnd.github+json") on every request
      - 30-second timeout on the underlying httpx client
      - get() raises httpx.HTTPStatusError on 4xx/5xx
      - get_raw() returns the raw Response (e.g. for diff text)

    Usage:
        client = GitHubClient(token="ghp_xxx")
        user = client.get("/user")
        print(user["login"])

        # Anonymous access (lower rate limits):
        client = GitHubClient()
        repo = client.get("/repos/python/cpython")
    """

    USER_AGENT = "code-review-agent/0.1.0"
    ACCEPT = "application/vnd.github+json"

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
    ) -> None:
        """Initialise the GitHub API client.

        Args:
            token: GitHub Personal Access Token (PAT). Optional -- anonymous
                   access is supported but subject to lower rate limits (60 req/h
                   vs 5000 req/h with a token).
            base_url: API base URL. Defaults to GitHub's public API. Override
                      for GitHub Enterprise Server.
        """
        self.base_url = base_url
        self.token = token

        headers: dict[str, str] = {
            "Accept": self.ACCEPT,
            "User-Agent": self.USER_AGENT,
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=30.0,
        )

    def get(self, path: str) -> dict[str, Any]:
        """Send a GET request and return parsed JSON.

        Args:
            path: API path relative to base_url (e.g. "/user", "/repos/owner/repo").

        Returns:
            Parsed JSON response body as a dict.

        Raises:
            httpx.HTTPStatusError: If the response status is 4xx or 5xx.
            httpx.RequestError: On network-level errors.
        """
        response = self._client.get(path)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def get_raw(self, path: str) -> httpx.Response:
        """Send a GET request and return the raw Response object.

        Useful for endpoints that return non-JSON bodies, e.g. diff text from
        /repos/{owner}/{repo}/pulls/{number}.diff.

        Args:
            path: API path relative to base_url.

        Returns:
            The raw httpx.Response object (caller must check status).

        Raises:
            httpx.RequestError: On network-level errors. Does NOT raise on 4xx/5xx
                                -- caller should inspect response.status_code.
        """
        return self._client.get(path)

    def post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request with a JSON body and return parsed JSON.

        Args:
            path: API path relative to base_url.
            json_data: JSON-serialisable body to send.

        Returns:
            Parsed JSON response body as a dict.

        Raises:
            httpx.HTTPStatusError: If the response status is 4xx or 5xx.
            httpx.RequestError: On network-level errors.
        """
        response = self._client.post(path, json=json_data)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def close(self) -> None:
        """Close the underlying httpx Client, releasing connections."""
        self._client.close()
