"""PATValidator -- validate a GitHub Personal Access Token via GET /user.

Checks whether a token is accepted by the GitHub API and inspects the
X-OAuth-Scopes response header to determine which permissions (scopes) the
token carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from study_agent.github.client import GitHubClient


@dataclass
class TokenValidationResult:
    """Outcome of a token validation call.

    Attributes:
        valid: True when GitHub accepted the token (HTTP 200).
        username: GitHub login name of the token owner (empty on failure).
        scopes: Permission scopes granted to the token (empty on failure).
        error: Human-readable error description (empty on success).
    """

    valid: bool
    username: str = ""
    scopes: list[str] = field(default_factory=list)
    error: str = ""

    def has_repo_scope(self) -> bool:
        """Return True if the token grants repository access.

        Checks for either the full "repo" scope (private repos) or the
        limited "public_repo" scope (public repos only).
        """
        return "repo" in self.scopes or "public_repo" in self.scopes


class PATValidator:
    """Validates a GitHub Personal Access Token against the REST API.

    Calls GET /user with the provided token. A 200 response means the token
    is valid; the X-OAuth-Scopes header tells us which permissions it has.
    """

    @staticmethod
    def validate(token: str) -> TokenValidationResult:
        """Validate a GitHub PAT and return its metadata.

        Args:
            token: The GitHub Personal Access Token string to validate.

        Returns:
            A TokenValidationResult indicating validity, username, scopes,
            and any error encountered.
        """
        if not token or not token.strip():
            return TokenValidationResult(valid=False, error="token is empty")

        client = GitHubClient(token=token)
        try:
            resp = client.get_raw("/user")
        except httpx.RequestError as exc:
            return TokenValidationResult(valid=False, error=f"network error: {exc}")

        try:
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                username: str = data.get("login", "")
                raw_scopes = resp.headers.get("X-OAuth-Scopes", "")
                scopes = (
                    [s.strip() for s in raw_scopes.split(",") if s.strip()] if raw_scopes else []
                )
                return TokenValidationResult(valid=True, username=username, scopes=scopes)

            return TokenValidationResult(
                valid=False,
                error=f"token rejected: HTTP {resp.status_code}",
            )
        finally:
            client.close()
