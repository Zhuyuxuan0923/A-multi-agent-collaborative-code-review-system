"""Tests for PATValidator -- GitHub personal access token validation."""

from pytest_httpx import HTTPXMock

from study_agent.github.token_validator import PATValidator, TokenValidationResult


class TestTokenValidationResult:
    def test_has_repo_scope_detects_repo(self) -> None:
        result = TokenValidationResult(valid=True, username="testuser", scopes=["repo", "user"])
        assert result.has_repo_scope() is True

    def test_has_repo_scope_detects_public_repo(self) -> None:
        result = TokenValidationResult(
            valid=True, username="testuser", scopes=["public_repo", "gist"]
        )
        assert result.has_repo_scope() is True

    def test_has_repo_scope_returns_false_when_missing(self) -> None:
        result = TokenValidationResult(valid=True, username="testuser", scopes=["user", "gist"])
        assert result.has_repo_scope() is False

    def test_has_repo_scope_returns_false_for_empty_scopes(self) -> None:
        result = TokenValidationResult(valid=True, username="testuser")
        assert result.has_repo_scope() is False


class TestPATValidator:
    def test_valid_token(self, httpx_mock: HTTPXMock) -> None:
        """Token is valid, returns username and scopes."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            status_code=200,
            json={"login": "testuser"},
            headers={"X-OAuth-Scopes": "repo, user"},
        )

        result = PATValidator.validate("ghp_valid_token")

        assert result.valid is True
        assert result.username == "testuser"
        assert result.scopes == ["repo", "user"]
        assert result.error == ""
        assert result.has_repo_scope() is True

    def test_invalid_token(self, httpx_mock: HTTPXMock) -> None:
        """Token returns 401, result.valid is False."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            status_code=401,
            json={"message": "Bad credentials"},
        )

        result = PATValidator.validate("ghp_invalid_token")

        assert result.valid is False
        assert result.username == ""
        assert result.scopes == []
        assert "401" in result.error

    def test_token_missing_repo_scope(self, httpx_mock: HTTPXMock) -> None:
        """Token valid but has no repo scope -> has_repo_scope() returns False."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            status_code=200,
            json={"login": "testuser"},
            headers={"X-OAuth-Scopes": "user, gist"},
        )

        result = PATValidator.validate("ghp_no_repo_token")

        assert result.valid is True
        assert result.username == "testuser"
        assert result.scopes == ["user", "gist"]
        assert result.has_repo_scope() is False

    def test_token_has_repo_scope(self, httpx_mock: HTTPXMock) -> None:
        """Token has repo scope -> has_repo_scope() returns True."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            status_code=200,
            json={"login": "testuser"},
            headers={"X-OAuth-Scopes": "repo, user"},
        )

        result = PATValidator.validate("ghp_repo_token")

        assert result.valid is True
        assert result.has_repo_scope() is True

    def test_token_has_public_repo_scope(self, httpx_mock: HTTPXMock) -> None:
        """Token has public_repo scope -> has_repo_scope() returns True."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.github.com/user",
            status_code=200,
            json={"login": "testuser"},
            headers={"X-OAuth-Scopes": "public_repo"},
        )

        result = PATValidator.validate("ghp_public_repo_token")

        assert result.valid is True
        assert result.has_repo_scope() is True

    def test_empty_token(self) -> None:
        """Empty string token -> invalid without API call."""
        result = PATValidator.validate("")

        assert result.valid is False
        assert result.username == ""
        assert result.scopes == []
        assert result.error == "token is empty"

    def test_whitespace_only_token(self) -> None:
        """Whitespace-only token -> invalid without API call."""
        result = PATValidator.validate("   ")

        assert result.valid is False
        assert result.username == ""
        assert result.scopes == []
        assert result.error == "token is empty"
