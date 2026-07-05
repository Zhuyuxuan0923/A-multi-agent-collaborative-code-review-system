"""Tests for WebhookVerifier -- GitHub webhook HMAC-SHA256 signature verification."""

import hashlib
import hmac

from study_agent.github.webhook import WebhookVerifier, WebhookVerifyResult

WEBHOOK_SECRET = "my_secret_key"


def _make_signature(body: bytes, secret: str) -> str:
    """Compute a valid GitHub-style signature header for the given body and secret."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class TestWebhookVerifyResult:
    def test_valid_result_defaults(self) -> None:
        result = WebhookVerifyResult(valid=True)
        assert result.valid is True
        assert result.error == ""

    def test_invalid_result_with_error(self) -> None:
        result = WebhookVerifyResult(valid=False, error="something went wrong")
        assert result.valid is False
        assert result.error == "something went wrong"


class TestWebhookVerifier:
    def test_valid_signature(self) -> None:
        """Correct HMAC -> valid=True."""
        body = b'{"action":"opened","number":1}'
        header = _make_signature(body, WEBHOOK_SECRET)
        result = WebhookVerifier.verify(body, header, WEBHOOK_SECRET)

        assert result.valid is True
        assert result.error == ""

    def test_invalid_signature(self) -> None:
        """Wrong HMAC -> valid=False."""
        body = b'{"action":"opened","number":1}'
        header = f"sha256={'a' * 64}"
        result = WebhookVerifier.verify(body, header, WEBHOOK_SECRET)

        assert result.valid is False
        assert result.error == "signature mismatch"

    def test_missing_signature_header(self) -> None:
        """Empty header -> valid=False."""
        body = b'{"action":"opened"}'
        result = WebhookVerifier.verify(body, "", WEBHOOK_SECRET)

        assert result.valid is False
        assert result.error == "missing X-Hub-Signature-256 header"

    def test_tampered_body(self) -> None:
        """Signature computed for original body, but body was changed -> mismatch."""
        original_body = b'{"action":"opened"}'
        header = _make_signature(original_body, WEBHOOK_SECRET)

        tampered_body = b'{"action":"closed"}'
        result = WebhookVerifier.verify(tampered_body, header, WEBHOOK_SECRET)

        assert result.valid is False
        assert result.error == "signature mismatch"

    def test_empty_secret(self) -> None:
        """Empty webhook_secret -> valid=False."""
        body = b'{"action":"opened"}'
        header = _make_signature(body, WEBHOOK_SECRET)
        result = WebhookVerifier.verify(body, header, "")

        assert result.valid is False
        assert result.error == "webhook secret is empty"

    def test_unsupported_algorithm(self) -> None:
        """Header does not start with 'sha256=' -> valid=False."""
        body = b'{"action":"opened"}'
        header = "sha1=abc123"
        result = WebhookVerifier.verify(body, header, WEBHOOK_SECRET)

        assert result.valid is False
        assert result.error == "unsupported signature algorithm"
