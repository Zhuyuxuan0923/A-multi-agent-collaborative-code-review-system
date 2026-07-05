"""WebhookVerifier -- GitHub webhook HMAC-SHA256 signature verification.

GitHub signs each webhook payload with a secret key using HMAC-SHA256.
The signature arrives in the ``X-Hub-Signature-256`` header as::

    sha256=<64-character hex digest>

By re-computing the HMAC on the raw request body and comparing, the
receiver can be sure the payload is authentic and untampered.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass
class WebhookVerifyResult:
    """Outcome of a webhook signature verification."""

    valid: bool
    """True when the computed signature matches the header."""

    error: str = ""
    """Human-readable reason when ``valid`` is False."""


class WebhookVerifier:
    """Verify that an incoming webhook request genuinely originated from GitHub.

    Usage::

        result = WebhookVerifier.verify(
            request_body=request.body,
            signature_header=request.headers.get("X-Hub-Signature-256", ""),
            webhook_secret=os.environ["GITHUB_WEBHOOK_SECRET"],
        )
        if not result.valid:
            raise HTTPException(status_code=401, detail=result.error)
    """

    @staticmethod
    def verify(
        request_body: bytes,
        signature_header: str,
        webhook_secret: str,
    ) -> WebhookVerifyResult:
        """Check the HMAC-SHA256 signature against the request body.

        Args:
            request_body: Raw HTTP request body **as bytes** (do not decode).
            signature_header: Value of the ``X-Hub-Signature-256`` header.
            webhook_secret: The secret string configured in the GitHub webhook
                settings.

        Returns:
            ``WebhookVerifyResult`` with ``valid=True`` when the signature
            matches; ``valid=False`` plus an ``error`` string otherwise.
        """
        # 1. Guard: webhook secret must be set.
        if not webhook_secret:
            return WebhookVerifyResult(valid=False, error="webhook secret is empty")

        # 2. Guard: signature header must be present.
        if not signature_header:
            return WebhookVerifyResult(valid=False, error="missing X-Hub-Signature-256 header")

        # 3. Guard: only sha256 is supported.
        if not signature_header.startswith("sha256="):
            return WebhookVerifyResult(valid=False, error="unsupported signature algorithm")

        # 4. Parse the hex digest from the header.
        expected = signature_header[7:]

        # 5. Re-compute the HMAC on the request body.
        computed = hmac.new(
            webhook_secret.encode(),
            request_body,
            hashlib.sha256,
        ).hexdigest()

        # 6. Constant-time comparison (resists timing side-channel attacks).
        if hmac.compare_digest(computed, expected):
            return WebhookVerifyResult(valid=True)

        return WebhookVerifyResult(valid=False, error="signature mismatch")
