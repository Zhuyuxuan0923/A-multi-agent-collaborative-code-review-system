# GitHub Integration Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GitHub integration layer (`src/study_agent/github/`) — a pure Python module that wraps GitHub REST API for PR diff fetching, PAT validation, comment posting, and webhook verification.

**Architecture:** Five small modules, each with a single responsibility. `GitHubClient` is the shared HTTP layer (thin wrapper around `httpx`). The four feature modules depend on `GitHubClient`. Each module is independently testable with mocked HTTP responses.

**Tech Stack:** Python 3.11, httpx (async HTTP), hmac (stdlib), hashlib (stdlib), re (stdlib), pytest + pytest-httpx for testing

---

## File Structure

```
src/study_agent/github/
+-- __init__.py           # exports all public classes
+-- client.py             # GitHubClient: shared httpx wrapper with auth headers
+-- token_validator.py    # PATValidator: validate GitHub PAT + check scopes
+-- diff_fetcher.py       # PRDiffFetcher: fetch PR diff, parse file stats
+-- comment_poster.py     # CommentPoster: post review summary as PR comment
+-- webhook.py            # WebhookVerifier: verify HMAC-SHA256 signature
tests/
+-- test_github_client.py
+-- test_token_validator.py
+-- test_diff_fetcher.py
+-- test_comment_poster.py
+-- test_webhook.py
```

Each file is < 80 lines. No class exceeds 3 public methods. All HTTP calls go through `GitHubClient`.

---

### Task 1: GitHubClient — shared HTTP layer

**Files:**
- Create: `src/study_agent/github/__init__.py`
- Create: `src/study_agent/github/client.py`
- Create: `tests/test_github_client.py`

- [ ] **Step 1: Create `src/study_agent/github/__init__.py` (empty for now)**

```python
"""GitHub integration layer for Multi-Agent Code Review System."""
```

- [ ] **Step 2: Write the failing test for GitHubClient**

Create `tests/test_github_client.py`:

```python
"""Tests for GitHubClient."""
import httpx
import pytest
from pytest_httpx import HTTPXMock

from study_agent.github.client import GitHubClient


def test_client_sets_auth_header(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        json={"login": "testuser"},
        status_code=200,
    )
    client = GitHubClient(token="ghp_test123")
    with client._client as http:
        resp = http.get("https://api.github.com/user")
    assert resp.status_code == 200
    assert resp.json()["login"] == "testuser"


def test_client_user_agent_header(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        json={"login": "testuser"},
        status_code=200,
    )
    client = GitHubClient()
    with client._client as http:
        resp = http.get("https://api.github.com/user")
    request_headers = resp.request.headers
    assert "User-Agent" in request_headers
    assert "code-review-agent" in request_headers["User-Agent"]


def test_client_base_url_default():
    client = GitHubClient()
    assert client.base_url == "https://api.github.com"


def test_client_get_method(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/test/repo",
        json={"full_name": "test/repo"},
        status_code=200,
    )
    client = GitHubClient(token="ghp_test123")
    data = client.get("/repos/test/repo")
    assert data["full_name"] == "test/repo"


def test_client_get_raises_on_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/test/repo",
        status_code=404,
        json={"message": "Not Found"},
    )
    client = GitHubClient(token="ghp_test123")
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/repos/test/repo")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/test_github_client.py -v`
Expected: all 5 tests FAIL — `ModuleNotFoundError: No module named 'study_agent.github.client'`

- [ ] **Step 4: Install httpx dependency**

Run: `poetry add httpx`

- [ ] **Step 5: Write GitHubClient implementation**

Create `src/study_agent/github/client.py`:

```python
"""GitHub REST API HTTP client with auth and User-Agent headers."""
from typing import Optional
import httpx

GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "code-review-agent/0.1.0"


class GitHubClient:
    """Thin wrapper around httpx.Client for GitHub API calls.

    Every request automatically gets:
      - Authorization: Bearer {token}  (if token provided)
      - User-Agent: code-review-agent/0.1.0  (GitHub requires this)
      - Accept: application/vnd.github+json
    """

    def __init__(self, token: Optional[str] = None, base_url: str = GITHUB_API_BASE):
        self.token = token
        self.base_url = base_url.rstrip("/")
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=30.0)

    def get(self, path: str, **kwargs) -> dict:
        """GET request, returns parsed JSON. Raises HTTPStatusError on 4xx/5xx."""
        resp = self._client.get(path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_raw(self, path: str, **kwargs) -> httpx.Response:
        """GET request, returns raw Response (for diff text, not JSON)."""
        resp = self._client.get(path, **kwargs)
        resp.raise_for_status()
        return resp

    def post(self, path: str, json_data: Optional[dict] = None, **kwargs) -> dict:
        """POST request, returns parsed JSON."""
        resp = self._client.post(path, json=json_data, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/test_github_client.py -v`
Expected: 5 PASS

- [ ] **Step 7: Commit**

```bash
git add src/study_agent/github/__init__.py src/study_agent/github/client.py tests/test_github_client.py
git commit -m "feat: add GitHubClient — shared HTTP layer for GitHub API"
```

---

### Task 2: PATValidator — validate GitHub Personal Access Token

**Files:**
- Create: `src/study_agent/github/token_validator.py`
- Create: `tests/test_token_validator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_token_validator.py`:

```python
"""Tests for PATValidator."""
from pytest_httpx import HTTPXMock

from study_agent.github.token_validator import PATValidator, TokenValidationResult


def test_valid_token(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        json={"login": "testuser"},
        status_code=200,
        headers={"X-OAuth-Scopes": "repo, user"},
    )
    result = PATValidator.validate("ghp_valid")
    assert result.valid is True
    assert result.username == "testuser"
    assert "repo" in result.scopes


def test_invalid_token(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        status_code=401,
        json={"message": "Bad credentials"},
    )
    result = PATValidator.validate("ghp_bad")
    assert result.valid is False
    assert result.username == ""


def test_token_missing_repo_scope(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        json={"login": "testuser"},
        status_code=200,
        headers={"X-OAuth-Scopes": "user, gist"},
    )
    result = PATValidator.validate("ghp_norepo")
    assert result.valid is True
    assert result.has_repo_scope() is False


def test_token_has_repo_scope(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/user",
        json={"login": "testuser"},
        status_code=200,
        headers={"X-OAuth-Scopes": "repo, user"},
    )
    result = PATValidator.validate("ghp_valid")
    assert result.has_repo_scope() is True


def test_empty_token():
    result = PATValidator.validate("")
    assert result.valid is False
    assert "empty" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_token_validator.py -v`
Expected: 5 FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write PATValidator implementation**

Create `src/study_agent/github/token_validator.py`:

```python
"""Validate GitHub Personal Access Tokens."""
from dataclasses import dataclass, field
import httpx
from study_agent.github.client import GitHubClient


@dataclass
class TokenValidationResult:
    valid: bool
    username: str = ""
    scopes: list[str] = field(default_factory=list)
    error: str = ""

    def has_repo_scope(self) -> bool:
        return any(s in ["repo", "public_repo"] for s in self.scopes)


class PATValidator:
    """Validate a GitHub Personal Access Token by calling /user endpoint.

    Usage:
        result = PATValidator.validate("ghp_xxx")
        if result.valid and result.has_repo_scope():
            print(f"Token OK, logged in as {result.username}")
    """

    @staticmethod
    def validate(token: str) -> TokenValidationResult:
        if not token or not token.strip():
            return TokenValidationResult(valid=False, error="token is empty")

        client = GitHubClient(token=token.strip())
        try:
            data = client.get("/user")
            scopes = client._client.headers.get("X-OAuth-Scopes", "")
            scope_list = [s.strip() for s in scopes.split(",") if s.strip()]
            return TokenValidationResult(
                valid=True,
                username=data.get("login", ""),
                scopes=scope_list,
            )
        except httpx.HTTPStatusError as e:
            return TokenValidationResult(
                valid=False,
                error=f"GitHub API returned {e.response.status_code}",
            )
        except Exception as e:
            return TokenValidationResult(valid=False, error=str(e))
        finally:
            client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_token_validator.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/study_agent/github/token_validator.py tests/test_token_validator.py
git commit -m "feat: add PATValidator — GitHub token validation with scope check"
```

---

### Task 3: PRDiffFetcher — fetch and parse PR diff

**Files:**
- Create: `src/study_agent/github/diff_fetcher.py`
- Create: `tests/test_diff_fetcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_diff_fetcher.py`:

```python
"""Tests for PRDiffFetcher."""
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


def test_parse_pr_url():
    owner, repo, num = PRDiffFetcher.parse_pr_url(YOUR_REPO_URL)
    assert owner == "testuser"
    assert repo == "testrepo"
    assert num == 1


def test_parse_pr_url_with_trailing_slash():
    owner, repo, num = PRDiffFetcher.parse_pr_url(
        "https://github.com/abc/def/pull/42/"
    )
    assert owner == "abc"
    assert repo == "def"
    assert num == 42


def test_parse_pr_url_invalid():
    result = PRDiffFetcher.parse_pr_url("https://google.com")
    assert result is None


def test_fetch_diff(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/testuser/testrepo/pulls/1",
        text=SAMPLE_DIFF,
        status_code=200,
        headers={"Content-Type": "application/vnd.github.v3.diff"},
    )
    result = PRDiffFetcher.fetch(YOUR_REPO_URL, token="ghp_test")
    assert isinstance(result, PRDiffResult)
    assert "app.py" in result.diff_raw
    assert result.files_changed >= 1
    assert result.additions >= 1


def test_fetch_diff_pr_not_found(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/testuser/testrepo/pulls/1",
        status_code=404,
        json={"message": "Not Found"},
    )
    result = PRDiffFetcher.fetch(YOUR_REPO_URL, token="ghp_test")
    assert result.error != ""
    assert result.diff_raw == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_diff_fetcher.py -v`
Expected: 5 FAIL

- [ ] **Step 3: Write PRDiffFetcher implementation**

Create `src/study_agent/github/diff_fetcher.py`:

```python
"""Fetch and parse GitHub PR diffs."""
import re
from dataclasses import dataclass, field
from typing import Optional
import httpx
from study_agent.github.client import GitHubClient


@dataclass
class PRDiffResult:
    diff_raw: str = ""
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0
    error: str = ""


class PRDiffFetcher:
    """Fetch a PR's diff from GitHub API and extract basic stats.

    Uses the v3.diff media type to get raw unified diff instead of JSON.

    Usage:
        result = PRDiffFetcher.fetch("https://github.com/u/r/pull/1", token="ghp_xxx")
        if not result.error:
            print(f"{result.files_changed} files, +{result.additions} -{result.deletions}")
    """

    PR_URL_PATTERN = re.compile(
        r"^https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?$"
    )

    @classmethod
    def parse_pr_url(cls, pr_url: str) -> Optional[tuple[str, str, int]]:
        """Extract (owner, repo, pr_number) from PR URL. Returns None if invalid."""
        m = cls.PR_URL_PATTERN.match(pr_url.strip().rstrip("/"))
        if not m:
            return None
        return (m.group(1), m.group(2), int(m.group(3)))

    @classmethod
    def fetch(cls, pr_url: str, token: str) -> PRDiffResult:
        parsed = cls.parse_pr_url(pr_url)
        if parsed is None:
            return PRDiffResult(error=f"Invalid PR URL: {pr_url}")

        owner, repo, pr_number = parsed
        client = GitHubClient(token=token)
        try:
            resp = client.get_raw(
                f"/repos/{owner}/{repo}/pulls/{pr_number}",
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            diff_text = resp.text
            additions = diff_text.count("\n+") - diff_text.count("\n+++")
            deletions = diff_text.count("\n-") - diff_text.count("\n---")
            files_changed = diff_text.count("diff --git ")

            return PRDiffResult(
                diff_raw=diff_text,
                files_changed=files_changed,
                additions=max(additions, 0),
                deletions=max(deletions, 0),
            )
        except httpx.HTTPStatusError as e:
            return PRDiffResult(error=f"GitHub API error: {e.response.status_code}")
        except Exception as e:
            return PRDiffResult(error=str(e))
        finally:
            client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_diff_fetcher.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/study_agent/github/diff_fetcher.py tests/test_diff_fetcher.py
git commit -m "feat: add PRDiffFetcher — fetch PR diff with file stats"
```

---

### Task 4: CommentPoster — post review summary as PR comment

**Files:**
- Create: `src/study_agent/github/comment_poster.py`
- Create: `tests/test_comment_poster.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_comment_poster.py`:

```python
"""Tests for CommentPoster."""
from pytest_httpx import HTTPXMock

from study_agent.github.comment_poster import CommentPoster

PR_URL = "https://github.com/testuser/testrepo/pull/1"


def test_post_comment_success(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/testuser/testrepo/issues/1/comments",
        method="POST",
        status_code=201,
        json={"id": 123, "html_url": "https://github.com/t/r/issues/1#issuecomment-123"},
    )
    result = CommentPoster.post(
        pr_url=PR_URL,
        token="ghp_test",
        report_md="# Review\nLooks good",
        score=8,
        issues_count=3,
    )
    assert result.success is True
    assert "123" in result.comment_url


def test_post_comment_invalid_pr_url():
    result = CommentPoster.post(
        pr_url="https://google.com",
        token="ghp_test",
        report_md="# Review",
        score=5,
        issues_count=0,
    )
    assert result.success is False
    assert "Invalid PR URL" in result.error


def test_build_comment_body_short():
    body = CommentPoster.build_comment_body(
        report_md="# OK", score=9, issues_count=1
    )
    assert "9/10" in body
    assert "1" in body
    assert "# OK" in body


def test_build_comment_body_truncates_long_report():
    long_report = "x" * 70000
    body = CommentPoster.build_comment_body(
        report_md=long_report, score=5, issues_count=10
    )
    assert len(body) < 66000
    assert "..." in body


def test_post_comment_api_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/testuser/testrepo/issues/1/comments",
        method="POST",
        status_code=403,
        json={"message": "Resource not accessible by integration"},
    )
    result = CommentPoster.post(
        pr_url=PR_URL,
        token="ghp_test",
        report_md="# Review",
        score=5,
        issues_count=0,
    )
    assert result.success is False
    assert "403" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_comment_poster.py -v`
Expected: 5 FAIL

- [ ] **Step 3: Write CommentPoster implementation**

Create `src/study_agent/github/comment_poster.py`:

```python
"""Post review results as PR comments on GitHub."""
from dataclasses import dataclass
import httpx
from study_agent.github.client import GitHubClient
from study_agent.github.diff_fetcher import PRDiffFetcher

MAX_COMMENT_LENGTH = 65536


@dataclass
class CommentResult:
    success: bool
    comment_url: str = ""
    error: str = ""


class CommentPoster:
    """Post a code review summary as a comment on a GitHub PR.

    Usage:
        result = CommentPoster.post(
            pr_url="https://github.com/u/r/pull/1",
            token="ghp_xxx",
            report_md=review_report,
            score=7,
            issues_count=5,
        )
    """

    @staticmethod
    def build_comment_body(report_md: str, score: int, issues_count: int) -> str:
        header = (
            f"## Code Review by AI Agent\n\n"
            f"**Score: {score}/10** | **Issues found: {issues_count}**\n\n"
            f"---\n\n"
        )
        body = header + report_md
        if len(body) > MAX_COMMENT_LENGTH:
            truncation_note = (
                "\n\n---\n\n"
                "> [注意] 报告过长，已截断。完整报告请查看审查系统页面。\n"
            )
            available = MAX_COMMENT_LENGTH - len(truncation_note)
            body = body[:available] + "...(truncated)" + truncation_note
        return body

    @classmethod
    def post(
        cls,
        pr_url: str,
        token: str,
        report_md: str,
        score: int,
        issues_count: int,
    ) -> CommentResult:
        parsed = PRDiffFetcher.parse_pr_url(pr_url)
        if parsed is None:
            return CommentResult(success=False, error=f"Invalid PR URL: {pr_url}")

        owner, repo, pr_number = parsed
        body = cls.build_comment_body(report_md, score, issues_count)
        client = GitHubClient(token=token)
        try:
            data = client.post(
                f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
                json_data={"body": body},
            )
            return CommentResult(
                success=True,
                comment_url=data.get("html_url", ""),
            )
        except httpx.HTTPStatusError as e:
            return CommentResult(
                success=False,
                error=f"GitHub API error: {e.response.status_code}",
            )
        except Exception as e:
            return CommentResult(success=False, error=str(e))
        finally:
            client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_comment_poster.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/study_agent/github/comment_poster.py tests/test_comment_poster.py
git commit -m "feat: add CommentPoster — post review summary as PR comment"
```

---

### Task 5: WebhookVerifier — verify GitHub webhook HMAC signature

**Files:**
- Create: `src/study_agent/github/webhook.py`
- Create: `tests/test_webhook.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_webhook.py`:

```python
"""Tests for WebhookVerifier."""
import hashlib
import hmac

from study_agent.github.webhook import WebhookVerifier

WEBHOOK_SECRET = "my_secret_key"


def test_valid_signature():
    body = b'{"action":"opened","pull_request":{}}'
    mac = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256)
    signature = f"sha256={mac.hexdigest()}"
    result = WebhookVerifier.verify(
        request_body=body,
        signature_header=signature,
        webhook_secret=WEBHOOK_SECRET,
    )
    assert result.valid is True


def test_invalid_signature():
    body = b'{"action":"opened"}'
    result = WebhookVerifier.verify(
        request_body=body,
        signature_header="sha256=abc123def456",
        webhook_secret=WEBHOOK_SECRET,
    )
    assert result.valid is False
    assert "signature mismatch" in result.error.lower()


def test_missing_signature_header():
    result = WebhookVerifier.verify(
        request_body=b"{}",
        signature_header="",
        webhook_secret=WEBHOOK_SECRET,
    )
    assert result.valid is False
    assert "missing" in result.error.lower()


def test_tampered_body():
    original_body = b'{"action":"opened","pull_request":{"id":1}}'
    mac = hmac.new(WEBHOOK_SECRET.encode(), original_body, hashlib.sha256)
    signature = f"sha256={mac.hexdigest()}"
    tampered_body = b'{"action":"closed","pull_request":{"id":1}}'
    result = WebhookVerifier.verify(
        request_body=tampered_body,
        signature_header=signature,
        webhook_secret=WEBHOOK_SECRET,
    )
    assert result.valid is False


def test_empty_secret():
    result = WebhookVerifier.verify(
        request_body=b"{}",
        signature_header="sha256=abc",
        webhook_secret="",
    )
    assert result.valid is False
    assert "secret" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_webhook.py -v`
Expected: 5 FAIL

- [ ] **Step 3: Write WebhookVerifier implementation**

Create `src/study_agent/github/webhook.py`:

```python
"""Verify GitHub webhook payload signatures using HMAC-SHA256."""
import hashlib
import hmac
from dataclasses import dataclass


@dataclass
class WebhookVerifyResult:
    valid: bool
    error: str = ""


class WebhookVerifier:
    """Verify GitHub webhook payload authenticity.

    GitHub signs webhook payloads with HMAC-SHA256 using the webhook secret.
    The signature comes in the X-Hub-Signature-256 header as "sha256=<hex>".

    Uses hmac.compare_digest() to prevent timing attacks.

    Usage:
        result = WebhookVerifier.verify(
            request_body=request.body(),
            signature_header=request.headers.get("X-Hub-Signature-256", ""),
            webhook_secret=os.environ["WEBHOOK_SECRET"],
        )
        if not result.valid:
            raise HTTPException(status_code=401, detail="Invalid signature")
    """

    @staticmethod
    def verify(
        request_body: bytes,
        signature_header: str,
        webhook_secret: str,
    ) -> WebhookVerifyResult:
        if not webhook_secret:
            return WebhookVerifyResult(valid=False, error="webhook secret is empty")

        if not signature_header:
            return WebhookVerifyResult(
                valid=False, error="missing X-Hub-Signature-256 header"
            )

        if not signature_header.startswith("sha256="):
            return WebhookVerifyResult(
                valid=False, error=f"unsupported signature algorithm"
            )

        expected_hex = signature_header[7:]  # remove "sha256=" prefix
        computed = hmac.new(
            webhook_secret.encode(),
            request_body,
            hashlib.sha256,
        )
        computed_hex = computed.hexdigest()

        if not hmac.compare_digest(computed_hex, expected_hex):
            return WebhookVerifyResult(valid=False, error="signature mismatch")

        return WebhookVerifyResult(valid=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_webhook.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/study_agent/github/webhook.py tests/test_webhook.py
git commit -m "feat: add WebhookVerifier — HMAC-SHA256 signature check"
```

---

### Task 6: Update `__init__.py` with all exports

**Files:**
- Modify: `src/study_agent/github/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
"""GitHub integration layer for Multi-Agent Code Review System."""
from study_agent.github.client import GitHubClient
from study_agent.github.token_validator import PATValidator, TokenValidationResult
from study_agent.github.diff_fetcher import PRDiffFetcher, PRDiffResult
from study_agent.github.comment_poster import CommentPoster, CommentResult
from study_agent.github.webhook import WebhookVerifier, WebhookVerifyResult

__all__ = [
    "GitHubClient",
    "PATValidator",
    "TokenValidationResult",
    "PRDiffFetcher",
    "PRDiffResult",
    "CommentPoster",
    "CommentResult",
    "WebhookVerifier",
    "WebhookVerifyResult",
]
```

- [ ] **Step 2: Verify import works**

Run: `poetry run python -c "from study_agent.github import GitHubClient, PATValidator, PRDiffFetcher, CommentPoster, WebhookVerifier; print('[OK] All imports successful')"`

- [ ] **Step 3: Run all tests together**

Run: `poetry run pytest tests/test_github_client.py tests/test_token_validator.py tests/test_diff_fetcher.py tests/test_comment_poster.py tests/test_webhook.py -v`
Expected: 25 PASS

- [ ] **Step 4: Format and lint**

Run: `poetry run ruff check src/study_agent/github/ tests/`
Then: `poetry run black src/study_agent/github/ tests/`

- [ ] **Step 5: Commit**

```bash
git add src/study_agent/github/__init__.py
git commit -m "feat: complete GitHub integration layer — 5 modules, 25 tests"
```

---

## Self-Review

1. **Spec coverage:** Maps to design doc Section 6 (GitHub Integration) — PATValidator, PRDiffFetcher, CommentPoster, WebhookVerifier all covered.
2. **No placeholders:** Verified — all code blocks are complete, all test assertions are concrete.
3. **Type consistency:** `TokenValidationResult`, `PRDiffResult`, `CommentResult`, `WebhookVerifyResult` — each has consistent field names. `PRDiffFetcher.parse_pr_url()` is used by both `PRDiffFetcher.fetch()` and `CommentPoster.post()` — same signature.
4. **Testability:** All modules use `GitHubClient` which wraps `httpx.Client` — easy to mock via `pytest-httpx`. No real network calls in tests.
