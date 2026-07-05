"""GitHub integration layer for the Multi-Agent Code Review System.

This package provides HTTP-level and operation-level access to the GitHub REST API:
  - GitHubClient: shared HTTP layer (token, User-Agent, Accept headers)
  - PRDiffFetcher: pull request diff retrieval and parsing
  - CommentPoster: review comment posting to PRs
  - WebhookVerifier: GitHub webhook signature verification
  - PATValidator: personal access token scope validation
"""

from study_agent.github.client import GitHubClient
from study_agent.github.comment_poster import CommentPoster, CommentResult
from study_agent.github.diff_fetcher import PRDiffFetcher, PRDiffResult
from study_agent.github.token_validator import PATValidator, TokenValidationResult
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
