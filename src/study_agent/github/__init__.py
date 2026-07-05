"""GitHub integration layer for the Multi-Agent Code Review System.

This package provides HTTP-level and operation-level access to the GitHub REST API:
  - GitHubClient: shared HTTP layer (token, User-Agent, Accept headers)
  - PRDiffFetcher: pull request diff retrieval and parsing
  - CommentPoster: review comment posting to PRs
  - WebhookVerifier: GitHub webhook signature verification
  - PATValidator: personal access token scope validation
"""
