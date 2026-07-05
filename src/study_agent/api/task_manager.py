"""TaskManager -- core orchestration module for the code review API.

Connects the database layer, code review orchestrator, and GitHub integration
into a unified async task lifecycle:

  1. Receives a review request (code snippet or PR URL)
  2. Creates a task record in SQLite
  3. Launches an async background task to run the review
  4. Updates task status as it progresses (queued -> running -> completed/failed)
  5. Stores the final report (Markdown, score, issues, research data) in the DB
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from study_agent.agent.code_review_agents import (
    CodeReviewOrchestrator,
    ResearchResult,
    ReviewResult,
)
from study_agent.api.database import create_task, get_connection, get_task, init_db, update_task
from study_agent.github import PRDiffFetcher
from study_agent.llm.client import LLMClient


class TaskManager:
    """Manages review task lifecycle with async background execution.

    Design decisions:
      - Lazy init: LLMClient and CodeReviewOrchestrator are created on first use,
        so the API server can start without LLM credentials.
      - Thread pool: ``asyncio.to_thread()`` wraps synchronous LLM calls so they
        do not block the event loop.
      - Token hashing: GitHub tokens are SHA256-hashed before storage. Raw tokens
        never touch the database.
    """

    def __init__(self, db_path: str = "data/reviews.db", llm_provider: str = "deepseek") -> None:
        self.db_path = db_path
        self.llm_provider = llm_provider
        self._llm: LLMClient | None = None
        self._orchestrator: CodeReviewOrchestrator | None = None
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        init_db(db_path)

    # -- Properties (lazy init) ------------------------------------------------

    @property
    def llm(self) -> LLMClient:
        """Lazy init LLMClient (shared across tasks)."""
        if self._llm is None:
            self._llm = LLMClient(provider=self.llm_provider)
        return self._llm

    @property
    def orchestrator(self) -> CodeReviewOrchestrator:
        """Lazy init CodeReviewOrchestrator."""
        if self._orchestrator is None:
            self._orchestrator = CodeReviewOrchestrator(self.llm)
        return self._orchestrator

    # -- Public API ------------------------------------------------------------

    def submit_review(
        self,
        code: str,
        language: str = "python",
        github_token: str | None = None,
        pr_url: str | None = None,
    ) -> str:
        """Submit code for review. Returns task_id immediately.

        The review runs in the background via ``asyncio.create_task``.
        Callers poll ``get_task(task_id)`` or ``get_report(task_id)`` for results.
        """
        task_id = uuid.uuid4().hex[:12]

        with get_connection(self.db_path) as conn:
            create_task(
                conn,
                {
                    "id": task_id,
                    "type": "review",
                    "code": code,
                    "language": language,
                    "pr_url": pr_url or "",
                    "github_token_hash": _hash_token(github_token) if github_token else "",
                },
            )

        # Launch background review
        task = asyncio.create_task(self._run_review(task_id))
        self._running_tasks[task_id] = task
        return task_id

    def submit_pr_review(self, pr_url: str, github_token: str) -> str:
        """Fetch PR diff and submit for review. Returns task_id immediately.

        If the PR diff fetch fails, the task is created but immediately marked
        as "failed" -- no background task is spawned.
        """
        task_id = uuid.uuid4().hex[:12]

        # Fetch PR diff synchronously (one HTTP call, no LLM)
        diff_result = PRDiffFetcher.fetch(pr_url, github_token)
        if diff_result.error:
            with get_connection(self.db_path) as conn:
                create_task(
                    conn,
                    {
                        "id": task_id,
                        "type": "pr",
                        "code": "",
                        "language": "",
                        "pr_url": pr_url,
                        "github_token_hash": _hash_token(github_token),
                    },
                )
                update_task(
                    conn,
                    task_id,
                    {
                        "status": "failed",
                        "error": f"Failed to fetch PR diff: {diff_result.error}",
                    },
                )
            return task_id

        with get_connection(self.db_path) as conn:
            create_task(
                conn,
                {
                    "id": task_id,
                    "type": "pr",
                    "code": diff_result.diff_raw,
                    "language": "",
                    "pr_url": pr_url,
                    "github_token_hash": _hash_token(github_token),
                },
            )

        task = asyncio.create_task(self._run_review(task_id))
        self._running_tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task status from DB. Returns None if not found."""
        with get_connection(self.db_path) as conn:
            return get_task(conn, task_id)

    def get_report(self, task_id: str) -> dict[str, Any] | None:
        """Get review report. Only returns data when task is completed."""
        task = self.get_task(task_id)
        if task is None or task["status"] != "completed":
            return None

        issues: list[dict[str, Any]] = json.loads(task.get("issues_json") or "[]")
        technologies: list[str] = []
        if task.get("researcher_data_json"):
            try:
                researcher_data: dict[str, Any] = json.loads(task["researcher_data_json"])
                technologies = researcher_data.get("technologies", [])
            except json.JSONDecodeError:
                pass

        return {
            "task_id": task_id,
            "report_md": task.get("report_md", ""),
            "score": task.get("score", 0),
            "issues": issues,
            "technologies": technologies,
            "reviewed_by": ["reviewer", "researcher", "reporter"],
            "created_at": task.get("created_at", ""),
            "latency_ms": 0,  # TODO: compute from completed_at - created_at
        }

    # -- Internal --------------------------------------------------------------

    async def _run_review(self, task_id: str) -> None:
        """Background task: run orchestrator, update DB with results.

        All LLM calls are wrapped in ``asyncio.to_thread`` so the event loop
        stays free to handle other requests.
        """
        try:
            # Load task from DB
            with get_connection(self.db_path) as conn:
                task = get_task(conn, task_id)
            if task is None:
                return

            code: str = task.get("code", "")
            language: str = task.get("language", "python")

            # Mark running
            with get_connection(self.db_path) as conn:
                update_task(conn, task_id, {"status": "running", "progress": "reviewing"})

            # Run review (blocking LLM calls, offloaded to thread pool)
            review: ReviewResult
            research: ResearchResult
            report_md: str
            review, research, report_md = await asyncio.to_thread(
                self.orchestrator.review_with_intermediates, code, language
            )

            # Persist results
            now = datetime.now(UTC).isoformat()
            with get_connection(self.db_path) as conn:
                update_task(
                    conn,
                    task_id,
                    {
                        "status": "completed",
                        "progress": None,
                        "report_md": report_md,
                        "score": review.score,
                        "issues_json": json.dumps(review.issues, ensure_ascii=False),
                        "reviewer_data_json": json.dumps(
                            {
                                "summary": review.summary,
                                "score": review.score,
                                "issues": review.issues,
                            },
                            ensure_ascii=False,
                        ),
                        "researcher_data_json": json.dumps(
                            {
                                "technologies": research.technologies,
                                "best_practices": research.best_practices,
                                "common_pitfalls": research.common_pitfalls,
                                "recommendations": research.recommendations,
                                "references": research.references,
                            },
                            ensure_ascii=False,
                        ),
                        "completed_at": now,
                    },
                )

        except Exception as exc:
            with get_connection(self.db_path) as conn:
                update_task(
                    conn,
                    task_id,
                    {
                        "status": "failed",
                        "progress": None,
                        "error": str(exc),
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                )
        finally:
            self._running_tasks.pop(task_id, None)


def _hash_token(token: str) -> str:
    """SHA256 hash of token. We never store raw tokens."""
    return hashlib.sha256(token.encode()).hexdigest()
