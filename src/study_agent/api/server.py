# mypy: disable-error-code="misc,no-untyped-def,untyped-decorator,type-arg,import-not-found"
"""FastAPI server for the Multi-Agent Code Review System.

Exposes 5 endpoints:
  - POST /api/review          (manual code submit)
  - POST /api/review/pr       (PR URL submit)
  - POST /api/webhook/github  (GitHub webhook receiver)
  - GET  /api/task/{task_id}  (task status)
  - GET  /api/report/{task_id} (review report)
  - GET  /health              (health check)

Start with:
    poetry run uvicorn study_agent.api.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from study_agent.api.models import (
    PRReviewRequest,
    ReportResponse,
    ReviewRequest,
    TaskResponse,
)
from study_agent.api.task_manager import TaskManager
from study_agent.github import WebhookVerifier

logger = logging.getLogger(__name__)

# Global TaskManager instance -- initialized in the lifespan handler so
# the server can start without LLM credentials configured.
task_manager: TaskManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and tear down the shared TaskManager singleton.

    The database path and LLM provider are read from environment variables
    so the same server binary can target different configurations without
    recompilation.
    """
    global task_manager
    db_path = os.environ.get("DB_PATH", "data/reviews.db")
    llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
    task_manager = TaskManager(db_path=db_path, llm_provider=llm_provider)
    logger.info("TaskManager initialized: db=%s, llm=%s", db_path, llm_provider)
    yield
    logger.info("Shutting down")
    task_manager = None


app = FastAPI(
    title="Code Review Agent API",
    description="Multi-Agent Code Review System -- submit code or PRs for automated review",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Helper -------------------------------------------------------------------


def _require_tm() -> TaskManager:
    """Return the global TaskManager or raise 500 if not initialized."""
    if task_manager is None:
        raise HTTPException(500, "TaskManager not initialized")
    return task_manager


# -- Endpoints ----------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check -- always returns ok so load balancers can probe."""
    return {"status": "ok"}


@app.post("/api/review", status_code=202)
async def submit_review(request: ReviewRequest):
    """Submit code for review. Returns a task_id immediately; the review
    runs in the background.
    """
    tm = _require_tm()
    task_id = tm.submit_review(
        code=request.code,
        language=request.language,
        github_token=request.github_token,
        pr_url=request.pr_url,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/api/review/pr", status_code=202)
async def submit_pr_review(request: PRReviewRequest):
    """Submit a GitHub PR URL for review. The system fetches the diff,
    launches a background review, and returns a task_id.
    """
    tm = _require_tm()
    task_id = tm.submit_pr_review(
        pr_url=request.pr_url,
        github_token=request.github_token,
    )

    # PRDiffFetcher may have failed immediately -- check for that.
    task = tm.get_task(task_id)
    if task and task.get("status") == "failed":
        raise HTTPException(400, detail=task.get("error", "Failed to fetch PR diff"))

    return {"task_id": task_id, "status": "queued"}


@app.post("/api/webhook/github", status_code=202)
async def github_webhook(request: Request):
    """Receive GitHub webhook events.

    When the webhook payload contains a ``pull_request`` object the system
    enqueues a review of the affected PR.

    If ``WEBHOOK_SECRET`` is set in the environment the ``X-Hub-Signature-256``
    header is validated via HMAC-SHA256 before the payload is processed.
    """
    tm = _require_tm()

    body: bytes = await request.body()
    signature: str = request.headers.get("X-Hub-Signature-256", "")
    secret: str = os.environ.get("WEBHOOK_SECRET", "")

    # -- Signature verification (when secret is configured) --------------------
    if secret:
        result = WebhookVerifier.verify(body, signature, secret)
        if not result.valid:
            raise HTTPException(401, detail=f"Invalid signature: {result.error}")

    # -- Parse JSON body -------------------------------------------------------
    try:
        payload: dict = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid JSON body")

    event_type: str = request.headers.get("X-GitHub-Event", "unknown")

    # -- Handle pull_request events --------------------------------------------
    pr_data = payload.get("pull_request")
    if event_type == "pull_request" or pr_data is not None:
        if pr_data is None:
            return {"message": "pull_request event but no PR data in payload", "event": event_type}

        pr_url: str = pr_data.get("html_url", "")
        if not pr_url:
            return {"message": "PR URL not found in webhook payload", "event": event_type}

        bot_token: str = os.environ.get("GITHUB_BOT_TOKEN", "")
        if not bot_token:
            return {
                "message": "Webhook received but GITHUB_BOT_TOKEN not configured",
                "event": event_type,
            }

        task_id = tm.submit_pr_review(pr_url=pr_url, github_token=bot_token)
        return {"task_id": task_id, "status": "queued", "event": event_type}

    return {
        "message": f"Event type '{event_type}' ignored (only pull_request is handled)",
        "event": event_type,
    }


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Return the current status of a review task.

    Returns 404 when the task_id does not exist.
    """
    tm = _require_tm()
    task = tm.get_task(task_id)
    if task is None:
        raise HTTPException(404, detail=f"Task '{task_id}' not found")

    return TaskResponse(
        task_id=task_id,
        status=task.get("status", "queued"),
        progress=task.get("progress"),
        created_at=task.get("created_at", ""),
        result=None,
        error=task.get("error"),
    )


@app.get("/api/report/{task_id}")
async def get_report(task_id: str):
    """Return the full review report for a completed task.

    Returns 404 when the task does not exist or the review has not completed yet.
    """
    tm = _require_tm()
    report = tm.get_report(task_id)
    if report is None:
        task = tm.get_task(task_id)
        if task is None:
            raise HTTPException(404, detail=f"Task '{task_id}' not found")
        raise HTTPException(404, detail=f"Report not ready. Task status: {task.get('status')}")
    return ReportResponse(**report)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
