"""API 路由模块 -- FastAPI 端点。

/upload   -> 文件上传 + 文档索引
/chat     -> 问答接口
/kb       -> 知识库管理

Code Review API (server.py):
  POST /api/review        -> 手动提交代码审查
  POST /api/review/pr     -> PR URL 提交
  POST /api/webhook/github -> GitHub Webhook 接收
  GET  /api/task/{id}     -> 任务状态查询
  GET  /api/report/{id}   -> 审查报告
"""

from study_agent.api.database import (
    create_task,
    get_connection,
    get_task,
    get_tasks,
    init_db,
    update_task,
)
from study_agent.api.models import (
    ErrorResponse,
    Issue,
    PRReviewRequest,
    ReportResponse,
    ReviewRequest,
    TaskProgress,
    TaskResponse,
    TaskStatus,
    WebhookPayload,
)
from study_agent.api.server import app as code_review_app
from study_agent.api.task_manager import TaskManager

__all__ = [
    # Database
    "get_connection",
    "create_task",
    "update_task",
    "get_task",
    "get_tasks",
    "init_db",
    # Models
    "TaskStatus",
    "TaskProgress",
    "ReviewRequest",
    "PRReviewRequest",
    "WebhookPayload",
    "TaskResponse",
    "Issue",
    "ReportResponse",
    "ErrorResponse",
    # TaskManager
    "TaskManager",
    # Server
    "code_review_app",
]
