"""Lightweight SQLite database layer for the code review API.

Provides schema initialization, a connection context manager,
and basic CRUD operations for the tasks and review_history tables.
"""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

# Project root is two levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_PATH = str(PROJECT_ROOT / "data" / "reviews.db")

# ── Schema ───────────────────────────────────────────────────────────────────

CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress TEXT,
    code TEXT,
    language TEXT,
    pr_url TEXT,
    github_token_hash TEXT,
    report_md TEXT,
    score INTEGER,
    issues_json TEXT,
    reviewer_data_json TEXT,
    researcher_data_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
)
"""

CREATE_REVIEW_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    issues_found INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    raw_response_length INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
)
"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Run CREATE TABLE IF NOT EXISTS on the given connection.

    Called by both ``init_db`` (for explicit setup) and ``get_connection``
    (so every new connection is guaranteed to have the schema, which is
    essential for ``:memory:`` databases that start fresh every time).
    """
    conn.execute(CREATE_TASKS_TABLE)
    conn.execute(CREATE_REVIEW_HISTORY_TABLE)
    conn.commit()


# ── Public API ───────────────────────────────────────────────────────────────


def init_db(db_path: str = DATABASE_PATH) -> None:
    """Create the database file (and its parent directory) plus tables.

    Args:
        db_path: Path to the SQLite database file.
                 Use ":memory:" for an in-memory database (no filesystem access).
    """
    # Ensure the parent directory exists (skip for :memory:).
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(conn)
    finally:
        conn.close()


@contextmanager
def get_connection(db_path: str = DATABASE_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a sqlite3.Connection configured for safe access.

    The connection uses:
    - ``row_factory = sqlite3.Row`` so that fetch results behave like dicts.
    - ``journal_mode = WAL`` for better concurrent read performance.
    - Schema is auto-created so the connection is always ready for CRUD operations.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def create_task(conn: sqlite3.Connection, task_data: dict[str, Any]) -> str:
    """Insert a new task row and return its id.

    Args:
        conn: An active database connection.
        task_data: A dict with keys:
            id, type, code, language, pr_url, github_token_hash.
            All other columns use their schema defaults.

    Returns:
        The ``id`` of the newly inserted task.
    """
    from datetime import UTC, datetime

    conn.execute(
        """
        INSERT INTO tasks (id, type, code, language, pr_url, github_token_hash, created_at)
        VALUES (:id, :type, :code, :language, :pr_url, :github_token_hash, :created_at)
        """,
        {
            "id": task_data["id"],
            "type": task_data["type"],
            "code": task_data.get("code"),
            "language": task_data.get("language"),
            "pr_url": task_data.get("pr_url"),
            "github_token_hash": task_data.get("github_token_hash"),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    conn.commit()
    return str(task_data["id"])


def update_task(conn: sqlite3.Connection, task_id: str, updates: dict[str, Any]) -> None:
    """Update one or more columns on an existing task row.

    Args:
        conn: An active database connection.
        task_id: The ``id`` of the task to update.
        updates: A dict mapping column names to new values.
                 Keys that do not match any column are silently ignored.
    """
    if not updates:
        return

    allowed_columns = {
        "status",
        "progress",
        "code",
        "language",
        "pr_url",
        "github_token_hash",
        "report_md",
        "score",
        "issues_json",
        "reviewer_data_json",
        "researcher_data_json",
        "error",
        "completed_at",
    }
    safe = {k: v for k, v in updates.items() if k in allowed_columns}
    if not safe:
        return

    set_clause = ", ".join(f"{col} = :{col}" for col in safe)
    safe["id"] = task_id
    conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = :id", safe)
    conn.commit()


def get_task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    """Return a single task as a dict, or ``None`` if not found.

    Args:
        conn: An active database connection.
        task_id: The ``id`` of the task to look up.

    Returns:
        A dict with all columns, or ``None``.
    """
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def get_tasks(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recently created tasks.

    Args:
        conn: An active database connection.
        limit: Maximum number of rows to return (default 20).

    Returns:
        A list of dicts, ordered by ``created_at DESC``.
    """
    rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
