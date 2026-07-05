"""Database layer tests -- SQLite schema + CRUD for tasks.

Uses :memory: for most tests (no filesystem hits).
File-based tests use tmp_path only where needed (e.g. WAL journal mode).
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from study_agent.api.database import (
    create_task,
    get_connection,
    get_task,
    get_tasks,
    init_db,
    update_task,
)


@pytest.fixture  # type: ignore[untyped-decorator]  # mypy <2.0 needs this
def conn() -> Generator[sqlite3.Connection, None, None]:
    """Create an in-memory database with schema, yield connection, then close."""
    init_db(":memory:")
    with get_connection(":memory:") as c:
        yield c


class TestInitDb:
    """Tests for init_db -- schema creation."""

    def test_init_db_creates_tasks_table(self) -> None:
        """init_db should create the 'tasks' table."""
        init_db(":memory:")
        with get_connection(":memory:") as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "tasks"

    def test_init_db_creates_review_history_table(self) -> None:
        """init_db should create the 'review_history' table."""
        init_db(":memory:")
        with get_connection(":memory:") as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='review_history'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "review_history"

    def test_init_db_is_idempotent(self) -> None:
        """Calling init_db twice should not raise an error."""
        init_db(":memory:")
        init_db(":memory:")  # Second call should be safe (IF NOT EXISTS)

    def test_init_db_creates_data_directory(self, tmp_path: Path) -> None:
        """init_db should create the parent directory for a file-based path."""
        db_path = str(tmp_path / "subdir" / "test.db")
        init_db(db_path)
        assert (tmp_path / "subdir").exists()
        assert (tmp_path / "subdir" / "test.db").exists()


class TestCreateTask:
    """Tests for create_task."""

    def test_create_task_inserts_and_returns_id(self, conn: sqlite3.Connection) -> None:
        """create_task should INSERT a row and return the task id."""
        task_data = {
            "id": "task-001",
            "type": "code_review",
            "code": "def foo(): pass",
            "language": "python",
            "pr_url": "https://github.com/example/repo/pull/1",
            "github_token_hash": "abc123hash",
        }
        result = create_task(conn, task_data)
        assert result == "task-001"

        row = conn.execute("SELECT * FROM tasks WHERE id = ?", ("task-001",)).fetchone()
        assert row is not None
        assert row["type"] == "code_review"
        assert row["status"] == "queued"
        assert row["code"] == "def foo(): pass"
        assert row["language"] == "python"

    def test_create_task_defaults_status_to_queued(self, conn: sqlite3.Connection) -> None:
        """create_task should set status to 'queued' by default."""
        task_data = {
            "id": "task-002",
            "type": "pr_review",
            "code": None,
            "language": None,
            "pr_url": "https://github.com/example/repo/pull/2",
            "github_token_hash": "hash456",
        }
        create_task(conn, task_data)
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", ("task-002",)).fetchone()
        assert row["status"] == "queued"


class TestUpdateTask:
    """Tests for update_task."""

    def test_update_task_modifies_fields(self, conn: sqlite3.Connection) -> None:
        """update_task should change specified fields on an existing row."""
        task_data = {
            "id": "task-003",
            "type": "code_review",
            "code": "def bar(): pass",
            "language": "python",
            "pr_url": "",
            "github_token_hash": "",
        }
        create_task(conn, task_data)

        update_task(conn, "task-003", {"status": "completed", "score": 95})

        row = conn.execute("SELECT * FROM tasks WHERE id = ?", ("task-003",)).fetchone()
        assert row["status"] == "completed"
        assert row["score"] == 95
        # Unchanged fields should remain
        assert row["type"] == "code_review"

    def test_update_task_nonexistent_does_not_raise(self, conn: sqlite3.Connection) -> None:
        """update_task on a missing id should not crash (UPDATE affects 0 rows)."""
        update_task(conn, "nonexistent", {"status": "failed"})
        # Should not raise

    def test_update_task_ignores_unknown_columns(self, conn: sqlite3.Connection) -> None:
        """update_task should silently drop keys that are not real columns."""
        task_data = {
            "id": "task-ignored",
            "type": "code_review",
            "code": "",
            "language": "",
            "pr_url": "",
            "github_token_hash": "",
        }
        create_task(conn, task_data)
        # "fake_column" is not a real column -- should be ignored without error
        update_task(conn, "task-ignored", {"status": "running", "fake_column": "oops"})
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", ("task-ignored",)).fetchone()
        assert row["status"] == "running"


class TestGetTask:
    """Tests for get_task."""

    def test_get_task_returns_dict(self, conn: sqlite3.Connection) -> None:
        """get_task should return a dict for an existing task."""
        task_data = {
            "id": "task-004",
            "type": "code_review",
            "code": "print(1)",
            "language": "python",
            "pr_url": "",
            "github_token_hash": "",
        }
        create_task(conn, task_data)

        result = get_task(conn, "task-004")
        assert isinstance(result, dict)
        assert result["id"] == "task-004"
        assert result["type"] == "code_review"
        assert result["code"] == "print(1)"

    def test_get_task_returns_none_for_missing_id(self, conn: sqlite3.Connection) -> None:
        """get_task should return None when the task does not exist."""
        result = get_task(conn, "nonexistent-id")
        assert result is None


class TestGetTasks:
    """Tests for get_tasks."""

    def test_get_tasks_returns_list_ordered_by_created_at_desc(
        self, conn: sqlite3.Connection
    ) -> None:
        """get_tasks should return tasks newest-first."""
        for i in range(3):
            create_task(
                conn,
                {
                    "id": f"task-{i:03d}",
                    "type": "code_review",
                    "code": f"code-{i}",
                    "language": "python",
                    "pr_url": "",
                    "github_token_hash": "",
                },
            )

        results = get_tasks(conn, limit=10)
        assert len(results) == 3
        # Newest first: task-002, task-001, task-000
        assert results[0]["id"] == "task-002"
        assert results[1]["id"] == "task-001"
        assert results[2]["id"] == "task-000"

    def test_get_tasks_respects_limit(self, conn: sqlite3.Connection) -> None:
        """get_tasks should return at most `limit` rows."""
        for i in range(5):
            create_task(
                conn,
                {
                    "id": f"task-{i:03d}",
                    "type": "code_review",
                    "code": "",
                    "language": "",
                    "pr_url": "",
                    "github_token_hash": "",
                },
            )

        results = get_tasks(conn, limit=2)
        assert len(results) == 2

    def test_get_tasks_empty_returns_empty_list(self, conn: sqlite3.Connection) -> None:
        """get_tasks should return an empty list when no tasks exist."""
        results = get_tasks(conn)
        assert results == []


class TestGetConnection:
    """Tests for get_connection context manager."""

    def test_get_connection_uses_wal_journal_mode(self, tmp_path: Path) -> None:
        """get_connection should set journal_mode to WAL on a file-based db."""
        db_path = str(tmp_path / "wal_test.db")
        init_db(db_path)
        with get_connection(db_path) as c:
            row = c.execute("PRAGMA journal_mode").fetchone()
        assert row[0].upper() == "WAL"

    def test_get_connection_uses_row_factory(self) -> None:
        """get_connection should set row_factory to sqlite3.Row."""
        init_db(":memory:")
        with get_connection(":memory:") as c:
            c.execute("CREATE TABLE IF NOT EXISTS test_row (col1 TEXT)")
            c.execute("INSERT INTO test_row VALUES ('hello')")
            row = c.execute("SELECT * FROM test_row").fetchone()
        # sqlite3.Row allows dict-like access by column name
        assert row["col1"] == "hello"
        # Also check it's actually a sqlite3.Row
        assert isinstance(row, sqlite3.Row)

    def test_get_connection_auto_creates_schema(self) -> None:
        """get_connection should ensure schema exists even without prior init_db."""
        # Skip init_db entirely -- get_connection alone should create the schema.
        with get_connection(":memory:") as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["name"] == "tasks"
