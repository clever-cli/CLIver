"""SQLite-backed task store for the gateway.

Schema:
    tasks(name, yaml_path, session_id, created_at, updated_at,
          origin_*, state_status, state_suspend_reason)    — task registry
    task_runs(task_name, execution_id, status, ...)        — run history
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from cliver.db import get_store
from cliver.task_manager import TaskRun

if TYPE_CHECKING:
    from cliver.task_manager import TaskOrigin

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    name TEXT PRIMARY KEY,
    yaml_path TEXT NOT NULL,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    origin_source TEXT,
    origin_platform TEXT,
    origin_channel_id TEXT,
    origin_thread_id TEXT,
    origin_user_id TEXT,
    state_status TEXT,
    state_suspend_reason TEXT
);

CREATE TABLE IF NOT EXISTS task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    result TEXT,
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_runs_task_name
    ON task_runs(task_name);

CREATE INDEX IF NOT EXISTS idx_task_runs_started_at
    ON task_runs(task_name, started_at DESC);
"""


class TaskStore:
    """SQLite store for task registry, execution history, and live state."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._store = get_store(db_path)
        self._store.execute_schema(_SCHEMA)

    # -- Task registry --------------------------------------------------------

    def register_task(self, name: str, yaml_path: str) -> None:
        """Register a task in the database (UPSERT)."""
        from cliver.util import format_datetime

        now = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
        with self._store.write() as db:
            db.execute(
                """INSERT INTO tasks (name, yaml_path, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name)
                   DO UPDATE SET yaml_path=excluded.yaml_path,
                                 updated_at=excluded.updated_at""",
                (name, yaml_path, now, now),
            )

    def get_registered_task(self, name: str) -> Optional[dict]:
        """Get a registered task by name."""
        with self._store.read() as db:
            row = db.execute(
                "SELECT name, yaml_path, session_id, created_at, updated_at FROM tasks WHERE name = ?",
                (name,),
            ).fetchone()
            return dict(row) if row else None

    def list_registered_tasks(self) -> List[dict]:
        """List all registered tasks."""
        with self._store.read() as db:
            rows = db.execute(
                "SELECT name, yaml_path, session_id, created_at, updated_at FROM tasks ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]

    def unregister_task(self, name: str) -> bool:
        """Remove a task from the registry. Returns True if a row was deleted."""
        with self._store.write() as db:
            cursor = db.execute("DELETE FROM tasks WHERE name = ?", (name,))
            return cursor.rowcount > 0

    # -- Session linkage ------------------------------------------------------

    def set_session_id(self, task_name: str, session_id: str) -> None:
        """Link a task to a conversation session."""
        from cliver.util import format_datetime

        now = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
        with self._store.write() as db:
            db.execute(
                "UPDATE tasks SET session_id=?, updated_at=? WHERE name=?",
                (session_id, now, task_name),
            )

    def get_session_id(self, task_name: str) -> Optional[str]:
        """Get the session ID linked to a task."""
        with self._store.read() as db:
            row = db.execute(
                "SELECT session_id FROM tasks WHERE name = ?",
                (task_name,),
            ).fetchone()
            return row["session_id"] if row else None

    # -- Task origin (columns on the tasks table) -----------------------------

    def save_origin(self, task_name: str, origin: TaskOrigin) -> None:
        """Update origin columns on an existing task row."""
        from cliver.util import format_datetime

        now = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
        with self._store.write() as db:
            db.execute(
                """UPDATE tasks
                   SET origin_source=?, origin_platform=?,
                       origin_channel_id=?, origin_thread_id=?,
                       origin_user_id=?, updated_at=?
                   WHERE name=?""",
                (
                    origin.source,
                    origin.platform,
                    origin.channel_id,
                    origin.thread_id,
                    origin.user_id,
                    now,
                    task_name,
                ),
            )

    def get_origin(self, task_name: str) -> Optional[TaskOrigin]:
        """Read origin from the tasks row. Returns None if no origin is set."""
        from cliver.task_manager import TaskOrigin

        with self._store.read() as db:
            row = db.execute(
                """SELECT origin_source, origin_platform, origin_channel_id,
                          origin_thread_id, origin_user_id
                   FROM tasks WHERE name = ?""",
                (task_name,),
            ).fetchone()
        if not row or not row["origin_source"]:
            return None
        return TaskOrigin(**{k.removeprefix("origin_"): row[k] for k in row.keys() if row[k] is not None})

    def delete_origin(self, task_name: str) -> None:
        """Clear origin columns on a task row."""
        with self._store.write() as db:
            db.execute(
                """UPDATE tasks
                   SET origin_source=NULL, origin_platform=NULL,
                       origin_channel_id=NULL, origin_thread_id=NULL,
                       origin_user_id=NULL
                   WHERE name=?""",
                (task_name,),
            )

    # -- Task state (columns on the tasks table) ------------------------------

    def set_task_state(self, task_name: str, status: str, reason: Optional[str] = None) -> None:
        """Update live state columns on an existing task row."""
        from cliver.util import format_datetime

        now = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
        with self._store.write() as db:
            db.execute(
                """UPDATE tasks
                   SET state_status=?, state_suspend_reason=?, updated_at=?
                   WHERE name=?""",
                (status, reason, now, task_name),
            )

    def get_task_state(self, task_name: str) -> Optional[dict]:
        """Read live state from the tasks row. Returns None if no state is set."""
        with self._store.read() as db:
            row = db.execute(
                """SELECT state_status, state_suspend_reason, updated_at
                   FROM tasks WHERE name = ?""",
                (task_name,),
            ).fetchone()
        if not row or not row["state_status"]:
            return None
        return {
            "task_name": task_name,
            "status": row["state_status"],
            "updated_at": row["updated_at"],
            "suspend_reason": row["state_suspend_reason"],
        }

    def get_tasks_by_status(self, status: str) -> List[dict]:
        """Find tasks with a specific live state."""
        with self._store.read() as db:
            rows = db.execute(
                "SELECT name, state_status, updated_at, state_suspend_reason FROM tasks WHERE state_status = ?",
                (status,),
            ).fetchall()
        return [
            {
                "task_name": r["name"],
                "status": r["state_status"],
                "updated_at": r["updated_at"],
                "suspend_reason": r["state_suspend_reason"],
            }
            for r in rows
        ]

    def delete_task_state(self, task_name: str) -> None:
        """Clear state columns on a task row."""
        with self._store.write() as db:
            db.execute(
                "UPDATE tasks SET state_status=NULL, state_suspend_reason=NULL WHERE name=?",
                (task_name,),
            )

    # -- Run history ---------------------------------------------------------

    def record_run(self, run: TaskRun) -> None:
        with self._store.write() as db:
            db.execute(
                """INSERT INTO task_runs
                   (task_name, execution_id, status, started_at, finished_at, error, result, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.task_name,
                    run.execution_id,
                    run.status,
                    run.started_at,
                    run.finished_at,
                    run.error,
                    run.result,
                    run.session_id,
                ),
            )

    def get_runs(self, task_name: str, limit: int = 10) -> List[TaskRun]:
        with self._store.read() as db:
            rows = db.execute(
                """SELECT task_name, execution_id, status,
                          started_at, finished_at, error, result, session_id
                   FROM task_runs
                   WHERE task_name = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (task_name, limit),
            ).fetchall()
            return [TaskRun(**dict(r)) for r in rows]

    def delete_runs(self, task_name: str) -> int:
        with self._store.write() as db:
            cursor = db.execute(
                "DELETE FROM task_runs WHERE task_name = ?",
                (task_name,),
            )
            return cursor.rowcount

    def get_last_run_time(self, task_name: str) -> Optional[float]:
        with self._store.read() as db:
            row = db.execute(
                """SELECT started_at FROM task_runs
                   WHERE task_name = ?
                   ORDER BY id DESC LIMIT 1""",
                (task_name,),
            ).fetchone()
            if not row:
                return None
            return _parse_timestamp(row["started_at"])

    def get_all_task_names(self) -> List[str]:
        with self._store.read() as db:
            rows = db.execute("SELECT DISTINCT task_name FROM task_runs").fetchall()
            return [r["task_name"] for r in rows]

    def close(self) -> None:
        self._store.close()


def _parse_timestamp(ts: str) -> float:
    ts = ts.strip()
    if ts.endswith(" UTC"):
        ts = ts[:-4]
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return 0.0
