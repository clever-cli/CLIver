"""
Task Manager — scheduled prompt-based tasks with agent ownership.

A task is a YAML definition that provides a prompt for the LLM,
optional model override, and optional cron schedule. Tasks are stored
per-agent in {config_dir}/agents/{agent_name}/tasks/.

The **database** (``tasks`` table in cliver.db) is the source of truth
for which tasks exist.  Each row stores a ``yaml_path`` (relative to
tasks_dir) that points to the YAML file holding the task content.
YAML files without a corresponding database row are ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cliver.gateway.task_store import TaskStore

logger = logging.getLogger(__name__)


class TaskOrigin(BaseModel):
    """Where a task was created and where results should be delivered.

    This is purely about routing — platform, channel, thread, user.
    Session linkage is on TaskDefinition.session_id, not here.
    """

    source: str = Field(..., description="Origin source: cli, api, slack, telegram, discord, feishu")
    platform: Optional[str] = Field(None, description="Adapter name for IM sources")
    channel_id: Optional[str] = Field(None, description="IM channel to reply to")
    thread_id: Optional[str] = Field(None, description="IM thread to reply in")
    user_id: Optional[str] = Field(None, description="Who created the task")


class TaskDefinition(BaseModel):
    """A task definition — a prompt with optional skill activation."""

    name: str = Field(..., description="Unique task name")
    description: Optional[str] = Field(None, description="What this task does")
    prompt: str = Field(..., description="The prompt to send to the LLM")
    skills: Optional[List[str]] = Field(None, description="Skills to pre-activate in system prompt")
    model: Optional[str] = Field(None, description="Model override for this task")
    agent: Optional[str] = Field(None, description="Agent name to use (from config.yaml agents)")
    schedule: Optional[str] = Field(None, description="Cron expression for recurring execution")
    run_at: Optional[str] = Field(None, description="ISO 8601 datetime for one-shot execution")
    permissions: Optional[Any] = Field(None, description="Permission overrides for this task")
    origin: Optional[TaskOrigin] = Field(None, description="Where the task was created from")
    session_id: Optional[str] = Field(None, description="Linked conversation session ID")


class TaskRun(BaseModel):
    """Record of a single task execution."""

    task_name: str
    execution_id: str
    status: str  # completed, failed, cancelled
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None
    session_id: Optional[str] = None


class TaskEntry(BaseModel):
    """A task registry entry with YAML load status.

    Used by admin and list views that need to show broken tasks.
    """

    name: str
    yaml_path: str
    status: str  # 'active', 'yaml_missing', 'yaml_invalid'
    error: Optional[str] = None
    definition: Optional[TaskDefinition] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskManager:
    """Manages task definitions for a specific agent.

    The **database** is the source of truth for task existence.
    Each registered task has a ``yaml_path`` pointing to its YAML file.
    YAML files without a DB row are not recognised as valid tasks.

    Directory layout:
        {tasks_dir}/
        ├── daily-research.yaml        # task content
        └── code-review.yaml

    Run history is stored in cliver.db via TaskStore.
    """

    def __init__(self, tasks_dir: Path, run_store: TaskStore):
        self.tasks_dir = tasks_dir
        self.run_store = run_store

    def _ensure_dir(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    # -- CRUD ---------------------------------------------------------------

    def list_tasks(self) -> List[TaskDefinition]:
        """List all *valid* task definitions (DB-first).

        Returns only tasks whose YAML is present and parseable.
        Used by scheduler and runner — they only care about runnable tasks.
        """
        tasks = []
        for row in self.run_store.list_registered_tasks():
            path = self.tasks_dir / row["yaml_path"]
            if not path.exists():
                continue
            try:
                task = self._load_task_file(path)
                if task:
                    tasks.append(task)
            except Exception as e:
                logger.warning("Skipping invalid task '%s': %s", row["name"], e)
        return tasks

    def list_task_entries(self) -> List[TaskEntry]:
        """List all registered tasks with YAML load status.

        Returns every DB-registered task including those with missing
        or invalid YAML — for admin and CLI list views.
        """
        entries: List[TaskEntry] = []
        for row in self.run_store.list_registered_tasks():
            path = self.tasks_dir / row["yaml_path"]
            if not path.exists():
                entries.append(
                    TaskEntry(
                        name=row["name"],
                        yaml_path=row["yaml_path"],
                        status="yaml_missing",
                        error=f"YAML file not found: {row['yaml_path']}",
                        created_at=row.get("created_at"),
                        updated_at=row.get("updated_at"),
                    )
                )
                continue
            try:
                task = self._load_task_file(path)
                if task:
                    entries.append(
                        TaskEntry(
                            name=row["name"],
                            yaml_path=row["yaml_path"],
                            status="active",
                            definition=task,
                            created_at=row.get("created_at"),
                            updated_at=row.get("updated_at"),
                        )
                    )
                else:
                    entries.append(
                        TaskEntry(
                            name=row["name"],
                            yaml_path=row["yaml_path"],
                            status="yaml_invalid",
                            error="YAML file is empty or not a dict",
                            created_at=row.get("created_at"),
                            updated_at=row.get("updated_at"),
                        )
                    )
            except Exception as e:
                entries.append(
                    TaskEntry(
                        name=row["name"],
                        yaml_path=row["yaml_path"],
                        status="yaml_invalid",
                        error=str(e),
                        created_at=row.get("created_at"),
                        updated_at=row.get("updated_at"),
                    )
                )
        return entries

    def get_task(self, name: str) -> Optional[TaskDefinition]:
        """Get a task definition by name (DB-first).

        Returns None if the task is not registered in the DB,
        or if the YAML file is missing/invalid.
        """
        row = self.run_store.get_registered_task(name)
        if not row:
            return None
        path = self.tasks_dir / row["yaml_path"]
        if not path.exists():
            return None
        try:
            return self._load_task_file(path)
        except Exception as e:
            logger.warning("Invalid task file for '%s': %s", name, e)
            return None

    def get_task_entry(self, name: str) -> Optional[TaskEntry]:
        """Get a task registry entry with YAML load status.

        Returns the entry even if YAML is missing/invalid (with status info).
        Returns None only if the task is not registered in the DB at all.
        """
        row = self.run_store.get_registered_task(name)
        if not row:
            return None
        path = self.tasks_dir / row["yaml_path"]
        if not path.exists():
            return TaskEntry(
                name=row["name"],
                yaml_path=row["yaml_path"],
                status="yaml_missing",
                error=f"YAML file not found: {row['yaml_path']}",
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
        try:
            task = self._load_task_file(path)
            if task:
                return TaskEntry(
                    name=row["name"],
                    yaml_path=row["yaml_path"],
                    status="active",
                    definition=task,
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                )
            return TaskEntry(
                name=row["name"],
                yaml_path=row["yaml_path"],
                status="yaml_invalid",
                error="YAML file is empty or not a dict",
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
        except Exception as e:
            return TaskEntry(
                name=row["name"],
                yaml_path=row["yaml_path"],
                status="yaml_invalid",
                error=str(e),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )

    def save_task(self, task: TaskDefinition) -> Path:
        """Save a task definition to YAML and register in DB.

        Origin and session_id are NOT stored in YAML — they belong in the
        database (columns on the tasks table). Callers that need to persist
        origin should use TaskStore.save_origin() separately. session_id
        is persisted via TaskStore.set_session_id().
        """
        self._ensure_dir()
        yaml_path = f"{task.name}.yaml"
        path = self.tasks_dir / yaml_path
        data = task.model_dump(exclude_none=True, exclude={"origin", "session_id"})
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        self.run_store.register_task(task.name, yaml_path)
        if task.session_id:
            self.run_store.set_session_id(task.name, task.session_id)
        logger.info("Task '%s' saved to %s and registered in DB", task.name, path)
        return path

    def remove_task(self, name: str) -> bool:
        """Remove a task from the registry and delete its YAML file."""
        row = self.run_store.get_registered_task(name)
        if not row:
            return False
        self.run_store.unregister_task(name)
        yaml_file = self.tasks_dir / row["yaml_path"]
        if yaml_file.exists():
            yaml_file.unlink()
        return True

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _load_task_file(path: Path) -> Optional[TaskDefinition]:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not data or not isinstance(data, dict):
            return None
        return TaskDefinition(**data)

    @staticmethod
    def timestamp_now() -> str:
        from cliver.util import format_datetime

        return format_datetime(fmt="%Y-%m-%d %H:%M:%S")
