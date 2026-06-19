"""Task scheduler backed by APScheduler (AsyncIOScheduler).

Jobs are synced from task configuration on startup and every 10 minutes.
Uses MemoryJobStore (default) — DB persistence is unnecessary since
sync_tasks() rebuilds all jobs from config on every startup.
Supports one-shot (run_at) and recurring (cron) tasks.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Coroutine

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskDefinition, TaskManager

logger = logging.getLogger(__name__)

RunTaskFn = Callable[[TaskDefinition], Coroutine[Any, Any, None]]


class Scheduler:
    """Wraps APScheduler (AsyncIOScheduler) with in-memory job storage.

    Tasks are synced on startup via sync_tasks(). Every registered task
    gets a scheduled job — one-shot (run_at) or recurring (cron).

    Uses MemoryJobStore (the APScheduler default) because all jobs are
    rebuilt from configuration on every startup — no DB persistence needed.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        run_store: TaskStore,
        run_task_fn: RunTaskFn,
    ):
        self._task_manager = task_manager
        self._run_store = run_store
        self._run_task_fn = run_task_fn

        # Use default MemoryJobStore — jobs are rebuilt from config
        # on every startup via sync_tasks(), so DB persistence isn't needed.
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        self._scheduler.shutdown(wait=wait)
        logger.info("Scheduler stopped")

    # ── Task sync ───────────────────────────────────────────

    def sync_tasks(self) -> None:
        """Sync all configured tasks to APScheduler jobs.

        Adds jobs for scheduled tasks, removes jobs for manual/deleted tasks.
        """
        existing_jobs = {j.id for j in self._scheduler.get_jobs()}
        all_tasks = {t.name: t for t in self._task_manager.list_tasks()}
        configured = set(all_tasks)
        scheduled = {n for n, t in all_tasks.items() if t.schedule or t.run_at}

        # Remove jobs for deleted or now-manual tasks
        for job_id in existing_jobs - scheduled:
            self._scheduler.remove_job(job_id)
            reason = "deleted" if job_id not in configured else "now manual"
            logger.info("Removed job for %s task '%s'", reason, job_id)

        # Add or update jobs for scheduled tasks
        for name in scheduled:
            self._add_or_update_job(all_tasks[name])
            if not all_tasks[name].prompt or not all_tasks[name].prompt.strip():
                logger.warning("Task '%s' has an empty prompt", name)

    def _add_or_update_job(self, task: TaskDefinition) -> None:
        """Add or replace an APScheduler job for a scheduled task.

        Only called for tasks with a schedule or run_at — manual tasks
        never reach this method.
        """
        trigger = self._build_trigger(task)
        assert trigger is not None, f"Expected trigger for scheduled task '{task.name}'"

        task_name = task.name

        async def run():
            t = self._task_manager.get_task(task_name)
            if not t:
                logger.warning("Task '%s' not found — removing stale job", task_name)
                try:
                    self._scheduler.remove_job(task_name)
                except Exception:
                    pass
                return
            try:
                await self._run_task_fn(t)
            except Exception as e:
                logger.error("Task '%s' failed: %s", task_name, e)

        self._scheduler.add_job(
            func=run,
            trigger=trigger,
            id=task_name,
            name=task_name,
            replace_existing=True,
        )

    def _build_trigger(self, task: TaskDefinition):
        """Build an APScheduler trigger for a task.  Returns None if not schedulable."""
        # One-shot
        if task.run_at:
            try:
                run_time = datetime.fromisoformat(task.run_at)
                if run_time.tzinfo is None:
                    from cliver.util import get_effective_timezone

                    run_time = run_time.replace(tzinfo=get_effective_timezone())
                return DateTrigger(run_date=run_time)
            except ValueError as e:
                logger.warning("Invalid run_at for task '%s': %s", task.name, e)
                return None

        # Cron
        if task.schedule:
            try:
                return CronTrigger.from_crontab(task.schedule)
            except (ValueError, KeyError) as e:
                logger.warning("Invalid cron for task '%s': %s", task.name, e)
                return None

        return None

    # ── Validation ──────────────────────────────────────────

    def validate_tasks(self) -> None:
        """Validate cron expressions, run_at datetimes, and empty prompts."""
        for task in self._task_manager.list_tasks():
            if not task.prompt or not task.prompt.strip():
                logger.warning("Task '%s' has an empty prompt", task.name)
            if task.schedule:
                try:
                    CronTrigger.from_crontab(task.schedule)
                except (ValueError, KeyError):
                    logger.warning("Task '%s' has invalid cron '%s'", task.name, task.schedule)
            if task.run_at:
                try:
                    datetime.fromisoformat(task.run_at)
                except ValueError:
                    logger.warning("Task '%s' has invalid run_at '%s'", task.name, task.run_at)

    def cleanup_orphan_runs(self) -> int:
        """Delete run records for tasks not in the database."""
        registered = {r["name"] for r in self._run_store.list_registered_tasks()}
        recorded = set(self._run_store.get_all_task_names())
        orphans = recorded - registered

        total = 0
        for name in orphans:
            deleted = self._run_store.delete_runs(name)
            total += deleted
            logger.info("Cleaned up %d orphan run record(s) for unregistered task '%s'", deleted, name)
        return total
