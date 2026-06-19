"""Task management routes for the admin portal."""

from __future__ import annotations

import logging
import re
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Fields clients are allowed to set on task create/update.
_ALLOWED_TASK_FIELDS = frozenset(
    {
        "name",
        "description",
        "prompt",
        "skills",
        "model",
        "agent",
        "schedule",
        "run_at",
        "workflow",
    }
)

# Task names must be filesystem-safe: alphanumeric, hyphens, underscores only.
_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_task_name(name: str) -> None:
    """Reject task names containing path separators, null bytes, or other
    dangerous characters that could enable path traversal."""
    if not _TASK_NAME_RE.match(name):
        raise ValueError(
            "Task name must be 1-128 characters, start with an alphanumeric, "
            "and contain only letters, digits, hyphens, underscores, or dots."
        )


def _pick_allowed(body: dict, *, extra: dict | None = None) -> dict:
    """Return only the allowed task fields from *body*, merged with *extra*."""
    data = {k: v for k, v in body.items() if k in _ALLOWED_TASK_FIELDS}
    if extra:
        data.update(extra)
    return data


def _get_tasks(ctx: dict) -> list:
    try:
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        gateway = ctx.get("gateway")
        if not gateway:
            return []
        store = TaskStore(gateway._agent_profile.db_path)
        tm = TaskManager(gateway._agent_profile.tasks_dir, store)
        entries = tm.list_task_entries()

        result = []
        for entry in entries:
            if entry.status == "active" and entry.definition:
                data = entry.definition.model_dump(exclude_none=True)
            else:
                data = {"name": entry.name}
            data["task_status"] = entry.status
            if entry.error:
                data["task_error"] = entry.error
            if entry.created_at:
                data["created_at"] = entry.created_at
            if entry.updated_at:
                data["updated_at"] = entry.updated_at
            state = store.get_task_state(entry.name)
            if state:
                data["live_status"] = state
            runs = store.get_runs(entry.name, limit=1)
            if runs:
                data["last_run"] = runs[0].model_dump(exclude_none=True)
            origin = store.get_origin(entry.name)
            if origin:
                data["origin"] = origin.model_dump(exclude_none=True)
            result.append(data)

        store.close()
        return result
    except Exception as e:
        logger.warning("Failed to get tasks: %s", e)
        return []


def _get_task_detail(ctx, task_name):
    try:
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        gateway = ctx.get("gateway")
        if not gateway:
            return None
        store = TaskStore(gateway._agent_profile.db_path)
        tm = TaskManager(gateway._agent_profile.tasks_dir, store)
        task_entry = tm.get_task_entry(task_name)
        if not task_entry:
            return None
        if task_entry.status == "active" and task_entry.definition:
            result = task_entry.definition.model_dump(exclude_none=True)
        else:
            result = {"name": task_entry.name}
        result["task_status"] = task_entry.status
        if task_entry.error:
            result["task_error"] = task_entry.error
        if task_entry.created_at:
            result["created_at"] = task_entry.created_at
        if task_entry.updated_at:
            result["updated_at"] = task_entry.updated_at
        state = store.get_task_state(task_name)
        if state:
            result["live_status"] = state
        runs = store.get_runs(task_name, limit=10)
        result["runs"] = [r.model_dump(exclude_none=True) for r in runs]
        origin = store.get_origin(task_name)
        if origin:
            result["origin"] = origin.model_dump(exclude_none=True)
        store.close()
        return result
    except Exception as e:
        logger.warning("Failed to get task detail: %s", e)
        return None


async def _run_task(ctx: dict, task_name: str) -> dict:
    try:
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        gateway = ctx.get("gateway")
        if not gateway:
            return {"status": "error", "message": "gateway not available"}

        store = TaskStore(gateway._agent_profile.db_path)
        tm = TaskManager(gateway._agent_profile.tasks_dir, store)
        task = tm.get_task(task_name)
        if not task:
            return {"status": "error", "message": f"task '{task_name}' not found"}

        import asyncio

        asyncio.create_task(gateway._run_task(task))
        return {"status": "started"}
    except Exception as e:
        logger.warning("Failed to run task: %s", e)
        return {"status": "error", "message": str(e)}


def get_task_routes(context: dict, require_auth: Callable) -> list:
    """Return task management API routes."""

    @require_auth
    async def handle_tasks(request: Request):
        from cliver.gateway.admin import _run_in_thread

        tasks = await _run_in_thread(_get_tasks, context)
        return JSONResponse(tasks)

    @require_auth
    async def handle_task_detail_api(request: Request):
        from cliver.gateway.admin import _run_in_thread

        name = request.path_params["name"]
        try:
            _validate_task_name(name)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        detail = await _run_in_thread(_get_task_detail, context, name)
        return JSONResponse(detail)

    async def _save_task(body: dict, *, action: str) -> JSONResponse:
        """Shared create/update logic."""
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskDefinition, TaskManager

        gateway = context.get("gateway")
        if not gateway:
            return JSONResponse({"error": "Gateway not available"}, status_code=500)

        task = TaskDefinition(**body)

        store = TaskStore(gateway._agent_profile.db_path)
        tm = TaskManager(gateway._agent_profile.tasks_dir, store)
        tm.save_task(task)
        store.close()

        if gateway._scheduler:
            gateway._scheduler.sync_tasks()

        logger.info("[admin] Task '%s' %s via admin portal", task.name, action)
        status = 201 if action == "created" else 200
        return JSONResponse(task.model_dump(exclude_none=True), status_code=status)

    @require_auth
    async def handle_create_task(request: Request):
        try:
            body = _pick_allowed(await request.json())
            _validate_task_name(body["name"])
            return await _save_task(body, action="created")
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.error("Failed to create task: %s", e)
            return JSONResponse({"error": str(e)}, status_code=400)

    @require_auth
    async def handle_update_task(request: Request):
        try:
            task_name = request.path_params["name"]
            _validate_task_name(task_name)
            body = _pick_allowed(await request.json(), extra={"name": task_name})
            return await _save_task(body, action="updated")
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.error("Failed to update task '%s': %s", task_name, e)
            return JSONResponse({"error": str(e)}, status_code=400)

    @require_auth
    async def handle_run_task(request: Request):
        task_name = request.path_params["name"]
        try:
            _validate_task_name(task_name)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        logger.info("[admin] Task '%s' triggered via admin portal", task_name)
        result = await _run_task(context, task_name)
        status_code = 200 if result.get("status") == "started" else 400
        return JSONResponse(result, status_code=status_code)

    @require_auth
    async def handle_delete_task(request: Request):
        task_name = request.path_params["name"]
        try:
            _validate_task_name(task_name)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        logger.info("[admin] Task '%s' deleted via admin portal", task_name)
        try:
            from cliver.gateway.task_store import TaskStore
            from cliver.task_manager import TaskManager

            gateway = context.get("gateway")
            if not gateway:
                return JSONResponse({"error": "Gateway not available"}, status_code=500)

            store = TaskStore(gateway._agent_profile.db_path)
            tm = TaskManager(gateway._agent_profile.tasks_dir, store)
            removed = tm.remove_task(task_name)

            deleted_runs = 0
            if removed:
                deleted_runs = store.delete_runs(task_name)
            store.close()

            if not removed:
                return JSONResponse({"error": "Task not found"}, status_code=404)

            return JSONResponse({"status": "deleted", "runs_removed": deleted_runs})
        except Exception as e:
            logger.error("Failed to delete task '%s': %s", task_name, e)
            return JSONResponse({"error": str(e)}, status_code=500)

    return [
        Route("/admin/api/tasks", handle_tasks),
        Route("/admin/api/tasks", handle_create_task, methods=["POST"]),
        Route("/admin/api/tasks/{name}", handle_task_detail_api),
        Route("/admin/api/tasks/{name}", handle_update_task, methods=["PUT"]),
        Route("/admin/api/tasks/{name}/run", handle_run_task, methods=["POST"]),
        Route("/admin/api/tasks/{name}", handle_delete_task, methods=["DELETE"]),
    ]
