"""Conversation CRUD routes for the admin portal general chat."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_conversations_routes(context: dict, require_auth: Callable) -> list:
    """Return conversation CRUD API routes (wraps SessionManager)."""
    from cliver.gateway.admin import _run_in_thread

    @require_auth
    async def handle_list(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        sessions = await _run_in_thread(session_manager.list_general_sessions)
        return JSONResponse(sessions)

    @require_auth
    async def handle_create(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        title = ""
        if request.headers.get("content-type") == "application/json":
            body = await request.json()
            if isinstance(body, dict):
                title = body.get("title", "")
        session_id = await _run_in_thread(session_manager.create_session, title)
        return JSONResponse({"id": session_id, "title": title or None})

    @require_auth
    async def handle_get(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        session_id = request.path_params["id"]
        info = await _run_in_thread(session_manager.get_session_info, session_id)
        if not info:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        turns = await _run_in_thread(session_manager.load_turns, session_id)
        return JSONResponse({"session": info, "turns": turns})

    @require_auth
    async def handle_delete(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        session_id = request.path_params["id"]
        deleted = await _run_in_thread(session_manager.delete_session, session_id)
        if not deleted:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_delete_turn(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        session_id = request.path_params["id"]
        try:
            turn_id = int(request.path_params["turn_id"])
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid turn_id"}, status_code=400)
        deleted = await _run_in_thread(session_manager.delete_turn, session_id, turn_id)
        if not deleted:
            return JSONResponse({"error": "Turn not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_update(request: Request):
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        session_id = request.path_params["id"]
        body = await request.json()
        title = body.get("title")
        options = body.get("options")
        if title is not None:
            await _run_in_thread(session_manager.update_title, session_id, title)
        if options is not None:
            await _run_in_thread(session_manager.merge_options, session_id, options)
        return JSONResponse({"status": "updated"})

    return [
        Route("/admin/api/conversations", handle_list),
        Route("/admin/api/conversations", handle_create, methods=["POST"]),
        Route("/admin/api/conversations/{id}", handle_get),
        Route("/admin/api/conversations/{id}", handle_delete, methods=["DELETE"]),
        Route("/admin/api/conversations/{id}", handle_update, methods=["PATCH"]),
        Route("/admin/api/conversations/{id}/turns/{turn_id}", handle_delete_turn, methods=["DELETE"]),
    ]
