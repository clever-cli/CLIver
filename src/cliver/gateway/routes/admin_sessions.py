"""Session management routes for the admin portal.

Sessions are stored in the ``sessions`` / ``turns`` tables in cliver.db.
One SessionManager instance is shared across all sources (CLI, gateway, lab).
"""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _get_session_manager(ctx) -> object | None:
    """Return the shared SessionManager — prefer gateway, fall back to CLI."""
    gw = ctx.get("gateway")
    if gw and getattr(gw, "_session_manager", None):
        return gw._session_manager
    return ctx.get("cli_session_manager")


def _enrich_session(s: dict) -> dict:
    """Add display_title and optional platform to a session dict."""
    title = s.get("title") or ""
    if ":" in title:
        parts = title.split(":", 2)
        s["platform"] = parts[0]
        s["display_title"] = parts[2] if len(parts) > 2 else parts[1]
    elif title:
        s["display_title"] = title[:80]
    return s


def _list_all_sessions(ctx) -> list[dict]:
    sm = _get_session_manager(ctx)
    if not sm:
        return []
    try:
        sessions = sm.list_sessions()
        return [_enrich_session(s) for s in sessions]
    except Exception as e:
        logger.warning("Failed to list sessions: %s", e)
        return []


def _get_session_turns(ctx, session_id: str) -> list[dict]:
    sm = _get_session_manager(ctx)
    if not sm:
        return []
    try:
        return sm.load_turns(session_id)
    except Exception as e:
        logger.warning("Failed to load session turns: %s", e)
        return []


def _delete_session(ctx, session_id: str) -> bool:
    sm = _get_session_manager(ctx)
    if not sm:
        return False
    try:
        return sm.delete_session(session_id)
    except Exception as e:
        logger.warning("Failed to delete session: %s", e)
        return False


def get_session_routes(context: dict, require_auth: Callable) -> list:
    """Return session management API routes."""

    @require_auth
    async def handle_list_sessions(request: Request):
        from cliver.gateway.admin import _run_in_thread

        sessions = await _run_in_thread(_list_all_sessions, context)
        return JSONResponse(sessions)

    @require_auth
    async def handle_get_turns(request: Request):
        from cliver.gateway.admin import _run_in_thread

        session_id = request.path_params["id"]
        turns = await _run_in_thread(_get_session_turns, context, session_id)
        return JSONResponse(turns)

    @require_auth
    async def handle_delete_session(request: Request):
        session_id = request.path_params["id"]
        logger.info("[admin] Session '%s' deleted via admin portal", session_id)
        deleted = _delete_session(context, session_id)
        if deleted:
            return JSONResponse({"status": "deleted"})
        return JSONResponse({"error": "session not found"}, status_code=404)

    @require_auth
    async def handle_delete_turn(request: Request):
        sm = _get_session_manager(context)
        if not sm:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)
        session_id = request.path_params["id"]
        try:
            turn_id = int(request.path_params["turn_id"])
        except (ValueError, TypeError):
            return JSONResponse({"error": "Invalid turn_id"}, status_code=400)
        try:
            deleted = sm.delete_turn(session_id, turn_id)
        except Exception as e:
            logger.warning("Failed to delete turn: %s", e)
            return JSONResponse({"error": "Failed to delete turn"}, status_code=500)
        if not deleted:
            return JSONResponse({"error": "Turn not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    return [
        Route("/admin/api/sessions", handle_list_sessions),
        Route("/admin/api/sessions/{id}", handle_get_turns),
        Route("/admin/api/sessions/{id}", handle_delete_session, methods=["DELETE"]),
        Route("/admin/api/sessions/{id}/turns/{turn_id}", handle_delete_turn, methods=["DELETE"]),
    ]
