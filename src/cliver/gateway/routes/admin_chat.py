"""Chat SSE route for the admin portal."""

from __future__ import annotations

import json
import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _strip_tool_calls(text: str) -> str:
    """Strip inline JSON tool call blocks from text output."""
    import re

    return re.sub(r'\s*\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}', "", text, flags=re.DOTALL).strip()


def get_chat_routes(context: dict, require_auth: Callable) -> list:
    """Return chat API route (SSE streaming)."""

    @require_auth
    async def handle_chat(request: Request):
        gateway = context.get("gateway")
        if not gateway:
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        session_manager = context.get("cli_session_manager")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        prompt = (body.get("message") or body.get("prompt", "")).strip()
        if not prompt:
            return JSONResponse({"error": "'message' (or 'prompt') is required"}, status_code=400)

        model = body.get("model") or gateway._get_default_model_name()
        system_message = body.get("system_message")
        agent_name = body.get("agent", "").strip()
        raw_history = body.get("conversation_history") or []
        tool_names = body.get("filter_tools")
        conversation_id = body.get("session_id") or body.get("conversation_id")

        # Resolve agent config if specified
        agent_role = None
        if agent_name:
            config = getattr(gateway, "_get_config_manager", None)
            if config:
                try:
                    cfg = config().config
                    agents = getattr(cfg, "agents", {}) or {}
                    agent_cfg = agents.get(agent_name)
                    if agent_cfg:
                        agent_model = getattr(agent_cfg, "model", None)
                        if agent_model:
                            model = agent_model
                        agent_role = getattr(agent_cfg, "role", None)
                except Exception:
                    pass

        # -- Conversation persistence --
        session_id = None
        if conversation_id and session_manager:
            existing = session_manager.get_session_info(conversation_id)
            if not existing:
                return JSONResponse({"error": "Conversation not found"}, status_code=404)
            session_id = conversation_id
            # Load persisted turns as history if client didn't pass it explicitly
            if not raw_history:
                loaded = session_manager.load_turns(session_id)
                raw_history = [{"role": t["role"], "content": t["content"]} for t in loaded]
        elif session_manager:
            # Capture options at session creation time so the full configuration
            # is immediately persisted and available for reload/fork.
            session_options = {}
            if agent_name:
                session_options["agent"] = agent_name
            if model:
                session_options["model"] = model
            if system_message:
                session_options["system_message"] = system_message
            if tool_names:
                session_options["filter_tools"] = tool_names
            session_id = session_manager.create_session(
                kind="chat",
                options=session_options if session_options else None,
            )

        if session_manager and session_id:
            try:
                from cliver.messages import CLIverMessage

                session_manager.append_turn(
                    session_id,
                    "user",
                    prompt,
                    message=CLIverMessage(role="user", content=prompt),
                )
            except Exception as e:
                logger.warning("Failed to persist user turn: %s", e)

        conversation_history = None
        if raw_history:
            from cliver.messages import CLIverMessage

            conversation_history = []
            for msg in raw_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                vendor = {}
                if "reasoning_content" in msg:
                    vendor["reasoning_content"] = msg["reasoning_content"]
                if role in ("user", "assistant"):
                    conversation_history.append(CLIverMessage(role=role, content=content, vendor_ext=vendor))

        def _system_appender():
            parts = []
            if agent_role:
                parts.append(agent_role)
            if system_message:
                parts.append(system_message)
            parts.append(
                "\n## Server Mode\n\n"
                "You are running as a backend API service via the admin portal. "
                "Make autonomous decisions. Be concise and direct. "
                "After a tool call completes, summarize the result briefly — "
                "do NOT repeat the tool input or arguments you already provided."
            )
            return "\n".join(parts)

        _tool_filter = None
        if tool_names:
            allowed = set(tool_names)

            async def _tool_filter(user_input, tools):
                return [t for t in tools if t.name in allowed]

        uses_thinking = True

        async def generate():
            if session_id:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n".encode()

            logger.info("Chat start — session=%s model=%s", session_id or "new", model)

            full_text = ""
            vendor_ext: dict = {}
            media_files = []
            try:
                agent = gateway._get_agent(model)
                async for chunk in agent.stream(
                    prompt=prompt,
                    system_prompt=gateway._build_agent_system_prompt(agent, extra=_system_appender()),
                    conversation=conversation_history,
                ):
                    if chunk.content:
                        full_text += chunk.content
                        data = json.dumps({"type": "content", "content": chunk.content})
                        yield f"data: {data}\n\n".encode()
                    if chunk.vendor_ext:
                        for k, v in chunk.vendor_ext.items():
                            if k == "__llm_trace__":
                                # Structured diagnostic data — parse, don't accumulate
                                vendor_ext[k] = json.loads(v) if isinstance(v, str) else v
                            else:
                                vendor_ext.setdefault(k, "")
                                vendor_ext[k] += v

                clean_text = _strip_tool_calls(full_text)

                # Persist assistant turn
                if session_manager and session_id:
                    try:
                        from cliver.messages import CLIverMessage

                        session_manager.append_turn(
                            session_id,
                            "assistant",
                            clean_text,
                            message=CLIverMessage(role="assistant", content=clean_text, vendor_ext=vendor_ext),
                        )
                    except Exception as e:
                        logger.warning("Failed to persist assistant turn: %s", e)

                done_data = {
                    "type": "done",
                    "content": clean_text,
                    "media": media_files,
                    "media_files": media_files,
                    "session_id": session_id,
                }
                if vendor_ext.get("__llm_trace__"):
                    done_data["__llm_trace__"] = vendor_ext["__llm_trace__"]
                if uses_thinking:
                    done_data["reasoning_content"] = ""
                yield f"data: {json.dumps(done_data)}\n\n".encode()

            except Exception as e:
                logger.error("Chat streaming error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return [Route("/admin/api/chat", handle_chat, methods=["POST"])]
