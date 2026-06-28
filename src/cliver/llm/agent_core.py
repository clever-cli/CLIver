"""AgentCore — Re-Act loop with CLIverMessage, Provider, and MCPClient.

No langchain dependency. No model fallback. No compression (v1).
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

from cliver.events import (
    EventHandler,
    ToolEvent,
    ToolEventType,
)
from cliver.mcp import MCPClient
from cliver.media import MediaContent
from cliver.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    ToolCallAccumulator,
)
from cliver.provider import CLIverRequest, CLIverResponse, Provider
from cliver.tool import CLIverTool, ToolRegistry

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_ERRORS = 5


class AgentCore:
    """Re-Act loop: send messages to LLM, execute tools, repeat.

    One AgentCore instance per model/provider combination.
    For multiple models, create multiple AgentCore instances
    (they can share the same ``mcp_client`` and ``builtin_tools``).

    **Instance state** (bound at construction, reused across calls):
        provider, model, builtin_tools, mcp_client, on_event, builtin_system_prompt

    **Call state** (passed per chat/stream, varies each turn):
        user_input, system_prompt, conversation, extra_tools, mcp_servers, options

    ``system_prompt`` (call-time) is appended to ``builtin_system_prompt``
    (construction-time) — the caller only needs to pass persona / extra context.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        builtin_tools: list[CLIverTool] | None = None,
        mcp_client: MCPClient | None = None,
        *,
        on_event: EventHandler | None = None,
        max_consecutive_errors: int = _MAX_CONSECUTIVE_ERRORS,
        builtin_system_prompt: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.model = model
        self.tool_registry = ToolRegistry(builtin_tools or [])
        self.mcp_client = mcp_client
        self.on_event = on_event
        self.max_consecutive_errors = max_consecutive_errors
        self._builtin_system_prompt = builtin_system_prompt
        self.logger = logger or logging.getLogger(__name__)

    # ── Public API ────────────────────────────────────────────

    async def chat(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        conversation: list[CLIverMessage] | None = None,
        media: list[MediaContent] | None = None,
        extra_tools: list[CLIverTool] | None = None,
        mcp_servers: list[str] | None = None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
        options: dict[str, Any] | None = None,
        max_iterations: int = 50,
    ) -> CLIverResponse:
        """Non-streaming chat.  Returns the final response.

        ``media`` is attached to the user message as content blocks
        (images as data URLs, audio/video as text placeholders).
        """
        messages = self._build_messages(user_input, system_prompt, conversation, media)
        all_tools = await self._gather_tools(extra_tools, mcp_servers, tool_filter)
        opts = options or {}

        if self.mcp_client:
            await self.mcp_client.start()

        return await self._run_loop(messages, all_tools, opts, max_iterations)

    async def stream(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        conversation: list[CLIverMessage] | None = None,
        media: list[MediaContent] | None = None,
        extra_tools: list[CLIverTool] | None = None,
        mcp_servers: list[str] | None = None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
        options: dict[str, Any] | None = None,
        max_iterations: int = 50,
    ) -> AsyncIterator[CLIverMessageChunk]:
        """Streaming chat.  Yields chunks, executes tools between iterations."""
        messages = self._build_messages(user_input, system_prompt, conversation, media)
        all_tools = await self._gather_tools(extra_tools, mcp_servers, tool_filter)
        opts = options or {}

        if self.mcp_client:
            await self.mcp_client.start()

        async for chunk in self._run_stream_loop(messages, all_tools, opts, max_iterations):
            yield chunk

    async def generate(
        self,
        prompt: str,
        *,
        media_type: str = "image",
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate media — image, audio, or video.

        ``media_type``: ``"image"`` (default), ``"audio"``, or ``"video"``.
        Reference media for editing is passed via ``media``.
        Generated files are saved to ``output_dir`` if given.
        """
        return await self.provider.generate(
            prompt=prompt,
            model=self.model,
            media_type=media_type,
            media=media,
            output_dir=output_dir,
            **options,
        )

    # ── Re-Act Loop (non-streaming) ───────────────────────────

    async def _run_loop(
        self,
        messages: list[CLIverMessage],
        tools: list[CLIverTool],
        options: dict[str, Any],
        max_iterations: int,
    ) -> CLIverResponse:
        consecutive_errors = 0
        diagnostic_trace: list[dict] = []

        for _iteration in range(max_iterations):
            request = CLIverRequest(
                messages=messages,
                tools=tools or None,
                model=self.model,
                options=options,
            )

            response = await self.provider.chat(request)
            msg = response.message

            # Collect diagnostic data for this Re-Act iteration
            diagnostic_trace.append(
                {
                    "iteration": _iteration,
                    "model": self.model,
                    "provider": self.provider.provider_name(),
                    "request": msg.vendor_ext.pop("__llm_request__", None),
                    "response": msg.vendor_ext.pop("__llm_raw_response__", None),
                    "message": msg.model_dump(exclude_none=True),
                }
            )

            if not msg.tool_calls:
                msg.vendor_ext["__llm_trace__"] = diagnostic_trace
                return response

            messages.append(msg)
            consecutive_errors, stop = await self._execute_tool_calls(messages, msg.tool_calls, consecutive_errors)
            if stop:
                error_msg = CLIverMessage(
                    role="assistant",
                    content="Tool calls failed repeatedly. Please check the tool arguments or try a different "
                    "approach.",
                )
                error_msg.vendor_ext["__llm_trace__"] = diagnostic_trace
                return CLIverResponse(message=error_msg, usage=response.usage)

        max_msg = CLIverMessage(
            role="assistant",
            content="Reached maximum iterations without a final answer.",
        )
        max_msg.vendor_ext["__llm_trace__"] = diagnostic_trace
        return CLIverResponse(message=max_msg)

    # ── Re-Act Loop (streaming) ───────────────────────────────

    async def _run_stream_loop(
        self,
        messages: list[CLIverMessage],
        tools: list[CLIverTool],
        options: dict[str, Any],
        max_iterations: int,
    ) -> AsyncIterator[CLIverMessageChunk]:
        import json

        consecutive_errors = 0
        diagnostic_trace: list[dict] = []

        for _iteration in range(max_iterations):
            request = CLIverRequest(
                messages=messages,
                tools=tools or None,
                model=self.model,
                options=options,
            )

            content_parts: list[str] = []
            vendor_buffers: dict[str, str] = {}
            tool_acc = ToolCallAccumulator()

            async for raw in self.provider.stream(request):
                if raw.content:
                    content_parts.append(raw.content)
                for key, delta in raw.vendor_ext.items():
                    vendor_buffers[key] = vendor_buffers.get(key, "") + delta
                for tc_chunk in raw.tool_call_chunks or []:
                    tool_acc.feed(tc_chunk)

                if raw.content or raw.vendor_ext:
                    yield CLIverMessageChunk(
                        content=raw.content,
                        vendor_ext=raw.vendor_ext,
                    )

            tool_calls = tool_acc.finalize()
            vendor = {k: v for k, v in vendor_buffers.items() if v}

            # Collect diagnostic data for this streaming iteration.
            # We don't have __llm_request__ / __llm_raw_response__ from the
            # streaming engine path, but we capture what we can.
            diagnostic_trace.append(
                {
                    "iteration": _iteration,
                    "model": self.model,
                    "provider": self.provider.provider_name(),
                    "request": {
                        "model": request.model,
                        "options": request.options,
                        "system_prompt": next(
                            (m.content for m in request.messages if m.role == "system"),
                            None,
                        ),
                        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                        "tools": [t.model_dump(exclude={"execute"}) for t in (request.tools or [])],
                    },
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts) if content_parts else None,
                        "tool_calls": [tc.model_dump(exclude_none=True) for tc in (tool_calls or [])],
                        "vendor_ext": vendor,
                    },
                }
            )

            if not tool_calls:
                # Final answer — emit trace as a sentinel chunk
                yield CLIverMessageChunk(
                    vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
                )
                return

            messages.append(
                CLIverMessage(
                    role="assistant",
                    content="".join(content_parts) if content_parts else None,
                    tool_calls=tool_calls,
                    vendor_ext=vendor,
                )
            )

            consecutive_errors, stop = await self._execute_tool_calls(messages, tool_calls, consecutive_errors)
            if stop:
                yield CLIverMessageChunk(
                    content="Tool calls failed repeatedly. Please check the tool arguments or try a different "
                    "approach.",
                )
                yield CLIverMessageChunk(
                    vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
                )
                return

        yield CLIverMessageChunk(
            content="Reached maximum iterations without a final answer.",
        )
        yield CLIverMessageChunk(
            vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
        )

    # ── Tool execution (shared by both loops) ─────────────────

    async def _execute_tool_calls(
        self,
        messages: list[CLIverMessage],
        tool_calls: list[ToolCall],
        consecutive_errors: int,
    ) -> tuple[int, bool]:
        """Execute a batch of tool calls and append results to messages.

        Returns (consecutive_errors, should_stop).
        """
        for tc in tool_calls:
            result = await self._execute_tool(tc)
            messages.append(
                CLIverMessage(
                    role="tool",
                    content=self._format_tool_result(result),
                    tool_call_id=tc.id,
                )
            )

            if self._is_error(result):
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    return consecutive_errors, True
            else:
                consecutive_errors = 0

        return consecutive_errors, False

    # ── Helpers ────────────────────────────────────────────────

    def _build_messages(
        self,
        user_input: str,
        system_prompt: str | None,
        conversation: list[CLIverMessage] | None,
        media: list[MediaContent] | None = None,
    ) -> list[CLIverMessage]:
        # Merge builtin system prompt + caller's extra context into one message.
        # Engines (Anthropic's _split_system, OpenAI's msg_to_native) expect a
        # single system message — merging here avoids engine-level complexity.
        parts: list[str] = []
        if self._builtin_system_prompt:
            parts.append(self._builtin_system_prompt)
        if system_prompt:
            parts.append(system_prompt)
        merged_system = "\n\n".join(parts) if parts else None

        messages: list[CLIverMessage] = []
        if merged_system:
            messages.append(CLIverMessage(role="system", content=merged_system))
        if conversation:
            messages.extend(conversation)

        if media:
            # Embed media into the user message as content blocks
            content_parts: list[dict] = [{"type": "text", "text": user_input}]
            from cliver.media import add_media_content_to_message_parts

            add_media_content_to_message_parts(content_parts, media)
            messages.append(CLIverMessage(role="user", content=content_parts))
        else:
            messages.append(CLIverMessage(role="user", content=user_input))
        return messages

    async def _gather_tools(
        self,
        extra: list[CLIverTool] | None,
        mcp_servers: list[str] | None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
    ) -> list[CLIverTool]:
        tools = list(self.tool_registry.all_tools)
        if self.mcp_client:
            tools.extend(await self.mcp_client.get_tools(servers=mcp_servers or None))
        if extra:
            tools.extend(extra)
        if tool_filter:
            tools = [t for t in tools if tool_filter(t)]
        return tools

    async def _execute_tool(self, tc: ToolCall) -> list[dict]:
        """Execute a single tool call. Emits TOOL_START/TOOL_END/TOOL_ERROR events."""
        await self._emit(
            ToolEvent(
                event=ToolEventType.START,
                tool_name=tc.name,
                tool_call_id=tc.id,
                args=tc.args,
            )
        )
        start = time.monotonic()
        try:
            if "#" in tc.name:
                if not self.mcp_client:
                    result = [{"error": "No MCP client configured"}]
                else:
                    result = await self.mcp_client.call_tool(tc.name, tc.args)
            else:
                tool = self.tool_registry.get(tc.name)
                if tool is None:
                    result = [{"error": f"Tool '{tc.name}' not found"}]
                else:
                    result = await asyncio.to_thread(tool.execute, **tc.args)

            duration = (time.monotonic() - start) * 1000
            await self._emit(
                ToolEvent(
                    event=ToolEventType.END,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    result=_truncate(self._format_tool_result(result), 500),
                    duration_ms=duration,
                )
            )
            return result
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            self.logger.warning("Tool '%s' failed: %s", tc.name, e)
            await self._emit(
                ToolEvent(
                    event=ToolEventType.ERROR,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    error=str(e),
                    duration_ms=duration,
                )
            )
            return [{"error": str(e)}]

    async def _emit(self, event: ToolEvent) -> None:
        if self.on_event:
            await self.on_event(event)

    @staticmethod
    def _is_error(result: list[dict]) -> bool:
        """Check if any result dict contains an 'error' key."""
        if not result:
            return False
        return any(isinstance(r, dict) and "error" in r for r in result)

    @staticmethod
    def _format_tool_result(result: list[dict]) -> str:
        """Format tool result for a ToolMessage."""
        if not result:
            return "(no output)"

        if len(result) == 1 and isinstance(result[0], dict):
            r = result[0]
            if "error" in r:
                return f"Error: {r['error']}"
            if "text" in r:
                return str(r["text"])
            if "tool_result" in r:
                return str(r["tool_result"])

        import json

        return json.dumps(result, ensure_ascii=False, default=str)


def _truncate(text: str, max_len: int) -> str:
    """Truncate a string for event display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
