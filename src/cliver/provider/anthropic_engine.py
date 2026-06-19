"""Anthropic-native messages API protocol engine."""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from uuid import uuid4

from anthropic import NOT_GIVEN, AsyncAnthropic

from cliver.events import EventHandler, InferenceEvent, InferenceEventType
from cliver.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    ToolCallChunk,
    UsageInfo,
)
from cliver.provider import CLIverResponse
from cliver.provider.engine import ProtocolEngine
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)

# Anthropic tool name constraint: ^[a-zA-Z0-9_-]{1,128}$
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class AnthropicEngine(ProtocolEngine):
    """Anthropic-native messages API protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        on_event: EventHandler | None = None,
        user_agent: str | None = None,
    ):
        super().__init__(api_key, base_url, on_event, user_agent=user_agent)
        extra_headers = {"User-Agent": user_agent} if user_agent else {}
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url, default_headers=extra_headers)

    # ── Conversion ──────────────────────────────────────────

    def msg_to_native(self, msg: CLIverMessage) -> dict:
        content: list[dict] = []

        # Preserve thinking blocks from vendor_ext (required by MiniMax & Anthropic)
        if msg.role == "assistant" and "thinking" in msg.vendor_ext:
            thinking_text = msg.vendor_ext["thinking"]
            if thinking_text:
                content.append({"type": "thinking", "thinking": thinking_text})

        if isinstance(msg.content, str):
            if msg.content:
                content.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            content = list(msg.content)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,
                    }
                )

        if msg.role == "tool":
            native: dict[str, Any] = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }
                ],
            }
        else:
            native = {"role": msg.role, "content": content}

        return native

    def tool_to_native(self, tool: CLIverTool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def extract_cliver_message(self, response, tool_name_map: dict[str, str] | None = None) -> CLIverMessage:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        vendor_ext: dict[str, Any] = {}

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                name = block.name
                if tool_name_map and name in tool_name_map:
                    name = tool_name_map[name]
                tool_calls.append(ToolCall(id=block.id, name=name, args=dict(block.input)))
            elif block.type == "thinking":
                vendor_ext.setdefault("thinking", "")
                vendor_ext["thinking"] += block.thinking

        return CLIverMessage(
            role="assistant",
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            vendor_ext=vendor_ext,
        )

    def extract_chunk(self, event) -> CLIverMessageChunk:
        chunk = CLIverMessageChunk(vendor_ext={})

        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                chunk.content = event.delta.text
            elif event.delta.type == "thinking_delta":
                chunk.vendor_ext["thinking"] = event.delta.thinking
            elif event.delta.type == "input_json_delta":
                chunk.tool_call_chunks = [
                    ToolCallChunk(
                        index=event.index,
                        args_delta=event.delta.partial_json,
                    )
                ]

        elif event.type == "content_block_start":
            if event.content_block.type == "tool_use":
                chunk.tool_call_chunks = [
                    ToolCallChunk(
                        index=event.index,
                        id=event.content_block.id,
                        name=event.content_block.name,
                    )
                ]
            elif event.content_block.type == "thinking":
                chunk.vendor_ext["thinking"] = getattr(event.content_block, "thinking", "")

        return chunk

    # ── Tool name sanitization ──────────────────────────────

    def _sanitize_tool_names(self, tools: list[CLIverTool]) -> tuple[dict[str, str], dict[str, str]]:
        """Ensure tool names match Anthropic's ^[a-zA-Z0-9_-]{1,128}$ constraint.

        Returns (forward_map, reverse_map):
            forward_map: original_name → sanitized_name  (for request building)
            reverse_map: sanitized_name → original_name  (for response restoration)
        """
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        seen: set[str] = set()

        for tool in tools:
            name = tool.name
            if _NAME_RE.match(name) and name not in seen:
                forward[name] = name
                reverse[name] = name
                seen.add(name)
                continue

            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:128]
            if not sanitized:
                sanitized = "tool"

            base = sanitized
            counter = 1
            while sanitized in seen:
                suffix = f"_{counter}"
                sanitized = base[: 128 - len(suffix)] + suffix
                counter += 1

            forward[name] = sanitized
            reverse[sanitized] = name
            seen.add(sanitized)

        return forward, reverse

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _split_system(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        system_parts = []
        conversation = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                text = m.get("content", "")
                if isinstance(text, str) and text:
                    system_parts.append(text)
            else:
                conversation.append(m)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, conversation

    @staticmethod
    def _merge_tool_results(conv_messages: list[dict]) -> list[dict]:
        """Merge consecutive tool_result-only user messages into one.

        Anthropic requires all ``tool_result`` blocks for a given assistant
        message to live in a single subsequent user message.  CLIver's
        Re-Act loop emits one message per tool call, so we merge them here.
        """
        merged: list[dict] = []
        for m in conv_messages:
            if not isinstance(m, dict) or m.get("role") != "user":
                merged.append(m)
                continue
            content = m.get("content")
            if not isinstance(content, list) or not content:
                merged.append(m)
                continue
            is_all_tool_results = all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
            if is_all_tool_results and merged and merged[-1].get("role") == "user":
                prev_content = merged[-1].get("content")
                if isinstance(prev_content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in prev_content
                ):
                    prev_content.extend(content)
                    continue
            merged.append(m)
        return merged

    @staticmethod
    def _build_params(options: dict[str, Any]) -> dict[str, Any]:
        """Extract known Anthropic params. Does NOT mutate the input dict."""
        known_keys = {
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "thinking",
        }
        max_tokens = options.get("max_tokens", options.get("max_completion_tokens", 4096))
        params: dict[str, Any] = {"max_tokens": max_tokens}
        for key in ("temperature", "top_p", "top_k", "thinking"):
            if key in options:
                params[key] = options[key]
        # Pass through anything else
        params.update({k: v for k, v in options.items() if k not in known_keys})
        return params

    # ── API Calls ───────────────────────────────────────────

    async def chat(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="anthropic",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        system, conv_messages = self._split_system(messages)
        conv_messages = self._merge_tool_results(conv_messages)

        name_forward, name_reverse = self._sanitize_tool_names(tools) if tools else ({}, {})
        native_tools = NOT_GIVEN
        if tools:
            native_tools = []
            for tool in tools:
                nt = self.tool_to_native(tool)
                nt["name"] = name_forward.get(tool.name, tool.name)
                native_tools.append(nt)

        create_kwargs = self._build_params(options)
        create_kwargs.update(
            model=model,
            messages=conv_messages,
            tools=native_tools,
        )
        if system:
            create_kwargs["system"] = system

        try:
            response = await self.client.messages.create(**create_kwargs)
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="anthropic",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        usage = None
        if response.usage:
            usage = UsageInfo(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", None),
            )

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="anthropic",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

        cliver_msg = self.extract_cliver_message(response, name_reverse)
        # Store raw SDK response for diagnostic display
        try:
            cliver_msg.vendor_ext["__llm_raw_response__"] = response.model_dump()
        except Exception:
            pass

        return CLIverResponse(
            message=cliver_msg,
            usage=usage,
        )

    async def stream(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()
        first_token_emitted = False

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="anthropic",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        system, conv_messages = self._split_system(messages)
        conv_messages = self._merge_tool_results(conv_messages)

        name_forward, name_reverse = self._sanitize_tool_names(tools) if tools else ({}, {})
        native_tools = NOT_GIVEN
        if tools:
            native_tools = []
            for tool in tools:
                nt = self.tool_to_native(tool)
                nt["name"] = name_forward.get(tool.name, tool.name)
                native_tools.append(nt)

        create_kwargs = self._build_params(options)
        create_kwargs.update(
            model=model,
            messages=conv_messages,
            tools=native_tools,
        )
        if system:
            create_kwargs["system"] = system

        try:
            async with self.client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    chunk = self.extract_chunk(event)

                    if not first_token_emitted and (chunk.content or chunk.tool_call_chunks or chunk.vendor_ext):
                        first_token_emitted = True
                        await self._emit(
                            InferenceEvent(
                                event=InferenceEventType.FIRST_TOKEN,
                                model=model,
                                provider="anthropic",
                                request_id=request_id,
                                data={"latency_ms": (time.monotonic() - start) * 1000},
                            )
                        )

                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            if tc.id:
                                await self._emit(
                                    InferenceEvent(
                                        event=InferenceEventType.TOOL_CALL,
                                        model=model,
                                        provider="anthropic",
                                        request_id=request_id,
                                        data={
                                            "tool_name": tc.name or "?",
                                            "tool_call_id": tc.id,
                                        },
                                    )
                                )

                    yield chunk
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="anthropic",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="anthropic",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

    async def _emit(self, event: InferenceEvent) -> None:
        if self.on_event:
            await self.on_event(event)
