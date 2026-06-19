"""MiniMax: strips unsupported params, normalises thinking key,
and parses proprietary XML tool-call format."""

from __future__ import annotations

import json
import re
import uuid
from typing import List

from cliver.messages import CLIverMessageChunk, ToolCall
from cliver.provider import CLIverResponse
from cliver.provider.providers import _EngineProvider

# MiniMax wraps tool calls in proprietary XML inside text content
# instead of using Anthropic-native ``tool_use`` blocks.  Format:
#
#   <minimax:tool_call>
#   <invoke name="tool_name">
#   <parameter name="arg1">value</parameter>
#   <parameter name="arg2">["json", "array"]</parameter>
#   </invoke>
#   </minimax:tool_call>
#
# Multiple <invoke> blocks may appear inside a single <minimax:tool_call>.

# The opening tag may or may not have a leading "<" (both forms seen).
_TOOL_CALL_RE = re.compile(
    r"<?minimax:tool_call[>\s](.*?)</minimax:tool_call>",
    re.DOTALL,
)

_INVOKE_RE = re.compile(
    r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>',
    re.DOTALL,
)

_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>',
    re.DOTALL,
)

# MiniMax built-in tools that cannot be executed locally.
# The XML is stripped from the response and no ToolCall is created,
# so the Re-Act loop won't attempt to execute them.
_MINIMAX_BUILTIN_TOOLS = frozenset({"generate_images"})


def _parse_param_value(raw: str) -> str | int | float | bool | list | dict:
    """Convert a parameter value string to its natural Python type."""
    val = raw.strip()
    # Strip leading/trailing newlines
    if val.startswith("\n"):
        val = val[1:]
    if val.endswith("\n"):
        val = val[:-1]
    val = val.strip()
    # Try JSON (arrays, objects, numbers, booleans, null)
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return val


def _parse_minimax_tool_calls(text: str) -> tuple[str, List[ToolCall]]:
    """Extract MiniMax XML tool calls from *text*.

    Returns the cleaned text (XML blocks removed) and a list of
    :class:`ToolCall` objects.
    """
    tool_calls: List[ToolCall] = []

    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        # A single <minimax:tool_call> may contain multiple <invoke> blocks.
        for invoke in _INVOKE_RE.finditer(inner):
            raw_name = invoke.group(1)
            if raw_name in _MINIMAX_BUILTIN_TOOLS:
                # MiniMax built-in tool (e.g. generate_images) — not a CLIver
                # tool.  Skip it so the Re-Act loop doesn't try to execute it.
                continue
            args = {}
            for pm in _PARAM_RE.finditer(invoke.group(2)):
                args[pm.group(1)] = _parse_param_value(pm.group(2))
            tool_calls.append(
                ToolCall(
                    id=f"minimax_{uuid.uuid4().hex[:8]}",
                    name=raw_name,
                    args=args,
                )
            )
        return ""

    cleaned = _TOOL_CALL_RE.sub(_replace, text).strip()
    return cleaned, tool_calls


class MiniMaxProvider(_EngineProvider):
    """MiniMax provider.

    Filters out params MiniMax doesn't support, remaps the
    Anthropic-native ``thinking`` key to the canonical ``reasoning_content``
    key in streaming chunks, and parses MiniMax's proprietary XML tool-call
    format from text responses.
    """

    supported_protocols = ["openai", "anthropic"]
    default_base_url = "https://api.minimax.chat/v1"

    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "logit_bias",
        "parallel_tool_calls",
    }

    def filter_options(self, options: dict) -> dict:
        return {k: v for k, v in options.items() if k not in self.UNSUPPORTED_PARAMS}

    def on_response(self, response: CLIverResponse) -> CLIverResponse:
        """Parse MiniMax XML tool calls from text content."""
        msg = response.message
        if msg.content and isinstance(msg.content, str) and "minimax:tool_call" in msg.content:
            cleaned, tool_calls = _parse_minimax_tool_calls(msg.content)
            msg.content = cleaned or None
            if tool_calls:
                if msg.tool_calls:
                    msg.tool_calls.extend(tool_calls)
                else:
                    msg.tool_calls = tool_calls
        return response

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        if "thinking" in chunk.vendor_ext:
            chunk.vendor_ext["reasoning_content"] = chunk.vendor_ext.pop("thinking")
        return chunk
