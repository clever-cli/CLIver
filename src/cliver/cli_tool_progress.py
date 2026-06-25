"""
CLI tool execution progress display using Rich.

This module provides a Rich-based renderer for tool execution events.
It is ONLY used by the CLI layer — AgentCore and the API layer
have no dependency on this module.
"""

import logging
import threading

from rich.console import Console

from cliver.events import EventHandler, ToolEvent, ToolEventType

logger = logging.getLogger(__name__)

# ─── Last Output Buffer (for Ctrl+O expand) ──────────────────────────────────

_last_full_output: dict = {"text": "", "tool": ""}


def get_last_full_output() -> tuple[str, str]:
    """Return (text, tool_name) of the last truncated tool output."""
    return _last_full_output["text"], _last_full_output["tool"]


# ─── Status Icons ─────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    ToolEventType.START: "⟳",
    ToolEventType.END: "✓",
    ToolEventType.ERROR: "✗",
}

# ─── Tool Activity Descriptions ──────────────────────────────────────────────
# Template-based human-readable descriptions for tool calls.
# Each entry maps a tool name to a function that takes the args dict and
# returns a short description string.  Tools not listed here fall back to
# showing the raw tool name.

_TOOL_DESCRIPTIONS = {
    "Read": lambda a: f"Reading {_short_path(a.get('file_path', ''))}",
    "Write": lambda a: f"Writing {_short_path(a.get('file_path', ''))}",
    "LS": lambda a: f"Listing {_short_path(a.get('path', '.'))}",
    "Grep": lambda a: f"Searching for '{_trunc(a.get('pattern', ''), 40)}'"
    + (f" in {_short_path(a.get('path', ''))}" if a.get("path", ".") != "." else ""),
    "Bash": lambda a: f"Running {_trunc(a.get('description') or a.get('command', ''), 50)}",
    "Exec": lambda a: "Executing Python code",
    "WebFetch": lambda a: f"Fetching {_trunc(a.get('url', ''), 50)}",
    "WebSearch": lambda a: f"Searching web for '{_trunc(a.get('query', ''), 40)}'",
    "Browse": lambda a: f"Browsing {_trunc(a.get('url', ''), 50)}",
    "Browser": lambda a: f"Browser {a.get('action', 'action')}",
    "TodoRead": lambda _: "Reading plan",
    "TodoWrite": lambda _: "Updating plan",
    "MemoryRead": lambda _: "Reading memory",
    "MemoryWrite": lambda _: "Saving to memory",
    "Identity": lambda _: "Updating identity",
    "Skill": lambda a: f"Activating skill '{a.get('skill_name', '')}'",
    "Docker": lambda a: f"Running container {_trunc(a.get('image', ''), 40)}",
    "Transcribe": lambda a: f"Transcribing {_short_path(a.get('file_path', ''))}",
    "SearchSessions": lambda a: f"Searching sessions for '{_trunc(a.get('query', ''), 40)}'",
    "ImageGenerate": lambda a: f"Generating image with {a.get('model', 'image model')}",
}


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_path(p: str) -> str:
    """Show just filename or last 2 path components."""
    if not p:
        return "."
    parts = p.replace("\\", "/").rstrip("/").split("/")
    if len(parts) <= 2:
        return p
    return "…/" + "/".join(parts[-2:])


def _describe_tool(tool_name: str, args: dict | None) -> str:
    """Generate a human-readable description for a tool call."""
    fn = _TOOL_DESCRIPTIONS.get(tool_name)
    if fn and args is not None:
        try:
            return fn(args)
        except Exception:
            pass
    # Fallback: just the tool name
    return tool_name


# ─── Plan Status Icons ────────────────────────────────────────────────────────

_PLAN_ICONS = {
    "pending": "○",
    "in_progress": "◉",
    "completed": "●",
    "failed": "✗",
}


# ─── Thinking Spinner ────────────────────────────────────────────────────────


class ThinkingIndicator:
    """Shows a live animated thinking indicator while the LLM is processing.

    Started when inference begins, stopped on first tool call or content chunk.
    Runs a background thread that prints color-cycling dots.
    """

    _COLORS = ["#ff6b6b", "#ffa06b", "#ffd86b", "#6bffa0", "#6bd8ff", "#a06bff"]
    _PHRASES = [
        "Thinking…",
        "Reasoning…",
        "Analyzing…",
        "Considering…",
        "Processing…",
        "Reflecting…",
        "Pondering…",
        "Evaluating…",
        "Working…",
        "Figuring out…",
    ]

    def __init__(self, console: Console):
        self._console = console
        self._active = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._model = ""

    def start(self, model: str = "") -> None:
        """Start the animated thinking indicator."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._model = model
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def _animate(self) -> None:
        """Background animation loop."""
        import random
        import sys
        import time

        label = f"{self._model} " if self._model else ""
        frame = 0
        phrase = random.choice(self._PHRASES)
        while self._active:
            color = self._COLORS[frame % len(self._COLORS)]
            dots = "●" * ((frame % 3) + 1)
            # Switch phrase every ~3 seconds (10 frames × 0.3s)
            if frame > 0 and frame % 10 == 0:
                phrase = random.choice(self._PHRASES)
            sys.stdout.write(f"\r  \033[38;2;{_hex_to_rgb(color)}m{dots}\033[0m \033[2m{label}{phrase}\033[0m  ")
            sys.stdout.flush()
            frame += 1
            time.sleep(0.3)
        # Clear the line when done
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    def stop(self, blank_line: bool = True) -> None:
        """Stop the animated indicator.

        Args:
            blank_line: Print an extra blank line after stopping (before LLM content).
                        Set to False when the spinner will be restarted immediately.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        if blank_line:
            print()

    @property
    def active(self) -> bool:
        return self._active


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#rrggbb' to 'r;g;b' for ANSI escape."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"


# ─── Tool Progress Handler ────────────────────────────────────────────────────


def create_tool_progress_handler(
    console: Console,
    thinking: ThinkingIndicator | None = None,
) -> EventHandler:
    """Create a tool event handler that displays progress using Rich.

    Args:
        console: Rich Console instance for output
        thinking: Optional ThinkingIndicator to stop when tools start

    Returns:
        A EventHandler callback for use with AgentCore
    """
    # Track whether we're inside a tool execution block for spacing
    state = {"in_block": False}

    async def handler(event: ToolEvent) -> None:
        icon = _STATUS_ICONS.get(event.event, "")
        tool = event.tool_name

        if event.event == ToolEventType.START:
            # Stop thinking spinner when tool execution begins (no blank line — spinner restarts after)
            if thinking:
                thinking.stop(blank_line=False)

            if not state["in_block"]:
                console.print()  # blank line before first tool in a batch
                state["in_block"] = True

            # Human-readable activity description
            desc = _describe_tool(event.tool_name, event.args)
            console.print(f"  {icon} {desc}")

        elif event.event == ToolEventType.END:
            duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")

            # Show tool result preview
            if event.result and event.result not in ("denied", "(no output)"):
                _render_tool_result(console, event.result, event.tool_name)

            state["in_block"] = False

            # Render plan progress when TodoWrite completes
            if event.tool_name == "TodoWrite":
                _render_plan_progress(console)

            console.print()  # blank line after completion

            # Restart spinner while LLM processes the tool result
            if thinking:
                thinking.start(thinking._model)

        elif event.event == ToolEventType.ERROR:
            duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else ""
            console.print(f"  {icon} {tool} {duration}")
            if event.error:
                # Truncate very long errors
                err = event.error if len(event.error) <= 200 else event.error[:197] + "…"
                console.print(f"      {err}")
            state["in_block"] = False
            console.print()  # blank line after error

            # Restart spinner while LLM processes the error
            if thinking:
                thinking.start(thinking._model)

    return handler


# ─── Tool Result Display ─────────────────────────────────────────────────────

_MAX_RESULT_LINES = 15
_MAX_LINE_WIDTH = 120


def _render_tool_result(console: Console, result: str, tool_name: str = "") -> None:
    """Render a compact preview of the tool result output."""
    lines = result.splitlines()
    total = len(lines)

    # Truncate lines that are too wide
    display_lines = []
    for line in lines[:_MAX_RESULT_LINES]:
        if len(line) > _MAX_LINE_WIDTH:
            line = line[:_MAX_LINE_WIDTH] + "…"
        display_lines.append(line)

    was_truncated = total > _MAX_RESULT_LINES or any(len(ln) > _MAX_LINE_WIDTH for ln in lines)

    body = "\n".join(display_lines)
    if total > _MAX_RESULT_LINES:
        body += f"\n… ({total - _MAX_RESULT_LINES} more lines, Ctrl+O to expand)"

    console.print(f"      {body}")

    # Store full output for Ctrl+O expansion
    if was_truncated:
        _last_full_output["text"] = result
        _last_full_output["tool"] = tool_name


# ─── Plan Progress Display ────────────────────────────────────────────────────


def _render_plan_progress(console: Console) -> None:
    """Render the current todo plan with clear status markers."""
    from cliver.tools.todo_write import get_current_todos

    todos = get_current_todos()
    if not todos:
        return

    lines = []
    pending = in_progress = completed = failed = 0
    for item in todos:
        status = item.get("status", "pending")
        icon = _PLAN_ICONS.get(status, _PLAN_ICONS["pending"])
        content = item.get("content", "")

        styled = content

        lines.append(f"  {icon} {styled}")
        if status == "pending":
            pending += 1
        elif status == "in_progress":
            in_progress += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1

    total = len(todos)
    # Progress bar
    done_ratio = completed / total if total > 0 else 0
    bar_width = 20
    filled = int(done_ratio * bar_width)
    bar = f"{'━' * filled}{'─' * (bar_width - filled)}"

    summary_parts = [f"{completed}/{total} done"]
    if in_progress:
        summary_parts.append(f"{in_progress} active")
    if pending:
        summary_parts.append(f"{pending} pending")
    if failed:
        summary_parts.append(f"{failed} failed")
    summary = ", ".join(summary_parts)

    console.print()
    console.print(f"  Plan  {bar}  {summary}")
    for line in lines:
        console.print(line)
