"""Built-in tool for executing Python code in a sandboxed subprocess.

Enables the LLM to write and run Python scripts that perform multi-step
operations in a single tool call, dramatically reducing token usage.
The code runs in an isolated subprocess with stdout/stderr capture.
"""

import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional

from cliver.tool import tool
from cliver.util import _safe_int

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 300  # 5 minutes
MAX_OUTPUT_LENGTH = 50000  # characters


def _build_env() -> dict:
    """Build environment for the subprocess.

    Inherits the parent environment so installed packages (yaml, etc.) are available.
    """
    env = os.environ.copy()
    # Ensure UTF-8 output
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@tool(
    name="Exec",
    description=(
        "Execute a Python script in an isolated subprocess and return its output. "
        "Use this for multi-step data processing, file manipulation, calculations, "
        "or any task that would otherwise require many sequential tool calls. "
        "The script runs in the current working directory with full filesystem access. "
        "Print results to stdout -- that's what you'll see. "
        "Common libraries are available: json, csv, yaml, os, pathlib, re, math, etc."
    ),
)
def execute_code(code: str, timeout: Optional[int] = None) -> list[dict]:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: Python code to execute. Has access to the filesystem (cwd) and standard
            library. Print results to stdout -- the output is returned to you.
        timeout: Timeout in seconds (max 300). Defaults to 30s.
    """
    timeout_s = min(_safe_int(timeout, DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S)

    # Write code to a temp file (avoids shell escaping issues)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            script_path = f.name
    except Exception as e:
        return [{"error": f"Error creating script file: {e}"}]

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=os.getcwd(),
            env=_build_env(),
        )

        output_parts = []

        if result.stdout:
            stdout = result.stdout
            if len(stdout) > MAX_OUTPUT_LENGTH:
                stdout = stdout[:MAX_OUTPUT_LENGTH] + f"\n\n... (truncated, {len(result.stdout)} total chars)"
            output_parts.append(stdout)

        if result.stderr:
            stderr = result.stderr
            if len(stderr) > MAX_OUTPUT_LENGTH:
                stderr = stderr[:MAX_OUTPUT_LENGTH] + f"\n\n... (truncated, {len(result.stderr)} total chars)"
            output_parts.append(f"STDERR:\n{stderr}")

        if result.returncode != 0:
            output_parts.append(f"\nExit code: {result.returncode}")

        if not output_parts:
            return [{"text": "(no output)"}]

        return [{"text": "\n".join(output_parts)}]

    except subprocess.TimeoutExpired:
        return [{"error": f"Script timed out after {timeout_s} seconds."}]
    except Exception as e:
        logger.warning(f"Code execution error: {e}")
        return [{"error": f"Error executing code: {e}"}]
    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except OSError:
            pass
