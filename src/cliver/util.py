import logging
import os
import platform
import select
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from cliver.constants import APP_NAME, CONFIG_DIR

logger = logging.getLogger(__name__)

# -- Timezone utilities -------------------------------------------------------

_configured_tz: ZoneInfo | None = None


def configure_timezone(tz_name: str | None) -> None:
    """Set the app-wide timezone from config. Call once at startup."""
    global _configured_tz
    if tz_name:
        _configured_tz = ZoneInfo(tz_name)
    else:
        _configured_tz = None


def get_effective_timezone():
    """Return configured timezone, or system local."""
    if _configured_tz:
        return _configured_tz
    return datetime.now().astimezone().tzinfo


def format_datetime(dt: datetime | None = None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime in the effective timezone. Defaults to now."""
    tz = get_effective_timezone()
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(tz).strftime(fmt)


# -- Config directory ---------------------------------------------------------


BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def get_app_user_agent() -> str:
    """Read user_agent from app config, falling back to 'CLIver'."""
    try:
        from cliver.agent_profile import get_current_profile

        profile = get_current_profile()
        if profile:
            from cliver.config import ConfigManager

            cm = ConfigManager(profile.config_dir)
            if cm.config.user_agent:
                return cm.config.user_agent
    except Exception:
        pass
    return "CLIver"


def url_request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    user_agent: str | None = None,
    timeout: int = 30,
):
    """Build a ``urllib.request.Request`` with the app-configured User-Agent.

    Args:
        url: Target URL.
        data: Optional POST body (makes the request a POST).
        headers: Extra headers merged on top of defaults.
        user_agent: Override User-Agent (default: ``config.user_agent``).
        timeout: Not set on the Request itself — pass to ``urlopen(req, timeout=...)``.

    Returns:
        A ready-to-use ``urllib.request.Request``.
    """
    import urllib.request as _ur

    ua = user_agent or get_app_user_agent()
    merged: dict[str, str] = {"User-Agent": ua}
    if headers:
        merged.update(headers)
    return _ur.Request(url, data=data, headers=merged)


def get_config_dir() -> Path:
    """
    Returns the config directory for CLiver.
    It can be overridden by the environment of 'CLIVER_CONF_DIR'
    """
    config_dir = os.getenv(CONFIG_DIR)
    if config_dir is not None:
        return Path(config_dir)
    system = platform.system()
    if system == "Windows":
        return Path(os.getenv("APPDATA")) / APP_NAME
    elif system == "Darwin":
        return Path.home() / f".{APP_NAME}"
    else:
        return Path.home() / f".{APP_NAME}"


def stdin_is_piped():
    try:
        fd = sys.stdin.fileno()
        mode = os.fstat(fd).st_mode
        # True if stdin is a FIFO (pipe)
        return not os.isatty(fd) and stat.S_ISFIFO(mode)
    except Exception:
        return True


def read_piped_input(timeout=5.0):
    """Non-blocking read from stdin if data is available."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read()
    return None


def retry_with_confirmation(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    confirm_on_retry: bool = True,
    confirmation_prompt: str = "Operation failed. Retry?",
    **kwargs,
) -> Any:
    """
    Retry a function until it succeeds or reaches max retries.

    Args:
        func: The function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        confirm_on_retry: Whether to ask for confirmation before
            retrying (default: True)
        confirmation_prompt: Prompt to show when asking for
            confirmation (default: "Operation failed. Retry?")
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call if successful

    Raises:
        The last exception raised by the function if all
        retries are exhausted
    """

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logging.debug(f"Attempt {attempt + 1} failed with error: {str(e)}")

            # If this was the last attempt, don't ask to retry
            if attempt == max_retries:
                break

            # Ask for confirmation before retrying (if enabled)
            if confirm_on_retry:
                if not _confirm_tool_execution(f"{confirmation_prompt} (y/n): "):
                    raise e

            # Wait before retrying
            if retry_delay > 0:
                time.sleep(retry_delay)

    # If we get here, all retries were exhausted
    raise last_exception


async def retry_with_confirmation_async(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    confirm_on_retry: bool = True,
    confirmation_prompt: str = "Operation failed. Retry?",
    **kwargs,
) -> Any:
    """
    Retry an async function until it succeeds or reaches max retries.

    Args:
        func: The async function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        confirm_on_retry: Whether to ask for confirmation before
            retrying (default: True)
        confirmation_prompt: Prompt to show when asking for
            confirmation (default: "Operation failed. Retry?")
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call if successful

    Raises:
        The last exception raised by the function if all
        retries are exhausted
    """
    import asyncio

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logging.debug(f"Attempt {attempt + 1} failed with error: {str(e)}")

            # If this was the last attempt, don't ask to retry
            if attempt == max_retries:
                break

            # Ask for confirmation before retrying (if enabled)
            if confirm_on_retry:
                if not _confirm_tool_execution(f"{confirmation_prompt} (y/n): "):
                    raise e

            # Wait before retrying
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)

    # If we get here, all retries were exhausted
    raise last_exception


def _confirm_tool_execution(prompt="Are you sure? (y/n): ") -> bool:
    """Helper function to get user confirmation."""
    from cliver.agent_profile import get_cli_instance, get_input_fn

    cliver_inst = get_cli_instance()
    if cliver_inst:
        response = cliver_inst.ui.ask_input(prompt, choices=["y", "yes", "n", "no"])
        return response.lower() in ("y", "yes")

    # Fallback via get_input_fn (supports test mocking)
    _input = get_input_fn()
    try:
        response = _input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return response.lower() in ("y", "yes")


# Primary context file; falls back to CLAUDE.md if Cliver.md is absent.
_DEFAULT_CONTEXT_FILES = [
    "Cliver.md",
    "CLAUDE.md",
]

MAX_CONTEXT_FILE_CHARS = 4000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + "\n...(truncated)"
    return text


def _safe_int(value, default: int) -> int:
    """Convert value to int, returning default on failure.

    Handles strings, floats, and ints.  Returns *default* when conversion
    fails or the value is None/falsy.
    """
    if not value:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def read_context_files(
    base_path: str = ".",
    file_filter: list[str] = None,
    max_chars: int = MAX_CONTEXT_FILE_CHARS,
) -> str:
    """
    Read context from context files found in the project tree.

    Default behaviour (no *file_filter*): reads only the **first** file
    found from ``[Cliver.md, CLAUDE.md]`` — Cliver.md is preferred,
    CLAUDE.md serves as fallback.  This keeps the system prompt lean.

    When *file_filter* is provided explicitly, **all** matching files are
    included (backwards-compatible with callers that pass a custom list).

    Each file is truncated to *max_chars* to avoid bloating the system
    prompt.

    Args:
        base_path: The base path to start looking (default: current directory)
        file_filter: List of filenames to look for
        max_chars: Maximum characters per file (default: 4000, ~1000 tokens)
    """
    import os

    use_defaults = file_filter is None
    context_files = _DEFAULT_CONTEXT_FILES if use_defaults else file_filter
    dirs_to_scan = _collect_context_dirs(base_path)

    context = ""
    seen_files: set = set()

    for dir_path in dirs_to_scan:
        for filename in context_files:
            if filename in seen_files:
                continue
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if not content.strip():
                            continue
                        content = _truncate(content, max_chars)
                        context += f"\n# Content from {filename}:\n{content}\n"
                        seen_files.add(filename)
                        if use_defaults:
                            return context.strip()
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")

    return context.strip()


def _collect_context_dirs(base_path: str) -> list[str]:
    """Collect directories to scan for context files.

    Returns [base_path, parent, grandparent, ...] up to the git root
    or filesystem root, whichever comes first. Max 10 levels.
    """
    import os

    dirs = []
    current = os.path.abspath(base_path)

    for _ in range(10):  # safety cap
        dirs.append(current)
        # Stop at git root
        if os.path.isdir(os.path.join(current, ".git")):
            break
        parent = os.path.dirname(current)
        if parent == current:
            break  # filesystem root
        current = parent

    return dirs


def parse_key_value_options(option_list: tuple, console=None) -> dict:
    """
    Parse a list of key=value strings into a dictionary with appropriate type conversion.

    Args:
        option_list: Tuple of strings in key=value format
        console: Optional console object for printing warnings

    Returns:
        Dictionary with parsed key-value pairs
    """
    options_dict = {}

    if not option_list:
        return options_dict

    for opt in option_list:
        if "=" in opt:
            key, value = opt.split("=", 1)
            # Try to convert value to appropriate type
            try:
                # Try integer first
                if "." not in value and value.isdigit():
                    options_dict[key] = int(value)
                # Try float
                elif "." in value and all(c.isdigit() or c == "." for c in value):
                    options_dict[key] = float(value)
                # Keep as string
                else:
                    options_dict[key] = value
            except ValueError:
                options_dict[key] = value
        else:
            # Print warning if console is provided
            if console:
                console.print(f"Warning: Invalid option format '{opt}', expected key=value")

    return options_dict


def read_file_content(file_path: str, max_size: int = 100000) -> str:
    """
    Read the content of a file and return it as a string.

    Args:
        file_path: Path to the file to read
        max_size: Maximum file size to read in bytes (default: 100KB)

    Returns:
        String content of the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is too large or cannot be read as text
    """
    import os

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValueError(f"File is too large ({file_size} bytes). Maximum allowed size is {max_size} bytes.")

    # Try to read as text with common encodings
    encodings = ["utf-8", "utf-16", "latin-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}") from e

    raise ValueError(f"Unable to read file {file_path} with any of the supported encodings: {encodings}")
