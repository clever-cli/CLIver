"""Shared logging configuration for CLIver.

Gateway uses :func:`configure_gateway_logging` (in ``gateway/logging_config.py``)
to write to ``~/.cliver/gateway.log``.  The TUI / CLI uses
:func:`configure_tui_logging` to write to ``~/.cliver/cliver.log``.

Both processes configure the root logger independently so that every
module-level ``logging.getLogger(__name__)`` automatically routes to the
correct file for its process.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_tui_logging(
    log_path: str | Path | None = None,
    *,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Set up logging for the CLI / TUI process.

    Writes all logs to a rotating file (default: ``~/.cliver/cliver.log``)
    and optionally mirrors to stderr for interactive use.

    Args:
        log_path: Path to the log file.  Defaults to ``{config_dir}/cliver.log``.
        level: Root logger level.  Defaults to INFO; set to DEBUG for dev.
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    if log_path is None:
        from cliver.util import get_config_dir

        log_path = get_config_dir() / "cliver.log"

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers (e.g. basicConfig from early imports)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)

    # Also mirror to stderr so TUI users see log output in the terminal.
    # Log format is shorter (no timestamp) for readability.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root.addHandler(stderr_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
