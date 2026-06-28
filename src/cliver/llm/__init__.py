import logging
import os

from cliver.llm.agent_core import AgentCore
from cliver.logging_config import configure_tui_logging

_level = logging.DEBUG if os.environ.get("MODE") == "dev" else logging.INFO
configure_tui_logging(level=_level)

__all__ = ["AgentCore"]
