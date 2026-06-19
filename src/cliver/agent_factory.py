"""AgentCore factory — creates AgentCore from configuration.

All context is derived from ``config_manager``.  The simplest usage::

    agent = create_agent_core(model_config=mc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cliver.config import ConfigManager, ModelConfig
    from cliver.events import EventHandler
    from cliver.llm.agent_core import AgentCore
    from cliver.mcp import MCPClient
    from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)

# Module-level caches
_builtin_tools_cache: list["CLIverTool"] | None = None
_mcp_client_cache: "MCPClient | None" = None


def _get_builtin_tools(config_manager: "ConfigManager") -> list["CLIverTool"]:
    """Return builtin tools, caching the discovery result."""
    global _builtin_tools_cache
    if _builtin_tools_cache is not None:
        return _builtin_tools_cache
    from cliver.tool import ToolRegistry, discover_builtin_tools

    all_tools = discover_builtin_tools()
    reg = ToolRegistry(all_tools)
    reg.configure(config_manager.config.enabled_toolsets)
    _builtin_tools_cache = reg.all_tools
    return _builtin_tools_cache


def _get_mcp_client(config_manager: "ConfigManager") -> "MCPClient":
    """Return MCP client, caching the instance."""
    global _mcp_client_cache
    if _mcp_client_cache is not None:
        return _mcp_client_cache
    from cliver.mcp import MCPClient

    _mcp_client_cache = MCPClient(config_manager.list_mcp_servers_for_mcp_caller())
    return _mcp_client_cache


def create_agent_core(
    model_config: "ModelConfig",
    *,
    config_manager: "ConfigManager | None" = None,
    on_event: "EventHandler | None" = None,
    max_consecutive_errors: int = 5,
) -> "AgentCore":
    """Create an AgentCore from a model config.

    Everything is derived from ``config_manager`` (or auto-loaded from
    ``~/.cliver`` if not provided).

    The builtin system prompt is always included.  Callers add per-call
    persona / extra context via ``system_prompt`` on ``chat()`` / ``stream()``.

    Args:
        model_config: The ModelConfig for the desired model.
        config_manager: ConfigManager (auto-loaded from default config dir if None).
        tool_filter: Optional predicate to filter tools (receives CLIverTool,
            returns True to include).
        on_event: Optional handler for ToolEvent/InferenceEvent callbacks.
        max_consecutive_errors: Max consecutive tool errors before stopping.

    Returns:
        A configured AgentCore ready for ``chat()`` / ``stream()``.
    """
    from cliver.agent_profile import CliverProfile
    from cliver.llm.agent_core import AgentCore
    from cliver.provider.providers import create_provider
    from cliver.system_prompt import build as build_system_prompt

    if config_manager is None:
        from cliver.config import ConfigManager
        from cliver.util import get_config_dir

        config_manager = ConfigManager(get_config_dir())

    profile = CliverProfile(config_manager.config_dir)

    models = config_manager.list_llm_models()
    agents = getattr(config_manager.config, "agents", None)
    user_agent = config_manager.config.user_agent

    tools = _get_builtin_tools(config_manager)

    pc = config_manager.get_provider_config(model_config.provider)
    protocol = model_config.get_provider_protocol(pc)
    provider = create_provider(
        api_key=pc.get_api_key() if pc else None,
        base_url=model_config.get_resolved_url(pc) or None,
        protocol=protocol,
        provider_class=model_config.provider,
        user_agent=user_agent,
    )

    builtin_sp = build_system_prompt(
        agent_name=profile.profile_name,
        available_tools={t.name for t in tools},
        models=models,
        agents=agents,
        current_model=model_config.api_model_name,
        current_provider=protocol,
    )

    return AgentCore(
        provider=provider,
        model=model_config.api_model_name,
        builtin_tools=tools,
        mcp_client=_get_mcp_client(config_manager),
        on_event=on_event,
        max_consecutive_errors=max_consecutive_errors,
        builtin_system_prompt=builtin_sp,
    )


def resolve_model(
    model_name: str | None,
    config_manager: "ConfigManager",
) -> "ModelConfig | None":
    """Resolve a model name to a ModelConfig, using the default if None.

    Returns None if no model is configured at all.
    """
    if model_name:
        mc = _resolve_name(model_name, config_manager)
        if mc:
            return mc

    default = config_manager.get_llm_model()
    if default:
        return default

    # Fallback: first available text model
    text_models = config_manager.config.models.get("text", {})
    if text_models:
        return next(iter(text_models.values()))

    return None


def _resolve_name(name: str, config_manager: "ConfigManager") -> "ModelConfig | None":
    all_models = config_manager.all_models()
    if name in all_models:
        return all_models[name]
    matches = [mc for key, mc in all_models.items() if key.endswith(f"/{name}") or key == name]
    if len(matches) == 1:
        return matches[0]
    return None
