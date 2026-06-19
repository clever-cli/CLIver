"""
Configuration module for Cliver client.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from cliver.template_utils import render_template_if_needed

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limit configuration for an LLM provider."""

    requests: int = Field(description="Number of requests allowed in the period")
    period: str = Field(description="Time period, e.g. '5h', '30m', '1d', or seconds as string")
    margin: float = Field(default=0.1, description="Safety margin (0.1 = 10% slower than limit)")


class PricingConfig(BaseModel):
    """Pricing configuration per million tokens."""

    currency: Optional[str] = Field(default=None, description="Currency code: USD, CNY, EUR, etc.")
    input: Optional[float] = Field(default=None, description="Cost per million input tokens")
    output: Optional[float] = Field(default=None, description="Cost per million output tokens")
    cached_input: Optional[float] = Field(default=None, description="Cost per million cached input tokens")


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider (API endpoint + credentials + rate limit)."""

    name: str
    type: str = Field(default="openai", description="API protocol: openai (OpenAI-compatible) or anthropic")
    api_url: str = Field(description="Base URL for the provider API")
    api_key: Optional[str] = Field(default=None, description="API key (supports Jinja2 templates)")
    rate_limit: Optional[RateLimitConfig] = Field(default=None, description="Rate limit for API calls")
    pricing: Optional[PricingConfig] = Field(default=None, description="Token pricing for cost tracking")
    model_config = {"extra": "allow"}

    def get_api_key(self) -> Optional[str]:
        if self.api_key is None:
            return None
        return render_template_if_needed(self.api_key)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        return {k: v for k, v in data.items() if v is not None}


class ModelOptions(BaseModel):
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling cutoff")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Models are organized by category (text/image/audio/video).  ``name``
    is a short config key (e.g. ``minimax-2.7``).  ``model`` is the
    API-facing model name (e.g. ``MiniMax-M2.7``).
    """

    name: str
    provider: str
    model: str = Field(description="API-facing model name sent to the provider")
    category: str = Field(default="text", description="Category: text, image, audio, or video")
    api_url: Optional[str] = Field(default=None, description="Per-model endpoint override")
    options: Optional[ModelOptions] = Field(default=None, description="Options for model")

    model_config = {"extra": "allow"}

    @property
    def api_model_name(self) -> str:
        """The model name as sent to the provider API."""
        return self.model

    def get_provider_protocol(self, pc: "ProviderConfig | None" = None) -> str:
        """Return the protocol type (``"openai"`` or ``"anthropic"``).

        Pass *pc* from :meth:`ConfigManager.get_provider_config` to resolve
        directly from the user's ProviderConfig.
        """
        return pc.type if pc else "openai"

    def get_resolved_url(self, pc: "ProviderConfig | None" = None) -> Optional[str]:
        """Return the API base URL.  *pc* is the linked ProviderConfig."""
        if self.api_url:
            return self.api_url
        return pc.api_url if pc else None

    def get_api_key(self, pc: "ProviderConfig | None" = None) -> Optional[str]:
        """Return the API key.  *pc* is the linked ProviderConfig."""
        return pc.get_api_key() if pc else None

    def get_resolved_pricing(self, pc: "ProviderConfig | None" = None) -> Optional[tuple]:
        """Get provider-level pricing.

        Returns (input, output, cached_input, currency) per million tokens.
        """
        if pc and pc.pricing and pc.pricing.input is not None and pc.pricing.output is not None:
            cached = pc.pricing.cached_input or pc.pricing.input
            return (pc.pricing.input, pc.pricing.output, cached, pc.pricing.currency or "USD")
        return None

    @property
    def is_text(self) -> bool:
        return self.category == "text"

    @property
    def can_strict_tool_call(self) -> bool:
        return self.get_provider_protocol() == "openai"

    def model_dump(self, **kwargs):
        """Override to exclude internal fields and null values."""
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        data.pop("provider", None)
        return {k: v for k, v in data.items() if v is not None}


class MCPServerConfig(BaseModel):
    """Base class for MCP server configurations."""

    name: str
    transport: str

    def model_dump(self, **kwargs):
        """Override to exclude name field and null values."""
        data = super().model_dump(**kwargs)
        # Remove name field since it's redundant (key in mcpServers dict)
        data.pop("name", None)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}


class StdioMCPServerConfig(MCPServerConfig):
    """Configuration for stdio MCP servers."""

    transport: str = "stdio"
    command: str
    args: Optional[List[str]] = Field(default=None, description="Arguments to start the stdio mcp server")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables for the stdio mcp server")


class SSEMCPServerConfig(MCPServerConfig):
    """Configuration for SSE MCP servers (deprecated)."""

    transport: str = "sse"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="The HTTP headers to interact with the SSE MCP server"
    )


class StreamableHttpMCPServerConfig(MCPServerConfig):
    """Configuration for Streamable HTTP MCP servers."""

    transport: str = "streamable_http"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="The HTTP headers to interact with the streamable_http MCP server",
    )


class WebSocketMCPServerConfig(MCPServerConfig):
    """Configuration for WebSocket MCP servers."""

    transport: str = "websocket"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="The HTTP headers to interact with the websocket MCP server",
    )


class PlatformConfig(BaseModel):
    """Configuration for a single messaging platform adapter."""

    name: str = Field(default="", description="Platform name (set from config key)")
    type: str = Field(description="Adapter type: builtin name or module path")
    token: Optional[str] = Field(default=None, description="Bot/API token (supports Jinja2 templates)")
    app_token: Optional[str] = Field(default=None, description="App-level token (e.g., Slack Socket Mode)")
    home_channel: Optional[str] = Field(default=None, description="Default channel for cron delivery")
    allowed_users: Optional[List[str]] = Field(default=None, description="Whitelist of user IDs")

    model_config = {"extra": "allow"}

    @property
    def is_builtin(self) -> bool:
        return "." not in self.type

    def get_token(self) -> Optional[str]:
        if self.token is None:
            return None
        return render_template_if_needed(self.token)

    def get_app_token(self) -> Optional[str]:
        if self.app_token is None:
            return None
        return render_template_if_needed(self.app_token)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        return {k: v for k, v in data.items() if v is not None}


class GatewayConfig(BaseModel):
    """Configuration for the gateway daemon process."""

    host: str = Field(default="127.0.0.1", description="Host to bind the gateway HTTP server")
    port: int = Field(default=8321, description="Port for the gateway HTTP server")
    api_key: Optional[str] = Field(default=None, description="API key for authentication (optional)")
    platforms: Dict[str, PlatformConfig] = Field(default_factory=dict, description="Platform adapter configurations")
    log_file: Optional[str] = Field(
        default=None,
        description="Path to the gateway rotating log file. Default: {config_dir}/gateway.log",
    )
    log_max_bytes: int = Field(default=10 * 1024 * 1024, description="Max size per log file in bytes (default 10MB)")
    log_backup_count: int = Field(default=5, description="Number of rotated log files to keep (default 5)")
    admin_username: Optional[str] = Field(default=None, description="Admin portal username")
    admin_password: Optional[str] = Field(default=None, description="Admin portal password (supports Jinja2 templates)")


class SessionConfig(BaseModel):
    """Session storage limits — shared by CLI and gateway."""

    max_sessions: int = Field(
        default=300,
        description="Max sessions to keep; oldest deleted when exceeded",
    )
    max_turns_per_session: int = Field(default=100, description="Max turns per session")
    max_age_days: int = Field(default=365, description="Delete sessions idle for this many days")


class AgentConfig(BaseModel):
    """Configuration profile for a named agent around AgentCore."""

    name: str = Field(description="Agent name (derived from config.agents key, not serialized)")
    description: Optional[str] = Field(default=None, description="Human-readable purpose")
    role: Optional[str] = Field(default=None, description="Role name (e.g. 'code-reviewer', 'research-assistant')")
    system_prompt: Optional[str] = Field(default=None, description="Full system prompt / persona text")
    model: Optional[str] = Field(default=None, description="Model name from models config")
    skills: Optional[List[str]] = Field(default=None, description="Skill names to activate for this agent")
    toolsets: Optional[List[str]] = Field(
        default=None, description="Tool groups to enable (e.g. ['code', 'web']). None = all"
    )
    auto_fallback: Optional[bool] = Field(default=None, description="Model auto-fallback")
    timeout_s: Optional[int] = Field(default=None, description="Max seconds per chat call (None = 300)")
    max_retries: int = Field(default=0, description="Max retries on failure (0 = no retries)")

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        return {k: v for k, v in data.items() if v is not None}


class AppConfig(BaseModel):
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    mcpServers: Dict[str, MCPServerConfig] = {}
    models: Dict[str, Dict[str, ModelConfig]] = Field(
        default_factory=dict, description="Models grouped by category (text/image/audio/video)"
    )
    default_model: Optional[str] = Field(default=None, description="The default LLM model")
    user_agent: Optional[str] = Field(default="CLIver", description="User-Agent header for LLM provider HTTP requests")
    enabled_toolsets: Optional[List[str]] = Field(
        default=None,
        description="Override which tool groups are enabled. Default: auto-detect from environment.",
    )
    gateway: Optional[GatewayConfig] = Field(default=None, description="Gateway daemon configuration")
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session storage limits")
    search_engines: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of search engines for WebSearch tool (e.g. [bing, baidu]). "
        "First engine is tried; others are fallbacks. "
        "Available: duckduckgo, bing, google, baidu, sogou. Default: [duckduckgo].",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="IANA timezone (e.g. Asia/Shanghai). Defaults to system local.",
    )
    theme: Optional[str] = Field(default=None, description="UI theme: dark (default), light, dracula")
    skill_auto_learn: bool = Field(
        default=False,
        description="Enable autonomous skill creation after complex tasks (default: off).",
    )
    model_auto_fallback: bool = Field(
        default=True,
        description="Automatically fall back to another model when the current one fails (default: on).",
    )
    agents: Dict[str, AgentConfig] = Field(
        default_factory=dict,
        description="Named agent configurations (profiles around AgentCore)",
    )
    default_agent: Optional[str] = Field(default=None, description="Default agent name")

    def resolve_secrets(self, key_store=None) -> None:
        """Resolve all secret references in the config tree.

        Secret fields (api_key, token, app_token, admin_password) are resolved
        via the three-layer chain: KeyStore -> env var -> literal.
        Non-secret template fields ({{ }}) use Jinja2 rendering.
        """
        if key_store is None:
            from cliver.key_store import KeyStore
            from cliver.util import get_config_dir

            ks_path = Path(get_config_dir()) / "cliver.db"
            key_store = KeyStore(ks_path)

        from cliver.template_utils import resolve_secret

        for provider in self.providers.values():
            if provider.api_key:
                provider.api_key = resolve_secret(provider.api_key, key_store)

        if self.gateway:
            if self.gateway.admin_password:
                self.gateway.admin_password = resolve_secret(self.gateway.admin_password, key_store)
            if self.gateway.api_key:
                self.gateway.api_key = resolve_secret(self.gateway.api_key, key_store)

        if self.gateway and self.gateway.platforms:
            for plat in self.gateway.platforms.values():
                if plat.token:
                    plat.token = resolve_secret(plat.token, key_store)
                if plat.app_token:
                    plat.app_token = resolve_secret(plat.app_token, key_store)

        _resolve_non_secret_templates(self)

    def model_dump(self, **kwargs):
        """Override to exclude null values."""
        data = super().model_dump(**kwargs)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}


_SECRET_FIELDS = frozenset({"api_key", "token", "app_token", "admin_password"})


def _resolve_non_secret_templates(obj) -> None:
    """Resolve {{ }} Jinja2 templates in non-secret fields only."""
    if isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            if field_name in _SECRET_FIELDS:
                continue
            val = getattr(obj, field_name, None)
            if val is None:
                continue
            if isinstance(val, str) and "{{" in val:
                setattr(obj, field_name, render_template_if_needed(val))
            elif isinstance(val, BaseModel):
                _resolve_non_secret_templates(val)
            elif isinstance(val, dict):
                _resolve_non_secret_template_dict(val)
            elif isinstance(val, list):
                _resolve_non_secret_template_list(val)
        if hasattr(obj, "__pydantic_extra__") and obj.__pydantic_extra__:
            _resolve_non_secret_template_dict(obj.__pydantic_extra__)
    elif isinstance(obj, dict):
        _resolve_non_secret_template_dict(obj)


def _resolve_non_secret_template_dict(d: dict) -> None:
    for key, val in list(d.items()):
        if key in _SECRET_FIELDS:
            continue
        if isinstance(val, str) and "{{" in val:
            d[key] = render_template_if_needed(val)
        elif isinstance(val, BaseModel):
            _resolve_non_secret_templates(val)
        elif isinstance(val, dict):
            _resolve_non_secret_template_dict(val)
        elif isinstance(val, list):
            _resolve_non_secret_template_list(val)


def _resolve_non_secret_template_list(lst: list) -> None:
    for i, val in enumerate(lst):
        if isinstance(val, str) and "{{" in val:
            lst[i] = render_template_if_needed(val)
        elif isinstance(val, BaseModel):
            _resolve_non_secret_templates(val)
        elif isinstance(val, dict):
            _resolve_non_secret_template_dict(val)


# TODO: support the configuration from others like from a k8s ConfigMap


class ConfigManager:
    """Configuration manager for Cliver client."""

    def __init__(self, config_dir: Path, config: Optional[AppConfig] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Configuration directory
            config: Pre-loaded config (skips reading from disk if provided)
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / "config.yaml"
        self.config = config if config is not None else self._load_config()
        self.config.resolve_secrets()

    def _load_config(self) -> AppConfig:
        """Load configuration from YAML file.

        Returns:
            Cliver configuration
        """
        if not self.config_file.exists():
            logger.info(f"No configuration file found at {str(self.config_dir)}, using default configuration.")
            return AppConfig()

        try:
            with open(self.config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # safe_load returns None for empty files
            if not config_data:
                return AppConfig()

            # Remove legacy agent_name fields
            config_data.pop("agent_name", None)
            config_data.pop("default_agent_name", None)

            # Parse providers
            providers: Dict[str, ProviderConfig] = {}
            if "providers" in config_data and isinstance(config_data["providers"], dict):
                for pname, pdata in config_data["providers"].items():
                    if isinstance(pdata, dict):
                        pdata["name"] = pname
                        prov = ProviderConfig(**pdata)
                        providers[pname] = prov
                        config_data["providers"][pname] = prov

            # Parse models by category
            flat_models: Dict[str, ModelConfig] = {}
            cat_models: Dict[str, Dict[str, ModelConfig]] = {}
            if "models" in config_data and isinstance(config_data["models"], dict):
                for cat_name, cat_data in config_data["models"].items():
                    if isinstance(cat_data, dict) and cat_data:
                        cat_models[cat_name] = {}
                        for model_key, model_data in cat_data.items():
                            if isinstance(model_data, dict):
                                model_data["name"] = model_key
                                model_data["category"] = cat_name
                                mc = ModelConfig(**model_data)
                                flat_models[mc.name] = mc
                                cat_models[cat_name][mc.name] = mc

            config_data["models"] = cat_models

            mcp_servers_data = config_data.get("mcpServers")
            if mcp_servers_data and isinstance(mcp_servers_data, dict):
                converted_servers = {}
                for name, server in mcp_servers_data.items():
                    if isinstance(server, dict):
                        server_dict = server.copy()
                        server_dict.pop("name", None)
                        transport = server_dict.get("transport")
                        server_config = {"name": name, **server_dict}
                        if transport == "stdio":
                            converted_servers[name] = StdioMCPServerConfig(**server_config)
                        elif transport == "sse":
                            converted_servers[name] = SSEMCPServerConfig(**server_config)
                        elif transport == "streamable_http":
                            converted_servers[name] = StreamableHttpMCPServerConfig(**server_config)
                        elif transport == "websocket":
                            converted_servers[name] = WebSocketMCPServerConfig(**server_config)
                        else:
                            raise ValueError(f"Unknown transport {transport}")
                config_data["mcpServers"] = converted_servers

            # Inject agent name from dict key (matches provider/MCP pattern)
            if "agents" in config_data and isinstance(config_data["agents"], dict):
                for aname, adata in list(config_data["agents"].items()):
                    if isinstance(adata, dict):
                        adata["name"] = aname

            return AppConfig(**config_data)
        except Exception as e:
            logger.error("Error loading configuration: %s", e, stack_info=True, exc_info=True)
            raise e

    def _save_config(self) -> None:
        """Save configuration to YAML file.

        Models are grouped by category, providers by name.
        """
        try:
            if not self.config_dir.exists():
                self.config_dir.mkdir(parents=True, exist_ok=True)

            config_data = self.config.model_dump()

            # Group models by category
            cat_models: Dict[str, dict] = {}
            for model in self.all_models().values():
                cat = model.category or "text"
                if cat not in cat_models:
                    cat_models[cat] = {}
                entry = {"provider": model.provider, "model": model.model}
                if model.api_url:
                    entry["api_url"] = model.api_url
                if model.options:
                    opts = model.options.model_dump(exclude_unset=True)
                    if opts:
                        entry["options"] = opts
                cat_models[cat][model.name] = entry
            config_data["models"] = cat_models

            # Serialize providers
            serialized_providers = {}
            for name, prov in self.config.providers.items():
                serialized_providers[name] = prov.model_dump()
            config_data["providers"] = serialized_providers

            if "mcpServers" in config_data:
                serialized_servers = {}
                for name, server in self.config.mcpServers.items():
                    serialized_servers[name] = server.model_dump()
                config_data["mcpServers"] = serialized_servers

            with open(self.config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        except Exception as e:
            logger.error("Error saving configuration: %s", e)
            raise e

    def list_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """List all mcp servers.

        Returns:
            List of mcp server information
        """
        return self.config.mcpServers

    def list_mcp_servers_for_mcp_caller(self) -> Dict[str, Dict]:
        """List all mcp servers as dictionaries for the MCP caller.

        Returns:
            List of mcp server information as dictionaries
        """
        # Convert Pydantic models to dictionaries for compatibility with
        # MCP server caller
        return {name: server.model_dump() for name, server in self.config.mcpServers.items()}

    def add_or_update_mcp_server(
        self,
        name: str,
        transport: str = "stdio",
        command: str = None,
        args: list = None,
        env: dict = None,
        url: str = None,
        headers: dict = None,
    ) -> None:
        """Add or update an MCP server in config."""
        # Normalize short transport names
        if transport == "streamable":
            transport = "streamable_http"
        if transport == "stdio":
            if not command:
                raise ValueError("Command is required for stdio transport")
            self.config.mcpServers[name] = StdioMCPServerConfig(name=name, command=command, args=args, env=env)
        elif transport == "sse":
            if not url:
                raise ValueError("URL is required for SSE transport")
            self.config.mcpServers[name] = SSEMCPServerConfig(name=name, url=url, headers=headers)
        elif transport == "streamable_http":
            if not url:
                raise ValueError("URL is required for streamable_http transport")
            self.config.mcpServers[name] = StreamableHttpMCPServerConfig(name=name, url=url, headers=headers)
        elif transport == "websocket":
            if not url:
                raise ValueError("URL is required for websocket transport")
            self.config.mcpServers[name] = WebSocketMCPServerConfig(name=name, url=url)
        else:
            raise ValueError(f"Unsupported transport: {transport}")
        self._save_config()

    def remove_mcp_server(self, name: str) -> bool:
        """Remove an MCP server by name. Returns True if found and removed."""
        if name in self.config.mcpServers:
            del self.config.mcpServers[name]
            self._save_config()
            return True
        return False

    def all_models(self) -> Dict[str, ModelConfig]:
        """Return all models as a flat dict (name -> ModelConfig) across all categories."""
        result: Dict[str, ModelConfig] = {}
        for cat_data in self.config.models.values():
            if isinstance(cat_data, dict):
                result.update(cat_data)
        return result

    def list_llm_models(self) -> Dict[str, ModelConfig]:
        """List all LLM Models (flat dict)."""
        return self.all_models()

    def add_or_update_llm_model(
        self,
        provider: str,
        model_name: str,
        api_model_name: str = "",
        category: str = "text",
        api_url: Optional[str] = None,
        options: Dict[str, Any] = None,
        is_default: bool = None,
    ) -> None:
        """Add or update a model under a provider.

        Args:
            provider: Provider name (must exist in config.providers)
            model_name: Config key (short name, e.g. 'minimax-2.7')
            api_model_name: API-facing model name (e.g. 'MiniMax-M2.7')
            category: One of 'text', 'image', 'audio', 'video'
            api_url: Optional per-model endpoint override
            options: Optional model options dict
            is_default: Set this model as the default
        """
        if provider not in self.config.providers:
            raise ValueError(f"Provider '{provider}' not found. Add it first with /provider add.")

        if not api_model_name:
            api_model_name = model_name

        if category not in self.config.models:
            self.config.models[category] = {}

        if model_name in self.config.models.get(category, {}):
            llm = self.config.models[category][model_name]
        else:
            llm = ModelConfig(
                name=model_name, provider=provider, model=api_model_name, category=category, api_url=api_url
            )
            self.config.models.setdefault(category, {})[model_name] = llm
            if self.config.default_model is None:
                self.config.default_model = model_name

        if api_model_name:
            llm.model = api_model_name
        if api_url is not None:
            llm.api_url = api_url
        if options:
            llm.options = ModelOptions(**options)
        if is_default:
            self.config.default_model = model_name

        self._save_config()

    def remove_llm_model(self, name: str) -> bool:
        """Remove a model by config key."""
        mc = self._resolve_model_name(name)
        if not mc:
            return False

        cat = mc.category or "text"
        self.config.models.get(cat, {}).pop(mc.name, None)

        if self.config.default_model == mc.name:
            defaults = self.all_models()
            self.config.default_model = next(iter(defaults)) if defaults else None

        self._save_config()
        return True

    def set_default_model(self, name: str) -> bool:
        """Set the default LLM model. Accepts canonical or short-form."""
        mc = self._resolve_model_name(name)
        if not mc:
            return False
        if self.config.default_model == mc.name:
            return True
        self.config.default_model = mc.name
        self._save_config()
        return True

    def set_user_agent(self, user_agent: str) -> None:
        """Set the User-Agent header string."""
        self.config.user_agent = user_agent
        self._save_config()

    def add_or_update_provider(
        self,
        name: str,
        type: str,
        api_url: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ) -> None:
        prov = self.config.providers.get(name)
        if prov:
            prov.type = type
            prov.api_url = api_url
            if api_key is not None:
                prov.api_key = api_key
            if rate_limit is not None:
                prov.rate_limit = rate_limit
            if pricing is not None:
                prov.pricing = PricingConfig(**pricing)
        else:
            self.config.providers[name] = ProviderConfig(
                name=name,
                type=type,
                api_url=api_url,
                api_key=api_key,
                rate_limit=rate_limit,
                pricing=PricingConfig(**pricing) if pricing else None,
            )
        self._save_config()

    def remove_provider(self, name: str) -> bool:
        if name in self.config.providers:
            referencing = [m.name for m in self.all_models().values() if m.provider == name]
            if referencing:
                raise ValueError(f"Cannot remove provider '{name}': used by models: {', '.join(referencing)}")
            self.config.providers.pop(name)
            self._save_config()
            return True
        return False

    def list_providers(self) -> Dict[str, ProviderConfig]:
        return self.config.providers

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Return the ProviderConfig for *name*, or None if not found."""
        return self.config.providers.get(name)

    def get_llm_model(self, name: Optional[str] = None) -> Optional[ModelConfig]:
        if not name:
            name = self.config.default_model
        if not name or not self.config.models:
            return None
        return self._resolve_model_name(name)

    def _resolve_model_name(self, name: str) -> Optional[ModelConfig]:
        """Resolve a model name to a ModelConfig across all categories."""
        all_models = self.all_models()
        if name in all_models:
            return all_models[name]
        # Short-form fallback: match by suffix
        matches = [mc for key, mc in all_models.items() if key.endswith(f"/{name}") or key == name]
        if len(matches) == 1:
            return matches[0]
        return None
