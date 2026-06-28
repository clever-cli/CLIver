"""Concrete Provider implementations and auto-detection.

Each provider is a thin subclass of _EngineProvider.
Brand-specific behavior is added via hook overrides.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from cliver.messages import CLIverMessage, CLIverMessageChunk
from cliver.provider import CLIverRequest, CLIverResponse, Provider
from cliver.provider.engine import ProtocolEngine, create_engine


class _EngineProvider(Provider):
    """Base for providers that delegate to a ProtocolEngine.

    Creates the engine, implements msg_to_native/tool_to_native by
    delegating to the engine, and routes chat/stream through
    message conversion → engine → response hooks.

    Subclasses override hooks (msg_to_native, on_response, on_chunk,
    filter_options) for brand-specific behavior.
    """

    def __init__(
        self,
        protocol: str,
        api_key: str,
        base_url: str,
        user_agent: str | None = None,
        *,
        logger: "logging.Logger | None" = None,
    ):
        super().__init__(protocol, api_key, base_url, logger=logger)
        use_bearer = getattr(self.__class__, "_anthropic_use_bearer_auth", False)
        self.engine: ProtocolEngine = create_engine(
            protocol,
            api_key,
            base_url,
            user_agent=user_agent,
            use_bearer_auth=use_bearer,
        )

    def msg_to_native(self, msg: CLIverMessage) -> Any:
        return self.engine.msg_to_native(msg)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        native_messages = [self.msg_to_native(m) for m in request.messages]
        options = self.filter_options(request.options)
        response = await self.engine.chat(native_messages, request.tools or [], request.model, options)

        # Attach request diagnostic data for session turn inspection.
        # Convert Pydantic models to plain dicts so vendor_ext serialises cleanly.
        response.message.vendor_ext["__llm_request__"] = {
            "model": request.model,
            "provider": self.provider_name(),
            "options": options,
            "system_prompt": next(
                (m.content for m in request.messages if m.role == "system"),
                None,
            ),
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "tools": [t.model_dump(exclude={"execute"}) for t in (request.tools or [])],
        }

        return self.on_response(response)

    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]:
        messages = [self.msg_to_native(m) for m in request.messages]
        options = self.filter_options(request.options)
        async for chunk in self.engine.stream(messages, request.tools or [], request.model, options):
            yield self.on_chunk(chunk)

    async def generate(
        self, prompt: str, *, model: str, media_type: str = "image", media=None, output_dir=None, **options
    ) -> CLIverResponse:
        return await self.engine.generate(
            prompt=prompt, model=model, media_type=media_type, media=media, output_dir=output_dir, **options
        )


# ── Lazy imports for individual providers ────────────────────
# Each provider module defines a class that extends _EngineProvider.
# Using lazy imports here avoids circular dependencies since
# provider modules import from this __init__.py.


def _get_deepseek_provider():
    from cliver.provider.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider


def _get_minimax_provider():
    from cliver.provider.providers.minimax import MiniMaxProvider

    return MiniMaxProvider


def _get_openai_provider():
    from cliver.provider.providers.openai import OpenAIProvider

    return OpenAIProvider


def _get_anthropic_provider():
    from cliver.provider.providers.anthropic import AnthropicProvider

    return AnthropicProvider


def _get_glm_provider():
    from cliver.provider.providers.glm import GLMProvider

    return GLMProvider


def _get_ollama_provider():
    from cliver.provider.providers.ollama import OllamaProvider

    return OllamaProvider


# ── Auto-detection ───────────────────────────────────────────

# URL substring → provider factory. Used to detect the provider class from a
# configured ``api_url``.  Order matters: first match wins.
_URL_PROVIDER_MAP: list[tuple[str, callable]] = [
    ("deepseek", _get_deepseek_provider),
    ("minimax", _get_minimax_provider),
    ("api.minimax", _get_minimax_provider),
    ("openai", _get_openai_provider),
    ("api.openai", _get_openai_provider),
    ("anthropic", _get_anthropic_provider),
    ("api.anthropic", _get_anthropic_provider),
    ("glm", _get_glm_provider),
    ("zhipu", _get_glm_provider),
    ("bigmodel", _get_glm_provider),
    ("ollama", _get_ollama_provider),
    ("localhost:11434", _get_ollama_provider),
]

# Provider name substring → provider factory.  Used when no ``api_url`` is
# configured — the only signal available is the user-defined provider config
# name (e.g. ``providers.deepseek`` in config.yaml).
#
# Matching is *substring* (case-insensitive), so user-chosen names like
# ``my-deepseek``, ``deepseek-prod``, or ``deepseek/china`` all resolve to
# DeepSeekProvider.  Order matters: first match wins, so more specific
# patterns (e.g. ``zhipu`` before ``glm``) are listed first.
_NAME_PROVIDER_MAP: list[tuple[str, callable]] = [
    ("deepseek", _get_deepseek_provider),
    ("minimax", _get_minimax_provider),
    ("zhipu", _get_glm_provider),
    ("bigmodel", _get_glm_provider),
    ("glm", _get_glm_provider),
    ("ollama", _get_ollama_provider),
    ("openai", _get_openai_provider),
    ("anthropic", _get_anthropic_provider),
]


def detect_provider_class(api_url: str, provider_name: str | None = None) -> type[Provider]:
    """Detect provider entity from the base URL or provider config name.

    Resolution order:
    1. *provider_name* — substring match (case-insensitive) against
       ``_NAME_PROVIDER_MAP`` patterns
    2. *api_url* — substring match against ``_URL_PROVIDER_MAP``
    3. Fall back to OpenAIProvider (most APIs are OpenAI-compatible).
    """
    if provider_name:
        name_lower = provider_name.lower()
        for pattern, factory in _NAME_PROVIDER_MAP:
            if pattern in name_lower:
                return factory()

    url_lower = api_url.lower()
    for pattern, factory in _URL_PROVIDER_MAP:
        if pattern in url_lower:
            return factory()
    return _get_openai_provider()


def resolve_base_url(
    model_api_url: str | None = None,
    provider_api_url: str | None = None,
    *,
    protocol: str = "openai",
    provider_cls: type[Provider] | None = None,
) -> str:
    """Resolve the full base URL for a provider, including class defaults.

    Resolution order:
    1. *model_api_url* — per-model override
    2. *provider_api_url* — per-provider config
    3. ``_default_base_urls[protocol]`` on *provider_cls*
    4. ``default_base_url`` on *provider_cls*

    Callers should use this BEFORE ``create_provider()`` so the URL is
    always fully resolved when the provider is created.
    """
    url = model_api_url or provider_api_url or ""
    if url:
        return url

    if provider_cls is None:
        return ""

    urls = getattr(provider_cls, "_default_base_urls", None)
    return (urls or {}).get(protocol) or getattr(provider_cls, "default_base_url", "")


def create_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    protocol: str = "openai",
    provider_class: type[Provider] | None = None,
    provider_name: str | None = None,
    user_agent: str | None = None,
    logger: "logging.Logger | None" = None,
) -> Provider:
    """Create a Provider instance.

    The *base_url* is resolved inside this factory:
    1. Use *base_url* if given
    2. Fall back to the provider class's ``_default_base_urls[protocol]``
    3. Fall back to ``default_base_url`` on the class

    Args:
        api_key: API key for the provider.
        base_url: Base URL (optional — defaults are used if empty).
        protocol: ``"openai"`` (default) or ``"anthropic"``.
        provider_class: Provider class. If ``None``, auto-detected from
            *base_url* and/or *provider_name*.
        provider_name: Provider config name (e.g. ``"deepseek"``, ``"minimax"``).
            Used to look up the correct provider class when *base_url* is empty.
        user_agent: Optional User-Agent header.
        logger: Optional logger for this provider.  When provided, all log
            output from the provider is routed through this logger.

    Returns:
        A Provider instance ready to use.
    """
    api_key = api_key or ""
    url = base_url or ""

    if provider_class is None:
        cls = detect_provider_class(url, provider_name=provider_name)
    else:
        cls = provider_class

    if protocol not in cls.supported_protocols:
        cls = _get_anthropic_provider() if protocol == "anthropic" else _get_openai_provider()

    # Resolve URL from class defaults if not explicitly provided
    if not url:
        url = resolve_base_url(protocol=protocol, provider_cls=cls)

    return cls(api_key=api_key, base_url=url, protocol=protocol, user_agent=user_agent, logger=logger)
