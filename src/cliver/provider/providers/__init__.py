"""Concrete Provider implementations and auto-detection.

Each provider is a thin subclass of _EngineProvider.
Brand-specific behavior is added via hook overrides.
"""

from __future__ import annotations

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

    def __init__(self, protocol: str, api_key: str, base_url: str, user_agent: str | None = None):
        super().__init__(protocol, api_key, base_url)
        self.engine: ProtocolEngine = create_engine(protocol, api_key, base_url, user_agent=user_agent)

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

_NAME_PROVIDER_MAP: dict[str, callable] = {
    "deepseek": _get_deepseek_provider,
    "minimax": _get_minimax_provider,
    "openai": _get_openai_provider,
    "anthropic": _get_anthropic_provider,
    "glm": _get_glm_provider,
    "ollama": _get_ollama_provider,
}


def detect_provider_class(api_url: str) -> type[Provider]:
    """Detect provider entity from the base URL.

    Falls back to OpenAIProvider for unknown URLs (most are OpenAI-compatible).
    """
    url_lower = api_url.lower()
    for pattern, factory in _URL_PROVIDER_MAP:
        if pattern in url_lower:
            return factory()
    return _get_openai_provider()


def create_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    protocol: str = "openai",
    provider_class: type[Provider] | str | None = None,
    user_agent: str | None = None,
) -> Provider:
    """Create a Provider instance.

    Args:
        api_key: API key for the provider (may be None).
        base_url: Base URL for the provider API.
            Falls back to the provider's ``default_base_url`` when empty/None.
        protocol: ``"openai"`` (default) or ``"anthropic"``.
        provider_class: Optional override for auto-detection.
            Can be a class, a string name (e.g. ``"deepseek"``, ``"minimax"``),
            or ``None`` for auto-detection from ``base_url``.
        user_agent: Optional User-Agent header sent with all HTTP requests.

    Returns:
        A Provider instance ready to use.
    """
    api_key = api_key or ""
    url = base_url or ""

    if provider_class is None:
        cls = detect_provider_class(url)
    elif isinstance(provider_class, str):
        factory = _NAME_PROVIDER_MAP.get(provider_class.lower())
        cls = factory() if factory else _get_openai_provider()
    else:
        cls = provider_class

    # If the named provider doesn't support the requested protocol, fall
    # back to the canonical provider for that protocol.  This can happen
    # when a model's ``provider`` field names a brand whose default
    # protocol differs from the one in ProviderConfig (e.g. MiniMax
    # using ``type: anthropic`` but ``provider: minimax``).
    if protocol not in cls.supported_protocols:
        cls = _get_anthropic_provider() if protocol == "anthropic" else _get_openai_provider()

    # Use provider's default URL if none specified
    if not url:
        url = getattr(cls, "default_base_url", "")

    return cls(api_key=api_key, base_url=url, protocol=protocol, user_agent=user_agent)
