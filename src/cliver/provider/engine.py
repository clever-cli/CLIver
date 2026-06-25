"""Protocol engine base and factory.

Each engine handles ONE API protocol
(OpenAI-compatible or Anthropic-native).
Engines are stateless beyond the SDK client — Providers delegate to them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator

from cliver.events import EventHandler
from cliver.messages import CLIverMessage, CLIverMessageChunk
from cliver.provider import CLIverResponse, MessageConverter


class ProtocolEngine(MessageConverter):
    """Owns an SDK client, handles message/tool conversion and API calls.

    Stateless beyond the client — safe to share across provider instances
    using the same protocol.

    Inherits ``msg_to_native`` and ``tool_to_native`` from MessageConverter.
    Subclasses implement those for their specific protocol, plus ``chat()``
    and ``stream()`` for the API calls.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        on_event: EventHandler | None = None,
        user_agent: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.on_event = on_event
        self.user_agent = user_agent

    async def close(self) -> None:
        """Close the underlying SDK client and release connections.

        Subclasses that hold SDK clients MUST override this to close them.
        The default is a no-op for engines without persistent connections.
        """
        return

    @abstractmethod
    async def chat(
        self,
        messages: list[Any],
        tools: list[Any],
        model: str,
        options: dict[str, Any],
    ) -> CLIverResponse:
        """Non-streaming chat completion."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Any],
        tools: list[Any],
        model: str,
        options: dict[str, Any],
    ) -> AsyncIterator[CLIverMessageChunk]:
        """Streaming chat completion."""
        ...

    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        media_type: str = "image",
        media: list[Any] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate media (image, audio, video).

        Engines that support media generation override this.
        The default returns an error response — no exception is raised,
        so callers can safely call generate() on any engine.
        """
        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content=f"{type(self).__name__} does not support {media_type} generation.",
            ),
        )


# ── Engine factory ─────────────────────────────────────────────


def create_engine(
    protocol: str,
    api_key: str,
    base_url: str,
    on_event: EventHandler | None = None,
    user_agent: str | None = None,
    use_bearer_auth: bool = False,
) -> ProtocolEngine:
    """Create a ProtocolEngine for the given protocol.

    Uses lazy imports to avoid circular dependencies.
    """
    if protocol == "openai":
        from cliver.provider.openai_engine import OpenAIEngine

        return OpenAIEngine(api_key=api_key, base_url=base_url, on_event=on_event, user_agent=user_agent)
    elif protocol == "anthropic":
        from cliver.provider.anthropic_engine import AnthropicEngine

        return AnthropicEngine(
            api_key=api_key,
            base_url=base_url,
            on_event=on_event,
            user_agent=user_agent,
            use_bearer_auth=use_bearer_auth,
        )
    else:
        raise ValueError(f"Unknown protocol '{protocol}'. Supported: ['openai', 'anthropic']")
