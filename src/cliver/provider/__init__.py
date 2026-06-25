"""Provider interface and request/response models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from cliver.media import MediaContent
from cliver.messages import CLIverMessage, CLIverMessageChunk, UsageInfo
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


class CLIverRequest(BaseModel):
    """A request to an LLM provider.

    Media (images/audio/video) is embedded into the user message content
    as content blocks by AgentCore before the request is built.
    """

    messages: list[CLIverMessage]
    tools: list[CLIverTool] | None = None
    model: str
    options: dict[str, Any] = Field(default_factory=dict)
    # temperature, top_p, max_tokens, thinking, etc.
    # Passed through — each engine filters what it supports.


class CLIverResponse(BaseModel):
    """A response from an LLM provider."""

    message: CLIverMessage
    media: list[MediaContent] | None = None
    # Generated or returned media files (images, audio, video).
    usage: UsageInfo | None = None


class MessageConverter(ABC):
    """Converts CLIverMessage to native provider format.

    Shared by both ProtocolEngine (protocol-level conversion) and
    Provider (brand-specific injection on top of the engine).
    """

    @abstractmethod
    def msg_to_native(self, msg: CLIverMessage) -> Any:
        """Convert CLIverMessage → native format for the target protocol."""
        ...


class Provider(MessageConverter):
    """Interface for LLM inference.

    A Provider wraps a specific brand (DeepSeek, MiniMax, OpenAI, ...)
    and delegates to a ProtocolEngine (OpenAI or Anthropic) for the
    actual API calls and message conversion.

    Subclasses MUST define ``supported_protocols`` — a list of protocol
    names this provider supports (e.g. ``["openai", "anthropic"]``).
    The check runs at class-definition time, so a missing declaration
    is caught at import, not at the first API call.
    """

    supported_protocols: list[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only validate classes that declare supported_protocols in their
        # own __dict__ (concrete providers).  Intermediate abstract bases
        # (e.g. _EngineProvider) skip the check and let subclasses define it.
        if "supported_protocols" in cls.__dict__:
            if not cls.supported_protocols:
                raise TypeError(
                    f"{cls.__name__} must define supported_protocols "
                    f"(e.g. supported_protocols = ['openai', 'anthropic']), "
                    f"not an empty list"
                )

    def __init__(self, protocol: str, api_key: str, base_url: str):
        if protocol not in self.supported_protocols:
            raise ValueError(
                f"Provider '{self.provider_name()}' does not support protocol '{protocol}'. "
                f"Supported: {', '.join(self.supported_protocols)}"
            )
        self.protocol = protocol
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def provider_name(cls) -> str:
        """Short name for logging/registration."""
        return cls.__name__.removesuffix("Provider").lower()

    # ── Hooks (subclasses override for brand-specific behavior) ──

    def on_response(self, response: CLIverResponse) -> CLIverResponse:
        """Post-process a response. Override to extract vendor_ext fields."""
        return response

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        """Post-process a streaming chunk. Override to remap vendor_ext keys."""
        return chunk

    def filter_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Filter/transform options before passing to the engine."""
        return options

    # ── Public API ─────────────────────────────────────────

    @abstractmethod
    async def chat(self, request: CLIverRequest) -> CLIverResponse: ...

    @abstractmethod
    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]: ...

    async def close(self) -> None:
        """Close the provider and its underlying engine/client.

        Call this when done using the provider to release httpx connections.
        """
        if hasattr(self, "engine") and hasattr(self.engine, "close"):
            await self.engine.close()

    # ── Media generation (HTTP-based, shared across providers) ────

    # Subclasses override these to configure media generation endpoints.
    # Keys: "image", "audio", "video".  Values: full URL or path appended
    # to ``self.base_url`` when the value starts with "/".
    _media_gen_urls: dict[str, str] = {}

    def _build_media_gen_body(self, prompt: str, model: str, media_type: str, **options) -> dict:
        """Build the request body for media generation.

        Override to customise field names or add provider-specific params.
        The default uses ``{"model": ..., "prompt": ...}`` which works for
        most OpenAI-compatible image/audio/video APIs.
        """
        return {"model": model, "prompt": prompt, **options}

    def _build_media_gen_headers(self) -> dict[str, str]:
        """Build HTTP headers for media generation requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _check_media_response_error(self, data: dict, media_type: str) -> str | None:
        """Check for provider-specific error envelopes in the JSON body.

        Some providers (e.g. MiniMax) return HTTP 200 with an error code
        inside the response body.  Override this to extract and return
        the error message, or ``None`` if the response is genuinely
        successful.
        """
        return None

    def _extract_media_items(self, data: dict, media_type: str) -> list[dict]:
        """Extract raw media items from a provider's JSON response.

        Each returned dict should have at least one of ``url`` or
        ``b64_json``.  Override this to handle provider-specific response
        shapes (nested wrappers, different field names, etc.).

        The default handles ``{"data": [{"url": "...", ...}]}`` — the
        OpenAI / MiniMax / compatible format.
        """
        raw = data.get("data", data)
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _resolve_media_gen_url(self, media_type: str) -> str | None:
        """Resolve the URL for a media generation endpoint.

        Looks up ``_media_gen_urls[media_type]`` and resolves it against
        ``self.base_url`` (the model's configured ``api_url``):

        - ``"https://..."`` → as-is (absolute URL, bypasses base_url)
        - ``"/path"``       → ``base_url + "/path"``
        - ``""``            → ``base_url`` (model api_url IS the full endpoint)
        - absent            → ``None`` (media type not supported by this provider)
        """
        path = self._media_gen_urls.get(media_type)
        if path is None:
            return None
        if not path:
            return self.base_url or None
        # Absolute URL (e.g. different domain) — use as-is, ignore base_url
        if path.startswith(("http://", "https://")):
            return path
        if path.startswith("/") and self.base_url:
            return self.base_url.rstrip("/") + path
        if self.base_url:
            return self.base_url.rstrip("/") + "/" + path.lstrip("/")
        return path

    async def _http_generate(
        self,
        prompt: str,
        *,
        model: str,
        media_type: str = "image",
        output_dir: str | None = None,
        timeout: float = 120.0,
        **options,
    ) -> CLIverResponse:
        """Generate media via direct HTTP POST.

        This is the shared implementation used by providers that expose
        dedicated media endpoints (MiniMax, etc.).  Subclasses only need
        to declare ``_media_gen_urls`` and optionally override the body /
        response parser methods.

        Providers that use SDK-based generation (e.g. OpenAIEngine)
        should override ``generate()`` instead of using this method.
        """
        import logging as _logging
        from pathlib import Path

        import httpx

        from cliver.messages import CLIverMessage

        _logger = _logging.getLogger(__name__)

        url = self._resolve_media_gen_url(media_type)
        if not url:
            raise ValueError(
                f"{self.provider_name()} does not support {media_type} generation. "
                f"Add a '{media_type}' entry to _media_gen_urls on the provider class, "
                f"or set api_url on the model config."
            )
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError(
                f"{self.provider_name()} {media_type} URL is invalid: {url!r}. "
                f"Set a valid api_url on the model or provider in config.yaml."
            )

        body = self._build_media_gen_body(prompt, model, media_type, **options)
        headers = self._build_media_gen_headers()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body, headers=headers)
                data = resp.json()

            if not resp.is_success:
                err_msg = data.get("msg") or data.get("error", {}).get("message", resp.text)
                code = data.get("code") or resp.status_code
                _logger.warning(
                    "%s %s API error: code=%s detail=%s",
                    self.provider_name(),
                    media_type,
                    code,
                    err_msg,
                )
                return CLIverResponse(
                    message=CLIverMessage(
                        role="assistant",
                        content=f"{self.provider_name()} {media_type} generation failed: {err_msg}",
                    ),
                )
        except httpx.TimeoutException:
            _logger.warning("%s %s API timeout: url=%s", self.provider_name(), media_type, url)
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=f"{self.provider_name()} {media_type} generation timed out.",
                ),
            )
        except httpx.HTTPStatusError as e:
            _logger.warning("%s %s HTTP error: %s", self.provider_name(), media_type, e)
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=f"{self.provider_name()} {media_type} generation failed: HTTP {e.response.status_code}",
                ),
            )
        except Exception as e:
            _logger.warning("%s %s generation failed: %s", self.provider_name(), media_type, e)
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=f"{self.provider_name()} {media_type} generation failed: {e}",
                ),
            )

        # Check for provider-specific error envelopes in the JSON body
        # (e.g. MiniMax returns HTTP 200 with base_resp.status_code ≠ 0).
        err = self._check_media_response_error(data, media_type)
        if err:
            _logger.warning(
                "%s %s API error: %s",
                self.provider_name(),
                media_type,
                err,
            )
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=f"{self.provider_name()} {media_type} generation failed: {err}",
                ),
            )

        # Let the provider extract raw items from its JSON response.
        raw_items = self._extract_media_items(data, media_type)

        if not raw_items:
            _logger.warning(
                "%s %s returned no media items. Raw response: %s",
                self.provider_name(),
                media_type,
                data,
            )

        # Build MediaContent — same pattern for ALL providers.
        from cliver.media import MediaType

        mime_map = {"image": "image/png", "audio": "audio/mpeg", "video": "video/mp4"}
        mime = mime_map.get(media_type, "application/octet-stream")
        mt = MediaType(media_type)

        media_items: list[MediaContent] = []
        for item in raw_items:
            media_data = item.get("url") or item.get("b64_json") or ""
            if media_data:
                media_items.append(MediaContent(type=mt, data=media_data, mime_type=mime))

        # Save to disk — same pattern for ALL providers.
        if output_dir and media_items:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ext = {"image": "png", "audio": "mp3", "video": "mp4"}.get(media_type, "bin")
            for i, mc in enumerate(media_items):
                mc.save(out / f"generated_{i}.{ext}")

        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content=f"Generated {len(media_items)} {media_type}(s) via {self.provider_name()}.",
            ),
            media=media_items if media_items else None,
        )

    async def generate(
        self, prompt: str, *, model: str, media_type: str = "image", media=None, output_dir=None, **options
    ) -> CLIverResponse:
        """Generate media (image, audio, video).

        The default implementation uses :meth:`_http_generate` — a direct
        HTTP POST to the provider's media endpoint.  Subclasses that use
        an SDK should override this method.
        """
        return await self._http_generate(prompt, model=model, media_type=media_type, output_dir=output_dir, **options)
