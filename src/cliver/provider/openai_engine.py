"""OpenAI-compatible chat/completions protocol engine."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

from openai import NOT_GIVEN, AsyncOpenAI

from cliver.events import EventHandler, InferenceEvent, InferenceEventType
from cliver.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    ToolCallChunk,
    UsageInfo,
)
from cliver.provider import CLIverResponse
from cliver.provider.engine import ProtocolEngine
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


class OpenAIEngine(ProtocolEngine):
    """OpenAI-compatible chat/completions protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        on_event: EventHandler | None = None,
        user_agent: str | None = None,
    ):
        super().__init__(api_key, base_url, on_event, user_agent=user_agent)
        extra_headers = {"User-Agent": user_agent} if user_agent else {}
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)

    # ── Conversion ──────────────────────────────────────────

    def msg_to_native(self, msg: CLIverMessage) -> dict:
        native: dict[str, Any] = {"role": msg.role, "content": msg.content}

        if msg.tool_calls:
            native["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            native["tool_call_id"] = msg.tool_call_id

        return native

    def tool_to_native(self, tool: CLIverTool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def extract_cliver_message(self, choice) -> CLIverMessage:
        m = choice.message
        vendor_ext: dict[str, Any] = {}

        rc = getattr(m, "reasoning_content", None)
        if rc:
            vendor_ext["reasoning_content"] = rc

        content = m.content
        tool_calls = None
        if m.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
                for tc in m.tool_calls
            ]
            if not content:
                content = None

        return CLIverMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            vendor_ext=vendor_ext,
        )

    def extract_chunk(self, delta) -> CLIverMessageChunk:
        chunk = CLIverMessageChunk(
            content=delta.content,
            vendor_ext={},
        )

        rc = getattr(delta, "reasoning_content", None)
        if rc:
            chunk.vendor_ext["reasoning_content"] = rc

        if delta.tool_calls:
            tc_chunks = []
            for tc in delta.tool_calls:
                tc_chunk = ToolCallChunk(index=tc.index or 0)
                if tc.id:
                    tc_chunk.id = tc.id
                if tc.function and tc.function.name:
                    tc_chunk.name = tc.function.name
                if tc.function and tc.function.arguments:
                    tc_chunk.args_delta = tc.function.arguments
                tc_chunks.append(tc_chunk)
            chunk.tool_call_chunks = tc_chunks

        return chunk

    # ── API Calls ───────────────────────────────────────────

    async def chat(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()
        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        try:
            native_tools = [self.tool_to_native(t) for t in tools] if tools else NOT_GIVEN
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=native_tools,
                **options,
            )
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="openai",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = UsageInfo(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

        cliver_msg = self.extract_cliver_message(choice)
        # Store raw SDK response for diagnostic display
        try:
            cliver_msg.vendor_ext["__llm_raw_response__"] = response.model_dump()
        except Exception:
            pass

        return CLIverResponse(
            message=cliver_msg,
            usage=usage,
        )

    async def stream(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()
        first_token_emitted = False

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        try:
            native_tools = [self.tool_to_native(t) for t in tools] if tools else NOT_GIVEN
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=native_tools,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="openai",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        async for sdk_chunk in stream:
            if not sdk_chunk.choices:
                continue

            delta = sdk_chunk.choices[0].delta

            if not first_token_emitted and (delta.content or (delta.tool_calls and delta.tool_calls[0].function)):
                first_token_emitted = True
                await self._emit(
                    InferenceEvent(
                        event=InferenceEventType.FIRST_TOKEN,
                        model=model,
                        provider="openai",
                        request_id=request_id,
                        data={"latency_ms": (time.monotonic() - start) * 1000},
                    )
                )

            chunk = self.extract_chunk(delta)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        await self._emit(
                            InferenceEvent(
                                event=InferenceEventType.TOOL_CALL,
                                model=model,
                                provider="openai",
                                request_id=request_id,
                                data={
                                    "tool_name": tc.function.name if tc.function else "?",
                                    "tool_call_id": tc.id,
                                },
                            )
                        )

            yield chunk

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

    async def _emit(self, event: InferenceEvent) -> None:
        if self.on_event:
            await self.on_event(event)

    # ── Media generation ────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        media: list[Any] | None = None,
        output_dir: str | None = None,
        media_type: str = "image",
        **options,
    ) -> CLIverResponse:
        """Generate media — image, audio, or video.

        Dispatches to the appropriate SDK method based on ``media_type``.
        ``media`` holds reference files for editing/variation.
        Generated files are saved to ``output_dir`` if given.
        """
        if media_type == "image":
            return await self._generate_image(prompt, model, media=media, output_dir=output_dir, **options)
        elif media_type == "audio":
            return await self._generate_audio(prompt, model, output_dir=output_dir, **options)
        elif media_type == "video":
            return self._generate_video(prompt, model, output_dir=output_dir, **options)
        else:
            raise ValueError(f"Unknown media_type '{media_type}'. Supported: image, audio, video")

    async def _generate_image(
        self,
        prompt: str,
        model: str,
        *,
        media: list[Any] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate images via the OpenAI Images API."""
        from pathlib import Path

        from cliver.media import MediaContent, MediaType

        reference_image: Any | None = None
        if media:
            for m in media:
                if hasattr(m, "type") and m.type == MediaType.IMAGE:
                    reference_image = m
                    break

        kwargs: dict[str, Any] = {"model": model, "prompt": prompt, **options}
        if reference_image:
            kwargs["image"] = reference_image.data
            if not reference_image.data.startswith("data:"):
                kwargs["image"] = f"data:image/png;base64,{reference_image.data}"

        response = await self.client.images.generate(**kwargs)

        media_items: list[MediaContent] = []
        out_dir = Path(output_dir) if output_dir else None

        for i, img in enumerate(response.data):
            url = img.url
            b64 = getattr(img, "b64_json", None)
            data = url or b64 or ""

            mc = MediaContent(type=MediaType.IMAGE, data=data, mime_type="image/png")
            if out_dir and data:
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = getattr(img, "filename", None) or f"generated_{i}.png"
                mc.save(out_dir / fname)
            media_items.append(mc)

        return self._build_generate_response(media_items)

    async def _generate_audio(
        self,
        prompt: str,
        model: str,
        *,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate audio via the OpenAI TTS API."""
        from pathlib import Path

        from cliver.media import MediaContent, MediaType

        voice = options.pop("voice", "alloy")
        response_format = options.pop("response_format", "mp3")
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "voice": voice,
            "response_format": response_format,
            **options,
        }

        response = await self.client.audio.speech.create(**kwargs)
        audio_bytes = response.content  # raw bytes from TTS

        mime = f"audio/{response_format}"
        mc = MediaContent(type=MediaType.AUDIO, data="", mime_type=mime)

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"speech_{uuid4().hex[:8]}.{response_format}"
            path = out_dir / fname
            path.write_bytes(audio_bytes)
            mc.saved_path = str(path)
            mc.data = str(path)

        return self._build_generate_response([mc])

    _DEFAULT_VIDEO_PATH = "/videos/generations"

    async def _generate_video(
        self,
        prompt: str,
        model: str,
        *,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate video via OpenAI-compatible endpoint.

        The endpoint path defaults to ``/videos/generations`` and can be
        overridden via ``options["path"]`` (set in model config options).
        """
        from pathlib import Path

        from cliver.media import MediaContent, MediaType

        path = options.pop("path", self._DEFAULT_VIDEO_PATH)
        body: dict[str, Any] = {"model": model, "prompt": prompt, **options}

        try:
            response = await self.client.post(path, body=body)
            result = response.json()
        except Exception as e:
            logger.warning("Video generation failed: %s", e)
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=f"Video generation failed: {e}",
                ),
            )

        media_items: list[MediaContent] = []
        out_dir = Path(output_dir) if output_dir else None

        data_list = result.get("data", [])
        if not isinstance(data_list, list):
            data_list = [result]

        for i, item in enumerate(data_list):
            url = item.get("url") or item.get("video_url", "")
            mc = MediaContent(type=MediaType.VIDEO, data=url or "", mime_type="video/mp4")
            if out_dir and url:
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = item.get("filename") or f"generated_{i}.mp4"
                mc.save(out_dir / fname)
            media_items.append(mc)

        return self._build_generate_response(media_items)

    @staticmethod
    def _build_generate_response(media_items: list) -> CLIverResponse:
        paths = [mc.saved_path for mc in media_items if mc.saved_path]
        label = media_items[0].type.value if media_items else "media"
        content = f"Generated {len(media_items)} {label}(s)."
        if paths:
            content += "\n" + "\n".join(f"- {p}" for p in paths)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content=content),
            media=media_items if media_items else None,
        )
