import asyncio
import contextvars
import structlog
from enum import Enum
from typing import AsyncGenerator, List, Optional
from google import genai
from google.genai import types as genai_types

from app.config import settings

logger = structlog.get_logger()

GEMINI_MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]

_usage_accumulator: contextvars.ContextVar = contextvars.ContextVar(
    "rihla_usage_accumulator", default=None
)


def begin_usage_tracking():
    """Start accumulating Gemini token usage for the current request scope."""
    _usage_accumulator.set([])


def consume_usage() -> list:
    """Return accumulated Gemini usage entries for the current request and reset."""
    entries = _usage_accumulator.get() or []
    _usage_accumulator.set(None)
    return entries


class KeyStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"


class GeminiKey:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.status = KeyStatus.ACTIVE
        self.fail_count = 0
        self.cooldown_until = 0.0
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 120000},
        )

    def mark_failed(self, cooldown_seconds: float = 60.0):
        self.fail_count += 1
        self.status = KeyStatus.DEGRADED
        self.cooldown_until = asyncio.get_event_loop().time() + cooldown_seconds
        logger.warning("Key marked degraded", key_suffix=self.api_key[-4:], cooldown=cooldown_seconds)

    def mark_success(self):
        self.fail_count = 0
        if self.status == KeyStatus.DEGRADED:
            self.status = KeyStatus.ACTIVE
            logger.info("Key revived", key_suffix=self.api_key[-4:])

    def is_available(self) -> bool:
        loop_time = asyncio.get_event_loop().time()
        if self.status in (KeyStatus.DEGRADED, KeyStatus.COOLDOWN):
            if loop_time >= self.cooldown_until:
                self.status = KeyStatus.ACTIVE
                logger.info("Key cooldown expired", key_suffix=self.api_key[-4:])
                return True
            return False
        return self.status == KeyStatus.ACTIVE


class GeminiClient:
    MAX_RETRIES = 10

    def __init__(self, api_keys: List[str], cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self.keys = [GeminiKey(k) for k in api_keys]
        self._round_robin_index = 0
        if not api_keys:
            logger.error("GeminiClient initialized with NO API keys — every request will fail")
        else:
            logger.info("GeminiClient initialized", key_count=len(self.keys))

    def _get_next_available_key(self) -> Optional[GeminiKey]:
        for _ in range(len(self.keys)):
            key = self.keys[self._round_robin_index % len(self.keys)]
            self._round_robin_index = (self._round_robin_index + 1) % len(self.keys)
            if key.is_available():
                return key
        return None

    def _model_for_retry(self, retry_count: int) -> str:
        models = [settings.gemini_model]
        for m in GEMINI_MODEL_FALLBACKS:
            if m not in models:
                models.append(m)
        return models[min(retry_count, len(models) - 1)]

    def _extract_text(self, response) -> str:
        if response is None:
            return ""
        if hasattr(response, "text") and response.text is not None:
            return response.text
        return str(response)

    def _extract_usage(self, response) -> dict:
        """Extract token usage from a Gemini response."""
        if response is None:
            return {"model": None, "inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        meta = getattr(response, "usage_metadata", None)
        model = getattr(response, "model", None)
        input_tokens = getattr(meta, "prompt_token_count", None) or 0
        output_tokens = getattr(meta, "candidates_token_count", None) or 0
        total_tokens = getattr(meta, "total_token_count", None) or 0
        if not total_tokens:
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        return {
            "model": model,
            "inputTokens": int(input_tokens or 0),
            "outputTokens": int(output_tokens or 0),
            "totalTokens": int(total_tokens or 0),
        }

    def _record_usage(self, response, model: Optional[str] = None) -> None:
        entries = _usage_accumulator.get()
        if entries is None:
            return
        usage = self._extract_usage(response)
        if not usage.get("model") and model:
            usage["model"] = model
        if usage["totalTokens"] > 0:
            entries.append(usage)

    async def _stream_to_async(self, sync_gen, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        try:
            for chunk in sync_gen:
                if hasattr(chunk, "text") and chunk.text is not None:
                    yield chunk.text
                self._record_usage(chunk, model)
        except Exception as e:
            logger.error("Stream error during iteration", error=str(e))
            raise

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        stream: bool = False,
        _retry_count: int = 0,
    ):
        if _retry_count > self.MAX_RETRIES:
            raise RuntimeError("Max retries exceeded for Gemini API call")

        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = self._model_for_retry(_retry_count)
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        try:
            if stream:
                sync_gen = key.client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                key.mark_success()
                return self._stream_to_async(sync_gen, model=model)
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            self._record_usage(response, model)
            return response
        except Exception as e:
            logger.warning("Gemini API call failed", error=str(e), key_suffix=key.api_key[-4:])
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=stream,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        temperature: float = 0.7,
        _retry_count: int = 0,
    ):
        if _retry_count > self.MAX_RETRIES:
            raise RuntimeError("Max retries exceeded for Gemini tool call")

        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = self._model_for_retry(_retry_count)
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            tools=[genai_types.Tool(function_declarations=tools)],
        )

        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            self._record_usage(response, model)
            return response
        except Exception as e:
            logger.warning("Gemini tool call failed", error=str(e), key_suffix=key.api_key[-4:])
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_tools(
                system_prompt=system_prompt,
                user_message=user_message,
                tools=tools,
                temperature=temperature,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_image(
        self,
        system_prompt: str,
        user_message: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        _retry_count: int = 0,
    ):
        if _retry_count > self.MAX_RETRIES:
            raise RuntimeError("Max retries exceeded for Gemini vision call")

        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = self._model_for_retry(_retry_count)
        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(text=user_message),
                    genai_types.Part(
                        inline_data=genai_types.Blob(mime_type=mime_type, data=image_bytes)
                    ),
                ],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        )

        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            self._record_usage(response, model)
            return response
        except Exception as e:
            logger.warning("Gemini vision call failed", error=str(e), key_suffix=key.api_key[-4:])
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_image(
                system_prompt=system_prompt,
                user_message=user_message,
                image_bytes=image_bytes,
                mime_type=mime_type,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_audio(
        self,
        system_prompt: str,
        audio_bytes: bytes,
        mime_type: str = "audio/mpeg",
        _retry_count: int = 0,
    ):
        if _retry_count > self.MAX_RETRIES:
            raise RuntimeError("Max retries exceeded for Gemini audio call")

        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = self._model_for_retry(_retry_count)
        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(
                        inline_data=genai_types.Blob(mime_type=mime_type, data=audio_bytes)
                    ),
                    genai_types.Part(text="Process this audio and respond appropriately."),
                ],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
        )

        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            self._record_usage(response, model)
            return response
        except Exception as e:
            logger.warning("Gemini audio call failed", error=str(e), key_suffix=key.api_key[-4:])
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_audio(
                system_prompt=system_prompt,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                _retry_count=_retry_count + 1,
            )

    async def generate_speech(
        self,
        text: str,
        voice_name: str = "Zephyr",
        _retry_count: int = 0,
    ) -> Optional[dict]:
        if not text:
            return None
        if _retry_count > 2:
            raise RuntimeError("Gemini TTS unavailable after retries")

        voice = voice_name or settings.tts_voice
        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = "gemini-3.1-flash-tts-preview"
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=text[:500])],
            )
        ]
        config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            self._record_usage(response, model)
            parts = response.candidates[0].content.parts
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    return {
                        "audio_bytes": inline.data,
                        "mime": inline.mime_type or "audio/l16",
                    }
            return None
        except Exception as e:
            code = getattr(e, "code", None)
            logger.warning(
                "Gemini TTS call failed",
                error=str(e),
                key_suffix=key.api_key[-4:],
                code=code,
            )
            if code != 503:
                key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_speech(
                text,
                voice_name=voice_name,
                _retry_count=_retry_count + 1,
            )

    def get_key_statuses(self) -> list[dict]:
        return [
            {
                "key_suffix": k.api_key[-4:],
                "status": k.status.value,
                "fail_count": k.fail_count,
            }
            for k in self.keys
        ]
