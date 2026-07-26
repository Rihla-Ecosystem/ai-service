import asyncio
import structlog
from enum import Enum
from typing import AsyncGenerator, List, Optional
from google import genai
from google.genai import types as genai_types

logger = structlog.get_logger()


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
        self.client = genai.Client(api_key=api_key)

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

    def _extract_text(self, response) -> str:
        if response is None:
            return ""
        if hasattr(response, "text") and response.text is not None:
            return response.text
        return str(response)

    async def _stream_to_async(self, sync_gen) -> AsyncGenerator[str, None]:
        try:
            for chunk in sync_gen:
                if hasattr(chunk, "text") and chunk.text is not None:
                    yield chunk.text
        except Exception as e:
            logger.error("Stream error during iteration", error=str(e))
            yield ""

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

        model = "gemini-2.0-flash"
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
                return self._stream_to_async(sync_gen)
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
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

        model = "gemini-2.0-flash"
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            tools=tools,
        )

        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
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

        model = "gemini-2.0-flash"
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

        model = "gemini-2.0-flash"
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

    def get_key_statuses(self) -> list[dict]:
        return [
            {
                "key_suffix": k.api_key[-4:],
                "status": k.status.value,
                "fail_count": k.fail_count,
            }
            for k in self.keys
        ]
