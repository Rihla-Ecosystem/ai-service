import asyncio
import datetime
import os
import structlog
from enum import Enum
from typing import AsyncGenerator, List, Optional
from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.core.gemini_usage import extract_response_model, extract_token_counts
from app.core.usage import (
    ATTEMPT_OUTCOME_FAILED,
    ATTEMPT_OUTCOME_INDETERMINATE,
    ATTEMPT_OUTCOME_SUCCEEDED,
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_CONNECTION_ERROR,
    ERROR_CATEGORY_INVALID_REQUEST,
    ERROR_CATEGORY_LOCAL_PROCESSING,
    ERROR_CATEGORY_RATE_LIMIT,
    ERROR_CATEGORY_SERVER_ERROR,
    ERROR_CATEGORY_TIMEOUT,
    ERROR_CATEGORY_UNSUPPORTED_OPERATION,
    ERROR_CATEGORY_UNKNOWN,
    OP_AUDIO_UNDERSTANDING,
    OP_IMAGE_ANALYSIS,
    OP_TEXT_CHAT,
    OP_TEXT_CHAT_STREAM,
    OP_TEXT_GENERATION,
    OP_TEXT_TO_SPEECH,
    PROVIDER_GOOGLE,
    USAGE_COMPLETENESS_COMPLETE,
    USAGE_COMPLETENESS_UNAVAILABLE,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    USAGE_SOURCE_STREAM_FINAL,
    final_stream_usage,
    make_provider_attempt,
    make_provider_call,
    record_provider_attempt,
    record_provider_call,
)

logger = structlog.get_logger()

GEMINI_MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]

def safety_settings() -> list:
    """Standard harmful-content blocking for every Gemini generation call."""
    categories = [
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        genai_types.HarmCategory.HARM_CATEGORY_JAILBREAK,
    ]
    return [
        genai_types.SafetySetting(
            category=c,
            threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        )
        for c in categories
    ]


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
        # Process-env override (default keeps historical behavior).
        env_max_retries = os.environ.get("GEMINI_MAX_RETRIES")
        if env_max_retries is not None:
            try:
                parsed = int(env_max_retries)
                self.MAX_RETRIES = max(0, parsed)
            except ValueError:
                logger.warning(
                    "GEMINI_MAX_RETRIES is not an integer; ignoring override",
                    value=env_max_retries,
                )
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

    @staticmethod
    def _now_iso() -> str:
        """Provider-neutral UTC timestamp for `providerCallStartedAt`."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _classify_error(e: Exception) -> tuple:
        """Map an SDK exception to (outcome, errorCategory, httpStatus).

        Conservative billing-safety policy: a failure is classified FAILED only
        when there is reliable evidence the provider operation definitively did
        NOT execute (a confirmed request/auth/unsupported rejection). Every
        uncertain failure — timeouts, connection drops, HTTP 5xx server errors,
        HTTP 429 rate limits, and unknown exceptions after the call started — is
        INDETERMINATE because the provider call MAY have executed and incurred
        cost. The presence of an HTTP status never by itself forces FAILED.
        Values are provider-neutral; no native identifiers, messages, or stack
        traces are surfaced.
        """
        code = getattr(e, "code", None)
        status = code if isinstance(code, int) else None
        message = str(e).lower()

        # Timeouts / deadlines (with or without a numeric 408 status).
        if (
            status == 408
            or "timeout" in message
            or "deadline" in message
            or "timed out" in message
        ):
            return ATTEMPT_OUTCOME_INDETERMINATE, ERROR_CATEGORY_TIMEOUT, status
        # HTTP 5xx provider/server errors: the request may have executed.
        if status is not None and status >= 500:
            return ATTEMPT_OUTCOME_INDETERMINATE, ERROR_CATEGORY_SERVER_ERROR, status
        # HTTP 429 / rate limit: INDETERMINATE by default (the request may have
        # been accepted before the throttle).
        if status == 429:
            return ATTEMPT_OUTCOME_INDETERMINATE, ERROR_CATEGORY_RATE_LIMIT, status
        # Confirmed authentication/authorization rejection before execution.
        if status in (401, 403):
            return ATTEMPT_OUTCOME_FAILED, ERROR_CATEGORY_AUTH_ERROR, status
        # Confirmed unsupported operation/model rejection before execution.
        if status == 404:
            return ATTEMPT_OUTCOME_FAILED, ERROR_CATEGORY_UNSUPPORTED_OPERATION, status
        # Confirmed invalid-request rejection before execution.
        if status is not None and 400 <= status < 500:
            return ATTEMPT_OUTCOME_FAILED, ERROR_CATEGORY_INVALID_REQUEST, status
        # Connection drops / resets / transport failures (no numeric status).
        if any(
            keyword in message
            for keyword in (
                "connection",
                "reset",
                "refused",
                "broken pipe",
                "transport",
                "network unreachable",
                "remote host",
            )
        ):
            return ATTEMPT_OUTCOME_INDETERMINATE, ERROR_CATEGORY_CONNECTION_ERROR, status
        # Unknown exception after the provider call started.
        return ATTEMPT_OUTCOME_INDETERMINATE, ERROR_CATEGORY_UNKNOWN, status

    def _record_attempt(
        self,
        *,
        operation: str,
        requested_model: Optional[str],
        actual_model: Optional[str],
        attempt_number: int,
        outcome: str,
        provider_call_started_at: str,
        provider_response_received: bool,
        provider_call_id: Optional[str] = None,
        error_category: Optional[str] = None,
        http_status: Optional[int] = None,
    ) -> None:
        """Record one diagnostic ProviderAttempt; no content, never priced.

        ``_record_attempt`` is only ever invoked from inside a provider try
        block, i.e. after a real provider SDK call began, so
        ``providerCallStarted`` is always ``True`` and
        ``provider_call_started_at`` carries the recorded start time.
        Pre-provider local validation/preparation failures never reach here.
        """
        attempt = make_provider_attempt(
            provider=PROVIDER_GOOGLE,
            operation=operation,
            requested_model=requested_model,
            actual_model=actual_model,
            attempt_number=attempt_number,
            outcome=outcome,
            provider_call_started=True,
            provider_call_started_at=provider_call_started_at,
            provider_response_received=provider_response_received,
            provider_call_id=provider_call_id,
            error_category=error_category,
            http_status=http_status,
        )
        record_provider_attempt(attempt)

    def _extract_text(self, response) -> str:
        if response is None:
            return ""
        if hasattr(response, "text") and response.text is not None:
            return response.text
        return str(response)

    def _record_provider_call(
        self,
        response,
        requested_model: Optional[str],
        operation: str,
        usage_source: str = USAGE_SOURCE_PROVIDER_RESPONSE,
        accounting_semantics: Optional[str] = None,
    ) -> Optional[str]:
        """Record one ProviderCallUsage entry for a real provider call.

        The entry always represents the provider call that executed. Token
        counts are only included when the provider reported them; otherwise the
        record is emitted with no usage fields and
        ``usageCompleteness=UNAVAILABLE``. ``providerRequestId`` is left absent
        because the current Gemini SDK path does not expose a request id.
        Returns the assigned ``providerCallId`` for attempt linkage.
        """
        counts = extract_token_counts(response)
        actual_model = extract_response_model(response)
        call = make_provider_call(
            provider=PROVIDER_GOOGLE,
            requested_model=requested_model,
            actual_model=actual_model,
            operation=operation,
            provider_call_made=True,
            usage_source=usage_source,
            usage_completeness=(
                USAGE_COMPLETENESS_COMPLETE if counts else USAGE_COMPLETENESS_UNAVAILABLE
            ),
            accounting_semantics=accounting_semantics,
            **counts,
        )
        return record_provider_call(call)

    def _record_stream_final(
        self,
        usage_fields: dict,
        actual_model: Optional[str],
        requested_model: Optional[str],
        operation: str,
    ) -> Optional[str]:
        """Record exactly one ProviderCallUsage entry for a streamed call.

        ``usage_fields`` must already be the final cumulative snapshot (the last
        non-empty snapshot observed across chunks). This guarantees one entry
        per streamed provider call and never sums cumulative snapshots.
        Returns the assigned ``providerCallId`` for attempt linkage.
        """
        call = make_provider_call(
            provider=PROVIDER_GOOGLE,
            requested_model=requested_model,
            actual_model=actual_model,
            operation=operation,
            provider_call_made=True,
            usage_source=USAGE_SOURCE_STREAM_FINAL,
            usage_completeness=(
                USAGE_COMPLETENESS_COMPLETE if usage_fields else USAGE_COMPLETENESS_UNAVAILABLE
            ),
            **usage_fields,
        )
        return record_provider_call(call)

    async def _stream_to_async(
        self,
        sync_gen,
        requested_model: Optional[str] = None,
        operation: str = OP_TEXT_CHAT_STREAM,
        attempt_number: int = 1,
        provider_call_started_at: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        snapshots = []
        last_model = None
        outcome = ATTEMPT_OUTCOME_SUCCEEDED
        error_category = None
        http_status = None
        try:
            for chunk in sync_gen:
                if hasattr(chunk, "text") and chunk.text is not None:
                    yield chunk.text
                snapshot = extract_token_counts(chunk)
                if snapshot:
                    snapshots.append(snapshot)
                model = extract_response_model(chunk)
                if model:
                    last_model = model
        except Exception as e:
            # A mid-stream failure may have executed provider work; treat as
            # indeterminate unless the provider definitively rejected it.
            outcome, error_category, http_status = self._classify_error(e)
            raise
        finally:
            last_usage = final_stream_usage(snapshots)
            call_id = self._record_stream_final(last_usage, last_model, requested_model, operation)
            started_at = provider_call_started_at or self._now_iso()
            received = last_model is not None or bool(last_usage) or outcome == ATTEMPT_OUTCOME_SUCCEEDED
            self._record_attempt(
                operation=operation,
                requested_model=requested_model,
                actual_model=last_model,
                attempt_number=attempt_number,
                outcome=outcome,
                provider_call_started_at=started_at,
                provider_response_received=received,
                provider_call_id=call_id,
                error_category=error_category,
                http_status=http_status,
            )

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        stream: bool = False,
        operation: str = OP_TEXT_GENERATION,
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
            safety_settings=safety_settings(),
        )

        attempt_number = _retry_count + 1
        started = self._now_iso()
        try:
            if stream:
                sync_gen = key.client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
                key.mark_success()
                return self._stream_to_async(
                    sync_gen,
                    requested_model=model,
                    operation=OP_TEXT_CHAT_STREAM,
                    attempt_number=attempt_number,
                    provider_call_started_at=started,
                )
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            call_id = self._record_provider_call(response, requested_model=model, operation=operation)
            actual_model = extract_response_model(response)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started_at=started,
                provider_response_received=True,
                provider_call_id=call_id,
            )
            return response
        except Exception as e:
            logger.warning("Gemini API call failed", error=str(e), key_suffix=key.api_key[-4:])
            outcome, error_category, http_status = self._classify_error(e)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=None,
                attempt_number=attempt_number,
                outcome=outcome,
                provider_call_started_at=started,
                provider_response_received=False,
                error_category=error_category,
                http_status=http_status,
            )
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=stream,
                operation=operation,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        temperature: float = 0.7,
        operation: str = OP_TEXT_CHAT,
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
            safety_settings=safety_settings(),
        )

        attempt_number = _retry_count + 1
        started = self._now_iso()
        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            call_id = self._record_provider_call(response, requested_model=model, operation=operation)
            actual_model = extract_response_model(response)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started_at=started,
                provider_response_received=True,
                provider_call_id=call_id,
            )
            return response
        except Exception as e:
            logger.warning("Gemini tool call failed", error=str(e), key_suffix=key.api_key[-4:])
            outcome, error_category, http_status = self._classify_error(e)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=None,
                attempt_number=attempt_number,
                outcome=outcome,
                provider_call_started_at=started,
                provider_response_received=False,
                error_category=error_category,
                http_status=http_status,
            )
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_tools(
                system_prompt=system_prompt,
                user_message=user_message,
                tools=tools,
                temperature=temperature,
                operation=operation,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_image(
        self,
        system_prompt: str,
        user_message: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        operation: str = OP_IMAGE_ANALYSIS,
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
            safety_settings=safety_settings(),
        )

        attempt_number = _retry_count + 1
        started = self._now_iso()
        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            call_id = self._record_provider_call(response, requested_model=model, operation=operation)
            actual_model = extract_response_model(response)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started_at=started,
                provider_response_received=True,
                provider_call_id=call_id,
            )
            return response
        except Exception as e:
            logger.warning("Gemini vision call failed", error=str(e), key_suffix=key.api_key[-4:])
            outcome, error_category, http_status = self._classify_error(e)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=None,
                attempt_number=attempt_number,
                outcome=outcome,
                provider_call_started_at=started,
                provider_response_received=False,
                error_category=error_category,
                http_status=http_status,
            )
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_image(
                system_prompt=system_prompt,
                user_message=user_message,
                image_bytes=image_bytes,
                mime_type=mime_type,
                operation=operation,
                _retry_count=_retry_count + 1,
            )

    async def generate_with_audio(
        self,
        system_prompt: str,
        audio_bytes: bytes,
        mime_type: str = "audio/mpeg",
        extra_user_context: str = "",
        operation: str = OP_AUDIO_UNDERSTANDING,
        _retry_count: int = 0,
    ):
        if _retry_count > self.MAX_RETRIES:
            raise RuntimeError("Max retries exceeded for Gemini audio call")

        key = self._get_next_available_key()
        if not key:
            raise RuntimeError("All API keys are degraded or in cooldown")

        model = self._model_for_retry(_retry_count)
        user_text = "Process this audio and respond appropriately."
        if extra_user_context:
            user_text += "\n" + extra_user_context
        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(
                        inline_data=genai_types.Blob(mime_type=mime_type, data=audio_bytes)
                    ),
                    genai_types.Part(text=user_text),
                ],
            )
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            safety_settings=safety_settings(),
        )

        attempt_number = _retry_count + 1
        started = self._now_iso()
        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            call_id = self._record_provider_call(response, requested_model=model, operation=operation)
            actual_model = extract_response_model(response)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started_at=started,
                provider_response_received=True,
                provider_call_id=call_id,
            )
            return response
        except Exception as e:
            logger.warning("Gemini audio call failed", error=str(e), key_suffix=key.api_key[-4:])
            outcome, error_category, http_status = self._classify_error(e)
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=None,
                attempt_number=attempt_number,
                outcome=outcome,
                provider_call_started_at=started,
                provider_response_received=False,
                error_category=error_category,
                http_status=http_status,
            )
            key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_with_audio(
                system_prompt=system_prompt,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                extra_user_context=extra_user_context,
                operation=operation,
                _retry_count=_retry_count + 1,
            )

    async def generate_speech(
        self,
        text: str,
        voice_name: str = "Zephyr",
        operation: str = OP_TEXT_TO_SPEECH,
        _retry_count: int = 0,
    ) -> Optional[dict]:
        if not text:
            return None
        if _retry_count > self.MAX_RETRIES:
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
            safety_settings=safety_settings(),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        attempt_number = _retry_count + 1
        started = self._now_iso()
        response = None
        call_id = None
        try:
            response = key.client.models.generate_content(
                model=model, contents=contents, config=config
            )
            key.mark_success()
            call_id = self._record_provider_call(response, requested_model=model, operation=operation)
            actual_model = extract_response_model(response)
            parts = response.candidates[0].content.parts
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    self._record_attempt(
                        operation=operation,
                        requested_model=model,
                        actual_model=actual_model,
                        attempt_number=attempt_number,
                        outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                        provider_call_started_at=started,
                        provider_response_received=True,
                        provider_call_id=call_id,
                    )
                    return {
                        "audio_bytes": inline.data,
                        "mime": inline.mime_type or "audio/l16",
                    }
            self._record_attempt(
                operation=operation,
                requested_model=model,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started_at=started,
                provider_response_received=True,
                provider_call_id=call_id,
            )
            return None
        except Exception as e:
            code = getattr(e, "code", None)
            logger.warning(
                "Gemini TTS call failed",
                error=str(e),
                key_suffix=key.api_key[-4:],
                code=code,
            )
            if response is not None:
                # The provider responded but local processing failed (e.g. no
                # usable audio part); the call may have executed. Record as
                # indeterminate with providerResponseReceived=true.
                self._record_attempt(
                    operation=operation,
                    requested_model=model,
                    actual_model=extract_response_model(response),
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started_at=started,
                    provider_response_received=True,
                    provider_call_id=call_id,
                    error_category=ERROR_CATEGORY_LOCAL_PROCESSING,
                )
            else:
                outcome, error_category, http_status = self._classify_error(e)
                self._record_attempt(
                    operation=operation,
                    requested_model=model,
                    actual_model=None,
                    attempt_number=attempt_number,
                    outcome=outcome,
                    provider_call_started_at=started,
                    provider_response_received=False,
                    error_category=error_category,
                    http_status=http_status,
                )
            if code != 503:
                key.mark_failed(cooldown_seconds=self.cooldown_seconds)
            return await self.generate_speech(
                text,
                voice_name=voice_name,
                operation=operation,
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
