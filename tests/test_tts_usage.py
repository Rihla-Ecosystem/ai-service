import asyncio
import io
import pytest
from unittest.mock import MagicMock, patch

from app.core.llm_client import GeminiClient
from app.core.usage import (
    ATTEMPT_OUTCOME_FAILED,
    ATTEMPT_OUTCOME_INDETERMINATE,
    ATTEMPT_OUTCOME_SUCCEEDED,
    ERROR_CATEGORY_LOCAL_PROCESSING,
    OP_AUDIO_UNDERSTANDING,
    OP_TEXT_TO_SPEECH,
    PROVIDER_GOOGLE,
    USAGE_COMPLETENESS_COMPLETE,
    USAGE_COMPLETENESS_UNAVAILABLE,
    begin_usage_tracking,
    consume_usage_and_attempts,
)


def make_mock_tts_response(
    model_version="gemini-3.1-flash-tts-preview",
    prompt_tokens=12,
    candidates_tokens=250,
    total_tokens=262,
    audio_output_tokens=250,
    has_usage=True,
    audio_bytes=b"fake_pcm_audio_data_1234567890",
):
    mock_resp = MagicMock()
    mock_resp.model_version = model_version

    if has_usage:
        meta = MagicMock()
        meta.prompt_token_count = prompt_tokens
        meta.candidates_token_count = candidates_tokens
        meta.total_token_count = total_tokens
        meta.cached_content_token_count = None
        meta.thoughts_token_count = None
        meta.prompt_tokens_details = []

        detail = MagicMock()
        detail.modality = "AUDIO"
        detail.token_count = audio_output_tokens
        meta.candidates_tokens_details = [detail]

        mock_resp.usage_metadata = meta
    else:
        mock_resp.usage_metadata = None

    if audio_bytes is not None:
        part = MagicMock()
        inline = MagicMock()
        inline.data = audio_bytes
        inline.mime_type = "audio/l16; rate=24000; channels=1"
        part.inline_data = inline

        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        mock_resp.candidates = [candidate]
    else:
        mock_resp.candidates = []

    return mock_resp


def test_successful_gemini_tts_provider_call_telemetry():
    async def _test():
        client = GeminiClient(api_keys=["test-key-12345"])
        mock_sdk_client = MagicMock()
        mock_response = make_mock_tts_response(
            model_version="gemini-3.1-flash-tts-preview",
            prompt_tokens=12,
            candidates_tokens=250,
            total_tokens=262,
            audio_output_tokens=250,
            has_usage=True,
        )
        mock_sdk_client.models.generate_content = MagicMock(return_value=mock_response)
        client.keys[0].client = mock_sdk_client

        begin_usage_tracking()
        result = await client.generate_speech("Hello world, this is a test speech synthesis.")
        assert result is not None
        assert result["audio_bytes"] == b"fake_pcm_audio_data_1234567890"

        calls, attempts = consume_usage_and_attempts()

        assert len(calls) == 1
        call = calls[0]
        assert call["provider"] == PROVIDER_GOOGLE
        assert call["requestedModel"] == "gemini-3.1-flash-tts-preview"
        assert call["actualModel"] == "gemini-3.1-flash-tts-preview"
        assert call["operation"] == OP_TEXT_TO_SPEECH
        assert call["inputTokens"] == 12
        assert call["outputTokens"] == 250
        assert call["totalTokens"] == 262
        assert call["audioOutputTokens"] == 250
        assert call["usageCompleteness"] == USAGE_COMPLETENESS_COMPLETE

        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt["provider"] == PROVIDER_GOOGLE
        assert attempt["operation"] == OP_TEXT_TO_SPEECH
        assert attempt["outcome"] == ATTEMPT_OUTCOME_SUCCEEDED
        assert attempt["providerResponseReceived"] is True
        assert attempt["providerCallId"] == call["providerCallId"]

    asyncio.run(_test())


def test_voice_endpoint_end_to_end_combines_audio_understanding_and_tts_calls():
    async def _test():
        from app.api.voice import voice_endpoint
        from fastapi import UploadFile

        mock_audio_resp = MagicMock()
        mock_audio_resp.text = "Tell me about the Pyramids of Giza."
        mock_audio_resp.model_version = "gemini-2.5-flash"
        mock_audio_resp.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=30,
            total_token_count=130,
            cached_content_token_count=None,
            thoughts_token_count=None,
            prompt_tokens_details=[],
            candidates_tokens_details=[],
        )

        mock_tts_resp = make_mock_tts_response(
            model_version="gemini-3.1-flash-tts-preview",
            prompt_tokens=15,
            candidates_tokens=180,
            total_tokens=195,
            audio_output_tokens=180,
        )

        fake_audio_file = UploadFile(filename="input.mp3", file=io.BytesIO(b"fake_audio_input_bytes"))
        fake_request = MagicMock()

        llm_client = GeminiClient(api_keys=["test-key-12345"])
        mock_sdk_client = MagicMock()
        mock_sdk_client.models.generate_content = MagicMock(side_effect=[mock_audio_resp, mock_tts_resp])
        llm_client.keys[0].client = mock_sdk_client

        with patch("app.main.llm_client", llm_client), \
             patch("app.api.voice.enforce_rate_limit"):
            response = await voice_endpoint(
                request=fake_request,
                audio=fake_audio_file,
                lat=29.9792,
                lon=31.1342,
                conversation_id="conv-123",
                user={"sub": "user1"},
            )

        assert response.text_response == "Tell me about the Pyramids of Giza."
        assert response.providerCalls is not None
        assert len(response.providerCalls) == 2

        ops = [c["operation"] for c in response.providerCalls]
        assert OP_AUDIO_UNDERSTANDING in ops
        assert OP_TEXT_TO_SPEECH in ops

        tts_calls = [c for c in response.providerCalls if c["operation"] == OP_TEXT_TO_SPEECH]
        assert len(tts_calls) == 1
        tts_call = tts_calls[0]
        assert tts_call["requestedModel"] == "gemini-3.1-flash-tts-preview"
        assert tts_call["actualModel"] == "gemini-3.1-flash-tts-preview"
        assert tts_call["inputTokens"] == 15
        assert tts_call["outputTokens"] == 180
        assert tts_call["totalTokens"] == 195
        assert tts_call["audioOutputTokens"] == 180

    asyncio.run(_test())


def test_tts_missing_usage_metadata_records_unavailable():
    async def _test():
        client = GeminiClient(api_keys=["test-key-12345"])
        mock_sdk_client = MagicMock()
        mock_response = make_mock_tts_response(has_usage=False)
        mock_sdk_client.models.generate_content = MagicMock(return_value=mock_response)
        client.keys[0].client = mock_sdk_client

        begin_usage_tracking()
        result = await client.generate_speech("Testing missing usage metadata.")
        assert result is not None

        calls, _ = consume_usage_and_attempts()
        assert len(calls) == 1
        call = calls[0]
        assert call["provider"] == PROVIDER_GOOGLE
        assert call["requestedModel"] == "gemini-3.1-flash-tts-preview"
        assert call["actualModel"] == "gemini-3.1-flash-tts-preview"
        assert call["operation"] == OP_TEXT_TO_SPEECH
        assert "inputTokens" not in call
        assert "outputTokens" not in call
        assert "audioOutputTokens" not in call
        assert "totalTokens" not in call
        assert call["usageCompleteness"] == USAGE_COMPLETENESS_UNAVAILABLE

    asyncio.run(_test())


def test_gtts_fallback_preserves_recorded_gemini_usage():
    async def _test():
        client = GeminiClient(api_keys=["test-key-12345"])
        mock_sdk_client = MagicMock()

        # Gemini responds with usage, but no inline audio candidates (local processing failure)
        mock_response = make_mock_tts_response(
            model_version="gemini-3.1-flash-tts-preview",
            prompt_tokens=10,
            candidates_tokens=100,
            total_tokens=110,
            audio_output_tokens=100,
            audio_bytes=None,
        )
        mock_sdk_client.models.generate_content = MagicMock(return_value=mock_response)
        client.keys[0].client = mock_sdk_client

        begin_usage_tracking()
        with patch("app.api.voice.gtts_audio_bytes", return_value=(b"gtts_audio_bytes", "audio/mpeg")):
            from app.api.voice import synthesize_speech
            result = await synthesize_speech("Fallback test text", client)

        assert result is not None
        assert result[0] == b"gtts_audio_bytes"
        assert result[1] == "audio/mpeg"

        calls, attempts = consume_usage_and_attempts()
        # The Gemini provider call recorded when response returned MUST NOT be deleted
        assert len(calls) == 1
        call = calls[0]
        assert call["provider"] == PROVIDER_GOOGLE
        assert call["operation"] == OP_TEXT_TO_SPEECH
        assert call["inputTokens"] == 10
        assert call["outputTokens"] == 100

        # Attempt should record INDETERMINATE with LOCAL_PROCESSING
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt["outcome"] == ATTEMPT_OUTCOME_INDETERMINATE
        assert attempt["providerResponseReceived"] is True
        assert attempt["errorCategory"] == ERROR_CATEGORY_LOCAL_PROCESSING

    asyncio.run(_test())


def test_gtts_fallback_when_gemini_fails_before_response():
    async def _test():
        client = GeminiClient(api_keys=["test-key-12345"])
        client.MAX_RETRIES = 0
        mock_sdk_client = MagicMock()
        mock_sdk_client.models.generate_content = MagicMock(side_effect=RuntimeError("SDK Connection Failed"))
        client.keys[0].client = mock_sdk_client

        begin_usage_tracking()
        with patch("app.api.voice.gtts_audio_bytes", return_value=(b"gtts_audio_bytes", "audio/mpeg")):
            from app.api.voice import synthesize_speech
            result = await synthesize_speech("Fallback failure test text", client)

        assert result is not None
        assert result[0] == b"gtts_audio_bytes"

        calls, attempts = consume_usage_and_attempts()
        # No providerCall was made / received
        assert len(calls) == 0
        # Attempt recorded as INDETERMINATE/FAILED without providerResponseReceived
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt["providerResponseReceived"] is False

    asyncio.run(_test())
