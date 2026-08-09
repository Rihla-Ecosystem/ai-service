"""Local media-boundary tests for the billed Voice path; no provider calls."""

from io import BytesIO
import wave

import pytest

from app.api.voice import (
    TtsGenerationError,
    VoiceMediaValidationError,
    synthesize_speech,
    validate_voice_media,
)


def _wav_bytes(duration_seconds: int) -> bytes:
    sample_rate = 8_000
    with BytesIO() as out:
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * sample_rate * duration_seconds)
        return out.getvalue()


def test_valid_wav_at_sixty_seconds_is_accepted():
    assert validate_voice_media(_wav_bytes(60), "audio/wav") == pytest.approx(60.0)


def test_wav_above_sixty_seconds_is_rejected_before_provider_dispatch():
    with pytest.raises(VoiceMediaValidationError, match="duration exceeds"):
        validate_voice_media(_wav_bytes(61), "audio/wav")


def test_spoofed_or_malformed_audio_is_rejected():
    with pytest.raises(VoiceMediaValidationError):
        validate_voice_media(_wav_bytes(1), "audio/mpeg")
    with pytest.raises(VoiceMediaValidationError):
        validate_voice_media(b"not media", "audio/wav")


def test_gemini_tts_failure_does_not_fall_back_to_gtts():
    class FailingGemini:
        async def generate_speech(self, _text):
            raise RuntimeError("Gemini unavailable")

    async def run():
        with pytest.raises(TtsGenerationError):
            await synthesize_speech("hello", FailingGemini())

    import asyncio
    asyncio.run(run())


def test_oversized_pcm_tts_response_is_rejected():
    class OversizedGemini:
        async def generate_speech(self, _text):
            # 31 seconds of 24kHz mono 16-bit raw PCM.
            return {
                "audio_bytes": b"\x00\x00" * 24_000 * 31,
                "mime": "audio/l16; rate=24000; channels=1",
            }

    async def run():
        with pytest.raises(TtsGenerationError, match="Gemini TTS generation failed"):
            await synthesize_speech("hello", OversizedGemini())

    import asyncio
    asyncio.run(run())
