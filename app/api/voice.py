import base64
import logging
import re
import struct
from io import BytesIO
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from gtts import gTTS

from app.core.guardrails import check_input, check_output
from app.core.auth import allow_access
from app.core.llm_client import begin_usage_tracking, consume_usage

logger = logging.getLogger("app.api.voice")

router = APIRouter()


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(pcm),
    )
    return header + pcm


def gtts_audio(text: str) -> Optional[str]:
    try:
        text = text[:200]
        lang = "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"
        mp3 = BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(mp3)
        audio_b64 = base64.b64encode(mp3.getvalue()).decode("utf-8")
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception:
        return None


async def synthesize_speech(text: str, llm_client) -> Optional[str]:
    if not text:
        return None
    try:
        tts = await llm_client.generate_speech(text)
        if tts:
            audio = tts["audio_bytes"]
            mime = tts.get("mime") or "audio/l16"
            if mime.startswith("audio/l16"):
                m = re.match(r"audio/l16;\s*rate=(\d+);\s*channels=(\d+)", mime)
                sample_rate = int(m.group(1)) if m else 24000
                channels = int(m.group(2)) if m else 1
                wav = pcm_to_wav(audio, sample_rate, channels, 16)
                return f"data:audio/wav;base64,{base64.b64encode(wav).decode('utf-8')}"
            return f"data:{mime};base64,{base64.b64encode(audio).decode('utf-8')}"
    except Exception as e:
        logger.warning("Gemini TTS failed, falling back to gTTS", error=str(e))
    return gtts_audio(text)


class VoiceResponse(BaseModel):
    text_response: str
    audio_response: Optional[str] = None
    conversation_id: Optional[str] = None
    usage: Optional[dict] = None
    model: Optional[str] = None


@router.post("", response_model=VoiceResponse)
async def voice_endpoint(
    audio: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    conversation_id: Optional[str] = Form(None),
    user: dict = Depends(allow_access),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio data received")

    from app.main import llm_client

    if not llm_client:
        raise HTTPException(status_code=503, detail="AI service not initialized")

    system_prompt = (
        "You are Rihla, a helpful Egyptian tour assistant. "
        "The user has sent you an audio message. Listen to it and respond appropriately. "
        "If they ask about tourism in Egypt, provide helpful information. "
        "Be concise and friendly."
    )

    mime_type = audio.content_type or "audio/mpeg"

    try:
        begin_usage_tracking()
        response = await llm_client.generate_with_audio(
            system_prompt=system_prompt,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
        )

        text = ""
        if response is not None and hasattr(response, "text") and response.text is not None:
            text = response.text

        guard_result = check_output(text)
        if guard_result.requires_regeneration:
            text = "I understand your concern, but let me help you with tourism information about Egypt instead."

        usage_entries = consume_usage()
        usage = None
        model = None
        if usage_entries:
            entry = usage_entries[0]
            usage = {
                "inputTokens": entry.get("inputTokens", 0),
                "outputTokens": entry.get("outputTokens", 0),
                "totalTokens": entry.get("totalTokens", 0),
            }
            model = entry.get("model")

        audio_response = None
        if text:
            audio_response = await synthesize_speech(text, llm_client)

        return VoiceResponse(
            text_response=text,
            audio_response=audio_response,
            conversation_id=conversation_id,
            usage=usage,
            model=model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
