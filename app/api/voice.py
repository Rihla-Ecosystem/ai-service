import base64
import logging
import re
import secrets
import struct
from io import BytesIO
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gtts import gTTS

from app.config import settings
from app.core.guardrails import check_input, check_output
from app.core.auth import allow_access
from app.core.ratelimit import rate_limit
from app.core.usage import begin_usage_tracking, consume_usage, derive_legacy_usage

logger = logging.getLogger("app.api.voice")

router = APIRouter()

# Short-lived cache for generated speech served through /audio (streamed playback).
_AUDIO_CACHE: Dict[str, Dict] = {}
_AUDIO_CACHE_MAX = 50

_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # regional indicators
    "\U0001F300-\U0001F5FF"  # misc symbols and pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport and map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining enclosing keycap
    "]"
)


def strip_for_speech(text: str) -> str:
    """Remove emojis and markdown syntax so TTS does not read them aloud."""
    if not text:
        return text
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>~|\\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def gtts_audio_bytes(text: str) -> Optional[Tuple[bytes, str]]:
    try:
        text = strip_for_speech(text)[:200]
        if not text:
            return None
        lang = "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"
        mp3 = BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(mp3)
        return mp3.getvalue(), "audio/mpeg"
    except Exception:
        return None


async def synthesize_speech(text: str, llm_client) -> Optional[Tuple[bytes, str]]:
    if not text:
        return None
    try:
        tts = await llm_client.generate_speech(strip_for_speech(text))
        if tts:
            audio = tts["audio_bytes"]
            mime = tts.get("mime") or "audio/l16"
            if mime.startswith("audio/l16"):
                m = re.match(r"audio/l16;\s*rate=(\d+);\s*channels=(\d+)", mime)
                sample_rate = int(m.group(1)) if m else 24000
                channels = int(m.group(2)) if m else 1
                wav = pcm_to_wav(audio, sample_rate, channels, 16)
                return wav, "audio/wav"
            return audio, mime
    except Exception as e:
        logger.warning("Gemini TTS failed, falling back to gTTS", error=str(e))
    return gtts_audio_bytes(text)


def _cache_audio(audio: bytes, mime: str) -> str:
    token = secrets.token_urlsafe(24)
    _AUDIO_CACHE[token] = {"bytes": audio, "mime": mime}
    if len(_AUDIO_CACHE) > _AUDIO_CACHE_MAX:
        _AUDIO_CACHE.clear()
    return token


class VoiceResponse(BaseModel):
    text_response: str
    audio_response: Optional[str] = None
    audio_url: Optional[str] = None
    conversation_id: Optional[str] = None
    usage: Optional[dict] = None
    model: Optional[str] = None
    providerCalls: Optional[list] = None


@router.get("/audio")
async def voice_audio(token: str = Query(...)):
    """Serve a previously generated audio clip as a streaming response."""
    entry = _AUDIO_CACHE.pop(token, None)
    if not entry:
        raise HTTPException(status_code=404, detail="Audio not found or expired")
    return StreamingResponse(iter([entry["bytes"]]), media_type=entry["mime"])


@router.post("", response_model=VoiceResponse)
async def voice_endpoint(
    audio: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    conversation_id: Optional[str] = Form(None),
    user: dict = Depends(rate_limit),
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

    if lat is not None and lon is not None:
        system_prompt += (
            f"\nThe user is currently at latitude {lat}, longitude {lon}."
            " This IS the user's current location. Never claim you don't know "
            "where the user is. Use this location to give relevant nearby advice."
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

        audio_response = None
        audio_url = None
        if text:
            synthesized = await synthesize_speech(text, llm_client)
            if synthesized:
                audio_bytes_resp, mime = synthesized
                token = _cache_audio(audio_bytes_resp, mime)
                audio_url = f"/voice/audio?token={token}"
                audio_response = f"data:{mime};base64,{base64.b64encode(audio_bytes_resp).decode('utf-8')}"

        provider_calls = consume_usage()
        usage = derive_legacy_usage(provider_calls)
        model = usage.get("model") if usage else None

        return VoiceResponse(
            text_response=text,
            audio_response=audio_response,
            audio_url=audio_url,
            conversation_id=conversation_id,
            usage=usage,
            model=model,
            providerCalls=provider_calls,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
