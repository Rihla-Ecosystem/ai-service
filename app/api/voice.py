import base64
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.guardrails import check_input, check_output

router = APIRouter()


class VoiceResponse(BaseModel):
    text_response: str
    audio_response: Optional[str] = None
    conversation_id: Optional[str] = None


@router.post("", response_model=VoiceResponse)
async def voice_endpoint(
    audio: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    conversation_id: Optional[str] = Form(None),
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

        return VoiceResponse(
            text_response=text,
            conversation_id=conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
