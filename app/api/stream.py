import json
import structlog
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from app.core.system_prompt import build_system_prompt
from app.core.guardrails import check_input
from app.core.auth import allow_access
from app.core.ratelimit import rate_limit
from app.core.llm_client import begin_usage_tracking, consume_usage

logger = structlog.get_logger()

router = APIRouter()


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    persona: str = "auto"
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    geography: Optional[Dict[str, Any]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@router.post("/stream")
async def chat_stream(req: StreamRequest, user: dict = Depends(rate_limit)):
    guard_result = check_input(req.message)
    if guard_result.blocked:
        async def blocked_stream():
            yield f"data: {json.dumps({'error': 'Message blocked', 'reason': guard_result.reason})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    context = {}
    if req.user:
        context["user"] = req.user
    if req.environment:
        context["environment"] = req.environment
    if req.geography:
        context["geography"] = req.geography
    if req.lat is not None and req.lon is not None:
        context["coordinates"] = {"lat": req.lat, "lon": req.lon}

    system_prompt = build_system_prompt(persona=req.persona, context=context)

    from app.main import llm_client

    if not llm_client:
        async def no_client():
            yield f"data: {json.dumps({'error': 'AI service not initialized'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_client(), media_type="text/event-stream")

    try:
        begin_usage_tracking()
        stream = await llm_client.generate(
            system_prompt=system_prompt,
            user_message=req.message,
            stream=True,
        )

        async def generate():
            full_text = ""
            try:
                async for chunk in stream:
                    if chunk:
                        full_text += chunk
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
            except Exception as e:
                logger.error("Stream iteration error", error=str(e))
                yield f"data: {json.dumps({'error': 'AI temporarily unavailable', 'reason': str(e)})}\n\n"
                return
            usage_entries = consume_usage()
            usage = None
            model = None
            if usage_entries:
                usage = {
                    "inputTokens": sum(e.get("inputTokens", 0) for e in usage_entries),
                    "outputTokens": sum(e.get("outputTokens", 0) for e in usage_entries),
                    "totalTokens": sum(e.get("totalTokens", 0) for e in usage_entries),
                }
                model = next((e.get("model") for e in usage_entries if e.get("model")), None)
            yield f"data: {json.dumps({'done': True, 'full_response': full_text, 'usage': usage, 'model': model})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
