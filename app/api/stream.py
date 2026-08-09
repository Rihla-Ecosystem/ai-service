import json
import structlog
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

from app.agent.supervisor import route_and_respond
from app.core.system_prompt import build_system_prompt, build_user_context
from app.core.guardrails import check_input
from app.core.auth import allow_access
from app.core.ratelimit import rate_limit
from app.core.usage import (
    begin_usage_tracking,
    consume_usage_and_attempts,
    derive_legacy_usage,
)
from app.monitoring.langfuse import get_user_id
from app.monitoring.tracing import trace_turn
from app.core.rate_limit import enforce_rate_limit
from app.monitoring import metrics

logger = structlog.get_logger()

router = APIRouter()


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    persona: str = "auto"
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    geography: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    user_journeys: Optional[Any] = None
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
    if req.safety:
        context["safety"] = req.safety
    if req.user_journeys:
        context["user_journeys"] = req.user_journeys
    if req.lat is not None and req.lon is not None:
        context["coordinates"] = {"lat": req.lat, "lon": req.lon}

    begin_usage_tracking()

    try:
        async with trace_turn(
            feature="stream",
            user_id=get_user_id(user),
            session_id=req.conversation_id,
            persona=req.persona,
            input_text=req.message,
            tags=["chat", "stream", req.persona],
        ) as span:
            result = await route_and_respond(
                message=req.message,
                persona=req.persona,
                context=context,
            )
            if span is not None:
                span.update(output={"response": result.get("response", "")[:2000]})

        provider_calls, provider_attempts = consume_usage_and_attempts()
        usage = derive_legacy_usage(provider_calls)
        model = usage.get("model") if usage else None

        async def generate():
            # Single-chunk response (non-streaming fallback)
            response_text = result.get("response", "")
            if response_text:
                yield f"data: {json.dumps({'token': response_text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full_response': response_text, 'usage': usage, 'model': model, 'providerCalls': provider_calls, 'providerAttempts': provider_attempts})}\n\n"
            yield "data: [DONE]\n\n"

        metrics.llm_requests_total.labels(endpoint="chat_stream", status="ok").inc()
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error("Stream processing error", error=str(e))
        async def error_stream():
            yield f"data: {json.dumps({'error': 'AI temporarily unavailable'})}\n\n"
            yield "data: [DONE]\n\n"
        metrics.llm_requests_total.labels(endpoint="chat_stream", status="error").inc()
        return StreamingResponse(error_stream(), media_type="text/event-stream")