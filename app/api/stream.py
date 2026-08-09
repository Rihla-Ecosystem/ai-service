import json
import structlog
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

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
from app.core.llm_client import CHAT_MAX_OUTPUT_TOKENS
from app.core.execution_limits import (
    AI_CHAT_QUERY,
    begin_execution_budget,
    end_execution_budget,
    estimate_text_tokens,
)

logger = structlog.get_logger()

router = APIRouter()
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_TOKENS = 6000


class StreamHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


def _format_history(history: List[StreamHistoryMessage]) -> str:
    selected = []
    used = 0
    for item in reversed(history):
        tokens = estimate_text_tokens(item.role, item.content)
        if used + tokens > MAX_HISTORY_TOKENS:
            break
        selected.append(item)
        used += tokens
    return "\n".join(["Previous conversation (oldest first):"] + [
        f"{item.role}: {item.content}" for item in reversed(selected)
    ]) if selected else ""


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    persona: str = "auto"
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    geography: Optional[Dict[str, Any]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    history: List[StreamHistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


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
    history = _format_history(req.history)
    user_turn = f"{history}\n\nCurrent user message: {req.message}" if history else req.message
    user_context_data = build_user_context(context)
    if user_context_data:
        user_turn = f"{req.message}\n\n{user_context_data}"

    from app.main import llm_client

    if not llm_client:
        async def no_client():
            yield f"data: {json.dumps({'error': 'AI service not initialized'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_client(), media_type="text/event-stream")

    try:
        begin_usage_tracking()
        begin_execution_budget(AI_CHAT_QUERY)
        async with trace_turn(
            feature="stream",
            user_id=get_user_id(user),
            session_id=req.conversation_id,
            persona=req.persona,
            input_text=req.message,
            tags=["chat", "stream", req.persona],
        ) as span:
            stream = await llm_client.generate(
                system_prompt=system_prompt,
                user_message=user_turn,
                max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
                stream=True,
            )

            async def generate():
                full_text = ""
                consumed = False
                try:
                    async for chunk in stream:
                        if chunk:
                            full_text += chunk
                            yield f"data: {json.dumps({'token': chunk})}\n\n"
                except Exception as e:
                    logger.error("Stream iteration error", error=str(e))
                    provider_calls, provider_attempts = consume_usage_and_attempts()
                    consumed = True
                    usage = derive_legacy_usage(provider_calls)
                    model = usage.get("model") if usage else None
                    yield f"data: {json.dumps({'error': 'AI temporarily unavailable', 'usage': usage, 'model': model, 'providerCalls': provider_calls, 'providerAttempts': provider_attempts})}\n\n"
                    yield "data: [DONE]\n\n"
                    end_execution_budget()
                    return
                if not consumed:
                    provider_calls, provider_attempts = consume_usage_and_attempts()
                    usage = derive_legacy_usage(provider_calls)
                    model = usage.get("model") if usage else None
                    yield f"data: {json.dumps({'done': True, 'full_response': full_text, 'usage': usage, 'model': model, 'providerCalls': provider_calls, 'providerAttempts': provider_attempts})}\n\n"
                    yield "data: [DONE]\n\n"
                    end_execution_budget()

            return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error("Stream setup error", error=str(e))
        async def error_stream():
            yield f"data: {json.dumps({'error': 'AI temporarily unavailable'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
