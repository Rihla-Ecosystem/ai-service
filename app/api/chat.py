from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.agent.supervisor import route_and_respond
from app.core.auth import allow_access

from app.core.ratelimit import rate_limit
from app.core.usage import (
    begin_usage_tracking,
    consume_usage_and_attempts,
    derive_legacy_usage,
)

from app.core.rate_limit import enforce_rate_limit
from app.monitoring import metrics
from app.monitoring.langfuse import get_user_id
from app.monitoring.tracing import trace_turn


router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    persona: str = "auto"
    lat: Optional[float] = None
    lon: Optional[float] = None
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    geography: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    user_journeys: Optional[Any] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    persona: Optional[str] = None
    blocked: Optional[bool] = None
    reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    providerCalls: Optional[List[Dict[str, Any]]] = None
    providerAttempts: Optional[List[Dict[str, Any]]] = None


@router.post("", response_model=ChatResponse)

async def chat_endpoint(req: ChatRequest, user: dict = Depends(rate_limit)):
    begin_usage_tracking()

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
        if "geography" not in context:
            context["geography"] = {}
        context["geography"]["lat"] = req.lat
        context["geography"]["lon"] = req.lon

    metrics.llm_requests_total.labels(endpoint="chat", status="started").inc()

    async with trace_turn(
        feature="chat",
        user_id=get_user_id(user),
        session_id=req.conversation_id,
        persona=req.persona,
        input_text=req.message,
        tags=["chat", req.persona],
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
    metrics.llm_requests_total.labels(endpoint="chat", status="ok").inc()


    return ChatResponse(
        response=result.get("response", ""),
        conversation_id=req.conversation_id,
        persona=result.get("persona"),
        blocked=result.get("blocked", False),
        reason=result.get("reason"),
        usage=usage,
        model=model,
        providerCalls=provider_calls,
        providerAttempts=provider_attempts,
    )
