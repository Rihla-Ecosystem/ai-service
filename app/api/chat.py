from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.agent.supervisor import route_and_respond
from app.core.auth import allow_access

from app.core.ratelimit import rate_limit
from app.core.llm_client import begin_usage_tracking, consume_usage

from app.core.rate_limit import enforce_rate_limit
from app.monitoring import metrics


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


def _sum_usage(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    model = None
    for e in entries:
        if e.get("model"):
            model = e.get("model")
            break
    return {
        "model": model,
        "inputTokens": sum(e.get("inputTokens", 0) for e in entries),
        "outputTokens": sum(e.get("outputTokens", 0) for e in entries),
        "totalTokens": sum(e.get("totalTokens", 0) for e in entries),
    }


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

    result = await route_and_respond(
        message=req.message,
        persona=req.persona,
        context=context,
    )


    usage_entries = consume_usage()
    usage = _sum_usage(usage_entries)
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
    )
