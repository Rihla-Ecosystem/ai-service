from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.agent.supervisor import route_and_respond

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
    user_journeys: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    persona: Optional[str] = None
    blocked: Optional[bool] = None
    reason: Optional[str] = None


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
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

    result = await route_and_respond(
        message=req.message,
        persona=req.persona,
        context=context,
    )

    return ChatResponse(
        response=result.get("response", ""),
        conversation_id=req.conversation_id,
        persona=result.get("persona"),
        blocked=result.get("blocked", False),
        reason=result.get("reason"),
    )
