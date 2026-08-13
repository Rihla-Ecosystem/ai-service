from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

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
from app.core.execution_limits import (
    AI_CHAT_QUERY,
    begin_execution_budget,
    end_execution_budget,
    execution_budget_limit,
    estimate_text_tokens,
)


router = APIRouter()

MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_MESSAGE_CHARS = 4000
MAX_HISTORY_TOKENS = 6000


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_HISTORY_MESSAGE_CHARS)


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
    currency: Optional[Dict[str, Any]] = None
    history: List[ChatHistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    executionBudget: Optional[Dict[str, Any]] = None


def format_history(history: List[ChatHistoryMessage], max_messages: int, max_tokens: int) -> str:
    """Render Core's bounded role/content history without accepting metadata."""
    if not history:
        return ""
    selected = []
    used = 0
    for item in reversed(history[-max_messages:]):
        tokens = estimate_text_tokens(item.role, item.content)
        if used + tokens > max_tokens:
            break
        selected.append(item)
        used += tokens
    lines = ["Previous conversation (oldest first):"]
    for item in reversed(selected):
        lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines)


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
    try:
        budget = begin_execution_budget(AI_CHAT_QUERY, req.executionBudget)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    requested = req.executionBudget or {}
    try:
        max_current = execution_budget_limit(requested, "maxCurrentMessageTokens", budget.max_input_tokens, budget.max_input_tokens)
        max_history_messages = execution_budget_limit(requested, "maxHistoryMessages", MAX_HISTORY_MESSAGES, MAX_HISTORY_MESSAGES, allow_zero=True)
        max_history_tokens = execution_budget_limit(requested, "maxHistoryTokens", budget.max_input_tokens, budget.max_input_tokens, allow_zero=True)
    except ValueError as exc:
        end_execution_budget()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if estimate_text_tokens(req.message) > max_current:
        end_execution_budget()
        raise HTTPException(status_code=422, detail="Current message exceeds execution budget")

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
    if req.currency:
        context["currency"] = req.currency

    if req.lat is not None and req.lon is not None:
        if "geography" not in context:
            context["geography"] = {}
        context["geography"]["lat"] = req.lat
        context["geography"]["lon"] = req.lon

    metrics.llm_requests_total.labels(endpoint="chat", status="started").inc()

    try:
        async with trace_turn(
            feature="chat", user_id=get_user_id(user), session_id=req.conversation_id,
            persona=req.persona, input_text=req.message, tags=["chat", req.persona],
        ) as span:
            history = format_history(req.history, max_history_messages, max_history_tokens)
            message = f"{history}\n\nCurrent user message: {req.message}" if history else req.message
            result = await route_and_respond(message=message, persona=req.persona, context=context)
            if span is not None:
                span.update(output={"response": result.get("response", "")[:2000]})
    finally:
        provider_calls, provider_attempts = consume_usage_and_attempts()
        end_execution_budget()
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
