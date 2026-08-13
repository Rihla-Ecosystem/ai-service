import json
from typing import Any, Dict, List, Literal, Optional

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.supervisor import route_and_respond
from app.core.execution_limits import AI_CHAT_QUERY, begin_execution_budget, end_execution_budget, estimate_text_tokens, execution_budget_limit
from app.core.guardrails import check_input
from app.core.ratelimit import rate_limit
from app.core.usage import begin_usage_tracking, consume_usage_and_attempts, derive_legacy_usage
from app.monitoring import metrics
from app.monitoring.langfuse import get_user_id
from app.monitoring.tracing import trace_turn

logger = structlog.get_logger()
router = APIRouter()
MAX_HISTORY_MESSAGES = 100


class StreamHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


def _format_history(history: List[StreamHistoryMessage], max_messages: int, max_tokens: int) -> str:
    selected = []
    used = 0
    for item in reversed(history[-max_messages:]):
        tokens = estimate_text_tokens(item.role, item.content)
        if used + tokens > max_tokens:
            break
        selected.append(item)
        used += tokens
    if not selected:
        return ""
    return "\n".join(["Previous conversation (oldest first):"] + [f"{item.role}: {item.content}" for item in reversed(selected)])


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    persona: str = "auto"
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    geography: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    user_journeys: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    history: List[StreamHistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    executionBudget: Optional[Dict[str, Any]] = None


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/stream")
async def chat_stream(req: StreamRequest, user: dict = Depends(rate_limit)):
    guard_result = check_input(req.message)
    if guard_result.blocked:
        async def blocked_stream():
            yield _sse({"error": "Message blocked", "reason": guard_result.reason})
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    context: Dict[str, Any] = {}
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
    if req.context:
        context["session"] = req.context
    if req.lat is not None and req.lon is not None:
        context["coordinates"] = {"lat": req.lat, "lon": req.lon}

    # Latest main's supervisor may issue tool and regeneration calls; install
    # the feature policy before it so every nested call shares the Chat budget.
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
    history = _format_history(req.history, max_history_messages, max_history_tokens)
    user_turn = f"{history}\n\nCurrent user message: {req.message}" if history else req.message
    provider_calls: List[Dict[str, Any]] = []
    provider_attempts: List[Dict[str, Any]] = []
    result: Optional[Dict[str, Any]] = None
    failure: Optional[Exception] = None
    try:
        async with trace_turn(feature="stream", user_id=get_user_id(user), session_id=req.conversation_id, persona=req.persona, input_text=req.message, tags=["chat", "stream", req.persona]) as span:
            result = await route_and_respond(message=user_turn, persona=req.persona, context=context, user_id=get_user_id(user))
            if span is not None:
                span.update(output={"response": result.get("response", "")[:2000]})
    except Exception as exc:
        failure = exc
        logger.error("Stream processing error", error=str(exc))
    finally:
        provider_calls, provider_attempts = consume_usage_and_attempts()
        end_execution_budget()

    # Do not emit a providerCalls array when execution is uncertain: Core then
    # retains its pre-dispatch reservation for recovery instead of zero-settling.
    if failure is not None or not provider_calls:
        async def error_stream():
            yield _sse({"error": "AI temporarily unavailable", "providerAttempts": provider_attempts})
            yield "data: [DONE]\n\n"
        metrics.llm_requests_total.labels(endpoint="chat_stream", status="error").inc()
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    usage = derive_legacy_usage(provider_calls)
    model = usage.get("model") if usage else None
    response_text = result.get("response", "") if result else ""

    async def generate():
        # Core accepts a complete response in one token event; settlement uses
        # this final evidence event only, exactly once.
        if response_text:
            yield _sse({"token": response_text})
        yield _sse({"done": True, "full_response": response_text, "usage": usage, "model": model, "providerCalls": provider_calls, "providerAttempts": provider_attempts})
        yield "data: [DONE]\n\n"

    metrics.llm_requests_total.labels(endpoint="chat_stream", status="ok").inc()
    return StreamingResponse(generate(), media_type="text/event-stream")
