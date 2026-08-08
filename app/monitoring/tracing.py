"""Request-level Langfuse tracing helpers.

Each user turn (chat / stream / identify / voice) becomes one trace whose name
describes the feature (e.g. ``chat-response``). The Google GenAI OTel
instrumentor then nests each Gemini generation under this span automatically.
Sensitive args are never captured: input is restricted to the user message and
a minimal context summary.
"""

import structlog
from contextlib import asynccontextmanager
from typing import Any, Optional, AsyncGenerator

from app.monitoring.langfuse import is_initialized

logger = structlog.get_logger()

TRACE_NAMES = {
    "chat": "chat-response",
    "stream": "chat-stream-response",
    "identify": "image-identify",
    "voice": "voice-response",
    "itinerary": "itinerary-response",
}


@asynccontextmanager
async def trace_turn(
    *,
    feature: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    persona: Optional[str] = None,
    input_text: Optional[str] = None,
    tags: Optional[list] = None,
) -> AsyncGenerator[Any, None]:
    """Open a root span for one user turn and propagate trace attributes.

    Yields the active span so callers can attach output/metadata. When Langfuse
    is not configured this is a no-op (avoids overhead and import errors).
    """
    if not is_initialized():
        yield None
        return

    from langfuse import get_client, propagate_attributes

    client = get_client()
    name = TRACE_NAMES.get(feature, f"{feature}-turn")
    input_payload = {"message": input_text} if input_text else None
    if persona:
        input_payload = {**(input_payload or {}), "persona": persona}

    with client.start_as_current_observation(
        as_type="span",
        name=name,
        input=input_payload,
    ) as span, propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=tags or [feature],
    ):
        yield span
