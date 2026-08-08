import structlog
from typing import Any, Dict, Optional

from app.config import settings

logger = structlog.get_logger()

_initialized = False


def initialize():
    """Set up Langfuse tracing: client + Google GenAI OTel instrumentor.

    Langfuse v4 is OpenTelemetry-native: creating the client registers a global
    TracerProvider whose spans are exported to Langfuse. The GenAI instrumentor
    attaches to that provider, so every Gemini call becomes a `generation`
    observation (model, tokens, cost) nested under whatever span is active.
    """
    global _initialized
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("Langfuse not configured — skipping")
        return

    try:
        from langfuse import get_client

        get_client()
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        GoogleGenAIInstrumentor().instrument()
        _initialized = True
        logger.info("Langfuse initialized (Google GenAI OTel instrumentor)")
    except Exception as e:
        logger.warning("Langfuse initialization failed", error=str(e))


def is_initialized() -> bool:
    return _initialized


def get_user_id(auth_user: Optional[Dict[str, Any]]) -> Optional[str]:
    """Best-effort user identifier from the verified JWT payload (sub claim)."""
    if not auth_user:
        return None
    sub = auth_user.get("sub")
    if isinstance(sub, str) and sub:
        return sub
    return None
