import structlog
from typing import Optional

from app.config import settings

logger = structlog.get_logger()

_initialized = False


def initialize():
    global _initialized
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("Langfuse not configured — skipping")
        return

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _initialized = True
        logger.info("Langfuse initialized")
    except Exception as e:
        logger.warning("Langfuse initialization failed", error=str(e))


def is_initialized() -> bool:
    return _initialized
