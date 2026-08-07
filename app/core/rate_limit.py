from typing import Optional
from fastapi import Request, HTTPException

from app.monitoring.instrument import limiter
from app.monitoring import metrics


def client_key(request: Request, user: Optional[dict] = None) -> str:
    """Stable rate-limit key: authenticated user sub, else direct peer host.

    X-Forwarded-For is intentionally NOT trusted: it is client-spoofable and
    the direct peer address is the only non-forgeable source.
    """
    if user and user.get("sub"):
        return f"user:{user['sub']}"
    if request.client:
        return f"ip:{request.client.host}"
    return "unknown"


def enforce_rate_limit(request: Request, endpoint: str, user: Optional[dict] = None) -> None:
    key = client_key(request, user)
    if not limiter.allow(key):
        metrics.rate_limit_blocks_total.labels(endpoint=endpoint).inc()
        remaining = limiter.remaining(key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {endpoint}. Try again later.",
            headers={"Retry-After": str(limiter.window_seconds)},
        )
