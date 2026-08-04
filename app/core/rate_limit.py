from fastapi import Request, HTTPException

from app.monitoring.instrument import limiter
from app.monitoring import metrics


def client_key(request: Request) -> str:
    """Best-effort stable client key: forwarded-for IP, else request host."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request, endpoint: str) -> None:
    key = client_key(request)
    if not limiter.allow(key):
        metrics.rate_limit_blocks_total.labels(endpoint=endpoint).inc()
        remaining = limiter.remaining(key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {endpoint}. Try again later.",
            headers={"Retry-After": str(limiter.window_seconds)},
        )
