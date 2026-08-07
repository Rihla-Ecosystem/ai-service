import time
import threading
from collections import deque
from fastapi import Depends, HTTPException

from app.core.auth import allow_access
from app.config import settings

_log = threading.Lock()
_requests: dict[str, deque] = {}

WINDOW_SECONDS = 60

_INTERNAL_GATEWAY_KEY = "internal-gateway"


def _enforce(key: str, limit: int) -> None:
    now = time.monotonic()
    with _log:
        dq = _requests.setdefault(key, deque())
        while dq and dq[0] <= now - WINDOW_SECONDS:
            dq.popleft()
        if len(dq) >= limit:
            retry_after = WINDOW_SECONDS - (now - dq[0]) if dq else WINDOW_SECONDS
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(int(max(1, retry_after)))},
            )
        dq.append(now)


def rate_limit(user: dict = Depends(allow_access)) -> dict:
    """Per-user sliding-window rate limiter.

    Only genuine admin JWTs are exempt. Internal-key callers share a single
    high-but-bounded bucket so a leaked key cannot trigger unlimited traffic.
    """
    if user.get("role") == "admin" and user.get("source") != "internal":
        return user

    limit = settings.rate_limit_per_user
    if limit <= 0:
        return user

    if user.get("sub") == _INTERNAL_GATEWAY_KEY:
        _enforce(_INTERNAL_GATEWAY_KEY, settings.rate_limit_internal)
        return user

    _enforce(str(user.get("sub", "anonymous")), limit)
    return user
