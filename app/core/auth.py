from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import jwt as pyjwt
import structlog

from app.config import settings

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)


def _verify_jwt(token: str) -> dict:
    payload = pyjwt.decode(
        token,
        settings.jwt_access_secret,
        algorithms=["HS256"],
        options={"verify_aud": False}
    )
    if "sub" not in payload or "role" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token contract")
    return payload


def _check_internal_api_key(request: Request) -> bool:
    header = request.headers.get("X-Internal-Api-Key")
    if not header or not settings.internal_api_key:
        return False
    return secrets.compare_digest(header, settings.internal_api_key)


async def allow_access(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    has_internal_key = _check_internal_api_key(request)

    user = None
    if credentials:
        try:
            user = _verify_jwt(credentials.credentials)
        except pyjwt.PyJWTError as e:
            logger.warning("JWT verification failed", error=str(e))

    # Internal key present: this is Core proxying a request (or pure
    # service-to-service). Prefer the real user JWT when present so per-user
    # rate limiting keys on the actual user instead of the shared gateway.
    if has_internal_key:
        if user:
            return user
        return {"sub": "internal-gateway", "role": "admin", "source": "internal"}

    # Direct access (no internal key): only admins may call ai-service
    # directly; regular users must go through Core.
    if user:
        if user.get("role") == "admin":
            return user
        raise HTTPException(status_code=403, detail="Admin privileges required for direct access")

    raise HTTPException(status_code=401, detail="Authentication required")