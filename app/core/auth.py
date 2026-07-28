from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
    if not settings.internal_api_key:
        return False
    return request.headers.get("X-Internal-Api-Key") == settings.internal_api_key


async def allow_access(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    if _check_internal_api_key(request):
        return {"sub": "internal-gateway", "role": "admin", "source": "internal"}
    if credentials:
        try:
            user = _verify_jwt(credentials.credentials)
            if user.get("role") == "admin":
                return user
            raise HTTPException(status_code=403, detail="Admin privileges required for direct access")
        except pyjwt.PyJWTError as e:
            logger.warning("JWT verification failed", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    raise HTTPException(status_code=401, detail="Authentication required")