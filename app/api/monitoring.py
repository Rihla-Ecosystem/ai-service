from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.config import settings
from app.monitoring.instrument import metrics_response, build_stats, limiter

router = APIRouter()
security = HTTPBearer(auto_error=False)


def _is_admin(request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    if settings.internal_api_key and request.headers.get("X-Internal-Api-Key") == settings.internal_api_key:
        return True
    if credentials and credentials.scheme.lower() == "bearer":
        try:
            import jwt as pyjwt

            payload = pyjwt.decode(
                credentials.credentials,
                settings.jwt_access_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            return payload.get("role") == "admin"
        except pyjwt.PyJWTError:
            return False
    return False


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics exposition. Unauthenticated for scraping by a metrics collector."""
    return metrics_response()


@router.get("/admin/stats")
async def admin_stats(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not _is_admin(request, credentials):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    from app.main import vector_store

    collections = await vector_store.list_collections() if vector_store else []
    return build_stats(collections=collections)


@router.get("/admin/rate-limit")
async def rate_limit_status(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not _is_admin(request, credentials):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return {"rate_limiter": limiter.stats()}
