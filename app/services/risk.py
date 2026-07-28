import httpx
import structlog
from typing import Any, Dict, Optional

from app.config import settings

logger = structlog.get_logger()


async def get_safety_info(city: str) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.risk_service_url}/safety/current",
                params={"city": city.lower()},
                headers={"X-Internal-Api-Key": settings.internal_api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Risk service returned non-200", status=resp.status_code)
            return None
    except Exception as e:
        logger.error("Risk service request failed", error=str(e))
        return None


async def get_all_safety() -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.risk_service_url}/safety/current",
                headers={"X-Internal-Api-Key": settings.internal_api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
    except Exception as e:
        logger.error("Risk service request failed", error=str(e))
        return None
