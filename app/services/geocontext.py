import httpx
import structlog
from typing import Any, Dict, List, Optional

from app.config import settings

logger = structlog.get_logger()


async def get_nearby_sites(
    lat: float,
    lon: float,
    radius: int = 1000,
) -> Optional[List[Dict[str, Any]]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.gis_service_url}/api/v1/nearby-sites",
                params={"lat": lat, "lon": lon, "radius": radius},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("GeoContext returned non-200", status=resp.status_code)
            return None
    except Exception as e:
        logger.error("GeoContext request failed", error=str(e))
        return None


async def get_context(
    lat: float,
    lon: float,
    radius: int = 1000,
) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.gis_service_url}/api/v1/context",
                params={"lat": lat, "lon": lon, "radius": radius},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
    except Exception as e:
        logger.error("GeoContext context request failed", error=str(e))
        return None
