from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.agent.tools import _recommend_itinerary
from app.core.auth import allow_access
from app.core.ratelimit import rate_limit
from app.core.guardrails import check_input
from app.core.usage import (
    begin_usage_tracking,
    consume_usage_and_attempts,
    derive_legacy_usage,
)

router = APIRouter()

BUDGETS = ["budget", "mid", "luxury"]
STYLES = ["cultural", "adventure", "relaxation", "family", "solo", "romantic"]


class ItineraryRequest(BaseModel):
    interests: List[str] = Field(..., min_length=1, max_length=10)
    days: int = Field(..., ge=1, le=14)
    budget: str = Field(..., min_length=1)
    style: str = Field("cultural", min_length=1)
    cities: Optional[List[str]] = Field(None, max_length=10)
    base_currency: Optional[str] = Field(None, min_length=3, max_length=3)


class ItineraryResponse(BaseModel):
    itinerary: str
    blocked: Optional[bool] = None
    reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    providerCalls: Optional[List[Dict[str, Any]]] = None
    providerAttempts: Optional[List[Dict[str, Any]]] = None


@router.post("", response_model=ItineraryResponse)
async def itinerary_endpoint(req: ItineraryRequest, user: dict = Depends(rate_limit)):
    if req.budget not in BUDGETS:
        raise HTTPException(status_code=422, detail=f"budget must be one of {BUDGETS}")
    if req.style not in STYLES:
        raise HTTPException(status_code=422, detail=f"style must be one of {STYLES}")

    probe_text = " ".join(req.interests) + (" " + " ".join(req.cities) if req.cities else "")
    guard = check_input(probe_text)
    if guard.blocked:
        raise HTTPException(status_code=400, detail=f"Request blocked: {guard.reason}")

    begin_usage_tracking()
    try:
        itinerary = await _recommend_itinerary(
            interests=req.interests,
            days=req.days,
            budget=req.budget,
            cities=req.cities,
            style=req.style,
            base_currency=req.base_currency or "",
        )
    finally:
        provider_calls, provider_attempts = consume_usage_and_attempts()

    usage = derive_legacy_usage(provider_calls)
    model = usage.get("model") if usage else None

    return ItineraryResponse(
        itinerary=itinerary,
        blocked=False,
        usage=usage,
        model=model,
        providerCalls=provider_calls,
        providerAttempts=provider_attempts,
    )
