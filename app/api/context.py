import json as json_mod
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.guardrails import check_output
from app.core.llm_client import OP_TEXT_GENERATION
from app.core.execution_limits import AI_CONTEXT_ANALYZE, begin_execution_budget, end_execution_budget
from app.core.ratelimit import rate_limit
from app.core.usage import (
    begin_usage_tracking,
    consume_usage_and_attempts,
    derive_legacy_usage,
)

router = APIRouter()


class ContextAnalyzeRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="Complete aggregated Context Object")
    operationId: Optional[str] = None
    executionBudget: Optional[Dict[str, Any]] = None


class GeneratedNotification(BaseModel):
    rule: Optional[str] = None
    title: Optional[str] = None
    message: Optional[str] = None
    priority: Optional[str] = None
    category: Optional[str] = None


class ContextReportResult(BaseModel):
    executiveSummary: str
    currentSituation: str
    safetyAssessment: str
    riskAnalysis: str
    personalizedRecommendations: List[str] = []
    touristTips: List[str] = []
    historicalSummary: str
    interestingFacts: List[str] = []
    thingsToAvoid: List[str] = []
    recommendedActions: List[str] = []
    emergencyInstructions: List[str] = []


class ContextAnalyzeResponse(BaseModel):
    summary: Dict[str, Any]
    report: ContextReportResult
    generatedNotifications: List[GeneratedNotification] = []
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    providerCalls: Optional[List[Dict[str, Any]]] = None
    providerAttempts: Optional[List[Dict[str, Any]]] = None


def _require_llm():
    from app.main import llm_client

    if llm_client is None:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    return llm_client


@router.post("/analyze", response_model=ContextAnalyzeResponse)
async def analyze_context(req: ContextAnalyzeRequest, user: dict = Depends(rate_limit)):
    try:
        begin_execution_budget(AI_CONTEXT_ANALYZE, req.executionBudget)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    context = req.context

    geo = context.get("geoContext") or {}
    risk = context.get("riskContext") or {}

    area = geo.get("currentArea") or geo.get("governorate") or "your area"
    risk_level = risk.get("riskLevel") or "info"
    safety_score = risk.get("safetyScore")
    threats_raw = risk.get("threats") or []
    threats = [
        (t.get("headline") or t.get("category"))
        for t in threats_raw
        if (t.get("headline") or t.get("category"))
    ]
    restricted = geo.get("restrictedAreas") or []
    restricted_names = [r.get("name") or r.get("reason") for r in restricted if isinstance(r, dict)]
    photo_restrictions = geo.get("photographyRestrictions") or []
    historical = geo.get("historicalPlaces") or []
    historical_names = [h.get("name") for h in historical if isinstance(h, dict) and h.get("name")]
    attractions = geo.get("nearbyAttractions") or []
    attraction_names = [a.get("name") for a in attractions if isinstance(a, dict) and a.get("name")]
    hotels = geo.get("nearbyHotels") or []
    hotel_names = [h.get("name") for h in hotels if isinstance(h, dict) and h.get("name")]
    restaurants = geo.get("nearbyRestaurants") or []
    restaurant_names = [r.get("name") for r in restaurants if isinstance(r, dict) and r.get("name")]

    context_block = (
        f"Current area: {area}\n"
        f"Risk level: {risk_level} (safety score {safety_score if safety_score is not None else 'N/A'}/100)\n"
        f"Nearby threats/alerts: {', '.join(t for t in threats if t) or 'none reported'}\n"
        f"Restricted areas: {', '.join(n for n in restricted_names if n) or 'none nearby'}\n"
        f"Photography restrictions: {', '.join(photo_restrictions) or 'none nearby'}\n"
        f"Historical sites: {', '.join(historical_names) or 'none nearby'}\n"
        f"Notable attractions: {', '.join(attraction_names) or 'none nearby'}\n"
        f"Nearby hotels: {', '.join(hotel_names) or 'none'}\n"
        f"Nearby restaurants: {', '.join(restaurant_names) or 'none'}\n"
    )

    system_prompt = (
        "You are Rihla's Context Intelligence engine. Analyze the traveler's current"
        " context based ONLY on the provided aggregated context object. Do not search"
        " or invent external information. Respond in raw JSON only, with exactly these fields:\n"
        "{\n"
        '  "executiveSummary": "1-2 sentence overview",\n'
        '  "currentSituation": "where they are and current risk",\n'
        '  "safetyAssessment": "concrete safety evaluation",\n'
        '  "riskAnalysis": "specific nearby threats",\n'
        '  "personalizedRecommendations": ["..."],\n'
        '  "touristTips": ["..."],\n'
        '  "historicalSummary": "notable history nearby",\n'
        '  "interestingFacts": ["..."],\n'
        '  "thingsToAvoid": ["..."],\n'
        '  "recommendedActions": ["..."],\n'
        '  "emergencyInstructions": ["..."]\n'
        "}\n"
        "Do NOT include markdown code fences."
    )

    try:
        begin_usage_tracking()
        response = await _require_llm().generate(
            system_prompt=system_prompt,
            user_message=context_block,
            temperature=0.4,
            operation=OP_TEXT_GENERATION,
        )
        text = ""
        if response is not None and hasattr(response, "text") and response.text is not None:
            text = response.text

        guard_result = check_output(text)
        if guard_result.requires_regeneration:
            raise HTTPException(status_code=400, detail="Analysis did not pass safety checks")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("\n", 1)[0]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        report = json_mod.loads(cleaned)
        if not isinstance(report, dict):
            report = {}

        report = {k: report.get(k) for k in [
            "executiveSummary", "currentSituation", "safetyAssessment", "riskAnalysis",
            "personalizedRecommendations", "touristTips", "historicalSummary",
            "interestingFacts", "thingsToAvoid", "recommendedActions", "emergencyInstructions",
        ]}

        generated = _derive_notifications(report)

        provider_calls, provider_attempts = consume_usage_and_attempts()
        usage = derive_legacy_usage(provider_calls)
        model = usage.get("model") if usage else None

        return ContextAnalyzeResponse(
            summary={
                "area": area,
                "riskLevel": risk_level,
                "safetyScore": safety_score,
                "threatCount": len(threats),
                "restrictedAreas": restricted_names,
                "photographyRestrictions": photo_restrictions,
            },
            report=ContextReportResult(**report),
            generatedNotifications=[GeneratedNotification(**n) for n in generated],
            model=model,
            usage=usage,
            providerCalls=provider_calls,
            providerAttempts=provider_attempts,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")
    finally:
        end_execution_budget()


def _derive_notifications(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    notifications: List[Dict[str, Any]] = []
    if report.get("riskAnalysis"):
        notifications.append({
            "rule": "ai_risk_summary",
            "title": "Context Risk Summary",
            "message": str(report["riskAnalysis"])[:280],
            "priority": "NORMAL",
            "category": "SAFETY",
        })
    for t in report.get("recommendedActions") or []:
        notifications.append({"rule": "ai_recommendation", "title": "Recommended Action", "message": str(t)[:280], "priority": "LOW", "category": "RECOMMENDATION"})
    if report.get("emergencyInstructions"):
        notifications.append({"rule": "ai_emergency", "title": "Emergency What to Know", "message": str(report["emergencyInstructions"][0])[:280], "priority": "HIGH", "category": "SAFETY"})
    return notifications
