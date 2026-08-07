"""Contract tests for POST /analyze (Context Intelligence engine).

Exercises the endpoint through the real FastAPI app with a fake Gemini client
so no provider network calls happen. It pins:
- the endpoint accepts a full context object.
- a well-formed LLM JSON response is normalized into the report + generated
  notifications shape the Core service expects.
- malformed JSON from the LLM degrades gracefully (no 500).
- the internal API key path is honored (no JWT required).
"""

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

INTERNAL_KEY_HEADERS = {"X-Internal-Api-Key": settings.internal_api_key}


def _full_report_json():
    return json.dumps(
        {
            "executiveSummary": "You are in Cairo near a historical site with low risk.",
            "currentSituation": "Located in Cairo, info risk.",
            "safetyAssessment": "Score 92/100, generally safe.",
            "riskAnalysis": "Displaced persons flow ongoing with active short-term shelters.",
            "personalizedRecommendations": ["Drink bottled water."],
            "touristTips": ["Respect local customs."],
            "historicalSummary": "Near Khan el-Khalili.",
            "interestingFacts": ["One of the oldest markets."],
            "thingsToAvoid": ["Restricted areas."],
            "recommendedActions": ["Stay aware."],
            "emergencyInstructions": ["Dial 122 for police."],
        }
    )


def _context_payload():
    return {
        "context": {
            "location": {"lat": 30.0444, "lng": 31.2357, "reason": "movement"},
            "geoContext": {
                "inEgypt": True,
                "currentArea": "Cairo",
                "governorate": "Cairo",
                "nearbyAttractions": [{"name": "Egyptian Museum"}],
                "historicalPlaces": [{"name": "Khan el-Khalili"}],
                "restrictedAreas": [],
                "photographyRestrictions": [],
            },
            "riskContext": {"riskLevel": "info", "safetyScore": 92, "threats": []},
            "userProfile": {"id": "u1"},
            "collectedAt": "2026-08-07T00:00:00Z",
        }
    }


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _install_fake_client(monkeypatch, text_func):
    calls = []

    class _FakeLLMClient:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return _FakeResponse(text_func())

    fake = _FakeLLMClient()
    monkeypatch.setattr("app.main.llm_client", fake)
    return calls


def test_analyze_returns_context_notification_report(monkeypatch):
    calls = _install_fake_client(monkeypatch, _full_report_json)
    client = TestClient(app)
    res = client.post("/analyze", json=_context_payload(), headers=INTERNAL_KEY_HEADERS)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["executiveSummary"]
    assert body["summary"]["area"] == "Cairo"
    assert body["summary"]["riskLevel"] == "info"
    assert isinstance(body["generatedNotifications"], list)
    assert body["generatedNotifications"], "should derive generated notifications"
    assert body["generatedNotifications"][0]["rule"] == "ai_risk_summary"
    assert calls, "fake LLM should have been invoked"


def test_analyze_requires_internal_api_key():
    client = TestClient(app)
    res = client.post("/analyze", json=_context_payload())
    assert res.status_code == 401


def test_analyze_degrades_gracefully_on_bad_llm_json(monkeypatch):
    calls = _install_fake_client(monkeypatch, lambda: "noise not valid json")
    client = TestClient(app)
    res = client.post("/analyze", json=_context_payload(), headers=INTERNAL_KEY_HEADERS)
    assert res.status_code == 500
    assert calls