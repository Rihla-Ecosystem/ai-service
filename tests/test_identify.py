"""Endpoint contract tests for POST /identify caching behavior.

These tests run the real FastAPI app via TestClient with the module-level
Gemini client replaced by a fake, so no provider network calls happen. They pin
the cache contract: cache hits must return ``providerCalls: []`` explicitly,
must keep ``usage`` and ``model`` absent, must never call the provider or touch
usage tracking, must never mutate the stored cache entry, and must never reuse
provider calls recorded during the originating cache miss.
"""

import base64
import json
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import identify as identify_module
from app.core.usage import (
    OP_IMAGE_ANALYSIS,
    consume_usage,
    make_provider_call,
    record_provider_call,
)
from app.main import app
from app.config import settings

INTERNAL_KEY_HEADERS = {"X-Internal-Api-Key": settings.internal_api_key}

_IMG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/"
    "4arq5AAAAABJRU5ErkJggg=="
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _fake_landmark_json(name="Great Pyramid of Giza"):
    return json.dumps(
        {
            "name": name,
            "name_ar": "الهرم الأكبر",
            "description": "The oldest and largest pyramid in the Giza complex.",
            "category": "pyramid",
            "historical_period": "Old Kingdom",
        }
    )


def _install_fake_client(monkeypatch):
    calls = []

    class _FakeLLMClient:
        def __init__(self):
            self.calls = calls

        async def generate_with_image(
            self,
            system_prompt,
            user_message,
            image_bytes,
            mime_type,
            operation=OP_IMAGE_ANALYSIS,
            _retry_count=0,
        ):
            calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "image_bytes": image_bytes,
                    "mime_type": mime_type,
                    "operation": operation,
                }
            )
            record_provider_call(
                make_provider_call(
                    requested_model="gemini-3.6-flash",
                    actual_model="gemini-3.6-flash",
                    operation=OP_IMAGE_ANALYSIS,
                    usage_source="PROVIDER_RESPONSE",
                    usage_completeness="COMPLETE",
                    inputTokens=100,
                    outputTokens=40,
                    totalTokens=140,
                )
            )
            return _FakeResponse(_fake_landmark_json())

    fake = _FakeLLMClient()
    monkeypatch.setattr("app.main.llm_client", fake)
    return fake


def _identify(client, image_bytes=_IMG, lat=None, lon=None):
    data = {}
    if lat is not None:
        data["lat"] = str(lat)
    if lon is not None:
        data["lon"] = str(lon)
    return client.post(
        "/identify",
        headers=INTERNAL_KEY_HEADERS,
        files={"image": ("photo.png", image_bytes, "image/png")},
        data=data,
    )


def _reset_cache():
    identify_module._cache.clear()


class TestCacheHitProviderCalls:
    def test_cache_hit_returns_empty_provider_calls(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        first = _identify(client)
        second = _identify(client)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert len(first.json()["providerCalls"]) == 1
        assert second.json()["providerCalls"] == []

    def test_cache_hit_usage_is_null(self, monkeypatch):
        _reset_cache()
        _install_fake_client(monkeypatch)
        client = TestClient(app)

        first = _identify(client)
        second = _identify(client)

        assert first.json()["usage"] is not None
        assert second.json()["usage"] is None

    def test_cache_hit_model_is_null(self, monkeypatch):
        _reset_cache()
        _install_fake_client(monkeypatch)
        client = TestClient(app)

        first = _identify(client)
        second = _identify(client)

        assert first.json()["model"] == "gemini-3.6-flash"
        assert second.json()["model"] is None

    def test_cache_hit_does_not_call_provider(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        _identify(client)
        assert len(fake.calls) == 1

        _identify(client)
        _identify(client)
        assert len(fake.calls) == 1

    def test_cache_hit_does_not_touch_usage_tracking(self, monkeypatch):
        _reset_cache()
        _install_fake_client(monkeypatch)
        client = TestClient(app)

        _identify(client)
        _identify(client)

        assert consume_usage() == []

    def test_cache_hit_does_not_reuse_miss_provider_calls(self, monkeypatch):
        _reset_cache()
        _install_fake_client(monkeypatch)
        client = TestClient(app)

        first = _identify(client)
        second = _identify(client)

        miss_calls = first.json()["providerCalls"]
        assert len(miss_calls) == 1
        assert miss_calls[0]["operation"] == OP_IMAGE_ANALYSIS
        assert miss_calls[0]["totalTokens"] == 140
        assert second.json()["providerCalls"] == []

    def test_cache_hit_does_not_mutate_stored_entry(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        _identify(client)
        cache_key = list(identify_module._cache.keys())[0]
        stored = identify_module._cache[cache_key]

        assert stored["cached"] is False
        assert "providerCalls" not in stored
        assert "usage" not in stored
        assert "model" not in stored


class TestImageDimensionLimit:
    def test_decoded_pixel_limit_rejects_an_oversized_image(self):
        class _OversizedImage:
            size = (5000, 5000)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def verify(self):
                return None

        with patch("app.api.identify.Image.open", return_value=_OversizedImage()):
            try:
                identify_module._enforce_image_dimensions(b"image bytes")
                assert False, "expected decoded image pixel limit rejection"
            except HTTPException as exc:
                assert exc.status_code == 413

        _identify(client)
        _identify(client)

        assert identify_module._cache[cache_key] is stored
        assert stored["cached"] is False
        assert "providerCalls" not in stored
        assert "usage" not in stored
        assert "model" not in stored


class TestCacheKeyAndMissBehavior:
    def test_cache_key_distinguishes_image_content(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        _identify(client, image_bytes=_IMG)
        _identify(client, image_bytes=_IMG + b"other")

        assert len(fake.calls) == 2

    def test_cache_key_distinguishes_lat_lon(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        _identify(client, lat=29.9792, lon=31.1342)
        _identify(client, lat=29.9793, lon=31.1343)

        assert len(fake.calls) == 2

    def test_cache_miss_preserves_usage_and_provider_calls(self, monkeypatch):
        _reset_cache()
        _install_fake_client(monkeypatch)
        client = TestClient(app)

        response = _identify(client)

        assert response.status_code == 200
        body = response.json()
        assert body["cached"] is False
        assert body["usage"]["inputTokens"] == 100
        assert body["usage"]["totalTokens"] == 140
        assert body["model"] == "gemini-3.6-flash"
        assert len(body["providerCalls"]) == 1
        assert body["providerCalls"][0]["usageSource"] == "PROVIDER_RESPONSE"
        assert body["name"] == "Great Pyramid of Giza"

    def test_empty_image_rejected(self, monkeypatch):
        _reset_cache()
        fake = _install_fake_client(monkeypatch)
        client = TestClient(app)

        response = client.post(
            "/identify",
            headers=INTERNAL_KEY_HEADERS,
            files={"image": ("photo.png", b"", "image/png")},
        )

        assert response.status_code == 400
        assert len(fake.calls) == 0
