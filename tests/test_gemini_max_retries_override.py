"""Phase 2E-C: GEMINI_MAX_RETRIES process-env override.

Verifies that GeminiClient honours the GEMINI_MAX_RETRIES environment
variable, clamped to the demo-safe 0..2 range. No provider network call is
made.
"""

import os

from app.core.llm_client import GeminiClient


def _dummy_client():
    return GeminiClient(api_keys=["dummy-key"])


def test_default_max_retries_is_two(monkeypatch):
    monkeypatch.delenv("GEMINI_MAX_RETRIES", raising=False)
    client = _dummy_client()
    assert client.MAX_RETRIES == 2


def test_env_override_to_zero(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "0")
    client = _dummy_client()
    assert client.MAX_RETRIES == 0


def test_env_override_above_ceiling_is_clamped(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "10")
    client = _dummy_client()
    assert client.MAX_RETRIES == 2


def test_env_override_clamped_negative(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "-5")
    client = _dummy_client()
    assert client.MAX_RETRIES == 0


def test_env_override_non_integer_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "not-a-number")
    client = _dummy_client()
    assert client.MAX_RETRIES == 2
