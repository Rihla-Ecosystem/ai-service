"""Phase 2E-A2.1 tests for providerAttempts observability and contract.

These tests cover the mandated scenarios:

  A. success                                -> SUCCEEDED attempt linked to a provider call
  B. confirmed request/auth/unsupported rejection (400/401/403/404)
                                           -> FAILED attempt with errorCategory + httpStatus
  C. timeout / connection drop / unknown after start / 5xx / 429
                                           -> INDETERMINATE attempt
  D. retry then success                    -> INDETERMINATE then SUCCEEDED attempts, attemptNumber 1,2
  E. response received but local processing fails -> INDETERMINATE, providerResponseReceived=true
  F. voice: audio ok + TTS fails + gTTS fallback  -> SUCCEEDED + INDETERMINATE attempts, gTTS fallback
  G. identify cache hit                    -> providerAttempts: [] and providerCalls: []

No provider network call is made: the model client is always replaced by a
fake. Attempts must never contain prompts, responses, media, or secrets.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.api import identify as identify_module
from app.core.llm_client import GeminiClient, KeyStatus
from app.core.usage import (
    ATTEMPT_OUTCOME_INDETERMINATE,
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_CONNECTION_ERROR,
    ERROR_CATEGORY_INVALID_REQUEST,
    ERROR_CATEGORY_LOCAL_PROCESSING,
    ERROR_CATEGORY_RATE_LIMIT,
    ERROR_CATEGORY_SERVER_ERROR,
    ERROR_CATEGORY_TIMEOUT,
    ERROR_CATEGORY_UNKNOWN,
    ERROR_CATEGORY_UNSUPPORTED_OPERATION,
    OP_AUDIO_UNDERSTANDING,
    OP_IMAGE_ANALYSIS,
    OP_TEXT_CHAT,
    OP_TEXT_GENERATION,
    OP_TEXT_TO_SPEECH,
    PROVIDER_GOOGLE,
    begin_usage_tracking,
    consume_attempts,
    consume_usage_and_attempts,
    make_provider_attempt,
    make_provider_call,
    record_provider_attempt,
    record_provider_call,
)
from app.main import app

INTERNAL_KEY_HEADERS = {"X-Internal-Api-Key": "change-me-in-production"}

_IMG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _ApiError(Exception):
    def __init__(self, code=None, message=""):
        super().__init__(message)
        self.code = code


class _UsageMeta:
    def __init__(self, prompt, candidates, total):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class _Resp:
    def __init__(self, text="ok", model_version="gemini-3.6-flash", meta=None, candidates=None):
        self.text = text
        self.model_version = model_version
        self.usage_metadata = meta
        self.candidates = candidates if candidates is not None else []


class _Chunk:
    def __init__(self, text=None, meta=None, model_version=None):
        self.text = text
        self.usage_metadata = meta
        self.model_version = model_version


class _Inline:
    def __init__(self, data, mime_type):
        self.data = data
        self.mime_type = mime_type


class _Part:
    def __init__(self, inline_data=None):
        self.inline_data = inline_data


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Candidate:
    def __init__(self, content):
        self.content = content


def _tts_ok_response():
    return _Resp(
        model_version="gemini-3.1-flash-tts-preview",
        candidates=[_Candidate(_Content([_Part(_Inline(b"\x00\x01", "audio/l16"))]))],
    )


def _make_fake(monkeypatch, responses, key_overrides=None):
    """Non-stream fake: each call pops the next response/error (last repeats).

    Uses two keys so a retry after a failed attempt can rotate to an available
    key (single-key clients enter cooldown and retries would raise instead).
    """
    from app.core.llm_client import GeminiClient

    client = GeminiClient(api_keys=["dummy-key-1", "dummy-key-2"])
    if key_overrides:
        for key in client.keys:
            key_overrides(key)
    state = {"idx": 0}

    class _Models:
        def generate_content(self, **kw):
            i = state["idx"]
            state["idx"] += 1
            item = responses[min(i, len(responses) - 1)]
            if isinstance(item, BaseException):
                raise item
            return item

        def generate_content_stream(self, **kw):
            raise AssertionError("stream not expected in this fake")

    class _FakeClient:
        def __init__(self):
            self.models = _Models()

    fake = _FakeClient()
    for key in client.keys:
        monkeypatch.setattr(key, "client", fake)
    return client, fake


def _make_stream_fake(monkeypatch, chunk_gen):
    from app.core.llm_client import GeminiClient

    client = GeminiClient(api_keys=["dummy-key-1", "dummy-key-2"])
    key = client.keys[0]

    class _Models:
        def generate_content(self, **kw):
            raise AssertionError("non-stream not expected in this fake")

        def generate_content_stream(self, **kw):
            return chunk_gen()

    class _FakeClient:
        def __init__(self):
            self.models = _Models()

    fake = _FakeClient()
    monkeypatch.setattr(key, "client", fake)
    return client, fake


def _no_key(key):
    key.status = KeyStatus.COOLDOWN
    key.cooldown_until = float("inf")


# ---------------------------------------------------------------------------
# Scenario A — success
# ---------------------------------------------------------------------------

class TestSuccess:
    def test_non_stream_success_records_succeeded_attempt(self, monkeypatch):
        resp = _Resp(text="ok", model_version="gemini-3.6-flash", meta=_UsageMeta(100, 40, 140))
        client, _ = _make_fake(monkeypatch, [resp])
        begin_usage_tracking()

        async def _run():
            await client.generate(system_prompt="", user_message="hi", operation=OP_TEXT_GENERATION)

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1
        assert len(attempts) == 1
        a = attempts[0]
        assert a["attemptId"] == "attempt-1"
        assert a["attemptNumber"] == 1
        assert a["outcome"] == "SUCCEEDED"
        assert a["provider"] == PROVIDER_GOOGLE
        assert a["operation"] == OP_TEXT_GENERATION
        assert a["requestedModel"] == "gemini-3.6-flash"
        assert a["actualModel"] == "gemini-3.6-flash"
        assert a["providerResponseReceived"] is True
        assert a["providerCallId"] == "call-1"
        assert a["providerCallStarted"] is True
        assert "providerCallStartedAt" in a
        assert a["providerCallStartedAt"]
        assert "errorCategory" not in a
        assert "httpStatus" not in a

    def test_stream_success_records_succeeded_attempt(self, monkeypatch):
        def gen():
            return iter([
                _Chunk("Hello", _UsageMeta(100, 10, 110), "gemini-3.6-flash"),
                _Chunk("Hello world", _UsageMeta(100, 25, 125), "gemini-3.6-flash"),
            ])

        client, _ = _make_stream_fake(monkeypatch, gen)
        begin_usage_tracking()

        async def _run():
            stream = await client.generate(system_prompt="", user_message="hi", stream=True)
            async for _ in stream:
                pass

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1
        assert len(attempts) == 1
        a = attempts[0]
        assert a["outcome"] == "SUCCEEDED"
        assert a["attemptNumber"] == 1
        assert a["providerResponseReceived"] is True
        assert a["providerCallId"] == "call-1"
        assert a["actualModel"] == "gemini-3.6-flash"
        assert a["operation"] == "TEXT_CHAT_STREAM"


# ---------------------------------------------------------------------------
# Scenario B — explicit provider failure
# ---------------------------------------------------------------------------

class TestExplicitFailure:
    def test_4xx_is_failed_invalid_request(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(400, "bad request")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert calls == []
        assert len(attempts) == 1
        a = attempts[0]
        assert a["outcome"] == "FAILED"
        assert a["errorCategory"] == ERROR_CATEGORY_INVALID_REQUEST
        assert a["httpStatus"] == 400
        assert a["providerResponseReceived"] is False
        assert a["attemptNumber"] == 1
        assert "providerCallId" not in a

    def test_5xx_is_indeterminate_server_error(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(503, "unavailable")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_SERVER_ERROR
        assert attempts[0]["httpStatus"] == 503

    @pytest.mark.parametrize("status", [500, 502, 504])
    def test_5xx_variants_are_indeterminate_server_error(self, monkeypatch, status):
        client, _ = _make_fake(monkeypatch, [_ApiError(status, "server error")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_SERVER_ERROR
        assert attempts[0]["httpStatus"] == status

    def test_429_is_indeterminate_rate_limit(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(429, "quota")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_RATE_LIMIT
        assert attempts[0]["httpStatus"] == 429

    @pytest.mark.parametrize("status", [401, 403])
    def test_401_403_are_failed_auth_error(self, monkeypatch, status):
        client, _ = _make_fake(monkeypatch, [_ApiError(status, "not allowed")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "FAILED"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_AUTH_ERROR
        assert attempts[0]["httpStatus"] == status

    def test_404_is_failed_unsupported_operation(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(404, "not found")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "FAILED"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_UNSUPPORTED_OPERATION
        assert attempts[0]["httpStatus"] == 404


# ---------------------------------------------------------------------------
# Scenario C — timeout / connection drop
# ---------------------------------------------------------------------------

class TestIndeterminate:
    def test_timeout_is_indeterminate(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(None, "deadline exceeded before operation completed")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert calls == []
        assert len(attempts) == 1
        a = attempts[0]
        assert a["outcome"] == "INDETERMINATE"
        assert a["errorCategory"] == ERROR_CATEGORY_TIMEOUT
        assert "httpStatus" not in a
        assert a["providerResponseReceived"] is False

    def test_connection_error_is_indeterminate(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [_ApiError(None, "connection reset by peer")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_CONNECTION_ERROR

    def test_unknown_exception_after_start_is_indeterminate(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [ValueError("something odd happened")])
        client.MAX_RETRIES = 0
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        _, attempts = consume_usage_and_attempts()
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_UNKNOWN
        assert "httpStatus" not in attempts[0]

    def test_stream_mid_failure_is_indeterminate_with_partial_call(self, monkeypatch):
        def gen():
            yield _Chunk("Hello", _UsageMeta(100, 10, 110), "gemini-3.6-flash")
            raise _ApiError(None, "connection reset by peer")

        client, _ = _make_stream_fake(monkeypatch, gen)
        begin_usage_tracking()

        async def _run():
            stream = await client.generate(system_prompt="", user_message="hi", stream=True)
            with pytest.raises(_ApiError):
                async for _ in stream:
                    pass

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1, "partial stream still records the final snapshot call"
        assert len(attempts) == 1
        a = attempts[0]
        assert a["outcome"] == "INDETERMINATE"
        assert a["errorCategory"] == ERROR_CATEGORY_CONNECTION_ERROR
        assert a["providerResponseReceived"] is True
        assert a["providerCallId"] == "call-1"


# ---------------------------------------------------------------------------
# Scenario D — retry then success
# ---------------------------------------------------------------------------

class TestRetryThenSuccess:
    def test_retry_then_success_records_two_attempts(self, monkeypatch):
        err = _ApiError(503, "server busy")
        resp = _Resp(text="ok", model_version="gemini-3.6-flash", meta=_UsageMeta(100, 40, 140))
        client, _ = _make_fake(monkeypatch, [err, resp])
        begin_usage_tracking()

        async def _run():
            await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1, "only the successful call records a provider call"
        assert len(attempts) == 2
        assert attempts[0]["attemptId"] == "attempt-1"
        assert attempts[0]["attemptNumber"] == 1
        assert attempts[0]["outcome"] == "INDETERMINATE"
        assert attempts[0]["errorCategory"] == ERROR_CATEGORY_SERVER_ERROR
        assert attempts[0]["httpStatus"] == 503
        assert "providerCallId" not in attempts[0]
        assert attempts[1]["attemptId"] == "attempt-2"
        assert attempts[1]["attemptNumber"] == 2
        assert attempts[1]["outcome"] == "SUCCEEDED"
        assert attempts[1]["providerCallId"] == "call-1"


# ---------------------------------------------------------------------------
# Scenario E — response received but local processing fails
# ---------------------------------------------------------------------------

class TestLocalProcessingFailure:
    def test_tts_no_candidates_is_indeterminate_then_retry_succeeds(self, monkeypatch):
        no_candidates = _Resp(model_version="gemini-3.1-flash-tts-preview", candidates=[])
        client, _ = _make_fake(monkeypatch, [no_candidates, _tts_ok_response()])
        begin_usage_tracking()

        async def _run():
            result = await client.generate_speech(text="hello")
            assert result is not None

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 2
        assert len(attempts) == 2
        a0 = attempts[0]
        assert a0["outcome"] == "INDETERMINATE"
        assert a0["errorCategory"] == ERROR_CATEGORY_LOCAL_PROCESSING
        assert a0["providerResponseReceived"] is True
        assert a0["providerCallId"] == "call-1"
        assert a0["attemptNumber"] == 1
        a1 = attempts[1]
        assert a1["outcome"] == "SUCCEEDED"
        assert a1["attemptNumber"] == 2
        assert a1["providerCallId"] == "call-2"


# ---------------------------------------------------------------------------
# Scenario F — voice: audio ok + TTS fails + gTTS fallback
# ---------------------------------------------------------------------------

class TestVoiceScenarioF:
    def test_voice_audio_success_tts_fail_gtts_fallback(self, monkeypatch):
        from app.api import voice as voice_module

        monkeypatch.setattr(voice_module, "gtts_audio_bytes", lambda text: (b"GFAKE", "audio/mpeg"))

        class _FakeLLM:
            async def generate_with_audio(self, **kw):
                record_provider_call(make_provider_call(
                    requested_model="gemini-3.6-flash",
                    actual_model="gemini-3.6-flash",
                    operation=OP_AUDIO_UNDERSTANDING,
                    usage_source="PROVIDER_RESPONSE",
                    usage_completeness="COMPLETE",
                    inputTokens=100,
                    outputTokens=40,
                    totalTokens=140,
                ))
                record_provider_attempt(make_provider_attempt(
                    operation=OP_AUDIO_UNDERSTANDING,
                    requested_model="gemini-3.6-flash",
                    actual_model="gemini-3.6-flash",
                    attempt_number=1,
                    outcome="SUCCEEDED",
                    provider_call_started=True,
                    provider_call_started_at="2026-08-05T00:00:00+00:00",
                    provider_response_received=True,
                    provider_call_id="call-1",
                ))
                return _Resp(text="Egyptian tourism is wonderful.", model_version="gemini-3.6-flash")

            async def generate_speech(self, text, **kw):
                record_provider_attempt(make_provider_attempt(
                    operation=OP_TEXT_TO_SPEECH,
                    requested_model="gemini-3.1-flash-tts-preview",
                    attempt_number=1,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started=True,
                    provider_call_started_at="2026-08-05T00:00:00+00:00",
                    provider_response_received=True,
                    error_category=ERROR_CATEGORY_LOCAL_PROCESSING,
                ))
                raise RuntimeError("Gemini TTS failed")

        monkeypatch.setattr("app.main.llm_client", _FakeLLM())
        client = TestClient(app)
        resp = client.post(
            "/voice",
            headers=INTERNAL_KEY_HEADERS,
            files={"audio": ("a.mp3", b"\xff\xfb", "audio/mpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["audio_url"] is not None, "gTTS fallback must still produce audio"
        assert len(body["providerCalls"]) == 1
        assert body["providerCalls"][0]["operation"] == OP_AUDIO_UNDERSTANDING
        attempts = body["providerAttempts"]
        assert len(attempts) == 2
        assert attempts[0]["outcome"] == "SUCCEEDED"
        assert attempts[0]["operation"] == OP_AUDIO_UNDERSTANDING
        assert attempts[1]["outcome"] == "INDETERMINATE"
        assert attempts[1]["operation"] == OP_TEXT_TO_SPEECH
        assert attempts[1]["errorCategory"] == ERROR_CATEGORY_LOCAL_PROCESSING


# ---------------------------------------------------------------------------
# Scenario G — identify cache hit
# ---------------------------------------------------------------------------

def _fake_landmark_json(name="Great Pyramid of Giza"):
    return json.dumps({
        "name": name,
        "name_ar": "الهرم الأكبر",
        "description": "The oldest and largest pyramid in the Giza complex.",
        "category": "pyramid",
        "historical_period": "Old Kingdom",
    })


def _install_attempt_fake_client(monkeypatch):
    class _FakeLLMClient:
        async def generate_with_image(
            self,
            system_prompt,
            user_message,
            image_bytes,
            mime_type,
            operation=OP_IMAGE_ANALYSIS,
            _retry_count=0,
        ):
            record_provider_call(make_provider_call(
                requested_model="gemini-3.6-flash",
                actual_model="gemini-3.6-flash",
                operation=OP_IMAGE_ANALYSIS,
                usage_source="PROVIDER_RESPONSE",
                usage_completeness="COMPLETE",
                inputTokens=100,
                outputTokens=40,
                totalTokens=140,
            ))
            record_provider_attempt(make_provider_attempt(
                operation=OP_IMAGE_ANALYSIS,
                requested_model="gemini-3.6-flash",
                actual_model="gemini-3.6-flash",
                attempt_number=1,
                outcome="SUCCEEDED",
                provider_call_started=True,
                provider_call_started_at="2026-08-05T00:00:00+00:00",
                provider_response_received=True,
                provider_call_id="call-1",
            ))
            return _Resp(text=_fake_landmark_json(), model_version="gemini-3.6-flash")

    fake = _FakeLLMClient()
    monkeypatch.setattr("app.main.llm_client", fake)
    return fake


def _identify(client, image_bytes=_IMG):
    return client.post(
        "/identify",
        headers=INTERNAL_KEY_HEADERS,
        files={"image": ("photo.png", image_bytes, "image/png")},
    )


class TestIdentifyScenarioG:
    def _setup(self, monkeypatch):
        identify_module._cache.clear()
        _install_attempt_fake_client(monkeypatch)
        return TestClient(app)

    def test_identify_cache_hit_returns_empty_attempts(self, monkeypatch):
        client = self._setup(monkeypatch)
        first = _identify(client)
        second = _identify(client)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["cached"] is True
        assert second.json()["providerAttempts"] == []

    def test_identify_miss_exposes_succeeded_attempt(self, monkeypatch):
        client = self._setup(monkeypatch)
        resp = _identify(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["cached"] is False
        assert len(body["providerCalls"]) == 1
        attempts = body["providerAttempts"]
        assert len(attempts) == 1
        assert attempts[0]["outcome"] == "SUCCEEDED"
        assert attempts[0]["operation"] == OP_IMAGE_ANALYSIS
        assert attempts[0]["providerCallId"] == "call-1"

    def test_identify_cache_hit_does_not_mutate_stored_entry(self, monkeypatch):
        client = self._setup(monkeypatch)
        _identify(client)
        cache_key = list(identify_module._cache.keys())[0]
        stored = identify_module._cache[cache_key]
        assert "providerAttempts" not in stored
        assert "providerCalls" not in stored
        _identify(client)
        assert identify_module._cache[cache_key] is stored


# ---------------------------------------------------------------------------
# contract-level tests
# ---------------------------------------------------------------------------

class TestAttemptContract:
    def test_make_provider_attempt_omits_unknown_fields(self):
        attempt = make_provider_attempt(
            attempt_number=1,
            outcome="FAILED",
            provider_response_received=False,
        )
        assert attempt["provider"] == PROVIDER_GOOGLE
        assert attempt["attemptNumber"] == 1
        assert attempt["outcome"] == "FAILED"
        assert attempt["providerResponseReceived"] is False
        assert "attemptId" not in attempt
        assert "providerCallId" not in attempt
        assert "errorCategory" not in attempt
        assert "httpStatus" not in attempt
        assert "providerCallStarted" not in attempt

    def test_make_provider_attempt_includes_explicit_fields(self):
        attempt = make_provider_attempt(
            operation=OP_TEXT_CHAT,
            requested_model="gemini-3.6-flash",
            actual_model="gemini-3.6-flash",
            attempt_number=2,
            outcome="INDETERMINATE",
            provider_call_started=True,
            provider_call_started_at="2026-08-05T00:00:00+00:00",
            provider_response_received=True,
            provider_call_id="call-1",
            error_category="TIMEOUT",
            http_status=504,
        )
        assert attempt["providerCallId"] == "call-1"
        assert attempt["errorCategory"] == "TIMEOUT"
        assert attempt["httpStatus"] == 504
        assert attempt["providerCallStarted"] is True
        assert attempt["providerCallStartedAt"] == "2026-08-05T00:00:00+00:00"

    def test_attempt_ids_deterministic_and_sequential(self):
        begin_usage_tracking()
        record_provider_attempt(make_provider_attempt(
            attempt_number=1, outcome="FAILED", provider_response_received=False,
        ))
        record_provider_attempt(make_provider_attempt(
            attempt_number=2, outcome="SUCCEEDED", provider_response_received=True,
        ))
        attempts = consume_attempts()
        assert [a["attemptId"] for a in attempts] == ["attempt-1", "attempt-2"]
        assert [a["attemptNumber"] for a in attempts] == [1, 2]

    def test_consume_usage_and_attempts_returns_both_and_resets(self):
        begin_usage_tracking()
        record_provider_call(make_provider_call(operation=OP_TEXT_CHAT))
        record_provider_attempt(make_provider_attempt(
            attempt_number=1, outcome="SUCCEEDED", provider_response_received=True,
        ))
        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1
        assert len(attempts) == 1
        assert consume_usage_and_attempts() == ([], [])

    def test_record_attempt_without_scope_is_noop(self):
        consume_attempts()
        record_provider_attempt(make_provider_attempt(
            attempt_number=1, outcome="FAILED", provider_response_received=False,
        ))
        assert consume_attempts() == []


# ---------------------------------------------------------------------------
# pre-provider local failures are never attempts
# ---------------------------------------------------------------------------

class TestNoAttemptForLocalFailures:
    def test_no_attempt_when_no_key_available(self, monkeypatch):
        client, _ = _make_fake(monkeypatch, [], key_overrides=_no_key)
        begin_usage_tracking()

        async def _run():
            with pytest.raises(RuntimeError):
                await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert calls == []
        assert attempts == []

    def test_empty_tts_text_produces_no_attempt(self):
        client = GeminiClient(api_keys=["dummy-key"])
        begin_usage_tracking()

        async def _run():
            assert await client.generate_speech(text="") is None

        asyncio.run(_run())
        calls, attempts = consume_usage_and_attempts()
        assert calls == []
        assert attempts == []
