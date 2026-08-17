import asyncio
import io
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.usage import (
    ATTEMPT_OUTCOME_FAILED,
    ATTEMPT_OUTCOME_INDETERMINATE,
    ATTEMPT_OUTCOME_SUCCEEDED,
    ERROR_CATEGORY_LOCAL_PROCESSING,
    OP_EMBEDDING,
    OP_IMAGE_ANALYSIS,
    PROVIDER_GOOGLE,
    PROVIDER_JINA,
    begin_usage_tracking,
    consume_usage_and_attempts,
    make_provider_attempt,
    make_provider_call,
    record_provider_attempt,
    record_provider_call,
)
from app.rag.retriever import _embed_http, EMBEDDING_MODEL
from app.config import settings


@pytest.fixture(autouse=True)
def setup_jina_key():
    with patch.object(settings, "jina_api_key", "test-jina-key"):
        yield


def test_successful_jina_embedding_in_active_scope():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {
                    "prompt_tokens": 25,
                    "total_tokens": 25,
                },
                "data": [{"embedding": [0.1, 0.2]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        calls, attempts = consume_usage_and_attempts()

        assert len(calls) == 1
        jina_call = calls[0]
        assert jina_call["provider"] == PROVIDER_JINA
        assert jina_call["operation"] == OP_EMBEDDING
        assert jina_call["requestedModel"] == EMBEDDING_MODEL
        assert jina_call["actualModel"] == "jina-embeddings-v4"
        assert jina_call["inputTokens"] == 25
        assert jina_call["totalTokens"] == 25
        assert jina_call["providerCallMade"] is True
        assert jina_call["usageSource"] == "PROVIDER_RESPONSE"
        assert jina_call["usageCompleteness"] == "COMPLETE"
        assert jina_call["accountingSemantics"] == "SEPARATELY_BILLABLE"

        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt["provider"] == PROVIDER_JINA
        assert attempt["operation"] == OP_EMBEDDING
        assert attempt["outcome"] == ATTEMPT_OUTCOME_SUCCEEDED
        assert attempt["providerResponseReceived"] is True
        assert attempt["providerCallId"] == jina_call["providerCallId"]

    asyncio.run(_test())


def test_total_tokens_only_surface_is_billable_input():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {
                    "total_tokens": 9,
                },
                "data": [{"embedding": [0.1, 0.2]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        calls, _ = consume_usage_and_attempts()

        assert len(calls) == 1
        jina_call = calls[0]
        assert jina_call["provider"] == PROVIDER_JINA
        assert jina_call["operation"] == OP_EMBEDDING
        assert jina_call["inputTokens"] == 9
        assert jina_call["totalTokens"] == 9
        assert jina_call["providerCallMade"] is True
        assert jina_call["usageSource"] == "PROVIDER_RESPONSE"
        assert jina_call["usageCompleteness"] == "PARTIAL"
        assert jina_call["accountingSemantics"] == "SEPARATELY_BILLABLE"

    asyncio.run(_test())


def test_batch_request_produces_single_provider_call():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {
                    "prompt_tokens": 50,
                    "total_tokens": 50,
                },
                "data": [{"embedding": [0.1]}, {"embedding": [0.2]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["query one", "query two"])

        assert len(embeddings) == 2
        calls, _ = consume_usage_and_attempts()
        assert len(calls) == 1
        assert calls[0]["inputTokens"] == 50

    asyncio.run(_test())


def test_missing_provider_usage_does_not_fabricate_zeros():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        calls, _ = consume_usage_and_attempts()

        assert len(calls) == 1
        jina_call = calls[0]
        assert "inputTokens" not in jina_call
        assert "totalTokens" not in jina_call
        assert jina_call["usageCompleteness"] == "UNAVAILABLE"

    asyncio.run(_test())


def test_matching_succeeded_provider_attempt_linked():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
                "data": [{"embedding": [0.1]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            await _embed_http(["query"])

        calls, attempts = consume_usage_and_attempts()
        assert len(calls) == 1
        assert len(attempts) == 1
        assert attempts[0]["outcome"] == ATTEMPT_OUTCOME_SUCCEEDED
        assert attempts[0]["providerCallId"] == calls[0]["providerCallId"]

    asyncio.run(_test())


def test_provider_http_failure_produces_attempts_without_provider_call():
    async def _test():
        mock_response = httpx.Response(401, text="Unauthorized")

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["test query"])

        assert embeddings == []
        calls, attempts = consume_usage_and_attempts()

        assert len(calls) == 0
        assert len(attempts) == 1
        assert attempts[0]["outcome"] == ATTEMPT_OUTCOME_FAILED
        assert attempts[0]["errorCategory"] == "AUTH_ERROR"
        assert attempts[0]["httpStatus"] == 401

    asyncio.run(_test())


def test_identify_cache_miss_includes_jina_and_gemini_calls():
    async def _test():
        from app.api.identify import identify_landmark, _cache
        from fastapi import UploadFile

        _cache.clear()

        mock_jina_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": 15, "total_tokens": 15},
                "data": [{"embedding": [0.1] * 1024}],
            },
        )

        mock_gemini_resp = MagicMock()
        mock_gemini_resp.text = '{"name": "Pyramids", "description": "Ancient pyramids"}'

        async def fake_generate_with_image(*args, **kwargs):
            call_id = record_provider_call(
                make_provider_call(
                    provider=PROVIDER_GOOGLE,
                    requested_model="gemini-2.5-flash",
                    actual_model="gemini-2.5-flash",
                    operation=OP_IMAGE_ANALYSIS,
                    provider_call_made=True,
                    usage_source="PROVIDER_RESPONSE",
                    usage_completeness="COMPLETE",
                    inputTokens=100,
                    outputTokens=50,
                    totalTokens=150,
                )
            )
            record_provider_attempt(
                make_provider_attempt(
                    provider=PROVIDER_GOOGLE,
                    operation=OP_IMAGE_ANALYSIS,
                    requested_model="gemini-2.5-flash",
                    actual_model="gemini-2.5-flash",
                    attempt_number=1,
                    outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                    provider_call_started=True,
                    provider_call_started_at="2026-08-07T12:00:00Z",
                    provider_response_received=True,
                    provider_call_id=call_id,
                )
            )
            return mock_gemini_resp

        mock_vector_store = AsyncMock()
        mock_vector_store.search = AsyncMock(return_value=[
            MagicMock(payload={"text": "Giza Pyramids | Monument"}, score=0.9)
        ])

        fake_image = UploadFile(filename="test.jpg", file=io.BytesIO(b"fake_image_bytes"))

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_jina_response), \
             patch("app.main.vector_store", mock_vector_store), \
             patch("app.main.llm_client") as mock_llm:
            mock_llm.generate_with_image = AsyncMock(side_effect=fake_generate_with_image)

            response = await identify_landmark(
                image=fake_image,
                lat=29.9792,
                lon=31.1342,
                user={"sub": "user1"},
            )

        assert response.cached is False
        assert response.providerCalls is not None
        assert len(response.providerCalls) == 2

        jina_calls = [c for c in response.providerCalls if c["provider"] == PROVIDER_JINA]
        google_calls = [c for c in response.providerCalls if c["provider"] == PROVIDER_GOOGLE]

        assert len(jina_calls) == 1
        assert len(google_calls) == 1

        assert response.providerAttempts is not None
        assert len(response.providerAttempts) == 2

    asyncio.run(_test())


def test_identify_cache_hit_returns_empty_calls_and_no_http_work():
    async def _test():
        from app.api.identify import identify_landmark, _cache
        from fastapi import UploadFile

        _cache.clear()
        fake_image_bytes = b"cache_test_image_bytes"

        mock_jina_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
                "data": [{"embedding": [0.1] * 1024}],
            },
        )
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.text = '{"name": "Karnak", "description": "Karnak Temple"}'

        mock_vector_store = AsyncMock()
        mock_vector_store.search = AsyncMock(return_value=[])

        # First call: Cache miss
        fake_image1 = UploadFile(filename="test.jpg", file=io.BytesIO(fake_image_bytes))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_jina_response), \
             patch("app.main.vector_store", mock_vector_store), \
             patch("app.main.llm_client") as mock_llm:
            mock_llm.generate_with_image = AsyncMock(return_value=mock_gemini_resp)
            first_resp = await identify_landmark(image=fake_image1, lat=25.7188, lon=32.6573, user={"sub": "u1"})

        assert first_resp.cached is False

        # Second call: Cache hit (same image, lat, lon)
        fake_image2 = UploadFile(filename="test.jpg", file=io.BytesIO(fake_image_bytes))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
             patch("app.main.llm_client") as mock_llm:
            mock_llm.generate_with_image = AsyncMock()
            second_resp = await identify_landmark(image=fake_image2, lat=25.7188, lon=32.6573, user={"sub": "u1"})

            assert mock_post.call_count == 0
            assert mock_llm.generate_with_image.call_count == 0

        assert second_resp.cached is True
        assert second_resp.providerCalls == []
        assert second_resp.providerAttempts == []

    asyncio.run(_test())


def test_jina_retry_429_then_200():
    async def _test():
        resp_429 = httpx.Response(429, text="Rate limited")
        resp_200 = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": 15, "total_tokens": 15},
                "data": [{"embedding": [0.1]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=[resp_429, resp_200]), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        assert mock_sleep.call_count == 1

        calls, attempts = consume_usage_and_attempts()

        assert len(calls) == 1
        jina_call = calls[0]
        assert jina_call["provider"] == PROVIDER_JINA

        assert len(attempts) == 2

        att1 = attempts[0]
        assert att1["attemptNumber"] == 1
        assert att1["outcome"] == ATTEMPT_OUTCOME_INDETERMINATE
        assert att1["providerResponseReceived"] is True
        assert att1["httpStatus"] == 429

        att2 = attempts[1]
        assert att2["attemptNumber"] == 2
        assert att2["outcome"] == ATTEMPT_OUTCOME_SUCCEEDED
        assert att2["providerResponseReceived"] is True
        assert att2["providerCallId"] == jina_call["providerCallId"]

    asyncio.run(_test())


def test_negative_provider_token_counts_omitted():
    async def _test():
        # Case 1: prompt_tokens negative, total_tokens positive
        mock_response_partial = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": -10, "total_tokens": 25},
                "data": [{"embedding": [0.1]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response_partial):
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        calls, _ = consume_usage_and_attempts()
        assert len(calls) == 1
        call1 = calls[0]
        assert call1["inputTokens"] == 25
        assert call1["totalTokens"] == 25
        assert call1["usageCompleteness"] == "PARTIAL"

        # Case 2: both negative
        mock_response_all_neg = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": -10, "total_tokens": -20},
                "data": [{"embedding": [0.1]}],
            },
        )

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response_all_neg):
            embeddings = await _embed_http(["test query"])

        assert len(embeddings) == 1
        calls, _ = consume_usage_and_attempts()
        assert len(calls) == 1
        call2 = calls[0]
        assert "inputTokens" not in call2
        assert "totalTokens" not in call2
        assert call2["usageCompleteness"] == "UNAVAILABLE"

    asyncio.run(_test())


def test_malformed_json_response_records_provider_response_received_true():
    async def _test():
        mock_response = httpx.Response(200, text="invalid json {{{")

        begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            embeddings = await _embed_http(["test query"])

        assert embeddings == []
        calls, attempts = consume_usage_and_attempts()

        assert len(calls) == 0
        assert len(attempts) == 3

        for att in attempts:
            assert att["outcome"] == ATTEMPT_OUTCOME_INDETERMINATE
            assert att["providerResponseReceived"] is True
            assert att["httpStatus"] == 200
            assert att["errorCategory"] == ERROR_CATEGORY_LOCAL_PROCESSING

    asyncio.run(_test())


def test_no_active_tourist_scope_does_not_accumulate_usage_or_attempts():
    async def _test():
        mock_response = httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v4",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
                "data": [{"embedding": [0.1, 0.2]}],
            },
        )

        # Do NOT call begin_usage_tracking()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            embeddings = await _embed_http(["ingest doc text"])

        assert len(embeddings) == 1
        calls, attempts = consume_usage_and_attempts()
        assert calls == []
        assert attempts == []

    asyncio.run(_test())
