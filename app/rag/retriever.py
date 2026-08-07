import asyncio
import datetime
import structlog
from typing import List, Optional
from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue

import httpx

from app.config import settings
from app.core.usage import (
    ACCOUNTING_SEPARATELY_BILLABLE,
    ATTEMPT_OUTCOME_FAILED,
    ATTEMPT_OUTCOME_INDETERMINATE,
    ATTEMPT_OUTCOME_SUCCEEDED,
    ERROR_CATEGORY_AUTH_ERROR,
    ERROR_CATEGORY_CONNECTION_ERROR,
    ERROR_CATEGORY_INVALID_REQUEST,
    ERROR_CATEGORY_LOCAL_PROCESSING,
    ERROR_CATEGORY_RATE_LIMIT,
    ERROR_CATEGORY_SERVER_ERROR,
    ERROR_CATEGORY_TIMEOUT,
    ERROR_CATEGORY_UNSUPPORTED_OPERATION,
    ERROR_CATEGORY_UNKNOWN,
    OP_EMBEDDING,
    PROVIDER_JINA,
    USAGE_COMPLETENESS_COMPLETE,
    USAGE_COMPLETENESS_PARTIAL,
    USAGE_COMPLETENESS_UNAVAILABLE,
    USAGE_SOURCE_PROVIDER_RESPONSE,
    make_provider_attempt,
    make_provider_call,
    record_provider_attempt,
    record_provider_call,
)

logger = structlog.get_logger()

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
EMBEDDING_MODEL = "jina-embeddings-v4"
EMBED_DIMENSIONS = 1024
EMBED_BATCH_SIZE = 1000
EMBED_MAX_RETRIES = 3

_embedding_client: Optional[httpx.AsyncClient] = None
_embedding_lock = asyncio.Lock()


async def _get_client() -> Optional[httpx.AsyncClient]:
    global _embedding_client
    if not settings.jina_api_key:
        logger.error("JINA_API_KEY is not set — embedding calls will fail")
        return None
    async with _embedding_lock:
        if _embedding_client is None or _embedding_client.is_closed:
            _embedding_client = httpx.AsyncClient(timeout=60.0)
    return _embedding_client


async def _embed_http(inputs: List[str]) -> List[List[float]]:
    client = await _get_client()
    if client is None or not inputs:
        return []

    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": inputs,
        "dimensions": EMBED_DIMENSIONS,
        "embedding_type": "float",
        "normalized": True,
    }

    for attempt in range(EMBED_MAX_RETRIES):
        attempt_number = attempt + 1
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        response = None
        try:
            response = await client.post(JINA_EMBED_URL, json=payload, headers=headers)
            status_code = response.status_code

            if status_code == 429 or status_code >= 500:
                error_cat = ERROR_CATEGORY_RATE_LIMIT if status_code == 429 else ERROR_CATEGORY_SERVER_ERROR
                record_provider_attempt(make_provider_attempt(
                    provider=PROVIDER_JINA,
                    operation=OP_EMBEDDING,
                    requested_model=EMBEDDING_MODEL,
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started=True,
                    provider_call_started_at=started_at,
                    provider_response_received=True,
                    error_category=error_cat,
                    http_status=status_code,
                ))
                wait = 2 ** attempt
                logger.warning(
                    "Embedding API rate/error, retrying",
                    status=status_code,
                    attempt=attempt,
                    wait=wait,
                )
                await asyncio.sleep(wait)
                continue

            if status_code != 200:
                if status_code in (401, 403):
                    error_cat = ERROR_CATEGORY_AUTH_ERROR
                elif status_code == 404:
                    error_cat = ERROR_CATEGORY_UNSUPPORTED_OPERATION
                elif 400 <= status_code < 500:
                    error_cat = ERROR_CATEGORY_INVALID_REQUEST
                else:
                    error_cat = ERROR_CATEGORY_UNKNOWN

                record_provider_attempt(make_provider_attempt(
                    provider=PROVIDER_JINA,
                    operation=OP_EMBEDDING,
                    requested_model=EMBEDDING_MODEL,
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_FAILED,
                    provider_call_started=True,
                    provider_call_started_at=started_at,
                    provider_response_received=True,
                    error_category=error_cat,
                    http_status=status_code,
                ))
                logger.error(
                    "Embedding API error",
                    status=status_code,
                    body=response.text[:300],
                )
                return []

            data = response.json()
            embeddings = [
                item.get("embedding", [])
                for item in data.get("data", [])
            ]
            if len(embeddings) != len(inputs):
                record_provider_attempt(make_provider_attempt(
                    provider=PROVIDER_JINA,
                    operation=OP_EMBEDDING,
                    requested_model=EMBEDDING_MODEL,
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started=True,
                    provider_call_started_at=started_at,
                    provider_response_received=True,
                    error_category=ERROR_CATEGORY_LOCAL_PROCESSING,
                    http_status=200,
                ))
                logger.error("Embedding count mismatch", got=len(embeddings), expected=len(inputs))
                return []

            raw_model = data.get("model")
            actual_model = raw_model if (isinstance(raw_model, str) and raw_model.strip()) else EMBEDDING_MODEL

            usage_dict = data.get("usage") if isinstance(data.get("usage"), dict) else None
            prompt_tokens = usage_dict.get("prompt_tokens") if usage_dict else None
            total_tokens = usage_dict.get("total_tokens") if usage_dict else None

            valid_prompt = isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0
            valid_total = isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens >= 0

            call_counts = {}
            if valid_prompt:
                call_counts["inputTokens"] = prompt_tokens
            if valid_total:
                call_counts["totalTokens"] = total_tokens

            if valid_prompt and valid_total:
                completeness = USAGE_COMPLETENESS_COMPLETE
            elif valid_prompt or valid_total:
                completeness = USAGE_COMPLETENESS_PARTIAL
            else:
                completeness = USAGE_COMPLETENESS_UNAVAILABLE

            provider_call_dict = make_provider_call(
                provider=PROVIDER_JINA,
                requested_model=EMBEDDING_MODEL,
                actual_model=actual_model,
                operation=OP_EMBEDDING,
                usage_source=USAGE_SOURCE_PROVIDER_RESPONSE,
                usage_completeness=completeness,
                accounting_semantics=ACCOUNTING_SEPARATELY_BILLABLE,
                provider_call_made=True,
                **call_counts,
            )

            call_id = record_provider_call(provider_call_dict)

            record_provider_attempt(make_provider_attempt(
                provider=PROVIDER_JINA,
                operation=OP_EMBEDDING,
                requested_model=EMBEDDING_MODEL,
                actual_model=actual_model,
                attempt_number=attempt_number,
                outcome=ATTEMPT_OUTCOME_SUCCEEDED,
                provider_call_started=True,
                provider_call_started_at=started_at,
                provider_response_received=True,
                provider_call_id=call_id,
            ))

            return embeddings

        except Exception as e:
            if response is not None:
                record_provider_attempt(make_provider_attempt(
                    provider=PROVIDER_JINA,
                    operation=OP_EMBEDDING,
                    requested_model=EMBEDDING_MODEL,
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started=True,
                    provider_call_started_at=started_at,
                    provider_response_received=True,
                    error_category=ERROR_CATEGORY_LOCAL_PROCESSING,
                    http_status=response.status_code,
                ))
            else:
                msg = str(e).lower()
                if "timeout" in msg or "timed out" in msg:
                    error_cat = ERROR_CATEGORY_TIMEOUT
                elif any(k in msg for k in ("connection", "reset", "refused", "network", "broken pipe")):
                    error_cat = ERROR_CATEGORY_CONNECTION_ERROR
                else:
                    error_cat = ERROR_CATEGORY_UNKNOWN

                record_provider_attempt(make_provider_attempt(
                    provider=PROVIDER_JINA,
                    operation=OP_EMBEDDING,
                    requested_model=EMBEDDING_MODEL,
                    attempt_number=attempt_number,
                    outcome=ATTEMPT_OUTCOME_INDETERMINATE,
                    provider_call_started=True,
                    provider_call_started_at=started_at,
                    provider_response_received=False,
                    error_category=error_cat,
                ))

            logger.warning("Embedding request exception", error=str(e)[:200], attempt=attempt)
            if attempt < EMBED_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error("Embedding failed after retries")
    return []


async def get_embedding(text: str) -> Optional[List[float]]:
    embeddings = await _embed_http([text])
    return embeddings[0] if embeddings else None


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    results: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        vectors = await _embed_http(batch)
        results.extend(vectors)
    return results


async def retrieve(
    vector_store,
    query: str,
    collection_name: str,
    top_k: int = 5,
    strategy: str = "semantic",
    filters: Optional[dict] = None,
) -> List[dict]:
    query_vec = await get_embedding(query)
    if not query_vec:
        return []

    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if conditions:
            qdrant_filter = QdrantFilter(must=conditions)

    results = await vector_store.search(
        collection_name=collection_name,
        query_vector=query_vec,
        top_k=top_k,
        qdrant_filter=qdrant_filter,
    )

    return [
        {
            "text": r.payload.get("text", "") if r.payload else "",
            "score": r.score,
            "metadata": {k: v for k, v in (r.payload or {}).items() if k != "text"},
        }
        for r in results
    ]


async def retrieve_hybrid(
    vector_store,
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> List[dict]:
    results = await retrieve(vector_store, query, collection_name, top_k=top_k * 2, strategy="semantic")
    seen = set()
    deduped = []
    for r in results:
        text_hash = hash(r.get("text", ""))
        if text_hash not in seen:
            seen.add(text_hash)
            deduped.append(r)
    return deduped[:top_k]


ALL_COLLECTIONS = ["attractions", "monuments", "emergency", "legal", "currency", "scams", "advisories"]
