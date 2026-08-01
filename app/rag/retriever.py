import asyncio
import structlog
from typing import List, Optional
from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue

import httpx

from app.config import settings

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
        try:
            response = await client.post(JINA_EMBED_URL, json=payload, headers=headers)
            if response.status_code == 429 or response.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Embedding API rate/error, retrying",
                    status=response.status_code,
                    attempt=attempt,
                    wait=wait,
                )
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                logger.error(
                    "Embedding API error",
                    status=response.status_code,
                    body=response.text[:300],
                )
                return []
            data = response.json()
            embeddings = [
                item.get("embedding", [])
                for item in data.get("data", [])
            ]
            if len(embeddings) != len(inputs):
                logger.error("Embedding count mismatch", got=len(embeddings), expected=len(inputs))
                return []
            return embeddings
        except Exception as e:
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
