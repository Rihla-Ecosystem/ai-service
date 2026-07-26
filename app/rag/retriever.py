import structlog
from typing import Any, List, Optional
from google import genai
from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue

from app.config import settings

logger = structlog.get_logger()

_embedding_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _embedding_client
    if _embedding_client is None:
        key = settings.gemini_key_list[0] if settings.gemini_key_list else ""
        _embedding_client = genai.Client(api_key=key)
    return _embedding_client


async def get_embedding(text: str) -> Optional[List[float]]:
    try:
        client = _get_client()
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=[text],
        )
        return result.embeddings[0].values if result.embeddings else None
    except Exception as e:
        logger.error("Embedding failed", error=str(e))
        return None


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
