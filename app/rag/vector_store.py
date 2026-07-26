import structlog
from typing import Any, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter as QdrantFilter,
    ScoredPoint,
)

logger = structlog.get_logger()

COLLECTION_CONFIG = {
    "attractions": {"size": 768},
    "monuments": {"size": 768},
    "emergency": {"size": 768},
    "legal": {"size": 768},
    "currency": {"size": 768},
    "scams": {"size": 768},
    "advisories": {"size": 768},
}


class VectorStore:
    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.host = host
        self.port = port
        self.client: Optional[AsyncQdrantClient] = None

    async def initialize(self):
        self.client = AsyncQdrantClient(host=self.host, port=self.port, prefer_grpc=True)
        existing = await self.client.get_collections()
        existing_names = {c.name for c in existing.collections}

        for name, config in COLLECTION_CONFIG.items():
            collection_name = f"rihla_{name}"
            if collection_name not in existing_names:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config["size"],
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection", collection=collection_name)

    async def list_collections(self) -> List[str]:
        if not self.client:
            return []
        collections = await self.client.get_collections()
        return [c.name for c in collections.collections]

    async def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
    ):
        if not self.client:
            raise RuntimeError("Vector store not initialized")
        await self.client.upsert(
            collection_name=f"rihla_{collection_name}",
            points=points,
            wait=True,
        )

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        qdrant_filter: Optional[QdrantFilter] = None,
    ) -> List[ScoredPoint]:
        if not self.client:
            raise RuntimeError("Vector store not initialized")
        results = await self.client.search(
            collection_name=f"rihla_{collection_name}",
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return results

    async def delete_collection(self, collection_name: str):
        if not self.client:
            return
        await self.client.delete_collection(collection_name=f"rihla_{collection_name}")
        logger.info("Deleted collection", collection=collection_name)

    async def close(self):
        if self.client:
            await self.client.close()
