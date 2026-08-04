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
    "attractions": {"size": 1024},
    "monuments": {"size": 1024},
    "emergency": {"size": 1024},
    "legal": {"size": 1024},
    "currency": {"size": 1024},
    "scams": {"size": 1024},
    "advisories": {"size": 1024},
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


    async def get_collections_info(self) -> List[dict]:
        """Return each collection with its current point count."""
        if not self.client:
            return []
        collections = await self.client.get_collections()
        result = []
        for c in collections.collections:
            try:
                count = await self.client.count(collection_name=c.name)
                result.append({"name": c.name, "count": count.count})
            except Exception:
                result.append({"name": c.name, "count": 0})
        return result

    async def ensure_collection(self, collection_name: str, size: int = 768):
        if not self.client:
            raise RuntimeError("Vector store not initialized")
        full_name = f"rihla_{collection_name}" if not collection_name.startswith("rihla_") else collection_name
        existing = await self.client.get_collections()
        existing_names = {c.name for c in existing.collections}
        if full_name not in existing_names:
            await self.client.create_collection(
                collection_name=full_name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection", collection=full_name)


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
        result = await self.client.query_points(
            collection_name=f"rihla_{collection_name}",
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return list(result.points)

    async def delete_collection(self, collection_name: str):
        if not self.client:
            return
        await self.client.delete_collection(collection_name=f"rihla_{collection_name}")
        logger.info("Deleted collection", collection=collection_name)

    async def close(self):
        if self.client:
            await self.client.close()
