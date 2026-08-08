"""Copy all Qdrant collections from the local instance to a remote (cloud) instance.

Idempotent: creates missing collections, upserts all points, preserves IDs and
vector sizes / distance metrics. Run from the repo root:

    .venv/bin/python scripts/migrate_qdrant.py \
        --from-url http://127.0.0.1:6333 \
        --to-url https://<cluster>.cloud.qdrant.io \
        --api-key <key>

Optional flags:
    --batch 32            points per upsert batch (cloud free tier throttles writes)
    --delete-existing     drop remote collections first (destructive)
    --only attractions,scams    limit to specific collection names (no rihla_ prefix)
"""
import argparse
import asyncio

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

LOCAL_URL = "http://127.0.0.1:6333"


async def upsert_with_retry(
    client: AsyncQdrantClient,
    collection: str,
    points: list[PointStruct],
    attempts: int = 3,
    delay: float = 10.0,
) -> None:
    for i in range(attempts):
        try:
            await client.upsert(collection_name=collection, points=points, wait=True)
            return
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1}/{attempts} after error: {str(e)[:120]}")
            await asyncio.sleep(delay)


async def scroll_all(client: AsyncQdrantClient, collection: str, batch: int):
    offset = None
    while True:
        res = await client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        points = res[0]
        for p in points:
            yield p
        next_offset = res[1]
        if next_offset is None or not points:
            break
        offset = next_offset


async def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Qdrant collections local->remote")
    parser.add_argument("--from-url", default=LOCAL_URL, help="Source Qdrant base URL")
    parser.add_argument("--to-url", required=True, help="Destination Qdrant base URL")
    parser.add_argument("--api-key", required=True, help="Destination API key")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds")
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--only", type=str, default="", help="Comma-separated collection names (no rihla_ prefix)")
    args = parser.parse_args()

    src = AsyncQdrantClient(url=args.from_url)
    dst = AsyncQdrantClient(url=args.to_url, api_key=args.api_key, prefer_grpc=False, timeout=args.timeout)

    only = [f"rihla_{n.strip()}" for n in args.only.split(",") if n.strip()]

    src_cols = await src.get_collections()
    names = sorted(c.name for c in src_cols.collections)
    if only:
        names = [n for n in names if n in only]

    if args.delete_existing:
        for name in names:
            try:
                await dst.delete_collection(collection_name=name)
                print(f"deleted remote {name}")
            except Exception as e:
                print(f"delete {name} failed (continuing): {e}")

    for name in names:
        info = await src.get_collection(collection_name=name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", 768)
        distance = getattr(vectors, "distance", None)
        try:
            await dst.get_collection(collection_name=name)
            print(f"{name}: already exists on remote, reusing")
        except Exception:
            await dst.create_collection(
                collection_name=name,
                vectors_config={
                    "size": size,
                    "distance": str(distance.value),
                },
            )
            print(f"{name}: created remote (size={size}, distance={distance})")

        total = 0
        buffer = []
        async for p in scroll_all(src, name, args.batch):
            buffer.append(
                PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload or {},
                )
            )
            if len(buffer) >= args.batch:
                await upsert_with_retry(dst, name, buffer, attempts=3, delay=10.0)
                total += len(buffer)
                buffer = []
                print(f"  {name}: {total} points copied")
        if buffer:
            await upsert_with_retry(dst, name, buffer, attempts=3, delay=10.0)
            total += len(buffer)
        print(f"{name}: done, {total} points total")

    await src.close()
    await dst.close()
    print("migration complete")


if __name__ == "__main__":
    asyncio.run(main())
