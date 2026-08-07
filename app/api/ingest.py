import json
import os
import structlog
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, Query
from pydantic import BaseModel

from app.core.auth import allow_access
from app.core.guardrails import check_input
from app.rag.chunking import chunk_text, chunk_json_array, chunk_json_object
from app.rag.retriever import get_embedding
from app.config import settings

logger = structlog.get_logger()

router = APIRouter()

# RAG collections that are safe to write via /ingest. Anything else is rejected
# so a key holder cannot create/poison arbitrary collections.
ALLOWED_COLLECTIONS = {
    "attractions",
    "scams",
    "legal",
    "emergency",
    "currency",
    "advisories",
    "general",
    "uploaded",
}


class IngestResponse(BaseModel):
    collection: str
    chunks_indexed: int
    source_file: str


class CollectionInfo(BaseModel):
    name: str
    points_count: int
    vectors_size: int


class PointResponse(BaseModel):
    id: int
    payload: Dict[str, Any]
    score: Optional[float] = None


class DeleteResponse(BaseModel):
    collection: str
    deleted: bool


def _detect_file_type(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".json":
        try:
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                return "json_array"
            return "json_object"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "unknown"
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext in (".txt",):
        return "text"
    return "unknown"


def _chunk_content(
    content: str,
    source_file: str,
    category: str,
    file_type: str,
) -> List[Dict[str, Any]]:
    if file_type == "json_array":
        data = json.loads(content)
        if isinstance(data, list):
            text_fields = list(
                {k for item in data if isinstance(item, dict) for k in item.keys()}
            )
            return chunk_json_array(data, source_file, category, text_fields)
        return chunk_json_object(data, source_file, category)

    if file_type == "json_object":
        data = json.loads(content)
        return chunk_json_object(data, source_file, category)

    if file_type in ("markdown", "text"):
        return chunk_text(content, source_file, category)

    return []


@router.post("", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    category: str = Form("uploaded"),
    collection: Optional[str] = Form(None),
    user: dict = Depends(allow_access),
):
    from app.main import vector_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file data received")

    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds maximum allowed size")

    source_file = file.filename or "unknown"
    file_type = _detect_file_type(source_file, file_bytes)

    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Provide JSON, Markdown, or Text files.",
        )

    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    chunks = _chunk_content(content, source_file, category, file_type)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks could be extracted from file")

    collection_name = collection or category
    if collection_name.startswith("rihla_"):
        collection_name = collection_name[len("rihla_"):]

    if collection_name not in ALLOWED_COLLECTIONS:
        logger.warning("Ingest rejected: collection not allowed", collection=collection_name)
        raise HTTPException(status_code=403, detail=f"Collection '{collection_name}' is not writable via ingest")

    # RAG data poisoning guard: drop chunks that try to inject instructions.
    clean_chunks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        guard_result = check_input(text)
        if guard_result.blocked:
            logger.warning(
                "Ingest chunk blocked by guardrails",
                collection=collection_name,
                reason=guard_result.reason,
            )
            continue
        clean_chunks.append(chunk)

    if not clean_chunks:
        raise HTTPException(status_code=400, detail="All chunks were rejected by content guardrails")
    chunks = clean_chunks

    points = []
    for i, chunk in enumerate(chunks):
        embedding = await get_embedding(chunk["text"])
        if embedding is None:
            continue
        from qdrant_client.models import PointStruct

        points.append(
            PointStruct(
                id=hash(f"{category}_{i}_{chunk['text'][:50]}") % (2**63),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"],
                },
            )
        )

    if not points:
        raise HTTPException(
            status_code=500, detail="Failed to generate embeddings for any chunk"
        )

    embedding_size = len(points[0].vector)
    await vector_store.ensure_collection(collection_name, size=embedding_size)
    await vector_store.upsert_points(collection_name, points)

    return IngestResponse(
        collection=f"rihla_{collection_name}",
        chunks_indexed=len(points),
        source_file=source_file,
    )


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections(user: dict = Depends(allow_access)):
    from app.main import vector_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    collections = await vector_store.list_collections()
    result = []
    for col_name in collections:
        try:
            info = await vector_store.client.get_collection(col_name)
            result.append(
                CollectionInfo(
                    name=col_name,
                    points_count=info.points_count,
                    vectors_size=info.config.vectors.size,
                )
            )
        except Exception:
            result.append(
                CollectionInfo(
                    name=col_name,
                    points_count=0,
                    vectors_size=0,
                )
            )

    return result


@router.get("/collections/{collection_name}/points", response_model=List[PointResponse])
async def get_points(
    collection_name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user: dict = Depends(allow_access),
):
    from app.main import vector_store
    from qdrant_client.models import ScrollRequest

    if not vector_store or not vector_store.client:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    full_name = f"rihla_{collection_name}" if not collection_name.startswith("rihla_") else collection_name

    try:
        result = await vector_store.client.scroll(
            collection_name=full_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.warning("Collection scroll failed", collection=full_name, error=str(e))
        raise HTTPException(status_code=404, detail="Collection not found")

    points = []
    for point in result[0]:
        points.append(
            PointResponse(
                id=point.id,
                payload=point.payload or {},
            )
        )

    return points


@router.delete("/collections/{collection_name}", response_model=DeleteResponse)
async def delete_collection(
    collection_name: str,
    user: dict = Depends(allow_access),
):
    from app.main import vector_store

    if not vector_store or not vector_store.client:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    if user.get("role") != "admin" or user.get("source") == "internal":
        raise HTTPException(status_code=403, detail="Admin privileges required for deletion")

    full_name = f"rihla_{collection_name}" if not collection_name.startswith("rihla_") else collection_name

    try:
        await vector_store.client.delete_collection(collection_name=full_name)
    except Exception as e:
        logger.warning("Collection delete failed", collection=full_name, error=str(e))
        raise HTTPException(status_code=404, detail="Collection not found")

    return DeleteResponse(collection=full_name, deleted=True)


@router.delete("/collections/{collection_name}/points/{point_id}", response_model=DeleteResponse)
async def delete_point(
    collection_name: str,
    point_id: int,
    user: dict = Depends(allow_access),
):
    from app.main import vector_store

    if not vector_store or not vector_store.client:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    if user.get("role") != "admin" or user.get("source") == "internal":
        raise HTTPException(status_code=403, detail="Admin privileges required for deletion")

    full_name = f"rihla_{collection_name}" if not collection_name.startswith("rihla_") else collection_name

    try:
        await vector_store.client.delete(
            collection_name=full_name,
            points_selector=[point_id],
        )
    except Exception as e:
        logger.warning("Point delete failed", collection=full_name, error=str(e))
        raise HTTPException(status_code=404, detail="Point not found")

    return DeleteResponse(collection=full_name, deleted=True)