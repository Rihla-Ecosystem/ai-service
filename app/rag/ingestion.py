import json
import os
import structlog
from typing import Any, AsyncGenerator, Dict, List, Tuple

from app.rag.chunking import chunk_json_array, chunk_json_object, chunk_text
from app.rag.vector_store import VectorStore

logger = structlog.get_logger()

DATA_SOURCES = {
    "attractions": {
        "path": "Archiological/EgyptAttractions_rag.json",
        "type": "json_array",
        "text_fields": ["name_en", "description", "wikipedia_summary", "type", "category"],
    },
    "monuments": {
        "path": "egymonuments.com.json",
        "type": "json_array",
        "text_fields": ["title", "description", "location", "opening_hours"],
    },
    "emergency": {
        "path": "Emergency_Contacts/Emergency_Contacts.json",
        "type": "json_object",
    },
    "currency": {
        "path": "EG_Curruncy/CurrunciesEG.json",
        "type": "json_object",
    },
    "legal": {
        "path": "Legal_Frameworks_and_Culture_Regulations",
        "type": "json_array",
        "text_fields": ["summary", "sections"],
    },
    "scams": {
        "path": "egypt_scam_scenarios/scams",
        "type": "json_array",
        "text_fields": ["name", "setup", "mechanism", "countermeasure", "hook_dialogue"],
    },
    "advisories": {
        "path": "egypt_travel_advisories",
        "type": "markdown",
    },
}


def read_json_file(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def read_markdown_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def discover_files(rag_dir: str, source_config: dict) -> List[str]:
    path = os.path.join(rag_dir, source_config["path"])
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                files.append(fp)
        return sorted(files)
    return []


async def ingest_all(vector_store: VectorStore, rag_dir: str):
    from app.rag.retriever import get_embedding

    for category, config in DATA_SOURCES.items():
        files = discover_files(rag_dir, config)
        if not files:
            logger.warning("No files found for category", category=category, path=config["path"])
            continue

        all_chunks = []
        for filepath in files:
            rel_path = os.path.relpath(filepath, rag_dir)
            try:
                if config["type"] == "json_array":
                    data = read_json_file(filepath)
                    if isinstance(data, list):
                        chunks = chunk_json_array(data, rel_path, category, config.get("text_fields", []))
                    elif isinstance(data, dict):
                        chunks = chunk_json_object(data, rel_path, category)
                    else:
                        continue
                elif config["type"] == "json_object":
                    data = read_json_file(filepath)
                    chunks = chunk_json_object(data, rel_path, category)
                elif config["type"] == "markdown":
                    text = read_markdown_file(filepath)
                    chunks = chunk_text(text, rel_path, category)
                else:
                    continue

                all_chunks.extend(chunks)
                logger.info("Ingested file", file=rel_path, chunks=len(chunks))
            except Exception as e:
                logger.error("Failed to ingest file", file=rel_path, error=str(e))

        if not all_chunks:
            logger.warning("No chunks for category", category=category)
            continue

        from qdrant_client.models import PointStruct
        from app.rag.retriever import get_embeddings_batch

        vectors = await get_embeddings_batch([chunk["text"] for chunk in all_chunks])
        points = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, vectors)):
            points.append(PointStruct(
                id=hash(f"{category}_{i}_{chunk['text'][:50]}") % (2**63),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"],
                },
            ))

        if points:
            await vector_store.upsert_points(category, points)
            logger.info("Indexed collection", collection=category, points=len(points))
