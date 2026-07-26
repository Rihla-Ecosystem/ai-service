from typing import Any, Dict, List


def chunk_text(
    text: str,
    source_file: str,
    category: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source_file": source_file,
                "category": category,
                "chunk_index": len(chunks),
            },
        })

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_json_array(
    data: List[Dict[str, Any]],
    source_file: str,
    category: str,
    text_fields: List[str],
) -> List[Dict[str, Any]]:
    chunks = []
    for i, item in enumerate(data):
        text_parts = []
        for field in text_fields:
            value = item.get(field)
            if value:
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    text_parts.append(str(value))
                elif isinstance(value, list):
                    text_parts.append("; ".join(str(v) for v in value))

        combined = " | ".join(text_parts) if text_parts else str(item)
        chunks.append({
            "text": combined,
            "metadata": {
                "source_file": source_file,
                "category": category,
                "chunk_index": i,
                "item_id": item.get("id") or item.get("scam_id") or str(i),
            },
        })

    return chunks


def chunk_json_object(
    data: Dict[str, Any],
    source_file: str,
    category: str,
) -> List[Dict[str, Any]]:
    chunks = []
    for key, value in data.items():
        text = str(value)
        if len(text) > 50:
            chunks.append({
                "text": text,
                "metadata": {
                    "source_file": source_file,
                    "category": category,
                    "section": key,
                },
            })
    return chunks
