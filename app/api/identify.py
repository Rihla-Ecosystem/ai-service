import hashlib
import json as json_mod
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.core.guardrails import check_output
from app.core.auth import allow_access

router = APIRouter()

_cache: Dict[str, Any] = {}


class IdentifyResponse(BaseModel):
    name: str
    name_ar: Optional[str] = None
    description: str
    category: Optional[str] = None
    historical_period: Optional[str] = None
    wikipedia_url: Optional[str] = None
    image_url: Optional[str] = None
    nearby_sites: Optional[list] = None
    cached: bool = False


@router.post("", response_model=IdentifyResponse)
async def identify_landmark(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    radius: int = Form(500),
    user: dict = Depends(allow_access),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data received")

    img_hash = hashlib.md5(image_bytes).hexdigest()
    cache_key = f"{img_hash}_{lat}_{lon}"
    if cache_key in _cache:
        cached = _cache[cache_key]
        cached["cached"] = True
        return IdentifyResponse(**cached)

    from app.main import llm_client, vector_store

    if not llm_client:
        raise HTTPException(status_code=503, detail="AI service not initialized")

    nearby_context = ""
    if lat is not None and lon is not None and vector_store:
        try:
            from app.rag.retriever import retrieve
            results = await retrieve(
                vector_store, "tourist attraction landmark", "attractions",
                top_k=5,
            )
            if results:
                names = []
                for r in results:
                    text = r.get("text", "")
                    name = text.split(" | ")[0] if " | " in text else text[:80]
                    names.append(name)
                nearby_context = "Nearby known sites for reference: " + "; ".join(names[:5])
        except Exception as e:
            nearby_context = ""

    system_prompt = (
        "You are an expert Egyptologist. Identify this landmark in Egypt. "
        f"{nearby_context}\n\n"
        "Respond in JSON format with exactly these fields:\n"
        "{\n"
        '  "name": "English name of the landmark",\n'
        '  "name_ar": "Arabic name if known",\n'
        '  "description": "Brief description (2-3 sentences)",\n'
        '  "category": "Type (mosque, temple, museum, pyramid, church, monument, etc.)",\n'
        '  "historical_period": "e.g. Old Kingdom, Ptolemaic, Islamic, Modern",\n'
        '  "wikipedia_url": "Wikipedia URL if known"\n'
        "}\n"
        "Do NOT include markdown formatting, just raw JSON."
    )

    mime_type = image.content_type or "image/jpeg"

    try:
        response = await llm_client.generate_with_image(
            system_prompt=system_prompt,
            user_message="Identify this landmark in Egypt.",
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        text = ""
        if response is not None and hasattr(response, "text") and response.text is not None:
            text = response.text

        guard_result = check_output(text)
        if guard_result.requires_regeneration:
            raise HTTPException(status_code=400, detail="Could not identify this image")

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("\n", 1)[0]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            result = json_mod.loads(cleaned)
        except json_mod.JSONDecodeError:
            result = {"name": "Unknown", "description": text[:200]}

        result["nearby_sites"] = None
        result["cached"] = False
        _cache[cache_key] = result
        if len(_cache) > 100:
            _cache.clear()

        return IdentifyResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")
