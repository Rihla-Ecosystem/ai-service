from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import allow_access
from app.core.guardrails import check_output
from app.core.llm_client import OP_TEXT_GENERATION

router = APIRouter(prefix="/legal", tags=["legal"])

# Zone class -> RAG legal query. Identity of the specific area is NEVER passed;
# only the broad class, keeping alerting anonymous.
CLASS_QUERIES: "dict[str, str]" = {
    "restricted": "military restricted zones drone no-fly photography prohibition law",
    "caution": "public photography filming drone regulations Egypt law",
    "protected": "antiquities protection heritage photography rules protected areas law",
}

ALLOWED_CLASSES = {"restricted", "caution", "protected"}


class LegalRule(BaseModel):
    heading: str
    points: List[str] = []


class LegalResponse(BaseModel):
    source: str  # "rag" or "ai"
    class_name: str
    title: str
    summary: str
    rules: List[LegalRule] = []
    citations: List[str] = []
    advice: Optional[str] = None


def _require_vector_store():
    from app.main import vector_store

    if not vector_store:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    return vector_store


@router.get("", response_model=LegalResponse)
async def legal_guidelines(
    class_name: str = Query(...),
    synthesize: bool = Query(False, description="Compose a short AI advice summary"),
    user: dict = Depends(allow_access),
):
    if class_name not in ALLOWED_CLASSES:
        raise HTTPException(status_code=400, detail=f"class must be one of {sorted(ALLOWED_CLASSES)}")

    from app.rag.retriever import retrieve

    vector_store = _require_vector_store()
    query = CLASS_QUERIES[class_name]

    results = await retrieve(vector_store, query, "legal", top_k=4)
    if not results:
        return LegalResponse(
            source="rag",
            class_name=class_name,
            title="Legal guidance unavailable",
            summary="No matching legal guidance found in the knowledge base.",
            rules=[],
        )

    # Build deterministic rules from retrieved RAG chunks (the file). The
    # `legal` collection stores one chunk per top-level JSON key with the raw
    # string in `text` and `metadata.section` naming the key. The `sections`
    # chunk contains a serialized list of {heading, content[]} blocks.
    rules: List[LegalRule] = []
    citations: List[str] = []
    texts: List[str] = []

    def _add_sections_value(value: object) -> None:
        """Turn a `sections` value (list of {heading, content}) into rules."""
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading") or item.get("title") or "Guidance")
            content = item.get("content")
            if isinstance(content, list):
                points = [str(c) for c in content if str(c).strip()]
            elif isinstance(content, str):
                points = [content]
            else:
                points = []
            if points:
                rules.append(LegalRule(heading=heading, points=points))
                texts.append(f"{heading}: " + " ".join(points))

    used_sections = False
    for r in results:
        payload = r.get("metadata") or {}
        section = payload.get("section")
        text = r.get("text") or ""
        if text:
            texts.append(text)

        if section == "sections":
            value = None
            try:
                import json

                value = json.loads(text)
            except Exception:
                try:
                    import ast

                    value = ast.literal_eval(text)
                except Exception:
                    value = None
            if value is not None:
                _add_sections_value(value)
                used_sections = True
            else:
                # fall through: treat the raw text as a single guidance block
                if text and text.strip():
                    rules.append(LegalRule(heading="Guidance", points=[text]))
                used_sections = True
        elif text and text.strip():
            heading = str(section or "Guidance").capitalize()
            rules.append(LegalRule(heading=heading, points=[text]))

        src = payload.get("source_file") or payload.get("title") or section or ""
        if src and src not in citations:
            citations.append(str(src))

    if not used_sections and not rules and texts:
        # No structured blocks were recovered — surface the raw excerpts so the
        # client still gets useful, anonymous legal context.
        rules.append(LegalRule(heading="Key points", points=[t for t in texts if t.strip()]))

    summary = (
        "Egyptian laws and regulations for this zone class. "
        "Specifics may vary; when in doubt follow the onsite authority."
    )

    advice: Optional[str] = None
    source = "rag"
    if synthesize and texts:
        try:
            from app.main import llm_client

            if llm_client is None:
                raise HTTPException(status_code=503, detail="AI service not initialized")

            system = (
                "You are Rihla's anonymous safety advisor. The user is in or near a "
                f"'{class_name}' area. Using ONLY the supplied Egyptian law excerpts, write "
                "3-4 short, practical, neutral sentences on what to be careful about "
                "(drone use, photography, entry, or specialized rules). Never reveal or "
                "guess the name, description, or location of any specific area. Respond "
                "in plain text only, no markdown."
            )
            user_message = "Excerpts:\n" + "\n\n".join(texts)

            response = await llm_client.generate(
                system_prompt=system,
                user_message=user_message,
                temperature=0.3,
                operation=OP_TEXT_GENERATION,
            )
            text = ""
            if response is not None and hasattr(response, "text") and response.text:
                text = response.text.strip()

            if text:
                guard = check_output(text)
                if not guard.requires_regeneration:
                    advice = text
                    source = "ai"
        except HTTPException:
            raise
        except Exception:
            advice = None
            source = "rag"

    return LegalResponse(
        source=source,
        class_name=class_name,
        title="Egyptian Law & Safety Guidance",
        summary=summary,
        rules=rules,
        citations=citations,
        advice=advice,
    )