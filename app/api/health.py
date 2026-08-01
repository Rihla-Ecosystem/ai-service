from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.project_name,
        "version": "0.1.0",
    }


@router.get("/readyz")
async def readyz():
    from app.main import llm_client, vector_store

    status = "ok"
    checks = {}

    if llm_client:
        key_count = len(llm_client.keys)
        active_keys = sum(1 for k in llm_client.keys if k.is_available())
        checks["llm"] = {"status": "ok", "active_keys": active_keys, "total_keys": key_count}
    else:
        checks["llm"] = {"status": "not_initialized"}
        status = "degraded"

    if vector_store:
        try:
            collections = await vector_store.list_collections()
            checks["vector_store"] = {"status": "ok", "collections": collections}
        except Exception as e:
            checks["vector_store"] = {"status": "error", "message": str(e)}
            status = "degraded"
    else:
        checks["vector_store"] = {"status": "not_initialized"}
        status = "degraded"

    return {"status": status, "checks": checks}


@router.get("/health/keys")
async def health_keys():
    from app.main import llm_client

    if not llm_client:
        return {"status": "not_initialized"}

    return {
        "keys": llm_client.get_key_statuses(),
        "total_keys": len(llm_client.keys),
        "available_keys": sum(1 for k in llm_client.keys if k.is_available()),
    }


@router.get("/health/collections")
async def health_collections():
    from app.main import vector_store

    if not vector_store:
        return {"status": "not_initialized", "collections": []}

    try:
        collections = await vector_store.get_collections_info()
        return {
            "status": "ok",
            "collections": collections,
            "total_collections": len(collections),
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "collections": []}
