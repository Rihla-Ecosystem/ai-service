import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.config import settings

from app.api import (
    admin_assistant,
    chat,
    context,
    health,
    identify,
    ingest,
    itinerary,
    monitoring,
    stream,
    voice,
)

from app.rag.vector_store import VectorStore
from app.core.llm_client import GeminiClient
from app.monitoring.instrument import HttpMetricsMiddleware
from app.monitoring import metrics
from app.monitoring.langfuse import initialize as initialize_langfuse

logger = structlog.get_logger()

vector_store: VectorStore | None = None
llm_client: GeminiClient | None = None


async def _auto_ingest():
    from app.rag.ingestion import ingest_all

    if not vector_store:
        logger.warning("Vector store not ready, skipping auto-ingestion")
        return

    try:
        collections = await vector_store.list_collections()
        populated = False
        for c in collections:
            from qdrant_client import AsyncQdrantClient
            client: AsyncQdrantClient = vector_store.client
            if client:
                info = await client.get_collection(c)
                if info.points_count > 0:
                    populated = True
                    break
        if populated:
            logger.info("RAG collections already populated, skipping ingestion")
            return

        logger.info("Starting background RAG ingestion", data_dir=settings.rag_data_dir)
        await ingest_all(vector_store, settings.rag_data_dir)
        logger.info("RAG ingestion complete")
    except Exception as e:
        logger.warning("Auto-ingestion failed (non-fatal)", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, llm_client
    logger.info("Starting Rihla AI Service", env=settings.environment)

    initialize_langfuse()

    llm_client = GeminiClient(api_keys=settings.gemini_key_list)
    logger.info("Gemini client initialized", key_count=len(settings.gemini_key_list))
    metrics.active_api_keys.set(len(settings.gemini_key_list))

    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    await vector_store.initialize()
    logger.info("Qdrant initialized", collections=await vector_store.list_collections())

    asyncio.create_task(_auto_ingest())

    yield

    logger.info("Shutting down Rihla AI Service")
    if vector_store:
        await vector_store.close()


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(HttpMetricsMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=exc, path=str(request.url))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router, tags=["health"])
app.include_router(monitoring.router, tags=["monitoring"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(stream.router, prefix="/chat", tags=["chat"])
app.include_router(voice.router, prefix="/voice", tags=["voice"])
app.include_router(identify.router, prefix="/identify", tags=["identify"])

app.include_router(itinerary.router, prefix="/itinerary", tags=["itinerary"])

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(admin_assistant.router, prefix="/admin", tags=["admin"])

app.include_router(context.router, tags=["context"])

