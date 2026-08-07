# ai-service — Context

> Auto-loaded when working here. Keep SHORT.
> Mid-task state: `read CONTEXT.md` (relevant section) first.

## What
FastAPI + Google Gemini + Qdrant. 3 personas (tour_guide / local_expert / safety_guru), 9 agent tools, RAG (7 collections, 768-d), guardrails, landmark identify, voice, SSE streaming, round-robin LLM key failover.

## Run / test
- Dev: `.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 3003 --reload` (needs Qdrant up)
- Tests: `.venv/bin/pytest -v` (test_guardrails, test_tools)
- Types: `mypy app`
- Manual re-ingest: `python -c "import asyncio; from app.rag.ingestion import ingest_all; from app.main import vector_store; asyncio.run(ingest_all(vector_store, 'data/rag'))"`

## External contract
- Called by Core-Server: `POST /chat`, `/chat/stream`, `/voice`, `/identify` (JWT or `X-Internal-Api-Key`)
- Calls GeoContext `/api/v1/nearby-sites` + `/api/v1/context`; Risk `/safety/current` (10s timeouts)
- Qdrant gRPC at `qdrant:6333` (env `QDRANT_HOST`/`QDRANT_PORT`)
- `GEMINI_API_KEYS` comma-separated; on 429/500 marks key degraded → 60s cooldown → auto-revive

## Key files
- `app/main.py` (lifespan: init LLM + vector store + background auto-ingest)
- `app/agent/supervisor.py` · `app/agent/tools.py` (9 tools) · `app/core/llm_client.py` (failover)
- `app/core/guardrails.py` · `app/core/system_prompt.py`
- `app/rag/vector_store.py` · `app/rag/ingestion.py` · `app/rag/retriever.py`
- `data/rag/` — 44 source files → 7 collections

## Standing rules (enforced reflex)
1. At the end of every task, append a 3–6 line checkpoint to this module's `CONTEXT.md`.
2. At session start, `read` the needed `CONTEXT.md` section before working.
3. Only read sections you need — never dump whole files into replies.
4. Never commit/log `.env` secrets. Match `JWT_ACCESS_SECRET` + `INTERNAL_API_KEY` across services.