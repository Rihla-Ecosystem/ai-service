# ai-service — Handoff

> Read relevant section only. Appends 3–6 lines. Prune Changelog > 25.
> Last updated: 2026-08-08

## Current status
- Service UP on 3003 (`.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 3003`; setsid nohup). Health 200.
- Qdrant UP with all 7 collections populated — ingestion skips on boot (expected). Currency collection was re-indexed after a delete test (3 points confirmed).
- Security hardening B1–B9 complete and live (see SECURITY_HARDENING.md status).

## In-progress / next
- C-items (client) + verification done; next session: git-commit the hardening changes across ai-service, Core-Server, rihla-client, and update root CONTEXT.md.

## Timestamp applied (remote main regressions fixed)
- `app/api/voice.py`: added missing `from fastapi import Request` import + `from app.core.rate_limit import enforce_rate_limit` (remote used rate-limit code without imports → NameError on boot).
- `app/main.py`: restored `itinerary` in the `from app.api import ...` line (remote dropped it but kept `app.include_router(itinerary...)`) → NameError on boot.
- Remote added gated endpoints: `/admin/assistant`, `/ingest` (collections CRUD incl. destructive DELETE, auth via `allow_access`), `/metrics`, `/monitoring/*`. CORS tightened from `*` to configurable `cors_origins` (default gge 3000/3001).

## Architecture snapshot
- Lifespan: init GeminiClient (round-robin keys + cooldown) → init VectorStore (auto-create 7 collections) → background `_auto_ingest` (idempotent, non-blocking).
- Supervisor flow: input guardrails → intent detect (keyword scoring) → build system prompt → LLM w/ tools → call_tool → 2nd LLM → output guardrails.
- Tools hit GeoContext/Risk (HTTP) or Qdrant collections. `recommend_itinerary` does multi-city parallel fetch + Gemini composition → markdown + structured JSON (HTML comment).
- Guardrails: military/PII/injection regex (in+out); output often regenerates on military match, redacts PII.

## Key facts
- Config in `app/config.py` (Pydantic). Embeddings via `text-embedding-004` (768-d). Distance cosine.
- FULL request flows: `ARCHITECTURE.md` · Specs: `SPECS.md`.

## Gotchas
- Must have Qdrant up or startup logs RAG missing / `/readyz` degraded.
- Auto-ingest only if collections empty; to force re-ingest delete the `rihla_*` collections then restart service.
- Do not `docker compose up` qdrant (wipes `ai-service_qdrant_data` volume).

## Decision log
- LLM key failover chosen over single-key (round-robin, cooldown, recursion-safe).
- No direct DB access yet — user data passed in request body (per SPECS §15.4).

## Changelog
- 2026-08-08: Merged teammate main (84e69ce) after committing local security work (9437579). Resolved 5 conflicts keeping B-hardening (safety_settings on all 5 configs, generic errors, MAX_RETRIES default 10 + env override) while adopting remote usage telemetry (`app/core/usage.py` + `gemini_usage.py`, `providerCalls`/`providerAttempts` on /identify /voice /chat/stream, new `/analyze` context route mounted at bare `/analyze`). Fixed teammate tests to use settings-internal key. 162/162 tests pass. Merged commit 291703e pushed; service restarted on 3003.
- 2026-08-08: SECURITY HARDENING (B1–B9, per `Modules/SECURITY_HARDENING.md`) — untrusted-data delimiters for tool/RAG results (B1), client context out of system_instruction (B2), tool-call cap + per-persona gating + arg validation + 4-city cap (B3), ingest allow-list + check_input + size cap + admin-only deletes (B4), raw str(e) removed (B5), /readyz+/health/keys+/health/collections behind allow_access (B6), Gemini safety_settings (B7), internal-gateway throttled + no XFF trust + MAX_RETRIES 3 (B8), compare_digest + fail-fast weak secrets + user-JWT precedence over internal key (B9). Tests 24/25 (pre-existing tools>=9 still fails, 8 tools). E2E live: /chat injection blocked, /chat normal OK, protected health 401 without key / 200 with key, deletes 403 with key-only.
- 2026-08-07: Created `AGENTS.md` + `CONTEXT.md`. Qdrant collections probed → all 7 present.
- 2026-08-07: Merged origin/main + fixed 2 import regressions (voice rate-limit, main itinerary). Tests 24/25 (1 pre-existing tools>=9 fails, 8 tools defined). E2E: /chat (18s gemini tool calls), /chat/stream SSE, /admin/assistant, /ingest/collections, /metrics all OK.