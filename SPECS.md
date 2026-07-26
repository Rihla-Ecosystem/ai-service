# Rihla AI Service — Specifications

## Overview

The AI Service is the 4th microservice of the Rihla platform. It provides intelligent conversational capabilities, landmark identification, voice interaction, and safety-oriented guidance to tourists in Egypt. Built from scratch using Python/FastAPI with Google Gemini as the primary LLM.

---

## 1. Architecture

### Service Layer
```
┌──────────────┐     HTTP/JSON      ┌──────────────────┐
│   Client     │ ◄──────────────►   │   AI Service     │
│ (App/Browser)│    port 3003       │   (FastAPI)       │
└──────────────┘                    └────────┬─────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
             ┌──────────┐           ┌──────────────┐        ┌──────────────┐
             │  Qdrant  │           │  GeoContext   │        │Risk_Intellig.│
             │ (Vector) │           │  (port 8000)  │        │ (port 3001)  │
             └──────────┘           └──────────────┘        └──────────────┘
```

### Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Runtime | Python | 3.11+ |
| Web framework | FastAPI | 0.115+ |
| LLM | Google Gemini 2.0 Flash | via `google-genai` SDK |
| Orchestration | LangChain | 0.3+ |
| Vector DB | Qdrant | latest (Docker) |
| Embeddings | Gemini text-embedding-004 | via `langchain-google-genai` |
| Monitoring | Langfuse | 2.55+ |
| Metrics | Prometheus | via `prometheus-client` |
| Container | Docker | multi-stage Python 3.11-slim |

### Ports

| Service | Port |
|---------|------|
| AI Service | 3003 |
| Qdrant (gRPC) | 6333 |
| Qdrant (HTTP) | 6334 |

---

## 2. API Endpoints

### 2.1 Chat — `POST /chat`
Primary conversational endpoint. Accepts user message with optional context.

**Request:**
```json
{
  "message": "What are the best things to do in Luxor?",
  "conversation_id": "uuid-optional",
  "persona": "auto | tour_guide | local_expert | safety_guru",
  "lat": 25.6872,
  "lon": 32.6396,
  "user": {
    "display_name": "Ahmed",
    "gender": "MALE",
    "nationality": "US",
    "language": ["en"],
    "budget_level": "mid",
    "travel_style": "cultural",
    "interests": ["history", "photography"],
    "accommodation_type": "hotel",
    "preferences": {}
  },
  "environment": {
    "weather": {},
    "airQuality": {},
    "prayerTimes": {}
  },
  "geography": {
    "pois": [],
    "route": {},
    "geocode": {}
  },
  "safety": {},
  "user_journeys": {}
}
```

**Response:**
```json
{
  "response": "Great choice! Luxor has...",
  "conversation_id": "uuid",
  "persona": "tour_guide",
  "blocked": false,
  "reason": null
}
```

### 2.2 Chat Stream — `POST /chat/stream`
SSE streaming version. Same request body as `/chat`. Returns `text/event-stream`:
```
data: {"token": "Luxor"}
data: {"token": " is"}
data: {"token": " home"}
...
data: {"done": true, "full_response": "Luxor is home to..."}
data: [DONE]
```

### 2.3 Voice — `POST /voice`
Audio input, text (optional audio) output. Uses Gemini native audio understanding.

**Request:** `multipart/form-data`
- `audio`: audio file (WAV/MP3/OGG)
- `lat` (optional): number
- `lon` (optional): number
- `conversation_id` (optional): string

**Response:**
```json
{
  "text_response": "I heard your question about...",
  "audio_response": null,
  "conversation_id": "uuid"
}
```

### 2.4 Identify Landmark — `POST /identify`
Image + optional GPS coordinates → landmark identification with cross-referencing.

**Request:** `multipart/form-data`
- `image`: image file (JPEG/PNG)
- `lat` (optional): number
- `lon` (optional): number
- `radius` (optional, default 500): meters

**Response:**
```json
{
  "name": "Mosque of Muhammad Ali",
  "name_ar": "مسجد محمد علي",
  "description": "A stunning Ottoman-era mosque located within the Cairo Citadel...",
  "category": "mosque",
  "historical_period": "ottoman",
  "wikipedia_url": "https://en.wikipedia.org/wiki/Muhammad_Ali_Mosque",
  "image_url": null,
  "nearby_sites": null,
  "cached": false
}
```

### 2.5 Health — `GET /health`
```json
{
  "status": "ok",
  "service": "Rihla AI Service",
  "version": "0.1.0"
}
```

### 2.6 Readiness — `GET /readyz`
```json
{
  "status": "ok",
  "checks": {
    "llm": {"status": "ok", "active_keys": 2, "total_keys": 3},
    "vector_store": {"status": "ok", "collections": ["rihla_attractions", "rihla_monuments", ...]}
  }
}
```

### 2.7 Key Health — `GET /health/keys`
```json
{
  "keys": [
    {"key_suffix": "abcd", "status": "active", "fail_count": 0},
    {"key_suffix": "efgh", "status": "cooldown", "fail_count": 2}
  ],
  "total_keys": 3,
  "available_keys": 2
}
```

---

## 3. Personas & Routing

### 3.1 Auto-Detection
When `persona: "auto"`, the supervisor detects intent from message keywords:

| Persona | Triggers |
|---------|----------|
| `tour_guide` | history, attraction, site, museum, pyramid, temple, tour, visit, ancient, pharaoh, culture |
| `safety_guru` | safe, danger, scam, risk, emergency, warning, police, hospital, crime, protect |
| `local_expert` | food, eat, restaurant, shop, bargain, local, insider, hidden, authentic, custom |

### 3.2 Persona Details

#### Tour Guide
- **Identity**: Enthusiastic Egyptologist / professional tour guide
- **Tools**: `search_attractions`, `get_nearby_attractions`, `get_currency_info`, `get_legal_guidelines`, `recommend_itinerary`
- **Access**: RAG collections (attractions, monuments, legal)

#### Local Expert
- **Identity**: Friendly local Egyptian
- **Tools**: `get_scam_warnings`, `get_currency_info`, `get_legal_guidelines`, `get_nearby_attractions`
- **Access**: RAG collections (scams, currency, legal)
- **Tone**: Warm, practical, occasionally Arabic phrases

#### Safety Guru
- **Identity**: Proactive but calm safety advisor
- **Tools**: `get_safety_info`, `get_emergency_contacts`, `get_scam_warnings`, `get_legal_guidelines`
- **Access**: RAG collections (emergency, scams, legal)
- **Tone**: Informative, never alarmist — "be aware" not "be scared"

### 3.3 Hard Rules (applied to ALL personas)
1. NEVER reveal, mention, or describe military sites, restricted zones, or security installations
2. NEVER speak negatively about Egypt, its people, culture, or government
3. Frame challenges factually: "be aware that..." not "this is dangerous/scary"
4. If asked about restricted data, politely redirect to tourist-appropriate topics
5. Always represent Egypt positively — tourism ambassador
6. When discussing scams, always pair the warning with a concrete countermeasure
7. Do not invent or hallucinate information

---

## 4. Agent System

### 4.1 Tools (9 total)

| Tool | Input | Output | Backend |
|------|-------|--------|---------|
| `get_nearby_attractions` | lat, lon, radius | List of nearby sites | GeoContext HTTP |
| `search_attractions` | query, category, city | Ranked attractions | Qdrant RAG |
| `get_safety_info` | city | Risk state | Risk_Intelligence HTTP |
| `get_emergency_contacts` | context_type | Phone numbers + procedures | Qdrant RAG |
| `get_legal_guidelines` | topic | Egyptian laws | Qdrant RAG |
| `get_currency_info` | denomination, base_currency | EGP details + rates | Qdrant RAG |
| `get_scam_warnings` | category, severity | Scam scenarios + countermeasures | Qdrant RAG |
| `recommend_itinerary` | interests, days, budget, cities, style, base_currency | Markdown + structured JSON itinerary | Multi-tool + Gemini |

### 4.2 Multi-Agent Architecture
```
User Message
    │
    ▼
Supervisor Agent
    ├── Intent detection (keyword-based scoring)
    ├── Guardrails check (input)
    ├── Gemini function calling with tool definitions
    ├── Tool execution (if function call triggered)
    ├── Gemini text generation (with tool results)
    ├── Guardrails check (output)
    └── Response
```

If persona is `auto`, supervisor detects intent and routes. If a specific persona is requested, it uses that agent directly with its restricted tool set.

### 4.3 Error Handling
- Tool call timeout: 10 seconds
- Retry: 2 attempts on transient failures
- Degradation: agent failure → supervisor falls back to direct Gemini response
- All errors logged to Langfuse

---

## 5. LLM Token Failover

### 5.1 Configuration
```env
GEMINI_API_KEYS=key1,key2,key3
```

### 5.2 Behavior
- **Round-robin** across all keys for even distribution
- **On 429 (rate limit)**: mark key `degraded`, immediate switch to next
- **On 500**: mark key `degraded`, switch to next, retry once
- **On success**: reset fail count, mark as `active`
- **Cooldown**: degraded keys wait 60s before automatic reactivation
- **Recursion safety**: if all keys are degraded and recursive call happens, raises `RuntimeError`

### 5.3 Gemini Methods

| Method | Use Case | Model |
|--------|----------|-------|
| `generate()` | Text-only chat | gemini-2.0-flash |
| `generate_with_tools()` | Function calling | gemini-2.0-flash |
| `generate_with_image()` | Landmark identification | gemini-2.0-flash (vision) |
| `generate_with_audio()` | Voice understanding | gemini-2.0-flash (audio) |

---

## 6. RAG Pipeline

### 6.1 Data Sources (8 categories, ~30+ files)

| Collection | Source Files | Format | Chunk Method |
|-----------|-------------|--------|-------------|
| `rihla_attractions` | `Archiological/EgyptAttractions_rag.json` | JSON array | Per-object, text fields |
| `rihla_monuments` | `egymonuments.com.json` | JSON array | Per-object, text fields |
| `rihla_emergency` | `Emergency_Contacts/Emergency_Contacts.json` | JSON object | Per-key section |
| `rihla_legal` | `Legal_Frameworks/*.json` (6 files) | JSON array | Per-object, text fields |
| `rihla_currency` | `EG_Curruncy/CurrunciesEG.json` | JSON object | Per-key section |
| `rihla_scams` | `egypt_scam_scenarios/scams/*.json` (6 files) | JSON array | Per-scam, text fields |
| `rihla_advisories` | `egypt_travel_advisories/*.md` (9 files) | Markdown | Semantic chunking |

### 6.2 Chunking
- JSON arrays: 1 chunk per object, fields concatenated with ` | `
- JSON objects: 1 chunk per top-level key
- Markdown: semantic chunking, 800 words per chunk, 120 word overlap

### 6.3 Embeddings
- Model: `models/text-embedding-004`
- Vector size: 768 dimensions
- Distance: Cosine

### 6.4 Retrieval Strategies
| Strategy | Description |
|----------|-------------|
| **Semantic** | Cosine similarity, top-k=5 |
| **Hybrid MMR** | Maximum Marginal Relevance for diversity *(planned)* |
| **Keyword + Dense** | BM25 + embedding RRF *(planned)* |

### 6.4 Auto-Ingestion on Startup

On service start, `lifespan()` in `main.py` launches a background task (`_auto_ingest()`) that:
1. Checks if any RAG collection in Qdrant already has data (via `client.get_collection().points_count`)
2. If populated → skips (idempotent)
3. If empty → runs `ingest_all()` to populate all 7 collections
4. Runs as `asyncio.create_task()` — does NOT block the service from accepting requests
5. Ingestion failures are logged as warnings (non-fatal — service runs without RAG)

### 6.5 Qdrant Collections
Each collection is named `rihla_{category}` and stores:
- `vector`: 768-d embedding
- `payload.text`: chunk text
- `payload.source_file`: origin file
- `payload.category`: data category
- `payload.chunk_index`: position in sequence

---

## 7. Guardrails System

### 7.1 Input Guard
Blocks requests containing:
- **Military keywords**: military, army, naval, air force, missile, weapon, barracks, restricted area, military base
- **PII patterns**: SSN, credit card numbers, passport numbers
- **Prompt injection**: "ignore previous instructions", "you are now...", "DAN", override attempts

### 7.2 Output Guard
Scans AI responses for:
- **Military mentions** → triggers regeneration with stricter prompt
- **PII leakage** → auto-redacts with `[REDACTED]`
- All hits logged to Langfuse

### 7.3 Gemini Safety Settings
- `HarmCategory.HARASSMENT`: BLOCK_MEDIUM_AND_ABOVE
- `HarmCategory.HATE_SPEECH`: BLOCK_MEDIUM_AND_ABOVE
- `HarmCategory.SEXUALLY_EXPLICIT`: BLOCK_MEDIUM_AND_ABOVE
- `HarmCategory.DANGEROUS_CONTENT`: BLOCK_MEDIUM_AND_ABOVE

---

## 8. Monitoring

### 8.1 Langfuse
- Traces on every LLM call (chat, tool, vision, audio)
- Spans for each tool execution
- Scores: faithfulness (LLM-as-Judge), context relevance, user feedback
- Cost tracking via token counts

### 8.2 Prometheus Metrics
| Metric | Type | Labels |
|--------|------|--------|
| `llm_requests_total` | Counter | endpoint, status |
| `llm_latency_seconds` | Histogram | endpoint |
| `llm_token_usage` | Counter | type, key_suffix |
| `rag_retrieval_count` | Counter | collection, strategy |
| `guardrail_hits_total` | Counter | rule_type |
| `agent_calls_total` | Counter | agent_name |
| `active_api_keys` | Gauge | — |

### 8.3 Quality Evaluation
- Batch script: `python -m app.monitoring.evaluate`
- Dataset: 50+ Q&A pairs across all personas
- Metrics: faithfulness, relevance, conciseness, safety compliance

---

## 9. Configuration

### 9.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEYS` | Yes | — | Comma-separated Gemini API keys |
| `JWT_ACCESS_SECRET` | Yes | — | Shared with Core-Server |
| `INTERNAL_API_KEY` | Yes | — | Service-to-service auth |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse project key |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse secret |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse URL |
| `GIS_SERVICE_URL` | No | `http://geocontext:8000` | GeoContext base |
| `RISK_SERVICE_URL` | No | `http://risk-intelligence:3001` | Risk base |
| `CORE_SERVER_URL` | No | `http://core-server:3000` | Core-Server base |
| `QDRANT_HOST` | No | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | No | 6333 | Qdrant gRPC port |
| `RATE_LIMIT_PER_USER` | No | 30 | Requests/min per user |
| `MAX_TOOL_CALLS` | No | 5 | Tools per turn |
| `LOG_LEVEL` | No | INFO | Logging level |

### 9.2 Rate Limits
- Per-user: 30 req/min (configurable)
- Per-IP: 60 req/min
- Per-token: 100K tokens/min per Gemini key (enforced by Google)

---

## 10. Testing

### 10.1 Test Coverage
| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_guardrails.py` | 13 | Military detection, PII, prompt injection, output sanitization, regeneration triggers |
| `test_tools.py` | 12 | Tool definition structure, required params, enum values, EGYPT_CITIES validation |

### 10.2 Planned Tests
- `test_llm_client.py`: Token failover, key rotation, degraded state recovery
- `test_rag.py`: Ingestion, chunking, retrieval with mocked Qdrant
- `test_agent.py`: Supervisor routing, persona switching, tool calling

---

## 11. Persona Prompt Templates

### 11.1 Structure
```python
PERSONAS = {
    "tour_guide": {
        "identity": "You are an enthusiastic and knowledgeable Egyptian tour guide...",
        "tone": "Speak with passion about Egypt's 7000-year history...",
        "knowledge_boundaries": "You specialize in Egyptian history, archaeology...",
        "tools": ["search_attractions", "get_nearby_attractions", "get_currency_info", "get_legal_guidelines", "recommend_itinerary"],
    },
    "local_expert": {
        "identity": "You are a friendly local Egyptian...",
        "tone": "Speak warmly and conversationally...",
        "knowledge_boundaries": "You know the best local food spots...",
        "tools": ["get_scam_warnings", "get_currency_info", ...],
    },
    "safety_guru": {
        "identity": "You are a proactive but calm travel safety advisor...",
        "tone": "Be informative and watchful, never alarmist...",
        "knowledge_boundaries": "You cover travel advisories, common scams...",
        "tools": ["get_safety_info", "get_emergency_contacts", ...],
    },
}
```

### 11.2 Dynamic Context Injection
The system prompt builder (`build_system_prompt()`) injects:
- User profile (name, nationality, language, budget, style, interests)
- Current environment (weather, air quality, prayer times)
- Nearby geography (POIs, routes, geocode)

---

## 12. Multimodal Features

### 12.1 Landmark Identification
1. Image uploaded + optional GPS coordinates
2. If GPS provided: query GeoContext for nearby sites (cross-reference list)
3. Send to Gemini Vision with system prompt including cross-reference data
4. Parse JSON response into structured output
5. Cache by image MD5 hash + coordinates (24h TTL, max 100 entries)
6. Output: name, Arabic name, description, category, historical period, Wikipedia URL

### 12.2 Voice Processing
1. Audio file uploaded (WAV/MP3/OGG)
2. Sent to Gemini as `inline_data` with `audio/mpeg` MIME type
3. Gemini handles STT + understanding + response generation natively
4. Returns text response
5. Optional TTS via gTTS (planned)

---

## 13. Docker

### 13.1 Dockerfile
- **Builder stage**: `python:3.11-slim`, installs dependencies from `requirements.txt`
- **Runtime stage**: `python:3.11-slim`, copies installed packages, application code, RAG data
- Exposes port 3003
- Runs `uvicorn app.main:app --host 0.0.0.0 --port 3003`

### 13.2 Docker Compose
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: [6333, 6334]
    volumes: [qdrant_data:/qdrant/storage]
    healthcheck: curl localhost:6333/healthz

  ai-service:
    build: .
    ports: [3003]
    volumes: [./data, ../../RAG:/app/data/rag:ro]
    env_file: .env
    depends_on: qdrant (condition: service_healthy)
```

---

## 14. File Inventory

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # 82 lines — FastAPI app, lifespan, CORS, routes, background auto-ingestion
│   ├── config.py                  # 47 lines — Pydantic BaseSettings
│   ├── api/
│   │   ├── chat.py                # 63 lines — POST /chat
│   │   ├── stream.py              # 71 lines — POST /chat/stream (SSE)
│   │   ├── voice.py               # 60 lines — POST /voice
│   │   ├── identify.py            # 117 lines — POST /identify
│   │   └── health.py              # 56 lines — GET /health, /readyz, /health/keys
│   ├── core/
│   │   ├── llm_client.py          # 265 lines — Gemini with token failover, key revive, recursion guard, sync→async stream
│   │   ├── system_prompt.py       # 107 lines — Persona templates + rules + dynamic context
│   │   └── guardrails.py          # 126 lines — Input/output guardrails
│   ├── agent/
│   │   ├── supervisor.py          # 107 lines — Intent detection, routing, tool orchestration
│   │   ├── tour_guide.py          # 33 lines — Tour Guide agent handler
│   │   ├── local_expert.py        # 33 lines — Local Expert agent handler
│   │   ├── safety_guru.py         # 33 lines — Safety Guru agent handler
│   │   └── tools.py               # 479 lines — 9 tool definitions + implementations
│   ├── rag/
│   │   ├── vector_store.py        # 92 lines — Qdrant client wrapper
│   │   ├── ingestion.py           # 127 lines — Multi-format file reader
│   │   ├── chunking.py            # 87 lines — Semantic + recursive chunking
│   │   └── retriever.py           # 88 lines — Embedding + search strategies
│   ├── services/
│   │   ├── geocontext.py          # 48 lines — GeoContext HTTP client
│   │   └── risk.py                # 39 lines — Risk_Intelligence HTTP client
│   └── monitoring/
│       ├── langfuse.py            # 32 lines — Langfuse initialization
│       └── metrics.py             # 42 lines — Prometheus metric definitions
├── tests/
│   ├── test_guardrails.py         # 67 lines — 13 tests
│   └── test_tools.py              # 55 lines — 8 tests
├── Dockerfile                     # Multi-stage Python 3.11-slim
├── docker-compose.yml             # AI service + Qdrant
├── .env.example                   # 15 env vars
├── pyproject.toml                 # Dependencies + dev extras
├── requirements.txt               # Pinned dependencies
└── SPECS.md                       # This file
```

**Total: 38 files, ~2,390 lines of Python**

---

## 15. Integration Points

### 15.1 Core-Server (planned, not implemented)
The following changes will be needed in Core-Server when remote updates are ready:
- `chat.service.ts`: Send `persona` field in payload, attach `safety` data
- `chat.routes.ts`: Add optional `persona` to Zod schema
- `geo.service.ts`: Fix proxy paths to GeoContext API
- `risk.service.ts`: New — Risk_Intelligence client
- `internal.routes.ts`: New — internal AI data endpoints
- `docker-compose.yml`: Add ai-service container

### 15.2 GeoContext (already wired)
- AI service calls `GET /api/v1/nearby-sites` for nearby attractions
- AI service calls `GET /api/v1/context` for location context

### 15.3 Risk_Intelligence (already wired)
- AI service calls `GET /safety/current?city=` for safety data

### 15.4 Database Access (not yet implemented)
Currently the AI service has **no database access**. To read user data:
- **Option A**: Direct read access to Core-Server's PostgreSQL
- **Option B**: Client sends user data in request body (simpler, currently used)

---

## 16. Future Enhancements

- [ ] Direct PostgreSQL connection to read user profiles, preferences, journeys
- [ ] MMR and keyword-dense fusion retrieval strategies
- [ ] TTS for voice responses (gTTS integration)
- [ ] Batch quality evaluation pipeline
- [ ] CI/CD GitHub Actions workflow
- [ ] LangChain Hub prompt versioning
