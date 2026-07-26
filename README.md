# Rihla AI Service

Intelligent conversational AI assistant for tourists in Egypt — 4th microservice of the Rihla platform. Built from scratch with Python/FastAPI and Google Gemini.

## Features

- **3 Persona Chat** — Tour Guide, Local Expert, Safety Guru with auto-intent detection
- **9 Agent Tools** — RAG search, GeoContext integration, Risk data, itinerary planner
- **Multi-Format RAG** — 8 Qdrant collections from JSON, Markdown, and GeoJSON sources
- **LLM Token Failover** — Round-robin Gemini keys, degrade on 429/500, 60s cooldown, auto-revive
- **Input/Output Guardrails** — Military content, PII, prompt injection detection and blocking
- **Landmark Identification** — Image + GPS → Gemini Vision → structured response with caching
- **Voice Processing** — Native Gemini audio understanding (STT + NLU + response)
- **Streaming Responses** — SSE token-by-token streaming
- **Itinerary Planning** — Multi-city, multi-day trip planner with budget, timing, and safety context
- **Monitoring** — Langfuse tracing + Prometheus metrics (7 metric types)
- **Dockerized** — Multi-stage Python 3.11-slim, compose file with Qdrant

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Gemini API key(s) from [Google AI Studio](https://makersuite.google.com/)

### Setup

```bash
# Clone and enter directory
cd ai-service

# Copy environment config
cp .env.example .env
# Edit .env: add GEMINI_API_KEYS, INTERNAL_API_KEY, JWT_ACCESS_SECRET

# Install dependencies
pip install -r requirements.txt

# Run with Docker (Qdrant + AI Service)
docker compose up --build
```

### Configuration

All configuration via environment variables (see `.env.example`):

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

## API Reference

### `POST /chat`

Primary conversational endpoint.

**Request:**
```json
{
  "message": "What are the best things to do in Luxor?",
  "conversation_id": "optional-uuid",
  "persona": "auto | tour_guide | local_expert | safety_guru",
  "lat": 25.6872,
  "lon": 32.6396,
  "user": {
    "display_name": "Ahmed",
    "nationality": "US",
    "language": ["en"],
    "budget_level": "mid",
    "travel_style": "cultural",
    "interests": ["history", "photography"]
  }
}
```

**Response:**
```json
{
  "response": "Great choice! Luxor is home to...",
  "conversation_id": "uuid",
  "persona": "tour_guide",
  "blocked": false,
  "reason": null
}
```

### `POST /chat/stream`

SSE streaming — same body as `/chat`. Returns tokens as server-sent events:
```
data: {"token": "Luxor"}
data: {"token": " is"}
data: {"done": true, "full_response": "Luxor is home to..."}
data: [DONE]
```

### `POST /voice`

**Request:** `multipart/form-data` with `audio` file (WAV/MP3/OGG), optional `lat`/`lon`.

**Response:**
```json
{
  "text_response": "I heard your question about...",
  "audio_response": null,
  "conversation_id": "uuid"
}
```

### `POST /identify`

**Request:** `multipart/form-data` with `image` (JPEG/PNG), optional `lat`/`lon`.

**Response:**
```json
{
  "name": "Mosque of Muhammad Ali",
  "name_ar": "مسجد محمد علي",
  "description": "A stunning Ottoman-era mosque...",
  "category": "mosque",
  "historical_period": "ottoman",
  "wikipedia_url": "https://en.wikipedia.org/...",
  "cached": false
}
```

### Health Endpoints

| Path | Purpose |
|------|---------|
| `GET /health` | Liveness check |
| `GET /readyz` | Readiness (LLM + Qdrant status) |
| `GET /health/keys` | Per-key status and fail counts |

## Personas

### Auto-Detection

When `persona: "auto"`, the supervisor scores message keywords:

| Persona | Triggers |
|---------|----------|
| `tour_guide` | history, attraction, museum, pyramid, temple, ancient, pharaoh, culture |
| `safety_guru` | safe, danger, scam, risk, emergency, police, hospital, crime |
| `local_expert` | food, eat, restaurant, shop, bargain, local, hidden, authentic |

### Tour Guide
Enthusiastic Egyptologist. Covers history, sites, fees, hours, and itineraries.
- **Tools**: `search_attractions`, `get_nearby_attractions`, `get_currency_info`, `get_legal_guidelines`, `recommend_itinerary`

### Local Expert
Friendly Egyptian insider. Gives local tips, haggling advice, food recommendations.
- **Tools**: `get_scam_warnings`, `get_currency_info`, `get_legal_guidelines`, `get_nearby_attractions`

### Safety Guru
Calm, proactive safety advisor. Covers scams, emergencies, health, and legal rights.
- **Tools**: `get_safety_info`, `get_emergency_contacts`, `get_scam_warnings`, `get_legal_guidelines`

### Hard Rules (all personas)
1. Never mention military sites or restricted zones
2. Never disparage Egypt, its people, or government
3. Frame challenges factually ("be aware") not alarmingly
4. Politely redirect restricted queries to tourist topics
5. Always represent Egypt positively
6. Pair every warning with a concrete countermeasure
7. Do not hallucinate — say "I don't know" if unsure

## Agent Tools

| Tool | Description | Backend |
|------|-------------|---------|
| `get_nearby_attractions` | Find sites near lat/lon | GeoContext HTTP |
| `search_attractions` | Search attractions by keyword/city | Qdrant RAG |
| `get_safety_info` | Current safety for a city | Risk_Intelligence HTTP |
| `get_emergency_contacts` | Emergency numbers by type | Qdrant RAG |
| `get_legal_guidelines` | Egyptian laws for tourists | Qdrant RAG |
| `get_currency_info` | EGP details + exchange info | Qdrant RAG |
| `get_scam_warnings` | Scam scenarios + countermeasures | Qdrant RAG |
| `recommend_itinerary` | Full multi-day trip planner | Multi-tool + Gemini |

### `recommend_itinerary`

Creates a complete day-by-day itinerary. Flow: auto-suggest cities → parallel fetch attractions/safety/scams per city → Gemini composes day-by-day plan.

**Parameters:**
- `interests` (required): `["history", "photography", "food"]`
- `days` (required): 1-14
- `budget` (required): `"budget" | "mid" | "luxury"`
- `cities` (optional): specific cities, or empty for AI to suggest
- `style` (optional): `"cultural" | "adventure" | "relaxation" | "family" | "solo" | "romantic"`
- `base_currency` (optional): `"USD"`, `"EUR"`, `"GBP"` for budget estimates

**Output:** Markdown itinerary + structured JSON (embedded in HTML comment) with per-day activity items, timing, fees, safety tips, and scam warnings.

## RAG Pipeline

### Collections (8 in Qdrant)

| Collection | Source | Chunking |
|-----------|--------|----------|
| `rihla_attractions` | `EgyptAttractions_rag.json` | Per-object |
| `rihla_monuments` | `egymonuments.com.json` | Per-object |
| `rihla_emergency` | `Emergency_Contacts.json` | Per-key |
| `rihla_legal` | 6 legal framework files | Per-object |
| `rihla_currency` | `CurrunciesEG.json` | Per-key |
| `rihla_scams` | 6 scam scenario files | Per-scam |
| `rihla_advisories` | 9 travel advisory markdown files | Semantic (800 words) |

### Embeddings

- Model: `text-embedding-004` (Gemini)
- Vector size: 768 dimensions
- Distance: Cosine

### Ingestion

Auto-ingested on first startup via background task (non-blocking). Checks if Qdrant collections already have data before running.

```bash
# Manual re-ingestion:
python -c "import asyncio; from app.rag.ingestion import ingest_all; from app.main import vector_store; asyncio.run(ingest_all(vector_store, 'data/rag'))"
```

## Guardrails

### Input Guard (blocks before processing)

- Military/restrcted content keywords
- PII (SSN, credit cards, passport numbers)
- Prompt injection attempts ("ignore previous instructions", "DAN", etc.)

### Output Guard (sanitizes AI responses)

- Military keyword detection → triggers regeneration with stricter prompt
- PII leakage → auto-redacted with `[REDACTED]`
- All hits logged to Langfuse

## Monitoring

### Langfuse

- Traces on every LLM call (chat, tool, vision, audio)
- Spans for tool execution
- Cost tracking via token counts
- Enabled via `LANGFUSE_*` env vars

### Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `llm_requests_total` | Counter | endpoint, status |
| `llm_latency_seconds` | Histogram | endpoint |
| `llm_token_usage` | Counter | type, key_suffix |
| `rag_retrieval_count` | Counter | collection, strategy |
| `guardrail_hits_total` | Counter | rule_type |
| `agent_calls_total` | Counter | agent_name |
| `active_api_keys` | Gauge | — |

Metrics at `GET /metrics` (auto-mounted by prometheus_client).

## LLM Token Failover

- **Round-robin** across all configured Gemini keys
- **On 429 (rate limit)** or **500**: marks key `degraded`, 60s cooldown
- **On success**: resets fail count, marks `active`
- **Cooldown expiry**: auto-revives the key
- **All keys degraded**: raises `RuntimeError`, caught by supervisor → user-friendly message

## Project Structure

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    FastAPI app, lifespan, CORS, routes
│   ├── config.py                  Pydantic BaseSettings
│   ├── api/
│   │   ├── chat.py                POST /chat
│   │   ├── stream.py              POST /chat/stream (SSE)
│   │   ├── voice.py               POST /voice
│   │   ├── identify.py            POST /identify
│   │   └── health.py              GET /health, /readyz, /health/keys
│   ├── core/
│   │   ├── llm_client.py          GeminiClient with token failover
│   │   ├── system_prompt.py       Persona templates, hard rules, context injection
│   │   └── guardrails.py          Input/output guardrails
│   ├── agent/
│   │   ├── supervisor.py          Intent detection, routing, tool orchestration
│   │   ├── tools.py               9 tool definitions + implementations
│   │   ├── tour_guide.py          Tour Guide persona
│   │   ├── local_expert.py        Local Expert persona
│   │   └── safety_guru.py         Safety Guru persona
│   ├── rag/
│   │   ├── vector_store.py        Qdrant client wrapper
│   │   ├── ingestion.py           Multi-format ingestion pipeline
│   │   ├── chunking.py            Semantic + recursive chunking
│   │   └── retriever.py           Embedding + search strategies
│   ├── services/
│   │   ├── geocontext.py          GeoContext HTTP client
│   │   └── risk.py                Risk_Intelligence HTTP client
│   └── monitoring/
│       ├── langfuse.py            Langfuse initialization
│       └── metrics.py             Prometheus metrics
├── tests/
│   ├── test_guardrails.py         13 tests
│   └── test_tools.py              12 tests
├── Dockerfile                     Multi-stage Python 3.11-slim
├── docker-compose.yml             AI service + Qdrant
├── .env.example
├── pyproject.toml
├── requirements.txt
├── SPECS.md                       Full specifications
├── ARCHITECTURE.md                Request flow documentation
└── README.md                      This file
```

**Total: 38 files, ~2,320 lines of Python**

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run with hot reload
uvicorn app.main:app --reload --port 3003

# Check types
mypy app
```

## Docker

```bash
# Build and run with Qdrant
docker compose up --build

# Run only the service (with external Qdrant)
docker build -t rihla-ai-service .
docker run -p 3003:3003 --env-file .env rihla-ai-service
```

## Integration Points

| Service | Connection | Protocol | Endpoint |
|---------|-----------|----------|----------|
| Qdrant | Vector DB | gRPC | `qdrant:6333` |
| GeoContext | GIS data | HTTP | `geocontext:8000/api/v1/` |
| Risk_Intelligence | Safety data | HTTP | `risk-intelligence:3001/` |
| Core-Server | Platform backend | HTTP *(planned)* | `core-server:3000/` |

## Docs

- [SPECS.md](SPECS.md) — Full service specifications (16 sections)
- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed request/data flow diagrams for every feature
