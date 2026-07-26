# Rihla AI Service — Architecture & Request Flows

## Table of Contents

1. [Service Topology](#1-service-topology)
2. [Chat Request Flow (Happy Path)](#2-chat-request-flow-happy-path)
3. [Persona Routing](#3-persona-routing)
4. [Tool Execution Flow](#4-tool-execution-flow)
5. [Itinerary Generation Flow](#5-itinerary-generation-flow)
6. [Guardrails Flow](#6-guardrails-flow)
7. [LLM Token Failover Flow](#7-llm-token-failover-flow)
8. [RAG Retrieval Flow](#8-rag-retrieval-flow)
9. [RAG Ingestion Pipeline](#9-rag-ingestion-pipeline)
10. [Streaming Flow](#10-streaming-flow)
11. [Landmark Identification Flow](#11-landmark-identification-flow)
12. [Voice Processing Flow](#12-voice-processing-flow)
13. [Health Check Flow](#13-health-check-flow)
14. [Error Handling](#14-error-handling)

---

## 1. Service Topology

```
┌──────────────┐     HTTP/JSON      ┌──────────────────────────────────────────┐
│   Client     │ ◄─────────────────►│           AI Service (port 3003)          │
│ (App/Browser)│                    │              FastAPI + Uvicorn             │
└──────────────┘                    └──────────────┬───────────────────────────┘
                                                   │
         ┌─────────────────────────────────────────┼─────────────────────────────┐
         │                                         │                             │
         ▼                                         ▼                             ▼
┌─────────────────┐                      ┌──────────────────┐          ┌───────────────────┐
│     Qdrant      │                      │   GeoContext     │          │  Risk_Intelligence │
│  (Vector DB)    │                      │  (port 8000)     │          │   (port 3001)      │
│  gRPC :6333     │                      │                  │          │                    │
│  HTTP :6334     │                      └──────────────────┘          └───────────────────┘
└─────────────────┘
```

### Internal Module Map

```
app/
├── main.py                 FastAPI app, lifespan (init Qdrant + LLM), CORS, global exception handler, router mounts
├── config.py               Pydantic BaseSettings — all env vars, gemini_key_list property
├── api/
│   ├── chat.py             POST /chat — builds context dict, calls supervisor.route_and_respond()
│   ├── stream.py           POST /chat/stream — SSE streaming via llm_client.generate(stream=True)
│   ├── voice.py            POST /voice — audio → Gemini native STT + response
│   ├── identify.py         POST /identify — image + GPS → Gemini Vision + RAG cross-ref
│   └── health.py           GET /health, /readyz, /health/keys
├── core/
│   ├── llm_client.py       GeminiClient — 4 generation methods, round-robin key failover, cooldown/revive
│   ├── system_prompt.py    PERSONAS dict, HARD_RULES, build_system_prompt() with dynamic context injection
│   └── guardrails.py       InputGuard (military/PII/injection) + OutputGuard (military/PII redact/regenerate)
├── agent/
│   ├── supervisor.py       Intent detection (keyword scoring), route_and_respond(), tool orchestration, guardrail wrap
│   ├── tools.py            9 tool definitions (TOOL_DEFINITIONS), EGYPT_CITIES, call_tool() dispatcher, implementations
│   ├── tour_guide.py       Tour Guide persona handler — restricted tool set
│   ├── local_expert.py     Local Expert persona handler
│   └── safety_guru.py      Safety Guru persona handler
├── rag/
│   ├── vector_store.py     AsyncQdrantClient wrapper — initialize, create collections, upsert, search, delete
│   ├── ingestion.py        DATA_SOURCES config, multi-format reader, ingest_all() pipeline
│   ├── chunking.py         chunk_text (semantic 800 words), chunk_json_array (per-object), chunk_json_object (per-key)
│   └── retriever.py        Gemini text-embedding-004, retrieve() (semantic), retrieve_hybrid(), ALL_COLLECTIONS
├── services/
│   ├── geocontext.py       HTTP client for GeoContext (/api/v1/nearby-sites, /api/v1/context)
│   └── risk.py             HTTP client for Risk_Intelligence (/safety/current)
└── monitoring/
    ├── langfuse.py         Langfuse init (conditional on env vars)
    └── metrics.py          Prometheus metric definitions (7 metrics)
```

---

## 2. Chat Request Flow (Happy Path)

```
Client                          AI Service
  │                                 │
  │  POST /chat {message, persona,  │
  │    user, environment, geography}│
  │────────────────────────────────►│
  │                                 │
  │                           ┌─────┴──────────┐
  │                           │ chat.chat_      │
  │                           │ endpoint()      │
  │                           │ - build context │
  │                           │   dict from req │
  │                           └─────┬──────────┘
  │                                 │
  │                           ┌─────┴──────────┐
  │                           │ supervisor.     │
  │                           │ route_and_      │
  │                           │ respond()       │
  │                           └─────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 1. check_input()    │
  │                      │    guardrails       │
  │                      │    ┌── blocked? ──► return polite redirect
  │                      │    └── passed       │
  │                      └──────────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 2. detect_intent()  │
  │                      │    (if persona=auto)│
  │                      │    keyword scoring  │
  │                      │    ─► tour_guide/   │
  │                      │       local_expert/ │
  │                      │       safety_guru   │
  │                      └──────────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 3. build_system_    │
  │                      │    prompt(persona,  │
  │                      │    context)          │
  │                      │    - identity + tone│
  │                      │    - HARD_RULES     │
  │                      │    - dynamic user/  │
  │                      │      env/geo context│
  │                      └──────────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 4. llm_client.      │
  │                      │    generate_with_   │
  │                      │    tools(            │
  │                      │      system_prompt,  │
  │                      │      user_message,   │
  │                      │      TOOL_DEFINITIONS│
  │                      │    )                 │
  │                      │    ┌─── Gemini ────┐ │
  │                      │    │  round-robin  │ │
  │                      │    │  key select   │ │
  │                      │    │  func call or │ │
  │                      │    │  text response│ │
  │                      │    └───────────────┘ │
  │                      └──────────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 5. If function_calls │
  │                      │    ─► call_tool()   │
  │                      │    ─► append results │
  │                      │    ─► llm_client.   │
  │                      │       generate()    │
  │                      │       (text-only    │
  │                      │        with context)│
  │                      └──────────┬──────────┘
  │                                 │
  │                      ┌──────────┴──────────┐
  │                      │ 6. check_output()   │
  │                      │    guardrails       │
  │                      │    ┌── requires_re- │
  │                      │    │   generation?  │
  │                      │    │   ─► regenerate│
  │                      │    │   with stricter│
  │                      │    │   prompt       │
  │                      │    ├── modified?    │
  │                      │    │   ─► sanitize  │
  │                      │    └── passed       │
  │                      └──────────┬──────────┘
  │                                 │
  │  {response, persona,            │
  │   blocked, reason}              │
  │◄────────────────────────────────│
```

---

## 3. Persona Routing

### 3.1 Intent Detection Algorithm

```
Input: message string
Output: persona name

1. Lowercase message
2. For each persona (tour_guide, safety_guru, local_expert):
     For each keyword in INTENT_KEYWORDS[persona]:
       If keyword in message → increment score
3. max_score = max(scores.values())
4. If max_score == 0 → return "tour_guide" (default)
5. Return persona with highest score (first if tie)
```

| Persona | Keywords |
|---------|----------|
| `tour_guide` | history, attraction, site, museum, pyramid, temple, monument, tour, visit, see, ancient, pharaoh, culture, heritage |
| `safety_guru` | safe, danger, scam, risk, emergency, warning, police, hospital, ambulance, crime, protect, avoid, alert |
| `local_expert` | food, eat, restaurant, shop, bargain, haggle, local, insider, tip, hidden, best, authentic, custom, culture |

### 3.2 Persona Tool Boundaries

When `persona=auto`, the supervisor uses the **full** tool set via `TOOL_DEFINITIONS`. When a specific persona is requested via the dedicated handler (`app/agent/{persona}.py`), only `ALLOWED_TOOLS` for that persona are passed to Gemini:

| Persona | Allowed Tools |
|---------|--------------|
| `tour_guide` | search_attractions, get_nearby_attractions, get_currency_info, get_legal_guidelines, recommend_itinerary |
| `local_expert` | get_scam_warnings, get_currency_info, get_legal_guidelines, get_nearby_attractions |
| `safety_guru` | get_safety_info, get_emergency_contacts, get_scam_warnings, get_legal_guidelines |

---

## 4. Tool Execution Flow

### 4.1 Generic Tool Flow

```
Gemini decides function call
        │
        ▼
response.function_calls = [...]
        │
        ▼
For each function_call:
  │
  ├─ fc.name     → tool_name
  ├─ dict(fc.args) → arguments
  │
  ▼
call_tool(tool_name, arguments)
  │
  ├─ Dispatcher chain (tools.py):
  │   if "get_nearby_attractions" → _get_nearby_attractions(lat, lon, radius)
  │   if "search_attractions"     → _search_attractions(query, category, city)
  │   if "get_safety_info"        → _get_safety_info(city)
  │   if "get_emergency_contacts" → _get_emergency_contacts(context_type)
  │   if "get_legal_guidelines"   → _get_legal_guidelines(topic)
  │   if "get_currency_info"      → _get_currency_info(denomination, base_currency)
  │   if "get_scam_warnings"      → _get_scam_warnings(category, severity)
  │   if "recommend_itinerary"    → _recommend_itinerary(...)
  │
  ▼
Result string appended to conversation
        │
        ▼
Gemini text generation with tool results
```

### 4.2 Tool Behind-the-Scenes Detail

| Tool | Internal Calls | Data Sources | Fallback |
|------|---------------|--------------|----------|
| `_get_nearby_attractions` | HTTP GET to GeoContext `/api/v1/nearby-sites` | GeoContext PostgreSQL | Returns error message string |
| `_search_attractions` | `retrieve()` → embed query → Qdrant search `rihla_attractions` | Qdrant collection | "Vector store not available" |
| `_get_safety_info` | HTTP GET to Risk_Intelligence `/safety/current?city=` | Risk_Intelligence data | "No safety data for {city}" |
| `_get_emergency_contacts` | `retrieve()` → Qdrant search `rihla_emergency` | Qdrant collection | "No emergency contacts found" |
| `_get_legal_guidelines` | Topic map → `retrieve()` → Qdrant search `rihla_legal` | Qdrant collection | "No legal guidelines found" |
| `_get_currency_info` | `retrieve()` → Qdrant search `rihla_currency` | Qdrant collection | "Currency information not available" |
| `_get_scam_warnings` | `retrieve()` → Qdrant search `rihla_scams` | Qdrant collection | "No scam warnings found" |
| `_recommend_itinerary` | AI city suggestion + parallel fetches + Gemini composition | Multiple (see below) | Falls back to Gemini response |

---

## 5. Itinerary Generation Flow

`recommend_itinerary` is the most complex tool — it orchestrates multiple sub-tools and two Gemini calls.

```
_recommend_itinerary(interests, days, budget, cities, style, base_currency)
  │
  ├── [CITIES EMPTY?] ──yes──► _suggest_cities(interests, days, style)
  │                                   │
  │                                   └── Gemini generate() with structured prompt
  │                                       Return: ["Cairo", "Luxor"]
  │
  ├── [FOR EACH CITY in parallel (asyncio.gather)]:
  │     │
  │     ├── Lookup EGYPT_CITIES[city] for lat/lon
  │     │
  │     ├── _search_attractions(query=f"{interests} attractions in {city}")
  │     │     └── retrieve(embeddings, "attractions")
  │     │
  │     ├── _get_safety_info(city)
  │     │     └── HTTP GET → Risk_Intelligence
  │     │
  │     ├── _get_nearby_attractions(lat, lon, radius=5000)
  │     │     └── HTTP GET → GeoContext
  │     │
  │     └── [_get_scam_warnings(category)] for each interest-mapped category
  │           └── retrieve(embeddings, "scams")
  │
  ├── _get_currency_info(base_currency)
  │     └── retrieve(embeddings, "currency")
  │
  ├── [COMPOSE ITINERARY]
  │     │
  │     └── Gemini generate() with HARD_RULES + budget guides + collected data
  │         │
  │         └── JSON parse: {markdown, json: {days: [...]}}
  │
  └── Return: markdown string + structured JSON in HTML comment
```

### Output Format

The tool returns a markdown-rendered itinerary with a hidden structured JSON payload embedded in an HTML comment:

```
# 3-Day Cairo Itinerary

**Day 1: Pyramids & Ancient Wonders**...

<!-- structured: {"title":"3-Day Cairo...","days":[...]} -->
```

The structured JSON contains:
- `title`: Trip title
- `budget_estimate`: `{egp, usd}` amounts
- `currency_note`: Exchange rate disclaimer
- `days[]`: Array of day objects, each with:
  - `day`: Day number
  - `city`: City name
  - `theme`: Day theme label
  - `items[]`: Activity items with:
    - `time`: Time of day (e.g. "06:00")
    - `activity`: Activity name
    - `type`: Category (attraction/meal/transport/rest/other)
    - `fee_egp`: Entry fee in EGP
    - `duration_hours`: Estimated time
    - `safety_tip`: Contextual safety advice
    - `scam_warning`: Applicable scam warning
- `trip_notes[]`: General travel tips

---

## 6. Guardrails Flow

### 6.1 Input Guard

```
User Message
    │
    ├── check_prompt_injection()
    │     Regex patterns: "ignore previous instructions", "you are now", "DAN", etc.
    │     └── Match? → BLOCK (reason: "prompt_injection_attempt")
    │
    ├── check_military_content()
    │     Regex patterns: military, army, naval, missile, weapon, barracks, etc.
    │     └── Match? → BLOCK (reason: "restricted_content_request")
    │
    └── check_pii()
          Regex patterns: SSN, credit card, passport formats
          └── Match? → BLOCK (reason: "pii_detected")

Blocked response:
  "I'm here to help with tourism information about Egypt.
   Let's keep our conversation focused on making your visit
   to Egypt wonderful and safe! How can I assist you today?"
```

### 6.2 Output Guard

```
AI Response Text
    │
    ├── check_military_content()
    │     └── Match? → requires_regeneration=True
    │         → Generate again with stricter prompt
    │           "IMPORTANT: Do not mention restricted areas."
    │
    └── sanitize_output()
          └── PII found? → modified=True, text redacted with [REDACTED]
```

---

## 7. LLM Token Failover Flow

```
GeminiClient initialized with N keys
        │
        ▼
generate() / generate_with_tools() / generate_with_image() / generate_with_audio()
        │
        ▼
_get_next_available_key()
  │
  ├── Round-robin index++
  ├── If key.status == ACTIVE → return key
  ├── If key.status == COOLDOWN:
  │     ├── Check current time ≥ cooldown_until
  │     ├── Yes → status→ACTIVE, return key
  │     └── No  → skip
  ├── If key.status == DEGRADED:
  │     ├── Check cooldown expiry
  │     └── No expiry → skip
  └── No keys available → raise RuntimeError("All keys degraded")
        │
        ▼
API call on selected key
        │
        ├── Success → key.mark_success() [fail_count=0, status=ACTIVE]
        │
        └── Exception (429/500)
              └── key.mark_failed() [fail_count++, status=DEGRADED,
                                     cooldown_until=now+60s]
                  └── RECURSIVE call to same generate method
                        (auto-retry with next key)
```

### Key State Machine

```
                    success
    ┌─────── ACTIVE ─────────┐
    │          │             │
    │     fail (429/500)     │  cooldown expired
    │          ▼             │
    │       DEGRADED ────────┘
    │          │
    │     60s timer
    │          │
    └─► COOLDOWN ───────────► ACTIVE (auto-revive)
         (unavailable)
```

### Recursion Safety

If all keys fail consecutively, the recursive call in `except` block will eventually exhaust the key pool and raise `RuntimeError("All API keys are degraded or in cooldown")`. This propagates up to the supervisor which returns a user-friendly error message.

---

## 8. RAG Retrieval Flow

```
User query (text string)
        │
        ▼
retrieve(vector_store, query, collection_name, top_k=5, strategy="semantic")
        │
        ├── get_embedding(query)
        │     └── Gemini text-embedding-004 → List[float] (768-d)
        │
        ├── [Optional] build QdrantFilter from filters dict
        │
        ├── vector_store.search(collection_name, query_vector, top_k, qdrant_filter)
        │     └── AsyncQdrantClient.search() → List[ScoredPoint]
        │
        └── Parse results:
              return [{
                  "text": payload.text,
                  "score": cosine_similarity,
                  "metadata": {everything except "text"}
              }]
```

### Retrieval Strategies

| Strategy | Implementation | When Used |
|----------|---------------|-----------|
| **Semantic** | `retrieve()` — pure cosine similarity | Default for all tool queries |
| **Hybrid** | `retrieve_hybrid()` — fetch 2× top_k, deduplicate by text hash | Planned for itinerary |

### Collections Accessed Per Tool

| Tool | Collection(s) |
|------|--------------|
| `search_attractions` | `rihla_attractions` |
| `get_emergency_contacts` | `rihla_emergency` |
| `get_legal_guidelines` | `rihla_legal` |
| `get_currency_info` | `rihla_currency` |
| `get_scam_warnings` | `rihla_scams` |

---

## 9. RAG Ingestion Pipeline

```
ingest_all(vector_store, rag_dir)
  │
  ├── For each category in DATA_SOURCES:
  │     │
  │     ├── discover_files(rag_dir, config)
  │     │     └── Returns list of file paths (single file or directory)
  │     │
  │     ├── For each file:
  │     │     │
  │     │     ├── json_array type → read_json_file() → chunk_json_array()
  │     │     │     └── Each object → 1 chunk: text_fields joined with " | "
  │     │     │
  │     │     ├── json_object type → read_json_file() → chunk_json_object()
  │     │     │     └── Each top-level key → 1 chunk (if value > 50 chars)
  │     │     │
  │     │     └── markdown type → read_markdown_file() → chunk_text()
  │     │           └── 800-word sliding window, 120-word overlap
  │     │
  │     └── All chunks → embed with text-embedding-004 → PointStruct[]
  │           └── vector_store.upsert_points(collection_name, points)
  │                 └── AsyncQdrantClient.upsert()
  │
  └── Done
```

### Data Source Configuration

```python
DATA_SOURCES = {
    "attractions": {"path": "Archiological/EgyptAttractions_rag.json", "type": "json_array", "text_fields": [...]},
    "monuments":   {"path": "egymonuments.com.json",                    "type": "json_array", "text_fields": [...]},
    "emergency":   {"path": "Emergency_Contacts/Emergency_Contacts.json","type": "json_object"},
    "legal":       {"path": "Legal_Frameworks_and_Culture_Regulations",  "type": "json_array", "text_fields": [...]},
    "currency":    {"path": "EG_Curruncy/CurrunciesEG.json",            "type": "json_object"},
    "scams":       {"path": "egypt_scam_scenarios/scams",               "type": "json_array", "text_fields": [...]},
    "advisories":  {"path": "egypt_travel_advisories",                  "type": "markdown"},
}
```

---

## 10. Streaming Flow

```
Client                          AI Service
  │                                 │
  │  POST /chat/stream {message}    │
  │────────────────────────────────►│
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ check_input()    │
  │                         │ guardrails       │
  │                         │ └── blocked?     │
  │                         │  ─► SSE error    │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ llm_client.      │
  │                         │ generate(        │
  │                         │  stream=True)     │
  │                         │                   │
  │                         │ ┌── Gemini ────┐  │
  │                         │ │ 2.0 Flash    │  │
  │                         │ │ token stream │  │
  │                         │ └──────────────┘  │
  │                         └───────┬──────────┘
  │                                 │
  │  data: {"token": "Luxor"}       │
  │◄────────────────────────────────│
  │  data: {"token": " is"}         │
  │◄────────────────────────────────│
  │  ...                            │
  │  data: {"done": true,           │
  │         "full_response": "..."} │
  │◄────────────────────────────────│
  │  data: [DONE]                   │
  │◄────────────────────────────────│
```

### SSE Event Format

```
data: {"token": "<word or chunk>"}
data: {"done": true, "full_response": "<complete text>"}
data: [DONE]
```

On error:
```
data: {"error": "Message blocked", "reason": "..."}
data: [DONE]
```

---

## 11. Landmark Identification Flow

```
Client                          AI Service
  │                                 │
  │  POST /identify                 │
  │  image + lat + lon              │
  │────────────────────────────────►│
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 1. Read image    │
  │                         │    bytes         │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 2. Compute MD5   │
  │                         │    hash          │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 3. Check cache   │
  │                         │    ┌── hit? ──►  │
  │                         │    │  return      │
  │                         │    │  cached=True │
  │                         │    └── miss       │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 4. If GPS present│
  │                         │    retrieve()    │
  │                         │    Qdrant for    │
  │                         │    nearby sites  │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 5. Gemini Vision │
  │                         │    with system   │
  │                         │    prompt +      │
  │                         │    image +       │
  │                         │    nearby context│
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 6. Parse JSON    │
  │                         │    response      │
  │                         │    └── guardrails │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 7. Cache result  │
  │                         │    (24h TTL,     │
  │                         │     max 100)     │
  │                         └───────┬──────────┘
  │                                 │
  │  {name, name_ar, description,   │
  │   category, historical_period,  │
  │   wikipedia_url, cached}        │
  │◄────────────────────────────────│
```

---

## 12. Voice Processing Flow

```
Client                          AI Service
  │                                 │
  │  POST /voice                    │
  │  audio (multipart)              │
  │────────────────────────────────►│
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 1. Read audio    │
  │                         │    bytes         │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 2. Gemini audio  │
  │                         │    generate_with │
  │                         │    _audio()      │
  │                         │    - inline_data │
  │                         │    - audio/mpeg  │
  │                         │    - system      │
  │                         │      prompt      │
  │                         │    - Gemini      │
  │                         │      handles     │
  │                         │      STT + NLU   │
  │                         │      + response  │
  │                         └───────┬──────────┘
  │                                 │
  │                         ┌───────┴──────────┐
  │                         │ 3. guardrails    │
  │                         │    check_output  │
  │                         └───────┬──────────┘
  │                                 │
  │  {text_response,                │
  │   conversation_id}              │
  │◄────────────────────────────────│
```

---

## 13. Health Check Flow

### GET /health (Liveness)
```
Simple static response:
{
  "status": "ok",
  "service": "Rihla AI Service",
  "version": "0.1.0"
}
```

### GET /readyz (Readiness)
```
1. Check llm_client:
   └── Initialized? → {status: "ok", active_keys, total_keys}
   └── Not init?    → {status: "not_initialized"}, overall=degraded

2. Check vector_store:
   └── Initialized? → list_collections() → {status: "ok", collections: [...]}
   └── Error?       → {status: "error", message}, overall=degraded
   └── Not init?    → {status: "not_initialized"}, overall=degraded

Response:
{
  "status": "ok" | "degraded",
  "checks": {llm, vector_store}
}
```

### GET /health/keys (Key Status)
```
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

## 14. Error Handling

### Error Propagation Chain

```
Layer                Error Type                     Handling
─────                ──────────                     ────────
API Layer            HTTP 400/500/503               FastAPI exception handler
  │                                                    │
  ▼                                                    ▼
Supervisor           LLM failure, tool crash         Return user-friendly message
  │                  guardrail block                  Log to Langfuse
  ▼
Tool Layer           HTTP timeout, Qdrant down        Return error string to LLM
  │                  embedding failure                 LLM may retry or adapt
  ▼
LLM Client           Key exhaustion, API error        Recursive retry → RuntimeError
  ▼
Guardrails           Military content detected        Block or regenerate
                     PII detected                     Redact or block
                     Prompt injection                 Block with redirect
```

### Specific Failure Modes

| Scenario | What Happens | User Sees |
|----------|-------------|-----------|
| All Gemini keys degraded | `RuntimeError("All keys degraded")` | "I encountered an issue processing your request. Please try again." |
| GeoContext down | Tool returns error string | LLM adapts response, may say "I couldn't fetch nearby sites" |
| Qdrant not initialized | Tool returns "Vector store not available" | LLM responds without RAG data |
| Input blocked by guardrails | Early return from supervisor | Polite redirect message |
| Output needs regeneration | Second Gemini call with stricter prompt | Normal response (transparent) |
| Prompt injection detected | Request blocked immediately | Polite redirect message |
| Identify endpoint fails | HTTP 500 with detail | "Identification failed: ..." |
| Voice processing fails | HTTP 500 with detail | "Voice processing failed: ..." |

### Timeouts

| Component | Timeout |
|-----------|---------|
| Gemini API call | No explicit (model-dependent) |
| GeoContext HTTP | 10 seconds |
| Risk_Intelligence HTTP | 10 seconds |
| Tool execution | Configurable via `max_tool_timeout` (default 10s) |
| Total tool calls per turn | Configurable via `max_tool_calls_per_turn` (default 5) |
