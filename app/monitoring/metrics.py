from prometheus_client import Counter, Histogram, Gauge

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["endpoint", "status"],
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM request latency",
    ["endpoint"],
)

llm_token_usage = Counter(
    "llm_token_usage",
    "LLM token usage",
    ["type", "key_suffix"],
)

rag_retrieval_count = Counter(
    "rag_retrieval_count",
    "RAG retrieval count",
    ["collection", "strategy"],
)

guardrail_hits_total = Counter(
    "guardrail_hits_total",
    "Guardrail hits",
    ["rule_type"],
)

agent_calls_total = Counter(
    "agent_calls_total",
    "Agent calls",
    ["agent_name"],
)

active_api_keys = Gauge(
    "active_api_keys",
    "Number of active API keys",
)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the AI service",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

http_errors_total = Counter(
    "http_errors_total",
    "Total HTTP 4xx/5xx errors",
    ["status"],
)

rate_limit_blocks_total = Counter(
    "rate_limit_blocks_total",
    "Requests rejected by the rate limiter",
    ["endpoint"],
)
