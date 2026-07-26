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
