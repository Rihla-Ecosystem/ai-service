import time
from collections import defaultdict, deque
from typing import Dict, Deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.monitoring import metrics


class HttpMetricsMiddleware(BaseHTTPMiddleware):
    """Records per-request Prometheus metrics for the AI service."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.http_errors_total.labels(status="500").inc()
            raise

        duration = time.perf_counter() - started
        path = request.url.path
        # Normalize dynamic segments for stable metric cardinality.
        metrics.http_requests_total.labels(method=request.method, path=path, status=response.status_code).inc()
        metrics.http_request_duration_seconds.labels(method=request.method, path=path).observe(duration)
        if response.status_code >= 400:
            metrics.http_errors_total.labels(status=str(response.status_code)).inc()
        return response


class InMemoryRateLimiter:
    """Simple sliding-window in-memory rate limiter keyed by client.

    Buckets requests per client per window (default 60 seconds) up to
    ``max_requests``. Safe for a single process deployment.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._cleanup_counter = 0

    def _prune(self, key: str, now: float) -> None:
        bucket = self._buckets[key]
        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()
        if not bucket:
            self._buckets.pop(key, None)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        self._prune(key, now)
        bucket = self._buckets[key]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        self._prune(key, now)
        return max(0, self.max_requests - len(self._buckets.get(key, ())))

    def stats(self) -> dict:
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_clients": len(self._buckets),
        }


limiter = InMemoryRateLimiter(max_requests=settings.rate_limit_per_user)


def metrics_response() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def _counter_total(counter) -> int:
    total = 0
    for metric in counter.collect():
        for sample in metric.samples:
            total += int(sample.value)
    return total


def _counter_by_label(counter, label: str) -> dict:
    result: dict = {}
    for metric in counter.collect():
        for sample in metric.samples:
            value = sample.labels.get(label, "unknown")
            result[value] = result.get(value, 0) + int(sample.value)
    return result


def build_stats(collections: list | None = None) -> dict:
    from app.main import llm_client

    llm_status = None
    if llm_client:
        llm_status = {
            "total_keys": len(llm_client.keys),
            "available_keys": sum(1 for k in llm_client.keys if k.is_available()),
            "keys": llm_client.get_key_statuses(),
        }

    return {
        "llm": llm_status,
        "rate_limiter": limiter.stats(),
        "counters": {
            "http_requests": _counter_total(metrics.http_requests_total),
            "http_errors": _counter_total(metrics.http_errors_total),
            "llm_requests": _counter_total(metrics.llm_requests_total),
            "guardrail_hits": _counter_total(metrics.guardrail_hits_total),
            "agent_calls": _counter_total(metrics.agent_calls_total),
            "rag_retrievals": _counter_total(metrics.rag_retrieval_count),
        },
        "by_endpoint": _counter_by_label(metrics.llm_requests_total, "endpoint"),
        "by_rule": _counter_by_label(metrics.guardrail_hits_total, "rule_type"),
        "by_collection": _counter_by_label(metrics.rag_retrieval_count, "collection"),
        "collections": collections,
    }
