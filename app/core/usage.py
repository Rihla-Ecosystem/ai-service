"""Provider-neutral per-provider-call usage accounting for Rihla.

This module is intentionally dependency-free (stdlib only). It defines the
ProviderCallUsage contract emitted to the Core Server, the per-request usage
scope that assigns deterministic providerCallId values, the enums used across
the contract, and small pure helpers for deriving legacy aggregate usage and
reducing cumulative stream snapshots to a single final snapshot.

The Gemini-specific extraction of token counts lives in
``app.core.gemini_usage``; nothing in this module reads provider-native field
names.
"""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# operation enums
# ---------------------------------------------------------------------------
OP_TEXT_CHAT = "TEXT_CHAT"
OP_TEXT_CHAT_STREAM = "TEXT_CHAT_STREAM"
OP_IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
OP_AUDIO_UNDERSTANDING = "AUDIO_UNDERSTANDING"
OP_SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
OP_TEXT_GENERATION = "TEXT_GENERATION"
OP_TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
OP_REALTIME_AUDIO = "REALTIME_AUDIO"
OP_ITINERARY_GENERATION = "ITINERARY_GENERATION"
OP_OTHER = "OTHER"

# ---------------------------------------------------------------------------
# usageSource enums
# ---------------------------------------------------------------------------
USAGE_SOURCE_PROVIDER_RESPONSE = "PROVIDER_RESPONSE"
USAGE_SOURCE_STREAM_FINAL = "STREAM_FINAL"
USAGE_SOURCE_STREAM_EVENT = "STREAM_EVENT"
USAGE_SOURCE_DERIVED_FROM_PROVIDER_FIELDS = "DERIVED_FROM_PROVIDER_FIELDS"
USAGE_SOURCE_NO_PROVIDER_CALL = "NO_PROVIDER_CALL"
USAGE_SOURCE_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# usageCompleteness enums
# ---------------------------------------------------------------------------
USAGE_COMPLETENESS_COMPLETE = "COMPLETE"
USAGE_COMPLETENESS_PARTIAL = "PARTIAL"
USAGE_COMPLETENESS_UNAVAILABLE = "UNAVAILABLE"
USAGE_COMPLETENESS_UNVERIFIED = "UNVERIFIED"

# ---------------------------------------------------------------------------
# accountingSemantics enums
# ---------------------------------------------------------------------------
ACCOUNTING_INCLUDED_IN_AGGREGATE = "INCLUDED_IN_AGGREGATE"
ACCOUNTING_SEPARATELY_BILLABLE = "SEPARATELY_BILLABLE"
ACCOUNTING_BREAKDOWN_ONLY = "BREAKDOWN_ONLY"
ACCOUNTING_DERIVED = "DERIVED"
ACCOUNTING_TELEMETRY_ONLY = "TELEMETRY_ONLY"
ACCOUNTING_UNKNOWN = "UNKNOWN"

# Provider identifiers are provider-neutral tokens; the only provider wired in
# Phase 1 is Google Gemini, but the field name carries no provider-native
# schema.
PROVIDER_GOOGLE = "google"

# All optional numeric fields of the ProviderCallUsage contract.
TOKEN_FIELD_NAMES: tuple = (
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cachedInputTokens",
    "cachedOutputTokens",
    "cacheWriteInputTokens",
    "reasoningTokens",
    "imageInputTokens",
    "imageOutputTokens",
    "audioInputTokens",
    "audioOutputTokens",
    "cachedAudioInputTokens",
    "cachedAudioOutputTokens",
    "audioInputSeconds",
    "audioOutputSeconds",
    "transcriptionSeconds",
    "inputCharacters",
    "outputCharacters",
    "generatedImageCount",
)


def make_provider_call(
    *,
    provider: str = PROVIDER_GOOGLE,
    requested_model: Optional[str] = None,
    actual_model: Optional[str] = None,
    operation: Optional[str] = None,
    provider_call_made: bool = True,
    provider_request_id: Optional[str] = None,
    usage_source: Optional[str] = None,
    usage_completeness: Optional[str] = None,
    accounting_semantics: Optional[str] = None,
    **counts: Any,
) -> Dict[str, Any]:
    """Build a ProviderCallUsage record with unknown values left absent.

    Unknown optional values are omitted from the returned dictionary rather
    than represented as zero. Zeros only appear when an explicit count is
    passed in (which producers only do when the provider reported it).
    ``providerCallId`` is intentionally not set here; it is assigned by the
    request-scope ``record_provider_call`` so ids are deterministic and unique
    within a single user operation.
    """
    call: Dict[str, Any] = {
        "provider": provider,
        "providerCallMade": provider_call_made,
        "requestedModel": requested_model,
        "actualModel": actual_model,
        "operation": operation,
        "usageSource": usage_source,
        "usageCompleteness": usage_completeness,
        "accountingSemantics": accounting_semantics,
    }
    if provider_request_id is not None:
        call["providerRequestId"] = provider_request_id
    for field in TOKEN_FIELD_NAMES:
        value = counts.get(field)
        if value is not None:
            call[field] = value
    return {k: v for k, v in call.items() if v is not None}


class UsageScope:
    """Per-request accumulator of ProviderCallUsage records.

    Each record receives a deterministic, test-friendly ``providerCallId``
    (``call-1``, ``call-2``, ...) assigned in append order, so two provider
    calls in the same operation always have distinct ids even when they use the
    same model.
    """

    def __init__(self) -> None:
        self._calls: List[Dict[str, Any]] = []

    def next_call_id(self) -> str:
        return f"call-{len(self._calls) + 1}"

    def add(self, call: Dict[str, Any]) -> None:
        call["providerCallId"] = self.next_call_id()
        self._calls.append(call)

    def calls(self) -> List[Dict[str, Any]]:
        return list(self._calls)


_usage_accumulator: ContextVar = ContextVar("rihla_usage_accumulator", default=None)


def begin_usage_tracking() -> None:
    """Start a fresh provider-call scope for the current request context."""
    _usage_accumulator.set(UsageScope())


def consume_usage() -> List[Dict[str, Any]]:
    """Return accumulated provider-call records for the current request and reset."""
    scope = _usage_accumulator.get()
    _usage_accumulator.set(None)
    if scope is None:
        return []
    return scope.calls()


def record_provider_call(call: Dict[str, Any]) -> None:
    """Assign the next providerCallId and append the record to the current scope."""
    scope = _usage_accumulator.get()
    if scope is None:
        return
    scope.add(call)


def derive_legacy_usage(
    calls: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Derive the legacy aggregate ``usage`` object for backward compatibility.

    The legacy object is a compatibility artifact only. It sums token counts
    across distinct provider-call records and attributes the aggregate to the
    first record that exposes a model. It is NOT authoritative for pricing; the
    ``providerCalls`` array is.

    Unknown values are never converted into misleading zeros: a token field is
    only emitted when at least one provider call reported it, and ``totalTokens``
    is never derived from ``inputTokens + outputTokens`` because provider
    accounting may include categories beyond those two.
    """
    if not calls:
        return None

    aggregated: Dict[str, int] = {}
    for field in ("inputTokens", "outputTokens", "totalTokens"):
        values = [c[field] for c in calls if c.get(field) is not None]
        if values:
            aggregated[field] = sum(values)
    if not aggregated:
        return None

    model = None
    for c in calls:
        m = c.get("actualModel") or c.get("requestedModel")
        if m:
            model = m
            break
    result: Dict[str, Any] = {"model": model}
    result.update(aggregated)
    return result


def final_stream_usage(
    snapshots: List[Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Return the final cumulative usage snapshot for a streamed provider call.

    Gemini streaming ``usageMetadata`` is cumulative: every chunk reports the
    running totals so far. The correct single final usage is therefore the last
    non-empty snapshot, never the sum of all snapshots. If no chunk carried a
    snapshot, returns ``None`` (no fabricated usage).
    """
    for snapshot in reversed(snapshots):
        if snapshot:
            return snapshot
    return None
