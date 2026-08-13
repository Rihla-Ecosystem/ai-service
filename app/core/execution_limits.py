"""Request-scoped pre-provider execution limits for billed AI operations."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Optional

AI_CHAT_QUERY = "AI_CHAT_QUERY"
AI_IMAGE_ANALYSIS = "AI_IMAGE_ANALYSIS"
REAL_TIME_TRANSLATION = "REAL_TIME_TRANSLATION"
AI_TRIP_ITINERARY = "AI_TRIP_ITINERARY"
AI_CONTEXT_ANALYZE = "AI_CONTEXT_ANALYZE"

# Absolute local safety ceilings. Business budgets are supplied by Core per
# request and are clamped here; these values are never a second business policy.
AI_EXECUTION_SAFETY_CEILINGS = {
    AI_CHAT_QUERY: {"max_input_tokens": 16_000, "max_output_tokens": 2_048},
    AI_IMAGE_ANALYSIS: {"max_input_tokens": 4_000, "max_output_tokens": 1_024},
    REAL_TIME_TRANSLATION: {"max_input_tokens": 2_000, "max_output_tokens": 1_024},
    AI_TRIP_ITINERARY: {"max_input_tokens": 12_000, "max_output_tokens": 2_048},
    AI_CONTEXT_ANALYZE: {"max_input_tokens": 4_000, "max_output_tokens": 1_024},
}

# Voice media exposure is deliberately separate from text ExecutionBudget.
VOICE_MEDIA_EXECUTION_POLICY = {
    "max_audio_duration_seconds": 60,
    "max_audio_input_tokens": 1_920,
    "max_tts_input_chars": 500,
    "max_tts_output_tokens": 750,
    "max_tts_calls_per_operation": 1,
}

# Compatibility aliases for callers/tests that only need the Chat policy.
CHAT_MAX_INPUT_TOKENS = AI_EXECUTION_SAFETY_CEILINGS[AI_CHAT_QUERY]["max_input_tokens"]
OPERATION_MAX_OUTPUT_TOKENS = AI_EXECUTION_SAFETY_CEILINGS[AI_CHAT_QUERY]["max_output_tokens"]


class ExecutionLimitExceeded(RuntimeError):
    pass


@dataclass
class ExecutionBudget:
    max_input_tokens: int = CHAT_MAX_INPUT_TOKENS
    max_output_tokens: int = OPERATION_MAX_OUTPUT_TOKENS
    consumed_output_tokens: int = 0
    consumed_input_tokens: int = 0

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.consumed_output_tokens)

    @property
    def remaining_input_tokens(self) -> int:
        return max(0, self.max_input_tokens - self.consumed_input_tokens)


_budget: ContextVar[Optional[ExecutionBudget]] = ContextVar("rihla_execution_budget", default=None)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"executionBudget.{name} must be a positive integer")
    return value


def parse_execution_budget(feature: str, requested: Optional[Mapping[str, Any]]) -> ExecutionBudget:
    """Validate Core's contract and clamp it to local non-business ceilings."""
    try:
        ceiling = AI_EXECUTION_SAFETY_CEILINGS[feature]
    except KeyError as exc:
        raise ValueError(f"Unknown AI execution policy: {feature}") from exc
    if requested is None:
        # Compatibility for internal/direct callers; public billed Core calls
        # always transmit a budget.
        return ExecutionBudget(**ceiling)
    max_input = _positive_int(requested.get("maxInputTokens"), "maxInputTokens")
    max_output = _positive_int(requested.get("maxOutputTokens"), "maxOutputTokens")
    return ExecutionBudget(
        max_input_tokens=min(max_input, ceiling["max_input_tokens"]),
        max_output_tokens=min(max_output, ceiling["max_output_tokens"]),
    )


def execution_budget_limit(
    requested: Optional[Mapping[str, Any]], key: str, default: int, ceiling: int, allow_zero: bool = False,
) -> int:
    """Read an optional non-financial limit from the validated wire object."""
    if requested is None or key not in requested:
        return min(default, ceiling)
    value = requested[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"executionBudget.{key} must be a {'non-negative' if allow_zero else 'positive'} integer")
    return min(value, ceiling)


def begin_execution_budget(
    feature: str = AI_CHAT_QUERY,
    requested: Optional[Mapping[str, Any]] = None,
) -> ExecutionBudget:
    """Install a policy only when this is the outermost billed operation.

    A tool can invoke itinerary generation while servicing a chat turn. In
    that case the existing ContextVar budget is deliberately retained, so the
    nested Gemini calls draw down the Chat policy rather than opening a fresh
    itinerary allocation.
    """
    existing = _budget.get()
    if existing is not None:
        return existing
    budget = parse_execution_budget(feature, requested)
    _budget.set(budget)
    return budget


def end_execution_budget() -> None:
    _budget.set(None)


def current_execution_budget() -> Optional[ExecutionBudget]:
    return _budget.get()


def estimate_text_tokens(*texts: str) -> int:
    """Conservative, dependency-free preflight estimate (one token per 4 chars)."""
    return sum((len(text) + 3) // 4 for text in texts if text)


def enforce_input_budget(system_prompt: str, user_message: str) -> None:
    budget = current_execution_budget()
    if budget is None:
        return
    est = estimate_text_tokens(system_prompt, user_message)
    if est > budget.remaining_input_tokens:
        import logging
        logging.getLogger(__name__).warning(
            "Input budget exceeded estimated=%s remaining=%s consumed=%s",
            est, budget.remaining_input_tokens, budget.consumed_input_tokens,
        )
        raise ExecutionLimitExceeded("Provider-visible input exceeds the operation limit")


def output_limit(requested_limit: int) -> int:
    budget = current_execution_budget()
    if budget is None:
        return requested_limit
    remaining = budget.remaining_output_tokens
    if remaining <= 0:
        raise ExecutionLimitExceeded("Operation output budget is exhausted")
    return min(requested_limit, remaining)


def record_output_tokens(tokens: Optional[int], operation: Optional[str]) -> None:
    # TTS uses AUDIO response semantics and intentionally remains outside this
    # text-output budget until those provider semantics are verified.
    if operation == "TEXT_TO_SPEECH" or tokens is None:
        return
    budget = current_execution_budget()
    if budget is not None:
        budget.consumed_output_tokens += max(0, tokens)


def record_input_tokens(
    tokens: Optional[int], operation: Optional[str], audio_tokens: Optional[int] = None
) -> None:
    """Consume actual provider-reported input once per completed logical call."""
    if operation == "TEXT_TO_SPEECH" or tokens is None:
        return
    budget = current_execution_budget()
    if budget is not None:
        # Gemini's aggregate input count includes audio. The Voice text budget
        # must consume only the text portion; audio is bounded separately by
        # VOICE_MEDIA_EXECUTION_POLICY.
        text_tokens = tokens
        if operation == "AUDIO_UNDERSTANDING" and audio_tokens is not None:
            text_tokens = max(0, tokens - max(0, audio_tokens))
        budget.consumed_input_tokens += max(0, text_tokens)
