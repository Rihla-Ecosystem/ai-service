"""Request-scoped pre-provider execution limits for billed AI operations."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

AI_CHAT_QUERY = "AI_CHAT_QUERY"
AI_IMAGE_ANALYSIS = "AI_IMAGE_ANALYSIS"
REAL_TIME_TRANSLATION = "REAL_TIME_TRANSLATION"
AI_TRIP_ITINERARY = "AI_TRIP_ITINERARY"

# This is the single provider-execution policy. Budgets are installed once at
# a billed operation boundary and are cumulative across all nested Gemini calls.
AI_EXECUTION_POLICIES = {
    AI_CHAT_QUERY: {"max_input_tokens": 6_000, "max_output_tokens": 800},
    AI_IMAGE_ANALYSIS: {"max_input_tokens": 3_000, "max_output_tokens": 400},
    REAL_TIME_TRANSLATION: {"max_input_tokens": 1_000, "max_output_tokens": 500},
    AI_TRIP_ITINERARY: {"max_input_tokens": 8_000, "max_output_tokens": 1_000},
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
CHAT_MAX_INPUT_TOKENS = AI_EXECUTION_POLICIES[AI_CHAT_QUERY]["max_input_tokens"]
OPERATION_MAX_OUTPUT_TOKENS = AI_EXECUTION_POLICIES[AI_CHAT_QUERY]["max_output_tokens"]


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


def begin_execution_budget(feature: str = AI_CHAT_QUERY) -> ExecutionBudget:
    """Install a policy only when this is the outermost billed operation.

    A tool can invoke itinerary generation while servicing a chat turn. In
    that case the existing ContextVar budget is deliberately retained, so the
    nested Gemini calls draw down the Chat policy rather than opening a fresh
    itinerary allocation.
    """
    existing = _budget.get()
    if existing is not None:
        return existing
    try:
        policy = AI_EXECUTION_POLICIES[feature]
    except KeyError as exc:
        raise ValueError(f"Unknown AI execution policy: {feature}") from exc
    budget = ExecutionBudget(**policy)
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
        import structlog
        structlog.get_logger().warning(
            "DEBUG input budget exceeded",
            estimated=est,
            remaining=budget.remaining_input_tokens,
            consumed=budget.consumed_input_tokens,
            sys_prompt_chars=len(system_prompt or ""),
            user_msg_chars=len(user_message or ""),
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
