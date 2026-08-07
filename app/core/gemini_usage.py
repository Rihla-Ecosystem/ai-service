"""Gemini-specific token-usage extraction for the Rihla AI Service.

Provider-native field names (``prompt_token_count``, ``total_token_count``,
``prompt_tokens_details``, ...) must stay inside this module and the AI
Service. The functions below consume Gemini response objects and return only
provider-neutral camelCase counts that the generic ``app.core.usage`` contract
understands. Duck typing (``getattr``) is used so this module is importable and
unit-testable without the ``google-genai`` SDK installed; callers may pass any
object exposing the documented attributes.
"""

from typing import Any, Dict, Optional


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def extract_response_model(response: Any) -> Optional[str]:
    """Return the actual model used by the provider response, or None.

    Reads the pinned google-genai SDK field ``model_version`` (the SDK exposes
    ``GenerateContentResponse.model_version``, not ``model``). ``actualModel``
    is never fabricated from ``requestedModel``; when ``model_version`` is
    absent, null, empty, or invalid, the field stays absent.
    """
    model = getattr(response, "model_version", None)
    if isinstance(model, str) and model:
        return model
    return None


def extract_token_counts(response: Any) -> Dict[str, Any]:
    """Extract provider-neutral token counts from a Gemini response.

    Only fields actually reported by the provider are included; missing
    fields stay absent (never zero). Reasoning, cached-input, and modality
    breakdowns (image/audio) are surfaced when present. Because Gemini reports
    per-modality counts inside the aggregate token counts (non-additive), the
    breakdown fields are BREAKDOWN_ONLY and must never be added to the
    aggregate for pricing.
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}

    out: Dict[str, Any] = {}

    prompt = getattr(meta, "prompt_token_count", None)
    candidates = getattr(meta, "candidates_token_count", None)
    total = getattr(meta, "total_token_count", None)
    cached = getattr(meta, "cached_content_token_count", None)
    thoughts = getattr(meta, "thoughts_token_count", None)

    if _is_non_negative_int(prompt):
        out["inputTokens"] = prompt
    if _is_non_negative_int(candidates):
        out["outputTokens"] = candidates
    if _is_non_negative_int(total):
        out["totalTokens"] = total
    if _is_non_negative_int(cached):
        out["cachedInputTokens"] = cached
    if _is_non_negative_int(thoughts):
        out["reasoningTokens"] = thoughts

    image_input = audio_input = 0
    for entry in getattr(meta, "prompt_tokens_details", None) or []:
        modality = getattr(entry, "modality", None)
        count = getattr(entry, "token_count", None)
        if not _is_non_negative_int(count):
            continue
        if modality == "IMAGE":
            image_input += count
        elif modality == "AUDIO":
            audio_input += count
    if image_input:
        out["imageInputTokens"] = image_input
    if audio_input:
        out["audioInputTokens"] = audio_input

    image_output = audio_output = 0
    for entry in getattr(meta, "candidates_tokens_details", None) or []:
        modality = getattr(entry, "modality", None)
        count = getattr(entry, "token_count", None)
        if not _is_non_negative_int(count):
            continue
        if modality == "IMAGE":
            image_output += count
        elif modality == "AUDIO":
            audio_output += count
    if image_output:
        out["imageOutputTokens"] = image_output
    if audio_output:
        out["audioOutputTokens"] = audio_output

    return out
