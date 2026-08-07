"""Unit tests for Gemini-specific token-usage extraction.

The extraction functions use duck typing so tests pass plain objects that mimic
the shapes returned by the ``google-genai`` SDK without importing it.
"""

from app.core.gemini_usage import extract_response_model, extract_token_counts


class _Detail:
    def __init__(self, modality, token_count):
        self.modality = modality
        self.token_count = token_count


class _Meta:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, model_version=None, meta=None):
        self.model_version = model_version
        if meta is not None:
            self.usage_metadata = meta


def _full_meta():
    return _Meta(
        prompt_token_count=1024,
        candidates_token_count=64,
        total_token_count=1100,
        cached_content_token_count=512,
        thoughts_token_count=0,
        prompt_tokens_details=[_Detail("IMAGE", 10), _Detail("AUDIO", 4)],
        candidates_tokens_details=[_Detail("IMAGE", 2), _Detail("AUDIO", 1)],
    )


class TestExtractResponseModel:
    def test_reads_model_version_from_response(self):
        assert (
            extract_response_model(_Resp(model_version="gemini-3.6-flash"))
            == "gemini-3.6-flash"
        )

    def test_response_model_alone_is_not_accepted(self):
        # The pinned google-genai SDK exposes model_version, not model. A fake
        # that only sets model must not be treated as the real SDK field.
        response = _Resp()
        response.model = "gemini-3.6-flash"
        assert extract_response_model(response) is None

    def test_none_when_missing(self):
        assert extract_response_model(_Resp()) is None

    def test_none_when_model_version_none(self):
        assert extract_response_model(_Resp(model_version=None)) is None

    def test_none_when_model_version_empty(self):
        assert extract_response_model(_Resp(model_version="")) is None

    def test_none_when_model_version_invalid(self):
        assert extract_response_model(_Resp(model_version=12345)) is None

    def test_none_when_response_is_none(self):
        assert extract_response_model(None) is None


class TestExtractTokenCounts:
    def test_full_snapshot(self):
        counts = extract_token_counts(_Resp(meta=_full_meta()))
        assert counts["inputTokens"] == 1024
        assert counts["outputTokens"] == 64
        assert counts["totalTokens"] == 1100
        assert counts["cachedInputTokens"] == 512
        assert counts["imageInputTokens"] == 10
        assert counts["audioInputTokens"] == 4
        assert counts["imageOutputTokens"] == 2
        assert counts["audioOutputTokens"] == 1

    def test_reasoning_tokens_surfaced_when_present(self):
        meta = _Meta(
            prompt_token_count=5,
            candidates_token_count=3,
            total_token_count=8,
            thoughts_token_count=2,
        )
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["reasoningTokens"] == 2

    def test_zero_reported_counts_are_preserved(self):
        meta = _Meta(
            prompt_token_count=0, candidates_token_count=0, total_token_count=0
        )
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["inputTokens"] == 0
        assert counts["outputTokens"] == 0
        assert counts["totalTokens"] == 0

    def test_absent_fields_stay_absent(self):
        meta = _Meta(prompt_token_count=10, candidates_token_count=2)
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["inputTokens"] == 10
        assert counts["outputTokens"] == 2
        assert "totalTokens" not in counts

    def test_no_metadata_returns_empty(self):
        assert extract_token_counts(_Resp()) == {}

    def test_none_returns_empty(self):
        assert extract_token_counts(None) == {}

    def test_modality_breakdowns_are_additive_for_pricing_never_used(self):
        # Breakdown counts are only surfaced under their own keys and must not
        # be merged into the aggregate input/output totals.
        counts = extract_token_counts(_Resp(meta=_full_meta()))
        assert counts["inputTokens"] == 1024
        assert counts["outputTokens"] == 64
        assert (
            counts["imageInputTokens"] + counts["audioInputTokens"]
            != counts["inputTokens"]
        )

    def test_cached_content_token_count_not_readded_to_aggregates(self):
        # cachedContentTokenCount is surfaced as cachedInputTokens and must not
        # be added back into inputTokens (which already includes it).
        meta = _Meta(
            prompt_token_count=1024,
            candidates_token_count=64,
            total_token_count=1100,
            cached_content_token_count=512,
        )
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["inputTokens"] == 1024
        assert counts["cachedInputTokens"] == 512
        assert counts["inputTokens"] != counts["inputTokens"] + counts["cachedInputTokens"]

    def test_total_tokens_is_provider_reported_not_input_plus_output(self):
        # totalTokens is taken verbatim from the provider; it is never derived
        # as inputTokens + outputTokens.
        meta = _Meta(
            prompt_token_count=100,
            candidates_token_count=40,
            total_token_count=180,
        )
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["inputTokens"] == 100
        assert counts["outputTokens"] == 40
        assert counts["totalTokens"] == 180
        assert counts["totalTokens"] != counts["inputTokens"] + counts["outputTokens"]

    def test_total_tokens_absent_when_only_parts_reported(self):
        # When the provider reports only parts, totalTokens must stay absent
        # rather than being fabricated from inputTokens + outputTokens.
        meta = _Meta(prompt_token_count=100, candidates_token_count=40)
        counts = extract_token_counts(_Resp(meta=meta))
        assert counts["inputTokens"] == 100
        assert counts["outputTokens"] == 40
        assert "totalTokens" not in counts
