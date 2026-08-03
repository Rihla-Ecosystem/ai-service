"""Unit tests for the provider-neutral usage accounting contract.

These tests exercise ``app.core.usage`` in isolation (stdlib only) so they do
not require the Google SDK, structlog, or a live provider.
"""

from app.core.usage import (
    OP_IMAGE_ANALYSIS,
    OP_TEXT_CHAT,
    PROVIDER_GOOGLE,
    begin_usage_tracking,
    consume_usage,
    derive_legacy_usage,
    final_stream_usage,
    make_provider_call,
    record_provider_call,
)


class TestMakeProviderCall:
    def test_omits_unknown_fields(self):
        call = make_provider_call(
            requested_model="gemini-3.6-flash",
            actual_model="gemini-3.6-flash",
            operation=OP_TEXT_CHAT,
        )
        assert call["provider"] == PROVIDER_GOOGLE
        assert call["providerCallMade"] is True
        assert call["requestedModel"] == "gemini-3.6-flash"
        assert "totalTokens" not in call
        assert "inputTokens" not in call

    def test_includes_reported_counts(self):
        call = make_provider_call(
            requested_model="m",
            operation=OP_TEXT_CHAT,
            inputTokens=100,
            outputTokens=40,
            totalTokens=140,
        )
        assert call["inputTokens"] == 100
        assert call["outputTokens"] == 40
        assert call["totalTokens"] == 140

    def test_no_provider_request_id_by_default(self):
        call = make_provider_call(operation=OP_TEXT_CHAT)
        assert "providerRequestId" not in call

    def test_explicit_provider_request_id(self):
        call = make_provider_call(operation=OP_TEXT_CHAT, provider_request_id="req-1")
        assert call["providerRequestId"] == "req-1"


class TestUsageScope:
    def test_deterministic_call_ids(self):
        begin_usage_tracking()
        record_provider_call(make_provider_call(operation=OP_TEXT_CHAT))
        record_provider_call(make_provider_call(operation=OP_TEXT_CHAT))
        calls = consume_usage()
        assert [c["providerCallId"] for c in calls] == ["call-1", "call-2"]

    def test_reset_between_scopes(self):
        begin_usage_tracking()
        record_provider_call(make_provider_call(operation=OP_TEXT_CHAT))
        consume_usage()
        begin_usage_tracking()
        record_provider_call(make_provider_call(operation=OP_IMAGE_ANALYSIS))
        calls = consume_usage()
        assert [c["providerCallId"] for c in calls] == ["call-1"]

    def test_consume_usage_returns_list_when_no_scope(self):
        consume_usage()
        assert consume_usage() == []

    def test_record_without_scope_is_noop(self):
        consume_usage()
        record_provider_call(make_provider_call(operation=OP_TEXT_CHAT))
        assert consume_usage() == []


class TestDeriveLegacyUsage:
    def test_none_when_empty(self):
        assert derive_legacy_usage([]) is None

    def test_sums_across_calls(self):
        calls = [
            {
                "inputTokens": 100,
                "outputTokens": 10,
                "totalTokens": 110,
                "actualModel": "a",
            },
            {
                "inputTokens": 50,
                "outputTokens": 20,
                "totalTokens": 70,
                "actualModel": "b",
            },
        ]
        usage = derive_legacy_usage(calls)
        assert usage["inputTokens"] == 150
        assert usage["outputTokens"] == 30
        assert usage["totalTokens"] == 180
        assert usage["model"] == "a"

    def test_first_model_prefers_actual(self):
        calls = [
            {
                "inputTokens": 1,
                "outputTokens": 1,
                "totalTokens": 2,
                "requestedModel": "req",
                "actualModel": "act",
            },
        ]
        assert derive_legacy_usage(calls)["model"] == "act"

    def test_omits_unreported_fields(self):
        calls = [{"inputTokens": 5, "outputTokens": 3}]
        usage = derive_legacy_usage(calls)
        assert usage["inputTokens"] == 5
        assert usage["outputTokens"] == 3
        assert "totalTokens" not in usage

    def test_does_not_fabricate_zeros_when_no_tokens(self):
        calls = [{"provider": PROVIDER_GOOGLE, "providerCallMade": False}]
        assert derive_legacy_usage(calls) is None

    def test_never_derives_total_from_sum(self):
        calls = [{"inputTokens": 7, "outputTokens": 2}]
        usage = derive_legacy_usage(calls)
        assert "totalTokens" not in usage


class TestFinalStreamUsage:
    def test_last_non_empty_snapshot_wins(self):
        snapshots = [
            {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            {"inputTokens": 100, "outputTokens": 25, "totalTokens": 125},
            {"inputTokens": 100, "outputTokens": 40, "totalTokens": 140},
        ]
        assert final_stream_usage(snapshots)["totalTokens"] == 140

    def test_skips_empty_snapshots_between(self):
        snapshots = [{"totalTokens": 5}, {}, {"totalTokens": 9}]
        assert final_stream_usage(snapshots)["totalTokens"] == 9

    def test_none_when_no_snapshot(self):
        assert final_stream_usage([]) is None
        assert final_stream_usage([None, {}, None]) is None


class TestContractEnumsSurfaced:
    def test_token_fields_exposed(self):
        from app.core.usage import TOKEN_FIELD_NAMES

        assert "totalTokens" in TOKEN_FIELD_NAMES
        assert "cachedInputTokens" in TOKEN_FIELD_NAMES
        assert "reasoningTokens" in TOKEN_FIELD_NAMES
