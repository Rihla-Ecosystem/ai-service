"""Focused tests for stream usage consumption — success and failure paths."""

import asyncio

from app.core.llm_client import GeminiClient
from app.core.usage import (
    begin_usage_tracking,
    consume_usage,
    derive_legacy_usage,
    final_stream_usage,
    make_provider_call,
    record_provider_call,
)


class TestConsumeUsageExactlyOnce:
    def test_consume_called_exactly_once_on_success(self):
        begin_usage_tracking()
        record_provider_call(
            make_provider_call(
                inputTokens=100,
                outputTokens=40,
                totalTokens=140,
                operation="TEXT_CHAT_STREAM",
            )
        )
        calls = consume_usage()
        assert len(calls) == 1
        assert calls[0]["totalTokens"] == 140
        assert consume_usage() == []

    def test_consume_returns_empty_when_nothing_recorded(self):
        begin_usage_tracking()
        assert consume_usage() == []
        assert consume_usage() == []

    def test_consume_after_partial_stream_returns_partial_usage(self):
        begin_usage_tracking()
        record_provider_call(
            make_provider_call(
                inputTokens=50,
                outputTokens=10,
                totalTokens=60,
                operation="TEXT_CHAT_STREAM",
            )
        )
        calls = consume_usage()
        assert len(calls) == 1
        assert calls[0]["inputTokens"] == 50
        assert consume_usage() == []


class TestPartialStreamFailure:
    def test_partial_failure_records_last_snapshot(self, monkeypatch):
        class _Meta:
            def __init__(self, prompt, candidates, total):
                self.prompt_token_count = prompt
                self.candidates_token_count = candidates
                self.total_token_count = total

        class _Chunk:
            def __init__(self, text, meta=None, model_version=None):
                self.text = text
                self.usage_metadata = meta
                self.model_version = model_version

        client = GeminiClient(api_keys=["dummy-key"])
        key = client.keys[0]

        class _FakeModels:
            def generate_content_stream(self, **kw):
                yield _Chunk("Hello", _Meta(100, 10, 110), "gemini-3.6-flash")
                yield _Chunk("Hello world", _Meta(100, 25, 125), "gemini-3.6-flash")
                raise RuntimeError("provider exploded after two chunks")

        class _FakeClient:
            models = _FakeModels()

        monkeypatch.setattr(key, "client", _FakeClient())

        begin_usage_tracking()

        collected = []
        loop_errored = False

        async def _run():
            nonlocal loop_errored
            gen = await client.generate(
                system_prompt="",
                user_message="hi",
                stream=True,
            )
            try:
                async for text in gen:
                    collected.append(text)
            except RuntimeError:
                loop_errored = True

        asyncio.run(_run())

        calls = consume_usage()
        assert loop_errored is True
        assert len(calls) == 1
        latest = calls[0]
        assert latest["totalTokens"] == 125
        assert latest["inputTokens"] == 100
        assert latest["outputTokens"] == 25
        assert latest["usageSource"] == "STREAM_FINAL"
        assert "".join(collected) == "HelloHello world"

    def test_failure_before_any_usage_yields_unavailable(self):
        begin_usage_tracking()
        record_provider_call(
            make_provider_call(
                provider_call_made=True,
                usage_completeness="UNAVAILABLE",
                usage_source="STREAM_FINAL",
            )
        )
        calls = consume_usage()
        assert len(calls) == 1
        assert "totalTokens" not in calls[0]
        assert calls[0]["usageCompleteness"] == "UNAVAILABLE"

    def test_no_duplicate_provider_calls(self):
        begin_usage_tracking()
        record_provider_call(
            make_provider_call(
                inputTokens=1,
                outputTokens=1,
                totalTokens=2,
                providerCallId="call-1",
            )
        )
        calls1 = consume_usage()
        consume_usage()
        begin_usage_tracking()
        record_provider_call(
            make_provider_call(
                inputTokens=3,
                outputTokens=3,
                totalTokens=6,
                providerCallId="call-1",
            )
        )
        calls2 = consume_usage()
        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_derive_legacy_from_partial_data(self):
        calls = [
            {
                "inputTokens": 100,
                "outputTokens": 25,
                "totalTokens": 125,
                "actualModel": "gemini-3.6-flash",
            },
        ]
        usage = derive_legacy_usage(calls)
        assert usage["inputTokens"] == 100
        assert usage["totalTokens"] == 125


class TestSuccessfulStreamUnchanged:
    def test_snapshots_100_40_140_not_summed(self):
        snapshots = [
            {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            {"inputTokens": 100, "outputTokens": 25, "totalTokens": 125},
            {"inputTokens": 100, "outputTokens": 40, "totalTokens": 140},
        ]
        result = final_stream_usage(snapshots)
        assert result["inputTokens"] == 100
        assert result["outputTokens"] == 40
        assert result["totalTokens"] == 140

    def test_no_snapshots_returns_none(self):
        result = final_stream_usage([])
        assert result is None

    def test_all_none_snapshots_returns_none(self):
        result = final_stream_usage([None, {}, None])
        assert result is None
