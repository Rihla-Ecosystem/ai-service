"""Tests for provider-call recording in the Gemini client.

These tests patch out only the model client (no network / no real API keys) and
verify that ``generate`` and streaming record exactly the right ProviderCallUsage
records on the request-scoped accumulator.
"""

import asyncio

from app.core.usage import begin_usage_tracking, consume_usage


class _UsageMeta:
    def __init__(self, prompt, candidates, total):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class _Chunk:
    def __init__(self, text=None, meta=None, model_version=None):
        self.text = text
        self.usage_metadata = meta
        self.model_version = model_version


def _make_client(monkeypatch):
    from app.core.llm_client import GeminiClient

    client = GeminiClient(api_keys=["dummy-key"])
    key = client.keys[0]

    class _Models:
        def __init__(self, chunks):
            self._chunks = chunks
            self._last_response = None

        def generate_content(self, **kw):
            return self._last_response

        def generate_content_stream(self, **kw):
            yield from self._chunks

    class _FakeClient:
        def __init__(self):
            self.models = _Models([])

    fake = _FakeClient()
    monkeypatch.setattr(key, "client", fake)
    return client, fake


class TestGenerateStreamRecordsSingleEntry:
    def test_stream_records_final_snapshot_once(self, monkeypatch):
        client, fake = _make_client(monkeypatch)
        fake.models._chunks = [
            _Chunk(
                text="Hello",
                meta=_UsageMeta(100, 10, 110),
                model_version="gemini-3.6-flash",
            ),
            _Chunk(
                text="Hello world",
                meta=_UsageMeta(100, 25, 125),
                model_version="gemini-3.6-flash",
            ),
            _Chunk(
                text="Hello world!",
                meta=_UsageMeta(100, 40, 140),
                model_version="gemini-3.6-flash",
            ),
        ]
        begin_usage_tracking()
        collected = []

        async def _run():
            gen = await client.generate(
                system_prompt="",
                user_message="hi",
                stream=True,
            )
            async for text in gen:
                collected.append(text)

        import asyncio

        asyncio.run(_run())
        calls = consume_usage()

        assert len(calls) == 1
        entry = calls[0]
        assert entry["providerCallId"] == "call-1"
        assert entry["totalTokens"] == 140
        assert entry["inputTokens"] == 100
        assert entry["outputTokens"] == 40
        assert entry["usageSource"] == "STREAM_FINAL"
        assert "".join(collected) == "HelloHello worldHello world!"
        assert entry["operation"] == "TEXT_CHAT_STREAM"


class TestNonStreamRecordsSingleEntry:
    def test_generate_records_provider_response(self, monkeypatch):
        client, fake = _make_client(monkeypatch)

        class _Resp:
            def __init__(self):
                self.model_version = "gemini-3.6-flash"
                self.usage_metadata = _UsageMeta(100, 40, 140)

        fake.models._last_response = _Resp()
        begin_usage_tracking()

        async def _run():
            await client.generate(
                system_prompt="", user_message="hi", operation="TEXT_GENERATION"
            )

        asyncio.run(_run())
        calls = consume_usage()

        assert len(calls) == 1
        entry = calls[0]
        assert entry["usageSource"] == "PROVIDER_RESPONSE"
        assert entry["totalTokens"] == 140
        assert entry["actualModel"] == "gemini-3.6-flash"

    def test_generate_emits_unavailable_without_usage(self, monkeypatch):
        client, fake = _make_client(monkeypatch)

        class _RespNoUsage:
            text = "ok"

        fake.models._last_response = _RespNoUsage()
        begin_usage_tracking()

        async def _run():
            await client.generate(system_prompt="", user_message="hi")

        asyncio.run(_run())
        calls = consume_usage()

        assert len(calls) == 1
        assert calls[0]["usageCompleteness"] == "UNAVAILABLE"
        assert "totalTokens" not in calls[0]


class TestMultipleCallsGetDistinctIds:
    def test_two_calls_same_model_two_ids(self, monkeypatch):
        client, fake = _make_client(monkeypatch)

        class _Resp:
            model_version = "gemini-3.6-flash"

        fake.models._last_response = _Resp()

        async def _run():
            await client.generate_with_tools(
                system_prompt="", user_message="a", tools=[]
            )
            await client.generate_with_tools(
                system_prompt="", user_message="b", tools=[]
            )

        begin_usage_tracking()
        asyncio.run(_run())
        calls = consume_usage()

        assert [c["providerCallId"] for c in calls] == ["call-1", "call-2"]


class TestDemoOutputCaps:
    def test_tool_image_and_audio_configs_are_capped_at_chat_limit(self, monkeypatch):
        from app.core.llm_client import CHAT_MAX_OUTPUT_TOKENS

        client, fake = _make_client(monkeypatch)
        configs = []

        class _Resp:
            model_version = "gemini-3.6-flash"

        def capture_config(**kwargs):
            configs.append(kwargs["config"])
            return _Resp()

        fake.models.generate_content = capture_config

        async def _run():
            await client.generate_with_tools(system_prompt="", user_message="hi", tools=[])
            await client.generate_with_image(
                system_prompt="", user_message="identify", image_bytes=b"image"
            )
            await client.generate_with_audio(system_prompt="", audio_bytes=b"audio")

        asyncio.run(_run())

        assert CHAT_MAX_OUTPUT_TOKENS == 1200
        assert [config.max_output_tokens for config in configs] == [1200, 1200, 1200]
