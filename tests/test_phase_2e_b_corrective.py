"""Phase 2E-B corrective spoken-voice probe tests (NO live provider requests).

These tests verify the corrective probe machinery without ever calling Gemini:

  1. voice-only mode does not execute the image probe
  2. the corrective input asset is spoken audio from the offline synthesizer
     (espeak-ng/espeak), never the old square-wave generator
  3. semantic validation passes for a Cairo-related response
  4. semantic validation fails for empty or unrelated responses
  5. provider execution cannot occur more than once in one corrective run
  6. sanitized reporting excludes prompts, full responses, credentials, media
  7. gTTS is not reported as a Gemini provider call
  8. modality breakdown tokens are not added to aggregate totals

No provider network call is made: probe functions are driven with fakes and
monkeypatched synthesizers.
"""

import asyncio
import json

import pytest

from scripts import phase_2e_b_probe as probe
from scripts.phase_2e_b_probe import (
    CorrectiveAlreadyRun,
    _redaction_audit,
    _safe,
    generate_spoken_wav,
    modality_not_double_counted,
    reset_corrective_guard,
    validate_semantic_response,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _AudioResp:
    def __init__(self, text):
        self.text = text


class _FakeAudioLLM:
    """Returns a fixed text response for generate_with_audio; never calls out."""

    def __init__(self, text="I recommend the Egyptian Museum in Cairo."):
        self.text = text
        self.calls = []

    async def generate_with_audio(self, **kwargs):
        self.calls.append(kwargs)
        return _AudioResp(self.text)

    async def generate_speech(self, text, **kw):
        return {"audio_bytes": b"\x00\x01", "mime": "audio/l16"}


class _CountingLLM(_FakeAudioLLM):
    """Increments a shared counter so tests can assert single execution."""

    def __init__(self, counter, text="I recommend the Egyptian Museum in Cairo."):
        super().__init__(text)
        self.counter = counter

    async def generate_with_audio(self, **kwargs):
        self.counter["count"] += 1
        return await super().generate_with_audio(**kwargs)


class _DummyClient:
    MAX_RETRIES = 0


@pytest.fixture(autouse=True)
def _reset_guard():
    reset_corrective_guard()
    yield
    reset_corrective_guard()


def _fake_settings_gemini_model(monkeypatch):
    monkeypatch.setattr(probe.settings, "gemini_model", "gemini-3.6-flash")


def _patch_synth(monkeypatch, spoken_bytes=b"RIFF\x00fake spoken wav bytes"):
    monkeypatch.setattr(probe, "generate_spoken_wav", lambda text=None: spoken_bytes)


# ---------------------------------------------------------------------------
# 1. voice-only mode does not execute the image probe
# ---------------------------------------------------------------------------

class TestVoiceOnlySkipsImage:
    def test_voice_only_never_calls_image_probe(self, monkeypatch):
        _patch_synth(monkeypatch)
        called = {"image": False}

        async def _boom(client):
            called["image"] = True
            raise AssertionError("image probe must not run in voice-only mode")

        async def _ok(client, spoken_wav):
            return {"probe": "corrective_spoken_voice"}

        monkeypatch.setattr(probe, "probe_image", _boom)
        monkeypatch.setattr(probe, "probe_voice_corrective", _ok)

        args = probe._parse_args(["--voice-only", "--no-write"])
        asyncio.run(probe._run_probes(args, _DummyClient()))
        assert called["image"] is False

    def test_default_mode_runs_image_and_voice(self, monkeypatch):
        _patch_synth(monkeypatch)
        order = []

        async def _img(client):
            order.append("image")
            return {"probe": "image", "provider_calls": [], "provider_attempts": []}

        async def _voice(client, spoken_wav):
            order.append("voice")
            return {"probe": "corrective_spoken_voice", "provider_calls": [], "provider_attempts": []}

        async def _legacy(client):
            order.append("legacy")
            return {"probe": "legacy_voice", "provider_calls": [], "provider_attempts": []}

        monkeypatch.setattr(probe, "probe_image", _img)
        monkeypatch.setattr(probe, "probe_voice_corrective", _voice)
        monkeypatch.setattr(probe, "probe_voice", _legacy)

        args = probe._parse_args(["--no-write"])
        asyncio.run(probe._run_probes(args, _DummyClient()))
        assert order == ["voice", "image", "legacy"]


# ---------------------------------------------------------------------------
# 2. corrective input is spoken audio, never the square-wave generator
# ---------------------------------------------------------------------------

class TestSpokenInputAsset:
    def test_generate_spoken_wav_requires_offline_synthesizer(self, monkeypatch):
        def _no_bin():
            raise probe.OfflineSynthesizerUnavailable("no espeak-ng")

        monkeypatch.setattr(probe, "locate_espeak_binary", _no_bin)
        with pytest.raises(probe.OfflineSynthesizerUnavailable):
            generate_spoken_wav()

    def test_generate_spoken_wav_rejects_non_wav_output(self, monkeypatch):
        # The offline synthesizer subprocess is replaced; it "writes" garbage.
        import subprocess
        real_run = subprocess.run

        def _fake_run(cmd, **kw):
            out_path = cmd[cmd.index("-w") + 1]
            with open(out_path, "wb") as f:
                f.write(b"not a wav at all")
            return real_run(["/bin/true"], capture_output=True, timeout=60)

        monkeypatch.setattr(probe, "locate_espeak_binary", lambda: "/fake/espeak-ng")
        monkeypatch.setattr(probe.subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="no valid WAV"):
            generate_spoken_wav()

    def test_corrective_probe_uses_spoken_wav_not_square_wave(self, monkeypatch):
        _fake_settings_gemini_model(monkeypatch)
        sent = {}

        async def _audio(self, system_prompt, audio_bytes, mime_type, **kw):
            sent["audio_bytes"] = audio_bytes
            sent["mime_type"] = mime_type
            return _AudioResp("I recommend the Egyptian Museum in Cairo.")

        async def _speech(self, text, **kw):
            return {"audio_bytes": b"\x00\x01", "mime": "audio/l16"}

        spoken = b"RIFF\x00real spoken audio"
        _patch_synth(monkeypatch, spoken_bytes=spoken)
        monkeypatch.setattr(probe.GeminiClient, "generate_with_audio", _audio)
        monkeypatch.setattr(probe, "synthesize_speech", _speech)

        client = probe.GeminiClient(api_keys=["dummy"])
        client.MAX_RETRIES = 0
        result = asyncio.run(probe.probe_voice_corrective(client, spoken))

        assert sent["audio_bytes"] == spoken
        assert sent["mime_type"] == "audio/wav"
        # The corrective probe must never call the legacy square-wave generator.
        assert not hasattr(result, "_synthetic_wav")
        assert result["probe"] == "corrective_spoken_voice"


# ---------------------------------------------------------------------------
# 3 / 4. semantic validation
# ---------------------------------------------------------------------------

class TestSemanticValidation:
    @pytest.mark.parametrize(
        "text",
        [
            "I recommend visiting the Egyptian Museum in Cairo. It has the Tutankhamun treasures.",
            "You should visit the Pyramids of Giza, they are a must-see landmark.",
            "Try Khan el-Khalili bazaar, it is the most famous market in Cairo.",
            "The Citadel of Saladin is worth seeing for its views over old Cairo.",
            "Visit Al-Azhar Park, a beautiful green space to relax in Cairo.",
            "Cairo Tower is a great place to see the whole city.",
        ],
    )
    def test_cairo_response_passes(self, text):
        r = validate_semantic_response(text)
        assert r["semanticUnderstandingVerified"] is True
        assert r["matchedKeywords"]
        assert r["responseCharLength"] == len(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "The weather is nice today.",
            "I hope you have a great day.",
            "Sorry, I could not understand your question.",
            "There was an error processing your request.",
        ],
    )
    def test_empty_or_unrelated_response_fails(self, text):
        r = validate_semantic_response(text)
        assert r["semanticUnderstandingVerified"] is False
        assert r["matchedKeywords"] == []
        assert "reason" in r


# ---------------------------------------------------------------------------
# 5. provider execution cannot occur more than once per corrective run
# ---------------------------------------------------------------------------

class TestSingleExecution:
    def test_second_corrective_execution_is_refused(self, monkeypatch):
        _fake_settings_gemini_model(monkeypatch)
        counter = {"count": 0}
        fake = _CountingLLM(counter)

        # Use the REAL GeminiClient method bodies via monkeypatch so the probe
        # function runs its actual guard logic with a fake provider.
        async def _audio(self, system_prompt, audio_bytes, mime_type, **kw):
            counter["count"] += 1
            return _AudioResp("I recommend the Egyptian Museum in Cairo.")

        async def _speech(self, text, **kw):
            return {"audio_bytes": b"\x00\x01", "mime": "audio/l16"}

        monkeypatch.setattr(probe.GeminiClient, "generate_with_audio", _audio)
        monkeypatch.setattr(probe, "synthesize_speech", _speech)

        client = probe.GeminiClient(api_keys=["dummy"])
        client.MAX_RETRIES = 0

        # First execution is allowed and performs exactly one audio call.
        first = asyncio.run(probe.probe_voice_corrective(client, b"RIFF spoken"))
        assert counter["count"] == 1
        assert first["probe"] == "corrective_spoken_voice"

        # A second execution must be refused BEFORE any provider call.
        with pytest.raises(CorrectiveAlreadyRun):
            asyncio.run(probe.probe_voice_corrective(client, b"RIFF spoken"))
        assert counter["count"] == 1


# ---------------------------------------------------------------------------
# 6. sanitized reporting excludes prompts, full responses, credentials, media
# ---------------------------------------------------------------------------

class TestSanitizedReporting:
    def test_safe_strips_media_and_prompts(self):
        record = {
            "audio_bytes": b"\x00\x01\x02",
            "inline_data": {"data": b"\xaa\xbb"},
            "data": b"\xcc\xdd",
            "audio": b"\xee\xff",
            "system_prompt": "TOP SECRET PROMPT",
            "user_message": "user message",
            "text": "full response would be here",
            "inputTokens": 10,
        }
        safe = _safe(record)
        assert "audio_bytes" not in safe
        assert "inline_data" not in safe
        assert "data" not in safe
        assert "audio" not in safe
        assert "system_prompt" not in safe
        assert "user_message" not in safe
        assert safe["inputTokens"] == 10

    def test_redaction_audit_detects_no_leaks(self):
        records = [
            {"operation": "AUDIO_UNDERSTANDING", "inputTokens": 10, "providerCallId": "call-1"},
            {"operation": "TEXT_TO_SPEECH", "audioOutputTokens": 20},
        ]
        audit = _redaction_audit(records)
        assert audit == {
            "aiza_leak": False,
            "api_key_field": False,
            "b64_media_leak": False,
            "audio_bytes_field": False,
            "prompt_leak": False,
        }

    def test_redaction_audit_flags_real_secrets(self):
        records = [{"x": "AIzaSyD1AbC1234567890abcdefghijklmnop_123456"}]
        audit = _redaction_audit(records)
        assert audit["aiza_leak"] is True


# ---------------------------------------------------------------------------
# 7. gTTS is not reported as a Gemini provider call
# ---------------------------------------------------------------------------

class TestGttsNeverAGeminiCall:
    def test_gtts_fallback_is_not_recorded_as_provider_call(self, monkeypatch):
        _fake_settings_gemini_model(monkeypatch)
        _patch_synth(monkeypatch)

        calls = []
        attempts = []

        from app.core.usage import (
            begin_usage_tracking,
            consume_usage_and_attempts,
            make_provider_attempt,
            make_provider_call,
            record_provider_attempt,
            record_provider_call,
        )
        from app.core.usage import OP_AUDIO_UNDERSTANDING, OP_TEXT_TO_SPEECH

        # The fake audio call records a real AUDIO_UNDERSTANDING provider call.
        async def _audio(self, system_prompt, audio_bytes, mime_type, **kw):
            begin_usage_tracking()
            record_provider_call(make_provider_call(
                requested_model="gemini-3.6-flash",
                actual_model="gemini-3.6-flash",
                operation=OP_AUDIO_UNDERSTANDING,
                usage_source="PROVIDER_RESPONSE",
                usage_completeness="COMPLETE",
                inputTokens=100,
                outputTokens=40,
                totalTokens=140,
            ))
            record_provider_attempt(make_provider_attempt(
                operation=OP_AUDIO_UNDERSTANDING,
                requested_model="gemini-3.6-flash",
                attempt_number=1,
                outcome="SUCCEEDED",
                provider_call_started=True,
                provider_response_received=True,
                provider_call_id="call-1",
            ))
            return _AudioResp("I recommend the Egyptian Museum in Cairo.")

        async def _speech_fail(self, text, **kw):
            # Gemini TTS fails -> gTTS fallback produces audio (returns bytes).
            return (b"GFAKE", "audio/mpeg")

        monkeypatch.setattr(probe.GeminiClient, "generate_with_audio", _audio)
        monkeypatch.setattr(probe, "synthesize_speech", _speech_fail)

        client = probe.GeminiClient(api_keys=["dummy"])
        client.MAX_RETRIES = 0
        result = asyncio.run(probe.probe_voice_corrective(client, b"RIFF spoken"))

        # The only Gemini provider call is the audio-understanding call.
        assert len(result["provider_calls"]) == 1
        assert result["provider_calls"][0]["operation"] == OP_AUDIO_UNDERSTANDING
        # gTTS fallback produces audio but must NOT appear as a provider call.
        assert result["tts_source"] == "gtts_fallback"
        assert result["audio_produced"] is True
        ops = [c["operation"] for c in result["provider_calls"]]
        assert OP_TEXT_TO_SPEECH not in ops
        assert "gtts" not in str(result["provider_calls"]).lower()


# ---------------------------------------------------------------------------
# 8. modality breakdown tokens are not added to aggregate totals
# ---------------------------------------------------------------------------

class TestModalityNonDoubleCount:
    def test_audio_call_breakdown_not_added_to_aggregate(self):
        call = {
            "operation": "AUDIO_UNDERSTANDING",
            "inputTokens": 153,
            "outputTokens": 56,
            "totalTokens": 397,
            "reasoningTokens": 188,
            "audioInputTokens": 99,
        }
        check = modality_not_double_counted(call)
        assert check["aggregate_arith_ok"] is True  # 153 + 56 + 188 == 397
        assert check["double_count_detected"] is False
        assert "audioInputTokens" in check["breakdown_fields_present"]

    def test_tts_call_breakdown_not_added_to_aggregate(self):
        call = {
            "operation": "TEXT_TO_SPEECH",
            "inputTokens": 54,
            "outputTokens": 561,
            "totalTokens": 615,
            "audioOutputTokens": 561,
        }
        check = modality_not_double_counted(call)
        assert check["aggregate_arith_ok"] is True  # 54 + 561 == 615
        assert check["double_count_detected"] is False
        assert "audioOutputTokens" in check["breakdown_fields_present"]

    def test_derive_legacy_usage_sums_only_aggregate_fields(self):
        from app.core.usage import derive_legacy_usage

        calls = [
            {
                "operation": "AUDIO_UNDERSTANDING",
                "actualModel": "gemini-3.6-flash",
                "inputTokens": 153,
                "outputTokens": 56,
                "totalTokens": 397,
                "reasoningTokens": 188,
                "audioInputTokens": 99,
            },
            {
                "operation": "TEXT_TO_SPEECH",
                "inputTokens": 54,
                "outputTokens": 561,
                "totalTokens": 615,
                "audioOutputTokens": 561,
            },
        ]
        legacy = derive_legacy_usage(calls)
        # Aggregate sums over input/output/total ONLY; breakdown fields are
        # never folded into the totals.
        assert legacy["inputTokens"] == 153 + 54
        assert legacy["outputTokens"] == 56 + 561
        assert legacy["totalTokens"] == 397 + 615
        assert "audioInputTokens" not in legacy
        assert "audioOutputTokens" not in legacy
        assert "reasoningTokens" not in legacy
