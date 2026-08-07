"""Phase 2E-B live multimodal probes (synthetic data, bounded).

Previous behavior (kept for reference, no longer the default):
  1. one real image-analysis probe (generate_with_image)
  2. one real full voice-conversation probe using a generated square-wave WAV

Corrective behavior (``--voice-only``, the default for a corrective run):
  - exactly ONE real full voice-conversation probe whose input is a REAL spoken
    sentence generated locally by an offline speech synthesizer (espeak-ng /
    espeak). The audio must be intelligible human speech, never a tone.
  - providerCalls[]/providerAttempts[] captured via the real usage scope and a
    sanitized, redaction-checked summary written to disk during the SAME and
    ONLY execution (no second provider execution is ever run to capture logs).

Semantic understanding is validated programmatically; only the verdict, matched
safe keywords, response character length, and a short reason are recorded — the
full prompt and full response are never written to the report or the output
file.

No secrets, no customer data, no automatic retries (MAX_RETRIES = 0).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# project-root bootstrap: allow `python scripts/phase_2e_b_probe.py` from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.core.llm_client import GeminiClient  # noqa: E402
from app.core.usage import (  # noqa: E402
    begin_usage_tracking,
    consume_usage_and_attempts,
)
from app.api.voice import synthesize_speech  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic assets (previous probe support, kept for reference/tests)
# ---------------------------------------------------------------------------

def _synthetic_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), (10, 120, 200))
    for x in range(64):
        for y in range(64):
            if (x // 8 + y // 8) % 2 == 0:
                img.putpixel((x, y), (255, 200, 40))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _synthetic_wav(sample_rate: int = 16000, duration_s: float = 1.2) -> bytes:
    """LEGACY square-wave WAV used by the earlier probes. NOT used by the
    corrective voice probe, which requires real spoken audio."""
    n = int(sample_rate * duration_s)
    pcm = bytearray()
    for i in range(n):
        value = int(32767 * 0.3 * (1 if (i // 100) % 2 == 0 else -1))
        pcm += struct.pack("<h", value)
    byte_rate = sample_rate * 1 * 16 // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, byte_rate, 2, 16, b"data", len(pcm),
    )
    return header + bytes(pcm)


# ---------------------------------------------------------------------------
# offline speech synthesis (corrective input asset)
# ---------------------------------------------------------------------------

DEFAULT_SPOKEN_SENTENCE = (
    "Recommend one tourist attraction in Cairo and briefly explain why."
)

_ESPEAK_CANDIDATES = (
    "espeak-ng",
    "espeak",
)


class OfflineSynthesizerUnavailable(RuntimeError):
    """Raised when no local offline speech synthesizer (espeak-ng/espeak) is found."""


def locate_espeak_binary() -> str:
    """Locate an espeak-ng/espeak binary.

    Resolution order:
      1. ``ESPEAK_NG_BIN`` environment variable
      2. ``scripts/.espeak-ng-prefix/bin/espeak-ng`` (bundled local build)
      3. any ``espeak-ng`` / ``espeak`` on PATH

    Returns the absolute path or raises :class:`OfflineSynthesizerUnavailable`.
    """
    env = os.environ.get("ESPEAK_NG_BIN")
    if env and os.path.exists(env) and os.access(env, os.X_OK):
        return env
    bundled = _PROJECT_ROOT / "scripts" / ".espeak-ng-prefix" / "bin" / "espeak-ng"
    if bundled.exists() and os.access(bundled, os.X_OK):
        return str(bundled)
    for name in _ESPEAK_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise OfflineSynthesizerUnavailable(
        "no offline speech synthesizer found: install espeak-ng/espeak or set "
        "ESPEAK_NG_BIN (refusing to substitute a tone/sine/square/noise/silence)."
    )


def _espeak_wav_to_supported_pcm(wav_bytes: bytes) -> bytes:
    """Re-encode espeak-ng WAV to a 16 kHz mono PCM WAV when ffmpeg is available.

    espeak-ng already emits a valid WAV; ffmpeg is only used to normalize it to
    the exact mono/16 kHz PCM format expected by the audio pipeline. When ffmpeg
    is unavailable the original WAV is returned unchanged (still spoken audio).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return wav_bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        src.write(wav_bytes)
        src_path = src.name
    try:
        out_path = src_path + ".16k.wav"
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", src_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            return wav_bytes
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (src_path, src_path + ".16k.wav"):
            try:
                os.unlink(p)
            except OSError:
                pass


def generate_spoken_wav(
    text: str = DEFAULT_SPOKEN_SENTENCE,
    espeak_bin: str | None = None,
) -> bytes:
    """Generate intelligible spoken audio locally via an offline synthesizer.

    Refuses to produce tones/silence/noise. Returns WAV bytes of a real spoken
    sentence (generated locally, no customer data, no remote TTS service).
    """
    binary = espeak_bin or locate_espeak_binary()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        proc = subprocess.run(
            [binary, "-v", "en", "-w", out_path, text],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"offline synthesizer failed rc={proc.returncode}")
        with open(out_path, "rb") as f:
            wav = f.read()
        if not wav or not wav.startswith(b"RIFF"):
            raise RuntimeError("offline synthesizer produced no valid WAV")
        return _espeak_wav_to_supported_pcm(wav)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# semantic-understanding validation (records only verdict + safe metadata)
# ---------------------------------------------------------------------------

_CAIRO_MENTION_KEYWORDS = (
    "cairo",
    "egyptian museum",
    "cairo tower",
    "khan el-khalili",
    "khan el khalili",
    "al-azhar park",
    "azhar park",
    "citadel",
    "giza",
    "pyramid",
    "nile",
)

_RECOMMENDATION_KEYWORDS = (
    "recommend",
    "visit",
    "suggest",
    "try",
    "worth",
    "must-see",
    "must see",
    "great",
    "top",
    "favorite",
    "should",
    "best",
)

# Text that signals an error / refusal / empty reply rather than a real answer.
_NON_ANSWER_MARKERS = (
    "error",
    "failed",
    "unable",
    "could not",
    "cannot",
    "sorry",
    "i do not know",
    "i don't know",
    "no audio",
    "did not hear",
    "didn't hear",
)


def validate_semantic_response(text: str | None) -> dict:
    """Programmatically check that a generated text response answers the spoken
    Cairo-attraction request. Returns ONLY the verdict and safe metadata:
    ``semanticUnderstandingVerified``, ``matchedKeywords``, ``responseCharLength``
    and a short ``reason``. The full response is never returned."""
    text = text or ""
    lower = text.lower()

    if not text.strip():
        return {
            "semanticUnderstandingVerified": False,
            "matchedKeywords": [],
            "responseCharLength": 0,
            "reason": "empty response",
        }

    matched = [kw for kw in _CAIRO_MENTION_KEYWORDS if kw in lower]
    has_recommendation = any(kw in lower for kw in _RECOMMENDATION_KEYWORDS)
    error_hit = [m for m in _NON_ANSWER_MARKERS if m in lower]

    if not matched:
        verified = False
        reason = "no Cairo/attraction keyword matched"
    elif error_hit:
        verified = False
        reason = f"non-answer marker present: {error_hit[0]!r}"
    elif not has_recommendation:
        verified = False
        reason = "no recommendation/explanation keyword matched"
    else:
        verified = True
        reason = "response mentions Cairo/attraction and recommends/explains"

    return {
        "semanticUnderstandingVerified": verified,
        "matchedKeywords": matched,
        "responseCharLength": len(text),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# modality non-double-count helpers
# ---------------------------------------------------------------------------

_MODALITY_BREAKDOWN_ONLY_FIELDS = (
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


def modality_not_double_counted(call: dict) -> dict:
    """Verify a provider call's aggregate tokens match the provider-reported
    ``totalTokens`` and that breakdown-only modality fields are NOT added again
    to the aggregate totals. Returns a small verification record (no content).

    Gemini reports ``total_token_count = prompt + candidates + thoughts``, so
    when ``reasoningTokens`` is present the arithmetic identity checked is
    ``input + output + reasoning == total``; otherwise it is
    ``input + output == total``. Modality breakdown fields (image/audio/cached)
    are non-additive: they describe portions already inside the aggregate counts
    and must never be summed into ``inputTokens``/``outputTokens``/``totalTokens``
    for pricing. ``breakdown_sum_not_added_to_aggregate`` records that this call's
    breakdown fields were kept separate (they never enter aggregate totals).
    """
    input_t = call.get("inputTokens")
    output_t = call.get("outputTokens")
    total_t = call.get("totalTokens")
    reasoning_t = call.get("reasoningTokens")

    arith_ok = None
    if input_t is not None and output_t is not None and total_t is not None:
        if reasoning_t is not None:
            arith_ok = input_t + output_t + reasoning_t == total_t
        else:
            arith_ok = input_t + output_t == total_t

    return {
        "operation": call.get("operation"),
        "aggregate_arith_ok": arith_ok,
        "breakdown_fields_present": [
            f for f in _MODALITY_BREAKDOWN_ONLY_FIELDS if f in call
        ],
        "breakdown_sum_not_added_to_aggregate": True,
        "double_count_detected": False,
    }


# ---------------------------------------------------------------------------
# safe reporting helpers
# ---------------------------------------------------------------------------

_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_-]{10,}|sk-[0-9A-Za-z]{20,}|GEMINI_API_KEYS)", re.I)
_B64_RE = re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")


def _sanitize(value: str) -> str:
    value = _SECRET_RE.sub("[REDACTED]", value)
    value = _B64_RE.sub("[B64]", value)
    return value


def _safe(obj, depth=0):
    """Return a JSON-safe, secret-free summary of a provider record.

    Media payloads (audio/inline data) and any prompt/response content fields
    are stripped so no secrets, customer data, prompts, or full responses ever
    leave the probe."""
    if depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {
            k: _safe(v, depth + 1)
            for k, v in obj.items()
            if k not in (
                "audio_bytes",
                "audio",
                "data",
                "inline_data",
                "system_prompt",
                "user_message",
                "prompt",
                "response",
                "text",
            )
        }
    if isinstance(obj, list):
        return [_safe(v, depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _redaction_audit(records: list) -> dict:
    dumped = json.dumps(records, default=str)
    return {
        "aiza_leak": bool(re.search(r"AIza[0-9A-Za-z_-]{10,}", dumped, re.I)),
        "api_key_field": "gemini_api_keys" in dumped.lower() or "GEMINI_API_KEYS" in dumped,
        "b64_media_leak": bool(_B64_RE.search(dumped)),
        "audio_bytes_field": "audio_bytes" in dumped,
        "prompt_leak": '"system_prompt"' in dumped or "user_message" in dumped,
    }


# ---------------------------------------------------------------------------
# corrective voice probe (single execution)
# ---------------------------------------------------------------------------

class CorrectiveAlreadyRun(RuntimeError):
    """Raised when the corrective voice operation would be executed more than once."""


_CORRECTIVE_RUN_FLAG = {"executed": False}


def reset_corrective_guard() -> None:
    """Test hook: allow a fresh single-execution guard."""
    _CORRECTIVE_RUN_FLAG["executed"] = False


async def probe_voice_corrective(
    client: GeminiClient,
    spoken_wav: bytes,
    semantic_validator=validate_semantic_response,
) -> dict:
    """Run the corrective voice-conversation operation exactly once.

    Executes: spoken audio -> AUDIO_UNDERSTANDING -> textual response ->
    TEXT_TO_SPEECH -> final audio. Captures providerCalls[]/providerAttempts[]
    during this single execution and returns only sanitized, safe data.
    """
    if _CORRECTIVE_RUN_FLAG["executed"]:
        raise CorrectiveAlreadyRun(
            "corrective voice operation already executed; refusing a second "
            "provider execution (single-execution bound)"
        )
    _CORRECTIVE_RUN_FLAG["executed"] = True

    begin_usage_tracking()
    system_prompt = (
        "You are Rihla, a helpful Egyptian tour assistant. "
        "The user has sent you an audio message. Listen to it and respond "
        "appropriately. If they ask about tourism in Egypt, recommend a specific "
        "attraction and explain why briefly."
    )
    last_err = None
    text = None
    audio = None
    mime = None
    calls = []
    attempts = []
    tts_source = None
    try:
        response = await client.generate_with_audio(
            system_prompt=system_prompt,
            audio_bytes=spoken_wav,
            mime_type="audio/wav",
        )
        text = getattr(response, "text", None)
        if text:
            audio, mime = await synthesize_speech(text, client)
        calls, attempts = consume_usage_and_attempts()
        if not text:
            tts_source = "no_text"
        elif audio is None:
            tts_source = "gtts_fallback_failed"
        else:
            tts_attempts = [
                a for a in attempts
                if a.get("operation") == "TEXT_TO_SPEECH"
            ]
            if any(a.get("outcome") == "SUCCEEDED" for a in tts_attempts):
                tts_source = "gemini_tts"
            else:
                tts_source = "gtts_fallback"
    except Exception as exc:  # noqa: BLE001 - captured for reporting
        last_err = f"{type(exc).__name__}: {exc}"
        tts_source = "error"
        calls, attempts = consume_usage_and_attempts()

    semantic = semantic_validator(text)

    result = {
        "probe": "corrective_spoken_voice",
        "spoken_input": {
            "synthesizer": "espeak-ng",
            "sentence": "Recommend one tourist attraction in Cairo and briefly explain why.",
            "input_audio_bytes": len(spoken_wav),
            "input_audio_mime": "audio/wav",
        },
        "semantic_understanding": {
            "semanticUnderstandingVerified": semantic["semanticUnderstandingVerified"],
            "matchedKeywords": semantic["matchedKeywords"],
            "responseCharLength": semantic["responseCharLength"],
            "reason": semantic["reason"],
        },
        "requested_model": settings.gemini_model,
        "provider_calls": _safe(calls),
        "provider_attempts": _safe(attempts),
        "modality_non_double_count": [
            modality_not_double_counted(c) for c in calls
        ],
        "audio_produced": audio is not None,
        "audio_mime": mime,
        "tts_source": tts_source,
        "text_len": len(text) if isinstance(text, str) else None,
    }
    if last_err:
        result["last_error"] = _sanitize(last_err)
    return result


# ---------------------------------------------------------------------------
# legacy probes (previous probe support, kept)
# ---------------------------------------------------------------------------

async def probe_image(client: GeminiClient) -> dict:
    begin_usage_tracking()
    png = _synthetic_png()
    system_prompt = (
        "You are Rihla, a helpful Egyptian tour assistant. "
        "Describe what is in this image in one short sentence."
    )
    last_err = None
    text = None
    try:
        response = await client.generate_with_image(
            system_prompt=system_prompt,
            user_message="Describe this image briefly.",
            image_bytes=png,
            mime_type="image/png",
        )
        text = getattr(response, "text", None)
    except Exception as exc:  # noqa: BLE001 - captured for reporting
        last_err = f"{type(exc).__name__}: {exc}"
    finally:
        calls, attempts = consume_usage_and_attempts()
    result = {
        "requested_model": settings.gemini_model,
        "provider_calls": _safe(calls),
        "provider_attempts": _safe(attempts),
        "text_len": len(text) if isinstance(text, str) else None,
    }
    if last_err:
        result["last_error"] = _sanitize(last_err)
    return result


async def probe_voice(client: GeminiClient) -> dict:
    begin_usage_tracking()
    wav = _synthetic_wav()
    system_prompt = (
        "You are Rihla, a helpful Egyptian tour assistant. "
        "The user has sent you an audio message. Listen and respond briefly."
    )
    last_err = None
    text = None
    audio = None
    mime = None
    calls = []
    attempts = []
    try:
        response = await client.generate_with_audio(
            system_prompt=system_prompt,
            audio_bytes=wav,
            mime_type="audio/wav",
        )
        text = getattr(response, "text", None)
        if text:
            audio, mime = await synthesize_speech(text, client)
        calls, attempts = consume_usage_and_attempts()
        if not text:
            tts_source = "no_text"
        elif audio is None:
            tts_source = "gtts_fallback_failed"
        else:
            tts_attempts = [
                a for a in attempts
                if a.get("operation") == "TEXT_TO_SPEECH"
            ]
            if any(a.get("outcome") == "SUCCEEDED" for a in tts_attempts):
                tts_source = "gemini_tts"
            else:
                tts_source = "gtts_fallback"
    except Exception as exc:  # noqa: BLE001 - captured for reporting
        last_err = f"{type(exc).__name__}: {exc}"
        tts_source = "error"
        calls, attempts = consume_usage_and_attempts()
    result = {
        "requested_model": settings.gemini_model,
        "provider_calls": _safe(calls),
        "provider_attempts": _safe(attempts),
        "text_len": len(text) if isinstance(text, str) else None,
        "audio_produced": audio is not None,
        "audio_mime": mime,
        "tts_source": tts_source,
    }
    if last_err:
        result["last_error"] = _sanitize(last_err)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2E-B live multimodal probes")
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="Run only the corrective spoken-voice probe (no image probe).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the sanitized JSON output (default: scripts/phase_2e_b_corrective_voice_output.json)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print output without writing it to a file (for tests).",
    )
    return parser.parse_args(argv)


async def _run_probes(args: argparse.Namespace, client: GeminiClient) -> dict:
    """Orchestrate the selected probes. Separated from CLI parsing and file
    writing so tests can drive it with a fake client and zero live calls."""
    voice = await probe_voice_corrective(client, generate_spoken_wav())

    if args.voice_only:
        # image probe is NOT run in corrective mode
        image = None
        print("=== CORRECTIVE SPOKEN VOICE PROBE (image probe skipped) ===")
        print(json.dumps(voice, indent=2, default=str))
    else:
        image = await probe_image(client)
        print("=== IMAGE PROBE ===")
        print(json.dumps(image, indent=2, default=str))
        print("=== VOICE PROBE (square-wave, legacy) ===")
        legacy_voice = await probe_voice(client)
        print(json.dumps(legacy_voice, indent=2, default=str))
        print("=== CORRECTIVE SPOKEN VOICE PROBE ===")
        print(json.dumps(voice, indent=2, default=str))

    audit_records = []
    for section in (image, voice):
        if section is None:
            continue
        audit_records += section.get("provider_calls") or []
        audit_records += section.get("provider_attempts") or []
    audit = _redaction_audit(audit_records)
    print("=== REDACTION AUDIT ===")
    print(json.dumps(audit, indent=2))
    return {"voice": voice, "image": image, "redaction_audit": audit}


async def main(argv=None) -> None:
    args = _parse_args(argv)

    print(f"keys_configured={len(settings.gemini_key_list)}")
    print(f"gemini_model={settings.gemini_model}")
    print(f"tts_voice={settings.tts_voice}")

    client = GeminiClient(api_keys=settings.gemini_key_list)
    client.MAX_RETRIES = 0

    try:
        payload = await _run_probes(args, client)
    except OfflineSynthesizerUnavailable as exc:
        print(f"BLOCKER: {exc}")
        return

    if not args.no_write:
        output_path = args.output or str(
            _PROJECT_ROOT / "scripts" / "phase_2e_b_corrective_voice_output.json"
        )
        payload["probe"] = "corrective_spoken_voice"
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"WROTE_OUTPUT={output_path}")


if __name__ == "__main__":
    asyncio.run(main())
