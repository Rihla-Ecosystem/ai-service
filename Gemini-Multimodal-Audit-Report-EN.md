# Technical Audit Report: Gemini Multimodal Integration — Pre-Pricing Verification

**Scope:** Image Analysis (via `POST /identify`), Audio Understanding, and Text-to-Speech (via `POST /voice`)
**Status:** Verification only — no production code was modified.
**Files inspected:** `app/api/identify.py`, `app/api/voice.py`, `app/core/usage.py`, `app/core/gemini_usage.py`, `app/core/llm_client.py`, `requirements.txt`, `pyproject.toml`, `uv.lock`, `tests/test_gemini_usage.py`, `tests/test_usage_contract.py`, `tests/test_llm_usage.py`, `tests/test_stream_usage.py`, `tests/test_tools.py`, `tests/test_guardrails.py`.
**Files not found that could materially affect conclusions:** No `tests/test_identify.py` or `tests/test_voice.py` (endpoint-level tests), and no fixture based on a captured real Gemini response. This does not stop the audit, but it limits confidence in some conclusions (flagged below as `UNKNOWN_REQUIRES_LIVE_PROBE`).

---

## 1. Executive Summary

**What is confirmed:**
- The SDK in use is `google-genai==2.14.0` (the official Google Gen AI SDK — not the legacy `google-generativeai`, not the Vertex AI SDK), and all image, audio, and TTS operations go through the same underlying `client.models.generate_content` (non-streaming).
- The actual `usageMetadata` schema confirms that **modality breakdown fields are part of the aggregate total (breakdown-only), not additive** — no double counting is currently occurring.
- A **real, SDK-source-confirmed bug** was found: `extract_response_model()` reads `response.model`, and this field **does not exist at all** on a real `GenerateContentResponse` object (the correct field is `model_version`). Result: **`actualModel` is never populated in production** for any operation (image, audio, TTS, text), despite the current tests passing — the tests use fake stub objects with a `.model` attribute that does not match the real SDK schema.
- **Cache Hit on `/identify` does not return `providerCalls: []`** as documented in the spec; it returns `providerCalls: null` (field absent entirely), due to code ordering.
- **Failed provider calls are never recorded** in `providerCalls[]` — if all retries are exhausted, there is zero trace of that real, network-sent provider request.
- **gTTS is correctly classified** today: it never appears as `provider=google` in `providerCalls[]`.
- When Gemini TTS succeeds after Audio Understanding, **two calls (call-1, call-2) are correctly produced** — matching the example given in the spec.

**What remains uncertain:**
- Does Gemini actually return complete `usage_metadata` on every response for Vision, Audio Understanding, and TTS (especially the preview TTS model)? — requires a Live Probe.
- Could a failed provider request (network/5xx) still be billed by Google despite returning no usable result? — needs official confirmation/support.
- The final-streaming-usage path has never been exercised on any real Vision or Audio call — the paths actually used in `/identify` and `/voice` are non-streaming.

**What blocks pricing right now:**
1. Fix `extract_response_model` (critical bug) before relying on `actualModel` for any monitoring or Rate Card.
2. A Live Probe to confirm the actual shape of `usageMetadata` for Vision, Audio, and TTS.
3. Decide the semantics of failed calls (should they be recorded as `providerCallMade=true, status=FAILED`, or remain unrecorded?).

**Is the current `providerCalls[]` contract sufficient?**
Yes, structurally (schema-wise) — the existing fields (`inputTokens`, `outputTokens`, `totalTokens`, breakdown fields) are sufficient to represent what Gemini actually returns, and no ground-up redesign is warranted. The problem is not the contract; it is the **implementation**: the `actualModel` bug, the missing recording of failed calls, and the Cache Hit behavior not matching the spec.

---

## 2. Current Architecture

```
Gemini Native Response (GenerateContentResponse)
        │  (usage_metadata, model_version, response_id, candidates)
        ▼
AI Service Normalization
        │  app/core/gemini_usage.py  → extract_token_counts() / extract_response_model()
        │  app/core/llm_client.py    → _record_provider_call() / _record_stream_final()
        ▼
providerCalls[]  (app/core/usage.py: make_provider_call + UsageScope contextvar)
        │
        ├─► IdentifyResponse.providerCalls   (app/api/identify.py)
        └─► VoiceResponse.providerCalls      (app/api/voice.py)
        ▼
Core Server
        ├─ AI usage recording
        ├─ Shadow pricing
        ├─ Pricing coverage metrics
        └─ UNPRICED reason tracking
```

The third layer (`providerCalls[]` as received by the Core Server) is already **known and tested** — this audit focuses on the first two layers (provider-native response and normalization), which had not been fully verified before.

---

## 3. Scope Confirmation

- The current image feature is **Image Analysis / Landmark Identification only**, via `POST /identify` (line 101 in `identify.py`: `llm_client.generate_with_image(...)`). No call anywhere in the codebase uses `response_modalities=["IMAGE"]` (confirmed via `grep`), so **Image Generation is not implemented at all** — `NOT_CURRENTLY_IMPLEMENTED`.
- The voice flow (`/voice`) may produce more than one provider call (Audio Understanding then TTS) — **confirmed by code**.
- gTTS is **not invented** as a paid Gemini call — **confirmed by code** (see Section 9).

---

## 4. Project Code Audit

### 4.1 `app/core/llm_client.py` — the main driver of Gemini calls

| Item | Value |
|---|---|
| Package | `google-genai` (`from google import genai` + `from google.genai import types as genai_types`) |
| Mode | Primarily non-streaming for Vision/Audio/TTS; streaming is optional for `generate()` only, via `stream=True` |
| API method | `client.models.generate_content(model=..., contents=..., config=...)` for all of: `generate` (text), `generate_with_tools`, `generate_with_image` (Vision), `generate_with_audio`, `generate_speech` (TTS) — **all the same underlying call**, differing only in `contents` and `config` (e.g. `response_modalities=["AUDIO"]` for TTS) |
| Model selection | `_model_for_retry()` (lines 97-102) uses `settings.gemini_model`, then a fixed fallback list `GEMINI_MODEL_FALLBACKS` based on retry count — **except TTS**, which uses a fixed model literal `"gemini-3.1-flash-tts-preview"` directly (line 420), with no model escalation |
| `requestedModel` | Explicitly passed as the chosen `model` before the call (the same value sent to the provider) — `CONFIRMED_BY_PROJECT_CODE` |
| `actualModel` | Produced by `extract_response_model(response)`, which reads `getattr(response, "model", None)` — **incorrect**; see Sections 6/7 |
| `providerCallId` | Generated deterministically and sequentially (`call-1`, `call-2`, ...) inside `UsageScope.add()` in `app/core/usage.py` lines 141-146, **not** any identifier returned by the provider (the SDK does not expose a usable request id on this path — explicit comment in the code, lines 124-125) |
| Recording a call | Only on success, inside the `try` block right after `generate_content` (e.g. line 339 for Vision, line 390 for Audio, line 441 for TTS) |
| On failure | `except Exception` logs a warning, puts the key in cooldown (`key.mark_failed`), then **recursively retries** (`return await self.generate_with_image(...)` etc.) with no call to `record_provider_call` — **there is no trace whatsoever of a failed attempt** |
| Retry cap | `MAX_RETRIES = 10` for each of `generate/generate_with_tools/generate_with_image/generate_with_audio`, while `generate_speech` (TTS) is capped at 2 (`_retry_count > 2`) |
| TTS 503 handling | If `getattr(e, "code", None) == 503`, the key is **not** put into cooldown (line 459) — behavior specific to TTS only |

### 4.2 `app/core/gemini_usage.py` — extraction from the response

- `extract_response_model` (lines 19-24): reads `response.model` only. **This field does not exist on a real `google.genai.types.GenerateContentResponse`** (SDK Source confirmation below). Practical result: always returns `None` on a real response.
- `extract_token_counts` (lines 27-90): reads `response.usage_metadata`, then the fields `prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`, as well as `prompt_tokens_details` / `candidates_tokens_details` (lists of `ModalityTokenCount` with `modality` and `token_count` fields). **All these field names exactly match** the real field names in `google-genai==2.14.0` (verified directly against the installed package's source — see the next section).
- `IMAGE`/`AUDIO` breakdown entries are classified as **additional, separate fields only** (`imageInputTokens`, `audioInputTokens`, ...) and are never added into `inputTokens`/`outputTokens` — correct behavior, matching the SDK schema.

### 4.3 `app/core/usage.py` — the intermediate contract

- `UsageScope` is a `ContextVar` initialized once via `begin_usage_tracking()` at the start of each request (`identify.py` line 100, `voice.py` line 177), and consumed once via `consume_usage()` (returns and resets). This correctly guarantees **isolation between concurrent requests** through asyncio's `contextvars` mechanism.
- `derive_legacy_usage`: sums `inputTokens`/`outputTokens`/`totalTokens` across recorded calls only, **never touches** breakdown fields (`imageInputTokens`, etc.) — no path produces double counting.
- `final_stream_usage`: picks the last **non-empty snapshot** (not the sum) — correct, since Gemini's streaming `usageMetadata` is cumulative, as confirmed by official docs below.

### 4.4 `app/api/identify.py` — Cache Hit/Miss

- `_cache` is an in-memory `dict`, keyed on `md5(image_bytes) + lat + lon`.
- **Cache Miss** (lines 99-141): `begin_usage_tracking()` → `generate_with_image()` → `consume_usage()` → `derive_legacy_usage()` → a `payload` is built by adding `usage`/`model`/`providerCalls`, **then** `result` (without these fields) is stored in `_cache` — storage happens at line 129, **before** those fields are added at lines 133-140. This ordering is the root cause of the Cache Hit issue below.
- **Cache Hit** (lines 46-49): `cached["cached"] = True`, then `IdentifyResponse(**cached)` is built directly. Since `cached` (the stored copy of `result`) **does not contain** the keys `usage`/`model`/`providerCalls`, Pydantic falls back to the optional fields' defaults: `usage: Optional[dict] = None`, `model: Optional[str] = None`, `providerCalls: Optional[list] = None`.
- **Conclusion:** no fake provider call is invented on a Cache Hit (no fabricated `providerCallMade=true`) — this part is **correct and safe for pricing**. However, the actual shape is `providerCalls: null`, not `providerCalls: []` as documented in the spec — a **literal contract mismatch** that could confuse API consumers who expect an always-present, type-safe empty array instead of `null`.

### 4.5 `app/api/voice.py` — Audio + TTS

- `begin_usage_tracking()` is called once (line 177) before `generate_with_audio()`, then (if text was produced) `synthesize_speech()`, which internally calls `llm_client.generate_speech()` (real Gemini TTS).
- `synthesize_speech()` (lines 96-113): if Gemini TTS succeeds, the bytes are returned directly (with PCM→WAV conversion when needed). If `generate_speech` **raises** (i.e., all three retry attempts were exhausted) → `except Exception` (line 111) logs a warning only and falls through to `return gtts_audio_bytes(text)` (local fallback).
- **There is no way in `providerCalls[]` to distinguish** between: (a) TTS was never attempted because `text` was empty, and (b) Gemini TTS was attempted and failed after consuming real network resources, followed by a gTTS fallback. Both cases produce the exact same final trace: a single `AUDIO_UNDERSTANDING` entry in `providerCalls[]`, with no signal that a TTS attempt occurred.
- `consume_usage()` is called **once** at the end of `voice_endpoint` (line 202), after both Audio Understanding and (an attempted) TTS have completed — so if both succeed, a **two-element array** is correctly returned.

---

## 5. Official Research

Official Google sources (`ai.google.dev`, `docs.cloud.google.com`) were consulted, along with the actual installed source code of `google-genai==2.14.0` (installed from PyPI and verified directly against the Pydantic model definitions — this is classified as `CONFIRMED_BY_SDK_SOURCE`, the strongest confidence level available short of a live probe against a real Google server).

| Source | Access date |
|---|---|
| `https://ai.google.dev/api/tokens` (Counting tokens) | 2026-08-04 |
| `https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/ModalityTokenCount` | 2026-08-04 |
| `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse` | 2026-08-04 |
| `https://ai.google.dev/gemini-api/docs/live-api/capabilities` | 2026-08-04 |
| `https://ai.google.dev/gemini-api/docs/pricing` (Gemini Developer API pricing) | 2026-08-04 |
| Cloud TTS pricing: `https://cloud.google.com/text-to-speech/pricing` | 2026-08-04 |
| Secondary community sources (context only, for tokens/second estimates and example real fields): OpenRouter model card, third-party pricing comparison sites (`benchlm.ai`, `geotoolbox.ai`, etc.) — **not used as official evidence, contextual estimation only (`COMMUNITY_REPORTED`)** |

---

## 6. Image Analysis Findings

| # | Question | Answer | Confidence | Classification |
|---|---|---|---|---|
| 1 | Native response type | `google.genai.types.GenerateContentResponse` (via non-streaming `generate_content`) | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 2 | Is `usageMetadata` returned? | The field exists in the schema (`usage_metadata: Optional[...]`) but is **fully optional** — actual runtime presence for a specific Vision call is not established by any real fixture | MEDIUM | `CONFIRMED_BY_SDK_SOURCE` (schema) / `UNKNOWN_REQUIRES_LIVE_PROBE` (actual population) |
| 3 | Is `modelVersion` returned? | Yes, `model_version: Optional[str]` genuinely exists on `GenerateContentResponse` — but **the project never reads it** (it reads the non-existent `model` instead) | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 4 | Is `responseId` returned? | Yes, `response_id: Optional[str]` exists in the schema, **not currently used in the project** | HIGH (schema) | `CONFIRMED_BY_SDK_SOURCE` |
| 5 | Equivalent actual-model field? | `model_version` is the correct equivalent field | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 6 | Is the returned model guaranteed to be the model actually used? | Yes, by intent (`model_version` is documented as "Output only. The model version used to generate the response"), though not live-verified here | MEDIUM | `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` (field documentation) |
| 7 | Are images converted into tokens? | Yes, all modalities are converted into tokens | HIGH | `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` |
| 8 | Are image tokens included in `promptTokenCount`? | Yes — officially documented and confirmed with a concrete example (615 = 99 text + 516 image) | HIGH | `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` |
| 9-11 | Is a modality breakdown returned, and what fields? | Yes: `promptTokensDetails[]` (type `ModalityTokenCount`, fields `modality`, `tokenCount`), possible values include `TEXT/IMAGE/AUDIO/VIDEO/DOCUMENT` | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 12-13 | Are modality tokens part of the aggregate or additive? | **Part of the aggregate (breakdown-only)** — confirmed by the docstring of `prompt_token_count` itself in the SDK: "When cached_content is set, this also includes..." and by the official example above | HIGH | `CONFIRMED_BY_SDK_SOURCE` + `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` |
| 14 | Would adding image breakdown details to `promptTokenCount` cause double counting? | **Yes, if done manually** — but the current code **does not do this** (see 4.2/4.3) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 15 | Does token calculation depend on resolution/image count/model family...? | Yes, per moderately reliable secondary sources (e.g., fixed resolution tiers determine a fixed token count per image), but exact figures vary by model and are not officially confirmed for every model used here | LOW-MEDIUM | `COMMUNITY_REPORTED` |
| 16-17 | Is cached content usage separate? Included in the total? | `cachedContentTokenCount` is a separate field, but it is **already included** within `promptTokenCount` (per the field's own official docstring) — i.e., it is a breakdown, not additive, just like the modality breakdown | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 18-21 | Internal representation of Cache Hit/Miss | **A confirmed problem**: Miss correctly returns a `providerCalls` array with one entry, but Hit returns `providerCalls: null` (not `[]`) due to storage ordering in `identify.py` — no fabricated call is invented (positive), but the shape does not match the spec (negative) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 22-23 | Can image usage be fully priced with current fields? Any additional monitoring fields needed? | The aggregate (`inputTokens`/`outputTokens`) can be priced once actual population is confirmed. `actualModel` (post-fix) and `responseId` are useful as optional monitoring fields only, without changing billing totals | MEDIUM | `INFERRED` |

**Required decisive answer (A/B/C):** **(A)** — modality tokens (image) are a breakdown already included in the aggregate total, not additive. Confirmed with matching SDK-source and official-documentation evidence.

---

## 7. Audio Understanding Findings

| # | Question | Answer | Confidence | Classification |
|---|---|---|---|---|
| 1 | Native response type | Same `GenerateContentResponse` (no separate Audio Understanding API — it goes through `generate_content` with an audio `inline_data` part) | HIGH | `CONFIRMED_BY_PROJECT_CODE` + `CONFIRMED_BY_SDK_SOURCE` |
| 2-3 | `usageMetadata`/`modelVersion`? | Exactly the same schema as the section above (same response type) | HIGH (schema) / UNKNOWN (live population) | as above |
| 4-6 | Is audio represented as input tokens? What unit? | Yes, audio is converted into tokens (not a direct time value in the response). Moderately reliable pricing sources estimate **~32 tokens/second** as the input conversion rate, but this is a **third-party commercial figure**, not a field in the response itself — the response only returns `promptTokenCount`, no explicit audio duration | LOW (exact rate) / MEDIUM (that the unit is "tokens" at all) | `COMMUNITY_REPORTED` (the rate) |
| 7 | Is the billing quantity exposed directly? | Yes, via `promptTokenCount` itself — no need for a separate calculation from file duration | MEDIUM | `INFERRED` from the official pricing structure (per-token) |
| 8-10 | Audio modality breakdown, and is it added to prompt? | Yes, via `promptTokensDetails` with `modality=AUDIO` — **part of the aggregate, not additive** (same logic as image) | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 11 | Difference between uploaded file / inline / File API / streaming? | The project uses **inline bytes only** (`genai_types.Blob(mime_type=..., data=audio_bytes)`) — no File API path, no audio streaming currently | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 12-14 | Output text tokens / thought tokens / cached tokens? | All are supported in the schema (`candidates_token_count`, `thoughts_token_count`, `cached_content_token_count`) and are actually extracted in `extract_token_counts` when present | HIGH | `CONFIRMED_BY_SDK_SOURCE` + `CONFIRMED_BY_PROJECT_CODE` |
| 15-17 | Is usage only present in the final streaming chunk? | Yes — the code handles this correctly via `final_stream_usage` (only the last non-empty snapshot, never a sum) — but **this path is not actually used** for either image or audio in `/identify` or `/voice`, only for the generic `generate()` function when `stream=True` | HIGH (for the logic) / N/A (not used here) | `CONFIRMED_BY_PROJECT_CODE` |
| 18-19 | Can audio succeed with no usage? How is completeness classified? | Yes, theoretically possible (the field is Optional) — the code handles this correctly: when no counter is present, `usageCompleteness = UNAVAILABLE`, not a fabricated zero | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 20 | Can the current contract represent all confirmed audio fields? | Yes — `TOKEN_FIELD_NAMES` in `usage.py` already includes `audioInputTokens`, `audioOutputTokens`, `audioInputSeconds`, `audioOutputSeconds`, `transcriptionSeconds` as ready-made fields (even though not all are currently populated from `gemini_usage.py`) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |

**Important note:** `audioInputSeconds`/`audioOutputSeconds`/`transcriptionSeconds` are **defined in the contract** (`usage.py`) but **no code in `gemini_usage.py` populates them** — because Gemini does not return an explicit duration in `usageMetadata`, only tokens. This means these fields will always stay absent for the current Gemini path; this is **expected, not a bug** (the fields exist to support other providers in the future, per the file's own comment).

---

## 8. Gemini Text-to-Speech Findings

| # | Question | Answer | Confidence | Classification |
|---|---|---|---|---|
| 1-3 | Response type / usageMetadata / modelVersion | Exactly the same `GenerateContentResponse` (TTS is called via the same `generate_content` with `response_modalities=["AUDIO"]`) — same schema, same `actualModel`-reading bug | HIGH (response type) | `CONFIRMED_BY_PROJECT_CODE` + `CONFIRMED_BY_SDK_SOURCE` |
| 4-9 | Input/output tokens, character count, duration, sample/byte count? | Only `promptTokenCount`/`candidatesTokenCount` via the same `usageMetadata` — **there is no explicit character-count or duration field in the response**; the generated audio is returned as raw PCM bytes via `inline_data`, with no metadata about its duration | MEDIUM (general structure) | `CONFIRMED_BY_SDK_SOURCE` (absence of duration/character fields) |
| 10-11 | Official billing unit | **Tokens**: text input priced per million text tokens, audio output priced per million "audio tokens" (commercially calculated as 25 tokens/second) — confirmed by Google's official pricing page (`ai.google.dev/gemini-api/docs/pricing`) and corroborating third-party pricing pages | HIGH (unit = tokens) / MEDIUM (the exact 25 tokens/sec rate, from a secondary source) | `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` (partially) + `COMMUNITY_REPORTED` (exact rate) |
| 12 | Is the billing unit directly present in the runtime response? | Yes structurally (`candidates_token_count` = output audio tokens), assuming it is actually populated by the server for this specific model | MEDIUM | `INFERRED` |
| 13 | Does the installed SDK expose the required fields? | Yes, same `GenerateContentResponseUsageMetadata` (confirmed directly from the installed package source) | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| 14 | Are fields always populated or conditional? | **Always conditional** (every field is `Optional`) — no real population guarantee without a live test, especially since the model used (`gemini-3.1-flash-tts-preview`) is a **Preview** model, and the official pricing page explicitly warns that preview models "may change before becoming stable and have more restrictive rate limits" | MEDIUM | `CONFIRMED_BY_OFFICIAL_DOCUMENTATION` (preview warning) |
| 15 | Differences between Preview/Stable, single/multi-speaker, streaming? | The project exclusively uses **Preview + single-speaker + non-streaming** (no `voice2` in `SpeechConfig`, no streaming TTS API) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 16 | Double counting risk when adding audio-output modality details to the total? | No risk — the code does not add breakdown output into `outputTokens` (same logic as image/audio) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 17-20 | TTS failure: what happens? Is any usage recorded on a failed call? | **No `providerCalls` entry is created for the failed call at all** — `_record_provider_call` is only invoked after `generate_content` succeeds (see 4.1). On failure after all retries are exhausted, a `RuntimeError` is raised and caught in `voice.py` (falling back to gTTS) with zero accounting trace | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| 21 | Can Gemini TTS be priced reliably right now? | **Not yet** — the contract is structurally able to hold the fields, but (a) there is no live evidence of actual population for the Preview model, and (b) failed calls (which may in fact still be billed by Google) are entirely invisible | — | `BLOCKED_PENDING_LIVE_PROBE` |
| 22 | Should Gemini TTS stay UNPRICED until a live probe? | **Yes, recommended** for the official Rate Card entry, while still accumulating within Shadow Pricing whenever real data is actually available | — | Recommendation |

---

## 9. gTTS and Non-Gemini Speech Generation Findings

- `gtts_audio_bytes()` (`voice.py` lines 83-93) calls the `gTTS` package (version `gtts==2.5.4` per `uv.lock`), which in turn **makes a real HTTP request** to an undocumented public Google Translate endpoint (a well-known behavior of this open-source package — it is **not** part of the official Google Cloud AI or Gemini API, and is not billed through the project's Google Cloud/AI Studio account). This conclusion is based on general knowledge of how the gTTS package works, not on an official Google document — since the dependency itself is unofficial to begin with — `COMMUNITY_REPORTED`.
- **There is no call to `record_provider_call` or `_record_provider_call` anywhere inside `gtts_audio_bytes()` or its call path** — verified directly against the code. — `CONFIRMED_BY_PROJECT_CODE`.
- **Therefore:** the gTTS path **never** appears as `provider=google, operation=TEXT_TO_SPEECH, providerCallMade=true` — the current contract is **correct and compliant** with the strict rule required for this audit.
- **Recommended classification:** `NOT_A_PAID_GEMINI_CALL` — it could optionally be tracked in the future as "local/non-priced operation" or "infrastructure usage" entirely separate from `providerCalls[]`, but this is **not required now** (`OPTIONAL_OBSERVABILITY_FIELD`) since it has no impact on billing.
- **Additional operational risk (non-accounting):** relying on an unofficial gTTS endpoint carries a risk of sudden breakage/change outside the project's control (ToS risk) — listed in the risk register (Section 16) as an operational, not accounting, risk.

---

## 10. Partial Failure Matrix

| Scenario | `providerCalls[]` count | Details | Overall pricing status |
|---|---|---|---|
| **A**: Audio succeeds + Gemini TTS succeeds | **2** | call-1: `AUDIO_UNDERSTANDING`, `providerCallMade=true`, usage per availability. call-2: `TEXT_TO_SPEECH`, `provider=google`, `providerCallMade=true` | `FULLY_PRICED` if usage is complete for both, otherwise `PARTIALLY_PRICED` |
| **B**: Audio succeeds + TTS fails **before** sending a real request (e.g., empty `text`) | **1** | Only `AUDIO_UNDERSTANDING`. No TTS call at all (`synthesize_speech` returns `None` immediately for empty text, lines 97-98) | `PARTIALLY_PRICED` (audio priced, TTS = `SKIPPED` with no trace) |
| **C**: Audio succeeds + TTS request **was actually sent** but fails (after all retries) | **1** only (same audio entry) | **No trace whatsoever** of the TTS attempt despite a real network request being sent — this is the **most serious gap** found in this audit: a real, potentially billable call is **entirely invisible** to the Core Server | `PARTIALLY_PRICED` on the surface, but **may hide UNPRICED real cost** |
| **D**: Audio succeeds + gTTS fallback succeeds (Gemini TTS was never attempted at all — a theoretical scenario, since the current code always tries Gemini first) | **1** | gTTS never appears in `providerCalls[]` (correct) | `PARTIALLY_PRICED` (TTS = `NOT_A_PAID_GEMINI_CALL`) |
| **E**: Audio succeeds + Gemini TTS fails + gTTS fallback succeeds **(the actual most common real-world path when TTS fails)** | **1** | Identical to Scenario C from the `providerCalls[]` perspective — **no visible difference between "TTS failed then gTTS succeeded" and "TTS was never attempted"** | Same risks as Scenario C |
| **F**: Audio fails, TTS is never attempted | **0** (since `_record_provider_call` is only called after `generate_content` succeeds; a failure in `generate_with_audio` after all retries raises an exception that reaches `except Exception` in `voice_endpoint`, returning HTTP 500 with no `providerCalls` at all) | No `VoiceResponse` body is produced at all (the endpoint raises `HTTPException(500)`) | `ZERO_CALL` from the perspective of a successful response, but **there may be real network attempts that go unrecorded** (same gap as C) |
| **G**: Both calls succeed but one returns incomplete usage | **2** | The incomplete entry is automatically classified `usageCompleteness=UNAVAILABLE` (correct, confirmed behavior — no fabricated zeros) | `PARTIALLY_PRICED` |

**Explicit rule derived from the evidence:** the current contract has **no mechanism whatsoever to distinguish** "no attempt was made" from "an attempt was made and failed after consuming real network resources," for any of `generate/generate_with_image/generate_with_audio/generate_speech`. This is a general design gap in `llm_client.py` (not specific to audio) — `REQUIRED_NOW`, with relatively high severity for accurate billing.

---

## 11. Assessment of the Current `providerCalls[]` Contract

| Field | Source | Current reliability |
|---|---|---|
| `provider` | Hardcoded `"google"` via `PROVIDER_GOOGLE` | Always reliable (no other source is actually used) |
| `providerCallId` | Generated internally (sequential) | Reliable for ordering purposes within a request, **not** a real provider identifier |
| `providerCallMade` | Always `True` when recorded | **Not reliable as a failure signal** — there is never a `providerCallMade=False` or "sent but failed" state anywhere in the current code (the parameter's own default is `True`, and no call site ever passes `False`) |
| `requestedModel` | Directly from project code (the same value sent) | Fully reliable |
| `actualModel` | **Broken** (reads a non-existent field) | **Never reliable — always absent** |
| `usageSource` / `usageCompleteness` | Logically derived from the presence/absence of counters | Reliable and consistent with the required rules (no fabricated zeros) |
| `inputTokens`/`outputTokens`/`totalTokens` | From real `usage_metadata` when present | Reliable **provided** Google actually populates them (needs a Live Probe) |
| Breakdown fields (`imageInputTokens`, etc.) | From `prompt_tokens_details`/`candidates_tokens_details` | Structurally reliable, breakdown-only confirmed |

- `totalTokens` is **never derived** from `inputTokens + outputTokens` (confirmed by code and by the `test_never_derives_total_from_sum` test) — and this is **genuinely correct**, because the real `total_token_count` = `prompt + candidates + tool_use + thoughts` (confirmed from SDK source), not just the first two fields.
- The `billingQuantities[]` structure proposed in the spec (a more general future alternative): **not needed now** — current evidence (Gemini bills entirely on a unified Token unit across all modalities used here) does not justify redesigning the contract. It is classified `FUTURE_COMPATIBILITY` only, in case a future provider bills in a non-token unit (seconds/characters), which is a real possibility but **not required for this phase**.

**Final classification of possible changes:**

| Proposed change | Classification |
|---|---|
| Fix `extract_response_model` to read `model_version` instead of `model` | `REQUIRED_NOW` |
| Make Cache Hit return `providerCalls: []` instead of `null` | `REQUIRED_NOW` |
| Add a `providerCallMade=False` / `status=FAILED` state for calls that were sent but failed | `REQUIRED_NOW` (before fully relying on `providerCalls[]` for Shadow Pricing of TTS/unstable-network operations) |
| Add `responseId` as an optional monitoring field | `OPTIONAL_OBSERVABILITY_FIELD` |
| A general `billingQuantities[]` structure | `FUTURE_COMPATIBILITY` |
| A ground-up contract redesign | `NOT_NEEDED` |
| Confirm TTS/Audio billing units via live data | `BLOCKED_PENDING_LIVE_PROBE` |

---

## 12. Usage Interpretation Rules

The following rules are supported only by the evidence gathered above:

1. Modality breakdown details (`imageInputTokens`, `audioInputTokens`, ...) are **breakdown-only** and must never be added into `inputTokens`/`outputTokens` — confirmed.
2. Missing `usage` **does not mean** zero usage — the current contract genuinely honors this (`UNAVAILABLE` rather than a fabricated zero).
3. `UNPRICED` does not mean free — this distinction must remain explicit in the Core Server (outside the scope of this AI Service code).
4. A Cache Hit with no provider call should produce `providerCalls: []` — **currently required as a fix**, not yet true today.
5. Local fallback (gTTS) **must not** be represented as a paid Gemini call — **currently true**.
6. `actualModel` **must not** be invented from `requestedModel` — **currently true in `usage.py`** (no code does this), but **because of the bug**, `actualModel` ends up permanently absent instead of being filled with a real value — a distinction between "not fabricating" and "lacking the correct tool to read the real value."
7. **New rule that must be added:** a provider call that was **actually sent** (even if it later fails) should leave a traceable accounting record (even if not priced), because the real potential cost to the provider has nothing to do with whether the application-level response succeeded — this rule is **not currently honored** and requires an explicit engineering decision.

---

## 13. Contract Test Specification

> Note: all tests below use fake provider responses only, following the same style already used in `tests/test_gemini_usage.py` and `tests/test_llm_usage.py` (simple duck-typed objects). **No real Gemini requests.** The helper stub objects (`_Resp`, `_Meta`, `_Detail`) already exist in `tests/test_gemini_usage.py`; anything marked `(proposed helper)` below does not currently exist and is only a suggestion.

### Image Analysis

| Test name | Fixture | Fake response | Expected `providerCalls` | `usageCompleteness` | `providerCallMade` | Double-counting assertion |
|---|---|---|---|---|---|---|
| `test_identify_cache_hit_returns_empty_provider_calls` *(proposed — endpoint-level)* | An image already cached in `_cache` | No network call | **Should be `[]`** (currently `null` — this test will fail on today's code, which is exactly the point) | N/A | N/A | N/A |
| `test_identify_cache_miss_records_one_call` *(proposed)* | A new image | `_Resp(model_version="gemini-3.6-flash", meta=_full_meta())` | One entry, `operation=IMAGE_ANALYSIS` | `COMPLETE` | `True` | Confirm `imageInputTokens != inputTokens` |
| `test_missing_model_version_leaves_actual_model_absent` | — | `_Resp(meta=_full_meta())` without `model_version` | `actualModel` absent from the entry (not an explicit `null`) | per token availability | `True` | — |
| `test_missing_usage_metadata_marks_unavailable` | — | `_Resp(model_version="m")` without `usage_metadata` | `usageCompleteness=UNAVAILABLE`, no token fields | `UNAVAILABLE` | `True` | — |
| `test_image_modality_breakdown_surfaced_separately` | — | `_full_meta()` | `imageInputTokens=10`, `inputTokens=1024`, `imageInputTokens != inputTokens` | `COMPLETE` | `True` | Explicit assertion: `imageInputTokens + audioInputTokens != inputTokens` |
| `test_provider_error_records_nothing_currently` *(proposed, documents the current behavior as a regression guard)* | — | Exception raised by `generate_content` after all retries | **No entry recorded** (documents the current gap — should later be converted to a "should be recorded" test after the fix) | — | — | — |

### Audio Understanding

| Test name | Fixture | Note |
|---|---|---|
| `test_audio_complete_usage` | Full `_full_meta()` | One `AUDIO_UNDERSTANDING` entry, `COMPLETE` |
| `test_audio_partial_usage` | `prompt_token_count` only, missing `candidates_token_count` | `outputTokens` absent — partially already covered by `test_absent_fields_stay_absent` at the `gemini_usage.py` level |
| `test_audio_modality_breakdown` | `_Detail("AUDIO", 4)` | `audioInputTokens=4` kept separate from `inputTokens` |
| `test_streaming_final_usage_last_snapshot_wins` | Already exists: `TestGenerateStreamRecordsSingleEntry` in `test_llm_usage.py` | Covers the general logic; **no actual audio streaming path exists in the project** to test end-to-end today |
| `test_streaming_missing_final_usage` | `snapshots=[]` | Already exists as part of `TestFinalStreamUsage.test_none_when_no_snapshot` |
| `test_model_field_missing_uses_correct_sdk_attribute` *(proposed, post-fix)* | `_Resp(model_version="gemini-3.6-flash")` (not `model=`) | Should only pass **after** fixing `extract_response_model` — the current tests use `model=`, which is unrealistic |

### TTS

| Test name | Fixture | Note |
|---|---|---|
| `test_gemini_tts_complete_usage` | Fake TTS response with full `usage_metadata` and `candidates[0].content.parts[0].inline_data` | `TEXT_TO_SPEECH` entry, `COMPLETE` |
| `test_gemini_tts_missing_usage` | Same as above but no `usage_metadata` | `UNAVAILABLE` |
| `test_tts_provider_error_currently_unrecorded` *(proposed, regression guard for the current gap)* | Exception after both retries exhausted | **Zero TTS entries** despite a real network request having been sent — documents Scenario C/E from Section 10 |
| `test_tts_timeout_currently_unrecorded` | Same as above with `code=503` | Documents that `key.mark_failed` is not called for 503, yet the recording gap remains |
| `test_tts_success_with_no_model_version` | `usage_metadata` present, `model_version` absent | `actualModel` absent, everything else clean |
| `test_gtts_only_path_produces_no_gemini_call` *(proposed — endpoint-level, on `voice.py`)* | Gemini TTS fails → gTTS succeeds | `providerCalls` contains only `AUDIO_UNDERSTANDING` (one entry), **no** `TEXT_TO_SPEECH` with `provider=google` |
| `test_audio_success_tts_failure_two_vs_one_call` *(proposed)* | Same as above | Explicitly documents that the resulting count is **1, not 2**, despite the real TTS attempt |
| `test_multiple_real_calls_preserved_in_order` | Both Audio and TTS succeed | `[c["providerCallId"] for c in calls] == ["call-1", "call-2"]`, and `operation` in the correct order |

Example pseudocode in the project's own pytest style (no real network):

```python
# tests/test_voice_contract.py  (proposed — does not currently exist)
import asyncio
from app.core.usage import begin_usage_tracking, consume_usage

def _make_tts_success_response(model_version="gemini-3.1-flash-tts-preview"):
    class _Inline:
        data = b"\x00\x01"
        mime_type = "audio/l16;rate=24000;channels=1"

    class _Part:
        inline_data = _Inline()

    class _Content:
        parts = [_Part()]

    class _Candidate:
        content = _Content()

    class _UsageMeta:
        prompt_token_count = 40
        candidates_token_count = 300
        total_token_count = 340

    class _Resp:
        model_version = model_version   # the correct field, post-fix
        usage_metadata = _UsageMeta()
        candidates = [_Candidate()]

    return _Resp()


class TestAudioSuccessTtsSuccessTwoCalls:
    def test_two_provider_calls_in_order(self, monkeypatch):
        # patch key.client.models.generate_content to return an audio response
        # first, then a TTS response
        # ... (follows the pattern of _make_client in tests/test_llm_usage.py)
        begin_usage_tracking()
        # ... call generate_with_audio, then generate_speech
        calls = consume_usage()
        assert len(calls) == 2
        assert calls[0]["operation"] == "AUDIO_UNDERSTANDING"
        assert calls[1]["operation"] == "TEXT_TO_SPEECH"
```

---

## 14. Safe Live Probe Plan

**Cost warning:** every real execution of these scripts **consumes real quota/credit** on the Gemini API key used. **These scripts will not be executed by this audit without explicit permission and actual runtime access.**

### 14.1 Image probe

```python
"""probe_image.py — One live probe for Image Analysis. Not run automatically."""
import json, sys
from google import genai
from google.genai import types

print("google-genai version:", __import__("google.genai").__version__ if hasattr(__import__("google.genai"), "__version__") else "unknown")

client = genai.Client(api_key="<PROVIDED_SEPARATELY>")  # never print the key
MODEL = "gemini-3.6-flash"

# Synthetic, non-sensitive test image
with open("synthetic_test_image.png", "rb") as f:
    image_bytes = f.read()

resp = client.models.generate_content(
    model=MODEL,
    contents=[types.Content(role="user", parts=[
        types.Part(text="Describe this synthetic test image briefly."),
        types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
    ])],
)

print("requested model:", MODEL)
print("model_version present:", hasattr(resp, "model_version"), getattr(resp, "model_version", None))
print("response_id present:", hasattr(resp, "response_id"), getattr(resp, "response_id", None))

meta = getattr(resp, "usage_metadata", None)
if meta is None:
    print("usage_metadata: ABSENT")
else:
    for field in ("prompt_token_count", "candidates_token_count", "total_token_count",
                  "cached_content_token_count", "thoughts_token_count"):
        v = getattr(meta, field, None)
        print(f"{field}: {'ABSENT' if v is None else v} (type={type(v).__name__})")
    details = getattr(meta, "prompt_tokens_details", None) or []
    print("prompt_tokens_details:", [(d.modality, d.token_count) for d in details])

# Never print the real user prompt, image bytes, or the full raw response object
```

### 14.2 Audio probe

```python
"""probe_audio.py — One live probe for Audio Understanding."""
from google import genai
from google.genai import types

client = genai.Client(api_key="<PROVIDED_SEPARATELY>")
MODEL = "gemini-3.6-flash"

# Very short synthetic audio file (generic, non-sensitive sentence)
with open("synthetic_test_audio.mp3", "rb") as f:
    audio_bytes = f.read()

resp = client.models.generate_content(
    model=MODEL,
    contents=[types.Content(role="user", parts=[
        types.Part(inline_data=types.Blob(mime_type="audio/mpeg", data=audio_bytes)),
        types.Part(text="Process this audio and respond appropriately."),
    ])],
)

print("model_version:", getattr(resp, "model_version", None))
meta = getattr(resp, "usage_metadata", None)
print("usage_metadata present:", meta is not None)
if meta:
    print({f: getattr(meta, f, None) for f in
           ("prompt_token_count", "candidates_token_count", "total_token_count")})
    details = getattr(meta, "prompt_tokens_details", None) or []
    print("audio modality entries:", [(d.modality, d.token_count) for d in details])
```

### 14.3 TTS probe

```python
"""probe_tts.py — One live probe for Gemini TTS."""
from google import genai
from google.genai import types

client = genai.Client(api_key="<PROVIDED_SEPARATELY>")
MODEL = "gemini-3.1-flash-tts-preview"

resp = client.models.generate_content(
    model=MODEL,
    contents=[types.Content(role="user", parts=[types.Part(text="Welcome to Egypt.")])],
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
    ),
)

print("model_version:", getattr(resp, "model_version", None))
meta = getattr(resp, "usage_metadata", None)
print("usage_metadata present:", meta is not None)
if meta:
    print({f: getattr(meta, f, None) for f in
           ("prompt_token_count", "candidates_token_count", "total_token_count")})

part = resp.candidates[0].content.parts[0]
inline = getattr(part, "inline_data", None)
print("audio returned:", inline is not None and bool(getattr(inline, "data", None)))
print("mime_type:", getattr(inline, "mime_type", None) if inline else None)
print("audio byte length:", len(inline.data) if inline and inline.data else 0)
# No base64/audio content is ever printed
```

**Strict rules for all probes:** exactly one request per probe (unless the API requires otherwise), never print an API key/Authorization header, never print real user prompts/content, sanitized JSON output only.

---

## 15. Evidence Matrix

| Internal Endpoint | Scenario | Operation | Project File | Project Function | SDK Ver | Native Field | Meaning | Always/Conditional | Additive/Breakdown | Current Mapping | Double-Count Risk | Confidence | Classification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `/identify` | Cache Hit | — | `identify.py:46-49` | `identify_landmark` | — | — | No provider call | — | — | `providerCalls=null` (should be `[]`) | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/identify` | Cache Miss | `IMAGE_ANALYSIS` | `identify.py:99-141` | `identify_landmark` | 2.14.0 | `usage_metadata` | Request usage | Conditional | — | Direct, via `_record_provider_call` | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/identify` | Total input for image | `IMAGE_ANALYSIS` | `gemini_usage.py:43,49-50` | `extract_token_counts` | 2.14.0 | `prompt_token_count` | Total input incl. image | Conditional | Includes breakdown already | `inputTokens` | No (no addition performed) | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| `/identify` | Image breakdown | `IMAGE_ANALYSIS` | `gemini_usage.py:60-73` | `extract_token_counts` | 2.14.0 | `prompt_tokens_details[modality=IMAGE]` | Image-only tokens | Conditional | Breakdown-only | `imageInputTokens` | No | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| `/identify` | Actual model | `IMAGE_ANALYSIS` | `gemini_usage.py:19-24` | `extract_response_model` | 2.14.0 | **Should be** `model_version` | Real model used | Conditional (though the correct field exists in the schema always) | — | **Broken — reads non-existent `model`** | No | HIGH | `CONFIRMED_BY_SDK_SOURCE` (bug) |
| `/voice` | Audio Understanding | `AUDIO_UNDERSTANDING` | `voice.py:178-182`, `llm_client.py:353-401` | `generate_with_audio` | 2.14.0 | `usage_metadata` | Audio usage | Conditional | — | Direct | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/voice` | Audio breakdown | `AUDIO_UNDERSTANDING` | `gemini_usage.py:75-88` | `extract_token_counts` | 2.14.0 | `prompt_tokens_details[modality=AUDIO]` | Audio-only tokens | Conditional | Breakdown-only | `audioInputTokens` | No | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| `/voice` | Actual model for audio | `AUDIO_UNDERSTANDING` | Same as above | `extract_response_model` | 2.14.0 | `model_version` | — | Conditional | — | **Also broken** | No | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| `/voice` | TTS input usage | `TEXT_TO_SPEECH` | `llm_client.py:403-450` | `generate_speech` | 2.14.0 | `prompt_token_count` | Input text | Conditional, not live-verified for preview model | — | `inputTokens` | No | MEDIUM | `CONFIRMED_BY_SDK_SOURCE` (shape) / `UNKNOWN_REQUIRES_LIVE_PROBE` (population) |
| `/voice` | TTS output usage | `TEXT_TO_SPEECH` | Same as above | `generate_speech` | 2.14.0 | `candidates_token_count` | Output audio tokens | Conditional | — | `outputTokens` | No | MEDIUM | Same as above |
| `/voice` | TTS actual model | `TEXT_TO_SPEECH` | Same as above | `extract_response_model` | 2.14.0 | `model_version` | — | Conditional | — | **Broken** | No | HIGH | `CONFIRMED_BY_SDK_SOURCE` |
| `/voice` | TTS missing usage | `TEXT_TO_SPEECH` | `llm_client.py:127-141` | `_record_provider_call` | — | — | — | — | — | `usageCompleteness=UNAVAILABLE` (correct) | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/voice` | gTTS path | `TEXT_TO_SPEECH` (local) | `voice.py:83-93` | `gtts_audio_bytes` | `gtts==2.5.4` | — | No `usage_metadata` from Gemini | — | — | **Does not appear in `providerCalls[]` (correct)** | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/voice` | Audio success + TTS failure | `TEXT_TO_SPEECH` | `voice.py:96-113`, `llm_client.py:436-466` | `synthesize_speech` / `generate_speech` | — | — | — | — | — | **Zero trace of the failed TTS attempt** | No (but real undercounting risk exists) | HIGH | `CONFIRMED_BY_PROJECT_CODE` |
| `/generate` (text, not used directly here) | Streaming final usage | `TEXT_CHAT_STREAM` | `llm_client.py:171-191` | `_stream_to_async` | 2.14.0 | Last cumulative snapshot | — | — | — | `_record_stream_final` (correct) | No | HIGH | `CONFIRMED_BY_PROJECT_CODE` + existing test |
| `/generate` | Streaming partial usage | `TEXT_CHAT_STREAM` | `usage.py:214-227` | `final_stream_usage` | — | — | — | — | — | Skips empty intermediate snapshots (correct) | No | HIGH | Existing test `test_skips_empty_snapshots_between` |

---

## 16. Risk Register

| Risk | Probability | Impact | Evidence | Mitigation | Blocks pricing? |
|---|---|---|---|---|---|
| Double counting of modality breakdown | Low | High if it occurred | Confirmed not currently happening (code + SDK) | Keep `test_modality_breakdowns_are_additive_for_pricing_never_used` as a permanent regression guard | Not currently |
| **`actualModel` permanently broken** | **Confirmed (100%)** | Medium-High (loses the ability to detect Google-side model substitution, and weakens trust in any "actual" field in reporting) | `CONFIRMED_BY_SDK_SOURCE` | Immediate fix: `getattr(response, "model_version", None)` instead of `"model"` | **Yes — blocks reliance on `actualModel` in monitoring/billing until fixed** |
| Cache Hit returns `null` instead of `[]` | Confirmed | Low-Medium (type-safety issue for consumers) | `CONFIRMED_BY_PROJECT_CODE` | Move `payload` construction (usage/model/providerCalls) before the `_cache` write, or explicitly store `providerCalls: []` on read from cache | Partially yes |
| **Provider-call undercounting on failure** | Medium-High (especially for the preview TTS model with tighter rate limits) | **High** — real potential spend fully unaccounted for on the Core Server side | `CONFIRMED_BY_PROJECT_CODE` (Section 10, Scenarios C/E/F) | Record a call entry with `providerCallMade=true` plus an explicit failure state (`status=FAILED` or equivalent), even without usage | **Yes — must be resolved before fully relying on `providerCalls[]` to cover 100% of actual spend** |
| Streaming usage loss | Low | Medium | Handled correctly and tested in code | No change needed | No (the path is not actually used for image/audio today) |
| TTS failure after successful audio with no trace | High (same risk phrased differently) | High | Confirmed | Same mitigation as above | Yes |
| Incorrect gTTS classification in the future | Low today (current code is correct) but risk of future drift on careless changes | High if it occurred | No endpoint-level test currently protects this behavior | Add `test_gtts_only_path_produces_no_gemini_call` (proposed, Section 13) as a permanent regression guard | Not currently, but the test is recommended before any future change |
| Pricing an unimplemented operation (Image Generation) | Low (no code for it exists) | High if mistakenly added later | `CONFIRMED_BY_PROJECT_CODE` (no `response_modalities=["IMAGE"]` anywhere) | Keep `IMAGE_GENERATION` out of any Rate Card until it is actually developed | No (not implemented) |
| SDK version mismatch between environments (dev/prod) | Low-Medium | Medium | Production environment's actual version not directly verified, only the accompanying `uv.lock` (`google-genai==2.14.0`) | Confirm production uses the same lockfile (`uv.lock`), and monitor `pyproject.toml`'s wide floor of `>=1.0.0` (too broad, may allow versions with a different schema) | Not currently, but tightening `google-genai` to a narrower range (e.g., `>=2.0.0,<3.0.0`) is worthwhile |

---

## 17. Recommended Next Steps

**Intended sequence:** Verify actual responses → Freeze the response contract → Define usage interpretation rules → Add verified rate-card entries → Run shadow pricing → Connect wallet conversion later.

**Must do before pricing:**
1. Fix `extract_response_model` (read `model_version` instead of `model`).
2. Fix the `/identify` Cache Hit behavior to return `providerCalls: []`.
3. Run the three Live Probes (image/audio/TTS) and document actual results as an update appendix to this report.
4. Make an explicit design decision on how failed provider calls should be represented in `providerCalls[]`.

**Can do during shadow pricing:**
- Enable `IMAGE_ANALYSIS` and `AUDIO_UNDERSTANDING` for Shadow Pricing (without an actual Rate Card entry yet) once a Live Probe succeeds, since they do not carry the same uncertainty as the preview TTS model.
- Keep `TEXT_TO_SPEECH` (Gemini) in monitoring-only mode with no Rate Card entry until the model's stability is confirmed (preview → stable) or enough live data becomes available.

**Can be deferred:**
- The general `billingQuantities[]` structure — unnecessary as long as every current modality bills in tokens.
- `audioInputSeconds`/`transcriptionSeconds` fields — remain empty for the current Gemini path, and this is expected.

**Must not be included in this phase:**
- Any Rate Card entry for `IMAGE_GENERATION` (not implemented at all).
- Any wallet deduction/conversion logic — explicitly outside this audit's scope.

---

## 18. Final Decision

| Question | Answer |
|---|---|
| Can `IMAGE_ANALYSIS` be priced now? | **Not yet** — `READY_FOR_SHADOW_PRICING_ONLY` once a Live Probe is run; a real Rate Card entry needs further confirmation of resolution/image-count effects |
| Can `AUDIO_UNDERSTANDING` be priced now? | **Not yet** — same status: `READY_FOR_SHADOW_PRICING_ONLY` after a Live Probe |
| Can Gemini `TEXT_TO_SPEECH` be priced now? | **No** — `BLOCKED_PENDING_LIVE_PROBE`; recommended to remain `UNPRICED` on the Rate Card until the preview model's stability and `usage_metadata` population are confirmed |
| Is the current `providerCalls[]` contract sufficient? | **Sufficient structurally (schema-wise)**, but **not currently trustworthy at the implementation level** due to the `actualModel` bug and the missing recording of failed calls — the implementation must be fixed before full reliance |
| What are the exact remaining Live Probes? | One for Vision, one for Audio Understanding, one for Gemini TTS (scripts in Section 14) |
| Which Rate Card entries can safely be added? | None yet, definitively — Shadow Pricing can begin for `IMAGE_ANALYSIS`/`AUDIO_UNDERSTANDING` only, after the fixes above, without enabling real deductions |
| Which entries must remain excluded? | `IMAGE_GENERATION` entirely, `TEXT_TO_SPEECH` via gTTS (not billed at all), and `TEXT_TO_SPEECH` via Gemini until a Live Probe |

---

## 19. Appendix

### Official sources
- Counting tokens — Gemini API: `https://ai.google.dev/api/tokens`
- ModalityTokenCount reference: `https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/ModalityTokenCount`
- GenerateContentResponse reference (Vertex): `https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse`
- Live API capabilities: `https://ai.google.dev/gemini-api/docs/live-api/capabilities`
- Gemini Developer API pricing: `https://ai.google.dev/gemini-api/docs/pricing`
- Cloud Text-to-Speech pricing: `https://cloud.google.com/text-to-speech/pricing`

### SDK source (the strongest evidence used in this report)
- Package installed and directly inspected: `google-genai==2.14.0` (matches `uv.lock`)
- `google.genai.types.GenerateContentResponse` → actual fields: `candidates`, `create_time`, `model_version`, `prompt_feedback`, `response_id`, `usage_metadata`, `model_status`, ... — **no `model` field exists**.
- `google.genai.types.GenerateContentResponseUsageMetadata` → `prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`, `prompt_tokens_details`, `candidates_tokens_details`, `cache_tokens_details`, `tool_use_prompt_token_count`, `tool_use_prompt_tokens_details`, `traffic_type` — `total_token_count` is explicitly documented as the "sum of prompt_token_count, candidates_token_count, tool_use_prompt_token_count, and thoughts_token_count".
- `google.genai.types.MediaModality` → `str, Enum` with values `TEXT/IMAGE/VIDEO/AUDIO/DOCUMENT/MODALITY_UNSPECIFIED` — the comparison `modality == "IMAGE"` used in `gemini_usage.py` is **technically correct** because the type inherits from `str`.

### Secondary community sources (context only, not officially relied upon)
- Estimated tokens/second rates for audio (32 for input, 25 for output) — third-party pricing-comparison sources, not directly from Google.

### Unknowns requiring a Live Probe
- Actual population of `usage_metadata` on a real successful Vision/Audio/preview-TTS call.
- Whether a real cost is billed to the provider account when a request fails after being sent (timeout/5xx).
- The actual behavior of `model_version` across retry/fallback transitions between the different models in `GEMINI_MODEL_FALLBACKS`.

### Proposed sanitized Live Probe output schema
```json
{
  "requested_model": "string",
  "model_version_present": true,
  "model_version": "string|null",
  "response_id_present": true,
  "usage_metadata_present": true,
  "usage_fields": {
    "prompt_token_count": "int|null",
    "candidates_token_count": "int|null",
    "total_token_count": "int|null",
    "thoughts_token_count": "int|null",
    "cached_content_token_count": "int|null"
  },
  "modality_breakdown": [{"modality": "string", "token_count": "int"}]
}
```

### Proposed test fixtures (not yet implemented)
- `synthetic_test_image.png`: a simple programmatically generated image (no real user images).
- `synthetic_test_audio.mp3`: a short local TTS clip (a generic, non-sensitive sentence, e.g., "This is a test recording.").
- Generic test TTS sentence: "Welcome to Egypt."
