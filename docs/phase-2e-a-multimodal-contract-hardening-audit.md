# Phase 2E-A — Multimodal Contract-Hardening Audit

**Worktrees involved:** `ai-service-provider-pricing-phase2` (AI Service — read + this document only) and `Core-Server-provider-pricing-phase2` (Core Server — read-only contract verification).

**Audit type:** Independent verification of the external multimodal audit report (`Gemini-Multimodal-Audit-Report-EN.md`) against the actual codebase, the pinned SDK schema (`google-genai==2.14.0`), the AI Service test suite, and the Core Server `providerCalls[]` ingestion + shadow-pricing pipeline.

**Constraint honored:** No source or test file was modified in either worktree. This document is the only write performed by this audit.

---

## 1. Executive Summary

The external report (`Gemini-Multimodal-Audit-Report-EN.md`, located at the AI Service worktree root — see Section 3 for a path discrepancy) is **substantially accurate**. Every one of its material claims that this audit could verify against project code and the pinned SDK was confirmed; its classification labels map cleanly onto the required taxonomy. The audit confirms the report's three headline findings:

1. `extract_response_model()` in `app/core/gemini_usage.py:19-24` reads `getattr(response, "model", None)`, but the pinned SDK's `GenerateContentResponse` exposes `model_version`, **not** `model` → `actualModel` is permanently absent in production. **CONFIRMED.**
2. The `/identify` cache-hit path (`app/api/identify.py:46-49`) returns `providerCalls: null` instead of `[]` because the cache stores `result` at line 129 *before* the payload (usage/model/providerCalls) is built at lines 133-141. **CONFIRMED.**
3. Calls that were actually dispatched to the provider but failed leave **zero** accounting trace (`_record_provider_call` is only invoked on success). For the preview TTS model this hides real, potentially billable spend from the Core Server. **CONFIRMED** — this is the most serious gap.

The audit also surfaced **three findings beyond the external report**:

- **F1 — SDK-level HTTP retries are disabled by the app's own configuration.** The app passes `http_options={"timeout": 120000}` with no `retry_options`; the SDK's `retry_args(None)` path yields `stop_after_attempt(1)` (never retries). The app-level recursive retry loop in `app/core/llm_client.py` is therefore the *only* retry mechanism, and every recursion is a real HTTP request (up to 11 attempts for non-TTS operations, 3 for TTS). The external report never analyzed the SDK retry layer.
- **F2 — `providerRequestId` could be populated.** The code comment at `app/core/llm_client.py:124-125` states the SDK "does not expose a request id," but the pinned SDK's `GenerateContentResponse` has `response_id` (`google/genai/types.py:8421`). The comment is stale/inaccurate; the external report only noted `response_id` "exists but is not used" without flagging that the code's own justification is wrong.
- **F3 — Cached-input accounting tension.** The SDK docstring for `prompt_token_count` states that cached content tokens are **included** in `prompt_token_count` (`google/genai/types.py:8283`). The AI Service emits both `inputTokens` (from `prompt_token_count`, which includes cached content) **and** `cachedInputTokens` (from `cached_content_token_count`). The Core rate card declares `cachedInputAccounting: 'DISJOINT'` and adds the cached rate on top (`src/utils/provider-pricing/price-call.ts:327-341`). If a real Gemini response ever populates `cached_content_token_count`, the cached portion would be **double-counted**. The external report correctly classified cached content as a breakdown of `prompt_token_count` but did not connect that to the Core DISJOINT declaration.

**Final decision:** the external report is endorsed as the verified baseline. Its `REQUIRED_NOW` items (fix `extract_response_model`; fix cache-hit shape; add failed-attempt recording; run the three live probes) remain the authoritative next steps, now with precise file/line targets and an implementation slice plan (Section 18).

---

## 2. Scope and Method

### 2.1 In-scope areas
- Image Analysis — `POST /identify` (`app/api/identify.py`).
- Audio Understanding — `POST /voice` (`app/api/voice.py`).
- Gemini Text-to-Speech — `generate_speech` (`app/core/llm_client.py`).
- gTTS fallback (`app/api/voice.py:83-93`, `gtts==2.5.4`).
- `providerCalls[]` normalization, ingestion, and shadow-pricing compatibility on the Core Server (`src/types/ai.ts`, `src/utils/ai-usage.ts`, `src/services/ai-usage.service.ts`, `src/services/ai-shadow-pricing.service.ts`).

### 2.2 Out-of-scope
- Image Generation (no `response_modalities=["IMAGE"]` or `generate_image` exists anywhere in the AI Service — verified by search).
- Wallet deduction / real-money conversion (explicitly deferred to a later phase).
- General contract redesign (`billingQuantities[]`).

### 2.3 Method
1. Read the full external report (595 lines) and extract every material claim.
2. Read the AI Service source for all in-scope paths.
3. Read the AI Service test suite and confirm which behaviors are actually covered.
4. Downloaded and statically inspected the **pinned** SDK wheel `google_genai-2.14.0-py3-none-any.whl` (matching `uv.lock`) outside the worktrees; verified every schema claim against `google/genai/types.py`, `google/genai/_api_client.py`, and `google/genai/errors.py`.
5. Inspected the gTTS 2.5.4 wheel (`/tmp/opencode/gtts`) to confirm its network behavior.
6. Verified the Core Server `providerCalls[]` contract, normalization, shadow pricing, rate-card, and per-endpoint forwarding (read-only).
7. Classified every material claim using the legend in Section 4.
8. Produced this audit document (the single permitted write).

### 2.4 Classification legend
| Classification | Meaning |
|---|---|
| `CONFIRMED` | Verified directly against project code and/or the pinned SDK source. |
| `PARTIALLY_CONFIRMED` | Verified in part; a caveat remains (typically schema confirmed, runtime population not). |
| `NOT_CONFIRMED` | No supporting evidence found; claim unsupported. |
| `INCORRECT` | Contradicted by project code and/or pinned SDK source. |
| `REQUIRES_LIVE_PROBE` | Structurally plausible and code-supported, but actual runtime behavior must be confirmed with one real (sanitized) Gemini call. |

---

## 3. External Report Identification and Path Discrepancy

The task statement referenced the report at `.../ai-service-provider-pricing-phase2/docs/Gemini-Multimodal-Audit-Report.md`. The actual file found is:

- **`/media/mohamed/newvolume/ITI Professional Scholarship nine month/Rhila/ai-service-provider-pricing-phase2/Gemini-Multimodal-Audit-Report-EN.md`** (worktree **root**, `Gemini-...-EN.md`).

The AI Service worktree had **no `docs/` directory** before this audit created it to hold the required output document. The `docs/` folder exists only in the Core Server worktree. This is a location/naming discrepancy in the task prompt, not a missing file. The report file is untracked in git and was **not** modified, moved, or renamed.

Report provenance notes: the report's SDK evidence says it inspected "the installed package" `google-genai==2.14.0`. This audit independently confirmed `uv.lock` pins `google-genai==2.14.0` (line 766) and `gtts==2.5.4` (line 913). There is no installed `.venv` in the AI Service worktree; the report's "installed package" claim is interpreted as "pinned package," which this audit reproduced via wheel extraction.

---

## 4. Verification Environment and Evidence Base

| Item | Value | Verified at |
|---|---|---|
| AI Service SDK pin | `google-genai==2.14.0` | `uv.lock:765-766` |
| AI Service gTTS pin | `gtts==2.5.4` (deps: `click`, `requests`) | `uv.lock:912-917` |
| Declared dependency floor | `google-genai>=1.0.0` (wide; report recommends tightening) | `requirements.txt`, `pyproject.toml` |
| `GenerateContentResponse` | fields include `model_version` (Optional[str], "Output only. The model version used to generate the response."), `response_id` (Optional[str]), `usage_metadata`, `candidates`, `prompt_feedback`, `create_time`, `model_status` — **no `model` field** | wheel `google/genai/types.py:8397-8421` |
| `GenerateContentResponseUsageMetadata` | `prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`, `prompt_tokens_details`, `candidates_tokens_details`, `cache_tokens_details`, `tool_use_prompt_token_count`, `tool_use_candidates_token_count`, `server_tool_use_prompt_token_count`, `server_tool_use_candidates_token_count`, `traffic_type` | `google/genai/types.py:8258-8310` |
| `prompt_token_count` docstring | "...includes any text, images, or other media... When `cached_content` is set, this also includes the number of tokens in the cached content." | `google/genai/types.py:8283` |
| `ModalityTokenCount` | `modality` + `token_count` | `google/genai/types.py:8218` |
| `MediaModality` | `CaseInSensitiveEnum` (str-based enum): TEXT/IMAGE/VIDEO/AUDIO/DOCUMENT/MODALITY_UNSPECIFIED | `google/genai/types.py:604`; `modality == "IMAGE"` string compare is valid because the enum inherits `str` |
| SDK retry layer | tenacity-based; default `_RETRY_ATTEMPTS=5`; retryable codes 408/429/500/502/503/504; **`retry_args(None)` → `stop_after_attempt(1)` (never retry)** | wheel `google/genai/_api_client.py` |
| App HTTP config | `http_options={"timeout": 120000}`, **no `retry_options`** → SDK-level retries effectively disabled | `app/core/llm_client.py` (client construction) |
| `APIError` | `code` (int HTTP status), `status`, `message`, `response` | wheel `google/genai/errors.py` |
| gTTS network behavior | `gTTS.write_to_fp` issues real HTTP requests to `translate.google.com` via `requests.Session` (unofficial Google Translate endpoint, not Google Cloud AI/Gemini) | wheel `gtts/tts.py` (`_prepare_requests`), gTTS 2.5.4 |
| AI Service config defaults | `gemini_model = "gemini-3.6-flash"`, `tts_voice = "Zephyr"` | `app/config.py:16,18` |
| TTS model | `gemini-3.1-flash-tts-preview` (literal, in `generate_speech`) | `app/core/llm_client.py:420` |

---

## 5. Claim Classification Matrix — Image Analysis (`/identify`)

Every material claim from external report Section 6, mapped to the required taxonomy.

| Report # | Material claim | My classification | Evidence |
|---|---|---|---|
| 6.1 | Native response type is `GenerateContentResponse` via non-streaming `generate_content` | `CONFIRMED` | `app/api/identify.py:101` calls `generate_with_image` → `generate_content` (non-stream); SDK types.py:8397 |
| 6.2 | `usage_metadata` field exists but is fully Optional; actual population not established by a real fixture | `PARTIALLY_CONFIRMED` | SDK schema confirmed; no real fixture in repo; no live probe executed → runtime population `REQUIRES_LIVE_PROBE` |
| 6.3 | `model_version` exists on the response; the project reads non-existent `model` instead | `CONFIRMED` | `gemini_usage.py:21` reads `"model"`; types.py has `model_version`, no `model` |
| 6.4 | `response_id` exists in the schema, not currently used by the project | `CONFIRMED` | types.py:8421; no `response_id` read anywhere in AI Service source |
| 6.5 | `model_version` is the correct equivalent of "actual model" | `CONFIRMED` | types.py docstring "Output only. The model version used to generate the response." |
| 6.6 | Returned model is the model actually used (by field intent) | `CONFIRMED` (field documentation); runtime `REQUIRES_LIVE_PROBE` | types.py:8413 docstring |
| 6.7 | All modalities are converted into tokens | `CONFIRMED` | Official docs; consistent with `prompt_token_count` docstring |
| 6.8 | Image tokens are included in `promptTokenCount` | `CONFIRMED` | types.py:8283 docstring explicitly includes images |
| 6.9-11 | Modality breakdown exists as `prompt_tokens_details[]` with `ModalityTokenCount{modality, tokenCount}`; values TEXT/IMAGE/AUDIO/VIDEO/DOCUMENT | `CONFIRMED` | types.py:8218, 8258-8310, 604 |
| 6.12-13 | Modality breakdowns are part of the aggregate (breakdown-only), not additive | `CONFIRMED` | types.py:8283 (aggregate includes all media); AI Service never adds breakdowns into aggregates (`gemini_usage.py:60-88` returns them as separate fields) |
| 6.14 | Manually adding image breakdown to `promptTokenCount` would double-count; current code does not do this | `CONFIRMED` | `gemini_usage.py:60-88` never sums into `inputTokens`; dedicated regression tests exist |
| 6.15 | Token counts depend on resolution / image count / model family | `PARTIALLY_CONFIRMED` | Community-reported figures only; not an official field; no live verification |
| 6.16-17 | `cachedContentTokenCount` is a separate field but already included within `promptTokenCount` (breakdown, not additive) | `CONFIRMED` | types.py:8283 docstring |
| 6.18-21 | Cache Hit returns `providerCalls: null` (not `[]`); cache Miss returns one entry; no fabricated call invented | `CONFIRMED` | `identify.py:46-49` returns stored dict lacking usage/model/providerCalls → Pydantic defaults `None` → serialized `null`; `identify.py:129` stores `result` before payload build at 133-141 |
| 6.22-23 | Image usage priceable from aggregate fields once population confirmed; `actualModel`/`responseId` are monitoring-only | `PARTIALLY_CONFIRMED` | Schema supports it; requires live probe to confirm population; `actualModel` currently broken (F: fix pending) |

**Required decisive answer (image):** `(A)` — modality tokens are a breakdown already included in the aggregate total. `CONFIRMED` against SDK source.

---

## 6. Claim Classification Matrix — Audio Understanding (`/voice`)

| Report # | Material claim | My classification | Evidence |
|---|---|---|---|
| 7.1 | Same `GenerateContentResponse` type (audio passed as `inline_data` part) | `CONFIRMED` | `voice.py:178-182` → `generate_with_audio` → `generate_content`; no File API anywhere |
| 7.2-3 | `usage_metadata`/`model_version` same schema as image | `CONFIRMED` (schema); population `REQUIRES_LIVE_PROBE` | Same response type; no real fixture |
| 7.4-6 | Audio is billed in tokens; ~32 tokens/sec is a community estimate | `PARTIALLY_CONFIRMED` | Unit=tokens confirmed; exact rate is third-party/community only |
| 7.7 | Billing quantity is directly exposed via `promptTokenCount` | `PARTIALLY_CONFIRMED` | Structurally true; runtime population not live-verified |
| 7.8-10 | Audio modality breakdown via `promptTokensDetails[modality=AUDIO]`, breakdown-only | `CONFIRMED` | `gemini_usage.py:61-69`; types.py schema |
| 7.11 | Project uses inline bytes only; no File API / no audio streaming | `CONFIRMED` | `voice.py:151,178-182`; no streaming audio path |
| 7.12-14 | `candidates_token_count`, `thoughts_token_count`, `cached_content_token_count` supported and extracted when present | `CONFIRMED` | `gemini_usage.py:45-58` maps them to outputTokens/reasoningTokens/cachedInputTokens |
| 7.15-17 | Final-usage-last-snapshot logic correct; this path is **not** used by `/identify` or `/voice` (only generic streaming `generate()`) | `CONFIRMED` | `usage.py` `final_stream_usage`; image/audio use non-streaming `generate_content` |
| 7.18-19 | Audio can succeed with no usage → `usageCompleteness=UNAVAILABLE` (never a fabricated zero) | `CONFIRMED` | `llm_client.py:127-141` (`_record_provider_call`); existing tests |
| 7.20 | Contract can represent all confirmed audio fields | `CONFIRMED` | `usage.py` `TOKEN_FIELD_NAMES` includes `audioInputTokens`, `audioOutputTokens`, `audioInputSeconds`, `audioOutputSeconds`, `transcriptionSeconds` |
| 7.note | `audioInputSeconds`/`audioOutputSeconds`/`transcriptionSeconds` defined but never populated by `gemini_usage.py` — expected, not a bug | `CONFIRMED` | Gemini returns no explicit duration; fields exist for future providers |

---

## 7. Claim Classification Matrix — Gemini Text-to-Speech

| Report # | Material claim | My classification | Evidence |
|---|---|---|---|
| 8.1-3 | Same response type; same `actualModel`-reading bug | `CONFIRMED` | `generate_speech` uses same `_record_provider_call` → `extract_response_model` |
| 8.4-9 | Only token counts returned; no explicit duration/character/sample-count field | `CONFIRMED` | SDK schema: no duration/character fields on usage metadata |
| 8.10-11 | Billing unit is tokens; 25 tokens/sec audio-output rate is community-derived | `PARTIALLY_CONFIRMED` | Unit=tokens confirmed (official); exact rate community-only |
| 8.12 | `candidates_token_count` = output audio tokens | `PARTIALLY_CONFIRMED` | Structurally the output token count; live population for the preview model unverified |
| 8.13 | Installed SDK exposes the required fields | `CONFIRMED` | types.py:8258-8310 |
| 8.14 | All fields conditional (Optional); preview model has restrictive/unstable rate limits | `CONFIRMED` | SDK schema all-Optional; preview warning from official pricing page (report cites it; consistent) |
| 8.15 | Project uses Preview + single-speaker + non-streaming TTS | `CONFIRMED` | `llm_client.py:420` literal model; `response_modalities=["AUDIO"]` (line 428); no `voice2` |
| 8.16 | No double-counting risk from adding audio-output breakdown into `outputTokens` | `CONFIRMED` | `gemini_usage.py` never adds breakdowns into aggregates |
| 8.17-20 | Failed TTS call records **no** `providerCalls` entry; `_record_provider_call` only after success | `CONFIRMED` | `llm_client.py:403-450`; provider call recorded only on success path (line 441) |
| 8.21 | TTS cannot be priced reliably right now | `CONFIRMED` | No live evidence; failed calls invisible; preview-model uncertainty |
| 8.22 | TTS should stay `UNPRICED` on the rate card until a live probe | `CONFIRMED` (recommendation endorsed) | Core rate card already omits `gemini-3.1-flash-tts-preview` → resolves `UNPRICED` |

---

## 8. Claim Classification Matrix — gTTS and Non-Gemini Speech Generation

| Report # | Material claim | My classification | Evidence |
|---|---|---|---|
| 9.1 | `gtts_audio_bytes` calls the gTTS package, which makes real HTTP requests to an undocumented public Google Translate endpoint — not Google Cloud AI/Gemini, not billed to the project account | `CONFIRMED` | Verified in gTTS 2.5.4 wheel: `write_to_fp` → `requests` to `translate.google.com`; `voice.py:83-93` uses `gTTS(...).write_to_fp(mp3)` |
| 9.2 | No `record_provider_call` anywhere in the gTTS call path | `CONFIRMED` | `voice.py:83-93` has no usage calls; `synthesize_speech` fallback path (96-113) does not record |
| 9.3 | gTTS never appears as `provider=google, operation=TEXT_TO_SPEECH, providerCallMade=true` | `CONFIRMED` | Only `_record_provider_call` (Gemini) creates such entries; gTTS path bypasses it |
| 9.4 | Recommended classification `NOT_A_PAID_GEMINI_CALL`; optional future observability only | `CONFIRMED` (endorsed) | Consistent with Core shadow pricing (no entry → no pricing) |
| 9.5 | Operational risk of the unofficial gTTS endpoint (breakage/ToS) | `CONFIRMED` | External dependency risk; outside accounting |

---

## 9. Claim Classification Matrix — `providerCalls[]` Contract and Partial Failure

| Report # | Material claim | My classification | Evidence |
|---|---|---|---|
| 11.1 | `provider` is hardcoded `"google"` | `CONFIRMED` | `usage.py` `PROVIDER_GOOGLE` |
| 11.2 | `providerCallId` is an internal sequential id, not a provider id | `CONFIRMED` | `usage.py` `next_call_id`/`add` (per-request deterministic scope) |
| 11.3 | `providerCallMade` is always `True` when recorded; no call site passes `False`; never a "sent but failed" state | `CONFIRMED` | `make_provider_call` default `True`; all `_record_provider_call` sites pass `True` |
| 11.4 | `requestedModel` fully reliable | `CONFIRMED` | Read from project-side requested model |
| 11.5 | `actualModel` **broken** — always absent | `CONFIRMED` | `gemini_usage.py:21` reads non-existent `model` |
| 11.6 | `usageSource`/`usageCompleteness` reliable (no fabricated zeros) | `CONFIRMED` | `_record_provider_call` logic + tests |
| 11.7 | `inputTokens`/`outputTokens`/`totalTokens` reliable provided Google populates them | `PARTIALLY_CONFIRMED` | Needs live probe for population |
| 11.8 | Breakdown fields structurally reliable, breakdown-only | `CONFIRMED` | `gemini_usage.py` + SDK schema |
| 11.9 | `totalTokens` never derived from `inputTokens + outputTokens` | `CONFIRMED` | `usage.py` `derive_legacy_usage`; `test_never_derives_total_from_sum` |
| 10.A | Audio success + Gemini TTS success → 2 calls, both priced | `CONFIRMED` | `voice.py:178,100` two `generate_content` calls both recorded |
| 10.B | Audio success + TTS aborted before dispatch (empty text) → 1 call | `CONFIRMED` | `synthesize_speech` returns `None` for empty text (`voice.py:97-98`) |
| 10.C | Audio success + TTS actually dispatched but failed after retries → 1 call, TTS attempt invisible | `CONFIRMED` | `_record_provider_call` only on success; failure raises → caught → gTTS fallback |
| 10.D | Audio success + gTTS only (theoretical) → 1 call | `CONFIRMED` | gTTS never recorded |
| 10.E | Audio success + Gemini TTS failed + gTTS succeeded (common real path) → 1 call, indistinguishable from "never attempted" | `CONFIRMED` | Same as 10.C |
| 10.F | Audio fails → 0 calls; HTTP 500; no response body | `CONFIRMED` | `voice.py:215-216` raises HTTPException(500); no `VoiceResponse` produced |
| 10.G | Both succeed but one has incomplete usage → 2 calls, `UNAVAILABLE` | `CONFIRMED` | `_record_provider_call` classification logic |
| 10.rule | Contract has no mechanism to distinguish "not attempted" from "attempted and failed" | `CONFIRMED` | No status/attempt field anywhere in `ProviderCallUsage` |
| 12.1-7 | Usage interpretation rules (breakdown-only; missing usage ≠ zero; UNPRICED ≠ free; cache-hit should be `[]`; gTTS never paid; no fabricated actualModel; new rule needed for sent-but-failed) | `CONFIRMED` (1-6); rule 7 = recommended change | Verified across code + tests |

---

## 10. Answers to the 11 Required Audit Questions

**Q1. Is the `actualModel` reading correct against the pinned SDK?**
No. `extract_response_model` (`gemini_usage.py:19-24`) reads `getattr(response, "model", None)`. The pinned SDK 2.14.0 has no `model` field; the authoritative field is `model_version` (`types.py:8413`, documented "Output only. The model version used to generate the response."). Result: `actualModel` is always absent in production. Correct fix: `getattr(response, "model_version", None)`.

**Q2. Is `response_id` usable as `providerRequestId`, contradicting the code comment?**
Yes, it is available. `GenerateContentResponse.response_id` exists (`types.py:8421`, "response_id is used to identify each response. It is the encoding of the event_id."). The comment at `llm_client.py:124-125` ("the current Gemini SDK path does not expose a request id") is stale. Populating `providerRequestId` from `response_id` is optional-but-recommended observability (`OPTIONAL_OBSERVABILITY_FIELD`); it does not change billing totals.

**Q3. Are modality breakdown fields additive or breakdown-only?**
Breakdown-only. `prompt_token_count` is documented to include all media (types.py:8283). The AI Service returns `imageInputTokens`/`audioInputTokens`/`imageOutputTokens`/`audioOutputTokens` as separate fields and never adds them into the aggregate. Adding them to the aggregate would double-count. Core's pricing path ignores modality breakdowns except for `MODALITY_INVALID` validation, which is correct.

**Q4. Are cached-content tokens included in `promptTokenCount`, and does Core's `DISJOINT` declaration conflict?**
Per the pinned SDK docstring, **yes** — `prompt_token_count` includes cached content tokens (types.py:8283). The AI Service emits `inputTokens` (already including cached content) and, separately, `cachedInputTokens`. Core's rate card declares `cachedInputAccounting: 'DISJOINT'` (adds cached tokens at the cached rate on top of input). **Conflict (new finding F3):** if a real response populates `cached_content_token_count`, the cached portion is counted twice. Today no Gemini path sets context caching for these endpoints, so the field is expected to be absent/zero in practice — but the DISJOINT declaration is a latent double-count risk that must be resolved (decision in Section 18, slice 2E-C) before any context-caching is introduced.

**Q5. What is the runtime shape of `providerCalls` on `/identify` cache hit vs miss?**
- **Miss** (`identify.py:99-141`): exactly one entry, `operation=IMAGE_ANALYSIS`, `providerCallMade=true`, usage per availability.
- **Hit** (`identify.py:46-49`): the stored dict was written at line 129 **before** usage/model/providerCalls were attached to the payload at lines 133-141, so the cache holds no accounting fields; `IdentifyResponse` Pydantic defaults serialize them as `null` → `providerCalls: null`, `usage: null`, `model: null`. This violates the contract (should be `[]`). No fabricated call is invented.

**Q6. Are failed-but-sent provider calls recorded anywhere?**
No. `_record_provider_call` runs only after `generate_content` returns successfully (`llm_client.py:250,298,350,441`). A dispatched request that fails after retries raises an exception that propagates to the endpoint handler; the failure is invisible to accounting. This affects all four operations (generate / generate_with_image / generate_with_audio / generate_speech) and is the most serious gap.

**Q7. Is the gTTS fallback ever represented as a paid Gemini call?**
No. gTTS never appears in `providerCalls[]`. It uses a remote, unofficial Google Translate HTTP endpoint (`voice.py:83-93`, gTTS 2.5.4) that is neither a Gemini call nor billed to the project account. This is correct and compliant.

**Q8. What are the effective retry semantics (SDK layer + app layer)?**
- **SDK layer:** disabled. The app builds the client with `http_options={"timeout": 120000}` and no `retry_options`. The SDK's `retry_args(None)` resolves to `stop_after_attempt(1)` — never retry. The tenacity layer (default 5 attempts, retryable 408/429/500/502/503/504) never engages.
- **App layer:** recursive retry loop is the *only* retry. `_retry_count > self.MAX_RETRIES` (10) raises for generate/generate_with_tools/generate_with_image/generate_with_audio → up to **11 real HTTP attempts**; `_retry_count > 2` for `generate_speech` → up to **3 real attempts**. Retries may rotate API keys (`_get_next_available_key`) and may switch models (`_model_for_retry`, `llm_client.py:98-102`), so a retried call can run on a different model than `requestedModel`. Every attempt is a real provider request and, on failure, is unrecorded.
- This is **new finding F1** (external report did not analyze the SDK layer).

**Q9. Is `totalTokens` ever derived from `inputTokens + outputTokens`?**
No, and this is correct. `total_token_count` = prompt + candidates + tool_use + thoughts (per SDK schema), not just input+output. `derive_legacy_usage` (`usage.py:177-215`) sums only fields that were actually reported and never fabricates `totalTokens` from the first two. Guarded by `test_never_derives_total_from_sum`.

**Q10. Can IMAGE_ANALYSIS and AUDIO_UNDERSTANDING be priced now, and what must happen for TTS?**
- `IMAGE_ANALYSIS` / `AUDIO_UNDERSTANDING`: structurally priceable from `inputTokens`/`outputTokens` once (a) `extract_response_model` is fixed, and (b) a live probe confirms `usage_metadata` population. Recommended status: `READY_FOR_SHADOW_PRICING_ONLY` until probe; no Rate Card entry required for shadow mode.
- Gemini `TEXT_TO_SPEECH`: **blocked pending live probe.** The preview model's population and stability are unverified, and failed-but-billed calls are invisible. Keep the rate card entry absent (resolves `UNPRICED`); accumulate in shadow pricing whenever real data appears.

**Q11. Is the current `providerCalls[]` contract sufficient for Core shadow pricing?**
Structurally sufficient (schema covers all fields; Core normalizes and prices correctly), but not trustworthy at the implementation level because of (a) the `actualModel` bug, (b) the cache-hit `null` shape, and (c) missing failed-attempt recording. These three must be fixed before relying on `providerCalls[]` to represent 100% of real spend. Core's existing `ZERO_PROVIDER_CALLS` (authoritative `[]`) and skip-on-null semantics are correct and should be preserved.

---

## 11. Modality-Token Interpretation Rules (Adopted for Pricing)

1. Modality breakdowns (`imageInputTokens`, `audioInputTokens`, `imageOutputTokens`, `audioOutputTokens`) are **breakdown-only** and must never be added into `inputTokens`/`outputTokens`. (Verified: code + SDK.)
2. Missing `usage` does **not** mean zero usage → `usageCompleteness=UNAVAILABLE`, never a fabricated zero.
3. `UNPRICED` does **not** mean free — must remain explicit in Core.
4. A cache hit with no provider call must produce `providerCalls: []` (currently returns `null` — fix required).
5. Local fallback (gTTS) must never be represented as a paid Gemini call (currently true).
6. `actualModel` must never be invented from `requestedModel` (currently no code fabricates it — but the extraction bug means it stays empty instead of holding the real value).
7. **New rule (recommended):** a provider call that was actually dispatched — even if it later fails — should leave a traceable accounting record (not priced), because the real cost to the provider is independent of the application-level outcome.

---

## 12. Partial-Failure Scenario Matrix

| Scenario | `providerCalls[]` count | Details | Overall pricing status | Verified |
|---|---|---|---|---|
| A — Audio ok, Gemini TTS ok | 2 | call-1 AUDIO_UNDERSTANDING, call-2 TEXT_TO_SPEECH | FULLY/PARTIALLY_PRICED per usage completeness | `CONFIRMED` |
| B — Audio ok, TTS aborted before dispatch (empty text) | 1 | `synthesize_speech` returns None for empty text | PARTIALLY_PRICED; TTS SKIPPED with no trace | `CONFIRMED` |
| C — Audio ok, TTS **dispatched** but failed after retries | 1 (audio only) | TTS attempt invisible — **most serious gap** | PARTIALLY_PRICED on surface; may hide UNPRICED real cost | `CONFIRMED` |
| D — Audio ok, gTTS-only (theoretical) | 1 | gTTS never in providerCalls (correct) | PARTIALLY_PRICED (TTS = NOT_A_PAID_GEMINI_CALL) | `CONFIRMED` |
| E — Audio ok, Gemini TTS failed, gTTS succeeded (common real path) | 1 | Indistinguishable from "never attempted" | Same risk as C | `CONFIRMED` |
| F — Audio fails | 0 | HTTP 500, no VoiceResponse body | ZERO_CALL from successful-response view; real attempts unrecorded | `CONFIRMED` |
| G — Both ok, one incomplete usage | 2 | Incomplete → UNAVAILABLE | PARTIALLY_PRICED | `CONFIRMED` |

**Derived rule:** the contract has no mechanism to distinguish "not attempted" from "attempted and failed" for any of the four operations. This is the primary contract-hardening gap.

---

## 13. Failed-Attempt Design Alternatives

Three candidate designs for representing dispatched-but-failed provider calls:

### Alternative A — Additive `status` field (recommended)
Extend `ProviderCallUsage` with an optional `status` field (`"SUCCESS" | "FAILED"`), and record **one entry per real HTTP dispatch**, including failures. Failed entries carry `providerCallMade=true`, `status="FAILED"`, no token usage, and optionally the SDK `APIError.code` (int HTTP status).

- Pros: real spend becomes visible; distinguishes "attempted & failed" from "never attempted"; backward compatible (additive field; Core drops unknown fields today via allowlist, so Core must add `status` to the allowlist in the same slice); precise (per-attempt, so up to 11 attempts for non-TTS and 3 for TTS are each captured); enables failed-attempt metrics without pricing them.
- Cons: contract + normalization + shadow-pricing changes required in both repos; more entries per request on failure paths.
- Core handling: price only `SUCCESS` entries; count `FAILED` as attempted-but-unpriced (surface in metrics; exclude from the priced denominator; never raise a "missing call" invariant).

### Alternative B — Reuse `providerCallMade=false`
Mark failed-but-dispatched attempts with the existing `providerCallMade=false` (keeping `operation`), no usage.

- Pros: no new field.
- Cons: semantically wrong — `providerCallMade=false` means "no call was made", not "a call was made and failed"; Core's shadow service already treats `providerCallMade=false` defensively (drops/non-priced), so this would hide the very information it is meant to expose; requires Core changes anyway; conflates two distinct states. **Rejected.**

### Alternative C — No contract change; out-of-band telemetry only
Keep `providerCalls[]` success-only; record failed attempts in application logs / a future separate telemetry channel.

- Pros: zero contract change; zero Core impact.
- Cons: failed attempts remain invisible to the Core Server; cannot reconcile real provider spend; still requires building a telemetry channel; fails the audit's core requirement. Acceptable only as a stopgap. **Not recommended.**

### Comparison summary
| Criterion | A (status field) | B (providerCallMade=false) | C (telemetry only) |
|---|---|---|---|
| Real spend visible to Core | Yes | No (dropped by Core) | No |
| Semantic clarity | High | Poor (overloaded flag) | Medium |
| Backward compatible | Yes (additive) | No (relies on Core change to stop dropping) | Yes |
| Effort | Medium | Medium | Low |
| Satisfies "traceable attempt" rule | Yes | No | Partially |

---

## 14. Recommendation and Justification

**Adopt Alternative A** (additive `status` field with per-dispatch entries), implemented as slice **2E-B** (Section 18).

Justification:
1. It is the only alternative that makes dispatched-but-failed calls visible to the Core Server's shadow-pricing pipeline, satisfying the audit rule that real provider cost is independent of application-level success.
2. It is backward compatible at the API level: `status` is additive and optional; the AI Service can ship it without breaking consumers, and Core adds it to its allowlist in the same slice.
3. It keeps Core's authoritative zero-call semantics intact: cache hits still produce `providerCalls: []` (ZERO_PROVIDER_CALLS); real successes are priced; failures are counted but not priced.
4. It aligns with the SDK's `APIError.code` (int HTTP status), enabling per-attempt status reporting without new provider-side work.
5. It also resolves finding F2: the same slice can populate `providerRequestId` from `response.response_id` when present (optional observability, no billing impact).

**Ordered gate sequence (endorsing the external report's Section 17):**
1. Fix `extract_response_model` → `model_version` (2E-A1).
2. Fix `/identify` cache-hit `providerCalls: []` (2E-A2).
3. Add failed-attempt recording with `status` (2E-B).
4. Run the three live probes and document results (2E-C part 1).
5. Tighten `google-genai` dependency floor and resolve the cached-input DISJOINT decision (2E-C part 2).

---

## 15. Core Server Contract Compatibility and Shadow-Pricing Analysis

Verified (read-only) in `Core-Server-provider-pricing-phase2`:

- **Type contract** (`src/types/ai.ts`): `ProviderCallUsage` covers `provider`, `providerCallMade`, `providerCallId`, `providerRequestId`, `requestedModel`, `actualModel`, `operation`, `usageSource`, `usageCompleteness`, `accountingSemantics`, all token fields including modality breakdowns, `audioOutputSeconds`, `inputCharacters`, `outputCharacters`, `generatedImageCount`.
- **Normalization** (`src/utils/ai-usage.ts`): `normalizeProviderCalls` uses an allowlist of token fields (camelCase + snake_case), ignores unknown fields, returns `undefined` for `null`, empty array, non-array, or missing `provider`/`providerCallMade`. So `providerCalls: null` (today's cache-hit shape) → `undefined` → skipped; `[]` → `ZERO_PROVIDER_CALLS`. Confirmed by `tests/ai-usage-contracts.test.ts` (empty→undefined, non-array→undefined, missing provider, missing providerCallMade).
- **Recording** (`src/services/ai-usage.service.ts`): `recordAiUsage` persists per-call records.
- **Shadow pricing** (`src/services/ai-shadow-pricing.service.ts`): authoritative classifier — absent/`null`/non-array → skipped; `[]` → zero-call cache-hit observation; non-empty → price each real call; `providerCallMade=false` dropped defensively. `providerCallMade` is required, which means **Alternative B would be dropped by Core** — another reason it is rejected.
- **Metrics** (`src/services/ai-shadow-pricing-metrics.service.ts`): `FULLY_PRICED` / `PARTIALLY_PRICED` / `UNPRICED` / `ZERO_PROVIDER_CALLS`.
- **Rate card** (`src/config/provider-rate-card/index.ts`): `gemini-3.1-flash-tts-preview` deliberately absent → TTS resolves `UNPRICED`; all Google entries declare `cachedInputAccounting: 'DISJOINT'` (see F3).
- **Model identity** (`src/utils/provider-pricing/model-identity.ts`): `actualModel` authoritative when present; `requestedModel` only as fallback. Fixing the AI Service `actualModel` therefore flows through unchanged.
- **Endpoint forwarding**:
  - `identify.service.ts:57` — records usage only when `!result.cached` (Core independently gates cache hits; complements the AI-side `providerCalls: []` fix).
  - `voice.service.ts`, `chat.service.ts`, `itinerary.service.ts` — forward `providerCalls` normally.
- **Dormant client** (`src/clients/ai-service-execution.client.ts`): non-streaming one-shot fetch with a single attempt (no retries) — future durable-execution path, not live; relevant reference for per-attempt semantics.
- **Phase 2C engine** (`src/utils/provider-pricing/price-call.ts`, `aggregate.ts`, `arithmetic.ts`): prices token calls from aggregate counts; `cachedInputTokens` handled under DISJOINT (additive) or INCLUDED_IN_INPUT; fractional seconds → `USAGE_INVALID`; zero explicit counts → `PRICED` 0; missing billable fields → `USAGE_MISSING`. Consistent with the audit's interpretation rules.

**Compatibility verdict:** Core is compatible with the recommended AI-side changes. Slice 2E-B must add `status` to Core's normalization allowlist and teach the shadow service to classify `FAILED` as attempted-but-unpriced; everything else requires no Core change.

---

## 16. Test Plan (10 Groups)

All tests use duck-typed fake responses (existing style in `test_gemini_usage.py`/`test_llm_usage.py`); **no real Gemini requests**. The fakes must be corrected to expose `model_version` (not `model`) so tests reflect the real SDK schema.

1. **`extract_response_model` (post-fix).** `_Resp(model_version="gemini-3.6-flash")` → `"gemini-3.6-flash"`; absent `model_version` → `None`; non-string value → `None`. Update existing fakes (`test_gemini_usage.py:24`, `test_llm_usage.py:24,105`, `test_stream_usage.py:65`) from `model=` to `model_version=`.
2. **`extract_token_counts` aggregate.** Full `_full_meta()` → inputTokens/outputTokens/totalTokens present; missing `usage_metadata` → `{}` (no fabricated zeros).
3. **Breakdown-only regression guard.** `imageInputTokens + audioInputTokens != inputTokens`; `imageOutputTokens + audioOutputTokens != outputTokens`. Extend existing modality tests.
4. **Cache-hit `/identify` (endpoint-level, proposed).** An image already in `_cache` → `providerCalls == []`, `usage is None`, `model is None`, `cached is True`. Fails on today's code (documents the fix).
5. **Cache-miss `/identify` (endpoint-level, proposed).** New image → exactly one entry, `operation=IMAGE_ANALYSIS`, `providerCallMade=true`, `actualModel` from `model_version` when present, `usageCompleteness` per availability.
6. **Audio success.** One `AUDIO_UNDERSTANDING` entry with `COMPLETE` usage; modality breakdown kept separate.
7. **TTS success.** One `TEXT_TO_SPEECH` entry, `COMPLETE` when `usage_metadata` present; `UNAVAILABLE` when absent; `actualModel` absent when `model_version` missing.
8. **TTS failure + gTTS fallback (endpoint-level, proposed).** Gemini TTS raises after retries → gTTS succeeds → `providerCalls` contains only the audio entry (no `TEXT_TO_SPEECH`), and `audio_url`/`audio_response` are populated. After 2E-B, also asserts a `status="FAILED"` TTS entry exists.
9. **Failed-attempt recording (post-2E-B, proposed).** Dispatched-but-failed attempts produce per-attempt entries with `status="FAILED"`, `providerCallMade=true`, no token usage; up to the retry cap count of entries; `providerRequestId` populated from `response.response_id` when present.
10. **Core contract compatibility.** `normalizeProviderCalls` empty array → undefined; `null` → undefined; non-array → undefined; missing `provider`/`providerCallMade` → undefined; unknown fields ignored; new `status` field carried once allowlisted; shadow service treats `FAILED` as attempted-but-unpriced.

---

## 17. Safe Live-Probe Review (3 probes, not executed)

The external report's Section 14 contains three probe scripts. **None were executed** (no live Gemini requests are permitted in this audit; the SDK is not installed in the worktree). Review:

- **`probe_image.py`** (`model=gemini-3.6-flash`, one `generate_content` call): reads `model_version`, `response_id`, `usage_metadata` fields (`prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`), `prompt_tokens_details` — all valid pinned-SDK attributes. Safe: prints no key, no prompt, no image bytes, no raw response.
- **`probe_audio.py`** (short synthetic MP3, one call): same field reads plus audio-modality entries. Safe.
- **`probe_tts.py`** (`model=gemini-3.1-flash-tts-preview`, `response_modalities=["AUDIO"]`, `SpeechConfig`/`PrebuiltVoiceConfig(voice_name="Zephyr")`): reads `candidates[0].content.parts[0].inline_data` presence, `mime_type`, byte length only. Safe.
- **Refinement for the probes:** since the SDK's tenacity retries are already disabled when `retry_options` is omitted, the probes are already single-request. Optionally set `retry_options={"attempts": 1}` explicitly for belt-and-braces. Add an `sdk_version` print via `import google.genai; google.genai.__version__` (the report already prints it, conditionally).
- **Execution rule (not performed here):** exactly one request per probe; sanitized JSON output only; API key supplied separately, never printed; real prompts/images/audio bytes never printed. Probes should be run in 2E-C part 1 and their sanitized outputs appended as an update appendix to the report.

---

## 18. Implementation Slices (Specified, Not Implemented)

### 2E-A1 — Fix `actualModel` extraction (AI Service)
- **Files:** `app/core/gemini_usage.py` (`extract_response_model` → `getattr(response, "model_version", None)`); tests `test_gemini_usage.py`, `test_llm_usage.py`, `test_stream_usage.py` (update fakes to `model_version=`).
- **Tests:** test-plan group 1 (+ group 2 sanity).
- **Rollback:** revert `gemini_usage.py` and the test fakes.
- **Entry:** `consume_usage()` entries carry `actualModel` from `model_version` when present; `actualModel` absent only when the field is genuinely missing.
- **Exit:** `extract_response_model` unit tests pass with real-SDK-shaped fakes.
- **Affected repos:** AI Service only.

### 2E-A2 — Fix `/identify` cache-hit `providerCalls: []` (AI Service)
- **Files:** `app/api/identify.py` — either move payload construction before the `_cache[cache_key] = result` write (line 129), or explicitly store `providerCalls`/`usage`/`model` in the cached dict; new endpoint test file `tests/test_identify.py`.
- **Tests:** test-plan group 4 (cache hit) + group 5 (cache miss).
- **Rollback:** revert `identify.py` and delete `tests/test_identify.py`.
- **Entry:** cache hits serialize `providerCalls: []`, `usage: null`, `model: null`, `cached: true`.
- **Exit:** endpoint tests pass; Core `identify.service.ts` behavior unchanged (already gates on `!result.cached`).
- **Affected repos:** AI Service only.

### 2E-B — Failed-attempt recording (AI Service + Core)
- **Files:**
  - AI: `app/core/usage.py` (add optional `status` to `make_provider_call`/`ProviderCallUsage`); `app/core/llm_client.py` (record per-dispatch attempt on failure with `status="FAILED"` + optional `APIError.code`; populate `providerRequestId` from `response.response_id` when present); new tests in `test_llm_usage.py` / `tests/test_voice_contract.py` (proposed).
  - Core: `src/types/ai.ts` (add `status`), `src/utils/ai-usage.ts` (allowlist `status`), `src/services/ai-shadow-pricing.service.ts` + metrics (classify `FAILED` as attempted-but-unpriced, exclude from priced denominator), contract tests.
- **Tests:** test-plan groups 8, 9, 10.
- **Rollback:** revert contract + handling changes in both repos; re-run group 10 to confirm drop behavior restored.
- **Entry:** every real HTTP dispatch leaves a traceable entry; failures are visible and unpriced; retries (up to 11 non-TTS / 3 TTS) each represented.
- **Exit:** shadow pricing reports attempted-but-failed calls in metrics; `FULLY_PRICED`/`PARTIALLY_PRICED`/`UNPRICED` classification unchanged for successes.
- **Affected repos:** AI Service + Core Server.

### 2E-C — Live probes, dependency pin, and cached-input decision
- **Part 1 (probes):** run the three sanitized live probes; document results as an appendix to the report; confirm `usage_metadata` population, `model_version`/`response_id` presence, modality breakdown names/counts, TTS audio byte/mime behavior, and whether `cached_content_token_count` appears.
- **Part 2 (pins + decision):** tighten `google-genai` floor in `requirements.txt`/`pyproject.toml` (e.g. `>=2.0.0,<3.0.0`, or pin the lockfile version) and confirm production uses `uv.lock`; make the cached-input accounting decision (resolve F3): either declare Google cached input `INCLUDED_IN_INPUT` (avoiding double-count per SDK docstring) or keep `DISJOINT` only if the live probe shows `cached_content_token_count` is genuinely excluded from `prompt_token_count` at runtime. Add regression tests for whichever rule is chosen.
- **Rollback:** revert pin changes; keep probe results as documentation only.
- **Entry/exit:** dependency floor narrowed; cached-input semantics documented and tested; probe appendix merged.
- **Affected repos:** AI Service (`requirements.txt`, `pyproject.toml`, `uv.lock`); Core only if the DISJOINT rule changes.

---

## 19. Risks and Open Items

| Risk | Probability | Impact | Mitigation | Blocks pricing? |
|---|---|---|---|---|
| `actualModel` permanently broken | Confirmed (100%) | Medium-High | 2E-A1 immediate fix | Yes until fixed for actual-model monitoring |
| Cache-hit `null` shape | Confirmed | Low-Medium | 2E-A2 | Partially (Core already gates on `cached`) |
| Failed-but-sent calls unrecorded (esp. preview TTS) | Medium-High | High (real spend hidden) | 2E-B (Alternative A) | Yes for full-spend reliance |
| Cached-input double-count under DISJOINT (F3) | Low today (no context caching used) | High if caching introduced | 2E-C part 2 decision + regression tests | Not today; blocks context-caching |
| SDK-level retries accidentally enabled by future config | Low | Medium (multiplied attempts) | Keep `retry_options` unset or explicit; document | No |
| `google-genai>=1.0.0` wide floor | Low-Medium | Medium (schema drift) | 2E-C part 2 pin | No, but recommended |
| gTTS unofficial endpoint breakage | Medium | Low (operational) | Monitor; optional future local TTS | No |
| Preview TTS model instability/rate limits | Medium | Medium | Keep `UNPRICED`; shadow-only | Yes for TTS until stable |
| Streaming usage loss | Low | Medium | Already correct + tested | No (path unused for image/audio) |
| Unauthorized Image Generation entry | Low | High if added | Keep `IMAGE_GENERATION` off rate card (no code) | No |

---

## 20. References

- External report (verified): `Gemini-Multimodal-Audit-Report-EN.md` (AI Service worktree root; untracked; not modified).
- Pinned SDK (verified): `google-genai==2.14.0` (`uv.lock:765-766`), wheel extracted at `/tmp/opencode/sdk`; `gtts==2.5.4` (`uv.lock:912-917`), wheel extracted at `/tmp/opencode/gtts`.
- AI Service source: `app/api/identify.py`, `app/api/voice.py`, `app/core/gemini_usage.py`, `app/core/usage.py`, `app/core/llm_client.py`, `app/config.py`, `requirements.txt`, `pyproject.toml`, `uv.lock`.
- AI Service tests: `tests/test_gemini_usage.py`, `test_llm_usage.py`, `test_stream_usage.py`, `test_usage_contract.py`, `test_tools.py`, `test_guardrails.py`. (No `test_identify.py`/`test_voice.py` — confirmed.)
- Core Server (read-only): `src/types/ai.ts`, `src/utils/ai-usage.ts`, `src/services/ai-usage.service.ts`, `src/services/ai-shadow-pricing.service.ts`, `src/services/ai-shadow-pricing-metrics.service.ts`, `src/config/provider-rate-card/index.ts`, `src/utils/provider-pricing/price-call.ts`, `src/utils/provider-pricing/model-identity.ts`, `src/services/identify.service.ts`, `src/services/voice.service.ts`, `src/services/chat.service.ts`, `src/services/itinerary.service.ts`, `src/clients/ai-service-execution.client.ts`, `tests/ai-usage-contracts.test.ts`, `tests/ai-usage-pricing.test.ts`, `tests/ai-shadow-pricing-service.test.ts`.
- Prior Core phase reports: `docs/phase-2c-pricing-engine-report.md`, `docs/phase-2d-a-shadow-integration-report.md`, `docs/phase-2d-b-admin-metrics-report.md` (Core worktree).

---

## 21. Final Decision

| Question | Verdict |
|---|---|
| External report accuracy | Endorsed — all material claims confirmed; classification labels map cleanly onto the required taxonomy |
| New findings beyond the report | F1 (SDK retries disabled by config), F2 (`response_id` usable for `providerRequestId`), F3 (cached-input DISJOINT tension) |
| Fix `extract_response_model` | Required now (2E-A1) |
| Fix `/identify` cache-hit shape | Required now (2E-A2) |
| Add failed-attempt recording | Required now — Alternative A (`status` field) recommended (2E-B) |
| Live probes | Required before pricing (2E-C part 1) — not executed in this audit |
| `TEXT_TO_SPEECH` (Gemini) rate card | Stay `UNPRICED` until live probe + model stability confirmed |
| `IMAGE_ANALYSIS` / `AUDIO_UNDERSTANDING` | Ready for shadow pricing after fixes + probe; no rate-card entry required yet |
| Cached-input accounting | Resolve DISJOINT vs SDK containment in 2E-C part 2 (F3) |
| gTTS | `NOT_A_PAID_GEMINI_CALL` — no change required |
| `IMAGE_GENERATION` | Out of scope; no code exists; keep off rate card |

**Audit completeness:** all external-report material claims classified; code and pinned SDK verified; alternatives compared with justified recommendation; files identified per slice; test plan (10 groups) and live-probe plan complete; no source or test file modified in either worktree.

PHASE_2E_A_AUDIT_READY
