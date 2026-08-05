# Phase 2E-A1 — Confirmed Contract Fixes Report

**Worktree:** `ai-service-provider-pricing-phase2` (AI Service). **Phase:** 2E-A1 (the first slice of the Phase 2E-A confirmed-contract-fixes workstream).

**This slice implements the two confirmed, uncontroversial fixes** that the Phase 2E-A audit (`docs/phase-2e-a-multimodal-contract-hardening-audit.md`) and the external report (`Gemini-Multimodal-Audit-Report-EN.md`) both marked as `REQUIRED_NOW`, and pins them with the required regression and endpoint tests. All other confirmed gaps (failed-attempt recording, live probes, dependency pinning, cached-input accounting decision) remain **out of scope** for this slice by design.

**Final marker:** `PHASE_2E_A1_READY` (see Section 14).

---

## 1. Executive Summary

Two production bugs confirmed by the Phase 2E-A audit were fixed and covered by tests:

1. **Fix 1 — `actualModel` was permanently absent.** `extract_response_model()` in `app/core/gemini_usage.py` read `getattr(response, "model", None)`, but the pinned `google-genai` SDK exposes `GenerateContentResponse.model_version`, **not** `model`. The function now reads `model_version`. `actualModel` is populated from the real provider field and is **never** fabricated from `requestedModel`; when `model_version` is absent/null/empty/invalid, `actualModel` stays absent. The fix is applied through the single shared extraction function, so text, image, audio, TTS, tools, and streaming all benefit.

2. **Fix 2 — `/identify` cache hits returned `providerCalls: null`.** The cache stored `result` (`app/api/identify.py:133`) **before** the response payload was augmented with `usage`/`model`/`providerCalls` (lines 141-144), so cache-hit responses serialized those fields as `null`. The cache-hit path now builds a per-hit shallow copy of the stored dict, sets `cached=true` on the copy only (the stored entry is never mutated), explicitly sets `providerCalls=[]`, and keeps `usage`/`model` absent (`None`). Cache hits never call the Gemini client and never start/consume usage tracking.

Verification summary:

| Check | Result |
|---|---|
| Focused suites (`test_gemini_usage`, `test_usage_contract`, `test_llm_usage`, `test_stream_usage`, `test_identify`) | **60 passed** |
| Full suite (85 tests collected) | **84 passed, 1 failed** |
| Pre-existing unrelated failure (`test_tools.py::TestTools::test_tool_definitions_exist`) | Confirmed pre-existing and unrelated; left untouched (see Section 11) |

**Decisive finding for the required tests:** all 7 required extraction tests and all 10 required endpoint cache tests pass on the fixed code, and the 3 modality-token regression guards hold.

---

## 2. Scope and Constraints

### 2.1 In scope (this slice)
- `app/core/gemini_usage.py` — `extract_response_model` reads `model_version`.
- `app/api/identify.py` — cache-hit response shape (`providerCalls: []`, `usage: null`, `model: null`, copy-per-hit, no client call, no usage-tracking touch).
- Test fakes updated to the real SDK shape (`model_version`, not `model`): `tests/test_gemini_usage.py`, `tests/test_llm_usage.py`, `tests/test_stream_usage.py`.
- New endpoint tests: `tests/test_identify.py`.
- This report: `docs/phase-2e-a1-confirmed-contract-fixes-report.md`.

### 2.2 Explicitly out of scope (not implemented, per instructions)
- **No Core Server changes** — Core Server is read-only for this slice.
- **No changes to the `providerCalls` schema** — no `status`, no `providerAttempts[]`, no failed-attempt recording (`status=FAILED`), no `providerRequestId`/`responseId` population (deferred to 2E-B).
- **No pricing changes** — no Rate Card, Wallet, Durable Billing, or Shadow Pricing edits.
- **No retry changes**, **no Image Generation**, **no live Gemini probes**.
- **No dependency or environment changes** — the existing venv
  (`/media/mohamed/newvolume/ITI Professional Scholarship nine month/Rhila/.venvs/ai-service-usage-accounting-phase1`)
  was used; nothing was installed or upgraded.
- **No commits/pushes.**

### 2.3 Hard contract rules honored (unchanged by these fixes)
- `actualModel` is **never** derived from `requestedModel` (no fabrication fallback).
- `actualModel` is a `Optional[str]`; absent/empty/invalid `model_version` → field absent.
- The field name `actualModel` is preserved.
- Cache-hit responses keep `usage: null` and `model: null` (they are **not** populated with stale miss data).
- `providerCalls` on cache hits is explicitly `[]` (authoritative zero-call signal for Core's shadow pricing).
- The stored cached dict is **never** mutated in place; each hit builds its own payload copy.
- Cache-miss `providerCalls` are **never** stored in the cache and therefore never reused on hits.
- Cache hits make **zero** Gemini client calls and **never** `begin_usage_tracking()`/`consume_usage()`.
- Cache keys are unchanged (`f"{md5(image_bytes)}_{lat}_{lon}"`).
- `totalTokens` is never derived as `inputTokens + outputTokens`.
- Modality breakdowns (`imageInputTokens`, `audioInputTokens`, `imageOutputTokens`, `audioOutputTokens`) are breakdown-only and never added into aggregates.
- `cachedContentTokenCount` is surfaced as `cachedInputTokens` and never re-added into `inputTokens`.

---

## 3. Verification Environment and Evidence Base

| Item | Value | Evidence |
|---|---|---|
| Worktree | `ai-service-provider-pricing-phase2` | `git branch --show-current` → `feature/provider-pricing-phase2` |
| Python venv | `/media/mohamed/newvolume/ITI Professional Scholarship nine month/Rhila/.venvs/ai-service-usage-accounting-phase1` | Used for all runs; no installs performed |
| Test runner | pytest 9.1.1 | `python -m pytest --version` |
| HTTP test stack | httpx 0.28.1, fastapi 0.141.1, pydantic 2.13.4 | `fastapi.testclient.TestClient` works in this venv |
| Installed SDK | `google-genai==2.16.0` (venv) | `GenerateContentResponse.model_version` present, `model` absent |
| Pinned SDK | `google-genai==2.14.0` (`uv.lock:766`) | Verified earlier via extracted wheel: `model_version` (Optional[str], "Output only. The model version used to generate the response."), `response_id`, `usage_metadata`; **no `model` field** |
| `usage_metadata` fields | `prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`, `prompt_tokens_details`, `candidates_tokens_details` | Confirmed on installed 2.16.0 |
| Auth for endpoint tests | `X-Internal-Api-Key` = `settings.internal_api_key` (default `"change-me-in-production"`); admins exempt from rate limit | `app/core/auth.py`, `app/core/ratelimit.py` |
| Baseline suite (before this slice) | **66 collected; 65 passed, 1 failed** | Pre-existing `test_tools.py` failure only |
| Suite after this slice | **85 collected; 84 passed, 1 failed** | Same single pre-existing failure (Section 11) |

---

## 4. Fix 1 — `actualModel` reads `model_version`

### 4.1 Before
`app/core/gemini_usage.py:19-24`:
```python
def extract_response_model(response: Any) -> Optional[str]:
    """Return the actual model used by the provider response, or None."""
    model = getattr(response, "model", None)
    if isinstance(model, str) and model:
        return model
    return None
```
The SDK never sets `response.model` → `actualModel` was always absent in production for every operation (text, image, audio, TTS, tools, streaming).

### 4.2 After
`app/core/gemini_usage.py:19-30`:
```python
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
```

### 4.3 Why this is correct and complete
- The single shared extraction function is consumed by every path: `app/core/llm_client.py:128` (`_record_provider_call` → non-streaming text/tools/image/audio/TTS) and `app/core/llm_client.py:186` (`_stream_to_async` streaming final model). Fixing it once fixes all paths.
- The `isinstance(model, str) and model` guard preserves the existing convention: only truthy strings are accepted; `None`, `""`, and non-strings (e.g. `12345`) yield `None`.
- No dual `model`/`model_version` compatibility was added: the task's authoritative SDK contract is `model_version`, and no production integration outside `google-genai` is proven to need `model`.

---

## 5. Fix 2 — `/identify` cache-hit `providerCalls: []`

### 5.1 Root cause (confirmed)
- Cache write: `app/api/identify.py:133` stores `result` (the parsed landmark dict plus `nearby_sites`/`cached`).
- Payload augmentation happens **after** the store at lines 141-144 (`usage`, `model`, `providerCalls` are attached to `payload`, not to the stored `result`).
- Cache hit: old lines 46-49 did `cached = _cache[cache_key]; cached["cached"] = True; return IdentifyResponse(**cached)`.
  - It **mutated the stored dict in place** (`cached["cached"] = True`).
  - It returned the stored dict, which has no `usage`/`model`/`providerCalls` keys → Pydantic defaults (`None`) → JSON `providerCalls: null` instead of `[]`.

### 5.2 After
`app/api/identify.py:46-53`:
```python
    if cache_key in _cache:
        cached = _cache[cache_key]
        payload = dict(cached)
        payload["cached"] = True
        payload["usage"] = None
        payload["model"] = None
        payload["providerCalls"] = []
        return IdentifyResponse(**payload)
```

### 5.3 Contract guarantees implemented
- **Copy per hit:** `payload = dict(cached)`; the stored dict is never mutated (guarantees the stored `cached: false` marker and the absence of accounting fields persist across hits).
- **`providerCalls: []` explicit:** an authoritative zero-call signal (Core shadow pricing treats `[]` as `ZERO_PROVIDER_CALLS`).
- **`usage: null`, `model: null`:** cache hits never reuse the originating miss's usage/model (must stay absent).
- **No client call, no usage tracking:** the hit path returns before `from app.main import llm_client` (line 55) and before `begin_usage_tracking()` (line 104), so no Gemini call and no usage-scope init/consume can occur on a hit.
- **Miss path unchanged:** one `IMAGE_ANALYSIS` provider call, `cached: false`, `usage`/`model` derived when available, `providerCalls` carrying the recorded call. `consume_usage()` is still called once, exactly on the miss path.

---

## 6. Required Extraction Tests (7)

All pass (`tests/test_gemini_usage.py`, class `TestExtractResponseModel`).

| # | Test | Assertion |
|---|---|---|
| 1 | `test_reads_model_version_from_response` | `extract_response_model(_Resp(model_version="gemini-3.6-flash")) == "gemini-3.6-flash"` |
| 2 | `test_response_model_alone_is_not_accepted` | A fake exposing only `model` returns `None` (guards against the old wrong field) |
| 3 | `test_none_when_missing` | No `model_version` attribute → `None` |
| 4 | `test_none_when_model_version_none` | `model_version=None` → `None` |
| 5 | `test_none_when_model_version_empty` | `model_version=""` → `None` |
| 6 | `test_none_when_model_version_invalid` | `model_version=12345` → `None` |
| 7 | `test_none_when_response_is_none` | `extract_response_model(None)` → `None` |

**Duck-typed fakes updated to the real SDK shape** (this mirrors the SDK contract and would fail loudly on a regression to `model`):
- `tests/test_gemini_usage.py`: `_Resp.__init__(model_version=...)`.
- `tests/test_llm_usage.py`: `_Chunk.__init__(model_version=...)`, `_Resp.model_version`.
- `tests/test_stream_usage.py`: `_Chunk.__init__(model_version=...)`.

---

## 7. Required Endpoint Cache Tests (10)

All pass (`tests/test_identify.py`, real FastAPI app via `TestClient`, module-level `llm_client` monkeypatched to a fake that records one `IMAGE_ANALYSIS` provider call per miss; auth via `X-Internal-Api-Key`).

| # | Test | Verifies |
|---|---|---|
| 1 | `test_cache_hit_returns_empty_provider_calls` | Miss: `cached:false`, 1 provider call. Hit: `cached:true`, `providerCalls == []` |
| 2 | `test_cache_hit_usage_is_null` | Miss has `usage`; hit has `usage is None` |
| 3 | `test_cache_hit_model_is_null` | Miss `model == "gemini-3.6-flash"`; hit `model is None` |
| 4 | `test_cache_hit_does_not_call_provider` | After 3 requests (1 miss + 2 hits), fake client called exactly once |
| 5 | `test_cache_hit_does_not_touch_usage_tracking` | After a hit, `consume_usage() == []` (no dangling/leaked scope) |
| 6 | `test_cache_hit_does_not_reuse_miss_provider_calls` | Miss call (`operation=IMAGE_ANALYSIS`, `totalTokens=140`) never reappears; hit returns `[]` |
| 7 | `test_cache_hit_does_not_mutate_stored_entry` | Stored dict still `cached:false` and has no `providerCalls`/`usage`/`model` after two hits; same object identity |
| 8 | `test_cache_key_distinguishes_image_content` | Different image bytes → cache miss → second provider call |
| 9 | `test_cache_key_distinguishes_lat_lon` | Different lat/lon → cache miss → second provider call |
| 10 | `test_cache_miss_preserves_usage_and_provider_calls` | Miss body: `cached:false`, `usage.inputTokens==100`, `usage.totalTokens==140`, `model=="gemini-3.6-flash"`, one call with `usageSource=PROVIDER_RESPONSE`, `name` parsed correctly |

**Plus one guard test:** `test_empty_image_rejected` (empty upload → HTTP 400, provider never called) — confirms the empty-image guard and prevents the fake client from being invoked on invalid input.

---

## 8. Modality-Token Regression Guard Tests

Added/extended to prevent the pricing-rule violations the audit flagged. All pass.

| # | Guard | File / test |
|---|---|---|
| 1 | **Breakdowns never added into aggregates** | `test_gemini_usage.py::test_modality_breakdowns_are_additive_for_pricing_never_used` — `imageInputTokens + audioInputTokens != inputTokens` (with a full `_full_meta()` fixture) |
| 2 | **`cachedContentTokenCount` not re-added** | `test_gemini_usage.py::test_cached_content_token_count_not_readded_to_aggregates` — `inputTokens` stays `1024`, `cachedInputTokens==512`, and `inputTokens != inputTokens + cachedInputTokens` |
| 3 | **`totalTokens` never derived as input+output** | `test_gemini_usage.py::test_total_tokens_is_provider_reported_not_input_plus_output` — with `prompt=100, candidates=40, total=180`, asserts `totalTokens == 180 != 140`; plus `test_total_tokens_absent_when_only_parts_reported` — with only `input`/`output` reported, `totalTokens` stays **absent** |

These complement the pre-existing guard in `tests/test_usage_contract.py::test_never_derives_total_from_sum` (aggregate-level) and keep the contract consistent with Core's shadow-pricing arithmetic.

---

## 9. Focused Test Results

Command:
```
.../.venvs/ai-service-usage-accounting-phase1/bin/python -m pytest \
    tests/test_gemini_usage.py tests/test_usage_contract.py \
    tests/test_llm_usage.py tests/test_stream_usage.py tests/test_identify.py -q
```

**Result: `60 passed`** (5 files; includes all 7 extraction tests, all 10+1 endpoint tests, and the modality-token guards). One benign deprecation warning (`StarletteDeprecationWarning: install httpx2`) — pre-existing, unrelated.

---

## 10. Full Suite Results

Command: `python -m pytest -q`

**Result: `85 collected; 84 passed, 1 failed`** (up from the 66-test baseline).

The only failure is the pre-existing, unrelated one — see Section 11. No other test regressed.

---

## 11. Pre-Existing Unrelated Failure (documented precisely, left untouched)

| Item | Detail |
|---|---|
| Test | `tests/test_tools.py::TestTools::test_tool_definitions_exist` |
| Failure | `assert len(TOOL_DEFINITIONS) >= 9` → `AssertionError: assert 8 >= 9` |
| Cause | The test expects at least 9 tool definitions; the module currently defines 8 |
| Related to this slice? | **No** — tool definitions are unrelated to usage accounting, `actualModel`, or `/identify` caching |
| Pre-existing? | **Yes** — present in the baseline run before any change in this slice (66 collected, same single failure) |
| Action taken | **None.** Left untouched per instructions ("do not alter unrelated tests") |
| Remaining risk | If `TOOL_DEFINITIONS` grows to 9, the test passes; the discrepancy is a separate feature/test-coordination item outside 2E-A1 |

---

## 12. Confirmed Out-of-Scope Items (not implemented)

Explicitly **not** done in this slice, per the task's constraint list — listed for completeness so the report is unambiguous about what 2E-A1 does and does not cover:

1. Core Server modifications (read-only in this slice).
2. `providerCalls` schema changes: no `status`, no `providerAttempts[]`, no failed-attempt recording (`status=FAILED`).
3. `responseId` / `providerRequestId` population.
4. Retry behavior changes.
5. Pricing changes: no Rate Card, Wallet, Durable Billing, Shadow Pricing edits.
6. Image Generation.
7. Live Gemini probes.
8. Dependency / environment changes (venv untouched; no installs).
9. Git commits or pushes.

These remain open for subsequent slices (2E-B, 2E-C) as described in the Phase 2E-A audit's Section 18.

---

## 13. Compliance Checklist

| Constraint | Status |
|---|---|
| Worked only in `ai-service-provider-pricing-phase2` | ✅ |
| No Core Server edits | ✅ |
| No `providerCalls` schema / pricing / Wallet / Durable Billing / Shadow Pricing / Rate Card edits | ✅ |
| No Image Generation, `responseId`, failed-attempt recording, retry changes | ✅ |
| No live Gemini probes | ✅ |
| No dependency / environment changes | ✅ |
| No commits/pushes | ✅ |
| `actualModel` never fabricated from `requestedModel` | ✅ (only `model_version`, else absent) |
| `actualModel` field name preserved | ✅ |
| Fix applied via shared extraction function (text/image/audio/TTS/tools/streaming) | ✅ |
| Cache hits: `providerCalls: []` explicit; `usage: null`; `model: null` | ✅ |
| Stored cached dict never mutated in place (copy per hit) | ✅ |
| Cache-miss `providerCalls` not stored / not reused | ✅ |
| Cache hits never call Gemini client or init/consume usage tracking | ✅ |
| Cache keys unchanged | ✅ |
| 7 required extraction tests | ✅ (all pass) |
| 10 required endpoint cache tests | ✅ (all pass) |
| Modality-token regression guards (breakdowns not added; `cachedContentTokenCount` not re-added; `totalTokens` never input+output) | ✅ (all pass) |
| Pre-existing unrelated failure documented, not altered | ✅ (Section 11) |
| Used existing venv only | ✅ |

---

## 14. Final Decision

| Question | Verdict |
|---|---|
| Is `actualModel` now extracted from the real SDK field (`model_version`)? | **Yes** — Fix 1 applied and tested (Section 4, 6) |
| Do `/identify` cache hits now return `providerCalls: []` without calling the provider, touching usage tracking, or mutating the cache? | **Yes** — Fix 2 applied and tested (Section 5, 7) |
| Are the 7 required extraction tests in place and passing? | **Yes** |
| Are the 10 required endpoint cache tests in place and passing? | **Yes** |
| Are the modality-token regression guards in place and passing? | **Yes** |
| Full suite | **84 passed, 1 failed** — the sole failure is pre-existing and unrelated (`test_tools.py`, Section 11) |
| Out-of-scope items (2E-B, 2E-C, etc.) | Confirmed not implemented (Section 12) |

**All gates for 2E-A1 pass.** The two confirmed `REQUIRED_NOW` fixes are implemented, covered by the required test matrix, and verified against the full suite with no new failures introduced.

PHASE_2E_A1_READY
