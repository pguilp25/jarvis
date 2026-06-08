"""
NVIDIA NIM API client — 5 frontier models, rate-limited at 40 RPM shared.
Uses OpenAI-compatible /v1/chat/completions endpoint.
Supports both sequential and PARALLEL calls, and SSE streaming to thought_logger.
"""

import json as _json
import os
import asyncio
import aiohttp
from core.http_timeout import http_timeout
import fcntl


_GPTOSS_LOCK_PATH = "/tmp/jarvis_gptoss.lock"


class _gptoss_serialize:
    """Cross-process advisory lock so only ONE JARVIS process calls gpt-oss at a
    time — the free tier is concurrency-1, so two concurrent callers 429 each
    other into degraded fallbacks. ENV-GATED (JARVIS_GPTOSS_LOCK) and FAIL-OPEN:
    any locking error → proceed UNLOCKED, never block or crash a real call. So
    with the env unset (default) this is a pure no-op — zero risk to normal runs.

    Granularity is PER gpt-oss CALL (one coder round), so two runs sharing the
    lock interleave round-by-round; the PLANNING phase never uses gpt-oss, so it
    never touches the lock and overlaps freely. This is the mechanism behind the
    night interleave (SWE-bench + a real-world coding run on the same box)."""

    def __init__(self, model_id: str):
        # Per-call flock is fine for gpt-oss: within ONE run the native coder makes
        # one gpt-oss call at a time (sequential rounds), so there's no intra-run
        # concurrency to preserve here (unlike the parallel planner drafts).
        self.on = bool(os.environ.get("JARVIS_INTERLEAVE")) and ("gpt-oss" in (model_id or ""))
        self.fh = None

    async def __aenter__(self):
        if not self.on:
            return self
        try:
            self.fh = open(_GPTOSS_LOCK_PATH, "w")
            waited = 0.0
            while True:
                try:
                    fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.25)
                    waited += 0.25
                    if waited > 1800:   # 30-min safety valve: never deadlock a run
                        break
        except Exception:
            try:
                if self.fh:
                    self.fh.close()
            except Exception:
                pass
            self.fh = None   # fail-open
        return self

    async def __aexit__(self, *exc):
        if self.fh is not None:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
                self.fh.close()
            except Exception:
                pass
            self.fh = None
from typing import Optional
from config import NVIDIA_MODEL_IDS, NVIDIA_SLEEP_BETWEEN, STREAM_TTFT_TIMEOUT
from core.cli import thinking, warn
from core.rate_limiter import nvidia_limiter
from core.stream_guard import DegenerationDetector

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LIGHTNING_API_URL = "https://lightning.ai/api/v1/chat/completions"
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _apply_gptoss_pin(payload: dict, url: str, api_model: str) -> None:
    """Pin PAID gpt-oss-120b to DeepInfra@bf16 on the OpenRouter route ONLY (user-approved
    2026-06-03; the :free pool 503s, bf16 is the reliable full-precision route).
    allow_fallbacks=False → never silently land on a cheaper/lower-quant provider; if DeepInfra is
    down the call errors and the coder chain falls to the next model. Guarded on the ENDPOINT, not
    the slug (bughunt #18 — the NIM last-resort route resolves to the SAME slug but a different URL
    and must NOT get this OR-only pin). Shared by call_nvidia_tools AND call_nvidia_stream so the
    streaming JSON-ops coder is pinned too (bughunt #11)."""
    if url != OPENROUTER_API_URL:
        return
    if api_model == "openai/gpt-oss-120b":
        payload["provider"] = {"order": ["DeepInfra"], "quantizations": ["bf16"],
                               "allow_fallbacks": False}
    elif "nemotron-3-ultra" in api_model:
        # User directive (2026-06-08, refined): keep the EXPENSIVE planner free-only. Nemotron 3
        # Ultra bills up to $0.94/request ($3.95 burned on the first bad run) — a bare `:free` slug
        # SILENTLY ESCALATES to that paid provider when the free pool is busy (allow_fallbacks
        # defaults true). Pinning allow_fallbacks=False keeps Ultra on the FREE provider only:
        # serves free or 402s (chain falls through) — never bills. The OTHER :free planners
        # (nemotron-super ~$0.02/req, gemma-4 ~$0.04/req) are CHEAP, so we leave allow_fallbacks at
        # the default → they fall back to paid when the free pool is exhausted instead of crippling
        # the planner pool to owl-alpha alone (free-only 402s were starving it). gpt-oss-120b is the
        # paid coder (DeepInfra-pinned above); owl-alpha is a free alpha (no :free suffix, no pin).
        payload["provider"] = {"allow_fallbacks": False}

# Models we deliberately route to DeepInfra. Pro is intentionally NOT here:
# DeepInfra serves Pro FP4-quantized at only 66k context (vs 200k+ on NVIDIA
# and 1M native), so we keep Pro on NVIDIA/Lightning. Flash on DeepInfra
# keeps the full 1M context, which is what we want for huge code repos.
DEEPINFRA_MODELS = {
    # deepseek-v4-flash REMOVED (ckpt-178): DeepInfra returns HTTP 402 "need positive
    # balance" → it 402'd on EVERY planner round in the ckpt-177 run. ckpt-174 un-forced
    # it from OR but left it here, so _route hit DeepInfra (line 210) BEFORE NIM. With it
    # gone (and LIGHTNING_API_KEY unset), _route falls through to NIM (line 226) →
    # "deepseek-ai/deepseek-v4-flash" which is VERIFIED 200 @ ~5.9s — the FASTEST planner.
}

# OpenRouter slugs — every entry MUST be a :free model (user-confirmed
# constraint 2026-05-18: no paid OR usage). When NIM hosts the same
# model, the routing prefers NIM unless the model is also in the
# OPENROUTER_FORCED set (below), in which case OR is the primary route.
#
# Free-tier OR upstream rate-limits hit ~1 call/sec per model; the retry
# layer in core/retry.py absorbs short 429 bursts.
OPENROUTER_MODELS = {
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash:free",
    "minimax-m2.5":      "minimax/minimax-m2.5:free",
    "gpt-oss-120b":      "openai/gpt-oss-120b",       # coder primary — PAID OR, pinned to
                                                      # DeepInfra@bf16 in call_nvidia_tools
                                                      # (user-approved 2026-06-03; the :free
                                                      # upstream pool 503s "Provider returned error")
    "qwen3-coder":       "qwen/qwen3-coder:free",     # 1st text-coder fallback: 429s
                                                      # INSTANTLY when full → ~ms failover
    # User reported (2026-05-18 dashboard inspection): glm-4.5-air:free on
    # OR is the source of free-tier 429 storms. glm-5.1 stays on NIM where
    # it works reliably; do NOT route glm-* to OR.
    # No deepseek-v4-pro — no :free OR variant and paid is off-limits.
    # No kimi-k2.6     — replaced by minimax-m2.5 in the planner pool.
    # ── ckpt-187 NON-FRONTIER planner pool (user 2026-06-06: prove the workflow,
    #    not a frontier model like glm-5.1 / kimi-k2.6). All :free; owl-alpha is a
    #    free alpha/stealth slug (no :free suffix). Verified live 2026-06-06. ──
    "nemotron-3-ultra-550b-a55b": "nvidia/nemotron-3-ultra-550b-a55b:free",   # NEW Nemotron 3 Ultra
    "nemotron-3-super-120b-a12b": "nvidia/nemotron-3-super-120b-a12b:free",   # Nemotron 3 Super (flakier → fallback)
    "owl-alpha":                  "openrouter/owl-alpha",                     # Owl Alpha (free alpha, ~2s, stable)
    "gemma-4-31b-it":             "google/gemma-4-31b-it:free",               # Google Gemma 4 (open, 2 providers)
}

# Models that ALWAYS route via OpenRouter regardless of NVIDIA_API_KEY
# presence — NIM endpoints for these are unresponsive 2026-05-18.
OPENROUTER_FORCED = {
    # deepseek-v4-flash REMOVED from forced-OR (ckpt-174): OR :free returns "No
    # endpoints found" (dead pool) — it 404'd EVERY call on the ckpt-173 run, a dead
    # planner that forced slow cascades. With the new NVIDIA key, NIM serves it
    # (deepseek-ai/deepseek-v4-flash → 200, verified), so let it route to NIM.
    "minimax-m2.5",   # NOT on NIM at all (404) — only OR :free (also dead); stays here
                      # (fails fast → cascade). TODO: replace in PLAN_MODELS, it's dead weight.
    "gpt-oss-120b",   # coder: route OR FIRST (now PAID DeepInfra@bf16 pin), not NIM
    "qwen3-coder",    # text-coder fallback on OR :free (instant-429 fast failover)
    # ckpt-187 non-frontier planner pool — force to OR :free (their NIM/native
    # routes don't apply; OR is the only home for these slugs).
    "nemotron-3-ultra-550b-a55b", "nemotron-3-super-120b-a12b",
    "owl-alpha", "gemma-4-31b-it",
}

# ── OpenRouter key pool ──────────────────────────────────────────────
# The user runs TWO OpenRouter accounts. Free (:free) models are quota-
# capped PER ACCOUNT: once a key's daily :free allotment is spent the
# upstream provider returns HTTP 402 "Provider returned error" (observed
# 1,178× in the v14 overnight run). retry.py then fell straight to a
# weaker model, silently degrading planning/coding. Rotating round-robin
# across both keys (a) doubles the effective free budget and (b) — paired
# with the 3× same-model retry in core/retry.py — lets a 402'd call land
# on the OTHER account and stay on the SAME model instead of degrading.
# Keys come from OPENROUTER_API_KEYS (comma/space separated); falls back
# to the single OPENROUTER_API_KEY. No paid usage — both keys hit :free.
_OR_KEY_IDX = 0


def _openrouter_keys() -> list[str]:
    raw = (os.environ.get("OPENROUTER_API_KEYS", "").strip()
           or os.environ.get("OPENROUTER_API_KEY", "").strip())
    seen: set[str] = set()
    keys: list[str] = []
    for k in raw.replace(",", " ").split():
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _next_openrouter_key() -> str:
    """Round-robin the OpenRouter key pool. Each call advances the index
    so consecutive retries of the same model land on different accounts.
    asyncio is single-threaded and there is no await between the read and
    the increment, so the bump is race-free across concurrent tasks."""
    global _OR_KEY_IDX
    keys = _openrouter_keys()
    if not keys:
        return ""
    k = keys[_OR_KEY_IDX % len(keys)]
    _OR_KEY_IDX += 1
    return k


def _route_provider(model_id: str, provider: str) -> "tuple[str, str, str] | None":
    """Force a SPECIFIC provider's endpoint for `model_id` (used to cycle a model
    like gpt-oss-120b across OpenRouter → NVIDIA NIM before the fallback chain
    switches to a DIFFERENT model). Returns (url, key, slug) or None if that
    provider can't serve it (no key / not mapped)."""
    base = model_id.split("/", 1)[-1]
    if provider == "openrouter":
        k = _next_openrouter_key()
        if k and base in OPENROUTER_MODELS:
            return OPENROUTER_API_URL, k, OPENROUTER_MODELS[base]
    elif provider == "nvidia":
        k = os.environ.get("NVIDIA_API_KEY", "")
        if k:
            return NVIDIA_API_URL, k, NVIDIA_MODEL_IDS.get(model_id, base)
    return None


def _route(model_id: str) -> tuple[str, str, str]:
    """Pick endpoint, auth key, and provider-specific model slug.

    Priority per model:
      1. OPENROUTER_FORCED — these models go to OR :free regardless of
         JARVIS_PREFER_OPENROUTER. Used when NIM is hosting a broken
         endpoint (e.g. deepseek-v4-flash returning 300s ReadTimeout).
      2. OpenRouter if JARVIS_PREFER_OPENROUTER=1 AND key is set AND
         model is in OPENROUTER_MODELS — global fallback.
      3. DeepInfra — only for models in DEEPINFRA_MODELS.
      4. Lightning AI — if LIGHTNING_API_KEY is set.
      5. NVIDIA NIM — integrate.api.nvidia.com (free, occasionally flaky).

    For routes that fail at call time, retry layer in core/retry.py
    falls through to the per-model chain in config.NVIDIA_FALLBACKS.
    """
    base = model_id.split("/", 1)[-1]
    orkey = _next_openrouter_key()

    # 1. Forced OR routes
    if base in OPENROUTER_FORCED and orkey and base in OPENROUTER_MODELS:
        return OPENROUTER_API_URL, orkey, OPENROUTER_MODELS[base]

    # 2. Global OR-preferred mode
    prefer_or = os.environ.get("JARVIS_PREFER_OPENROUTER", "0") == "1"
    if prefer_or and orkey and base in OPENROUTER_MODELS:
        return OPENROUTER_API_URL, orkey, OPENROUTER_MODELS[base]

    dkey = os.environ.get("DEEPINFRA_API_KEY", "")
    if dkey and base in DEEPINFRA_MODELS:
        return DEEPINFRA_API_URL, dkey, DEEPINFRA_MODELS[base]

    lkey = os.environ.get("LIGHTNING_API_KEY", "")
    if lkey:
        return LIGHTNING_API_URL, lkey, f"lightning-ai/{base}"

    nkey = os.environ.get("NVIDIA_API_KEY", "")
    if not nkey:
        # Last-resort: try OR even without prefer flag.
        if orkey and base in OPENROUTER_MODELS:
            return OPENROUTER_API_URL, orkey, OPENROUTER_MODELS[base]
        raise RuntimeError(
            "None of OPENROUTER_API_KEY / DEEPINFRA_API_KEY / "
            "LIGHTNING_API_KEY / NVIDIA_API_KEY is set"
        )
    return NVIDIA_API_URL, nkey, NVIDIA_MODEL_IDS.get(model_id, base)


def _get_key() -> str:
    # Kept for callers (clients/imagen.py) that still need the NVIDIA key directly.
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


def _max_thinking_payload(model_id: str) -> dict:
    """Per-model parameters that force the strongest available reasoning mode.

    Defaults vary by provider/family:
      • DeepSeek V4 Pro/Flash → `reasoning_effort: "high"` by default; "xhigh"
        is the documented map for the "max" budget. We also set the explicit
        `thinking: {type: enabled}` so hosts that key off it (rather than
        reasoning_effort) still surface reasoning_content.
      • Kimi K2.6 → thinking is ON by default; we still send the explicit
        enable so a host that flipped the default doesn't silently disable it.
      • GLM-5.1 → thinking is ON by default; same belt-and-suspenders
        approach. We send the canonical `thinking` plus the vLLM-style
        `chat_template_kwargs` so it works against either parser.

    All values are OpenAI-compatible JSON fields. A host that does not
    recognize a field generally ignores it; if a provider returns HTTP 400
    on one of these, narrow this map for that model.
    """
    base = model_id.split("/", 1)[-1].lower()
    if base.startswith("deepseek-v4"):
        return {
            "reasoning_effort": "xhigh",
            "thinking": {"type": "enabled"},
        }
    if base.startswith("kimi-"):
        return {
            "thinking": {"type": "enabled"},
        }
    if base.startswith("glm-"):
        return {
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": True},
        }
    return {}


async def call_nvidia(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    Call an NVIDIA model. model_id is our config name like 'nvidia/deepseek-v4-pro'.
    Acquires rate limiter before calling. Returns response text.
    """
    await nvidia_limiter.acquire()
    thinking(model_id)

    url, key, api_model = _route(model_id)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # Output-token floor — see call_nvidia_stream for rationale.
        "max_tokens": max(int(max_tokens), 4096),
        **_max_thinking_payload(model_id),
    }

    _apply_gptoss_pin(payload, url, api_model)   # free-only pin for :free planners + gpt-oss DeepInfra pin

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=http_timeout(url, payload)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {body[:200]}")
            data = await resp.json()

    return data["choices"][0]["message"]["content"]


async def call_nvidia_tools(
    model_id: str,
    messages: list,
    tools: list,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    tool_choice: str = "auto",
    force_provider: str = "",
) -> dict:
    """NATIVE tool-calling call (2026-05-27). Unlike call_nvidia (text in/out),
    this takes a full `messages` list + OpenAI `tools` schemas and returns the
    raw assistant MESSAGE dict (role/content/tool_calls) so the caller can run
    a structured tool-use loop. Used for models built for native function
    calling (gpt-oss), which don't speak JARVIS's text-tag protocol. Routes via
    _route (gpt-oss → OpenRouter :free). Non-streaming — tool-calling turns are
    bounded and we need the whole tool_calls array at once. Raises on non-200 so
    the caller can retry / fall over (same error strings retry.py classifies)."""
    # TEST HOOK (inert unless set) — see clients/api.py. Forces this native model
    # to raise so the coder fallback chain can be exercised under controlled cond.
    _ff = os.environ.get("JARVIS_FORCE_FAIL_MODELS", "")
    if _ff and any(s and s in model_id for s in _ff.split(",")):
        raise RuntimeError(f"HTTP 404: forced test failure for {model_id} (JARVIS_FORCE_FAIL_MODELS)")
    await nvidia_limiter.acquire()
    thinking(model_id)
    if force_provider:
        _forced = _route_provider(model_id, force_provider)
        if _forced is None:
            raise RuntimeError(
                f"{model_id} not serviceable on provider '{force_provider}' "
                f"(no key / not mapped)")
        url, key, api_model = _forced
    else:
        # OpenAI-compatible non-NVIDIA providers (e.g. mistral/medium → api.mistral.ai)
        # speak the standard tools= function-calling API, so the same payload + parser
        # work here — only the endpoint/key/slug differ. Route them via openai_compat
        # so a former text-only model can serve as a NATIVE-tool coder. gpt-oss never
        # reaches this branch (it always passes force_provider). (user 2026-06-02)
        from clients.openai_compat import PROVIDERS as _OAI_PROV, _resolve as _oai_resolve
        if model_id.split("/", 1)[0] in _OAI_PROV:
            url, key, api_model = _oai_resolve(model_id)
        else:
            url, key, api_model = _route(model_id)
    payload = {
        "model": api_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "max_tokens": max(int(max_tokens), 4096),
        **_max_thinking_payload(model_id),
    }
    _apply_gptoss_pin(payload, url, api_model)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # gpt-oss serialization (night interleave): hold the cross-process lock only for
    # the duration of the actual gpt-oss HTTP call. No-op unless JARVIS_GPTOSS_LOCK is
    # set; fail-open on any lock error.
    async with _gptoss_serialize(model_id):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=http_timeout(url, payload)) as resp:
                raw = await resp.text()
                if resp.status != 200:
                    # tool_choice="required" isn't universally supported on :free
                    # endpoints — some 400 on it. Fall back to "auto" ONCE on the same
                    # endpoint rather than cascading to a weaker model. (user 2026-06-02)
                    # ckpt-145: this downgrade SILENTLY disabled the empty-turn guard —
                    # once on "auto" the model is free to stop with no tool call. LOG it
                    # (with the 400 body) so we can see when/why it fires; the caller
                    # also retries the empty-turn with required forced.
                    if resp.status == 400 and tool_choice == "required":
                        import sys as _sys
                        print(f"⚠️  [nvidia] {api_model}: HTTP 400 on tool_choice=required "
                              f"→ DOWNGRADING to auto (empty-turn guard OFF this call). "
                              f"400 body: {raw[:240]}", file=_sys.stderr, flush=True)
                        payload["tool_choice"] = "auto"
                        async with session.post(url, json=payload, headers=headers,
                                                timeout=http_timeout(url, payload)) as resp2:
                            raw = await resp2.text()
                            if resp2.status != 200:
                                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp2.status}: {raw[:200]}")
                    else:
                        raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {raw[:200]}")
    return _extract_tool_message(raw, api_model)


def _extract_tool_message(raw: str, api_model: str) -> dict:
    """Parse a chat-completions body into the assistant message, raising CLEAR,
    retry-classifiable errors for every malformed shape instead of a bare
    KeyError/IndexError. Free providers (OpenRouter :free) sometimes return an
    error object or an empty `choices` with HTTP 200 — those must surface as a
    real error (and 429/rate strings stay retryable), not crash the coder."""
    try:
        data = _json.loads(raw)
    except Exception:
        raise RuntimeError(f"NVIDIA {api_model} returned non-JSON body: {raw[:200]}")
    if not isinstance(data, dict):
        raise RuntimeError(f"NVIDIA {api_model} returned non-object JSON: {str(data)[:200]}")
    if data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            msg = err.get("message", str(err))
            code = err.get("code", err.get("type", ""))
        else:
            msg, code = str(err), ""
        raise RuntimeError(f"NVIDIA {api_model} error {code}: {str(msg)[:200]}")
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"NVIDIA {api_model} returned no choices: {str(data)[:200]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        fr = choices[0].get("finish_reason", "") if isinstance(choices[0], dict) else ""
        raise RuntimeError(f"NVIDIA {api_model} choice has no message "
                           f"(finish_reason={fr}): {str(choices[0])[:200]}")
    # Surface finish_reason so the coder loop can diagnose empty-turns: a no-tool-call
    # with finish_reason=="stop" is the harmony analysis→commentary boundary stop;
    # =="length" means the reasoning channel exhausted the output budget. (synthetic
    # key — the loop builds its own assistant turn, never re-sends this dict.)
    message["_finish_reason"] = choices[0].get("finish_reason", "")
    return message


async def call_nvidia_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    log_label: str = "",
    stop_check: object = None,
    messages_override: list = None,
) -> str:
    """
    Call an NVIDIA model with SSE streaming.
    Streams each response chunk to thought_logger as it arrives.
    If stop_check(accumulated_text) returns True, stops early.
    `messages_override`: a full chat history to send instead of [system, prompt]
    (used by the JSON-ops coder loop, which needs multi-round conversation). When
    given, `prompt`/`system` are ignored for the request (prompt may still be "").
    Returns the complete response text.
    """
    from core import thought_logger

    await nvidia_limiter.acquire()
    thinking(model_id)
    thought_logger.write_header(model_id, log_label)

    url, key, api_model = _route(model_id)

    if messages_override:
        messages = messages_override
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    # ── Pre-flight context-budget check ──────────────────────────────
    # Rough estimate: ~4 chars per token for English/code. If our prompt
    # is already over the model's typical input cap, fail loudly with a
    # clear message instead of letting the server return the cryptic
    # "requested 0 output tokens" HTTP 400 (which happens when the server
    # computes max_output = context_limit - input and gets 0 or negative).
    # The threshold is a soft hint — we'd rather warn early than truncate
    # silently and lose the model's work.
    _approx_input_chars = sum(len(m.get("content", "")) for m in messages)
    _approx_input_tokens = _approx_input_chars // 4
    # Most NVIDIA models we use have 200k-256k context. We reserve 8k
    # for output and warn at 90% of a conservative 200k input cap.
    _SOFT_INPUT_CAP = 190_000  # tokens
    if _approx_input_tokens > _SOFT_INPUT_CAP:
        from core.cli import warn as _warn
        _warn(
            f"  [{model_id.split('/')[-1]}] prompt is ~{_approx_input_tokens:,} "
            f"tokens — over the {_SOFT_INPUT_CAP:,} soft cap. The model may "
            f"refuse with HTTP 400 'requested 0 output tokens'. Consider "
            f"narrowing [KEEP:] ranges or splitting the step."
        )

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # Reserve a floor for output. Without this, when input nearly fills
        # the context the server computes max_output = 0 and returns the
        # opaque "requested 0 output tokens" error. With an explicit floor,
        # an overflowing request fails with a clear "context exceeded"
        # message we can surface and handle.
        "max_tokens": max(int(max_tokens), 4096),
        "stream": True,
        **_max_thinking_payload(model_id),
    }
    _apply_gptoss_pin(payload, url, api_model)   # bughunt #11: pin gpt-oss on the streaming
    # (JSON-ops) coder path too — it was unpinned, so OpenRouter could route it to any provider/quant.

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    chunks: list[str] = []
    # Visible-only buffer: stop_check must NEVER see reasoning_content. A
    # reasoning model that mentions "[STOP]" in its CoT would otherwise trigger
    # an early-stop while still thinking. Track visible content separately.
    visible_chunks: list[str] = []
    # Degeneration / prompt-leak guard. Aborts the stream as soon as the
    # model starts repeating or emits prompt-only scaffolding. Saves both
    # tokens and the next round's context (degenerate output here gets
    # echoed into YOUR WORK SO FAR otherwise).
    degen_guard = DegenerationDetector()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=http_timeout(url, payload),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {body[:200]}")

            buf = b""
            done = False
            in_thinking_block = False
            # Time-to-first-token / between-chunk idle cap — the uniform
            # STREAM_TTFT_TIMEOUT (10 min, config.py). NIM is the canonical
            # "hangs ~5 min then 504" provider; this watchdog turns that hang
            # into a clean stall error that core/retry.py fails over on (to the
            # NEXT model, not a re-queue of NIM). The aiohttp `total` is 1 hour
            # so it can't catch a dead connection; this between-chunk cap does.
            STREAM_IDLE_TIMEOUT = STREAM_TTFT_TIMEOUT
            while True:
                try:
                    raw = await asyncio.wait_for(
                        resp.content.readany(), timeout=STREAM_IDLE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Raise as a regular RuntimeError so retry.py treats it
                    # as a normal recoverable error (bounded retries +
                    # fallback). asyncio.TimeoutError would trigger the
                    # infinite-retry timeout path in retry.py — wrong for
                    # an idle stream that's likely a dead connection.
                    raise RuntimeError(
                        f"NVIDIA {api_model} stream idle "
                        f"{STREAM_IDLE_TIMEOUT:.0f}s — server stalled"
                    )
                if not raw:
                    break  # EOF
                buf += raw
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").rstrip("\r")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        obj = _json.loads(data)
                        delta_obj = obj["choices"][0]["delta"]
                        # ── Reasoning content (hidden CoT) ──
                        # Some servers use `reasoning`, others `reasoning_content`.
                        reasoning = (
                            delta_obj.get("reasoning_content")
                            or delta_obj.get("reasoning")
                            or ""
                        )
                        if reasoning:
                            if not in_thinking_block:
                                opener = "<think>"
                                chunks.append(opener)
                                thought_logger.write_chunk(model_id, opener)
                                in_thinking_block = True
                            chunks.append(reasoning)
                            thought_logger.write_chunk(model_id, reasoning)
                        # ── Visible content ──
                        delta = delta_obj.get("content") or ""
                        if delta:
                            if in_thinking_block:
                                closer = "</think>\n\n"
                                chunks.append(closer)
                                thought_logger.write_chunk(model_id, closer)
                                in_thinking_block = False
                            chunks.append(delta)
                            visible_chunks.append(delta)
                            thought_logger.write_chunk(model_id, delta)
                            if stop_check and ("]" in delta or "\n" in delta):
                                if stop_check("".join(visible_chunks)):
                                    done = True
                                    break
                            # Degeneration / prompt-leak guard — check on
                            # newline-bearing deltas so we re-scan when a
                            # line completes. Cheap when not tripped.
                            if "\n" in delta:
                                reason = degen_guard.check("".join(visible_chunks))
                                if reason:
                                    warn(
                                        f"  [{model_id.split('/')[-1]}] stream "
                                        f"aborted — {reason}"
                                    )
                                    done = True
                                    break
                    except (ValueError, KeyError, IndexError):
                        pass
                if done:
                    break
            if in_thinking_block:
                chunks.append("</think>\n\n")
                thought_logger.write_chunk(model_id, "</think>\n\n")

    return "".join(chunks)


async def call_nvidia_parallel(calls: list[dict]) -> list[str]:
    """
    Run multiple NVIDIA calls IN PARALLEL. All fire at once, rate limiter
    still enforces 40 RPM. At ~5 RPM real usage, this is totally safe.
    Returns list of response texts in same order as calls.
    """
    async def _one(c):
        return await call_nvidia(
            model_id=c["model_id"],
            prompt=c["prompt"],
            system=c.get("system", ""),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 4096),
            json_mode=c.get("json_mode", False),
        )

    return await asyncio.gather(*[_one(c) for c in calls])


async def call_nvidia_sequential(calls: list[dict], sleep: float = NVIDIA_SLEEP_BETWEEN) -> list[str]:
    """
    Run multiple NVIDIA calls sequentially (old method, kept as fallback).
    """
    results = []
    for i, c in enumerate(calls):
        result = await call_nvidia(
            model_id=c["model_id"],
            prompt=c["prompt"],
            system=c.get("system", ""),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 4096),
            json_mode=c.get("json_mode", False),
        )
        results.append(result)
        if i < len(calls) - 1:
            await asyncio.sleep(sleep)
    return results
