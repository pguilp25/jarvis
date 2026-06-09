"""
OpenRouter API client — OpenAI-compatible endpoint.
Used for free models like Nemotron Super.
"""

import json as _json
import os
import asyncio
import aiohttp
from core.http_timeout import http_timeout
from core.cli import thinking
from core import thought_logger
from config import STREAM_TTFT_TIMEOUT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model ID mapping
OPENROUTER_MODELS = {
    "openrouter/qwen3.6-plus": "qwen/qwen3.6-plus-preview:free",
}


def _resolve_and_pin(model_id: str, payload: dict) -> str:
    """Resolve the OpenRouter slug and apply the shared free-only / max_tokens pin.

    The PLANNER models (nemotron-3-ultra/super, gemma-4) are provider="openrouter" in config.py, so
    call_api_stream routes them HERE — NOT to clients/nvidia.py. This module's local OPENROUTER_MODELS
    only knew qwen, so it sent the BARE base slug (e.g. `nemotron-3-ultra-550b-a55b`) with NO `:free`
    suffix → OpenRouter served the PAID variant at 16384 max_tokens → the persistent HTTP 402
    "requires more credits, or fewer max_tokens". Reuse the nvidia client's :free mapping + its pin so
    the planners here get the SAME treatment: :free slugs, max_tokens capped to 4096 for :free,
    nemotron-3-ultra pinned allow_fallbacks=False (never billed), cheap planners allowed paid fallback.
    Never raises. (ckpt-227, 2026-06-08.)"""
    # gemma mitigation (ckpt-233, user 2026-06-08): the weak free planners (esp. gemma-4) sometimes
    # collapse into token-repetition degeneration ("same same sameL_use_use", "額額額…") with no output
    # cap — a mild frequency_penalty discourages the verbatim loop at the sampler level (the
    # DegenerationDetector still aborts the worst cases; this REDUCES how often they happen). Applied
    # to every planner call routed through this client (free + :paid). Conservative 0.3 — high enough
    # to break loops, low enough not to hurt plan quality. setdefault → never clobbers a caller value.
    payload.setdefault("frequency_penalty", 0.2)
    # ":paid" variant (ckpt-232, user 2026-06-08): the PAID fallback for a free planner that just
    # failed (stall/429). Strip the marker → use the BARE slug (no :free) so OpenRouter serves a PAID
    # provider; allow_fallbacks stays default-true. Cheap planners (gemma ~$0.04, super ~$0.02) use
    # this; nemotron-ultra never does (its fallback is owl-alpha, not ultra-paid).
    if model_id.endswith(":paid"):
        api_model = model_id[:-5]          # e.g. "google/gemma-4-31b-it:paid" → "google/gemma-4-31b-it"
        payload["model"] = api_model
        payload.pop("max_tokens", None)    # full context window, like the free path
        return api_model
    base = model_id.split("/", 1)[-1]
    api_model = OPENROUTER_MODELS.get(model_id)
    if not api_model:
        try:
            from clients.nvidia import OPENROUTER_MODELS as _NV
            api_model = _NV.get(base)
        except Exception:
            api_model = None
    api_model = api_model or base
    payload["model"] = api_model
    try:
        from clients.nvidia import _apply_gptoss_pin
        _apply_gptoss_pin(payload, OPENROUTER_URL, api_model)
    except Exception:
        pass
    # Every model routed through THIS client is a FREE planner (nemotron/gemma :free + owl-alpha
    # free-alpha, incl. the MERGER). User directive: free models run at MAX context window — drop our
    # output max_tokens so the provider grants max-available completion (a fixed cap reserves part of a
    # small free window and truncates / 402s). Free = $0, so no reservation/billing concern. (ckpt-228.)
    payload.pop("max_tokens", None)
    return api_model


def _get_key() -> str:
    # Share the round-robin key pool with clients/nvidia.py so both
    # OpenRouter accounts are used here too (doubles the :free budget and
    # spreads quota — see _openrouter_keys there for the full rationale).
    try:
        from clients.nvidia import _next_openrouter_key
        key = _next_openrouter_key()
    except Exception:
        key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


async def call_openrouter(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Call an OpenRouter model. Returns response text."""
    thinking(model_id)

    api_model = OPENROUTER_MODELS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    api_model = _resolve_and_pin(model_id, payload)   # :free slug + cap + free-only pin (ckpt-227)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    _key = _get_key()
    headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=http_timeout(OPENROUTER_URL, payload)) as resp:
            if resp.status != 200:
                body = await resp.text()
                if resp.status == 402:
                    try:
                        from clients.nvidia import _mark_or_key_dead
                        _mark_or_key_dead(_key, body)
                    except Exception:
                        pass
                raise RuntimeError(f"OpenRouter {resp.status}: {body[:300]}")
            data = await resp.json()
            try:   # bughunt ckpt-248: malformed 200 -> clean retryable error, not a raw KeyError/IndexError
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"OpenRouter: malformed 200 response "
                                   f"(no choices/message/content): {str(data)[:200]}")


async def call_openrouter_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    log_label: str = "",
    stop_check=None,
) -> str:
    """Stream an OpenRouter model response via SSE."""
    thinking(model_id)
    thought_logger.write_header(model_id, log_label)

    api_model = OPENROUTER_MODELS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    api_model = _resolve_and_pin(model_id, payload)   # :free slug + cap + free-only pin (ckpt-227)

    _key = _get_key()
    headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
    }

    full = ""
    # stop_check must only see VISIBLE content (no <think> block).
    visible = ""
    in_thinking = False
    done = False
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=http_timeout(OPENROUTER_URL, payload)) as resp:
            if resp.status != 200:
                body = await resp.text()
                if resp.status == 402:
                    try:
                        from clients.nvidia import _mark_or_key_dead
                        _mark_or_key_dead(_key, body)
                    except Exception:
                        pass
                raise RuntimeError(f"OpenRouter {resp.status}: {body[:300]}")

            buf = b""
            # Time-to-first-token / idle watchdog (uniform STREAM_TTFT_TIMEOUT,
            # 10 min). OpenRouter sends ': OPENROUTER PROCESSING' SSE heartbeats
            # while an upstream works, so this trips only on a genuine stall —
            # and the heartbeat bytes reset the timer (readany sees them). On a
            # stall we raise, and core/retry.py fails over to the NEXT model.
            while True:
                try:
                    raw = await asyncio.wait_for(
                        resp.content.readany(), timeout=STREAM_TTFT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"OpenRouter {api_model} stream idle "
                        f"{STREAM_TTFT_TIMEOUT:.0f}s — server stalled"
                    )
                if not raw:
                    break  # EOF
                buf += raw
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\r").strip()
                    if not line.startswith("data: "):
                        continue  # skips ': OPENROUTER PROCESSING' heartbeats + blanks
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        done = True
                        break
                    try:
                        data = _json.loads(data_str)
                    except _json.JSONDecodeError:
                        continue
                    # Mid-stream error: OpenRouter keeps HTTP 200 but delivers a
                    # top-level `error` object (e.g. an upstream 429/5xx). Raise
                    # so the retry layer fails over to the next model in the chain.
                    if isinstance(data, dict) and data.get("error"):
                        raise RuntimeError(
                            f"OpenRouter {api_model} stream error: "
                            f"{str(data['error'])[:200]}"
                        )
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning:
                        if not in_thinking:
                            full += "<think>"
                            thought_logger.write_chunk(model_id, "<think>")
                            in_thinking = True
                        full += reasoning
                        thought_logger.write_chunk(model_id, reasoning)
                    chunk = delta.get("content") or ""
                    if chunk:
                        if in_thinking:
                            full += "</think>\n\n"
                            thought_logger.write_chunk(model_id, "</think>\n\n")
                            in_thinking = False
                        full += chunk
                        visible += chunk
                        thought_logger.write_chunk(model_id, chunk)
                        if stop_check and ("]" in chunk or "\n" in chunk):
                            if stop_check(visible):
                                done = True
                                break
                if done:
                    break
            if in_thinking:
                full += "</think>\n\n"
                thought_logger.write_chunk(model_id, "</think>\n\n")

    return full
