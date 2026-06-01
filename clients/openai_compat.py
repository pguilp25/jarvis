"""
Generic OpenAI-compatible streaming client. ONE implementation for every
provider whose API is OpenAI `/v1/chat/completions`-shaped — they differ only
in base URL, auth key env var(s), and model-id slugs, captured in PROVIDERS.

Currently registered: Cerebras, z.ai (Zhipu GLM-Flash), Mistral (Codestral /
Devstral), Pollinations (anonymous free tier — no key).

Every provider here inherits the uniform 10-min time-to-first-token watchdog
(config.STREAM_TTFT_TIMEOUT): a provider that accepts the connection (HTTP 200)
but emits no token within the window raises a "stream idle … stalled" error
that core/retry.py fails over on (to the NEXT model in the chain, not a
re-queue). Fast providers reject with an immediate 429 at the HTTP level, so
they fail over in milliseconds — the watchdog only governs genuine stalls.
"""

import json as _json
import os
import asyncio
import aiohttp
from config import STREAM_TTFT_TIMEOUT
from core.cli import thinking, warn
from core import thought_logger
from core.stream_guard import DegenerationDetector


# provider key → endpoint, auth env var(s) (first one set wins; () = anonymous),
# and config-model-id → provider-API-slug map.
PROVIDERS = {
    "zai": {
        "url": "https://api.z.ai/api/paas/v4/chat/completions",
        "key_env": ("ZAI_API_KEY", "ZHIPU_API_KEY"),
        "models": {
            "zai/glm-4.7-flash": "glm-4.7-flash",
            "zai/glm-4.5-flash": "glm-4.5-flash",
        },
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key_env": ("MISTRAL_API_KEY",),
        "models": {
            "mistral/codestral": "codestral-latest",        # dedicated free coder
            "mistral/devstral":  "devstral-small-latest",
            "mistral/medium":    "mistral-medium-latest",   # current flagship (newer/better than large; replaced the non-existent magistral)
        },
    },
    "pollinations": {
        # The full roster (deepseek/minimax/glm/qwen-coder) lives on the newer
        # gen.pollinations.ai endpoint, which now requires a FREE key (sign up
        # at enter.pollinations.ai). The old anon text.pollinations.ai/openai
        # endpoint was gutted to a single model. Without the key these links
        # 401 → core/retry.py skips to the next model in the chain.
        "url": "https://gen.pollinations.ai/v1/chat/completions",
        "key_env": ("POLLINATIONS_API_KEY",),
        "models": {
            # NOTE: pollinations "deepseek" is PAID (402 insufficient balance);
            # minimax / glm / qwen-coder are free (HTTP 200).
            "pollinations/minimax-m2.7":      "minimax",
            "pollinations/glm-5.1":           "glm",
            "pollinations/qwen-coder":        "qwen-coder",
        },
    },
}


def _provider_of(model_id: str) -> str:
    return model_id.split("/", 1)[0]


def _resolve(model_id: str) -> tuple[str, str, str]:
    """Return (url, key, api_model) for a config model_id. Raises a recognizable
    'API key not set' error for a provider that needs a key but has none — so
    core/retry.py fails over to the next model in the chain instantly (a model
    whose provider key isn't configured is simply skipped, no wasted attempt)."""
    prov = _provider_of(model_id)
    cfg = PROVIDERS[prov]
    key = ""
    for env in cfg["key_env"]:
        key = os.environ.get(env, "").strip()
        if key:
            break
    if cfg["key_env"] and not key:
        raise RuntimeError(
            f"{prov} API key not set ({' / '.join(cfg['key_env'])}) — skipping"
        )
    api_model = cfg["models"].get(model_id, model_id.split("/", 1)[-1])
    return cfg["url"], key, api_model


def _headers(key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _thinking_payload(model_id: str) -> dict:
    """Per-provider params that ENABLE reasoning/thinking in the request. Without
    these a reasoning model may not surface its chain-of-thought (z.ai GLM needs
    the explicit toggle; gpt-oss honors reasoning_effort). Magistral and the
    Cerebras GLM reason by default and reject unknown params, so they get none.
    Mirrors clients/nvidia._max_thinking_payload for the OpenAI-compat providers."""
    prov = _provider_of(model_id)
    base = model_id.split("/", 1)[-1].lower()
    if prov == "zai":                       # GLM native reasoning toggle (confirmed)
        return {"thinking": {"type": "enabled"}}
    if "gpt-oss" in base:                   # gpt-oss reasoning effort (Cerebras)
        return {"reasoning_effort": "high"}
    return {}                               # magistral / cerebras-glm / pollinations: default-on


def _flatten_content(content) -> str:
    """Mistral reasoning models return `content` as a LIST of typed parts
    (e.g. {"type":"text","text":...}) instead of a plain string. Flatten to the
    visible text; other providers pass a string through unchanged."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", None):
                out.append(part.get("text") or "")
        return "".join(out)
    return ""


def _extract_delta(delta: dict) -> tuple[str, str]:
    """Return (reasoning_text, visible_text) from a streaming delta, handling:
    (a) reasoning in `reasoning_content`/`reasoning` (z.ai GLM, DeepSeek, gpt-oss),
    (b) Mistral's structured LIST content where reasoning arrives as
        {"type":"thinking","thinking":[{"type":"text","text":...}]} and visible
        text as {"type":"text","text":...}, and
    (c) plain string content."""
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    visible = ""
    content = delta.get("content")
    if isinstance(content, str):
        visible = content
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "thinking":
                th = part.get("thinking")
                if isinstance(th, list):
                    reasoning += "".join(
                        p.get("text", "") for p in th if isinstance(p, dict)
                    )
                elif isinstance(th, str):
                    reasoning += th
            elif ptype in ("text", None):
                visible += part.get("text") or ""
    return reasoning, visible


async def call_openai_compat(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Non-streaming call (used by clients.api.call_api)."""
    thinking(model_id)
    url, key, api_model = _resolve(model_id)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max(int(max_tokens), 1024),
        **_thinking_payload(model_id),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=_headers(key),
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"{_provider_of(model_id)} {api_model} HTTP {resp.status}: {body[:200]}"
                )
            data = await resp.json()
            return _flatten_content(data["choices"][0]["message"].get("content"))


async def call_openai_compat_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    log_label: str = "",
    stop_check=None,
) -> str:
    """SSE-streaming call with the uniform 10-min TTFT/idle watchdog. Mirrors
    the NVIDIA/OpenRouter stream clients (reasoning_content + content handling,
    early stop_check, degeneration guard, mid-stream error detection)."""
    thinking(model_id)
    thought_logger.write_header(model_id, log_label)
    url, key, api_model = _resolve(model_id)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max(int(max_tokens), 1024),
        "stream": True,
        **_thinking_payload(model_id),   # enable reasoning in the request
    }

    full, visible = "", ""
    in_thinking = False
    done = False
    degen = DegenerationDetector()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=_headers(key),
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"{_provider_of(model_id)} {api_model} HTTP {resp.status}: {body[:200]}"
                )
            buf = b""
            while True:
                try:
                    raw = await asyncio.wait_for(
                        resp.content.readany(), timeout=STREAM_TTFT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"{_provider_of(model_id)} {api_model} stream idle "
                        f"{STREAM_TTFT_TIMEOUT:.0f}s — server stalled"
                    )
                if not raw:
                    break  # EOF
                buf += raw
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\r").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        done = True
                        break
                    try:
                        data = _json.loads(data_str)
                    except _json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and data.get("error"):
                        raise RuntimeError(
                            f"{_provider_of(model_id)} {api_model} stream error: "
                            f"{str(data['error'])[:200]}"
                        )
                    try:
                        delta = data["choices"][0]["delta"]
                    except (KeyError, IndexError, TypeError):
                        continue
                    reasoning, chunk = _extract_delta(delta)
                    if reasoning:
                        if not in_thinking:
                            full += "<think>"
                            thought_logger.write_chunk(model_id, "<think>")
                            in_thinking = True
                        full += reasoning
                        thought_logger.write_chunk(model_id, reasoning)
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
                        if "\n" in chunk:
                            reason = degen.check(visible)
                            if reason:
                                warn(f"  [{model_id.split('/')[-1]}] stream aborted — {reason}")
                                done = True
                                break
                if done:
                    break
            if in_thinking:
                full += "</think>\n\n"
                thought_logger.write_chunk(model_id, "</think>\n\n")

    return full
