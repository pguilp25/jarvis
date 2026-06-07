#!/usr/bin/env python3
"""Probe: does `continue_final_message=true` make gpt-oss (DeepInfra/vLLM via OpenRouter) CONTINUE
a trailing assistant prefill instead of RESTARTing?

vLLM's chat endpoint closes the trailing assistant turn by default (add_generation_prompt=True) →
the model starts a fresh turn → RESTART. `continue_final_message=true` + `add_generation_prompt=false`
leaves the turn OPEN → the model continues it. DeepInfra runs vLLM; OpenRouter may or may not forward
these extra params. This tests it EMPIRICALLY, controlling the full payload (raw aiohttp).

CONTINUE → visible is just the tail ('lazy dog.'), NO 'quick'.   RESTART → full sentence (has 'quick').
"""
import asyncio, json, os, sys
sys.path.insert(0, "/home/pguilp25/jarvis")
import aiohttp

URL = "https://openrouter.ai/api/v1/chat/completions"
SLUG = "openai/gpt-oss-120b"
PIN = {"order": ["DeepInfra"], "quantizations": ["bf16"], "allow_fallbacks": False}
USER = ("Reply with EXACTLY this sentence and nothing else (no quotes): "
        "The quick brown fox jumps over the lazy dog.")
PREFILL = "The quick brown fox jumps over the"


def _key():
    k = os.environ.get("OPENROUTER_API_KEY") or ""
    if not k:
        ks = os.environ.get("OPENROUTER_API_KEYS") or ""
        k = ks.split(",")[0].strip() if ks else ""
    return k


async def call(extra: dict, label: str):
    payload = {
        "model": SLUG,
        "messages": [{"role": "user", "content": USER},
                     {"role": "assistant", "content": PREFILL}],
        "provider": PIN,
        "temperature": 0.2,
        "max_tokens": 600,
        **extra,
    }
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as s:
        async with s.post(URL, json=payload, headers=headers, timeout=120) as r:
            raw = await r.text()
    try:
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        vis = (msg.get("content") or "").strip()
        rsn = (msg.get("reasoning") or msg.get("reasoning_content") or "")
        fr = data["choices"][0].get("finish_reason")
    except Exception as e:
        print(f"=== {label}: PARSE/HTTP ERROR {type(e).__name__}: {raw[:200]}")
        return
    low = vis.lower()
    verdict = ("CONTINUE" if ("lazy" in low and "quick" not in low)
               else "RESTART" if "quick" in low else "UNCLEAR")
    print(f"=== {label}")
    print(f"    verdict: {verdict}  (finish_reason={fr})")
    print(f"    visible: {vis[:120]!r}")
    if rsn:
        print(f"    reasoning[:120]: {rsn[:120]!r}")
    print()


async def main():
    if not _key():
        print("no OPENROUTER key in env"); return
    await call({}, "BASELINE (no continue param)")
    await call({"continue_final_message": True, "add_generation_prompt": False},
               "continue_final_message=true + add_generation_prompt=false")
    # also try the param alone (some servers want just one)
    await call({"continue_final_message": True}, "continue_final_message=true ONLY")


if __name__ == "__main__":
    asyncio.run(main())
