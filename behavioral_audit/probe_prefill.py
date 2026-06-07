#!/usr/bin/env python3
"""Probe: does <model> CONTINUE a content-prefill (trailing assistant turn) or RESTART/ERROR?

This decides whether the rolling-prefill design is viable per model, via the EXACT code path
the coder/planner use (call_nvidia_stream with messages_override + meta). $0 (free models), fast.

Verdict per model:
  CONTINUE  → the response continues the prefix (does NOT re-emit it) → prefill is usable
  RESTART   → the response re-lists from the start (ignores the prefill) → prefill NOT usable
  ERROR     → the request failed (provider rejects a trailing assistant) → prefill NOT usable
"""
import asyncio, sys
sys.path.insert(0, "/home/pguilp25/jarvis")
from clients.nvidia import call_nvidia_stream

# CONSISTENT incomplete prefill (the API returns continuation-ONLY, no echo — proven by the prior
# probe returning just 'Red/Blue/Yellow' without the prefill). So:
#   CONTINUE → visible is the SHORT completion ('lazy dog.') — starts with 'lazy', no 'quick'.
#   RESTART  → visible is the FULL sentence ('The quick brown fox ... lazy dog.') — contains 'quick'.
# This avoids the earlier flaw: a CONTRADICTORY prefill induces the model to CORRECT, not continue.
USER = ("Reply with EXACTLY this sentence and nothing else (no quotes): "
        "The quick brown fox jumps over the lazy dog.")
PREFILL = "The quick brown fox jumps over the"

MODELS = [
    "nvidia/gpt-oss-120b",        # the coder (pinned to DeepInfra@bf16)
    "openrouter/owl-alpha",       # planner + merger (LongCat family)
    "nvidia/nemotron-3-super-120b-a12b",
    "google/gemma-4-31b-it",
    "nvidia/qwen3-coder",
]


async def probe(model):
    msgs = [{"role": "user", "content": USER},
            {"role": "assistant", "content": PREFILL}]
    meta = {}
    try:
        out = await call_nvidia_stream(model, prompt="", system="", messages_override=msgs,
                                       max_tokens=150, meta=meta)
    except Exception as e:
        return model, "ERROR", f"{type(e).__name__}: {str(e)[:140]}", {}
    vis = (meta.get("visible") or "").strip()
    low = vis.lower()
    # CONTINUE → completion only ('lazy dog.'), does NOT contain 'quick'. RESTART → full sentence.
    has_quick = "quick" in low
    has_lazy = "lazy" in low
    if has_lazy and not has_quick:
        verdict = "CONTINUE"          # only the tail → the model appended to the prefill
    elif has_quick:
        verdict = "RESTART"          # re-emitted the whole sentence from the start
    else:
        verdict = "UNCLEAR"
    return model, verdict, vis[:140].replace("\n", " ⏎ "), meta


async def main():
    print(f"PREFILL ends with '3.' → CONTINUE should produce 'Earth' first; RESTART re-emits '1. Mercury'.\n")
    for m in MODELS:
        model, verdict, sample, meta = await probe(m)
        fr = meta.get("finish_reason", "?")
        print(f"=== {model}")
        print(f"    verdict: {verdict}   (finish_reason={fr})")
        print(f"    visible continuation: {sample!r}\n")


if __name__ == "__main__":
    asyncio.run(main())
