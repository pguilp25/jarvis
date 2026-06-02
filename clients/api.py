"""
Unified API — routes call to correct provider client.
"""
import os
import fcntl
import asyncio

from clients.groq import call_groq, call_groq_stream
from clients.nvidia import call_nvidia, call_nvidia_stream
from clients.gemini import call_gemini
from clients.openrouter import call_openrouter, call_openrouter_stream
from clients.openai_compat import call_openai_compat, call_openai_compat_stream, PROVIDERS as _OAI_PROVIDERS
from config import MODELS


# ── Night interleave: PLANNING-model serialization across two JARVIS processes ──
# The free planning pool (mistral/glm/minimax/deepseek/…) is concurrency-1, so two
# runs both planning at once 429 each other. This lock excludes the OTHER process
# during a planning call — but is REFCOUNTED per-process so the parallel Layer-1
# drafts WITHIN one run still share the single held lock (intra-run concurrency
# preserved; only cross-run is excluded). Pairs with the gpt-oss lock in nvidia.py
# so the two runs are mutually exclusive on BOTH phases: one plans while the other
# codes (different locks → overlap), but two-planning or two-coding serialize.
# ENV-GATED (JARVIS_INTERLEAVE) + FAIL-OPEN: default unset = pure no-op; any lock
# error or a 10-min stall → proceed unlocked, never deadlock a run.
_PLANNING_LOCK_PATH = "/tmp/jarvis_planning.lock"


class _RefcountFlock:
    def __init__(self, path):
        self.path = path
        self.n = 0
        self.fh = None
        self.guard = asyncio.Lock()

    async def __aenter__(self):
        if not os.environ.get("JARVIS_INTERLEAVE"):
            return self
        async with self.guard:
            self.n += 1
            if self.n == 1:                     # first holder acquires the cross-process flock
                try:
                    fh = open(self.path, "w")
                    waited = 0.0
                    while True:
                        try:
                            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            await asyncio.sleep(0.25)
                            waited += 0.25
                            if waited > 600:     # safety valve: never block planning > 10 min
                                break
                    self.fh = fh
                except Exception:
                    self.fh = None               # fail-open
        return self

    async def __aexit__(self, *exc):
        if not os.environ.get("JARVIS_INTERLEAVE"):
            return
        async with self.guard:
            self.n -= 1
            if self.n <= 0:
                self.n = 0
                if self.fh is not None:
                    try:
                        fcntl.flock(self.fh, fcntl.LOCK_UN)
                        self.fh.close()
                    except Exception:
                        pass
                    self.fh = None


# gpt-oss has its own per-call lock in nvidia.py and is NATIVE (never routed through
# call_api*), so everything reaching this module is planning/review class — wrap it all.
_PLANNING = _RefcountFlock(_PLANNING_LOCK_PATH)


async def call_api(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    Call any model by its config name (e.g. 'groq/kimi-k2', 'nvidia/deepseek-v4-pro').
    Routes to the correct provider client automatically.
    """
    provider = MODELS[model_id]["provider"]

    async with _PLANNING:   # night interleave: serialize planning across runs (no-op unless JARVIS_INTERLEAVE)
        if provider == "groq":
            return await call_groq(model_id, prompt, system, temperature, max_tokens, json_mode)
        elif provider == "nvidia":
            return await call_nvidia(model_id, prompt, system, temperature, max_tokens, json_mode)
        elif provider == "gemini":
            return await call_gemini(model_id, prompt, system, temperature, max_tokens)
        elif provider == "openrouter":
            return await call_openrouter(model_id, prompt, system, temperature, max_tokens, json_mode)
        elif provider in _OAI_PROVIDERS:  # cerebras / zai / mistral / pollinations
            return await call_openai_compat(model_id, prompt, system, temperature, max_tokens, json_mode)
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'")


async def call_api_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Stream any model's response to the thought_logger.
    NVIDIA and Groq: true SSE streaming (chunks written as they arrive).
    Gemini: full response written to log after completion (no SSE endpoint).
    If stop_check(accumulated_text) returns True, stops the stream early.
    Returns the complete response text.
    """
    from core import thought_logger

    # TEST HOOK (inert unless set): JARVIS_FORCE_FAIL_MODELS=comma,substrings makes
    # any matching model raise a permanent error — used to FORCE the coder fallback
    # chain to walk through each link under controlled conditions. Zero prod impact.
    _ff = os.environ.get("JARVIS_FORCE_FAIL_MODELS", "")
    if _ff and any(s and s in model_id for s in _ff.split(",")):
        raise RuntimeError(f"HTTP 404: forced test failure for {model_id} (JARVIS_FORCE_FAIL_MODELS)")

    provider = MODELS[model_id]["provider"]

    async with _PLANNING:   # night interleave: serialize planning across runs (no-op unless JARVIS_INTERLEAVE)
        if provider == "groq":
            return await call_groq_stream(
                model_id, prompt, system, temperature, max_tokens, json_mode, log_label,
                stop_check=stop_check,
            )
        elif provider == "nvidia":
            return await call_nvidia_stream(
                model_id, prompt, system, temperature, max_tokens, log_label,
                stop_check=stop_check,
            )
        elif provider == "gemini":
            # Gemini uses a non-SSE REST API — write the full response to the log
            result = await call_gemini(model_id, prompt, system, temperature, max_tokens)
            thought_logger.write_header(model_id, log_label)
            thought_logger.write_chunk(model_id, result)
            return result
        elif provider == "openrouter":
            return await call_openrouter_stream(
                model_id, prompt, system, temperature, max_tokens, log_label,
                stop_check=stop_check,
            )
        elif provider in _OAI_PROVIDERS:  # cerebras / zai / mistral / pollinations
            return await call_openai_compat_stream(
                model_id, prompt, system, temperature, max_tokens, log_label,
                stop_check=stop_check,
            )
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'")
