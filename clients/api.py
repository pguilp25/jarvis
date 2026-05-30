"""
Unified API — routes call to correct provider client.
"""
import os

from clients.groq import call_groq, call_groq_stream
from clients.nvidia import call_nvidia, call_nvidia_stream
from clients.gemini import call_gemini
from clients.openrouter import call_openrouter, call_openrouter_stream
from clients.openai_compat import call_openai_compat, call_openai_compat_stream, PROVIDERS as _OAI_PROVIDERS
from config import MODELS


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
