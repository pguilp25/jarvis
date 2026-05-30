"""
JARVIS Configuration — Models, reserves, pairs, fallbacks, budget.
"""

# ─── Token Reserves by Task Type ─────────────────────────────────────────────

RESERVES = {
    "simple":  {"think": 0,     "output": 8_000,  "total": 8_000},
    "medium":  {"think": 15_000, "output": 8_000,  "total": 23_000},
    "hard":    {"think": 40_000, "output": 16_000, "total": 56_000},
    "extreme": {"think": 60_000, "output": 16_000, "total": 76_000},
}

# ─── Model Definitions ───────────────────────────────────────────────────────

MODELS = {
    # Groq (free, TPM-limited)
    "groq/llama-3.1-8b":     {"window": 128_000, "tpm": 6_000,  "provider": "groq"},
    "groq/qwen3-32b":        {"window": 32_000,  "tpm": 6_000,  "provider": "groq"},
    "groq/gpt-oss-120b":     {"window": 131_000, "tpm": 8_000,  "provider": "groq"},
    "groq/llama-3.3-70b":    {"window": 128_000, "tpm": 12_000, "provider": "groq"},
    "groq/llama-4-scout":    {"window": 128_000, "tpm": 30_000, "provider": "groq"},

    # NVIDIA (free, 40 RPM shared, no TPM limit)
    "nvidia/deepseek-v4-pro":   {"window": 1_000_000, "tpm": None, "provider": "nvidia"},
    "nvidia/deepseek-v4-flash": {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/kimi-k2.6":         {"window": 256_000, "tpm": None, "provider": "nvidia"},
    # minimax-m2.5 — actually routed to OpenRouter (:free) via
    # OPENROUTER_FORCED. The "nvidia/" prefix is kept for callsite
    # consistency with the rest of the planner pool.
    "nvidia/minimax-m2.5":      {"window": 200_000, "tpm": None, "provider": "nvidia"},
    "nvidia/glm-5":             {"window": 200_000, "tpm": None, "provider": "nvidia"},
    "nvidia/glm-5.1":           {"window": 200_000, "tpm": None, "provider": "nvidia"},
    # nemotron-super kept as a fallback target only — active workflows now
    # route to nvidia/kimi-k2.6 instead. If it's deprecated server-side, the
    # 410 fast-fail in core/retry.py routes downstream automatically.
    "nvidia/nemotron-super": {"window": 1_000_000, "tpm": None, "provider": "nvidia"},
    "nvidia/ultralong-8b":   {"window": 4_000_000, "tpm": None, "provider": "nvidia"},
    # minimax-m2.7 on NIM (NOT in OPENROUTER_FORCED → routes to NVIDIA NIM,
    # unlike minimax-m2.5 which is forced to OR :free). Planner fallback link.
    "nvidia/minimax-m2.7":   {"window": 200_000, "tpm": None, "provider": "nvidia"},
    # gpt-oss-120b on NIM + qwen3-coder on NIM — deep cross-provider fallback
    # targets for the most-redundant backbone models.
    "nvidia/gpt-oss-120b":   {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/gpt-oss-nim":    {"window": 128_000, "tpm": None, "provider": "nvidia"},  # gpt-oss on NIM (coder chain slot 4, native)
    "nvidia/qwen3-coder":    {"window": 256_000, "tpm": None, "provider": "nvidia"},

    # z.ai / Zhipu (free GLM-Flash tier — the realistic free GLM)
    "zai/glm-4.7-flash":     {"window": 200_000, "tpm": None, "provider": "zai"},
    "zai/glm-4.5-flash":     {"window": 128_000, "tpm": None, "provider": "zai"},

    # Mistral La Plateforme (free Experiment tier — Codestral = dedicated coder)
    "mistral/codestral":     {"window": 256_000, "tpm": None, "provider": "mistral"},
    "mistral/devstral":      {"window": 128_000, "tpm": None, "provider": "mistral"},
    "mistral/magistral":     {"window": 128_000, "tpm": None, "provider": "mistral"},  # reasoning → planner
    "mistral/large":         {"window": 128_000, "tpm": None, "provider": "mistral"},

    # Pollinations (anonymous free tier — no key)
    "pollinations/minimax-m2.7":      {"window": 200_000, "tpm": None, "provider": "pollinations"},
    "pollinations/glm-5.1":           {"window": 200_000, "tpm": None, "provider": "pollinations"},
    "pollinations/qwen-coder":        {"window": 128_000, "tpm": None, "provider": "pollinations"},

    # Gemini (free Flash Lite for utility, paid Pro for tiebreakers)
    "gemini/flash-lite":     {"window": 1_000_000, "tpm": None, "provider": "gemini"},
    "gemini/3.1-flash-lite": {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.0, "cost_per_1k_out": 0.0},
    "gemini/2.5-pro":        {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.00125, "cost_per_1k_out": 0.01},
    "gemini/3.1-pro":        {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.002, "cost_per_1k_out": 0.012},

    # OpenRouter (free tier)
    "openrouter/qwen3.6-plus": {"window": 1_000_000, "tpm": None, "provider": "openrouter"},
}

# ─── Groq Model ID Mapping (config name → API model string) ─────────────────

GROQ_MODEL_IDS = {
    "groq/llama-3.1-8b":  "llama-3.1-8b-instant",
    "groq/qwen3-32b":     "qwen/qwen-3-32b",
    "groq/gpt-oss-120b":  "openai/gpt-oss-120b",
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}

# ─── NVIDIA Model ID Mapping ────────────────────────────────────────────────
# NOTE: the deepseek-v4-flash API id is my best guess at the NVIDIA NIM
# slug — verify against build.nvidia.com if requests come back with
# "model not found" and update this single line.

NVIDIA_MODEL_IDS = {
    "nvidia/deepseek-v4-pro":   "deepseek-ai/deepseek-v4-pro",
    "nvidia/deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "nvidia/kimi-k2.6":         "moonshotai/kimi-k2.6",
    # NVIDIA retired z-ai/glm5 on 2026-05-18T00:00Z. New ID is z-ai/glm-5.1.
    "nvidia/glm-5":             "z-ai/glm-5.1",
    "nvidia/glm-5.1":           "z-ai/glm-5.1",
    # minimax-m2.5: not hosted on NIM at all — OPENROUTER_FORCED routes it to
    # OR :free. NIM_MODEL_IDS entry kept for callsite consistency.
    "nvidia/minimax-m2.5":      "minimax/minimax-m2.5",
    "nvidia/minimax-m2.7":      "minimaxai/minimax-m2.7",
    "nvidia/nemotron-super":    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/ultralong-8b":      "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct",
    "nvidia/gpt-oss-120b":      "openai/gpt-oss-120b",
    "nvidia/gpt-oss-nim":       "openai/gpt-oss-120b",  # NOT in OPENROUTER_FORCED → routes to NIM
    "nvidia/qwen3-coder":       "qwen/qwen3-coder-480b-a35b-instruct",
}

# ─── Priority Order per Role ─────────────────────────────────────────────────

PRIORITY_ORDER = {
    "decorticator":  ["nvidia/deepseek-v4-pro", "nvidia/glm-5.1", "nvidia/deepseek-v4-flash"],
    "fast_chat":     ["zai/glm-4.7-flash", "nvidia/minimax-m2.5", "nvidia/glm-5.1"],
    "synthesizer":   ["zai/glm-4.7-flash", "nvidia/kimi-k2.6", "nvidia/glm-5.1", "nvidia/deepseek-v4-flash"],
    "verifier":      ["nvidia/glm-5.1", "zai/glm-4.7-flash", "nvidia/deepseek-v4-flash"],
    "search_exec":   ["zai/glm-4.7-flash"],
    "self_eval":     ["zai/glm-4.7-flash", "nvidia/minimax-m2.5", "nvidia/deepseek-v4-pro"],
    "plan_compare":  ["nvidia/glm-5.1"],
    "formatter":     ["nvidia/kimi-k2.6"],
}

# ─── Domain-Matched Pairs ────────────────────────────────────────────────────
# Each domain → two distinct models. Where the previous pair had two slots
# that would now both be glm-5.1 (code/arduino) we substitute the second
# slot with deepseek-v4-flash to preserve diversity.

BEST_PAIRS = {
    "math":    ("nvidia/deepseek-v4-pro", "nvidia/deepseek-v4-flash"),
    "code":    ("nvidia/glm-5.1",         "nvidia/deepseek-v4-flash"),
    "science": ("nvidia/deepseek-v4-flash", "nvidia/deepseek-v4-pro"),
    "cfd":     ("nvidia/deepseek-v4-pro", "nvidia/glm-5.1"),
    "arduino": ("nvidia/glm-5.1",         "nvidia/deepseek-v4-flash"),
    "web":     ("nvidia/glm-5.1",         "nvidia/deepseek-v4-flash"),
    "general": ("nvidia/deepseek-v4-pro", "nvidia/glm-5.1"),
}

# ─── Fallback Maps ───────────────────────────────────────────────────────────

NVIDIA_FALLBACKS = {
    # Each value is a TUPLE — core/retry.py tries them in order. A CIRCUIT
    # BREAKER there remembers a model that just failed (429/5xx/stall) and skips
    # it for a cooldown, so the chain is NOT re-walked into the same dead
    # endpoint on every call (the planner fires dozens of calls).
    #
    # ORDER = OR :free FIRST (fails fast in ~0.2s; the breaker remembers a storm
    # and skips it next call) → reliable BIG-CONTEXT providers (z.ai GLM-4.7-Flash,
    # Mistral, Pollinations) → NIM LAST (NIM hangs ~5 min on overload). NO Groq /
    # Cerebras (too little context). OR :free = nvidia/deepseek-v4-flash &
    # nvidia/minimax-m2.5 (FORCED to OpenRouter); all other nvidia/* = NIM.

    # ── PLANNER lead (zai/glm-4.7-flash) — OR :free first → reliable → NIM last ──
    "zai/glm-4.7-flash": (
        "nvidia/deepseek-v4-flash", "nvidia/minimax-m2.5",     # OR :free (fast-fail)
        "mistral/magistral", "pollinations/minimax-m2.7",
        "nvidia/glm-5.1", "nvidia/kimi-k2.6",                  # NIM — last
    ),
    "mistral/magistral": (
        "nvidia/minimax-m2.5", "nvidia/deepseek-v4-flash",     # OR :free
        "zai/glm-4.7-flash", "pollinations/minimax-m2.7",
        "nvidia/glm-5.1", "nvidia/kimi-k2.6",                  # NIM — last
    ),

    # ── OR :free models (forced to OpenRouter) ──
    "nvidia/minimax-m2.5": (
        "nvidia/deepseek-v4-flash",                            # other OR
        "mistral/magistral", "zai/glm-4.7-flash", "pollinations/minimax-m2.7",
        "nvidia/minimax-m2.7", "nvidia/kimi-k2.6",             # NIM — last
    ),
    "nvidia/deepseek-v4-flash": (
        "nvidia/minimax-m2.5",                                 # other OR
        "mistral/codestral", "zai/glm-4.7-flash", "pollinations/qwen-coder",
        "nvidia/glm-5.1", "nvidia/deepseek-v4-pro",            # NIM — last
    ),

    # ── CODER primary + reviewer + understand (nvidia/glm-5.1) — coding-
    #    capable: OR :free (fast-fail) → reliable → NIM last ──
    "nvidia/glm-5.1": (
        "nvidia/deepseek-v4-flash", "nvidia/minimax-m2.5",     # OR :free
        "mistral/codestral", "zai/glm-4.7-flash", "pollinations/qwen-coder",
        "nvidia/deepseek-v4-pro", "nvidia/qwen3-coder",        # NIM — last
    ),
    "nvidia/glm-5": (                                          # legacy alias
        "nvidia/glm-5.1", "nvidia/deepseek-v4-flash", "mistral/codestral",
        "zai/glm-4.7-flash",
    ),

    # ── deep NIM fallback targets (reached rarely; OR/reliable first, NIM last) ──
    "nvidia/minimax-m2.7": (
        "nvidia/minimax-m2.5", "mistral/magistral", "zai/glm-4.7-flash",
        "pollinations/minimax-m2.7", "nvidia/kimi-k2.6",
    ),
    "nvidia/kimi-k2.6": (
        "nvidia/minimax-m2.5", "mistral/magistral", "zai/glm-4.7-flash",
        "pollinations/minimax-m2.7", "nvidia/minimax-m2.7",
    ),
    "nvidia/deepseek-v4-pro": (
        "nvidia/deepseek-v4-flash", "mistral/codestral", "zai/glm-4.7-flash",
        "pollinations/qwen-coder", "nvidia/glm-5.1",
    ),
    "nvidia/qwen3-coder": (
        # Text-coder fallback within call_with_retry. The EXACT coder order
        # (gpt-OR → qwen → mistral → gpt-NIM(native) → glm) is orchestrated in code
        # (_implement_one_step), since gpt-NIM must run the NATIVE loop, not text —
        # so it is NOT in this text chain. mistral/large → glm-5.1 only.
        "mistral/large", "nvidia/glm-5.1",
    ),
    "nvidia/nemotron-super": (
        "nvidia/minimax-m2.5", "zai/glm-4.7-flash", "mistral/magistral", "nvidia/glm-5.1",
    ),
    # MERGER model (2026-05-28): mistral/large is now the merger primary — it's a
    # flagship (128K ctx, no 8K throttle) that holds a structured plan better than
    # glm-5.1 did under load (glm degraded to empty/salvaged plans on the heavier
    # merger prompt → django/pylint regressions). glm-5.1 is the fallback, then
    # the off-NIM coders. (User-chosen 2026-05-28.)
    "mistral/large": (
        "nvidia/glm-5.1", "zai/glm-4.7-flash", "nvidia/deepseek-v4-flash",
        "nvidia/minimax-m2.5", "mistral/codestral",
    ),
    "nvidia/gpt-oss-120b": (
        # Text-path fallback for gpt-oss (rare; the native coder path orchestrates
        # the real chain in code). qwen3-coder → mistral/large → glm-5.1.
        "nvidia/qwen3-coder", "mistral/large", "nvidia/glm-5.1",
    ),
    # gpt-oss on NVIDIA NIM (coder chain slot 4, run NATIVE via code). If it's ever
    # reached through call_with_retry, fall to glm-5.1.
    "nvidia/gpt-oss-nim": ("nvidia/glm-5.1",),

    # ── reliable-provider primaries' onward chains ──
    "mistral/codestral": (
        "nvidia/deepseek-v4-flash", "zai/glm-4.7-flash", "pollinations/qwen-coder",
        "mistral/devstral", "nvidia/glm-5.1", "nvidia/qwen3-coder",
    ),
    "pollinations/minimax-m2.7": (
        "nvidia/minimax-m2.5", "mistral/magistral", "zai/glm-4.7-flash",
        "nvidia/minimax-m2.7", "nvidia/kimi-k2.6",
    ),
    "zai/glm-4.5-flash": (
        "zai/glm-4.7-flash", "nvidia/deepseek-v4-flash", "mistral/magistral", "nvidia/glm-5.1",
    ),
    "pollinations/glm-5.1": (
        "zai/glm-4.7-flash", "nvidia/deepseek-v4-flash", "mistral/codestral", "nvidia/glm-5.1",
    ),
    "pollinations/qwen-coder": (
        "nvidia/deepseek-v4-flash", "mistral/codestral", "zai/glm-4.7-flash", "nvidia/qwen3-coder",
    ),
}

GROQ_FALLBACKS = {}   # Groq removed (low context); kept as {} so retry.py import is stable.

# ─── Compression ─────────────────────────────────────────────────────────────

COMPRESS_THRESHOLD = 72_000
COMPRESS_TARGET = 50_000

# ─── Budget ──────────────────────────────────────────────────────────────────

MONTHLY_BUDGET = 45.0

# ─── NVIDIA Rate Limit ───────────────────────────────────────────────────────

NVIDIA_MAX_RPM = 40
NVIDIA_SLEEP_BETWEEN = 1.6  # seconds between sequential NVIDIA calls

# ─── Streaming watchdog (time-to-first-token / idle) ─────────────────────────
# ONE uniform cap for EVERY streaming client. A provider that accepts the
# connection (HTTP 200) but emits no token within this window is treated as
# stalled, and the retry layer fails over to the NEXT model in the chain — it
# does NOT re-queue the same endpoint (that would just re-enter the same queue).
#
# Deliberately 10 min, applied uniformly: fast providers reject with an
# immediate 429/503 at the HTTP level (handled BEFORE this watchdog ever
# applies, so they fail over in milliseconds), while slow/queueing providers —
# which are placed LAST in every chain — get a full 10-min shot at reaching the
# front of their queue instead of us bailing after 30s and losing our place.
# Generation can run past this; the watchdog only measures the gap BETWEEN
# tokens (and the wait for the first one).
STREAM_TTFT_TIMEOUT = 600.0

# ─── Abort Signals ───────────────────────────────────────────────────────────

ABORT_SIGNALS = ["stop", "cancel", "abort", "nevermind", "start over", "scratch that"]

# ─── Override Prefixes ───────────────────────────────────────────────────────

OVERRIDE_MAP = {
    "!!simple":     2,
    "!!medium":     5,
    "!!hard":       10,
    "!!deep":       99,  # Special: routes to deep thinking mode
    "!!conjecture": 99,
    "!!compute":    99,
    "!!prove":      99,
}
