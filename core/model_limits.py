"""
Per-model CONCURRENCY caps + RATE pacing for tight-limit free providers.

Two free-tier realities the plain fallback chain can't handle alone:

  • z.ai GLM-Flash is ~1 concurrent request — a 2nd in-flight call returns
    HTTP 429 code 1302 ("high concurrency").
  • Mistral's free Experiment tier is ~2 requests/min — a 3rd call inside a
    minute returns 429 rate_limited.

So:
  (a) we track IN-FLIGHT calls per model and treat a model AT its cap as
      "busy" — the router (core/retry.py) SKIPS a busy model and picks the
      next fallback instead of piling concurrent calls onto it; and
  (b) we pace low-rpm models with a REAL, VARIABLE delay — the committed
      caller waits only the remaining time to its next allowed slot, so the
      model is driven to its limit EXACTLY (2 rpm), never more.

asyncio is single-threaded and there is no await between the read and the
mutation of the counters below, so the bookkeeping is race-free across
concurrent tasks.
"""
import time
import asyncio

# Concurrency caps. A model at its cap is "busy" → concurrent callers skip it.
# Models not listed get _DEFAULT_CAP (NIM / OpenRouter tolerate many in-flight).
_CAP = {
    "zai/glm-4.7-flash": 1,            # z.ai free ≈ 1 concurrent (1302 otherwise)
    "zai/glm-4.5-flash": 1,
    "mistral/codestral": 1,            # Mistral free is rpm-limited → serialize
    "mistral/magistral": 1,
    "mistral/devstral":  1,
    "mistral/large":     1,
    "pollinations/minimax-m2.7": 2,    # hobby tier — keep concurrency low
    "pollinations/glm-5.1":      2,
    "pollinations/qwen-coder":   2,
}
_DEFAULT_CAP = 8

# Minimum seconds between the START of consecutive calls to a model (rate cap).
# Mistral free = 2 rpm → 30s spacing. The delay is VARIABLE: the committed
# caller waits only `interval - elapsed`, so it achieves 2 rpm, not less.
_MIN_INTERVAL = {
    "mistral/codestral": 30.0,
    "mistral/magistral": 30.0,
    "mistral/devstral":  30.0,
    "mistral/large":     30.0,
}

_inflight: dict[str, int] = {}
_last_start: dict[str, float] = {}


def _cap(model_id: str) -> int:
    return _CAP.get(model_id, _DEFAULT_CAP)


def rate_wait(model_id: str) -> float:
    """Seconds to wait before STARTING this model to honor its rate cap (0 if
    enough time has already elapsed since the last start)."""
    iv = _MIN_INTERVAL.get(model_id, 0.0)
    if iv <= 0:
        return 0.0
    return max(0.0, iv - (time.time() - _last_start.get(model_id, 0.0)))


def is_busy(model_id: str) -> bool:
    """True if a CONCURRENT caller should SKIP this model right now: it's at its
    concurrency cap, OR it would have to wait to honor its rate limit. The
    router prefers a not-busy fallback; only as a last resort does a caller
    actually commit to a busy model (and then `slot` waits for it)."""
    return _inflight.get(model_id, 0) >= _cap(model_id) or rate_wait(model_id) > 0.0


class slot:
    """Async context manager: reserve a concurrency slot for `model_id` and pace
    it to its rate limit. On enter, waits the (variable) remaining time to the
    next allowed slot, then marks one call in-flight; on exit, releases it."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    async def __aenter__(self):
        w = rate_wait(self.model_id)
        if w > 0:
            await asyncio.sleep(w)
        _inflight[self.model_id] = _inflight.get(self.model_id, 0) + 1
        _last_start[self.model_id] = time.time()
        return self

    async def __aexit__(self, *exc):
        _inflight[self.model_id] = max(0, _inflight.get(self.model_id, 0) - 1)
        return False
