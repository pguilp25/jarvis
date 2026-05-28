"""
Retry wrapper — timeout, retry with backoff, automatic fallback.
v5: connectivity check before API calls.

Timeout policy:
  - Timeout errors: infinite retries with exponential backoff (10s, 20s, 40s … capped at 5min)
  - Other errors: up to max_retries, then fallback model
  - PERMANENT errors (HTTP 410 Gone, 404 Not Found, 401/403 auth): skip
    retries entirely, go straight to fallback. Retrying a deprecated
    model just wastes 6+ seconds of backoff per call — observed in
    practice when MiniMax 2.5 was sunset on NVIDIA NIM.
"""

import re
import time
import asyncio
from clients.api import call_api, call_api_stream
from config import NVIDIA_FALLBACKS, GROQ_FALLBACKS, MODELS
from core.cli import status, warn, error
from core import model_limits as _ml

# Errors that CANNOT recover with a retry. Detected from the exception
# message. We match on the literal HTTP status prefix the API clients
# include, e.g. "HTTP 410:".
_PERMANENT_STATUS = re.compile(r'HTTP\s*(?:410|404|401|403)\b', re.IGNORECASE)


def _is_permanent_error(exc: BaseException) -> bool:
    """True if `exc` represents a failure that retrying THIS model won't fix —
    skip the backoff and go straight to the fallback chain.

    HTTP 410 (Gone) = model was deprecated/removed.
    HTTP 404 = model name not recognised by the provider.
    HTTP 401/403 = auth wrong — won't get better with a sleep.
    HTTP 429 on a `:free` model = OpenRouter free-tier DAILY quota exhausted —
      persistent, not a transient rate-limit; retrying 2s/4s just wastes time
      on every planner/coder call. Fall straight back to the NIM model.
    """
    s = str(exc)
    if _PERMANENT_STATUS.search(s):
        return True
    if "429" in s and ":free" in s:
        return True
    return False


# Errors meaning "this endpoint can't serve right NOW — go straight to the
# next model in the fallback chain instead of retrying the SAME model." The
# fallback chain is the redundancy; retrying the same endpoint on a capacity
# rejection (429/5xx) just thrashes it, and retrying after the 10-min TTFT
# stall just re-enters the SAME queue for another 10 min. Either way we want
# the NEXT provider, not a re-queue.
#   - stall: our uniform STREAM_TTFT_TIMEOUT elapsed with no token (see
#     STREAM_TTFT_TIMEOUT in config.py; clients raise "stream idle … stalled").
#   - capacity: 429 (rate-limit), 5xx (overload/gateway), or the textual
#     "rate-limited upstream" / "Provider returned error" / "overloaded".
# Deliberately EXCLUDES HTTP 402 — that's the OpenRouter free-tier per-account
# quota wall, where the 2-key round-robin retry (clients/nvidia.py) lands the
# next attempt on the OTHER account and keeps the SAME model. Let that retry.
_FAILOVER_NOW = re.compile(
    r'stream idle|server stalled|no first token|'
    r'HTTP\s*(?:429|500|502|503|504)\b|'
    r'temporarily rate-limited|provider returned error|'
    r'overloaded|out of capacity|capacity exceeded|'
    r'api key not set|'   # provider key absent → skip this model, try the next
    r'insufficient balance',  # Pollinations 402 (paid model, no Pollen) → skip
    re.IGNORECASE,
)


def _is_failover_now_error(exc: BaseException) -> bool:
    """True if retrying THIS endpoint is pointless right now — fail over to the
    next model in the chain immediately (busy/overloaded/stalled)."""
    return bool(_FAILOVER_NOW.search(str(exc)))


# ── Circuit breaker ──────────────────────────────────────────────────────────
# A model that just failed with a failover-now error (429 / 5xx / 10-min stall /
# key-absent) is parked on a short cooldown. Subsequent calls SKIP it instead of
# re-walking the chain into the same dead endpoint — so once a fallback is in
# effect it's "remembered", and the planner's dozens of calls don't each re-hit
# a storming OR :free model or a hanging NIM endpoint. The cooldown is a soft
# preference: if EVERY candidate in a chain is cooling down, we still try them
# (they may have recovered) rather than hard-failing.
_COOLDOWN_SEC = 120.0
_down_until: dict[str, float] = {}


def _is_down(model_id: str) -> bool:
    return time.time() < _down_until.get(model_id, 0.0)


# A transient GATEWAY 5xx (502/503/504 — NVIDIA NIM's load-balancer hiccuping
# on a large/slow request, returns an HTML error page) is NOT a capacity wall
# like a 429. Banishing the model for the full 120s on a blip exiles the strong
# coder (glm-5.1) to weak fallbacks for 2 min and churns rounds (django-14792
# audit). Park it only briefly so the next round retries it.
_GATEWAY_5XX_COOLDOWN_SEC = 20.0
_TRANSIENT_GATEWAY = re.compile(r'HTTP\s*(?:502|503|504)\b', re.IGNORECASE)


def _is_transient_gateway_error(exc: BaseException) -> bool:
    return bool(_TRANSIENT_GATEWAY.search(str(exc)))


def _mark_down(model_id: str, seconds: float = _COOLDOWN_SEC) -> None:
    _down_until[model_id] = time.time() + seconds


# ── "Continue, not restart" — positive counterpart to the circuit breaker ────
# _down_until is NEGATIVE memory (who just failed). Without a POSITIVE memory,
# once a role's strong primary starts storming, every following round re-probes
# the primary first and re-pays its stall/5xx before failing over again — the
# user's "it should CONTINUE, not RESTART; behave like ONE model" complaint.
# When a primary's call falls over to a fallback that WORKS, remember it: the
# next calls go straight to that fallback. A short TTL (NOT refreshed on reuse)
# means the primary is re-probed at most once per window, so it reclaims the
# slot soon after it recovers — bounding both the waste and the time pinned to a
# weaker model.
_last_good: dict[str, tuple[str, float]] = {}
_LAST_GOOD_TTL = 90.0


def _remember_good(primary: str, served_by: str) -> None:
    """Record which model actually served `primary`. If the primary served
    itself, it's healthy → drop any stickiness so it keeps the slot."""
    if served_by and served_by != primary:
        _last_good[primary] = (served_by, time.time())
    else:
        _last_good.pop(primary, None)


def _sticky_fallback(primary: str) -> "str | None":
    """The fallback that recently served `primary`, if still fresh and currently
    usable (not cooling/busy). Else None (evicting a stale entry). Not time-
    refreshed by callers, so it naturally expires `_LAST_GOOD_TTL` after the
    fallover that set it — re-probing the primary then."""
    rec = _last_good.get(primary)
    if not rec:
        return None
    fb, ts = rec
    if time.time() - ts > _LAST_GOOD_TTL:
        _last_good.pop(primary, None)
        return None
    if fb == primary or _is_down(fb) or _ml.is_busy(fb):
        return None
    return fb

try:
    from tools.connectivity import is_online, wait_for_connection
    _HAS_CONNECTIVITY = True
except ImportError:
    _HAS_CONNECTIVITY = False

# Timeout backoff: starts at 10s, doubles each retry, capped at 300s (5 min)
_TIMEOUT_BACKOFF_START = 10
_TIMEOUT_BACKOFF_CAP   = 300


def _default_timeout(model_id: str) -> float:
    """No practical time limit — let models finish thinking."""
    return 3600.0  # 1 hour — effectively unlimited


def _timeout_wait(timeout_attempt: int) -> float:
    """Exponential backoff for timeout retries: 10s → 20s → 40s … capped at 5min."""
    return min(_TIMEOUT_BACKOFF_START * (2 ** timeout_attempt), _TIMEOUT_BACKOFF_CAP)


async def call_with_retry(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 16384,
    json_mode: bool = False,
    max_retries: int = 3,
    timeout: float = 0,  # 0 = auto-detect from provider
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Call a model with retries + exponential backoff + automatic fallback.
    Streams thinking to terminal via thought_logger.
    If stop_check(accumulated_text) returns True, stops the stream early.

    Timeout errors: infinite retries with increasing wait (10s → 20s → 40s … 5min max).
    Other errors:   up to max_retries, then tries the fallback model once.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    # v5: pause if WiFi dropped
    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"API call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during call to {model_id}")

    last_error = None
    error_attempt = 0   # counts non-timeout failures (has a limit)
    timeout_attempt = 0  # counts timeout failures (no limit)

    # Continue, not restart: if a fallback recently served this primary and is
    # still usable, go straight to it instead of re-probing the (likely still
    # storming) primary and re-paying its stall. On success return; on failure
    # drop the stickiness and fall through to the normal primary-first path.
    _sticky = _sticky_fallback(model_id)
    if _sticky:
        try:
            async with _ml.slot(_sticky):
                _sres = await asyncio.wait_for(
                    call_api_stream(_sticky, prompt, system, temperature, max_tokens,
                                    json_mode, log_label, stop_check=stop_check),
                    timeout=_default_timeout(_sticky),
                )
            status(f"  ↻ {model_id} still cooling — continuing on {_sticky}")
            return _sres
        except Exception as _se:
            if _is_failover_now_error(_se) or _is_permanent_error(_se):
                _mark_down(_sticky)
            _last_good.pop(model_id, None)
            warn(f"  sticky {_sticky} for {model_id} failed ({str(_se)[:60]}) — reverting to chain")

    while True:
        if _is_down(model_id) or _ml.is_busy(model_id):
            last_error = last_error or f"{model_id} busy/cooling — skip to fallback"
            break
        try:
            async with _ml.slot(model_id):
                result = await asyncio.wait_for(
                    call_api_stream(model_id, prompt, system, temperature, max_tokens,
                                    json_mode, log_label, stop_check=stop_check),
                    timeout=timeout,
                )
            _remember_good(model_id, model_id)   # primary healthy → drop stickiness
            return result

        except asyncio.TimeoutError:
            wait = _timeout_wait(timeout_attempt)
            last_error = f"Timeout after {timeout}s"
            warn(f"  ⚠️  {model_id}: {last_error} — waiting {wait:.0f}s then retrying (no limit)...")
            timeout_attempt += 1
            await asyncio.sleep(wait)
            continue  # infinite retry on timeout

        except Exception as e:
            last_error = str(e)[:120]
            # Don't retry on 400 (bad request) — prompt/params are wrong
            if "HTTP 400" in str(e):
                warn(f"  {model_id}: Bad request — {last_error}")
                break
            # Don't retry on permanent failures (model deprecated, 410/404/
            # 401/403). Retrying a sunset model just burns backoff seconds.
            # Skip straight to fallback.
            if _is_permanent_error(e):
                warn(
                    f"  {model_id}: permanent error ({last_error}) — "
                    f"skipping retries, going to fallback"
                )
                _mark_down(model_id)
                break
            # Busy / overloaded / 10-min-TTFT stall → fail over to the NEXT
            # model in the chain now; retrying the same endpoint would just
            # re-queue behind the same overloaded provider.
            if _is_failover_now_error(e):
                _gw = _is_transient_gateway_error(e)
                warn(
                    f"  {model_id}: {last_error} — "
                    + ("transient gateway 5xx, brief cooldown (retried soon)"
                       if _gw else "busy/stalled")
                    + ", failing over to next in chain (no re-queue)"
                )
                _mark_down(model_id,
                           _GATEWAY_5XX_COOLDOWN_SEC if _gw else _COOLDOWN_SEC)
                break
            error_attempt += 1
            if error_attempt >= max_retries:
                break
            wait = 2 * error_attempt
            warn(f"  ⚠️  {model_id}: {last_error}. Retry {error_attempt}/{max_retries} in {wait}s...")
            await asyncio.sleep(wait)

    # v9.1 fix: walk the fallback chain (NVIDIA_FALLBACKS values are
    # TUPLES per config.py — previously the whole tuple was passed as
    # a model_id, silently breaking every fallback attempt).
    fb_raw = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    chain = ((fb_raw,) if isinstance(fb_raw, str)
             else tuple(fb_raw) if fb_raw else ())
    if chain:
        fb_errors: list[str] = []
        # Live (not-cooling-down) fallbacks first; cooled-down ones only as a
        # last resort (they may have recovered) — never re-walk into a known-dead
        # endpoint while a working one is available.
        # Prefer fallbacks that are neither cooling-down nor busy (at their
        # concurrency cap / inside their rate window); only then the busy ones
        # (slot waits for them), and cooled-down ones as the last resort.
        ready  = [fb for fb in chain if not _is_down(fb) and not _ml.is_busy(fb)]
        busy   = [fb for fb in chain if not _is_down(fb) and _ml.is_busy(fb)]
        cooled = [fb for fb in chain if _is_down(fb)]
        for fb in (ready + busy + cooled):
            error(f"{model_id} unreachable ({last_error}). Falling back to {fb}...")
            try:
                async with _ml.slot(fb):
                    _fbres = await asyncio.wait_for(
                        call_api_stream(fb, prompt, system, temperature, max_tokens,
                                        json_mode, log_label, stop_check=stop_check),
                        timeout=_default_timeout(fb),
                    )
                _remember_good(model_id, fb)   # this fallback worked — prefer it next round
                return _fbres
            except Exception as e2:
                if _is_failover_now_error(e2) or _is_permanent_error(e2):
                    _mark_down(fb)
                fb_errors.append(f"{fb}={str(e2)[:80]}")
                continue
        raise RuntimeError(
            f"All retries failed for {model_id} ({last_error}) "
            f"AND all {len(chain)} fallbacks failed: "
            f"{'; '.join(fb_errors)}"
        )

    raise RuntimeError(f"All retries failed for {model_id}: {last_error}")


async def call_with_retry_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 16384,
    json_mode: bool = False,
    max_retries: int = 3,
    timeout: float = 0,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Stream a model call with retry + backoff + fallback.
    Timeout errors: infinite retries with increasing wait.
    Other errors:   up to max_retries, then fallback model.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"stream call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during stream call to {model_id}")

    last_error = None
    error_attempt = 0
    timeout_attempt = 0

    # Continue, not restart: if a fallback recently served this primary and is
    # still usable, go straight to it instead of re-probing the (likely still
    # storming) primary and re-paying its stall. On success return; on failure
    # drop the stickiness and fall through to the normal primary-first path.
    _sticky = _sticky_fallback(model_id)
    if _sticky:
        try:
            async with _ml.slot(_sticky):
                _sres = await asyncio.wait_for(
                    call_api_stream(_sticky, prompt, system, temperature, max_tokens,
                                    json_mode, log_label, stop_check=stop_check),
                    timeout=_default_timeout(_sticky),
                )
            status(f"  ↻ {model_id} still cooling — continuing on {_sticky}")
            return _sres
        except Exception as _se:
            if _is_failover_now_error(_se) or _is_permanent_error(_se):
                _mark_down(_sticky)
            _last_good.pop(model_id, None)
            warn(f"  sticky {_sticky} for {model_id} failed ({str(_se)[:60]}) — reverting to chain")

    while True:
        if _is_down(model_id) or _ml.is_busy(model_id):
            last_error = last_error or f"{model_id} busy/cooling — skip to fallback"
            break
        try:
            async with _ml.slot(model_id):
                result = await asyncio.wait_for(
                    call_api_stream(model_id, prompt, system, temperature, max_tokens,
                                    json_mode, log_label, stop_check=stop_check),
                    timeout=timeout,
                )
            _remember_good(model_id, model_id)   # primary healthy → drop stickiness
            return result

        except asyncio.TimeoutError:
            wait = _timeout_wait(timeout_attempt)
            last_error = f"Timeout after {timeout}s"
            warn(f"  ⚠️  {model_id}: {last_error} — waiting {wait:.0f}s then retrying (no limit)...")
            timeout_attempt += 1
            await asyncio.sleep(wait)
            continue

        except Exception as e:
            last_error = str(e)[:120]
            if "HTTP 400" in str(e):
                warn(f"  {model_id}: Bad request — {last_error}")
                break
            if _is_permanent_error(e):
                warn(
                    f"  {model_id}: permanent error ({last_error}) — "
                    f"skipping retries, going to fallback"
                )
                _mark_down(model_id)
                break
            # Busy / overloaded / 10-min-TTFT stall → fail over to the NEXT
            # model in the chain now; retrying the same endpoint would just
            # re-queue behind the same overloaded provider.
            if _is_failover_now_error(e):
                _gw = _is_transient_gateway_error(e)
                warn(
                    f"  {model_id}: {last_error} — "
                    + ("transient gateway 5xx, brief cooldown (retried soon)"
                       if _gw else "busy/stalled")
                    + ", failing over to next in chain (no re-queue)"
                )
                _mark_down(model_id,
                           _GATEWAY_5XX_COOLDOWN_SEC if _gw else _COOLDOWN_SEC)
                break
            error_attempt += 1
            if error_attempt >= max_retries:
                break
            wait = 2 * error_attempt
            warn(f"  ⚠️  {model_id}: {last_error}. Retry {error_attempt}/{max_retries} in {wait}s...")
            await asyncio.sleep(wait)

    # v9.1 fix: walk the fallback tuple chain (mirror of call_with_retry).
    fb_raw = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    chain = ((fb_raw,) if isinstance(fb_raw, str)
             else tuple(fb_raw) if fb_raw else ())
    if chain:
        fb_errors: list[str] = []
        # Live (not-cooling-down) fallbacks first; cooled-down ones only as a
        # last resort (they may have recovered) — never re-walk into a known-dead
        # endpoint while a working one is available.
        # Prefer fallbacks that are neither cooling-down nor busy (at their
        # concurrency cap / inside their rate window); only then the busy ones
        # (slot waits for them), and cooled-down ones as the last resort.
        ready  = [fb for fb in chain if not _is_down(fb) and not _ml.is_busy(fb)]
        busy   = [fb for fb in chain if not _is_down(fb) and _ml.is_busy(fb)]
        cooled = [fb for fb in chain if _is_down(fb)]
        for fb in (ready + busy + cooled):
            error(f"{model_id} unreachable ({last_error}). Falling back to {fb}...")
            try:
                async with _ml.slot(fb):
                    _fbres = await asyncio.wait_for(
                        call_api_stream(fb, prompt, system, temperature, max_tokens,
                                        json_mode, log_label, stop_check=stop_check),
                        timeout=_default_timeout(fb),
                    )
                _remember_good(model_id, fb)   # this fallback worked — prefer it next round
                return _fbres
            except Exception as e2:
                if _is_failover_now_error(e2) or _is_permanent_error(e2):
                    _mark_down(fb)
                fb_errors.append(f"{fb}={str(e2)[:80]}")
                continue
        raise RuntimeError(
            f"All retries failed for {model_id} ({last_error}) "
            f"AND all {len(chain)} fallbacks failed: "
            f"{'; '.join(fb_errors)}"
        )

    raise RuntimeError(f"All retries failed for {model_id}: {last_error}")
