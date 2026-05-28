"""Sliding-window rate limiter (core.model_limits) — count, delay only when full.

The limiter must NOT impose fixed inter-call spacing or wait for a calendar
reset: while there's budget left in the trailing window it fires immediately
(even at :45 into a minute); only when the window is full does it wait, and only
by the minimal time for the oldest call to age out.
"""
import time
from collections import deque

import core.model_limits as ml

M = "zai/glm-4.7-flash"   # configured (1, 60) in _RATE_LIMITS


def _set(model, times):
    ml._call_times[model] = deque(times)


def test_empty_window_fires_now():
    _set(M, [])
    assert ml.rate_wait(M) == 0.0


def test_under_limit_fires_now_midwindow():
    # limit is N per window; with N-1 recent calls there's still budget → 0 wait,
    # regardless of how far into the window we are (the user's key requirement).
    n, win = ml._RATE_LIMITS[M]
    if n >= 2:
        _set(M, [time.time() - 30] * (n - 1))   # 30s into the window, budget left
        assert ml.rate_wait(M) == 0.0


def test_full_window_waits_only_minimal():
    n, win = ml._RATE_LIMITS[M]
    now = time.time()
    # window full, oldest call was `win-12`s ago → must wait ~12s (NOT the full
    # window, NOT until a fixed reset point).
    oldest_age = win - 12
    _set(M, [now - oldest_age] * n)
    w = ml.rate_wait(M)
    assert 0 < w <= 12.5, w
    assert w < win                      # never a full-window/fixed wait


def test_aged_out_call_frees_the_slot():
    n, win = ml._RATE_LIMITS[M]
    now = time.time()
    # n calls but the oldest is older than the window → it's pruned → budget → 0
    _set(M, [now - win - 1] + [now] * (n - 1))
    assert ml.rate_wait(M) == 0.0
    assert len(ml._call_times[M]) == n - 1   # the expired one was dropped


def test_unconfigured_model_no_wait():
    assert ml.rate_wait("nvidia/glm-5.1") == 0.0   # not rate-capped → fire freely


def test_record_start_appends_only_for_capped_models():
    _set(M, [])
    ml._record_start(M)
    assert len(ml._call_times[M]) == 1
    # an unconfigured model isn't tracked
    ml._call_times.pop("nvidia/gpt-oss-120b", None)
    ml._record_start("nvidia/gpt-oss-120b")
    assert "nvidia/gpt-oss-120b" not in ml._call_times
