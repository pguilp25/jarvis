"""'Continue, not restart': retry.py remembers a working fallback (sticky memory).

When a primary storms and a fallback serves the call, the role should KEEP using
that fallback on following rounds instead of re-probing the dead primary every
time (the user's "behave like ONE model" requirement). A short TTL lets the
primary reclaim once it recovers. These tests pin both the decision helpers and
the end-to-end behavior (primary not re-probed while a fallback is sticky).
"""
import asyncio
import time

import pytest

import core.retry as R


@pytest.fixture(autouse=True)
def _clean_state():
    R._last_good.clear()
    R._down_until.clear()
    yield
    R._last_good.clear()
    R._down_until.clear()


# ── helper logic ─────────────────────────────────────────────────────────────
def test_remember_and_get_sticky():
    R._remember_good("p", "fb")
    assert R._sticky_fallback("p") == "fb"


def test_primary_serving_itself_clears_stickiness():
    R._remember_good("p", "fb")
    R._remember_good("p", "p")          # primary healthy again
    assert R._sticky_fallback("p") is None


def test_sticky_expires_after_ttl():
    R._remember_good("p", "fb")
    R._last_good["p"] = ("fb", time.time() - R._LAST_GOOD_TTL - 1)
    assert R._sticky_fallback("p") is None
    assert "p" not in R._last_good          # stale entry evicted


def test_sticky_skipped_when_fallback_is_down():
    R._remember_good("p", "fb")
    R._mark_down("fb", 60)
    assert R._sticky_fallback("p") is None


def test_no_sticky_when_unset():
    assert R._sticky_fallback("never-seen") is None


# ── end-to-end: primary not re-probed while a fallback is sticky ─────────────
def _install_fake(monkeypatch, behavior, calls):
    async def fake(model_id, prompt, system, temperature, max_tokens,
                   json_mode, log_label, stop_check=None):
        calls.append(model_id)
        act = behavior.get(model_id, "ok")
        if act == "fail":
            raise RuntimeError("HTTP 503 overloaded")
        return f"OK:{model_id}"
    monkeypatch.setattr(R, "call_api_stream", fake)
    monkeypatch.setattr(R, "_HAS_CONNECTIVITY", False)
    monkeypatch.setattr(R, "NVIDIA_FALLBACKS", {"test/primary": ("test/fb",)})
    monkeypatch.setattr(R, "GROQ_FALLBACKS", {})


def test_fallover_then_continue_on_fallback(monkeypatch):
    calls = []
    # primary always storms; fb always works
    _install_fake(monkeypatch, {"test/primary": "fail", "test/fb": "ok"}, calls)

    # round 1: primary fails -> chain walk -> fb serves; stickiness recorded
    r1 = asyncio.run(R.call_with_retry("test/primary", "p"))
    assert r1 == "OK:test/fb"
    assert R._sticky_fallback("test/primary") == "test/fb"
    assert calls.count("test/primary") == 1   # primary probed once

    # round 2: sticky fb is used DIRECTLY — primary is NOT re-probed
    calls.clear()
    r2 = asyncio.run(R.call_with_retry("test/primary", "p"))
    assert r2 == "OK:test/fb"
    assert "test/primary" not in calls         # the whole point: no re-probe
    assert calls == ["test/fb"]


def test_primary_recovers_reclaims_after_ttl(monkeypatch):
    calls = []
    behavior = {"test/primary": "fail", "test/fb": "ok"}
    _install_fake(monkeypatch, behavior, calls)
    asyncio.run(R.call_with_retry("test/primary", "p"))   # sets stickiness
    # primary fully recovers: it serves again AND its circuit-breaker cooldown
    # has lapsed; also age the sticky entry past its TTL.
    behavior["test/primary"] = "ok"
    R._down_until.clear()
    R._last_good["test/primary"] = ("test/fb", time.time() - R._LAST_GOOD_TTL - 1)
    calls.clear()
    r = asyncio.run(R.call_with_retry("test/primary", "p"))
    assert r == "OK:test/primary"              # primary re-probed and reclaimed
    assert calls[0] == "test/primary"
    assert R._sticky_fallback("test/primary") is None


def test_sticky_failure_reverts_to_chain(monkeypatch):
    calls = []
    behavior = {"test/primary": "ok", "test/fb": "ok"}
    _install_fake(monkeypatch, behavior, calls)
    # pretend fb was last-good, but now fb is broken and primary is healthy
    R._remember_good("test/primary", "test/fb")
    behavior["test/fb"] = "fail"
    r = asyncio.run(R.call_with_retry("test/primary", "p"))
    # sticky fb fails -> cleared -> primary serves
    assert r == "OK:test/primary"
    assert "test/fb" in calls and "test/primary" in calls
    assert R._sticky_fallback("test/primary") is None
