"""Audit the two-tag signal detection: STOP_TAG, DONE_TAG, FORCE_DONE_TAG,
CONTINUE_TAG, PLAN_DONE_TAG.

Contracts:
  - Each signal requires BOTH halves in order: e.g. `[STOP][CONFIRM_STOP]`
  - Bare halves are inert text
  - Adjacent (no space), newline-separated, and space-separated all fire
  - Arbitrary text BETWEEN the two halves prevents firing
  - Wrong order prevents firing
  - FORCE_DONE_TAG must NOT match plain DONE_TAG
"""
import re
import pytest

from core.tool_call import (
    STOP_TAG, DONE_TAG, FORCE_DONE_TAG, CONTINUE_TAG, PLAN_DONE_TAG,
    _BARE_STOP, _BARE_DONE, _BARE_FORCE_DONE, _BARE_CONTINUE, _BARE_PLAN_DONE,
)


# ───────────────────── STOP ─────────────────────

def test_stop__adjacent_fires():
    assert STOP_TAG.search("[STOP][CONFIRM_STOP]")


def test_stop__newline_separated_fires():
    assert STOP_TAG.search("[STOP]\n[CONFIRM_STOP]")


def test_stop__space_separated_fires():
    assert STOP_TAG.search("[STOP] [CONFIRM_STOP]")


def test_stop__text_between_does_NOT_fire():
    assert not STOP_TAG.search("[STOP] then [CONFIRM_STOP]")


def test_stop__wrong_order_does_NOT_fire():
    assert not STOP_TAG.search("[CONFIRM_STOP][STOP]")


def test_stop__bare_alone_does_NOT_fire_full():
    assert not STOP_TAG.search("[STOP]")
    assert not STOP_TAG.search("[CONFIRM_STOP]")


def test_stop__bare_detection_fires():
    """The BARE detector fires on lone halves so we can correct the model."""
    assert _BARE_STOP.search("[STOP]")
    # Already-paired should NOT trigger bare detection
    # (the regex uses negative lookahead)


def test_stop__case_insensitive():
    """Lowercase variants — should still fire (re.IGNORECASE used)."""
    assert STOP_TAG.search("[stop][confirm_stop]")


# ───────────────────── DONE ─────────────────────

def test_done__adjacent_fires():
    assert DONE_TAG.search("[DONE][CONFIRM_DONE]")


def test_done__newline_separated_fires():
    assert DONE_TAG.search("[DONE]\n[CONFIRM_DONE]")


def test_done__bare_alone_does_NOT_fire_full():
    assert not DONE_TAG.search("[DONE]")


# ───────────────────── FORCE DONE ─────────────────────

def test_force_done__adjacent_fires():
    assert FORCE_DONE_TAG.search("[FORCE DONE][CONFIRM_FORCE_DONE]")


def test_force_done__does_NOT_match_plain_done():
    """[FORCE DONE][CONFIRM_FORCE_DONE] must not match the DONE_TAG regex."""
    s = "[FORCE DONE][CONFIRM_FORCE_DONE]"
    assert not DONE_TAG.search(s), f"DONE_TAG should not match FORCE_DONE: {s!r}"


def test_force_done__plain_done_does_NOT_match_force():
    """The reverse: plain `[DONE][CONFIRM_DONE]` should NOT match FORCE_DONE."""
    s = "[DONE][CONFIRM_DONE]"
    assert not FORCE_DONE_TAG.search(s)


def test_force_done__bare_detection():
    assert _BARE_FORCE_DONE.search("[FORCE DONE]")


# ───────────────────── CONTINUE ─────────────────────

def test_continue__adjacent_fires():
    assert CONTINUE_TAG.search("[CONTINUE][CONFIRM_CONTINUE]")


def test_continue__bare_detection():
    assert _BARE_CONTINUE.search("[CONTINUE]")


# ───────────────────── PLAN DONE ─────────────────────

def test_plan_done__adjacent_fires():
    assert PLAN_DONE_TAG.search("[PLAN DONE][CONFIRM_PLAN_DONE]")


def test_plan_done__case_insensitive():
    assert PLAN_DONE_TAG.search("[plan done][confirm_plan_done]")


def test_plan_done__space_variation():
    """`[PLAN DONE]` (with space) — make sure the regex accepts the space."""
    assert PLAN_DONE_TAG.search("[PLAN DONE][CONFIRM_PLAN_DONE]")


def test_plan_done__bare_detection():
    assert _BARE_PLAN_DONE.search("[PLAN DONE]")


# ───────────────────── DISTINCTNESS ─────────────────────

def test_signals__no_cross_match():
    """Each signal's regex matches ONLY its own pair, not others."""
    pairs = {
        "STOP": "[STOP][CONFIRM_STOP]",
        "DONE": "[DONE][CONFIRM_DONE]",
        "FORCE_DONE": "[FORCE DONE][CONFIRM_FORCE_DONE]",
        "CONTINUE": "[CONTINUE][CONFIRM_CONTINUE]",
        "PLAN_DONE": "[PLAN DONE][CONFIRM_PLAN_DONE]",
    }
    regexes = {
        "STOP": STOP_TAG,
        "DONE": DONE_TAG,
        "FORCE_DONE": FORCE_DONE_TAG,
        "CONTINUE": CONTINUE_TAG,
        "PLAN_DONE": PLAN_DONE_TAG,
    }
    for name, regex in regexes.items():
        for pair_name, pair_text in pairs.items():
            should_match = (name == pair_name)
            actually_matches = bool(regex.search(pair_text))
            assert actually_matches == should_match, (
                f"{name} regex on {pair_name}: expected {should_match}, "
                f"got {actually_matches}"
            )


def test_signals__embedded_in_prose_fires():
    """A signal at the end of a long response — should still match."""
    text = "Lots of prose here.\nMore prose.\n" + ("filler " * 100) + "\n[STOP][CONFIRM_STOP]"
    assert STOP_TAG.search(text)


def test_signals__nested_in_quotes_still_matches_regex():
    """`"[STOP][CONFIRM_STOP]"` — the regex matches even in quotes.
    The masking layer above is what actually filters; the raw regex matches."""
    text = '"[STOP][CONFIRM_STOP]"'
    assert STOP_TAG.search(text)


def test_signals__multiple_signals_in_response():
    """If both STOP and DONE appear, both match (the runtime decides which fires)."""
    text = "tool calls...\n[STOP][CONFIRM_STOP]\nfollowed by [DONE][CONFIRM_DONE]"
    assert STOP_TAG.search(text)
    assert DONE_TAG.search(text)
