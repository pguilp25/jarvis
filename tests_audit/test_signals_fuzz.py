"""THIRD-PASS FUZZ audit of signal detection.

Combined with the adversarial signal tests, this ensures no edge-case
arrangement of input can:
  (A) Make a partial signal fire (bare half).
  (B) Suppress a real signal that should fire.
  (C) Distort the position of a signal across re-runs.

100s of random arrangements per signal.
"""
import pytest
import random
import re
import string
from core.tool_call import (
    STOP_TAG, DONE_TAG, FORCE_DONE_TAG, CONTINUE_TAG, PLAN_DONE_TAG,
    _BARE_STOP, _BARE_DONE, _BARE_CONTINUE,
    _mask_for_signals,
)


SIGNAL_TAGS = [
    ("STOP", STOP_TAG, "[STOP][CONFIRM_STOP]", "[STOP]"),
    ("DONE", DONE_TAG, "[DONE][CONFIRM_DONE]", "[DONE]"),
    ("FORCE_DONE", FORCE_DONE_TAG, "[FORCE DONE][CONFIRM_FORCE_DONE]", "[FORCE DONE]"),
    ("CONTINUE", CONTINUE_TAG, "[CONTINUE][CONFIRM_CONTINUE]", "[CONTINUE]"),
    ("PLAN_DONE", PLAN_DONE_TAG, "[PLAN DONE][CONFIRM_PLAN_DONE]", "[PLAN DONE]"),
]


# ─────────────── PROPERTY: PARTIAL HALVES DON'T FIRE ───────────────


@pytest.mark.parametrize("name,tag,form,first_half", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_partial__bare_half_never_fires(name, tag, form, first_half, seed):
    """Random prose with the FIRST HALF only — no signal should fire."""
    rng = random.Random(seed)
    safe = string.ascii_letters + " 0123456789 "
    prose_before = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    prose_after = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    text = f"{prose_before}{first_half}{prose_after}"
    masked = _mask_for_signals(text)
    assert tag.search(masked) is None, (
        f"{name} fired on bare half: {text!r}"
    )


@pytest.mark.parametrize("seed", range(30))
def test_partial__1000_bare_stops_no_fire(seed):
    rng = random.Random(seed)
    text = ""
    for _ in range(1000):
        text += "[STOP]"
        # Random whitespace between (but no second half)
        text += " " if rng.random() < 0.5 else "\n"
    masked = _mask_for_signals(text)
    assert STOP_TAG.search(masked) is None


# ─────────────── PROPERTY: BARE STOP/DONE detection (separate from two-tag) ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_bare__stop_detected_by_bare_pattern(seed):
    """`_BARE_STOP` regex finds bare [STOP] but only if not followed by CONFIRM_STOP."""
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    text = f"{prose}[STOP]{prose}"
    # _BARE_STOP should match (no CONFIRM_STOP follows)
    assert _BARE_STOP.search(text) is not None


def test_bare__stop_NOT_when_full_signal_present():
    """`[STOP][CONFIRM_STOP]` — full signal — _BARE_STOP must NOT fire."""
    text = "[STOP][CONFIRM_STOP]"
    assert _BARE_STOP.search(text) is None


def test_bare__double_bracket_starts_not_bare():
    """`[[STOP]` — extra preceding `[` — _BARE_STOP should not match."""
    text = "[[STOP]"
    # Per the lookbehind `(?<!\\[)`, the second `[STOP]` has a `[` before it → no match
    # Document the actual behavior
    result = _BARE_STOP.search(text)
    assert result is None


# ─────────────── PROPERTY: TWO-TAG FIRES ONLY ON ADJACENT HALVES ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_full__fires_with_random_prose_around(name, tag, form, _, seed):
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose_before = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    prose_after = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    text = f"{prose_before}\n{form}\n{prose_after}"
    masked = _mask_for_signals(text)
    assert tag.search(masked) is not None, f"{name} did not fire"


# ─────────────── PROPERTY: WHITESPACE BETWEEN HALVES ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_ws__varying_whitespace_between_halves(name, tag, form, _, seed):
    """Random whitespace combinations between halves still fire."""
    rng = random.Random(seed)
    ws_chars = " \t\n"
    ws = "".join(rng.choice(ws_chars) for _ in range(rng.randint(0, 10)))
    spaced = form.replace("][", f"]{ws}[")
    masked = _mask_for_signals(spaced)
    assert tag.search(masked) is not None


# ─────────────── PROPERTY: WRONG-PAIRING NEVER FIRES ───────────────


WRONG_PAIRS = [
    "[STOP][CONFIRM_DONE]",
    "[STOP][CONFIRM_CONTINUE]",
    "[STOP][CONFIRM_PLAN_DONE]",
    "[STOP][CONFIRM_FORCE_DONE]",
    "[DONE][CONFIRM_STOP]",
    "[DONE][CONFIRM_CONTINUE]",
    "[DONE][CONFIRM_FORCE_DONE]",
    "[CONTINUE][CONFIRM_STOP]",
    "[CONTINUE][CONFIRM_DONE]",
    "[PLAN DONE][CONFIRM_DONE]",
    "[PLAN DONE][CONFIRM_STOP]",
    "[PLAN DONE][CONFIRM_CONTINUE]",
    "[FORCE DONE][CONFIRM_DONE]",
    "[FORCE DONE][CONFIRM_STOP]",
]


@pytest.mark.parametrize("wrong", WRONG_PAIRS)
@pytest.mark.parametrize("seed", range(10))
def test_wrongpair__never_fires(wrong, seed):
    """No signal regex should match wrong-paired halves."""
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 50)))
    text = f"{prose}{wrong}{prose}"
    masked = _mask_for_signals(text)
    for name, tag, _, _ in SIGNAL_TAGS:
        result = tag.search(masked)
        # The signal regex must NOT match this wrong pair
        assert result is None, f"{name} fired on wrong pair {wrong!r}: {result.group()!r}"


# ─────────────── PROPERTY: SIGNALS IN INERT ZONES NEVER FIRE ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_inert__never_fires_in_fence(name, tag, form, _, seed):
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 50)))
    text = f"{prose}\n```\n{form}\n```\n{prose}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(form)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_inert__never_fires_in_think_xml(name, tag, form, _, seed):
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 50)))
    text = f"{prose}<think>{form}</think>{prose}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(form)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_inert__never_fires_in_think_bracket(name, tag, form, _, seed):
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 50)))
    text = f"{prose}[think]{form}[/think]{prose}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(form)
    assert masked[sig_pos] != '['


# ─────────────── PROPERTY: MULTIPLE SIGNALS DON'T INTERFERE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_multi__random_5_signal_ordering_all_fire(seed):
    """Generate random ordering of the 5 signals — all should fire."""
    rng = random.Random(seed)
    signals = [form for _, _, form, _ in SIGNAL_TAGS]
    rng.shuffle(signals)
    safe = string.ascii_letters + " "
    sep = "\n" + "".join(rng.choice(safe) for _ in range(rng.randint(0, 20))) + "\n"
    text = sep.join(signals)
    masked = _mask_for_signals(text)
    for name, tag, form, _ in SIGNAL_TAGS:
        assert tag.search(masked) is not None, f"{name} did not fire in: {text[:100]}"


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(30))
def test_det__same_input_same_match(name, tag, form, _, seed):
    rng = random.Random(seed)
    safe = string.ascii_letters + " "
    prose = "".join(rng.choice(safe) for _ in range(rng.randint(0, 100)))
    text = f"{prose}\n{form}\n{prose}"
    masked = _mask_for_signals(text)
    m1 = tag.search(masked)
    m2 = tag.search(masked)
    m3 = tag.search(masked)
    if m1 is not None:
        assert m1.span() == m2.span() == m3.span()


# ─────────────── EDGE: ADJACENT SIGNAL HALVES OK ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
def test_adj__no_whitespace_canonical(name, tag, form, _):
    """Canonical form has 0 whitespace between halves — fires."""
    masked = _mask_for_signals(form)
    assert tag.search(masked) is not None


# ─────────────── EDGE: MANY SAME SIGNALS ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
def test_many__same_signal_100x(name, tag, form, _):
    """100× the same signal — all findable."""
    text = "\n".join([form] * 100)
    masked = _mask_for_signals(text)
    assert len(tag.findall(masked)) == 100


# ─────────────── EDGE: CASE INSENSITIVITY ───────────────


@pytest.mark.parametrize("name,tag,form,_", SIGNAL_TAGS)
@pytest.mark.parametrize("seed", range(20))
def test_case__random_case_variants(name, tag, form, _, seed):
    rng = random.Random(seed)
    cased = "".join(
        ch.upper() if rng.random() < 0.5 else ch.lower()
        for ch in form
    )
    masked = _mask_for_signals(cased)
    assert tag.search(masked) is not None, f"{name} failed for cased: {cased!r}"
