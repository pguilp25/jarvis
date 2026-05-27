"""ADVERSARIAL SECOND-PASS audit of the 5 two-tag signals:
  [STOP][CONFIRM_STOP], [DONE][CONFIRM_DONE], [FORCE DONE][CONFIRM_FORCE_DONE],
  [CONTINUE][CONFIRM_CONTINUE], [PLAN DONE][CONFIRM_PLAN_DONE].

Each must fire ONLY when both halves appear correctly adjacent and outside
any masked context (think, fence, backtick, plan body, edit body).

Adversarial coverage:
  • Whitespace between the two halves: 0/1/many spaces, tabs, newlines.
  • Case variations on the inner words.
  • Partial signals (only first half) MUST NOT fire.
  • Wrong second half (e.g. [STOP][CONFIRM_DONE]) MUST NOT fire.
  • Bare-half spam: 1000× [STOP] alone — no false positives.
  • Multiple signals in the same text.
"""
import pytest
import re
from core.tool_call import (
    STOP_TAG, DONE_TAG, FORCE_DONE_TAG, CONTINUE_TAG, PLAN_DONE_TAG,
    _mask_for_signals,
)


SIGNAL_TAGS = [
    ("STOP", STOP_TAG, "[STOP][CONFIRM_STOP]"),
    ("DONE", DONE_TAG, "[DONE][CONFIRM_DONE]"),
    ("FORCE_DONE", FORCE_DONE_TAG, "[FORCE DONE][CONFIRM_FORCE_DONE]"),
    ("CONTINUE", CONTINUE_TAG, "[CONTINUE][CONFIRM_CONTINUE]"),
    ("PLAN_DONE", PLAN_DONE_TAG, "[PLAN DONE][CONFIRM_PLAN_DONE]"),
]


# ─────────────── PROPERTY: each signal fires on canonical form ───────────────


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_canon__fires_on_canonical_form(name, tag, form):
    assert tag.search(_mask_for_signals(form)) is not None


# ─────────────── WHITESPACE BETWEEN HALVES ───────────────


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_ws__no_space_fires(name, tag, form):
    """Canonical form has 0 spaces — fires."""
    assert tag.search(_mask_for_signals(form)) is not None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_ws__single_space_between_fires(name, tag, form):
    """`[STOP] [CONFIRM_STOP]` with single space — fires (regex \\s*)."""
    spaced = form.replace("][", "] [")
    assert tag.search(_mask_for_signals(spaced)) is not None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_ws__many_spaces_between_fires(name, tag, form):
    """5 spaces between halves — fires."""
    spaced = form.replace("][", "]     [")
    assert tag.search(_mask_for_signals(spaced)) is not None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_ws__newline_between_fires(name, tag, form):
    """Newline between halves — `\\s*` includes \\n."""
    spaced = form.replace("][", "]\n[")
    assert tag.search(_mask_for_signals(spaced)) is not None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_ws__tab_between_fires(name, tag, form):
    """Tab between halves — fires."""
    spaced = form.replace("][", "]\t[")
    assert tag.search(_mask_for_signals(spaced)) is not None


# ─────────────── PARTIAL SIGNALS DON'T FIRE ───────────────


def test_partial__bare_stop_no_fire():
    assert STOP_TAG.search(_mask_for_signals("[STOP]")) is None


def test_partial__bare_done_no_fire():
    assert DONE_TAG.search(_mask_for_signals("[DONE]")) is None


def test_partial__bare_continue_no_fire():
    assert CONTINUE_TAG.search(_mask_for_signals("[CONTINUE]")) is None


def test_partial__bare_plan_done_no_fire():
    assert PLAN_DONE_TAG.search(_mask_for_signals("[PLAN DONE]")) is None


def test_partial__bare_force_done_no_fire():
    assert FORCE_DONE_TAG.search(_mask_for_signals("[FORCE DONE]")) is None


def test_partial__bare_confirm_no_fire():
    """`[CONFIRM_STOP]` alone — no fire."""
    assert STOP_TAG.search(_mask_for_signals("[CONFIRM_STOP]")) is None


def test_partial__bare_halves_spammed_no_fires():
    """1000 [STOP] alone — should NOT fire (no confirm half adjacent)."""
    text = "[STOP] " * 1000
    assert STOP_TAG.search(_mask_for_signals(text)) is None


def test_partial__1000_first_halves_no_fires():
    """1000 [DONE] without confirms."""
    text = "[DONE]\n" * 1000
    assert DONE_TAG.search(_mask_for_signals(text)) is None


# ─────────────── WRONG-PAIRING DOESN'T FIRE ───────────────


def test_wrong_pair__stop_done_no_fire():
    """`[STOP][CONFIRM_DONE]` — wrong second half."""
    assert STOP_TAG.search(_mask_for_signals("[STOP][CONFIRM_DONE]")) is None
    assert DONE_TAG.search(_mask_for_signals("[STOP][CONFIRM_DONE]")) is None


def test_wrong_pair__continue_stop_no_fire():
    assert CONTINUE_TAG.search(_mask_for_signals("[CONTINUE][CONFIRM_STOP]")) is None
    assert STOP_TAG.search(_mask_for_signals("[CONTINUE][CONFIRM_STOP]")) is None


def test_wrong_pair__plan_done_done_no_fire():
    """`[PLAN DONE][CONFIRM_DONE]` — mixing terminators."""
    assert PLAN_DONE_TAG.search(_mask_for_signals("[PLAN DONE][CONFIRM_DONE]")) is None
    assert DONE_TAG.search(_mask_for_signals("[PLAN DONE][CONFIRM_DONE]")) is None


def test_wrong_pair__force_done_done_no_fire():
    """`[FORCE DONE][CONFIRM_DONE]` — missing FORCE in confirm."""
    assert FORCE_DONE_TAG.search(_mask_for_signals("[FORCE DONE][CONFIRM_DONE]")) is None
    assert DONE_TAG.search(_mask_for_signals("[FORCE DONE][CONFIRM_DONE]")) is None


# ─────────────── CASE VARIATIONS ───────────────


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_case__lowercase_fires(name, tag, form):
    assert tag.search(_mask_for_signals(form.lower())) is not None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_case__mixed_case_fires(name, tag, form):
    """Mixed-case inner words still match (re.IGNORECASE)."""
    # Manually-mixed: random capitalization
    mixed = "".join(
        ch.upper() if i % 2 == 0 else ch.lower()
        for i, ch in enumerate(form)
    )
    assert tag.search(_mask_for_signals(mixed)) is not None


# ─────────────── CONTEXT MASKING ───────────────


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_mask__inside_fence_does_not_fire(name, tag, form):
    text = f"```\n{form}\n```"
    assert tag.search(_mask_for_signals(text)) is None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_mask__inside_inline_backtick_does_not_fire(name, tag, form):
    text = f"Discussion of `{form}` syntax"
    assert tag.search(_mask_for_signals(text)) is None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_mask__inside_think_xml_does_not_fire(name, tag, form):
    text = f"<think>\n{form}\n</think>"
    assert tag.search(_mask_for_signals(text)) is None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_mask__inside_think_bracket_does_not_fire(name, tag, form):
    text = f"[think]\n{form}\n[/think]"
    assert tag.search(_mask_for_signals(text)) is None


@pytest.mark.parametrize("name,tag,form", SIGNAL_TAGS)
def test_mask__inside_plan_block_does_not_fire(name, tag, form):
    text = f"=== PLAN ===\n{form}\n=== END PLAN ==="
    assert tag.search(_mask_for_signals(text)) is None


# ─────────────── ADJACENCY EDGE CASES ───────────────


def test_adjacency__no_text_between_halves():
    """`[STOP]X[CONFIRM_STOP]` — junk between halves → no fire."""
    text = "[STOP]X[CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    # The regex \s* between halves — X is not whitespace
    assert STOP_TAG.search(masked) is None


def test_adjacency__only_whitespace_between():
    """`[STOP] \\t \\n [CONFIRM_STOP]` — mixed whitespace → fire."""
    text = "[STOP] \t \n [CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    assert STOP_TAG.search(masked) is not None


def test_adjacency__prose_outside_signal_doesnt_block():
    """Prose around the signal doesn't matter."""
    text = "Some prose. [STOP][CONFIRM_STOP] then more prose."
    assert STOP_TAG.search(_mask_for_signals(text)) is not None


# ─────────────── MULTIPLE SIGNALS ───────────────


def test_multi__same_signal_twice():
    """Two [STOP][CONFIRM_STOP] in same text — both findable."""
    text = "first [STOP][CONFIRM_STOP] then second [STOP][CONFIRM_STOP]"
    matches = STOP_TAG.findall(_mask_for_signals(text))
    assert len(matches) == 2


def test_multi__different_signals_present():
    """Multiple signal kinds in one text."""
    text = "[STOP][CONFIRM_STOP] then [DONE][CONFIRM_DONE]"
    masked = _mask_for_signals(text)
    assert STOP_TAG.search(masked) is not None
    assert DONE_TAG.search(masked) is not None


def test_multi__all_5_in_one_text():
    text = (
        "[STOP][CONFIRM_STOP] [DONE][CONFIRM_DONE] "
        "[FORCE DONE][CONFIRM_FORCE_DONE] "
        "[CONTINUE][CONFIRM_CONTINUE] "
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    masked = _mask_for_signals(text)
    for name, tag, _ in SIGNAL_TAGS:
        assert tag.search(masked) is not None, f"{name} did not fire"


# ─────────────── POSITION ───────────────


def test_pos__signal_at_start():
    text = "[STOP][CONFIRM_STOP] then prose"
    assert STOP_TAG.search(_mask_for_signals(text)) is not None


def test_pos__signal_at_end():
    text = "prose then [STOP][CONFIRM_STOP]"
    assert STOP_TAG.search(_mask_for_signals(text)) is not None


def test_pos__signal_middle():
    text = "prose [STOP][CONFIRM_STOP] more prose"
    assert STOP_TAG.search(_mask_for_signals(text)) is not None


# ─────────────── INTERNAL WORD SPACING ───────────────


def test_internal__force_done_extra_spaces():
    """`[FORCE  DONE][CONFIRM_FORCE_DONE]` — extra space inside FORCE DONE."""
    text = "[FORCE  DONE][CONFIRM_FORCE_DONE]"
    # The regex uses `\\s+` between FORCE and DONE — accepts multiple spaces
    result = FORCE_DONE_TAG.search(_mask_for_signals(text))
    # Document: should fire on multiple spaces
    assert result is not None


def test_internal__plan_done_extra_spaces():
    """`[PLAN   DONE][CONFIRM_PLAN_DONE]` — extra space inside PLAN DONE."""
    text = "[PLAN   DONE][CONFIRM_PLAN_DONE]"
    result = PLAN_DONE_TAG.search(_mask_for_signals(text))
    assert result is not None


def test_internal__force_done_no_space_NO_fire():
    """`[FORCEDONE][CONFIRM_FORCE_DONE]` — no space → no match."""
    text = "[FORCEDONE][CONFIRM_FORCE_DONE]"
    result = FORCE_DONE_TAG.search(_mask_for_signals(text))
    assert result is None


# ─────────────── DEFENSIVE ───────────────


def test_def__empty_text():
    masked = _mask_for_signals("")
    for name, tag, _ in SIGNAL_TAGS:
        assert tag.search(masked) is None


def test_def__just_brackets():
    text = "[][]"
    masked = _mask_for_signals(text)
    for name, tag, _ in SIGNAL_TAGS:
        assert tag.search(masked) is None


def test_def__unicode_around_signal():
    text = "北京 [STOP][CONFIRM_STOP] 测试"
    assert STOP_TAG.search(_mask_for_signals(text)) is not None
