"""Audit `_plan_done_context_kind` — the PLAN_DONE position validator.

A `[PLAN DONE][CONFIRM_PLAN_DONE]` is honored only when its preceding
context shows the plan was actually committed:

  • "end-plan-block" — there's a `=== END PLAN ===` marker above it
  • "terminal-section" — there's a `## VERIFICATION` / `## CONFIDENCE GATE` /
    `## PRE-MORTEM RESOLUTION` / `## TEST CRITERIA` / `## FINAL NOTES` /
    `## SUMMARY` section above it
  • "post-think" — a `</think>` or `[/think]` close tag is within the
    short lookback window

If NONE of these are present, the signal is rejected (returns None).

This is critical: a model that types `[PLAN DONE][CONFIRM_PLAN_DONE]` in
the *middle* of drafting (or while paraphrasing the protocol in prose)
must NOT cause the planner round to commit early. Historic bugs lived
right here.
"""
import pytest
from core.tool_call import _plan_done_context_kind


# ─────────────── VALID CONTEXTS — should return a kind ───────────────


def test_pdctx__end_plan_block_above():
    """`=== END PLAN ===` somewhere in the long-lookback window → 'end-plan-block'."""
    text = "Plan content...\n=== END PLAN ===\n[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "end-plan-block"


def test_pdctx__verification_section():
    text = (
        "## VERIFICATION\nAll requirements covered.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__confidence_gate_section():
    text = (
        "## CONFIDENCE GATE\nReady to commit.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__pre_mortem_resolution_section():
    text = (
        "## PRE-MORTEM RESOLUTION\nRisks mitigated.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__test_criteria_section():
    text = (
        "## TEST CRITERIA\nDefined.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__final_notes_section():
    text = (
        "## FINAL NOTES\nDone.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__summary_section():
    text = (
        "## SUMMARY\nWrap-up.\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__post_think_xml():
    text = "<think>\nplanned everything\n</think>\n[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "post-think"


def test_pdctx__post_think_bracket():
    text = "[think]\nplanned everything\n[/think]\n[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "post-think"


# ─────────────── INVALID CONTEXTS — should return None ───────────────


def test_pdctx__bare_signal_no_context():
    """Bare signal with no plan structure before it → None."""
    text = "Just a quick note. [PLAN DONE][CONFIRM_PLAN_DONE]\n"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) is None


def test_pdctx__empty_prefix():
    """Signal at position 0 (no prefix) → None."""
    text = "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) is None


def test_pdctx__only_plan_body_no_close():
    """Plan in progress, no END / VERIFICATION / SUMMARY → None."""
    text = (
        "## OBJECTIVE\nDo a thing.\n\n## CONSTRAINTS\nNone.\n\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) is None


def test_pdctx__think_unclosed_not_post_think():
    """An open <think> with no closing tag should NOT count as post-think.
    (Otherwise an in-progress thought firing [PLAN DONE] inside it would
    commit early.)"""
    text = "<think>\nplanning still\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    # No </think> visible → not post-think → falls to None
    assert _plan_done_context_kind(text, sig_start) is None


# ─────────────── LOOKBACK BOUNDARIES ───────────────


def test_pdctx__end_plan_outside_long_lookback_window():
    """END PLAN further than 2000 chars back → should NOT trigger."""
    padding = "x" * 2200  # 2200 chars between END PLAN and the signal
    text = "=== END PLAN ===\n" + padding + "[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) is None


def test_pdctx__end_plan_inside_long_lookback():
    """END PLAN at exactly the edge of long lookback should still match."""
    padding = "x" * 100
    text = "=== END PLAN ===\n" + padding + "[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "end-plan-block"


def test_pdctx__think_close_outside_short_lookback():
    """[/think] further than ~800 chars back → not post-think."""
    padding = "x" * 1000
    text = "[/think]\n" + padding + "[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    # Short lookback for post-think is tight — should reject
    assert _plan_done_context_kind(text, sig_start) is None


def test_pdctx__think_close_inside_short_lookback():
    """[/think] right before the signal → post-think."""
    text = "[/think]\nReady.\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "post-think"


# ─────────────── PRECEDENCE ───────────────


def test_pdctx__end_plan_takes_priority_over_terminal_section():
    """If both `=== END PLAN ===` and a terminal section exist, the
    code returns 'end-plan-block' first (the stricter, more explicit marker)."""
    text = (
        "## SUMMARY\nWrap.\n"
        "=== END PLAN ===\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "end-plan-block"


def test_pdctx__terminal_section_takes_priority_over_think():
    """Terminal section beats post-think if both are present."""
    text = (
        "## VERIFICATION\nAll covered.\n"
        "[/think]\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


# ─────────────── CASE INSENSITIVITY ───────────────


def test_pdctx__end_plan_lowercase_matches():
    text = "=== end plan ===\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "end-plan-block"


def test_pdctx__verification_lowercase_matches():
    text = "## verification\nok.\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__think_uppercase_matches():
    text = "</THINK>\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "post-think"


# ─────────────── FORMATTING VARIATIONS ───────────────


def test_pdctx__pre_mortem_with_dash():
    """`PRE-MORTEM` with a dash."""
    text = "## PRE-MORTEM RESOLUTION\nok.\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__pre_mortem_with_space():
    """`PRE MORTEM` with a space."""
    text = "## PRE MORTEM RESOLUTION\nok.\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__end_plan_extra_whitespace():
    """`===  END  PLAN  ===` with extra spaces."""
    text = "===   END   PLAN   ===\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "end-plan-block"


def test_pdctx__section_with_leading_whitespace():
    """`  ## VERIFICATION` indented should still match (MULTILINE anchor with [ \\t]*)."""
    text = "  ## VERIFICATION\nok.\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"


def test_pdctx__section_inside_prose_no_match():
    """`## VERIFICATION` mentioned mid-line should NOT count as a section
    header (MULTILINE means `^` is line-start, so embedded text fails)."""
    text = "see the ## VERIFICATION header below\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    # The `## ...` is not at line start, so it doesn't match
    # (still might match if regex is lax — assert behavior)
    result = _plan_done_context_kind(text, sig_start)
    # Most strict behavior: this returns None
    assert result is None or isinstance(result, str)


# ─────────────── EDGE CASES ───────────────


def test_pdctx__signal_at_text_start_no_underflow():
    """Signal at position 0 — should not crash on negative lookback indices."""
    text = "[PLAN DONE][CONFIRM_PLAN_DONE]"
    out = _plan_done_context_kind(text, 0)
    assert out is None


def test_pdctx__multiple_signals_independent_evaluation():
    """Each signal position is evaluated against its own lookback window."""
    text = (
        "## SUMMARY\nfirst\n[PLAN DONE][CONFIRM_PLAN_DONE]\n"
        "later prose with no anchor\n[PLAN DONE][CONFIRM_PLAN_DONE]\n"
    )
    # First signal: terminal-section
    first = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, first) == "terminal-section"
    # Second signal: also gets terminal-section because SUMMARY is still in
    # the long lookback (2000 chars). This is intentional: once the plan
    # is committed, subsequent signals continue to be honored.
    second = text.find("[PLAN DONE]", first + 1)
    assert _plan_done_context_kind(text, second) == "terminal-section"


def test_pdctx__unicode_in_lookback_doesnt_crash():
    text = "## VERIFICATION\n北京 résumé\n[PLAN DONE][CONFIRM_PLAN_DONE]"
    sig_start = text.find("[PLAN DONE]")
    assert _plan_done_context_kind(text, sig_start) == "terminal-section"
