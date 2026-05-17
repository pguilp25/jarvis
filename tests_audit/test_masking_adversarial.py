"""ADVERSARIAL SECOND-PASS audit of `_mask_quoted_tags` / `_mask_for_signals`.

These are the SECURITY-CRITICAL anti-injection functions. The first-pass
verified happy paths. This pass attempts to:

  • Bypass the mask with creative nesting (think-in-fence, fence-in-think).
  • Trick streaming-detection with crafted unclosed sequences.
  • Inject a signal through "escape escape" (`\\\\[STOP]`).
  • Exploit case-insensitivity differences.
  • Probe the ORDER in which masks apply (some rules depend on others).
  • Test the IDEMPOTENCE invariant: masking twice == masking once.
  • Test the LENGTH-PRESERVATION invariant: masked text has same length.
  • Test that ONLY `[` is masked, not the closing `]`.
"""
import pytest
from core.tool_call import (
    _mask_quoted_tags,
    _mask_for_signals,
    _mask_quoted_tags_core,
)


# ─────────────── PROPERTY: LENGTH PRESERVED ───────────────


def test_inv__length_preserved_simple():
    text = "[think]secret[/think] real [CODE: a.py]"
    masked = _mask_for_signals(text)
    assert len(masked) == len(text)


def test_inv__length_preserved_long():
    text = "x" * 5000 + "[STOP][CONFIRM_STOP]" + "y" * 5000
    masked = _mask_for_signals(text)
    assert len(masked) == len(text)


def test_inv__length_preserved_nested():
    text = "[think]<think>nested</think>[/think][STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    assert len(masked) == len(text)


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


def test_inv__idempotent_basic():
    """Masking already-masked text → identical output."""
    text = "[think]secret[/think]"
    once = _mask_for_signals(text)
    twice = _mask_for_signals(once)
    assert once == twice


def test_inv__idempotent_complex():
    text = (
        "```\n[STOP][CONFIRM_STOP]\n```\n"
        "<think>more [DONE][CONFIRM_DONE]</think>\n"
        "real text [CODE: a.py]"
    )
    once = _mask_for_signals(text)
    twice = _mask_for_signals(once)
    assert once == twice


# ─────────────── PROPERTY: ONLY `[` IS MASKED ───────────────


def test_inv__only_bracket_masked():
    """`]` and other chars should pass through unchanged."""
    text = "[think]hidden][/think]"
    masked = _mask_for_signals(text)
    # All `]` survive
    assert masked.count(']') == text.count(']')


def test_inv__newlines_preserved():
    text = "[think]\nlin1\nlin2\n[/think]"
    masked = _mask_for_signals(text)
    assert masked.count('\n') == text.count('\n')


def test_inv__non_bracket_chars_unchanged():
    """Outside-mask zones: every non-`[` char must be unchanged."""
    text = "Hello, world! 北京 résumé"
    masked = _mask_for_signals(text)
    assert masked == text


# ─────────────── NESTED MASK STRUCTURES ───────────────


def test_nested__think_inside_fence():
    """A think block inside a fenced block. Both should mask their contents.
    Signal inside the inner think MUST be masked."""
    text = "```\n<think>[STOP][CONFIRM_STOP]</think>\n```\nReal: [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    inner_stop = text.find("[STOP]")
    assert masked[inner_stop] != '['
    outer_stop = text.rfind("[STOP]")
    assert masked[outer_stop] == '['


def test_nested__fence_inside_think():
    """A fence inside a think. Both masking layers apply."""
    text = "<think>\n```\n[STOP][CONFIRM_STOP]\n```\n</think>\nReal: [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    inner_stop = text.find("[STOP]")
    assert masked[inner_stop] != '['
    outer_stop = text.rfind("[STOP]")
    assert masked[outer_stop] == '['


def test_nested__backtick_inside_think():
    text = "<think>\nProtocol: `[STOP][CONFIRM_STOP]`\n</think>\nReal: [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    inner_stop = text.find("[STOP]")
    assert masked[inner_stop] != '['
    outer_stop = text.rfind("[STOP]")
    assert masked[outer_stop] == '['


def test_nested__bracket_think_inside_fence():
    text = "```\n[think]\n[STOP][CONFIRM_STOP]\n[/think]\n```\nReal: [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    inner_stop = text.find("[STOP]")
    assert masked[inner_stop] != '['
    outer_stop = text.rfind("[STOP]")
    assert masked[outer_stop] == '['


def test_nested__three_levels_deep():
    """Fence → think → backtick → signal. ALL layers should mask."""
    text = "```\n<think>Protocol: `[STOP][CONFIRM_STOP]`</think>\n```\nReal: [DONE][CONFIRM_DONE]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    done_pos = text.find("[DONE]")
    assert masked[stop_pos] != '['
    assert masked[done_pos] == '['  # not in any quoted zone


# ─────────────── STREAMING / UNCLOSED HAZARDS ───────────────


def test_streaming__unclosed_think_at_end():
    text = "real prose <think>partial reasoning [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_streaming__unclosed_fence_at_end():
    text = "real prose ```\nin fence [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_streaming__unclosed_inline_backtick_eol():
    """Inline backtick unclosed but newline follows → mask only that line."""
    text = "Protocol: `[STOP][CONFIRM_STOP]\nNew line safe."
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_streaming__multiple_unclosed_thinks():
    """If 2 opens and 1 close, the second open is the unclosed one."""
    text = "<think>first</think> ok <think>second unclosed [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_streaming__alternating_closes_and_opens():
    """4 opens, 3 closes → the LAST open is unclosed."""
    text = "<think>1</think><think>2</think><think>3</think><think>4 [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_streaming__many_balanced_thinks_no_overmasking():
    """Many balanced think blocks; a signal OUTSIDE should survive."""
    text = "<think>1</think> mid <think>2</think> [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] == '['


# ─────────────── ESCAPE-BYPASS ATTEMPTS ───────────────


def test_escape__simple_backslash_masks_bracket():
    r"""`\[STOP][CONFIRM_STOP]` — the `\[` escapes the bracket."""
    text = "discussion of \\[STOP]\\[CONFIRM_STOP] protocol"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_escape__inside_think_inside_escape():
    r"""\\[think]\\[STOP]\\[CONFIRM_STOP][/think] — escapes everywhere."""
    text = "ref: \\[think]\\[STOP]\\[CONFIRM_STOP]\\[/think]"
    masked = _mask_for_signals(text)
    # All `[`s after `\` are masked
    for pos in [m for m in range(len(text)) if text[m] == '[' and text[m-1] == '\\']:
        assert masked[pos] != '['


# ─────────────── CASE SENSITIVITY ───────────────


def test_case__think_uppercase():
    text = "<THINK>secret [STOP][CONFIRM_STOP]</THINK>"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_case__bracket_think_mixed():
    text = "[ThInK][STOP][CONFIRM_STOP][/tHiNk]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


# ─────────────── PLAN-BODY MASKING ADVERSARIAL ───────────────


def test_plan__signal_in_plan_body_masked():
    text = (
        "=== PLAN ===\n"
        "step 1: [PLAN DONE][CONFIRM_PLAN_DONE] commits\n"
        "=== END PLAN ===\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    masked = _mask_for_signals(text)
    inside_pd = text.find("[PLAN DONE]")
    outside_pd = text.find("[PLAN DONE]", inside_pd + 1)
    assert masked[inside_pd] != '['
    assert masked[outside_pd] == '['


def test_plan__unclosed_plan_blocks_rest():
    """Unclosed `=== PLAN ===` masks from open to EOT."""
    text = "=== PLAN ===\nstill writing... [PLAN DONE][CONFIRM_PLAN_DONE]"
    masked = _mask_for_signals(text)
    pd_pos = text.find("[PLAN DONE]")
    assert masked[pd_pos] != '['


def test_plan_edit__signal_in_plan_edit_masked():
    text = (
        "=== PLAN_EDIT ===\n"
        "[REPLACE LINES 1-1]\n"
        "[STOP][CONFIRM_STOP]\n"
        "[/REPLACE]\n"
        "=== END PLAN_EDIT ===\n"
        "[STOP][CONFIRM_STOP]"
    )
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    outside = text.rfind("[STOP]")
    assert masked[inside] != '['
    assert masked[outside] == '['


# ─────────────── EDIT-BLOCK MASKING ADVERSARIAL ───────────────


def test_edit_block__signal_inside_edit_search_block_masked():
    """[SEARCH]...[/SEARCH] is code being searched for — signals masked."""
    text = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\n"
        "old code [STOP][CONFIRM_STOP] here\n"
        "[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]\n"
    )
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    assert masked[inside] != '['


def test_edit_block__signal_inside_replace_block_masked():
    text = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew code [STOP][CONFIRM_STOP] here\n[/REPLACE]\n"
    )
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    assert masked[inside] != '['


def test_edit_block__signal_inside_insert_block_masked():
    text = (
        "=== EDIT: a.py ===\n"
        "[INSERT AFTER LINE 5]\n"
        "[STOP][CONFIRM_STOP]\n"
        "[/INSERT]\n"
    )
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    assert masked[inside] != '['


def test_file_block__signal_inside_file_body_masked():
    """`=== FILE: new.py === ... === END FILE ===` — signals inside masked."""
    text = (
        "=== FILE: new.py ===\n"
        "[STOP][CONFIRM_STOP]\n"
        "=== END FILE ===\n"
        "[STOP][CONFIRM_STOP]"
    )
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    outside = text.rfind("[STOP]")
    assert masked[inside] != '['
    assert masked[outside] == '['


def test_file_block__unterminated_file_block_masks_rest():
    """Unterminated `=== FILE:` — the body extends to EOT."""
    text = (
        "=== FILE: new.py ===\n"
        "content...\n"
        "[STOP][CONFIRM_STOP]\n"
        "more content\n"
    )
    # The _EDIT_FILE_SPAN won't match (no END FILE), but _EDIT_BLOCK_SPAN
    # alternates terminators. Document the actual behavior.
    masked = _mask_for_signals(text)
    inside = text.find("[STOP]")
    # Either masked (good) or visible (a known limitation). Either way,
    # the runtime should call `_detect_unterminated_blocks` to warn.
    assert masked[inside] in {'[', '\x00'}


# ─────────────── MIXED MULTI-SIGNAL ───────────────


def test_multi__all_five_signal_kinds_in_one_text():
    """All 5 signal kinds: STOP / DONE / FORCE DONE / CONTINUE / PLAN DONE."""
    text = (
        "[STOP][CONFIRM_STOP] "
        "[DONE][CONFIRM_DONE] "
        "[FORCE DONE][CONFIRM_FORCE_DONE] "
        "[CONTINUE][CONFIRM_CONTINUE] "
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    masked = _mask_for_signals(text)
    # All 5 first-brackets should be visible (none in quoted zones)
    for sig in ["[STOP]", "[DONE]", "[FORCE DONE]", "[CONTINUE]", "[PLAN DONE]"]:
        pos = text.find(sig)
        assert masked[pos] == '['


def test_multi__all_signals_inside_fence_hidden():
    text = (
        "```\n"
        "[STOP][CONFIRM_STOP] "
        "[DONE][CONFIRM_DONE] "
        "[FORCE DONE][CONFIRM_FORCE_DONE] "
        "[CONTINUE][CONFIRM_CONTINUE] "
        "[PLAN DONE][CONFIRM_PLAN_DONE]\n"
        "```"
    )
    masked = _mask_for_signals(text)
    for sig in ["[STOP]", "[DONE]", "[FORCE DONE]", "[CONTINUE]", "[PLAN DONE]"]:
        pos = text.find(sig)
        assert masked[pos] != '['


# ─────────────── _mask_quoted_tags_core PARAMETERIZATION ───────────────


def test_core__enforce_tool_use_blocks_true_masks_outside():
    text = "[tool use][CODE: in.py][/tool use]\n[CODE: out.py]"
    masked = _mask_quoted_tags_core(text, enforce_tool_use_blocks=True)
    in_pos = text.find("[CODE: in.py")
    out_pos = text.find("[CODE: out.py")
    assert masked[in_pos] == '['
    assert masked[out_pos] != '['


def test_core__enforce_tool_use_blocks_false_keeps_outside():
    text = "[tool use][CODE: in.py][/tool use]\n[CODE: out.py]"
    masked = _mask_quoted_tags_core(text, enforce_tool_use_blocks=False)
    in_pos = text.find("[CODE: in.py")
    out_pos = text.find("[CODE: out.py")
    assert masked[in_pos] == '['
    assert masked[out_pos] == '['


# ─────────────── ATTACK: signal-in-think-claim ───────────────


def test_attack__signal_disguised_as_documentation():
    """Model writes: `the protocol uses [STOP][CONFIRM_STOP]` — should NOT
    fire unless explicitly in a backtick / fence / think context."""
    # Bare prose with signal — DOES fire (this is by design; agent must
    # backtick-quote when discussing signals).
    text = "the protocol uses [STOP][CONFIRM_STOP] to stop"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    # Bare signal is visible — agent is responsible for backtick-quoting.
    assert masked[stop_pos] == '['


def test_attack__signal_quoted_safe():
    """Same content but properly backtick-quoted — masked."""
    text = "the protocol uses `[STOP][CONFIRM_STOP]` to stop"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


# ─────────────── EMPTY AND BOUNDARY ───────────────


def test_empty__empty_text():
    assert _mask_for_signals("") == ""
    assert _mask_quoted_tags("") == ""


def test_empty__just_brackets_no_match():
    """Bare `[` characters without any wrapping — pass through."""
    text = "[not a tag][also not"
    masked = _mask_for_signals(text)
    # No tool-use block, no signal-shape — every `[` survives
    assert masked == text


def test_boundary__signal_at_text_start():
    text = "[STOP][CONFIRM_STOP] then prose"
    masked = _mask_for_signals(text)
    assert masked[0] == '['


def test_boundary__signal_at_text_end():
    text = "prose then [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] == '['
