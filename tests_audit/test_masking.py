"""Audit `_mask_quoted_tags` and `_mask_for_signals`.

These are the security-critical functions that prevent prompt injection
from the model's own output. A model that types

    Here's how the protocol works: [STOP][CONFIRM_STOP] aborts the round.

inside a `<think>` block, fenced code block, or backtick span MUST NOT
trigger the runtime — the model is documenting the protocol, not invoking
it.

Bugs in masking → model thinks it's documenting → runtime fires the
signal → stream aborts mid-thought → silent failure.
"""
import pytest
from core.tool_call import (
    _mask_quoted_tags,
    _mask_for_signals,
)


def _is_masked(masked: str, original: str, snippet: str) -> bool:
    """Returns True if `snippet`'s `[` characters are masked (replaced
    with \\x00) in `masked` vs `original`."""
    idx = original.find(snippet)
    if idx < 0:
        return False
    # Check that the first `[` of the snippet was masked.
    bracket_idx = original.find('[', idx)
    if bracket_idx < 0 or bracket_idx >= idx + len(snippet):
        return False
    return masked[bracket_idx] != '['


# ─────────────── _mask_quoted_tags — fence / backtick / think ───────────────


def test_mask__inside_fenced_block():
    text = "```\n[CODE: secret.py]\n```\n[tool use][CODE: real.py][/tool use]"
    masked = _mask_quoted_tags(text)
    # The bracket inside the fence should be masked
    fence_pos = text.find("[CODE: secret.py]")
    assert masked[fence_pos] != '['
    # The real tag should NOT be masked
    real_pos = text.find("[CODE: real.py]")
    assert masked[real_pos] == '['


def test_mask__inside_inline_backticks():
    text = "use `[CODE: docs.py]` syntax; [tool use][CODE: real.py][/tool use]"
    masked = _mask_quoted_tags(text)
    quoted_pos = text.find("[CODE: docs.py]")
    assert masked[quoted_pos] != '['


def test_mask__inside_think_bracket():
    text = "[think][CODE: hidden.py][/think] [tool use][CODE: real.py][/tool use]"
    masked = _mask_quoted_tags(text)
    hidden_pos = text.find("[CODE: hidden.py]")
    assert masked[hidden_pos] != '['


def test_mask__inside_think_xml():
    text = "<think>[CODE: hidden.py]</think> [tool use][CODE: real.py][/tool use]"
    masked = _mask_quoted_tags(text)
    hidden_pos = text.find("[CODE: hidden.py]")
    assert masked[hidden_pos] != '['


def test_mask__outside_tool_use_block_masked_for_extraction():
    """When AT LEAST ONE [tool use] block exists, tags outside it are
    masked. (If no [tool use] block exists at all, enforcement is a
    no-op — the model hasn't opted into the protocol yet.)"""
    text = (
        "[CODE: lonely.py]\n"
        "[tool use][REFS: real_func][/tool use]"
    )
    masked = _mask_quoted_tags(text)
    # The lonely tag (outside [tool use]) is masked
    lonely_pos = text.find("[CODE:")
    assert masked[lonely_pos] != '['
    # The real tag (inside [tool use]) survives
    real_pos = text.find("[REFS:")
    assert masked[real_pos] == '['


def test_mask__no_tool_use_block_no_enforcement():
    """If no [tool use] block exists in the text, the enforcement is a
    no-op (otherwise legacy callers without the wrapper would break)."""
    text = "[CODE: lonely.py]"
    masked = _mask_quoted_tags(text)
    # No tool-use block exists → enforcement no-op → bracket visible
    assert masked[0] == '['


def test_mask__signal_outside_tool_use_visible():
    """`_mask_for_signals` (NO tool-use enforcement) leaves signals
    written outside [tool use] visible — that's their canonical position."""
    text = "[STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    # First [ is the STOP's [
    assert masked[0] == '['


def test_mask__signal_inside_fence_HIDDEN_for_signals_too():
    """Even with signal masking (no tool-use enforcement), signals inside
    a fence MUST be hidden. Otherwise the protocol-documentation problem
    fires."""
    text = "```\nProtocol: [STOP][CONFIRM_STOP] aborts.\n```"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_mask__signal_inside_think_hidden_for_signals():
    text = "<think>I should not write [STOP][CONFIRM_STOP] here</think>"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_mask__signal_inside_backticks_hidden_for_signals():
    text = "Two-tag protocol: `[STOP][CONFIRM_STOP]`"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


# ─────────────── UNCLOSED ───────────────


def test_mask__unclosed_think_blocks_all_after():
    """An unclosed <think> at end masks everything from open to EOT
    so partial-stream signals don't fire prematurely."""
    text = "before <think>\nstreaming... [STOP][CONFIRM_STOP] not arrived yet"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_mask__unclosed_fence_blocks_all_after():
    text = "before ```\nin fence [STOP][CONFIRM_STOP] streaming"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


def test_mask__unclosed_backtick_blocks_eol():
    """Inline backtick unclosed → mask to end of line (only)."""
    text = "Protocol: `[STOP][CONFIRM_STOP]\nNew line should be OK"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    assert masked[stop_pos] != '['


# ─────────────── PLAN BODY MASKING ───────────────


def test_mask__signal_inside_plan_block_hidden():
    """Signals inside `=== PLAN === ... === END PLAN ===` are data,
    not commands — masked even by signal extractor."""
    text = (
        "=== PLAN ===\n"
        "Example: [PLAN DONE][CONFIRM_PLAN_DONE] commits.\n"
        "=== END PLAN ===\n"
        "[PLAN DONE][CONFIRM_PLAN_DONE]"
    )
    masked = _mask_for_signals(text)
    # The first occurrence (inside plan body) is masked
    inside_pos = text.find("[PLAN DONE]")
    assert masked[inside_pos] != '['
    # The second occurrence (after END PLAN) is visible
    outside_pos = text.find("[PLAN DONE]", inside_pos + 1)
    assert masked[outside_pos] == '['


def test_mask__unclosed_plan_blocks_all_after():
    """Unclosed `=== PLAN ===` (no END marker) — mask from open to EOT."""
    text = "=== PLAN ===\nstreaming... [PLAN DONE][CONFIRM_PLAN_DONE]"
    masked = _mask_for_signals(text)
    pd_pos = text.find("[PLAN DONE]")
    assert masked[pd_pos] != '['


# ─────────────── EDIT BLOCK MASKING ───────────────


def test_mask__signal_inside_edit_block_hidden():
    """Signals inside `=== EDIT: ... [SEARCH] ... [/SEARCH]` are masked —
    the model might be illustrating an edit that mentions the protocol."""
    text = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold [STOP][CONFIRM_STOP] code\n[/SEARCH]\n"
        "[REPLACE]\nnew code\n[/REPLACE]"
    )
    masked = _mask_for_signals(text)
    inside_pos = text.find("[STOP]")
    assert masked[inside_pos] != '['


# ─────────────── EXPLICIT ESCAPES ───────────────


def test_mask__backslash_escape_masks_bracket():
    r"""`\[STOP][CONFIRM_STOP]` — the leading `\[` is an explicit escape."""
    text = "writing about \\[STOP]\\[CONFIRM_STOP] safely"
    masked = _mask_for_signals(text)
    stop_pos = text.find("[STOP]")
    # `\[` should mask the `[`
    assert masked[stop_pos] != '['


# ─────────────── DIDN'T MASK WHAT IT SHOULDN'T ───────────────


def test_mask__plain_text_unchanged():
    text = "plain text with no brackets at all"
    assert _mask_for_signals(text) == text


def test_mask__legit_tool_call_unchanged():
    text = "[tool use][CODE: a.py][/tool use]"
    masked = _mask_quoted_tags(text)
    # The actual tag's brackets remain `[`
    code_pos = text.find("[CODE:")
    assert masked[code_pos] == '['


def test_mask__multiple_tags_in_tool_use_all_visible():
    text = "[tool use][CODE: a.py] [REFS: foo] [SEARCH: bar][/tool use]"
    masked = _mask_quoted_tags(text)
    # All three opening `[` of the actual tags should be visible
    code_pos = text.find("[CODE:")
    refs_pos = text.find("[REFS:")
    search_pos = text.find("[SEARCH:")
    assert masked[code_pos] == '['
    assert masked[refs_pos] == '['
    assert masked[search_pos] == '['


# ─────────────── REGRESSION / OBSERVED BUGS ───────────────


def test_mask__streaming_partial_backtick_protected():
    """Bug observed (glm-5.1): model streaming
        "Two-tag protocol: `[STOP][CONFIRM_STOP]`, `[DONE][CONFIRM_DONE]..."
    When [CONFIRM_DONE] arrived, the second ` was still unclosed (delta).
    Without the unclosed-backtick guard, DONE would fire. With the guard,
    both signals are masked while the line is incomplete."""
    text = "Two-tag protocol: `[STOP][CONFIRM_STOP]`, `[DONE][CONFIRM_DONE]"
    masked = _mask_for_signals(text)
    done_pos = text.find("[DONE]")
    assert masked[done_pos] != '['


def test_mask__triple_fence_distinguished_from_inline():
    """``` opens a fence (multiline), ` opens an inline (single-line).
    Both should mask their content, not interfere with each other."""
    text = (
        "Inline: `[STOP][CONFIRM_STOP]` then fence:\n"
        "```\n[DONE][CONFIRM_DONE]\n```\nReal: [STOP][CONFIRM_STOP]"
    )
    masked = _mask_for_signals(text)
    # Inline-quoted STOP masked
    first_stop = text.find("[STOP]")
    assert masked[first_stop] != '['
    # Fenced DONE masked
    done_pos = text.find("[DONE]")
    assert masked[done_pos] != '['
    # Real STOP at end visible
    last_stop = text.rfind("[STOP]")
    assert masked[last_stop] == '['
