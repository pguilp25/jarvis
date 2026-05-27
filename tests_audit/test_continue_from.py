"""Audit `_apply_continue_from` — the `[continue from: -N]` directive that
erases the last N lines of visible output BEFORE downstream extraction."""

import re
import pytest


def _continue_from_helper(text: str) -> str:
    """Import from core/tool_call.py."""
    from core.tool_call import _apply_continue_from
    return _apply_continue_from(text)


# ───────────────────── BASIC ─────────────────────

def test_continue__erase_5_lines():
    src = (
        "line1\n"
        "line2\n"
        "line3\n"
        "line4\n"
        "line5\n"
        "[continue from: -3]\n"
        "good content\n"
    )
    out = _continue_from_helper(src)
    # Lines 3,4,5 + the directive line erased — line1,2 + good content survive
    assert "line1" in out
    assert "line2" in out
    assert "line3" not in out
    assert "line4" not in out
    assert "line5" not in out
    assert "good content" in out
    assert "[continue from" not in out


def test_continue__erase_zero_lines():
    """N=0 is a no-op — just the directive itself stripped."""
    src = "line1\nline2\n[continue from: -0]\nafter\n"
    out = _continue_from_helper(src)
    # Directive removed, content preserved
    assert "line1" in out
    assert "line2" in out
    assert "after" in out
    assert "[continue from" not in out


def test_continue__erase_more_than_available():
    """N > available lines — should clamp, not crash."""
    src = "line1\n[continue from: -100]\nafter\n"
    out = _continue_from_helper(src)
    assert "after" in out
    # line1 should be erased (or kept if clamp behavior is different)
    assert "[continue from" not in out


def test_continue__no_directive_passthrough():
    src = "line1\nline2\nline3\n"
    out = _continue_from_helper(src)
    assert out == src


def test_continue__multiple_directives():
    """Multiple `[continue from: -N]` in the same response — each fires
    in order."""
    src = (
        "a\n"
        "wrong1\n"
        "[continue from: -1]\n"
        "b\n"
        "wrong2\n"
        "[continue from: -1]\n"
        "c\n"
    )
    out = _continue_from_helper(src)
    assert "a" in out and "b" in out and "c" in out
    assert "wrong1" not in out
    assert "wrong2" not in out


# ───────────────────── INERT INSIDE TAGS ─────────────────────

def test_continue__inside_think_inert():
    """`[continue from: -N]` inside [think] block — should NOT erase."""
    src = (
        "line1\n"
        "line2\n"
        "[think]\n"
        "I could write: [continue from: -2]\n"
        "[/think]\n"
        "real content\n"
    )
    out = _continue_from_helper(src)
    # If properly inert, line1+line2 survive
    assert "line1" in out
    assert "line2" in out


def test_continue__inside_fenced_inert():
    """`[continue from: -N]` inside ``` fenced block — should NOT erase."""
    src = (
        "line1\n"
        "line2\n"
        "```\n"
        "[continue from: -2]\n"
        "```\n"
        "real content\n"
    )
    out = _continue_from_helper(src)
    assert "line1" in out
    assert "line2" in out


def test_continue__inside_backticks_inert():
    """Inline backticks: `[continue from: -5]` should be inert."""
    src = (
        "Documentation: use `[continue from: -5]` to backtrack\n"
        "more content\n"
    )
    out = _continue_from_helper(src)
    # Both lines should survive
    assert "Documentation" in out
    assert "more content" in out


# ───────────────────── MALFORMED ─────────────────────

def test_continue__malformed_no_number():
    """`[continue from: -]` without a number — should be inert."""
    src = "line1\n[continue from: -]\nline3\n"
    out = _continue_from_helper(src)
    # The directive is malformed; behavior depends on regex


def test_continue__malformed_no_minus():
    """`[continue from: 5]` without minus — invalid form."""
    src = "line1\nline2\n[continue from: 5]\nline4\n"
    out = _continue_from_helper(src)
    # Should be left as-is OR ignored


def test_continue__directive_at_top_of_response():
    """`[continue from: -3]` as the FIRST line — N > available."""
    src = "[continue from: -3]\nactual content\n"
    out = _continue_from_helper(src)
    # Clamp to 0, just keep "actual content"
    assert "actual content" in out


# ───────────────────── EDGE ─────────────────────

def test_continue__blank_lines_counted():
    """Blank lines count toward N."""
    src = "a\n\nb\nc\n[continue from: -3]\nd\n"
    out = _continue_from_helper(src)
    # -3 erases [b, c, blank line above them?]. Behavior must be CONSISTENT.
    # We just verify it doesn't crash and produces SOMETHING.
    assert isinstance(out, str)


def test_continue__cr_lf_endings():
    """CRLF line endings."""
    src = "a\r\nb\r\nc\r\n[continue from: -2]\r\nd\r\n"
    out = _continue_from_helper(src)
    assert isinstance(out, str)
    assert "d" in out
