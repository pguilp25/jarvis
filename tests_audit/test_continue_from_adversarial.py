"""ADVERSARIAL SECOND-PASS audit of `_apply_continue_from`.

The directive `[continue from: -N]` erases the N lines IMMEDIATELY
PRECEDING it (plus the directive's own line). The function applies
the FIRST directive, then loops to find the next one in the REWRITTEN
text — directives chain.

Critical properties:
  • Directives inside masked contexts (think/fence/backtick) are DOCUMENTATION
    and must NOT fire.
  • N=0 or N>500 — malformed; directive stripped without erasing content.
  • Each directive operates on the state AFTER previous directives applied.

Adversarial:
  • Nested directives interacting with chained backticks.
  • Directive in a quoted string that mentions the syntax.
  • Multiple directives in document order.
  • Directive at line 0 (no content above).
  • Whitespace + tabs variations.
"""
import pytest
from core.tool_call import _apply_continue_from


# ─────────────── BASIC ───────────────


def test_basic__erase_3_lines():
    text = "A\nB\nC\nD\n[continue from: -3]\nE"
    out = _apply_continue_from(text)
    # Lines B, C, D erased + directive line erased
    # Result: A + (newline) + E... actually let's check exact format
    # newline_positions in "A\nB\nC\nD\n" = [1, 3, 5, 7]
    # n=3, needed = 4 → newline_positions[-4] = 1 → cut_at = 2 (start of B)
    # So everything from B onwards through the directive is erased
    assert "B" not in out
    assert "C" not in out
    assert "D" not in out
    assert "A" in out
    assert "E" in out


def test_basic__erase_zero_invalid():
    """N=0 → directive stripped, no content erased."""
    text = "A\nB\n[continue from: -0]\nC"
    out = _apply_continue_from(text)
    # All content preserved (directive itself removed)
    assert "A" in out
    assert "B" in out
    assert "C" in out
    assert "[continue from" not in out


def test_basic__no_directive_passthrough():
    text = "no directive\nin here\n"
    assert _apply_continue_from(text) == text


# ─────────────── INERT-ZONE PROTECTION ───────────────


def test_inert__inside_fenced_no_fire():
    """Directive in a fenced code block is documentation, not command."""
    text = (
        "real content\n"
        "```\n"
        "[continue from: -100]\n"
        "```\n"
        "more content"
    )
    out = _apply_continue_from(text)
    # No content erased; directive preserved as documentation
    assert "real content" in out
    assert "more content" in out
    # The directive itself stays as part of the fenced block
    assert "[continue from: -100]" in out


def test_inert__inside_think_bracket_no_fire():
    text = (
        "real content\n"
        "[think]\n"
        "[continue from: -100]\n"
        "[/think]\n"
        "more content"
    )
    out = _apply_continue_from(text)
    assert "real content" in out
    assert "more content" in out
    assert "[continue from: -100]" in out


def test_inert__inside_think_xml_no_fire():
    text = (
        "real content\n"
        "<think>[continue from: -100]</think>\n"
        "more content"
    )
    out = _apply_continue_from(text)
    assert "real content" in out
    assert "more content" in out


def test_inert__inside_inline_backtick_no_fire():
    text = "real `[continue from: -5]` syntax explained"
    out = _apply_continue_from(text)
    assert "real " in out
    assert "syntax explained" in out


# ─────────────── REAL DIRECTIVES FIRE OUTSIDE INERT ZONES ───────────────


def test_real__outside_inert_zone_fires():
    """A real directive outside any inert zone — fires."""
    text = "A\nB\nC\nD\n[continue from: -2]\nE"
    out = _apply_continue_from(text)
    assert "C" not in out
    assert "D" not in out
    assert "A" in out
    assert "B" in out
    assert "E" in out


def test_real__after_inert_fires():
    """Directive after a fenced block — the directive itself is real."""
    text = (
        "A\nB\n```\nexample\n```\n"
        "C\nD\n[continue from: -2]\nE"
    )
    out = _apply_continue_from(text)
    # C and D erased; directive erased
    assert "C" not in out
    assert "D" not in out
    assert "A" in out
    assert "B" in out
    assert "E" in out


# ─────────────── MALFORMED ───────────────


def test_malformed__n_too_large_stripped():
    """N > 500 → malformed → directive stripped without erasing."""
    text = "A\nB\n[continue from: -9999]\nC"
    out = _apply_continue_from(text)
    # All content preserved
    assert "A" in out
    assert "B" in out
    assert "C" in out
    assert "[continue from" not in out


def test_malformed__negative_n_in_syntax_not_supported():
    """`[continue from: 5]` (no minus) — should not match the regex."""
    text = "A\nB\n[continue from: 5]\nC"
    out = _apply_continue_from(text)
    # Pattern requires `-`, so this doesn't match → text unchanged
    assert out == text


def test_malformed__non_numeric_arg():
    """`[continue from: -abc]` — non-numeric, no match."""
    text = "A\nB\n[continue from: -abc]\nC"
    out = _apply_continue_from(text)
    assert out == text


# ─────────────── BOUNDARIES ───────────────


def test_boundary__directive_at_top_no_content_above():
    """Directive at first line — cut_at=0 (erase all prefix, but nothing to erase)."""
    text = "[continue from: -5]\ncontent"
    out = _apply_continue_from(text)
    assert "[continue from" not in out
    assert "content" in out


def test_boundary__erase_more_than_exists():
    """N=100 but only 3 lines above → cut_at = 0 (erase everything)."""
    text = "A\nB\nC\n[continue from: -100]\nD"
    out = _apply_continue_from(text)
    # A, B, C all erased — D survives
    assert "A" not in out
    assert "B" not in out
    assert "C" not in out
    assert "D" in out


def test_boundary__exactly_at_n_500():
    """N=500 is the upper bound (inclusive)."""
    text = "x\n" * 500 + "[continue from: -500]\ntail"
    out = _apply_continue_from(text)
    assert "tail" in out
    # All 500 x's erased
    assert out.count("x\n") == 0 or out.count("x") < 500


def test_boundary__n_501_too_large():
    """N=501 — over the limit → stripped."""
    text = "A\n[continue from: -501]\nB"
    out = _apply_continue_from(text)
    assert "A" in out
    assert "B" in out
    assert "[continue from" not in out


# ─────────────── CHAINING ───────────────


def test_chain__two_directives_applied_in_order():
    """First directive erases B, C; in the new state, second directive
    operates and erases the 2 lines immediately above it (E, F)."""
    text = (
        "A\nB\nC\n[continue from: -2]\n"  # erases B, C, and directive
        "D\nE\nF\n[continue from: -2]\n"  # erases E, F, and 2nd directive
        "G"
    )
    out = _apply_continue_from(text)
    # After both: A, D, G survive; B, C, E, F erased
    assert "A" in out
    assert "D" in out
    assert "G" in out
    assert "B" not in out
    assert "C" not in out
    assert "E" not in out
    assert "F" not in out


def test_chain__three_directives_chain():
    text = (
        "A\nB\n[continue from: -1]\n"  # erases B + directive
        "C\nD\n[continue from: -1]\n"  # erases D + directive
        "E\nF\n[continue from: -1]\n"  # erases F + directive
        "G"
    )
    out = _apply_continue_from(text)
    assert "A" in out and "C" in out and "E" in out and "G" in out
    assert "B" not in out and "D" not in out and "F" not in out


# ─────────────── WHITESPACE ───────────────


def test_ws__directive_with_spaces():
    """`[continue from:   -3]` — extra whitespace around N."""
    text = "A\nB\nC\nD\n[continue from:   -3]\nE"
    out = _apply_continue_from(text)
    # Should still match — the regex is lenient
    assert "E" in out


def test_ws__directive_uppercase():
    """Case-insensitive match."""
    text = "A\nB\n[CONTINUE FROM: -1]\nC"
    out = _apply_continue_from(text)
    assert "C" in out


def test_ws__directive_mixed_case():
    text = "A\nB\n[Continue From: -1]\nC"
    out = _apply_continue_from(text)
    assert "C" in out


# ─────────────── CRLF / EDGE LINE ENDINGS ───────────────


def test_ws__crlf_endings():
    """CRLF endings — directive works on \\n boundaries.
    With N=1, the line IMMEDIATELY before the directive is erased.
    For `A\\r\\nB\\r\\nC\\r\\n[continue from: -1]\\r\\nD`, that's `C`."""
    text = "A\r\nB\r\nC\r\n[continue from: -1]\r\nD"
    out = _apply_continue_from(text)
    # C is the line just before the directive → erased
    assert "C" not in out
    # A, B preserved; D preserved
    assert "A" in out
    assert "B" in out
    assert "D" in out


# ─────────────── IDEMPOTENCE / NO-OP ───────────────


def test_idem__text_with_no_directives_unchanged():
    text = "totally\nclean\ntext\n"
    assert _apply_continue_from(text) == text


def test_idem__after_full_apply_no_more_directives():
    """Running again on output should be a no-op (no remaining directives)."""
    text = "A\nB\nC\n[continue from: -2]\nD"
    once = _apply_continue_from(text)
    twice = _apply_continue_from(once)
    assert once == twice


# ─────────────── ADVERSARIAL ───────────────


def test_adv__directive_immediately_at_start():
    text = "[continue from: -5]"
    out = _apply_continue_from(text)
    # Directive at start, nothing else → just removed
    assert out == ""


def test_adv__multiple_directives_one_in_inert():
    """One real, one inside fence — only real fires."""
    text = (
        "A\nB\n"
        "```\n[continue from: -100]\n```\n"
        "C\n[continue from: -1]\n"  # real → erases C and directive
        "D"
    )
    out = _apply_continue_from(text)
    assert "A" in out
    assert "B" in out
    assert "D" in out
    # C erased
    assert "C" not in out
    # Fenced documentation preserved
    assert "[continue from: -100]" in out


def test_adv__directive_with_trailing_content_same_line():
    """`[continue from: -2] more text` — same line content after."""
    text = "A\nB\nC\n[continue from: -2] same line more\nD"
    out = _apply_continue_from(text)
    # Directive removed, but "same line more" might be preserved
    # depending on implementation. Document either way.
    assert "A" in out
    assert "D" in out


def test_adv__unicode_around_directive():
    text = "北京\n中文\n测试\n[continue from: -2]\nresume"
    out = _apply_continue_from(text)
    assert "北京" in out
    assert "中文" not in out
    assert "测试" not in out
    assert "resume" in out


def test_adv__1000_lines_then_directive():
    """Big text — directive at end erases last 100."""
    text = "\n".join(f"line_{i}" for i in range(1000)) + "\n[continue from: -100]\nend"
    out = _apply_continue_from(text)
    # Lines 900-999 erased
    assert "line_899" in out  # boundary kept
    assert "line_950" not in out
    assert "end" in out


def test_adv__empty_text():
    assert _apply_continue_from("") == ""


def test_adv__only_directive_no_other_content():
    text = "[continue from: -1]"
    out = _apply_continue_from(text)
    assert "[continue from" not in out


# ─────────────── LENGTH INVARIANT ───────────────


def test_inv__output_never_longer_than_input():
    """The directive can only ERASE content, never add. Output length ≤ input."""
    cases = [
        "A\nB\nC\n[continue from: -2]\nD",
        "X" * 1000 + "\n[continue from: -1]\n" + "Y" * 100,
        "[continue from: -5]\nbody",
        "nothing here",
    ]
    for text in cases:
        out = _apply_continue_from(text)
        assert len(out) <= len(text), f"Output longer: input={len(text)} output={len(out)}"


def test_inv__directive_always_removed_when_processed():
    """Any FIRED directive must be absent from output."""
    text = "A\nB\nC\n[continue from: -2]\nD"
    out = _apply_continue_from(text)
    assert "[continue from" not in out
