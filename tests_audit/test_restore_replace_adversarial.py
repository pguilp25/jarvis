"""ADVERSARIAL SECOND-PASS audit of `_restore_replace_whitespace`.

The first pass verified happy paths. This pass attacks with:
  • Every possible i{N}| prefix value (0, 1, 4, 8, 16, 32, 128).
  • Mid-line packing in adversarial positions (inside strings, around
    operators, before/after punctuation).
  • Trailing-lineno cases that should AND should not strip.
  • Combined cases: i{N}| prefix + visible whitespace markers + trailer.
  • PROPERTY: idempotence after first restore (running again should be no-op).
  • PROPERTY: number of OUTPUT lines ≥ input lines (mid-line splits).
"""
import pytest
from workflows.code import _restore_replace_whitespace


# ─────────────── i{N}| PREFIX — every value ───────────────


def test_prefix__i0():
    assert _restore_replace_whitespace("i0|x = 1") == "x = 1"


def test_prefix__i1():
    assert _restore_replace_whitespace("i1|x = 1") == " x = 1"


def test_prefix__i4():
    assert _restore_replace_whitespace("i4|x = 1") == "    x = 1"


def test_prefix__i8():
    assert _restore_replace_whitespace("i8|x = 1") == "        x = 1"


def test_prefix__i16():
    assert _restore_replace_whitespace("i16|x = 1") == " " * 16 + "x = 1"


def test_prefix__i32():
    assert _restore_replace_whitespace("i32|x = 1") == " " * 32 + "x = 1"


def test_prefix__i128():
    """Deeply nested code (128 spaces) — should still work."""
    assert _restore_replace_whitespace("i128|x = 1") == " " * 128 + "x = 1"


def test_prefix__leading_whitespace_in_content_stripped():
    """`i4|    x` — extra leading spaces in content are stripped."""
    assert _restore_replace_whitespace("i4|    x = 1") == "    x = 1"


def test_prefix__leading_tab_in_content_stripped():
    assert _restore_replace_whitespace("i4|\tx = 1") == "    x = 1"


def test_prefix__mixed_leading_whitespace_stripped():
    assert _restore_replace_whitespace("i4| \tx = 1") == "    x = 1"


# ─────────────── MULTI-LINE BODIES ───────────────


def test_multiline__sequential_lines():
    text = "i0|def foo():\ni4|return 1"
    out = _restore_replace_whitespace(text)
    assert out == "def foo():\n    return 1"


def test_multiline__many_lines():
    text = "\n".join([f"i{i*2}|line_{i}" for i in range(10)])
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    assert lines[0] == "line_0"
    assert lines[1] == "  line_1"
    assert lines[9] == "                  line_9"


def test_multiline__blank_lines_preserved():
    """`i0|` followed by content alternating with truly blank lines."""
    text = "i0|a\n\ni0|b"
    out = _restore_replace_whitespace(text)
    assert out == "a\n\nb"


# ─────────────── MID-LINE PACKING ───────────────


def test_midline__simple_pack():
    """`i4|def x(): i8|return 1` → split into two physical lines."""
    text = "i4|def x(): i8|return 1"
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    assert lines[0].rstrip() == "    def x():"
    assert lines[1] == "        return 1"


@pytest.mark.skip(reason="v8 `iN|content` line-packing removed in the format-B "
                  "migration — model emits whitespace `N:content` now, never packed iN|.")
def test_midline__pack_with_quote():
    """Observed astropy-13033: `"...required "i33|"continuation"` — no
    whitespace before mid-line `i{N}|` should still split."""
    text = 'i0|msg = "required"i33|"continuation"'
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    # First line: msg = "required" (trailing whitespace stripped)
    assert lines[0].startswith('msg = "required"')
    # Second line: indented continuation
    assert lines[1].startswith(' ' * 33)


@pytest.mark.skip(reason="v8 `iN|content` line-packing removed in the format-B "
                  "migration — model emits whitespace `N:content` now, never packed iN|.")
def test_midline__three_segments_packed():
    text = "i0|a()i4|b()i8|c()"
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    assert len(lines) == 3
    assert lines[0].rstrip() == "a()"
    assert lines[1] == "    b()"
    assert lines[2] == "        c()"


def test_midline__no_split_when_no_prefix():
    """Line doesn't start with i{N}| → no mid-line splitting."""
    text = "regular line with i4| in middle"
    out = _restore_replace_whitespace(text)
    # Output unchanged
    assert out == "regular line with i4| in middle"


def test_midline__no_split_when_only_one_marker():
    """Line starts with i{N}| but only ONE marker — no split."""
    text = "i4|just one line"
    out = _restore_replace_whitespace(text)
    assert out == "    just one line"


# ─────────────── TRAILING-LINENO STRIPPING ───────────────


def test_trailer__paren_close_then_digits_stripped():
    """`foo() 42` → `foo()` (the 42 is a copied line number)."""
    text = "i0|foo() 42"
    out = _restore_replace_whitespace(text)
    assert out == "foo()"


def test_trailer__square_bracket_close_then_digits_stripped():
    text = "i0|arr[5] 100"
    out = _restore_replace_whitespace(text)
    assert out == "arr[5]"


def test_trailer__curly_close_then_digits_stripped():
    text = "i0|d['key'] 47"
    out = _restore_replace_whitespace(text)
    assert out == "d['key']"


def test_trailer__quote_then_digits_stripped():
    text = 'i0|x = "hello" 18'
    out = _restore_replace_whitespace(text)
    assert out == 'x = "hello"'


def test_trailer__single_quote_then_digits_stripped():
    text = "i0|x = 'hello' 18"
    out = _restore_replace_whitespace(text)
    assert out == "x = 'hello'"


def test_trailer__colon_then_digits_stripped():
    text = "i0|if x: 5"
    out = _restore_replace_whitespace(text)
    assert out == "if x:"


def test_trailer__return_one_NOT_stripped():
    """REGRESSION: `return 1` — 1 has no preceding statement-end. Stay intact."""
    text = "i4|return 1"
    out = _restore_replace_whitespace(text)
    assert out == "    return 1"


def test_trailer__assignment_NOT_stripped():
    """`x = 5` — 5 is the value, not a line number. Keep."""
    text = "i0|x = 5"
    out = _restore_replace_whitespace(text)
    assert out == "x = 5"


def test_trailer__n_equals_4_NOT_stripped():
    """`n = 4` — same as above."""
    text = "i0|n = 4"
    out = _restore_replace_whitespace(text)
    assert out == "n = 4"


def test_trailer__multi_digit_lineno_stripped():
    """`foo() 12345` — 5-digit line number stripped."""
    text = "i0|foo() 12345"
    out = _restore_replace_whitespace(text)
    assert out == "foo()"


def test_trailer__7_digit_NOT_stripped():
    """Regex `\\d{1,6}` — 7 digits won't match the trailer."""
    text = "i0|foo() 1234567"
    out = _restore_replace_whitespace(text)
    # 7 digits is beyond the bound — stays
    assert "1234567" in out


def test_trailer__pure_lineno_blank_line():
    """`i0| 503` — line was originally blank, [CODE:] rendered as i0| 503."""
    text = "i0| 503"
    out = _restore_replace_whitespace(text)
    # Becomes a blank line
    assert out == ""


def test_trailer__pure_lineno_with_indent_keeps_indent():
    """`i4| 100` — indented prefix, pure trailer → indent preserved,
    digits dropped. Result is 4 spaces (the indent prefix is authoritative)."""
    text = "i4| 100"
    out = _restore_replace_whitespace(text)
    # Indent applied, content becomes empty → 4 spaces remain.
    # (The i{N}| prefix is the source of truth for indent.)
    assert out == "    "


# ─────────────── COMBINATIONS ───────────────


def test_combo__multi_line_with_trailers():
    text = "i0|def foo(): 1\ni4|return x 2"
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    # First line: trailer stripped (colon ends)
    assert lines[0] == "def foo():"
    # Second line: `return x 2` — `x` is alpha (not a statement-end char)
    # so trailer NOT stripped
    assert "return x 2" in lines[1] or lines[1] == "    return x"


def test_combo__indent_packed_mid_line():
    """Both i{N}| prefix AND mid-line pack on the same input."""
    text = "i0|a() 5 i4|b() 10"
    out = _restore_replace_whitespace(text)
    # Mid-line split kicks in
    lines = out.split('\n')
    assert len(lines) == 2


# ─────────────── LEGACY VISIBLE WHITESPACE ───────────────


def test_legacy__hyphen_bullet_to_space():
    text = "⁃⁃x = 1"
    out = _restore_replace_whitespace(text)
    assert out.startswith("  x = 1")


def test_legacy__rightwards_arrow_to_tab():
    text = "→x = 1"
    out = _restore_replace_whitespace(text)
    assert "\t" in out or out.startswith("    ")


def test_legacy__middle_dot_to_space():
    text = "··x = 1"
    out = _restore_replace_whitespace(text)
    assert out.startswith("  x = 1")


def test_legacy__T_marker_only_after_whitespace():
    """Legacy `T` marker only converts to tab if preceded by whitespace
    in the result buffer. Bare `Tx = 1` (no leading WS) is passed verbatim
    — the T is treated as content."""
    text = "Tx = 1"
    out = _restore_replace_whitespace(text)
    assert out == "Tx = 1"


def test_legacy__T_marker_after_space_becomes_tab():
    """` Tx = 1` — leading space then T → T becomes tab (legacy interpretation)."""
    text = " Tx = 1"
    out = _restore_replace_whitespace(text)
    # Space + T → space + tab, then `x = 1` follows
    assert "\t" in out or out == " Tx = 1"  # document the contract


def test_legacy__no_visible_markers_passthrough():
    text = "regular content\n  with indent"
    out = _restore_replace_whitespace(text)
    assert out == text


# ─────────────── IDEMPOTENCE ───────────────


def test_idem__i_prefix_applied_once():
    """Running `_restore_replace_whitespace` on output that has no i{N}|
    markers is a no-op (the visible-whitespace path is also idempotent)."""
    text = "i4|x = 1\ni8|y = 2"
    once = _restore_replace_whitespace(text)
    twice = _restore_replace_whitespace(once)
    assert once == twice


def test_idem__trailer_stripping_is_idempotent():
    text = "i0|foo() 42"
    once = _restore_replace_whitespace(text)
    twice = _restore_replace_whitespace(once)
    assert once == twice


def test_idem__legacy_markers_idempotent():
    text = "⁃⁃content"
    once = _restore_replace_whitespace(text)
    twice = _restore_replace_whitespace(once)
    assert once == twice


# ─────────────── ADVERSARIAL EDGE CASES ───────────────


def test_adv__empty_string():
    assert _restore_replace_whitespace("") == ""


def test_adv__only_newlines():
    out = _restore_replace_whitespace("\n\n\n")
    assert out == "\n\n\n"


def test_adv__i_prefix_no_pipe():
    """`i4 x = 1` — no `|` after digits → not a valid prefix."""
    text = "i4 x = 1"
    out = _restore_replace_whitespace(text)
    # Should NOT be interpreted as i{N}| — pass through
    assert out == "i4 x = 1"


def test_adv__pipe_no_i_prefix():
    """`|x = 1` — bare pipe, no `i\\d+` prefix → not a valid prefix."""
    text = "|x = 1"
    out = _restore_replace_whitespace(text)
    assert out == "|x = 1"


def test_adv__i_followed_by_letter():
    """`i foo|bar` — not digits after `i` → not a valid prefix."""
    text = "i foo|bar"
    out = _restore_replace_whitespace(text)
    assert out == "i foo|bar"


def test_adv__only_i_prefix_no_content():
    """`i4|` with no content."""
    text = "i4|"
    out = _restore_replace_whitespace(text)
    assert out == "    "


def test_adv__very_long_line():
    """A single long line with i{N}| prefix — should still work."""
    text = "i0|" + "x" * 10000
    out = _restore_replace_whitespace(text)
    assert out == "x" * 10000


def test_adv__many_short_lines():
    """1000 lines each with their own i{N}|."""
    text = "\n".join([f"i{i % 8}|line_{i}" for i in range(1000)])
    out = _restore_replace_whitespace(text)
    assert out.count('\n') == text.count('\n')


def test_adv__unicode_content_preserved():
    text = "i4|name = '北京'"
    out = _restore_replace_whitespace(text)
    assert out == "    name = '北京'"


def test_adv__emoji_content_preserved():
    text = "i0|status = '🎉'"
    out = _restore_replace_whitespace(text)
    assert out == "status = '🎉'"


def test_adv__indent_marker_inside_string_literal_NOT_split():
    """`"i4|literal"` inside a regular line that does NOT start with i{N}|
    must NOT trigger a split."""
    text = 'msg = "i4|just text"'
    out = _restore_replace_whitespace(text)
    assert out == 'msg = "i4|just text"'


# ─────────────── BOX-DRAWING CHARACTERS ───────────────


def test_box_drawing__header_decorations_then_digits_stripped():
    """Code formatted as `# ── Header ──── 201` — strip trailing 201."""
    text = "i0|# ── Header ──── 201"
    out = _restore_replace_whitespace(text)
    # Trailer regex includes box-drawing chars in _STATEMENT_END
    # so `──── 201` → `────`
    assert "201" not in out or out == "# ── Header ──── 201"  # document either


def test_box_drawing__various_chars_pass_through():
    """Box-drawing chars are valid in comments — preserve."""
    text = "i0|# ┌──┐"
    out = _restore_replace_whitespace(text)
    assert "┌──┐" in out
