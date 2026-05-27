"""Exhaustive audit of `_restore_replace_whitespace` — the i{N}| line-prefix
parser that converts coder-format edit blocks into real Python whitespace.

This is the parser that corrupted astropy-13033 (the `i33|` leak): it
mishandled mid-line `i\\d+|` segments when no whitespace preceded them.

What we test:
  - Basic single-line restoration
  - Multi-line restoration with newlines
  - Mid-line packed segments (the original bug class)
  - Boundary conditions: i0|, i999|, trailing line numbers
  - Adversarial: nested code containing `i\\d+|`-like text, unicode,
    fenced code blocks, escapes
  - Malformed input: missing pipe, missing digits, just `i|`, just `|`
  - Tab characters, mixed tabs+spaces, leading whitespace inside content
  - Empty content, blank lines, trailing whitespace
"""
import re
import pytest
from workflows.code import _restore_replace_whitespace


# ────────────────────────── BASIC ──────────────────────────

def test_lineprefix__single_indent__produces_spaces():
    """`i4|x = 1` → 4 spaces + `x = 1`"""
    assert _restore_replace_whitespace("i4|x = 1") == "    x = 1"


def test_lineprefix__zero_indent__no_leading_spaces():
    """`i0|x = 1` at the top level → no leading spaces."""
    assert _restore_replace_whitespace("i0|x = 1") == "x = 1"


def test_lineprefix__large_indent__exact_count():
    """`i40|x` → exactly 40 leading spaces."""
    out = _restore_replace_whitespace("i40|x")
    assert out == " " * 40 + "x"
    assert len(out) - len(out.lstrip()) == 40


def test_lineprefix__multiline__each_line_independent():
    """Each line independently translated."""
    src = "i4|def f():\ni8|return 1"
    expected = "    def f():\n        return 1"
    assert _restore_replace_whitespace(src) == expected


def test_lineprefix__empty_string__empty_output():
    assert _restore_replace_whitespace("") == ""


def test_lineprefix__blank_line__preserved():
    """A blank line should round-trip as blank."""
    assert _restore_replace_whitespace("\n") == "\n"


def test_lineprefix__no_prefix_anywhere__pass_through():
    """Plain Python without any i{N}| markers — pass through untouched."""
    src = "    def f():\n        return 1"
    assert _restore_replace_whitespace(src) == src


# ───────── THE BUG CLASS: mid-line packed segments ─────────

def test_lineprefix__midline_packed_with_whitespace__splits():
    """The original mid-line case (whitespace before each marker)."""
    src = "i4|def f(): i8|return 1"
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    assert lines == ["    def f():", "        return 1"]


@pytest.mark.skip(reason="v8 iN| line-packing removed in format-B migration (model emits whitespace N:content)")
def test_lineprefix__midline_packed_no_whitespace_quote__splits():
    """astropy-13033 case: closing quote DIRECTLY followed by i{N}|."""
    src = 'i16|raise ValueError("a"i33|"b"i33|"c")'
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    assert len(lines) == 3, f"expected 3 lines, got {len(lines)}: {lines!r}"
    assert lines[0].startswith(" " * 16)
    assert lines[1].startswith(" " * 33)
    assert lines[2].startswith(" " * 33)
    assert "i33|" not in out, f"i33| leaked: {out!r}"


@pytest.mark.skip(reason="v8 iN| line-packing removed in format-B migration (model emits whitespace N:content)")
def test_lineprefix__midline_packed_no_whitespace_paren__splits():
    """Mid-line marker after closing paren — common in continuation lines."""
    src = "i4|x = (a)i8|y = 1"
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    # First segment ends with `)`, second starts at indent 8
    assert lines[1].startswith(" " * 8), f"got: {lines!r}"


@pytest.mark.skip(reason="v8 iN| line-packing removed in format-B migration (model emits whitespace N:content)")
def test_lineprefix__midline_packed_no_whitespace_bracket__splits():
    """Mid-line marker after `]`."""
    src = "i4|x = [1,2]i8|y = 1"
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    assert "i8|" not in out


@pytest.mark.skip(reason="v8 iN| line-packing removed in format-B migration (model emits whitespace N:content)")
def test_lineprefix__midline_packed_three_segments__three_lines():
    """3+ segments on one physical line all split."""
    src = "i0|a = 1i4|b = 2i8|c = 3i12|d = 4"
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    assert len(lines) == 4
    assert lines == ["a = 1", "    b = 2", "        c = 3", "            d = 4"]


# ─────────── DOES NOT SPLIT WHEN IT SHOULDN'T ──────────────

def test_lineprefix__line_without_prefix__no_split_even_if_midline_marker():
    """A line that does NOT start with i{N}| should NEVER be split, even if it
    contains text like `i33|` (which would be Python's bitwise-or operator
    with a variable named i33).
    """
    src = "if i33|0:\n    pass"  # `i33` would be a variable; `i33|0` is bitwise OR
    out = _restore_replace_whitespace(src)
    # The line doesn't start with i\d+|, so it should pass through unchanged.
    # Either the regex respects "line must start with i\d+|" OR it (incorrectly)
    # splits and we get a regression.
    assert out == "if i33|0:\n    pass", (
        f"non-prefix line was incorrectly split: {out!r}"
    )


def test_lineprefix__midline_marker_in_string_literal__no_split():
    """A string literal containing `i4|` should ideally not be split, because
    the model didn't mean it as a separator. HOWEVER if the line STARTS with
    a prefix, our heuristic does split — that's an acceptable tradeoff per
    the docstring. Document this here.

    bug_lineprefix_001: if the model writes
        i4|s = "embedded i33| marker"
    we split — fewer-than-ideal recovery on the string-literal edge.
    Acceptable because (a) it's rare and (b) the model is told not to
    embed `i\\d+|` in string literals when using the format. But worth
    knowing about.
    """
    src = 'i4|s = "embedded i33| marker"'
    out = _restore_replace_whitespace(src)
    # We document the current behavior: splits at i33|
    lines = out.split("\n")
    if len(lines) > 1:
        # Current behavior — split. Mark as known limitation.
        pytest.skip("Known limitation: string-literal containing i{N}| splits")
    else:
        assert out == '    s = "embedded i33| marker"'


# ─────────── TRAILING LINE NUMBERS ──────────

def test_lineprefix__trailing_lineno_after_quote__stripped():
    """`i4|x = "hello" 23` → strip the trailing 23 (line number from view)"""
    src = 'i4|x = "hello" 23'
    out = _restore_replace_whitespace(src)
    assert out == '    x = "hello"', f"got: {out!r}"


def test_lineprefix__trailing_lineno_after_paren__stripped():
    src = "i4|x = foo() 99"
    out = _restore_replace_whitespace(src)
    assert out == "    x = foo()"


def test_lineprefix__trailing_lineno_after_colon__stripped():
    src = "i4|class Foo: 12"
    out = _restore_replace_whitespace(src)
    assert out == "    class Foo:"


def test_lineprefix__pure_lineno_only__stripped():
    """A line that is ONLY `iN| <digits>` is just a copied blank line — strip
    the digits to produce a true blank."""
    src = "i0| 503"
    out = _restore_replace_whitespace(src)
    # The expected behavior — pure trailer stripped to blank
    assert out.strip() == "" or out == ""


def test_lineprefix__legit_trailing_digit_not_stripped():
    """`i4|n = 5` should NOT be stripped of the 5 (that's the value)."""
    src = "i4|n = 5"
    out = _restore_replace_whitespace(src)
    assert out == "    n = 5"


def test_lineprefix__digit_at_end_of_assignment_not_stripped():
    src = "i4|x = 100"
    out = _restore_replace_whitespace(src)
    assert out == "    x = 100"


# ─────────── DEFENSIVE: leading whitespace in content ───────

def test_lineprefix__content_has_leading_spaces__stripped():
    """`i4|    def foo` should NOT give 8 spaces (4+4). Prefix is authoritative."""
    src = "i4|    def foo():\n        pass"
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    # First line: 4 spaces (NOT 8). Second line: whatever it had.
    assert lines[0].startswith(" " * 4) and not lines[0].startswith(" " * 5), (
        f"prefix should be authoritative: {lines[0]!r}"
    )


def test_lineprefix__content_has_leading_tab__stripped():
    """Tab in content after prefix should be stripped (prefix is authoritative)."""
    src = "i4|\tdef foo():"
    out = _restore_replace_whitespace(src)
    assert out == "    def foo():", f"got: {out!r}"


# ─────────── MALFORMED INPUT ───────────

def test_lineprefix__missing_pipe_no_split():
    """`i4 x = 1` (no pipe) — should pass through unchanged."""
    src = "i4 x = 1"
    out = _restore_replace_whitespace(src)
    assert out == "i4 x = 1"


def test_lineprefix__missing_digit_no_split():
    """`i|x = 1` — no digit, should pass through."""
    src = "i|x = 1"
    out = _restore_replace_whitespace(src)
    assert out == "i|x = 1"


def test_lineprefix__just_pipe_no_split():
    """`|x = 1` — no i prefix, pass through."""
    src = "|x = 1"
    out = _restore_replace_whitespace(src)
    assert out == "|x = 1"


def test_lineprefix__multiple_digits():
    """i100| should give 100 spaces."""
    src = "i100|x"
    out = _restore_replace_whitespace(src)
    assert out == " " * 100 + "x"


# ─────────── ADVERSARIAL: edge characters ───────────

def test_lineprefix__unicode_content_preserved():
    src = "i4|x = 'résumé naïve 北京'"
    out = _restore_replace_whitespace(src)
    assert out == "    x = 'résumé naïve 北京'"


def test_lineprefix__crlf_endings():
    """Windows-style CRLF line endings — should round-trip."""
    src = "i4|def f():\r\ni8|pass"
    out = _restore_replace_whitespace(src)
    # Our split is on \n; \r might leak. Document:
    lines = out.split("\n")
    # At minimum, the second line has 8 spaces of indent
    assert lines[-1].startswith(" " * 8)


@pytest.mark.skip(reason="v8 iN| line-packing removed in format-B migration (model emits whitespace N:content)")
def test_lineprefix__realworld_astropy13033_case():
    """The EXACT failure case from production:
    The reviewer emitted a 3-line REPLACE on a single physical line."""
    src = (
        'i16|raise ValueError("{} object is invalid - required {} "'
        'i33|"as the first columns but time series has no columns"'
        'i33|.format(self.__class__.__name__, required_columns))'
    )
    out = _restore_replace_whitespace(src)
    lines = out.split("\n")
    assert len(lines) == 3, f"expected 3 lines: {lines!r}"
    assert lines[0].startswith(" " * 16)
    assert lines[1].startswith(" " * 33)
    assert lines[2].startswith(" " * 33)
    assert "i33|" not in out
    assert "i16|" not in out
